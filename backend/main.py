"""
main.py
-------
FastAPI backend for the Sign Language to Text Translator.

Endpoints:
  GET  /                  — serves the frontend app (index.html)
  GET  /health            — health check
  WS   /ws/translate      — real-time sign recognition via WebSocket

WebSocket protocol:
  Client → Server:  base64-encoded JPEG string (raw text message)
  Server → Client:  JSON { "sign": str, "text": str, "confidence": float, "status": str }
"""

import base64
import json
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# MediaPipe Tasks API (mediapipe >= 0.10)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_model import GestureClassifier

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Sign Language Translator API", version="1.0.0")

# Allow any localhost origin so the plain-HTML frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve frontend static files from ../frontend
# ---------------------------------------------------------------------------

_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))


@app.get("/")
async def serve_index():
    """Serve the frontend index.html at the root URL."""
    return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))


@app.get("/style.css")
async def serve_css():
    return FileResponse(os.path.join(_FRONTEND_DIR, "style.css"), media_type="text/css")


@app.get("/script.js")
async def serve_js():
    return FileResponse(os.path.join(_FRONTEND_DIR, "script.js"), media_type="application/javascript")

# ---------------------------------------------------------------------------
# MediaPipe HandLandmarker — new Tasks API (mediapipe >= 0.10)
# ---------------------------------------------------------------------------

_MODEL_TASK_PATH = os.path.join(os.path.dirname(__file__), "model", "hand_landmarker.task")

_base_options = mp_python.BaseOptions(model_asset_path=_MODEL_TASK_PATH)
_hand_options = mp_vision.HandLandmarkerOptions(
    base_options=_base_options,
    running_mode=mp_vision.RunningMode.IMAGE,   # per-frame (sync), simplest for WebSocket
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.4,
)
_hand_landmarker = mp_vision.HandLandmarker.create_from_options(_hand_options)

# ---------------------------------------------------------------------------
# Gesture classifier — tries to load ML model, falls back to rule-based
# ---------------------------------------------------------------------------

_CLASSIFIER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "gesture_model.pkl")
classifier = GestureClassifier(model_path=_CLASSIFIER_MODEL_PATH)

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Basic health check — useful for judge demos and integration tests."""
    return {
        "status": "ok",
        "mediapipe": "loaded (tasks api)",
        "classifier": "ml" if classifier.ml_model else "rule-based",
    }


# ---------------------------------------------------------------------------
# Per-connection translation state
# ---------------------------------------------------------------------------

class TranslationSession:
    """
    Holds per-WebSocket state:
      - recent predictions deque for smoothing/debouncing
      - running sentence buffer
      - last confirmed sign (to avoid repeating the same letter)

    Rules:
      1. No spaces between letters.
      2. The same letter is NOT repeated until the user removes their hand
         AND shows the same sign again after a 2-second gap.
      3. First letter is always capitalised; all subsequent letters are
         lowercase — except the standalone letter 'I' which is always
         printed as 'I'.
    """

    SMOOTHING_WINDOW = 4          # frames that must agree before confirming
    REPEAT_HAND_ABSENT_SEC = 2.0  # hand must be gone this long before same letter repeats
    MAX_SENTENCE_SIGNS = 200      # clear buffer if it grows too long

    def __init__(self):
        self.prediction_history: deque = deque(maxlen=self.SMOOTHING_WINDOW)
        self.sentence: list[str] = []          # list of formatted letter strings
        self.last_confirmed_sign: Optional[str] = None   # raw sign (e.g. 'A')
        self.last_confirmed_time: float = 0.0
        # Tracks the moment the hand last disappeared from frame
        self.hand_absent_since: Optional[float] = None
        self.hand_present: bool = False
        self.frame_count: int = 0

    def reset(self):
        """Called when the client sends a 'clear' command."""
        self.prediction_history.clear()
        self.sentence.clear()
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0.0
        self.hand_absent_since = None
        self.hand_present = False

    def on_no_hand(self):
        """Call every frame when no hand is detected."""
        now = time.time()
        if self.hand_present:
            # Hand just disappeared — start the absence timer
            self.hand_absent_since = now
        self.hand_present = False
        self.prediction_history.clear()  # flush stale predictions

    def _format_letter(self, sign: str) -> str:
        """
        Apply capitalisation rules:
          - 'I' is always uppercase.
          - First letter of the whole output is uppercase.
          - All other letters are lowercase.
        """
        if sign == 'I':
            return 'I'
        if not self.sentence:
            return sign.upper()   # very first letter → capital
        return sign.lower()

    def update(self, sign: str, confidence: float) -> Optional[str]:
        """
        Feed a new raw prediction (hand is present).
        Returns the formatted confirmed letter if accepted, else None.
        """
        now = time.time()
        self.hand_present = True
        self.prediction_history.append(sign)

        if len(self.prediction_history) < self.SMOOTHING_WINDOW:
            return None  # not enough frames yet

        # Check consensus
        counts: dict[str, int] = {}
        for s in self.prediction_history:
            counts[s] = counts.get(s, 0) + 1

        best_sign = max(counts, key=counts.__getitem__)
        best_count = counts[best_sign]

        if best_count < self.SMOOTHING_WINDOW * 0.75:
            return None  # no clear consensus

        if best_sign == "UNKNOWN":
            return None

        # ── Rule 2: same letter repeat check ──────────────────────────
        if best_sign == self.last_confirmed_sign:
            # Only allow repeat if hand was absent for ≥ 2 s
            if self.hand_absent_since is None:
                return None  # hand never left
            absence_duration = now - self.hand_absent_since
            if absence_duration < self.REPEAT_HAND_ABSENT_SEC:
                return None  # not absent long enough

        # ── Confirm ────────────────────────────────────────────────────
        formatted = self._format_letter(best_sign)
        self.last_confirmed_sign = best_sign
        self.last_confirmed_time = now
        self.hand_absent_since = None   # reset absence timer on new confirmed sign
        if len(self.sentence) < self.MAX_SENTENCE_SIGNS:
            self.sentence.append(formatted)
        return formatted

    @property
    def text(self) -> str:
        # Rule 1: no spaces between letters
        return "".join(self.sentence)


# ---------------------------------------------------------------------------
# Frame processing helpers
# ---------------------------------------------------------------------------

def _decode_frame(b64_data: str) -> Optional[np.ndarray]:
    """Decode a base64 JPEG string into an OpenCV BGR image."""
    try:
        # Strip data-URI prefix if present ("data:image/jpeg;base64,...")
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        raw = base64.b64decode(b64_data)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[decode_frame] Error: {e}")
        return None


def _extract_landmarks(img: np.ndarray) -> Optional[list]:
    """
    Run MediaPipe HandLandmarker on an image and return 21 landmarks as
    [[x, y, z], ...] in normalised image coordinates, or None.
    Uses the new Tasks API (mediapipe >= 0.10).
    """
    # Convert BGR → RGB, then wrap in MediaPipe Image
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _hand_landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]   # first detected hand
    return [[lm.x, lm.y, lm.z] for lm in hand]


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/translate")
async def ws_translate(websocket: WebSocket):
    """
    Real-time sign-language translation over WebSocket.

    Message flow:
      Client sends: base64-encoded JPEG frame (text message)
                    OR the string "CLEAR" to reset the sentence buffer
      Server sends: JSON { sign, text, confidence, status }
    """
    await websocket.accept()
    session = TranslationSession()
    print("[WS] Client connected.")

    try:
        while True:
            # ---- Receive frame or command ----
            data = await websocket.receive_text()

            # Handle special commands
            if data.strip().upper() == "CLEAR":
                session.reset()
                await websocket.send_text(json.dumps({
                    "sign": "",
                    "text": "",
                    "confidence": 0.0,
                    "status": "cleared",
                }))
                continue

            session.frame_count += 1

            # ---- Decode & analyse frame ----
            img = _decode_frame(data)
            if img is None:
                await websocket.send_text(json.dumps({
                    "sign": "ERROR",
                    "text": session.text,
                    "confidence": 0.0,
                    "status": "decode_error",
                }))
                continue

            landmarks = _extract_landmarks(img)

            if landmarks is None:
                # No hand detected — update absence state, send current text unchanged
                session.on_no_hand()
                await websocket.send_text(json.dumps({
                    "sign": "No hand",
                    "text": session.text,
                    "confidence": 0.0,
                    "status": "no_hand",
                }))
                continue

            # ---- Classify ----
            raw_sign, confidence = classifier.predict(landmarks)

            # ---- Smooth + debounce ----
            confirmed = session.update(raw_sign, confidence)

            # ---- Respond ----
            await websocket.send_text(json.dumps({
                "sign": raw_sign,
                "text": session.text,
                "confidence": round(confidence, 3),
                "status": "confirmed" if confirmed else "detecting",
            }))

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "sign": "ERROR",
                "text": session.text,
                "confidence": 0.0,
                "status": f"error: {str(e)}",
            }))
        except Exception:
            pass
