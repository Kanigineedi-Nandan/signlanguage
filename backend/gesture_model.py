"""
gesture_model.py
----------------
Rule-based ASL alphabet classifier using MediaPipe 21-landmark geometry.

Official ASL letter definitions:
  A  -- fist, thumb rests on side of index finger (thumb sideways)
  B  -- four fingers straight up, thumb folded flat across palm
  C  -- all fingers + thumb curved in a C-arc (open toward camera)
  D  -- index UP, middle+ring+pinky curl to touch thumb (forming a circle)
  E  -- all fingers curled toward palm, thumb tucked under/across
  F  -- index+thumb tips touch (small circle), middle+ring+pinky extended up
  G  -- index + thumb pointing sideways like a gun (horizontal)
  H  -- index + middle both extended sideways (horizontal)
  I  -- only pinky extended upward, rest in fist
  J  -- same static shape as I (motion letter)
  K  -- index up, middle up, thumb tip between/near middle finger
  L  -- index pointing up, thumb pointing out to side (L-shape)
  M  -- index+middle+ring folded OVER thumb (thumb tucked under 3 fingers)
  N  -- index+middle folded over thumb (thumb tucked under 2 fingers)
  O  -- all tips curve to meet thumb tip (oval/O shape)
  P  -- like K but rotated so fingers point forward/down
  Q  -- like G but pointing downward
  R  -- index + middle crossed over each other, pointing up
  S  -- tight fist, thumb across FRONT of all fingers
  T  -- thumb tip tucked between index and middle fingers
  U  -- index + middle together, both pointing up
  V  -- index + middle spread apart in V-shape, pointing up
  W  -- index + middle + ring all extended and spread
  X  -- only index finger, hooked/bent (hook shape)
  Y  -- thumb + pinky extended, middle 3 curled
  Z  -- only index extended upward (traces Z -- static pose)
"""

import math
import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# MediaPipe 21-landmark indices
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist(a, b) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _angle(a, b, c) -> float:
    """Angle at vertex B (degrees)."""
    ba = [a[i] - b[i] for i in range(3)]
    bc = [c[i] - b[i] for i in range(3)]
    dot = sum(ba[i]*bc[i] for i in range(3))
    mag_ba = math.sqrt(sum(x**2 for x in ba)) + 1e-9
    mag_bc = math.sqrt(sum(x**2 for x in bc)) + 1e-9
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))))


def _pip_angle(lm, mcp, pip, tip) -> float:
    return _angle(lm[mcp], lm[pip], lm[tip])


def _finger_extended(lm, mcp, pip, tip, thr=150) -> bool:
    return _pip_angle(lm, mcp, pip, tip) > thr


def _pointing_up(lm, mcp, tip) -> bool:
    dy = lm[mcp][1] - lm[tip][1]
    dx = abs(lm[tip][0] - lm[mcp][0])
    return dy > 0.03 and dy > dx * 0.6


def _pointing_sideways(lm, mcp, tip) -> bool:
    dx = abs(lm[tip][0] - lm[mcp][0])
    dy = abs(lm[tip][1] - lm[mcp][1])
    return dx > dy * 0.8 and dx > 0.025


def _pointing_down(lm, mcp, tip) -> bool:
    return lm[tip][1] > lm[mcp][1] + 0.04


def _thumb_extended(lm) -> bool:
    """Thumb out to the side (not tucked into fist). Works for left/right hands."""
    if lm[WRIST][0] < lm[PINKY_MCP][0]:   # right hand
        return lm[THUMB_TIP][0] < lm[THUMB_IP][0] - 0.01
    else:                                   # left hand
        return lm[THUMB_TIP][0] > lm[THUMB_IP][0] + 0.01


def _finger_states(lm):
    thumb  = _thumb_extended(lm)
    index  = _finger_extended(lm, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)
    middle = _finger_extended(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    ring   = _finger_extended(lm, RING_MCP,   RING_PIP,   RING_TIP)
    pinky  = _finger_extended(lm, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)
    return thumb, index, middle, ring, pinky


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_rule_based(landmarks: list) -> Tuple[str, float]:
    lm = landmarks
    thumb, index, middle, ring, pinky = _finger_states(lm)

    hand_span = _dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-9

    # Tip-to-tip distances (normalised)
    ti = _dist(lm[THUMB_TIP], lm[INDEX_TIP])  / hand_span
    tm = _dist(lm[THUMB_TIP], lm[MIDDLE_TIP]) / hand_span
    tr = _dist(lm[THUMB_TIP], lm[RING_TIP])   / hand_span
    tp = _dist(lm[THUMB_TIP], lm[PINKY_TIP])  / hand_span
    im = _dist(lm[INDEX_TIP], lm[MIDDLE_TIP]) / hand_span

    # PIP angles
    idx_pip = _pip_angle(lm, INDEX_MCP,  INDEX_PIP,  INDEX_TIP)
    mid_pip = _pip_angle(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
    rng_pip = _pip_angle(lm, RING_MCP,   RING_PIP,   RING_TIP)
    pky_pip = _pip_angle(lm, PINKY_MCP,  PINKY_PIP,  PINKY_TIP)
    avg_pip = (idx_pip + mid_pip + rng_pip + pky_pip) / 4

    # Direction flags
    idx_up   = _pointing_up(lm,       INDEX_MCP,  INDEX_TIP)
    idx_side = _pointing_sideways(lm, INDEX_MCP,  INDEX_TIP)
    idx_down = _pointing_down(lm,     INDEX_MCP,  INDEX_TIP)
    mid_up   = _pointing_up(lm,       MIDDLE_MCP, MIDDLE_TIP)
    mid_side = _pointing_sideways(lm, MIDDLE_MCP, MIDDLE_TIP)
    mid_down = _pointing_down(lm,     MIDDLE_MCP, MIDDLE_TIP)
    pky_up   = _pointing_up(lm,       PINKY_MCP,  PINKY_TIP)

    ext = sum([thumb, index, middle, ring, pinky])

    # =========================================================================
    # 1. FIVE FINGERS
    # =========================================================================

    # B: four fingers up + thumb folded (not extended)
    if index and middle and ring and pinky and not thumb:
        return "B", 0.91

    # Open palm (all 5) -- still treat as B
    if ext == 5:
        return "B", 0.68

    # =========================================================================
    # 2. FOUR-FINGER COMBINATIONS
    # =========================================================================

    # W: index+middle+ring extended and spread (no pinky, no thumb)
    if index and middle and ring and not pinky and not thumb:
        return "W", 0.88

    # =========================================================================
    # 3. THREE-FINGER COMBINATIONS
    # =========================================================================

    # K: thumb + index + middle up; thumb tip near middle finger
    if thumb and index and middle and not ring and not pinky:
        if idx_up and mid_up:
            if tm < 0.60:
                return "K", 0.85
            return "K", 0.72
        # P: like K but fingers pointing down/forward
        if idx_down or mid_down:
            return "P", 0.82
        return "K", 0.65

    # F: middle + ring + pinky up; index+thumb touch (circle)
    if middle and ring and pinky and not index:
        if ti < 0.45:
            return "F", 0.88
        return "F", 0.72

    # =========================================================================
    # 4. TWO-FINGER COMBINATIONS
    # =========================================================================

    # Y: thumb + pinky (shaka)
    if thumb and pinky and not index and not middle and not ring:
        return "Y", 0.93

    # L: index + thumb  (L-shape -- index UP, thumb SIDEWAYS)
    if index and thumb and not middle and not ring and not pinky:
        if idx_up:
            return "L", 0.91
        if idx_side:
            return "G", 0.82    # more sideways than up
        if idx_down:
            return "Q", 0.78
        return "L", 0.68

    # H/U/V/R: index + middle (no thumb, no ring, no pinky)
    if index and middle and not ring and not pinky and not thumb:
        # H: both fingers pointing sideways
        if idx_side and mid_side:
            return "H", 0.89
        # R: fingers crossed -- tips very close together, nearly overlapping
        if im < 0.10:
            return "R", 0.82
        # V: fingers clearly spread in a V
        if im > 0.24 and idx_up and mid_up:
            return "V", 0.88
        # U: fingers together pointing up
        if idx_up and mid_up:
            return "U", 0.86
        # H fallback (if somewhat sideways)
        if idx_side or mid_side:
            return "H", 0.72
        return "U", 0.62

    # =========================================================================
    # 5. SINGLE-FINGER / SINGLE-DIGIT
    # =========================================================================

    # I: only pinky extended upward
    if pinky and not index and not middle and not ring and not thumb:
        if pky_up:
            return "I", 0.91
        return "I", 0.75

    # Index-only group: D, G, Q, X, Z
    if index and not middle and not ring and not pinky and not thumb:
        # X: index hooked (PIP tightly bent)
        if idx_pip < 115:
            return "X", 0.85
        # Z: index pointing straight up (no thumb)
        if idx_up:
            return "Z", 0.83
        # G: index sideways
        if idx_side:
            return "G", 0.75
        # Q: index down
        if idx_down:
            return "Q", 0.72
        return "Z", 0.55

    # D: index UP + thumb makes circle with middle/ring/pinky
    # (index extended, others curled DOWN, thumb tip near those curled fingers)
    if index and thumb and not middle and not ring and not pinky:
        # This is actually L (handled above). But in D the thumb forms the circle
        # so it should NOT be extended outward. If we reach here with thumb+index
        # and idx_up but tm is small, it is D
        if idx_up and tm < 0.50:
            return "D", 0.84
        return "L", 0.70

    # =========================================================================
    # 6. ALL-CLOSED / FIST VARIANTS (no fingers extended)
    # =========================================================================

    if not index and not middle and not ring and not pinky:

        # O: all tips curve to meet each other and thumb
        if ti < 0.48 and tm < 0.55 and avg_pip > 95:
            return "O", 0.85

        # C: fingers curved but open (wide arc, tips far from thumb)
        if 85 < avg_pip < 148 and ti > 0.55:
            return "C", 0.80

        # Now all fingers are curled (avg_pip likely < 130)

        # T: thumb tip tucked between index-MCP and middle-MCP (x-axis)
        idx_mcp_x = lm[INDEX_MCP][0]
        mid_mcp_x = lm[MIDDLE_MCP][0]
        rng_mcp_x = lm[RING_MCP][0]
        lo_im = min(idx_mcp_x, mid_mcp_x)
        hi_im = max(idx_mcp_x, mid_mcp_x)
        lo_ir = min(idx_mcp_x, rng_mcp_x)
        hi_ir = max(idx_mcp_x, rng_mcp_x)
        tx = lm[THUMB_TIP][0]
        ty = lm[THUMB_TIP][1]
        thumb_between_idx_mid  = lo_im < tx < hi_im
        thumb_between_idx_ring = lo_ir < tx < hi_ir

        # T check: thumb between index+middle, tip at roughly index-PIP height
        if not thumb and thumb_between_idx_mid and ty < lm[INDEX_PIP][1] + 0.06:
            return "T", 0.79

        # E: all fingers tightly curled, thumb tucked UNDER fingers
        if avg_pip < 118 and not thumb:
            if ty > lm[INDEX_MCP][1] - 0.04:   # thumb not above index MCP
                return "E", 0.79

        # M: 3 fingers (index+middle+ring) folded over thumb
        if not thumb and thumb_between_idx_ring:
            return "M", 0.77

        # N: 2 fingers (index+middle) folded over thumb
        if not thumb and thumb_between_idx_mid:
            return "N", 0.74

        # A: thumb extended to side of fist
        if thumb:
            return "A", 0.87

        # S: tight fist, thumb across front
        return "S", 0.77

    # =========================================================================
    # 7. FALLBACK
    # =========================================================================

    if 80 < avg_pip < 145:
        if ti > 0.55:
            return "C", 0.55
        return "E", 0.50

    return "UNKNOWN", 0.30


# ---------------------------------------------------------------------------
# ML model wrapper (optional)
# ---------------------------------------------------------------------------

class GestureClassifier:
    """Loads an sklearn model if present, otherwise falls back to rule-based."""

    def __init__(self, model_path: Optional[str] = None):
        self.ml_model = None
        self.label_encoder: list = []

        if model_path:
            try:
                import pickle
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                self.ml_model = saved["model"]
                self.label_encoder = saved["labels"]
                print(f"[GestureClassifier] Loaded ML model from {model_path}")
            except FileNotFoundError:
                print(f"[GestureClassifier] No model at {model_path}, using rule-based fallback.")
            except Exception as e:
                print(f"[GestureClassifier] Failed to load model ({e}), using rule-based fallback.")

    def predict(self, landmarks: list) -> Tuple[str, float]:
        if self.ml_model is not None and self.label_encoder:
            try:
                flat = np.array([[coord for pt in landmarks for coord in pt]])
                proba = self.ml_model.predict_proba(flat)[0]
                idx = int(np.argmax(proba))
                label = self.label_encoder[idx]
                conf = float(proba[idx])
                return label, conf
            except Exception as e:
                print(f"[GestureClassifier] ML inference error: {e}. Falling back.")

        return classify_rule_based(landmarks)
