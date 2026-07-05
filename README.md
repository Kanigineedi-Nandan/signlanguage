# 🤟 Sign Language Translator

A real-time **American Sign Language (ASL) → Text** web application built for hackathon demos.

Uses your webcam to detect hand gestures via MediaPipe, classifies them with a rule-based gesture recogniser (or optional scikit-learn RandomForest), and streams translated text to the browser over WebSocket.

---

## Demo Signs Supported

| Sign | Gesture |
|------|---------|
| **A** | Closed fist |
| **B** | 4 fingers up, thumb folded in |
| **C** | Curved open hand |
| **D** | Index up, thumb + index form circle |
| **HELLO** | Full open hand (all 5 fingers) |
| **YES** | Thumbs up |
| **NO** | Index + middle fingers up together |
| **THANK YOU** | Index finger only pointing up |
| **I LOVE YOU** | Thumb + index + pinky up |
| **PEACE** | Index + middle spread apart |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Plain HTML + CSS + Vanilla JS |
| Backend | Python 3.13 · FastAPI · Uvicorn |
| Computer Vision | OpenCV · MediaPipe Hands |
| Classification | Rule-based geometry (+ optional sklearn RandomForest) |
| Transport | WebSocket (`ws://localhost:8000/ws/translate`) |

---

## Project Structure

```
signlanguage/
├── backend/
│   ├── venv/                  ← Python virtual environment
│   ├── main.py                ← FastAPI app + WebSocket endpoint
│   ├── gesture_model.py       ← Rule-based + ML classifier
│   ├── train_model.py         ← Optional: retrain with your own data
│   ├── model/                 ← Trained model saved here (gesture_model.pkl)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── .gitignore
└── README.md
```

---

## ⚡ Quick Start — Copy-Paste Commands

### Step 1 — One-time Setup (already done if venv/ exists)

```powershell
cd C:\Users\kanig\Downloads\signlanguage\backend

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If you get a PowerShell execution policy error, run this once first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 2 — Start the Backend

Open **Terminal 1**:

```powershell
cd C:\Users\kanig\Downloads\signlanguage\backend
.\venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
[GestureClassifier] No model at ...\model\gesture_model.pkl, using rule-based fallback.
```

✅ **Health check:** Open http://localhost:8000/health → `{"status":"ok","mediapipe":"loaded","classifier":"rule-based"}`

---

### Step 3 — Serve the Frontend

Open **Terminal 2**:

```powershell
cd C:\Users\kanig\Downloads\signlanguage\frontend
python -m http.server 5500
```

Then open your browser at: **http://localhost:5500**

> ⚠️ **Do NOT open `index.html` as a `file://` URL** — camera access requires `http://localhost` or `https://`.

---

### Step 4 — Use the App

1. Browser asks for **camera + microphone** → click **Allow**
2. Webcam feed fills the top 60% of screen
3. Click **▶ Start Translation**
4. Hold your hand up to the camera and sign
5. Detected sign appears in the status bar; confirmed words accumulate in the text box
6. **🗑 Clear** to reset · **📋 Copy** to copy text

---

## Backend API

### `GET /health`
```json
{ "status": "ok", "mediapipe": "loaded", "classifier": "rule-based" }
```

### `WS /ws/translate`

| Direction | Format |
|-----------|--------|
| Client → Server | Base64 JPEG string (or `"CLEAR"` to reset) |
| Server → Client | `{ "sign": "HELLO", "text": "HELLO YES", "confidence": 0.90, "status": "confirmed" }` |

**Status values:** `confirmed`, `detecting`, `no_hand`, `cleared`, `error`

---

## Optional: Train Your Own ML Model

```powershell
# 1. Create data/ folder with labelled CSVs (one per sign, 63 floats/row)
mkdir backend\data

# 2. Train
cd backend
.\venv\Scripts\Activate.ps1
python train_model.py

# 3. Restart backend — it auto-loads model\gesture_model.pkl
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv: `.\venv\Scripts\Activate.ps1` |
| WebSocket won't connect | Start backend on port 8000 before clicking Start |
| Camera permission denied | Click 🔒 in browser address bar → allow camera |
| Black video / `file://` blocked | Use `python -m http.server 5500` not `file://` |
| PowerShell script blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Architecture

```
Browser (http://localhost:5500)
  │  getUserMedia() → <video>
  │  <canvas> captures @ 8 FPS → base64 JPEG
  │
  │  WebSocket ──────────────────────────────────────────►
  │                                              FastAPI (port 8000)
  │                                                │ Decode JPEG → OpenCV
  │                                                │ MediaPipe Hands → 21 landmarks
  │                                                │ GestureClassifier.predict()
  │                                                │ TranslationSession.update() (smooth+debounce)
  │  ◄─────────────────────────────────────────── │ JSON { sign, text, confidence, status }
  │
  └─ Update UI: sign chip, confidence bar, text output
```

---

*Built with ❤️ for hackathon demos*
