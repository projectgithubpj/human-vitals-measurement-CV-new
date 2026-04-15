<<<<<<< HEAD
# human-vitals-measurement-CV
This repository contains code that helps in detecting human vitals such as Heart Rate, Respiratory Rate, HRV etc. in real-time using webcam.
=======
# CardiacVision — Real-Time Heart Rate Monitor (rPPG)

Contactless heart rate detection from a standard webcam using
**remote photoplethysmography (rPPG)** — no wearables, no contact sensors.

---

## How It Works

### Science behind rPPG
Blood volume in skin capillaries changes with each heartbeat.
These changes cause subtle, periodic variations in skin colour
(typically green channel) that are invisible to the naked eye
but detectable by a camera.

### Algorithm: CHROM (Chrominance-Based rPPG)
Based on De Haan & Jeanne (2013):

```
Frame → Face Detection (MediaPipe) → Extract ROIs (forehead + cheeks)
     → RGB mean per frame → Temporal normalisation
     → CHROM projection (Xs = 3R−2G,  Ys = 1.5R+G−1.5B)
     → Alpha-blend to cancel specular noise
     → Bandpass filter 0.75–3 Hz (45–180 BPM)
     → Welch PSD → dominant frequency → BPM
     → Temporal median smoothing → Output
```

---

## Project Structure

```
heartrate_monitor/
├── app.py                    # Flask + Socket.IO server (entry point)
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-command setup & launch
│
├── src/
│   ├── __init__.py
│   ├── rppg_processor.py     # CHROM rPPG algorithm (core)
│   ├── face_roi.py           # MediaPipe face mesh ROI extractor
│   └── video_processor.py    # Camera thread + pipeline orchestrator
│
├── templates/
│   └── index.html            # Single-page web UI
│
└── static/
    ├── css/style.css          # Dark clinical stylesheet
    └── js/app.js              # Socket.IO client + Chart.js waveform
```

---

## Quick Start

### Prerequisites
- Python 3.10 or newer
- Webcam connected
- Modern browser (Chrome / Firefox)

### Linux / macOS
```bash
cd heartrate_monitor
chmod +x setup.sh
./setup.sh
```

### Windows
```powershell
cd heartrate_monitor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Usage Tips for Accurate Readings

| Tip | Why |
|-----|-----|
| Sit still | Motion is the #1 source of noise |
| Face camera squarely | Maximises face ROI area |
| Even lighting (no backlighting) | Prevents ROI under-exposure |
| Neutral background | Reduces JPEG compression artefacts |
| Wait ~10 seconds | Buffer needs ≥3 s; 10 s gives best accuracy |
| Remove glasses if possible | Glass reduces forehead visibility |

---

## Accuracy

- Typical accuracy: **±5 BPM** under good lighting conditions
- Comparable to: consumer-grade PPG wristbands
- **Not** suitable for medical diagnosis

---

## Configuration

Edit `app.py` or pass environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT`   | `5000`  | Server port |
| `HOST`   | `0.0.0.0` | Bind address |
| `DEBUG`  | `false` | Flask debug mode |

To change window length or FPS, edit `VideoProcessor` init in `app.py`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` + `flask-socketio` | Web server + real-time WebSocket |
| `opencv-python` | Camera capture + frame encoding |
| `mediapipe` | Face landmark detection (468 points) |
| `numpy` | Numerical signal arrays |
| `scipy` | Butterworth filter + Welch PSD |
| `eventlet` | Async I/O for Socket.IO |

---

## References

- G. de Haan, V. Jeanne — *Robust Pulse Rate From Chrominance-Based rPPG* (2013)
- W. Wang et al. — *Algorithmic Principles of Remote PPG* (2017)
- MediaPipe Face Mesh — https://mediapipe.dev
>>>>>>> fb3b78fe (Initial commit)
