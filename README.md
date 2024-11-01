# Nudge: IoT Drowsiness & Yawn Detection System

An intelligent IoT-enabled real-time driver drowsiness, blink rate, and yawn monitoring system built using computer vision, OpenCV, Dlib facial landmark estimation, and custom deep learning pipelines.

## Features

- 🚘 **Real-time Driver Monitoring**: Live video stream evaluation for eye aspect ratio (EAR), closure duration, and yawn frequency.
- 🔔 **Multi-stage Audio Alerts**: Differentiating warning triggers (`alarm.wav`, `drowsy_alarm.wav`, `yawn_alarm.mp3`, `mobile_alarm.wav`).
- 👤 **Face Recognition & Encodings**: User identification and session logging.
- 📹 **Automated Recording**: Captures driver distraction & fatigue video snippets into `recorded_video/`.
- 🌐 **Web Dashboard**: Flask-based interactive monitoring application.
- 📊 **Training & Analytics**: Evaluation runs, dataset split pipelines, and performance logging.

## Directory Structure

```text
.
├── app.py                      # Primary Flask application server
├── app1.py / APP2.PY           # Web portal variants & API routes
├── blink_rate.py               # Eye closure & EAR calculation module
├── detection_final.py          # Core real-time detection pipeline
├── shape_predictor_68_face_landmarks.dat  # Dlib 68-point facial landmark predictor
├── datasets/                   # Driver facial dataset images & annotations
├── recorded_video/             # Captured monitoring snippets
├── static/ & templates/        # Web dashboard UI assets
└── runs/                       # Training logs and detection metrics
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application server:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://localhost:5000`.

## License

MIT License.
