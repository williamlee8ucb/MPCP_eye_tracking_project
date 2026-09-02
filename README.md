# Real-Time Eye Tracking System
**Computer Vision Pipeline**

A real-time eye tracking system built with MediaPipe FaceLandmarker and OpenCV. Detects facial landmarks, computes per-eye Eye Aspect Ratio (EAR), classifies each eye as OPEN or CLOSED, counts blinks, detects winks, and logs session data to CSV. Supports both live webcam input and pre-recorded video files.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Algorithm](#algorithm)
5. [Visual Overlay](#visual-overlay)
6. [CSV Output](#csv-output)
7. [Project Structure](#project-structure)
8. [Running Tests](#running-tests)
9. [Known Limitations](#known-limitations)

---

## Features

**Core**
- 30 fps webcam feed with MediaPipe 478-point face mesh
- Per-eye EAR calculation and OPEN / CLOSED classification
- Noise-robust blink detection (requires N consecutive closed frames to confirm)

**Level 1 — Enhanced analytics**
- Per-eye blink counts tracked independently (left and right)
- Wink detection — one eye closed while the other stays open
- Blink duration measurement (ms from confirmation to reopening)
- Rolling blink frequency (blinks per minute over a 60-second window)

**Level 2 — Production features**
- Temporal EAR smoothing (rolling average, configurable window) to reduce jitter
- Session CSV logging — one row per frame with raw EAR, smoothed EAR, state, threshold
- Live threshold adjustment via `[` / `]` keyboard keys, no restart needed
- Video file input in addition to webcam (`--video path/to/file.mp4`)

---

## Installation

**Requires Python 3.10 or later.** The tracker uses the `X | Y` union type syntax introduced in 3.10. Check your version with `python --version`.

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. MediaPipe model

The face landmark model (`face_landmarker.task`, ~5 MB) is downloaded automatically on first run and cached in the working directory. An internet connection is required only for this first run. To pre-download manually:

```bash
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

---

## Usage

### Webcam (default)

```bash
python eye_tracker.py
```

### External or virtual camera

```bash
python eye_tracker.py --camera 1
```

### Video file

```bash
python eye_tracker.py --video path/to/recording.mp4
```

Any format OpenCV supports works: `.mp4`, `.avi`, `.mov`, `.mkv`. The tracker reads to the end of the file and exits cleanly.

### Custom threshold

```bash
python eye_tracker.py --threshold 0.19
```

Recommended range: `0.18`–`0.25`. Lower values reduce false CLOSED detections for people with naturally smaller eyes; higher values catch subtle blinks more aggressively.

### Disable CSV logging

```bash
python eye_tracker.py --no-csv
```

### Custom smoothing window

```bash
python eye_tracker.py --smooth 8
```

Number of frames averaged for the smoothed EAR value. Higher values reduce jitter but add slight lag to blink detection. Default is `5` (≈167 ms at 30 fps).

### All options

```
python eye_tracker.py --help

  --camera INT      Webcam device index (default: 0)
  --video PATH      Video file path — overrides --camera
  --threshold FLOAT Initial EAR threshold (default: 0.21)
  --smooth INT      Smoothing window in frames (default: 5)
  --no-csv          Disable CSV session logging
```

### Keyboard controls (while running)

| Key | Action |
|-----|--------|
| `[` | Decrease threshold by 0.01 |
| `]` | Increase threshold by 0.01 |
| `Q` or `ESC` | Quit |

Threshold changes take effect immediately on the next frame and are reflected in the HUD and CSV output.

---

## Algorithm

### Eye Aspect Ratio (EAR)

```
EAR = ( ||P2−P6|| + ||P3−P5|| ) / ( 2 × ||P1−P4|| )
```

Six landmarks are extracted per eye, positioned around the eyelid:

```
        P2  P3
   P1           P4
        P6  P5
```

| Point | Role |
|-------|------|
| P1 | Left corner |
| P4 | Right corner |
| P2, P3 | Upper eyelid |
| P5, P6 | Lower eyelid |

When the eye is open the vertical distances (P2–P6, P3–P5) are large relative to the horizontal (P1–P4), producing a high EAR. As the eye closes, the vertical distances collapse while the horizontal stays roughly constant, driving EAR toward zero.

**Default threshold: `EAR ≥ 0.21 → OPEN`, `EAR < 0.21 → CLOSED`**

### Temporal smoothing

Raw EAR values are passed through a rolling-average smoother before classification. This prevents single noisy frames from triggering false blinks. The smoothed value is what drives the state machine and gets written to the CSV; the raw value is also recorded for reference.

### Blink confirmation

A blink is confirmed only after the smoothed EAR stays below the threshold for `CONSEC_FRAMES_NEEDED` (default 2) consecutive frames. This debounces landmark jitter that might dip below the threshold briefly without representing a real eye closure. Duration is measured from the confirmation frame to the frame where the eye reopens.

### Wink detection

A wink is registered when one eye's state machine fires `blink_start` while the other eye is currently open. Left and right winks are counted separately.

### MediaPipe landmark indices

| Eye | EAR indices |
|-----|-------------|
| Left | `362, 385, 387, 263, 373, 380` |
| Right | `33, 160, 158, 133, 153, 144` |

These index into MediaPipe's 478-point FaceLandmarker topology.

---

## Visual Overlay

| Element | Meaning |
|---------|---------|
| Grey dots | All 478 face mesh landmarks |
| **Green** polygon | Eye contour — OPEN |
| **Red** polygon | Eye contour — CLOSED |
| Yellow dots | The 6 EAR key-points per eye |
| Top-left HUD | Raw → smoothed EAR, state, threshold, blink/wink stats |
| Bottom banner | `EYES CLOSED`, `LEFT WINK`, or `RIGHT WINK` |
| Centre banner | `NO FACE DETECTED` |

The HUD shows both raw and smoothed EAR for each eye in the format `L EAR: 0.248 → 0.251`, making it easy to see how much smoothing is being applied.

---

## CSV Output

A file named `session_YYYYMMDD_HHMMSS.csv` is written to the working directory. One row is written per frame where a face is detected.

| Column | Description |
|--------|-------------|
| `timestamp_ms` | Monotonic millisecond timestamp |
| `frame` | Frame counter |
| `left_ear` | Raw left EAR |
| `left_ear_smooth` | Smoothed left EAR |
| `left_state` | `OPEN` or `CLOSED` |
| `right_ear` | Raw right EAR |
| `right_ear_smooth` | Smoothed right EAR |
| `right_state` | `OPEN` or `CLOSED` |
| `avg_ear` | Average raw EAR |
| `avg_ear_smooth` | Average smoothed EAR |
| `blink_count` | Bilateral blink count at this frame |
| `left_winks` | Left wink count at this frame |
| `right_winks` | Right wink count at this frame |
| `threshold` | EAR threshold in effect at this frame |

The threshold column changes value mid-session if you use `[` / `]` during recording, making it straightforward to analyse the effect of different thresholds in post.

---

## Project Structure

```
eye_tracking_project/
├── eye_tracker.py              # Main implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── face_landmarker.task        # MediaPipe model (auto-downloaded)
├── session_YYYYMMDD_HHMMSS.csv # Session log (created on each run)
└── tests/
    ├── conftest.py              # Shared fixtures and mock helpers
    ├── test_ear.py              # EAR formula unit tests
    ├── test_classification.py   # OPEN/CLOSED classification + blink counting
    ├── test_edge_cases.py       # No face, degenerate input, process_frame
    ├── test_camera.py           # Camera init, clean exit, error handling
    ├── test_performance.py      # Frame speed, 7-minute sustained run
    ├── test_enhanced_features.py # Blink duration, winks, per-eye counts, frequency
    └── test_level2_features.py  # Smoothing, CSV, keyboard threshold, video input
```

---

## Running Tests

All tests mock out MediaPipe and OpenCV's camera — no webcam or model file is required.

```bash
# Run everything
python -m pytest tests/ -v

# One module at a time
python -m pytest tests/test_ear.py -v
python -m pytest tests/test_level2_features.py -v

# With coverage
pip install pytest-cov
python -m pytest tests/ --cov=eye_tracker --cov-report=term-missing
```

---

## Known Limitations

**Threshold is not universal.** The default EAR threshold of 0.21 works well for most adults looking directly at the camera, but the ideal value varies with eye shape, lighting, face angle, and distance from the camera. People with naturally narrow eyes (common in some ethnicities) may need a lower threshold (e.g. `0.16`–`0.18`). Use the `[` / `]` keys to tune live, or run with `--threshold` and a pre-measured value.

**Single face only.** The tracker processes the first detected face only. If multiple people are in frame the closest face is used and others are ignored.

**Head pose sensitivity.** EAR degrades at large yaw or pitch angles (roughly beyond ±30°). Looking far to the side or up/down causes the eyelid landmarks to compress, producing artificially low EAR values that can trigger false blink detections.

**Lighting conditions.** MediaPipe's face detection confidence drops in very low light, strong backlighting, or high-contrast shadows across the face. The `NO FACE DETECTED` banner will appear more frequently in these conditions.

**Glasses and reflections.** Thick-framed glasses can partially occlude the eyelid landmarks. Reflections from lens coatings occasionally cause landmark positions to shift, resulting in transient EAR spikes. The smoothing window mitigates this but does not eliminate it.

**Video file timestamps.** Some video codecs produce non-monotonic or duplicate timestamps via `CAP_PROP_POS_MSEC`. The tracker handles this by incrementing the last known timestamp by 1 ms, but this means blink durations logged for such files may be off by a few milliseconds.

**No GPU acceleration.** The tracker runs on CPU only. On machines without hardware acceleration, frame processing may drop below 30 fps at higher resolutions. Use `--smooth 3` or lower to reduce per-frame work if needed.

**CSV grows unboundedly.** A 30-minute session at 30 fps with a face present will produce roughly 54,000 rows (~4–5 MB). There is no automatic rotation or size cap.
