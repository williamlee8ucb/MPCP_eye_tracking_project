"""
eye_tracker.py
==============
Real-time eye tracking system using MediaPipe FaceLandmarker + OpenCV.

Features
--------
  - Per-eye OPEN / CLOSED classification via Eye Aspect Ratio (EAR)
  - Blink count, blink frequency, blink duration, wink detection
  - Temporal EAR smoothing (rolling average, reduces jitter)
  - CSV session logging  (timestamp, EAR, smoothed EAR, state per frame)
  - Live threshold adjustment via keyboard  ([ to lower, ] to raise)
  - Webcam OR video file input

Algorithm
---------
    EAR = ( ||P2-P6|| + ||P3-P5|| ) / ( 2 * ||P1-P4|| )
    EAR >= threshold  →  OPEN  /  EAR < threshold  →  CLOSED

Usage
-----
    python eye_tracker.py                          # webcam
    python eye_tracker.py --camera 1               # second webcam
    python eye_tracker.py --video path/to/file.mp4 # video file
    python eye_tracker.py --threshold 0.19         # custom threshold
    python eye_tracker.py --no-csv                 # disable CSV logging

Keyboard controls (while running)
----------------------------------
    [        decrease threshold by 0.01
    ]        increase threshold by 0.01
    Q / ESC  quit
"""

import csv
import sys
import time
import argparse
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path("face_landmarker.task")

# ---------------------------------------------------------------------------
# Eye landmark indices  (MediaPipe 478-point topology)
# ---------------------------------------------------------------------------

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

LEFT_EYE_CONTOUR  = [362, 382, 381, 380, 374, 373, 390, 249,
                     263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_CONTOUR = [33,   7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_SKIP           = 1
FREQUENCY_WINDOW_SEC = 60
THRESHOLD_STEP       = 0.01
THRESHOLD_MIN        = 0.10
THRESHOLD_MAX        = 0.40
SMOOTH_WINDOW        = 5      # frames used in rolling EAR average

CSV_COLUMNS = [
    "timestamp_ms", "frame",
    "left_ear", "left_ear_smooth", "left_state",
    "right_ear", "right_ear_smooth", "right_state",
    "avg_ear", "avg_ear_smooth",
    "blink_count", "left_winks", "right_winks",
    "threshold",
]


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class _EarSmoother:
    """
    Rolling-average smoother for EAR values.

    Keeps a fixed-length deque of recent raw EAR values and returns
    their mean. This reduces frame-to-frame jitter without introducing
    the lag of a heavier filter.

    Parameters
    ----------
    window : int
        Number of frames to average (default 5).
    """

    def __init__(self, window: int = SMOOTH_WINDOW):
        self._window = window
        self._buf: deque = deque(maxlen=window)

    def update(self, ear: float) -> float:
        """Add a new EAR value and return the smoothed value."""
        self._buf.append(ear)
        return round(float(np.mean(self._buf)), 3)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def window(self) -> int:
        return self._window


class _EyeState:
    """
    Per-eye blink state machine.

    Tracks consecutive frames below threshold, confirms blinks,
    measures blink duration.
    """

    def __init__(self, consec_needed: int = 2):
        self.consec_needed      = consec_needed
        self.consec_below       = 0
        self.blink_count        = 0
        self.is_blinking        = False
        self.blink_start_ms: Optional[int]   = None
        self.last_duration_ms: Optional[float] = None

    def update(self, is_open: bool, timestamp_ms: int) -> str:
        """
        Feed one frame. Returns event string:
        'blink_start' | 'blinking' | 'closed_unconfirmed' | 'open'
        """
        if not is_open:
            self.consec_below += 1
            if not self.is_blinking and self.consec_below >= self.consec_needed:
                self.is_blinking    = True
                self.blink_start_ms = timestamp_ms
                return "blink_start"
            return "blinking" if self.is_blinking else "closed_unconfirmed"
        else:
            if self.is_blinking:
                if self.blink_start_ms is not None:
                    self.last_duration_ms = timestamp_ms - self.blink_start_ms
                self.blink_count   += 1
                self.is_blinking    = False
                self.blink_start_ms = None
            self.consec_below = 0
            return "open"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EyeTracker:
    """
    Real-time eye tracker with smoothing, CSV logging, and video file support.

    Parameters
    ----------
    ear_threshold : float
        Initial EAR classification threshold (default 0.21).
    source : int | str
        Camera index (int) or path to a video file (str).
        Default: 0 (first webcam).
    enable_csv : bool
        Write a per-frame CSV log to the working directory (default True).
    smooth_window : int
        Number of frames used in the rolling EAR average (default 5).
    """

    def __init__(
        self,
        ear_threshold: float = 0.21,
        source: Union[int, str]    = 0,
        enable_csv: bool     = True,
        smooth_window: int   = SMOOTH_WINDOW,
    ):
        # ── Model ─────────────────────────────────────────────────────────
        if not MODEL_PATH.exists():
            print(f"[EyeTracker] Downloading model → {MODEL_PATH} ...")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            except Exception as exc:
                print(f"[EyeTracker] Download failed: {exc}")
                sys.exit(1)
            print("[EyeTracker] Download complete.")

        # ── MediaPipe ─────────────────────────────────────────────────────
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # ── Threshold ─────────────────────────────────────────────────────
        self.ear_threshold = float(ear_threshold)

        # ── Source ────────────────────────────────────────────────────────
        # int → webcam index, str → video file path
        self.source      = source
        self._is_video   = isinstance(source, str)

        # ── Per-eye state + smoothers ──────────────────────────────────────
        self.left_eye    = _EyeState(consec_needed=2)
        self.right_eye   = _EyeState(consec_needed=2)
        self._left_smooth  = _EarSmoother(window=smooth_window)
        self._right_smooth = _EarSmoother(window=smooth_window)

        # ── Wink counters ─────────────────────────────────────────────────
        self.left_wink_count  = 0
        self.right_wink_count = 0

        # ── Blink frequency ───────────────────────────────────────────────
        self._blink_times: deque = deque()

        # ── Frame counter + cache ─────────────────────────────────────────
        self.frame_count       = 0
        self._last_annotated: Optional[np.ndarray] = None
        self._session_start_ms: Optional[int]      = None

        # ── CSV logging ───────────────────────────────────────────────────
        self.enable_csv  = enable_csv
        self._csv_file   = None
        self._csv_writer = None
        if enable_csv:
            self._open_csv()

        print(f"[EyeTracker] Ready  |  threshold={self.ear_threshold}  "
              f"source={source}  csv={enable_csv}  "
              f"smooth_window={smooth_window}")

    # ------------------------------------------------------------------ #
    # CSV helpers
    # ------------------------------------------------------------------ #

    def _open_csv(self) -> None:
        """Open a timestamped CSV file and write the header row."""
        stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(f"csv_logs/session_{stamp}.csv")
        self._csv_file   = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS)
        self._csv_writer.writeheader()
        print(f"[EyeTracker] Logging to {csv_path}")

    def _log_row(
        self,
        timestamp_ms: int,
        left_ear: float,  left_smooth: float,  left_open: bool,
        right_ear: float, right_smooth: float, right_open: bool,
        avg_ear: float,   avg_smooth: float,
    ) -> None:
        """Write one data row to the CSV (no-op if CSV is disabled)."""
        if self._csv_writer is None:
            return
        self._csv_writer.writerow({
            "timestamp_ms":   timestamp_ms,
            "frame":          self.frame_count,
            "left_ear":       left_ear,
            "left_ear_smooth": left_smooth,
            "left_state":     "OPEN" if left_open  else "CLOSED",
            "right_ear":      right_ear,
            "right_ear_smooth": right_smooth,
            "right_state":    "OPEN" if right_open else "CLOSED",
            "avg_ear":        avg_ear,
            "avg_ear_smooth": avg_smooth,
            "blink_count":    self.blink_count,
            "left_winks":     self.left_wink_count,
            "right_winks":    self.right_wink_count,
            "threshold":      self.ear_threshold,
        })

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def blink_count(self) -> int:
        return min(self.left_eye.blink_count, self.right_eye.blink_count)

    @property
    def blinks_per_minute(self) -> float:
        if not self._blink_times:
            return 0.0
        now    = time.monotonic_ns() // 1_000_000
        cutoff = now - FREQUENCY_WINDOW_SEC * 1000
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        elapsed_min = min(
            (now - (self._session_start_ms or now)) / 60_000,
            FREQUENCY_WINDOW_SEC / 60,
        )
        if elapsed_min < 1e-6:
            return 0.0
        return round(len(self._blink_times) / elapsed_min, 1)

    # ------------------------------------------------------------------ #

    def calculate_ear(self, eye_landmarks: list) -> float:
        """
        Compute the Eye Aspect Ratio (EAR) for one eye.

        EAR = ( ||P2-P6|| + ||P3-P5|| ) / ( 2 * ||P1-P4|| )

        Returns 0.0 on bad or degenerate input.
        """
        if len(eye_landmarks) < 6:
            return 0.0
        p1, p2, p3, p4, p5, p6 = (np.array(pt, dtype=float)
                                   for pt in eye_landmarks)
        v_a = np.linalg.norm(p2 - p6)
        v_b = np.linalg.norm(p3 - p5)
        h   = np.linalg.norm(p1 - p4)
        if h < 1e-6:
            return 0.0
        return round(float((v_a + v_b) / (2.0 * h)), 3)

    # ------------------------------------------------------------------ #

    def get_eye_landmarks(
        self, landmarks: list, indices: list,
        frame_w: int, frame_h: int,
    ) -> list:
        """Convert normalised MediaPipe landmarks to pixel (x, y) tuples."""
        return [
            (int(landmarks[i].x * frame_w),
             int(landmarks[i].y * frame_h))
            for i in indices
        ]

    # ------------------------------------------------------------------ #

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """
        Detect landmarks on one BGR frame, update state, log to CSV,
        and return the annotated frame.

        Smoothing
        ---------
        Raw EAR is passed through _EarSmoother before classification.
        The smoothed value is used for both the open/closed decision and
        the CSV log; the raw value is also recorded for reference.

        CSV columns written per frame (face detected only):
            timestamp_ms, frame, left_ear, left_ear_smooth, left_state,
            right_ear, right_ear_smooth, right_state, avg_ear,
            avg_ear_smooth, blink_count, left_winks, right_winks, threshold
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        annotated = frame.copy()

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── No face ───────────────────────────────────────────────────────
        if not result.face_landmarks:
            self._draw_banner(annotated, w, h // 2, "NO FACE DETECTED",
                              bg_color=(0, 100, 200))
            self._draw_hud(annotated, w, None, None, None, None,
                           None, None, None, None)
            return annotated

        if self._session_start_ms is None:
            self._session_start_ms = timestamp_ms

        landmarks = result.face_landmarks[0]

        # ── Mesh dots ─────────────────────────────────────────────────────
        for lm in landmarks:
            cv2.circle(annotated,
                       (int(lm.x * w), int(lm.y * h)),
                       1, (80, 80, 80), -1, cv2.LINE_AA)

        # ── Landmarks ────────────────────────────────────────────────────
        left_pts      = self.get_eye_landmarks(landmarks, LEFT_EYE_IDX,      w, h)
        right_pts     = self.get_eye_landmarks(landmarks, RIGHT_EYE_IDX,     w, h)
        left_contour  = self.get_eye_landmarks(landmarks, LEFT_EYE_CONTOUR,  w, h)
        right_contour = self.get_eye_landmarks(landmarks, RIGHT_EYE_CONTOUR, w, h)

        # ── Raw EAR ───────────────────────────────────────────────────────
        left_ear  = self.calculate_ear(left_pts)
        right_ear = self.calculate_ear(right_pts)
        avg_ear   = round((left_ear + right_ear) / 2.0, 3)

        # ── Smoothed EAR (used for classification) ────────────────────────
        left_smooth  = self._left_smooth.update(left_ear)
        right_smooth = self._right_smooth.update(right_ear)
        avg_smooth   = round((left_smooth + right_smooth) / 2.0, 3)

        left_open  = left_smooth  >= self.ear_threshold
        right_open = right_smooth >= self.ear_threshold

        # ── Per-eye state + wink/blink detection ──────────────────────────
        left_event  = self.left_eye.update(left_open,  timestamp_ms)
        right_event = self.right_eye.update(right_open, timestamp_ms)

        if left_event == "blink_start" and right_event == "blink_start":
            self._blink_times.append(timestamp_ms)
        if left_event  == "blink_start" and right_open:
            self.left_wink_count  += 1
        if right_event == "blink_start" and left_open:
            self.right_wink_count += 1

        # ── CSV row ───────────────────────────────────────────────────────
        self._log_row(
            timestamp_ms,
            left_ear,  left_smooth,  left_open,
            right_ear, right_smooth, right_open,
            avg_ear,   avg_smooth,
        )

        # ── Draw contours ─────────────────────────────────────────────────
        for contour, is_open in ((left_contour, left_open),
                                 (right_contour, right_open)):
            color = (0, 210, 90) if is_open else (0, 50, 220)
            hull  = cv2.convexHull(
                        np.array(contour, dtype=np.int32).reshape((-1, 1, 2)))
            cv2.polylines(annotated, [hull], isClosed=True,
                          color=color, thickness=2, lineType=cv2.LINE_AA)

        for pt in left_pts + right_pts:
            cv2.circle(annotated, pt, 3, (0, 220, 255), -1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────
        self._draw_hud(
            annotated, w,
            left_ear, left_smooth, left_open,
            right_ear, right_smooth, right_open,
            avg_smooth, self.ear_threshold,
        )

        # ── State banners ─────────────────────────────────────────────────
        if not left_open and right_open:
            self._draw_banner(annotated, w, h - 27, "LEFT WINK",
                              bg_color=(180, 100, 0), banner_h=54, anchor="bottom")
        elif not right_open and left_open:
            self._draw_banner(annotated, w, h - 27, "RIGHT WINK",
                              bg_color=(180, 100, 0), banner_h=54, anchor="bottom")
        elif not left_open and not right_open:
            self._draw_banner(annotated, w, h - 27, "EYES CLOSED",
                              bg_color=(0, 50, 220), banner_h=54, anchor="bottom")

        return annotated

    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Main capture loop.

        Handles both webcam and video file sources.  For a video file,
        timestamps come from CAP_PROP_POS_MSEC (position in the file)
        which is already monotonically increasing.  For a webcam,
        timestamps come from the system monotonic clock.

        Keyboard controls
        -----------------
        [        lower threshold by THRESHOLD_STEP (min 0.10)
        ]        raise threshold by THRESHOLD_STEP (max 0.40)
        Q / ESC  quit
        """
        # ── Open source ───────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            msg = (f"video file '{self.source}'"
                   if self._is_video
                   else f"camera {self.source}")
            print(f"[ERROR] Cannot open {msg}.")
            sys.exit(1)

        if not self._is_video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

        src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_label = (f"video '{self.source}'"
                     if self._is_video else f"camera {self.source}")
        print(f"[EyeTracker] {src_label}  |  "
              f"{src_w}x{src_h} @ {src_fps:.0f} fps")
        print("[EyeTracker] Controls: [ / ] adjust threshold  |  Q/ESC quit\n")

        consecutive_failures = 0
        MAX_FAILURES         = 10
        # Track last timestamp to guarantee monotonically increasing values
        # for video files where CAP_PROP_POS_MSEC can occasionally repeat
        _last_ts = -1

        try:
            while True:
                ret, frame = cap.read()

                # End of video file → clean exit (not a failure)
                if not ret and self._is_video:
                    print("[EyeTracker] End of video file.")
                    break

                if not ret:
                    consecutive_failures += 1
                    print(f"[WARNING] Frame grab failed "
                          f"({consecutive_failures}/{MAX_FAILURES}).")
                    if consecutive_failures >= MAX_FAILURES:
                        print("[ERROR] Camera disconnected. Exiting.")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0

                # ── Timestamp ─────────────────────────────────────────────
                if self._is_video:
                    ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    # Guarantee strict monotonicity (some codecs repeat timestamps)
                    if ts <= _last_ts:
                        ts = _last_ts + 1
                else:
                    ts = time.monotonic_ns() // 1_000_000
                _last_ts = ts

                # ── Process ───────────────────────────────────────────────
                if self.frame_count % FRAME_SKIP == 0:
                    annotated = self.process_frame(frame, ts)
                    self._last_annotated = annotated
                else:
                    self.frame_count += 1
                    annotated = (self._last_annotated
                                 if self._last_annotated is not None
                                 else frame)

                cv2.imshow("Eye Tracker  |  [ ] threshold  Q quit", annotated)

                # ── Keyboard ──────────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    print("\n[EyeTracker] Quit by user.")
                    break
                elif key == ord("["):
                    self.ear_threshold = round(
                        max(THRESHOLD_MIN,
                            self.ear_threshold - THRESHOLD_STEP), 2)
                    print(f"[EyeTracker] Threshold → {self.ear_threshold:.2f}")
                elif key == ord("]"):
                    self.ear_threshold = round(
                        min(THRESHOLD_MAX,
                            self.ear_threshold + THRESHOLD_STEP), 2)
                    print(f"[EyeTracker] Threshold → {self.ear_threshold:.2f}")

        except Exception as exc:
            print(f"\n[ERROR] Unexpected exception: {exc}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()
            self._close_csv()
            print(f"[EyeTracker] Session ended  |  "
                  f"frames={self.frame_count}  "
                  f"blinks={self.blink_count}  "
                  f"left_winks={self.left_wink_count}  "
                  f"right_winks={self.right_wink_count}  "
                  f"blinks/min={self.blinks_per_minute}")

    # ================================================================== #
    # Drawing helpers
    # ================================================================== #

    def _put_text(self, frame, text, pos,
                  scale=0.60, color=(255, 255, 255), thickness=2) -> None:
        x, y = pos
        cv2.putText(frame, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)

    def _draw_banner(self, frame, w, cy, text,
                     bg_color=(0, 100, 200), banner_h=60,
                     anchor="center") -> None:
        y0 = cy - banner_h if anchor == "bottom" else cy - banner_h // 2
        y1 = cy             if anchor == "bottom" else cy + banner_h // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (w, y1), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        tw     = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        text_y = y0 + (y1 - y0) // 2 + 9
        cv2.putText(frame, text, ((w - tw) // 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_hud(
        self, frame, w,
        left_ear, left_smooth, left_open,
        right_ear, right_smooth, right_open,
        avg_smooth, threshold,
    ) -> None:
        if left_ear is None:
            self._put_text(frame, "Searching for face...",
                           (10, 30), color=(180, 180, 180))
            return

        l_dur = (f"{self.left_eye.last_duration_ms:.0f}ms"
                 if self.left_eye.last_duration_ms is not None else "--")
        r_dur = (f"{self.right_eye.last_duration_ms:.0f}ms"
                 if self.right_eye.last_duration_ms is not None else "--")

        entries = [
            (f"L EAR: {left_ear:.3f} → {left_smooth:.3f}  "
             f"{'OPEN' if left_open else 'CLOSED'}",
             (0, 210, 90) if left_open  else (0, 50, 220), (10, 28)),
            (f"R EAR: {right_ear:.3f} → {right_smooth:.3f}  "
             f"{'OPEN' if right_open else 'CLOSED'}",
             (0, 210, 90) if right_open else (0, 50, 220), (10, 54)),
            (f"Avg smooth: {avg_smooth:.3f}",
             (255, 220, 0), (10, 80)),
            (f"Threshold : {threshold:.2f}  ([ / ] to adjust)",
             (200, 200, 200), (10, 106)),
            (f"Blinks: {self.blink_count}  |  {self.blinks_per_minute}/min",
             (255, 255, 255), (10, 132)),
            (f"Winks  L: {self.left_wink_count}  R: {self.right_wink_count}",
             (255, 180, 0), (10, 158)),
            (f"Dur  L: {l_dur}  R: {r_dur}",
             (200, 200, 200), (10, 184)),
            (f"Frame: {self.frame_count}",
             (120, 120, 120), (10, 210)),
        ]
        for text, color, pos in entries:
            self._put_text(frame, text, pos, color=color)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time eye tracking — webcam or video file.")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0). Ignored if --video is set.")
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a video file (mp4, avi, …). Overrides --camera.")
    parser.add_argument(
        "--threshold", type=float, default=0.21,
        help="Initial EAR threshold (default 0.21, range 0.10–0.40).")
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Disable CSV session logging.")
    parser.add_argument(
        "--smooth", type=int, default=SMOOTH_WINDOW,
        help=f"Smoothing window in frames (default {SMOOTH_WINDOW}).")
    return parser.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    source = args.video if args.video else args.camera
    EyeTracker(
        ear_threshold = args.threshold,
        source        = source,
        enable_csv    = not args.no_csv,
        smooth_window = args.smooth,
    ).run()