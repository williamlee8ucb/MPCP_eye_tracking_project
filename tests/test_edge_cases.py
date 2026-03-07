"""
tests/test_edge_cases.py
------------------------
Tests for edge-case and robustness handling.

Covers acceptance criteria:
  - System handles "no face detected" without crashing
  - Partial occlusion / degenerate landmarks don't raise exceptions
  - process_frame always returns a valid annotated frame

Run with:
    python -m pytest tests/test_edge_cases.py -v
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker, make_landmark, make_blank_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(num_faces: int = 1, ear_open: bool = True):
    """
    Build a fake FaceLandmarkerResult with `num_faces` detected faces.
    All 478 landmarks are placed at the centre of a normalised 1x1 space,
    with the 6 EAR landmarks per eye spread just enough to produce a
    realistic (open or closed) EAR.
    """
    if num_faces == 0:
        result = MagicMock()
        result.face_landmarks = []
        return result

    # Build 478 neutral landmarks at (0.5, 0.5)
    lms = [make_landmark(0.5, 0.5) for _ in range(478)]

    # Patch in 6-point eye landmarks that yield a meaningful EAR
    # Using pixel-equivalent normalised coords on a 640x480 frame:
    #   open  → height ≈ 16px / 640  = 0.025
    #   closed → height ≈  5px / 640  = 0.0078
    h_norm = 0.025 if ear_open else 0.0078
    w_norm = 0.05

    # LEFT eye indices [362, 385, 387, 263, 373, 380]
    cx, cy = 0.35, 0.40
    lms[362] = make_landmark(cx - w_norm / 2, cy)
    lms[385] = make_landmark(cx - w_norm / 4, cy - h_norm / 2)
    lms[387] = make_landmark(cx + w_norm / 4, cy - h_norm / 2)
    lms[263] = make_landmark(cx + w_norm / 2, cy)
    lms[373] = make_landmark(cx + w_norm / 4, cy + h_norm / 2)
    lms[380] = make_landmark(cx - w_norm / 4, cy + h_norm / 2)

    # RIGHT eye indices [33, 160, 158, 133, 153, 144]
    cx = 0.65
    lms[33]  = make_landmark(cx - w_norm / 2, cy)
    lms[160] = make_landmark(cx - w_norm / 4, cy - h_norm / 2)
    lms[158] = make_landmark(cx + w_norm / 4, cy - h_norm / 2)
    lms[133] = make_landmark(cx + w_norm / 2, cy)
    lms[153] = make_landmark(cx + w_norm / 4, cy + h_norm / 2)
    lms[144] = make_landmark(cx - w_norm / 4, cy + h_norm / 2)

    face_list = [lms] * num_faces
    result = MagicMock()
    result.face_landmarks = face_list
    return result


# ---------------------------------------------------------------------------
# 1.  No face detected
# ---------------------------------------------------------------------------

class TestNoFaceDetected:

    def test_returns_frame_not_none(self, tracker, blank_frame):
        """process_frame must return a frame even when no face is present."""
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert out is not None

    def test_returns_ndarray(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert isinstance(out, np.ndarray)

    def test_output_same_shape_as_input(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert out.shape == blank_frame.shape

    def test_no_exception_raised(self, tracker, blank_frame):
        """Critically: no face must never raise an exception."""
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        try:
            tracker.process_frame(blank_frame, timestamp_ms=1000)
        except Exception as exc:
            pytest.fail(f"process_frame raised unexpectedly: {exc}")

    def test_no_face_banner_drawn(self, tracker, blank_frame):
        """
        The 'NO FACE DETECTED' banner should alter the frame from the
        original (pixels should differ due to the overlay).
        """
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        out = tracker.process_frame(blank_frame.copy(), timestamp_ms=1000)
        assert not np.array_equal(out, blank_frame), (
            "Frame should be annotated with a 'no face' banner")


# ---------------------------------------------------------------------------
# 2.  Multiple faces in frame (only first is processed)
# ---------------------------------------------------------------------------

class TestMultipleFaces:

    def test_does_not_crash_with_multiple_faces(self, tracker, blank_frame):
        """num_faces=1 caps detection but simulate result with 2 to be safe."""
        tracker.landmarker.detect_for_video.return_value = _make_result(2)
        try:
            tracker.process_frame(blank_frame, timestamp_ms=1000)
        except Exception as exc:
            pytest.fail(f"Raised with 2 faces in result: {exc}")

    def test_blink_count_unchanged_with_open_eyes(self, tracker, blank_frame):
        """No blink should be counted when eyes are open."""
        tracker.landmarker.detect_for_video.return_value = (
            _make_result(1, ear_open=True))
        before = tracker.blink_count
        tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert tracker.blink_count == before


# ---------------------------------------------------------------------------
# 3.  Degenerate / bad EAR inputs
# ---------------------------------------------------------------------------

class TestDegenerateLandmarks:

    def test_empty_landmark_list(self, tracker):
        assert tracker.calculate_ear([]) == 0.0

    def test_fewer_than_six_points(self, tracker):
        assert tracker.calculate_ear([(0, 0)] * 5) == 0.0

    def test_all_points_coincident(self, tracker):
        """All 6 points at the same location → zero distances → return 0.0."""
        pts = [(50, 50)] * 6
        assert tracker.calculate_ear(pts) == 0.0

    def test_zero_horizontal_distance(self, tracker):
        """P1 == P4 → division by zero guard → return 0.0."""
        pts = [
            (100, 100),  # P1
            (100, 90),   # P2
            (100, 90),   # P3
            (100, 100),  # P4  ← same x as P1
            (100, 110),  # P5
            (100, 110),  # P6
        ]
        assert tracker.calculate_ear(pts) == 0.0

    def test_negative_coordinates(self, tracker):
        """Negative pixel coords (e.g. off-frame) should not raise."""
        pts = [
            (-60,   0),
            (-30, -10),
            ( 30, -10),
            ( 60,   0),
            ( 30,  10),
            (-30,  10),
        ]
        try:
            ear = tracker.calculate_ear(pts)
        except Exception as exc:
            pytest.fail(f"Raised on negative coords: {exc}")
        assert isinstance(ear, float)

    def test_very_large_coordinates(self, tracker):
        """Large pixel values (e.g. 4K resolution) should not overflow."""
        scale = 3840
        pts = [
            (0,        scale // 2),
            (scale // 4, 0),
            (scale * 3 // 4, 0),
            (scale,    scale // 2),
            (scale * 3 // 4, scale),
            (scale // 4, scale),
        ]
        ear = tracker.calculate_ear(pts)
        assert 0.0 <= ear <= 2.0


# ---------------------------------------------------------------------------
# 4.  process_frame robustness
# ---------------------------------------------------------------------------

class TestProcessFrameRobustness:

    def test_open_eyes_returns_annotated_frame(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = (
            _make_result(1, ear_open=True))
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert out.shape == blank_frame.shape

    def test_closed_eyes_returns_annotated_frame(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = (
            _make_result(1, ear_open=False))
        out = tracker.process_frame(blank_frame, timestamp_ms=2000)
        assert out.shape == blank_frame.shape

    def test_frame_counter_increments(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        before = tracker.frame_count
        tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert tracker.frame_count == before + 1

    def test_original_frame_not_mutated(self, tracker, blank_frame):
        """process_frame should work on a copy and leave the input intact."""
        original = blank_frame.copy()
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert np.array_equal(blank_frame, original)

    def test_different_frame_sizes_accepted(self, tracker):
        """Tracker should handle non-standard resolutions without crashing."""
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        for h, w in [(240, 320), (720, 1280), (1080, 1920)]:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            try:
                tracker.process_frame(frame, timestamp_ms=1000)
            except Exception as exc:
                pytest.fail(f"Raised for {w}x{h} frame: {exc}")

    def test_consecutive_frames_with_increasing_timestamps(self, tracker, blank_frame):
        """Monotonically increasing timestamps must not raise."""
        tracker.landmarker.detect_for_video.return_value = _make_result(0)
        for ts in range(1000, 1500, 33):   # ~30 fps
            try:
                tracker.process_frame(blank_frame, timestamp_ms=ts)
            except Exception as exc:
                pytest.fail(f"Raised at timestamp {ts}: {exc}")


# ---------------------------------------------------------------------------
# 5.  get_eye_landmarks coordinate conversion
# ---------------------------------------------------------------------------

class TestGetEyeLandmarks:

    def _make_landmarks(self, n: int = 478, x: float = 0.5, y: float = 0.5):
        return [make_landmark(x, y) for _ in range(n)]

    def test_normalised_to_pixel_conversion(self, tracker):
        """x=0.5, y=0.5 on a 640x480 frame → (320, 240)."""
        lms  = self._make_landmarks(x=0.5, y=0.5)
        pts  = tracker.get_eye_landmarks(lms, [0, 1, 2], 640, 480)
        assert pts[0] == (320, 240)

    def test_top_left_normalised_is_pixel_zero(self, tracker):
        lms = self._make_landmarks(x=0.0, y=0.0)
        pts = tracker.get_eye_landmarks(lms, [0], 640, 480)
        assert pts[0] == (0, 0)

    def test_bottom_right_normalised_is_frame_size(self, tracker):
        lms = self._make_landmarks(x=1.0, y=1.0)
        pts = tracker.get_eye_landmarks(lms, [0], 640, 480)
        assert pts[0] == (640, 480)

    def test_returns_correct_count(self, tracker):
        from eye_tracker import LEFT_EYE_IDX
        lms = self._make_landmarks()
        pts = tracker.get_eye_landmarks(lms, LEFT_EYE_IDX, 640, 480)
        assert len(pts) == len(LEFT_EYE_IDX)

    def test_returns_int_tuples(self, tracker):
        lms = self._make_landmarks(x=0.33, y=0.67)
        pts = tracker.get_eye_landmarks(lms, [0], 640, 480)
        x, y = pts[0]
        assert isinstance(x, int)
        assert isinstance(y, int)
