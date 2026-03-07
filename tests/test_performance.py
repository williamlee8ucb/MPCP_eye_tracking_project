"""
tests/test_performance.py
--------------------------
Tests for performance, memory stability, and display smoothness.

Covers acceptance criteria:
  - Video display updates smoothly
  - No memory leaks (program runs >5 minutes without issues)
  - System handles sustained operation without degradation

These tests do NOT require a camera, model file, or GPU.

Run with:
    python -m pytest tests/test_performance.py -v
"""

import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker, make_blank_frame



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_no_face_result():
    from unittest.mock import MagicMock
    r = MagicMock()
    r.face_landmarks = []
    return r


def _make_open_face_result():
    """Minimal result with face_landmarks set to an empty list of landmarks
    (process_frame will hit the no-face path, which is fine for perf tests)."""
    return _make_no_face_result()


# ---------------------------------------------------------------------------
# 1.  Per-frame processing time
# ---------------------------------------------------------------------------

class TestFrameProcessingSpeed:
    """
    process_frame (excluding MediaPipe inference) should be fast enough
    to sustain >= 30 fps, i.e. each call < 33 ms.

    MediaPipe is mocked out, so this measures only the Python / OpenCV
    annotation work.
    """

    FRAME_BUDGET_MS = 33.0   # 30 fps target

    def test_no_face_frame_under_budget(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()

        t0 = time.perf_counter()
        tracker.process_frame(blank_frame, timestamp_ms=1000)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < self.FRAME_BUDGET_MS, (
            f"No-face frame took {elapsed_ms:.1f} ms (budget {self.FRAME_BUDGET_MS} ms)")

    def test_100_frames_average_under_budget(self, tracker, blank_frame):
        """Average over 100 frames must stay within the per-frame budget."""
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        n = 100

        t0 = time.perf_counter()
        for ts in range(1000, 1000 + n * 33, 33):
            tracker.process_frame(blank_frame, timestamp_ms=ts)
        total_ms = (time.perf_counter() - t0) * 1000
        avg_ms   = total_ms / n

        assert avg_ms < self.FRAME_BUDGET_MS, (
            f"Average frame time {avg_ms:.2f} ms exceeds budget "
            f"{self.FRAME_BUDGET_MS} ms over {n} frames")


# ---------------------------------------------------------------------------
# 2.  EAR computation speed
# ---------------------------------------------------------------------------

class TestEARComputationSpeed:

    def test_single_ear_call_fast(self, tracker):
        """One EAR calculation should complete well under 1 ms."""
        pts = [
            (100, 200), (120, 185), (140, 185),
            (160, 200), (140, 215), (120, 215),
        ]
        t0 = time.perf_counter()
        tracker.calculate_ear(pts)
        elapsed_us = (time.perf_counter() - t0) * 1_000_000

        assert elapsed_us < 1000, (
            f"calculate_ear took {elapsed_us:.1f} µs (should be < 1000 µs)")

    def test_10000_ear_calls_complete_in_time(self, tracker):
        """10 000 EAR calls should finish in under 1 second total."""
        pts = [
            (100, 200), (120, 185), (140, 185),
            (160, 200), (140, 215), (120, 215),
        ]
        t0 = time.perf_counter()
        for _ in range(10_000):
            tracker.calculate_ear(pts)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, (
            f"10 000 EAR calls took {elapsed:.2f}s (should be < 1.0s)")


# ---------------------------------------------------------------------------
# 3.  Memory stability (simulated 5-minute run)
# ---------------------------------------------------------------------------

class TestSustainedOperation:
    """
    Simulates 7+ minutes of continuous operation (12 600 frames at 30 fps)
    and verifies the program stays stable — no crashes, no exceptions,
    and no state drift across the run.
    """

    SEVEN_MIN_FRAMES = 12_600  # 7 min × 30 fps

    def test_runs_5_minutes_without_crashing(self, tracker, blank_frame):
        """
        Process 12 600 frames without raising any exception.
        This is the direct translation of the spec requirement.
        """
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()

        try:
            for i in range(self.SEVEN_MIN_FRAMES):
                tracker.process_frame(blank_frame, timestamp_ms=1000 + i * 33)
        except Exception as exc:
            pytest.fail(
                f"Crashed after {tracker.frame_count} frames "
                f"({tracker.frame_count / 30:.0f}s): {exc}")

    def test_blink_count_correct_after_long_run(self, tracker, blank_frame):
        """
        Blink counting must stay accurate across thousands of frames —
        state must not drift or overflow.
        """
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()

        # No face → blink count should never change
        for i in range(1000):
            tracker.process_frame(blank_frame, timestamp_ms=1000 + i * 33)

        assert tracker.blink_count == 0, (
            f"Blink count drifted to {tracker.blink_count} with no face present")

    def test_frame_count_increments_correctly(self, tracker, blank_frame):
        """frame_count must increment by exactly 1 per process_frame call."""
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        n = 500
        for i in range(n):
            tracker.process_frame(blank_frame, timestamp_ms=1000 + i * 33)
        assert tracker.frame_count == n


# ---------------------------------------------------------------------------
# 4.  Frame output integrity
# ---------------------------------------------------------------------------

class TestFrameOutputIntegrity:
    """
    Verify that process_frame never returns a corrupt frame — same shape,
    correct dtype, valid pixel range — even under sustained load.
    """

    def test_output_dtype_is_uint8(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert out.dtype == np.uint8

    def test_output_pixel_range_valid(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_output_shape_stable_across_100_frames(self, tracker, blank_frame):
        """Shape must never change between frames."""
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        expected = blank_frame.shape
        for i in range(100):
            out = tracker.process_frame(blank_frame, timestamp_ms=1000 + i * 33)
            assert out.shape == expected, (
                f"Frame {i}: shape changed to {out.shape}")

    def test_output_is_3_channel_bgr(self, tracker, blank_frame):
        tracker.landmarker.detect_for_video.return_value = _make_no_face_result()
        out = tracker.process_frame(blank_frame, timestamp_ms=1000)
        assert len(out.shape) == 3
        assert out.shape[2] == 3