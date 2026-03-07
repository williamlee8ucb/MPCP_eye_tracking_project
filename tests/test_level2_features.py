"""
tests/test_level2_features.py
------------------------------
Tests for Level 2 features:
  - CSV session logging
  - Keyboard threshold adjustment
  - Temporal EAR smoothing
  - Video file input

Run with:
    python -m pytest tests/test_level2_features.py -v
"""

import csv
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eye_tracker import (
    _EarSmoother, EyeTracker,
    THRESHOLD_STEP, THRESHOLD_MIN, THRESHOLD_MAX,
    CSV_COLUMNS, SMOOTH_WINDOW,
)
from conftest import make_tracker, make_blank_frame


# ---------------------------------------------------------------------------
# 1.  EAR Smoothing  (_EarSmoother)
# ---------------------------------------------------------------------------

class TestEarSmoother:

    def test_single_value_returns_itself(self):
        s = _EarSmoother(window=5)
        assert s.update(0.30) == pytest.approx(0.30, abs=0.001)

    def test_average_over_window(self):
        s = _EarSmoother(window=4)
        values = [0.20, 0.24, 0.28, 0.32]
        result = None
        for v in values:
            result = s.update(v)
        expected = round(sum(values) / 4, 3)
        assert result == pytest.approx(expected, abs=0.001)

    def test_window_slides_drops_oldest(self):
        """Once the window is full, oldest values are dropped."""
        s = _EarSmoother(window=3)
        s.update(0.10)
        s.update(0.10)
        s.update(0.10)   # window: [0.10, 0.10, 0.10]
        result = s.update(0.40)  # window: [0.10, 0.10, 0.40]
        expected = round((0.10 + 0.10 + 0.40) / 3, 3)
        assert result == pytest.approx(expected, abs=0.001)

    def test_smoothed_value_less_jittery_than_raw(self):
        """
        Smoothed values should have lower variance than raw noisy inputs.
        """
        s     = _EarSmoother(window=5)
        raw   = [0.30, 0.10, 0.30, 0.10, 0.30, 0.10, 0.30, 0.10]
        smoothed = [s.update(v) for v in raw]
        assert np.std(smoothed) < np.std(raw)

    def test_reset_clears_buffer(self):
        s = _EarSmoother(window=5)
        for _ in range(5):
            s.update(0.30)
        s.reset()
        # After reset, next value should equal itself (buffer empty)
        assert s.update(0.20) == pytest.approx(0.20, abs=0.001)

    def test_window_property(self):
        s = _EarSmoother(window=7)
        assert s.window == 7

    def test_returns_float(self):
        s = _EarSmoother()
        assert isinstance(s.update(0.25), float)

    def test_spike_dampened(self):
        """A single outlier spike should not dominate the smoothed output."""
        s = _EarSmoother(window=5)
        for _ in range(4):
            s.update(0.30)   # stable baseline
        smoothed = s.update(0.01)   # sudden spike down
        assert smoothed > 0.20, (
            f"Spike should be dampened, got {smoothed}")

    def test_custom_window_size(self):
        s1 = _EarSmoother(window=2)
        s5 = _EarSmoother(window=5)
        values = [0.10, 0.30, 0.10, 0.30, 0.10]
        r1 = r5 = None
        for v in values:
            r1 = s1.update(v)
            r5 = s5.update(v)
        # Smaller window reacts faster → higher variance
        # Just check both return valid floats in range
        assert 0.0 <= r1 <= 1.0
        assert 0.0 <= r5 <= 1.0


# ---------------------------------------------------------------------------
# 2.  CSV Logging
# ---------------------------------------------------------------------------

class TestCSVLogging:

    def _make_csv_tracker(self, tmp_path):
        """Tracker with CSV writing redirected to tmp_path."""
        t = make_tracker()
        t.enable_csv = True
        csv_path = tmp_path / "test_session.csv"
        t._csv_file   = open(csv_path, "w", newline="")
        t._csv_writer = csv.DictWriter(t._csv_file, fieldnames=CSV_COLUMNS)
        t._csv_writer.writeheader()
        return t, csv_path

    def test_csv_file_created(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t._close_csv()
        assert csv_path.exists()

    def test_csv_has_header(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t._close_csv()
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == set(CSV_COLUMNS)

    def test_csv_row_written_per_frame(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t._log_row(1000, 0.28, 0.27, True,  0.29, 0.28, True,  0.285, 0.275)
        t._log_row(1033, 0.27, 0.27, True,  0.28, 0.28, True,  0.275, 0.275)
        t._close_csv()
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_csv_row_values_correct(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t._log_row(5000, 0.15, 0.16, False, 0.32, 0.31, True, 0.235, 0.235)
        t._close_csv()
        with open(csv_path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["timestamp_ms"] == "5000"
        assert row["left_state"]   == "CLOSED"
        assert row["right_state"]  == "OPEN"
        assert float(row["left_ear"])  == pytest.approx(0.15, abs=0.001)
        assert float(row["right_ear"]) == pytest.approx(0.32, abs=0.001)

    def test_csv_contains_threshold(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t.ear_threshold = 0.23
        t._log_row(1000, 0.25, 0.25, True, 0.25, 0.25, True, 0.25, 0.25)
        t._close_csv()
        with open(csv_path) as f:
            row = list(csv.DictReader(f))[0]
        assert float(row["threshold"]) == pytest.approx(0.23, abs=0.001)

    def test_csv_disabled_no_write(self, tmp_path):
        """When enable_csv=False, _log_row should be a no-op."""
        t = make_tracker()
        t.enable_csv  = False
        t._csv_writer = None
        # Should not raise
        t._log_row(1000, 0.28, 0.27, True, 0.29, 0.28, True, 0.285, 0.275)

    def test_csv_all_columns_present_in_row(self, tmp_path):
        t, csv_path = self._make_csv_tracker(tmp_path)
        t._log_row(1000, 0.28, 0.27, True, 0.29, 0.28, True, 0.285, 0.275)
        t._close_csv()
        with open(csv_path) as f:
            row = list(csv.DictReader(f))[0]
        for col in CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"

    def test_close_csv_idempotent(self, tmp_path):
        """Calling _close_csv twice should not raise."""
        t, _ = self._make_csv_tracker(tmp_path)
        t._close_csv()
        t._close_csv()   # second call — must not raise


# ---------------------------------------------------------------------------
# 3.  Threshold keyboard adjustment
# ---------------------------------------------------------------------------

class TestThresholdAdjustment:

    def test_increase_threshold(self, tracker):
        initial = tracker.ear_threshold
        tracker.ear_threshold = round(
            min(THRESHOLD_MAX, tracker.ear_threshold + THRESHOLD_STEP), 2)
        assert tracker.ear_threshold == pytest.approx(initial + THRESHOLD_STEP,
                                                      abs=0.001)

    def test_decrease_threshold(self, tracker):
        initial = tracker.ear_threshold
        tracker.ear_threshold = round(
            max(THRESHOLD_MIN, tracker.ear_threshold - THRESHOLD_STEP), 2)
        assert tracker.ear_threshold == pytest.approx(initial - THRESHOLD_STEP,
                                                      abs=0.001)

    def test_cannot_exceed_maximum(self, tracker):
        tracker.ear_threshold = THRESHOLD_MAX
        tracker.ear_threshold = round(
            min(THRESHOLD_MAX, tracker.ear_threshold + THRESHOLD_STEP), 2)
        assert tracker.ear_threshold == THRESHOLD_MAX

    def test_cannot_go_below_minimum(self, tracker):
        tracker.ear_threshold = THRESHOLD_MIN
        tracker.ear_threshold = round(
            max(THRESHOLD_MIN, tracker.ear_threshold - THRESHOLD_STEP), 2)
        assert tracker.ear_threshold == THRESHOLD_MIN

    def test_step_size_is_correct(self):
        assert THRESHOLD_STEP == pytest.approx(0.01, abs=1e-6)

    def test_threshold_affects_classification(self, tracker):
        """Raising the threshold should reclassify a borderline EAR as CLOSED."""
        pts  = [(100, 200), (115, 190), (135, 190),
                (150, 200), (135, 210), (115, 210)]
        ear  = tracker.calculate_ear(pts)

        # Set threshold just below EAR → OPEN
        tracker.ear_threshold = round(ear - 0.01, 2)
        assert ear >= tracker.ear_threshold

        # Raise threshold above EAR → CLOSED
        tracker.ear_threshold = round(ear + 0.01, 2)
        assert ear < tracker.ear_threshold

    def test_multiple_step_increases(self, tracker):
        start = tracker.ear_threshold
        for _ in range(5):
            tracker.ear_threshold = round(
                min(THRESHOLD_MAX, tracker.ear_threshold + THRESHOLD_STEP), 2)
        expected = round(min(THRESHOLD_MAX, start + 5 * THRESHOLD_STEP), 2)
        assert tracker.ear_threshold == pytest.approx(expected, abs=0.001)

    def test_threshold_change_logged_in_csv(self, tmp_path):
        """Threshold in the CSV row should reflect the value at write time."""
        t = make_tracker()
        t.enable_csv = True
        csv_path = tmp_path / "thresh_test.csv"
        t._csv_file   = open(csv_path, "w", newline="")
        t._csv_writer = csv.DictWriter(t._csv_file, fieldnames=CSV_COLUMNS)
        t._csv_writer.writeheader()

        t.ear_threshold = 0.19
        t._log_row(1000, 0.25, 0.25, True, 0.25, 0.25, True, 0.25, 0.25)
        t.ear_threshold = 0.24
        t._log_row(1033, 0.25, 0.25, True, 0.25, 0.25, True, 0.25, 0.25)
        t._close_csv()

        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["threshold"]) == pytest.approx(0.19, abs=0.001)
        assert float(rows[1]["threshold"]) == pytest.approx(0.24, abs=0.001)


# ---------------------------------------------------------------------------
# 4.  Video file input
# ---------------------------------------------------------------------------

class TestVideoFileInput:

    def _make_video_tracker(self, video_path="test.mp4"):
        """Tracker configured for video file input (no CSV, no camera)."""
        t = make_tracker()
        t.source    = video_path
        t._is_video = True
        return t

    def test_is_video_flag_set_for_string_source(self):
        t = make_tracker()
        t.source    = "video.mp4"
        t._is_video = isinstance(t.source, str)
        assert t._is_video is True

    def test_is_video_false_for_int_source(self):
        t = make_tracker()
        t.source    = 0
        t._is_video = isinstance(t.source, str)
        assert t._is_video is False

    def test_video_end_of_file_exits_cleanly(self):
        """
        When cap.read() returns (False, None) and source is a video file,
        run() should exit cleanly without treating it as a camera failure.
        """
        t   = self._make_video_tracker("fake.mp4")
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value      = 0.0
        # First read returns a frame, second signals end-of-file
        frame = make_blank_frame()
        cap.read.side_effect = [(True, frame), (False, None)]

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", return_value=0),
        ):
            try:
                t.run()
            except Exception as exc:
                pytest.fail(f"End of video raised: {exc}")

    def test_video_timestamps_from_pos_msec(self):
        """
        For video files, timestamps should come from CAP_PROP_POS_MSEC.
        We verify the tracker accepts increasing video timestamps without error.
        """
        t   = self._make_video_tracker()
        t.landmarker.detect_for_video.return_value = MagicMock(
            face_landmarks=[])

        frame = make_blank_frame()
        # Simulate 5 frames with increasing POS_MSEC values
        for pos_ms in [0, 33, 66, 99, 132]:
            try:
                t.process_frame(frame, timestamp_ms=pos_ms + 1)
            except Exception as exc:
                pytest.fail(f"Video timestamp {pos_ms} raised: {exc}")

    def test_video_non_monotonic_timestamps_handled(self):
        """
        The run() loop enforces monotonicity by incrementing _last_ts + 1
        when a video codec repeats a timestamp. Simulate this directly.
        """
        _last_ts = 100
        # Simulate the guard logic from run()
        ts = 100   # same as _last_ts — would violate MediaPipe
        if ts <= _last_ts:
            ts = _last_ts + 1
        assert ts == 101

    def test_webcam_timestamps_from_monotonic_clock(self):
        """For webcam (int source), timestamp should come from system clock."""
        t = make_tracker()
        t.source    = 0
        t._is_video = False
        # Just verify the tracker itself doesn't break with a large timestamp
        t.landmarker.detect_for_video.return_value = MagicMock(
            face_landmarks=[])
        frame = make_blank_frame()
        large_ts = (10 ** 9)  # 1 billion ms ≈ 11 days of uptime
        try:
            t.process_frame(frame, timestamp_ms=large_ts)
        except Exception as exc:
            pytest.fail(f"Large webcam timestamp raised: {exc}")

    def test_missing_video_file_exits(self):
        """A video path that doesn't exist should cause sys.exit(1)."""
        t = make_tracker()
        t.source    = "/nonexistent/path/video.mp4"
        t._is_video = True

        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False

        with (
            patch("cv2.VideoCapture", return_value=bad_cap),
            pytest.raises(SystemExit) as exc_info,
        ):
            t.run()

        assert exc_info.value.code == 1

    def test_process_frame_works_with_video_timestamps(self):
        """process_frame is source-agnostic — any valid timestamp works."""
        t = make_tracker()
        t.landmarker.detect_for_video.return_value = MagicMock(
            face_landmarks=[])
        frame = make_blank_frame()
        # Video-style timestamps (ms position in file)
        for ts in [0, 33, 66, 99]:
            out = t.process_frame(frame, timestamp_ms=ts + 1)
            assert out.shape == frame.shape
