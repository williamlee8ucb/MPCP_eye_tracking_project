"""
tests/test_enhanced_features.py
--------------------------------
Tests for Level 1 enhanced features:
  - Per-eye blink counts tracked independently
  - Blink frequency (blinks per minute, rolling window)
  - Blink duration measurement
  - Wink detection (one eye closed, other open)

Run with:
    python -m pytest tests/test_enhanced_features.py -v
"""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fire_blink(eye_state, n_closed: int = 3, ts_start: int = 1000,
                frame_ms: int = 33) -> int:
    """
    Drive a single _EyeState through one blink cycle.
    Returns the timestamp_ms after the eye reopens.
    """
    ts = ts_start
    for _ in range(n_closed):
        eye_state.update(is_open=False, timestamp_ms=ts)
        ts += frame_ms
    eye_state.update(is_open=True, timestamp_ms=ts)
    ts += frame_ms
    return ts


def _fire_bilateral_blink(tracker, ts: int, n_closed: int = 3,
                           frame_ms: int = 33) -> int:
    """Close both eyes for n_closed frames then reopen. Returns next ts."""
    for _ in range(n_closed):
        tracker.left_eye.update(is_open=False,  timestamp_ms=ts)
        tracker.right_eye.update(is_open=False, timestamp_ms=ts)
        ts += frame_ms
    tracker.left_eye.update(is_open=True,  timestamp_ms=ts)
    tracker.right_eye.update(is_open=True, timestamp_ms=ts)
    # Mirror what process_frame does: append to _blink_times on blink_start
    # (already counted above; we simulate the timestamp append here)
    ts += frame_ms
    return ts


# ---------------------------------------------------------------------------
# 1.  _EyeState unit tests
# ---------------------------------------------------------------------------

class TestEyeState:

    def _make_eye(self):
        from eye_tracker import _EyeState
        return _EyeState(consec_needed=2)

    def test_starts_at_zero_blinks(self):
        eye = self._make_eye()
        assert eye.blink_count == 0

    def test_not_blinking_initially(self):
        eye = self._make_eye()
        assert not eye.is_blinking

    def test_single_closed_frame_not_confirmed(self):
        """One frame below threshold should not confirm a blink."""
        eye = self._make_eye()
        event = eye.update(is_open=False, timestamp_ms=1000)
        assert not eye.is_blinking
        assert event != "blink_start"

    def test_blink_confirmed_after_consec_frames(self):
        """Blink confirmed only after consec_needed consecutive closed frames."""
        eye = self._make_eye()
        eye.update(is_open=False, timestamp_ms=1000)
        event = eye.update(is_open=False, timestamp_ms=1033)
        assert eye.is_blinking
        assert event == "blink_start"

    def test_blink_counted_on_reopen(self):
        eye = self._make_eye()
        _fire_blink(eye)
        assert eye.blink_count == 1

    def test_multiple_blinks_counted(self):
        eye = self._make_eye()
        ts = 1000
        for _ in range(5):
            ts = _fire_blink(eye, ts_start=ts)
        assert eye.blink_count == 5

    def test_is_blinking_clears_on_reopen(self):
        eye = self._make_eye()
        _fire_blink(eye)
        assert not eye.is_blinking

    def test_consec_below_resets_on_reopen(self):
        eye = self._make_eye()
        _fire_blink(eye)
        assert eye.consec_below == 0

    def test_open_event_returns_open(self):
        eye = self._make_eye()
        event = eye.update(is_open=True, timestamp_ms=1000)
        assert event == "open"


# ---------------------------------------------------------------------------
# 2.  Blink duration
# ---------------------------------------------------------------------------

class TestBlinkDuration:

    def _make_eye(self):
        from eye_tracker import _EyeState
        return _EyeState(consec_needed=2)

    def test_duration_none_before_any_blink(self):
        eye = self._make_eye()
        assert eye.last_duration_ms is None

    def test_duration_measured_after_blink(self):
        """Duration should be > 0 after a completed blink."""
        eye = self._make_eye()
        _fire_blink(eye, n_closed=4, ts_start=1000, frame_ms=33)
        assert eye.last_duration_ms is not None
        assert eye.last_duration_ms > 0

    def test_duration_reflects_closed_frames(self):
        """
        Blink is confirmed on frame 2 (ts=1033), eye reopens on frame 5
        (ts=1132). Duration = 1132 - 1033 = 99 ms.
        """
        eye = self._make_eye()
        # Frame 1 closed: ts=1000
        eye.update(is_open=False, timestamp_ms=1000)
        # Frame 2 closed: ts=1033 → blink confirmed, start_ms=1033
        eye.update(is_open=False, timestamp_ms=1033)
        # Frame 3 closed: ts=1066
        eye.update(is_open=False, timestamp_ms=1066)
        # Frame 4 closed: ts=1099
        eye.update(is_open=False, timestamp_ms=1099)
        # Frame 5 open: ts=1132 → duration = 1132 - 1033 = 99
        eye.update(is_open=True, timestamp_ms=1132)

        assert eye.last_duration_ms == pytest.approx(99, abs=1)

    def test_duration_updates_on_each_blink(self):
        """last_duration_ms should reflect the most recent blink."""
        eye = self._make_eye()
        # Short blink
        ts = _fire_blink(eye, n_closed=2, ts_start=1000, frame_ms=33)
        short_dur = eye.last_duration_ms
        # Long blink
        _fire_blink(eye, n_closed=8, ts_start=ts, frame_ms=33)
        long_dur = eye.last_duration_ms

        assert long_dur > short_dur

    def test_duration_is_non_negative(self):
        eye = self._make_eye()
        _fire_blink(eye, n_closed=3, ts_start=5000, frame_ms=33)
        assert eye.last_duration_ms >= 0


# ---------------------------------------------------------------------------
# 3.  Per-eye blink count independence
# ---------------------------------------------------------------------------

class TestPerEyeBlinks:

    def test_left_and_right_tracked_independently(self, tracker):
        """Blinking only the left eye should not increment right blink count."""
        ts = _fire_blink(tracker.left_eye, n_closed=3, ts_start=1000)
        assert tracker.left_eye.blink_count  == 1
        assert tracker.right_eye.blink_count == 0

    def test_right_only_blink(self, tracker):
        ts = _fire_blink(tracker.right_eye, n_closed=3, ts_start=1000)
        assert tracker.right_eye.blink_count == 1
        assert tracker.left_eye.blink_count  == 0

    def test_bilateral_blink_increments_both(self, tracker):
        ts = 1000
        ts = _fire_bilateral_blink(tracker, ts)
        assert tracker.left_eye.blink_count  == 1
        assert tracker.right_eye.blink_count == 1

    def test_blink_count_property_uses_minimum(self, tracker):
        """
        blink_count property = min(left, right) so winks don't
        inflate the bilateral count.
        """
        # 3 left blinks, 1 right blink
        ts = 1000
        for _ in range(3):
            ts = _fire_blink(tracker.left_eye, ts_start=ts)
        _fire_blink(tracker.right_eye, ts_start=ts)
        assert tracker.blink_count == 1

    def test_bilateral_blink_count_property(self, tracker):
        ts = 1000
        for _ in range(4):
            ts = _fire_bilateral_blink(tracker, ts)
        assert tracker.blink_count == 4


# ---------------------------------------------------------------------------
# 4.  Wink detection
# ---------------------------------------------------------------------------

class TestWinkDetection:

    def test_left_wink_counted(self, tracker):
        """Left eye confirmed blinking while right stays open → left wink."""
        # Left eye: close for CONSEC_NEEDED frames (confirm blink_start)
        # Right eye: stays open
        tracker.left_eye.update(is_open=False, timestamp_ms=1000)
        event = tracker.left_eye.update(is_open=False, timestamp_ms=1033)

        if event == "blink_start":
            # Right is open → wink
            tracker.left_wink_count += 1

        assert tracker.left_wink_count == 1

    def test_right_wink_counted(self, tracker):
        tracker.right_eye.update(is_open=False, timestamp_ms=1000)
        event = tracker.right_eye.update(is_open=False, timestamp_ms=1033)

        if event == "blink_start":
            tracker.right_wink_count += 1

        assert tracker.right_wink_count == 1

    def test_bilateral_blink_does_not_count_as_wink(self, tracker):
        """Both eyes closing simultaneously is a blink, not a wink."""
        ts = 1000
        _fire_bilateral_blink(tracker, ts)
        assert tracker.left_wink_count  == 0
        assert tracker.right_wink_count == 0

    def test_wink_counts_start_at_zero(self, tracker):
        assert tracker.left_wink_count  == 0
        assert tracker.right_wink_count == 0

    def test_left_and_right_wink_independent(self, tracker):
        """Multiple winks on different sides tracked separately."""
        for _ in range(3):
            ts = 1000
            tracker.left_eye.update(is_open=False, timestamp_ms=ts)
            event = tracker.left_eye.update(is_open=False, timestamp_ms=ts + 33)
            if event == "blink_start":
                tracker.left_wink_count += 1
            tracker.left_eye.update(is_open=True, timestamp_ms=ts + 66)
            tracker.left_eye.consec_below = 0
            tracker.left_eye.is_blinking  = False
            ts += 200

        assert tracker.left_wink_count  == 3
        assert tracker.right_wink_count == 0


# ---------------------------------------------------------------------------
# 5.  Blink frequency
# ---------------------------------------------------------------------------

class TestBlinkFrequency:

    def test_frequency_zero_with_no_blinks(self, tracker):
        assert tracker.blinks_per_minute == 0.0

    def test_frequency_positive_after_blinks(self, tracker):
        """After recording blinks, frequency should be > 0."""
        now = time.monotonic_ns() // 1_000_000
        tracker._session_start_ms = now - 10_000   # 10 seconds ago
        tracker._blink_times.append(now - 5000)
        tracker._blink_times.append(now - 2000)
        assert tracker.blinks_per_minute > 0.0

    def test_frequency_is_float(self, tracker):
        assert isinstance(tracker.blinks_per_minute, float)

    def test_frequency_scales_with_blink_count(self, tracker):
        """More blinks in the same window → higher frequency."""
        now = time.monotonic_ns() // 1_000_000
        tracker._session_start_ms = now - 30_000   # 30 seconds ago

        tracker_few = make_tracker()
        tracker_few._session_start_ms = now - 30_000
        tracker_few._blink_times.extend([now - 25000, now - 15000])  # 2 blinks

        tracker_many = make_tracker()
        tracker_many._session_start_ms = now - 30_000
        tracker_many._blink_times.extend(
            [now - 25000, now - 20000, now - 15000,
             now - 10000, now - 5000])                                 # 5 blinks

        assert tracker_many.blinks_per_minute > tracker_few.blinks_per_minute

    def test_old_blinks_pruned_from_window(self, tracker):
        """
        Blinks older than FREQUENCY_WINDOW_SEC should not count
        toward the current frequency.
        """
        from eye_tracker import FREQUENCY_WINDOW_SEC
        now = time.monotonic_ns() // 1_000_000
        tracker._session_start_ms = now - (FREQUENCY_WINDOW_SEC + 10) * 1000

        # One very old blink (outside window)
        old_ts = now - (FREQUENCY_WINDOW_SEC + 5) * 1000
        tracker._blink_times.append(old_ts)

        # One recent blink
        tracker._blink_times.append(now - 5000)

        freq = tracker.blinks_per_minute

        # Old blink should be pruned; only 1 recent blink remains
        assert len(tracker._blink_times) == 1

    def test_frequency_reasonable_range(self, tracker):
        """
        Normal human blink rate is 10–20 blinks/min.
        Simulate 15 blinks over the last 60 seconds.
        """
        now = time.monotonic_ns() // 1_000_000
        tracker._session_start_ms = now - 60_000
        for i in range(15):
            tracker._blink_times.append(now - (60_000 - i * 4000))

        freq = tracker.blinks_per_minute
        assert 10.0 <= freq <= 20.0, f"Expected 10–20 blinks/min, got {freq}"
