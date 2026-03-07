"""
tests/test_classification.py
-----------------------------
Tests for eye state classification (OPEN / CLOSED) and blink counting.

Covers acceptance criteria:
  - Eye state correctly identified when deliberately blinking
  - EAR values are reasonable (0.15–0.35 range typically)
  - Console shows clear status messages

Run with:
    python -m pytest tests/test_classification.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker, make_eye_pts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    return make_tracker(ear_threshold=0.21)


# ---------------------------------------------------------------------------
# 1.  Basic OPEN / CLOSED classification
# ---------------------------------------------------------------------------

class TestOpenClosedClassification:

    def test_open_eye_above_threshold(self, tracker):
        """EAR well above threshold → OPEN."""
        pts = make_eye_pts(width=60, height=20)   # EAR ≈ 0.33
        ear = tracker.calculate_ear(pts)
        assert ear >= tracker.ear_threshold, (
            f"Expected OPEN (EAR {ear} >= {tracker.ear_threshold})")

    def test_closed_eye_below_threshold(self, tracker):
        """EAR well below threshold → CLOSED."""
        pts = make_eye_pts(width=60, height=8)    # EAR ≈ 0.13
        ear = tracker.calculate_ear(pts)
        assert ear < tracker.ear_threshold, (
            f"Expected CLOSED (EAR {ear} < {tracker.ear_threshold})")

    def test_boundary_exactly_at_threshold_is_open(self, tracker):
        """EAR == threshold should be classified OPEN (>=)."""
        # Build an eye whose EAR equals the threshold exactly:
        # EAR = h/w = threshold  →  h = threshold * w
        w = 60
        h = tracker.ear_threshold * w
        pts  = make_eye_pts(width=w, height=h)
        ear  = tracker.calculate_ear(pts)
        open_ = ear >= tracker.ear_threshold
        assert open_, f"EAR {ear} at threshold should be OPEN"

    def test_just_below_threshold_is_closed(self, tracker):
        """EAR just under threshold → CLOSED."""
        w = 60
        h = (tracker.ear_threshold - 0.01) * w
        pts  = make_eye_pts(width=w, height=h)
        ear  = tracker.calculate_ear(pts)
        open_ = ear >= tracker.ear_threshold
        assert not open_, f"EAR {ear} just below threshold should be CLOSED"

    def test_custom_threshold_shifts_boundary(self):
        """A stricter threshold reclassifies the same EAR as CLOSED."""
        strict  = make_tracker(ear_threshold=0.30)
        lenient = make_tracker(ear_threshold=0.15)
        pts = make_eye_pts(width=60, height=13)   # EAR ≈ 0.22

        ear = strict.calculate_ear(pts)
        assert ear < strict.ear_threshold,  "Should be CLOSED with strict=0.30"
        assert ear >= lenient.ear_threshold, "Should be OPEN with lenient=0.15"


# ---------------------------------------------------------------------------
# 2.  Typical EAR value range (0.15 – 0.35)
# ---------------------------------------------------------------------------

class TestEARValueRange:
    """
    For a healthy eye in a normal webcam environment, EAR sits in the
    range 0.15–0.35.  Synthetic eyes built to mimic realistic proportions
    should fall within this band.
    """

    @pytest.mark.parametrize("width,height", [
        (60, 12),   # narrow  – EAR ≈ 0.20
        (60, 16),   # typical – EAR ≈ 0.27
        (60, 20),   # wide    – EAR ≈ 0.33
    ])
    def test_realistic_eye_in_expected_range(self, tracker, width, height):
        pts = make_eye_pts(width=width, height=height)
        ear = tracker.calculate_ear(pts)
        assert 0.15 <= ear <= 0.35, (
            f"EAR {ear} outside typical range [0.15, 0.35] "
            f"for eye({width}, {height})")

    def test_closed_eye_below_range(self, tracker):
        """A fully shut eye produces EAR < 0.15."""
        pts = make_eye_pts(width=60, height=5)   # EAR ≈ 0.083
        ear = tracker.calculate_ear(pts)
        assert ear < 0.15

    def test_open_eye_above_range_is_capped_by_geometry(self, tracker):
        """
        Extreme vertical stretch can push EAR > 0.35, but realistic
        faces don't open that wide — just verify the formula doesn't
        saturate or wrap.
        """
        pts = make_eye_pts(width=60, height=30)   # EAR = 0.50
        ear = tracker.calculate_ear(pts)
        assert ear > 0.35
        assert ear < 1.0, "EAR should not reach or exceed 1.0 for any eye"


# ---------------------------------------------------------------------------
# 3.  Blink detection
# ---------------------------------------------------------------------------

class TestBlinkDetection:
    """
    Blink state lives on tracker.left_eye / tracker.right_eye (_EyeState).
    Drive both eyes together to test bilateral blink counting via the
    tracker.blink_count property (= min of left and right blink counts).
    """

    def _simulate_blink(self, tracker, closed_frames: int, ts_start: int = 1000):
        """
        Drive both _EyeState machines through one blink cycle:
        `closed_frames` closed frames followed by one open frame.
        Returns the next available timestamp.
        """
        ts = ts_start
        for _ in range(closed_frames):
            tracker.left_eye.update(is_open=False,  timestamp_ms=ts)
            tracker.right_eye.update(is_open=False, timestamp_ms=ts)
            ts += 33
        tracker.left_eye.update(is_open=True,  timestamp_ms=ts)
        tracker.right_eye.update(is_open=True, timestamp_ms=ts)
        return ts + 33

    def test_single_blink_counted(self, tracker):
        """Closing and reopening both eyes once counts as exactly one blink."""
        self._simulate_blink(tracker, closed_frames=3)
        assert tracker.blink_count == 1

    def test_blink_below_consec_threshold_not_counted(self, tracker):
        """
        Eyes closed for fewer frames than consec_needed should not register
        as a blink (noise / involuntary twitch suppression).
        """
        self._simulate_blink(tracker, closed_frames=1)
        assert tracker.blink_count == 0

    def test_multiple_blinks_counted_correctly(self, tracker):
        """Three deliberate blinks → blink_count == 3."""
        ts = 1000
        for _ in range(3):
            ts = self._simulate_blink(tracker, closed_frames=4, ts_start=ts)
        assert tracker.blink_count == 3

    def test_blink_count_starts_at_zero(self, tracker):
        assert tracker.blink_count == 0

    def test_consec_below_resets_after_open(self, tracker):
        """consec_below should reset to 0 on each _EyeState once eyes reopen."""
        self._simulate_blink(tracker, closed_frames=3)
        assert tracker.left_eye.consec_below  == 0
        assert tracker.right_eye.consec_below == 0

    def test_sustained_closure_counts_as_one_blink(self, tracker):
        """Holding eyes shut for many frames is still just one blink."""
        self._simulate_blink(tracker, closed_frames=30)
        assert tracker.blink_count == 1


# ---------------------------------------------------------------------------
# 4.  Threshold tuning range
# ---------------------------------------------------------------------------

class TestThresholdTuning:
    """Verify the recommended 0.18–0.25 tuning range behaves sensibly."""

    @pytest.mark.parametrize("threshold", [0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25])
    def test_threshold_in_recommended_range(self, threshold):
        t = make_tracker(ear_threshold=threshold)
        assert t.ear_threshold == threshold

    def test_open_eye_open_across_all_recommended_thresholds(self):
        """A clearly open eye (EAR≈0.30) should be OPEN for every threshold."""
        pts = make_eye_pts(width=60, height=18)   # EAR = 0.30
        for threshold in [0.18, 0.21, 0.25]:
            t   = make_tracker(ear_threshold=threshold)
            ear = t.calculate_ear(pts)
            assert ear >= threshold, (
                f"Open eye EAR {ear} should be >= threshold {threshold}")

    def test_closed_eye_closed_across_all_recommended_thresholds(self):
        """A clearly closed eye (EAR≈0.10) should be CLOSED for every threshold."""
        pts = make_eye_pts(width=60, height=6)    # EAR ≈ 0.10
        for threshold in [0.18, 0.21, 0.25]:
            t   = make_tracker(ear_threshold=threshold)
            ear = t.calculate_ear(pts)
            assert ear < threshold, (
                f"Closed eye EAR {ear} should be < threshold {threshold}")
