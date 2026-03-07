"""
tests/test_ear.py
-----------------
Unit tests for the EyeTracker.calculate_ear() method.

Run with:
    python -m pytest tests/test_ear.py -v
"""

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker


@pytest.fixture
def tracker():
    """EyeTracker with MediaPipe and model-download fully mocked out."""
    return make_tracker(ear_threshold=0.21)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eye(width: float, height: float) -> list:
    """
    Build a synthetic 6-point eye landmark list.

    The eye is centred at the origin and shaped as a horizontal ellipse:
        P1 = left corner  (-w/2, 0)
        P2 = top-left     (-w/4, +h/2)
        P3 = top-right    (+w/4, +h/2)
        P4 = right corner (+w/2, 0)
        P5 = bot-right    (+w/4, -h/2)
        P6 = bot-left     (-w/4, -h/2)

    Expected EAR = (2 * h/2 + 2 * h/2) / (2 * w) ... wait, let's derive it:
        ||P2-P6|| = h,  ||P3-P5|| = h,  ||P1-P4|| = w
        EAR = (h + h) / (2 * w) = h / w
    So make_eye(w=1, h=0.3) -> EAR = 0.3
    """
    hw, hh = width / 2, height / 2
    return [
        (-hw,    0),    # P1 left corner
        (-hw/2, +hh),   # P2 top-left
        (+hw/2, +hh),   # P3 top-right
        (+hw,    0),    # P4 right corner
        (+hw/2, -hh),   # P5 bot-right
        (-hw/2, -hh),   # P6 bot-left
    ]


# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

class TestCalculateEAR:

    def test_wide_open_eye_high_ear(self, tracker):
        """A tall open eye should produce EAR well above threshold."""
        pts = make_eye(width=1.0, height=0.4)
        ear = tracker.calculate_ear(pts)
        assert ear >= 0.21, f"Expected >= 0.21 for open eye, got {ear}"

    def test_closed_eye_low_ear(self, tracker):
        """A flat (closed) eye should produce EAR below threshold."""
        pts = make_eye(width=1.0, height=0.05)
        ear = tracker.calculate_ear(pts)
        assert ear < 0.21, f"Expected < 0.21 for closed eye, got {ear}"

    def test_exact_ear_formula(self, tracker):
        """
        Verify the formula numerically.
        With make_eye(w, h): EAR should equal h / w.
        """
        w, h = 2.0, 0.6
        pts  = make_eye(w, h)
        ear  = tracker.calculate_ear(pts)
        expected = round(h / w, 3)
        assert math.isclose(ear, expected, abs_tol=1e-3), (
            f"Expected EAR = {expected}, got {ear}"
        )

    def test_threshold_boundary_open(self, tracker):
        """EAR exactly at threshold should be classified OPEN."""
        pts = make_eye(width=1.0, height=0.21)
        ear = tracker.calculate_ear(pts)
        assert ear >= tracker.ear_threshold

    def test_threshold_boundary_closed(self, tracker):
        """EAR just below threshold should be classified CLOSED."""
        pts = make_eye(width=1.0, height=0.20)
        ear = tracker.calculate_ear(pts)
        assert ear < tracker.ear_threshold

    def test_return_type_is_float(self, tracker):
        pts = make_eye(1.0, 0.3)
        ear = tracker.calculate_ear(pts)
        assert isinstance(ear, float)

    def test_return_value_is_positive(self, tracker):
        pts = make_eye(1.0, 0.3)
        ear = tracker.calculate_ear(pts)
        assert ear > 0.0

    def test_ear_rounded_to_3dp(self, tracker):
        pts = make_eye(3.0, 1.0)
        ear = tracker.calculate_ear(pts)
        assert ear == round(ear, 3)

    def test_pixel_coordinates(self, tracker):
        """Verify the formula works with realistic pixel-scale inputs."""
        pts = [
            (120, 240),   # P1
            (132, 232),   # P2
            (148, 232),   # P3
            (160, 240),   # P4
            (148, 248),   # P5
            (132, 248),   # P6
        ]
        ear = tracker.calculate_ear(pts)
        assert 0.0 < ear < 1.0, f"EAR should be in (0,1), got {ear}"


# ---------------------------------------------------------------------------
# Edge / error cases
# ---------------------------------------------------------------------------

class TestCalculateEAREdgeCases:

    def test_too_few_landmarks_returns_zero(self, tracker):
        """Fewer than 6 points -> return 0.0 without raising."""
        assert tracker.calculate_ear([])          == 0.0
        assert tracker.calculate_ear([(0, 0)])    == 0.0
        assert tracker.calculate_ear([(0,0)] * 5) == 0.0

    def test_zero_width_eye_returns_zero(self, tracker):
        """All points on the same x -> horizontal distance is 0 -> return 0.0."""
        pts = [(10, 10)] * 6
        assert tracker.calculate_ear(pts) == 0.0

    def test_completely_flat_eye_near_zero(self, tracker):
        """Eye height = 0 -> EAR should be very close to 0."""
        pts = make_eye(width=1.0, height=0.001)
        ear = tracker.calculate_ear(pts)
        assert ear < 0.01

    def test_different_thresholds(self):
        """Custom EAR threshold should shift open/closed boundary."""
        strict  = make_tracker(ear_threshold=0.30)
        lenient = make_tracker(ear_threshold=0.15)
        pts = make_eye(width=1.0, height=0.22)   # EAR = 0.22

        ear = strict.calculate_ear(pts)
        assert ear < strict.ear_threshold,   "Should be CLOSED with strict threshold"
        assert ear >= lenient.ear_threshold, "Should be OPEN with lenient threshold"
