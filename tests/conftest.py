"""
tests/conftest.py
-----------------
Shared fixtures and mock helpers used across all test modules.

Patches out the heavy MediaPipe / camera dependencies so every test
runs without a physical webcam or the face_landmarker.task model file.
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make sure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Lightweight EyeTracker factory
# ---------------------------------------------------------------------------

def make_tracker(ear_threshold: float = 0.21):
    """
    Return an EyeTracker instance with MediaPipe and model-download
    stubbed out — no network, no GPU, no .task file required.
    """
    mock_landmarker = MagicMock()

    with (
        patch("eye_tracker.MODEL_PATH") as mock_path,
        patch("eye_tracker.FaceLandmarker") as mock_fl,
        patch("urllib.request.urlretrieve"),
    ):
        mock_path.exists.return_value = True          # pretend model is cached
        mock_fl.create_from_options.return_value = mock_landmarker

        from eye_tracker import EyeTracker
        tracker = EyeTracker(ear_threshold=ear_threshold)

    # Swap in the mock so process_frame calls hit it
    tracker.landmarker = mock_landmarker
    return tracker


# ---------------------------------------------------------------------------
# Synthetic landmark helpers
# ---------------------------------------------------------------------------

def make_landmark(x: float, y: float, z: float = 0.0):
    """Return a simple object that mimics a MediaPipe NormalizedLandmark."""
    lm = types.SimpleNamespace(x=x, y=y, z=z)
    return lm


def make_eye_pts(width: float, height: float) -> list:
    """
    Build a 6-point synthetic eye in pixel coords.
    Eye is centred at (100, 100) with the given width and height.

        EAR = height / width  (see test_ear.py for derivation)
    """
    cx, cy = 100.0, 100.0
    hw, hh = width / 2, height / 2
    return [
        (int(cx - hw),     int(cy)),          # P1 left corner
        (int(cx - hw / 2), int(cy - hh)),     # P2 top-left
        (int(cx + hw / 2), int(cy - hh)),     # P3 top-right
        (int(cx + hw),     int(cy)),          # P4 right corner
        (int(cx + hw / 2), int(cy + hh)),     # P5 bot-right
        (int(cx - hw / 2), int(cy + hh)),     # P6 bot-left
    ]


def make_blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a plain grey BGR frame."""
    return np.full((h, w, 3), 100, dtype=np.uint8)


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    """EyeTracker with default threshold, no real MediaPipe."""
    return make_tracker(ear_threshold=0.21)


@pytest.fixture
def blank_frame():
    return make_blank_frame()
