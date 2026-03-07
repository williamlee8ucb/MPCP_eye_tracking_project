"""
tests/test_camera.py
--------------------
Tests for webcam initialisation, error handling, and clean program exit.

Covers acceptance criteria:
  - Webcam initialises successfully
  - 'q' key exits program cleanly
  - No memory leaks (resources always released)
  - Console shows clear status messages
  - Camera failures are handled gracefully

These tests mock cv2.VideoCapture so no physical camera is required.

Run with:
    python -m pytest tests/test_camera.py -v
"""

import sys
import os
import io
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conftest import make_tracker, make_blank_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cap(width: int = 640, height: int = 480, fps: float = 30.0,
             read_frames: int = 5, fail_after: int = None):
    """
    Build a mock cv2.VideoCapture that:
    - opens successfully
    - reports the requested properties
    - returns `read_frames` good frames then stops (or fails at `fail_after`)
    """
    cap = MagicMock()
    cap.isOpened.return_value = True

    def get_prop(prop_id):
        return {
            3:  width,   # CAP_PROP_FRAME_WIDTH
            4:  height,  # CAP_PROP_FRAME_HEIGHT
            5:  fps,     # CAP_PROP_FPS
        }.get(prop_id, 0.0)

    cap.get.side_effect = get_prop

    frame = make_blank_frame(width, height)
    reads = []
    for i in range(read_frames):
        if fail_after is not None and i >= fail_after:
            reads.append((False, None))
        else:
            reads.append((True, frame))

    cap.read.side_effect = reads
    return cap


def _run_with_mock_cap(tracker, cap, key_sequence=None):
    """
    Patch cv2.VideoCapture and cv2.waitKey, then call tracker.run().
    key_sequence: list of waitKey return values (default: [ord('q')]).
    """
    if key_sequence is None:
        key_sequence = [ord("q")]

    with (
        patch("cv2.VideoCapture", return_value=cap),
        patch("cv2.imshow"),
        patch("cv2.destroyAllWindows"),
        patch("cv2.waitKey", side_effect=key_sequence),
    ):
        tracker.run()


# ---------------------------------------------------------------------------
# 1.  Camera initialisation
# ---------------------------------------------------------------------------

class TestCameraInit:

    def test_successful_init_does_not_raise(self):
        """A valid camera index opens without errors."""
        tracker = make_tracker()
        cap = make_cap(read_frames=1)
        try:
            _run_with_mock_cap(tracker, cap)
        except SystemExit:
            pytest.fail("run() called sys.exit on a working camera")

    def test_sets_requested_resolution(self):
        """run() must request 640×480 from the camera."""
        tracker = make_tracker()
        cap = make_cap(read_frames=1)

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", return_value=ord("q")),
        ):
            tracker.run()

        set_calls = [c for c in cap.set.call_args_list]
        set_dict  = {c.args[0]: c.args[1] for c in set_calls}

        import cv2
        assert set_dict.get(cv2.CAP_PROP_FRAME_WIDTH)  == 640
        assert set_dict.get(cv2.CAP_PROP_FRAME_HEIGHT) == 480

    def test_sets_30fps(self):
        tracker = make_tracker()
        cap = make_cap(read_frames=1)

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", return_value=ord("q")),
        ):
            tracker.run()

        import cv2
        set_dict = {c.args[0]: c.args[1] for c in cap.set.call_args_list}
        assert set_dict.get(cv2.CAP_PROP_FPS) == 30

    def test_camera_not_found_exits_with_error(self, capsys):
        """When the camera cannot be opened, sys.exit(1) is called."""
        tracker = make_tracker()
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False

        with (
            patch("cv2.VideoCapture", return_value=bad_cap),
            pytest.raises(SystemExit) as exc_info,
        ):
            tracker.run()

        assert exc_info.value.code == 1

    def test_camera_not_found_prints_error(self, capsys):
        """A clear error message must be printed when camera is missing."""
        tracker = make_tracker()
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False

        with (
            patch("cv2.VideoCapture", return_value=bad_cap),
            pytest.raises(SystemExit),
        ):
            tracker.run()

        captured = capsys.readouterr()
        assert "ERROR" in captured.out or "error" in captured.out.lower()


# ---------------------------------------------------------------------------
# 2.  Clean exit
# ---------------------------------------------------------------------------

class TestCleanExit:

    def test_q_key_exits(self):
        """Pressing 'q' should end the loop without raising."""
        tracker = make_tracker()
        cap = make_cap(read_frames=10)
        try:
            _run_with_mock_cap(tracker, cap, key_sequence=[0, 0, ord("q")])
        except Exception as exc:
            pytest.fail(f"Clean exit raised: {exc}")

    def test_Q_uppercase_exits(self):
        """Uppercase 'Q' should also exit."""
        tracker = make_tracker()
        cap = make_cap(read_frames=10)
        try:
            _run_with_mock_cap(tracker, cap, key_sequence=[0, ord("Q")])
        except Exception as exc:
            pytest.fail(f"Uppercase Q raised: {exc}")

    def test_escape_key_exits(self):
        """ESC (key code 27) should also exit."""
        tracker = make_tracker()
        cap = make_cap(read_frames=10)
        try:
            _run_with_mock_cap(tracker, cap, key_sequence=[0, 27])
        except Exception as exc:
            pytest.fail(f"ESC raised: {exc}")

    def test_cap_released_on_quit(self):
        """cap.release() must be called after the loop exits."""
        tracker = make_tracker()
        cap = make_cap(read_frames=5)
        _run_with_mock_cap(tracker, cap, key_sequence=[ord("q")])
        cap.release.assert_called_once()

    def test_windows_destroyed_on_quit(self):
        """cv2.destroyAllWindows() must be called after the loop exits."""
        tracker = make_tracker()
        cap = make_cap(read_frames=5)

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows") as mock_destroy,
            patch("cv2.waitKey", return_value=ord("q")),
        ):
            tracker.run()

        mock_destroy.assert_called_once()

    def test_landmarker_closed_on_quit(self):
        """landmarker.close() must be called to release MediaPipe resources."""
        tracker = make_tracker()
        cap = make_cap(read_frames=5)
        _run_with_mock_cap(tracker, cap, key_sequence=[ord("q")])
        tracker.landmarker.close.assert_called_once()


# ---------------------------------------------------------------------------
# 3.  Camera failure handling
# ---------------------------------------------------------------------------

class TestCameraFailureHandling:

    def test_single_bad_read_does_not_crash(self):
        """
        One failed frame read should log a warning but not crash —
        the loop should retry.
        """
        tracker = make_tracker()
        # 1 bad frame then a good frame then 'q'
        frame = make_blank_frame()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.side_effect = [(False, None), (True, frame)]

        try:
            _run_with_mock_cap(tracker, cap, key_sequence=[0, ord("q")])
        except Exception as exc:
            pytest.fail(f"Single bad read raised: {exc}")

    def test_consecutive_failures_exit_cleanly(self, capsys):
        """
        MAX_FAILURES consecutive bad reads should exit cleanly and print
        an error — not hang or raise an unhandled exception.
        """
        tracker = make_tracker()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        # Return only failed reads (more than MAX_FAILURES)
        cap.read.return_value = (False, None)

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", return_value=0),
            patch("time.sleep"),        # skip the retry sleep
        ):
            try:
                tracker.run()
            except Exception as exc:
                pytest.fail(f"Consecutive failures raised: {exc}")

        out = capsys.readouterr().out
        assert "ERROR" in out or "error" in out.lower() or "WARNING" in out

    def test_resources_released_after_camera_failure(self):
        """cap.release() is called even when the camera drops mid-session."""
        tracker = make_tracker()
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = 0.0
        cap.read.return_value = (False, None)

        with (
            patch("cv2.VideoCapture", return_value=cap),
            patch("cv2.imshow"),
            patch("cv2.destroyAllWindows"),
            patch("cv2.waitKey", return_value=0),
            patch("time.sleep"),
        ):
            tracker.run()

        cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# 4.  Console status messages
# ---------------------------------------------------------------------------

class TestConsoleOutput:

    def test_startup_message_printed(self, capsys):
        """A startup message should be visible when the tracker begins."""
        tracker = make_tracker()
        cap = make_cap(read_frames=1)
        _run_with_mock_cap(tracker, cap, key_sequence=[ord("q")])
        out = capsys.readouterr().out
        # Should mention camera info or a ready/quit message
        assert any(kw in out.lower() for kw in
                   ["camera", "ready", "press", "quit", "fps"])

    def test_exit_summary_printed(self, capsys):
        """A summary line (frame count, blink count) should print on exit."""
        tracker = make_tracker()
        cap = make_cap(read_frames=3)
        _run_with_mock_cap(tracker, cap, key_sequence=[0, 0, ord("q")])
        out = capsys.readouterr().out
        assert "frame" in out.lower() or "blink" in out.lower()
