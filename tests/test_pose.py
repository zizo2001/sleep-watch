from pathlib import Path

import numpy as np
import pytest

from sleepwatch.pose import PoseOverlay

MODEL = Path("models/pose_landmarker_lite.task")


def frame():
    return np.full((120, 160, 3), 90, dtype=np.uint8)


def test_no_model_path_degrades_gracefully():
    p = PoseOverlay(model_path=None)
    assert p.available is False
    f = frame()
    out = p.annotate(f, ts_ms=0)
    assert out is f                      # unchanged, same object

def test_missing_model_file_degrades_gracefully(tmp_path):
    p = PoseOverlay(model_path=tmp_path / "nope.task")
    assert p.available is False
    assert p.annotate(frame(), ts_ms=0) is not None

@pytest.mark.skipif(not MODEL.exists(), reason="pose model not downloaded")
def test_real_model_annotates_without_crashing():
    p = PoseOverlay(model_path=MODEL)
    assert p.available is True
    out = p.annotate(frame(), ts_ms=0)   # no human in frame — must not crash
    assert out.shape == frame().shape
    out2 = p.annotate(frame(), ts_ms=66)
    assert out2 is not None
