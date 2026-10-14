import numpy as np
import cv2
from pathlib import Path
from sleepwatch.clipwriter import FrameRingBuffer, ClipWriter

def frame(v):
    return np.full((120, 160, 3), v, dtype=np.uint8)

def test_ring_buffer_caps_and_orders():
    rb = FrameRingBuffer(seconds=1.0, fps=5)          # cap 5
    for v in range(8):
        rb.push(frame(v))
    frames = rb.dump()
    assert len(frames) == 5
    assert [int(f[0, 0, 0]) for f in frames] == [3, 4, 5, 6, 7]

def test_ring_buffer_dump_copies():
    rb = FrameRingBuffer(seconds=1.0, fps=5)
    f = frame(9)
    rb.push(f)
    f[:] = 0
    assert int(rb.dump()[0][0, 0, 0]) == 9

def test_clip_writer_roundtrip(tmp_path):
    w = ClipWriter(clips_root=tmp_path, fps=15)
    assert not w.active
    w.start("2026-08-11", event_start_wall=1754947200.0,
            preroll=[frame(1), frame(2)])
    assert w.active
    for v in (3, 4, 5):
        w.write(frame(v))
    closed = w.close()
    assert not w.active and closed is not None and closed.path.exists()
    assert closed.path.parent.name == "2026-08-11"
    cap = cv2.VideoCapture(str(closed.path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 5

def test_close_without_start_returns_none(tmp_path):
    assert ClipWriter(tmp_path, fps=15).close() is None

def test_close_reports_path_and_duration(tmp_path):
    w = ClipWriter(clips_root=tmp_path, fps=15)
    w.start("2026-08-12", event_start_wall=1755000000.0,
            preroll=[frame(1), frame(2)])
    assert w.path is not None and w.path.parent.name == "2026-08-12"
    for v in (3, 4, 5):
        w.write(frame(v))
    closed = w.close()
    assert closed is not None
    assert closed.path.exists()
    assert abs(closed.duration_s - 5 / 15) < 1e-9
    assert w.path is None

def test_path_none_when_inactive(tmp_path):
    assert ClipWriter(tmp_path, fps=15).path is None
