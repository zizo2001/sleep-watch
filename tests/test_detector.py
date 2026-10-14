import numpy as np
from sleepwatch.detector import MotionScorer, ScoreResult

def gray_frame(value=80, w=1280, h=720):
    return np.full((h, w, 3), value, dtype=np.uint8)

def frame_with_square(x, value=200, size=200):
    f = gray_frame()
    f[300:300 + size, x:x + size] = value
    return f

def test_first_frame_scores_zero():
    s = MotionScorer()
    assert s.score(gray_frame()) == ScoreResult(0.0, False)

def test_static_scene_stays_quiet():
    s = MotionScorer()
    for _ in range(10):
        r = s.score(gray_frame())
    assert r.score < 0.001 and not r.lighting

def test_moving_square_scores():
    s = MotionScorer()
    s.score(gray_frame())
    for _ in range(5):
        s.score(gray_frame())          # settle background
    r = s.score(frame_with_square(x=100))
    assert r.score > 0.01 and not r.lighting

def test_global_change_is_lighting_event():
    s = MotionScorer()
    s.score(gray_frame(value=30))
    s.score(gray_frame(value=30))
    r = s.score(gray_frame(value=220))  # lights on
    assert r.lighting and r.score == 0.0

def test_background_relearns_after_lighting():
    s = MotionScorer()
    s.score(gray_frame(30))
    s.score(gray_frame(30))
    s.score(gray_frame(220))            # lighting event, re-learn
    r = s.score(gray_frame(220))        # new normal
    assert not r.lighting and r.score < 0.001

def test_motion_box_locates_the_square():
    s = MotionScorer()
    for _ in range(6):
        s.score(gray_frame())
    # square at x=100, y=300, 200px in 1280x720 → /4 in 320x180 space
    r = s.score(frame_with_square(x=100))
    assert r.frame_size == (320, 180)
    assert r.box is not None
    bx, by, bw, bh = r.box
    assert abs(bx - 25) <= 4 and abs(by - 75) <= 4
    assert 40 <= bw <= 60 and 40 <= bh <= 60

def test_no_motion_no_box():
    s = MotionScorer()
    s.score(gray_frame())
    r = s.score(gray_frame())
    assert r.box is None
