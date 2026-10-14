import random
from sleepwatch.baseline import AdaptiveBaseline

def make(warmup=1.0, fps=15, k=6.0, floor=0.004):
    return AdaptiveBaseline(window_seconds=20, fps=fps, k=k,
                            min_threshold=floor, warmup_seconds=warmup)

def test_not_ready_during_warmup():
    b = make(warmup=1.0)
    for _ in range(10):          # <15 samples
        b.update(0.001, in_movement=False)
    assert not b.ready and b.threshold == float("inf")

def test_ready_after_warmup_with_floor():
    b = make()
    for _ in range(30):
        b.update(0.0, in_movement=False)   # perfectly quiet
    assert b.ready
    assert b.threshold == 0.004            # floor wins over med+k*MAD=0

def test_threshold_tracks_noise():
    b = make()
    rng = random.Random(1)
    for _ in range(300):
        b.update(0.01 + rng.uniform(-0.002, 0.002), in_movement=False)
    assert 0.01 < b.threshold < 0.05

def test_movement_scores_excluded():
    b = make()
    for _ in range(300):
        b.update(0.001, in_movement=False)
    quiet_thr = b.threshold
    for _ in range(300):
        b.update(0.5, in_movement=True)    # long movement must not inflate
    assert b.threshold == quiet_thr
