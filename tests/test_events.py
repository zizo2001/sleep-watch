from sleepwatch.events import EventStateMachine, MovementEvent

FPS = 15
DT = 1.0 / FPS

def feed(fsm, seq, thr=0.01, t0=1000.0):
    """seq: list of scores. Returns list of (t, output)."""
    out = []
    for i, s in enumerate(seq):
        t = t0 + i * DT
        out.append((t, fsm.update(s, thr, wall_ts=t, mono_ts=t)))
    return out

def test_brief_blip_never_triggers():
    fsm = EventStateMachine(trigger_seconds=0.5, cooldown_seconds=3.0)
    outs = feed(fsm, [0.0] * 5 + [0.5] * 3 + [0.0] * 80)  # 0.2s blip
    assert all(o is None for _, o in outs)

def test_trigger_then_cooldown_closes_one_event():
    fsm = EventStateMachine(trigger_seconds=0.5, cooldown_seconds=3.0)
    seq = [0.0] * 5 + [0.5] * 30 + [0.0] * 60   # 2s movement, 4s quiet
    outs = [o for _, o in feed(fsm, seq)]
    starts = [o for o in outs if o == "started"]
    events = [o for o in outs if isinstance(o, MovementEvent)]
    assert len(starts) == 1 and len(events) == 1
    ev = events[0]
    assert abs(ev.duration_s - (29 * DT)) < 1e-6      # first..last above frame
    assert ev.peak_score == 0.5 and ev.frames == 30

def test_short_dip_merges_into_one_event():
    fsm = EventStateMachine(trigger_seconds=0.5, cooldown_seconds=3.0)
    seq = [0.0] * 5 + [0.5] * 15 + [0.0] * 15 + [0.5] * 15 + [0.0] * 60
    outs = [o for _, o in feed(fsm, seq)]           # 1s dip < 3s cooldown
    assert outs.count("started") == 1
    assert len([o for o in outs if isinstance(o, MovementEvent)]) == 1

def test_start_wall_is_first_above_frame():
    fsm = EventStateMachine(trigger_seconds=0.5, cooldown_seconds=3.0)
    outs = feed(fsm, [0.0] * 10 + [0.5] * 30 + [0.0] * 60)
    ev = [o for _, o in outs if isinstance(o, MovementEvent)][0]
    assert abs(ev.start_wall - (1000.0 + 10 * DT)) < 1e-6

def test_force_close_returns_open_event():
    fsm = EventStateMachine(trigger_seconds=0.5, cooldown_seconds=3.0)
    feed(fsm, [0.0] * 5 + [0.5] * 30)
    ev = fsm.force_close(wall_ts=2000.0, mono_ts=2000.0)
    assert isinstance(ev, MovementEvent)
    assert fsm.force_close(2001.0, 2001.0) is None
