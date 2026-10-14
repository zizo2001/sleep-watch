"""v2.7 — evidence-gated wakes, restless spans, magnitude, grouped clips."""
from types import SimpleNamespace as NS

import pytest

from sleepwatch.config import Config
from sleepwatch.monitor import classify_magnitude
from sleepwatch.report import generate_report, restlessness_score
from sleepwatch.storage import Storage

from tests.test_monitor import build_attributed, build_scripted

CFG = Config()


# -- magnitude classifier ----------------------------------------------------

def test_magnitude_by_coverage():
    assert classify_magnitude(0.7, 2.0, CFG) == "full"
    assert classify_magnitude(0.3, 2.0, CFG) == "partial"
    assert classify_magnitude(0.1, 2.0, CFG) == "twitch"


def test_magnitude_duration_fallback_when_pose_blind():
    assert classify_magnitude(None, 25.0, CFG) == "full"
    assert classify_magnitude(None, 10.0, CFG) == "partial"
    assert classify_magnitude(None, 2.0, CFG) == "twitch"


def test_movement_rows_carry_magnitude(tmp_path):
    m, st, cfg = build_scripted(tmp_path, [("q", 900), ("m", 45), ("q", 300)])
    m.run()
    evs = [e for e in st.events_for_night(st.latest_night()["id"])
           if e["kind"] == "movement"]
    assert evs and evs[0]["magnitude"] == "twitch"   # 3s, pose-blind


# -- evidence gate: twitch bursts are restless, not wake ----------------------

def test_twitch_burst_logs_restless_and_stays_asleep(tmp_path):
    # a real night: bursts of 3s twitches — no wake claim
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 45), ("q", 600), ("m", 45), ("q", 300)])
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    restless = [e for e in evs if e["kind"] == "restless"]
    assert len(restless) == 1
    assert restless[0]["duration_s"] > 0
    # stirs keep their honest label and the night stayed asleep throughout
    moves = [e for e in evs if e["kind"] == "movement"]
    assert len(moves) == 6                    # 5 burst stirs + the later one
    assert [e for e in evs if e["kind"] == "awake"] == []
    assert len([e for e in evs if e["kind"] == "onset"]) == 1


def test_partial_evidence_wake_demoted_on_quick_reonset(tmp_path):
    # a 10s stir fires the wake, but re-sleeping fast with nothing big and
    # no light demotes it (seen for real: 15s stir, re-slept in 14 min)
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 150), ("q", 900), ("m", 45), ("q", 300)])
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    restless = [e for e in evs if e["kind"] == "restless"]
    assert len(restless) == 1                 # the demoted wake marker
    assert [e for e in evs if e["kind"] == "awake"] == []
    assert len([e for e in evs if e["kind"] == "movement"]) == 6
    assert len([e for e in evs if e["kind"] == "onset"]) == 2


def test_full_body_wake_survives_reonset(tmp_path):
    # 20s = full magnitude: real waking, kept even though he re-slept
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 300), ("q", 900), ("m", 45), ("q", 300)])
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert len([e for e in evs if e["kind"] == "wake"]) == 1
    assert len([e for e in evs if e["kind"] == "awake"]) == 5


# -- ambient needs pose to actually watch the event ---------------------------

def test_flickering_pose_cannot_call_ambient(tmp_path):
    # pose sees a far-corner person in <50% of motion frames — the verdict
    # defaults to "you" (night 10: blanket lump misread as ambient)
    calls = {"n": 0}

    def flicker():
        calls["n"] += 1
        return (0.85, 0.85, 0.12, 0.12) if calls["n"] % 10 == 0 else None

    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)], person_box=flicker)
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert [e for e in evs if e["kind"] == "ambient"] == []
    assert len([e for e in evs if e["kind"] == "movement"]) == 1


def test_steady_pose_still_calls_ambient(tmp_path):
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)],
        person_box=lambda: (0.85, 0.85, 0.12, 0.12))
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert len([e for e in evs if e["kind"] == "ambient"]) == 1


def test_coverage_persisted_when_pose_saw(tmp_path):
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)],
        person_box=lambda: (0.0, 0.2, 0.7, 0.6))
    m.run()
    evs = [e for e in st.events_for_night(st.latest_night()["id"])
           if e["kind"] == "movement"]
    assert evs and evs[0]["coverage"] is not None
    assert evs[0]["magnitude"] in ("twitch", "partial", "full")


# -- clock stamp ---------------------------------------------------------------

def test_stamp_clock_returns_marked_copy(tmp_path):
    import numpy as np
    m, st, cfg = build_scripted(tmp_path, [("q", 30)])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    stamped = m._stamp_clock(frame, 1_787_097_469.0)
    assert stamped.shape == frame.shape
    assert stamped is not frame
    assert not frame.any()                    # raw frame untouched
    assert stamped.any()                      # clock pixels landed


# -- storage migration ---------------------------------------------------------

def test_migration_adds_magnitude_columns(tmp_path):
    db = tmp_path / "old.db"
    st = Storage(db)
    st.close()
    st = Storage(db)                          # idempotent reopen
    nid = st.begin_night("2026-08-19", 0.0, "{}")
    st.add_event(nid, "movement", 1.0, 2.0, 1.0, 0.1, 0.05, None,
                 coverage=0.4, magnitude="partial")
    row = st.events_for_night(nid)[0]
    assert row["coverage"] == 0.4 and row["magnitude"] == "partial"
    st.close()


# -- report: merged clips render once ------------------------------------------

def _night(date="2026-08-19"):
    return {"id": 1, "date": date, "start_ts": 1000.0,
            "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1300.0,
            "config_json": "{}"}


def _ev(start, dur, kind="movement", clip=None, eid=1, magnitude=None):
    return NS(start_ts=start, end_ts=start + dur, duration_s=dur,
              kind=kind, peak_score=0.3, mean_score=0.1, clip_path=clip,
              clip_duration_s=42.0 if clip else None, rating=None,
              id=eid, coverage=None, magnitude=magnitude)


def test_report_groups_merged_clips_into_one_row(tmp_path):
    clip = str(tmp_path / "c.mp4")
    (tmp_path / "c.mp4").write_bytes(b"x")
    events = [_ev(2000, 5, clip=clip, eid=1),
              _ev(2030, 4, clip=clip, eid=2),
              _ev(2060, 3, clip=clip, eid=3)]
    out = generate_report(_night(), events, tmp_path / "r.html")
    html = out.read_text()
    assert html.count('class="watch"') == 1          # one row per FILE
    assert "3 moves" in html


def test_report_renders_restless_band_and_magnitude_badge(tmp_path):
    events = [_ev(2000, 5, magnitude="full"),
              _ev(3000, 600, kind="restless")]
    out = generate_report(_night(), events, tmp_path / "r.html")
    html = out.read_text()
    assert "ev-restless" in html
    assert "mag-full" in html and "full-body" in html


def test_report_restlessness_tile(tmp_path):
    minutes = [NS(minute_ts=1300.0 + i * 60, max_motion=0.002,
                  threshold=0.004, max_speech=None) for i in range(60)]
    out = generate_report(_night(), [_ev(2000, 5)], tmp_path / "r.html",
                          signal_minutes=minutes)
    assert "restlessness" in out.read_text()


# -- restlessness score ----------------------------------------------------------

def test_restlessness_score_scales_with_motion():
    quiet = [NS(minute_ts=100.0 + i, max_motion=0.0004, threshold=0.004,
                max_speech=None) for i in range(30)]
    churn = [NS(minute_ts=100.0 + i, max_motion=0.004, threshold=0.004,
                max_speech=None) for i in range(30)]
    lo = restlessness_score(quiet, [])
    hi = restlessness_score(churn, [])
    assert lo is not None and hi is not None and hi > lo
    assert restlessness_score([], []) is None


def test_restlessness_score_only_counts_asleep_minutes():
    minutes = ([NS(minute_ts=t, max_motion=0.006, threshold=0.004,
                   max_speech=None) for t in (100.0, 160.0)]      # awake
               + [NS(minute_ts=t, max_motion=0.0004, threshold=0.004,
                     max_speech=None) for t in (300.0, 360.0)])   # asleep
    scoped = restlessness_score(minutes, [(250.0, 400.0)])
    everything = restlessness_score(minutes, [])
    assert scoped is not None and everything is not None
    assert scoped < everything


# -- review fixes (2026-08-19 adversarial pass) --------------------------------

def _leak_monitor(tmp_path, script):
    """Camera whose blip frames draw a BIG box and movement frames a SMALL
    one — the only shape that can expose a leaked motion union."""
    import numpy as np
    from sleepwatch.baseline import AdaptiveBaseline
    from sleepwatch.clipwriter import ClipWriter, FrameRingBuffer
    from sleepwatch.detector import MotionScorer
    from sleepwatch.events import EventStateMachine
    from sleepwatch.monitor import Monitor
    from sleepwatch.storage import Storage
    from tests.test_monitor import FakeClock

    class TwoSizeCamera:
        def __init__(self, clock, script):
            self.clock, self.i, self.monitor = clock, 0, None
            self.seq = [k for kind, n in script for k in [kind] * n]

        def read(self):
            if self.i >= len(self.seq):
                self.monitor.request_stop()
                return True, self._frame("q")
            f = self._frame(self.seq[self.i])
            self.i += 1
            self.clock.tick()
            return True, f

        def _frame(self, kind):
            f = np.full((120, 160, 3), 60, dtype=np.uint8)
            if kind == "B":                       # ~33% — big, but under
                x = 20 + (self.i % 4)             # lighting_frac (0.60)
                f[20:100, x:x + 80] = 220
            elif kind == "s":                     # ~4% of the frame
                x = 65 + (self.i % 5)
                f[45:70, x:x + 30] = 220
            return f

        def close(self):
            pass

    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0,
                 postroll_seconds=2.0, fps=15, sleep_onset_minutes=0.5)
    clock = FakeClock()
    cam = TwoSizeCamera(clock, script)
    st = Storage(cfg.data_dir / "sleepwatch.db")
    m = Monitor(
        config=cfg, storage=st, camera=cam,
        scorer=MotionScorer(alpha=cfg.alpha),
        baseline=AdaptiveBaseline(window_seconds=cfg.window_seconds,
                                  fps=cfg.fps, k=cfg.k,
                                  min_threshold=cfg.min_threshold,
                                  warmup_seconds=cfg.warmup_seconds),
        fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
        ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
        clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
        person_box=lambda: (0.0, 0.0, 0.7, 0.7),
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg


def test_aborted_blip_does_not_inflate_the_next_events_coverage(tmp_path):
    """The night-10 killer: a sub-trigger blip aborts in PENDING, returns no
    event, so the close branch never resets attribution. Its big union then
    leaked into the next event, turning a twitch into 'full' — fake wake
    evidence. Baseline first: the small movement alone is a twitch."""
    m, st, _ = _leak_monitor(tmp_path / "clean",
                             [("q", 900), ("s", 45), ("q", 300)])
    m.run()
    clean = [e for e in st.events_for_night(st.latest_night()["id"])
             if e["kind"] == "movement"]
    assert clean and clean[0]["magnitude"] == "twitch"
    baseline_cov = clean[0]["coverage"]

    # same movement, but a 2-frame big blip precedes it
    m2, st2, _ = _leak_monitor(
        tmp_path / "blip",
        [("q", 900), ("B", 2), ("q", 150), ("s", 45), ("q", 300)])
    m2.run()
    evs = st2.events_for_night(st2.latest_night()["id"])
    moves = [e for e in evs if e["kind"] == "movement"]
    assert moves, "the small movement should still be detected"
    assert moves[-1]["magnitude"] == "twitch"        # was "full" before
    assert moves[-1]["coverage"] == pytest.approx(baseline_cov, abs=0.05)


def test_coverage_needs_sustained_pose_not_one_lucky_frame(tmp_path):
    calls = {"n": 0}

    def flicker():
        calls["n"] += 1
        return (0.0, 0.2, 0.7, 0.6) if calls["n"] % 12 == 0 else None

    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)], person_box=flicker)
    m.run()
    evs = [e for e in st.events_for_night(st.latest_night()["id"])
           if e["kind"] == "movement"]
    assert evs and evs[0]["coverage"] is None      # not a measurement


def test_demoted_wake_becomes_a_spanning_restless_row(tmp_path):
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 150), ("q", 900), ("m", 45), ("q", 300)])
    m.run()
    restless = [e for e in st.events_for_night(st.latest_night()["id"])
                if e["kind"] == "restless"]
    assert len(restless) == 1
    r = restless[0]
    assert r["end_ts"] > r["start_ts"]             # a span, not a tick
    assert r["duration_s"] and r["duration_s"] > 0


def test_speech_sharing_a_clip_keeps_its_own_row(tmp_path):
    clip = str(tmp_path / "c.mp4")
    (tmp_path / "c.mp4").write_bytes(b"x")
    events = [_ev(2000, 5, clip=clip, eid=1, magnitude="twitch"),
              _ev(2003, 5, kind="speech", clip=clip, eid=2)]
    html = generate_report(_night(), events, tmp_path / "r.html").read_text()
    assert "0.300" in html                     # motion peak survives
    assert html.count("<tr") == 3              # header + movement + speech


def test_ambient_sharing_a_clip_keeps_its_label(tmp_path):
    clip = str(tmp_path / "c.mp4")
    (tmp_path / "c.mp4").write_bytes(b"x")
    events = [_ev(2000, 4, kind="ambient", clip=clip, eid=1,
                  magnitude="partial"),
              _ev(2007, 3, clip=clip, eid=2, magnitude="twitch")]
    html = generate_report(_night(), events, tmp_path / "r.html").read_text()
    assert "ambient — not you" in html
    assert 'class="muted"' in html          # ambient keeps its gray row
    assert html.count("<tr") == 3           # header + ambient + movement
    # the ambient's magnitude must not be promoted onto the movement row
    movement_row = html.split("</tr>")[2]
    assert "mag-twitch" in movement_row and "mag-partial" not in movement_row


def test_merged_row_carries_every_member_id_for_delete(tmp_path):
    clip = str(tmp_path / "c.mp4")
    (tmp_path / "c.mp4").write_bytes(b"x")
    events = [_ev(2000, 5, clip=clip, eid=7),
              _ev(2030, 4, clip=clip, eid=8),
              _ev(2060, 3, clip=clip, eid=9)]
    html = generate_report(_night(), events, tmp_path / "r.html").read_text()
    assert 'data-eids="7,8,9"' in html
