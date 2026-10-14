import numpy as np
from pathlib import Path
from sleepwatch.config import Config
from sleepwatch.monitor import Monitor
from sleepwatch.detector import MotionScorer
from sleepwatch.baseline import AdaptiveBaseline
from sleepwatch.events import EventStateMachine
from sleepwatch.clipwriter import FrameRingBuffer, ClipWriter
from sleepwatch.storage import Storage

class FakeClock:
    def __init__(self, start=1000.0, dt=1 / 15):
        self.t, self.dt = start, dt
    def wall(self):
        return self.t
    def mono(self):
        return self.t
    def tick(self):
        self.t += self.dt

class FakeCamera:
    """Static frames, then a moving square, then static again, then stop."""
    def __init__(self, clock, quiet1=900, move=45, quiet2=150):
        self.clock = clock
        self.seq = ["q"] * quiet1 + ["m"] * move + ["q"] * quiet2
        self.i = 0
        self.monitor = None
    def read(self):
        if self.i >= len(self.seq):
            self.monitor.request_stop()
            return True, self._frame("q")
        kind = self.seq[self.i]
        f = self._frame(kind)
        self.i += 1
        self.clock.tick()
        return True, f
    def _frame(self, kind):
        f = np.full((120, 160, 3), 60, dtype=np.uint8)
        if kind == "m":
            x = 10 + (self.i % 30) * 3
            f[30:90, x:x + 40] = 220
        return f
    def close(self):
        pass

def build(tmp_path):
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15,
                 sleep_onset_minutes=0.5)   # asleep after 30s quiet
    clock = FakeClock()
    cam = FakeCamera(clock)
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
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg

def test_synthetic_night_produces_one_movement_event(tmp_path):
    m, st, cfg = build(tmp_path)
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id) if e["kind"] == "movement"]
    assert len(evs) == 1
    ev = evs[0]
    assert 2.0 < ev["duration_s"] < 4.0
    assert ev["clip_path"] is not None
    assert Path(ev["clip_path"]).exists()
    night = st.latest_night()
    assert night["end_ts"] is not None

def test_report_generated_on_stop(tmp_path):
    m, st, cfg = build(tmp_path)
    m.run()
    reports = list((cfg.data_dir / "reports").glob("*.html"))
    assert len(reports) == 1

class ScriptCamera(FakeCamera):
    """FakeCamera with an arbitrary quiet/move frame script."""
    def __init__(self, clock, script):
        super().__init__(clock, quiet1=0, move=0, quiet2=0)
        self.seq = []
        for kind, n in script:
            self.seq += [kind] * n
        self.i = 0

def test_settling_before_sleep_onset(tmp_path):
    """Movements before sustained quiet are 'settling': no clips, and the
    first movement after onset is a real 'movement' with a clip."""
    from sleepwatch.storage import Storage
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15,
                 sleep_onset_minutes=0.5)          # 30s onset for the test
    clock = FakeClock()
    #                    10s quiet   3s move      40s quiet    3s move     10s quiet
    cam = ScriptCamera(clock, [("q", 150), ("m", 45), ("q", 600),
                               ("m", 45), ("q", 150)])
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
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] in ("settling", "movement")]
    assert [e["kind"] for e in evs] == ["settling", "movement"]
    assert evs[0]["clip_path"] is None
    assert evs[1]["clip_path"] is not None
    night = st.latest_night()
    assert night["sleep_onset_ts"] is not None
    # onset lands inside the long quiet stretch, before the real movement
    assert evs[0]["end_ts"] < night["sleep_onset_ts"] < evs[1]["start_ts"]


class SwappingCamera(FakeCamera):
    """FakeCamera that flips persistence on/off at given frame indices."""
    def __init__(self, clock, begin_at, end_at, storage_factory,
                 quiet1=600, move=45, quiet2=300, move2=45, quiet3=150):
        super().__init__(clock, quiet1=quiet1, move=move, quiet2=quiet2)
        self.seq += ["m"] * move2 + ["q"] * quiet3
        self.begin_at, self.end_at = begin_at, end_at
        self.storage_factory = storage_factory
    def read(self):
        if self.i == self.begin_at:
            self.monitor.begin_persist(self.storage_factory())
        if self.i == self.end_at:
            self.monitor.end_persist()
        return super().read()

def test_session_swap_records_only_between_begin_and_end(tmp_path):
    from sleepwatch.storage import NullStorage, Storage
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15,
                 sleep_onset_minutes=0.25)  # asleep 15s after persist begins
    clock = FakeClock()
    db = cfg.data_dir / "sleepwatch.db"
    real_storage = Storage(db)
    # First movement burst at frames 600-645 happens PERSISTED
    # (begin at 300); second burst at 945-990 happens after end (900).
    cam = SwappingCamera(clock, begin_at=300, end_at=900,
                         storage_factory=lambda: real_storage)
    m = Monitor(
        config=cfg, storage=NullStorage(), camera=cam,
        scorer=MotionScorer(alpha=cfg.alpha),
        baseline=AdaptiveBaseline(window_seconds=cfg.window_seconds,
                                  fps=cfg.fps, k=cfg.k,
                                  min_threshold=cfg.min_threshold,
                                  warmup_seconds=cfg.warmup_seconds),
        fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
        ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
        clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    m.run()
    nights = real_storage.conn.execute("SELECT * FROM nights").fetchall()
    assert len(nights) == 1
    assert nights[0]["end_ts"] is not None       # closed by end_persist
    evs = [e for e in real_storage.events_for_night(nights[0]["id"])
           if e["kind"] == "movement"]
    assert len(evs) == 1                          # only the persisted burst
    assert m.persisting is False

def test_auto_stop_ends_persisted_night(tmp_path):
    from sleepwatch.storage import NullStorage, Storage
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=2.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15,
                 sleep_onset_minutes=0.1)
    clock = FakeClock()
    real_storage = Storage(cfg.data_dir / "sleepwatch.db")
    # persist begins at frame 30; auto_stop 0.01h = 36s = 540 frames
    cam = SwappingCamera(clock, begin_at=30, end_at=10 ** 9,
                         storage_factory=lambda: real_storage,
                         quiet1=1200, move=0, quiet2=0, move2=0, quiet3=0)
    m = Monitor(
        config=cfg, storage=NullStorage(), camera=cam,
        scorer=MotionScorer(alpha=cfg.alpha),
        baseline=AdaptiveBaseline(window_seconds=cfg.window_seconds,
                                  fps=cfg.fps, k=cfg.k,
                                  min_threshold=cfg.min_threshold,
                                  warmup_seconds=cfg.warmup_seconds),
        fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
        ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
        clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
        auto_stop_hours=0.01,
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    m.run()
    night = real_storage.latest_night()
    assert night is not None and night["end_ts"] is not None
    # closed by auto-stop mid-run, not by the final shutdown
    assert night["end_ts"] < clock.t

def test_null_storage_test_mode_saves_nothing(tmp_path):
    from sleepwatch.storage import NullStorage
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15)
    clock = FakeClock()
    cam = FakeCamera(clock)
    m = Monitor(
        config=cfg, storage=NullStorage(), camera=cam,
        scorer=MotionScorer(alpha=cfg.alpha),
        baseline=AdaptiveBaseline(window_seconds=cfg.window_seconds,
                                  fps=cfg.fps, k=cfg.k,
                                  min_threshold=cfg.min_threshold,
                                  warmup_seconds=cfg.warmup_seconds),
        fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
        ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
        clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
        clips_enabled=False,
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    m.run()
    assert not (cfg.data_dir / "sleepwatch.db").exists()
    assert not list((cfg.data_dir / "clips").rglob("*.mp4")) \
        if (cfg.data_dir / "clips").exists() else True
    assert not (cfg.data_dir / "reports").exists()

def build_scripted(tmp_path, script, postroll=2.0):
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0,
                 postroll_seconds=postroll, fps=15,
                 sleep_onset_minutes=0.5)
    clock = FakeClock()
    cam = ScriptCamera(clock, script)
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
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg

def test_postroll_extends_clip_and_records_duration(tmp_path):
    # 60s quiet (warmup+onset) · 3s move · 20s quiet
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "movement"]
    assert len(evs) == 1
    dur = evs[0]["clip_duration_s"]
    assert dur is not None
    # preroll 1 + move ~3 + cooldown 3 + postroll 2 ≈ 9s
    assert 7.0 < dur < 11.0
    import cv2
    cap = cv2.VideoCapture(evs[0]["clip_path"])
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert abs(n / cfg.fps - dur) < 1e-6

def test_retrigger_in_postroll_merges_into_one_clip(tmp_path):
    # close at move1+3s cooldown; postroll deadline +2s; move2 1s after close
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 60), ("m", 45), ("q", 300)])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "movement"]
    assert len(evs) == 2
    assert evs[0]["clip_path"] == evs[1]["clip_path"]
    assert evs[0]["clip_duration_s"] == evs[1]["clip_duration_s"] is not None
    clips = list((cfg.data_dir / "clips").rglob("*.mp4"))
    assert len(clips) == 1

def test_movements_apart_get_separate_clips(tmp_path):
    # 10s gap > cooldown 3 + postroll 2 → two files
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 150), ("m", 45), ("q", 300)])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "movement"]
    assert len(evs) == 2
    assert evs[0]["clip_path"] != evs[1]["clip_path"]
    clips = list((cfg.data_dir / "clips").rglob("*.mp4"))
    assert len(clips) == 2

def test_stop_during_postroll_still_sets_duration(tmp_path):
    # camera script ends 1s after event close — inside the 2s postroll
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 60)])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "movement"]
    assert len(evs) == 1
    assert evs[0]["clip_duration_s"] is not None


class FakeAudio:
    """Scripted audio bundle: (mono_time, item) events + silent live tap."""
    def __init__(self, clock, events=()):
        import numpy as np
        from sleepwatch.audio import AudioRing
        self.clock = clock
        self.ring = AudioRing(seconds=1.0)
        self._events = sorted(events, key=lambda e: e[0])
        self._chunk = np.zeros(512, dtype="int16")

    def take_peak_prob(self):
        return None

    def take_chunks(self):
        ts = self.clock.mono()
        self.ring.push(self._chunk, ts)
        return [(self._chunk, ts)]

    def drain_events(self):
        out = []
        while self._events and self._events[0][0] <= self.clock.mono():
            out.append(self._events.pop(0)[1])
        return out


def build_audio_scripted(tmp_path, script, audio_events, postroll=2.0):
    from sleepwatch.config import Config as _C
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0,
                 postroll_seconds=postroll, fps=15,
                 sleep_onset_minutes=0.5)
    clock = FakeClock()
    cam = ScriptCamera(clock, script)
    st = Storage(cfg.data_dir / "sleepwatch.db")
    audio = FakeAudio(clock, audio_events)
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
        audio=audio,
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg


def test_speech_without_movement_triggers_clip(tmp_path):
    from sleepwatch.audio import SpeechEvent
    # FakeClock starts at 1000; asleep by ~1030; speech 1062→1064.5
    m, st, cfg = build_audio_scripted(
        tmp_path, [("q", 1200)],
        [(1062.0, ("started", 1062.0)),
         (1064.5, SpeechEvent(start_wall=1062.0, end_wall=1064.5,
                              duration_s=2.5, peak_prob=0.93))])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "speech"]
    assert len(evs) == 1
    ev = evs[0]
    assert ev["clip_path"] is not None
    assert Path(ev["clip_path"]).exists()
    assert ev["clip_duration_s"] is not None
    assert ev["peak_score"] == 0.93
    wavs = list((cfg.data_dir / "clips").rglob("*.wav"))
    assert wavs == []                              # muxed or cleaned up


def test_movement_clip_gets_audio_muxed(tmp_path):
    import shutil as _sh
    import subprocess
    m, st, cfg = build_audio_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)], [])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "movement"]
    assert len(evs) == 1
    assert list((cfg.data_dir / "clips").rglob("*.wav")) == []
    if _sh.which("ffprobe"):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             evs[0]["clip_path"]], capture_output=True, text=True)
        assert "audio" in probe.stdout


def test_speech_before_onset_is_clipless(tmp_path):
    from sleepwatch.audio import SpeechEvent
    # onset needs 30s sustained quiet → speech at mono 1020 is still awake
    m, st, cfg = build_audio_scripted(
        tmp_path, [("q", 1200)],
        [(1020.0, ("started", 1020.0)),
         (1022.0, SpeechEvent(start_wall=1020.0, end_wall=1022.0,
                              duration_s=2.0, peak_prob=0.9))])
    night_id = m.run()
    evs = [e for e in st.events_for_night(night_id)
           if e["kind"] == "speech"]
    assert len(evs) == 1
    assert evs[0]["clip_path"] is None


def build_attributed(tmp_path, script, person_box):
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0,
                 postroll_seconds=2.0, fps=15,
                 sleep_onset_minutes=0.5)
    clock = FakeClock()
    cam = ScriptCamera(clock, script)
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
        person_box=person_box,
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg


def test_motion_away_from_person_is_ambient(tmp_path):
    # fake square lives at x 10-97/160, y 30-90/120; person parked far corner
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)],
        person_box=lambda: (0.85, 0.85, 0.12, 0.12))
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert len([e for e in evs if e["kind"] == "ambient"]) == 1
    assert [e for e in evs if e["kind"] == "movement"] == []
    amb = [e for e in evs if e["kind"] == "ambient"][0]
    assert amb["clip_path"] is not None            # evidence clip kept


def test_motion_on_person_stays_movement(tmp_path):
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)],
        person_box=lambda: (0.0, 0.2, 0.7, 0.6))   # covers the square zone
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert len([e for e in evs if e["kind"] == "movement"]) == 1
    assert [e for e in evs if e["kind"] == "ambient"] == []


def test_ambient_does_not_feed_wake_burst(tmp_path):
    # 4 rapid ambient events must NOT end the night
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 150), ("m", 45), ("q", 150),
                   ("m", 45), ("q", 150), ("m", 45), ("q", 300)],
        person_box=lambda: (0.85, 0.85, 0.12, 0.12))
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    assert len([e for e in evs if e["kind"] == "ambient"]) == 4


def test_ambient_events_feed_nuisance_map(tmp_path):
    import json
    m, st, cfg = build_attributed(
        tmp_path, [("q", 900), ("m", 45), ("q", 300)],
        person_box=lambda: (0.85, 0.85, 0.12, 0.12))
    m.run()
    map_path = cfg.data_dir / "nuisance_map.json"
    assert map_path.exists()
    cells = json.loads(map_path.read_text())
    assert cells                                   # square's cells recorded
    assert all(0 < v < 1.0 for v in cells.values())  # 1 hit, then night decay


def test_wake_burst_flips_awake_and_keeps_night_open(tmp_path):
    # v2.5: the burst is a state transition, not a termination. A real
    # night died to end_persist here — whole night unrecorded.
    # v2.6: five DISTINCT stirs (>60s apart) inside the window mean awake.
    # v2.7: the burst needs evidence — the last stir is 20s (full).
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 300), ("q", 600)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    kinds = {}
    for e in evs:
        kinds.setdefault(e["kind"], []).append(e)
    assert "wake" in kinds and len(kinds["wake"]) == 1
    assert len(kinds.get("awake", [])) == 5           # burst reclassified
    assert kinds.get("movement", []) == []
    wake_ts = kinds["wake"][0]["start_ts"]
    assert wake_ts == kinds["awake"][0]["start_ts"]   # marker at burst start
    # the night survived: closed at script end, well after the wake
    assert night["end_ts"] > wake_ts + 30


def test_movement_while_awake_is_awake_kind_without_clip(tmp_path):
    # motion after the burst keeps being recorded — 'awake' kind, no clip
    # (same rationale as settling: he is not asleep, gray, stat-excluded)
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 300), ("q", 300), ("m", 45), ("q", 300)])
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    awake = [e for e in evs if e["kind"] == "awake"]
    assert len(awake) == 6                  # 5 reclassified + 1 post-burst
    assert all(a["clip_path"] for a in awake[:5])     # burst kept its clips
    assert awake[-1]["clip_path"] is None             # awake motion: clipless


def test_night5_replay_reonset_opens_second_sleep_segment(tmp_path):
    # false onset → burst → stillness → asleep AGAIN → movement is real.
    # The exact shape that lost a real night.
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 4
                  + [("m", 300), ("q", 600), ("m", 45), ("q", 300)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    onsets = [e for e in evs if e["kind"] == "onset"]
    wakes = [e for e in evs if e["kind"] == "wake"]
    assert len(onsets) == 2
    assert night["sleep_onset_ts"] == onsets[0]["start_ts"]  # column = FIRST
    assert onsets[1]["start_ts"] > wakes[0]["start_ts"]
    moves = [e for e in evs if e["kind"] == "movement"]
    assert len(moves) == 1                       # the post-re-onset one
    assert moves[0]["start_ts"] > onsets[1]["start_ts"]
    assert moves[0]["clip_path"] is not None


def test_first_onset_writes_marker_and_column(tmp_path):
    m, st, cfg = build_scripted(tmp_path, [("q", 900), ("m", 45), ("q", 300)])
    m.run()
    night = st.latest_night()
    onsets = [e for e in st.events_for_night(night["id"])
              if e["kind"] == "onset"]
    assert len(onsets) == 1
    assert onsets[0]["start_ts"] == night["sleep_onset_ts"]


def test_onset_backdated_to_stillness_start(tmp_path):
    # He falls asleep when he STOPS MOVING, not 10 minutes later when the
    # detector is finally sure. Onset is stamped at the start of the quiet
    # stretch, not at confirmation time (night 8: reads 10-20 min late).
    m, st, cfg = build_scripted(tmp_path, [("q", 150), ("m", 45), ("q", 900),
                                           ("m", 45), ("q", 300)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    settling = [e for e in evs if e["kind"] == "settling"][-1]
    onset = [e for e in evs if e["kind"] == "onset"][0]
    # confirmation takes sleep_onset_minutes (30s here); the marker must sit
    # near the settling movement's end, not 30s after it
    lag = onset["start_ts"] - settling["end_ts"]
    assert 0.0 <= lag < 10.0
    assert night["sleep_onset_ts"] == onset["start_ts"]


def test_four_spaced_stirs_do_not_wake(tmp_path):
    # one below the burst threshold: still asleep, still 'movement'
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 1125)] * 3
                  + [("m", 45), ("q", 600)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    assert len([e for e in evs if e["kind"] == "movement"]) == 4


def test_fragmented_turn_counts_as_one_stir(tmp_path):
    # Real-night false wakes: one physical turn fragments into
    # 2-3 event rows seconds apart. Rows inside
    # wake_stir_gap_s of each other are ONE stir — six rows in three
    # clusters must NOT wake.
    cluster = [("m", 45), ("q", 75), ("m", 45), ("q", 1125)]
    m, st, cfg = build_scripted(tmp_path, [("q", 900)] + cluster * 3
                                + [("q", 300)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    assert len([e for e in evs if e["kind"] == "movement"]) == 6


def test_sustained_stir_fires_wake(tmp_path):
    # continuous chained motion (the lying-awake-on-the-phone shape) is not a
    # sleep turn: one stir spanning wake_stir_span_s means awake NOW.
    # v2.7: one 20s move inside the chain is the evidence.
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900)] + [("m", 45), ("q", 75)] * 12
                  + [("m", 300), ("q", 75)] + [("m", 45), ("q", 75)] * 13
                  + [("q", 600)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    wakes = [e for e in evs if e["kind"] == "wake"]
    assert len(wakes) == 1
    # everything in the sustained stir got reclassified awake
    assert [e for e in evs if e["kind"] == "movement"] == []
    assert len([e for e in evs if e["kind"] == "awake"]) >= 2


def test_settling_bursts_never_wake(tmp_path):
    # rapid movements BEFORE sleep onset are settling — no wake
    m, st, cfg = build_scripted(
        tmp_path, [("q", 150), ("m", 45), ("q", 60), ("m", 45), ("q", 60),
                   ("m", 45), ("q", 60), ("m", 45), ("q", 900)])
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert [e for e in evs if e["kind"] == "wake"] == []
    assert len([e for e in evs if e["kind"] == "settling"]) >= 3


def test_signal_minutes_capture_subthreshold_motion(tmp_path):
    # the morning film shows moves the detector never fired on — per-minute
    # peak scores are the tuning data for that gap
    m, st, cfg = build_scripted(
        tmp_path, [("q", 900), ("m", 45), ("q", 1800)])   # 3+ minutes
    m.run()
    night = st.latest_night()
    rows = st.signal_minutes_for_night(night["id"])
    assert len(rows) >= 2                        # at least two full minutes
    assert all(r["threshold"] > 0 for r in rows)
    # the minute containing the movement recorded a real peak
    assert max(r["max_motion"] for r in rows) > 0.01
    # quiet minutes are near zero, NOT missing — absence of motion is data
    assert min(r["max_motion"] for r in rows) < 0.005


def test_timelapse_written_while_persisting(tmp_path):
    import cv2
    # 80s night at 1 frame/2.5s → ~32 frames in the timelapse film
    m, st, cfg = build_scripted(tmp_path, [("q", 1200)])
    m.run()
    night = st.latest_night()
    tl = cfg.data_dir / "timelapse" / f"{night['date']}.mp4"
    assert tl.exists()
    cap = cv2.VideoCapture(str(tl))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert 28 <= n <= 36


def build_absence(tmp_path, script, detector, person_box=None):
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=5.0,
                 window_seconds=20.0, preroll_seconds=1.0,
                 postroll_seconds=2.0, fps=15,
                 sleep_onset_minutes=0.5,
                 auto_stop_absence_minutes=0.05)     # 3s for the test
    clock = FakeClock()
    cam = ScriptCamera(clock, script)
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
        person_detector=detector, presence_every_s=1.0,
        person_box=person_box,
        wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = m
    return m, st, cfg, cam


def test_absence_after_departure_flips_awake_night_stays_open(tmp_path):
    # asleep, departure movement, then gone 45+ min: wake marker stamped
    # at the departure — but the night KEEPS recording (v2.5)
    state = {"cam": None}

    def detector(_frame):
        return state["cam"].i < 950        # absent after the movement

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 900), ("m", 45), ("q", 600)], detector)
    state["cam"] = cam
    m.run()
    night = st.latest_night()
    moves = [e for e in st.events_for_night(night["id"])
             if e["kind"] == "movement"]
    wakes = [e for e in st.events_for_night(night["id"])
             if e["kind"] == "wake"]
    assert len(wakes) == 1
    assert wakes[0]["start_ts"] == moves[-1]["end_ts"]   # departure time
    assert night["end_ts"] >= cam.clock.t - 1            # ran to script end


def test_empty_room_after_absence_wake_never_reonsets(tmp_path):
    # he left; the still empty room must not "fall asleep" again
    state = {"cam": None}

    def detector(_frame):
        return state["cam"].i < 950

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 900), ("m", 45), ("q", 900)], detector)
    state["cam"] = cam
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert len([e for e in evs if e["kind"] == "onset"]) == 1
    assert len([e for e in evs if e["kind"] == "wake"]) == 1


def test_return_after_absence_wake_reonsets(tmp_path):
    # bathroom trip > 45 min: wake, then he comes back, stills, and the
    # night opens a second sleep segment instead of being lost
    state = {"cam": None}

    def detector(_frame):
        return not (950 <= state["cam"].i < 1545)    # gone, then back

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 900), ("m", 45), ("q", 600), ("m", 45), ("q", 600),
                   ("m", 45), ("q", 150)], detector)
    state["cam"] = cam
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    kinds = {}
    for e in evs:
        kinds.setdefault(e["kind"], []).append(e)
    assert len(kinds.get("onset", [])) == 2
    assert len(kinds.get("awake", [])) == 1       # the return movement
    assert len(kinds.get("movement", [])) == 2    # departure + post-re-onset
    assert (kinds["movement"][-1]["start_ts"]
            > kinds["onset"][-1]["start_ts"])


def test_sun_in_empty_room_does_not_enable_reonset(tmp_path):
    # pose configured but sees nobody: motion with no person visible must
    # not count as "he came back" — re-onset stays blocked
    state = {"cam": None}

    def detector(_frame):
        return state["cam"].i < 950

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 900), ("m", 45), ("q", 600), ("m", 45), ("q", 600)],
        detector, person_box=lambda: None)
    state["cam"] = cam
    m.run()
    evs = st.events_for_night(st.latest_night()["id"])
    assert len([e for e in evs if e["kind"] == "onset"]) == 1
    assert len([e for e in evs if e["kind"] == "awake"]) == 1  # sun, gray


def test_settling_departure_blocks_first_onset(tmp_path):
    # start the night, walk out before ever sleeping: the empty room must
    # not phantom-onset 10 minutes later (v2.4 did exactly that)
    state = {"cam": None}

    def detector(_frame):
        return state["cam"].i < 390

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 300), ("m", 45), ("q", 900)], detector)
    state["cam"] = cam
    m.run()
    night = st.latest_night()
    evs = st.events_for_night(night["id"])
    assert night["sleep_onset_ts"] is None
    assert [e for e in evs if e["kind"] == "onset"] == []
    assert [e for e in evs if e["kind"] == "wake"] == []


def test_absence_without_departure_never_arms(tmp_path):
    # pose loses a still sleeper mid-night (blanket) — must NOT end night
    def detector(_frame):
        return False                       # "absent" the whole time

    m, st, cfg, cam = build_absence(
        tmp_path, [("q", 1200)], detector)
    night_id = m.run()
    assert [e for e in st.events_for_night(night_id)
            if e["kind"] == "wake"] == []
    night = st.latest_night()
    assert night["end_ts"] >= cam.clock.t - 1         # ran to script end


def test_short_speech_filtered_but_clip_closes(tmp_path):
    from sleepwatch.audio import SpeechEvent
    m, st, cfg = build_audio_scripted(
        tmp_path, [("q", 1200)],
        [(1062.0, ("started", 1062.0)),
         (1062.4, SpeechEvent(start_wall=1062.0, end_wall=1062.4,
                              duration_s=0.4, peak_prob=0.9))])
    night_id = m.run()
    assert [e for e in st.events_for_night(night_id)
            if e["kind"] == "speech"] == []         # under speech_min_s
    clips = list((cfg.data_dir / "clips").rglob("*.mp4"))
    assert len(clips) == 1                          # clip existed, closed fine
