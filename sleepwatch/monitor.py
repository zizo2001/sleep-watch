from __future__ import annotations

import json
import shutil
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2

from .audio import AudioClipWriter, SpeechEvent, mux_audio
from .baseline import AdaptiveBaseline
from .clipwriter import ClipWriter, FrameRingBuffer
from .config import Config
from .detector import MotionScorer
from .events import EventStateMachine, MovementEvent
from .nuisance import NuisanceMap
from .report import generate_report
from .storage import NullStorage, Storage

LIGHTING_LOG_INTERVAL_S = 60.0
CAMERA_RETRY_SLEEP_S = 0.5
CAMERA_GIVEUP_S = 60.0
DISK_CHECK_EVERY_FRAMES = 900
MAGNITUDE_PARTIAL_S = 8.0    # duration fallback when pose never saw the event
MAGNITUDE_FULL_S = 20.0


def classify_magnitude(coverage: float | None, duration_s: float,
                       cfg: Config) -> str:
    """twitch / partial / full — coverage (motion∪ ÷ person area) when pose
    saw the event, duration as the blanket-blind fallback."""
    if ((coverage is not None and coverage >= cfg.coverage_full_frac)
            or duration_s >= MAGNITUDE_FULL_S):
        return "full"
    if ((coverage is not None and coverage >= cfg.coverage_partial_frac)
            or duration_s >= MAGNITUDE_PARTIAL_S):
        return "partial"
    return "twitch"


class Monitor:
    def __init__(self, config: Config, storage: Storage, camera,
                 scorer: MotionScorer, baseline: AdaptiveBaseline,
                 fsm: EventStateMachine, ring: FrameRingBuffer,
                 clip: ClipWriter, clips_enabled: bool = True,
                 auto_stop_hours: float | None = None,
                 audio=None, person_detector=None, person_box=None,
                 presence_every_s: float = 30.0,
                 wall_clock=time.time,
                 mono_clock=time.monotonic, on_frame=None):
        self.cfg = config
        self.storage = storage
        self.camera = camera
        self.scorer = scorer
        self.baseline = baseline
        self.fsm = fsm
        self.ring = ring
        self.clip = clip
        self.wall = wall_clock
        self.mono = mono_clock
        self.on_frame = on_frame
        self._stop = False
        self._clips_enabled = clips_enabled
        self._night_id = 0
        self._night_date = ""
        self._swap_lock = threading.Lock()
        self._pending_storage = None
        self._asleep = False
        self._night_start_mono = 0.0
        self._last_motion_mono: float | None = None
        self._clip_close_mono: float | None = None
        self.auto_stop_hours = auto_stop_hours
        self.audio = audio
        self._audio_clip = AudioClipWriter()
        self._audio_from_mono = 0.0
        self.person_detector = person_detector
        self.presence_every_s = presence_every_s
        self._presence_checked_mono = -1e12
        self._absent_since_mono: float | None = None
        self._last_move_end_wall: float | None = None
        self._last_move_end_mono: float | None = None
        self._woke = False        # slept, then woke — night stays open
        self._had_onset = False   # sleep_onset_ts column = first onset only
        self._tl_writer: cv2.VideoWriter | None = None
        self._tl_last_mono = -1e12
        self._sig_minute: float | None = None   # per-minute signal bucket
        self._sig_max = 0.0
        # one dict per STIR (movement rows chained within wake_stir_gap_s):
        # {start_wall, end_wall, last_mono, ids} — wake burst counts stirs
        self._wake_deque: deque[dict] = deque()
        self.person_box = person_box
        self._mv_frames = 0          # motion frames where pose saw a person
        self._mv_hits = 0            # …of those, overlapping the person
        self._mv_motion_frames = 0   # ALL motion-box frames this event
        self._mv_person_seen = False
        self._mv_person_area = 0.0   # largest person box seen this event
        self._mv_union: tuple[float, float, float, float] | None = None
        self._last_lighting_wall: float | None = None
        self._pending_demote: dict | None = None   # burst wake awaiting onset
        self._last_restless: dict | None = None    # merge adjacent restless
        self.nuisance = NuisanceMap(config.data_dir / "nuisance_map.json")

    def request_stop(self) -> None:
        self._stop = True

    @property
    def persisting(self) -> bool:
        return not isinstance(self.storage, NullStorage)

    def begin_persist(self, storage: Storage) -> None:
        """Swap to a persisted night at the next loop iteration."""
        with self._swap_lock:
            self._pending_storage = storage

    def end_persist(self) -> None:
        """Swap back to idle (NullStorage) at the next loop iteration."""
        with self._swap_lock:
            self._pending_storage = NullStorage()

    def _apply_pending_swap(self) -> None:
        with self._swap_lock:
            pending, self._pending_storage = self._pending_storage, None
        if pending is None:
            return
        self._close_night()
        self.storage = pending
        self._begin_night()

    def _begin_night(self) -> None:
        self._asleep = False
        self._woke = False
        self._had_onset = False
        self._night_start_mono = self.mono()
        self._last_motion_mono = None
        self._absent_since_mono = None
        self._last_move_end_wall = None
        self._last_move_end_mono = None
        self._tl_last_mono = -1e12
        self._sig_minute = None
        self._sig_max = 0.0
        self._wake_deque.clear()
        self._last_lighting_wall = None
        self._pending_demote = None
        self._last_restless = None
        self._reset_attribution()      # never carry a union across nights
        self._night_date = time.strftime("%Y-%m-%d",
                                         time.localtime(self.wall()))
        cfg_json = json.dumps(asdict(self.cfg), default=str)
        self._night_id = self.storage.begin_night(self._night_date,
                                                  self.wall(), cfg_json)
        self.storage.purge_old_clips(self.cfg.data_dir / "clips",
                                     self.cfg.retention_days)
        # exports are regenerable; timelapse films are deliberately kept —
        # the morning film is the artifact worth hoarding (~28 MB/night)
        self.storage.purge_old_clips(self.cfg.data_dir / "exports",
                                     self.cfg.retention_days)

    def _start_audio_clip(self, mono_ts: float) -> None:
        if self.audio is None or self._audio_clip.active:
            return
        self._audio_clip.start(self.audio.ring.dump())
        self._audio_from_mono = mono_ts

    def _finish_clip(self, closed) -> None:
        """Common tail after ClipWriter.close(): mux audio, record truth."""
        if closed is None:
            self._audio_clip = AudioClipWriter()      # drop stray audio
            return
        if self.audio is not None:
            wav = self._audio_clip.close(closed.path.with_suffix(".wav"))
            if wav is not None:
                mux_audio(closed.path, wav)
        self.storage.set_clip_duration(str(closed.path), closed.duration_s)

    def _handle_audio(self, mono_ts: float) -> None:
        if self.audio is None:
            return
        for ch, ts in self.audio.take_chunks():
            if self._audio_clip.active and ts > self._audio_from_mono:
                self._audio_clip.push(ch)
        for item in self.audio.drain_events():
            if isinstance(item, SpeechEvent):
                self._on_speech_event(item, mono_ts)
            else:                                     # ("started", start_wall)
                self._on_speech_started(item[1], mono_ts)

    def _on_speech_started(self, start_wall: float, mono_ts: float) -> None:
        if not (self._clips_enabled and self.persisting and self._asleep):
            return
        if self.clip.active:
            self._clip_close_mono = None              # merge into open clip
        else:
            self.clip.start(self._night_date, start_wall, self.ring.dump())
            self._start_audio_clip(mono_ts)

    def _stamp_clock(self, frame, wall_ts: float):
        """Real time burned into every recorded pixel — film and clips both.
        Returns a stamped copy; the raw frame stays clean for the scorer
        and the mirrored live view."""
        stamped = frame.copy()
        label = time.strftime("%H:%M", time.localtime(wall_ts))
        h = frame.shape[0]
        scale = max(h / 720.0, 0.5)
        org = (int(18 * scale), int(46 * scale))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(stamped, label, org, font, scale, (0, 0, 0),
                    max(int(4 * scale), 3), cv2.LINE_AA)
        cv2.putText(stamped, label, org, font, scale, (255, 255, 255),
                    max(int(2 * scale), 1), cv2.LINE_AA)
        return stamped

    def _write_timelapse(self, frame, mono_ts: float) -> None:
        """One frame every timelapse_every_s → the night's morning film."""
        if not self.persisting:
            return
        if mono_ts - self._tl_last_mono < self.cfg.timelapse_every_s:
            return
        self._tl_last_mono = mono_ts
        if self._tl_writer is None:
            tl_dir = self.cfg.data_dir / "timelapse"
            tl_dir.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            path = tl_dir / f"{self._night_date}.mp4"
            for fourcc in ("avc1", "mp4v"):
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*fourcc), 24, (w, h))
                if writer.isOpened():
                    self._tl_writer = writer
                    break
                writer.release()
            if self._tl_writer is None:
                self._tl_last_mono = 1e18       # give up for this night
                return
        self._tl_writer.write(frame)

    def _track_signal(self, score: float, wall_ts: float) -> None:
        """Per-minute peak of the damped motion score — the tuning record
        for moves the detector never fired on (visible in the morning film,
        absent from events). Speech prob rides along for VAD calibration."""
        if not self.persisting or not self.baseline.ready:
            return
        minute = wall_ts - (wall_ts % 60.0)
        if self._sig_minute is None:
            self._sig_minute, self._sig_max = minute, score
        elif minute != self._sig_minute:
            self._flush_signal_minute()
            self._sig_minute, self._sig_max = minute, score
        elif score > self._sig_max:
            self._sig_max = score

    def _flush_signal_minute(self) -> None:
        if self._sig_minute is None:
            return
        speech = (self.audio.take_peak_prob()
                  if self.audio is not None else None)
        self.storage.add_signal_minute(self._night_id, self._sig_minute,
                                       self._sig_max,
                                       self.baseline.threshold, speech)
        self._sig_minute = None
        self._sig_max = 0.0

    def _track_attribution(self, res) -> None:
        """Accumulate motion↔person overlap while an event is running.
        The union box grows with or without pose — it feeds coverage and
        the nuisance map; the overlap counters need a person box."""
        if (not self.fsm.in_movement
                or res.box is None or res.frame_size == (0, 0)):
            return
        fw, fh = res.frame_size
        mx, my, mw, mh = res.box
        m = (mx / fw, my / fh, mw / fw, mh / fh)
        self._mv_motion_frames += 1
        if self._mv_union is None:
            self._mv_union = m
        else:
            u = self._mv_union
            x0, y0 = min(u[0], m[0]), min(u[1], m[1])
            x1 = max(u[0] + u[2], m[0] + m[2])
            y1 = max(u[1] + u[3], m[1] + m[3])
            self._mv_union = (x0, y0, x1 - x0, y1 - y0)
        if self.person_box is None:
            return
        pb = self.person_box()
        if pb is None:
            return
        self._mv_person_seen = True
        self._mv_frames += 1
        self._mv_person_area = max(self._mv_person_area, pb[2] * pb[3])
        pad = self.cfg.person_box_pad     # blanket > skeleton
        px, py = pb[0] - pb[2] * pad, pb[1] - pb[3] * pad
        pw, ph = pb[2] * (1 + 2 * pad), pb[3] * (1 + 2 * pad)
        if (m[0] < px + pw and px < m[0] + m[2]
                and m[1] < py + ph and py < m[1] + m[3]):
            self._mv_hits += 1

    def _event_is_ambient(self) -> bool:
        """Motion that never touched the person = sun/curtain/etc. v2.7:
        only when pose watched most of the event — a blanket lump with a
        flickering skeleton is the sleeper, not the sun."""
        if not self._mv_person_seen or self._mv_frames == 0:
            return False
        if (self._mv_frames / max(self._mv_motion_frames, 1)
                < self.cfg.ambient_pose_min_frac):
            return False
        return (self._mv_hits / self._mv_frames
                < self.cfg.ambient_overlap_frac)

    def _event_coverage(self) -> float | None:
        """Motion-union area over person area; None unless pose actually
        watched the event — a single lucky frame is not a measurement."""
        if (not self._mv_person_seen or self._mv_union is None
                or self._mv_person_area <= 0):
            return None
        if (self._mv_frames / max(self._mv_motion_frames, 1)
                < self.cfg.ambient_pose_min_frac):
            return None
        u = self._mv_union
        return min(u[2] * u[3] / self._mv_person_area, 1.5)

    def _reset_attribution(self) -> None:
        self._mv_frames = 0
        self._mv_hits = 0
        self._mv_motion_frames = 0
        self._mv_person_seen = False
        self._mv_person_area = 0.0
        self._mv_union = None

    def _check_wake_burst(self, start_wall: float, end_wall: float,
                          mono_ts: float, event_id: int,
                          duration_s: float = 0.0,
                          magnitude: str = "twitch") -> None:
        """Sleep onset, inverted: distinct stirs while 'asleep' mean the
        sleeper woke up. v2.5: transition to awake, keep the night open — stillness
        re-onsets a new sleep segment later. v2.6: rows within wake_stir_gap_s chain into ONE stir so restless sleep
        doesn't impersonate waking. A single stir spanning wake_stir_span_s
        is sustained activity (the lying-awake-on-the-phone shape). v2.7 (false
        wakes on twitch bursts): a burst alone can no
        longer claim wake — it needs evidence a camera can trust: one big
        move (full-body or ≥ wake_evidence_min_s) or a light coming on.
        Evidence-free bursts log a `restless` span and stay asleep."""
        if (self._wake_deque and start_wall - self._wake_deque[-1]["end_wall"]
                <= self.cfg.wake_stir_gap_s):
            stir = self._wake_deque[-1]
            stir["end_wall"] = end_wall
            stir["last_mono"] = mono_ts
            stir["ids"].append(event_id)
            stir["max_dur"] = max(stir["max_dur"], duration_s)
            stir["full"] = stir["full"] or magnitude == "full"
        else:
            self._wake_deque.append({"start_wall": start_wall,
                                     "end_wall": end_wall,
                                     "last_mono": mono_ts,
                                     "ids": [event_id],
                                     "max_dur": duration_s,
                                     "full": magnitude == "full"})
        window_s = self.cfg.wake_window_minutes * 60.0
        while (self._wake_deque
               and self._wake_deque[0]["last_mono"] < mono_ts - window_s):
            self._wake_deque.popleft()
        last = self._wake_deque[-1]
        sustained = (last["end_wall"] - last["start_wall"]
                     >= self.cfg.wake_stir_span_s)
        if len(self._wake_deque) < self.cfg.wake_burst_stirs and not sustained:
            return
        burst_start = self._wake_deque[0]["start_wall"]
        saw_full = any(d["full"] for d in self._wake_deque)
        evidence = (saw_full
                    or any(d["max_dur"] >= self.cfg.wake_evidence_min_s
                           for d in self._wake_deque)
                    or (self._last_lighting_wall is not None
                        and self._last_lighting_wall >= burst_start))
        ids = [i for d in self._wake_deque for i in d["ids"]]
        if not evidence:
            lr = self._last_restless
            if (lr is not None and burst_start - lr["end_wall"]
                    <= self.cfg.wake_stir_gap_s):
                # the same restless phase keeps re-firing: extend, don't spam
                self.storage.update_event_end(
                    lr["id"], end_wall, end_wall - lr["start_wall"])
                lr["end_wall"] = end_wall
            else:
                rid = self.storage.add_event(
                    self._night_id, "restless", burst_start, end_wall,
                    end_wall - burst_start, None, None, None)
                self._last_restless = {"id": rid, "start_wall": burst_start,
                                       "end_wall": end_wall}
            self._wake_deque.clear()
            return
        self.storage.reclassify_events(ids, "awake")
        wake_id = self.storage.add_event(self._night_id, "wake", burst_start,
                                         burst_start, 0.0, None, None, None)
        self._wake_deque.clear()
        self._asleep = False
        self._woke = True
        # armed for hindsight demotion: quick re-onset with nothing big and
        # no light after the wake means it was restless sleep after all
        self._pending_demote = {"wake_ts": burst_start, "wake_id": wake_id,
                                "ids": ids, "saw_full": saw_full,
                                "end_wall": end_wall}

    def _check_absence(self, frame, mono_ts: float) -> None:
        """Absence rule: out of bed 45+ min (anchored at a departure
        movement) means the sleeper woke up. Person reappearing disarms. v2.5:
        runs asleep OR awake — while armed it also blocks re-onset, so an
        empty room can never "fall asleep"."""
        if self.person_detector is None or not self.persisting:
            return
        if mono_ts - self._presence_checked_mono < self.presence_every_s:
            return
        self._presence_checked_mono = mono_ts
        present = self.person_detector(frame)
        if present is None:
            return
        if present:
            self._absent_since_mono = None
            return
        if self._absent_since_mono is None:
            recently_moved = (
                self._last_move_end_mono is not None
                and mono_ts - self._last_move_end_mono <= 120.0)
            if recently_moved:
                self._absent_since_mono = mono_ts
            return
        if (self._asleep and mono_ts - self._absent_since_mono
                >= self.cfg.auto_stop_absence_minutes * 60.0):
            wake_ts = self._last_move_end_wall or self.wall()
            self.storage.add_event(self._night_id, "wake", wake_ts,
                                   wake_ts, 0.0, None, None, None)
            self._asleep = False
            self._woke = True
            self._pending_demote = None    # left the bed — never demoted

    def _on_speech_event(self, ev: SpeechEvent, mono_ts: float) -> None:
        if not self.persisting:
            return
        clip_path = self.clip.path if self._asleep else None
        if ev.duration_s >= self.cfg.speech_min_s:
            self.storage.add_event(
                self._night_id, "speech", ev.start_wall, ev.end_wall,
                ev.duration_s, ev.peak_prob, None,
                str(clip_path) if clip_path else None)
        if self.clip.active:
            self._clip_close_mono = mono_ts + self.cfg.postroll_seconds

    def _close_night(self) -> None:
        leftover = self.fsm.force_close(self.wall(), self.mono())
        closed = self.clip.close()
        self._clip_close_mono = None
        if leftover is not None:
            coverage = self._event_coverage()
            self.storage.add_event(
                self._night_id, "truncated", leftover.start_wall,
                leftover.end_wall, leftover.duration_s,
                leftover.peak_score, leftover.mean_score,
                str(closed.path) if closed else None, coverage,
                classify_magnitude(coverage, leftover.duration_s, self.cfg))
        self._reset_attribution()
        if self._tl_writer is not None:
            self._tl_writer.release()
            self._tl_writer = None
        self._flush_signal_minute()            # partial last minute is data too
        if self.persisting:                    # nightly learning tick
            self.nuisance.decay()
            self.nuisance.save()
        self._finish_clip(closed)
        self.storage.end_night(self._night_id, self.wall())
        self._write_report(self._night_id)

    def _state_str(self) -> str:
        if not self.baseline.ready:
            return "CALIBRATING"
        if self.fsm.in_movement:
            return "MOVEMENT"
        if self.persisting:
            if self._asleep:
                return "SLEEPING"
            return "AWAKE" if self._woke else "SETTLING"
        return "QUIET"

    def _check_disk(self) -> None:
        p = Path(self.cfg.data_dir).resolve()
        while not p.exists() and p != p.parent:
            p = p.parent
        free_gb = shutil.disk_usage(str(p)).free / 1e9
        if free_gb < self.cfg.min_free_gb:
            self._clips_enabled = False

    def _handle_camera_gap(self, night_id: int) -> bool:
        """Retry until recovery or give-up. Returns True if recovered."""
        gap_start_wall, gap_start_mono = self.wall(), self.mono()
        while not self._stop:
            time.sleep(CAMERA_RETRY_SLEEP_S)
            ok, _ = self.camera.read()
            if ok:
                self.storage.add_event(night_id, "gap", gap_start_wall,
                                       self.wall(),
                                       self.mono() - gap_start_mono,
                                       None, None, None)
                return True
            if self.mono() - gap_start_mono > CAMERA_GIVEUP_S:
                break
        self.storage.add_event(night_id, "gap", gap_start_wall, self.wall(),
                               self.mono() - gap_start_mono, None, None, None)
        return False

    def run(self) -> int:
        self._begin_night()
        last_lighting_log = -1e12
        frames = 0
        try:
            while not self._stop:
                self._apply_pending_swap()
                if (self.auto_stop_hours is not None and self.persisting
                        and (self.mono() - self._night_start_mono
                             > self.auto_stop_hours * 3600)):
                    self.end_persist()
                    self._apply_pending_swap()
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    if not self._handle_camera_gap(self._night_id):
                        break
                    continue
                frames += 1
                if frames % DISK_CHECK_EVERY_FRAMES == 0:
                    self._check_disk()
                wall_ts, mono_ts = self.wall(), self.mono()
                self._handle_audio(mono_ts)
                res = self.scorer.score(frame)
                if res.lighting:
                    self._last_lighting_wall = wall_ts   # wake evidence
                    if mono_ts - last_lighting_log >= LIGHTING_LOG_INTERVAL_S:
                        last_lighting_log = mono_ts
                        self.storage.add_event(self._night_id, "lighting",
                                               wall_ts, wall_ts, 0.0,
                                               None, None, None)
                    continue
                score = res.score
                if res.box is not None and res.frame_size != (0, 0):
                    fw, fh = res.frame_size
                    bx, by, bw, bh = res.box
                    score *= self.nuisance.damp_factor(
                        (bx / fw, by / fh, bw / fw, bh / fh))
                self.baseline.update(score, in_movement=self.fsm.in_movement)
                self._track_signal(score, wall_ts)
                # recorded pixels carry the clock; raw frame stays clean for
                # the scorer background, pose presence and the mirrored live
                # view. Stamped unconditionally — the ring fills while idle,
                # so a night's first pre-roll would otherwise be half-clocked
                rec = self._stamp_clock(frame, wall_ts)
                self.ring.push(rec)
                self._track_attribution(res)
                out = self.fsm.update(score, self.baseline.threshold,
                                     wall_ts, mono_ts)
                if not self.fsm.in_movement and not isinstance(
                        out, MovementEvent):
                    # PENDING aborted below trigger_seconds: no event, so the
                    # close branch never fires. Drop the half-built union or
                    # it leaks into the next event's coverage — a noise blip
                    # would manufacture "full-body" and fake wake evidence.
                    self._reset_attribution()
                if self.fsm.in_movement:
                    self._last_motion_mono = mono_ts
                elif (self.persisting and not self._asleep
                      and self.baseline.ready
                      and self._absent_since_mono is None):
                    # armed absence blocks (re-)onset: an empty room is
                    # still, but it is not asleep
                    anchor = (self._last_motion_mono
                              if self._last_motion_mono is not None
                              else self._night_start_mono)
                    if (mono_ts - anchor
                            >= self.cfg.sleep_onset_minutes * 60):
                        self._asleep = True
                        # he fell asleep when he stopped moving, not when
                        # the detector became sure — backdate to quiet start
                        onset_wall = wall_ts - (mono_ts - anchor)
                        if not self._had_onset:
                            self._had_onset = True
                            self.storage.set_sleep_onset(self._night_id,
                                                         onset_wall)
                        self.storage.add_event(self._night_id, "onset",
                                               onset_wall, onset_wall, 0.0,
                                               None, None, None)
                        pend = self._pending_demote
                        if (pend is not None
                                and onset_wall - pend["wake_ts"]
                                <= self.cfg.wake_demote_minutes * 60
                                and not pend["saw_full"]
                                and (self._last_lighting_wall is None
                                     or self._last_lighting_wall
                                     < pend["wake_ts"])):
                            # back asleep fast, nothing big, no light —
                            # that "wake" was restless sleep. Repair labels.
                            self.storage.reclassify_events(
                                [pend["wake_id"]], "restless")
                            self.storage.reclassify_events(
                                pend["ids"], "movement")
                            # a wake is a point marker; restless is a SPAN —
                            # give the demoted row the phase it covers
                            self.storage.update_event_end(
                                pend["wake_id"], pend["end_wall"],
                                pend["end_wall"] - pend["wake_ts"])
                            self._last_restless = {
                                "id": pend["wake_id"],
                                "start_wall": pend["wake_ts"],
                                "end_wall": pend["end_wall"]}
                        self._pending_demote = None
                if out == "started":
                    if self._clips_enabled and self.persisting and self._asleep:
                        if self.clip.active:
                            # re-trigger inside post-roll: keep rolling —
                            # both events share this file
                            self._clip_close_mono = None
                            self.clip.write(rec)
                        else:
                            # ring already holds this frame — dump is the
                            # pre-roll plus the pending frames, so skip
                            # write() this loop.
                            self.clip.start(self._night_date,
                                            self.fsm.start_wall,
                                            self.ring.dump())
                            self._start_audio_clip(mono_ts)
                elif isinstance(out, MovementEvent):
                    kind = ("movement" if self._asleep
                            else "awake" if self._woke else "settling")
                    if kind == "movement" and self._event_is_ambient():
                        kind = "ambient"
                        if self._mv_union is not None:
                            self.nuisance.hit(self._mv_union)
                    person_seen = self._mv_person_seen
                    coverage = self._event_coverage()
                    magnitude = classify_magnitude(coverage, out.duration_s,
                                                   self.cfg)
                    self._reset_attribution()
                    if kind != "ambient":
                        # any human motion re-anchors absence arming and
                        # absence-wake timestamps
                        self._last_move_end_wall = out.end_wall
                        self._last_move_end_mono = mono_ts
                    if kind == "movement":
                        # asleep: unconditional disarm — blanket sleepers
                        # are invisible to pose but still here
                        self._absent_since_mono = None
                    elif kind != "ambient" and (person_seen
                                                or self.person_box is None):
                        # awake/settling: disarm only when pose actually
                        # saw him — sun in an empty room is not a return
                        self._absent_since_mono = None
                    clip_path = self.clip.path
                    eid = self.storage.add_event(
                        self._night_id, kind, out.start_wall,
                        out.end_wall, out.duration_s, out.peak_score,
                        out.mean_score,
                        str(clip_path) if clip_path else None,
                        coverage, magnitude)
                    if kind == "awake" and self._pending_demote is not None:
                        pend = self._pending_demote
                        pend["ids"].append(eid)
                        pend["saw_full"] = (pend["saw_full"]
                                            or magnitude == "full")
                        pend["end_wall"] = out.end_wall   # span, if demoted
                    if self.clip.active:
                        self.clip.write(rec)
                        self._clip_close_mono = (
                            mono_ts + self.cfg.postroll_seconds)
                    if kind == "movement":
                        self._check_wake_burst(out.start_wall, out.end_wall,
                                               mono_ts, eid,
                                               out.duration_s, magnitude)
                elif self.clip.active:
                    self.clip.write(rec)
                    if (self._clip_close_mono is not None
                            and mono_ts >= self._clip_close_mono):
                        closed = self.clip.close()
                        self._clip_close_mono = None
                        self._finish_clip(closed)
                self._write_timelapse(rec, mono_ts)
                self._check_absence(frame, mono_ts)
                if self.on_frame is not None:
                    if self.on_frame(frame, score, self.baseline.threshold,
                                     self._state_str()) is False:
                        self._stop = True
        finally:
            self._close_night()
            self.camera.close()
        return self._night_id

    def _write_report(self, night_id: int) -> Path | None:
        night = self.storage.latest_night()
        if night is None:
            return None
        events = self.storage.events_for_night(night_id)
        minutes = self.storage.signal_minutes_for_night(night_id)
        out = self.cfg.data_dir / "reports" / f"{night['date']}.html"
        try:
            return generate_report(night, events, out,
                                   signal_minutes=minutes)
        except Exception:
            import traceback
            traceback.print_exc()
            return None
