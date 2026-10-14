from __future__ import annotations

import json
import threading
import time

import cv2

from .baseline import AdaptiveBaseline
from .camera import Camera
from .clipwriter import ClipWriter, FrameRingBuffer
from .config import Config
from .detector import MotionScorer
from .events import EventStateMachine
from .monitor import Monitor
from .pose import PoseOverlay
from .storage import NullStorage, Storage

class DashboardRunner:
    """Owns the camera loop in a thread; publishes JPEG frames + status."""

    def __init__(self, cfg: Config, pose: PoseOverlay | None,
                 camera=None, auto_stop_hours: float | None = 12.0,
                 audio=None,
                 wall_clock=time.time,
                 mono_clock=time.monotonic):
        self.cfg = cfg
        self.pose = pose
        # user-chosen clip padding survives restarts via settings.json
        settings_path = cfg.data_dir / "settings.json"
        try:
            padding = float(json.loads(
                settings_path.read_text())["padding_s"])
            cfg.preroll_seconds = cfg.postroll_seconds = padding
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self._settings_path = settings_path
        self._camera = camera if camera is not None else Camera(
            cfg.camera_index, cfg.width, cfg.height, cfg.fps)
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._status: dict = {"state": "STARTING", "score": 0.0,
                              "threshold": None, "monitoring": False,
                              "night_date": None,
                              "skeleton": bool(pose and pose.available)}
        self._t0_mono = mono_clock()
        self._mono = mono_clock
        self._last_pub_mono: float | None = None
        self.monitor = Monitor(
            config=cfg, storage=NullStorage(), camera=self._camera,
            scorer=MotionScorer(cfg.alpha, cfg.blur_ksize, cfg.diff_threshold,
                                cfg.lighting_frac, cfg.target_width),
            baseline=AdaptiveBaseline(cfg.window_seconds, cfg.fps, cfg.k,
                                      cfg.min_threshold, cfg.warmup_seconds),
            fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
            ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
            clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
            auto_stop_hours=auto_stop_hours,
            audio=audio,
            person_detector=((lambda _f: pose.last_present)
                             if pose is not None else None),
            person_box=((lambda: pose.last_box)
                        if pose is not None else None),
            wall_clock=wall_clock, mono_clock=mono_clock,
            on_frame=self._publish)
        self._audio = audio
        self._thread = threading.Thread(target=self.monitor.run,
                                        name="sleepwatch-monitor",
                                        daemon=True)
        self._night_storage: Storage | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if isinstance(self._camera, Camera):
            self._camera.open()
        self._thread.start()

    def shutdown(self) -> None:
        self.monitor.request_stop()
        self._thread.join(timeout=10)
        if self._audio is not None:
            self._audio.stop()
        if self.pose is not None:
            self.pose.close()

    # -- night control -----------------------------------------------------

    def start_night(self) -> bool:
        if self.monitor.persisting:
            return False
        self._night_storage = Storage(self.cfg.data_dir / "sleepwatch.db")
        self.monitor.begin_persist(self._night_storage)
        return True

    def stop_night(self) -> bool:
        if not self.monitor.persisting:
            return False
        self.monitor.end_persist()
        return True

    def get_padding(self) -> float:
        return float(self.cfg.preroll_seconds)

    def set_padding(self, seconds: float) -> None:
        seconds = float(seconds)
        self.cfg.preroll_seconds = seconds
        self.cfg.postroll_seconds = seconds
        # atomic attribute swap; the loop picks it up next frame. The new
        # ring starts empty, so pre-roll is shallow for the first N seconds.
        self.monitor.ring = FrameRingBuffer(seconds, self.cfg.fps)
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(
            json.dumps({"padding_s": seconds}))

    # -- publishing --------------------------------------------------------

    def _publish(self, frame, score, threshold, state) -> bool:
        vis = cv2.flip(frame, 1)      # mirror for display — like a mirror,
                                      # your left is on your left
        if self.pose is not None and self.pose.available:
            ts_ms = int((self._mono() - self._t0_mono) * 1000)
            vis = self.pose.annotate(vis, ts_ms)
        ok, buf = cv2.imencode(".jpg", vis,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        persisting = self.monitor.persisting
        with self._lock:
            if ok:
                self._jpeg = buf.tobytes()
                self._last_pub_mono = self._mono()
            self._status = {
                "state": state,
                "score": round(float(score), 5),
                "threshold": (None if threshold == float("inf")
                              else round(float(threshold), 5)),
                "monitoring": persisting,
                "night_date": self.monitor._night_date if persisting else None,
                "skeleton": bool(self.pose and self.pose.available),
            }
        return True

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def status(self) -> dict:
        with self._lock:
            out = dict(self._status)
            out["camera_age_s"] = (
                None if self._last_pub_mono is None
                else round(self._mono() - self._last_pub_mono, 2))
            return out
