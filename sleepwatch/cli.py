from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import webbrowser
from pathlib import Path

import cv2

from .audio import build_audio
from .baseline import AdaptiveBaseline
from .camera import Camera
from .clipwriter import ClipWriter, FrameRingBuffer
from .config import Config, load_config
from .detector import MotionScorer
from .events import EventStateMachine
from .monitor import Monitor
from .report import generate_report
from .storage import NullStorage, Storage


def _build_monitor(cfg: Config, preview: bool,
                   persist: bool = True) -> tuple[Monitor, Storage | NullStorage]:
    storage = Storage(cfg.data_dir / "sleepwatch.db") if persist else NullStorage()
    camera = Camera(cfg.camera_index, cfg.width, cfg.height, cfg.fps)
    camera.open()
    monitor = Monitor(
        config=cfg, storage=storage, camera=camera,
        scorer=MotionScorer(cfg.alpha, cfg.blur_ksize, cfg.diff_threshold,
                            cfg.lighting_frac, cfg.target_width),
        baseline=AdaptiveBaseline(cfg.window_seconds, cfg.fps, cfg.k,
                                  cfg.min_threshold, cfg.warmup_seconds),
        fsm=EventStateMachine(cfg.trigger_seconds, cfg.cooldown_seconds),
        ring=FrameRingBuffer(cfg.preroll_seconds, cfg.fps),
        clip=ClipWriter(cfg.data_dir / "clips", cfg.fps),
        clips_enabled=persist,
        audio=build_audio(cfg) if persist else None,
        on_frame=_preview_callback() if preview else None)
    return monitor, storage


def _preview_callback():
    def on_frame(frame, score, threshold, state) -> bool:
        vis = frame.copy()
        thr = "…" if threshold == float("inf") else f"{threshold:.4f}"
        color = (60, 60, 230) if state == "MOVEMENT" else (90, 200, 90)
        cv2.putText(vis, f"{state}  score={score:.4f}  thr={thr}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        bar = int(min(score * 4000, vis.shape[1] - 24))
        cv2.rectangle(vis, (12, 44), (12 + bar, 56), color, -1)
        cv2.imshow("sleep-watch (q quits)", vis)
        return cv2.waitKey(1) & 0xFF != ord("q")
    return on_frame


def _spawn_caffeinate() -> subprocess.Popen | None:
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except FileNotFoundError:
        return None


def cmd_run(args, preview: bool, persist: bool = True) -> int:
    cfg = load_config(Path(args.config))
    if args.camera is not None:
        cfg.camera_index = args.camera
    caff = _spawn_caffeinate()
    monitor, storage = _build_monitor(cfg, preview=preview, persist=persist)
    signal.signal(signal.SIGINT, lambda *_: monitor.request_stop())
    signal.signal(signal.SIGTERM, lambda *_: monitor.request_stop())
    if not persist:
        print("TEST MODE — nothing is saved. q in window or Ctrl+C to quit")
    elif preview:
        print("monitoring WITH preview — q or Ctrl+C stops and saves")
    else:
        print("monitoring — Ctrl+C to stop and build the report")
    night_id = monitor.run()
    if preview:
        cv2.destroyAllWindows()
    if persist:
        print(f"night {night_id} saved. reports: "
              f"{(cfg.data_dir / 'reports').resolve()}")
    else:
        print("test session over — nothing was saved")
    storage.close()
    if caff is not None:
        caff.terminate()
    return 0


def cmd_dashboard(args) -> int:
    from .dashrunner import DashboardRunner
    from .pose import PoseOverlay
    from .webapp import create_app

    cfg = load_config(Path(args.config))
    if args.camera is not None:
        cfg.camera_index = args.camera
    model = Path("models/pose_landmarker_lite.task")
    pose = PoseOverlay(model if model.exists() else None)
    if not pose.available:
        print("skeleton off — models/pose_landmarker_lite.task not found")
    caff = _spawn_caffeinate()
    runner = DashboardRunner(cfg, pose=pose, audio=build_audio(cfg))
    runner.start()
    app = create_app(runner, cfg)
    from . import __version__
    print(f"sleep-watch v{__version__} — "
          f"dashboard: http://127.0.0.1:{args.port}")
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True)
    finally:
        runner.shutdown()
        if caff is not None:
            caff.terminate()
    return 0


def cmd_report(args) -> int:
    cfg = load_config(Path(args.config))
    storage = Storage(cfg.data_dir / "sleepwatch.db")
    storage.backfill_clip_durations()
    night = (storage.night_by_date(args.date) if args.date
             else storage.latest_night())
    if night is None:
        print("no recorded nights found", file=sys.stderr)
        storage.close()
        return 1
    events = storage.events_for_night(night["id"])
    minutes = storage.signal_minutes_for_night(night["id"])
    out = cfg.data_dir / "reports" / f"{night['date']}.html"
    path = generate_report(night, events, out, signal_minutes=minutes)
    print(path.resolve())
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    storage.close()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="sleep-watch")
    p.add_argument("--config", default="config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("test", "start"):
        sp = sub.add_parser(name)
        sp.add_argument("--camera", type=int, default=None)
    sub.choices["start"].add_argument("--preview", action="store_true")
    dp = sub.add_parser("dashboard")
    dp.add_argument("--camera", type=int, default=None)
    dp.add_argument("--port", type=int, default=5566)
    rp = sub.add_parser("report")
    rp.add_argument("--date", default=None)
    rp.add_argument("--open", action="store_true")
    args = p.parse_args()
    if args.cmd == "test":
        sys.exit(cmd_run(args, preview=True, persist=False))
    elif args.cmd == "start":
        sys.exit(cmd_run(args, preview=args.preview))
    elif args.cmd == "dashboard":
        sys.exit(cmd_dashboard(args))
    else:
        sys.exit(cmd_report(args))
