"""Dashboard with an injected black fake camera, port 5567 — UI verification.

Child processes without macOS camera TCC (Claude, CI) cannot open the real
camera; this serves the real UI and the real (read-only) DB without it.
NEVER click "Start night" here: it would record a fake night into the real
database. Leave port 5566 to the real dashboard.

Usage: .venv/bin/python scripts/fake_dashboard.py
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from sleepwatch.config import load_config
from sleepwatch.dashrunner import DashboardRunner
from sleepwatch.webapp import create_app


class BlackCam:
    def read(self):
        time.sleep(1 / 15)
        return True, np.zeros((720, 1280, 3), dtype=np.uint8)

    def close(self):
        pass


def main() -> None:
    cfg = load_config(REPO / "config.toml")
    cfg.data_dir = REPO / "data"
    runner = DashboardRunner(cfg, pose=None, camera=BlackCam(), audio=None)
    runner.start()
    app = create_app(runner, cfg)
    try:
        app.run(host="127.0.0.1", port=5567, threaded=True)
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
