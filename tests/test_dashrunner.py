import time as real_time

from sleepwatch.config import Config
from sleepwatch.dashrunner import DashboardRunner
from tests.test_monitor import FakeCamera, FakeClock


def make_runner(tmp_path, frames=2000):
    cfg = Config(data_dir=tmp_path / "data", warmup_seconds=2.0,
                 window_seconds=20.0, preroll_seconds=1.0, fps=15)
    clock = FakeClock()
    cam = FakeCamera(clock, quiet1=frames, move=0, quiet2=0)
    runner = DashboardRunner(cfg, pose=None, camera=cam,
                             wall_clock=clock.wall, mono_clock=clock.mono)
    cam.monitor = runner.monitor
    return runner, cfg


def wait_for(cond, timeout=5.0):
    t0 = real_time.time()
    while real_time.time() - t0 < timeout:
        if cond():
            return True
        real_time.sleep(0.01)
    return False


def test_runner_publishes_frames_and_status(tmp_path):
    runner, cfg = make_runner(tmp_path)
    runner.start()
    assert wait_for(lambda: runner.latest_jpeg() is not None)
    st = runner.status()
    assert st["monitoring"] is False
    assert st["skeleton"] is False
    assert st["state"] in ("CALIBRATING", "QUIET", "MOVEMENT")
    runner.shutdown()


def test_set_padding_swaps_ring_and_persists(tmp_path):
    runner, cfg = make_runner(tmp_path)
    runner.set_padding(15)
    assert cfg.preroll_seconds == 15.0
    assert cfg.postroll_seconds == 15.0
    assert runner.monitor.ring._buf.maxlen == 15 * cfg.fps
    import json
    saved = json.loads((cfg.data_dir / "settings.json").read_text())
    assert saved == {"padding_s": 15.0}


def test_padding_loads_from_settings_file(tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "settings.json").write_text('{"padding_s": 5}')
    runner, cfg = make_runner(tmp_path)     # make_runner passes preroll 1.0;
    assert cfg.preroll_seconds == 5.0       # settings.json must override it
    assert runner.monitor.ring._buf.maxlen == 5 * cfg.fps


def test_runner_start_stop_night_persists(tmp_path):
    runner, cfg = make_runner(tmp_path, frames=100000)
    runner.start()
    assert wait_for(lambda: runner.latest_jpeg() is not None)
    assert runner.start_night() is True
    assert wait_for(lambda: runner.status()["monitoring"] is True)
    assert runner.start_night() is False          # already running
    assert runner.stop_night() is True
    assert wait_for(lambda: runner.status()["monitoring"] is False)
    runner.shutdown()
    from sleepwatch.storage import Storage
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nights = st.conn.execute("SELECT * FROM nights").fetchall()
    assert len(nights) == 1 and nights[0]["end_ts"] is not None
    st.close()
