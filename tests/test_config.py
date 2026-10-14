from pathlib import Path
from sleepwatch.config import Config, load_config

def test_defaults():
    cfg = load_config(None)
    assert cfg.camera_index == 0
    assert cfg.width == 1280 and cfg.height == 720 and cfg.fps == 15
    assert cfg.k == 6.0 and cfg.min_threshold == 0.004
    assert cfg.trigger_seconds == 0.5 and cfg.cooldown_seconds == 3.0
    assert cfg.preroll_seconds == 10.0
    assert cfg.postroll_seconds == 10.0
    assert cfg.speech_enabled is True
    assert cfg.vad_threshold == 0.5
    assert cfg.speech_trigger_s == 0.3
    assert cfg.speech_cooldown_s == 2.0
    assert cfg.speech_min_s == 0.5
    assert cfg.audio_device == ""
    assert cfg.timelapse_every_s == 2.5   # v2.7: 1 film-second = 1 real minute
    assert cfg.auto_stop_absence_minutes == 45.0
    assert cfg.wake_burst_stirs == 5
    assert cfg.wake_stir_gap_s == 60.0
    assert cfg.wake_stir_span_s == 180.0
    assert cfg.wake_window_minutes == 10.0
    assert cfg.ambient_overlap_frac == 0.2
    assert cfg.retention_days == 7
    assert cfg.data_dir == Path("data")

def test_toml_override(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[sleepwatch]\ncamera_index = 2\nk = 4.5\n")
    cfg = load_config(p)
    assert cfg.camera_index == 2 and cfg.k == 4.5
    assert cfg.fps == 15  # untouched default
