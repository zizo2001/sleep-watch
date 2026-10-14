from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 15
    target_width: int = 320
    alpha: float = 0.35  # fast bg decay: ghost fades ~6 frames, well under cooldown
    blur_ksize: int = 7
    diff_threshold: int = 12
    lighting_frac: float = 0.60
    window_seconds: float = 300.0
    warmup_seconds: float = 30.0
    k: float = 6.0
    min_threshold: float = 0.004
    trigger_seconds: float = 0.5
    cooldown_seconds: float = 3.0
    preroll_seconds: float = 10.0   # ring: 10 s × 15 fps × 720p ≈ 415 MB RAM
    postroll_seconds: float = 10.0  # keep writing this long after event close
    speech_enabled: bool = True
    vad_threshold: float = 0.5      # Silero prob; fan noise scores ~0.002
    speech_trigger_s: float = 0.3
    speech_cooldown_s: float = 2.0
    speech_min_s: float = 0.5       # shorter voiced bursts: no event row
    audio_device: str = ""          # substring match; "" = OBSBOT, then default
    timelapse_every_s: float = 2.5  # ×24 fps: 1 film-second = 1 real minute
    auto_stop_absence_minutes: float = 45.0  # out of bed 45 min = woke up
    wake_burst_stirs: int = 5       # this many DISTINCT stirs inside the window…
    wake_window_minutes: float = 10.0   # …means awake (sleep onset, inverted)
    wake_stir_gap_s: float = 60.0   # movement rows this close = one stir
    wake_stir_span_s: float = 180.0  # one stir this long = sustained activity, awake now
    wake_evidence_min_s: float = 10.0    # burst needs one move this long (or
                                         # full-body, or light) to claim wake
    wake_demote_minutes: float = 20.0    # re-onset this soon after a burst
                                         # wake, nothing big → it was restless
    ambient_overlap_frac: float = 0.2   # motion∩person below this = ambient
    ambient_pose_min_frac: float = 0.5  # pose must see ≥ this frac of the
                                        # event's motion frames to call ambient
    person_box_pad: float = 0.15    # inflate person box — blanket > skeleton
    coverage_partial_frac: float = 0.25  # motion∪ ÷ person area → partial
    coverage_full_frac: float = 0.6      # …→ full-body
    sleep_onset_minutes: float = 10.0
    retention_days: int = 7
    min_free_gb: float = 10.0
    data_dir: Path = Path("data")


def load_config(path: Path | None) -> Config:
    cfg = Config()
    if path is None or not Path(path).exists():
        return cfg
    raw = tomllib.loads(Path(path).read_text()).get("sleepwatch", {})
    valid = {f.name for f in fields(Config)}
    for key, value in raw.items():
        if key in valid:
            if key == "data_dir":
                value = Path(value)
            setattr(cfg, key, value)
    return cfg
