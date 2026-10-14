from pathlib import Path

import numpy as np
import pytest

from sleepwatch.audio import (CHUNK, CHUNK_S, AudioRing, SileroVAD,
                              SpeechEvent, SpeechFSM)

MODEL = Path("models/silero_vad.onnx")


def chunk(value=0):
    return np.full(CHUNK, value, dtype=np.int16)


def test_ring_caps_and_dumps_window():
    ring = AudioRing(seconds=10 * CHUNK_S)          # cap 10 chunks
    for i in range(20):
        ring.push(chunk(i), mono_ts=i * CHUNK_S)
    out = ring.dump_since(0.0)
    assert len(out) == 10 * CHUNK                   # only last 10 kept
    assert out[0] == 10                             # oldest surviving chunk
    tail = ring.dump_since(19 * CHUNK_S)
    assert len(tail) == CHUNK
    assert tail[0] == 19


def test_ring_dump_empty():
    ring = AudioRing(seconds=1.0)
    assert len(ring.dump_since(0.0)) == 0


def test_speech_fsm_trigger_and_close():
    fsm = SpeechFSM(threshold=0.5, trigger_s=0.3,
                    cooldown_s=2.0)
    t = [0.0]

    def step(prob):
        t[0] += CHUNK_S
        return fsm.update(prob, wall_ts=1000 + t[0], mono_ts=t[0])

    for _ in range(5):                              # quiet stays quiet
        assert step(0.02) is None
    started = None
    for _ in range(12):                             # ~0.38s of speech
        out = step(0.9)
        if out == "started":
            started = t[0]
    assert started is not None and 0.3 <= started <= 0.45 + 5 * CHUNK_S
    for _ in range(8):                              # keep talking
        assert step(0.8) is None
    ev = None
    for _ in range(70):                             # 2s+ of silence closes
        out = step(0.05)
        if isinstance(out, SpeechEvent):
            ev = out
    assert ev is not None
    assert ev.peak_prob == 0.9
    assert 0.5 < ev.duration_s < 1.0                # ~20 voiced chunks
    assert ev.end_wall > ev.start_wall


def test_speech_fsm_blip_no_trigger():
    fsm = SpeechFSM(threshold=0.5, trigger_s=0.3, cooldown_s=2.0)
    t = 0.0
    outs = []
    for prob in [0.9, 0.9, 0.02, 0.02, 0.02]:       # 64ms blip < trigger
        t += CHUNK_S
        outs.append(fsm.update(prob, 1000 + t, t))
    assert all(o is None for o in outs)


@pytest.mark.skipif(not MODEL.exists(), reason="silero model not downloaded")
def test_vad_scores_silence_near_zero():
    vad = SileroVAD(MODEL)
    assert vad.available
    assert vad.prob(chunk(0)) < 0.1
    noise = (np.random.RandomState(0).randn(CHUNK) * 500).astype(np.int16)
    assert vad.prob(noise) < 0.3                    # fan-like noise stays low


def test_vad_missing_model_disabled(tmp_path):
    vad = SileroVAD(tmp_path / "nope.onnx")
    assert not vad.available
    assert vad.prob(chunk(3)) == 0.0


def test_audio_clip_writer_wav_roundtrip(tmp_path):
    import wave
    from sleepwatch.audio import AudioClipWriter
    w = AudioClipWriter()
    assert not w.active
    w.start(np.concatenate([chunk(1), chunk(2)]))
    assert w.active
    w.push(chunk(3))
    out = w.close(tmp_path / "c.wav")
    assert out is not None and out.exists() and not w.active
    with wave.open(str(out)) as f:
        assert f.getnchannels() == 1
        assert f.getframerate() == 16000
        assert f.getnframes() == 3 * CHUNK


def test_audio_clip_writer_close_without_start(tmp_path):
    from sleepwatch.audio import AudioClipWriter
    assert AudioClipWriter().close(tmp_path / "x.wav") is None


def test_mux_audio_attaches_stream(tmp_path):
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    import cv2
    from sleepwatch.audio import AudioClipWriter, mux_audio
    video = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                         15, (160, 120))
    for _ in range(30):
        vw.write(np.full((120, 160, 3), 60, dtype=np.uint8))
    vw.release()
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    aw = AudioClipWriter()
    aw.start(tone)
    wav = aw.close(tmp_path / "v.wav")
    assert mux_audio(video, wav) is True
    assert not wav.exists()                       # consumed
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0",
         str(video)], capture_output=True, text=True)
    assert "audio" in probe.stdout


DEVICES = [                       # mirrors real sd.query_devices() output
    {"name": "A50 Mic", "max_input_channels": 1},
    {"name": "Speakers", "max_input_channels": 0},
    {"name": "OBSBOT Meet SE Microphone", "max_input_channels": 2},
    {"name": "iPhone Microphone", "max_input_channels": 1},
]


def test_pick_device_prefers_obsbot_then_override():
    from sleepwatch.audio import pick_input_device
    assert pick_input_device(DEVICES) == 2
    assert pick_input_device(DEVICES, override="iphone") == 3
    assert pick_input_device(DEVICES, override="nonexistent") == 2
    assert pick_input_device([{"name": "X", "max_input_channels": 0}]) is None


def test_ingest_chunks_and_emits_events(tmp_path):
    from sleepwatch.audio import AudioCapture
    from sleepwatch.config import Config
    cfg = Config(preroll_seconds=1.0, vad_threshold=0.5,
                 speech_trigger_s=0.06, speech_cooldown_s=0.1)
    t = {"v": 0.0}

    def mono():
        t["v"] += CHUNK_S
        return t["v"]

    cap = AudioCapture(cfg, model_path=tmp_path / "none.onnx",
                       wall_clock=lambda: 5000 + t["v"], mono_clock=mono)
    probs = iter([0.9] * 4 + [0.0] * 10)
    cap._vad.prob = lambda chunk: next(probs)
    cap._ingest(np.zeros(700, dtype=np.int16))      # 1 chunk + 188 pending
    assert len(cap._pending) == 188
    cap._ingest(np.zeros(512 * 12, dtype=np.int16))
    chunks = cap.take_chunks()
    assert len(chunks) == 13                        # 700+6144 → 13×512 + 188
    events = cap.drain_events()
    assert events and events[0][0] == "started"
    from sleepwatch.audio import SpeechEvent
    assert any(isinstance(e, SpeechEvent) for e in events)


def test_take_peak_prob_returns_and_resets(tmp_path):
    # sub-threshold speech probs are calibration data: silent nights never
    # emit SpeechEvents, so the per-minute peak is the only VAD evidence
    from sleepwatch.audio import AudioCapture
    from sleepwatch.config import Config
    cfg = Config(preroll_seconds=1.0, vad_threshold=0.5)
    t = {"v": 0.0}

    def mono():
        t["v"] += CHUNK_S
        return t["v"]

    cap = AudioCapture(cfg, model_path=tmp_path / "none.onnx",
                       wall_clock=lambda: 5000 + t["v"], mono_clock=mono)
    probs = iter([0.1, 0.34, 0.2, 0.05])
    cap._vad.prob = lambda chunk: next(probs)
    cap._ingest(np.zeros(512 * 4, dtype=np.int16))
    assert cap.take_peak_prob() == 0.34
    assert cap.take_peak_prob() is None      # reset — nothing new since


def test_mux_audio_missing_ffmpeg_keeps_video(tmp_path, monkeypatch):
    import shutil as _sh
    from sleepwatch import audio as audio_mod
    monkeypatch.setattr(audio_mod.shutil, "which", lambda _: None)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"fake")
    assert audio_mod.mux_audio(video, wav) is False
    assert video.exists()
