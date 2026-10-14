from __future__ import annotations

import shutil
import subprocess
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np

SR = 16000
CHUNK = 512                     # Silero's native hop: 32 ms at 16 kHz
CHUNK_S = CHUNK / SR


class AudioRing:
    """Ring of (chunk, mono_ts) pairs covering the pre-roll window."""

    def __init__(self, seconds: float, sr: int = SR):
        n = max(1, round(seconds * sr / CHUNK))
        self._buf: deque[tuple[np.ndarray, float]] = deque(maxlen=n)

    def push(self, chunk: np.ndarray, mono_ts: float) -> None:
        self._buf.append((chunk.copy(), mono_ts))

    def dump_since(self, mono_ts: float) -> np.ndarray:
        parts = [c for c, t in self._buf if t >= mono_ts - CHUNK_S / 2]
        if not parts:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(parts)

    def dump(self) -> np.ndarray:
        return self.dump_since(-1e18)


class SileroVAD:
    """Speech probability per 512-sample chunk via the Silero ONNX model.

    Degrades to available=False (prob 0.0) when onnxruntime or the model
    file is missing — audio features switch off, video is untouched.
    """

    def __init__(self, model_path: Path):
        self.available = False
        self._sess = None
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"])
            self.available = True
        except Exception:
            pass
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(SR, dtype=np.int64)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        if not self.available:
            return 0.0
        x = chunk.astype(np.float32)
        if chunk.dtype == np.int16:
            x /= 32768.0
        out, self._state = self._sess.run(
            None, {"input": x.reshape(1, -1),
                   "state": self._state, "sr": self._sr})
        return float(out[0][0])


@dataclass(frozen=True)
class SpeechEvent:
    start_wall: float
    end_wall: float
    duration_s: float
    peak_prob: float


class _S(Enum):
    QUIET = auto()
    PENDING = auto()
    ACTIVE = auto()


class SpeechFSM:
    """EventStateMachine's shape, driven by VAD probability."""

    def __init__(self, threshold: float = 0.5, trigger_s: float = 0.3,
                 cooldown_s: float = 2.0):
        self.threshold = threshold
        self.trigger_s = trigger_s
        self.cooldown_s = cooldown_s
        self.state = _S.QUIET
        self._reset()

    def _reset(self) -> None:
        self.start_wall = 0.0
        self.start_mono = 0.0
        self.last_above_wall = 0.0
        self.last_above_mono = 0.0
        self.peak = 0.0

    @property
    def in_speech(self) -> bool:
        return self.state is not _S.QUIET

    def _close(self) -> SpeechEvent:
        ev = SpeechEvent(
            start_wall=self.start_wall,
            end_wall=self.last_above_wall,
            duration_s=self.last_above_mono - self.start_mono,
            peak_prob=self.peak)
        self.state = _S.QUIET
        self._reset()
        return ev

    def update(self, prob: float, wall_ts: float,
               mono_ts: float) -> None | str | SpeechEvent:
        above = prob >= self.threshold
        if self.state is _S.QUIET:
            if above:
                self.state = _S.PENDING
                self.start_wall, self.start_mono = wall_ts, mono_ts
                self.last_above_wall, self.last_above_mono = wall_ts, mono_ts
                self.peak = prob
            return None
        if self.state is _S.PENDING:
            if not above:
                self.state = _S.QUIET
                self._reset()
                return None
            self.peak = max(self.peak, prob)
            self.last_above_wall, self.last_above_mono = wall_ts, mono_ts
            if mono_ts - self.start_mono >= self.trigger_s:
                self.state = _S.ACTIVE
                return "started"
            return None
        # ACTIVE
        if above:
            self.peak = max(self.peak, prob)
            self.last_above_wall, self.last_above_mono = wall_ts, mono_ts
            return None
        if mono_ts - self.last_above_mono >= self.cooldown_s:
            return self._close()
        return None

    def force_close(self) -> SpeechEvent | None:
        if self.state is _S.ACTIVE:
            return self._close()
        self.state = _S.QUIET
        self._reset()
        return None


class AudioClipWriter:
    """Collects one clip's audio window; writes a 16-bit mono WAV at close."""

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._parts: list[np.ndarray] | None = None

    @property
    def active(self) -> bool:
        return self._parts is not None

    def start(self, preroll: np.ndarray) -> None:
        self._parts = [preroll.astype(np.int16)]

    def push(self, chunk: np.ndarray) -> None:
        if self._parts is not None:
            self._parts.append(chunk.astype(np.int16))

    def close(self, path: Path) -> Path | None:
        if self._parts is None:
            return None
        samples = (np.concatenate(self._parts) if self._parts
                   else np.zeros(0, dtype=np.int16))
        self._parts = None
        path = Path(path)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.sr)
            f.writeframes(samples.tobytes())
        return path


def pick_input_device(devices, override: str = "",
                      prefer: str = "OBSBOT") -> int | None:
    """Choose an input device index; None means system default."""
    def inputs():
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                yield i, d
    if override:
        for i, d in inputs():
            if override.lower() in d["name"].lower():
                return i
    for i, d in inputs():
        if prefer.lower() in d["name"].lower():
            return i
    return None


class AudioCapture:
    """Live mic → VAD → speech FSM, with thread-safe taps for the Monitor.

    The sounddevice callback runs on PortAudio's thread; Monitor drains
    `take_chunks()` / `drain_events()` from its own loop. Everything except
    the stream itself is exercised in tests through `_ingest`.
    """

    def __init__(self, cfg, model_path: Path = Path("models/silero_vad.onnx"),
                 wall_clock=None, mono_clock=None):
        import time as _time
        self.cfg = cfg
        self.ring = AudioRing(cfg.preroll_seconds)
        self._vad = SileroVAD(model_path)
        self._fsm = SpeechFSM(cfg.vad_threshold, cfg.speech_trigger_s,
                              cfg.speech_cooldown_s)
        self._wall = wall_clock or _time.time
        self._mono = mono_clock or _time.monotonic
        import threading
        self._lock = threading.Lock()
        self._chunks: list[tuple[np.ndarray, float]] = []
        self._events: list = []
        self._peak_prob: float | None = None
        self._pending = np.zeros(0, dtype=np.int16)
        self._decim = 1
        self._stream = None
        self.available = self._vad.available

    # -- hardware ----------------------------------------------------------

    def start(self) -> None:
        if not self.available:
            return
        try:
            import sounddevice as sd
            idx = pick_input_device(sd.query_devices(),
                                    override=self.cfg.audio_device)
            try:
                self._stream = sd.InputStream(
                    device=idx, samplerate=SR, channels=1, dtype="int16",
                    blocksize=CHUNK, callback=self._callback)
                self._stream.start()
            except Exception:
                info = (sd.query_devices(idx) if idx is not None
                        else sd.query_devices(kind="input"))
                native = int(info["default_samplerate"])
                self._decim = max(1, round(native / SR))
                self._stream = sd.InputStream(
                    device=idx, samplerate=native, channels=1,
                    dtype="int16", blocksize=CHUNK * self._decim,
                    callback=self._callback)
                self._stream.start()
        except Exception:
            self.available = False
            self._stream = None

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, indata, frames, t, status) -> None:
        self._ingest(indata[:, 0][::self._decim])

    # -- logic (tested) ----------------------------------------------------

    def _ingest(self, samples: np.ndarray) -> None:
        self._pending = np.concatenate(
            [self._pending, samples.astype(np.int16)])
        while len(self._pending) >= CHUNK:
            chunk, self._pending = (self._pending[:CHUNK],
                                    self._pending[CHUNK:])
            wall, mono = self._wall(), self._mono()
            self.ring.push(chunk, mono)
            prob = self._vad.prob(chunk)
            out = self._fsm.update(prob, wall, mono)
            with self._lock:
                if self._peak_prob is None or prob > self._peak_prob:
                    self._peak_prob = prob
                self._chunks.append((chunk, mono))
                if out == "started":
                    self._events.append(("started", self._fsm.start_wall))
                elif isinstance(out, SpeechEvent):
                    self._events.append(out)

    def take_chunks(self) -> list[tuple[np.ndarray, float]]:
        with self._lock:
            out, self._chunks = self._chunks, []
        return out

    def drain_events(self) -> list:
        with self._lock:
            out, self._events = self._events, []
        return out

    def take_peak_prob(self) -> float | None:
        """Max VAD prob since the last take — sub-threshold calibration
        data (silent nights emit no SpeechEvents, this is all we get)."""
        with self._lock:
            out, self._peak_prob = self._peak_prob, None
        return out


def build_audio(cfg, model_path: Path = Path("models/silero_vad.onnx")):
    """Audio bundle for the Monitor, or None with one honest warning."""
    if not cfg.speech_enabled:
        return None
    try:
        import sounddevice  # noqa: F401
    except Exception:
        print("audio off — sounddevice not installed")
        return None
    cap = AudioCapture(cfg, model_path=model_path)
    if not cap.available:
        print(f"audio off — {model_path} missing")
        return None
    cap.start()
    if not cap.available:
        print("audio off — no usable input device")
        return None
    return cap


def mux_audio(video: Path, wav: Path) -> bool:
    """Fold a WAV into the mp4 (video stream copied). Consumes the WAV.

    False → mp4 left untouched (still valid, silent); WAV removed either
    way so nights never accumulate stray audio files.
    """
    video, wav = Path(video), Path(wav)
    try:
        if shutil.which("ffmpeg") is None:
            return False
        tmp = video.with_suffix(".mux.mp4")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video), "-i", str(wav),
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)],
            capture_output=True, timeout=60)
        if proc.returncode != 0 or not tmp.exists():
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(video)
        return True
    except Exception:
        return False
    finally:
        wav.unlink(missing_ok=True)
