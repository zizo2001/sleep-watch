from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FOURCC_PRIMARY = "avc1"
FOURCC_FALLBACK = "mp4v"


@dataclass(frozen=True)
class ClosedClip:
    path: Path
    duration_s: float


class FrameRingBuffer:
    def __init__(self, seconds: float = 5.0, fps: int = 15):
        self._buf: deque[np.ndarray] = deque(maxlen=max(1, int(seconds * fps)))

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(frame.copy())

    def dump(self) -> list[np.ndarray]:
        return [f.copy() for f in self._buf]


class ClipWriter:
    def __init__(self, clips_root: Path, fps: int):
        self.clips_root = Path(clips_root)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._path: Path | None = None
        self._frames = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def _open_writer(self, path: Path, size: tuple[int, int]) -> cv2.VideoWriter:
        for fourcc in (FOURCC_PRIMARY, FOURCC_FALLBACK):
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                self.fps, size)
            if w.isOpened():
                return w
            w.release()
        raise RuntimeError(f"no working codec for {path}")

    def start(self, night_date: str, event_start_wall: float,
              preroll: list[np.ndarray]) -> None:
        if self.active or not preroll:
            self.close()
            if not preroll:
                return
        day_dir = self.clips_root / night_date
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S", time.localtime(event_start_wall))
        self._path = day_dir / f"event_{stamp}.mp4"
        first = preroll[0]
        h, w = first.shape[:2]
        self._writer = self._open_writer(self._path, (w, h))
        for f in preroll:
            self._writer.write(f)
        self._frames = len(preroll)

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            self._frames += 1

    def close(self) -> ClosedClip | None:
        if self._writer is None:
            return None
        self._writer.release()
        path, self._path = self._path, None
        self._writer = None
        frames, self._frames = self._frames, 0
        return ClosedClip(path=path, duration_s=frames / self.fps)
