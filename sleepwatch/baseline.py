from __future__ import annotations

from collections import deque

import numpy as np


class AdaptiveBaseline:
    """Rolling median + k*MAD threshold over quiet-period scores."""

    def __init__(self, window_seconds: float = 300.0, fps: int = 15,
                 k: float = 6.0, min_threshold: float = 0.004,
                 warmup_seconds: float = 30.0):
        self.scores: deque[float] = deque(maxlen=max(1, int(window_seconds * fps)))
        self.k = k
        self.min_threshold = min_threshold
        self.warmup_samples = max(1, int(warmup_seconds * fps))
        self._threshold: float | None = None
        self._since_recompute = 0

    def update(self, score: float, in_movement: bool) -> None:
        if not in_movement:
            self.scores.append(score)
            self._since_recompute += 1
            if self._threshold is None or self._since_recompute >= 15:
                self._recompute()

    def _recompute(self) -> None:
        self._since_recompute = 0
        if len(self.scores) < self.warmup_samples:
            return
        arr = np.fromiter(self.scores, dtype=np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) * 1.4826
        self._threshold = max(med + self.k * mad, self.min_threshold)

    @property
    def ready(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float:
        return self._threshold if self._threshold is not None else float("inf")
