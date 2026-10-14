from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ScoreResult:
    score: float
    lighting: bool
    box: tuple[int, int, int, int] | None = None    # x, y, w, h — downscale px
    frame_size: tuple[int, int] = (0, 0)            # (w, h) of downscale space


class MotionScorer:
    """Fraction-of-pixels-changed vs a running-average background."""

    def __init__(self, alpha: float = 0.35, blur_ksize: int = 7,
                 diff_threshold: int = 12, lighting_frac: float = 0.60,
                 target_width: int = 320):
        self.alpha = alpha
        self.blur_ksize = blur_ksize
        self.diff_threshold = diff_threshold
        self.lighting_frac = lighting_frac
        self.target_width = target_width
        self.bg: np.ndarray | None = None

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        small = cv2.resize(gray, (self.target_width,
                                  max(1, round(h * self.target_width / w))))
        small = cv2.GaussianBlur(small, (self.blur_ksize, self.blur_ksize), 0)
        return small.astype(np.float32)

    def score(self, frame_bgr: np.ndarray) -> ScoreResult:
        f32 = self._prepare(frame_bgr)
        if self.bg is None:
            self.bg = f32.copy()
            return ScoreResult(0.0, False)
        diff = cv2.absdiff(f32, self.bg)
        mask = diff > self.diff_threshold
        frac = float(mask.mean())
        h, w = f32.shape
        if frac > self.lighting_frac:
            self.bg = f32.copy()
            return ScoreResult(0.0, True, None, (w, h))
        cv2.accumulateWeighted(f32, self.bg, self.alpha)
        box = None
        if frac > 0:
            pts = cv2.findNonZero(mask.astype(np.uint8))
            if pts is not None:
                box = tuple(cv2.boundingRect(pts))
        return ScoreResult(frac, False, box, (w, h))
