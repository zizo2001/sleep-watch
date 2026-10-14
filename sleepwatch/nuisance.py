from __future__ import annotations

import json
from pathlib import Path

GRID_W, GRID_H = 16, 9
DAMP_THRESHOLD = 3.0     # cell must earn this many (decayed) ambient hits
DECAY = 0.9              # per-night multiplier — seasons un-learn themselves
PRUNE_BELOW = 0.5
DAMP_FACTOR = 0.5        # never zero: real motion must still break through


class NuisanceMap:
    """Learned map of screen regions that move without the sleeper in them.

    Day-by-day learning, no black box: ambient events (sun on the wall,
    curtain shadows) increment their grid cells; heavy cells damp the
    motion score by half; nightly decay forgets stale sun angles.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        try:
            self.cells = {k: float(v) for k, v in
                          json.loads(self.path.read_text()).items()}
        except Exception:
            self.cells = {}

    def _keys(self, box_norm) -> list[str]:
        x, y, w, h = box_norm
        x0 = max(0, min(GRID_W - 1, int(x * GRID_W)))
        x1 = max(0, min(GRID_W - 1, int((x + w) * GRID_W)))
        y0 = max(0, min(GRID_H - 1, int(y * GRID_H)))
        y1 = max(0, min(GRID_H - 1, int((y + h) * GRID_H)))
        return [f"{cx},{cy}"
                for cx in range(x0, x1 + 1) for cy in range(y0, y1 + 1)]

    def hit(self, box_norm) -> None:
        for k in self._keys(box_norm):
            self.cells[k] = self.cells.get(k, 0.0) + 1.0

    def damp_factor(self, box_norm) -> float:
        if any(self.cells.get(k, 0.0) >= DAMP_THRESHOLD
               for k in self._keys(box_norm)):
            return DAMP_FACTOR
        return 1.0

    def decay(self) -> None:
        self.cells = {k: v * DECAY for k, v in self.cells.items()
                      if v * DECAY >= PRUNE_BELOW}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cells))
