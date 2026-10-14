from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class MovementEvent:
    start_wall: float
    end_wall: float
    duration_s: float
    peak_score: float
    mean_score: float
    frames: int


class _State(Enum):
    QUIET = auto()
    PENDING = auto()
    MOVEMENT = auto()


class EventStateMachine:
    def __init__(self, trigger_seconds: float = 0.5, cooldown_seconds: float = 3.0):
        self.trigger_seconds = trigger_seconds
        self.cooldown_seconds = cooldown_seconds
        self.state = _State.QUIET
        self._reset_tracking()

    def _reset_tracking(self) -> None:
        self.start_wall = 0.0
        self.start_mono = 0.0
        self.last_above_wall = 0.0
        self.last_above_mono = 0.0
        self.scores: list[float] = []

    @property
    def in_movement(self) -> bool:
        return self.state is not _State.QUIET

    def _open(self, score: float, wall_ts: float, mono_ts: float) -> None:
        self.start_wall = wall_ts
        self.start_mono = mono_ts
        self.last_above_wall = wall_ts
        self.last_above_mono = mono_ts
        self.scores = [score]

    def _close(self) -> MovementEvent:
        ev = MovementEvent(
            start_wall=self.start_wall,
            end_wall=self.last_above_wall,
            duration_s=self.last_above_mono - self.start_mono,
            peak_score=max(self.scores),
            mean_score=sum(self.scores) / len(self.scores),
            frames=len(self.scores),
        )
        self.state = _State.QUIET
        self._reset_tracking()
        return ev

    def update(self, score: float, threshold: float,
               wall_ts: float, mono_ts: float) -> None | str | MovementEvent:
        above = score > threshold
        if self.state is _State.QUIET:
            if above:
                self.state = _State.PENDING
                self._open(score, wall_ts, mono_ts)
            return None
        if self.state is _State.PENDING:
            if not above:
                self.state = _State.QUIET
                self._reset_tracking()
                return None
            self.scores.append(score)
            self.last_above_wall, self.last_above_mono = wall_ts, mono_ts
            if mono_ts - self.start_mono >= self.trigger_seconds:
                self.state = _State.MOVEMENT
                return "started"
            return None
        # MOVEMENT
        if above:
            self.scores.append(score)
            self.last_above_wall, self.last_above_mono = wall_ts, mono_ts
            return None
        if mono_ts - self.last_above_mono >= self.cooldown_seconds:
            return self._close()
        return None

    def force_close(self, wall_ts: float, mono_ts: float) -> MovementEvent | None:
        if self.state is _State.MOVEMENT:
            return self._close()
        self.state = _State.QUIET
        self._reset_tracking()
        return None
