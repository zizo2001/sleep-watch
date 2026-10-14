#!/usr/bin/env python3
"""Replay a recorded night's event stream through the current wake gate.

Answers "what would today's code have called last night?" without a camera:
reads the movement/awake/onset/lighting rows a night already stored, feeds
them to Monitor._check_wake_burst plus the run-loop's re-onset demotion
branch, and prints the wake/restless verdicts it produces.

Use it after touching the gate — compare the output against what you
remember of the night: false wakes on twitch bursts should come out as
restless, the real wake should stay a wake.

    .venv/bin/python scripts/replay_wake_gate.py YYYY-MM-DD
"""
from __future__ import annotations

import sqlite3
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sleepwatch.config import Config, load_config          # noqa: E402
from sleepwatch.monitor import Monitor, classify_magnitude  # noqa: E402

MOTION_KINDS = ("movement", "awake", "truncated")


class _ReplayStorage:
    """Collects what the gate would have written, in order."""

    def __init__(self) -> None:
        self.rows: list[list] = []

    def add_event(self, night_id, kind, start_ts, end_ts, *_a, **_k) -> int:
        self.rows.append([kind, start_ts, end_ts])
        return len(self.rows)

    def reclassify_events(self, event_ids, kind) -> None:
        for i in event_ids:
            if 1 <= i <= len(self.rows):
                self.rows[i - 1][0] = kind

    def update_event_end(self, event_id, end_ts, duration_s) -> None:
        self.rows[event_id - 1][2] = end_ts


def replay(events, cfg: Config) -> list[list]:
    m = Monitor.__new__(Monitor)          # gate only — no camera, no threads
    m.cfg = cfg
    m.storage = _ReplayStorage()
    m._night_id = 0
    m._wake_deque = deque()
    m._last_restless = None
    m._pending_demote = None
    m._last_lighting_wall = None
    m._asleep = True
    m._woke = False
    for e in events:
        kind, ts = e["kind"], e["start_ts"]
        if kind == "wake":
            continue                      # the old verdict — being replaced
        if kind == "lighting":
            m._last_lighting_wall = ts
            continue
        if kind == "onset":
            pend = m._pending_demote
            if (pend is not None
                    and ts - pend["wake_ts"] <= cfg.wake_demote_minutes * 60
                    and not pend["saw_full"]
                    and (m._last_lighting_wall is None
                         or m._last_lighting_wall < pend["wake_ts"])):
                m.storage.reclassify_events([pend["wake_id"]], "restless")
            m._pending_demote = None
            m._asleep = True
            continue
        duration_s = e["duration_s"] or 0.0
        # replayed rows have no stored union box, so magnitude falls back to
        # duration — exactly what a pose-blind (blanket) night produces
        magnitude = classify_magnitude(e["coverage"], duration_s, cfg)
        if not m._asleep:
            if m._pending_demote is not None:
                m._pending_demote["saw_full"] |= magnitude == "full"
            continue
        m._check_wake_burst(ts, e["end_ts"] or ts, ts, 0,
                            duration_s, magnitude)
    return m.storage.rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    date = sys.argv[1]
    cfg = load_config(Path("config.toml"))
    conn = sqlite3.connect(str(cfg.data_dir / "sleepwatch.db"))
    conn.row_factory = sqlite3.Row
    night = conn.execute(
        "SELECT * FROM nights WHERE date=? ORDER BY id DESC LIMIT 1",
        (date,)).fetchone()
    if night is None:
        print(f"no night recorded for {date}")
        return 1
    kinds = MOTION_KINDS + ("onset", "lighting", "wake", "restless")
    events = conn.execute(
        f"SELECT * FROM events WHERE night_id=? AND kind IN "
        f"({','.join('?' * len(kinds))}) ORDER BY start_ts",
        (night["id"], *kinds)).fetchall()
    conn.close()

    clock = lambda ts: time.strftime("%H:%M:%S", time.localtime(ts))  # noqa: E731
    motion = [e for e in events if e["kind"] in MOTION_KINDS]
    print(f"night {date}: {len(motion)} motion rows, "
          f"{sum(1 for e in events if e['kind'] == 'lighting')} lighting, "
          f"{sum(1 for e in events if e['kind'] == 'onset')} onsets")
    print("\nas recorded:")
    for e in events:
        if e["kind"] in ("wake", "onset", "restless"):
            print(f"  {clock(e['start_ts'])}   {e['kind'].upper()}")
    if not any(e["kind"] == "onset" for e in events):
        print("\n  ⚠ no onset markers (pre-v2.5 night) — the replay assumes"
              "\n    asleep from the first row, so it fires earlier than the"
              "\n    live code would. Timings below are not comparable.")
    print("\nwhat the current gate would call:")
    verdicts = replay(events, cfg)
    if not verdicts:
        print("  (nothing — no burst ever reached the threshold)")
    for kind, start_ts, end_ts in verdicts:
        span = f" → {clock(end_ts)}" if end_ts and end_ts != start_ts else ""
        print(f"  {clock(start_ts)}{span}   {kind.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
