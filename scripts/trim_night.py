"""Trim a night that recorded past the real wake time.

Usage: python scripts/trim_night.py <db_path> <night_id> <HH:MM>

Finds the first movement at/after HH:MM local, stamps the night's end_ts
there, inserts a wake row, and reclassifies that movement and everything
after it to 'awake' (rows kept, stats exclude them — same shape the live
wake-burst detector produces). Regenerates the night's report.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleepwatch.report import generate_report        # noqa: E402
from sleepwatch.storage import Storage               # noqa: E402


def trim(db_path: str | Path, night_id: int, hhmm: str) -> int:
    st = Storage(Path(db_path))
    night = st.conn.execute("SELECT * FROM nights WHERE id=?",
                            (night_id,)).fetchone()
    if night is None:
        print(f"night {night_id} not found")
        st.close()
        return 0
    h, m = map(int, hhmm.split(":"))
    lt = time.localtime(night["start_ts"])
    cutoff = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                          h, m, 0, 0, 0, -1))
    if cutoff < night["start_ts"]:
        cutoff += 86400                       # wake fell on the next day
    evs = st.conn.execute(
        "SELECT id, start_ts FROM events WHERE night_id=? "
        "AND kind='movement' AND start_ts >= ? ORDER BY start_ts",
        (night_id, cutoff)).fetchall()
    if not evs:
        print("nothing to trim")
        st.close()
        return 0
    wake_ts = evs[0]["start_ts"]
    st.reclassify_events([e["id"] for e in evs], "awake")
    st.add_event(night_id, "wake", wake_ts, wake_ts, 0.0, None, None, None)
    st.conn.execute("UPDATE nights SET end_ts=? WHERE id=?",
                    (wake_ts, night_id))
    st.conn.commit()
    night = st.conn.execute("SELECT * FROM nights WHERE id=?",
                            (night_id,)).fetchone()
    out = Path(db_path).parent / "reports" / f"{night['date']}.html"
    generate_report(night, st.events_for_night(night_id), out)
    stamp = time.strftime("%H:%M:%S", time.localtime(wake_ts))
    print(f"trimmed night {night_id}: end -> {stamp}, "
          f"{len(evs)} movement(s) -> awake, report regenerated")
    st.close()
    return len(evs)


if __name__ == "__main__":
    trim(sys.argv[1], int(sys.argv[2]), sys.argv[3])
