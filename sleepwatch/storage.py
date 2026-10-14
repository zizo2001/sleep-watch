from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS nights(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  start_ts REAL NOT NULL,
  end_ts REAL,
  sleep_onset_ts REAL,
  config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  night_id INTEGER NOT NULL REFERENCES nights(id),
  kind TEXT NOT NULL,
  start_ts REAL NOT NULL,
  end_ts REAL,
  duration_s REAL,
  peak_score REAL,
  mean_score REAL,
  clip_path TEXT,
  clip_duration_s REAL,
  rating INTEGER,
  coverage REAL,
  magnitude TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_night ON events(night_id);
CREATE TABLE IF NOT EXISTS signal_minutes(
  night_id INTEGER NOT NULL REFERENCES nights(id),
  minute_ts REAL NOT NULL,
  max_motion REAL NOT NULL,
  threshold REAL NOT NULL,
  max_speech REAL
);
CREATE INDEX IF NOT EXISTS idx_signal_night ON signal_minutes(night_id);
"""


class Storage:
    def __init__(self, db_path: Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: dashboard creates Storage in the web
        # thread and hands it to the camera thread; WAL + single writer.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        # migrate databases created before these columns existed; each ALTER
        # gets its own try so one present column doesn't mask another missing
        for ddl in ("ALTER TABLE nights ADD COLUMN sleep_onset_ts REAL",
                    "ALTER TABLE events ADD COLUMN clip_duration_s REAL",
                    "ALTER TABLE events ADD COLUMN rating INTEGER",
                    "ALTER TABLE events ADD COLUMN coverage REAL",
                    "ALTER TABLE events ADD COLUMN magnitude TEXT"):
            try:
                self.conn.execute(ddl)
                self.conn.commit()
            except sqlite3.OperationalError:
                pass

    def begin_night(self, date: str, start_ts: float, config_json: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO nights(date, start_ts, config_json) VALUES (?,?,?)",
            (date, start_ts, config_json))
        self.conn.commit()
        return cur.lastrowid

    def end_night(self, night_id: int, end_ts: float) -> None:
        self.conn.execute("UPDATE nights SET end_ts=? WHERE id=?",
                          (end_ts, night_id))
        self.conn.commit()

    def set_sleep_onset(self, night_id: int, onset_ts: float) -> None:
        self.conn.execute("UPDATE nights SET sleep_onset_ts=? WHERE id=?",
                          (onset_ts, night_id))
        self.conn.commit()

    def add_event(self, night_id: int, kind: str, start_ts: float,
                  end_ts: float | None, duration_s: float | None,
                  peak_score: float | None, mean_score: float | None,
                  clip_path: str | None, coverage: float | None = None,
                  magnitude: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(night_id, kind, start_ts, end_ts, duration_s,"
            " peak_score, mean_score, clip_path, coverage, magnitude)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (night_id, kind, start_ts, end_ts, duration_s,
             peak_score, mean_score, clip_path, coverage, magnitude))
        self.conn.commit()
        return cur.lastrowid

    def add_signal_minute(self, night_id: int, minute_ts: float,
                          max_motion: float, threshold: float,
                          max_speech: float | None) -> None:
        self.conn.execute(
            "INSERT INTO signal_minutes(night_id, minute_ts, max_motion,"
            " threshold, max_speech) VALUES (?,?,?,?,?)",
            (night_id, minute_ts, max_motion, threshold, max_speech))
        self.conn.commit()

    def signal_minutes_for_night(self, night_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM signal_minutes WHERE night_id=? ORDER BY minute_ts",
            (night_id,)).fetchall()

    def set_clip_duration(self, clip_path: str, duration_s: float) -> None:
        # keyed by path: merged events share one file and update together
        self.conn.execute(
            "UPDATE events SET clip_duration_s=? WHERE clip_path=?",
            (duration_s, clip_path))
        self.conn.commit()

    def rate_event(self, event_id: int, rating: int | None) -> None:
        self.conn.execute("UPDATE events SET rating=? WHERE id=?",
                          (rating, event_id))
        self.conn.commit()

    def update_event_end(self, event_id: int, end_ts: float,
                         duration_s: float) -> None:
        # a restless phase that keeps re-firing extends one row in place
        self.conn.execute(
            "UPDATE events SET end_ts=?, duration_s=? WHERE id=?",
            (end_ts, duration_s, event_id))
        self.conn.commit()

    def reclassify_events(self, event_ids, kind: str) -> None:
        self.conn.executemany("UPDATE events SET kind=? WHERE id=?",
                              [(kind, i) for i in event_ids])
        self.conn.commit()

    def delete_event(self, event_id: int) -> bool:
        row = self.conn.execute(
            "SELECT clip_path FROM events WHERE id=?",
            (event_id,)).fetchone()
        if row is None:
            return False
        clip_path = row["clip_path"]
        self.conn.execute("DELETE FROM events WHERE id=?", (event_id,))
        self.conn.commit()
        if clip_path:
            others = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE clip_path=?",
                (clip_path,)).fetchone()[0]
            if others == 0:                 # merged clips die with last row
                Path(clip_path).unlink(missing_ok=True)
        return True

    def backfill_clip_durations(self) -> int:
        """Probe pre-v2.2 clips for their true length. Idempotent."""
        import cv2                    # lazy: keep storage importable without it
        rows = self.conn.execute(
            "SELECT DISTINCT clip_path FROM events "
            "WHERE clip_path IS NOT NULL AND clip_duration_s IS NULL"
        ).fetchall()
        done = 0
        for (path,) in rows:
            p = Path(path)
            if not p.exists():
                continue
            cap = cv2.VideoCapture(str(p))
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if frames > 0 and fps > 0:
                self.set_clip_duration(path, frames / fps)
                done += 1
        return done

    def events_for_night(self, night_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM events WHERE night_id=? ORDER BY start_ts",
            (night_id,)).fetchall()

    def night_by_date(self, date: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM nights WHERE date=? ORDER BY id DESC LIMIT 1",
            (date,)).fetchone()

    def latest_night(self) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM nights ORDER BY id DESC LIMIT 1").fetchone()

    def delete_night(self, night_id: int,
                     reports_dir: Path | None = None) -> bool:
        night = self.conn.execute("SELECT * FROM nights WHERE id=?",
                                  (night_id,)).fetchone()
        if night is None:
            return False
        for e in self.events_for_night(night_id):
            clip = e["clip_path"]
            if clip:
                p = Path(clip)
                if p.exists():
                    p.unlink()
                if p.parent.exists() and not any(p.parent.iterdir()):
                    p.parent.rmdir()
        if reports_dir is not None:
            report = Path(reports_dir) / f"{night['date']}.html"
            if report.exists():
                report.unlink()
        self.conn.execute("DELETE FROM events WHERE night_id=?", (night_id,))
        self.conn.execute("DELETE FROM nights WHERE id=?", (night_id,))
        self.conn.commit()
        return True

    def purge_old_clips(self, clips_root: Path, days: int,
                        now: float | None = None) -> list[Path]:
        now = time.time() if now is None else now
        cutoff = now - days * 86400
        deleted: list[Path] = []
        clips_root = Path(clips_root)
        if not clips_root.exists():
            return deleted
        for f in sorted(clips_root.rglob("*.mp4")):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted.append(f)
        for d in sorted(clips_root.iterdir()):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        return deleted

    def close(self) -> None:
        self.conn.close()


class NullStorage:
    """Storage stand-in for test/preview mode — records nothing."""

    def begin_night(self, date: str, start_ts: float, config_json: str) -> int:
        return 0

    def end_night(self, night_id: int, end_ts: float) -> None:
        pass

    def set_sleep_onset(self, night_id: int, onset_ts: float) -> None:
        pass

    def add_event(self, *args, **kwargs) -> int:
        return 0

    def set_clip_duration(self, clip_path: str, duration_s: float) -> None:
        pass

    def add_signal_minute(self, *args, **kwargs) -> None:
        pass

    def signal_minutes_for_night(self, night_id: int) -> list:
        return []

    def rate_event(self, event_id: int, rating: int | None) -> None:
        pass

    def update_event_end(self, event_id: int, end_ts: float,
                         duration_s: float) -> None:
        pass

    def reclassify_events(self, event_ids, kind: str) -> None:
        pass

    def events_for_night(self, night_id: int) -> list:
        return []

    def night_by_date(self, date: str) -> None:
        return None

    def latest_night(self) -> None:
        return None

    def purge_old_clips(self, clips_root, days, now=None) -> list:
        return []

    def close(self) -> None:
        pass
