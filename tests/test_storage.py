import json
import os
import time
from sleepwatch.storage import Storage

def test_night_lifecycle_and_events(tmp_path):
    st = Storage(tmp_path / "t.db")
    nid = st.begin_night("2026-08-11", 1000.0, json.dumps({"k": 6.0}))
    st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2, "clips/x.mp4")
    st.add_event(nid, "lighting", 1100.0, 1100.0, 0.0, 0.0, 0.0, None)
    st.end_night(nid, 2000.0)
    night = st.night_by_date("2026-08-11")
    assert night["end_ts"] == 2000.0
    evs = st.events_for_night(nid)
    assert [e["kind"] for e in evs] == ["movement", "lighting"]
    assert st.latest_night()["id"] == nid
    st.close()

def test_wal_mode(tmp_path):
    st = Storage(tmp_path / "t.db")
    assert st.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    st.close()

def test_delete_night_removes_everything(tmp_path):
    st = Storage(tmp_path / "t.db")
    clip = tmp_path / "clips" / "2026-08-11" / "event_010101.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    report = tmp_path / "reports" / "2026-08-11.html"
    report.parent.mkdir(parents=True)
    report.write_text("<html>")
    nid = st.begin_night("2026-08-11", 1000.0, "{}")
    st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2, str(clip))
    st.end_night(nid, 2000.0)
    assert st.delete_night(nid, reports_dir=report.parent) is True
    assert st.night_by_date("2026-08-11") is None
    assert st.events_for_night(nid) == []
    assert not clip.exists()
    assert not report.exists()
    assert st.delete_night(nid, reports_dir=report.parent) is False

def test_purge_old_clips(tmp_path):
    st = Storage(tmp_path / "t.db")
    clips = tmp_path / "clips"
    old_dir, new_dir = clips / "2026-08-01", clips / "2026-08-11"
    old_dir.mkdir(parents=True), new_dir.mkdir(parents=True)
    old, new = old_dir / "event_010101.mp4", new_dir / "event_020202.mp4"
    old.write_bytes(b"x"), new.write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 8 * 86400, now - 8 * 86400))
    deleted = st.purge_old_clips(clips, days=7, now=now)
    assert deleted == [old]
    assert not old.exists() and new.exists()
    assert not old_dir.exists()          # empty day dir removed
    st.close()

def test_purge_covers_exports_but_monitor_never_purges_timelapse(tmp_path):
    # exports are regenerable ffmpeg concats — same retention as clips.
    # timelapse films are deliberately NEVER purged: the morning film is
    # the artifact worth keeping, and it costs ~28 MB/night.
    st = Storage(tmp_path / "t.db")
    exports = tmp_path / "exports"
    exports.mkdir()
    old = exports / "2026-08-01.mp4"
    old.write_bytes(b"x")
    now = time.time()
    os.utime(old, (now - 8 * 86400, now - 8 * 86400))
    deleted = st.purge_old_clips(exports, days=7, now=now)
    assert deleted == [old] and not old.exists()
    st.close()


def test_signal_minutes_roundtrip(tmp_path):
    st = Storage(tmp_path / "s.db")
    nid = st.begin_night("2026-08-15", 1000.0, "{}")
    st.add_signal_minute(nid, 1020.0, 0.031, 0.05, 0.12)
    st.add_signal_minute(nid, 1080.0, 0.002, 0.05, None)
    rows = st.signal_minutes_for_night(nid)
    assert [r["minute_ts"] for r in rows] == [1020.0, 1080.0]
    assert rows[0]["max_motion"] == 0.031
    assert rows[0]["threshold"] == 0.05
    assert rows[0]["max_speech"] == 0.12
    assert rows[1]["max_speech"] is None
    st.close()


def test_clip_duration_column_and_setter(tmp_path):
    st = Storage(tmp_path / "s.db")
    nid = st.begin_night("2026-08-12", 1000.0, "{}")
    a = st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2,
                     "data/clips/2026-08-12/event_x.mp4")
    b = st.add_event(nid, "movement", 1020.0, 1022.0, 2.0, 0.3, 0.1,
                     "data/clips/2026-08-12/event_x.mp4")   # merged: same file
    c = st.add_event(nid, "settling", 1005.0, 1006.0, 1.0, 0.2, 0.1, None)
    st.set_clip_duration("data/clips/2026-08-12/event_x.mp4", 27.4)
    rows = {e["id"]: e for e in st.events_for_night(nid)}
    assert rows[a]["clip_duration_s"] == 27.4
    assert rows[b]["clip_duration_s"] == 27.4
    assert rows[c]["clip_duration_s"] is None
    st.close()

def test_delete_event_merge_aware(tmp_path):
    st = Storage(tmp_path / "d.db")
    nid = st.begin_night("2026-08-13", 1000.0, "{}")
    clip = tmp_path / "clips" / "2026-08-13" / "event_x.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")
    a = st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2,
                     str(clip))
    b = st.add_event(nid, "speech", 1016.0, 1018.0, 2.0, 0.9, None,
                     str(clip))                    # merged: same file
    assert st.delete_event(a) is True
    assert clip.exists()                           # b still references it
    assert st.delete_event(b) is True
    assert not clip.exists()                       # last referent gone
    assert st.events_for_night(nid) == []
    assert st.delete_event(999) is False


def test_rate_event_roundtrip(tmp_path):
    st = Storage(tmp_path / "r.db")
    nid = st.begin_night("2026-08-12", 1000.0, "{}")
    a = st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2, None)
    st.rate_event(a, 1)
    assert st.events_for_night(nid)[0]["rating"] == 1
    st.rate_event(a, -1)
    assert st.events_for_night(nid)[0]["rating"] == -1
    st.rate_event(a, None)
    assert st.events_for_night(nid)[0]["rating"] is None
    st.close()


def test_backfill_clip_durations_probes_files(tmp_path):
    import numpy as np
    from sleepwatch.clipwriter import ClipWriter
    st = Storage(tmp_path / "s.db")
    w = ClipWriter(clips_root=tmp_path / "clips", fps=15)
    f = np.full((120, 160, 3), 60, dtype=np.uint8)
    w.start("2026-08-12", event_start_wall=1755000000.0, preroll=[f])
    for _ in range(29):
        w.write(f)
    closed = w.close()                       # 30 frames = 2.0s
    nid = st.begin_night("2026-08-12", 1000.0, "{}")
    a = st.add_event(nid, "movement", 1010.0, 1011.0, 1.0, 0.4, 0.2,
                     str(closed.path))       # clip_duration_s left NULL
    b = st.add_event(nid, "movement", 1020.0, 1021.0, 1.0, 0.4, 0.2,
                     "data/clips/2026-08-12/gone.mp4")   # missing file
    done = st.backfill_clip_durations()
    assert done == 1
    rows = {e["id"]: e for e in st.events_for_night(nid)}
    assert abs(rows[a]["clip_duration_s"] - 2.0) < 0.2
    assert rows[b]["clip_duration_s"] is None
    assert st.backfill_clip_durations() == 0     # idempotent
    st.close()

def test_clip_duration_migrates_old_db(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
      CREATE TABLE nights(id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL, start_ts REAL NOT NULL, end_ts REAL,
        sleep_onset_ts REAL, config_json TEXT NOT NULL);
      CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id INTEGER NOT NULL, kind TEXT NOT NULL,
        start_ts REAL NOT NULL, end_ts REAL, duration_s REAL,
        peak_score REAL, mean_score REAL, clip_path TEXT);
    """)
    conn.commit()
    conn.close()
    st = Storage(db)                      # must ALTER TABLE without error
    cols = [r[1] for r in
            st.conn.execute("PRAGMA table_info(events)").fetchall()]
    assert "clip_duration_s" in cols
    st.close()
