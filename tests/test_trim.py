import time

from scripts.trim_night import trim
from sleepwatch.storage import Storage


def _local(h, m):
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0,
                        0, 0, -1))


def test_trim_reclassifies_and_stamps_end(tmp_path):
    db = tmp_path / "s.db"
    st = Storage(db)
    nid = st.begin_night("2026-08-13", _local(1, 0), "{}")
    st.set_sleep_onset(nid, _local(1, 20))
    st.add_event(nid, "movement", _local(3, 0), _local(3, 0) + 4,
                 4.0, 0.3, 0.1, None)               # real sleep movement
    st.add_event(nid, "movement", _local(9, 40), _local(9, 40) + 3,
                 3.0, 0.4, 0.2, None)               # after wake
    st.add_event(nid, "movement", _local(9, 50), _local(9, 50) + 2,
                 2.0, 0.4, 0.2, None)               # after wake
    st.end_night(nid, _local(10, 5))
    st.close()

    assert trim(db, nid, "09:34") == 2

    st = Storage(db)
    night = st.conn.execute("SELECT * FROM nights WHERE id=?",
                            (nid,)).fetchone()
    evs = st.events_for_night(nid)
    kinds = [e["kind"] for e in evs]
    assert kinds.count("movement") == 1             # 03:00 survives
    assert kinds.count("awake") == 2
    assert kinds.count("wake") == 1
    assert night["end_ts"] == _local(9, 40)
    # rerun: nothing left to trim
    assert trim(db, nid, "09:34") == 0
    st.close()
