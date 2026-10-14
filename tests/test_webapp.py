import json

from sleepwatch.config import Config
from sleepwatch.storage import Storage
from sleepwatch.webapp import create_app


class StubRunner:
    def __init__(self):
        self.monitoring = False
        self._jpeg = b"\xff\xd8fakejpeg"
    def latest_jpeg(self):
        return self._jpeg
    def status(self):
        return {"state": "QUIET", "score": 0.001, "threshold": 0.004,
                "monitoring": self.monitoring, "night_date": None,
                "skeleton": False}
    def start_night(self):
        if self.monitoring:
            return False
        self.monitoring = True
        return True
    def stop_night(self):
        if not self.monitoring:
            return False
        self.monitoring = False
        return True
    def get_padding(self):
        return getattr(self, "_padding", 10.0)
    def set_padding(self, seconds):
        self._padding = float(seconds)


def make_app(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nid = st.begin_night("2026-08-11", 1000.0, json.dumps({}))
    clip = cfg.data_dir / "clips" / "2026-08-11" / "event_001650.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"fake")
    st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2, str(clip))
    st.set_sleep_onset(nid, 1005.0)
    st.end_night(nid, 1000.0 + 8 * 3600)
    st.close()
    runner = StubRunner()
    return create_app(runner, cfg), runner


def test_status_endpoint(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    r = c.get("/api/status")
    assert r.status_code == 200
    assert r.get_json()["state"] == "QUIET"


def test_nights_stats_use_sleep_segments(tmp_path):
    # multi-segment night: 1000s + 1000s asleep inside an 8h monitored
    # span — per_hour must divide by 2000s asleep, not 8h
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nid = st.begin_night("2026-08-15", 1000.0, json.dumps({}))
    st.set_sleep_onset(nid, 2000.0)
    for kind, ts in (("onset", 2000.0), ("wake", 3000.0),
                     ("onset", 10000.0), ("wake", 11000.0)):
        st.add_event(nid, kind, ts, ts, 0.0, None, None, None)
    st.add_event(nid, "movement", 2500.0, 2510.0, 10.0, 0.4, 0.2, None)
    st.end_night(nid, 1000.0 + 8 * 3600)
    st.close()
    app = create_app(StubRunner(), cfg)
    n = app.test_client().get("/api/nights").get_json()["nights"][0]
    assert n["movements"] == 1
    assert abs(n["per_hour"] - 1 / (2000 / 3600)) < 0.01   # 1.8, not 0.13


def test_status_carries_build_version(tmp_path):
    # two nights ran on a stale build unnoticed — version must be visible
    import sleepwatch
    app, _ = make_app(tmp_path)
    body = app.test_client().get("/api/status").get_json()
    assert body["version"] == sleepwatch.__version__
    assert sleepwatch.__version__.count(".") == 2

def test_start_stop_cycle(tmp_path):
    app, runner = make_app(tmp_path)
    c = app.test_client()
    assert c.post("/api/start").get_json() == {"ok": True}
    assert runner.monitoring is True
    assert c.post("/api/start").get_json() == {"ok": False}
    assert c.post("/api/stop").get_json() == {"ok": True}
    assert runner.monitoring is False

def test_nights_endpoint(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    r = c.get("/api/nights")
    nights = r.get_json()["nights"]
    assert len(nights) == 1
    n = nights[0]
    assert n["date"] == "2026-08-11"
    assert n["movements"] == 1
    assert n["restless_pct"] > 0

def test_night_events_endpoint_with_clip_urls(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    nid = c.get("/api/nights").get_json()["nights"][0]["id"]
    events = c.get(f"/api/nights/{nid}/events").get_json()["events"]
    assert len(events) == 1
    assert events[0]["kind"] == "movement"
    assert events[0]["clip_url"] == "/clips/2026-08-11/event_001650.mp4"

def test_nights_include_sleep_onset(tmp_path):
    app, _ = make_app(tmp_path)
    n = app.test_client().get("/api/nights").get_json()["nights"][0]
    assert n["sleep_onset_ts"] == 1005.0

def test_delete_night_endpoint(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    nid = c.get("/api/nights").get_json()["nights"][0]["id"]
    assert c.delete(f"/api/nights/{nid}").get_json() == {"ok": True}
    assert c.get("/api/nights").get_json()["nights"] == []
    assert c.delete(f"/api/nights/{nid}").get_json() == {"ok": False}

def test_delete_refuses_open_night(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    from sleepwatch.storage import Storage as S
    from sleepwatch.config import Config as C2
    st = S((tmp_path / "data") / "sleepwatch.db")
    open_id = st.begin_night("2026-08-12", 2000.0, "{}")
    st.close()
    r = c.delete(f"/api/nights/{open_id}")
    assert r.status_code == 409
    assert r.get_json()["ok"] is False

def test_index_serves_page(tmp_path):
    app, _ = make_app(tmp_path)
    r = app.test_client().get("/")
    assert r.status_code == 200
    assert b"sleep-watch" in r.data

def test_stream_yields_jpeg(tmp_path):
    app, _ = make_app(tmp_path)
    r = app.test_client().get("/stream.mjpg")
    assert r.status_code == 200
    assert "multipart/x-mixed-replace" in r.content_type
    chunk = next(r.response)
    assert b"fakejpeg" in chunk

def test_events_include_clip_duration(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    nights = c.get("/api/nights").get_json()["nights"]
    evs = c.get(f"/api/nights/{nights[0]['id']}/events").get_json()["events"]
    assert "clip_duration_s" in evs[0]

def test_settings_roundtrip(tmp_path):
    app, runner = make_app(tmp_path)
    c = app.test_client()
    got = c.get("/api/settings").get_json()
    assert got == {"padding_s": 10.0, "choices": [5, 10, 15]}
    assert c.post("/api/settings", json={"padding_s": 15}
                  ).get_json() == {"ok": True, "padding_s": 15.0}
    assert runner.get_padding() == 15.0
    r = c.post("/api/settings", json={"padding_s": 42})
    assert r.status_code == 422

import shutil as _shutil

import numpy as np
import pytest

from sleepwatch.clipwriter import ClipWriter


def _real_clip(root, date, stamp_wall, n_frames):
    w = ClipWriter(clips_root=root, fps=15)
    f = np.full((120, 160, 3), 60, dtype=np.uint8)
    w.start(date, event_start_wall=stamp_wall, preroll=[f])
    for _ in range(n_frames - 1):
        w.write(f)
    return w.close()


@pytest.mark.skipif(_shutil.which("ffmpeg") is None,
                    reason="ffmpeg not installed")
def test_export_concatenates_night(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nid = st.begin_night("2026-08-12", 1000.0, "{}")
    c1 = _real_clip(cfg.data_dir / "clips", "2026-08-12", 1755000000.0, 30)
    c2 = _real_clip(cfg.data_dir / "clips", "2026-08-12", 1755000100.0, 30)
    st.add_event(nid, "movement", 1010.0, 1015.0, 5.0, 0.4, 0.2,
                 str(c1.path))
    st.add_event(nid, "movement", 1020.0, 1025.0, 5.0, 0.4, 0.2,
                 str(c2.path))
    st.end_night(nid, 1000.0 + 8 * 3600)
    st.close()
    app = create_app(StubRunner(), cfg)
    c = app.test_client()
    r = c.post(f"/api/nights/{nid}/export")
    body = r.get_json()
    assert r.status_code == 200 and body["ok"] is True
    assert body["url"] == "/exports/2026-08-12.mp4"
    assert (cfg.data_dir / "exports" / "2026-08-12.mp4").stat().st_size > 0
    assert c.get(body["url"]).status_code == 200


def test_delete_event_endpoint(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    nights = c.get("/api/nights").get_json()["nights"]
    evs = c.get(f"/api/nights/{nights[0]['id']}/events").get_json()["events"]
    eid = evs[0]["id"]
    assert c.delete(f"/api/events/{eid}").get_json() == {"ok": True}
    evs = c.get(f"/api/nights/{nights[0]['id']}/events").get_json()["events"]
    assert eid not in [e["id"] for e in evs]
    assert c.delete("/api/events/99999").status_code == 404


def test_nights_include_spoke_count(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nid = st.begin_night("2026-08-13", 1000.0, "{}")
    st.add_event(nid, "speech", 5000.0, 5004.0, 4.0, 0.9, None, None)
    st.add_event(nid, "speech", 9000.0, 9002.0, 2.0, 0.8, None, None)
    st.add_event(nid, "movement", 7000.0, 7003.0, 3.0, 0.4, 0.2, None)
    st.end_night(nid, 1000.0 + 8 * 3600)
    st.close()
    app = create_app(StubRunner(), cfg)
    nights = app.test_client().get("/api/nights").get_json()["nights"]
    assert nights[0]["spoke"] == 2


def test_timelapse_flag_route_and_delete(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    nid = st.begin_night("2026-08-12", 1000.0, "{}")
    st.end_night(nid, 1000.0 + 8 * 3600)
    st.close()
    tl = cfg.data_dir / "timelapse" / "2026-08-12.mp4"
    tl.parent.mkdir(parents=True)
    tl.write_bytes(b"fakefilm")
    app = create_app(StubRunner(), cfg)
    c = app.test_client()
    nights = c.get("/api/nights").get_json()["nights"]
    assert nights[0]["has_timelapse"] is True
    assert c.get("/timelapse/2026-08-12.mp4").status_code == 200
    c.delete(f"/api/nights/{nid}")
    assert not tl.exists()


def test_rate_event_endpoint(tmp_path):
    app, _ = make_app(tmp_path)
    c = app.test_client()
    nights = c.get("/api/nights").get_json()["nights"]
    evs = c.get(f"/api/nights/{nights[0]['id']}/events").get_json()["events"]
    eid = evs[0]["id"]
    assert "rating" in evs[0]
    r = c.post(f"/api/events/{eid}/rate", json={"rating": 1})
    assert r.get_json() == {"ok": True, "rating": 1}
    evs = c.get(f"/api/nights/{nights[0]['id']}/events").get_json()["events"]
    assert evs[0]["rating"] == 1
    assert c.post(f"/api/events/{eid}/rate",
                  json={"rating": 0}).get_json() == {"ok": True,
                                                     "rating": None}
    assert c.post(f"/api/events/{eid}/rate",
                  json={"rating": 5}).status_code == 422


def test_trends_aggregates_closed_nights(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    n1 = st.begin_night("2026-08-11", 1000.0, "{}")
    st.set_sleep_onset(n1, 1000.0 + 600)
    st.add_event(n1, "movement", 5000.0, 5004.0, 4.0, 0.4, 0.2, None)
    st.end_night(n1, 1000.0 + 8 * 3600)
    n2 = st.begin_night("2026-08-12", 90000.0, "{}")
    st.add_event(n2, "movement", 91000.0, 91002.0, 2.0, 0.3, 0.1, None)
    # open third night — must NOT appear
    st.begin_night("2026-08-13", 180000.0, "{}")
    st.end_night(n2, 90000.0 + 7 * 3600)
    st.close()
    app = create_app(StubRunner(), cfg)
    body = app.test_client().get("/api/trends").get_json()
    assert [n["date"] for n in body["nights"]] == ["2026-08-11", "2026-08-12"]
    assert body["nights"][0]["latency_s"] == 600
    assert body["nights"][1]["latency_s"] is None
    assert body["nights"][0]["movements"] == 1
    assert len(body["hour_counts"]) == 24
    assert sum(body["hour_counts"]) == 2


def test_export_rejects_open_or_empty_night(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    st = Storage(cfg.data_dir / "sleepwatch.db")
    open_id = st.begin_night("2026-08-13", 2000.0, "{}")
    empty_id = st.begin_night("2026-08-14", 3000.0, "{}")
    st.end_night(empty_id, 3000.0 + 3600)
    st.close()
    app = create_app(StubRunner(), cfg)
    c = app.test_client()
    assert c.post(f"/api/nights/{open_id}/export").status_code == 409
    assert c.post(f"/api/nights/{empty_id}/export").status_code == 422
    assert c.post("/api/nights/9999/export").status_code == 404
