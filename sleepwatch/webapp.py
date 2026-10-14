from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, render_template,
                   request, send_from_directory)

from . import __version__
from .config import Config
from .report import (compute_segment_stats, generate_report,
                     restlessness_score, sleep_segments)
from .storage import Storage

STREAM_BOUNDARY = b"--sleepwatchframe"


def create_app(runner, cfg: Config) -> Flask:
    app = Flask(__name__)
    # the code reloader restarts the process on .py edits but Jinja caches
    # templates — without this, dashboard.html changes silently do nothing
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    db_path = cfg.data_dir / "sleepwatch.db"

    def read_storage() -> Storage:
        return Storage(db_path)

    if db_path.exists():                  # measure pre-v2.2 clips once
        st = read_storage()
        st.backfill_clip_durations()
        st.close()

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.get("/api/status")
    def status():
        out = runner.status()
        out["version"] = __version__
        return jsonify(out)

    @app.post("/api/start")
    def start():
        return jsonify({"ok": runner.start_night()})

    @app.post("/api/stop")
    def stop():
        return jsonify({"ok": runner.stop_night()})

    PADDING_CHOICES = (5.0, 10.0, 15.0)

    @app.get("/api/settings")
    def get_settings():
        return jsonify({"padding_s": runner.get_padding(),
                        "choices": [int(c) for c in PADDING_CHOICES]})

    @app.post("/api/settings")
    def set_settings():
        body = request.get_json(silent=True) or {}
        try:
            padding = float(body.get("padding_s"))
        except (TypeError, ValueError):
            padding = None
        if padding not in PADDING_CHOICES:
            return jsonify({"ok": False,
                            "error": "padding_s must be one of 5, 10, 15"}), 422
        runner.set_padding(padding)
        return jsonify({"ok": True, "padding_s": padding})

    @app.get("/api/nights")
    def nights():
        st = read_storage()
        rows = st.conn.execute(
            "SELECT * FROM nights ORDER BY id DESC LIMIT 60").fetchall()
        out = []
        for n in rows:
            all_evs = st.events_for_night(n["id"])
            evs = [e for e in all_evs
                   if e["kind"] in ("movement", "truncated")]
            spoke = sum(1 for e in all_evs if e["kind"] == "speech")
            end = n["end_ts"] or (evs[-1]["end_ts"] if evs else n["start_ts"])
            segments = sleep_segments(n, all_evs)
            stats = compute_segment_stats(
                segments or [(n["start_ts"], end)], evs)
            out.append({
                "id": n["id"], "date": n["date"],
                "start_ts": n["start_ts"], "end_ts": n["end_ts"],
                "sleep_onset_ts": n["sleep_onset_ts"],
                "hours": round((end - n["start_ts"]) / 3600, 2),
                "movements": stats.total,
                "per_hour": round(stats.per_hour, 2),
                "longest_still_s": round(stats.longest_still_s),
                "restless_pct": round(stats.restless_pct, 2),
                "spoke": spoke,
                "open": n["end_ts"] is None,
                "has_timelapse": (cfg.data_dir / "timelapse" /
                                  f"{n['date']}.mp4").exists(),
            })
        st.close()
        return jsonify({"nights": out})

    @app.delete("/api/events/<int:event_id>")
    def delete_event(event_id: int):
        st = read_storage()
        ok = st.delete_event(event_id)
        st.close()
        if not ok:
            abort(404)
        return jsonify({"ok": True})

    @app.post("/api/events/<int:event_id>/rate")
    def rate_event(event_id: int):
        body = request.get_json(silent=True) or {}
        rating = body.get("rating")
        if rating not in (1, -1, 0):
            return jsonify({"ok": False,
                            "error": "rating must be 1, -1 or 0"}), 422
        st = read_storage()
        st.rate_event(event_id, None if rating == 0 else rating)
        st.close()
        return jsonify({"ok": True,
                        "rating": None if rating == 0 else rating})

    @app.get("/api/trends")
    def trends():
        st = read_storage()
        rows = st.conn.execute(
            "SELECT * FROM nights WHERE end_ts IS NOT NULL "
            "ORDER BY id").fetchall()
        nights_out = []
        hour_counts = [0] * 24
        for n in rows:
            all_evs = st.events_for_night(n["id"])
            evs = [e for e in all_evs
                   if e["kind"] in ("movement", "truncated")]
            onset = n["sleep_onset_ts"]
            sleep_start = onset if onset else n["start_ts"]
            segments = sleep_segments(n, all_evs)
            stats = compute_segment_stats(
                segments or [(sleep_start, n["end_ts"])], evs)
            for e in evs:
                hour_counts[time.localtime(e["start_ts"]).tm_hour] += 1
            score = restlessness_score(
                st.signal_minutes_for_night(n["id"]), segments)
            nights_out.append({
                "date": n["date"],
                "latency_s": (round(onset - n["start_ts"])
                              if onset else None),
                "movements": stats.total,
                "per_hour": round(stats.per_hour, 2),
                "longest_still_s": round(stats.longest_still_s),
                "restless_pct": round(stats.restless_pct, 2),
                "restlessness": (round(score, 1)
                                 if score is not None else None),
                "hours": round((n["end_ts"] - n["start_ts"]) / 3600, 2),
            })
        st.close()
        return jsonify({"nights": nights_out, "hour_counts": hour_counts})

    @app.get("/api/nights/<int:night_id>/events")
    def night_events(night_id: int):
        st = read_storage()
        clips_root = (cfg.data_dir / "clips").resolve()
        out = []
        for e in st.events_for_night(night_id):
            clip_url = None
            clip = e["clip_path"]
            if clip:
                try:
                    rel = Path(clip).resolve().relative_to(clips_root)
                    clip_url = f"/clips/{rel.as_posix()}"
                except ValueError:
                    clip_url = None
            out.append({"id": e["id"], "kind": e["kind"],
                        "start_ts": e["start_ts"], "end_ts": e["end_ts"],
                        "duration_s": e["duration_s"],
                        "clip_duration_s": e["clip_duration_s"],
                        "peak_score": e["peak_score"],
                        "rating": e["rating"],
                        "coverage": e["coverage"],
                        "magnitude": e["magnitude"],
                        "clip_url": clip_url})
        st.close()
        return jsonify({"events": out})

    @app.get("/api/nights/<int:night_id>/signal")
    def night_signal(night_id: int):
        st = read_storage()
        out = [{"minute_ts": m["minute_ts"],
                "max_motion": m["max_motion"],
                "threshold": m["threshold"],
                "max_speech": m["max_speech"]}
               for m in st.signal_minutes_for_night(night_id)]
        st.close()
        return jsonify({"minutes": out})

    @app.delete("/api/nights/<int:night_id>")
    def delete_night(night_id: int):
        st = read_storage()
        night = st.conn.execute("SELECT * FROM nights WHERE id=?",
                                (night_id,)).fetchone()
        if night is not None and night["end_ts"] is None:
            st.close()
            return jsonify({"ok": False, "error": "night still open"}), 409
        date = night["date"] if night is not None else None
        ok = st.delete_night(night_id,
                             reports_dir=cfg.data_dir / "reports")
        st.close()
        if ok and date:
            (cfg.data_dir / "exports" / f"{date}.mp4").unlink(missing_ok=True)
            (cfg.data_dir / "timelapse" /
             f"{date}.mp4").unlink(missing_ok=True)
        return jsonify({"ok": ok})

    @app.post("/api/nights/<int:night_id>/export")
    def export_night(night_id: int):
        st = read_storage()
        night = st.conn.execute("SELECT * FROM nights WHERE id=?",
                                (night_id,)).fetchone()
        if night is None:
            st.close()
            abort(404)
        if night["end_ts"] is None:
            st.close()
            return jsonify({"ok": False, "error": "night still open"}), 409
        seen: set[str] = set()
        clips: list[Path] = []
        for e in st.events_for_night(night_id):
            p = e["clip_path"]
            if (e["kind"] in ("movement", "truncated") and p
                    and p not in seen):
                seen.add(p)                 # merged events share one file
                if Path(p).exists():
                    clips.append(Path(p))
        st.close()
        if not clips:
            return jsonify({"ok": False, "error": "no clips"}), 422
        if shutil.which("ffmpeg") is None:
            return jsonify({"ok": False,
                            "error": "ffmpeg not installed"}), 501
        exports = cfg.data_dir / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        out = exports / f"{night['date']}.mp4"
        listfile = exports / f".{night['date']}.concat.txt"
        listfile.write_text(
            "".join(f"file '{c.resolve()}'\n" for c in clips))
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(out)],
            capture_output=True, timeout=120)
        listfile.unlink(missing_ok=True)
        if proc.returncode != 0 or not out.exists():
            return jsonify({"ok": False, "error": "ffmpeg failed"}), 500
        return jsonify({"ok": True, "url": f"/exports/{out.name}",
                        "bytes": out.stat().st_size})

    @app.get("/exports/<path:relpath>")
    def exports_files(relpath: str):
        root = (cfg.data_dir / "exports").resolve()
        return send_from_directory(root, relpath)

    @app.get("/timelapse/<path:relpath>")
    def timelapse_files(relpath: str):
        root = (cfg.data_dir / "timelapse").resolve()
        return send_from_directory(root, relpath)

    @app.get("/report/<date>")
    def report(date: str):
        st = read_storage()
        night = st.night_by_date(date)
        if night is None:
            st.close()
            abort(404)
        events = st.events_for_night(night["id"])
        minutes = st.signal_minutes_for_night(night["id"])
        out = cfg.data_dir / "reports" / f"{date}.html"
        generate_report(night, events, out, signal_minutes=minutes)
        st.close()
        # clip hrefs are ../clips/… for file:// opens; rewrite when served
        return Response(out.read_text()
                        .replace('href="../clips/', 'href="/clips/')
                        .replace('src="../timelapse/', 'src="/timelapse/'),
                        mimetype="text/html")

    @app.get("/clips/<path:relpath>")
    def clips(relpath: str):
        root = (cfg.data_dir / "clips").resolve()
        return send_from_directory(root, relpath)

    @app.get("/stream.mjpg")
    def stream():
        def gen():
            last = None
            while True:
                jpeg = runner.latest_jpeg()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield (STREAM_BOUNDARY + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " +
                           str(len(jpeg)).encode() + b"\r\n\r\n" +
                           jpeg + b"\r\n")
                time.sleep(1 / 30)
        return Response(
            gen(), mimetype="multipart/x-mixed-replace; "
                            "boundary=sleepwatchframe")

    return app
