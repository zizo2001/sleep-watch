from types import SimpleNamespace as NS
from sleepwatch.report import compute_stats, timeline_svg, generate_report

NIGHT_START, NIGHT_END = 1000.0, 1000.0 + 8 * 3600   # 8h night

def ev(start, dur):
    return NS(start_ts=start, end_ts=start + dur, duration_s=dur,
              kind="movement", peak_score=0.3, mean_score=0.1,
              clip_path=None, id=1)

def test_stats_empty_night():
    s = compute_stats(NIGHT_START, NIGHT_END, [])
    assert s.total == 0 and s.per_hour == 0.0
    assert s.longest_still_s == 8 * 3600 and s.restless_pct == 0.0

def test_stats_basic():
    events = [ev(NIGHT_START + 3600, 60), ev(NIGHT_START + 7200, 30)]
    s = compute_stats(NIGHT_START, NIGHT_END, events)
    assert s.total == 2
    assert abs(s.per_hour - 0.25) < 1e-9
    assert abs(s.longest_still_s - (NIGHT_END - (NIGHT_START + 7230))) < 1e-9
    assert abs(s.restless_pct - (90 / (8 * 3600) * 100)) < 1e-9

def test_timeline_svg_has_event_rects():
    events = [ev(NIGHT_START + 3600, 60)]
    svg = timeline_svg(NIGHT_START, NIGHT_END, events)
    assert svg.startswith("<svg") and svg.count('class="ev"') == 1

def test_report_separates_settling_from_sleep(tmp_path):
    onset = NIGHT_START + 24 * 60                 # fell asleep after 24m
    settling_ev = ev(NIGHT_START + 300, 40)
    settling_ev.kind = "settling"
    sleep_ev = ev(NIGHT_START + 2 * 3600, 60)
    night = {"id": 1, "date": "2026-08-11", "start_ts": NIGHT_START,
             "end_ts": NIGHT_END, "sleep_onset_ts": onset,
             "config_json": "{}"}
    out = generate_report(night, [settling_ev, sleep_ev], tmp_path / "r.html")
    html = out.read_text()
    assert "to fall asleep" in html and "24m" in html
    assert "settling" in html
    assert "<b>1</b><span>movements asleep</span>" in html   # not 2

def test_report_renders_signal_strip(tmp_path):
    # per-minute peak motion — shows the moves the detector never fired on
    night = {"id": 1, "date": "2026-08-15", "start_ts": NIGHT_START,
             "end_ts": NIGHT_END, "config_json": "{}"}
    minutes = [NS(minute_ts=NIGHT_START + i * 60, max_motion=0.002 * i,
                  threshold=0.05, max_speech=None) for i in range(30)]
    out = generate_report(night, [ev(NIGHT_START + 3600, 60)],
                          tmp_path / "r.html", signal_minutes=minutes)
    html = out.read_text()
    assert "sigstrip" in html             # the strip chart rendered
    assert html.count('<rect class="sig') == 30
    # near-miss minutes (>=half threshold, no event fired) stand out
    assert '<rect class="sig near"' in html


def test_report_without_signal_minutes_omits_strip(tmp_path):
    night = {"id": 1, "date": "2026-08-15", "start_ts": NIGHT_START,
             "end_ts": NIGHT_END, "config_json": "{}"}
    out = generate_report(night, [ev(NIGHT_START + 3600, 60)],
                          tmp_path / "r.html")
    assert "sigstrip" not in out.read_text()


def test_generate_report_writes_html(tmp_path):
    night = {"id": 1, "date": "2026-08-11", "start_ts": NIGHT_START,
             "end_ts": NIGHT_END, "config_json": "{}"}
    out = generate_report(night, [ev(NIGHT_START + 3600, 60)],
                          tmp_path / "r.html")
    html = out.read_text()
    assert "<html" in html and "2026-08-11" in html and "<svg" in html
    assert "http" not in html.split("<!--")[0].replace(
        "http://www.w3.org/2000/svg", "")  # no external resources (svg ns ok)

def test_report_shows_clip_length_column(tmp_path):
    night = {"date": "2026-08-12", "start_ts": 1000.0,
             "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1600.0}
    events = [{"kind": "movement", "start_ts": 2000.0, "end_ts": 2009.6,
               "duration_s": 9.6, "peak_score": 0.4, "mean_score": 0.2,
               "clip_path": None, "clip_duration_s": 26.6}]
    out = generate_report(night, events, tmp_path / "r.html")
    html_text = out.read_text()
    assert "Clip length" in html_text
    assert "27s" in html_text

def test_report_colors_kinds(tmp_path):
    night = {"date": "2026-08-12", "start_ts": 1000.0,
             "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1600.0}
    events = [
        {"kind": "movement", "start_ts": 2000.0, "end_ts": 2005.0,
         "duration_s": 5.0, "peak_score": 0.4, "mean_score": 0.2,
         "clip_path": None, "clip_duration_s": None},
        {"kind": "truncated", "start_ts": 3000.0, "end_ts": 3005.0,
         "duration_s": 5.0, "peak_score": 0.5, "mean_score": 0.2,
         "clip_path": None, "clip_duration_s": None},
    ]
    html_text = generate_report(night, events,
                                tmp_path / "r.html").read_text()
    assert 'class="ev-trunc"' in html_text
    assert "--violet" in html_text

def test_report_surfaces_awake_and_ambient(tmp_path):
    night = {"date": "2026-08-13", "start_ts": 1000.0,
             "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1600.0}
    events = [
        {"kind": "movement", "start_ts": 5000.0, "end_ts": 5004.0,
         "duration_s": 4.0, "peak_score": 0.4, "mean_score": 0.2,
         "clip_path": None, "clip_duration_s": None, "id": 1,
         "rating": None},
        {"kind": "ambient", "start_ts": 9000.0, "end_ts": 9003.0,
         "duration_s": 3.0, "peak_score": 0.02, "mean_score": None,
         "clip_path": None, "clip_duration_s": None, "id": 2,
         "rating": None},
        {"kind": "awake", "start_ts": 26000.0, "end_ts": 26004.0,
         "duration_s": 4.0, "peak_score": 0.3, "mean_score": None,
         "clip_path": None, "clip_duration_s": None, "id": 3,
         "rating": None},
    ]
    html_text = generate_report(night, events,
                                tmp_path / "r.html").read_text()
    assert "· ambient" in html_text
    assert "· awake" in html_text
    assert 'class="ev-ambient"' in html_text
    assert 'class="ev-awake"' in html_text
    # only the real movement counts in stats
    assert "<b>1</b><span>movements asleep</span>" in html_text


def test_report_surfaces_speech_and_wake(tmp_path):
    night = {"date": "2026-08-13", "start_ts": 1000.0,
             "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1600.0}
    events = [
        {"kind": "speech", "start_ts": 5000.0, "end_ts": 5004.0,
         "duration_s": 4.0, "peak_score": 0.9, "mean_score": None,
         "clip_path": None, "clip_duration_s": None, "id": 7,
         "rating": None},
        {"kind": "speech", "start_ts": 9000.0, "end_ts": 9002.0,
         "duration_s": 2.0, "peak_score": 0.8, "mean_score": None,
         "clip_path": None, "clip_duration_s": None, "id": 8,
         "rating": None},
        {"kind": "wake", "start_ts": 26000.0, "end_ts": 26000.0,
         "duration_s": 0.0, "peak_score": None, "mean_score": None,
         "clip_path": None, "clip_duration_s": None, "id": 9,
         "rating": None},
    ]
    html_text = generate_report(night, events,
                                tmp_path / "r.html").read_text()
    assert 'class="ev-speech"' in html_text        # violet timeline bars
    assert 'class="wakeline"' in html_text         # wake tick
    assert "sleep talking" in html_text            # spoke stat tile
    assert "· speech" in html_text                 # table rows labeled


def test_report_embeds_inline_player(tmp_path):
    clip = tmp_path / "clips" / "2026-08-12" / "event_011221.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"fake")
    night = {"date": "2026-08-12", "start_ts": 1000.0,
             "end_ts": 1000.0 + 8 * 3600, "sleep_onset_ts": 1600.0}
    events = [{"kind": "movement", "start_ts": 2000.0, "end_ts": 2009.6,
               "duration_s": 9.6, "peak_score": 0.4, "mean_score": 0.2,
               "clip_path": str(clip), "clip_duration_s": 16.8}]
    html_text = generate_report(night, events,
                                tmp_path / "r.html").read_text()
    assert 'id="rtheater"' in html_text          # player container
    assert 'class="watch"' in html_text          # intercepted links
    assert "rt-close" in html_text               # close control wired

def test_fmt_dur_precision():
    from sleepwatch.report import _fmt_dur
    assert _fmt_dur(0.9) == "0.9s"
    assert _fmt_dur(41.4) == "41s"
    assert _fmt_dur(9.6) == "10s"
    assert _fmt_dur(86) == "1m 26s"
    assert _fmt_dur(8135) == "2h 15m"


# ---- v2.5: multi-segment nights ------------------------------------------

def marker(kind, ts):
    return {"kind": kind, "start_ts": ts, "end_ts": ts, "duration_s": 0.0,
            "peak_score": None, "mean_score": None, "clip_path": None,
            "clip_duration_s": None, "id": 0, "rating": None}


def seg_night(**over):
    night = {"id": 1, "date": "2026-08-14", "start_ts": NIGHT_START,
             "end_ts": NIGHT_END, "sleep_onset_ts": NIGHT_START + 1000,
             "config_json": "{}"}
    night.update(over)
    return night


def test_sleep_segments_from_markers():
    from sleepwatch.report import sleep_segments
    events = [marker("onset", NIGHT_START + 1000),
              marker("wake", NIGHT_START + 2000),
              marker("onset", NIGHT_START + 4000),
              marker("wake", NIGHT_START + 8000)]
    assert sleep_segments(seg_night(), events) == [
        (NIGHT_START + 1000, NIGHT_START + 2000),
        (NIGHT_START + 4000, NIGHT_START + 8000)]


def test_sleep_segments_trailing_onset_closes_at_night_end():
    from sleepwatch.report import sleep_segments
    events = [marker("onset", NIGHT_START + 1000),
              marker("wake", NIGHT_START + 2000),
              marker("onset", NIGHT_START + 4000)]     # asleep at close
    assert sleep_segments(seg_night(), events) == [
        (NIGHT_START + 1000, NIGHT_START + 2000),
        (NIGHT_START + 4000, NIGHT_END)]


def test_sleep_segments_fallback_for_pre_v25_nights():
    from sleepwatch.report import sleep_segments
    # v2.4 data: sleep_onset_ts column, wake marker, NO onset markers
    events = [marker("wake", NIGHT_START + 8000)]
    assert sleep_segments(seg_night(), events) == [
        (NIGHT_START + 1000, NIGHT_START + 8000)]
    # ...and with no wake either: one open segment to night end
    assert sleep_segments(seg_night(), []) == [(NIGHT_START + 1000, NIGHT_END)]


def test_sleep_segments_never_slept():
    from sleepwatch.report import sleep_segments
    assert sleep_segments(seg_night(sleep_onset_ts=None), []) == []


def test_segment_stats_ignore_awake_gaps():
    from sleepwatch.report import compute_segment_stats
    segs = [(1000.0, 3000.0), (11000.0, 12000.0)]
    s = compute_segment_stats(segs, [ev(1500.0, 10)])
    assert s.total == 1
    assert abs(s.per_hour - 1 / (3000 / 3600)) < 1e-9   # 3000s asleep, not 11000
    # longest still is within a segment — the 8000s awake gap must not count
    assert abs(s.longest_still_s - 1490.0) < 1e-9       # 1510 → 3000
    assert abs(s.restless_pct - 10 / 3000 * 100) < 1e-9


def test_report_multisegment_tile_and_stats(tmp_path):
    events = [marker("onset", NIGHT_START + 1000),
              marker("wake", NIGHT_START + 2000),
              marker("onset", NIGHT_START + 4000),
              marker("wake", NIGHT_START + 8000),
              dict(kind="movement", start_ts=NIGHT_START + 1500,
                   end_ts=NIGHT_START + 1510, duration_s=10.0,
                   peak_score=0.4, mean_score=0.2, clip_path=None,
                   clip_duration_s=None, id=5, rating=None),
              dict(kind="awake", start_ts=NIGHT_START + 2500,
                   end_ts=NIGHT_START + 2504, duration_s=4.0,
                   peak_score=0.3, mean_score=None, clip_path=None,
                   clip_duration_s=None, id=6, rating=None)]
    html_text = generate_report(seg_night(), events,
                                tmp_path / "r.html").read_text()
    # slept 1000 + 4000 = 5000s in two stretches
    assert "1h 23m" in html_text and "2 stretches" in html_text
    assert "<b>1</b><span>movements asleep</span>" in html_text
    assert "slept" in html_text                     # sub line shows the span


def test_report_footer_shows_version(tmp_path):
    import sleepwatch
    html_text = generate_report(seg_night(), [],
                                tmp_path / "r.html").read_text()
    assert f"sleep-watch v{sleepwatch.__version__}" in html_text


def test_report_timeline_shades_awake_gaps(tmp_path):
    events = [marker("onset", NIGHT_START + 1000),
              marker("wake", NIGHT_START + 2000),
              marker("onset", NIGHT_START + 4000),
              marker("wake", NIGHT_START + 8000)]
    html_text = generate_report(seg_night(), events,
                                tmp_path / "r.html").read_text()
    # three awake regions: before first onset, the gap, after last wake
    assert html_text.count('class="settle"') == 3
