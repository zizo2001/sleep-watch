from __future__ import annotations

import html
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__

MAGNITUDE_LABEL = {"full": "full-body", "partial": "partial",
                   "twitch": "twitch"}
_MAG_RANK = {"full": 2, "partial": 1, "twitch": 0}


@dataclass(frozen=True)
class Stats:
    total: int
    per_hour: float
    longest_still_s: float
    restless_pct: float


def _get(e, name):
    try:
        return e[name]          # dict and sqlite3.Row
    except (TypeError, KeyError, IndexError):
        return getattr(e, name)


def _get_opt(e, name):
    """Like _get but None when the field simply doesn't exist (old rows)."""
    try:
        return _get(e, name)
    except (AttributeError, KeyError, IndexError):
        return None


def compute_segment_stats(segments, movement_events) -> Stats:
    """Stats over actual asleep time — awake gaps are not 'still'."""
    duration = max(sum(e - s for s, e in segments), 1e-9)
    events = sorted(movement_events, key=lambda e: _get(e, "start_ts"))
    total = len(events)
    moving = sum(_get(e, "duration_s") or 0.0 for e in events)
    stills: list[float] = []
    for seg_start, seg_end in segments:
        edges = [seg_start]
        for e in events:
            s = _get(e, "start_ts")
            if seg_start <= s <= seg_end:
                edges.extend([s, min(_get(e, "end_ts") or s, seg_end)])
        edges.append(seg_end)
        stills += [edges[i + 1] - edges[i]
                   for i in range(0, len(edges) - 1, 2)]
    return Stats(
        total=total,
        per_hour=total / (duration / 3600) if total else 0.0,
        longest_still_s=max(stills) if stills else duration,
        restless_pct=moving / duration * 100,
    )


def compute_stats(night_start: float, night_end: float, movement_events) -> Stats:
    return compute_segment_stats([(night_start, night_end)], movement_events)


def sleep_segments(night, events) -> list[tuple[float, float]]:
    """Sleep spans from onset/wake markers, oldest first. Pre-v2.5 nights
    have no onset markers — synthesize one segment from the column."""
    night_start = _get(night, "start_ts")
    night_end = _get(night, "end_ts") or max(
        [_get(e, "end_ts") or night_start for e in events] + [night_start])
    onsets = sorted(_get(e, "start_ts") for e in events
                    if _get_opt(e, "kind") == "onset")
    wakes = sorted(_get(e, "start_ts") for e in events
                   if _get_opt(e, "kind") == "wake")
    if not onsets:
        col = _get_opt(night, "sleep_onset_ts")
        if not col:
            return []
        end = next((w for w in wakes if w >= col), night_end)
        return [(col, end)]
    segs = []
    for i, onset in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else None
        wake = next((w for w in wakes
                     if w >= onset and (nxt is None or w < nxt)), None)
        segs.append((onset, wake if wake is not None
                     else (nxt if nxt is not None else night_end)))
    return segs


def _fmt_clock(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def _fmt_dur(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    h, m = total // 3600, (total % 3600) // 60
    return f"{h}h {m:02d}m" if h else f"{m}m {total % 60:02d}s"


def restlessness_score(minutes, segments) -> float | None:
    """One comparable number per night: mean per-minute motion vs threshold
    (capped at 1.5×) over asleep minutes only, ×100. ~10 = serene,
    ~50 = churning."""
    if not minutes:
        return None
    vals = []
    for m in minutes:
        ts = _get(m, "minute_ts")
        if segments and not any(s <= ts <= e for s, e in segments):
            continue
        thr = _get(m, "threshold") or 1e-9
        vals.append(min((_get(m, "max_motion") or 0.0) / thr, 1.5))
    if not vals:
        return None
    return 100 * sum(vals) / len(vals)


def signal_strip_svg(night_start: float, night_end: float, minutes,
                     width: int = 900, height: int = 78,
                     events=None) -> str:
    """Per-minute peak motion as a heat strip. The point is the sub-event
    texture: minutes at ≥50% of threshold that never fired an event are the
    moves visible in the morning film but missing from the timeline.
    With `events`, onset/wake ticks and restless bands overlay the strip —
    the film scrubber's map."""
    span = max(night_end - night_start, 1e-9)
    track_y, track_h = 2, 62

    def x(ts: float) -> float:
        return (ts - night_start) / span * width

    parts = [
        f'<svg class="sigstrip" viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Per-minute motion intensity">',
        f'<rect x="0" y="{track_y}" width="{width}" height="{track_h}" '
        f'rx="4" class="track"/>',
    ]
    for e in (events or []):
        if _get_opt(e, "kind") == "restless":
            s = _get(e, "start_ts")
            en = _get(e, "end_ts") or s
            parts.append(
                f'<rect class="sig-restless" x="{x(s):.1f}" y="{track_y}" '
                f'width="{max(x(en) - x(s), 2.0):.1f}" height="{track_h}">'
                f'<title>{_fmt_clock(s)}–{_fmt_clock(en)} · restless '
                f'(asleep, stirring)</title></rect>')
    # motion runs 0 → 50×+ threshold on a normal night, so a linear scale
    # clamped at 1× drew every active minute as the same full-height bar.
    # Log against the night's own peak keeps the shape readable.
    ratios = [(_get(m, "max_motion") or 0.0) / (_get(m, "threshold") or 1e-9)
              for m in minutes]
    peak = max(ratios, default=1.0) or 1.0
    big = max(5.0, peak / 4) if peak >= 10 else float("inf")
    denom = math.log(1 + peak)
    for m in minutes:
        ts = _get(m, "minute_ts")
        motion = _get(m, "max_motion") or 0.0
        thr = _get(m, "threshold") or 1e-9
        speech = _get_opt(m, "max_speech")
        ratio = motion / thr
        css = ("sig hot" if ratio >= big
               else "sig near" if ratio >= 1.0 else "sig")
        h = track_h * max(0.05, math.log(1 + ratio) / denom if ratio else 0)
        tip = (f"{_fmt_clock(ts)} · motion {motion:.3f} / thr {thr:.3f}"
               + (f" · speech {speech:.2f}" if speech is not None else ""))
        parts.append(
            f'<rect class="{css}" x="{x(ts):.1f}" '
            f'y="{track_y + track_h - h:.1f}" '
            f'width="{max(x(ts + 60.0) - x(ts), 1.0):.1f}" height="{h:.1f}">'
            f"<title>{tip}</title></rect>")
    for e in (events or []):
        kind = _get_opt(e, "kind")
        if kind not in ("onset", "wake"):
            continue
        s = _get(e, "start_ts")
        css = "wakeline" if kind == "wake" else "onsetline"
        label = "woke up" if kind == "wake" else "fell asleep"
        parts.append(
            f'<line class="{css}" x1="{x(s):.1f}" y1="0" '
            f'x2="{x(s):.1f}" y2="{track_y + track_h + 4}">'
            f"<title>{_fmt_clock(s)} · {label}</title></line>")
    # hour ticks — a shape with no scale is just decoration
    first_hour = math.ceil(night_start / 3600) * 3600
    ts = first_hour
    while ts < night_end:
        px = x(ts)
        if width * 0.02 < px < width * 0.98:
            parts.append(
                f'<text x="{px:.1f}" y="{height - 2}" class="tick" '
                f'text-anchor="middle">{_fmt_clock(ts)}</text>')
        ts += 3600
    parts.append("</svg>")
    return "".join(parts)


def timeline_svg(night_start: float, night_end: float, events,
                 width: int = 900, height: int = 150,
                 onset: float | None = None,
                 segments: list[tuple[float, float]] | None = None) -> str:
    span = max(night_end - night_start, 1e-9)
    track_y, track_h = 34, 72

    def x(ts: float) -> float:
        return (ts - night_start) / span * width

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Movement timeline">',
        f'<rect x="0" y="{track_y}" width="{width}" height="{track_h}" '
        f'rx="4" class="track"/>',
    ]
    if segments:
        # shade everything outside sleep segments as awake
        awake_spans, cursor = [], night_start
        for seg_start, seg_end in segments:
            if seg_start > cursor:
                awake_spans.append((cursor, seg_start))
            cursor = max(cursor, seg_end)
        if cursor < night_end:
            awake_spans.append((cursor, night_end))
        for a, b in awake_spans:
            parts.append(
                f'<rect x="{x(a):.1f}" y="{track_y}" '
                f'width="{x(b) - x(a):.1f}" height="{track_h}" rx="4" '
                f'class="settle"><title>awake</title></rect>')
    elif onset and onset > night_start:
        parts.append(
            f'<rect x="0" y="{track_y}" width="{x(onset):.1f}" '
            f'height="{track_h}" rx="4" class="settle">'
            f"<title>awake — falling asleep</title></rect>")
    hours = int(span // 3600)
    for i in range(hours + 1):
        ts = night_start + i * 3600
        if ts > night_end:
            break
        parts.append(
            f'<line x1="{x(ts):.1f}" y1="{track_y}" x2="{x(ts):.1f}" '
            f'y2="{track_y + track_h}" class="grid"/>'
            f'<text x="{x(ts):.1f}" y="{track_y + track_h + 24}" '
            f'class="tick" text-anchor="middle">{_fmt_clock(ts)}</text>')
    for e in events:
        s = _get(e, "start_ts")
        en = _get(e, "end_ts") or s
        kind = _get_opt(e, "kind")
        if kind == "restless":
            parts.append(
                f'<rect class="ev-restless" x="{x(s):.1f}" y="{track_y}" '
                f'width="{max(x(en) - x(s), 2.5):.1f}" height="{track_h}">'
                f'<title>{_fmt_clock(s)}–{_fmt_clock(en)} · restless '
                f'(asleep, stirring)</title></rect>')
            continue
        if kind == "wake":
            parts.append(
                f'<line class="wakeline" x1="{x(s):.1f}" y1="{track_y - 8}" '
                f'x2="{x(s):.1f}" y2="{track_y + track_h + 8}">'
                f"<title>{_fmt_clock(s)} · woke up</title></line>")
            continue
        w = max(x(en) - x(s), 2.5)
        peak = _get(e, "peak_score") or 0.0
        h = track_h * min(1.0, 0.35 + peak * 1.3)
        css = ("ev-trunc" if kind == "truncated"
               else "ev-speech" if kind == "speech"
               else "ev-ambient" if kind == "ambient"
               else "ev-awake" if kind == "awake"
               else "ev" if kind in (None, "movement")
               else "ev-settling")
        parts.append(
            f'<rect class="{css}" rx="2" x="{x(s):.1f}" '
            f'y="{track_y + track_h - h:.1f}" '
            f'width="{w:.1f}" height="{h:.1f}">'
            f'<title>{_fmt_clock(s)} · {_fmt_dur(_get(e, "duration_s") or 0)}'
            f' · peak {peak:.3f}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


# Palette: dataviz reference instance (validated both modes).
CSS = """
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --ev: #2a78d6;
  --border: rgba(11,11,11,0.10);
  --violet: #6d28d9; --teal: #0d9488;
  --movement: #e34948; --calib: #c98500;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ev: #3987e5;
    --border: rgba(255,255,255,0.10);
    --violet: #8055f0; --teal: #0fa593;
    --movement: #e66767; --calib: #eda100;
  }
}
body { background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 960px; margin: 2.5rem auto; padding: 0 1.25rem; }
h1 { font-size: 1.35rem; margin-bottom: .2rem; }
.sub { color: var(--ink-2); margin-top: 0; }
.stats { display: flex; gap: .8rem; flex-wrap: wrap; margin: 1.4rem 0; }
.stat { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: .75rem 1.1rem; min-width: 8.5rem;
  border-top: 3px solid var(--grid); }
.stat:nth-child(1) { border-top-color: var(--calib); }
.stat:nth-child(2) { border-top-color: var(--ev); }
.stat:nth-child(3) { border-top-color: var(--teal); }
.stat:nth-child(4) { border-top-color: var(--violet); }
.stat:nth-child(5) { border-top-color: var(--movement); }
.stat:nth-child(6) { border-top-color: var(--violet); }
.stat b { display: block; font-size: 1.6rem; font-weight: 650;
  font-variant-numeric: tabular-nums; }
.stat span { color: var(--ink-2); font-size: .85rem; }
tr:hover td { background: color-mix(in srgb, var(--ev) 6%, transparent); }
td a { display: inline-block; padding: 4px 10px;
  border: 1px solid var(--border); border-radius: 8px;
  text-decoration: none; font-weight: 600; font-size: .85rem;
  transition: background 0.2s ease; }
td a:hover { background: color-mix(in srgb, var(--ev) 10%, transparent); }
.rtheater { display: grid; grid-template-rows: 0fr;
  transition: grid-template-rows .4s cubic-bezier(.16,1,.3,1); }
.rtheater.open { grid-template-rows: 1fr; margin: 1.1rem 0 .4rem; }
.rtheater .tinner { overflow: hidden; min-height: 0; display: grid; gap: 8px; }
.rtheater .tstage { position: relative; background: #000;
  border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
.rtheater video { display: block; width: 100%; aspect-ratio: 16 / 9;
  max-height: 60vh; object-fit: contain; background: #000; }
.tnav { position: absolute; top: 50%; transform: translateY(-50%);
  width: 44px; height: 44px; border-radius: 50%; cursor: pointer;
  border: 1px solid rgba(255,255,255,.25); background: rgba(0,0,0,.45);
  color: #fff; font-size: 1.25rem; line-height: 1;
  transition: background .2s ease, opacity .2s ease; }
.tnav.prev { left: 10px; } .tnav.next { right: 10px; }
.tnav:hover { background: rgba(0,0,0,.7); }
.tnav:disabled { opacity: .25; cursor: default; }
.rtheater .tbar { display: flex; gap: 12px; align-items: center;
  flex-wrap: wrap; }
.rtheater .tbar b { font-size: .95rem; }
.tmono { color: var(--muted); font-size: .85rem;
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, monospace; }
#rt-count { margin-left: auto; }
#rt-close, #rt-del { cursor: pointer; border: 1px solid var(--border);
  background: none; color: var(--ink-2); border-radius: 8px;
  padding: 8px 12px; font: 600 .85rem system-ui, sans-serif;
  transition: background .2s ease, color .2s ease; min-height: 40px;
  display: inline-flex; align-items: center; gap: 6px; }
#rt-close:hover { background: color-mix(in srgb, var(--ev) 10%, transparent); }
#rt-del:hover { color: var(--movement); }
#rt-del.armed { color: var(--movement); border-color: var(--movement);
  background: color-mix(in srgb, var(--movement) 12%, transparent); }
tr.playing td { background: color-mix(in srgb, var(--ev) 12%, transparent); }
.trate { display: inline-flex; gap: 6px; }
.rateb { cursor: pointer; width: 38px; height: 38px; border-radius: 9px;
  border: 1px solid var(--border); background: none; color: var(--muted);
  display: inline-grid; place-items: center;
  transition: background .2s ease, color .2s ease; }
.rateb svg { width: 15px; height: 15px; }
.rateb:hover { color: var(--ink-2); }
.rateb[data-r="1"][aria-pressed="true"] { color: #1baf7a;
  background: rgba(27,175,122,.14); }
.rateb[data-r="-1"][aria-pressed="true"] { color: var(--movement);
  background: color-mix(in srgb, var(--movement) 14%, transparent); }
@media (prefers-reduced-motion: reduce) {
  .rtheater { transition: none; }
}
.chart { background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem 1.1rem .4rem; }
.chart h2 { font-size: .95rem; color: var(--ink-2); margin: 0 0 .5rem; }
.track { fill: var(--page); stroke: var(--baseline); stroke-width: 1; }
.ev { fill: var(--ev); }
.settle { fill: var(--muted); opacity: 0.16; }
.ev-settling { fill: var(--calib); opacity: 0.55; }
.ev-trunc { fill: var(--movement); }
.ev-speech { fill: var(--violet); }
.ev-ambient { fill: var(--muted); opacity: 0.4; }
.ev-awake { fill: var(--teal); opacity: 0.45; }
.wakeline { stroke: var(--teal); stroke-width: 2.5; }
.sig { fill: var(--ev); opacity: 0.35; }
.sig.near { fill: var(--calib); opacity: 0.85; }
.sig.hot { fill: var(--movement); opacity: 0.9; }
.sig-restless { fill: var(--calib); opacity: 0.16; }
.ev-restless { fill: var(--calib); opacity: 0.18; }
.onsetline { stroke: #1baf7a; stroke-width: 2.5; }
.mag { font-size: .72rem; padding: 2px 8px; border-radius: 9px;
  margin-left: 6px; font-weight: 600; white-space: nowrap; }
.mag-full { color: var(--movement);
  background: color-mix(in srgb, var(--movement) 14%, transparent); }
.mag-partial { color: var(--teal);
  background: color-mix(in srgb, var(--teal) 14%, transparent); }
.mag-twitch { color: var(--muted);
  background: color-mix(in srgb, var(--muted) 14%, transparent); }
#film { display: block; width: 100%; border-radius: 10px;
  background: #000; }
.filmstage { position: relative; }
#filmclock { position: absolute; top: 10px; left: 10px;
  background: rgba(0,0,0,.62); color: #fff; border-radius: 7px;
  padding: 4px 10px; pointer-events: none; letter-spacing: .02em;
  font: 600 1rem ui-monospace, "SF Mono", Menlo, monospace;
  font-variant-numeric: tabular-nums; }
#filmclock[hidden] { display: none; }
#filmstrip { cursor: pointer; margin-top: 6px; position: relative; }
#filmnow { position: absolute; top: 0; bottom: 14px; width: 2px;
  background: #fff; pointer-events: none; }
#filmnow[hidden] { display: none; }
.filmmeta { display: flex; gap: 12px; align-items: baseline;
  flex-wrap: wrap; margin: 0 0 .5rem; }
.chip { font-size: .78rem; color: var(--ink-2);
  border: 1px solid var(--border); border-radius: 999px;
  padding: 2px 10px; font-variant-numeric: tabular-nums; }
tr.muted td { color: var(--muted); }
.grid { stroke: var(--grid); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 12px;
  font-variant-numeric: tabular-nums; }
table { border-collapse: collapse; width: 100%; margin-top: 1.4rem; }
td, th { text-align: left; padding: .5rem .6rem;
  border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums; }
th { color: var(--ink-2); font-weight: 600; font-size: .85rem; }
.muted { color: var(--muted); }
a { color: var(--ev); }
"""


THEATER_HTML = """
<div class="rtheater" id="rtheater">
  <div class="tinner">
    <div class="tstage">
      <video controls playsinline preload="auto"></video>
      <button type="button" class="tnav prev" aria-label="Previous clip">&#8249;</button>
      <button type="button" class="tnav next" aria-label="Next clip">&#8250;</button>
    </div>
    <div class="tbar">
      <b id="rt-title">&mdash;</b>
      <span class="tmono" id="rt-sub">&mdash;</span>
      <span class="trate">
        <button type="button" class="rateb" data-r="1" aria-label="Good clip">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
               aria-hidden="true"><path d="M7 10v12M15 5.9 14 10h5.8a2 2 0 0 1
               1.9 2.6l-2.3 8a2 2 0 0 1-1.9 1.4H4a2 2 0 0 1-2-2v-8a2 2 0 0 1
               2-2h2.8a2 2 0 0 0 1.8-1.1L12 2a3.1 3.1 0 0 1 3 3.9Z"/></svg>
        </button>
        <button type="button" class="rateb" data-r="-1" aria-label="Junk clip">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
               stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
               aria-hidden="true"><path d="M17 14V2M9 18.1 10 14H4.2a2 2 0 0
               1-1.9-2.6l2.3-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0
               1-2 2h-2.8a2 2 0 0 0-1.8 1.1L12 22a3.1 3.1 0 0 1-3-3.9Z"/></svg>
        </button>
      </span>
      <span class="tmono" id="rt-count"></span>
      <button type="button" id="rt-del">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" aria-hidden="true"
             width="14" height="14"><path d="M3 6h18M8 6V4h8v2m1 0-1
             14H8L7 6"/></svg> delete</button>
      <button type="button" id="rt-close">&#10005; close</button>
    </div>
  </div>
</div>
"""

# plain string, not an f-string — JS braces stay literal
FILM_JS = """
(() => {
  const film = document.getElementById("film");
  const strip = document.getElementById("filmstrip");
  if (!film || !strip) return;
  const t0 = Number(film.dataset.start), t1 = Number(film.dataset.end);
  const span = Math.max(t1 - t0, 1);
  const clock = document.getElementById("filmclock");
  const now = document.getElementById("filmnow");
  strip.addEventListener("click", (e) => {
    const r = strip.getBoundingClientRect();
    const frac = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    if (film.duration) {
      film.currentTime = frac * film.duration;
      film.play().catch(() => {});
    }
  });
  // real wall time from playback position — works on films recorded before
  // the clock was burned into the frames
  const tick = () => {
    if (!film.duration) return;
    const frac = Math.min(Math.max(film.currentTime / film.duration, 0), 1);
    const d = new Date((t0 + frac * span) * 1000);
    clock.textContent = d.toLocaleTimeString([],
      { hour: "2-digit", minute: "2-digit" });
    clock.hidden = false;
    now.style.left = (100 * frac).toFixed(2) + "%";
    now.hidden = false;
  };
  film.addEventListener("timeupdate", tick);
  film.addEventListener("seeked", tick);
  film.addEventListener("loadedmetadata", tick);
  tick();
})();
"""

# plain string, not an f-string — JS braces stay literal
THEATER_JS = """
(() => {
  const links = Array.from(document.querySelectorAll("a.watch"));
  if (!links.length) return;
  const theater = document.getElementById("rtheater");
  const video = theater.querySelector("video");
  let i = 0;
  const rowOf = (a) => a.closest("tr");
  function show(n) {
    i = Math.max(0, Math.min(n, links.length - 1));
    video.src = links[i].getAttribute("href");
    video.play().catch(() => {});
    const tds = rowOf(links[i]).querySelectorAll("td");
    document.getElementById("rt-title").textContent = tds[0].textContent;
    document.getElementById("rt-sub").textContent = tds[2].textContent
      ? tds[2].textContent + " video \\u00b7 " + tds[1].textContent + " move"
      : tds[1].textContent + " move";
    document.getElementById("rt-count").textContent =
      (i + 1) + " / " + links.length + " \\u00b7 \\u2190 \\u2192";
    theater.querySelector(".tnav.prev").disabled = i === 0;
    theater.querySelector(".tnav.next").disabled = i === links.length - 1;
    const canRate = location.protocol !== "file:"
      && links[i].dataset.eid;
    theater.querySelectorAll(".rateb").forEach((b) => {
      b.style.display = canRate ? "" : "none";
      b.setAttribute("aria-pressed", String(
        b.dataset.r === (links[i].dataset.rating || "")));
    });
    const delBtn = document.getElementById("rt-del");
    delBtn.style.display = canRate ? "" : "none";
    delBtn.classList.remove("armed");
    document.querySelectorAll("tr.playing")
      .forEach((r) => r.classList.remove("playing"));
    rowOf(links[i]).classList.add("playing");
  }
  theater.querySelectorAll(".rateb").forEach((b) =>
    b.addEventListener("click", () => {
      const a = links[i];
      const next = a.dataset.rating === b.dataset.r ? "" : b.dataset.r;
      a.dataset.rating = next;
      fetch("/api/events/" + a.dataset.eid + "/rate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rating: next ? Number(next) : 0 }),
      }).catch(() => {});
      show(i);
    }));
  function openAt(n) {
    theater.classList.add("open");
    show(n);
    setTimeout(() => theater.scrollIntoView(
      { behavior: "smooth", block: "nearest" }), 380);
  }
  function close() {
    video.pause();
    video.removeAttribute("src");
    video.load();
    theater.classList.remove("open");
    document.querySelectorAll("tr.playing")
      .forEach((r) => r.classList.remove("playing"));
  }
  links.forEach((a, n) => a.addEventListener("click", (e) => {
    e.preventDefault();
    openAt(n);
  }));
  theater.querySelector(".tnav.prev")
    .addEventListener("click", () => show(i - 1));
  theater.querySelector(".tnav.next")
    .addEventListener("click", () => show(i + 1));
  document.getElementById("rt-close").addEventListener("click", close);
  const rtDel = document.getElementById("rt-del");
  rtDel.addEventListener("click", () => {
    const a = links[i];
    if (!a.dataset.eid) return;
    if (!rtDel.classList.contains("armed")) {     // two-step confirm
      rtDel.classList.add("armed");
      return;
    }
    rtDel.disabled = true;
    // a row can stand for several merged events — delete them all, else
    // the clip stays on disk and the row re-forms around the survivor
    const ids = (a.dataset.eids || a.dataset.eid).split(",").filter(Boolean);
    Promise.all(ids.map((id) =>
      fetch("/api/events/" + id, { method: "DELETE" })))
      .then(() => location.reload())
      .catch(() => {})
      .finally(() => {
        rtDel.disabled = false;
        rtDel.classList.remove("armed");
      });
  });
  document.addEventListener("keydown", (e) => {
    if (!theater.classList.contains("open")) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") show(i + 1);
    else if (e.key === "ArrowLeft") show(i - 1);
  });
})();
"""


def generate_report(night, event_rows, out_path: Path,
                    signal_minutes=None) -> Path:
    night_start = _get(night, "start_ts")
    night_end = _get(night, "end_ts") or max(
        [_get(e, "end_ts") or night_start for e in event_rows] + [night_start])
    onset = _get_opt(night, "sleep_onset_ts")
    sleep_start = onset if onset else night_start
    movement = [e for e in event_rows
                if _get(e, "kind") in ("movement", "truncated")]
    settling = [e for e in event_rows if _get(e, "kind") == "settling"]
    speech = [e for e in event_rows if _get(e, "kind") == "speech"]
    wakes = [e for e in event_rows if _get(e, "kind") == "wake"]
    restless = [e for e in event_rows if _get(e, "kind") == "restless"]
    muted_clips = [e for e in event_rows           # gray rows WITH clips
                   if _get(e, "kind") in ("ambient", "awake")]
    segments = sleep_segments(night, event_rows)
    stats = compute_segment_stats(segments or [(sleep_start, night_end)],
                                  movement)
    restless_score = restlessness_score(signal_minutes or [], segments)
    if onset:
        latency_tile = (f'<div class="stat"><b>{_fmt_dur(onset - night_start)}'
                        f'</b><span>to fall asleep</span></div>')
    else:
        latency_tile = ('<div class="stat"><b>—</b>'
                        '<span>never fell asleep</span></div>')
    slept_tile = span_note = ""
    if segments:
        slept_s = sum(e - s for s, e in segments)
        label = ("slept" if len(segments) == 1
                 else f"slept · {len(segments)} stretches")
        slept_tile = (f'<div class="stat"><b>{_fmt_dur(slept_s)}</b>'
                      f'<span>{label}</span></div>')
        span_note = (f" · slept {_fmt_clock(segments[0][0])} →"
                     f" {_fmt_clock(segments[-1][1])}")
    spoke_tile = ""
    if speech:
        talk_s = sum(_get(e, "duration_s") or 0.0 for e in speech)
        spoke_tile = (f'<div class="stat"><b>{len(speech)}×</b>'
                      f'<span>sleep talking · {_fmt_dur(talk_s)}</span></div>')
    restless_tile = ""
    if restless_score is not None:
        restless_tile = (f'<div class="stat"><b>{restless_score:.0f}</b>'
                         f'<span>restlessness (0–150)</span></div>')
    timed_rows = []
    # merged clips share one file (post-roll merging) — one ROW per file,
    # not per event, or the same 42-s clip lists twice (merged-clip bug).
    # Only movements merge: a speech row carries a VAD probability in
    # peak_score and an ambient row is explicitly "not you", so folding
    # either into a movement row would misreport both.
    by_clip: dict[str, list] = {}
    grouped: list[list] = []
    for e in sorted(movement + speech + muted_clips,
                    key=lambda e: _get(e, "start_ts")):
        clip = _get(e, "clip_path")
        mergeable = _get(e, "kind") in ("movement", "truncated")
        if clip and mergeable and clip in by_clip:
            by_clip[clip].append(e)
            continue
        members = [e]
        if clip and mergeable:
            by_clip[clip] = members
        grouped.append(members)
    for members in grouped:
        e = members[0]
        clip = _get(e, "clip_path")
        if clip and Path(clip).exists():
            parts = Path(clip).parts
            if "clips" in parts:
                rel = "/".join(parts[parts.index("clips") + 1:])
                href = html.escape(f"../clips/{rel}")
            else:
                href = html.escape(Path(clip).resolve().as_uri())
            eid = _get_opt(e, "id") or ""
            rating = _get_opt(e, "rating") or ""
            # every event sharing this file — deleting only the first would
            # leave the clip on disk and the row would simply re-form
            eids = ",".join(str(_get_opt(m, "id") or "") for m in members
                            if _get_opt(m, "id"))
            link = (f'<a class="watch" href="{href}" data-eid="{eid}"'
                    f' data-eids="{eids}" data-rating="{rating}">watch</a>')
        else:
            link = '<span class="muted">(clip expired)</span>' if clip else ""
        kinds = {_get(m, "kind") for m in members}
        kind_note = ("" + (" ⚠ truncated" if "truncated" in kinds else "")
                     + (" · speech" if "speech" in kinds else ""))
        if kinds <= {"ambient", "awake"}:
            kind_note += (" · ambient — not you" if kinds == {"ambient"}
                          else " · awake")
            row_class = ' class="muted"'
        else:
            row_class = ""
        move_s = sum(_get(m, "duration_s") or 0.0 for m in members)
        peak = max((_get(m, "peak_score") or 0.0) for m in members)
        mags = [m2 for m2 in (_get_opt(m, "magnitude") for m in members)
                if m2 in _MAG_RANK]
        badge = ""
        if mags:
            top = max(mags, key=_MAG_RANK.get)
            badge = (f' <span class="mag mag-{top}">'
                     f'{MAGNITUDE_LABEL[top]}</span>')
        move_cell = _fmt_dur(move_s) + kind_note + badge
        title = ""
        if len(members) > 1:
            times = ", ".join(
                time.strftime("%H:%M:%S",
                              time.localtime(_get(m, "start_ts")))
                for m in members)
            move_cell += (f' <span class="muted">· {len(members)} moves'
                          f"</span>")
            title = f' title="{times}"'
        clip_dur = _get_opt(e, "clip_duration_s")
        clip_cell = _fmt_dur(clip_dur) if clip_dur else ""
        timed_rows.append((_get(e, "start_ts"),
            f"<tr{row_class}><td>{_fmt_clock(_get(e, 'start_ts'))}</td>"
            f"<td{title}>{move_cell}</td>"
            f"<td>{clip_cell}</td>"
            f"<td>{peak:.3f}</td><td>{link}</td></tr>"))
    for e in settling:
        timed_rows.append((_get(e, "start_ts"),
            f'<tr class="muted"><td>{_fmt_clock(_get(e, "start_ts"))}</td>'
            f"<td>{_fmt_dur(_get(e, 'duration_s') or 0)} · settling"
            f" (awake)</td><td></td>"
            f"<td>{(_get(e, 'peak_score') or 0):.3f}</td><td></td></tr>"))
    rows = [r for _, r in sorted(timed_rows, key=lambda t: t[0])]
    date = _get(night, "date")
    tl_path = Path(out_path).parent.parent / "timelapse" / f"{date}.mp4"
    marker_events = wakes + restless + [
        e for e in event_rows if _get_opt(e, "kind") == "onset"]
    strip = (signal_strip_svg(night_start, night_end, signal_minutes,
                              events=marker_events)
             if signal_minutes else "")
    film_html = ""
    if tl_path.exists():
        every_s = 5.0
        try:
            every_s = float(json.loads(_get(night, "config_json"))
                            .get("timelapse_every_s") or 5.0)
        except (TypeError, ValueError, KeyError):
            pass
        real_s = every_s * 24.0            # film is written at 24 fps
        ratio = (f"1 s of film = {real_s / 60:g} min" if real_s >= 60
                 else f"1 s of film = {real_s:g} s")
        film_html = f'''<div class="chart" style="margin-bottom:.8rem">
<div class="filmmeta"><h2>Morning film — click the strip to jump ·
 green = fell asleep, teal = woke, amber = restless</h2>
<span class="chip">{ratio}</span></div>
<div class="filmstage">
<video id="film" controls preload="metadata"
 data-start="{night_start}" data-end="{night_end}"
 src="../timelapse/{date}.mp4"></video>
<div id="filmclock" hidden>—</div></div>
<div id="filmstrip">{strip}<div id="filmnow" hidden></div></div></div>'''
    gaps = [e for e in event_rows if _get(e, "kind") == "gap"]
    gap_note = (f'<p class="muted">⚠ {len(gaps)} camera gap(s) during the '
                f'night — timeline has blind spots.</p>' if gaps else "")
    settle_note = (f'<p class="muted">{len(settling)} movement(s) while '
                   f'falling asleep — excluded from sleep stats.</p>'
                   if settling else "")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sleep-watch · {_get(night, 'date')}</title><style>{CSS}</style></head><body>
<h1>Night of {_get(night, 'date')}</h1>
<p class="sub">{_fmt_clock(night_start)} → {_fmt_clock(night_end)}
 · {_fmt_dur(night_end - night_start)} monitored{span_note}</p>
<div class="stats">
{latency_tile}
{slept_tile}
<div class="stat"><b>{stats.total}</b><span>movements asleep</span></div>
<div class="stat"><b>{stats.per_hour:.1f}</b><span>per hour</span></div>
<div class="stat"><b>{_fmt_dur(stats.longest_still_s)}</b><span>longest still</span></div>
<div class="stat"><b>{stats.restless_pct:.1f}%</b><span>restless</span></div>
{restless_tile}
{spoke_tile}
</div>
{film_html}
<div class="chart"><h2>Movement timeline — bar height = peak intensity,
 shaded gray = awake, amber = restless</h2>
{timeline_svg(night_start, night_end,
              movement + settling + speech + wakes + restless + muted_clips,
              onset=onset, segments=segments)}</div>
{f'''<div class="chart" style="margin-top:.8rem"><h2>Motion per minute —
 orange = near the trigger line, red = fired</h2>
{strip}</div>''' if signal_minutes and not film_html else ''}
{settle_note}
{gap_note}
{THEATER_HTML}
<table><tr><th>Time</th><th>Movement</th><th>Clip length</th><th>Peak</th><th>Clip</th></tr>
{''.join(rows)}</table>
<p class="muted" style="margin-top:2rem">sleep-watch v{__version__}</p>
<script>{THEATER_JS}</script>
<script>{FILM_JS}</script>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path
