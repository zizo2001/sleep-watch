# sleep-watch

Camera-based sleep movement monitor for one bed. A webcam watches you sleep;
an adaptive frame-differencing pipeline detects every visible movement, saves a
short pre-rolled clip per movement, logs each event to SQLite, builds an
all-night timelapse film, and writes a static HTML morning report with a
timeline, stats, and clickable clips. Optional speech detection catches talking
in your sleep. Optional pose estimation tells sun-on-the-wall from you.

## Privacy

- **Runs fully local.** The only process that sees the camera is the one you
  start. The web dashboard binds `127.0.0.1` and there is no flag to change
  that. No telemetry, no cloud, no account.
- **No full-night recording.** Video is held in a RAM ring buffer and written
  to disk only around a detected movement or speech event (default 10 s before,
  10 s after). Everything else is dropped on the floor. The morning film is a
  timelapse at one frame per 2.5 s.
- **Everything the tool writes goes under `data/`** (clips, films, reports, the
  event database, the learned room map). Clips and exports are deleted after
  `retention_days` (7). `data/` is gitignored, and so is every clip, film and
  database extension at any depth. Keep it that way.
- Audio is the same: the microphone pre-roll lives in RAM and only event
  windows are muxed into clips.

The repository contains code, tests with synthetic fixtures, and nothing else.

## Requirements

- macOS or Linux, Python 3.11 to 3.14. The speech and pose extras depend on
  onnxruntime and mediapipe wheels, which lag new Python releases; on a newer
  interpreter install plain `-e .` (video only) until they catch up.
- A UVC webcam (developed on an OBSBOT Meet SE; any OpenCV-readable camera works)
- `ffmpeg` on PATH for "Export film" (`brew install ffmpeg`)
- Optional: a dim bedside light. The camera cannot see in pitch black.

## Install

```bash
git clone https://github.com/zizo2001/sleep-watch.git
cd sleep-watch
python3 -m venv .venv
.venv/bin/pip install -e ".[audio,pose]"     # or plain `-e .` for video only
```

Model files are downloaded once, into `models/` (gitignored):

```bash
# Silero VAD, for speech events (~2 MB)
curl -sL --create-dirs -o models/silero_vad.onnx \
  "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# MediaPipe pose landmarker, for the skeleton overlay and ambient attribution (~5.5 MB)
curl -sL --create-dirs -o models/pose_landmarker_lite.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
```

Both are optional. A missing model or library degrades to "feature off" with one
warning; video detection never depends on them.

**macOS permissions:** the app you launch from (Terminal, iTerm, …) needs Camera
and, for speech, Microphone access. The first run pops the dialogs.

## Use

```bash
.venv/bin/sleep-watch dashboard          # web UI at http://127.0.0.1:5566
.venv/bin/sleep-watch test               # live preview: score, threshold, state. Saves nothing.
.venv/bin/sleep-watch start              # headless night until Ctrl+C, then auto-report
.venv/bin/sleep-watch report --open      # rebuild the latest report and open it
.venv/bin/sleep-watch report --date YYYY-MM-DD
```

`--camera N` picks another camera index, `--config path` another config file,
`dashboard --port N` another port.

### Night checklist

1. Close anything else that uses the camera. UVC cameras are not reliably
   shareable.
2. Leave a dim light on, pointed at a wall.
3. If exposure hunts in `sleep-watch test`, lock it in your camera's own app;
   OpenCV's exposure lock is best-effort on macOS.
4. Start the night (dashboard button or `sleep-watch start`). The display can
   sleep; the machine will not (`caffeinate` is spawned automatically).
5. Morning: stop the night. The report path prints.

## What you get

**Dashboard.** Live view with optional skeleton overlay, Start/Stop night,
event cards with clip length and movement magnitude, an inline theater player
(←/→ between clips, Esc closes, "play all"), clip ratings, two-tap clip delete,
"Export film" (ffmpeg concat of the night's clips), and a Trends panel that
unlocks at two nights: sleep latency, movements per hour, a movement-by-hour
heat strip, and a restlessness score.

**Morning report.** One static HTML file per night: timeline with sleep
segments, awake gaps and restless spans shaded, stats tiles (slept, longest
still stretch, movements, restlessness), the embedded morning film with a
clickable per-minute spike strip, and one row per clip with magnitude badges.
Opens from disk or from the dashboard.

**Morning film.** All-night timelapse with the clock burned in. At the default
rate, one second of film is one real minute: an 8-hour night is an 8-minute
film (~56 MB). Films are never auto-deleted; they are the artifact worth
keeping.

## How detection works

- **Motion score.** Each frame is downscaled, blurred and diffed against a fast
  running-average background. Score = fraction of pixels that changed. A change
  covering more than 60 % of the frame is a lighting event (lamp toggled),
  logged separately and never recorded as a clip.
- **Adaptive threshold.** Rolling median + k × MAD of quiet-period scores.
  Calibrates in the first 30 s and keeps adapting all night, so sensitivity
  tunes itself to your light and sensor noise.
- **Events.** Score above threshold for 0.5 s opens an event; 3 s below closes
  it. Movements closer together than the post-roll merge into one clip and one
  report row.
- **Speech.** Silero VAD scores the mic ~31× per second. Talking opens a
  `speech` event even without movement; steady noise like a fan scores near
  zero and never triggers.
- **Sleep and wake.** Ten still minutes after settling = asleep, backdated to
  when you stopped moving. Waking is sleep onset inverted: five distinct stirs
  inside a rolling ten-minute window, or one stir lasting three minutes, with
  at least one piece of evidence (a long or full-body movement, or a light
  coming on). Bursts without evidence are logged as `restless`, not `wake`.
  Being out of bed for 45 minutes after a departure movement is a wake too.
  Waking never ends the night: you can fall asleep again and a new sleep
  segment opens. A wake that is followed by re-sleep within 20 minutes with
  nothing big in between is demoted to `restless` after the fact.
- **Ambient motion.** Sunlight crawling across a wall moves pixels but not
  you. Motion that never overlaps the pose skeleton's (padded) bounding box is
  classified `ambient`: gray, excluded from stats, clip kept as evidence. A
  persistent nuisance map (`data/nuisance_map.json`, 16×9 grid, nightly decay)
  halves the motion score in regions that repeatedly move with nobody in them,
  so the system learns your room and un-learns it as the seasons move the sun.
- **Magnitude.** Each movement stores its coverage (motion area ÷ person area)
  and a class: twitch, partial or full-body.
- **Signal minutes.** Per-minute peak motion score, threshold and peak speech
  probability are stored for every night. The report draws them as a heat strip
  under the timeline; this is the tuning record for movements the detector did
  not fire on.

## Data layout

```
data/
  sleepwatch.db          # SQLite (WAL): nights, events, signal minutes. Kept forever.
  clips/YYYY-MM-DD/      # per-event clips, auto-deleted after retention_days
  timelapse/YYYY-MM-DD.mp4
  reports/YYYY-MM-DD.html
  exports/               # ffmpeg concats, auto-deleted after retention_days
  nuisance_map.json      # learned ambient-motion map
  settings.json          # dashboard preferences
```

## Config

Copy `config.example.toml` to `config.toml`. Any field of `Config`
(`sleepwatch/config.py`) can be set in the `[sleepwatch]` table; the example
file lists every key with its default. The ones people actually touch:
`camera_index`, `k` (threshold stiffness, lower = more sensitive),
`preroll_seconds` / `postroll_seconds`, `retention_days`, `audio_device`,
`data_dir`.

## Development

```bash
.venv/bin/pip install -e ".[audio,pose,dev]"
.venv/bin/pytest -q                              # no camera, no models needed
.venv/bin/python scripts/fake_dashboard.py       # dashboard on :5567 with a black fake camera
.venv/bin/python scripts/replay_wake_gate.py YYYY-MM-DD   # re-run the wake gate on a stored night
```

See `CONTRIBUTING.md`. The one hard rule: no real footage, reports or databases
in issues or pull requests.

## License

MIT. See `LICENSE`.
