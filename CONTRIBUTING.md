# Contributing

Thanks for looking. This is a small personal tool that became useful; issues and
pull requests are welcome.

## The one hard rule

**Never attach real footage, night reports, timelapse films, event databases or
`nuisance_map.json` to an issue or pull request.** They show someone's bedroom.
Describe the problem, or reproduce it with the synthetic fixtures in `tests/`
(`build_scripted` and `build_attributed` in `tests/test_monitor.py` script a whole
fake night from a list of `("m", seconds)` / `("q", seconds)` tuples).

The `.gitignore` blocks `data/`, `*.mp4`, `*.db` and friends at every depth. Keep
it that way; do not add exceptions.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[audio,pose,dev]"
.venv/bin/pytest -q
```

The suite needs no camera and no model files. The dashboard can be exercised
without a camera and without the real database via
`.venv/bin/python scripts/fake_dashboard.py` (black fake camera on port 5567).

Before opening a pull request:

- `pytest -q` green.
- New behaviour comes with a scripted-night test, not a screenshot.
- No new network calls. The tool binds `127.0.0.1` and talks to nothing else;
  that is the privacy promise in the README and it is not negotiable.

## Style

Python 3.11+, type hints, `from __future__ import annotations`, no new
dependencies without a reason in the PR description.
