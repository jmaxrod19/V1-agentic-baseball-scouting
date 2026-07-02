# Automated Baseball Scouting Reports

A reproducible Python pipeline that turns raw MLB **Statcast** pitch data into
polished, front-office-style scouting reports. Type a pitcher's name, get a
self-contained HTML report — metrics, tables, an auto-written summary, and five
charts — in one file.

**🔗 [Live sample report](https://jmaxrod19.github.io/V1-agentic-baseball-scouting/)**

![Sample scouting report](docs/assets/report-screenshot.png)

## What it does

- Pulls pitch-level Statcast data for any pitcher and date range (via `pybaseball`).
- Computes an **arsenal summary** — usage, velocity, spin, movement (in inches,
  arm-side-positive convention), **whiff rate** and **chase rate** per pitch —
  plus plate-location and left/right batter splits.
- Renders a self-contained, offline-viewable HTML report: summary cards, an
  auto-generated narrative, three data tables, and five charts (pitch usage,
  velocity, movement plot, location heatmaps, handedness splits).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Generate a report

```bash
# Styled HTML report (with charts) -> data/processed/
python scout.py "Chase Burns" --start 2026-06-01 --end 2026-06-28 --html

# Plain-text report to the terminal
python scout.py "Chase Burns" --start 2026-06-01 --end 2026-06-28
```

Dates default to the current month. Raw pulls are cached in `data/raw/`, so
re-running the same request skips the network.

## Use the pieces directly

```python
from src import loaders, metrics

df = loaders.load_statcast_csv("data/raw/chase_burns.csv")
arsenal = metrics.pitcher_arsenal(df, pitcher=695505)   # per-pitch-type summary
splits  = metrics.pitcher_handedness_splits(df, pitcher=695505)
```

Each loader returns a clean, typed DataFrame; source-specific quirks are handled
inside `src/loaders.py`.

## Layout

- `scout.py` — the CLI entry point (name in → report out)
- `src/config.py` — paths and column constants (single source of truth)
- `src/loaders.py` — data loaders (Statcast, Savant, FanGraphs)
- `src/metrics.py` — arsenal, location, and handedness-split metrics
- `src/report.py` / `src/html_report.py` — text and HTML report formatters
- `src/charts.py` — the five report charts (matplotlib → embedded PNGs)
- `docs/` — the GitHub Pages site and sample report
- `data/raw/` — immutable downloads (gitignored)
- `data/processed/` — generated reports and cleaned outputs

See `CLAUDE.md` for project conventions.
