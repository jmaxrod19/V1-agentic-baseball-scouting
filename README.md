# baseball-scouting

A reproducible pipeline for baseball scouting + analytics. Turns raw data
(Statcast, Baseball Savant, FanGraphs) into clean DataFrames and,
eventually, formatted scouting reports.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Quickstart

```python
from src import loaders

# Pull live from pybaseball and cache the raw copy in data/raw/
df = loaders.pull_statcast("2024-06-14", "2024-06-15", save_as="sc.csv")

# Or load CSVs you already have
sc = loaders.load_statcast_csv("data/raw/sc.csv")          # pybaseball pull
sv = loaders.load_savant_csv("data/raw/savant_export.csv") # Savant search export
fg = loaders.load_fangraphs_csv("data/raw/fg_batting.csv")    # FanGraphs leaderboard export
```

Each loader returns a clean, typed DataFrame. Source-specific quirks (B-Ref's
repeated headers, name markers, multi-team rows, etc.) are handled inside the
loaders — see `src/loaders.py`.

## Layout

- `src/config.py` — paths and column constants (single source of truth)
- `src/loaders.py` — the three data loaders
- `data/raw/` — immutable downloads (gitignored)
- `data/processed/` — cleaned outputs
- `notebooks/` — exploration

See `CLAUDE.md` for project conventions.
