# CLAUDE.md

This file is read by Claude Code at the start of every session. It tells the
agent how this repo is organized and how I work. Keep it short and current —
if a convention changes, change it here.

## What this project is

A baseball scouting + analytics toolkit. The goal is a reproducible pipeline
that turns raw data (Statcast, Baseball Savant, FanGraphs) into professional
scouting reports that match internal front-office formatting.
This is a portfolio project: code quality and reproducibility matter as much
as the output.

Not building a web app right now. CLI / scripts / notebooks only until there's
a concrete reason for a frontend.

## How I want you to work

- **Explain code line by line.** When you write or change code, walk me
  through what each non-obvious line does and why. I'm using this project to
  learn the engineering side, not just to get working code. Don't hand me a
  black box.
- **Read before you write.** When fixing a bug, read the relevant module and
  run the code to see the real error before proposing a change.
- **Prefer clarity over cleverness.** Readable pandas beats a dense one-liner.
- **Flag tradeoffs.** If there's a meaningful choice (performance, accuracy,
  maintainability), tell me the options rather than silently picking one.
- **Don't add dependencies casually.** If a new package is worth it, say why.

## Repo layout

```
baseball-scouting/
  src/
    config.py     # paths + column-type constants. Single source of truth.
    loaders.py    # CSV loaders for the three data sources (see below).
  data/
    raw/          # immutable downloads / pulls. Never edit by hand. Gitignored.
    processed/    # cleaned, analysis-ready outputs.
  notebooks/      # exploration. Promote stable logic into src/ when it settles.
```

## Data sources and how to load them

All loaders live in `src/loaders.py` and return clean pandas DataFrames.

| Source | Loader | Notes |
|--------|--------|-------|
| pybaseball Statcast pull | `load_statcast_csv(path)` | Wide pitch-level schema. Use `pull_statcast(start, end, save_as=...)` to fetch + cache a raw copy. |
| Baseball Savant search export | `load_savant_csv(path)` | Same schema as Statcast; shares the cleaning core. |
| FanGraphs leaderboard export | `load_fangraphs_csv(path)` | Season-aggregated stats. Loader drops the leading rank column, strips `%`/`$`/commas, and coerces value columns to numeric. Percent columns keep their displayed value (25.4) by default — pass `as_proportion=True` for 0.254. `pull_fangraphs(season, kind)` pulls a leaderboard via pybaseball. |

**Convention:** raw files go in `data/raw/` and are never modified. Cleaning
happens at load time through the loaders, so the same raw file always produces
the same clean frame. If you need cleaned output on disk, write it to
`data/processed/`.

## Conventions

- Python 3.10+. Type hints on function signatures.
- New reusable logic goes in `src/`, not in notebooks.
- Statcast columns: `release_speed`, `pfx_x/pfx_z` (movement), `plate_x/plate_z`
  (location), `launch_speed`/`launch_angle`, plus bat-tracking
  (`bat_speed`, `swing_length`) on 2024+ data. Older pulls won't have the
  bat-tracking columns — code must not assume they exist.
- When in doubt about a Statcast column's meaning, check Savant's CSV docs
  rather than guessing.

## Roadmap (rough)

1. [done] Loaders for the three sources.
2. Metrics module: per-pitcher arsenal summary, per-hitter profile.
3. Report formatter: render a player's metrics into the internal report layout.
4. A single command: player name in -> formatted report out.
