"""
Project-wide paths and constants.

Keeping these in one place means every module references the same
directories and the same column definitions. If you move the project,
you change paths here and nowhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project version
# ---------------------------------------------------------------------------
# Single source of truth for the app's version string. Surfaced in the report
# footer, the web app, and the FastAPI docs — change it here and it updates
# everywhere. "-dev" marks work-in-progress between tagged releases (V1 shipped
# as git tag v1.0; V2 is in active development).
APP_VERSION = "2.0.0-dev"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT resolves to the repo root regardless of where a script is run
# from. config.py lives in src/, so the root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # untouched downloads / pulls
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, analysis-ready outputs

# Create the data directories if they don't exist yet, so a fresh clone
# of the repo doesn't error out on first run.
for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Statcast / Savant schema helpers
# ---------------------------------------------------------------------------
# Statcast pitch-level data is wide (90+ columns). We don't need to list all
# of them, but it helps to know which ones MUST be numeric and which is the
# date column, so the loader can coerce types reliably.

STATCAST_DATE_COL = "game_date"

# Columns that should always be numeric. pandas usually infers these, but a
# single stray string (e.g. an empty pull, a partial row) can flip a whole
# column to "object" dtype. Coercing explicitly avoids silent bugs later.
STATCAST_NUMERIC_COLS = [
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0", "vy0", "vz0",
    "ax", "ay", "az",
    "release_spin_rate",
    "release_extension",
    "launch_speed",
    "launch_angle",
    # Savant's 1–6 batted-ball classification (6 == Barrel). Categorical, but
    # coercing to numeric lets us compare `== 6` reliably across pulls.
    "launch_speed_angle",
    "hit_distance_sc",
    "effective_speed",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "babip_value",
    "iso_value",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    # Bat-tracking columns (added by Savant in 2024). Present in recent pulls,
    # absent in older ones — the loader only coerces columns that exist.
    "bat_speed",
    "swing_length",
]
