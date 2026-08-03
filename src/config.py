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
    # Swing-path / intercept columns (Savant, bat-tracking era: 2023 H2 onward).
    # Same story as bat_speed — only present in recent pulls, so downstream code
    # must never assume they exist. attack_angle/direction describe the swing
    # plane; the intercept_* columns are the point of contact relative to the
    # batter's body, in inches (the biomechanics signal we care about here).
    "attack_angle",
    "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]

# ---------------------------------------------------------------------------
# Savant leaderboard CSV endpoints (swing geometry)
# ---------------------------------------------------------------------------
# These two leaderboards are NOT wrapped by pybaseball, so we fetch the CSV
# export directly. Both accept a season and return one row per qualified hitter.
# `{season}` is filled in by the loader; `minSwings=q` restricts to Savant's
# own qualified-hitter set (matching how the site displays them).
#
# swing-path: attack angle/direction, swing tilt, ideal-attack-angle rate, and
#   the season-average intercept + batter-box position — the league baseline we
#   rank an individual hitter against.
# bat-tracking: average bat speed, swing length, squared-up / blast rates — the
#   "how hard/efficient is the swing" companion metrics.
SWING_PATH_LEADERBOARD_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking/swing-path-attack-angle"
    "?seasonStart={season}&seasonEnd={season}&type=batter&minSwings=q&csv=true"
)
BAT_TRACKING_LEADERBOARD_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    "?seasonStart={season}&seasonEnd={season}&type=batter"
    "&minSwings=q&minGroupSwings=1&gameType=Regular&csv=true"
)

# Batting-stance visual CSV — the "feet" geometry: avg_foot_sep (distance
# between the feet = stance width) and avg_stance_angle (how open/closed the
# feet are to the pitcher), plus box depth / distance off the plate.
# IMPORTANT: this endpoint is SEASON-AGNOSTIC — it ignores year/season params
# and always returns one fixed stance snapshot per hitter (a stance is stable
# year to year, so Savant serves a single profile). It also covers fewer
# hitters than the season leaderboards, so not every hitter has a row. There is
# no {season} to format; the loader takes one anyway for a consistent API.
BATTING_STANCE_URL = (
    "https://baseballsavant.mlb.com/visuals/batting-stance?csv=true"
)

# The earliest season with bat-tracking / swing-path data. Statcast began
# tracking the bat in the second half of 2023, so a full-season leaderboard
# only exists from 2024 on. Callers use this to fail gracefully (show a
# "no swing-tracking data" note) rather than pulling an empty/partial table.
SWING_TRACKING_FIRST_SEASON = 2024
