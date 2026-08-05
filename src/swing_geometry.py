"""Swing-geometry card: where a hitter sets up, meets the ball, and swings.

This is the project's first SQL-backed feature. The analysis is written in SQL
and run through DuckDB — an in-process engine that queries pandas DataFrames
directly, so nothing here needs a database server. The loaders still return
pandas (in), and every function here returns pandas / plain dicts (out); SQL is
only the layer in the middle that does the aggregating and ranking.

Two data channels, two jobs (see the design notes in the PR):

  * League leaderboard (one row per qualified hitter) -> rank THIS hitter
    against the league with a window function. `load_swing_path_baseline`
    fetches + caches it; `league_percentiles` runs the ranking query.

  * Pitch-level swings (one row per swing for this hitter) -> break the swing
    down by pitch type with a GROUP BY. `swing_profile_by_pitch` does that, and
    `intercept_points` pulls the per-swing contact points for the scatter plot.

`swing_geometry_card` ties them together into one dict the report/web layer can
render.

A note on percentiles: attack angle, swing tilt, and intercept depth have no
universal "good" direction (a steep swing isn't better than a flat one — it
depends on the hitter). So these percentiles are reported as neutral rank
context ("steeper than 63% of hitters"), NOT oriented so high = good the way
the batted-ball percentiles in percentiles.py are. Bat speed and ideal-attack-
angle rate are the exceptions where higher is genuinely better.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

try:
    from . import config, loaders
except ImportError:  # direct-script / non-package execution
    import config
    import loaders


# ---------------------------------------------------------------------------
# Metric metadata: which league columns we rank, and how to describe them.
# ---------------------------------------------------------------------------
# Keyed by the swing-path leaderboard column name. `label` is display text;
# `higher_is_better` is True/False for the two directional metrics and None for
# the neutral ones (so the display layer knows not to color them good/bad).
_LEAGUE_METRICS: dict[str, dict] = {
    # From the swing-path leaderboard (season-specific).
    "attack_angle":            {"label": "Attack angle",     "higher_is_better": None},
    "swing_tilt":              {"label": "Swing tilt",       "higher_is_better": None},
    "avg_intercept_y_vs_batter": {"label": "Intercept depth", "higher_is_better": None},
    "avg_bat_speed":           {"label": "Bat speed",        "higher_is_better": True},
    "ideal_attack_angle_rate": {"label": "Ideal-angle rate", "higher_is_better": True},
    # From the batting-stance snapshot (the "feet" geometry). Joined in on
    # (id, side); a hitter absent from the stance table gets None here.
    "avg_foot_sep":            {"label": "Stance width",     "higher_is_better": None},
    "avg_stance_angle":        {"label": "Stance angle",     "higher_is_better": None},
}


# ===========================================================================
# League baseline: fetch once, cache to disk (same pattern as percentiles.py)
# ===========================================================================

def load_swing_path_baseline(season: int) -> pd.DataFrame:
    """Return the season's swing-path leaderboard, cached to data/processed/.

    First call for a season pulls from Savant and writes a CSV; later calls read
    the CSV — instant and offline. Only *completed* seasons are cached, because a
    current-season leaderboard still moves daily (caching it would freeze stale
    percentiles until the next restart).
    """
    path = config.PROCESSED_DIR / f"league_swing_path_{season}.csv"
    if path.exists():
        return pd.read_csv(path)

    df = loaders.pull_swing_path_leaderboard(season)
    if not df.empty and season < date.today().year:
        df.to_csv(path, index=False)
    return df


def load_stance_baseline() -> pd.DataFrame:
    """Return the batting-stance snapshot (the feet geometry), cached to disk.

    No season argument: the Savant endpoint is season-agnostic (one fixed stance
    profile per hitter), so there's a single cache file rather than one per year.
    First call fetches + writes it; later calls read the CSV. To force a refresh
    (e.g. once a new season's stances settle), delete the cached file.
    """
    path = config.PROCESSED_DIR / "league_batting_stance.csv"
    if path.exists():
        return pd.read_csv(path)

    df = loaders.pull_batting_stance_leaderboard()
    if not df.empty:
        df.to_csv(path, index=False)
    return df


def player_stance(player_id: int, side: str | None = None) -> dict | None:
    """Return one hitter's raw stance row (for drawing the box/feet), or None.

    The chart needs the geometry itself — box position, foot separation, stance
    angle — not the percentiles. This pulls that straight from the stance
    snapshot. Switch hitters have a row per `side`; pass one to disambiguate,
    otherwise the first is returned.
    """
    stance = load_stance_baseline()
    if stance.empty or "id" not in stance.columns:
        return None
    match = stance[stance["id"] == player_id]
    if side is not None and "side" in match.columns:
        match = match[match["side"] == side]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def player_box_position(player_id: int, season: int, side: str | None = None) -> dict | None:
    """Batter's box position (avg_batter_x/y) from the swing-path leaderboard.

    A fallback for the chart when a hitter is absent from the batting-stance
    snapshot (which carries the feet) but still appears in the season's swing-path
    leaderboard (which also reports batter-box position). Returns a minimal
    stance-like dict with just the box position — enough to place the contact
    cloud over the plate and draw the measurements/bat, without the feet — or
    None if the hitter has no box position either.
    """
    sp = load_swing_path_baseline(season)
    if sp.empty or "id" not in sp.columns:
        return None
    match = sp[sp["id"] == player_id]
    if side is not None and "side" in match.columns:
        by_side = match[match["side"] == side]
        if not by_side.empty:
            match = by_side
    if match.empty:
        return None
    row = match.iloc[0]
    bx, by = row.get("avg_batter_x_position"), row.get("avg_batter_y_position")
    if bx is None or by is None or pd.isna(bx) or pd.isna(by):
        return None
    return {"avg_batter_x_position": float(bx), "avg_batter_y_position": float(by)}


def load_league_heights() -> pd.DataFrame:
    """Return {id, height_in} for every hitter in the stance snapshot, cached.

    Heights come from the MLB Stats API in one batched pull, then are cached to
    disk (season-agnostic, like the stance snapshot). This is the league height
    distribution the biomechanical percentiles rank a hitter against.
    """
    path = config.PROCESSED_DIR / "league_heights.csv"
    if path.exists():
        return pd.read_csv(path)

    stance = load_stance_baseline()
    if stance.empty or "id" not in stance.columns:
        return pd.DataFrame(columns=["id", "height_in"])

    heights = loaders.pull_player_heights(stance["id"].tolist())
    df = pd.DataFrame([{"id": k, "height_in": v} for k, v in heights.items()])
    if not df.empty:
        df.to_csv(path, index=False)
    return df


# Height-ADJUSTED league ranking: divide each stance metric by the hitter's
# height, then CUME_DIST over those ratios. Ranking the ratio (not the raw inch
# value) is what makes it "wide/deep FOR HIS SIZE" instead of just "wide/deep".
_STANCE_BIOMECH_SQL = """
WITH joined AS (
    SELECT
        s.id,
        s.side,
        s.avg_foot_sep,
        s.avg_intercept_y_vs_batter,
        s.avg_batter_y_position,
        h.height_in,
        s.avg_foot_sep              / h.height_in AS width_ratio,
        s.avg_intercept_y_vs_batter / h.height_in AS reach_ratio,
        s.avg_batter_y_position     / h.height_in AS depth_ratio
    FROM league_stance s
    JOIN league_heights h ON s.id = h.id
    WHERE h.height_in IS NOT NULL
      AND s.avg_foot_sep IS NOT NULL
      AND s.avg_intercept_y_vs_batter IS NOT NULL
      AND s.avg_batter_y_position IS NOT NULL
),
ranked AS (
    SELECT *,
        CUME_DIST() OVER (ORDER BY width_ratio) AS width_pct,
        CUME_DIST() OVER (ORDER BY reach_ratio) AS reach_pct,
        CUME_DIST() OVER (ORDER BY depth_ratio) AS depth_pct
    FROM joined
)
SELECT * FROM ranked WHERE id = ?
"""


def stance_biomech(player_id: int, side: str | None = None) -> dict | None:
    """Height-adjusted league percentiles for stance width, depth, and reach.

    Returns {height_in, height_text, stance_width, stance_width_pct,
    stance_depth, stance_depth_pct, reach, reach_pct}, or None if the hitter has
    no stance/height data to rank. Each percentile divides the raw stance metric
    by the hitter's own height before ranking against the league, so it answers
    "how wide/deep is his stance FOR A HITTER HIS SIZE" — a 6'6" and a 5'9"
    hitter are compared fairly instead of just by raw inches.
    """
    import duckdb

    stance = load_stance_baseline()
    heights = load_league_heights()
    if stance.empty or heights.empty:
        return None

    con = duckdb.connect()
    con.register("league_stance", stance)
    con.register("league_heights", heights)
    ranked = con.execute(_STANCE_BIOMECH_SQL, [player_id]).df()
    con.close()

    if side is not None and "side" in ranked.columns:
        ranked = ranked[ranked["side"] == side]
    if ranked.empty:
        return None
    row = ranked.iloc[0]

    # Clamp to 1..99 so the extremes read as "1st"/"99th" rather than a
    # confusing "0th"/"100th" (CUME_DIST hits exactly 0-adjacent and 1.0).
    def _pct(cume: float) -> int:
        return min(99, max(1, round(float(cume) * 100)))

    inches = int(row["height_in"])
    return {
        "height_in": inches,
        "height_text": f"{inches // 12}' {inches % 12}\"",
        "stance_width": float(row["avg_foot_sep"]),
        "stance_width_pct": _pct(row["width_pct"]),
        "stance_depth": float(row["avg_batter_y_position"]),
        "stance_depth_pct": _pct(row["depth_pct"]),
        "reach": float(row["avg_intercept_y_vs_batter"]),
        "reach_pct": _pct(row["reach_pct"]),
    }


# ===========================================================================
# SQL 1 — league percentiles: rank two tables, then JOIN them
# ===========================================================================
# CUME_DIST() OVER (ORDER BY col) is the "cumulative distribution": for each
# row, the fraction of rows with a value <= this one. That's exactly a rank
# percentile (0-1), matching the "share at or below" definition in percentiles.py.
#
# The two data sources have different populations (every swing-path hitter, but
# only some have stance data), so we rank each in its OWN table first — inside
# separate CTEs — then LEFT JOIN the ranked results on (id, side). Ranking
# before the join is what keeps the stance percentile honest: it's computed only
# among hitters who actually have a stance, not diluted by the ones who don't.
# The stance CTE filters out null rows so those nulls can't distort CUME_DIST.
# LEFT JOIN (not INNER) means a hitter missing from the stance snapshot still
# comes back — just with null stance columns.
_LEAGUE_PERCENTILE_SQL = """
WITH swing_ranked AS (
    SELECT
        id,
        side,
        name,
        attack_angle,
        swing_tilt,
        avg_intercept_y_vs_batter,
        avg_bat_speed,
        ideal_attack_angle_rate,
        CUME_DIST() OVER (ORDER BY attack_angle)               AS attack_angle_pct,
        CUME_DIST() OVER (ORDER BY swing_tilt)                 AS swing_tilt_pct,
        CUME_DIST() OVER (ORDER BY avg_intercept_y_vs_batter)  AS avg_intercept_y_vs_batter_pct,
        CUME_DIST() OVER (ORDER BY avg_bat_speed)              AS avg_bat_speed_pct,
        CUME_DIST() OVER (ORDER BY ideal_attack_angle_rate)   AS ideal_attack_angle_rate_pct
    FROM league_swing_path
),
stance_ranked AS (
    SELECT
        id,
        side,
        avg_foot_sep,
        avg_stance_angle,
        CUME_DIST() OVER (ORDER BY avg_foot_sep)      AS avg_foot_sep_pct,
        CUME_DIST() OVER (ORDER BY avg_stance_angle)  AS avg_stance_angle_pct
    FROM league_stance
    WHERE avg_foot_sep IS NOT NULL
)
SELECT
    sw.*,
    st.avg_foot_sep,
    st.avg_foot_sep_pct,
    st.avg_stance_angle,
    st.avg_stance_angle_pct
FROM swing_ranked sw
LEFT JOIN stance_ranked st
    ON sw.id = st.id AND sw.side = st.side
WHERE sw.id = ?
"""

# Columns the stance CTE needs. If the stance pull is empty or shaped
# unexpectedly, we swap in an empty frame with these columns so the SQL still
# runs (the LEFT JOIN just yields null stance) instead of raising.
_STANCE_COLS = ["id", "side", "avg_foot_sep", "avg_stance_angle"]


def league_percentiles(player_id: int, season: int) -> dict:
    """Rank one hitter's swing-geometry + stance metrics against the league.

    Returns a dict:
        {
          "found": bool,                     # was the hitter in the swing-path table?
          "metrics": {
             <league col>: {
                "label": str,
                "value": float | None,       # the hitter's own value
                "percentile": int | None,    # 0-100, share of league at or below
                "higher_is_better": bool | None,
             }, ...
          }
        }

    Swing-path metrics come from the season leaderboard; stance metrics (width,
    angle) are joined in from the season-agnostic stance snapshot. A hitter with
    no stance row still gets found=True with the swing-path metrics — the stance
    entries just have value/percentile = None. An unqualified hitter (absent from
    the swing-path leaderboard) comes back found=False.
    """
    import duckdb

    baseline = load_swing_path_baseline(season)
    empty = {"found": False, "metrics": {}}
    if baseline.empty or "id" not in baseline.columns:
        return empty

    # Load the stance snapshot; if it's missing/misshaped, fall back to an empty
    # frame with the right columns so the LEFT JOIN degrades gracefully to nulls.
    stance = load_stance_baseline()
    if not set(_STANCE_COLS).issubset(stance.columns):
        stance = pd.DataFrame(columns=_STANCE_COLS)

    # register() exposes each pandas frame to SQL under a table name.
    # execute(sql, params) binds player_id to the single "?" placeholder — the
    # parameterized form, so an id value can never be mis-read as SQL.
    con = duckdb.connect()
    con.register("league_swing_path", baseline)
    con.register("league_stance", stance)
    ranked = con.execute(_LEAGUE_PERCENTILE_SQL, [player_id]).df()
    con.close()

    if ranked.empty:
        return empty
    row = ranked.iloc[0]

    metrics: dict[str, dict] = {}
    for col, meta in _LEAGUE_METRICS.items():
        # Defensive: skip any metric the leaderboard didn't include this season.
        if col not in ranked.columns:
            continue
        value = row[col]
        pct = row[f"{col}_pct"]
        metrics[col] = {
            "label": meta["label"],
            "value": None if pd.isna(value) else float(value),
            # CUME_DIST is 0-1; scale to a 0-100 percentile for display.
            "percentile": None if pd.isna(pct) else round(float(pct) * 100),
            "higher_is_better": meta["higher_is_better"],
        }
    return {"found": True, "metrics": metrics}


# ===========================================================================
# SQL 2 — per-pitch-type swing breakdown (GROUP BY over pitch-level swings)
# ===========================================================================
# One row per pitch type, averaging the swing metrics. HAVING drops pitch types
# the hitter barely saw, so a two-swing sample can't masquerade as a tendency.
_BY_PITCH_SQL = """
SELECT
    pitch_type,
    COUNT(*)                                                       AS swings,
    ROUND(AVG(attack_angle), 1)                                   AS attack_angle,
    ROUND(AVG(bat_speed), 1)                                      AS bat_speed,
    ROUND(AVG(intercept_ball_minus_batter_pos_y_inches), 1)       AS intercept_depth_in,
    ROUND(AVG(intercept_ball_minus_batter_pos_x_inches), 1)       AS intercept_side_in
FROM swings
WHERE attack_angle IS NOT NULL          -- non-swings have null tracking; drop them
GROUP BY pitch_type
HAVING COUNT(*) >= ?
ORDER BY swings DESC
"""

# Pitch-level columns the by-pitch query needs. If a pull predates bat tracking
# (older season) these are absent, and we return an empty breakdown instead of
# letting SQL raise on a missing column.
_REQUIRED_SWING_COLS = [
    "pitch_type",
    "attack_angle",
    "bat_speed",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]


def swing_profile_by_pitch(swings: pd.DataFrame, min_swings: int = 15) -> pd.DataFrame:
    """Average a hitter's swing geometry by pitch type.

    Args:
        swings:     pitch-level frame for ONE hitter (from pull_statcast_batter).
        min_swings: minimum swings on a pitch type to include it (default 15).

    Returns a DataFrame (pitch_type, swings, attack_angle, bat_speed,
    intercept_depth_in, intercept_side_in), or an empty DataFrame when the pull
    has no bat-tracking columns.
    """
    import duckdb

    if not set(_REQUIRED_SWING_COLS).issubset(swings.columns):
        return pd.DataFrame()

    con = duckdb.connect()
    con.register("swings", swings)
    out = con.execute(_BY_PITCH_SQL, [min_swings]).df()
    con.close()
    return out


def intercept_points(swings: pd.DataFrame) -> pd.DataFrame:
    """Per-swing contact points for the intercept scatter plot.

    Returns the tidy columns the chart needs — horizontal and depth intercept
    (inches), attack angle, pitch type, and the batted-ball result — one row per
    swing that has tracking data. Empty DataFrame if the pull has no tracking.
    """
    if not set(_REQUIRED_SWING_COLS).issubset(swings.columns):
        return pd.DataFrame()

    cols = {
        "intercept_ball_minus_batter_pos_x_inches": "intercept_side_in",
        "intercept_ball_minus_batter_pos_y_inches": "intercept_depth_in",
        "attack_angle": "attack_angle",
        "pitch_type": "pitch_type",
    }
    # `events` (single, double, strikeout, ...) lets the chart color hits vs
    # outs. It's optional — include it only if the pull has it.
    if "events" in swings.columns:
        cols["events"] = "events"

    out = swings.loc[:, list(cols)].rename(columns=cols)
    # Keep only real swings (tracking present). dropna on the two intercept
    # columns is enough — non-swings are null in both.
    return out.dropna(subset=["intercept_side_in", "intercept_depth_in"]).reset_index(drop=True)


# ===========================================================================
# Orchestrator — one call assembles the whole card
# ===========================================================================

def swing_geometry_card(
    player_id: int,
    swings: pd.DataFrame,
    season: int,
    min_swings: int = 15,
) -> dict:
    """Assemble the full swing-geometry card for one hitter.

    Args:
        player_id: the hitter's key_mlbam (matches the leaderboard `id`).
        swings:    pitch-level frame for this hitter (pull_statcast_batter).
        season:    league season to rank against.
        min_swings: per-pitch-type minimum for the breakdown.

    Returns a dict with:
        season, has_tracking (bool), league (percentile block),
        by_pitch (DataFrame), points (DataFrame for the scatter).

    `has_tracking` is False when the season predates bat tracking or the pull
    lacks the swing columns — the render layer shows a graceful note instead.
    """
    has_tracking = (
        season >= config.SWING_TRACKING_FIRST_SEASON
        and set(_REQUIRED_SWING_COLS).issubset(swings.columns)
    )

    return {
        "season": season,
        "has_tracking": has_tracking,
        "league": league_percentiles(player_id, season) if has_tracking else {"found": False, "metrics": {}},
        "by_pitch": swing_profile_by_pitch(swings, min_swings) if has_tracking else pd.DataFrame(),
        "points": intercept_points(swings) if has_tracking else pd.DataFrame(),
    }
