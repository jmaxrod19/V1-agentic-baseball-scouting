"""League percentile ranking for hitter batted-ball metrics.

Turns a naked number ("88.8 mph avg EV") into context ("63rd percentile") by
ranking it against every qualified hitter's value from the season's Savant
exit-velocity leaderboard.

Flow:
    load_hitting_baseline(season)  -> the league table (pulled once, cached)
    hitter_percentiles(profile, season) -> {metric: percentile 0-100}

Scope: hitters only for now (pitcher percentiles are a different metric set —
velocity, spin, movement — and a separate leaderboard). All the metrics ranked
here are "higher is better," so a high percentile always reads as good.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

try:
    from . import config, loaders
except ImportError:  # direct-script / non-package execution
    import config
    import loaders


# Our metric key -> (league leaderboard column, scale to align units).
# Our rate metrics are decimals (0.39 = 39%); the leaderboard stores them as
# percents (39.0), so those map with scale=100. Exit-velocity columns are
# already the same unit (mph), so scale=1. avg_la is intentionally absent — it
# has no "better" direction, so a percentile would mislead.
_HITTER_METRIC_MAP: dict[str, tuple[str, float]] = {
    "avg_ev":          ("avg_hit_speed", 1.0),
    "max_ev":          ("max_hit_speed", 1.0),
    "hard_hit_rate":   ("ev95percent", 100.0),
    "barrel_rate":     ("brl_percent", 100.0),
    "sweet_spot_rate": ("anglesweetspotpercent", 100.0),
}


def load_hitting_baseline(season: int) -> pd.DataFrame:
    """Return the season's league hitting leaderboard, cached to data/processed/.

    First call for a season pulls from Savant and writes a CSV; later calls
    (this process or after a restart) read the CSV — instant and offline, the
    same disk-cache idea as the per-player pulls.
    """
    path = config.PROCESSED_DIR / f"league_hitting_{season}.csv"
    if path.exists():
        return pd.read_csv(path)

    df = loaders.pull_league_hitting_baseline(season)
    # Only persist a *completed* season — a current-season leaderboard still
    # changes daily, so caching it would freeze stale percentiles until restart.
    if not df.empty and season < date.today().year:
        df.to_csv(path, index=False)
    return df


def percentile_of(value: float | None, population: pd.Series) -> int | None:
    """Rank-based percentile (0-100): the share of the league at or below `value`.

    Returns None when there's nothing to rank against or the value is missing, so
    the display layer can simply omit a percentile rather than show a wrong 0.
    """
    if value is None or pd.isna(value):
        return None
    pop = population.dropna()
    if pop.empty:
        return None
    return round(float((pop <= value).mean()) * 100)


# Our display key -> Savant pitcher-percentile-rank column. These are already
# 0-100 percentiles (oriented high = good), so we read them straight through
# rather than computing — no scaling or ranking needed.
_PITCHER_PERCENTILE_COLS: dict[str, str] = {
    "fb_velocity": "fb_velocity",
    "fb_spin": "fb_spin",
    "whiff": "whiff_percent",
    "chase": "chase_percent",
    "k": "k_percent",
    "bb": "bb_percent",
    "hard_hit": "hard_hit_percent",
    "barrel": "brl_percent",
    "xera": "xera",
    "xwoba": "xwoba",
}


def load_pitcher_percentile_baseline(season: int) -> pd.DataFrame:
    """Return the season's per-pitcher percentile-rank table, cached to disk.

    Same completed-season caching rule as the hitter baseline: a current-season
    table still moves, so it's pulled fresh; finished seasons are frozen to CSV.
    """
    path = config.PROCESSED_DIR / f"league_pitcher_pctile_{season}.csv"
    if path.exists():
        return pd.read_csv(path)

    df = loaders.pull_pitcher_percentile_ranks(season)
    if not df.empty and season < date.today().year:
        df.to_csv(path, index=False)
    return df


def pitcher_percentiles(player_id: int, season: int) -> dict[str, int | None]:
    """Look up a pitcher's Savant percentile ranks by player id for a season.

    Returns {display key: percentile 0-100}. An empty dict when the pitcher
    isn't in the (qualified) table — the report then simply omits the section.
    """
    table = load_pitcher_percentile_baseline(season)
    if table.empty or "player_id" not in table.columns:
        return {}

    match = table[table["player_id"] == player_id]
    if match.empty:
        return {}
    row = match.iloc[0]

    out: dict[str, int | None] = {}
    for key, col in _PITCHER_PERCENTILE_COLS.items():
        if col not in table.columns:
            continue
        val = row[col]
        out[key] = None if pd.isna(val) else int(round(val))
    return out


def hitter_percentiles(profile: pd.Series | dict, season: int) -> dict[str, int | None]:
    """Map a hitter's batted-ball profile to a {metric: percentile} dict.

    Args:
        profile: one hitter's metric values (a row of hitter_batted_ball, or a
                 plain dict). Rate metrics are decimals (0.0-1.0), matching the
                 report's internal representation.
        season:  which league season to rank against.
    """
    baseline = load_hitting_baseline(season)
    out: dict[str, int | None] = {}
    for key, (col, scale) in _HITTER_METRIC_MAP.items():
        if key not in profile or col not in baseline.columns:
            continue
        raw = profile[key]
        value = None if raw is None or pd.isna(raw) else raw * scale
        out[key] = percentile_of(value, baseline[col])
    return out
