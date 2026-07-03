"""
Metrics module: compute scouting-relevant statistics from Statcast pitch data.

All rates are stored as decimals (0.0–1.0). Converting to display percentages
happens at the report layer, not here, so report math never mixes 0.254 and 25.4.

Public functions:
    pitcher_arsenal(df, pitcher=None)           -> per-pitch-type summary
    pitcher_location(df, pitcher=None)          -> per-pitch-type plate location
    pitcher_handedness_splits(df, pitcher=None) -> per (batter hand, pitch type)
    hitter_batted_ball(df, batter=None)         -> overall batted-ball quality
    hitter_platoon_batted_ball(df, batter=None) -> batted-ball quality vs LHP/RHP
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Plate-discipline classification sets
# ---------------------------------------------------------------------------
# Source: FanGraphs Plate Discipline Glossary + Statcast CSV docs.
# These are the exact `description` strings Statcast uses.

# Any pitch the batter offered at, regardless of outcome.
_SWING_CODES: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",   # swing-and-miss, catcher blocked it in dirt
    "foul",
    "foul_tip",                  # goes directly to catcher's glove — a miss by rule
    "foul_bunt",
    "bunt_foul_tip",
    "hit_into_play",
    "missed_bunt",
})

# A swing that results in zero contact. foul_tip counts (rulebook miss);
# plain "foul" does NOT (the barrel touched the ball). missed_bunt is also
# a zero-contact swing — the batter offered at a pitch and whiffed the bunt.
_WHIFF_CODES: frozenset[str] = frozenset({
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
})

# Statcast's `zone` column buckets the pitch location into a fixed grid:
#   zones 1–9  = inside the strike zone (a 3x3 grid)
#   zones 11–14 = outside the strike zone (four outer quadrants)
# There is no zone 10. So "out of zone" is simply zone > 10, and this is the
# same in/out split Baseball Savant's own chase-rate leaderboards use.
_OOZ_THRESHOLD = 10  # zone strictly greater than this is out of the strike zone

# Batted-ball quality thresholds (Statcast / Savant conventions, confirmed
# against MLB.com glossary + Savant CSV docs).
_BATTED_BALL = "hit_into_play"   # the `description` value marking a ball in play
_HARD_HIT_MPH = 95.0             # exit velocity >= this = a "hard-hit" ball
_SWEET_SPOT_LA = (8.0, 32.0)     # launch angle in this range (incl.) = "sweet spot"
_BARREL_BUCKET = 6               # launch_speed_angle == 6 is Savant's "Barrel" bucket


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_mean(series: pd.Series) -> float | None:
    """Return the mean of a numeric series, or None if all values are NaN.

    Using None instead of NaN makes the final DataFrame cleaner when a pitch
    type genuinely lacks a column (e.g. no spin data on older pulls).
    """
    dropped = series.dropna()
    return dropped.mean() if len(dropped) else None


def _whiff_rate(group: pd.DataFrame) -> float | None:
    """Whiffs / swings for one pitch-type group.

    Returns None (not 0) when there are zero swings, so callers can
    distinguish "no data" from "batter never missed."
    """
    if "description" not in group.columns:
        return None

    is_swing = group["description"].isin(_SWING_CODES)
    swings = is_swing.sum()
    if swings == 0:
        return None

    whiffs = group["description"].isin(_WHIFF_CODES).sum()
    return whiffs / swings


def _chase_rate(group: pd.DataFrame) -> float | None:
    """Out-of-zone swings / out-of-zone pitches for one group (a.k.a. O-Swing%).

    Needs both `zone` (to know which pitches were out of the zone) and
    `description` (to know which were swung at). Returns None when either
    column is missing or the pitcher threw nothing out of the zone.
    """
    if "zone" not in group.columns or "description" not in group.columns:
        return None

    out_of_zone = group["zone"] > _OOZ_THRESHOLD
    n_ooz = out_of_zone.sum()
    if n_ooz == 0:
        return None

    # A chase = a swing (any offer) on a pitch that was out of the zone.
    chases = (out_of_zone & group["description"].isin(_SWING_CODES)).sum()
    return chases / n_ooz


def _in_zone_rate(group: pd.DataFrame) -> float | None:
    """Share of pitches thrown inside the strike zone (zones 1–9).

    Denominator is pitches with a known zone (Statcast always assigns one when
    tracking succeeded), so a handful of untracked pitches don't skew the rate.
    """
    if "zone" not in group.columns:
        return None

    zone = group["zone"]
    n_known = zone.notna().sum()
    if n_known == 0:
        return None

    in_zone = zone.between(1, 9).sum()   # inclusive 1..9 = the 3x3 strike grid
    return in_zone / n_known


# -- Batted-ball rate helpers ------------------------------------------------
# Each returns (matching batted balls / batted balls with a known value), or
# None if the column is absent or entirely null. The denominator is per-column
# so a few untracked balls (null EV/LA) don't distort the rate.

def _rate_ge(series: pd.Series | None, threshold: float) -> float | None:
    """Share of values >= threshold (e.g. hard-hit: exit velo >= 95)."""
    if series is None:
        return None
    known = series.notna().sum()
    return (series >= threshold).sum() / known if known else None


def _rate_eq(series: pd.Series | None, value: float) -> float | None:
    """Share of values equal to `value` (e.g. barrel: launch_speed_angle == 6)."""
    if series is None:
        return None
    known = series.notna().sum()
    return (series == value).sum() / known if known else None


def _rate_between(series: pd.Series | None, lo: float, hi: float) -> float | None:
    """Share of values in [lo, hi] (e.g. sweet-spot: launch angle 8–32)."""
    if series is None:
        return None
    known = series.notna().sum()
    return series.between(lo, hi).sum() / known if known else None


def _prepare_batted_balls(
    df: pd.DataFrame,
    batter: int | None,
) -> pd.DataFrame:
    """Copy df, optionally filter to one batter, and keep only balls in play.

    The hitter analogue of _prepare_pitches. `batter` is an MLB player id (int):
    Statcast's `player_name` column holds the *pitcher's* name, so a hitter can
    only be selected by id, not by name.

    Bunts are excluded when the play-description text (`des`) is available —
    Statcast exposes no clean bunt flag, and bunts depress exit velocity without
    reflecting swing quality. When `des` is absent, bunts remain (documented
    limitation) — for most hitters they're a negligible share of batted balls.
    """
    df = df.copy()

    if batter is not None:
        if "batter" not in df.columns:
            raise KeyError("Column 'batter' not found — cannot filter by batter id.")
        df = df[df["batter"].astype("Int64") == batter]
        if df.empty:
            raise ValueError(f"No rows found for batter={batter!r}.")

    if "description" not in df.columns:
        raise KeyError("Column 'description' not found in DataFrame.")

    bbe = df[df["description"] == _BATTED_BALL].copy()

    if "des" in bbe.columns:
        is_bunt = bbe["des"].str.contains("bunt", case=False, na=False)
        bbe = bbe[~is_bunt]

    if bbe.empty:
        raise ValueError("No batted balls (hit_into_play) to summarize.")

    return bbe


def _batted_ball_metrics(bbe: pd.DataFrame) -> dict:
    """Compute the batted-ball-quality metric set for a set of balls in play.

    Rates are decimals (0.0–1.0). xwOBACON/xBACON are rate stats reported on
    their own scale (~.250–.400), not percentages — the display layer shows
    them as .XXX, everything else as a percent.
    """
    ev = bbe.get("launch_speed")
    la = bbe.get("launch_angle")
    lsa = bbe.get("launch_speed_angle")

    return {
        "n_bbe":           len(bbe),
        "avg_ev":          _safe_mean(ev) if ev is not None else None,
        "max_ev":          ev.max() if ev is not None and ev.notna().any() else None,
        "avg_la":          _safe_mean(la) if la is not None else None,
        "hard_hit_rate":   _rate_ge(ev, _HARD_HIT_MPH),
        "barrel_rate":     _rate_eq(lsa, _BARREL_BUCKET),
        "sweet_spot_rate": _rate_between(la, *_SWEET_SPOT_LA),
        # xwOBACON / xBACON: mean expected value over balls in play (contact only).
        "xwobacon": _safe_mean(bbe["estimated_woba_using_speedangle"])
                    if "estimated_woba_using_speedangle" in bbe.columns else None,
        "xbacon":   _safe_mean(bbe["estimated_ba_using_speedangle"])
                    if "estimated_ba_using_speedangle" in bbe.columns else None,
    }


# Column order shared by the overall and platoon batted-ball frames.
_BATTED_BALL_COLS = [
    "n_bbe", "avg_ev", "max_ev", "avg_la",
    "hard_hit_rate", "barrel_rate", "sweet_spot_rate",
    "xwobacon", "xbacon",
]


def _prepare_pitches(
    df: pd.DataFrame,
    pitcher: int | str | None,
) -> pd.DataFrame:
    """Copy df, optionally filter to one pitcher, and drop null pitch types.

    This is the shared front-end for every metric function: it guarantees a
    clean, non-mutating frame with only real pitches. Extracting it keeps the
    pitcher-selection logic in exactly one place, so a fix (like the Int64
    cast below) applies everywhere at once.

    Raises:
        KeyError   if a column needed for the requested filter is missing.
        ValueError if the filter or pitch-type cleaning leaves no rows.
    """
    # Never mutate the caller's frame.
    df = df.copy()

    if pitcher is not None:
        if isinstance(pitcher, int):
            if "pitcher" not in df.columns:
                raise KeyError("Column 'pitcher' not found — cannot filter by id.")
            # Cast to Int64 (nullable) before comparing: the column often loads
            # as float64 when any rows have NaN IDs, so int == float never matches.
            df = df[df["pitcher"].astype("Int64") == pitcher]
        else:
            # player_name is "Last, First" in pybaseball pulls.
            if "player_name" not in df.columns:
                raise KeyError("Column 'player_name' not found — cannot filter by name.")
            df = df[df["player_name"] == pitcher]

        if df.empty:
            raise ValueError(f"No rows found for pitcher={pitcher!r}.")

    # Statcast uses a null pitch_type for automatic balls and a few edge cases.
    # Exclude them before grouping so they don't pollute totals.
    if "pitch_type" not in df.columns:
        raise KeyError("Column 'pitch_type' not found in DataFrame.")

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]
    if df.empty:
        raise ValueError("No valid pitch_type rows to summarize.")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pitcher_arsenal(
    df: pd.DataFrame,
    pitcher: int | str | None = None,
) -> pd.DataFrame:
    """Summarize a pitcher's arsenal: one row per pitch type.

    Args:
        df:      A cleaned Statcast DataFrame from loaders.load_statcast_csv
                 (or a live pull). Must contain at least `pitch_type`.
        pitcher: Optional filter. Pass a player_id (int) or player_name (str)
                 to select one pitcher from a multi-pitcher frame. If None,
                 the entire df is treated as one pitcher's data.

    Returns:
        DataFrame with columns:
            pitch_type, n_pitches, usage, avg_velo, avg_spin,
            h_break_in, v_break_in, avg_extension, whiff_rate, chase_rate
        Sorted by usage descending (primary pitch first).
        All rates are decimals (0.0–1.0). Movement values are in inches.

        Note: `usage` is each pitch type's share of pitches in *this* DataFrame.
        If you pass a pre-filtered frame (e.g., only one game), usage reflects
        that sample — make sure df is a representative population for the pitcher.
    """
    df = _prepare_pitches(df, pitcher)
    total_pitches = len(df)

    rows: list[dict] = []

    # sort=False skips alphabetical sorting of group keys — we re-sort by usage
    # at the end anyway, so there's no point paying for an intermediate sort.
    for pitch_type, group in df.groupby("pitch_type", sort=False):
        n = len(group)

        row: dict = {
            "pitch_type": pitch_type,
            "n_pitches":  n,
            # usage: this pitch type's share of all pitches thrown.
            "usage":      n / total_pitches,
        }

        # Velocity — mean release speed in mph.
        row["avg_velo"] = (
            _safe_mean(group["release_speed"])
            if "release_speed" in group.columns else None
        )

        # Spin rate — mean in rpm.
        row["avg_spin"] = (
            _safe_mean(group["release_spin_rate"])
            if "release_spin_rate" in group.columns else None
        )

        # Horizontal break: pfx_x is in FEET (Statcast standard); ×12 → inches,
        # the front-office reporting unit.
        #
        # SIGN CONVENTION (industry standard): a right-handed pitcher's arm-side
        # run is POSITIVE, glove-side is negative — so a RHP's fastball/changeup
        # read positive and his slider/curve read negative. Raw Statcast pfx_x is
        # from the catcher's perspective, where a RHP's arm-side run comes out
        # NEGATIVE, so we negate it. This is a single fixed flip (not
        # handedness-aware): a left-handed pitcher's arm-side run is the mirror
        # image and therefore reads negative, which is exactly the scouting
        # convention ("righty fastball +HB, lefty fastball −HB").
        #
        # Capture the mean first, then transform — avoids `None * 12` TypeError
        # when _safe_mean returns None (all pfx_x NaN after dropna).
        _pfx_x = _safe_mean(group["pfx_x"]) if "pfx_x" in group.columns else None
        row["h_break_in"] = -(_pfx_x * 12) if _pfx_x is not None else None

        # Vertical break: pfx_z, same foot-to-inch conversion.
        # Positive = "rise" (less gravity drop than a theoretical spinless pitch);
        # negative = additional drop.
        _pfx_z = _safe_mean(group["pfx_z"]) if "pfx_z" in group.columns else None
        row["v_break_in"] = _pfx_z * 12 if _pfx_z is not None else None

        # Extension — how far in front of the rubber the ball is released (feet).
        # Higher extension = ball closer to the hitter, effectively shortens
        # their reaction time. No unit conversion needed (feet is standard here).
        row["avg_extension"] = (
            _safe_mean(group["release_extension"])
            if "release_extension" in group.columns else None
        )

        # Plate-discipline rates. See helper docstrings for the exact formulas.
        row["whiff_rate"] = _whiff_rate(group)
        row["chase_rate"] = _chase_rate(group)

        rows.append(row)

    result = pd.DataFrame(rows, columns=[
        "pitch_type", "n_pitches", "usage",
        "avg_velo", "avg_spin",
        "h_break_in", "v_break_in",
        "avg_extension", "whiff_rate", "chase_rate",
    ])

    # Primary pitch first — most-used pitch anchors the top of any report.
    result = result.sort_values("usage", ascending=False).reset_index(drop=True)

    return result


def pitcher_location(
    df: pd.DataFrame,
    pitcher: int | str | None = None,
) -> pd.DataFrame:
    """Per-pitch-type plate location summary.

    Args:
        df:      Cleaned Statcast DataFrame.
        pitcher: Optional player_id (int) or player_name (str) filter.

    Returns:
        DataFrame with columns:
            pitch_type, avg_plate_x, avg_plate_z, in_zone_rate
        Sorted by pitch count descending (matches the arsenal ordering).

        plate_x / plate_z are in FEET from the center of the plate, catcher's
        view: +plate_x is toward the batter's… it depends on batter hand, so we
        report the raw mean and leave interpretation to the location heatmaps.
        in_zone_rate is a decimal (0.0–1.0).
    """
    df = _prepare_pitches(df, pitcher)

    rows: list[dict] = []
    for pitch_type, group in df.groupby("pitch_type", sort=False):
        rows.append({
            "pitch_type":   pitch_type,
            "n_pitches":    len(group),  # kept only for sorting; dropped below
            "avg_plate_x":  _safe_mean(group["plate_x"]) if "plate_x" in group.columns else None,
            "avg_plate_z":  _safe_mean(group["plate_z"]) if "plate_z" in group.columns else None,
            "in_zone_rate": _in_zone_rate(group),
        })

    result = pd.DataFrame(rows, columns=[
        "pitch_type", "n_pitches", "avg_plate_x", "avg_plate_z", "in_zone_rate",
    ])
    # Order by usage so this table lines up row-for-row with pitcher_arsenal.
    result = result.sort_values("n_pitches", ascending=False).reset_index(drop=True)
    return result.drop(columns=["n_pitches"])


def pitcher_handedness_splits(
    df: pd.DataFrame,
    pitcher: int | str | None = None,
) -> pd.DataFrame:
    """Per (batter hand, pitch type) usage and outcome rates.

    Shows how the pitcher's plan changes against lefties vs righties — which
    pitches he leans on, and how they perform, split by the `stand` column
    (batter handedness, 'L' or 'R').

    Args:
        df:      Cleaned Statcast DataFrame.
        pitcher: Optional player_id (int) or player_name (str) filter.

    Returns:
        DataFrame with columns:
            stand, pitch_type, usage, avg_velo, whiff_rate, chase_rate
        `usage` is the pitch's share of pitches thrown *to that batter hand*
        (so each hand's usages sum to ~1.0). Sorted by hand, then usage desc.

    Raises:
        KeyError if the `stand` column is absent.
    """
    df = _prepare_pitches(df, pitcher)

    if "stand" not in df.columns:
        raise KeyError("Column 'stand' not found — cannot compute handedness splits.")

    # Drop rows with no batter hand (rare, but keeps the denominators honest).
    df = df[df["stand"].notna() & (df["stand"] != "")]
    if df.empty:
        raise ValueError("No rows with a known batter hand ('stand').")

    # Pitches thrown to each hand — the per-hand usage denominator.
    hand_totals = df.groupby("stand").size()

    rows: list[dict] = []
    for (stand, pitch_type), group in df.groupby(["stand", "pitch_type"], sort=False):
        rows.append({
            "stand":      stand,
            "pitch_type": pitch_type,
            "usage":      len(group) / hand_totals[stand],
            "avg_velo":   _safe_mean(group["release_speed"]) if "release_speed" in group.columns else None,
            "whiff_rate": _whiff_rate(group),
            "chase_rate": _chase_rate(group),
        })

    result = pd.DataFrame(rows, columns=[
        "stand", "pitch_type", "usage", "avg_velo", "whiff_rate", "chase_rate",
    ])
    # Group each hand together, most-used pitch first within the hand.
    result = result.sort_values(
        ["stand", "usage"], ascending=[True, False]
    ).reset_index(drop=True)
    return result


def hitter_batted_ball(
    df: pd.DataFrame,
    batter: int | None = None,
) -> pd.DataFrame:
    """Overall batted-ball-quality profile for a hitter (one row).

    Args:
        df:     Cleaned Statcast DataFrame.
        batter: Optional MLB player id (int). None treats df as one hitter's
                balls in play. (Name filtering isn't supported — `player_name`
                is the pitcher, not the batter.)

    Returns:
        One-row DataFrame with columns:
            n_bbe, avg_ev, max_ev, avg_la,
            hard_hit_rate, barrel_rate, sweet_spot_rate, xwobacon, xbacon
        Rates are decimals (0.0–1.0); xwobacon/xbacon are on the wOBA/BA scale.
    """
    bbe = _prepare_batted_balls(df, batter)
    metrics = _batted_ball_metrics(bbe)
    return pd.DataFrame([metrics], columns=_BATTED_BALL_COLS)


def hitter_platoon_batted_ball(
    df: pd.DataFrame,
    batter: int | None = None,
) -> pd.DataFrame:
    """Batted-ball quality split by pitcher handedness (vs LHP / vs RHP).

    Same metrics as hitter_batted_ball, one row per pitcher hand (`p_throws`),
    so you can see whether the hitter squares up lefties or righties better.

    Returns:
        DataFrame with a leading `vs_hand` column ('L'/'R') plus the batted-ball
        columns. Sorted by hand ('L' then 'R').

    Raises:
        KeyError if `p_throws` is absent.
    """
    bbe = _prepare_batted_balls(df, batter)

    if "p_throws" not in bbe.columns:
        raise KeyError("Column 'p_throws' not found — cannot split by pitcher hand.")

    bbe = bbe[bbe["p_throws"].notna() & (bbe["p_throws"] != "")]
    if bbe.empty:
        raise ValueError("No batted balls with a known pitcher hand ('p_throws').")

    rows: list[dict] = []
    for hand, group in bbe.groupby("p_throws", sort=False):
        rows.append({"vs_hand": hand, **_batted_ball_metrics(group)})

    result = pd.DataFrame(rows, columns=["vs_hand", *_BATTED_BALL_COLS])
    return result.sort_values("vs_hand").reset_index(drop=True)
