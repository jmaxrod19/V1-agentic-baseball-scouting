"""
Metrics module: compute scouting-relevant statistics from Statcast pitch data.

All rates are stored as decimals (0.0–1.0). Converting to display percentages
happens at the report layer, not here, so report math never mixes 0.254 and 25.4.

Public functions:
    pitcher_arsenal(df, pitcher=None) -> pd.DataFrame
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
            h_break_in, v_break_in, avg_extension, whiff_rate
        Sorted by usage descending (primary pitch first).
        All rates are decimals (0.0–1.0). Movement values are in inches.

        Note: `usage` is each pitch type's share of pitches in *this* DataFrame.
        If you pass a pre-filtered frame (e.g., only one game), usage reflects
        that sample — make sure df is a representative population for the pitcher.
    """
    # Work on a copy so we never mutate the caller's frame. CLAUDE.md:
    # "code must not assume [columns] exist" — every column access below
    # checks presence first.
    df = df.copy()

    # -- 1. Filter to one pitcher if requested ------------------------------
    if pitcher is not None:
        if isinstance(pitcher, int):
            # pitcher_id is the MLB player ID (integer in Statcast schema).
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

    # -- 2. Drop rows with no pitch type ------------------------------------
    # Statcast uses null pitch_type for automatic balls and a handful of
    # edge cases. Exclude them before grouping so they don't pollute totals.
    if "pitch_type" not in df.columns:
        raise KeyError("Column 'pitch_type' not found in DataFrame.")

    df = df[df["pitch_type"].notna() & (df["pitch_type"] != "")]

    total_pitches = len(df)
    if total_pitches == 0:
        raise ValueError("No valid pitch_type rows to summarize.")

    # -- 3. Build per-pitch-type aggregations -------------------------------
    # We group once and compute all metrics. The lambda approach is explicit
    # and readable — each metric is its own named line rather than a hidden
    # column inside a chained agg().

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

        # Horizontal break: pfx_x is in FEET (the Statcast standard). Multiply
        # by 12 to convert to inches, which is the front-office reporting unit.
        # Positive = arm-side run; negative = glove-side.
        # We capture the mean first, then multiply — avoids `None * 12` TypeError
        # if _safe_mean returns None (all values NaN after dropna).
        _pfx_x = _safe_mean(group["pfx_x"]) if "pfx_x" in group.columns else None
        row["h_break_in"] = _pfx_x * 12 if _pfx_x is not None else None

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

        # Whiff rate — misses per swing. See _whiff_codes rationale above.
        row["whiff_rate"] = _whiff_rate(group)

        rows.append(row)

    # -- 4. Assemble, sort, return ------------------------------------------
    result = pd.DataFrame(rows, columns=[
        "pitch_type", "n_pitches", "usage",
        "avg_velo", "avg_spin",
        "h_break_in", "v_break_in",
        "avg_extension", "whiff_rate",
    ])

    # Primary pitch first — most-used pitch anchors the top of any report.
    result = result.sort_values("usage", ascending=False).reset_index(drop=True)

    return result
