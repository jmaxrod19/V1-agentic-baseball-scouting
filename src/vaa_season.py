"""
A full season of four-seam VAA results, small enough to ship with the web app.

The raw league file is hundreds of megabytes. The chart only needs a handful of
averages per pitcher, so we compute those once and keep them in a small CSV:
    one "pitcher" row per pitcher, plus one "league" row per throwing hand.

    build_season_table(data, season, heights)   big pitch data -> small table
    comparison_from_table(table, pitcher_id)    small table    -> a Comparison
    find_pitchers(table, query)                 typed name     -> matching rows

Tradeoff: the table is full-season only. Date ranges would need the pitches.
Modeled paths still work at any comparison height, because they are recomputed
from the stored release state (see vaa_model.model_trajectory).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from src import vaa_model
from src.vaa_model import DEFAULT_PLATE_HEIGHT, Comparison, Side

# Side attributes saved as they are, and the release-state numbers saved beside them.
_SIDE_FIELDS = ("pitches", "extension", "arm_angle", "velocity", "ivb",
                "observed_vaa", "plate_height", "vaa_aa")
_STATE_FIELDS = ("release_y", "release_z", "speed_y", "ay", "az")
_NOTE_SEPARATOR = " | "


def _side_to_row(side: Side) -> dict:
    row = {name: getattr(side, name) for name in _SIDE_FIELDS}
    row.update({name: side.state[name] for name in _STATE_FIELDS})
    return row


def _row_to_side(row: pd.Series, plate_height: float) -> Side:
    state = {name: float(row[name]) for name in _STATE_FIELDS}
    values = {name: float(row[name]) for name in _SIDE_FIELDS}
    values["pitches"] = int(row["pitches"])
    return Side(
        release_height=state["release_z"],       # same number, stored once
        state=state,
        modeled=vaa_model.model_trajectory(state, plate_height),
        **values,
    )


def build_season_table(data: pd.DataFrame, season: int,
                       heights: dict[int, int]) -> pd.DataFrame:
    """Run the comparison for every four-seam thrower and keep one row each.

    data     league-wide pitch rows (loaders.load_statcast_csv).
    heights  {mlbam_id: listed height in inches}, for the figurines.
    """
    # Filter once here so each comparison searches a much smaller frame.
    keep = (data["pitch_type"] == "FF") & (data["game_year"] == season)
    if "game_type" in data.columns:
        keep &= data["game_type"] == "R"
    fastballs = data[keep]
    if fastballs.empty:
        raise ValueError(f"No regular-season four-seam fastballs for season {season}.")

    pitcher_rows, league_rows = [], {}
    for pitcher_id in sorted(fastballs["pitcher"].dropna().unique()):
        try:
            c = vaa_model.compare_fastball(fastballs, int(pitcher_id), season)
        except ValueError as problem:
            print(f"skipped pitcher {int(pitcher_id)}: {problem}")
            continue

        pitcher_rows.append({
            "role": "pitcher", "season": season, "hand": c.hand,
            "pitcher_id": int(pitcher_id), "name": c.name,
            "height_in": heights.get(int(pitcher_id), np.nan),
            "notes": _NOTE_SEPARATOR.join(c.notes),
            **_side_to_row(c.pitcher),
        })
        # The league side is identical for everyone of the same hand: save it once.
        if c.hand not in league_rows:
            listed = pd.Series(c.pitcher_ids).map(heights)
            league_rows[c.hand] = {
                "role": "league", "season": season, "hand": c.hand,
                "height_in": listed.mean(),
                "first_date": c.first_date, "last_date": c.last_date,
                "league_pitchers": c.league_pitchers,
                "fit_slope": c.fit_slope, "fit_intercept": c.fit_intercept,
                "zone_bottom": c.zone[0] if c.zone else np.nan,
                "zone_top": c.zone[1] if c.zone else np.nan,
                **_side_to_row(c.league),
            }

    return pd.DataFrame(list(league_rows.values()) + pitcher_rows)


def comparison_from_table(
    table: pd.DataFrame, pitcher_id: int, plate_height: float = DEFAULT_PLATE_HEIGHT,
) -> tuple[Comparison, float, float]:
    """Rebuild one pitcher's Comparison.

    Returns (comparison, pitcher height in inches, league average height).
    """
    mine = table[(table["role"] == "pitcher") & (table["pitcher_id"] == pitcher_id)]
    if mine.empty:
        raise ValueError("No four-seam fastball data for that pitcher this season.")
    mine = mine.iloc[0]
    league = table[(table["role"] == "league") & (table["hand"] == mine["hand"])].iloc[0]

    pitcher_side = _row_to_side(mine, plate_height)
    league_side = _row_to_side(league, plate_height)
    zone = (float(league["zone_bottom"]), float(league["zone_top"]))
    notes = str(mine["notes"]).split(_NOTE_SEPARATOR) if pd.notna(mine["notes"]) else []

    comparison = Comparison(
        name=str(mine["name"]),
        pitcher_id=int(pitcher_id),
        hand=str(mine["hand"]),
        season=int(mine["season"]),
        first_date=str(league["first_date"]),
        last_date=str(league["last_date"]),
        comparison_height=plate_height,
        pitcher=pitcher_side,
        league=league_side,
        league_pitchers=int(league["league_pitchers"]),
        pitcher_ids=(),          # only needed to average heights, already done at build time
        shares=vaa_model.explain_gap(pitcher_side.state, league_side.state, plate_height),
        fit_slope=float(league["fit_slope"]),
        fit_intercept=float(league["fit_intercept"]),
        notes=tuple(notes),
        zone=None if np.isnan(zone).any() else zone,
    )
    return comparison, float(mine["height_in"]), float(league["height_in"])


def _plain(text: str) -> str:
    """Lower-case with accents and punctuation removed: "José O'Neil Jr." -> "jose oneil jr"."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    letters = "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
    letters = letters.replace("-", " ").replace(",", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]", "", letters).split())


def find_pitchers(table: pd.DataFrame, query: str) -> pd.DataFrame:
    """Pitcher rows matching a typed name, most fastballs first.

    Every typed word must start some word of the name, so "skub", "joe ryan"
    and "Ryan, Joe" all work. An exact full name wins over partial matches.
    """
    pitchers = table[table["role"] == "pitcher"]
    words = _plain(query).split()
    if not words:
        return pitchers.iloc[0:0]

    names = pitchers["name"].map(_plain)
    exact = names.map(lambda n: sorted(n.split()) == sorted(words))
    if exact.any():
        found = pitchers[exact]
    else:
        found = pitchers[names.map(
            lambda n: all(any(part.startswith(w) for part in n.split()) for w in words)
        )]
    return found.sort_values("pitches", ascending=False)
