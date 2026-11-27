"""
A full season of four-seam VAA results, small enough to ship with the web app.

The raw league file is hundreds of megabytes. The chart only needs a handful of
averages per pitcher, so we compute those once and keep them in a small CSV.
For each part of the zone ("all", "top", "middle", "bottom") there is one
"pitcher" row per pitcher and one "league" row per throwing hand, using only
the pitches that crossed there.

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
from src.vaa_model import DEFAULT_PLATE_HEIGHT, ZONE_PARTS, Comparison, Side

# Side attributes saved as they are, and the release-state numbers saved beside them.
_SIDE_FIELDS = ("pitches", "extension", "arm_angle", "velocity", "ivb",
                "observed_vaa", "plate_height", "vaa_aa")
_STATE_FIELDS = ("release_y", "release_z", "speed_y", "ay", "az")
_NOTE_SEPARATOR = " | "
# The pitch columns the build reads. A raw pull has over 100; copying them all
# for every pitcher made the build several times slower.
_BUILD_COLUMNS = (
    "game_year", "game_type", "game_date", "pitch_type", "pitcher", "player_name", "p_throws",
    "release_speed", "release_pos_y", "release_pos_z", "release_extension", "arm_angle",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_z", "plate_x", "plate_z", "sz_top", "sz_bot",
)


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


def build_season_table(data: pd.DataFrame, season: int, heights: dict[int, int],
                       position_players: set[int] = frozenset()) -> pd.DataFrame:
    """Run the comparison for every four-seam thrower and keep one row each.

    data     league-wide pitch rows (loaders.load_statcast_csv).
    heights  {mlbam_id: listed height in inches}, for the figurines.
    position_players  ids to leave out, of both the pitcher rows and the
             league average (loaders.pull_position_players).
    """
    # Filter once here so each comparison searches a much smaller frame.
    keep = (data["pitch_type"] == "FF") & (data["game_year"] == season)
    if "game_type" in data.columns:
        keep &= data["game_type"] == "R"
    keep &= ~data["pitcher"].isin(position_players)
    fastballs = data.loc[keep, [c for c in _BUILD_COLUMNS if c in data.columns]]
    if fastballs.empty:
        raise ValueError(f"No regular-season four-seam fastballs for season {season}.")

    mix = vaa_model.pitch_mix(data, season)          # needs every pitch type
    located = vaa_model.zone_part(fastballs)
    rows = []
    for part in ZONE_PARTS:
        # The league benchmark comes from the same part of the zone as the pitcher.
        subset = fastballs if part == "all" else fastballs[located == part]
        rows += _rows_for(subset, season, heights, part)
        print(f"  {part}: done", flush=True)
    table = pd.DataFrame(rows)
    for column in mix.columns:      # league rows have no pitcher id, so stay blank
        table[column] = table["pitcher_id"].map(mix[column])
    return table


def _rows_for(fastballs: pd.DataFrame, season: int, heights: dict[int, int],
              part: str) -> list[dict]:
    """League and pitcher rows for one set of pitches."""
    pitcher_rows, league_rows = [], {}
    for pitcher_id in sorted(fastballs["pitcher"].dropna().unique()):
        try:
            c = vaa_model.compare_fastball(fastballs, int(pitcher_id), season)
        except ValueError as problem:
            print(f"skipped pitcher {int(pitcher_id)} ({part}): {problem}")
            continue

        pitcher_rows.append({
            "role": "pitcher", "part": part, "season": season, "hand": c.hand,
            "pitcher_id": int(pitcher_id), "name": c.name,
            "height_in": heights.get(int(pitcher_id), np.nan),
            "notes": _NOTE_SEPARATOR.join(c.notes),
            **_side_to_row(c.pitcher),
        })
        # The league side is identical for everyone of the same hand: save it once.
        if c.hand not in league_rows:
            listed = pd.Series(c.pitcher_ids).map(heights)
            league_rows[c.hand] = {
                "role": "league", "part": part, "season": season, "hand": c.hand,
                "height_in": listed.mean(),
                "first_date": c.first_date, "last_date": c.last_date,
                "league_pitchers": c.league_pitchers,
                "fit_slope": c.fit_slope, "fit_intercept": c.fit_intercept,
                "zone_bottom": c.zone[0] if c.zone else np.nan,
                "zone_top": c.zone[1] if c.zone else np.nan,
                **_side_to_row(c.league),
            }

    return list(league_rows.values()) + pitcher_rows


def comparison_from_table(
    table: pd.DataFrame, pitcher_id: int, part: str = "all",
    plate_height: float | None = None,
) -> tuple[Comparison, float, float]:
    """Rebuild one pitcher's Comparison for one part of the zone.

    The modeled paths cross the plate at `plate_height`. Left empty, that is
    2.75 ft for all pitches, or the league's typical height within the third.
    Returns (comparison, pitcher height in inches, league average height).
    """
    if part not in ZONE_PARTS:
        raise ValueError(f"Unknown part of the zone: {part!r}.")
    here = table[table["part"] == part]
    mine = here[(here["role"] == "pitcher") & (here["pitcher_id"] == pitcher_id)]
    if mine.empty:
        if part == "all":
            raise ValueError("No four-seam fastball data for that pitcher this season.")
        raise ValueError(f"No four-seam fastballs from this pitcher in the {ZONE_PARTS[part]}.")
    mine = mine.iloc[0]
    league = here[(here["role"] == "league") & (here["hand"] == mine["hand"])].iloc[0]
    if plate_height is None:
        plate_height = (DEFAULT_PLATE_HEIGHT if part == "all"
                        else round(float(league["plate_height"]), 1))

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
        zone_part=part,
        ff_pct=float(mine.get("ff_pct", np.nan)),
        main_pitch=str(mine["main_pitch"]) if pd.notna(mine.get("main_pitch")) else "",
        main_pct=float(mine.get("main_pct", np.nan)),
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
    pitchers = table[(table["role"] == "pitcher") & (table["part"] == "all")]
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
