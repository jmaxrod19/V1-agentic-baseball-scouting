"""
The numbers behind the four-seam fastball VAA chart. No drawing happens here.

Terms
    observed VAA   the average of each real pitch's arrival angle.
    modeled VAA    the angle of ONE average pitch, re-aimed so it crosses the
                   front of the plate at a chosen height (default 2.75 ft).
    expected VAA   what a league pitch (same hand, same season) typically has
                   at a given plate height: a straight-line fit.
    VAA AA         observed VAA minus expected VAA at the same height.
                   Positive = flatter than expected. Negative = steeper.

Physics: Statcast fits every pitch with constant acceleration in each direction.
That acceleration is what was measured, so gravity, drag and spin movement are
all already inside ax/ay/az. Axes: y = feet from the plate toward the mound,
z = height. A pitch heading home has negative vy, and a falling one negative vz.

"Release state" is the five numbers that fully set a pitch's side view:
    release_y   distance from the plate at release (ft)
    release_z   height at release (ft)
    speed_y     forward velocity at release (ft/s, negative)
    ay, az      accelerations (ft/s^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from src import vaa

DEFAULT_PLATE_HEIGHT = 2.75      # ft, at the front of the plate
CLASS_THRESHOLD = 0.3            # degrees of VAA AA that counts as different from average
SMALL_SAMPLE = 50                # fewer fastballs than this gets a warning
MIN_LEAGUE_PITCHERS = 20         # below this the data is not a league benchmark
_RUBBER_Y = 60.5
_STATE_COLS = ("release_y", "release_z", "speed_y", "ay", "az")

# Which parts of the release state belong to which explanation.
FACTORS: dict[str, tuple[str, ...]] = {
    "release_height": ("release_z",),
    "extension": ("release_y",),
    "velocity": ("speed_y", "ay"),
    "ride": ("az",),             # vertical acceleration: gravity plus spin and drag lift
}


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column, or all-NaN if the data does not have it."""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def release_states(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per pitch: release state, observed height at the plate front, observed VAA.

    Returns (frame, notes). The frame keeps only pitches with complete data.
    `notes` lists anything that had to be ESTIMATED instead of measured; an
    empty list means nothing was. Missing velocity or acceleration columns are
    never filled in: that raises ValueError.
    """
    missing = [c for c in ("vy0", "vz0", "ay", "az") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Pitch-tracking column(s) {missing} are missing, so no flight path can be "
            "computed. Nothing is filled in with made-up values."
        )
    notes: list[str] = []
    vy0, vz0 = _column(df, "vy0"), _column(df, "vz0")
    ay, az = _column(df, "ay"), _column(df, "az")

    # Release distance: measured, else worked out from extension.
    release_y = _column(df, "release_pos_y")
    from_extension = _RUBBER_Y - _column(df, "release_extension")
    if (release_y.isna() & from_extension.notna()).any():
        notes.append("some release distances derived from extension")
    release_y = release_y.fillna(from_extension)

    # Times are measured from y = 50 ft (where vy0 and vz0 are defined).
    # Speed at release: v^2 = vy0^2 + 2 * ay * (change in y). Release is ~4 ft
    # farther back than y = 50, where the ball was slightly faster.
    with np.errstate(invalid="ignore"):
        speed_y = -np.sqrt(vy0**2 + 2 * ay * (release_y - vaa.STATCAST_START_Y))
    release_time = (speed_y - vy0) / ay                # negative: before y = 50
    release_vz = vz0 + az * release_time

    # Release height: measured, else rebuilt backwards from the plate location.
    release_z = _column(df, "release_pos_z")
    lost = release_z.isna()
    if lost.any() and "plate_z" in df.columns:
        rebuilt = pd.Series(np.nan, index=df.index, dtype=float)
        paths = vaa.flight_paths(
            df.loc[lost].assign(
                release_pos_y=release_y[lost], vx0=0.0, ax=0.0, plate_x=0.0,   # side view only
            ),
            n_points=2,
        )
        rebuilt.loc[lost] = paths[:, 0, 2]
        if rebuilt.notna().any():
            notes.append("some release heights estimated from the trajectory fit")
        release_z = release_z.fillna(rebuilt)

    # Where the pitch really crossed the FRONT of the plate, flown from release.
    arrival = vaa.compute_vaa(df)
    state = pd.DataFrame(
        {"release_y": release_y, "release_z": release_z, "speed_y": speed_y, "ay": ay, "az": az}
    )
    flight_time, _ = _time_to_plate(state)
    plate_height = release_z + release_vz * flight_time + 0.5 * az * flight_time**2

    out = state.assign(plate_height=plate_height, vaa=arrival)
    return out.replace([np.inf, -np.inf], np.nan).dropna(), notes


def _time_to_plate(state):
    """Seconds from release to the plate front, and forward velocity on arrival.

    Works for one state (floats) or many (columns).
    """
    distance = state["release_y"] - vaa.PLATE_FRONT_Y
    with np.errstate(invalid="ignore"):
        final_speed = -np.sqrt(state["speed_y"] ** 2 - 2 * state["ay"] * distance)
    # distance = average velocity * time
    return 2 * distance / -(state["speed_y"] + final_speed), final_speed


@dataclass(frozen=True)
class Trajectory:
    """One modeled pitch, release to the front of the plate."""
    y: np.ndarray            # feet from the plate, at each point
    z: np.ndarray            # height in feet, at each point
    vy_final: float
    vz_final: float

    @property
    def vaa(self) -> float:
        """atan2(vz_final, |vy_final|) in degrees."""
        return float(np.degrees(np.arctan2(self.vz_final, abs(self.vy_final))))


def model_trajectory(state: dict, plate_height: float = DEFAULT_PLATE_HEIGHT,
                     n_points: int = 80) -> Trajectory:
    """Fly the pitch from its release point so it crosses the plate at `plate_height`.

    Release point, speed and accelerations stay as measured. Only the launch
    direction (vertical speed at release) is solved for:
        plate_height = release_z + vz * T + 0.5 * az * T^2
    so moving the target really does recompute the whole path.
    """
    flight_time, vy_final = _time_to_plate(state)
    az = state["az"]
    vz = (plate_height - state["release_z"] - 0.5 * az * flight_time**2) / flight_time

    t = np.linspace(0.0, flight_time, n_points)
    y = state["release_y"] + state["speed_y"] * t + 0.5 * state["ay"] * t**2
    z = state["release_z"] + vz * t + 0.5 * az * t**2
    return Trajectory(y=y, z=z, vy_final=float(vy_final), vz_final=float(vz + az * flight_time))


def explain_gap(pitcher: dict, league: dict,
                plate_height: float = DEFAULT_PLATE_HEIGHT) -> dict[str, float]:
    """Split the modeled VAA difference into one share per factor, in degrees.

    The idea: start with the league pitch and swap in the pitcher's values one
    factor at a time, noting how much the angle moves. The answer depends a
    little on the order, so we average over every order (the "Shapley value").
    The shares then add up EXACTLY to modeled pitcher VAA minus modeled league VAA.
    """
    names = list(FACTORS)
    count = len(names)

    def angle(swapped: tuple[str, ...]) -> float:
        state = dict(league)
        for factor in swapped:
            for key in FACTORS[factor]:
                state[key] = pitcher[key]
        return model_trajectory(state, plate_height, n_points=2).vaa

    shares = {}
    for name in names:
        others = [n for n in names if n != name]
        total = 0.0
        for size in range(count):
            # How often a group of this size comes before `name` across all orders.
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            for group in combinations(others, size):
                total += weight * (angle(group + (name,)) - angle(group))
        shares[name] = total
    return shares


def classify(vaa_above_expected: float) -> str:
    """Words for a VAA AA value (positive = flatter).

    Judged on the value rounded to 0.1, the way it is displayed, so the words
    can never disagree with the number printed next to them.
    """
    vaa_above_expected = round(vaa_above_expected, 1)
    if vaa_above_expected >= CLASS_THRESHOLD:
        return "Flatter than average"
    if vaa_above_expected <= -CLASS_THRESHOLD:
        return "Steeper than average"
    return "About average"


@dataclass(frozen=True)
class Side:
    """Everything shown for one side of the comparison (pitcher or league)."""
    pitches: int
    release_height: float
    extension: float
    arm_angle: float          # NaN when the data has none
    velocity: float           # mph
    ivb: float                # induced vertical break, inches
    observed_vaa: float
    plate_height: float       # observed average, front of plate
    vaa_aa: float             # 0 for the league by construction
    state: dict               # average release state
    modeled: Trajectory       # re-aimed to the comparison height


@dataclass(frozen=True)
class Comparison:
    """One pitcher's four-seamer against the league, ready to draw."""
    name: str
    pitcher_id: int
    hand: str
    season: int | None
    first_date: str
    last_date: str
    comparison_height: float
    pitcher: Side
    league: Side
    league_pitchers: int
    pitcher_ids: tuple[int, ...]       # league pitchers, for average listed height
    shares: dict[str, float]           # explain_gap result
    fit_slope: float                   # expected VAA = intercept + slope * height
    fit_intercept: float
    notes: tuple[str, ...]             # anything estimated or worth warning about
    zone: tuple[float, float] | None   # average strike-zone bottom and top (ft)

    @property
    def modeled_gap(self) -> float:
        return self.pitcher.modeled.vaa - self.league.modeled.vaa

    @property
    def classification(self) -> str:
        return classify(self.pitcher.vaa_aa)


def _side(rows: pd.DataFrame, states: pd.DataFrame, expected: np.ndarray,
          plate_height: float) -> Side:
    """Average one group of pitches. `rows` and `states` cover the same pitches."""
    state = {c: float(states[c].mean()) for c in _STATE_COLS}
    return Side(
        pitches=len(states),
        release_height=state["release_z"],
        extension=float(_column(rows, "release_extension").mean()),
        arm_angle=float(_column(rows, "arm_angle").mean()),
        velocity=float(_column(rows, "release_speed").mean()),
        ivb=float(_column(rows, "pfx_z").mean() * 12),          # pfx_z is in feet
        observed_vaa=float(states["vaa"].mean()),
        plate_height=float(states["plate_height"].mean()),
        vaa_aa=float((states["vaa"].to_numpy() - expected).mean()),
        state=state,
        modeled=model_trajectory(state, plate_height),
    )


def compare_fastball(data: pd.DataFrame, pitcher_id: int, season: int | None = None,
                     plate_height: float = DEFAULT_PLATE_HEIGHT) -> Comparison:
    """Compare one pitcher's four-seamer with the same-handed league that season.

    data        LEAGUE-WIDE pitch-level Statcast rows.
    pitcher_id  MLBAM id (the `pitcher` column).
    season      year (`game_year`); defaults to the only/latest year in the data.
    Only four-seam fastballs (FF) and, when marked, regular-season games are used.
    """
    df = data[data["pitch_type"] == "FF"]
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    if "game_year" in df.columns:
        if season is None:
            season = int(df["game_year"].max())
        years = sorted(int(y) for y in df["game_year"].dropna().unique())
        if season not in years:
            raise ValueError(
                f"This data covers season(s) {years}, not {season}. Pull that season "
                "league-wide (loaders.pull_statcast) and pass the file with --csv."
            )
        df = df[df["game_year"] == season]
    elif season is not None:
        raise ValueError("This data has no game_year column, so a season cannot be selected.")
    df = df.reset_index(drop=True)

    mine = df[df["pitcher"] == pitcher_id]
    if mine.empty:
        raise ValueError(f"No regular-season four-seam fastballs for pitcher {pitcher_id}"
                         f" in season {season}.")
    hand = str(mine["p_throws"].mode().iloc[0])
    league = df[df["p_throws"] == hand]
    if league["pitcher"].nunique() < MIN_LEAGUE_PITCHERS:
        raise ValueError(
            f"Only {league['pitcher'].nunique()} {hand}-handed pitchers in this data. "
            "A league benchmark needs league-wide data."
        )

    league_states, league_notes = release_states(league)
    mine_states, mine_notes = release_states(mine[mine["p_throws"] == hand])
    if mine_states.empty:
        raise ValueError("This pitcher has no four-seamers with complete tracking data.")

    # Expected VAA: a straight line through (plate height, VAA) for the league.
    slope, intercept = np.polyfit(league_states["plate_height"], league_states["vaa"], 1)

    def expected(states: pd.DataFrame) -> np.ndarray:
        return intercept + slope * states["plate_height"].to_numpy()

    pitcher_side = _side(mine.loc[mine_states.index], mine_states,
                         expected(mine_states), plate_height)
    league_side = _side(league.loc[league_states.index], league_states,
                        expected(league_states), plate_height)

    notes = list(dict.fromkeys(mine_notes + league_notes))      # no duplicates, keep order
    dropped = int((mine["p_throws"] == hand).sum()) - len(mine_states)
    if dropped:
        notes.append(f"{dropped} pitch(es) without complete tracking left out")
    if len(mine_states) < SMALL_SAMPLE:
        notes.append(f"small sample ({len(mine_states)} pitches)")

    dates = pd.to_datetime(df["game_date"]) if "game_date" in df.columns else pd.Series(dtype="datetime64[ns]")
    name = str(mine["player_name"].iloc[0]) if "player_name" in mine.columns else str(pitcher_id)
    last, _, first = name.partition(", ")                # Statcast gives "Last, First"

    zone = (float(_column(league, "sz_bot").mean()), float(_column(league, "sz_top").mean()))

    return Comparison(
        zone=None if np.isnan(zone).any() else zone,
        name=f"{first} {last}".strip(),
        pitcher_id=int(pitcher_id),
        hand=hand,
        season=season,
        first_date="" if dates.empty else f"{dates.min():%b} {dates.min().day}",
        last_date="" if dates.empty else f"{dates.max():%b} {dates.max().day}",
        comparison_height=plate_height,
        pitcher=pitcher_side,
        league=league_side,
        league_pitchers=int(league.loc[league_states.index, "pitcher"].nunique()),
        pitcher_ids=tuple(int(i) for i in league["pitcher"].dropna().unique()),
        shares=explain_gap(pitcher_side.state, league_side.state, plate_height),
        fit_slope=float(slope),
        fit_intercept=float(intercept),
        notes=tuple(notes),
    )
