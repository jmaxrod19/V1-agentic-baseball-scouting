"""
VAA module: vertical approach angle (VAA) from Statcast trajectory columns.

VAA is the angle, measured in the vertical plane, at which a pitch is
travelling as it reaches the plate. We report it the way it is commonly read
and talked about: as a NEGATIVE angle, because the pitch is heading downward.
0 degrees is a pitch flying perfectly level, and the MORE negative the number,
the more steeply the pitch is dropping (-6 is steeper than -4; a flat "rising"
fastball sits closer to zero).

Public functions:
    compute_vaa(df, plate_y=PLATE_FRONT_Y) -> pd.Series of degrees, row-aligned
    flight_paths(df, n_points=40)          -> (n_pitches, n_points, 3) array of
                                              x/y/z points, release to plate
    average_pitch(df)                      -> one-row frame: the "typical" pitch

The math is plain constant-acceleration physics (the same equations as a ball
thrown in a high-school physics problem). Statcast fits each pitch with a
starting position, velocity and acceleration measured at y = 50 ft from the
plate, so we only need `vy0`, `vz0`, `ay`, `az` to find the velocity when the
ball gets to the plate. Axes: y points from the plate toward the pitcher, so a
pitch heading to the plate has NEGATIVE vy; z is up, so a falling ball has
NEGATIVE vz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Statcast's trajectory fit (vx0/vy0/vz0, ax/ay/az) starts at y = 50 ft.
STATCAST_START_Y: float = 50.0

# The front edge of home plate is 17 inches deep. Measuring the angle here (not
# at the back of the plate) is the common convention for VAA.
PLATE_FRONT_Y: float = 17 / 12

# Where plate_x / plate_z are recorded. Savant's CSV docs: the FRONT of the plate
# through 2025, the MIDDLE of the plate (8.5 in from the back tip) from 2026 on.
PLATE_MIDDLE_Y: float = 8.5 / 12


def plate_reference_y(df: pd.DataFrame) -> pd.Series:
    """For each row, the distance at which its plate_x / plate_z were measured."""
    year = _num(df, "game_year") if "game_year" in df.columns else pd.Series(np.nan, index=df.index)
    # NaN >= 2026 is False, so data without a year keeps the older front-of-plate rule.
    return pd.Series(np.where(year >= 2026, PLATE_MIDDLE_Y, PLATE_FRONT_Y), index=df.index)


# Columns the calculation reads. Checked up front so a missing column fails
# with a readable message instead of a bare KeyError deep inside the math.
_REQUIRED_COLS: tuple[str, ...] = ("vy0", "vz0", "ay", "az")


def _check_columns(df: pd.DataFrame, needed: tuple[str, ...], who: str) -> None:
    """Raise one readable error naming every missing column."""
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{who}: missing column(s) {missing}.")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as plain floats; a stray string or None becomes NaN."""
    return pd.to_numeric(df[col], errors="coerce")


def _arrival_state(df: pd.DataFrame, plate_y: float) -> pd.DataFrame:
    """Flight time and speeds at the moment each pitch reaches `plate_y`.

    Returns a frame (same index as `df`) with columns:
        flight_time  seconds from y = 50 ft to the plate
        vy_final     speed toward the plate on arrival (negative)
        vz_final     vertical speed on arrival (negative = falling)
    Shared by compute_vaa (which only needs the angle) and flight_paths (which
    needs the whole trajectory), so the physics lives in exactly one place.
    """
    vy0 = _num(df, "vy0")
    vz0 = _num(df, "vz0")
    ay = _num(df, "ay")
    az = _num(df, "az")

    # How far the ball travels toward the plate, in feet (about 48.6 ft).
    distance = STATCAST_START_Y - plate_y

    # Step 1: speed toward the plate when the ball arrives.
    # Constant-acceleration identity: vy_final^2 = vy0^2 + 2 * ay * (change in y).
    # The ball moves in the -y direction, so the change in y is -distance.
    vy_final_sq = vy0**2 - 2 * ay * distance

    # Keep only rows that make physical sense: the ball is heading toward the
    # plate (vy0 < 0) and the square is positive. Everything else becomes NaN
    # here, which also keeps np.sqrt below from warning about negatives.
    valid = (vy0 < 0) & (vy_final_sq > 0)
    vy_final = -np.sqrt(vy_final_sq.where(valid))  # negative: still heading to plate

    # Step 2: flight time. With constant acceleration, distance = average
    # velocity * time, so time = (change in y) / ((vy0 + vy_final) / 2).
    # Using the average velocity (instead of dividing by ay) stays correct even
    # when ay is exactly zero.
    flight_time = (-distance * 2) / (vy0 + vy_final)

    # Step 3: vertical speed on arrival: vz_final = vz0 + az * time.
    vz_final = vz0 + az * flight_time

    return pd.DataFrame(
        {"flight_time": flight_time, "vy_final": vy_final, "vz_final": vz_final}
    )


def compute_vaa(df: pd.DataFrame, plate_y: float = PLATE_FRONT_Y) -> pd.Series:
    """Return each pitch's vertical approach angle in degrees (negative = descending).

    Rows missing any needed input (or with physically impossible values) come
    back as NaN rather than raising, so a few bad rows never sink a whole pull.
    The result has the same index as `df`, so it can be assigned straight back
    as a column. `df` itself is not modified.
    """
    _check_columns(df, _REQUIRED_COLS, "compute_vaa")
    arrival = _arrival_state(df, plate_y)

    # Step 4: the angle. atan2(opposite, adjacent) = atan2(vertical speed,
    # forward speed). vz_final is negative for a falling ball, so the angle comes
    # out negative. We negate vy_final only, because it is negative for every
    # pitch heading to the plate and we want the "forward" speed to be positive
    # (otherwise atan2 would return an angle near 180 degrees).
    vaa = np.degrees(np.arctan2(arrival["vz_final"], -arrival["vy_final"]))

    return vaa.rename("vaa")


_PATH_COLS: tuple[str, ...] = (
    "vx0", "vy0", "vz0", "ax", "ay", "az", "plate_x", "plate_z", "release_pos_y",
)


def flight_paths(df: pd.DataFrame, n_points: int = 40) -> np.ndarray:
    """Trace every pitch from its release point to the front of the plate.

    Returns an array shaped (n_pitches, n_points, 3). For pitch i, point j is
    (x, y, z) in feet: x = left/right (catcher's view, + is the catcher's right),
    y = distance from the plate (about 54 ft at release, 17/12 ft at the plate),
    z = height. Rows with missing or impossible inputs are all-NaN, so row i
    still lines up with row i of `df`.

    How it works: we know exactly where Statcast says each pitch crossed the
    plate (plate_x, plate_z; see plate_reference_y for the depth), and we know
    its speed and acceleration there. With
    constant acceleration, position at any other moment is
        position = plate position + velocity_at_plate * s + 0.5 * accel * s^2
    where s is the (negative) time relative to arrival. So we anchor the path at
    the plate and walk it backwards in time to the release point.
    """
    _check_columns(df, _PATH_COLS, "flight_paths")
    arrival = _arrival_state(df, PLATE_FRONT_Y)
    reference = _arrival_state(df, plate_reference_y(df))   # where plate_x/z were measured

    vx0, vy0 = _num(df, "vx0"), _num(df, "vy0")
    ax, ay, az = _num(df, "ax"), _num(df, "ay"), _num(df, "az")
    flight_time = arrival["flight_time"]
    reference_time = reference["flight_time"]

    # Sideways and vertical velocity at the moment plate_x / plate_z were recorded.
    vx_final = vx0 + ax * reference_time
    vz_final = reference["vz_final"]

    # When was release? The y-position follows y(t) = 50 + vy0*t + 0.5*ay*t^2
    # (t = 0 at y = 50 ft), so release is the t where y(t) = release_pos_y.
    # Solve that quadratic. Of its two roots we want the one closest to zero.
    # errstate: a bad row (negative under the square root, or ay == 0) should
    # quietly become NaN, not print a warning for every such row.
    c = STATCAST_START_Y - _num(df, "release_pos_y")
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(vy0**2 - 4 * (0.5 * ay) * c)
        t_a = (-vy0 + root) / ay
        t_b = (-vy0 - root) / ay
    release_time = t_a.where(t_a.abs() < t_b.abs(), t_b)

    # A grid of moments for every pitch: row i runs from its release time to
    # its arrival time in n_points even steps. Shape (n_pitches, n_points).
    fraction = np.linspace(0.0, 1.0, n_points)
    span = (flight_time - release_time).to_numpy()[:, None]
    t = release_time.to_numpy()[:, None] + fraction[None, :] * span

    # Time relative to the recorded plate location: the "s" in the docstring formula.
    s = t - reference_time.to_numpy()[:, None]

    x = _num(df, "plate_x").to_numpy()[:, None] + vx_final.to_numpy()[:, None] * s \
        + 0.5 * ax.to_numpy()[:, None] * s**2
    z = _num(df, "plate_z").to_numpy()[:, None] + vz_final.to_numpy()[:, None] * s \
        + 0.5 * az.to_numpy()[:, None] * s**2
    y = STATCAST_START_Y + vy0.to_numpy()[:, None] * t + 0.5 * ay.to_numpy()[:, None] * t**2

    paths = np.stack([x, y, z], axis=2)

    # A pitch missing only one input (say vz0) would still get valid x and y but
    # a NaN z, which makes a confusing half-drawn line. Make the whole row NaN
    # so "this pitch can't be traced" is one clear signal for the chart code.
    unusable = np.isnan(paths).any(axis=(1, 2))
    paths[unusable] = np.nan
    return paths


def average_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse many pitches into ONE typical pitch (a one-row frame).

    Averages every input flight_paths needs (speeds, accelerations, release
    distance and plate location), skipping missing values. The result can be fed
    straight to flight_paths or compute_vaa. Because the path model is built
    from these numbers, the "average pitch" is a real, physically consistent
    trajectory that passes through the average plate location.
    """
    _check_columns(df, _PATH_COLS, "average_pitch")
    row = {c: _num(df, c).mean() for c in _PATH_COLS}
    if "game_year" in df.columns:
        row["game_year"] = _num(df, "game_year").max()   # decides the measuring depth
    return pd.DataFrame([row])
