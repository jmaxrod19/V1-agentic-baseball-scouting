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

# Columns the calculation reads. Checked up front so a missing column fails
# with a readable message instead of a bare KeyError deep inside the math.
_REQUIRED_COLS: tuple[str, ...] = ("vy0", "vz0", "ay", "az")


def compute_vaa(df: pd.DataFrame, plate_y: float = PLATE_FRONT_Y) -> pd.Series:
    """Return each pitch's vertical approach angle in degrees (negative = descending).

    Rows missing any needed input (or with physically impossible values) come
    back as NaN rather than raising, so a few bad rows never sink a whole pull.
    The result has the same index as `df`, so it can be assigned straight back
    as a column. `df` itself is not modified.
    """
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_vaa: missing column(s) {missing}.")

    # Force plain floats so a stray string or None can't poison the math.
    vy0 = pd.to_numeric(df["vy0"], errors="coerce")
    vz0 = pd.to_numeric(df["vz0"], errors="coerce")
    ay = pd.to_numeric(df["ay"], errors="coerce")
    az = pd.to_numeric(df["az"], errors="coerce")

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

    # Step 4: the angle. atan2(opposite, adjacent) = atan2(vertical speed,
    # forward speed). vz_final is negative for a falling ball, so the angle comes
    # out negative. We negate vy_final only, because it is negative for every
    # pitch heading to the plate and we want the "forward" speed to be positive
    # (otherwise atan2 would return an angle near 180 degrees).
    vaa = np.degrees(np.arctan2(vz_final, -vy_final))

    return vaa.rename("vaa")
