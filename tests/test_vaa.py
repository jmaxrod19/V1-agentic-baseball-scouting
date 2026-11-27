"""Tests for src/vaa.py: vertical approach angle from Statcast trajectory columns.

Same technique as the other test files: hand-build tiny inputs whose answer we
can work out separately, then assert the code agrees.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import vaa


def _row(vy0=-130.0, vz0=-5.0, ay=0.0, az=-20.0) -> dict:
    """One pitch's trajectory inputs, with easy-to-reason-about defaults."""
    return {"vy0": vy0, "vz0": vz0, "ay": ay, "az": az}


def test_no_drag_matches_hand_calculation():
    # No drag (ay = 0) means the forward speed never changes: 130 ft/s the
    # whole way. Flight time is just distance / speed.
    distance = 50.0 - vaa.PLATE_FRONT_Y
    time = distance / 130.0
    vz_final = -5.0 + (-20.0) * time          # falling faster as it goes
    expected = math.degrees(math.atan(vz_final / 130.0))   # negative: heading down

    out = vaa.compute_vaa(pd.DataFrame([_row(ay=0.0)]))
    assert out.iloc[0] == pytest.approx(expected)
    assert -6.0 < out.iloc[0] < -5.0          # sanity: a realistic fastball angle


def test_with_drag_matches_a_different_formula():
    # Cross-check: work out flight time a DIFFERENT way than the code does
    # (time = change in speed / acceleration) and confirm the answers agree.
    ay = 25.0                                  # drag slows the ball down
    distance = 50.0 - vaa.PLATE_FRONT_Y
    vy_final = -math.sqrt(130.0**2 - 2 * ay * distance)
    time = (vy_final - (-130.0)) / ay
    vz_final = -5.0 + (-20.0) * time
    expected = math.degrees(math.atan2(vz_final, -vy_final))

    out = vaa.compute_vaa(pd.DataFrame([_row(ay=ay)]))
    assert out.iloc[0] == pytest.approx(expected)


def test_more_downward_acceleration_means_steeper():
    # A fastball with less "rise" (more negative az) must come in steeper, and
    # steeper means MORE negative, so the angles shrink as az gets more negative.
    df = pd.DataFrame([_row(az=-10.0), _row(az=-20.0), _row(az=-30.0)])
    out = vaa.compute_vaa(df)
    assert out.is_monotonic_decreasing
    assert (out < 0).all()


def test_plate_y_changes_the_answer():
    # Measuring further back (y = 5 ft) gives the ball less time to fall than
    # measuring at the front of the plate, so the angle is closer to zero.
    df = pd.DataFrame([_row()])
    at_plate = vaa.compute_vaa(df).iloc[0]
    earlier = vaa.compute_vaa(df, plate_y=5.0).iloc[0]
    assert at_plate < earlier < 0


def test_missing_column_raises_readable_error():
    df = pd.DataFrame([_row()]).drop(columns=["az"])
    with pytest.raises(ValueError, match="az"):
        vaa.compute_vaa(df)


def test_bad_rows_become_nan_and_good_rows_survive():
    df = pd.DataFrame(
        [
            _row(),                            # good
            _row(vz0=np.nan),                  # missing input
            _row(vy0=+130.0),                  # heading AWAY from the plate
            _row(ay=1e6),                      # impossible: speed^2 goes negative
        ]
    )
    out = vaa.compute_vaa(df)
    assert not np.isnan(out.iloc[0])
    assert out.iloc[1:].isna().all()


def test_output_is_row_aligned_and_input_untouched():
    df = pd.DataFrame([_row(), _row(az=-25.0)], index=[10, 20])
    before = df.copy()
    out = vaa.compute_vaa(df)
    assert list(out.index) == [10, 20]         # same index, so it assigns cleanly
    assert out.name == "vaa"
    pd.testing.assert_frame_equal(df, before)  # no column added to the caller's frame


def test_string_values_are_coerced_not_crashing():
    # A stray string in a numeric column (a partial row) becomes NaN, not an error.
    df = pd.DataFrame([_row(), _row()])
    df["vz0"] = df["vz0"].astype(object)
    df.loc[1, "vz0"] = "oops"
    out = vaa.compute_vaa(df)
    assert not np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])


# ---------------------------------------------------------------------------
# flight_paths: the full release-to-plate trajectory used by the 3D chart.
# ---------------------------------------------------------------------------

def _path_row(**overrides) -> dict:
    """A pitch with every column flight_paths needs (a realistic fastball)."""
    row = {
        "vx0": 6.0, "vy0": -130.0, "vz0": -5.0,
        "ax": -10.0, "ay": 25.0, "az": -20.0,
        "plate_x": 0.3, "plate_z": 2.6, "release_pos_y": 54.0,
    }
    row.update(overrides)
    return row


def test_flight_paths_shape_and_endpoints():
    df = pd.DataFrame([_path_row(), _path_row(plate_x=-0.5, plate_z=3.1)])
    paths = vaa.flight_paths(df, n_points=25)
    assert paths.shape == (2, 25, 3)
    # Last point is exactly where Statcast says the pitch crossed the plate.
    np.testing.assert_allclose(paths[0, -1], [0.3, vaa.PLATE_FRONT_Y, 2.6], atol=1e-9)
    np.testing.assert_allclose(paths[1, -1], [-0.5, vaa.PLATE_FRONT_Y, 3.1], atol=1e-9)
    # First point starts at the release distance.
    np.testing.assert_allclose(paths[:, 0, 1], 54.0, atol=1e-9)


def test_flight_paths_slope_at_plate_matches_vaa():
    # Two independent routes to the same angle must agree: the VAA formula, and
    # the slope of the last tiny step of the traced path (rise / run).
    df = pd.DataFrame([_path_row(), _path_row(az=-28.0, plate_z=1.8)])
    paths = vaa.flight_paths(df, n_points=4000)
    dz = paths[:, -1, 2] - paths[:, -2, 2]
    dy = paths[:, -2, 1] - paths[:, -1, 1]       # y shrinks toward the plate
    slope_deg = np.degrees(np.arctan2(dz, dy))
    np.testing.assert_allclose(slope_deg, vaa.compute_vaa(df).to_numpy(), atol=0.01)


def test_flight_paths_bad_rows_are_nan_and_stay_aligned():
    df = pd.DataFrame([_path_row(), _path_row(vz0=np.nan), _path_row()])
    paths = vaa.flight_paths(df, n_points=10)
    assert np.isfinite(paths[0]).all() and np.isfinite(paths[2]).all()
    assert np.isnan(paths[1]).all()


def test_flight_paths_missing_column_raises():
    df = pd.DataFrame([_path_row()]).drop(columns=["release_pos_y"])
    with pytest.raises(ValueError, match="release_pos_y"):
        vaa.flight_paths(df)


def test_average_pitch_is_one_row_of_column_means():
    df = pd.DataFrame([_path_row(vy0=-128.0, plate_z=2.0), _path_row(vy0=-132.0, plate_z=3.0),
                       _path_row(vy0=np.nan, plate_z=2.5)])
    avg = vaa.average_pitch(df)
    assert len(avg) == 1
    assert avg.loc[0, "vy0"] == pytest.approx(-130.0)    # NaN skipped
    assert avg.loc[0, "plate_z"] == pytest.approx(2.5)


def test_average_pitch_angle_is_close_to_average_of_angles():
    # For pitches that are similar, the angle of the "average pitch" and the
    # average of the individual angles should agree closely.
    rng = np.random.default_rng(2)
    df = pd.DataFrame([_path_row(vy0=-130 + rng.normal(0, 2), vz0=-5 + rng.normal(0, 1),
                                 az=-20 + rng.normal(0, 1.5)) for _ in range(200)])
    assert vaa.compute_vaa(vaa.average_pitch(df)).iloc[0] == pytest.approx(
        vaa.compute_vaa(df).mean(), abs=0.05)
