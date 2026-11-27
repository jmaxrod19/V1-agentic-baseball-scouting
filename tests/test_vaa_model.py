"""Tests for src/vaa_model.py: the physics and statistics behind the VAA chart.

The fake pitches below are built from the same constant-acceleration equations
Statcast uses, so every one of them is physically self-consistent.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import vaa, vaa_model as m


def _state(**kw) -> dict:
    base = {"release_y": 54.0, "release_z": 5.8, "speed_y": -139.0, "ay": 30.0, "az": -15.0}
    base.update(kw)
    return base


def _pitch(rng, pitcher: int, hand: str = "R", release_z: float = 5.8, year: int = 2026) -> dict:
    """One fastball. vy0/vz0 are defined at y = 50 ft, as in Statcast."""
    vy0 = rng.normal(-138, 2)
    ay, az = rng.normal(30, 1), rng.normal(-15, 1.5)
    release_y = rng.normal(54, 0.3)
    t_release = (-math.sqrt(vy0**2 + 2 * ay * (release_y - 50)) - vy0) / ay
    reference_y = vaa.PLATE_MIDDLE_Y if year >= 2026 else vaa.PLATE_FRONT_Y
    t_plate = (-math.sqrt(vy0**2 - 2 * ay * (50 - reference_y)) - vy0) / ay
    z_release = rng.normal(release_z, 0.1)
    # Pitchers AIM: pick where the ball should cross, then solve for the vz0 that gets it there.
    plate_z = rng.normal(2.6, 0.6)
    drop = 0.5 * az * (t_plate**2 - t_release**2)
    vz0 = (plate_z - z_release - drop) / (t_plate - t_release)
    return {
        "pitcher": pitcher, "player_name": f"Arm{pitcher}, Test", "p_throws": hand,
        "pitch_type": "FF", "game_type": "R", "game_year": year, "game_date": f"{year}-06-0{1 + pitcher % 9}",
        "vx0": 5.0, "ax": -8.0, "plate_x": 0.0, "vy0": vy0, "vz0": vz0, "ay": ay, "az": az,
        "release_pos_y": release_y, "release_pos_z": z_release, "plate_z": plate_z,
        "release_extension": 60.5 - release_y, "arm_angle": 40.0, "release_speed": 95.0,
        "pfx_z": 1.3, "sz_bot": 1.6, "sz_top": 3.3,
    }


def _league(n_pitchers: int = 30, each: int = 60, low_arm: int = 1) -> pd.DataFrame:
    """A fake league. Pitcher `low_arm` releases a foot lower than everyone else."""
    rng = np.random.default_rng(3)
    return pd.DataFrame([
        _pitch(rng, p, release_z=4.8 if p == low_arm else 5.8)
        for p in range(1, n_pitchers + 1) for _ in range(each)
    ])


# --- one trajectory ---------------------------------------------------------

def test_trajectory_runs_from_release_to_the_chosen_height():
    path = m.model_trajectory(_state(), plate_height=2.75)
    assert (path.y[0], path.z[0]) == pytest.approx((54.0, 5.8))
    assert (path.y[-1], path.z[-1]) == pytest.approx((vaa.PLATE_FRONT_Y, 2.75))
    assert path.vaa == pytest.approx(math.degrees(math.atan2(path.vz_final, abs(path.vy_final))))
    assert path.vaa < 0


def test_trajectory_is_curved_and_vaa_is_not_the_straight_line_slope():
    path = m.model_trajectory(_state())
    chord = path.z[0] + (path.z[-1] - path.z[0]) * (path.y[0] - path.y) / (path.y[0] - path.y[-1])
    assert np.abs(path.z - chord).max() > 0.1                     # really curved
    chord_angle = math.degrees(math.atan((path.z[-1] - path.z[0]) / (path.y[0] - path.y[-1])))
    assert abs(path.vaa - chord_angle) > 0.5


def test_changing_the_target_height_recomputes_the_whole_path():
    low, high = m.model_trajectory(_state(), 2.0), m.model_trajectory(_state(), 3.5)
    assert low.z[0] == high.z[0] == 5.8                           # same release point
    assert np.all(high.z[1:] > low.z[1:])                         # every point moved
    assert high.vaa > low.vaa                                     # higher = flatter


def test_model_reproduces_a_real_pitchs_observed_vaa_at_its_own_height():
    rows = _league(n_pitchers=1, each=20)
    states, notes = m.release_states(rows)
    assert notes == []
    for _, s in states.iterrows():
        path = m.model_trajectory(s.to_dict(), plate_height=s["plate_height"])
        assert path.vaa == pytest.approx(s["vaa"], abs=1e-9)


def test_observed_plate_height_agrees_with_plate_z_in_both_eras():
    for year in (2025, 2026):
        rng = np.random.default_rng(1)
        rows = pd.DataFrame([_pitch(rng, 1, year=year) for _ in range(10)])
        # plate_z was generated at that era's depth, so the path must start at release.
        start = vaa.flight_paths(rows, n_points=2)[:, 0, 2]
        np.testing.assert_allclose(start, rows["release_pos_z"], atol=1e-9)


# --- missing data -------------------------------------------------------------

def test_missing_acceleration_is_never_filled_in():
    with pytest.raises(ValueError, match="made-up"):
        m.release_states(_league(1, 5).drop(columns=["az"]))


def test_estimates_are_reported():
    rows = _league(1, 10)
    states, notes = m.release_states(rows.assign(release_pos_z=np.nan))
    assert "some release heights estimated from the trajectory fit" in notes
    np.testing.assert_allclose(states["release_z"], rows["release_pos_z"], atol=1e-6)
    _, notes = m.release_states(rows.drop(columns=["release_pos_y"]))
    assert "some release distances derived from extension" in notes


# --- explanation and classification ------------------------------------------

def test_shares_add_up_to_the_modeled_gap():
    pitcher = _state(release_z=4.7, release_y=53.5, speed_y=-136.0, az=-18.0)
    league = _state()
    shares = m.explain_gap(pitcher, league)
    gap = m.model_trajectory(pitcher).vaa - m.model_trajectory(league).vaa
    assert sum(shares.values()) == pytest.approx(gap)
    assert shares["release_height"] > 0 and shares["ride"] < 0   # low release flattens, less carry steepens
    assert all(v == pytest.approx(0) for v in m.explain_gap(league, league).values())


@pytest.mark.parametrize("value, words", [
    (0.9, "Flatter than average"), (-1.1, "Steeper than average"), (0.1, "About average"),
    (0.26, "Flatter than average"),     # shown as +0.3, so the words must agree
    (-0.24, "About average"),           # shown as -0.2
])
def test_classify_matches_the_displayed_number(value, words):
    assert m.classify(value) == words


# --- the whole comparison -------------------------------------------------------

def test_comparison_flags_the_low_release_pitcher_as_flatter():
    c = m.compare_fastball(_league(), 1)
    assert c.classification == "Flatter than average"
    assert c.pitcher.vaa_aa > 0 and c.modeled_gap > 0
    assert c.pitcher.vaa_aa == pytest.approx(c.modeled_gap, abs=0.1)
    assert max(c.shares, key=lambda k: abs(c.shares[k])) == "release_height"
    assert c.league.vaa_aa == pytest.approx(0, abs=1e-9)         # zero by construction
    assert c.name == "Test Arm1" and c.season == 2026 and c.pitcher.pitches == 60
    assert c.pitcher.ivb == pytest.approx(15.6)                  # 1.3 ft -> inches
    assert c.notes == ()


def test_comparison_uses_only_four_seamers_same_hand_regular_season():
    data = _league()
    extras = data[data["pitcher"] == 2]
    noisy = pd.concat([
        data,
        extras.assign(pitch_type="SL"),
        extras.assign(game_type="S"),
        extras.assign(pitcher=99, p_throws="L"),
    ])
    c = m.compare_fastball(noisy, 1)
    assert c.league.pitches == len(data) and c.league_pitchers == 30


def test_comparison_errors_and_warnings():
    data = _league()
    with pytest.raises(ValueError, match="not 2019"):
        m.compare_fastball(data, 1, season=2019)
    with pytest.raises(ValueError, match="No regular-season"):
        m.compare_fastball(data, 12345)
    with pytest.raises(ValueError, match="league-wide"):
        m.compare_fastball(data[data["pitcher"] == 1], 1)
    few = pd.concat([data[data["pitcher"] != 1], data[data["pitcher"] == 1].head(7)])
    assert "small sample (7 pitches)" in m.compare_fastball(few, 1).notes
