"""Smoke tests for the VAA chart prototypes: they draw, and they fail loudly."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import vaa, vaa_charts, vaa_figures


def _pitches(n: int = 5) -> pd.DataFrame:
    """n slightly different, realistic four-seamers with every column needed."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "vx0": rng.normal(6, 1, n), "vy0": rng.normal(-130, 2, n), "vz0": rng.normal(-5, 1, n),
        "ax": rng.normal(-10, 1, n), "ay": rng.normal(25, 1, n), "az": rng.normal(-20, 1, n),
        "plate_x": rng.normal(0, 0.5, n), "plate_z": rng.normal(2.6, 0.5, n),
        "release_pos_y": rng.normal(54, 0.3, n),
        "sz_bot": 1.6, "sz_top": 3.4,
    })


def test_flight_paths_3d_returns_a_figure():
    fig = vaa_charts.flight_paths_3d(_pitches(), "Test Pitcher FF")
    assert hasattr(fig, "savefig")
    plt.close(fig)


def test_flight_paths_3d_samples_down_to_max_paths():
    fig = vaa_charts.flight_paths_3d(_pitches(20), "Test", max_paths=7)
    assert len(fig.axes[0].lines) == 7 + 1      # 7 pitch paths + the strike-zone outline
    plt.close(fig)


def test_flight_paths_3d_works_without_zone_columns():
    df = _pitches().drop(columns=["sz_bot", "sz_top"])
    fig = vaa_charts.flight_paths_3d(df, "Test")
    plt.close(fig)


def test_flight_paths_3d_raises_when_nothing_is_traceable():
    df = _pitches()
    df["vz0"] = np.nan
    with pytest.raises(ValueError, match="no pitches"):
        vaa_charts.flight_paths_3d(df, "Test")


def test_compare_to_league_3d_draws_two_lines():
    fig = vaa_charts.compare_to_league_3d(_pitches(8), _pitches(30), "Pitcher", "League")
    # two path lines + the strike-zone outline
    assert len(fig.axes[0].lines) == 3
    plt.close(fig)


def test_compare_to_league_3d_raises_on_untraceable_side():
    bad = _pitches()
    bad["vz0"] = np.nan
    with pytest.raises(ValueError, match="no pitches"):
        vaa_charts.compare_to_league_3d(bad, _pitches(), "Pitcher", "League")


# ---------------------------------------------------------------------------
# Prototype A4: side view, figurines, angle panels, tiles. The text logic is pure functions, so test it directly.
# ---------------------------------------------------------------------------

def _prof(**kw) -> dict:
    base = {"n": 50, "vaa": -4.6, "release_z": 5.8, "extension": 6.5, "arm_angle": 39.0}
    base.update(kw)
    return base


def test_format_height():
    assert vaa_charts._format_height(74) == "6'2\""
    assert vaa_charts._format_height(74.6) == "6'3\""
    assert vaa_charts._format_height(None) == "n/a"
    assert vaa_charts._format_height(float("nan")) == "n/a"


def test_delta_arrows_and_ties():
    assert vaa_charts._delta(6.0, 5.5, 1, " ft") == "▲ 0.5 ft"
    assert vaa_charts._delta(5.0, 6.0, 1, " ft") == "▼ 1.0 ft"
    assert vaa_charts._delta(5.01, 5.0, 1, " ft") == "about the same"
    assert vaa_charts._delta(float("nan"), 5.0, 1, " ft") == ""


def test_takeaway_describes_release_and_angle():
    text = vaa_charts._takeaway(_prof(release_z=4.7, extension=7.0, vaa=-3.6), _prof())
    assert text == "Releases 1.1 ft lower and 0.5 ft closer to the plate than average; approaches 1.0° flatter."


def test_takeaway_calls_small_differences_average():
    text = vaa_charts._takeaway(_prof(release_z=5.9, extension=6.4, vaa=-4.7), _prof())
    assert text == "Release point is close to league average; approaches at about the league-average angle."


def test_takeaway_steeper_and_farther():
    text = vaa_charts._takeaway(_prof(extension=5.9, vaa=-5.5), _prof())
    assert "0.6 ft farther from the plate" in text and "0.9° steeper" in text


def test_tiles_survive_missing_arm_angle_and_height():
    tiles = vaa_charts._tiles(_prof(arm_angle=float("nan")), _prof(arm_angle=float("nan")), None, None)
    by_label = {t["label"]: t for t in tiles}
    assert by_label["HEIGHT"]["value"] == "n/a"
    assert by_label["ARM ANGLE"]["value"] == "n/a"
    assert by_label["ARM ANGLE"]["note"] == "not in this data"
    assert by_label["EXTENSION"]["value"] == "6.5 ft"


def test_side_view_with_profile_draws_with_and_without_optional_columns():
    df = _pitches(10)
    df["release_pos_z"] = 5.9
    df["release_extension"] = 6.4
    df["arm_angle"] = 40.0
    fig = vaa_charts.side_view_with_profile(df, _pitches(30).assign(
        release_pos_z=5.8, release_extension=6.5, arm_angle=39.0), "P", "L", 74, 74.5)
    assert len(fig.axes) == 1 + 1 + 2 + 4        # header, side view, angle + arm panels, tiles
    plt.close(fig)

    # Older pulls: no release_pos_z / extension / arm_angle columns at all.
    fig = vaa_charts.side_view_with_profile(_pitches(10), _pitches(30), "P", "L")
    plt.close(fig)


def _lines(ax, gid: str) -> list:
    """The plotted lines tagged with `gid`."""
    return [line for line in ax.lines if line.get_gid() == gid]


def test_side_view_pitch_lines_are_straight():
    fig = vaa_charts.side_view_with_profile(_pitches(10), _pitches(30), "P", "L")
    lines = _lines(fig.axes[1], "pitch-line")
    assert len(lines) == 2                              # pitcher and league
    assert all(len(line.get_xdata()) == 2 for line in lines)   # two points = straight
    plt.close(fig)


def test_angle_panel_slopes_are_the_real_vaa():
    pitcher, league = _pitches(10), _pitches(30)
    fig = vaa_charts.side_view_with_profile(pitcher, league, "P", "L")
    drawn = []
    for line in _lines(fig.axes[2], "angle-line"):
        (x0, x1), (y0, y1) = line.get_xdata(), line.get_ydata()
        drawn.append(np.degrees(np.arctan((y1 - y0) / (x1 - x0))))
    expected = [vaa.compute_vaa(league).mean(), vaa.compute_vaa(pitcher).mean()]
    np.testing.assert_allclose(drawn, expected, atol=1e-9)
    plt.close(fig)


def test_arm_end_matches_the_arm_angle_and_hand():
    shoulder_x, level = vaa_figures.FRONT_SHOULDER_HALF, vaa_figures.FRONT_SHOULDER_Z
    for hand, side in (("R", -1), ("L", 1)):            # righty shows on the viewer's left
        x, z = vaa_figures.arm_end(hand, 30.0)
        assert np.sign(x) == side
        assert np.degrees(np.arctan2(z - level, abs(x) - shoulder_x)) == pytest.approx(30.0)
    assert vaa_figures.arm_end("R", -10.0)[1] < level   # submarine: hand below the shoulder


def test_mound_height():
    assert vaa_figures.mound_height(60.5) == pytest.approx(10 / 12)   # at the rubber
    assert vaa_figures.mound_height(54.0) == pytest.approx(4 / 12)
    assert vaa_figures.mound_height(10.0) == 0                        # flat field


def test_capsule_ends_have_the_requested_radii():
    outline = vaa_figures.capsule((0.0, 0.0), (10.0, 0.0), 2.0, 1.0)
    assert outline[:, 0].min() == pytest.approx(-2.0, abs=0.02)    # thick end
    assert outline[:, 0].max() == pytest.approx(11.0, abs=0.02)    # thin end
    assert np.abs(outline[:, 1]).max() <= 2.0 + 1e-9


def test_capsule_with_swallowed_circle_is_just_the_big_circle():
    outline = vaa_figures.capsule((0.0, 0.0), (0.1, 0.0), 2.0, 0.5)
    np.testing.assert_allclose(np.hypot(outline[:, 0], outline[:, 1]), 2.0)


def test_pitchers_ball_sits_at_the_release_point_even_when_stretched():
    fig, ax = plt.subplots()
    vaa_figures.draw_pitcher_side(ax, (54.0, 5.8), 60.5, 6.2, "#cc3552", stretch=2.1)
    (ball,) = [patch for patch in ax.patches if patch.get_gid() == "ball"]
    points = ball.get_xy()[:-1]                  # last point repeats the first
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    np.testing.assert_allclose(center, [54.0, 5.8], atol=0.02)
    plt.close(fig)
