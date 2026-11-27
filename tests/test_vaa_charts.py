"""Smoke tests for the VAA chart prototypes: they draw, and they fail loudly."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import vaa, vaa_charts, vaa_figures, vaa_model
from tests.test_vaa_model import _league


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
# Prototype A6: modeled paths, tangents, cards.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def comparison():
    return vaa_model.compare_fastball(_league(), 1)


@pytest.fixture
def figure(comparison):
    fig = vaa_charts.fastball_vaa_chart(comparison, 74, 74.5)
    fig.canvas.draw()                       # settles the equal-aspect axes
    yield fig
    plt.close(fig)


def _lines(fig, gid: str) -> list:
    """Every plotted line in the figure tagged with `gid`, including inset panels."""
    return fig.findobj(lambda artist: artist.get_gid() == gid)


def test_paths_start_at_each_release_point_and_share_the_crossing_height(comparison, figure):
    league_line, pitcher_line = _lines(figure, "pitch-path")
    for line, side in ((league_line, comparison.league), (pitcher_line, comparison.pitcher)):
        y, z = line.get_xdata(), line.get_ydata()
        assert (y[0], z[0]) == pytest.approx((side.state["release_y"], side.state["release_z"]))
        assert (y[-1], z[-1]) == pytest.approx((vaa.PLATE_FRONT_Y, 2.75))
        assert len(y) > 2                   # a curve, not a straight segment


def test_tangent_lines_have_the_arrival_slope(comparison, figure):
    for line, side in zip(_lines(figure, "tangent-line"), (comparison.league, comparison.pitcher)):
        (y0, y1), (z0, z1) = line.get_xdata(), line.get_ydata()
        angle = -np.degrees(np.arctan2(z1 - z0, y1 - y0))
        assert angle == pytest.approx(side.modeled.vaa)


def test_cards_carry_every_metric_and_survive_missing_arm_angle(comparison):
    cards = vaa_charts._cards(comparison)
    assert [card["title"] for card in cards] == [
        "RELEASE HEIGHT", "EXTENSION", "ARM ANGLE", "VELOCITY", "RIDE (IVB)"]
    assert cards[0]["effect"].endswith("flatter")
    assert cards[2]["effect"] == "shown for context"

    blind = vaa_model.compare_fastball(_league().drop(columns=["arm_angle", "sz_bot", "sz_top"]), 1)
    assert vaa_charts._cards(blind)[2]["value"] == "n/a"
    plt.close(vaa_charts.fastball_vaa_chart(blind))          # no heights, no zone: still draws


def test_why_names_the_main_trait_in_plain_baseball_terms(comparison):
    why = vaa_charts._why(comparison)
    assert why.startswith("Why: His release height is about ")
    assert why.endswith("lower than most pitchers.")
    assert "°" not in why


def test_header_shows_two_angles_that_agree_with_the_verdict(comparison, figure):
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert "HIS VAA" in texts and "MLB VAA, SAME HEIGHT" in texts
    his, typical = (float(t.rstrip("°")) for t in texts if t.endswith("°"))
    assert his - typical == pytest.approx(round(comparison.pitcher.vaa_aa, 1))


def test_effects_and_lengths_are_words_not_decimals():
    assert vaa_charts._effect_text(0.1) == "little effect"
    assert vaa_charts._effect_text(0.4) == "makes it flatter"
    assert vaa_charts._effect_text(-0.4) == "makes it steeper"
    assert vaa_charts._length(-1.07) == "1 ft"
    assert vaa_charts._length(1.4) == "1.5 ft"
    assert vaa_charts._length(0.4) == "5 in"


def test_a_whole_season_is_named_instead_of_dated(comparison):
    whole = replace(comparison, first_date="Mar 25", last_date="Sep 27")
    part = replace(comparison, first_date="Apr 1", last_date="Jun 30")
    for c, expected, dated in ((whole, "full season", False), (part, "Apr 1–Jun 30", True)):
        fig = vaa_charts.fastball_vaa_chart(c)
        header = " ".join(t.get_text() for t in fig.axes[0].texts)
        plt.close(fig)
        assert expected in header and ("Jun 30" in header) == dated
    assert vaa_charts._sample_warning(whole).endswith("four-seamers all season")
    assert not vaa_charts._sample_warning(part).endswith("all season")


def test_format_height():
    assert vaa_charts._format_height(74) == "6'2\""
    assert vaa_charts._format_height(None) == "n/a"


# --- figurine geometry ----------------------------------------------------------
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
