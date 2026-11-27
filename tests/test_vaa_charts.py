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
    stretch = vaa_charts._vertical_stretch(figure.axes[1])
    anchor = comparison.league.state["release_y"]
    for line, side in ((league_line, comparison.league), (pitcher_line, comparison.pitcher)):
        y, z = line.get_xdata(), line.get_ydata()
        hand = vaa_charts._mound_x(side.state["release_y"], anchor, stretch)   # drawn hand
        assert (y[0], z[0]) == pytest.approx((hand, side.state["release_z"]))
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
    assert not any("effect" in card for card in cards)
    assert cards[2]["percentile"] == "" and cards[0]["percentile"].endswith("percentile")

    blind = vaa_model.compare_fastball(_league().drop(columns=["arm_angle", "sz_bot", "sz_top"]), 1)
    assert vaa_charts._cards(blind)[2]["value"] == "n/a"
    plt.close(vaa_charts.fastball_vaa_chart(blind))          # no heights, no zone: still draws


def test_percentiles_read_like_a_scouting_page():
    words = vaa_charts._percentile_words
    assert words(0.904) == "90th percentile"
    assert words(0.58) == "58th percentile"
    assert words(0.0) == "1st percentile" and words(1.0) == "99th percentile"
    assert words(None) == ""
    assert [vaa_charts._ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 63)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "63rd"]


def test_cards_rank_him_with_dots_instead_of_bars(comparison, figure):
    # The synthetic pitcher 1 releases a foot lower than everyone else.
    assert comparison.ranks["release_height"] == 0.0
    assert vaa_charts._cards(comparison)[0]["percentile"] == "1st percentile"
    cards = figure.axes[-5:]                              # the five card panels
    assert not any(card.containers for card in cards)     # no bar graphs left
    assert len(_lines(figure, "rank-dot")) == 4           # arm angle keeps its drawing
    words = " ".join(t.get_text() for t in figure.texts)
    assert "vs. 29 right-handers with 50+ four-seamers" in words
    assert "of 100" not in " ".join(t.get_text() for card in cards for t in card.texts)


def test_too_few_fastballs_to_rank_still_draws_the_cards():
    thin = vaa_model.compare_fastball(_league(each=20), 1)    # nobody reaches 50
    assert thin.ranks == {} and thin.rank_pool == 0
    fig = vaa_charts.fastball_vaa_chart(thin)
    assert not _lines(fig, "rank-dot")
    assert "Percentiles" not in " ".join(t.get_text() for t in fig.texts)
    plt.close(fig)


def test_why_names_the_main_trait_in_plain_baseball_terms(comparison):
    why = vaa_charts._why(comparison)
    assert why.startswith("Why: His release height is about ")
    assert why.endswith("lower than most pitchers.")
    assert "°" not in why


def test_header_shows_two_angles_that_agree_with_the_verdict(comparison, figure):
    texts = [t.get_text() for t in figure.axes[1].texts]      # tagged on the drawing
    assert "HIS VAA" in texts and "MLB VAA, SAME HEIGHT" in texts
    his, typical = (float(t.rstrip("°")) for t in texts if t.endswith("°"))
    assert his - typical == pytest.approx(round(comparison.pitcher.vaa_aa, 1))


def test_lengths_are_words_not_decimals():
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


def test_chart_text_stays_few_and_large(comparison, figure):
    shown = [t for ax in figure.axes for t in ax.texts] + list(figure.texts)
    assert min(t.get_fontsize() for t in shown) >= vaa_charts._TEXT
    assert len(figure.texts) == 2                         # two footer lines, not five
    assert any("atan2" in note for note in vaa_charts.method_notes(comparison))


def test_an_average_fastball_gets_no_heading_and_no_reason(comparison, figure):
    ordinary = vaa_model.compare_fastball(_league(), 2)       # releases like the rest
    assert ordinary.classification == "About average"
    fig = vaa_charts.fastball_vaa_chart(ordinary)
    header, drawing = (" ".join(t.get_text() for t in ax.texts) for ax in fig.axes[:2])
    plt.close(fig)
    assert "average" not in header.lower().replace("avg", "")
    assert "Why:" not in header
    assert "HIS VAA" in drawing and "MLB VAA, SAME HEIGHT" in drawing
    # A fastball that does stand out keeps its heading.
    assert comparison.classification in [t.get_text() for t in figure.axes[0].texts]


def test_extension_is_measured_from_the_rubber_to_the_ball(comparison, figure):
    (line,) = _lines(figure, "extension-line")
    start, end = line.get_xdata()
    stretch = vaa_charts._vertical_stretch(figure.axes[1])
    anchor = comparison.league.state["release_y"]
    assert start == pytest.approx(vaa_charts._mound_x(vaa_charts.RUBBER_Y, anchor, stretch))
    assert (start - end) / stretch == pytest.approx(comparison.pitcher.extension, abs=0.05)

    rubber, ball = _lines(figure, "extension-upright")
    assert ball.get_xdata()[0] == pytest.approx(
        vaa_charts._mound_x(comparison.pitcher.state["release_y"], anchor, stretch))
    assert ball.get_ydata()[0] == comparison.pitcher.state["release_z"]   # starts at the ball
    level = line.get_ydata()[0]
    assert level > comparison.pitcher.state["release_z"] and rubber.get_ydata()[1] > level

    length = vaa_charts._format_height(comparison.pitcher.extension * 12)
    words = [t.get_text() for ax in figure.axes for t in ax.texts]
    assert length in words and "extension" in words
    assert vaa_charts._cards(comparison)[1]["value"] == length          # card agrees
    assert not any("modeled" in w for w in words)


def _side_figure(height_ft: float):
    fig, ax = plt.subplots()
    vaa_figures.draw_pitcher_side(ax, (54.0, 6.0), 60.5, height_ft, "#cc3355", 2.0)
    return fig, ax


def test_a_taller_pitcher_is_drawn_with_a_longer_stride():
    feet = []
    for height in (5.75, 6.85):
        fig, ax = _side_figure(height)
        # Shapes that reach the ground: the one nearest the plate is the lead foot.
        low = [p.get_xy() for p in ax.patches if p.get_xy()[:, 1].min() < 0.9]
        feet.append(min(points[:, 0].min() for points in low))
        plt.close(fig)
    # Smaller distance = closer to the plate = a longer stride.
    assert feet[0] - feet[1] == pytest.approx(0.85 * (6.85 - 5.75), abs=0.2)


def _center(patch):
    points = patch.get_xy()
    return ((points[:, 0].min() + points[:, 0].max()) / 2,
            (points[:, 1].min() + points[:, 1].max()) / 2)


def test_both_pitchers_are_drawn_with_their_ball_where_the_line_starts(comparison, figure):
    league_line, pitcher_line = _lines(figure, "pitch-path")
    for gid, line, side in (("average-ball", league_line, comparison.league),
                            ("ball", pitcher_line, comparison.pitcher)):
        (ball,) = _lines(figure, gid)
        start = (line.get_xdata()[0], line.get_ydata()[0])
        assert _center(ball) == pytest.approx(start, abs=0.05)
        assert start[1] == pytest.approx(side.state["release_z"])    # true release height


def test_flight_paths_are_the_modeled_ones_near_the_plate_and_never_double_back(comparison, figure):
    for line, side in zip(_lines(figure, "pitch-path"), (comparison.league, comparison.pitcher)):
        x, z = line.get_xdata(), line.get_ydata()
        near = side.modeled.y <= vaa_charts._BEND_FROM
        assert x[near] == pytest.approx(side.modeled.y[near])
        assert z == pytest.approx(side.modeled.z)                    # heights never change
        assert np.all(np.diff(x) < 0)                                # always toward the plate
    gray = _lines(figure, "pitch-path")[0]
    assert gray.get_xdata() == pytest.approx(comparison.league.modeled.y)   # not moved at all


def test_the_whole_mound_fits_inside_the_drawing(comparison, figure):
    side = figure.axes[1]
    left = max(side.get_xlim())                       # the edge behind the mound
    shapes = [p.get_xy()[:, 0].max() for p in side.patches if hasattr(p, "get_xy")
              and np.ndim(p.get_xy()) == 2]
    assert max(shapes) < left


def test_the_figures_keep_true_proportions_on_a_stretched_view():
    widths = []
    for anchor in (None, 54.0):
        fig, ax = plt.subplots()
        vaa_figures.draw_pitcher_side(ax, (54.0, 6.0), 60.5, 6.33, "#cc3355", 2.5, anchor=anchor)
        xs = np.concatenate([p.get_xy()[:, 0] for p in ax.patches if p.get_gid() is None])
        widths.append(xs.max() - xs.min())
        plt.close(fig)
    assert widths[1] > 1.8 * widths[0]


def _delivery(extension: float):
    """(front edge of the lead foot, top of the head) for one 6'4" pitcher."""
    fig, ax = plt.subplots()
    vaa_figures.draw_pitcher_side(ax, (60.5 - extension, 6.0), 60.5, 6.33, "#cc3355", 2.0)
    foot = min(p.get_xy()[:, 0].min() for p in ax.patches if p.get_xy()[:, 1].min() < 0.9)
    (head,) = [p for p in ax.patches if p.get_gid() == "head"]
    top = head.get_xy()[:, 1].max()
    plt.close(fig)
    return foot, top


def test_a_longer_extension_changes_the_whole_delivery_not_just_the_hand():
    short_foot, short_top = _delivery(6.75)          # 6'9"
    long_foot, long_top = _delivery(7.75)            # 7'9"
    assert short_foot - long_foot > 0.9              # lands a foot or more farther down
    assert short_top - long_top > 0.2                # and the head drops lower


def test_the_throwing_arm_is_drawn_over_the_head():
    fig, ax = _side_figure(6.83)
    (head,) = [p for p in ax.patches if p.get_gid() == "head"]
    in_front = [p for p in ax.patches
                if p.get_zorder() > head.get_zorder() + 0.05 and p.get_gid() != "ball"]
    plt.close(fig)
    assert in_front                                    # the arm, not only the ball


def test_the_pitch_ends_at_a_labeled_zone_over_home_plate(comparison, figure):
    side = figure.axes[1]
    words = [t.get_text() for t in side.texts]
    assert "strike zone" in words
    assert "dashed line = the ball's direction\nas it reaches the plate" in words
    (plate,) = _lines(figure, "home-plate")
    (zone,) = _lines(figure, "zone-outline")
    assert plate.get_width() == pytest.approx(vaa.PLATE_FRONT_Y)
    assert (zone.get_y(), zone.get_y() + zone.get_height()) == pytest.approx(comparison.zone)
    assert not hasattr(vaa_figures, "draw_batter_side")


def test_without_zone_data_there_is_still_a_plate():
    blind = vaa_model.compare_fastball(_league().drop(columns=["sz_bot", "sz_top"]), 1)
    fig = vaa_charts.fastball_vaa_chart(blind)
    assert _lines(fig, "home-plate") and not _lines(fig, "zone-outline")
    plt.close(fig)


def test_angle_tags_sit_inside_the_drawing_above_the_path_names(comparison, figure):
    side = figure.axes[1]
    tags = [t for t in side.texts if t.get_gid() == "angle-tag"]
    assert len(tags) == 2
    names = [t for t in side.texts if t.get_text() in (comparison.name, "MLB average")]
    assert min(t.get_position()[1] for t in tags) > max(t.get_position()[1] for t in names) + 1.5
    assert max(t.get_position()[1] for t in tags) < side.get_ylim()[1]


def test_the_header_only_takes_room_for_a_verdict_when_there_is_one(comparison, figure):
    ordinary = vaa_charts.fastball_vaa_chart(vaa_model.compare_fastball(_league(), 2))
    assert figure.get_figheight() - ordinary.get_figheight() == pytest.approx(
        vaa_charts._HEADER_WITH_VERDICT_IN - vaa_charts._HEADER_IN)
    assert ordinary.get_figwidth() == figure.get_figwidth()
    plt.close(ordinary)


def test_a_lefty_is_drawn_as_the_mirror_image_of_a_righty(figure):
    lefty = vaa_model.compare_fastball(_league().assign(p_throws="L"), 1)
    fig = vaa_charts.fastball_vaa_chart(lefty)
    left_view, right_view = fig.axes[1], figure.axes[1]
    # Righty: distance from the plate grows to the left. Lefty: it grows to the right.
    assert right_view.get_xlim()[0] > right_view.get_xlim()[1]
    assert left_view.get_xlim()[0] < left_view.get_xlim()[1]
    (ball,) = _lines(fig, "ball")
    (plate,) = _lines(fig, "home-plate")
    to_screen = left_view.transData.transform
    assert to_screen(_center(ball))[0] > to_screen((plate.get_x(), 0))[0]   # mound on the right
    assert "left-handed" in " ".join(t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_only_low_arm_slots_are_named():
    angles = (51.0, 20.0, 19.4, 12.0, 0.0, -1.0, -11.0, -15.0, -18.0, -61.0)
    names = {a: vaa_charts._slot_name(a) for a in angles}
    assert names == {51.0: None, 20.0: None, 19.4: "SIDEARM", 12.0: "SIDEARM", 0.0: "SIDEARM",
                     -1.0: "SIDEARM", -11.0: "SIDEARM", -15.0: "SIDEARM",
                     -18.0: "SUBMARINE", -61.0: "SUBMARINE"}
    assert vaa_charts._slot_name(float("nan")) is None
    assert vaa_charts._fmt(-0.4, 0, "°") == "0°"                      # never "-0°"


@pytest.mark.parametrize("hand", ["R", "L"])
def test_a_sidearm_pitcher_is_tagged_between_his_hand_and_the_plate(hand, figure):
    low = _league().assign(p_throws=hand)
    low.loc[low["pitcher"] == 1, "arm_angle"] = 12.0
    fig = vaa_charts.fastball_vaa_chart(vaa_model.compare_fastball(low, 1))
    (tag,) = _lines(fig, "slot-tag")
    assert tag.get_text() == "SIDEARM\n12° arm angle"
    (ball,) = _lines(fig, "ball")
    assert 0 < tag.get_position()[0] < _center(ball)[0]               # plate is at x = 0
    plt.close(fig)
    assert not _lines(figure, "slot-tag")                             # a 40° arm: no tag
