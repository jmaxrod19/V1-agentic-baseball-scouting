"""Tests for src/charts.py — the "numbers in, PNG data-URI out" layer.

We don't assert pixels. Two cheaper things matter and are worth locking in:
  1. each chart returns a valid PNG data URI and doesn't crash on real input;
  2. the empty / all-null guards raise ValueError instead of silently emitting a
     blank chart (the behavior the chart functions promise).
"""

import numpy as np
import pandas as pd
import pytest

from src import charts

# The batted-ball frame the hitter charts expect: exit velocity, launch angle,
# and Savant's barrel bucket. Four rows is plenty to draw.
_BBE = pd.DataFrame(
    {
        "launch_speed": [100.0, 96.0, 80.0, 94.0],
        "launch_angle": [10.0, 40.0, 20.0, 5.0],
        "launch_speed_angle": [6, 6, 3, 1],
    }
)

# Every hitter chart takes the bbe frame and returns a data-URI string.
_HITTER_CHARTS = [
    charts.ev_distribution,
    charts.launch_angle_distribution,
    charts.ev_la_scatter,
]


@pytest.mark.parametrize("chart_fn", _HITTER_CHARTS)
def test_chart_returns_png_data_uri(chart_fn):
    uri = chart_fn(_BBE)
    # A real, embeddable PNG data URI — this is what the report inlines.
    assert isinstance(uri, str)
    assert uri.startswith("data:image/png;base64,")
    # A non-trivial payload means an actual image was encoded, not an empty stub.
    assert len(uri) > 100


def test_ev_distribution_raises_on_empty_frame():
    # No rows at all -> the _require_rows guard should fire.
    with pytest.raises(ValueError):
        charts.ev_distribution(pd.DataFrame({"launch_speed": []}))


def test_ev_distribution_raises_when_all_values_null():
    # Rows exist but every exit velocity is NaN (a pull that never tracked EV):
    # must raise, not embed a blank histogram.
    with pytest.raises(ValueError):
        charts.ev_distribution(pd.DataFrame({"launch_speed": [np.nan, np.nan]}))


# ---------------------------------------------------------------------------
# stance_contact_map — the top-down batter's-box view
# ---------------------------------------------------------------------------

# ~40 swings across a few pitch types, enough for at least one zone ellipse.
_POINTS = pd.DataFrame(
    {
        "intercept_side_in":  np.linspace(38, 48, 40),
        "intercept_depth_in": np.linspace(15, 30, 40),
        "pitch_type":         (["FF"] * 20) + (["SL"] * 12) + (["CH"] * 8),
    }
)

# A stance row shaped like swing_geometry.player_stance() output.
_STANCE = {
    "avg_batter_x_position": 34.7,
    "avg_batter_y_position": 20.2,
    "avg_foot_sep": 36.8,
    "avg_stance_angle": -2.7,
    "side": "R",
}


def test_stance_map_returns_png_data_uri():
    uri = charts.stance_contact_map(_POINTS, _STANCE, player_name="Test Hitter")
    assert uri.startswith("data:image/png;base64,")
    assert len(uri) > 100


def test_stance_map_with_biomech_context():
    # Passing biomech exercises the caption (width + depth percentiles) + title.
    uri = charts.stance_contact_map(
        _POINTS, _STANCE, player_name="Test Hitter",
        biomech={
            "height_text": "6' 2\"", "stance_width_pct": 87,
            "stance_depth": 20.2, "stance_depth_pct": 30, "reach_pct": 12,
        },
    )
    assert uri.startswith("data:image/png;base64,")


def test_biomech_caption_phrases_below_median_as_narrower_and_shallower():
    # A below-50 percentile should read "narrower"/"shallower", not the
    # confusing "wider/deeper than 1%" a naive phrasing would produce.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        charts._biomech_caption(
            ax,
            {"height_text": "5' 8\"", "stance_width": 8, "stance_width_pct": 1,
             "stance_depth": 30, "stance_depth_pct": 1},
            charts._STANCE_THEMES["report"],
        )
        texts = [t.get_text() for t in ax.texts]
        combined = " ".join(texts)
        assert "narrower than 99%" in combined
        assert "shallower in the box than 99%" in combined
    finally:
        plt.close(fig)


def test_stance_map_works_without_pitch_type():
    # No pitch_type column -> one neutral cloud, no legend, still valid.
    uri = charts.stance_contact_map(_POINTS.drop(columns=["pitch_type"]), _STANCE)
    assert uri.startswith("data:image/png;base64,")


def test_stance_map_raises_on_empty_points():
    with pytest.raises(ValueError):
        charts.stance_contact_map(
            pd.DataFrame({"intercept_side_in": [], "intercept_depth_in": []}), _STANCE
        )


def test_stance_map_raises_on_missing_coordinate_columns():
    with pytest.raises(ValueError):
        charts.stance_contact_map(pd.DataFrame({"foo": [1, 2]}), _STANCE)


def test_stance_map_raises_when_stance_has_no_box_position():
    with pytest.raises(ValueError):
        charts.stance_contact_map(_POINTS, {"avg_foot_sep": 36.0})  # no x/y
