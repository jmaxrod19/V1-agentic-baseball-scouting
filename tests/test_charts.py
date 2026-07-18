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
