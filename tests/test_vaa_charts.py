"""Smoke tests for the VAA chart prototypes: they draw, and they fail loudly."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import vaa_charts


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
