"""Tests for src/metrics.py — the "DataFrame in, scouting numbers out" layer.

The technique here is known input -> known output: we hand-build tiny DataFrames
where we can compute the right answer by hand, then assert the code agrees. That
way a wrong formula fails loudly instead of silently shipping a bad stat.
"""

import numpy as np
import pandas as pd
import pytest

from src import metrics


# ---------------------------------------------------------------------------
# A tiny, fully-controlled set of batted balls.
#   EV  = [100, 96, 80, 94]  -> 2 of 4 are >= 95  (hard-hit rate 0.50)
#   LA  = [10, 40, 20,  5]   -> 2 of 4 in [8, 32] (sweet-spot rate 0.50)
#   LSA = [ 6,  6,  3,  1]   -> 2 of 4 equal 6    (barrel rate     0.50)
# The trailing 'swinging_strike' row is NOT a ball in play and must be filtered
# out, so it also tests that _prepare_batted_balls drops non-contact events.
# ---------------------------------------------------------------------------

def _sample_bip() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": ["hit_into_play"] * 4 + ["swinging_strike"],
            "launch_speed": [100.0, 96.0, 80.0, 94.0, np.nan],
            "launch_angle": [10.0, 40.0, 20.0, 5.0, np.nan],
            "launch_speed_angle": [6, 6, 3, 1, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# Rate helpers — the small pure functions the metrics are built from.
# ---------------------------------------------------------------------------

def test_rate_ge_counts_share_at_or_above_threshold():
    # 1 of 2 known values is >= 95 (the NaN is excluded from the denominator).
    s = pd.Series([95.0, 94.0, np.nan])
    assert metrics._rate_ge(s, 95.0) == pytest.approx(0.5)


def test_rate_ge_returns_none_for_missing_series():
    # A column that doesn't exist arrives as None; the helper must say "no data".
    assert metrics._rate_ge(None, 95.0) is None


def test_rate_between_is_inclusive():
    # Both endpoints (8 and 32) count; 7 and 33 don't -> 2 of 4.
    s = pd.Series([8.0, 32.0, 7.0, 33.0])
    assert metrics._rate_between(s, 8.0, 32.0) == pytest.approx(0.5)


def test_rate_eq_matches_exact_value():
    s = pd.Series([6, 6, 3])
    assert metrics._rate_eq(s, 6) == pytest.approx(2 / 3)


def test_safe_mean_ignores_nan_and_returns_none_when_empty():
    assert metrics._safe_mean(pd.Series([2.0, 4.0])) == pytest.approx(3.0)
    # All-NaN -> None (not NaN), so downstream display stays clean.
    assert metrics._safe_mean(pd.Series([np.nan, np.nan])) is None


# ---------------------------------------------------------------------------
# batted_balls() — the public filter used by the chart layer.
# ---------------------------------------------------------------------------

def test_batted_balls_keeps_only_balls_in_play():
    bbe = metrics.batted_balls(_sample_bip())
    # The swinging_strike row is gone; the 4 hit_into_play rows remain.
    assert len(bbe) == 4
    assert (bbe["description"] == "hit_into_play").all()


def test_batted_balls_raises_without_description_column():
    # No 'description' column -> can't tell which pitches were put in play.
    with pytest.raises(KeyError):
        metrics.batted_balls(pd.DataFrame({"launch_speed": [100.0]}))


def test_batted_balls_raises_when_no_balls_in_play():
    df = pd.DataFrame({"description": ["ball", "swinging_strike"]})
    with pytest.raises(ValueError):
        metrics.batted_balls(df)


# ---------------------------------------------------------------------------
# hitter_batted_ball() — the one-row profile the report + charts consume.
# ---------------------------------------------------------------------------

def test_hitter_profile_matches_hand_computed_values():
    profile = metrics.hitter_batted_ball(_sample_bip())
    row = profile.iloc[0]

    assert row["n_bbe"] == 4
    assert row["avg_ev"] == pytest.approx(92.5)          # mean(100,96,80,94)
    assert row["max_ev"] == pytest.approx(100.0)
    assert row["avg_la"] == pytest.approx(18.75)         # mean(10,40,20,5)
    assert row["hard_hit_rate"] == pytest.approx(0.5)    # 2 of 4 >= 95
    assert row["sweet_spot_rate"] == pytest.approx(0.5)  # 2 of 4 in [8,32]
    assert row["barrel_rate"] == pytest.approx(0.5)      # 2 of 4 == 6


def test_hitter_profile_is_single_row_with_expected_columns():
    profile = metrics.hitter_batted_ball(_sample_bip())
    assert len(profile) == 1
    for col in ("n_bbe", "avg_ev", "hard_hit_rate", "barrel_rate", "sweet_spot_rate"):
        assert col in profile.columns
