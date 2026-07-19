"""Tests for src/percentiles.py — league percentile ranking for hitters.

No network: the league leaderboard pull is monkeypatched to a tiny synthetic
table with known values, and PROCESSED_DIR is redirected to a tmp_path so the
disk cache never touches the real data/processed/ folder.
"""

import pandas as pd
import pytest

from src import loaders, percentiles


# ---------------------------------------------------------------------------
# percentile_of() — the rank math
# ---------------------------------------------------------------------------

def test_percentile_of_is_share_at_or_below():
    pop = pd.Series([10, 20, 30, 40])
    assert percentiles.percentile_of(30, pop) == 75   # 3 of 4 values <= 30
    assert percentiles.percentile_of(40, pop) == 100  # everything <= 40
    assert percentiles.percentile_of(5, pop) == 0     # nothing <= 5


def test_percentile_of_handles_missing_and_empty():
    assert percentiles.percentile_of(None, pd.Series([1, 2])) is None
    assert percentiles.percentile_of(30, pd.Series([], dtype=float)) is None


# ---------------------------------------------------------------------------
# hitter_percentiles() — mapping + unit alignment against a synthetic league
# ---------------------------------------------------------------------------

def _fake_league() -> pd.DataFrame:
    # Four "players" so a value landing 3rd-of-4 is the 75th percentile.
    return pd.DataFrame(
        {
            "avg_hit_speed": [85, 88, 90, 92],
            "max_hit_speed": [100, 105, 110, 115],
            "ev95percent": [20, 30, 40, 50],           # percents in the league table
            "brl_percent": [2, 5, 8, 12],
            "anglesweetspotpercent": [25, 30, 35, 40],
        }
    )


def test_hitter_percentiles_aligns_units_and_ranks(monkeypatch, tmp_path):
    monkeypatch.setattr(percentiles.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_league_hitting_baseline", lambda *a, **k: _fake_league())

    # Rate metrics are decimals here (our internal units); the code scales them
    # to percents to match the league columns before ranking.
    profile = {
        "avg_ev": 90,
        "max_ev": 110,
        "hard_hit_rate": 0.40,
        "barrel_rate": 0.08,
        "sweet_spot_rate": 0.35,
    }
    pct = percentiles.hitter_percentiles(profile, 2024)

    assert pct["avg_ev"] == 75          # 90 -> 3 of 4
    assert pct["max_ev"] == 75          # 110 -> 3 of 4
    assert pct["hard_hit_rate"] == 75   # 0.40 -> 40% -> 3 of 4
    assert pct["barrel_rate"] == 75     # 0.08 -> 8% -> 3 of 4
    assert pct["sweet_spot_rate"] == 75  # 0.35 -> 35% -> 3 of 4


def test_missing_metric_value_yields_none(monkeypatch, tmp_path):
    monkeypatch.setattr(percentiles.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(
        loaders, "pull_league_hitting_baseline",
        lambda *a, **k: pd.DataFrame({"avg_hit_speed": [85, 90]}),
    )
    pct = percentiles.hitter_percentiles({"avg_ev": None}, 2024)
    assert pct["avg_ev"] is None


def test_baseline_is_cached_after_first_pull(monkeypatch, tmp_path):
    monkeypatch.setattr(percentiles.config, "PROCESSED_DIR", tmp_path)
    calls = {"n": 0}

    def counting_pull(*a, **k):
        calls["n"] += 1
        return pd.DataFrame({"avg_hit_speed": [85, 90]})

    monkeypatch.setattr(loaders, "pull_league_hitting_baseline", counting_pull)

    percentiles.load_hitting_baseline(2024)
    percentiles.load_hitting_baseline(2024)  # should read the CSV, not re-pull

    assert calls["n"] == 1
    assert (tmp_path / "league_hitting_2024.csv").exists()
