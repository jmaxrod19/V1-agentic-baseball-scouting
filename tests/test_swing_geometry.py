"""Tests for src/swing_geometry.py — the DuckDB-backed swing-geometry card.

No network: the league leaderboard pull is monkeypatched to a tiny synthetic
table, and PROCESSED_DIR is redirected to tmp_path so the disk cache never
touches the real data/processed/ folder. The pitch-level tests build small
in-memory frames, so the DuckDB queries run against known data with known
expected answers.
"""

import numpy as np
import pandas as pd

from src import loaders, swing_geometry


# ---------------------------------------------------------------------------
# league_percentiles() — the CUME_DIST window query
# ---------------------------------------------------------------------------

def _fake_league() -> pd.DataFrame:
    # Four qualified hitters. For target id=101 every metric is the 3rd-largest
    # of four values, so CUME_DIST = 3/4 = 0.75 -> the 75th percentile. Keeping
    # them all equal makes the expected answer obvious. `side` is needed because
    # the stance JOIN keys on (id, side).
    return pd.DataFrame(
        {
            "id":   [101, 102, 103, 104],
            "side": ["R", "R", "R", "R"],
            "name": ["Target, T", "Low, L", "Mid, M", "High, H"],
            "attack_angle":              [15,   5,    10,   20],
            "swing_tilt":                [30,   10,   20,   40],
            "avg_intercept_y_vs_batter": [28,   24,   26,   30],
            "avg_bat_speed":             [72,   68,   70,   74],
            "ideal_attack_angle_rate":   [0.55, 0.40, 0.50, 0.60],
            # Batter-box position (also carried by the swing-path leaderboard) —
            # used by player_box_position as the no-stance fallback.
            "avg_batter_x_position":     [34,   30,   32,   36],
            "avg_batter_y_position":     [20,   22,   21,   23],
        }
    )


def _fake_stance() -> pd.DataFrame:
    # Stance snapshot keyed on (id, side). Same 3rd-of-4 trick: target id=101 is
    # the 75th percentile in both feet metrics.
    return pd.DataFrame(
        {
            "id":   [101, 102, 103, 104],
            "side": ["R", "R", "R", "R"],
            "avg_foot_sep":     [33, 31, 32, 34],   # 33 is 3rd of 4 -> 75th pct
            "avg_stance_angle": [5,  3,  4,  6],     # 5  is 3rd of 4 -> 75th pct
            # Needed by the height-adjusted biomech query (reach metric).
            "avg_intercept_y_vs_batter": [26, 24, 25, 27],
        }
    )


def test_league_percentiles_ranks_with_window_function(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())
    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: _fake_stance())

    result = swing_geometry.league_percentiles(101, 2024)

    assert result["found"] is True
    # Every metric places the target 3rd of 4 -> 75th percentile. This spans
    # BOTH tables: swing-path metrics and the joined-in stance metrics.
    for col, block in result["metrics"].items():
        assert block["percentile"] == 75, f"{col} expected 75, got {block['percentile']}"

    # The stance feet metrics made it through the JOIN with their values + ranks.
    assert result["metrics"]["avg_foot_sep"]["value"] == 33
    assert result["metrics"]["avg_stance_angle"]["percentile"] == 75

    # The hitter's own value is carried through alongside the rank.
    assert result["metrics"]["attack_angle"]["value"] == 15
    # Directional flags survive: bat speed is higher-is-better, tilt is neutral.
    assert result["metrics"]["avg_bat_speed"]["higher_is_better"] is True
    assert result["metrics"]["swing_tilt"]["higher_is_better"] is None


def test_league_percentiles_hitter_missing_stance(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())
    # Stance snapshot lacks id 101 -> the LEFT JOIN yields null stance for it.
    stance_without_101 = _fake_stance().query("id != 101")
    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: stance_without_101)

    result = swing_geometry.league_percentiles(101, 2024)

    # Still found via swing-path; swing metrics rank normally...
    assert result["found"] is True
    assert result["metrics"]["attack_angle"]["percentile"] == 75
    # ...but the stance metrics are cleanly None rather than a wrong number.
    assert result["metrics"]["avg_foot_sep"]["value"] is None
    assert result["metrics"]["avg_foot_sep"]["percentile"] is None


def test_player_stance_returns_row_or_none(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: _fake_stance())

    row = swing_geometry.player_stance(101, side="R")
    assert row is not None
    assert row["avg_foot_sep"] == 33
    assert row["side"] == "R"

    # Unknown id -> None, so the chart layer can skip drawing feet cleanly.
    assert swing_geometry.player_stance(999) is None


def test_player_box_position_from_swing_path(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())

    # id 101 is in the swing-path leaderboard -> box position returned.
    pos = swing_geometry.player_box_position(101, 2024, side="R")
    assert pos is not None
    assert set(pos) == {"avg_batter_x_position", "avg_batter_y_position"}

    # Unknown id -> None (caller then omits the section).
    assert swing_geometry.player_box_position(999, 2024) is None


def test_stance_biomech_height_adjusted_percentiles(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: _fake_stance())
    # Heights chosen so that dividing foot_sep by height preserves the ranking:
    # id 101 is 3rd of 4 on width_ratio -> 75 -> clamped display stays 75.
    fake_heights = {101: 70, 102: 70, 103: 70, 104: 70}
    monkeypatch.setattr(loaders, "pull_player_heights", lambda ids: fake_heights)

    bio = swing_geometry.stance_biomech(101, side="R")
    assert bio is not None
    assert bio["height_in"] == 70
    assert bio["stance_width_pct"] == 75          # 3rd of 4 on width/height
    # Percentiles are clamped to 1..99 (no "0th"/"100th").
    assert 1 <= bio["stance_width_pct"] <= 99
    assert 1 <= bio["reach_pct"] <= 99


def test_league_percentiles_unqualified_hitter(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())
    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: _fake_stance())

    # id 999 isn't in the qualified table -> found False, no invented numbers.
    result = swing_geometry.league_percentiles(999, 2024)
    assert result == {"found": False, "metrics": {}}


def test_league_baseline_caches_completed_season(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)

    calls = {"n": 0}

    def _counting_pull(*a, **k):
        calls["n"] += 1
        return _fake_league()

    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", _counting_pull)

    # First call hits the (fake) network and writes the CSV; second reads cache.
    swing_geometry.load_swing_path_baseline(2024)
    swing_geometry.load_swing_path_baseline(2024)
    assert calls["n"] == 1
    assert (tmp_path / "league_swing_path_2024.csv").exists()


# ---------------------------------------------------------------------------
# swing_profile_by_pitch() — the GROUP BY + HAVING query
# ---------------------------------------------------------------------------

def _fake_swings() -> pd.DataFrame:
    # 3 fastballs, 2 sliders, and 1 non-swing (all-null tracking) that the
    # WHERE clause must drop. Values chosen so the fastball averages are round.
    return pd.DataFrame(
        {
            "pitch_type": ["FF", "FF", "FF", "SL", "SL", "FF"],
            "attack_angle": [10, 12, 14, 5, 7, np.nan],   # last row = non-swing
            "bat_speed":    [70, 72, 74, 60, 62, np.nan],
            "intercept_ball_minus_batter_pos_x_inches": [1, 2, 3, 4, 5, np.nan],
            "intercept_ball_minus_batter_pos_y_inches": [26, 28, 30, 20, 22, np.nan],
            "events": ["single", "field_out", "home_run", "field_out", "single", None],
        }
    )


def test_swing_profile_by_pitch_groups_and_filters():
    prof = swing_geometry.swing_profile_by_pitch(_fake_swings(), min_swings=3)

    # SL has only 2 swings -> dropped by HAVING; the non-swing FF row is dropped
    # by WHERE, so FF still counts 3.
    assert list(prof["pitch_type"]) == ["FF"]
    ff = prof.iloc[0]
    assert ff["swings"] == 3
    assert ff["attack_angle"] == 12.0        # mean(10,12,14)
    assert ff["bat_speed"] == 72.0           # mean(70,72,74)
    assert ff["intercept_depth_in"] == 28.0  # mean(26,28,30)
    assert ff["intercept_side_in"] == 2.0    # mean(1,2,3)


def test_swing_profile_missing_columns_returns_empty():
    # A pre-bat-tracking pull has none of the swing columns.
    old = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95, 85]})
    assert swing_geometry.swing_profile_by_pitch(old).empty


# ---------------------------------------------------------------------------
# intercept_points() — tidy per-swing points for the scatter
# ---------------------------------------------------------------------------

def test_intercept_points_are_tidy_and_drop_non_swings():
    pts = swing_geometry.intercept_points(_fake_swings())

    # 5 real swings; the null-tracking row is gone.
    assert len(pts) == 5
    assert set(pts.columns) == {"intercept_side_in", "intercept_depth_in", "attack_angle", "pitch_type", "events"}
    assert pts["intercept_depth_in"].notna().all()


# ---------------------------------------------------------------------------
# swing_geometry_card() — the orchestrator
# ---------------------------------------------------------------------------

def test_card_flags_pre_tracking_season(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())

    # 2022 predates bat tracking: even with good columns, has_tracking is False
    # and the heavy pieces stay empty (no league pull attempted).
    card = swing_geometry.swing_geometry_card(101, _fake_swings(), season=2022)
    assert card["has_tracking"] is False
    assert card["league"] == {"found": False, "metrics": {}}
    assert card["by_pitch"].empty
    assert card["points"].empty


def test_card_full_assembly(monkeypatch, tmp_path):
    monkeypatch.setattr(swing_geometry.config, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(loaders, "pull_swing_path_leaderboard", lambda *a, **k: _fake_league())

    monkeypatch.setattr(loaders, "pull_batting_stance_leaderboard", lambda *a, **k: _fake_stance())

    card = swing_geometry.swing_geometry_card(101, _fake_swings(), season=2024, min_swings=3)
    assert card["has_tracking"] is True
    assert card["league"]["found"] is True
    assert card["league"]["metrics"]["avg_foot_sep"]["percentile"] == 75
    assert list(card["by_pitch"]["pitch_type"]) == ["FF"]
    assert len(card["points"]) == 5
