"""Tests for src/validate.py (the inspector) and its wiring into the pipeline.

The inspector tests build tiny frames with a known defect and assert the report
flags it at the right severity. The pipeline test proves a structurally broken
pull raises DataQualityError and is never cached.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import loaders, pipeline, validate

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "statcast_sample.csv"


# ---------------------------------------------------------------------------
# validate_statcast() — the inspector
# ---------------------------------------------------------------------------

def test_clean_frame_reports_ok():
    df = loaders.load_statcast_csv(_FIXTURE)
    report = validate.validate_statcast(df)
    assert report.ok
    assert report.errors == []


def test_empty_frame_is_an_error():
    report = validate.validate_statcast(pd.DataFrame())
    assert not report.ok
    assert any("no rows" in i.message for i in report.errors)


def test_missing_required_column_is_an_error():
    # Has rows and a date, but no pitch_type -> structural error.
    df = pd.DataFrame({"game_date": pd.to_datetime(["2024-06-01"])})
    report = validate.validate_statcast(df)
    assert not report.ok
    assert any("pitch_type" in i.message for i in report.errors)


def test_implausible_value_is_a_warning_not_error():
    # 900 mph exit velocity is impossible, but it shouldn't be fatal — the report
    # stays "ok" (no errors) while flagging a warning.
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "pitch_type": ["FF", "SL"],
            "launch_speed": [900.0, 95.0],
        }
    )
    report = validate.validate_statcast(df)
    assert report.ok  # warnings don't flip ok
    assert any("launch_speed" in i.message for i in report.warnings)


def test_all_null_column_is_a_warning():
    df = pd.DataFrame(
        {
            "game_date": pd.to_datetime(["2024-06-01"]),
            "pitch_type": ["FF"],
            "release_speed": [None],
        }
    )
    report = validate.validate_statcast(df)
    assert any("entirely null" in i.message for i in report.warnings)


def test_duplicate_rows_are_a_warning():
    row = {"game_date": pd.to_datetime("2024-06-01"), "pitch_type": "FF"}
    df = pd.DataFrame([row, row])  # two identical rows
    report = validate.validate_statcast(df)
    assert any("duplicate" in i.message for i in report.warnings)


# ---------------------------------------------------------------------------
# Pipeline integration — a broken pull must raise and never be cached.
# ---------------------------------------------------------------------------

def test_pull_player_raises_and_skips_cache_on_invalid_data(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    # Rows present but no pitch_type -> validation error.
    bad = pd.DataFrame(
        {"game_date": pd.to_datetime(["2024-06-01"]), "release_speed": [95.0]}
    )
    monkeypatch.setattr(loaders, "pull_statcast_batter", lambda *a, **k: bad)

    with pytest.raises(pipeline.DataQualityError):
        pipeline._pull_player(999, "2024-06-01", "2024-06-10", "hitter")

    # A frame that failed validation must not have been written to the cache.
    assert list(tmp_path.glob("*.csv")) == []
