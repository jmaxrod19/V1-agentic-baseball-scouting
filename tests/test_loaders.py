"""Tests for src/loaders.py — the "messy CSV in, clean DataFrame out" layer.

These run fully offline against the tiny sample CSVs in tests/fixtures/, so
they're fast and deterministic (no network, no pybaseball).
"""

from pathlib import Path

import pandas as pd
import pytest

from src import loaders

# Sample CSVs live next to the tests (tests/fixtures/), NOT in data/raw/ — that
# folder is gitignored, so anything there is absent in CI. Committing the
# fixtures beside the tests keeps the suite self-contained and reproducible.
_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_STATCAST = _FIXTURES / "statcast_sample.csv"
_SAVANT = _FIXTURES / "savant_sample.csv"
_FANGRAPHS = _FIXTURES / "fangraphs_sample.csv"


# ---------------------------------------------------------------------------
# Statcast loader
# ---------------------------------------------------------------------------

def test_statcast_parses_date_column_to_datetime():
    df = loaders.load_statcast_csv(_STATCAST)
    # game_date must come back as real datetimes, not strings, so downstream
    # date math works.
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])


def test_statcast_coerces_value_columns_to_numeric():
    df = loaders.load_statcast_csv(_STATCAST)
    # Every listed value column should be a number dtype even though the CSV
    # holds them as text (and some cells are blank).
    for col in ("release_speed", "launch_speed", "bat_speed"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"


def test_statcast_blank_cells_become_nan():
    df = loaders.load_statcast_csv(_STATCAST)
    # Skubal's row has an empty release_speed; a blank must clean to NaN, not
    # crash or become the string "".
    skubal = df[df["player_name"] == "Skubal, Tarik"].iloc[0]
    assert pd.isna(skubal["release_speed"])


def test_statcast_rows_sorted_chronologically():
    df = loaders.load_statcast_csv(_STATCAST)
    # The CSV lists 2024-06-15 rows before the 2024-06-10 row; the cleaner sorts
    # by date/game/at-bat/pitch, so the 06-10 pitch (Skubal's CH) must land first.
    assert df.iloc[0]["pitch_type"] == "CH"
    # game_date must be non-decreasing across the whole frame.
    assert df["game_date"].is_monotonic_increasing


def test_statcast_preserves_all_rows():
    df = loaders.load_statcast_csv(_STATCAST)
    assert len(df) == 3  # cleaning tidies types/order, never drops rows


# ---------------------------------------------------------------------------
# Savant loader (shares the Statcast cleaning core)
# ---------------------------------------------------------------------------

def test_savant_loads_with_same_cleaning():
    df = loaders.load_savant_csv(_SAVANT)
    # Savant export shares Statcast's schema + cleaner, so the same guarantees
    # hold: datetime dates and numeric values.
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
    assert pd.api.types.is_numeric_dtype(df["launch_speed"])


# ---------------------------------------------------------------------------
# FanGraphs loader (its own quirks: %, $, commas, a leading rank column)
# ---------------------------------------------------------------------------

def test_fangraphs_drops_leading_rank_column():
    df = loaders.load_fangraphs_csv(_FANGRAPHS)
    # The unnamed rank column pandas reads as "Unnamed: 0" must be dropped.
    assert not any(c.startswith("Unnamed") for c in df.columns)
    assert list(df.columns)[0] == "Name"


def test_fangraphs_strips_percent_signs():
    df = loaders.load_fangraphs_csv(_FANGRAPHS)
    judge = df[df["Name"] == "Aaron Judge"].iloc[0]
    # "28.5 %" -> 28.5 by default (displayed value kept, not converted).
    assert judge["K%"] == pytest.approx(28.5)


def test_fangraphs_strips_dollar_and_commas():
    df = loaders.load_fangraphs_csv(_FANGRAPHS)
    # "$90.4" -> 90.4 (dollar stripped); "1,234" -> 1234.0 (comma stripped).
    assert df[df["Name"] == "Aaron Judge"].iloc[0]["Dol"] == pytest.approx(90.4)
    assert df[df["Name"] == "Bobby Witt Jr."].iloc[0]["Dol"] == pytest.approx(1234.0)


def test_fangraphs_keeps_text_columns_as_text():
    df = loaders.load_fangraphs_csv(_FANGRAPHS)
    # Name must stay a real string, not get coerced toward numbers.
    assert df.iloc[0]["Name"] == "Aaron Judge"


def test_fangraphs_as_proportion_divides_percents_by_100():
    df = loaders.load_fangraphs_csv(_FANGRAPHS, as_proportion=True)
    # With as_proportion=True, "28.5 %" -> 0.285 for math/joins with Statcast.
    judge = df[df["Name"] == "Aaron Judge"].iloc[0]
    assert judge["K%"] == pytest.approx(0.285)
