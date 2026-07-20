"""Tests for the disk cache in src/pipeline.py (_pull_player).

No network: we monkeypatch the pybaseball-backed pull functions so a real
Statcast call never happens, and redirect the cache dir to a pytest tmp_path so
tests never touch the real data/processed/ folder.
"""

import shutil
from datetime import date
from pathlib import Path

import pytest

from src import loaders, pipeline

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "statcast_sample.csv"


def _boom(*args, **kwargs):
    # Stand-in for a network pull that must NOT run on a cache hit.
    raise AssertionError("network pull should not be called when the cache hits")


def test_pull_player_reads_from_disk_cache(monkeypatch, tmp_path):
    # Redirect the cache dir, then pre-create the exact file _pull_player expects.
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    cached = tmp_path / "pull_pitcher_111_2024-06-01_2024-06-10.csv"
    shutil.copy(_FIXTURE, cached)

    # If the code tried to pull from Statcast, this would raise — so a clean
    # return proves the data came off disk.
    monkeypatch.setattr(loaders, "pull_statcast_pitcher", _boom)

    df = pipeline._pull_player(111, "2024-06-01", "2024-06-10", "pitcher")
    assert not df.empty
    assert "pitch_type" in df.columns  # reloaded + re-cleaned correctly


def test_pull_player_writes_cache_for_completed_range(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    fake = loaders.load_statcast_csv(_FIXTURE)  # pretend this is the pull result
    monkeypatch.setattr(loaders, "pull_statcast_batter", lambda *a, **k: fake)

    pipeline._pull_player(222, "2024-06-01", "2024-06-10", "hitter")

    # A completed range must be persisted for next time.
    assert (tmp_path / "pull_hitter_222_2024-06-01_2024-06-10.csv").exists()


def test_narrative_paragraph_caches_completed_range(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    calls = {"n": 0}

    def gen():
        calls["n"] += 1
        return "A plus-plus fastball headlines the profile."

    first = pipeline._narrative_paragraph("pitcher", 42, "2024-06-01", "2024-06-10", gen)
    second = pipeline._narrative_paragraph("pitcher", 42, "2024-06-01", "2024-06-10", gen)

    assert first == second
    assert calls["n"] == 1  # second call read the cached file, no re-generation
    assert (tmp_path / "narrative_pitcher_42_2024-06-01_2024-06-10.txt").exists()


def test_narrative_paragraph_not_cached_for_today(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    today = date.today().isoformat()
    pipeline._narrative_paragraph("hitter", 7, "2024-06-01", today, lambda: "text")
    assert list(tmp_path.glob("*.txt")) == []


def test_pull_player_does_not_cache_range_ending_today(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline.config, "PROCESSED_DIR", tmp_path)
    fake = loaders.load_statcast_csv(_FIXTURE)
    monkeypatch.setattr(loaders, "pull_statcast_pitcher", lambda *a, **k: fake)

    today = date.today().isoformat()
    pipeline._pull_player(333, "2024-06-01", today, "pitcher")

    # An in-progress range (ends today) must NOT be written to disk.
    assert list(tmp_path.glob("*.csv")) == []
