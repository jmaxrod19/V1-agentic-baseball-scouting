"""Tests for pipeline.resolve_player — name -> (id, canonical name).

No network: pybaseball.playerid_lookup is monkeypatched. Focus is the
accent-insensitive fallback (a user typing 'Jose Altuve' for 'José Altuve').
"""

import pandas as pd
import pytest

from src import pipeline


def _fuzzy_altuve() -> pd.DataFrame:
    # What playerid_lookup(..., fuzzy=True) returns for 'Altuve, Jose': the real
    # (accented) player plus a few near-miss names.
    return pd.DataFrame(
        {
            "name_last": ["altuve", "pate", "javier"],
            "name_first": ["josé", "joe", "jose"],
            "key_mlbam": [514888, 120278, 593733],
        }
    )


def _patch_lookup(monkeypatch, exact: pd.DataFrame, fuzzy: pd.DataFrame):
    exact_df, fuzzy_df = exact, fuzzy  # avoid shadowing by the `fuzzy` bool arg

    def fake(last, first=None, fuzzy=False):
        return fuzzy_df.copy() if fuzzy is True else exact_df.copy()
    monkeypatch.setattr("pybaseball.playerid_lookup", fake)


def test_strip_accents():
    assert pipeline._strip_accents("José") == "jose"
    assert pipeline._strip_accents("Rodríguez") == "rodriguez"
    assert pipeline._strip_accents("Smith") == "smith"


def test_unaccented_input_resolves_accented_player(monkeypatch):
    # Exact lookup misses (registry stores 'josé'); fuzzy + accent-strip finds him.
    _patch_lookup(monkeypatch, exact=pd.DataFrame(), fuzzy=_fuzzy_altuve())
    pid, canon = pipeline.resolve_player("Jose Altuve")
    assert pid == 514888
    assert canon == "José Altuve"


def test_exact_match_still_used_first(monkeypatch):
    # When the exact lookup already hits, we never touch the fuzzy path.
    exact = pd.DataFrame({"name_last": ["altuve"], "name_first": ["josé"], "key_mlbam": [514888]})
    _patch_lookup(monkeypatch, exact=exact, fuzzy=pd.DataFrame())
    pid, canon = pipeline.resolve_player("José Altuve")
    assert pid == 514888


def test_no_accent_match_raises_not_found(monkeypatch):
    # Fuzzy returns only genuine non-matches -> the accent filter empties them out.
    junk = pd.DataFrame({"name_last": ["mauer"], "name_first": ["joe"], "key_mlbam": [408045]})
    _patch_lookup(monkeypatch, exact=pd.DataFrame(), fuzzy=junk)
    with pytest.raises(pipeline.PlayerNotFound):
        pipeline.resolve_player("Zzz Nobody")
