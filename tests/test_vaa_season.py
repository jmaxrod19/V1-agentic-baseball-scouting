"""Tests for src/vaa_season.py (the season table) and the /vaa page in webapp.py.

No network and no real data: the league is the synthetic one from test_vaa_model.
"""

import pandas as pd
import pytest

import webapp
from src import vaa_model, vaa_season
from tests.test_vaa_model import _league


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _league()


@pytest.fixture(scope="module")
def table(data, tmp_path_factory) -> pd.DataFrame:
    """Build, write and re-read, exactly as the build script and web app do."""
    built = vaa_season.build_season_table(data, 2026, heights={p: 74 for p in range(1, 31)})
    path = tmp_path_factory.mktemp("vaa") / "table.csv"
    built.to_csv(path, index=False, float_format="%.8g")
    return pd.read_csv(path)


def _names(*names: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"role": "pitcher", "name": names, "pitches": range(100, 100 - len(names), -1)}
    )


# --- the table --------------------------------------------------------------

def test_one_row_per_pitcher_plus_one_per_hand(table):
    assert (table["role"] == "pitcher").sum() == 30
    assert table.loc[table["role"] == "league", "hand"].tolist() == ["R"]


@pytest.mark.parametrize("height", [1.8, 2.75, 3.4])
def test_table_gives_the_same_answer_as_the_pitches(data, table, height):
    direct = vaa_model.compare_fastball(data, 1, 2026, height)
    stored, pitcher_height, league_height = vaa_season.comparison_from_table(table, 1, height)

    assert (pitcher_height, league_height) == (74, 74)
    assert (stored.name, stored.notes, stored.classification) == (
        direct.name, direct.notes, direct.classification)
    assert stored.pitcher.vaa_aa == pytest.approx(direct.pitcher.vaa_aa, abs=1e-5)
    assert stored.pitcher.modeled.vaa == pytest.approx(direct.pitcher.modeled.vaa, abs=1e-5)
    assert stored.league.modeled.vaa == pytest.approx(direct.league.modeled.vaa, abs=1e-5)
    assert stored.shares == pytest.approx(direct.shares, abs=1e-5)
    assert stored.zone == pytest.approx(direct.zone)


def test_unknown_pitcher_is_a_readable_error(table):
    with pytest.raises(ValueError, match="No four-seam"):
        vaa_season.comparison_from_table(table, 999999)


# --- name search ------------------------------------------------------------

@pytest.mark.parametrize("query", ["Jose Perez", "pérez, josé", "PEREZ", "jos per"])
def test_search_ignores_accents_case_and_order(query):
    found = vaa_season.find_pitchers(_names("José Pérez", "Tarik Skubal"), query)
    assert found["name"].tolist() == ["José Pérez"]


def test_exact_name_beats_partial_matches():
    table = _names("Joe Ryan", "Joey Ryan", "Ryan Walker")
    assert vaa_season.find_pitchers(table, "joe ryan")["name"].tolist() == ["Joe Ryan"]
    assert len(vaa_season.find_pitchers(table, "ryan")) == 3


def test_same_name_returns_both_and_blank_returns_none():
    table = _names("Yunior Marte", "Yunior Marte")
    assert len(vaa_season.find_pitchers(table, "Yunior Marte")) == 2
    assert vaa_season.find_pitchers(table, "  ").empty
    assert vaa_season.find_pitchers(table, "xyz").empty


# --- the web page -----------------------------------------------------------
# The route is an ordinary function, so we call it directly: no test server and
# no extra HTTP-client dependency.

@pytest.fixture
def page(table, tmp_path, monkeypatch):
    """The real /vaa route pointed at the synthetic table; returns the HTML."""
    path = tmp_path / "table.csv"
    table.to_csv(path, index=False)
    monkeypatch.setattr(webapp, "_VAA_TABLE", path)
    webapp._vaa_table.cache_clear()
    webapp._vaa_png.cache_clear()

    def get(**params) -> str:
        response = webapp.vaa_page(**params)
        assert response.status_code == 200
        return response.body.decode()

    yield get
    webapp._vaa_table.cache_clear()          # do not leave fake data behind
    webapp._vaa_png.cache_clear()


def test_home_page_links_to_vaa():
    assert "href=/vaa" in webapp.home()


def test_search_page_lists_every_pitcher(page):
    assert page().count("<option value=") == 30


def test_a_full_name_returns_a_chart(page):
    assert "data:image/png;base64," in page(name="Test Arm1")


def test_a_partial_name_with_several_matches_asks_which(page):
    text = page(name="arm1")                                 # Arm1, Arm10..Arm19
    assert "Which pitcher?" in text and "/vaa?pitcher=10&amp;height=2.75" in text
    assert "data:image/png" not in text


@pytest.mark.parametrize("params, message", [
    ({"name": "Nobody"}, "No four-seam fastball found"),
    ({"pitcher": "999999"}, "No four-seam fastball data"),
    ({"pitcher": "abc"}, "not valid"),
    ({"name": "Test Arm1", "height": "99"}, "between 1.5 and 4"),
    ({"name": "Test Arm1", "height": "nan"}, "between 1.5 and 4"),
    ({"name": "Test Arm1", "height": "tall"}, "between 1.5 and 4"),
])
def test_bad_input_shows_a_message_not_a_crash(page, params, message):
    assert message in page(**params)


def test_typed_text_is_escaped(page):
    assert "<script>" not in page(name="<script>alert(1)</script>")


def test_the_shipped_table_is_present_and_draws():
    """The committed 2026 file must load and rebuild a comparison."""
    table = pd.read_csv(webapp._VAA_TABLE)
    pitcher_id = int(table.loc[table["role"] == "pitcher", "pitcher_id"].iloc[0])
    assert vaa_season.comparison_from_table(table, pitcher_id)[0].classification
