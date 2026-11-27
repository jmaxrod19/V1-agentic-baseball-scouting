"""Tests for src/vaa_season.py (the season table) and the /vaa page in webapp.py.

No network and no real data: the league is the synthetic one from test_vaa_model.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import webapp
from src import loaders, vaa_charts, vaa_model, vaa_season
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
        {"role": "pitcher", "part": "all", "name": names,
         "pitches": range(100, 100 - len(names), -1)}
    )


# --- the table --------------------------------------------------------------

def test_zone_part_splits_the_batters_own_zone_by_height():
    heights = [0.9, 1.0, 1.9, 2.0, 2.9, 3.0, 4.0, 4.1, np.nan]
    pitches = pd.DataFrame({"plate_z": heights, "sz_bot": 1.0, "sz_top": 4.0})
    assert vaa_model.zone_part(pitches).tolist() == [
        "below", "bottom", "bottom", "middle", "middle", "top", "top", "above", ""]


def test_one_row_per_pitcher_and_hand_for_every_part(table):
    for part in vaa_model.ZONE_PARTS:
        here = table[table["part"] == part]
        assert (here["role"] == "pitcher").sum() == 30
        assert here.loc[here["role"] == "league", "hand"].tolist() == ["R"]


@pytest.mark.parametrize("part", ["top", "middle", "bottom"])
def test_a_third_uses_only_the_pitches_that_crossed_there(data, table, part):
    inside = data[vaa_model.zone_part(data) == part]
    stored, _, _ = vaa_season.comparison_from_table(table, 1, part)
    direct = vaa_model.compare_fastball(inside, 1, 2026, stored.comparison_height)

    assert stored.zone_part == part
    assert stored.pitcher.pitches == (inside["pitcher"] == 1).sum()
    assert stored.league.pitches == len(inside)          # league from the same third
    assert stored.pitcher.observed_vaa == pytest.approx(direct.pitcher.observed_vaa, abs=1e-5)
    assert stored.pitcher.vaa_aa == pytest.approx(direct.pitcher.vaa_aa, abs=1e-5)
    # Drawn at the league's typical crossing height within that third.
    assert stored.comparison_height == round(float(direct.league.plate_height), 1)


def test_higher_thirds_are_drawn_higher_and_arrive_flatter(table):
    top, bottom = (vaa_season.comparison_from_table(table, 2, p)[0] for p in ("top", "bottom"))
    assert top.comparison_height > bottom.comparison_height
    assert top.league.observed_vaa > bottom.league.observed_vaa      # closer to zero


def test_chart_names_and_shades_the_chosen_third(table):
    c, height, league_height = vaa_season.comparison_from_table(table, 1, "top")
    fig = vaa_charts.fastball_vaa_chart(c, height, league_height)
    words = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    shaded = [p for ax in fig.axes for p in ax.patches if p.get_gid() == "zone-third"]
    plt.close(fig)

    assert "Top third of the zone only" in words
    bottom, top = c.zone
    assert len(shaded) == 1
    assert shaded[0].get_y() + shaded[0].get_height() == pytest.approx(top)
    assert shaded[0].get_height() == pytest.approx((top - bottom) / 3)


@pytest.mark.parametrize("part, words", [
    ("all", "Why approach angle matters"), ("top", "where flat plays"),
    ("middle", "less room for error"), ("bottom", "a different job"),
])
def test_each_part_of_the_zone_has_its_own_explanation(part, words):
    assert words in webapp._takeaway(part)


def test_the_explanation_informs_without_quoting_results():
    """The page text teaches why VAA matters: no stats, percentages or season."""
    text = " ".join(webapp._takeaway(p) for p in vaa_model.ZONE_PARTS) + webapp._VAA_STORY
    words = re.sub(r"<[^>]+>", " ", text)          # drop the HTML tags (h2, h3...)
    assert not re.search(r"\d|%", words)


@pytest.mark.parametrize("height", [1.8, 2.75, 3.4])
def test_table_gives_the_same_answer_as_the_pitches(data, table, height):
    direct = vaa_model.compare_fastball(data, 1, 2026, height)
    stored, pitcher_height, league_height = vaa_season.comparison_from_table(
        table, 1, "all", height)

    assert (pitcher_height, league_height) == (74, 74)
    assert (stored.name, stored.notes, stored.classification) == (
        direct.name, direct.notes, direct.classification)
    assert stored.pitcher.vaa_aa == pytest.approx(direct.pitcher.vaa_aa, abs=1e-5)
    assert stored.pitcher.modeled.vaa == pytest.approx(direct.pitcher.modeled.vaa, abs=1e-5)
    assert stored.league.modeled.vaa == pytest.approx(direct.league.modeled.vaa, abs=1e-5)
    assert stored.shares == pytest.approx(direct.shares, abs=1e-5)
    # The card rankings: same other pitchers, same answer from either route.
    assert stored.rank_pool == direct.rank_pool == 29
    assert set(stored.ranks) == set(direct.ranks) == set(vaa_model.RANKED_TRAITS)
    for trait in ("release_height", "extension"):        # others are identical for all
        assert stored.ranks[trait] == pytest.approx(direct.ranks[trait])
    assert stored.zone == pytest.approx(direct.zone)


def test_position_players_leave_the_list_and_the_league_average(data, table):
    heights = {p: 74 for p in range(1, 31)}
    without = vaa_season.build_season_table(data, 2026, heights, position_players={1})
    everyone = without[without["part"] == "all"]
    league = everyone.loc[everyone["role"] == "league", "pitches"].iloc[0]
    before = table.loc[(table["role"] == "league") & (table["part"] == "all"), "pitches"].iloc[0]

    assert 1 not in set(everyone.loc[everyone["role"] == "pitcher", "pitcher_id"])
    assert (everyone["role"] == "pitcher").sum() == 29
    assert league == before - (data["pitcher"] == 1).sum()


def test_position_lookup_keeps_pitchers_and_two_way_players(monkeypatch):
    positions = {1: "P", 2: "TWP", 3: "RF", 4: "C"}
    monkeypatch.setattr(loaders, "_fetch_statsapi_people", lambda ids: [
        {"id": i, "primaryPosition": {"abbreviation": positions[i]}} for i in ids])
    assert loaders.pull_position_players([1, 2, 3, 4]) == {3, 4}


def test_a_failed_position_lookup_stops_the_build(monkeypatch):
    monkeypatch.setattr(loaders, "_fetch_statsapi_people", lambda ids: [])
    with pytest.raises(RuntimeError, match="position lookup failed"):
        loaders.pull_position_players([1])


def test_unknown_pitcher_is_a_readable_error(table):
    with pytest.raises(ValueError, match="No four-seam"):
        vaa_season.comparison_from_table(table, 999999)


# --- is the four-seamer the main pitch? --------------------------------------

def test_pitch_mix_finds_the_main_pitch_and_ties_go_to_the_four_seamer():
    data = pd.DataFrame({
        "pitcher": [1] * 4 + [2] * 4 + [3] * 2,
        "pitch_type": ["SI", "SI", "SI", "FF", "FF", "FF", "SL", "SL", "SL", None],
        "game_year": 2026, "game_type": "R",
    })
    mix = vaa_model.pitch_mix(data, 2026)
    assert mix.loc[1].tolist() == [25.0, "SI", 75.0]
    assert mix.loc[2].tolist() == [50.0, "FF", 50.0]
    assert mix.loc[3].tolist() == [0.0, "SL", 100.0]     # the unlabeled pitch is ignored


def test_only_a_secondary_four_seamer_is_flagged():
    words = vaa_model.not_primary_words
    assert words(1.7, "SI", 52.1) == ("sinker", "2%", "52%")
    assert words(33.8, "SL", 34.2) == ("slider", "33.8%", "34.2%")
    assert words(0.4, "CU", 60.0)[1] == "under 1%"
    assert words(55.0, "FF", 55.0) is None
    assert words(np.nan, np.nan, np.nan) is None          # unknown mix: say nothing


def test_chart_and_page_say_when_it_is_not_the_main_pitch(page, table, tmp_path):
    one = table["pitcher_id"] == 1
    flagged = table.assign(ff_pct=table["ff_pct"].where(~one, 2.0),
                           main_pitch=table["main_pitch"].where(~one, "SI"),
                           main_pct=table["main_pct"].where(~one, 52.0))
    c, height, league_height = vaa_season.comparison_from_table(flagged, 1)
    fig = vaa_charts.fastball_vaa_chart(c, height, league_height)
    words = " ".join(t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)
    assert "NOT HIS PRIMARY PITCH" in words and "Sinker 52%" in words

    flagged.to_csv(tmp_path / "table.csv", index=False)
    assert "His main pitch was the sinker (52%)" in page(name="Test Arm1")
    assert "Not his primary pitch" not in page(name="Test Arm2")


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


def test_home_page_promotes_vaa_with_a_feature_card():
    text = webapp.home()
    assert "class=feature href=/vaa" in text and "Explore Fastball VAA" in text
    # The scouting report comes first; the advert follows its whole form.
    assert text.index("Build a scouting report") < text.index("</form>") < text.index("class=feature")


def test_search_bar_sits_above_the_results(page):
    chart = page(name="Test Arm1")
    assert chart.count("<form") == 1
    assert chart.index("<form") < chart.index("class=zonepanel") < chart.index("data:image/png")
    several = page(name="arm1")
    assert several.index("<form") < several.index("Which pitcher?")


def test_search_page_lists_every_pitcher_and_has_no_height_box(page):
    text = page()
    assert text.count("<option value=") == 30
    assert "type=number" not in text


def test_chart_page_offers_each_part_of_the_zone(page):
    text = page(name="Test Arm1", zone="top")
    for part in vaa_model.ZONE_PARTS:
        assert f"/vaa?pitcher=1&amp;zone={part}'" in text
    assert "class=' on' aria-current=true href='/vaa?pitcher=1&amp;zone=top'" in text
    assert "data:image/png;base64," in text


def test_a_third_with_no_pitches_keeps_the_picker(page, table, tmp_path):
    gap = (table["part"] == "top") & (table["pitcher_id"] == 1)
    table[~gap].to_csv(tmp_path / "table.csv", index=False)
    text = page(pitcher="1", zone="top")
    assert "in the top third of the zone" in text and "zone=middle" in text
    assert "data:image/png" not in text


def test_a_partial_name_with_several_matches_asks_which(page):
    text = page(name="arm1")                                 # Arm1, Arm10..Arm19
    assert "Which pitcher?" in text and "/vaa?pitcher=10&amp;zone=all" in text
    assert "data:image/png" not in text


@pytest.mark.parametrize("params, message", [
    ({"name": "Nobody"}, "No four-seam fastball found"),
    ({"pitcher": "999999"}, "No four-seam fastball data"),
    ({"pitcher": "abc"}, "not valid"),
    ({"name": "Test Arm1", "zone": "ankles"}, "not an option"),
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


def test_the_shipped_table_has_no_position_players():
    """Victor Robles (645302) and Jhonny Pereda (640902) pitched in 2026."""
    table = pd.read_csv(webapp._VAA_TABLE)
    assert not table["pitcher_id"].isin([645302, 640902]).any()


def test_method_notes_are_on_the_page_but_folded_away(page):
    text = page(name="Test Arm1")
    assert "<details class=method><summary>How this is calculated</summary>" in text
    assert text.index("data:image/png") < text.index("class=method")
