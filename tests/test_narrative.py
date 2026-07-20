"""Tests for src/narrative.py — the agentic scouting narrative.

No network: `_call_claude` (the single API boundary) is monkeypatched, and the
no-API-key path is asserted to short-circuit before any SDK import.
"""

import pandas as pd
import pytest

from src import narrative


# ---------------------------------------------------------------------------
# percentile -> 20-80 grade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pct,grade,term",
    [
        (99, 80, "elite"),
        (95, 70, "plus-plus"),
        (84, 60, "plus"),
        (70, 55, "above-average"),
        (50, 50, "average"),
        (30, 45, "fringe-average"),
        (20, 40, "below-average"),
        (5, 30, "well below-average"),
    ],
)
def test_grade_for_percentile(pct, grade, term):
    assert narrative.grade_for_percentile(pct) == (grade, term)


# ---------------------------------------------------------------------------
# grade line (deterministic header)
# ---------------------------------------------------------------------------

def test_hitter_grade_line():
    pct = {"max_ev": 95, "hard_hit_rate": 84, "barrel_rate": 50, "sweet_spot_rate": 30}
    assert narrative.grade_line("hitter", pct) == (
        "Raw Power 70  ·  Hard Contact 60  ·  Barrels 50  ·  Lift 45"
    )


def test_pitcher_grade_line():
    pct = {"fb_velocity": 100, "whiff": 100, "chase": 99, "bb": 44, "hard_hit": 92}
    assert narrative.grade_line("pitcher", pct) == (
        "Fastball 80  ·  Whiff 80  ·  Chase 80  ·  Command 50  ·  Contact 70"
    )


def test_grade_line_skips_missing_metrics():
    # Only two of the four hitter tools have percentiles -> only those appear.
    assert narrative.grade_line("hitter", {"max_ev": 92, "sweet_spot_rate": 70}) == (
        "Raw Power 70  ·  Lift 55"
    )


# ---------------------------------------------------------------------------
# fact sheets
# ---------------------------------------------------------------------------

def test_hitter_fact_lines_include_value_and_grade():
    profile = pd.Series({"max_ev": 116.2, "hard_hit_rate": 0.55})
    lines = narrative._hitter_fact_lines(profile, {"max_ev": 97, "hard_hit_rate": 84})
    joined = "\n".join(lines)
    assert "Max exit velocity: 116.2 mph — 97th percentile (70-grade, plus-plus)" in joined
    assert "Hard-hit rate: 55.0% — 84th percentile (60-grade, plus)" in joined


def test_pitcher_fact_lines_are_percentile_only():
    lines = narrative._pitcher_fact_lines({"fb_velocity": 100, "bb": 44})
    joined = "\n".join(lines)
    assert "Fastball velocity: 100th percentile (80-grade, elite)" in joined
    assert "44th percentile (50-grade, average)" in joined


def test_pitcher_fact_lines_spell_out_direction_for_inverted_stats():
    # Regression test: a real Ohtani report had Claude read a high hard-hit
    # percentile (= he SUPPRESSES hard contact well) as if it meant he ALLOWS a
    # lot of hard contact, inventing a fake "vulnerability" for a pitcher with
    # elite results. The labels for every stat where Savant's raw name reads
    # "bad-sounding" but is already flipped (high=good) must say so explicitly.
    lines = "\n".join(narrative._pitcher_fact_lines(
        {"hard_hit": 90, "barrel": 90, "xera": 90, "xwoba": 90, "bb": 90}
    ))
    assert "suppresses hard contact better" in lines
    assert "suppresses barrels better" in lines
    assert "lower/better xERA" in lines
    assert "lower/better xwOBA" in lines
    assert "fewer walks, better command" in lines


def test_system_prompt_states_percentile_direction_explicitly():
    prompt = narrative._SYSTEM_PROMPT
    assert "HIGHER percentile always means BETTER" in prompt
    lowered = prompt.lower()
    assert "hard-hit" in lowered and "strength, not a weakness" in lowered


def test_system_prompt_forbids_manufactured_weakness():
    prompt = narrative._SYSTEM_PROMPT
    assert "below average (45 or lower)" in prompt


# ---------------------------------------------------------------------------
# generate_paragraph — the orchestration, with the API boundary stubbed
# ---------------------------------------------------------------------------

def test_generate_paragraph_returns_model_text(monkeypatch):
    captured = {}

    def fake_call(system, user):
        captured["system"] = system
        captured["user"] = user
        return "Plus-plus raw power headlines the profile."

    monkeypatch.setattr(narrative, "_call_claude", fake_call)
    out = narrative.generate_paragraph(
        "hitter", name="Test Hitter", hand="Right-handed",
        sample="200 batted balls", percentiles={"max_ev": 97},
        profile=pd.Series({"max_ev": 116.0}),
    )
    assert out == "Plus-plus raw power headlines the profile."
    # The fact sheet actually reached the prompt.
    assert "Max exit velocity" in captured["user"]


def test_generate_paragraph_none_when_no_percentiles(monkeypatch):
    # Must not even call the API when there's nothing to describe.
    monkeypatch.setattr(narrative, "_call_claude", lambda s, u: pytest.fail("called"))
    assert narrative.generate_paragraph(
        "hitter", name="x", hand="R", sample="s", percentiles={}
    ) is None


def test_call_claude_returns_none_without_api_key(monkeypatch):
    # No key -> None, and it never reaches the (unimported) SDK.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert narrative._call_claude("sys", "user") is None


def test_prompt_forbids_lineup_and_role_guesses():
    # The data can't support batting order or team role — the prompt must forbid
    # both (a "middle-of-the-order bat" / "closer" projection would be a guess).
    prompt = narrative._SYSTEM_PROMPT.lower()
    assert "lineup" in prompt
    assert "starter" in prompt and "reliever" in prompt
    # The old, over-reaching instruction to project a role must be gone.
    assert "role projection" not in prompt
