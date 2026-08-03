"""Tests for the V2 swing-geometry section wired into the hitter HTML report.

We don't assert on the exact HTML — two things matter and are locked in here:
  1. the section degrades to "" (report renders without it) whenever the data
     isn't there — no inputs, or a pre-bat-tracking pull with no intercept cols;
  2. given a good pitch-level frame it emits the section with an embedded PNG.

The stance/height lookups are monkeypatched so the test never hits the network;
the chart itself renders for real (cheap, and proves the wiring end to end).
"""

import pandas as pd

from src import html_report, swing_geometry


def test_section_empty_without_inputs():
    # No pitch frame or no player id -> no section (report still renders).
    assert html_report._hitter_swing_geometry_section(None, 1, "X", 2024) == ""
    assert html_report._hitter_swing_geometry_section(pd.DataFrame(), None, "X", 2024) == ""


def test_section_disclaimer_when_pull_has_no_tracking():
    # An older pull without the intercept columns -> disclaimer, not silence.
    df = pd.DataFrame({"pitch_type": ["FF", "SL"], "release_speed": [95, 85]})
    html = html_report._hitter_swing_geometry_section(df, 123, "X", 2024)
    assert "Stance &amp; Contact Geometry" in html
    assert "2024 season" in html and "data:image" not in html


def test_section_renders_with_good_frame(monkeypatch):
    df = pd.DataFrame(
        {
            "pitch_type": ["FF"] * 20,
            "stand": ["R"] * 20,
            "bat_speed": [70.0] * 20,
            "attack_angle": [10.0] * 20,
            "intercept_ball_minus_batter_pos_x_inches": list(range(40, 60)),
            "intercept_ball_minus_batter_pos_y_inches": list(range(15, 35)),
            "events": [None] * 20,
        }
    )
    # Stance + height come from cached leaderboards in real life; stub them here.
    monkeypatch.setattr(
        swing_geometry, "player_stance",
        lambda pid, side=None: {
            "avg_batter_x_position": 34.7, "avg_batter_y_position": 20.2,
            "avg_foot_sep": 36.8, "avg_stance_angle": -2.7,
        },
    )
    monkeypatch.setattr(
        swing_geometry, "stance_biomech",
        lambda pid, side=None: {
            "height_text": "6' 2\"", "stance_width": 37.0, "stance_width_pct": 87,
        },
    )

    html = html_report._hitter_swing_geometry_section(df, 123, "Test Hitter", 2024)
    assert "Stance &amp; Contact Geometry" in html
    assert "data:image/png;base64," in html


def _tracking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pitch_type": ["FF"] * 20,
            "stand": ["R"] * 20,
            "bat_speed": [70.0] * 20,
            "attack_angle": [10.0] * 20,
            "intercept_ball_minus_batter_pos_x_inches": list(range(40, 60)),
            "intercept_ball_minus_batter_pos_y_inches": list(range(15, 35)),
        }
    )


def test_section_falls_back_to_box_position_without_stance(monkeypatch):
    # Not in the stance snapshot (no feet), but present in the swing-path
    # leaderboard -> use box position and still render the contact map.
    monkeypatch.setattr(swing_geometry, "player_stance", lambda pid, side=None: None)
    monkeypatch.setattr(
        swing_geometry, "player_box_position",
        lambda pid, season, side=None: {
            "avg_batter_x_position": 30.0, "avg_batter_y_position": 22.0,
        },
    )
    monkeypatch.setattr(swing_geometry, "stance_biomech", lambda pid, side=None: None)

    html = html_report._hitter_swing_geometry_section(_tracking_frame(), 999, "Fallback Guy", 2024)
    assert "Stance &amp; Contact Geometry" in html
    assert "data:image/png;base64," in html


def test_section_disclaimer_when_no_position_anywhere(monkeypatch):
    # Absent from BOTH the stance snapshot and the swing-path leaderboard ->
    # a disclaimer explaining why, not a silent drop.
    monkeypatch.setattr(swing_geometry, "player_stance", lambda pid, side=None: None)
    monkeypatch.setattr(swing_geometry, "player_box_position", lambda pid, season, side=None: None)
    html = html_report._hitter_swing_geometry_section(_tracking_frame(), 999, "Nobody", 2024)
    assert "Stance &amp; Contact Geometry" in html
    assert "isn't in Statcast" in html and "data:image" not in html
