"""Agentic scouting narrative — a Claude-written summary grounded in the metrics.

Two pieces make up the narrative block:

1. A **grade line** (e.g. "Raw Power 70 · Hard Contact 60 · Lift 45") — computed
   deterministically in code by mapping each metric's league percentile onto the
   20-80 scouting scale. No model involved, so it's always consistent.

2. A **paragraph** written by Claude (claude-opus-4-8) from a fact sheet the code
   builds (metric → value → percentile → grade → scout term) plus a style guide
   and a hard scope rule. Best-effort: if the API key is missing or the call
   fails, the paragraph is None and the report simply shows the grade line + its
   existing rule-based bullets.

The 20-80 ↔ percentile mapping is the standard-deviation convention documented by
FanGraphs (50 = MLB average, each 10 points = one standard deviation):
https://blogs.fangraphs.com/scouting-explained-the-20-80-scouting-scale/
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Percentile -> 20-80 grade
# ---------------------------------------------------------------------------
# Bands are (minimum percentile inclusive, grade, scout term). Derived from the
# normal-distribution breakpoints: 84th pct ≈ +1 SD = 60 "plus", 98th ≈ +3 SD =
# 80 "elite", 16th ≈ -1 SD = 40, and so on. Ordered high -> low.
_GRADE_BANDS: list[tuple[int, int, str]] = [
    (98, 80, "elite"),
    (90, 70, "plus-plus"),
    (80, 60, "plus"),
    (66, 55, "above-average"),
    (41, 50, "average"),
    (26, 45, "fringe-average"),
    (11, 40, "below-average"),
    (0, 30, "well below-average"),
]


def grade_for_percentile(pct: int) -> tuple[int, str]:
    """Map a league percentile (0-100) to a (grade, scout term) on the 20-80 scale."""
    for lo, grade, term in _GRADE_BANDS:
        if pct >= lo:
            return grade, term
    return 30, "well below-average"  # unreachable (last band starts at 0), but explicit


# ---------------------------------------------------------------------------
# Which metrics feed the grade line and the fact sheet, per report kind
# ---------------------------------------------------------------------------
# Grade-line "tools": a short, scannable header. (display label, percentile key)
_GRADE_LINE_TOOLS = {
    "hitter": [
        ("Raw Power", "max_ev"),
        ("Hard Contact", "hard_hit_rate"),
        ("Barrels", "barrel_rate"),
        ("Lift", "sweet_spot_rate"),
    ],
    "pitcher": [
        ("Fastball", "fb_velocity"),
        ("Whiff", "whiff"),
        ("Chase", "chase"),
        ("Command", "bb"),
        ("Contact", "hard_hit"),
    ],
}

# Fact-sheet rows fed to Claude. Hitters include the raw value (we have it);
# pitcher percentiles come pre-computed from Savant with no raw value attached.
# (display label, percentile key, optional profile key + formatter for the value)
_HITTER_FACTS = [
    ("Average exit velocity", "avg_ev", "avg_ev", lambda v: f"{v:.1f} mph"),
    ("Max exit velocity", "max_ev", "max_ev", lambda v: f"{v:.1f} mph"),
    ("Hard-hit rate", "hard_hit_rate", "hard_hit_rate", lambda v: f"{v * 100:.1f}%"),
    ("Barrel rate", "barrel_rate", "barrel_rate", lambda v: f"{v * 100:.1f}%"),
    ("Sweet-spot rate", "sweet_spot_rate", "sweet_spot_rate", lambda v: f"{v * 100:.1f}%"),
]
_PITCHER_FACTS = [
    ("Fastball velocity", "fb_velocity"),
    ("Fastball spin", "fb_spin"),
    ("Whiff rate", "whiff"),
    ("Chase rate", "chase"),
    ("Strikeout rate", "k"),
    ("Walk rate (command proxy)", "bb"),
    ("Hard-hit rate against", "hard_hit"),
    ("Barrel rate against", "barrel"),
    ("Expected ERA", "xera"),
    ("Expected wOBA", "xwoba"),
]


def grade_line(kind: str, percentiles: dict) -> str:
    """Build the deterministic grade line, e.g. 'Raw Power 70 · Hard Contact 60'."""
    parts = []
    for label, key in _GRADE_LINE_TOOLS.get(kind, []):
        pct = percentiles.get(key)
        if pct is None:
            continue
        grade, _ = grade_for_percentile(pct)
        parts.append(f"{label} {grade}")
    return "  ·  ".join(parts)


def _hitter_fact_lines(profile, percentiles: dict) -> list[str]:
    lines = []
    for label, pkey, vkey, fmt in _HITTER_FACTS:
        pct = percentiles.get(pkey)
        if pct is None:
            continue
        grade, term = grade_for_percentile(pct)
        val = profile[vkey] if profile is not None and vkey in profile else None
        value_str = f"{fmt(val)} — " if val is not None else ""
        lines.append(f"- {label}: {value_str}{pct}th percentile ({grade}-grade, {term})")
    return lines


def _pitcher_fact_lines(percentiles: dict) -> list[str]:
    lines = []
    for label, key in _PITCHER_FACTS:
        pct = percentiles.get(key)
        if pct is None:
            continue
        grade, term = grade_for_percentile(pct)
        lines.append(f"- {label}: {pct}th percentile ({grade}-grade, {term})")
    return lines


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a professional baseball scout writing the summary of \
an analytics-department scouting report. Write in authentic MLB scouting voice.

VOCABULARY — use these grade terms on the 20-80 scale (50 = MLB average, each 10 \
points = one standard deviation):
80 elite · 70 plus-plus · 60 plus · 55 above-average · 50 average · 45 \
fringe-average · 40 below-average · 30 well below-average.
Use natural scout descriptors: for hitters — "raw power", "loud contact", \
"barrels", "lift/carry", "bat-to-ball", "chases spin"; for pitchers — fastball \
"ride/carry/life, arm-side run, sink", breaking ball "sweep, depth, tilt, sharp, \
two-plane, downer", changeup "tumble/fade", "swing-and-miss / put-away / chase \
pitch", "plus velocity", "command".

RULES:
- Write ONE tight paragraph, 3-5 sentences.
- Ground every claim in the provided facts. Never invent numbers or observations.
- Lead with the headline strength; name the single biggest weakness honestly.
- End with a one-line profile summary based ONLY on the measured tools (e.g. "an \
elite-power profile that does its damage on contact" / "a swing-and-miss power \
arm"). Do NOT state or imply a batting-order/lineup slot (leadoff, \
middle-of-the-order, etc.) or a team role (starter, reliever, closer) — none of \
that is knowable from this data.
- This is batted-ball / pitch-tracking data only. NEVER grade, mention, or guess \
a hitter's hit tool, speed, fielding, arm, or lineup position, or a pitcher's \
delivery, mechanics, or role (starter/reliever) — there is no data on any of \
them. For pitchers, treat command only as a walk-rate proxy.
- Plain prose only — no markdown, headers, or bullet points."""


def _build_user_prompt(kind, name, hand, sample, fact_lines, extra=None) -> str:
    role = "hitter" if kind == "hitter" else "pitcher"
    facts = "\n".join(fact_lines)
    parts = [
        f"Player: {name} ({hand} {role})",
        f"Sample: {sample}",
        "",
        f"Metrics (percentile vs qualified {role}s, with 20-80 grade):",
        facts,
    ]
    if extra:
        parts += ["", extra]
    parts += ["", "Write the scouting paragraph."]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# The single network boundary (monkeypatched in tests, lazy-imports the SDK)
# ---------------------------------------------------------------------------

def _call_claude(system: str, user: str) -> str | None:
    """Send one request to Claude; return the text, or None on any failure.

    Best-effort by design: a missing key or any error returns None so the report
    still renders (with the grade line + rule-based bullets). The anthropic SDK
    is imported here, not at module top, so importing this module never requires
    the package — matching the lazy pybaseball imports in loaders.py.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return text or None
    except Exception as exc:  # noqa: BLE001 — narrative is optional; never crash a report
        print(f"[narrative] generation failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_paragraph(
    kind: str,
    *,
    name: str,
    hand: str,
    sample: str,
    percentiles: dict,
    profile=None,
    extra: str | None = None,
) -> str | None:
    """Build the fact sheet and return Claude's scouting paragraph (or None)."""
    if not percentiles:
        return None
    if kind == "hitter":
        fact_lines = _hitter_fact_lines(profile, percentiles)
    else:
        fact_lines = _pitcher_fact_lines(percentiles)
    if not fact_lines:
        return None
    user = _build_user_prompt(kind, name, hand, sample, fact_lines, extra)
    return _call_claude(_SYSTEM_PROMPT, user)
