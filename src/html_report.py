"""
HTML report layer: render pitcher metrics into the front-office report layout.

This mirrors the V1 browser report design (cream page, maroon accent rule,
Georgia serif, bordered summary cards, and pandas-styled tables). Like report.py
it is a pure display layer: every decimal→percentage conversion happens here,
never in metrics.py.

Charts are added in a later pass; this module renders the header, summary cards,
the auto-written narrative, and the three data tables.

Public functions:
    pitcher_html_report(...) -> str      (full HTML document)
    save_html_report(text, path)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from . import config
    from .report import pitch_display_name
except ImportError:  # direct-script execution fallback
    import config
    from report import pitch_display_name


# ===========================================================================
# Style — copied to match the V1 report exactly (design tokens in :root)
# ===========================================================================

_CSS = """
    :root {
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2a34;
      --muted: #6b7280;
      --accent: #8b1e3f;
      --line: #d7d2c8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.5;
    }
    .page { max-width: 1100px; margin: 0 auto; padding: 32px 24px 48px; }
    .header {
      border-bottom: 3px solid var(--accent);
      padding-bottom: 16px;
      margin-bottom: 24px;
    }
    .eyebrow {
      font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em;
      color: var(--muted); margin-bottom: 8px;
    }
    h1, h2, h3 { margin: 0 0 12px; }
    h1 { font-size: 34px; line-height: 1.1; }
    h2 {
      margin-top: 28px; font-size: 22px;
      border-bottom: 1px solid var(--line); padding-bottom: 8px;
    }
    h3 { font-size: 16px; }
    p { margin: 0; }
    .subhead { color: var(--muted); font-size: 16px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px; margin: 20px 0 12px;
    }
    .card {
      background: var(--panel); border: 1px solid var(--line);
      padding: 14px; border-radius: 8px;
    }
    .card-label {
      font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
      color: var(--muted); margin-bottom: 6px;
    }
    .card-value { font-size: 18px; font-weight: 700; }
    .section { margin-top: 24px; }
    .summary-list { margin: 0; padding-left: 20px; }
    .summary-list li { margin-bottom: 10px; }
    .report-table {
      width: 100%; border-collapse: collapse; background: var(--panel);
      border: 1px solid var(--line); border-radius: 8px; overflow: hidden;
      font-size: 14px;
    }
    .report-table th, .report-table td {
      padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left;
    }
    .report-table th {
      background: #ece6da; font-size: 12px;
      text-transform: uppercase; letter-spacing: 0.05em;
    }
    .report-table tr:last-child td { border-bottom: none; }
    .visual-grid { display: grid; grid-template-columns: 1fr; gap: 18px; }
    .visual-block {
      background: var(--panel); border: 1px solid var(--line);
      padding: 14px; border-radius: 8px; page-break-inside: avoid;
    }
    .visual-block img {
      width: 100%; height: auto; display: block;
      border: 1px solid var(--line); background: white;
    }
    .glossary {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; padding: 4px 18px 16px; margin: 0;
    }
    .glossary dt { font-weight: 700; margin-top: 14px; }
    .glossary dd { margin: 3px 0 0; color: var(--muted); }
    .glossary-src { margin-top: 12px; color: var(--muted); font-size: 12px; font-style: italic; }
    .footer-note { margin-top: 28px; color: var(--muted); font-size: 13px; }
    @media print {
      body { background: white; }
      .page { max-width: none; padding: 0; }
      .card, .report-table, .visual-block { break-inside: avoid; }
    }
"""


# ===========================================================================
# Small display formatters (return "NaN" on missing, mirroring the V1 report)
# ===========================================================================

def _fmt(val: float | None, decimals: int, scale: float = 1.0) -> str:
    """Format a number to fixed decimals, scaling first (e.g. 100x for %)."""
    if val is None or pd.isna(val):
        return "NaN"
    return f"{val * scale:.{decimals}f}"


def _pct(val: float | None, decimals: int = 1) -> str:
    """Decimal rate (0.637) -> percent string ('63.7')."""
    return _fmt(val, decimals, scale=100.0)


def _rate3(val: float | None) -> str:
    """Rate stat (0.385) -> baseball-style '.385' (leading zero dropped).

    Assumes 0 <= val < 10 (true for wOBA/BA-scale stats). Values >= 10 or
    negative would be mis-formatted by the leading-zero strip.
    """
    if val is None or pd.isna(val):
        return "NaN"
    return f"{val:.3f}".lstrip("0") or ".000"


# ===========================================================================
# Card + summary computation
# ===========================================================================

def _pitch_label_pct(row: pd.Series, rate_col: str) -> str:
    """'4-Seam Fastball (63.7%)' from a row's pitch_type + a rate column."""
    name = pitch_display_name(row["pitch_type"])
    return f"{name} ({_pct(row[rate_col])}%)"


def _best_row(arsenal: pd.DataFrame, rate_col: str) -> pd.Series | None:
    """Row with the highest value in `rate_col`, or None if the column is all-NaN.

    We must drop NaNs before idxmax: on an all-NaN column idxmax returns NaN
    (and `arsenal.loc[NaN]` then raises KeyError). That happens for real —
    a pitcher with zero swings, zero out-of-zone pitches, or a pull missing
    the `zone`/`description` columns — so the guard is not theoretical.
    """
    valid = arsenal[arsenal[rate_col].notna()]
    if valid.empty:
        return None
    return valid.loc[valid[rate_col].idxmax()]


def _build_cards(
    pitcher_name: str,
    throws: str,
    arsenal: pd.DataFrame,
) -> list[tuple[str, str]]:
    """Return the six (label, value) summary cards.

    arsenal is assumed sorted by usage descending, so row 0 is the primary
    pitch. Best-whiff/best-chase fall back to 'N/A' when no pitch has data.
    """
    primary = arsenal.iloc[0]
    best_whiff = _best_row(arsenal, "whiff_rate")
    best_chase = _best_row(arsenal, "chase_rate")

    return [
        ("Pitcher", pitcher_name),
        ("Handedness", throws),
        ("Pitch Mix", str(len(arsenal))),
        ("Primary Pitch", _pitch_label_pct(primary, "usage")),
        ("Best Whiff Pitch",
         _pitch_label_pct(best_whiff, "whiff_rate") if best_whiff is not None else "N/A"),
        ("Best Chase Pitch",
         _pitch_label_pct(best_chase, "chase_rate") if best_chase is not None else "N/A"),
    ]


def _zone_descriptor(in_zone_rate: float | None) -> str:
    """Qualitative read of a pitch's in-zone tendency, matching V1 phrasing."""
    if in_zone_rate is None or pd.isna(in_zone_rate):
        return "an unclear in-zone profile"
    if in_zone_rate >= 0.55:
        return "a zone-heavy profile"
    if in_zone_rate >= 0.45:
        return "a fairly balanced in-zone vs out-of-zone profile"
    return "a chase-oriented, out-of-zone profile"


def _handedness_lean(splits: pd.DataFrame) -> str | None:
    """Find the pitch with the largest usage gap between batter hands.

    Returns a sentence like 'Slider is featured more often versus right-handed
    hitters (48.4% vs 34.8%).', or None if there aren't two hands to compare.
    """
    if splits.empty or splits["stand"].nunique() < 2:
        return None

    # usage per (pitch_type, hand); pivot so each pitch has an L and R column.
    wide = splits.pivot_table(
        index="pitch_type", columns="stand", values="usage", aggfunc="first"
    )
    if "L" not in wide.columns or "R" not in wide.columns:
        return None

    wide = wide.fillna(0.0)

    # Only consider pitches thrown at least this often to BOTH hands. Without
    # this floor, a rare show-me pitch he happens to never throw to one side
    # (e.g. 8% vs 0%) posts a big fractional gap and buries the genuinely
    # differentiated pitch (e.g. slider 48% vs 35%). 5% ≈ a real part of the mix.
    min_usage = 0.05
    eligible = wide[(wide["L"] >= min_usage) & (wide["R"] >= min_usage)]
    if eligible.empty:
        return None

    gap = (eligible["R"] - eligible["L"]).abs()
    pitch = gap.idxmax()
    r_use, l_use = eligible.loc[pitch, "R"], eligible.loc[pitch, "L"]

    if r_use >= l_use:
        hand, more, less = "right", r_use, l_use
    else:
        hand, more, less = "left", l_use, r_use

    name = pitch_display_name(pitch)
    return (
        f"{name} is featured more often versus {hand}-handed hitters "
        f"({_pct(more)}% vs {_pct(less)}%)."
    )


def _build_summary(
    arsenal: pd.DataFrame,
    location: pd.DataFrame,
    splits: pd.DataFrame,
) -> list[str]:
    """Assemble the written-summary bullet strings (mirrors the V1 report)."""
    bullets: list[str] = []

    # 1. Primary mix — top three pitches by usage.
    top = arsenal.head(3)
    mix = ", ".join(_pitch_label_pct(r, "usage") for _, r in top.iterrows())
    bullets.append(f"Primary mix: {mix}.")

    # 2. Best swing-and-miss and best chase pitch. Build each clause only if
    #    that rate exists, so a data-sparse pull degrades gracefully instead
    #    of asserting a "best" pitch that has no underlying swings.
    best_whiff = _best_row(arsenal, "whiff_rate")
    best_chase = _best_row(arsenal, "chase_rate")
    clauses: list[str] = []
    if best_whiff is not None:
        clauses.append(
            f"Best swing-and-miss pitch: {pitch_display_name(best_whiff['pitch_type'])} "
            f"({_pct(best_whiff['whiff_rate'])}% whiff rate)."
        )
    if best_chase is not None:
        clauses.append(
            f"Best chase generator: {pitch_display_name(best_chase['pitch_type'])} "
            f"({_pct(best_chase['chase_rate'])}% chase rate)."
        )
    if clauses:
        bullets.append(" ".join(clauses))

    # 3. Handedness lean, if we can compute it.
    lean = _handedness_lean(splits)
    if lean:
        bullets.append(lean)

    # 4. In-zone tendency of the primary pitch.
    primary_code = arsenal.iloc[0]["pitch_type"]
    loc_row = location[location["pitch_type"] == primary_code]
    if not loc_row.empty:
        izr = loc_row.iloc[0]["in_zone_rate"]
        bullets.append(
            f"{pitch_display_name(primary_code)}, the most-used pitch, shows "
            f"{_zone_descriptor(izr)} ({_pct(izr)}% in-zone rate)."
        )

    return bullets


# ===========================================================================
# Table rendering (build a display DataFrame, then pandas .to_html)
# ===========================================================================

def _display_names(codes: pd.Series) -> pd.Series:
    """Map a column of pitch-type codes to full display names."""
    return codes.map(pitch_display_name)


def _table_html(display_df: pd.DataFrame) -> str:
    """Render a fully-formatted (string-valued) DataFrame to report HTML.

    escape=False is safe ONLY because every cell here is either a number
    formatted by _fmt/_pct or a pitch name from the internal _PITCH_NAMES
    dict — none of it is untrusted input. If you ever add a free-text column
    (scouting notes, a team/batter name from a CSV), escape it first or drop
    escape=False, or a stray '<' will break the layout / allow HTML injection.
    """
    return display_df.to_html(
        index=False,
        classes="report-table",
        border=0,
        escape=False,
        na_rep="NaN",
    )


def _arsenal_table(arsenal: pd.DataFrame) -> str:
    disp = pd.DataFrame({
        "Pitch":      _display_names(arsenal["pitch_type"]),
        "Usage %":    arsenal["usage"].map(lambda v: _pct(v)),
        "Velo (mph)": arsenal["avg_velo"].map(lambda v: _fmt(v, 1)),
        "HB (in)":    arsenal["h_break_in"].map(lambda v: _fmt(v, 1)),
        "VB (in)":    arsenal["v_break_in"].map(lambda v: _fmt(v, 1)),
        "Whiff %":    arsenal["whiff_rate"].map(lambda v: _pct(v)),
        "Chase %":    arsenal["chase_rate"].map(lambda v: _pct(v)),
    })
    return _table_html(disp)


def _handedness_table(splits: pd.DataFrame) -> str:
    disp = pd.DataFrame({
        "Batter Hand": splits["stand"],
        "Pitch":       _display_names(splits["pitch_type"]),
        "Usage %":     splits["usage"].map(lambda v: _pct(v)),
        "Velo (mph)":  splits["avg_velo"].map(lambda v: _fmt(v, 1)),
        "Whiff %":     splits["whiff_rate"].map(lambda v: _pct(v)),
        "Chase %":     splits["chase_rate"].map(lambda v: _pct(v)),
    })
    return _table_html(disp)


def _location_table(location: pd.DataFrame) -> str:
    disp = pd.DataFrame({
        "Pitch":        _display_names(location["pitch_type"]),
        "Avg Plate X":  location["avg_plate_x"].map(lambda v: _fmt(v, 2)),
        "Avg Plate Z":  location["avg_plate_z"].map(lambda v: _fmt(v, 2)),
        "In-Zone %":    location["in_zone_rate"].map(lambda v: _pct(v, 2)),
    })
    return _table_html(disp)


# ===========================================================================
# Glossary — official stat definitions (MLB Statcast glossary / FanGraphs)
# ===========================================================================
# Only the stats each report actually shows are listed. Wording follows the
# authoritative sources so the definitions can be trusted, not paraphrased loosely.

_PITCHER_GLOSSARY: list[tuple[str, str]] = [
    ("Usage %",
     "The share of a pitcher's total pitches thrown of a given pitch type."),
    ("Velocity (mph)",
     "Average release speed of the pitch, measured as it leaves the pitcher's hand."),
    ("Horizontal Break (HB, in)",
     "Average side-to-side movement of the pitch, in inches, shown arm-side-positive: "
     "a right-hander's arm-side run is positive and glove-side is negative "
     "(mirrored for left-handers)."),
    ("Vertical Break (VB, in)",
     "Average induced vertical break — movement caused by spin relative to a "
     "hypothetical spinless pitch, with gravity removed. Positive reads as “rise,” "
     "negative as extra drop."),
    ("Whiff %",
     "Swinging strikes divided by total swings — how often a swing misses. "
     "Distinct from swinging-strike rate, which is per pitch."),
    ("Chase %",
     "Swings at pitches outside the strike zone divided by total pitches outside "
     "the zone. Also called O-Swing%."),
    ("Avg Plate X / Z",
     "Average horizontal (X) and vertical (Z) location of the pitch as it crosses "
     "home plate, in feet from the middle of the plate (catcher's view)."),
    ("In-Zone %",
     "The share of pitches located inside the strike zone (Statcast zones 1–9)."),
]

_HITTER_GLOSSARY: list[tuple[str, str]] = [
    ("Exit Velocity (mph)",
     "The speed of the baseball as it comes off the bat, just after contact."),
    ("Launch Angle (°)",
     "The vertical angle at which the ball leaves the bat relative to the ground."),
    ("Hard-Hit %",
     "The share of batted balls hit with an exit velocity of 95 mph or higher."),
    ("Barrel %",
     "The share of batted balls classified as “barrels” — the exit-velocity/"
     "launch-angle combinations that have historically produced at least a .500 "
     "batting average and 1.500 slugging percentage."),
    ("Sweet-Spot %",
     "The share of batted balls hit with a launch angle between 8 and 32 degrees."),
    ("xwOBACON",
     "Expected weighted On-Base Average on Contact — the wOBA a hitter would be "
     "expected to earn on his batted balls based on each ball's exit velocity and "
     "launch angle, excluding walks and strikeouts."),
    ("xBACON",
     "Expected Batting Average on Contact — the batting average expected on batted "
     "balls given their exit velocity and launch angle."),
]

_GLOSSARY_SOURCE = "Definitions: MLB.com Statcast glossary and FanGraphs."


def _glossary_section(entries: list[tuple[str, str]]) -> str:
    """Render a Glossary section from (term, definition) pairs."""
    items = "\n".join(
        f"<dt>{term}</dt><dd>{definition}</dd>" for term, definition in entries
    )
    return (
        '<div class="section">\n'
        '      <h2>Glossary</h2>\n'
        f'      <dl class="glossary">\n{items}\n      </dl>\n'
        f'      <p class="glossary-src">{_GLOSSARY_SOURCE}</p>\n'
        '    </div>'
    )


# ===========================================================================
# Visuals section (charts)
# ===========================================================================

def _visual_block(title: str, data_uri: str) -> str:
    """One bordered panel: a heading plus an embedded PNG image."""
    return (
        f'<div class="visual-block"><h3>{title}</h3>'
        f'<img alt="{title}" src="{data_uri}"></div>'
    )


def _visuals_section(
    arsenal: pd.DataFrame,
    splits: pd.DataFrame,
    pitches: pd.DataFrame,
    throws: str,
) -> str:
    """Build the whole Visuals section from the five charts.

    charts is imported lazily (inside this function) so a table-only report
    never pays matplotlib's import cost or requires it to be installed.
    """
    try:
        from . import charts
    except ImportError:
        import charts

    blocks = [
        _visual_block("Pitch Usage", charts.pitch_usage_bar(arsenal)),
        _visual_block("Velocity by Pitch", charts.velocity_bar(arsenal)),
        _visual_block("Movement Plot", charts.movement_plot(arsenal, throws)),
        _visual_block("Location Heatmaps", charts.location_heatmaps(pitches, arsenal)),
        _visual_block("Handedness Split Usage", charts.handedness_split_bars(splits)),
    ]
    return _visuals_wrapper(blocks)


def _hitter_visuals_section(batted_balls: pd.DataFrame) -> str:
    """Build the hitter Visuals section (three batted-ball charts).

    charts is imported lazily so a table-only hitter report stays light.
    """
    try:
        from . import charts
    except ImportError:
        import charts

    blocks = [
        _visual_block("Exit Velocity Distribution", charts.ev_distribution(batted_balls)),
        _visual_block("Launch Angle Distribution", charts.launch_angle_distribution(batted_balls)),
        _visual_block("Exit Velocity vs Launch Angle", charts.ev_la_scatter(batted_balls)),
    ]
    return _visuals_wrapper(blocks)


def _visuals_wrapper(blocks: list[str]) -> str:
    """Wrap rendered visual blocks in the shared Visuals section markup."""
    return (
        '<div class="section">\n'
        '      <h2>Visuals</h2>\n'
        '      <div class="visual-grid">\n'
        + "\n".join(blocks)
        + "\n      </div>\n    </div>"
    )


# ===========================================================================
# Public API
# ===========================================================================

def pitcher_html_report(
    arsenal: pd.DataFrame,
    location: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    pitcher_name: str,
    throws: str,
    pitches: pd.DataFrame | None = None,
    subtitle: str = "Version 1 HTML report",
) -> str:
    """Render a full HTML scouting-report document.

    Args:
        arsenal:      metrics.pitcher_arsenal() output (sorted by usage desc).
        location:     metrics.pitcher_location() output.
        splits:       metrics.pitcher_handedness_splits() output.
        pitcher_name: Display name, e.g. 'Chase Burns'.
        throws:       Handedness phrase, e.g. 'Right-handed'.
        pitches:      Optional raw pitch-level rows for this pitcher. When given,
                      the report includes the Visuals section (five charts);
                      when None, tables only.
        subtitle:     Small print after the handedness in the subhead.

    Returns:
        A complete, self-contained HTML string (no external CSS or images).

    Keyword-only args after `*` force callers to name them at the call site —
    prevents accidentally swapping pitcher_name and throws.
    """
    cards = _build_cards(pitcher_name, throws, arsenal)
    bullets = _build_summary(arsenal, location, splits)

    # Charts are optional: only rendered when raw pitch data is supplied.
    visuals_html = ""
    if pitches is not None:
        visuals_html = _visuals_section(arsenal, splits, pitches, throws)

    cards_html = "\n".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
        for label, value in cards
    )
    bullets_html = "\n".join(f"<li>{b}</li>" for b in bullets)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{pitcher_name} Scouting Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="eyebrow">Automated Scouting Report</div>
      <h1>{pitcher_name}</h1>
      <p class="subhead">{throws} pitcher | {subtitle}</p>
    </div>

    <div class="cards">
      {cards_html}
    </div>

    <div class="section">
      <h2>Written Summary</h2>
      <ul class="summary-list">
        {bullets_html}
      </ul>
    </div>

    <div class="section">
      <h2>Pitch Metrics</h2>
      {_arsenal_table(arsenal)}
    </div>

    <div class="section">
      <h2>Handedness Splits</h2>
      {_handedness_table(splits)}
    </div>

    <div class="section">
      <h2>Location Summary</h2>
      {_location_table(location)}
    </div>

    {visuals_html}

    {_glossary_section(_PITCHER_GLOSSARY)}

    <div class="footer-note">
      Built for browser viewing now and structured to print cleanly to PDF later.
      <br>Automated Baseball Scouting · v{config.APP_VERSION}
    </div>
  </div>
</body>
</html>
"""


# ===========================================================================
# Hitter report — batted-ball quality (reuses the same design + helpers)
# ===========================================================================

# Barrel-rate tiers for the narrative. Rough league benchmarks: ~8% barrel rate
# is league-average, ~10%+ is a good power hitter, ~13%+ is elite. These are
# narrative judgment calls, not Statcast-defined constants — revisit per season.
_BARREL_GOOD = 0.10
_BARREL_ELITE = 0.13


def _quality_word(rate: float | None, good: float, elite: float) -> str:
    """Qualitative tier for a rate stat, using rough league benchmarks."""
    if rate is None or pd.isna(rate):
        return "unknown"
    if rate >= elite:
        return "elite"
    if rate >= good:
        return "above-average"
    return "average-or-below"


def _hitter_cards(hitter_name: str, bats: str, row: pd.Series) -> list[tuple[str, str]]:
    """Six summary cards for a hitter's batted-ball profile."""
    return [
        ("Hitter", hitter_name),
        ("Bats", bats),
        ("Batted Balls", str(int(row["n_bbe"]))),
        ("Avg Exit Velo", f"{_fmt(row['avg_ev'], 1)} mph"),
        ("Hard-Hit %", f"{_pct(row['hard_hit_rate'])}%"),
        ("Barrel %", f"{_pct(row['barrel_rate'])}%"),
    ]


def _hitter_summary(row: pd.Series, platoon: pd.DataFrame | None) -> list[str]:
    """Auto-written batted-ball-quality narrative bullets."""
    bullets = [
        f"{int(row['n_bbe'])} batted balls tracked — average exit velocity "
        f"{_fmt(row['avg_ev'], 1)} mph, topping out at {_fmt(row['max_ev'], 1)} mph.",
        f"Hard-hit rate {_pct(row['hard_hit_rate'])}% and "
        f"{_pct(row['barrel_rate'])}% barrels — "
        f"{_quality_word(row['barrel_rate'], _BARREL_GOOD, _BARREL_ELITE)} barrel production.",
        f"Sweet-spot rate {_pct(row['sweet_spot_rate'])}% at an average "
        f"{_fmt(row['avg_la'], 1)}° launch angle.",
        f"Expected production on contact: {_rate3(row['xwobacon'])} xwOBACON, "
        f"{_rate3(row['xbacon'])} xBACON.",
    ]

    # Platoon lean: compare xwOBACON vs LHP and vs RHP when both are present.
    # Skipped entirely if platoon data is missing (older pull without p_throws)
    # or the hitter only faced one hand.
    if platoon is not None and not platoon.empty:
        hands = {r["vs_hand"]: r for _, r in platoon.iterrows()}
        if "L" in hands and "R" in hands:
            l, r = hands["L"], hands["R"]
            if l["xwobacon"] >= r["xwobacon"]:
                strong, weak, sh, wh = l, r, "LHP", "RHP"
            else:
                strong, weak, sh, wh = r, l, "RHP", "LHP"
            bullets.append(
                f"Does more damage against {sh} ({_fmt(strong['avg_ev'], 1)} mph EV, "
                f"{_rate3(strong['xwobacon'])} xwOBACON) than {wh} "
                f"({_fmt(weak['avg_ev'], 1)} mph, {_rate3(weak['xwobacon'])})."
            )

    return bullets


def _batted_ball_profile_table(row: pd.Series) -> str:
    """Overall profile as a vertical Metric | Value table (reads better than
    one very wide row)."""
    rows = [
        ("Batted Balls", str(int(row["n_bbe"]))),
        ("Avg Exit Velo", f"{_fmt(row['avg_ev'], 1)} mph"),
        ("Max Exit Velo", f"{_fmt(row['max_ev'], 1)} mph"),
        ("Avg Launch Angle", f"{_fmt(row['avg_la'], 1)}°"),
        ("Hard-Hit %", f"{_pct(row['hard_hit_rate'])}%"),
        ("Barrel %", f"{_pct(row['barrel_rate'])}%"),
        ("Sweet-Spot %", f"{_pct(row['sweet_spot_rate'])}%"),
        ("xwOBACON", _rate3(row["xwobacon"])),
        ("xBACON", _rate3(row["xbacon"])),
    ]
    disp = pd.DataFrame(rows, columns=["Metric", "Value"])
    return _table_html(disp)


def _platoon_table(platoon: pd.DataFrame) -> str:
    """Batted-ball quality split by pitcher hand."""
    label = {"L": "vs LHP", "R": "vs RHP"}
    disp = pd.DataFrame({
        "Vs":           platoon["vs_hand"].map(lambda h: label.get(h, h)),
        "Batted Balls": platoon["n_bbe"].map(lambda v: str(int(v))),
        "Avg EV":       platoon["avg_ev"].map(lambda v: _fmt(v, 1)),
        "Max EV":       platoon["max_ev"].map(lambda v: _fmt(v, 1)),
        "Avg LA":       platoon["avg_la"].map(lambda v: _fmt(v, 1)),
        "Hard-Hit %":   platoon["hard_hit_rate"].map(lambda v: _pct(v)),
        "Barrel %":     platoon["barrel_rate"].map(lambda v: _pct(v)),
        "Sweet-Spot %": platoon["sweet_spot_rate"].map(lambda v: _pct(v)),
        "xwOBACON":     platoon["xwobacon"].map(_rate3),
        "xBACON":       platoon["xbacon"].map(_rate3),
    })
    return _table_html(disp)


def hitter_html_report(
    overall: pd.DataFrame,
    platoon: pd.DataFrame | None = None,
    *,
    hitter_name: str,
    bats: str,
    batted_balls_df: pd.DataFrame | None = None,
    subtitle: str = "Version 1 HTML report",
) -> str:
    """Render a hitter batted-ball-quality report in the shared design.

    Args:
        overall: metrics.hitter_batted_ball() output (one row).
        platoon: metrics.hitter_platoon_batted_ball() output. Optional — when
                 None or empty (e.g. a pull without `p_throws`), the Platoon
                 Splits section and its summary bullet are omitted.
        hitter_name: Display name, e.g. 'Ketel Marte'.
        bats:    Batting-hand phrase, e.g. 'Right-handed' / 'Switch'.
        batted_balls_df: Optional raw balls-in-play (metrics.batted_balls()).
                 When given, the report includes the Visuals section (three charts).
        subtitle: Small print after the hand in the subhead.

    Returns a complete, self-contained HTML string.
    """
    row = overall.iloc[0]
    cards = _hitter_cards(hitter_name, bats, row)
    bullets = _hitter_summary(row, platoon)

    visuals_html = ""
    if batted_balls_df is not None and not batted_balls_df.empty:
        visuals_html = _hitter_visuals_section(batted_balls_df)

    cards_html = "\n".join(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>'
        for label, value in cards
    )
    bullets_html = "\n".join(f"<li>{b}</li>" for b in bullets)

    # Platoon section only when we actually have split data.
    platoon_section = ""
    if platoon is not None and not platoon.empty:
        platoon_section = (
            '<div class="section">\n'
            '      <h2>Platoon Splits</h2>\n'
            f'      {_platoon_table(platoon)}\n'
            '    </div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{hitter_name} Scouting Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="eyebrow">Automated Scouting Report</div>
      <h1>{hitter_name}</h1>
      <p class="subhead">{bats} hitter | {subtitle}</p>
    </div>

    <div class="cards">
      {cards_html}
    </div>

    <div class="section">
      <h2>Written Summary</h2>
      <ul class="summary-list">
        {bullets_html}
      </ul>
    </div>

    <div class="section">
      <h2>Batted-Ball Profile</h2>
      {_batted_ball_profile_table(row)}
    </div>

    {platoon_section}

    {visuals_html}

    {_glossary_section(_HITTER_GLOSSARY)}

    <div class="footer-note">
      Batted-ball quality from Statcast. xwOBACON/xBACON are expected values on
      contact. Bunts excluded where identifiable.
      <br>Automated Baseball Scouting · v{config.APP_VERSION}
    </div>
  </div>
</body>
</html>
"""


def save_html_report(text: str, path: str | Path) -> None:
    """Write the HTML report to a file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"HTML report saved -> {path}")
