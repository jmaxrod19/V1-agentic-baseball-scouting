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
    .footer-note { margin-top: 28px; color: var(--muted); font-size: 13px; }
    @media print {
      body { background: white; }
      .page { max-width: none; padding: 0; }
      .card, .report-table { break-inside: avoid; }
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
# Public API
# ===========================================================================

def pitcher_html_report(
    arsenal: pd.DataFrame,
    location: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    pitcher_name: str,
    throws: str,
    subtitle: str = "Version 1 HTML report",
) -> str:
    """Render a full HTML scouting-report document.

    Args:
        arsenal:      metrics.pitcher_arsenal() output (sorted by usage desc).
        location:     metrics.pitcher_location() output.
        splits:       metrics.pitcher_handedness_splits() output.
        pitcher_name: Display name, e.g. 'Chase Burns'.
        throws:       Handedness phrase, e.g. 'Right-handed'.
        subtitle:     Small print after the handedness in the subhead.

    Returns:
        A complete, self-contained HTML string (no external CSS).

    Keyword-only args after `*` force callers to name them at the call site —
    prevents accidentally swapping pitcher_name and throws.
    """
    cards = _build_cards(pitcher_name, throws, arsenal)
    bullets = _build_summary(arsenal, location, splits)

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

    <div class="footer-note">
      Built for browser viewing now and structured to print cleanly to PDF later.
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
