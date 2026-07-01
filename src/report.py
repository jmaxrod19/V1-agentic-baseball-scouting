"""
Report formatter: renders pitcher metrics into a readable scouting report.

This is the display layer. All conversion from decimals to percentages happens
here — never in metrics.py. Nothing in this module should do math on rates;
it only formats numbers that metrics.py already computed.

Public functions:
    pitcher_report(arsenal, pitcher_name, date_range, n_starts) -> str
    save_report(text, path)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from . import config
except ImportError:
    import config


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _pct(val: float | None, decimals: int = 1) -> str:
    """Format a decimal rate (0.174) as a percentage string ('17.4%').

    Returns 'N/A' when the value is missing — avoids crashing on sparse
    pitch types that lack spin or movement data in older pulls.
    """
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def _mph(val: float | None) -> str:
    """Format velocity in mph to one decimal place."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f}"


def _rpm(val: float | None) -> str:
    """Format spin rate, rounded to the nearest whole rpm."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.0f}"


def _inch(val: float | None) -> str:
    """Format movement in inches to one decimal place, with sign."""
    if val is None or pd.isna(val):
        return "N/A"
    # Always show sign so arm-side vs glove-side direction is unambiguous.
    return f"{val:+.1f}"


def _ft(val: float | None) -> str:
    """Format extension in feet to one decimal place."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f}"


# Friendly names for the pitch type codes Statcast uses.
_PITCH_NAMES: dict[str, str] = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "SV": "Slurve",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "KN": "Knuckleball",
    "EP": "Eephus",
    "CS": "Slow Curve",
    "SC": "Screwball",
    "PO": "Pitchout",
    "FA": "Fastball",
}


def pitch_display_name(code: str) -> str:
    """Return a human-readable pitch name, falling back to the raw code.

    Public because the HTML report layer needs the same mapping — keeping it
    in one place means a new pitch code is added once, not twice.
    """
    return _PITCH_NAMES.get(code, code)


# ---------------------------------------------------------------------------
# Column widths for the arsenal table
# ---------------------------------------------------------------------------
# Fixed-width columns make the table readable in a monospace terminal or
# when pasted into a PDF template. Adjust widths here if the report layout
# changes — nowhere else.

_COL_WIDTHS = {
    "pitch":     20,
    "usage":      7,
    "velo":       7,
    "spin":       7,
    "h_break":    8,
    "v_break":    8,
    "ext":        6,
    "whiff":      8,
}


def _table_header() -> str:
    w = _COL_WIDTHS
    header = (
        f"{'Pitch':<{w['pitch']}}"
        f"{'Usage':>{w['usage']}}"
        f"{'Velo':>{w['velo']}}"
        f"{'Spin':>{w['spin']}}"
        f"{'H-Brk':>{w['h_break']}}"
        f"{'V-Brk':>{w['v_break']}}"
        f"{'Ext':>{w['ext']}}"
        f"{'Whiff%':>{w['whiff']}}"
    )
    # Underline each column with dashes the same width as the header cell.
    sep = (
        f"{'-'*w['pitch']:<{w['pitch']}}"
        f"{'-'*w['usage']:>{w['usage']}}"
        f"{'-'*w['velo']:>{w['velo']}}"
        f"{'-'*w['spin']:>{w['spin']}}"
        f"{'-'*w['h_break']:>{w['h_break']}}"
        f"{'-'*w['v_break']:>{w['v_break']}}"
        f"{'-'*w['ext']:>{w['ext']}}"
        f"{'-'*w['whiff']:>{w['whiff']}}"
    )
    return header + "\n" + sep


def _table_row(row: pd.Series) -> str:
    w = _COL_WIDTHS
    label = f"{pitch_display_name(row['pitch_type'])} ({row['pitch_type']})"
    return (
        f"{label:<{w['pitch']}}"
        f"{_pct(row['usage']):>{w['usage']}}"
        f"{_mph(row['avg_velo']):>{w['velo']}}"
        f"{_rpm(row['avg_spin']):>{w['spin']}}"
        f"{_inch(row['h_break_in']):>{w['h_break']}}"
        f"{_inch(row['v_break_in']):>{w['v_break']}}"
        f"{_ft(row['avg_extension']):>{w['ext']}}"
        f"{_pct(row['whiff_rate']):>{w['whiff']}}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pitcher_report(
    arsenal: pd.DataFrame,
    pitcher_name: str,
    date_range: str,
    n_starts: int,
) -> str:
    """Render a pitcher's arsenal metrics as a formatted scouting report string.

    Args:
        arsenal:      Output of metrics.pitcher_arsenal() — one row per pitch type.
        pitcher_name: Display name, e.g. 'Chase Burns'.
        date_range:   Human-readable date range, e.g. 'June 1 – June 28, 2026'.
        n_starts:     Number of starts included in the sample.

    Returns:
        A multi-line string ready to print or write to a file.

    Unit note: arsenal rates must be decimals (0.0–1.0). This function
    converts them to display percentages — don't pre-convert before passing.
    """
    total_pitches = int(arsenal["n_pitches"].sum())

    # Column legend goes below the table so it doesn't clutter the header.
    legend = (
        "  H-Brk: horizontal break, inches (+ = arm-side, − = glove-side)\n"
        "  V-Brk: induced vertical break, inches (+ = rise vs. spinless ball)\n"
        "  Ext:   release extension, feet\n"
        "  Whiff%: whiffs per swing"
    )

    lines: list[str] = [
        "=" * 71,
        f"  PITCHER SCOUTING REPORT — {pitcher_name.upper()}",
        f"  Sample: {date_range}  |  {n_starts} starts  |  {total_pitches:,} pitches",
        "=" * 71,
        "",
        "ARSENAL SUMMARY",
        "-" * 71,
        _table_header(),
    ]

    for _, row in arsenal.iterrows():
        lines.append(_table_row(row))

    lines += [
        "",
        legend,
        "",
        "=" * 71,
    ]

    return "\n".join(lines)


def save_report(text: str, path: str | Path) -> None:
    """Write the report string to a file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"Report saved -> {path}")
