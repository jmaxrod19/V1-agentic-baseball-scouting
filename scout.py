"""
scout.py — pitcher name in, scouting report out.

Usage:
    python scout.py "Chase Burns"
    python scout.py "Chase Burns" --start 2026-06-01 --end 2026-06-28
    python scout.py "Chase Burns" --save

The script:
  1. Looks up the pitcher's MLB ID via pybaseball's player lookup table.
  2. Pulls pitch-level Statcast data for the requested date range, or loads
     a cached raw CSV if one already exists (skips the network call).
  3. Filters to that pitcher's pitches and computes the arsenal summary.
  4. Renders and prints a formatted scouting report.
  5. Optionally saves the report to data/processed/.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

# src/ is a sibling of this script, so add it to the path for direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
from pybaseball import playerid_lookup

import config
from loaders import load_statcast_csv, pull_statcast
from metrics import (
    pitcher_arsenal,
    pitcher_handedness_splits,
    pitcher_location,
)
from report import pitcher_report, save_report
from html_report import pitcher_html_report, save_html_report


# ---------------------------------------------------------------------------
# Player ID resolution
# ---------------------------------------------------------------------------

def _resolve_pitcher_id(name: str) -> tuple[int, str]:
    """Look up a pitcher's MLB ID by name.

    Args:
        name: Full name as 'First Last' or 'Last, First'.

    Returns:
        (mlbam_id, canonical_name) tuple.

    Raises:
        SystemExit if no match or multiple ambiguous matches are found.
    """
    # Normalise: accept both "Chase Burns" and "Burns, Chase".
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.strip().split()
        # Treat the last word as the surname; everything before is the first name.
        # Works for two-word names and most hyphenated surnames.
        first, last = " ".join(parts[:-1]), parts[-1]

    results = playerid_lookup(last, first)

    if results.empty:
        print(f"ERROR: No player found for '{name}'. Check spelling.", file=sys.stderr)
        sys.exit(1)

    if len(results) > 1:
        # Narrow to players with MLB appearances. If still ambiguous, list them.
        with_mlb = results[results["key_mlbam"].notna()]
        if len(with_mlb) == 1:
            results = with_mlb
        else:
            print(f"ERROR: Multiple players match '{name}':", file=sys.stderr)
            for _, row in results.iterrows():
                print(
                    f"  {row['name_first'].title()} {row['name_last'].title()} "
                    f"(MLB ID {row['key_mlbam']}, "
                    f"played {row.get('mlb_played_first', '?')}–{row.get('mlb_played_last', '?')})",
                    file=sys.stderr,
                )
            print("Re-run with a more specific name.", file=sys.stderr)
            sys.exit(1)

    row = results.iloc[0]
    mlbam_id = int(row["key_mlbam"])
    canonical = f"{row['name_first'].title()} {row['name_last'].title()}"
    return mlbam_id, canonical


# ---------------------------------------------------------------------------
# Cache-aware data pull
# ---------------------------------------------------------------------------

def _safe_filename(s: str) -> str:
    """Convert a string to a safe filename fragment (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _load_or_pull(
    pitcher_name: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return a full-date-range Statcast DataFrame, pulling only if needed.

    The raw CSV is stored as data/raw/<name>_<start>_<end>.csv.
    On a second run with the same arguments, we load from disk and skip
    the network call — pybaseball pulls can take 30+ seconds for long ranges.
    """
    slug = _safe_filename(pitcher_name)
    filename = f"{slug}_{start}_{end}.csv"
    cache_path = config.RAW_DIR / filename

    if cache_path.exists():
        print(f"Loading cached pull: {cache_path}")
        return load_statcast_csv(cache_path)

    print(f"Pulling Statcast data {start} → {end} (this may take a moment)…")
    # pull_statcast saves the raw file and returns the cleaned DataFrame.
    return pull_statcast(start, end, save_as=filename)


# ---------------------------------------------------------------------------
# Default date range: first day of the current month → today
# ---------------------------------------------------------------------------

def _default_start() -> str:
    today = date.today()
    return today.replace(day=1).isoformat()


def _default_end() -> str:
    return date.today().isoformat()


def _friendly_range(start: str, end: str) -> str:
    """Format 'YYYY-MM-DD' dates as 'Month D – Month D, YYYY'."""
    fmt = "%B %-d"  # e.g. 'June 1'
    s = datetime.strptime(start, "%Y-%m-%d").strftime(fmt)
    e = datetime.strptime(end, "%Y-%m-%d")
    return f"{s} – {e.strftime(fmt)}, {e.year}"


# ---------------------------------------------------------------------------
# Count distinct game dates for the target pitcher
# ---------------------------------------------------------------------------

def _count_starts(df: pd.DataFrame, pitcher_id: int) -> int:
    """Return the number of distinct game dates for this pitcher in the frame."""
    mask = df["pitcher"].astype("Int64") == pitcher_id
    if "game_date" not in df.columns:
        return 0
    return int(df.loc[mask, "game_date"].nunique())


def _throws_label(df_pitcher: pd.DataFrame) -> str:
    """Derive 'Right-handed' / 'Left-handed' from the pitcher's `p_throws`.

    Read from the data rather than hardcoded so any pitcher works. Falls back
    to a neutral label if the column is missing (older pulls) or empty.
    """
    if "p_throws" not in df_pitcher.columns:
        return "Pitcher"
    hands = df_pitcher["p_throws"].dropna()
    if hands.empty:
        return "Pitcher"
    return {"R": "Right-handed", "L": "Left-handed"}.get(hands.iloc[0], "Pitcher")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pitcher scouting report from Statcast data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("pitcher", help="Pitcher name, e.g. 'Chase Burns'")
    parser.add_argument(
        "--start",
        default=_default_start(),
        metavar="YYYY-MM-DD",
        help="Start date (default: first day of current month)",
    )
    parser.add_argument(
        "--end",
        default=_default_end(),
        metavar="YYYY-MM-DD",
        help="End date (default: today)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the report to data/processed/",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate the styled HTML report instead of the text report",
    )
    args = parser.parse_args()

    # 1. Resolve the pitcher name to an MLB ID.
    print(f"Looking up '{args.pitcher}'…")
    pitcher_id, canonical_name = _resolve_pitcher_id(args.pitcher)
    print(f"Found: {canonical_name} (MLB ID {pitcher_id})")

    # 2. Load or pull the date-range data.
    df_all = _load_or_pull(canonical_name, args.start, args.end)

    # 3. Filter to just this pitcher's pitches.
    df_pitcher = df_all[df_all["pitcher"].astype("Int64") == pitcher_id].copy()

    if df_pitcher.empty:
        print(
            f"ERROR: No pitches found for {canonical_name} (ID {pitcher_id}) "
            f"between {args.start} and {args.end}.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_pitches = len(df_pitcher)
    n_starts = _count_starts(df_all, pitcher_id)
    print(f"Found {n_pitches:,} pitches across {n_starts} appearance(s).")

    # 4. Compute the arsenal metrics (needed by both report formats).
    arsenal = pitcher_arsenal(df_pitcher)
    date_label = _friendly_range(args.start, args.end)
    slug = _safe_filename(canonical_name)

    if args.html:
        # HTML path: also needs location + handedness splits, plus the
        # pitcher's throwing hand for the header.
        location = pitcher_location(df_pitcher)
        splits = pitcher_handedness_splits(df_pitcher)
        html = pitcher_html_report(
            arsenal,
            location,
            splits,
            pitcher_name=canonical_name,
            throws=_throws_label(df_pitcher),
            pitches=df_pitcher,  # raw rows enable the Visuals section (charts)
        )
        # HTML is a file format, not terminal output — always write it, and
        # tell the user where it landed rather than dumping markup to stdout.
        out_path = config.PROCESSED_DIR / f"{slug}_report_{args.start}_{args.end}.html"
        save_html_report(html, out_path)
        return

    # Text path (default).
    report = pitcher_report(
        arsenal,
        pitcher_name=canonical_name,
        date_range=date_label,
        n_starts=n_starts,
    )

    print()
    print(report)

    if args.save:
        out_path = config.PROCESSED_DIR / f"{slug}_report_{args.start}_{args.end}.txt"
        save_report(report, out_path)


if __name__ == "__main__":
    main()
