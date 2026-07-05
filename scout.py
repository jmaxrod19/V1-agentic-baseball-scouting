"""
scout.py — player name in, scouting report out (command line).

Usage:
    python scout.py "Chase Burns"                       # text pitcher report
    python scout.py "Chase Burns" --html                # styled HTML + charts
    python scout.py "Ketel Marte" --hitter              # hitter HTML report
    python scout.py "Chase Burns" --start 2026-06-01 --end 2026-06-28

The CLI pulls the whole league for the date range once and caches it under
data/raw/, which is handy for offline / repeated local use. The web app
(webapp.py) instead uses fast per-player pulls via src/pipeline.py. Both share
the same player resolution, label helpers, and report renderers.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# src/ is a sibling of this script, so add it to the path for direct execution.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

import config
import pipeline
from loaders import load_statcast_csv, pull_statcast
from metrics import (
    batted_balls,
    hitter_batted_ball,
    hitter_platoon_batted_ball,
    pitcher_arsenal,
    pitcher_handedness_splits,
    pitcher_location,
)
from report import pitcher_report, save_report
from html_report import hitter_html_report, pitcher_html_report, save_html_report


# ---------------------------------------------------------------------------
# Player resolution (shared logic in pipeline; CLI turns errors into exits)
# ---------------------------------------------------------------------------

def _resolve_or_exit(name: str) -> tuple[int, str]:
    """Resolve a player name to (id, canonical), printing + exiting on failure."""
    print(f"Looking up '{name}'…")
    try:
        return pipeline.resolve_player(name)
    except pipeline.ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Cache-aware whole-league pull (CLI-specific; the web app pulls per player)
# ---------------------------------------------------------------------------

def _load_or_pull(player_name: str, start: str, end: str) -> pd.DataFrame:
    """Return a full-date-range Statcast DataFrame, pulling only if needed.

    The raw CSV is cached as data/raw/<name>_<start>_<end>.csv; a second run
    with the same arguments loads from disk and skips the (slow) network call.
    """
    slug = pipeline.safe_filename(player_name)
    filename = f"{slug}_{start}_{end}.csv"
    cache_path = config.RAW_DIR / filename

    if cache_path.exists():
        print(f"Loading cached pull: {cache_path}")
        return load_statcast_csv(cache_path)

    print(f"Pulling Statcast data {start} → {end} (this may take a moment)…")
    return pull_statcast(start, end, save_as=filename)


def _default_start() -> str:
    return date.today().replace(day=1).isoformat()


def _default_end() -> str:
    return date.today().isoformat()


def _count_starts(df: pd.DataFrame, pitcher_id: int) -> int:
    """Number of distinct game dates for this pitcher in the frame."""
    if "game_date" not in df.columns:
        return 0
    mask = df["pitcher"].astype("Int64") == pitcher_id
    return int(df.loc[mask, "game_date"].nunique())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a scouting report from Statcast data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("player", help="Player name, e.g. 'Chase Burns' or 'Ketel Marte'")
    parser.add_argument("--start", default=_default_start(), metavar="YYYY-MM-DD",
                        help="Start date (default: first day of current month)")
    parser.add_argument("--end", default=_default_end(), metavar="YYYY-MM-DD",
                        help="End date (default: today)")
    parser.add_argument("--save", action="store_true",
                        help="Save the text report to data/processed/")
    parser.add_argument("--html", action="store_true",
                        help="Generate the styled HTML report instead of the text report")
    parser.add_argument("--hitter", action="store_true",
                        help="Treat the player as a hitter (batted-ball report)")
    args = parser.parse_args()

    # 1. Resolve the player name to an MLB ID.
    player_id, canonical_name = _resolve_or_exit(args.player)
    print(f"Found: {canonical_name} (MLB ID {player_id})")

    # 2. Load or pull the date-range data (all players in the window).
    df_all = _load_or_pull(canonical_name, args.start, args.end)
    slug = pipeline.safe_filename(canonical_name)

    # -- Hitter path: batted-ball-quality HTML report -----------------------
    if args.hitter:
        df_batter = df_all[df_all["batter"].astype("Int64") == player_id].copy()
        if df_batter.empty:
            print(
                f"ERROR: No batted-ball data for {canonical_name} between "
                f"{args.start} and {args.end}.\n"
                f"(If {canonical_name} is a pitcher, drop --hitter.)",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            bbe = batted_balls(df_batter)
            overall = hitter_batted_ball(df_batter)
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}\n(If {canonical_name} is a pitcher, drop --hitter.)",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Found {int(overall.iloc[0]['n_bbe'])} batted balls.")

        try:
            platoon = hitter_platoon_batted_ball(df_batter)
        except (KeyError, ValueError) as exc:
            print(f"NOTE: platoon splits unavailable ({exc}).", file=sys.stderr)
            platoon = None

        html = hitter_html_report(
            overall, platoon,
            hitter_name=canonical_name,
            bats=pipeline.bats_label(df_batter),
            batted_balls_df=bbe,
        )
        out_path = config.PROCESSED_DIR / f"{slug}_hitter_{args.start}_{args.end}.html"
        save_html_report(html, out_path)
        return

    # 3. Pitcher path: filter to just this pitcher's pitches.
    df_pitcher = df_all[df_all["pitcher"].astype("Int64") == player_id].copy()
    if df_pitcher.empty:
        print(
            f"ERROR: No pitches found for {canonical_name} (ID {player_id}) "
            f"between {args.start} and {args.end}.\n"
            f"(If {canonical_name} is a hitter, re-run with --hitter.)",
            file=sys.stderr,
        )
        sys.exit(1)

    n_starts = _count_starts(df_all, player_id)
    print(f"Found {len(df_pitcher):,} pitches across {n_starts} appearance(s).")

    arsenal = pitcher_arsenal(df_pitcher)

    if args.html:
        location = pitcher_location(df_pitcher)
        splits = pitcher_handedness_splits(df_pitcher)
        html = pitcher_html_report(
            arsenal, location, splits,
            pitcher_name=canonical_name,
            throws=pipeline.throws_label(df_pitcher),
            pitches=df_pitcher,  # raw rows enable the Visuals section (charts)
        )
        out_path = config.PROCESSED_DIR / f"{slug}_report_{args.start}_{args.end}.html"
        save_html_report(html, out_path)
        return

    # Text path (default).
    report = pitcher_report(
        arsenal,
        pitcher_name=canonical_name,
        date_range=pipeline.friendly_range(args.start, args.end),
        n_starts=n_starts,
    )
    print()
    print(report)

    if args.save:
        out_path = config.PROCESSED_DIR / f"{slug}_report_{args.start}_{args.end}.txt"
        save_report(report, out_path)


if __name__ == "__main__":
    main()
