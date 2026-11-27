"""Turn a season of league-wide pitches into the small table the web app reads.

Usage (from the repo root):
    .venv/bin/python build_vaa_table.py                  # 2026
    .venv/bin/python build_vaa_table.py --season 2025 --csv data/raw/statcast_2025.csv

Run it again whenever the raw data or the VAA math changes, then commit the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src import loaders, vaa_season

ROOT = Path(__file__).resolve().parent
HEIGHT_CACHE = ROOT / "data/processed/pitcher_heights.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the season VAA table for the web app.")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--csv", help="League-wide Statcast CSV; default data/raw/statcast_<season>.csv")
    args = parser.parse_args()

    csv = Path(args.csv) if args.csv else ROOT / f"data/raw/statcast_{args.season}.csv"
    data = loaders.load_statcast_csv(csv)
    ids = [int(i) for i in data["pitcher"].dropna().unique()]
    # Outfielders and catchers who mopped up an inning: their lobs are not fastballs.
    position_players = loaders.pull_position_players(ids)
    print(f"leaving out {len(position_players)} position players who pitched")
    # Listed heights from the MLB API, cached on disk so reruns are instant.
    heights = loaders.cached_player_heights(ids, HEIGHT_CACHE)
    table = vaa_season.build_season_table(data, args.season, heights, position_players)

    out = ROOT / f"app_data/vaa_fastball_{args.season}.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out, index=False, float_format="%.8g")   # 8 digits: plenty, and compact
    # One "all" row per pitcher; the other parts of the zone would count them again.
    pitchers = int(((table["role"] == "pitcher") & (table["part"] == "all")).sum())
    print(f"{pitchers} pitchers -> {out} ({out.stat().st_size / 1000:.0f} KB)")


if __name__ == "__main__":
    main()
