"""Runner for the fastball VAA chart: pick a pitcher and a season, get a picture.

Usage (from the repo root):
    .venv/bin/python vaa_prototype.py --pitcher "Skubal"
    .venv/bin/python vaa_prototype.py --pitcher "Ryan, Joe" --plate-height 3.0
    .venv/bin/python vaa_prototype.py --pitcher 669373 --season 2025 --csv data/raw/statcast_2025.csv

The CSV must be LEAGUE-WIDE pitch-level Statcast data, because the benchmark is
built from it. Four-seam fastballs only. The PNG goes to data/processed/.
The real logic lives in src/vaa_model.py (numbers) and src/vaa_charts.py (drawing).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import loaders, vaa_charts, vaa_model

ROOT = Path(__file__).resolve().parent
HEIGHT_CACHE = ROOT / "data/processed/pitcher_heights.csv"
DEFAULT_CSV = ROOT / "data/raw/statcast_2026.csv"   # league-wide, full 2026 season


def find_pitcher(df: pd.DataFrame, query: str) -> int:
    """Turn a name fragment (or a numeric MLBAM id) into one pitcher id."""
    if query.isdigit():
        return int(query)
    matches = df[df["player_name"].str.contains(query, case=False, na=False, regex=False)]
    people = matches[["pitcher", "player_name"]].drop_duplicates()
    if people.empty:
        raise SystemExit(f"No pitcher matching {query!r} in this data.")
    if len(people) > 1:
        raise SystemExit(f"{query!r} matches several pitchers: {people.to_dict('records')}")
    return int(people["pitcher"].iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the four-seam VAA chart for one pitcher.")
    parser.add_argument("--pitcher", required=True, help='Name fragment ("Skubal") or MLBAM id.')
    parser.add_argument("--season", type=int, help="Year; default is the latest in the data.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="League-wide Statcast CSV.")
    parser.add_argument("--plate-height", type=float, default=vaa_model.DEFAULT_PLATE_HEIGHT,
                        help="Height both modeled pitches cross the front of the plate (ft).")
    parser.add_argument("--out", default=str(ROOT / "data/processed"))
    args = parser.parse_args()

    df = loaders.load_statcast_csv(args.csv)
    pitcher_id = find_pitcher(df, args.pitcher)
    try:
        c = vaa_model.compare_fastball(df, pitcher_id, args.season, args.plate_height)
    except ValueError as problem:
        raise SystemExit(str(problem))
    c = replace(c, **vaa_model.pitch_mix(df, c.season).loc[pitcher_id].to_dict())

    # Listed heights from the MLB API, cached on disk so repeat runs are instant.
    heights = loaders.cached_player_heights([pitcher_id, *c.pitcher_ids], HEIGHT_CACHE)
    league_height = pd.Series(c.pitcher_ids).map(heights).mean()
    fig = vaa_charts.fastball_vaa_chart(c, heights.get(pitcher_id), league_height)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = c.name.lower().replace(".", "").replace(" ", "_")
    out_path = out_dir / f"vaa_proto_a6_{slug}_{c.season}_ff.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)

    print(f"{c.name} {c.season}: {c.pitcher.pitches} FF vs {c.league.pitches} {c.hand}HP FF")
    print(f"Observed VAA {c.pitcher.observed_vaa:.2f} (MLB {c.league.observed_vaa:.2f}), "
          f"VAA AA {c.pitcher.vaa_aa:+.2f} -> {c.classification}")
    print(f"Modeled at {c.comparison_height:g} ft: {c.pitcher.modeled.vaa:.2f} vs "
          f"{c.league.modeled.vaa:.2f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
