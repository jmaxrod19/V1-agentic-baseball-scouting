"""Runner for the VAA prototypes: pick a pitcher, get a picture.

Usage (from the repo root):
    .venv/bin/python vaa_prototype.py --pitcher "Skubal"
    .venv/bin/python vaa_prototype.py --pitcher "Burns, Chase" --pitch-type FF

It reads a league-wide Statcast CSV from data/raw/, takes one pitcher's pitches
of one type, compares his average to the league average for pitchers who throw
with the same hand, draws Prototype A2, and saves a PNG in data/processed/.
Throwaway glue: the real logic lives in src/vaa.py and src/vaa_charts.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import loaders, vaa, vaa_charts

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a VAA prototype for one pitcher.")
    parser.add_argument("--csv", default=str(ROOT / "data/raw/chase_burns_2026-06-01_2026-06-28.csv"))
    parser.add_argument("--pitcher", required=True, help='Part of the name, e.g. "Burns, Chase" or "Skubal".')
    parser.add_argument("--pitch-type", default="FF", help="Statcast code, default FF (four-seam).")
    parser.add_argument("--out", default=str(ROOT / "data/processed"))
    args = parser.parse_args()

    df = loaders.load_statcast_csv(args.csv)

    # Case-insensitive "contains" match so a last name is enough.
    matches = df[df["player_name"].str.contains(args.pitcher, case=False, na=False, regex=False)]
    if matches.empty:
        raise SystemExit(f"No pitcher matching {args.pitcher!r} in {args.csv}")
    names = matches["player_name"].unique()
    if len(names) > 1:
        raise SystemExit(f"{args.pitcher!r} matches several pitchers: {list(names)}. Be more specific.")
    name = names[0]

    same_type = df[df["pitch_type"] == args.pitch_type]
    mine = same_type[same_type["player_name"] == name]

    # League baseline: same pitch type, same throwing hand. Mixing hands would
    # compare a lefty's release side to a blend of both sides.
    hand = mine["p_throws"].mode().iloc[0]
    league = same_type[same_type["p_throws"] == hand]

    mine_vaa = vaa.compute_vaa(mine).dropna()
    league_vaa = vaa.compute_vaa(league).dropna()
    print(f"{name}: {len(mine_vaa)} {args.pitch_type}, mean VAA {mine_vaa.mean():.2f} deg, "
          f"mean plate height {mine['plate_z'].mean():.2f} ft")
    print(f"League {hand}HP: {len(league_vaa)} {args.pitch_type}, mean VAA {league_vaa.mean():.2f} deg, "
          f"mean plate height {league['plate_z'].mean():.2f} ft")

    fig = vaa_charts.compare_to_league_3d(
        mine, league, f"{name} {args.pitch_type}", f"MLB average {hand}HP {args.pitch_type}",
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(",", "").replace(" ", "_")
    out_path = out_dir / f"vaa_proto_a2_{slug}_{args.pitch_type.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
