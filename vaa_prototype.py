"""Runner for the VAA prototypes: pick a pitcher, get a picture.

Usage (from the repo root):
    .venv/bin/python vaa_prototype.py --pitcher "Skubal"
    .venv/bin/python vaa_prototype.py --pitcher "Ryan, Joe" --style 3d

It reads a league-wide Statcast CSV from data/raw/, takes one pitcher's pitches
of one type, compares his average to the league average for pitchers who throw
with the same hand, and saves a PNG in data/processed/.

Styles:  side (default)  Prototype A5: verdict, one side view, three release cards
         3d              Prototype A2: one line vs one line, in 3D
Throwaway glue: the real logic lives in src/vaa.py and src/vaa_charts.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src import loaders, vaa, vaa_charts

ROOT = Path(__file__).resolve().parent
HEIGHT_CACHE = ROOT / "data/processed/pitcher_heights.csv"
# Plain words for the chart; any other pitch code is shown as-is.
PITCH_NAMES = {"FF": "four-seam fastball", "SI": "sinker", "FC": "cutter"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a VAA prototype for one pitcher.")
    parser.add_argument("--csv", default=str(ROOT / "data/raw/chase_burns_2026-06-01_2026-06-28.csv"))
    parser.add_argument("--pitcher", required=True, help='Part of the name, e.g. "Burns, Chase" or "Skubal".')
    parser.add_argument("--pitch-type", default="FF", help="Statcast code, default FF (four-seam).")
    parser.add_argument("--style", choices=["side", "3d"], default="side")
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
    print(f"{name}: {len(mine_vaa)} {args.pitch_type}, mean VAA {mine_vaa.mean():.2f} deg")
    print(f"League {hand}HP: {len(league_vaa)} {args.pitch_type}, mean VAA {league_vaa.mean():.2f} deg")

    # Statcast names are "Last, First"; read better as "First Last".
    last, _, first = name.partition(", ")
    display_name = f"{first} {last}".strip()
    pitcher_label = f"{display_name} {args.pitch_type}"
    league_label = f"MLB average {hand}HP {args.pitch_type}"

    if args.style == "3d":
        fig = vaa_charts.compare_to_league_3d(mine, league, pitcher_label, league_label)
        stem = "a2"
    else:
        # Listed heights from the MLB API, cached on disk so repeat runs are instant.
        pitcher_id = int(mine["pitcher"].iloc[0])
        heights = loaders.cached_player_heights(
            [pitcher_id, *league["pitcher"].dropna().astype(int).unique()], HEIGHT_CACHE,
        )
        height_in = heights.get(pitcher_id)
        # League height averaged over PITCHES (like the other league numbers), not over pitchers.
        league_height_in = league["pitcher"].map(heights).mean()
        print(f"Heights found for {len(heights)} pitchers; {name}: {height_in} in, league avg {league_height_in:.1f} in")
        pitch_name = PITCH_NAMES.get(args.pitch_type, args.pitch_type)
        hand_word = "right" if hand == "R" else "left"
        fig = vaa_charts.side_view_with_profile(
            mine, league, display_name, f"average {hand_word}-handed {pitch_name}",
            pitcher_height_in=height_in, league_height_in=league_height_in,
        )
        stem = "a5"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name.lower().replace(",", "").replace(" ", "_")
    out_path = out_dir / f"vaa_proto_{stem}_{slug}_{args.pitch_type.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
