"""Runner for the VAA prototypes: pick a pitcher, get a picture.

Usage (from the repo root):
    .venv/bin/python vaa_prototype.py --pitcher "Burns, Chase"
    .venv/bin/python vaa_prototype.py --pitcher "Skubal" --pitch-type FF --max-paths 80

It reads a Statcast CSV from data/raw/, filters to one pitcher's pitches of one
type, draws Prototype A (3D flight paths), and saves a PNG in data/processed/.
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
    parser.add_argument("--max-paths", type=int, default=60)
    parser.add_argument("--out", default=str(ROOT / "data/processed"))
    args = parser.parse_args()

    df = loaders.load_statcast_csv(args.csv)

    # Case-insensitive "contains" match so a last name is enough.
    mine = df[df["player_name"].str.contains(args.pitcher, case=False, na=False, regex=False)]
    if mine.empty:
        raise SystemExit(f"No pitcher matching {args.pitcher!r} in {args.csv}")
    names = mine["player_name"].unique()
    if len(names) > 1:
        raise SystemExit(f"{args.pitcher!r} matches several pitchers: {list(names)}. Be more specific.")
    mine = mine[mine["pitch_type"] == args.pitch_type]

    angles = vaa.compute_vaa(mine).dropna()
    print(f"{names[0]}: {len(mine)} {args.pitch_type} pitches, {len(angles)} with a VAA; "
          f"mean {angles.mean():.2f} deg, range {angles.min():.1f} to {angles.max():.1f}")

    fig = vaa_charts.flight_paths_3d(
        mine, f"{names[0]} {args.pitch_type}", max_paths=args.max_paths,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = names[0].lower().replace(",", "").replace(" ", "_")
    out_path = out_dir / f"vaa_proto_a_{slug}_{args.pitch_type.lower()}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
