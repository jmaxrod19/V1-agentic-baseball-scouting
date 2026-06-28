"""
Example: load each data source and show what you get back.

Run from the repo root:
    python examples_quickstart.py

This is a scratch/demo script. It expects sample files in data/raw/ — swap in
your own pulls and exports. Nothing here is imported by the rest of the project;
it's just a guided tour of the loaders.
"""

import sys
from pathlib import Path

# Make src importable when running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import loaders  # noqa: E402
import config   # noqa: E402


def demo_statcast_pull():
    """Pull a small date range straight from pybaseball and cache it.

    Uncomment to run a live pull (needs internet). The raw pull is saved to
    data/raw/ so you don't have to re-download to experiment with cleaning.
    """
    # df = loaders.pull_statcast("2024-06-14", "2024-06-15", save_as="sc_2024-06-14_15.csv")
    # print(df.shape, "pitches pulled")
    # return df
    pass


def demo_load_from_disk():
    """Load each source from a CSV already sitting in data/raw/."""
    raw = config.RAW_DIR

    statcast_path = raw / "statcast_sample.csv"
    if statcast_path.exists():
        sc = loaders.load_statcast_csv(statcast_path)
        print(f"Statcast: {sc.shape[0]} rows x {sc.shape[1]} cols")
        print("  date dtype:", sc["game_date"].dtype)

    fg_path = raw / "fangraphs_sample.csv"
    if fg_path.exists():
        fg = loaders.load_fangraphs_csv(fg_path)
        print(f"FanGraphs: {fg.shape[0]} rows x {fg.shape[1]} cols")
        print(fg[["Name", "Team", "HR", "K%"]].to_string(index=False))


if __name__ == "__main__":
    print("Drop CSVs in data/raw/ and run the loaders. See loaders.py for the API.\n")
    demo_load_from_disk()
