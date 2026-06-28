"""
Loaders for the three CSV sources this project uses.

Each loader takes a file path and returns a clean pandas DataFrame, so the
rest of the codebase never has to worry about source-specific quirks. The
quirks (B-Ref's repeated headers, name annotations, etc.) are handled here
once, in functions you can read and trust.

Public functions:
    pull_statcast(start, end, ...)   -> pull from pybaseball, optionally save
    load_statcast_csv(path)          -> load a pybaseball Statcast pull CSV
    load_savant_csv(path)            -> load a Baseball Savant search export
    pull_fangraphs(season, kind)     -> pull a FanGraphs leaderboard
    load_fangraphs_csv(path)         -> load a FanGraphs leaderboard CSV export
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    # config is a sibling module. This import style works whether you run
    # scripts from the repo root or import src as a package.
    from . import config
except ImportError:  # pragma: no cover - fallback for direct script execution
    import config


# ===========================================================================
# Shared cleaning core for Statcast-like data
# ===========================================================================
# Statcast pulls (pybaseball) and Savant search exports share the same wide
# pitch-level schema. Rather than duplicate the cleaning logic, both loaders
# call this one helper. If the two sources ever diverge, you split here.

def _clean_statcast_like(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce and tidy a Statcast-schema DataFrame in place-ish.

    Returns a new DataFrame; the input is not mutated.
    """
    df = df.copy()

    # 1. Parse the date column to real datetimes (not strings). errors="coerce"
    #    turns anything unparseable into NaT instead of raising, so one bad
    #    row never kills the whole load.
    if config.STATCAST_DATE_COL in df.columns:
        df[config.STATCAST_DATE_COL] = pd.to_datetime(
            df[config.STATCAST_DATE_COL], errors="coerce"
        )

    # 2. Force the known numeric columns to numeric dtype. We only touch
    #    columns that actually exist in this pull (older pulls lack bat
    #    tracking, etc.), so the same loader works across seasons.
    for col in config.STATCAST_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Sort chronologically so downstream work (rolling windows, game logs)
    #    sees pitches in order. We sort by date, then game, then the pitch
    #    sequence within an at-bat when those columns are present.
    sort_cols = [
        c
        for c in (config.STATCAST_DATE_COL, "game_pk", "at_bat_number", "pitch_number")
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


# ===========================================================================
# Statcast (pybaseball pull)
# ===========================================================================

def pull_statcast(
    start_dt: str,
    end_dt: str,
    save_as: str | None = None,
) -> pd.DataFrame:
    """Pull pitch-level Statcast data via pybaseball for a date range.

    Args:
        start_dt: 'YYYY-MM-DD' (inclusive).
        end_dt:   'YYYY-MM-DD' (inclusive).
        save_as:  optional filename. If given, the RAW (uncleaned) pull is
                  written to data/raw/<save_as> so you keep an immutable copy
                  of exactly what came back, and can re-clean later without
                  re-hitting the network.

    Returns the cleaned DataFrame.

    Note: pybaseball is imported inside the function so that importing this
    module (e.g. to load CSVs) doesn't require pybaseball or a network call.
    """
    from pybaseball import statcast

    raw = statcast(start_dt=start_dt, end_dt=end_dt)

    if save_as:
        out = config.RAW_DIR / save_as
        raw.to_csv(out, index=False)
        print(f"Saved raw pull -> {out}  ({len(raw):,} rows)")

    return _clean_statcast_like(raw)


def load_statcast_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV produced by a pybaseball Statcast pull and clean it."""
    df = pd.read_csv(path, low_memory=False)
    return _clean_statcast_like(df)


# ===========================================================================
# Baseball Savant (browser search export)
# ===========================================================================

def load_savant_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV downloaded from baseballsavant.mlb.com's search tool.

    Savant's browser export shares Statcast's schema, so it runs through the
    same cleaner. It lives in its own function for two reasons:
      - intent is clearer at the call site (you know which source you loaded)
      - if Savant's export format diverges from pybaseball's, you change this
        function without touching load_statcast_csv.
    """
    df = pd.read_csv(path, low_memory=False)
    return _clean_statcast_like(df)


# ===========================================================================
# FanGraphs (leaderboard "Export Data" CSV)
# ===========================================================================
# FanGraphs exports are cleaner than B-Ref — no repeated header rows, no name
# markers — but they have their own signature quirks:
#   - Rate stats export as strings WITH a percent sign: "K%" -> "25.4 %",
#     "GB%", "O-Swing%", "SwStr%", etc. These need the '%' stripped and the
#     value coerced to float.
#   - Value stats export with a dollar sign: "Dol" / "$" -> "$12.3".
#   - Numbers can carry thousands separators (commas) in some views.
# The set of percent columns depends on which leaderboard you exported
# (Dashboard vs Advanced vs Batted Ball...), so instead of hardcoding a list
# we DETECT value columns and clean them. That makes one loader handle any
# FanGraphs export you throw at it.

# Columns that are always text and should never be coerced to numbers.
_FG_TEXT_COLS = {"Name", "NameASCII", "Team", "Tm", "Pos", "Bats", "Throws"}


def pull_fangraphs(season: int, kind: str = "batting", qual: int = 0, **kwargs):
    """Pull a FanGraphs season leaderboard via pybaseball.

    Args:
        season: e.g. 2024.
        kind:   'batting' or 'pitching'.
        qual:   minimum PA/IP to qualify; 0 returns everyone.
        kwargs: passed through to pybaseball (e.g. end_season for ranges).

    Returns the leaderboard DataFrame. pybaseball already returns numeric
    dtypes here (it pulls the API, not the CSV export), so no extra cleaning
    is needed — the cleaning below is only for CSVs you download by hand.
    """
    if kind == "batting":
        from pybaseball import batting_stats
        return batting_stats(season, qual=qual, **kwargs)
    elif kind == "pitching":
        from pybaseball import pitching_stats
        return pitching_stats(season, qual=qual, **kwargs)
    raise ValueError("kind must be 'batting' or 'pitching'")


def _clean_fg_value_column(s: pd.Series, as_proportion: bool) -> pd.Series:
    """Try to turn one string column of FanGraphs values into numbers.

    Strips '%', '$', and thousands-commas, then coerces to float. Returns the
    cleaned numeric series if most values convert, otherwise returns the
    original untouched (so genuine text columns are left alone).
    """
    text = s.astype(str)
    is_percent = text.str.contains("%", na=False).any()

    cleaned = (
        text.str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    converted = pd.to_numeric(cleaned, errors="coerce")

    # Only accept the conversion if at least half the non-null values became
    # numbers. Otherwise it's a real text column and we leave it as-is.
    non_null = s.notna().sum()
    if not non_null or converted.notna().sum() < 0.5 * non_null:
        return s

    # FanGraphs shows "25.4%" meaning 0.254 of PAs. Default keeps the
    # human-readable 25.4 (matches the site and most report layouts); pass
    # as_proportion=True to get 0.254 for math/joins with Statcast rates.
    if is_percent and as_proportion:
        converted = converted / 100.0

    return converted


def load_fangraphs_csv(
    path: str | Path,
    as_proportion: bool = False,
) -> pd.DataFrame:
    """Load and clean a FanGraphs leaderboard CSV export.

    Args:
        path: path to the exported CSV.
        as_proportion: if True, percent columns become proportions (25.4 ->
                  0.254). Default False keeps the displayed value (25.4).

    Returns a cleaned DataFrame with proper numeric dtypes.
    """
    df = pd.read_csv(path)

    # Drop FanGraphs' leading rank column if it came through unnamed.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # Clean every non-text column: strip %/$/commas and coerce to numeric.
    for col in df.columns:
        if col in _FG_TEXT_COLS:
            continue
        df[col] = _clean_fg_value_column(df[col], as_proportion)

    return df.reset_index(drop=True)
