"""
Report pipeline: name + date range + kind -> rendered HTML report.

One code path shared by the CLI (scout.py) and the web app (webapp.py), so the
report logic lives in exactly one place. Unlike scout.py's whole-league cached
pull, this uses fast per-player Statcast pulls (loaders.pull_statcast_pitcher /
_batter), suitable for a live request.

Errors are raised as ReportError subclasses (never sys.exit), so the web layer
can turn them into friendly responses and the CLI can print + exit.

Public API:
    resolve_player(name) -> (mlbam_id, canonical_name)
    generate_report(name, start, end, kind) -> ReportResult
    throws_label(df) / bats_label(df) / friendly_range(s, e) / safe_filename(s)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

try:
    from . import config, loaders, metrics, percentiles, validate
    from .html_report import hitter_html_report, pitcher_html_report
except ImportError:  # direct-script / non-package execution
    import config
    import loaders
    import metrics
    import percentiles
    import validate
    from html_report import hitter_html_report, pitcher_html_report


# ---------------------------------------------------------------------------
# Errors — raised instead of sys.exit so callers choose how to surface them
# ---------------------------------------------------------------------------

class ReportError(Exception):
    """Base class for any expected, user-facing report failure."""


class PlayerNotFound(ReportError):
    """No player matched the given name."""


class AmbiguousPlayer(ReportError):
    """More than one player matched and we can't safely pick one."""


class NoData(ReportError):
    """The player resolved, but there's no usable data in the range."""


class DataQualityError(ReportError):
    """The pulled data is present but structurally broken (failed validation)."""


# ---------------------------------------------------------------------------
# Pure label / format helpers (shared with the CLI)
# ---------------------------------------------------------------------------

def safe_filename(s: str) -> str:
    """Convert a string to a safe filename fragment (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def friendly_range(start: str, end: str) -> str:
    """Format 'YYYY-MM-DD' dates as 'Month D – Month D, YYYY'."""
    fmt = "%B %-d"  # e.g. 'June 1'
    s = datetime.strptime(start, "%Y-%m-%d").strftime(fmt)
    e = datetime.strptime(end, "%Y-%m-%d")
    return f"{s} – {e.strftime(fmt)}, {e.year}"


def throws_label(df: pd.DataFrame) -> str:
    """Derive 'Right-handed' / 'Left-handed' from a pitcher's `p_throws`.

    df is assumed to be one pitcher's pitches (per-player pull), so we read the
    first known value. Falls back to a neutral label if absent.
    """
    if "p_throws" not in df.columns:
        return "Pitcher"
    hands = df["p_throws"].dropna()
    if hands.empty:
        return "Pitcher"
    return {"R": "Right-handed", "L": "Left-handed"}.get(hands.iloc[0], "Pitcher")


def bats_label(df: pd.DataFrame) -> str:
    """Derive 'Right-handed' / 'Left-handed' / 'Switch' from a batter's `stand`.

    df is one batter's pitches (per-player pull). More than one distinct `stand`
    across his plate appearances means a switch hitter.
    """
    if "stand" not in df.columns:
        return "Hitter"
    stands = df["stand"].dropna().unique()
    if len(stands) > 1:
        return "Switch"
    if len(stands) == 1:
        return {"R": "Right-handed", "L": "Left-handed"}.get(stands[0], "Hitter")
    return "Hitter"


# ---------------------------------------------------------------------------
# Player resolution
# ---------------------------------------------------------------------------

def resolve_player(name: str) -> tuple[int, str]:
    """Look up a player's MLB id by name (works for pitchers and hitters).

    Args:
        name: 'First Last' or 'Last, First'.

    Returns:
        (mlbam_id, canonical_name).

    Raises:
        PlayerNotFound:  no match.
        AmbiguousPlayer: multiple matches we can't disambiguate.
    """
    # pybaseball is imported lazily so importing this module stays cheap.
    from pybaseball import playerid_lookup

    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.strip().split()
        if len(parts) < 2:
            raise PlayerNotFound(
                f"Please enter a full name (first and last), not '{name}'."
            )
        # Last word = surname; everything before = first name (handles middle
        # names and most hyphenated surnames).
        first, last = " ".join(parts[:-1]), parts[-1]

    results = playerid_lookup(last, first)

    if results.empty:
        raise PlayerNotFound(f"No player found for '{name}'. Check the spelling.")

    if len(results) > 1:
        # Prefer the lone player who actually has MLB appearances.
        with_mlb = results[results["key_mlbam"].notna()]
        if len(with_mlb) == 1:
            results = with_mlb
        else:
            names = ", ".join(
                f"{r['name_first'].title()} {r['name_last'].title()}"
                for _, r in results.head(5).iterrows()
            )
            raise AmbiguousPlayer(
                f"Multiple players match '{name}' ({names}). Try a more specific name."
            )

    row = results.iloc[0]
    return int(row["key_mlbam"]), f"{row['name_first'].title()} {row['name_last'].title()}"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    """A rendered report plus a little metadata for logging / the caller."""
    html: str
    player_name: str
    player_id: int
    kind: str          # 'pitcher' | 'hitter'
    n_events: int      # pitches (pitcher) or batted balls (hitter)


# Small bounded in-memory cache: repeat requests for the same (id, range, kind)
# return instantly instead of re-hitting Statcast. Keyed on resolved id, not the
# raw name, so 'Ketel Marte' and 'Marte, Ketel' share a cache entry.
# Tradeoff: there's no TTL — a range ending *today* can serve a slightly stale
# result until the entry is evicted, if more of today's games are played after
# the first request. Fine for a portfolio app; revisit if live same-day matters.
_CACHE: dict[tuple, ReportResult] = {}
_CACHE_MAX = 64


def _cache_get(key: tuple) -> ReportResult | None:
    return _CACHE.get(key)


def _cache_put(key: tuple, value: ReportResult) -> None:
    # Simple FIFO eviction — good enough for a low-traffic portfolio app.
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = value


def _pull_player(player_id: int, start: str, end: str, kind: str) -> pd.DataFrame:
    """Pull one player's cleaned Statcast frame, caching completed ranges to disk.

    The in-memory _CACHE above only survives while the process runs, so after
    every restart/redeploy the first request re-hits Statcast. This adds a disk
    layer under data/processed/: the cache IS the file.

    - Hit: reload + re-clean from CSV — turns a ~seconds network pull into an
      instant local read, and lets the app build reports fully offline.
    - Miss: pull from Statcast, then write the cache — but only for a *completed*
      date range (end before today). A range ending today could still gain more
      of today's games after we cache it, so persisting it risks serving stale
      data across restarts; those ranges fall back to the in-memory cache only.
    """
    puller = (
        loaders.pull_statcast_pitcher if kind == "pitcher"
        else loaders.pull_statcast_batter
    )
    # kind is in the filename so a pitcher and hitter pull for the same id/range
    # never collide (they're different Statcast queries).
    path = config.PROCESSED_DIR / f"pull_{kind}_{player_id}_{start}_{end}.csv"

    if path.exists():
        df = loaders.load_statcast_csv(path)
        _validate_or_raise(df, player_id)  # trust-but-verify even cached data
        return df

    df = puller(player_id, start, end)

    # Validate BEFORE writing the cache, so a broken pull is never persisted.
    _validate_or_raise(df, player_id)

    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    range_is_complete = end_date < datetime.now().date()
    if range_is_complete and not df.empty:
        df.to_csv(path, index=False)

    return df


def _validate_or_raise(df: pd.DataFrame, player_id: int) -> None:
    """Validate a pulled frame: log any issues, raise on structural errors.

    An empty frame is left for generate_report's per-kind NoData messages (they
    explain the pitcher/hitter mixup), so we only validate frames with rows.
    Warnings are logged to stderr for observability (visible in the HF logs);
    only error-level issues stop the request, via DataQualityError.
    """
    if df.empty:
        return

    report = validate.validate_statcast(df)
    if report.issues:
        # One line to stderr so data-quality problems are diagnosable in the logs.
        print(f"[validate] player {player_id}: {report.summary()}", file=sys.stderr)
    if not report.ok:
        raise DataQualityError(
            f"Pulled data for player {player_id} failed validation: {report.summary()}"
        )


def _pitcher_report(df: pd.DataFrame, name: str, player_id: int, season: int) -> str:
    """Render the pitcher HTML from a one-pitcher DataFrame.

    `player_id` + `season` drive the league percentile lookup (Savant's pitcher
    percentile card).
    """
    arsenal = metrics.pitcher_arsenal(df)          # df is already one pitcher
    location = metrics.pitcher_location(df)
    splits = metrics.pitcher_handedness_splits(df)

    # Best-effort percentiles: a lookup failure must not sink the report.
    pctiles = None
    try:
        pctiles = percentiles.pitcher_percentiles(player_id, season) or None
    except Exception as exc:  # noqa: BLE001 — percentiles are optional context
        print(f"[percentiles] skipped for {name} ({season}): {exc}", file=sys.stderr)

    return pitcher_html_report(
        arsenal, location, splits,
        pitcher_name=name,
        throws=throws_label(df),
        pitches=df,                                # enables the Visuals charts
        percentiles=pctiles,
        percentile_season=season,
    )


def _hitter_report(df: pd.DataFrame, name: str, season: int) -> tuple[str, int]:
    """Render the hitter HTML from a one-batter DataFrame; return (html, n_bbe).

    `season` selects which league leaderboard to rank the hitter against for the
    percentile section.
    """
    bbe = metrics.batted_balls(df)
    overall = metrics.hitter_batted_ball(df)
    # Platoon splits are a bonus — degrade gracefully if p_throws is missing.
    try:
        platoon = metrics.hitter_platoon_batted_ball(df)
    except (KeyError, ValueError):
        platoon = None

    # League percentiles are best-effort: a baseline-pull failure (network,
    # off-season) must not sink the whole report — it just renders without them.
    pctiles = None
    try:
        pctiles = percentiles.hitter_percentiles(overall.iloc[0], season)
    except Exception as exc:  # noqa: BLE001 — percentiles are optional context
        print(f"[percentiles] skipped for {name} ({season}): {exc}", file=sys.stderr)

    html = hitter_html_report(
        overall, platoon,
        hitter_name=name,
        bats=bats_label(df),
        batted_balls_df=bbe,
        percentiles=pctiles,
        percentile_season=season,
    )
    return html, int(overall.iloc[0]["n_bbe"])


def generate_report(
    name: str,
    start: str,
    end: str,
    kind: str,
) -> ReportResult:
    """Resolve a player, pull their data, and render the report.

    Args:
        name:  player name ('First Last' or 'Last, First').
        start: 'YYYY-MM-DD' (inclusive).
        end:   'YYYY-MM-DD' (inclusive).
        kind:  'pitcher' or 'hitter'.

    Returns a ReportResult (html + metadata).

    Raises:
        ValueError:      unknown kind.
        PlayerNotFound / AmbiguousPlayer: from resolve_player.
        NoData:          the player has no pitches / batted balls in the range.
    """
    if kind not in ("pitcher", "hitter"):
        raise ValueError(f"kind must be 'pitcher' or 'hitter', got {kind!r}.")

    player_id, canonical = resolve_player(name)

    cache_key = (player_id, start, end, kind)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    df = _pull_player(player_id, start, end, kind)

    if kind == "pitcher":
        if df.empty:
            raise NoData(
                f"No pitches found for {canonical} between {start} and {end}. "
                f"If {canonical} is a hitter, choose Hitter."
            )
        season = datetime.strptime(end, "%Y-%m-%d").year
        html = _pitcher_report(df, canonical, player_id, season)
        result = ReportResult(html, canonical, player_id, kind, len(df))
    else:
        if df.empty:
            raise NoData(
                f"No batted-ball data for {canonical} between {start} and {end}. "
                f"If {canonical} is a pitcher, choose Pitcher."
            )
        # Rank against the league season the range ends in.
        season = datetime.strptime(end, "%Y-%m-%d").year
        try:
            html, n_bbe = _hitter_report(df, canonical, season)
        except (KeyError, ValueError) as exc:
            # e.g. the player has pitches in range but zero balls in play.
            raise NoData(
                f"No batted balls for {canonical} between {start} and {end}."
            ) from exc
        result = ReportResult(html, canonical, player_id, kind, n_bbe)

    _cache_put(cache_key, result)
    return result
