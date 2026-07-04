"""
Chart module: render the scouting-report visuals as static PNGs.

Every function returns a base64-encoded PNG **data URI** (a string starting
`data:image/png;base64,...`) so the HTML report can embed the image inline with
no external files and no CDN — the whole report travels as one self-contained
.html that opens offline and prints to PDF cleanly.

We use matplotlib's non-interactive "Agg" backend (no display needed) and scipy's
gaussian_kde for the location heatmaps.

Public functions (each returns a data-URI string):
    pitch_usage_bar(arsenal)
    velocity_bar(arsenal)
    movement_plot(arsenal, throws)
    location_heatmaps(pitches, arsenal)
    handedness_split_bars(splits)
"""

from __future__ import annotations

import base64
import contextlib
import io

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

try:
    from .report import pitch_display_name
    from .metrics import _HARD_HIT_MPH, _SWEET_SPOT_LA, _BARREL_BUCKET
except ImportError:
    from report import pitch_display_name
    from metrics import _HARD_HIT_MPH, _SWEET_SPOT_LA, _BARREL_BUCKET


# ---------------------------------------------------------------------------
# Shared style — mirror the report's serif / ink palette
# ---------------------------------------------------------------------------

_INK = "#1f2a34"
_GRID = "#d7d2c8"

# Pitch-type colors chosen to match the V1 report's palette.
_PITCH_COLORS: dict[str, str] = {
    "FF": "#cc3552",  # 4-seam — crimson
    "FA": "#cc3552",
    "SI": "#e58f3c",  # sinker — orange
    "FC": "#8a4b38",  # cutter — brown
    "SL": "#e6d417",  # slider — yellow
    "ST": "#c9b81e",  # sweeper
    "SV": "#c9b81e",
    "CU": "#27c1dd",  # curveball — cyan
    "KC": "#27c1dd",
    "CS": "#27c1dd",
    "CH": "#2ca02c",  # changeup — green
    "FS": "#17a77e",  # splitter — teal
    "FO": "#17a77e",
}
_DEFAULT_COLOR = "#7f8c8d"  # gray fallback for unknown codes

# Batter-hand colors for the handedness-split chart (blue vs green, per V1).
_HAND_COLORS = {"R": "#1f5f8b", "L": "#2f9e6f"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Georgia", "Times New Roman", "DejaVu Serif"],
    "text.color": _INK,
    "axes.edgecolor": _INK,
    "axes.labelcolor": _INK,
    "xtick.color": _INK,
    "ytick.color": _INK,
    "axes.titlecolor": _INK,
})


def _color(code: str) -> str:
    """Color for a pitch-type code, gray if we don't have one mapped."""
    return _PITCH_COLORS.get(code, _DEFAULT_COLOR)


@contextlib.contextmanager
def _figure(*args, **kwargs):
    """Yield (fig, ax) from plt.subplots and ALWAYS close the figure after.

    matplotlib keeps every figure in a global registry, so a figure that isn't
    closed leaks memory. Putting the close in a `finally` means it runs even if
    the chart-building code raises partway through (e.g. a KDE blows up) — the
    happy-path-only `plt.close` we had before leaked on exceptions.
    """
    fig, ax = plt.subplots(*args, **kwargs)
    try:
        yield fig, ax
    finally:
        plt.close(fig)


def _fig_to_data_uri(fig) -> str:
    """Serialize a matplotlib figure to a base64 PNG data URI (does not close).

    Closing is the `_figure` context manager's job, so this is safe to call
    inside a `with _figure(...)` block — the URI is built before the finally
    runs, and the figure is closed on every path afterward.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _require_rows(df: pd.DataFrame, chart: str) -> None:
    """Guard: a chart can't be drawn from an empty frame. Fail loudly, early.

    scout.py already exits before this, but stating the precondition here makes
    the chart functions safe to reuse from a notebook without that guard.
    """
    if df.empty:
        raise ValueError(f"{chart}: no rows to plot.")


# ---------------------------------------------------------------------------
# 1. Pitch usage — vertical bars
# ---------------------------------------------------------------------------

def pitch_usage_bar(arsenal: pd.DataFrame) -> str:
    """Vertical bar chart of each pitch type's usage %, primary pitch first."""
    _require_rows(arsenal, "pitch_usage_bar")

    names = [pitch_display_name(c) for c in arsenal["pitch_type"]]
    usage_pct = arsenal["usage"] * 100  # decimal -> display percent
    colors = [_color(c) for c in arsenal["pitch_type"]]

    with _figure(figsize=(9, 4.5)) as (fig, ax):
        bars = ax.bar(names, usage_pct, color=colors, edgecolor=_INK, linewidth=0.5)

        # Label each bar with its exact percentage just above the top.
        for bar, pct in zip(bars, usage_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=10,
            )

        ax.set_ylabel("Usage (%)")
        ax.set_ylim(0, max(usage_pct) * 1.15)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)  # grid behind the bars

        return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# 2. Velocity — horizontal bars
# ---------------------------------------------------------------------------

def velocity_bar(arsenal: pd.DataFrame) -> str:
    """Horizontal bars of average velocity, hardest pitch on top."""
    _require_rows(arsenal, "velocity_bar")
    data = arsenal.dropna(subset=["avg_velo"]).sort_values("avg_velo")
    # Every pitch type can legitimately lack velocity (older pull without
    # release_speed). Empty here would make min()/max() NaN and crash set_xlim.
    if data.empty:
        raise ValueError("velocity_bar: no pitch has a velocity to plot.")

    names = [pitch_display_name(c) for c in data["pitch_type"]]
    colors = [_color(c) for c in data["pitch_type"]]

    with _figure(figsize=(9, 4.5)) as (fig, ax):
        bars = ax.barh(names, data["avg_velo"], color=colors, edgecolor=_INK, linewidth=0.5)

        for bar, velo in zip(bars, data["avg_velo"]):
            ax.text(
                bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{velo:.1f}", ha="left", va="center", fontsize=10,
            )

        ax.set_xlabel("Velocity (mph)")
        # Start the axis a bit below the slowest pitch so bars have visible length.
        lo = data["avg_velo"].min() - 3
        hi = data["avg_velo"].max() + 2
        ax.set_xlim(lo, hi)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# 3. Movement plot — HB vs VB scatter
# ---------------------------------------------------------------------------

def movement_plot(arsenal: pd.DataFrame, throws: str) -> str:
    """Scatter of horizontal vs induced-vertical break, one dot per pitch type.

    HB uses the arm-side-positive convention from metrics.py. Because that makes
    positive = the pitcher's arm side only for a RHP (a LHP's arm side is
    negative), the x-axis end labels are chosen from `throws` so "Arm-side"
    always points the right way for the pitcher in question.
    """
    _require_rows(arsenal, "movement_plot")
    data = arsenal.dropna(subset=["h_break_in", "v_break_in"])

    with _figure(figsize=(7.5, 7)) as (fig, ax):
        # Concentric dashed rings give a sense of total break magnitude (inches).
        for r in (10, 20, 30):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, linestyle="--",
                                     edgecolor=_GRID, linewidth=0.8))
        # Axes through the origin.
        ax.axhline(0, color=_GRID, linewidth=0.8)
        ax.axvline(0, color=_GRID, linewidth=0.8)

        # iterrows is fine here: at most ~10 pitch types, and we need both the
        # point and its label per row — clearer than parallel column iteration.
        for _, row in data.iterrows():
            ax.scatter(row["h_break_in"], row["v_break_in"], s=180,
                       color=_color(row["pitch_type"]), edgecolor=_INK,
                       linewidth=0.8, zorder=3)
            ax.annotate(
                pitch_display_name(row["pitch_type"]),
                (row["h_break_in"], row["v_break_in"]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9,
            )

        # Square, symmetric limits so the rings read as circles and 0,0 is centered.
        lim = 30
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")

        # Handedness-aware horizontal label: arm-side is positive for a RHP.
        # Arrows use mathtext ($\leftarrow$) so they render even when the serif
        # font lacks the Unicode arrow glyphs (Georgia does).
        if throws.lower().startswith("l"):
            x_label = r"$\leftarrow$ Arm-side      Horizontal Break (in)      Glove-side $\rightarrow$"
        else:
            x_label = r"$\leftarrow$ Glove-side      Horizontal Break (in)      Arm-side $\rightarrow$"
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(
            r"$\leftarrow$ More Drop      Induced Vertical Break (in)      More Rise $\rightarrow$",
            fontsize=10,
        )

        ax.spines[["top", "right"]].set_visible(False)
        return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# 4. Location heatmaps — KDE per pitch type
# ---------------------------------------------------------------------------

# Strike-zone rectangle (feet): plate half-width ~0.83, typical vertical zone.
_ZONE_X = (-0.83, 0.83)
_ZONE_Z = (1.5, 3.5)
_MIN_KDE_PITCHES = 5  # below this, a KDE is unstable — fall back to a scatter


def _draw_zone(ax) -> None:
    """Overlay the strike-zone rectangle (catcher's view)."""
    ax.add_patch(Rectangle(
        (_ZONE_X[0], _ZONE_Z[0]),
        _ZONE_X[1] - _ZONE_X[0], _ZONE_Z[1] - _ZONE_Z[0],
        fill=False, edgecolor="black", linewidth=1.4,
    ))


def location_heatmaps(pitches: pd.DataFrame, arsenal: pd.DataFrame) -> str:
    """Small-multiple KDE heatmaps of plate location, one panel per pitch type.

    Args:
        pitches: the pitcher's raw pitch-level rows (needs plate_x / plate_z).
        arsenal: used only to order panels by usage and label counts.

    Each panel is a blue→white→red density (red = where the pitch clusters)
    with the strike zone drawn on top, matching the V1 report.
    """
    _require_rows(arsenal, "location_heatmaps")
    codes = list(arsenal["pitch_type"])  # already usage-sorted
    n = len(codes)
    ncols = 2
    nrows = (n + ncols - 1) // ncols  # ceil division

    # A fixed evaluation grid over the plate region shared by every panel.
    xx, zz = np.mgrid[-2:2:100j, 0:5:100j]
    grid = np.vstack([xx.ravel(), zz.ravel()])

    with _figure(nrows, ncols, figsize=(9, 4.2 * nrows)) as (fig, axes):
        # ncols is always 2, so subplots returns a 2D array already; atleast_2d
        # is a harmless safety net (it would only matter if ncols became 1).
        axes = np.atleast_2d(axes)

        for i, code in enumerate(codes):
            ax = axes[i // ncols][i % ncols]
            pts = pitches[pitches["pitch_type"] == code][["plate_x", "plate_z"]].dropna()

            ax.set_title(f"{pitch_display_name(code)} ({len(pts)})", fontsize=11)
            ax.set_xlim(-2, 2)
            ax.set_ylim(0, 5)
            ax.set_xlabel("Plate X")
            ax.set_ylabel("Plate Z")

            # Draw a KDE density when there are enough points. gaussian_kde
            # raises LinAlgError on a singular covariance matrix — which happens
            # not just when an axis is constant but whenever the points are
            # collinear — so we catch that and fall back to a raw scatter rather
            # than trying to pre-detect every degenerate geometry.
            drew_kde = False
            if len(pts) >= _MIN_KDE_PITCHES:
                try:
                    kde = gaussian_kde(np.vstack([pts["plate_x"], pts["plate_z"]]))
                    density = kde(grid).reshape(xx.shape)
                    ax.contourf(xx, zz, density, levels=12, cmap="RdBu_r")
                    drew_kde = True
                except np.linalg.LinAlgError:
                    drew_kde = False
            if not drew_kde:
                ax.scatter(pts["plate_x"], pts["plate_z"], s=20, color=_color(code),
                           edgecolor=_INK, linewidth=0.4)

            _draw_zone(ax)

        # Blank out any unused panel in the last row (odd number of pitch types).
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout()
        return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# 5. Handedness split usage — grouped bars
# ---------------------------------------------------------------------------

def handedness_split_bars(splits: pd.DataFrame) -> str:
    """Grouped bars: each pitch type's usage vs RHB and vs LHB, side by side."""
    _require_rows(splits, "handedness_split_bars")

    # Pivot to pitch_type × hand, values = usage%. Order pitches by overall use.
    wide = splits.pivot_table(
        index="pitch_type", columns="stand", values="usage", aggfunc="first"
    ).fillna(0.0) * 100

    # We only know how to draw the R/L platoons; a stray `stand` value would
    # otherwise render an empty chart with no error. Fail visibly instead.
    if not ({"R", "L"} & set(wide.columns)):
        raise ValueError(
            f"handedness_split_bars: expected 'R'/'L' in stand, got {list(wide.columns)}."
        )

    order = (wide.sum(axis=1)).sort_values(ascending=False).index
    wide = wide.loc[order]

    names = [pitch_display_name(c) for c in wide.index]
    x = np.arange(len(names))
    width = 0.38

    with _figure(figsize=(9, 4.5)) as (fig, ax):
        # Draw R and L groups if present; skip a hand that isn't in the data.
        if "R" in wide.columns:
            ax.bar(x - width / 2, wide["R"], width, label="vs RHB",
                   color=_HAND_COLORS["R"], edgecolor=_INK, linewidth=0.5)
        if "L" in wide.columns:
            ax.bar(x + width / 2, wide["L"], width, label="vs LHB",
                   color=_HAND_COLORS["L"], edgecolor=_INK, linewidth=0.5)

        ax.set_ylabel("Usage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)


# ---------------------------------------------------------------------------
# Hitter batted-ball charts
# ---------------------------------------------------------------------------
# Reference lines reuse the metric thresholds imported from metrics.py
# (hard-hit 95 mph, sweet-spot LA 8–32°, barrel = launch_speed_angle == 6) so a
# chart's reference line can never drift out of sync with the numbers beside it.
_CRIMSON = "#cc3552"
_SWEET_SPOT_SHADE = "#2ca02c"


def ev_distribution(bbe: pd.DataFrame) -> str:
    """Histogram of exit velocity, with the 95 mph hard-hit line marked."""
    _require_rows(bbe, "ev_distribution")
    # _require_rows only checks row count; the column itself can be all-NaN on
    # a pull that never tracked exit velocity — guard that too so we fail loudly
    # instead of embedding a blank histogram.
    ev = bbe["launch_speed"].dropna()
    if ev.empty:
        raise ValueError("ev_distribution: no non-null launch_speed values to plot.")

    with _figure(figsize=(9, 4.2)) as (fig, ax):
        ax.hist(ev, bins=20, color=_CRIMSON, edgecolor=_INK, linewidth=0.5)
        ax.axvline(_HARD_HIT_MPH, color=_INK, linestyle="--", linewidth=1.2)
        # Place the label near the top of the axis so it doesn't collide with bars.
        ax.text(_HARD_HIT_MPH + 0.5, ax.get_ylim()[1] * 0.92,
                "95 mph — hard-hit", fontsize=9, va="top")

        ax.set_xlabel("Exit Velocity (mph)")
        ax.set_ylabel("Batted balls")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)


def launch_angle_distribution(bbe: pd.DataFrame) -> str:
    """Histogram of launch angle, with the 8–32° sweet-spot band shaded."""
    _require_rows(bbe, "launch_angle_distribution")
    la = bbe["launch_angle"].dropna()
    if la.empty:
        raise ValueError("launch_angle_distribution: no non-null launch_angle values to plot.")

    with _figure(figsize=(9, 4.2)) as (fig, ax):
        # Shade the sweet-spot band first so bars sit on top of it.
        ax.axvspan(*_SWEET_SPOT_LA, color=_SWEET_SPOT_SHADE, alpha=0.15,
                   label="Sweet spot (8–32°)")
        ax.hist(la, bins=24, color="#1f5f8b", edgecolor=_INK, linewidth=0.5)

        ax.set_xlabel("Launch Angle (°)")
        ax.set_ylabel("Batted balls")
        ax.legend(frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)


def ev_la_scatter(bbe: pd.DataFrame) -> str:
    """Exit velocity vs launch angle, with barrels highlighted.

    Rather than reimplement the barrel region (a curved EV/LA lookup), we color
    each ball by Savant's own classification (launch_speed_angle == 6). The
    hard-hit line and sweet-spot band give context for where damage lives.
    """
    _require_rows(bbe, "ev_la_scatter")
    d = bbe.dropna(subset=["launch_speed", "launch_angle"])
    if d.empty:
        raise ValueError("ev_la_scatter: no batted balls with both EV and LA to plot.")

    # Barrel flag: only if the classification column is present in this pull.
    if "launch_speed_angle" in d.columns:
        is_barrel = d["launch_speed_angle"] == _BARREL_BUCKET
    else:
        is_barrel = pd.Series(False, index=d.index)

    with _figure(figsize=(8, 6)) as (fig, ax):
        ax.axhline(_HARD_HIT_MPH, color=_INK, linestyle="--", linewidth=1.0)
        ax.axvspan(*_SWEET_SPOT_LA, color=_SWEET_SPOT_SHADE, alpha=0.12)

        non = d[~is_barrel]
        bar = d[is_barrel]
        ax.scatter(non["launch_angle"], non["launch_speed"], s=26,
                   color="#9aa3ab", alpha=0.6, linewidth=0, label="Batted ball")
        ax.scatter(bar["launch_angle"], bar["launch_speed"], s=40,
                   color=_CRIMSON, edgecolor=_INK, linewidth=0.5, label="Barrel",
                   zorder=3)

        ax.set_xlabel("Launch Angle (°)")
        ax.set_ylabel("Exit Velocity (mph)")
        ax.legend(frameon=False, loc="lower center", ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=_GRID, linewidth=0.5)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)
