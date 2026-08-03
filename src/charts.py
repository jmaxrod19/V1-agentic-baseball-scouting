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
import textwrap

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle
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

# Pitch FAMILIES — the classic scouting split, so every pitch a hitter sees is
# covered by one of a few panels (no pitch type gets orphaned). Codes not listed
# fall into "Other".
_PITCH_FAMILY_MAP: dict[str, str] = {
    # Fastballs (incl. cutter, a fastball-velocity pitch).
    "FF": "Fastball", "FA": "Fastball", "FT": "Fastball", "SI": "Fastball", "FC": "Fastball",
    # Breaking balls.
    "SL": "Breaking", "ST": "Breaking", "SV": "Breaking", "CU": "Breaking",
    "KC": "Breaking", "CS": "Breaking",
    # Offspeed.
    "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed", "SC": "Offspeed",
    "EP": "Offspeed", "KN": "Offspeed",
}
_FAMILY_ORDER = ["Fastball", "Breaking", "Offspeed", "Other"]
_FAMILY_COLORS = {
    "Fastball": "#cc3552",   # crimson
    "Breaking": "#27c1dd",   # cyan
    "Offspeed": "#2ca02c",   # green
    "Other": _DEFAULT_COLOR,
}


def _pitch_family(code: str) -> str:
    """Map a Statcast pitch-type code to Fastball / Breaking / Offspeed / Other."""
    return _PITCH_FAMILY_MAP.get(code, "Other")


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


def _fig_to_data_uri(fig, facecolor: str = "white") -> str:
    """Serialize a matplotlib figure to a base64 PNG data URI (does not close).

    Closing is the `_figure` context manager's job, so this is safe to call
    inside a `with _figure(...)` block — the URI is built before the finally
    runs, and the figure is closed on every path afterward. `facecolor` sets the
    exported background (white by default; dark themes pass their own).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=facecolor)
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


_SLATE = "#9aa3ab"  # muted gray-blue for "everything below the threshold"


def ev_distribution(bbe: pd.DataFrame) -> str:
    """Histogram of exit velocity, hard-hit balls shaded crimson.

    Upgrades over a plain histogram:
    - Bars sit on a fixed 5-mph grid, so a given exit velocity always lands in
      the same bin regardless of who's hitting — two players' charts line up.
    - Because 95 (the hard-hit cutoff) is itself a multiple of 5, it falls on a
      bin *edge*: every bar is wholly hard-hit or wholly not, so we can color the
      hard-hit bars crimson and the rest slate without any bar straddling the line.
    - A solid line marks the player's mean EV, and a caption states the two
      numbers the chart is really about (avg EV, hard-hit %) so the picture and
      the metric table can never disagree.
    """
    _require_rows(bbe, "ev_distribution")
    # _require_rows only checks row count; the column itself can be all-NaN on
    # a pull that never tracked exit velocity — guard that too so we fail loudly
    # instead of embedding a blank histogram.
    ev = bbe["launch_speed"].dropna()
    if ev.empty:
        raise ValueError("ev_distribution: no non-null launch_speed values to plot.")

    # Snap the range out to the enclosing multiples of 5 so every edge is on the
    # shared grid (…, 90, 95, 100, …). np.floor/ceil on ev/5 then *5 rounds the
    # min down and the max up; +5 on the stop makes arange include the top edge.
    lo = np.floor(ev.min() / 5) * 5
    hi = np.ceil(ev.max() / 5) * 5
    edges = np.arange(lo, hi + 5, 5)

    with _figure(figsize=(9, 4.2)) as (fig, ax):
        # hist returns (counts, edges, patches); we keep the patches to recolor
        # each bar individually after the fact.
        _, _, patches = ax.hist(ev, bins=edges, edgecolor=_INK, linewidth=0.5)
        for patch in patches:
            # patch.get_x() is the bar's left edge; a bar at >= 95 is hard-hit.
            hard = patch.get_x() >= _HARD_HIT_MPH
            patch.set_facecolor(_CRIMSON if hard else _SLATE)

        ax.axvline(_HARD_HIT_MPH, color=_INK, linestyle="--", linewidth=1.2)
        # Label to the RIGHT of the 95 line so it reads into the crimson region.
        ax.text(_HARD_HIT_MPH + 0.5, ax.get_ylim()[1] * 0.92,
                "95 mph — hard-hit", fontsize=9, va="top", ha="left")

        # Mean marker: solid line + a value label sitting to its LEFT (ha="right")
        # so it can't collide with the hard-hit label to the right of the 95 line.
        mean_ev = ev.mean()
        ax.axvline(mean_ev, color=_INK, linestyle="-", linewidth=1.0)
        ax.text(mean_ev - 0.5, ax.get_ylim()[1] * 0.92,
                f"Avg {mean_ev:.1f}", fontsize=9, va="top", ha="right")

        # Caption in the upper-left tying the chart to the numbers beside it.
        hard_hit_pct = (ev >= _HARD_HIT_MPH).mean() * 100
        ax.text(0.02, 0.97, f"Hard-hit {hard_hit_pct:.0f}%  ·  n={len(ev)}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                color=_INK)

        ax.set_xlabel("Exit Velocity (mph)")
        ax.set_ylabel("Batted balls")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)

        return _fig_to_data_uri(fig)


def launch_angle_distribution(bbe: pd.DataFrame) -> str:
    """Histogram of launch angle, with the 8–32° sweet-spot band shaded.

    Same upgrades as the EV histogram: a fixed 5° bin grid so bars line up across
    players (and so the batted-ball-type zones below always fall on the same
    edges), plus a mean-LA marker and a sweet-spot-% caption. Launch angle is
    strongly bimodal (grounders vs. balls in the air), which fixed edges preserve
    honestly instead of letting a data-dependent bin count smear the two humps.
    """
    _require_rows(bbe, "launch_angle_distribution")
    la = bbe["launch_angle"].dropna()
    if la.empty:
        raise ValueError("launch_angle_distribution: no non-null launch_angle values to plot.")

    # 5° grid snapped out to enclosing multiples of 5 (see ev_distribution for the
    # floor/ceil trick). Launch angle runs roughly -90..90, so this stays compact.
    lo = np.floor(la.min() / 5) * 5
    hi = np.ceil(la.max() / 5) * 5
    edges = np.arange(lo, hi + 5, 5)

    with _figure(figsize=(9, 4.2)) as (fig, ax):
        # Shade the sweet-spot band first so bars sit on top of it.
        ax.axvspan(*_SWEET_SPOT_LA, color=_SWEET_SPOT_SHADE, alpha=0.15,
                   label="Sweet spot (8–32°)")
        ax.hist(la, bins=edges, color="#1f5f8b", edgecolor=_INK, linewidth=0.5)

        # Mean-LA marker so the reader can place the center of mass at a glance.
        mean_la = la.mean()
        ax.axvline(mean_la, color=_INK, linestyle="-", linewidth=1.0)
        # Offset the label to the right of the line so the line doesn't bisect it.
        ax.text(mean_la + 1.5, ax.get_ylim()[1] * 0.90, f"Avg {mean_la:.1f}°",
                fontsize=9, va="top", ha="left")

        # Caption: sweet-spot rate is the number this chart is really about.
        lo_ss, hi_ss = _SWEET_SPOT_LA
        sweet_pct = ((la >= lo_ss) & (la <= hi_ss)).mean() * 100
        ax.text(0.02, 0.97, f"Sweet-spot {sweet_pct:.0f}%  ·  n={len(la)}",
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                color=_INK)

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


# ---------------------------------------------------------------------------
# Swing geometry — stance + contact map (top-down "batter's box" view)
# ---------------------------------------------------------------------------
# A bird's-eye view of the batter's box, in the same frame Savant uses, drawn
# entirely from our own data:
#   * home plate as the fixed anchor (right side)
#   * the batter's two feet, placed from stance depth, distance-off-plate, foot
#     separation, and stance angle
#   * the intercept cloud — one dot per swing where the bat met the ball — placed
#     in real box space via the mapping verified against Savant's own numbers:
#         off_plate = batter_x - intercept_side   (toward the plate)
#         depth     = batter_y - intercept_depth  (out in front)
# Our creative layer on top: the cloud is colored BY PITCH TYPE, with a soft
# 1.5-sigma ellipse per pitch type — so you can see, in real space, where he
# meets fastballs vs. breaking balls. Savant's tool doesn't segment contact that
# way; that's the piece that makes this ours.
#
# Coordinates are in inches. Axes are drawn "Savant-style": distance off plate
# grows to the LEFT, depth grows DOWNWARD (0 = front of plate, toward pitcher),
# so the plate sits at the right and the batter/box at the left.

# Home-plate outline as (off_plate, depth) vertices, inches. Inside edge at
# off_plate 0; the 17"-wide front faces the pitcher (depth 0); the point faces
# the catcher (positive depth).
_PLATE_XY = [(0, 0), (-17, 0), (-17, 8.5), (-8.5, 17), (0, 8.5)]
# Center of home plate (off_plate, depth): middle of the 17" width, mid-depth.
# Contact distance is measured from here.
_PLATE_CENTER = (-8.5, 8.5)

# Aesthetic themes for the stance/contact map. Only presentation colors change;
# the pitch-type palette (_PITCH_COLORS) stays because those colors carry meaning.
# `zone_alpha` is the per-pitch ellipse fill opacity — higher on the dark theme
# so the zones don't wash out against the background.
_STANCE_THEMES: dict[str, dict] = {
    # Crisp white report style (matches the rest of the report).
    "report":   {"bg": "white",   "ink": "#1f2a34", "grid": "#d7d2c8",
                 "foot": "#b7b0a4", "measure": "#e23b3b", "caption": "#fdf6e3",
                 "zone_alpha": 0.12},
    # Warm cream "paper" look — softer, less clinical.
    "paper":    {"bg": "#f4ecd8", "ink": "#3b3323", "grid": "#cdbf9f",
                 "foot": "#a99c7f", "measure": "#c0492f", "caption": "#eaddc0",
                 "zone_alpha": 0.14},
    # Dark "midnight" look — high contrast, modern.
    "midnight": {"bg": "#161b22", "ink": "#e7e2d7", "grid": "#39424c",
                 "foot": "#7f796b", "measure": "#ff6b6b", "caption": "#222a33",
                 "zone_alpha": 0.30},
}


class _HandlerEllipse(HandlerPatch):
    """Legend handler that draws the swatch as an ellipse (for feet / zones)."""

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        e = Ellipse((width / 2 - xd, height / 2 - yd), width, height * 0.95)
        e.set_facecolor(orig.get_facecolor())
        e.set_edgecolor(orig.get_edgecolor())
        e.set_alpha(orig.get_alpha())
        e.set_linewidth(orig.get_linewidth())
        e.set_transform(trans)
        return [e]


class _HandlerFoot(HandlerPatch):
    """Legend handler that draws the swatch as a foot silhouette."""

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        o = _FOOT_BODY.copy()
        # Fit the outline into the legend swatch box (toe up).
        xs, ys = o[:, 0], o[:, 1]
        nx = (xs - xs.min()) / (xs.max() - xs.min()) * width - xd
        ny = (ys - ys.min()) / (ys.max() - ys.min()) * height - yd
        p = Polygon(np.column_stack([nx, ny]), closed=True,
                    facecolor=orig.get_facecolor(), edgecolor=orig.get_edgecolor())
        p.set_transform(trans)
        return [p]


class _HandlerPlate(HandlerPatch):
    """Legend handler that draws the swatch as a home-plate pentagon."""

    def create_artists(self, legend, orig, xd, yd, width, height, fontsize, trans):
        w, h = width, height
        verts = [(0, h), (w, h), (w, 0.45 * h), (0.5 * w, 0), (0, 0.45 * h)]
        verts = [(x - xd, y - yd) for x, y in verts]
        p = Polygon(verts, closed=True, facecolor="none",
                    edgecolor=orig.get_edgecolor(), linewidth=orig.get_linewidth())
        p.set_transform(trans)
        return [p]


def _style_axis(ax, pal: dict) -> None:
    """Apply a theme's background / ink colors to one panel's axis furniture."""
    ax.set_facecolor(pal["bg"])
    for spine in ax.spines.values():
        spine.set_color(pal["ink"])
    ax.tick_params(colors=pal["ink"], labelsize=8)


def _draw_batters_box(ax, pal: dict) -> None:
    """Draw the true-size MLB batter's box (chalk outline) on the batter's side."""
    b = _BATTERS_BOX
    ax.add_patch(Rectangle(
        (b["off_lo"], b["depth_lo"]), b["off_hi"] - b["off_lo"],
        b["depth_hi"] - b["depth_lo"], fill=False, edgecolor=pal["ink"],
        linewidth=1.2, alpha=0.55, zorder=1,
    ))


# A bat is wood-toned regardless of theme.
_BAT_COLOR = "#b5793f"


def _draw_bat(ax, x0: float, y0: float, x1: float, y1: float, pal: dict) -> None:
    """Draw a tapered bat from the hands (x0, y0) toward a contact point (x1, y1).

    A schematic: narrow at the knob/handle, widening to the barrel, with the
    barrel end reaching just past the contact point. Conveys "the barrel gets
    out to here" without pretending to be a physically exact swing path.
    """
    d = np.array([x1 - x0, y1 - y0], dtype=float)
    length = float(np.hypot(*d))
    if length < 1e-6:
        return
    u = d / length                       # along the bat
    perp = np.array([-u[1], u[0]])       # across the bat
    knob = np.array([x0, y0]) - u * 1.5  # a touch behind the hands
    tip = np.array([x1, y1]) + u * 2.5   # barrel reaches just past contact
    hw_handle, hw_barrel = 0.7, 1.7      # half-widths (bat barrel ~2.6" across)
    poly = [knob + perp * hw_handle, tip + perp * hw_barrel,
            tip - perp * hw_barrel, knob - perp * hw_handle]
    ax.add_patch(Polygon(poly, closed=True, facecolor=_BAT_COLOR,
                         edgecolor=pal["ink"], linewidth=0.6, zorder=6))
    ax.add_patch(Ellipse(tuple(knob), 2.4, 2.4, facecolor=_BAT_COLOR,
                         edgecolor=pal["ink"], linewidth=0.6, zorder=6))  # knob

# Shared axis limits for every panel (inches), so all the mini-boxes are drawn
# to the exact same scale and can be compared at a glance.
_BOX_XLIM = (57, -20)   # inverted: distance off plate grows to the LEFT
_BOX_YLIM = (48, -30)   # inverted: depth grows DOWNWARD, front of plate at top

# True MLB batter's box (inches), in (off_plate, depth): 4 ft x 6 ft, inner edge
# 6" off the plate, centered lengthwise on the plate's center (depth 8.5").
# off_plate 6 -> 54 (6" gap + 48" wide); depth -27.5 -> 44.5 (72" long).
_BATTERS_BOX = {"off_lo": 6.0, "off_hi": 54.0, "depth_lo": -27.5, "depth_hi": 44.5}

# Below this many swings a pitch type doesn't get its own panel (too small a
# sample to read a zone from); those swings still appear in the "All pitches" one.
_MIN_PANEL_SWINGS = 20
# Below this the covariance ellipse is unstable, so we draw the dots without it.
_MIN_ELLIPSE_SWINGS = 12
# Cap the number of per-pitch panels so the grid stays compact.
_MAX_PITCH_TYPES = 5


def _pitch_zone_ellipse(ax, xs: pd.Series, ys: pd.Series, color: str,
                        fill_alpha: float = 0.12) -> None:
    """Draw a soft 1.5-sigma covariance ellipse around a pitch type's contacts.

    The ellipse's shape/orientation come from the 2x2 covariance of the points:
    its eigenvectors give the axis directions and the eigenvalues (variances)
    give the spread along each. This is the standard way to summarize a 2D point
    cloud as one shape — a compact "this is his contact zone for this pitch."
    """
    pts = np.vstack([xs.to_numpy(), ys.to_numpy()])
    cov = np.cov(pts)
    # eigh returns eigenvalues ascending; both are variances (>= 0). A singular
    # cloud (collinear points) can yield a tiny negative from rounding — clip.
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    # Angle of the largest-variance eigenvector, in degrees, for the ellipse tilt.
    major = vecs[:, np.argmax(vals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    # Full width/height = 2 * n_std * sqrt(variance) along each axis.
    width, height = 2 * 1.5 * np.sqrt(np.sort(vals)[::-1])
    # Translucent fill (alpha only on the FILL via rgba) + a solid outline, so
    # overlapping zones stay readable and the outline stays crisp on any theme.
    ax.add_patch(Ellipse(
        (float(xs.mean()), float(ys.mean())), width, height, angle=angle,
        facecolor=to_rgba(color, fill_alpha), edgecolor=color, linewidth=1.6, zorder=2,
    ))


# A stylized foot in a local frame: toe at +y, heel at -y, arch on the -x
# (inner) side. The sole is a polygon; the five toes are separate ellipses laid
# on the front so the foot reads clearly from above. Mirrored on x for the pair.
_FOOT_BODY = np.array([
    (0.00, -0.58), (0.18, -0.48), (0.25, -0.12), (0.31, 0.22), (0.24, 0.44),
    (0.05, 0.50), (-0.16, 0.45), (-0.25, 0.30), (-0.12, 0.02),
    (-0.23, -0.24), (-0.17, -0.48),
])
# Each toe: (x, y, radius_x, radius_y) — big toe (inner/-x, largest) to pinky.
_FOOT_TOES = [
    (-0.15, 0.54, 0.10, 0.14),
    (-0.03, 0.62, 0.075, 0.115),
    (0.07, 0.62, 0.065, 0.105),
    (0.16, 0.57, 0.060, 0.095),
    (0.24, 0.49, 0.050, 0.080),
]
_FOOT_LEN = 13.0   # inches, long axis (bigger than the old ellipse)
_FOOT_WID = 11.0   # scale for the normalized x extent
# Feet point toes straight at the plate (a batter stands sideways to the pitcher
# with toes toward the plate). 0 = no cant toward pitcher/catcher; the hitter's
# stance angle is still applied on top for open/closed.
_FOOT_TILT = 0.0


def _foot_transform(pts, x: float, y: float, angle_deg: float, mirror: bool) -> np.ndarray:
    """Scale/mirror/rotate local foot points and translate to (x, y) in data space.

    Rotates so the toe (+y) points toward the plate, canted toward the pitcher by
    _FOOT_TILT, plus the hitter's stance angle (open/closed) via `angle_deg`.
    """
    p = np.array(pts, dtype=float).reshape(-1, 2)
    if mirror:
        p = p * np.array([-1.0, 1.0])            # left vs right foot
    p = p * np.array([_FOOT_WID, _FOOT_LEN])
    theta = np.radians(90 + _FOOT_TILT + angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    p = p @ np.array([[c, -s], [s, c]]).T
    return p + np.array([x, y])


def _draw_foot(ax, x: float, y: float, angle_deg: float, pal: dict,
               mirror: bool = False) -> None:
    """Draw one foot: a sole silhouette plus five toe ellipses (anatomical anchor)."""
    sole = _foot_transform(_FOOT_BODY, x, y, angle_deg, mirror)
    ax.add_patch(Polygon(sole, closed=True, facecolor=pal["foot"],
                         edgecolor=pal["ink"], linewidth=0.8, zorder=5))
    # Toes: transform each center, and rotate the ellipse itself to match.
    toe_angle = 90 + _FOOT_TILT + angle_deg
    for tx, ty, rx, ry in _FOOT_TOES:
        cx, cy = _foot_transform([(tx, ty)], x, y, angle_deg, mirror)[0]
        ax.add_patch(Ellipse(
            (cx, cy), width=2 * rx * _FOOT_WID, height=2 * ry * _FOOT_LEN,
            angle=toe_angle, facecolor=pal["foot"], edgecolor=pal["ink"],
            linewidth=0.6, zorder=6,
        ))


def _draw_stance_box(ax, bx: float, by: float, foot_sep, angle: float, pal: dict,
                     show_box: bool = False) -> None:
    """Draw the shared frame for one panel: home plate, the two feet, the grid.

    Everything that's identical across panels lives here so each mini-box is
    drawn the same way and to the same scale — that sameness is what lets the
    panels be compared side by side.
    """
    _style_axis(ax, pal)

    if show_box:
        _draw_batters_box(ax, pal)

    # Feet: front foot toward the pitcher (smaller depth); the foot-line is the
    # "toward pitcher" direction (0,-1) rotated by the stance angle.
    if foot_sep is not None and not pd.isna(foot_sep):
        theta = np.radians(angle)
        fx, fy = np.sin(theta), -np.cos(theta)
        half = foot_sep / 2.0
        # mirror chosen so the insteps (arches) face each other (inward), like a
        # real stance — front foot toward the pitcher, back foot toward the catcher.
        _draw_foot(ax, bx + half * fx, by + half * fy, angle, pal, mirror=True)   # front
        _draw_foot(ax, bx - half * fx, by - half * fy, angle, pal, mirror=False)  # back

    # Home plate as the anchor, on top so the cloud never buries it.
    ax.add_patch(Polygon(_PLATE_XY, closed=True, facecolor="none",
                         edgecolor=pal["ink"], linewidth=1.4, zorder=7))

    ax.set_xlim(*_BOX_XLIM)
    ax.set_ylim(*_BOX_YLIM)
    ax.set_aspect("equal")
    ax.grid(color=pal["grid"], linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# Red, like Savant's own measurement callouts (default; themes may override).
_MEASURE_RED = "#e23b3b"


def _draw_measurements(ax, bx: float, by: float, pal: dict) -> None:
    """Draw Savant-style red measurement lines to the stance center.

    An L from the plate's inside-front corner (0, 0): straight back to the
    stance depth, then out to the stance center, labeled with the two numbers
    Savant reports — depth in box (vertical) and distance off plate (horizontal).
    Makes explicit *where* the stance position is measured from.
    """
    red = pal["measure"]
    ax.plot([0, 0], [0, by], color=red, linestyle="--", linewidth=1.3, zorder=9)
    ax.plot([0, bx], [by, by], color=red, linestyle="--", linewidth=1.3, zorder=9)
    ax.scatter([0], [0], s=28, color=red, zorder=10)                                 # plate corner
    ax.scatter([bx], [by], s=44, facecolor=pal["bg"], edgecolor=red,
               linewidth=1.6, zorder=10)                                             # center of mass
    # Say what that endpoint is: Savant measures to the batter's center of mass.
    ax.text(bx, by + 3, "center of mass", color=red, fontsize=8, ha="center",
            va="top", zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=pal["bg"],
                      edgecolor=red, linewidth=0.8))

    # Labels sit on a filled pill so they stay legible over the dots/grid and on
    # the dark theme (bold, larger, high-contrast box behind the red text).
    label_box = dict(boxstyle="round,pad=0.25", facecolor=pal["bg"],
                     edgecolor=red, linewidth=1.0)
    ax.text(-3.0, by / 2, f'{by:.1f}"', color=red, fontsize=10, fontweight="bold",
            ha="left", va="center", zorder=11, bbox=label_box)                       # depth in box
    ax.text(bx / 2, by - 1.5, f'{bx:.1f}"', color=red, fontsize=10, fontweight="bold",
            ha="center", va="bottom", zorder=11, bbox=label_box)                     # distance off plate


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 78 -> '78th'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _biomech_caption(ax, biomech: dict, pal: dict) -> None:
    """Upper-left callout: height + how wide the stance is FOR HIS SIZE.

    Made deliberately prominent (bold, boxed, accent color) because it's the
    headline biomechanical read. The stance-width percentile is height-adjusted
    (see stance_biomech). Phrasing it as "wider than N% of hitters his size"
    makes the direction unmistakable — a high number means a WIDE stance for his
    height — without the reader having to know how to read a percentile.
    """
    height_text = biomech.get("height_text")
    width = biomech.get("stance_width")
    pct = biomech.get("stance_width_pct")
    if height_text is None and width is None:
        return

    # One tidy stat box: the hard numbers (height, stance width in inches) on top,
    # then the plain-language league context below.
    stat_rows = []
    if height_text:
        stat_rows.append(f"Height:  {height_text}")
    if width is not None:
        stat_rows.append(f'Stance width:  {width:.0f}"')
    # Phrase it the natural way round: wide stances read "wider than X%", narrow
    # ones "narrower than (100-X)%" — so a 1st-pct stance says "narrower than 99%".
    if pct is None:
        context = ""
    elif pct >= 50:
        context = f"wider than {pct}% of hitters his size"
    else:
        context = f"narrower than {100 - pct}% of hitters his size"

    ax.text(0.035, 0.965, "\n".join(stat_rows), transform=ax.transAxes,
            fontsize=10.5, va="top", ha="left", color=pal["ink"], fontweight="bold",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=pal["caption"],
                      edgecolor=pal["ink"], linewidth=1.2))
    if context:
        ax.text(0.045, 0.965 - 0.075 * (len(stat_rows) + 1), context,
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                color=pal["ink"], style="italic")


def stance_contact_map(
    points: pd.DataFrame,
    stance: dict,
    player_name: str | None = None,
    biomech: dict | None = None,
    theme: str = "midnight",
    show_box: bool = True,
    show_bat: bool = True,
    note: str | None = None,
) -> str:
    """Small-multiple batter's-box view: one clean panel per pitch type.

    Each panel is the same top-down box (home plate + this hitter's SETUP feet)
    with ONE pitch type's contact zone. Splitting the pitch types into their own
    panels — instead of overlaying them — is what keeps it readable: his contact
    for every pitch lands in roughly the same place, so a single combined panel
    just turns into overlapping mud. A faint gray marker for his OVERALL average
    contact repeats in every panel as a fixed reference, so you can see how each
    pitch's zone shifts relative to his norm.

    The first "All pitches" panel carries the biomechanical context: Savant-style
    red measurement lines to the stance center, the feet labeled as the setup
    stance, and — if `height` is given — stance width and setup-to-contact reach
    expressed as a share of the hitter's height.

    Args:
        points: per-swing frame (swing_geometry.intercept_points) — needs
                `intercept_side_in`, `intercept_depth_in`, and `pitch_type`.
        stance: the hitter's stance row (swing_geometry.player_stance) — needs
                avg_batter_x_position, avg_batter_y_position, avg_foot_sep,
                avg_stance_angle.
        player_name: optional, used in the figure title.
        biomech: optional dict from swing_geometry.stance_biomech — height text
                plus height-adjusted stance-width / reach percentiles for the
                caption and title.
        theme: aesthetic preset — "report" (white), "paper" (cream), or
                "midnight" (dark). See _STANCE_THEMES.

    Returns a base64 PNG data URI, like the other chart functions.
    """
    pal = _STANCE_THEMES.get(theme, _STANCE_THEMES["report"])
    _require_rows(points, "stance_contact_map")
    if not {"intercept_side_in", "intercept_depth_in"}.issubset(points.columns):
        raise ValueError("stance_contact_map: missing intercept coordinate columns.")

    bx = stance.get("avg_batter_x_position")
    by = stance.get("avg_batter_y_position")
    if bx is None or by is None or pd.isna(bx) or pd.isna(by):
        raise ValueError("stance_contact_map: stance is missing box position (x/y).")
    foot_sep = stance.get("avg_foot_sep")
    angle = stance.get("avg_stance_angle") or 0.0

    d = points.dropna(subset=["intercept_side_in", "intercept_depth_in"]).copy()
    if d.empty:
        raise ValueError("stance_contact_map: no swings with intercept coordinates.")

    # Map each swing into real box coordinates (verified against Savant's numbers).
    d["off_plate"] = bx - d["intercept_side_in"]
    d["depth"] = by - d["intercept_depth_in"]
    overall_op, overall_dp = d["off_plate"].mean(), d["depth"].mean()

    # One panel per pitch FAMILY (fastball / breaking / offspeed / other) so
    # every pitch he sees is covered, plus a leading "All pitches" panel. Family
    # grouping keeps samples large and the grid compact.
    panels: list[tuple[str, pd.DataFrame, str]] = [("All pitches", d, _SLATE)]
    if "pitch_type" in d.columns and d["pitch_type"].notna().any():
        fam = d["pitch_type"].map(_pitch_family)
        fam_counts = fam.value_counts()
        for name in _FAMILY_ORDER:
            if fam_counts.get(name, 0) >= _MIN_PANEL_SWINGS:
                panels.append((name, d[fam == name], _FAMILY_COLORS[name]))

    # Grid shape: 4 panels look best as a tidy 2x2; otherwise up to 3 columns.
    ncols = {1: 1, 2: 2, 3: 3, 4: 2}.get(len(panels), 3)
    nrows = (len(panels) + ncols - 1) // ncols

    with _figure(nrows, ncols, figsize=(3.7 * ncols + 0.4, 4.1 * nrows + 0.6)) as (fig, axes):
        fig.patch.set_facecolor(pal["bg"])
        axes = np.atleast_1d(axes).ravel()

        for i, (label, sub, color) in enumerate(panels):
            ax = axes[i]
            is_hero = i == 0                       # the "All pitches" panel
            _draw_stance_box(ax, bx, by, foot_sep, angle, pal, show_box=show_box)

            # This panel's contact point: overall on the hero, else this pitch.
            cx = overall_op if is_hero else sub["off_plate"].mean()
            cy = overall_dp if is_hero else sub["depth"].mean()

            # Mark the center of the plate and draw a dashed line from it to the
            # contact point, so the "distance from plate center" is shown, not just
            # stated.
            ax.plot([_PLATE_CENTER[0], cx], [_PLATE_CENTER[1], cy], color=pal["ink"],
                    linestyle="--", linewidth=1.1, alpha=0.7, zorder=7)
            ax.scatter([_PLATE_CENTER[0]], [_PLATE_CENTER[1]], s=45, marker="+",
                       color=pal["ink"], linewidths=1.3, zorder=8)

            # Optional bat, drawn first so the contact X sits on top of the barrel.
            if show_bat:
                _draw_bat(ax, bx, by, cx, cy, pal)

            if is_hero:
                # Full context panel: faint dots, red measurement lines, callout.
                ax.scatter(sub["off_plate"], sub["depth"], s=9, color=pal["foot"],
                           alpha=0.30, linewidth=0, zorder=3)
                _draw_measurements(ax, bx, by, pal)
                if biomech:
                    _biomech_caption(ax, biomech, pal)
                elif note:
                    # No stance/biomech data — explain why the feet aren't drawn.
                    # Wrap to a fixed width so it stays inside the panel.
                    wrapped = textwrap.fill(note, width=34)
                    ax.text(0.035, 0.965, wrapped, transform=ax.transAxes, fontsize=8,
                            va="top", ha="left", color=pal["ink"], style="italic",
                            bbox=dict(boxstyle="round,pad=0.45", facecolor=pal["caption"],
                                      edgecolor=pal["ink"], linewidth=1.0))
            # The contact marker (overall on hero, this pitch elsewhere).
            ax.scatter([cx], [cy], s=150 if is_hero else 120, marker="X",
                       color=pal["ink"], edgecolor=pal["bg"], linewidth=1.2, zorder=8)
            if len(sub) >= _MIN_ELLIPSE_SWINGS:
                _pitch_zone_ellipse(ax, sub["off_plate"], sub["depth"], color,
                                    fill_alpha=pal["zone_alpha"])

            ax.set_title(f"{label}  ({len(sub)})", fontsize=11, color=color)
            # For a family panel, spell out which pitch types it includes (most
            # common first) so "Breaking" isn't a black box.
            if not is_hero and "pitch_type" in sub.columns:
                codes = list(sub["pitch_type"].value_counts().index)[:6]
                # Wrap to <=3 codes per line so the pill stays narrow enough to
                # sit under the cluster without running off the panel edge.
                lines = ["  ·  ".join(codes[i:i + 3]) for i in range(0, len(codes), 3)]
                # Straight-line distance from plate center to this family's average
                # contact — how far off-center / out front he meets these pitches.
                dist = float(np.hypot(cx - _PLATE_CENTER[0], cy - _PLATE_CENTER[1]))
                lines.append(f'{dist:.0f}" from plate center')
                # Place it just below THIS cluster (in data coords), pushed below
                # the plate so it never collides with it; on a filled pill so the
                # grid/box lines don't cut through it.
                ly = min(max(cy + 1.2 * sub["depth"].std() + 6, 22), 40)
                ax.text(cx, ly, "\n".join(lines), ha="center", va="top",
                        fontsize=8, color=pal["ink"],
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=pal["bg"],
                                  edgecolor=color, linewidth=1.0, alpha=0.95))
            # Only edge panels get axis labels/ticks, to cut repetition.
            if i % ncols == 0:
                ax.set_ylabel("Depth in box (in)", color=pal["ink"])
            else:
                ax.tick_params(labelleft=False)
            if i // ncols == nrows - 1:
                ax.set_xlabel("Distance off plate (in)", color=pal["ink"])
            else:
                ax.tick_params(labelbottom=False)

        # Blank any unused cells in the last row.
        for j in range(len(panels), len(axes)):
            axes[j].axis("off")

        title = "Stance & contact map — by pitch family"
        if player_name:
            header = player_name
            if biomech and biomech.get("height_text"):
                header = f"{player_name}  ·  {biomech['height_text']}"
            title = f"{header} — {title}"
        fig.suptitle(title, fontsize=13, color=pal["ink"])

        # A real key. Each swatch is drawn as the SHAPE it represents (custom
        # handlers) so nothing reads as a generic rectangle: a foot ellipse, the
        # X marker, a translucent zone ellipse, the red measurement line, and a
        # home-plate pentagon. The feet entry is only shown when feet were drawn
        # (i.e. the hitter had a stance row) — otherwise it would name a mark
        # that isn't there.
        has_feet = foot_sep is not None and not pd.isna(foot_sep)
        handles, hmap = [], {}
        if has_feet:
            foot_h = Patch(facecolor=pal["foot"], edgecolor=pal["ink"], label="Feet = setup stance")
            handles.append(foot_h)
            hmap[foot_h] = _HandlerFoot()
        contact_h = Line2D([], [], marker="X", color=pal["ink"], markeredgecolor=pal["bg"],
                           linestyle="None", markersize=11, label="Average contact")
        zone_h = Patch(facecolor="#9aa3ab", edgecolor="#9aa3ab", alpha=0.35,
                       label="Contact zone (per pitch)")
        measure_h = Line2D([], [], color=pal["measure"], linestyle="--",
                           label="Plate to center of mass")
        plate_h = Patch(facecolor="none", edgecolor=pal["ink"], label="Home plate")
        handles += [contact_h, zone_h, measure_h, plate_h]
        hmap[zone_h] = _HandlerEllipse()
        hmap[plate_h] = _HandlerPlate()
        fig.legend(
            handles=handles, handler_map=hmap,
            loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
            labelcolor=pal["ink"], bbox_to_anchor=(0.5, 0.085),
        )

        # Pitch-code glossary — only the codes this hitter actually saw, so the
        # FF/SI/ST abbreviations in the panels are all spelled out.
        if "pitch_type" in d.columns and d["pitch_type"].notna().any():
            present = list(d["pitch_type"].dropna().value_counts().index)
            entries = [f"{c} = {pitch_display_name(c)}" for c in present]
            per_line = 4
            gloss = "\n".join(
                "     ".join(entries[i:i + per_line]) for i in range(0, len(entries), per_line)
            )
            fig.text(0.5, 0.015, gloss, ha="center", va="bottom", fontsize=7.5,
                     color=pal["ink"])

        # Leave a strip at the bottom for the key + glossary and room up top for
        # the title; h_pad keeps a row's titles clear of the panels above them.
        fig.tight_layout(rect=(0, 0.13, 1, 0.96), h_pad=3.0)
        return _fig_to_data_uri(fig, facecolor=pal["bg"])
