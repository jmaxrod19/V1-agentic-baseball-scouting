"""
VAA charts: PROTOTYPES for visualizing vertical approach angle in 3D.

Everything here is experimental. Expect functions to change shape, split, or
disappear as we try ideas. Once a chart settles, it gets promoted into
src/charts.py (which returns base64 PNGs for the HTML report). Until then these
return a plain matplotlib Figure so a runner script or notebook can save/show it.

Prototype A1: flight_paths_3d. One pitcher's fastballs traced from release to
the plate in 3D, each colored by its VAA. (Too busy: shelved, kept for reference.)

Prototype A2: compare_to_league_3d. ONE line for the pitcher's average fastball
next to ONE line for the league-average fastball.

Prototype A3: side_view_with_profile. A flat side view (where VAA is a plain
slope) plus four tiles for the release traits that drive it: height, release
height, extension, arm angle. Each tile compares the pitcher to the league.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend, same convention as src/charts.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

from src import vaa

# Colors borrowed from src/charts.py: crimson = fastball, cyan = breaking.
_STEEP = "#cc3552"   # steep descent (more negative VAA)
_FLAT = "#27c1dd"    # flat approach (closer to zero)
_INK = "#1f2937"

# One fixed color scale for every pitcher, so two charts can be compared by eye.
# MLB fastballs mostly live between about -6.5 and -3 degrees.
_VAA_MIN, _VAA_MAX = -7.0, -2.5
_CMAP = LinearSegmentedColormap.from_list("vaa", [_STEEP, _FLAT])

# Plate geometry in feet. Home plate is 17 inches wide.
_HALF_PLATE = 17 / 12 / 2


def flight_paths_3d(
    pitches: pd.DataFrame,
    title: str,
    max_paths: int = 60,
    seed: int = 0,
    view: tuple[float, float] = (16.0, -58.0),
    exaggeration: float = 3.5,
) -> Figure:
    """Draw each pitch's path in 3D, colored by its vertical approach angle.

    pitches   Statcast rows for ONE pitcher and (usually) one pitch type.
    title     Chart title, e.g. the pitcher's name.
    max_paths Draw at most this many paths (random sample) so the picture stays
              readable. The average VAA in the title uses ALL the pitches.
    seed      Makes the random sample repeatable: same data, same picture.
    view      (elevation, azimuth) camera angles in degrees for ax.view_init.
    exaggeration  How much taller and wider than true scale the picture is drawn.
              At true scale (1.0) a 54-foot pitch is a razor-thin sliver, because
              the ball only falls a few feet over that distance. Stretching
              height and width makes the paths readable, but it ALSO makes every
              angle look steeper than it is, so the chart says so on its face.
    """
    paths = vaa.flight_paths(pitches)
    angles = vaa.compute_vaa(pitches).to_numpy()

    # Keep only pitches we could fully trace and measure.
    usable = np.isfinite(angles) & np.isfinite(paths).all(axis=(1, 2))
    paths, angles = paths[usable], angles[usable]
    if len(angles) == 0:
        raise ValueError("flight_paths_3d: no pitches with complete trajectory data.")

    average_vaa = float(angles.mean())   # computed BEFORE sampling, on every pitch
    total = len(angles)

    if total > max_paths:
        rng = np.random.default_rng(seed)
        pick = rng.choice(total, size=max_paths, replace=False)
        paths, angles = paths[pick], angles[pick]

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(projection="3d")
    norm = Normalize(vmin=_VAA_MIN, vmax=_VAA_MAX, clip=True)

    # Plot axes: matplotlib's x = distance from plate, y = left/right, z = height.
    for path, angle in zip(paths, angles):
        ax.plot(
            path[:, 1], path[:, 0], path[:, 2],
            color=_CMAP(norm(angle)), linewidth=1.3, alpha=0.75,
        )

    _draw_strike_zone(ax, pitches)

    ax.set_xlim(56, 0)           # pitcher on the left, plate on the right
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 8)
    ax.set_box_aspect((56, 8 * exaggeration, 8 * exaggeration), zoom=1.25)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("Distance from plate (ft)", labelpad=14)
    ax.set_ylabel("Left / right (ft)", labelpad=10)
    ax.set_zlabel("Height (ft)", labelpad=6)
    ax.set_yticks([-3, 0, 3])
    ax.set_zticks([0, 2, 4, 6, 8])
    ax.tick_params(pad=3, labelsize=8)

    fig.suptitle(
        f"{title}: {total} pitches, average VAA {average_vaa:.1f}°",
        fontsize=14, color=_INK, y=0.95,
    )
    ax.set_title(
        f"Height and width drawn {exaggeration:g}× true scale, so angles look steeper than they are",
        fontsize=9, color="#6b7280", pad=-4,
    )
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=_CMAP), ax=ax,
        shrink=0.5, pad=0.08, location="right",
    )
    colorbar.set_label("VAA (degrees)   ← steeper     flatter →")
    return fig


def _draw_strike_zone(ax, pitches: pd.DataFrame) -> None:
    """Outline the strike zone where the pitches cross the plate."""
    bottom = _column_mean(pitches, "sz_bot", default=1.5)
    top = _column_mean(pitches, "sz_top", default=3.5)
    y = vaa.PLATE_FRONT_Y
    corners = [(-_HALF_PLATE, bottom), (_HALF_PLATE, bottom),
               (_HALF_PLATE, top), (-_HALF_PLATE, top), (-_HALF_PLATE, bottom)]
    ax.plot([y] * 5, [c[0] for c in corners], [c[1] for c in corners],
            color=_INK, linewidth=1.6)


def _column_mean(df: pd.DataFrame, col: str, default: float) -> float:
    """Average of a column, or `default` if the column is absent or all-NaN."""
    if col not in df.columns:
        return default
    value = pd.to_numeric(df[col], errors="coerce").mean()
    return default if pd.isna(value) else float(value)


_PITCHER_COLOR = _STEEP    # crimson
_LEAGUE_COLOR = "#4b5563"  # slate gray: the baseline should recede


def compare_to_league_3d(
    pitcher: pd.DataFrame,
    league: pd.DataFrame,
    pitcher_label: str,
    league_label: str,
    view: tuple[float, float] = (16.0, -58.0),
    exaggeration: float = 3.5,
) -> Figure:
    """Prototype A2: the pitcher's average pitch vs the league's average pitch.

    pitcher / league   Statcast rows (same pitch type; league should include the
                       same handedness so left/right release side lines up).
    *_label            Names for the legend, e.g. "Skubal" / "MLB average LHP".
    exaggeration       Height/width stretch, as in flight_paths_3d.
    """
    lines = [
        _summarize(pitcher, pitcher_label, _PITCHER_COLOR),
        _summarize(league, league_label, _LEAGUE_COLOR),
    ]

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(projection="3d")

    for line in lines:
        path = line["path"]
        # Plot axes: matplotlib x = distance from plate, y = left/right, z = height.
        ax.plot(
            path[:, 1], path[:, 0], path[:, 2],
            color=line["color"], linewidth=3.2, label=line["legend"],
        )
        ax.scatter(*[path[0, i] for i in (1, 0, 2)], color=line["color"], s=22)   # release
        ax.scatter(*[path[-1, i] for i in (1, 0, 2)], color=line["color"], s=22)  # plate

    _draw_strike_zone(ax, pitcher)

    ax.set_xlim(56, 0)
    ax.set_ylim(-4, 4)
    ax.set_zlim(0, 8)
    ax.set_box_aspect((56, 8 * exaggeration, 8 * exaggeration), zoom=1.25)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("Distance from plate (ft)", labelpad=14)
    ax.set_ylabel("Left / right (ft)", labelpad=10)
    ax.set_zlabel("Height (ft)", labelpad=6)
    ax.set_yticks([-3, 0, 3])
    ax.set_zticks([0, 2, 4, 6, 8])
    ax.tick_params(pad=3, labelsize=8)

    fig.suptitle(f"{pitcher_label} vs {league_label}", fontsize=14, color=_INK, y=0.95)
    ax.set_title(
        f"Height and width drawn {exaggeration:g}x true scale, so angles look steeper than they are",
        fontsize=9, color="#6b7280", pad=-4,
    )
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    return fig


def _summarize(df: pd.DataFrame, label: str, color: str) -> dict:
    """Everything one line of the chart needs: its path, color and legend text."""
    angles = vaa.compute_vaa(df).dropna()
    if angles.empty:
        raise ValueError(f"compare_to_league_3d: no pitches with trajectory data for {label!r}.")
    typical = vaa.average_pitch(df)
    path = vaa.flight_paths(typical, n_points=80)[0]
    plate_height = float(path[-1, 2])
    legend = f"{label}: VAA {angles.mean():.1f}°  (n={len(angles)}, crosses plate at {plate_height:.1f} ft)"
    return {"path": path, "color": color, "legend": legend}


# ===========================================================================
# Prototype A3: side view + profile tiles
# ===========================================================================

RUBBER_Y = 60.5            # feet from the plate to the front of the pitching rubber
_GROUND = "#d6d3d1"
_TILE_FILL = "#f3f4f6"
_MUTED = "#6b7280"
_MIN_REL_GAP = 0.3         # ft: smaller release/extension differences count as "same"
_MIN_VAA_GAP = 0.3         # degrees: smaller angle differences count as "same"


def _mean_or_nan(df: pd.DataFrame, col: str) -> float:
    """Column average, or NaN if the column is absent (older pulls lack some)."""
    if col not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def _profile(df: pd.DataFrame, label: str) -> dict:
    """The handful of averages one side of the chart needs."""
    angles = vaa.compute_vaa(df).dropna()
    if angles.empty:
        raise ValueError(f"side_view_with_profile: no pitches with trajectory data for {label!r}.")
    return {
        "n": len(angles),
        "vaa": float(angles.mean()),
        "release_z": _mean_or_nan(df, "release_pos_z"),
        "extension": _mean_or_nan(df, "release_extension"),
        "arm_angle": _mean_or_nan(df, "arm_angle"),
    }


def _format_height(inches: float | None) -> str:
    """74 -> 6'2". Anything missing -> "n/a"."""
    if inches is None or pd.isna(inches):
        return "n/a"
    total = int(round(float(inches)))
    return f"{total // 12}'{total % 12}\""


def _fmt(value: float, decimals: int, unit: str) -> str:
    """Number with a unit, or "n/a" when missing."""
    if pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}{unit}"


def _delta(value: float, baseline: float, decimals: int, unit: str) -> str:
    """"▲ 0.5 ft" / "▼ 1.1 ft" / "about the same" / "" (when either is missing)."""
    if pd.isna(value) or pd.isna(baseline):
        return ""
    diff = value - baseline
    if round(abs(diff), decimals) == 0:
        return "about the same"
    arrow = "▲" if diff > 0 else "▼"
    return f"{arrow} {abs(diff):.{decimals}f}{unit}"


def _tiles(p: dict, lg: dict, height_in: float | None, league_height_in: float | None) -> list[dict]:
    """The four stat tiles as plain dicts (kept separate from drawing so it's testable)."""
    height_known = height_in is not None and not pd.isna(height_in)

    arm_missing = pd.isna(p["arm_angle"])
    return [
        {
            "label": "HEIGHT",
            "value": _format_height(height_in),
            "league": f"MLB avg {_format_height(league_height_in)}",
            "delta": _delta(height_in if height_known else float("nan"),
                            league_height_in if league_height_in is not None else float("nan"), 0, " in"),
            "note": "",
        },
        {
            "label": "RELEASE HEIGHT",
            "value": _fmt(p["release_z"], 1, " ft"),
            "league": f"MLB avg {_fmt(lg['release_z'], 1, ' ft')}",
            "delta": _delta(p["release_z"], lg["release_z"], 1, " ft"),
            "note": "measured above field level",
        },
        {
            "label": "EXTENSION",
            "value": _fmt(p["extension"], 1, " ft"),
            "league": f"MLB avg {_fmt(lg['extension'], 1, ' ft')}",
            "delta": _delta(p["extension"], lg["extension"], 1, " ft"),
            "note": "",
        },
        {
            "label": "ARM ANGLE",
            "value": _fmt(p["arm_angle"], 0, "°"),
            "league": f"MLB avg {_fmt(lg['arm_angle'], 0, '°')}",
            "delta": _delta(p["arm_angle"], lg["arm_angle"], 0, "°"),
            "note": "not in this data" if arm_missing else "0° sidearm, 90° over the top",
        },
    ]


def _takeaway(p: dict, lg: dict) -> str:
    """One plain-English sentence comparing release and angle to the league.

    Describes; never claims cause. Differences smaller than the _MIN_* gaps are
    treated as "about average" so the sentence doesn't over-read noise.
    """
    parts = []
    dz = p["release_z"] - lg["release_z"]
    if abs(dz) >= _MIN_REL_GAP:                      # NaN compares False: skipped
        parts.append(f"{abs(dz):.1f} ft {'higher' if dz > 0 else 'lower'}")
    de = p["extension"] - lg["extension"]
    if abs(de) >= _MIN_REL_GAP:
        parts.append(f"{abs(de):.1f} ft {'closer to the plate' if de > 0 else 'farther from the plate'}")

    dv = p["vaa"] - lg["vaa"]                        # positive = flatter (closer to 0)
    if abs(dv) >= _MIN_VAA_GAP:
        angle_text = f"approaches {abs(dv):.1f}° {'flatter' if dv > 0 else 'steeper'}"
    else:
        angle_text = "approaches at about the league-average angle"

    if parts:
        return f"Releases {' and '.join(parts)} than average; {angle_text}."
    return f"Release point is close to league average; {angle_text}."


def _avg_path(df: pd.DataFrame) -> np.ndarray:
    """The average pitch's path as an (n_points, 3) array of x, y, z."""
    return vaa.flight_paths(vaa.average_pitch(df), n_points=120)[0]


def side_view_with_profile(
    pitcher: pd.DataFrame,
    league: pd.DataFrame,
    pitcher_label: str,
    league_label: str,
    pitcher_height_in: float | None = None,
    league_height_in: float | None = None,
) -> Figure:
    """Prototype A3: flat side view of both average pitches + four profile tiles.

    pitcher_height_in / league_height_in are listed heights in inches. Leave them
    None if unknown: the height tile then shows "n/a" and the chart still draws.
    """
    p = _profile(pitcher, pitcher_label)
    lg = _profile(league, league_label)
    pitcher_path, league_path = _avg_path(pitcher), _avg_path(league)

    fig_w, fig_h = 11.0, 7.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = fig.add_gridspec(
        3, 4, height_ratios=[0.6, 3.4, 1.6], hspace=0.2, wspace=0.10,
        left=0.05, right=0.97, top=0.97, bottom=0.03,
    )

    _draw_header(fig.add_subplot(grid[0, :]), pitcher_label, p, lg, _takeaway(p, lg))

    side = fig.add_subplot(grid[1, :])
    _draw_side_view(side, pitcher, pitcher_path, league_path, pitcher_label, league_label, p, lg)
    _note_vertical_stretch(side, fig_w, fig_h)

    for column, tile in enumerate(_tiles(p, lg, pitcher_height_in, league_height_in)):
        _draw_tile(fig.add_subplot(grid[2, column]), tile)
    return fig


def _draw_header(ax, label: str, p: dict, lg: dict, takeaway: str) -> None:
    """Name and headline angle on the first line, plain-English takeaway under it."""
    ax.axis("off")
    ax.text(0.0, 0.78, label, fontsize=19, fontweight="bold", color=_INK,
            va="center", transform=ax.transAxes)
    ax.text(1.0, 0.78, f"VAA {p['vaa']:.1f}°", fontsize=19, fontweight="bold",
            color=_PITCHER_COLOR, va="center", ha="right", transform=ax.transAxes)
    ax.text(1.0, 0.22, f"MLB average {lg['vaa']:.1f}°", fontsize=11, color=_MUTED,
            va="center", ha="right", transform=ax.transAxes)
    ax.text(0.0, 0.22, takeaway, fontsize=11.5, color="#374151",
            va="center", transform=ax.transAxes)


def _draw_side_view(ax, pitcher, pitcher_path, league_path, pitcher_label, league_label, p, lg) -> None:
    """Flat side view: pitcher on the left, plate on the right, height up."""
    x_far, x_near, y_top = RUBBER_Y + 2.5, -1.5, 8.2
    ax.set_xlim(x_far, x_near)            # reversed: far (pitcher) on the left
    ax.set_ylim(-0.4, y_top)
    ax.axhspan(-0.4, 0, color=_GROUND, zorder=0)

    # Strike zone as it looks from the side: a thin box at the plate.
    bottom = _column_mean(pitcher, "sz_bot", default=1.5)
    top = _column_mean(pitcher, "sz_top", default=3.5)
    ax.add_patch(Rectangle((0, bottom), vaa.PLATE_FRONT_Y, top - bottom,
                           facecolor="#93c5fd", alpha=0.35, edgecolor="none", zorder=1))
    ax.text(vaa.PLATE_FRONT_Y / 2, top + 0.12, "zone", fontsize=8, color=_MUTED, ha="center")

    # Pitching rubber.
    ax.plot([RUBBER_Y, RUBBER_Y], [0, 0.35], color=_INK, linewidth=3, solid_capstyle="butt")
    ax.text(RUBBER_Y, -0.2, "rubber", fontsize=8, color=_MUTED, ha="center", va="center")

    for path, color, label in (
        (league_path, _LEAGUE_COLOR, league_label),
        (pitcher_path, _PITCHER_COLOR, pitcher_label),
    ):
        ax.plot(path[:, 1], path[:, 2], color=color, linewidth=3.2, label=label, zorder=3)
        ax.scatter([path[0, 1], path[-1, 1]], [path[0, 2], path[-1, 2]],
                   color=color, s=34, zorder=4)

    # Release height + extension drawn like a dimension diagram: a dashed drop
    # line from the release point to the ground, and a double arrow along the
    # ground-level from the rubber to that drop line.
    rel_y, rel_z = pitcher_path[0, 1], pitcher_path[0, 2]
    ax.plot([rel_y, rel_y], [0, rel_z], color=_PITCHER_COLOR, linestyle=(0, (3, 3)),
            linewidth=1.2, zorder=2)
    ax.text(rel_y + 0.8, rel_z * 0.62, f"release\n{rel_z:.1f} ft", fontsize=9,
            color=_PITCHER_COLOR, ha="right", va="center")

    bracket_z = 1.05
    ax.annotate("", xy=(RUBBER_Y, bracket_z), xytext=(rel_y, bracket_z),
                arrowprops=dict(arrowstyle="<->", color=_PITCHER_COLOR, linewidth=1.3))
    # Anchored just past the rubber and growing toward home plate, so the words
    # never run into the dashed drop line even for short extensions.
    ax.text(RUBBER_Y + 1.0, bracket_z - 0.15,
            f"extension {_fmt(p['extension'], 1, ' ft')}", fontsize=9,
            color=_PITCHER_COLOR, ha="left", va="top")

    ax.set_xlabel("Feet from home plate", color=_MUTED, fontsize=9)
    ax.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(colors=_MUTED, labelsize=8)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#d1d5db")
    ax.set_ylabel("Height (ft)", color=_MUTED, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def _note_vertical_stretch(ax, fig_w: float, fig_h: float) -> None:
    """Say how much taller than true scale the picture is drawn, from the real layout."""
    box = ax.get_position()                                  # fractions of the figure
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    inches_per_ft_x = box.width * fig_w / x_range
    inches_per_ft_y = box.height * fig_h / y_range
    stretch = inches_per_ft_y / inches_per_ft_x
    ax.text(0.005, 0.99, f"Height drawn {stretch:.1f}× true scale", transform=ax.transAxes,
            fontsize=8, color=_MUTED, ha="left", va="top")


def _draw_tile(ax, tile: dict) -> None:
    """One rounded stat tile: label, big number, league comparison, difference."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=_TILE_FILL, edgecolor="none",
    ))
    ax.text(0.5, 0.86, tile["label"], fontsize=9, fontweight="bold", color=_MUTED,
            ha="center", va="center")
    ax.text(0.5, 0.58, tile["value"], fontsize=24, fontweight="bold", color=_INK,
            ha="center", va="center")
    ax.text(0.5, 0.35, tile["league"], fontsize=9.5, color=_MUTED, ha="center", va="center")
    ax.text(0.5, 0.21, tile["delta"], fontsize=11, fontweight="bold", color=_INK,
            ha="center", va="center")
    if tile["note"]:
        ax.text(0.5, 0.08, tile["note"], fontsize=8, color=_MUTED, ha="center", va="center")
