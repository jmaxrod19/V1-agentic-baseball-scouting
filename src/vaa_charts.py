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

Prototype A5: side_view_with_profile. Built to be understood in seconds: a
verdict in words with the VAA, one clean side view with figurines and straight
release-to-plate lines, then three cards (release height, extension, arm angle).
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

from src import vaa, vaa_figures

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
_LEAGUE_COLOR = "#6b7280"  # gray: the baseline recedes, still distinct for colorblind readers


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
# Prototype A5: verdict + one side view + three cards
# ===========================================================================

RUBBER_Y = 60.5            # feet from the plate to the front of the pitching rubber
_GROUND = "#e7e5e4"
_TILE_FILL = "#f3f4f6"
_MUTED = "#6b7280"
_MIN_REL_GAP = 0.3         # ft: smaller release/extension differences count as "same"
_MIN_VAA_GAP = 0.3         # degrees: smaller angle differences count as "same"
_MIN_ARM_GAP = 5.0         # degrees of arm angle
_DEFAULT_HEIGHT_IN = 74    # figurine size when the listed height is unknown
_FIGURE_FILL = "#dbdee3"   # light gray for people who are only context


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


def _verdict(p: dict, lg: dict) -> str:
    """The answer in a few words. Positive difference = flatter (closer to 0)."""
    diff = p["vaa"] - lg["vaa"]
    if abs(diff) < _MIN_VAA_GAP:
        return "About average"
    return "Flatter than average" if diff > 0 else "Steeper than average"


def _compare(value: float, baseline: float, gap: float, decimals: int, unit: str,
             more: str, less: str) -> str:
    """"1.1 ft lower" / "about average" / "" (when either number is missing).

    Differences smaller than `gap` count as average, so noise is not over-read.
    `more` / `less` are the words for above and below the baseline.
    """
    if pd.isna(value) or pd.isna(baseline):
        return ""
    diff = value - baseline
    if abs(diff) < gap:
        return "about average"
    return f"{abs(diff):.{decimals}f}{unit} {more if diff > 0 else less}"


def _cards(p: dict, lg: dict) -> list[dict]:
    """The three release-trait cards as plain dicts (separate from drawing, so testable)."""
    specs = (
        # key,          title,            gap,           decimals, unit,  more,     less
        ("release_z",  "RELEASE HEIGHT", _MIN_REL_GAP,  1,        " ft", "higher", "lower"),
        ("extension",  "EXTENSION",      _MIN_REL_GAP,  1,        " ft", "longer", "shorter"),
        ("arm_angle",  "ARM ANGLE",      _MIN_ARM_GAP,  0,        "°",   "higher", "lower"),
    )
    cards = []
    for key, title, gap, decimals, unit, more, less in specs:
        cards.append({
            "key": key,
            "title": title,
            "value": _fmt(p[key], decimals, unit),
            "compare": _compare(p[key], lg[key], gap, decimals, unit, more, less),
            "league": f"MLB avg {_fmt(lg[key], decimals, unit)}",
        })
    return cards


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
    """Prototype A5: verdict, one picture, three release cards. Built to be skimmed.

    Reading order: (1) the verdict and VAA at the top, (2) the side view,
    (3) the three cards. Each number appears exactly once. Everywhere, crimson
    means this pitcher and gray means the MLB average.

    pitcher_height_in / league_height_in are listed heights in inches. Leave them
    None if unknown: the height line is then dropped and the chart still draws.
    """
    p = _profile(pitcher, pitcher_label)
    lg = _profile(league, league_label)
    hand = _throwing_hand(pitcher)

    fig = plt.figure(figsize=(12.0, 8.4))
    grid = fig.add_gridspec(
        3, 3, height_ratios=[1.25, 3.0, 1.75], hspace=0.12, wspace=0.07,
        left=0.04, right=0.96, top=0.96, bottom=0.07,
    )

    header = fig.add_subplot(grid[0, :])
    _draw_header(header, pitcher_label, league_label, p, lg,
                 pitcher_height_in, league_height_in)

    side = fig.add_subplot(grid[1, :])
    height_known = pitcher_height_in is not None and not pd.isna(pitcher_height_in)
    figure_height_ft = (pitcher_height_in if height_known else _DEFAULT_HEIGHT_IN) / 12
    _draw_side_view(side, pitcher, _avg_path(pitcher), _avg_path(league),
                    pitcher_label, figure_height_ft)

    for column, card in enumerate(_cards(p, lg)):
        _draw_card(fig.add_subplot(grid[2, column]), card, p, lg, hand)

    # All the fine print lives in one place, out of the way.
    fig.text(
        0.5, 0.022,
        f"Side view: straight release-to-plate lines, height stretched "
        f"{_vertical_stretch(side):.1f}×.  Top wedge: true arrival angle, exaggerated.  "
        f"Figures are illustrative.  Arm angle: 0° sidearm, 90° over the top.",
        fontsize=8, color=_MUTED, ha="center",
    )
    return fig


def _throwing_hand(df: pd.DataFrame) -> str:
    """"R" or "L" (most common value); "R" if the column is absent or empty."""
    if "p_throws" not in df.columns or df["p_throws"].dropna().empty:
        return "R"
    return str(df["p_throws"].mode().iloc[0])


def _draw_header(ax, pitcher_label: str, league_label: str, p: dict, lg: dict,
                 height_in: float | None, league_height_in: float | None) -> None:
    """Left: who, and the verdict. Right: the VAA number with its wedge."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.86, pitcher_label, fontsize=17, color=_INK, va="center")
    ax.text(0.0, 0.48, _verdict(p, lg), fontsize=30, fontweight="bold",
            color=_INK, va="center")
    context = f"vs. {league_label}"
    if height_in is not None and not pd.isna(height_in):
        average = ""
        if league_height_in is not None and not pd.isna(league_height_in):
            average = f" (avg {_format_height(league_height_in)})"
        context += f"   ·   {_format_height(height_in)} tall{average}"
    ax.text(0.0, 0.10, context, fontsize=10.5, color=_MUTED, va="center")

    ax.text(1.0, 0.64, f"{p['vaa']:.1f}°", fontsize=40, fontweight="bold",
            color=_PITCHER_COLOR, va="center", ha="right")
    ax.text(1.0, 0.22, f"VAA   ·   MLB avg {lg['vaa']:.1f}°", fontsize=10.5,
            color=_MUTED, va="center", ha="right")

    # A small picture of the two arrival angles, just left of the number.
    _draw_wedge(ax.inset_axes([0.63, 0.12, 0.20, 0.76]), p, lg)


def _draw_wedge(ax, p: dict, lg: dict) -> None:
    """The VAA as two straight lines arriving at the same spot.

    Both end at (0, 0) and rise going back toward the mound. The horizontal axis
    is one arbitrary unit, so a line's height at the left end is tan(angle).
    """
    rise = [float(np.tan(np.radians(-side["vaa"]))) for side in (lg, p)]
    ax.set_xlim(-1.03, 0.06)
    ax.set_ylim(-max(rise) * 0.12, max(rise) * 1.08)
    ax.axis("off")
    ax.plot([-1.0, 0.0], [0, 0], color="#9ca3af", linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(-1.0, -max(rise) * 0.03, "level", fontsize=8, color=_MUTED, va="top")
    for height, color in zip(rise, (_LEAGUE_COLOR, _PITCHER_COLOR)):   # pitcher on top
        ax.plot([-1.0, 0.0], [height, 0.0], color=color, linewidth=3.0,
                solid_capstyle="round", gid="angle-line")
    ax.scatter([0], [0], color=_INK, s=26, zorder=5)


def _draw_side_view(ax, pitcher, pitcher_path, league_path, pitcher_label,
                    figure_height_ft) -> None:
    """One clean picture: pitcher, two straight lines, plate. No axes or arrows."""
    x_far, x_near, y_bottom, y_top = RUBBER_Y + 3.0, -2.5, -0.45, 7.6
    ax.set_xlim(x_far, x_near)            # reversed: far (pitcher) on the left
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")

    ground_y = np.linspace(x_near, x_far, 400)
    ax.fill_between(ground_y, y_bottom, vaa_figures.mound_height(ground_y),
                    color=_GROUND, linewidth=0, zorder=0.5)

    bottom = _column_mean(pitcher, "sz_bot", default=1.5)
    top = _column_mean(pitcher, "sz_top", default=3.5)
    ax.add_patch(Rectangle((0, bottom), vaa.PLATE_FRONT_Y, top - bottom,
                           facecolor="#bfdbfe", alpha=0.6, edgecolor="none", zorder=1))

    release = (float(pitcher_path[0, 1]), float(pitcher_path[0, 2]))
    stretch = _vertical_stretch(ax)       # limits are set above, so this is final
    vaa_figures.draw_pitcher_side(ax, release, RUBBER_Y, figure_height_ft,
                                  _PITCHER_COLOR, stretch)
    vaa_figures.draw_batter_side(ax, _FIGURE_FILL, stretch)

    # STRAIGHT lines: only the release and plate end points are used.
    ends = {}
    for key, path, color in (("lg", league_path, _LEAGUE_COLOR),
                             ("p", pitcher_path, _PITCHER_COLOR)):
        ys, zs = [path[0, 1], path[-1, 1]], [path[0, 2], path[-1, 2]]
        ends[key] = (ys, zs)
        ax.plot(ys, zs, color=color, linewidth=3.0, zorder=3, gid="pitch-line")
        first_dot = 1 if key == "p" else 0            # the pitcher's ball is drawn already
        ax.scatter(ys[first_dot:], zs[first_dot:], color=color, s=40, zorder=4,
                   edgecolors="white", linewidths=1.2)

    # Name each line right on it (no legend to look back and forth to). The higher
    # line gets its name above, the lower one below.
    label_y = 36.0                         # feet from the plate
    heights = {}
    for key, (ys, zs) in ends.items():
        # np.interp needs increasing x, so list the plate end first.
        heights[key] = float(np.interp(label_y, [ys[1], ys[0]], [zs[1], zs[0]]))
    higher = "p" if heights["p"] >= heights["lg"] else "lg"
    for key, text, color in (("p", pitcher_label, _PITCHER_COLOR), ("lg", "MLB average", _LEAGUE_COLOR)):
        above = key == higher
        ax.text(label_y, heights[key] + (0.28 if above else -0.28), text, fontsize=12,
                fontweight="bold", color=color, ha="center",
                va="bottom" if above else "top", zorder=5)


def _vertical_stretch(ax) -> float:
    """How many times taller than true scale this axes draws things."""
    box = ax.get_position()                                  # fractions of the figure
    fig_w, fig_h = ax.figure.get_size_inches()
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    inches_per_ft_x = box.width * fig_w / x_range
    inches_per_ft_y = box.height * fig_h / y_range
    return float(inches_per_ft_y / inches_per_ft_x)


def _draw_card(ax, card: dict, p: dict, lg: dict, hand: str) -> None:
    """One trait: number and plain-word comparison on the left, mini picture on the right."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=_TILE_FILL, edgecolor="none",
    ))
    ax.text(0.07, 0.83, card["title"], fontsize=10, fontweight="bold", color=_MUTED, va="center")
    ax.text(0.07, 0.56, card["value"], fontsize=28, fontweight="bold", color=_INK, va="center")
    ax.text(0.07, 0.31, card["compare"], fontsize=12.5, fontweight="bold", color=_INK, va="center")
    ax.text(0.07, 0.15, card["league"], fontsize=10, color=_MUTED, va="center")

    mine, theirs = p[card["key"]], lg[card["key"]]
    if pd.isna(mine) or pd.isna(theirs):
        return                                   # nothing honest to draw
    picture = ax.inset_axes([0.58, 0.10, 0.37, 0.80])
    picture.axis("off")

    if card["key"] == "release_z":
        # Two upright bars from the ground: him, then average.
        picture.bar([0, 1], [mine, theirs], width=0.5, color=[_PITCHER_COLOR, _LEAGUE_COLOR])
        picture.axhline(0, color=_INK, linewidth=1.2)
        picture.set_xlim(-0.8, 1.8)
        picture.set_ylim(0, max(mine, theirs) * 1.08)
    elif card["key"] == "extension":
        # Two sideways bars reaching forward from the rubber: him on top.
        picture.barh([1, 0], [mine, theirs], height=0.42, color=[_PITCHER_COLOR, _LEAGUE_COLOR])
        picture.axvline(0, color=_INK, linewidth=1.2)
        picture.set_xlim(0, max(mine, theirs) * 1.03)
        picture.set_ylim(-0.9, 1.9)
    else:
        direction = -1 if hand == "R" else 1     # which side the throwing arm is on
        picture.set_xlim(*sorted((direction * 3.2, -direction * 1.4)))
        picture.set_ylim(2.72, 6.9)
        picture.set_aspect("equal")              # equal scales: the arm angle is TRUE
        start = direction * vaa_figures.FRONT_SHOULDER_HALF
        level = vaa_figures.FRONT_SHOULDER_Z
        picture.plot([start, start + direction * 2.5], [level, level], color="#9ca3af",
                     linewidth=1.0, linestyle=(0, (3, 3)), zorder=0.5)
        vaa_figures.draw_pitcher_front(picture, hand,
                                       [(theirs, _LEAGUE_COLOR), (mine, _PITCHER_COLOR)],
                                       _FIGURE_FILL)
