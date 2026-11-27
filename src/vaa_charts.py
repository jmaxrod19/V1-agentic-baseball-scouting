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

Prototype A6: fastball_vaa_chart. Modeled flight paths from Statcast velocity and
acceleration, tangent lines at the plate, a true-scale angle diagram, five trait
cards, and source/method notes. The numbers come from src/vaa_model.py.
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
from src.vaa_model import SMALL_SAMPLE, Comparison

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
# Prototype A6: modeled flight paths, true-scale angle diagram, five cards
# All numbers come from src/vaa_model.py; this section only draws them.
# ===========================================================================

RUBBER_Y = 60.5            # feet from the plate to the front of the pitching rubber
_GROUND = "#e7e5e4"
_TILE_FILL = "#f3f4f6"
_MUTED = "#6b7280"
_DEFAULT_HEIGHT_IN = 74    # figurine size when the listed height is unknown
_FIGURE_FILL = "#dbdee3"   # light gray for people who are only context
_TANGENT_RUN = 24.0        # how far back from the plate the tangent lines reach (ft)


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


def _signed(value: float) -> str:
    """+0.9° / -0.3° / 0.0° (never "-0.0")."""
    rounded = round(value, 1) + 0.0                  # adding 0.0 turns -0.0 into 0.0
    return f"{rounded:+.1f}°" if rounded else "0.0°"


def _effect_text(share: float) -> str:
    """What one factor does to the modeled angle, in words."""
    if abs(share) < 0.05:
        return "no real change"
    return f"{_signed(share)} {'flatter' if share > 0 else 'steeper'}"


def _factor_phrase(name: str, c: Comparison) -> str:
    """"a lower release", "less carry", ... decided from the data, per factor."""
    p, lg = c.pitcher.state, c.league.state
    if name == "release_height":
        return "a lower release" if p["release_z"] < lg["release_z"] else "a higher release"
    if name == "extension":                      # smaller release_y = reaches farther
        return "longer extension" if p["release_y"] < lg["release_y"] else "shorter extension"
    if name == "velocity":                       # speed_y is negative toward the plate
        return "more velocity" if p["speed_y"] < lg["speed_y"] else "less velocity"
    return "more carry" if p["az"] > lg["az"] else "less carry"   # az less negative = carry


def _reason(c: Comparison) -> str:
    """One sentence naming the biggest factor(s) behind the modeled difference."""
    ranked = sorted(c.shares.items(), key=lambda item: abs(item[1]), reverse=True)
    big = [(name, share) for name, share in ranked[:2] if abs(share) >= 0.15]
    if not big:
        return "No single trait moves the angle much."
    first = f"{_factor_phrase(big[0][0], c)} ({_signed(big[0][1])})"
    if len(big) == 1:
        return f"Mainly {first}."
    second = f"{_factor_phrase(big[1][0], c)} ({_signed(big[1][1])})"
    if big[0][1] * big[1][1] > 0:                # both push the same way
        return f"Mainly {first}, plus {second}."
    if c.classification == "About average":
        return f"{first[0].upper()}{first[1:]} and {second} largely offset."
    return f"Mainly {first}, partly offset by {second}."


def _cards(c: Comparison) -> list[dict]:
    """The five trait cards as plain dicts (separate from drawing, so testable)."""
    p, lg = c.pitcher, c.league
    specs = (
        # title, attribute, decimals, unit, factor that gets its effect
        ("RELEASE HEIGHT", "release_height", 1, " ft", "release_height"),
        ("EXTENSION", "extension", 1, " ft", "extension"),
        ("ARM ANGLE", "arm_angle", 0, "°", None),
        ("VELOCITY", "velocity", 1, " mph", "velocity"),
        ("VERTICAL BREAK", "ivb", 1, " in", "ride"),
    )
    cards = []
    for title, attribute, decimals, unit, factor in specs:
        mine, theirs = getattr(p, attribute), getattr(lg, attribute)
        cards.append({
            "key": attribute,
            "title": title,
            "mine": mine,
            "theirs": theirs,
            "value": _fmt(mine, decimals, unit),
            "league": f"MLB avg {_fmt(theirs, decimals, unit)}",
            # Arm angle is not one of the model's inputs, so it gets no number.
            "effect": _effect_text(c.shares[factor]) if factor else "not a direct input",
        })
    return cards


def fastball_vaa_chart(
    c: Comparison,
    pitcher_height_in: float | None = None,
    league_height_in: float | None = None,
) -> Figure:
    """Draw one pitcher's four-seamer comparison (build `c` with compare_fastball).

    Top: verdict, observed VAA, VAA above expected, and the two modeled arrival
    angles at TRUE 1:1 scale. Middle: both modeled flight paths with tangent
    lines at the plate. Bottom: five trait cards and the source and method notes.
    """
    fig = plt.figure(figsize=(13.0, 10.6))
    grid = fig.add_gridspec(
        3, 5, height_ratios=[2.15, 3.35, 2.5], hspace=0.10, wspace=0.07,
        left=0.04, right=0.97, top=0.975, bottom=0.095,
    )

    _draw_header(fig.add_subplot(grid[0, :]), c, pitcher_height_in, league_height_in)

    side = fig.add_subplot(grid[1, :])
    known = pitcher_height_in is not None and not pd.isna(pitcher_height_in)
    _draw_side_view(side, c, (pitcher_height_in if known else _DEFAULT_HEIGHT_IN) / 12)

    for column, card in enumerate(_cards(c)):
        _draw_card(fig.add_subplot(grid[2, column]), card, c.hand)

    _draw_footnotes(fig, c, _vertical_stretch(side))
    return fig


def _draw_header(ax, c: Comparison, height_in: float | None,
                 league_height_in: float | None) -> None:
    """Left: who, verdict, why, two headline numbers. Right: the angle diagram."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.95, c.name, fontsize=18, color=_INK, va="center")
    context = f"{c.season or ''} four-seam fastball vs. {_hand_word(c.hand)}-handed MLB".strip()
    if c.first_date:
        context += f", {c.first_date}–{c.last_date}"
    if height_in is not None and not pd.isna(height_in):
        average = ""
        if league_height_in is not None and not pd.isna(league_height_in):
            average = f" (avg {_format_height(league_height_in)})"
        context += f"   ·   {_format_height(height_in)} tall{average}"
    ax.text(0.0, 0.84, context, fontsize=10.5, color=_MUTED, va="center")
    if c.pitcher.pitches < SMALL_SAMPLE:
        ax.text(0.30, 0.95, f"SMALL SAMPLE: {c.pitcher.pitches} pitches", fontsize=10,
                fontweight="bold", color=_PITCHER_COLOR, va="center")
    ax.text(0.0, 0.66, c.classification, fontsize=30, fontweight="bold", color=_INK, va="center")
    ax.text(0.0, 0.50, _reason(c), fontsize=11.5, color="#374151", va="center")

    # The two headline numbers, each labelled with what kind of number it is.
    blocks = (
        (0.0, "OBSERVED VAA", f"{c.pitcher.observed_vaa:.1f}°", _PITCHER_COLOR,
         f"MLB {c.league.observed_vaa:.1f}°  ·  thrown {c.pitcher.plate_height:.1f} ft high"),
        (0.23, "HEIGHT-ADJUSTED (VAA AA)", _signed(c.pitcher.vaa_aa), _INK,
         "+ flatter than expected, − steeper"),
    )
    for x, label, value, color, note in blocks:
        ax.text(x, 0.31, label, fontsize=9.5, fontweight="bold", color=_MUTED, va="center")
        ax.text(x, 0.16, value, fontsize=29, fontweight="bold", color=color, va="center")
        ax.text(x, 0.01, note, fontsize=9.5, color=_MUTED, va="center")

    _draw_angle_diagram(ax.inset_axes([0.50, 0.05, 0.50, 0.80]), c)


def _hand_word(hand: str) -> str:
    return "right" if hand == "R" else "left"


def _draw_angle_diagram(ax, c: Comparison) -> None:
    """The two modeled arrival angles at true scale: one unit across = one unit up.

    Both lines end at (0, 0), the plate, and rise going back toward the mound at
    exactly their VAA. Nothing is stretched, which is why they look shallow:
    real fastballs arrive only a few degrees below level.
    """
    sides = (("lg", c.league, _LEAGUE_COLOR, "MLB avg"),
             ("p", c.pitcher, _PITCHER_COLOR, c.name))
    rise = {key: float(np.tan(np.radians(-s.modeled.vaa))) for key, s, _, _ in sides}
    top = max(0.125, max(rise.values()) * 1.3)

    ax.set_xlim(-1.55, 0.05)
    ax.set_ylim(-0.045, top)
    ax.set_aspect("equal", adjustable="box", anchor="C")     # the 1:1 guarantee
    ax.axis("off")

    ax.plot([-1.0, 0.04], [0, 0], color="#9ca3af", linewidth=1.0, linestyle=(0, (4, 3)),
            gid="diagram-level")
    ax.text(-1.02, 0, "level", fontsize=9, color=_MUTED, ha="right", va="center")

    # Keep the two labels apart even when the angles are nearly identical.
    label_z = dict(rise)
    min_gap = 0.045
    if abs(rise["p"] - rise["lg"]) < min_gap:
        middle = (rise["p"] + rise["lg"]) / 2
        direction = 1 if rise["p"] >= rise["lg"] else -1
        label_z["p"] = middle + direction * min_gap / 2
        label_z["lg"] = middle - direction * min_gap / 2

    for key, s, color, name in sides:
        ax.plot([-1.0, 0.0], [rise[key], 0.0], color=color, linewidth=2.2,
                solid_capstyle="round", gid="angle-line")
        ax.plot([-1.025, -1.0], [label_z[key], rise[key]], color=color, linewidth=0.8)
        ax.text(-1.035, label_z[key], f"{name}  {s.modeled.vaa:.1f}°", fontsize=10.5,
                fontweight="bold", color=color, ha="right", va="center")
    ax.scatter([0], [0], color=_INK, s=18, zorder=5)
    ax.text(-0.5, -0.04,
            f"Modeled arrival angle at {c.comparison_height:g} ft  ·  true 1:1 scale",
            fontsize=9.5, color=_MUTED, ha="center", va="center")


def _draw_side_view(ax, c: Comparison, figure_height_ft: float) -> None:
    """Both modeled flight paths on one set of axes, with their arrival tangents."""
    height = c.comparison_height
    x_far, x_near, y_bottom, y_top = RUBBER_Y + 3.0, -2.5, -0.45, max(7.6, height + 4.0)
    ax.set_xlim(x_far, x_near)            # reversed: mound on the left
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")

    ground_y = np.linspace(x_near, x_far, 400)
    ax.fill_between(ground_y, y_bottom, vaa_figures.mound_height(ground_y),
                    color=_GROUND, linewidth=0, zorder=0.5)
    if c.zone is not None:
        bottom, top = c.zone
        ax.add_patch(Rectangle((0, bottom), vaa.PLATE_FRONT_Y, top - bottom,
                               facecolor="#bfdbfe", alpha=0.6, edgecolor="none", zorder=1))

    release = (c.pitcher.state["release_y"], c.pitcher.state["release_z"])
    stretch = _vertical_stretch(ax)       # limits are set above, so this is final
    vaa_figures.draw_pitcher_side(ax, release, RUBBER_Y, figure_height_ft,
                                  _PITCHER_COLOR, stretch)
    vaa_figures.draw_batter_side(ax, _FIGURE_FILL, stretch)

    front, back = vaa.PLATE_FRONT_Y, vaa.PLATE_FRONT_Y + _TANGENT_RUN
    # A level line at the plate-crossing height, to read the tangents against.
    ax.plot([back, front], [height, height], color="#9ca3af", linewidth=1.1,
            linestyle=(0, (1, 2)), zorder=2.9, gid="level-line")
    ax.text(back, height - 0.12, f"level, {height:g} ft", fontsize=9, color=_MUTED,
            ha="left", va="top", zorder=5)

    for key, s, color in (("lg", c.league, _LEAGUE_COLOR), ("p", c.pitcher, _PITCHER_COLOR)):
        path = s.modeled
        ax.plot(path.y, path.z, color=color, linewidth=3.0, zorder=3, gid="pitch-path")
        # Tangent: carry the arrival direction straight back from the plate.
        slope = path.vz_final / path.vy_final          # feet up per foot toward the mound
        ax.plot([front, back], [height, height + slope * _TANGENT_RUN], color=color,
                linewidth=1.5, linestyle=(0, (5, 3)), zorder=3.1, gid="tangent-line")
        if key == "lg":                                # his ball is drawn by the figure
            ax.scatter([path.y[0]], [path.z[0]], color=color, s=40, zorder=4,
                       edgecolors="white", linewidths=1.2)
    ax.scatter([front], [height], color=_INK, s=34, zorder=4.2,
               edgecolors="white", linewidths=1.0)
    ax.text(back, height + 0.95 + max(0.0, c.pitcher.modeled.vz_final / c.pitcher.modeled.vy_final,
                                      c.league.modeled.vz_final / c.league.modeled.vy_final)
            * _TANGENT_RUN, "dashed = arrival angle at the plate", fontsize=9,
            color=_MUTED, ha="center", va="center", zorder=5)

    # Name each path right on it; the higher one gets its name above.
    label_y = 40.0
    at_label = {}
    for key, s in (("p", c.pitcher), ("lg", c.league)):
        # np.interp needs increasing x, and the path runs from far to near: reverse it.
        at_label[key] = float(np.interp(label_y, s.modeled.y[::-1], s.modeled.z[::-1]))
    higher = "p" if at_label["p"] >= at_label["lg"] else "lg"
    for key, text, color in (("p", c.name, _PITCHER_COLOR), ("lg", "MLB average", _LEAGUE_COLOR)):
        above = key == higher
        ax.text(label_y, at_label[key] + (0.28 if above else -0.28), text, fontsize=12,
                fontweight="bold", color=color, ha="center",
                va="bottom" if above else "top", zorder=5)

    ax.text(0.5, 0.995,
            f"Modeled average pitches, both aimed to cross the front of the plate at {height:g} ft",
            transform=ax.transAxes, fontsize=9.5, color=_MUTED, ha="center", va="top")


def _vertical_stretch(ax) -> float:
    """How many times taller than true scale this axes draws things."""
    box = ax.get_position()                                  # fractions of the figure
    fig_w, fig_h = ax.figure.get_size_inches()
    x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    inches_per_ft_x = box.width * fig_w / x_range
    inches_per_ft_y = box.height * fig_h / y_range
    return float(inches_per_ft_y / inches_per_ft_x)


def _draw_card(ax, card: dict, hand: str) -> None:
    """One trait: value, league value, small picture, and its effect on the angle."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=_TILE_FILL, edgecolor="none",
    ))
    ax.text(0.5, 0.91, card["title"], fontsize=9.5, fontweight="bold", color=_MUTED,
            ha="center", va="center")
    ax.text(0.5, 0.77, card["value"], fontsize=22, fontweight="bold", color=_INK,
            ha="center", va="center")
    ax.text(0.5, 0.645, card["league"], fontsize=9.5, color=_MUTED, ha="center", va="center")
    ax.plot([0.12, 0.88], [0.235, 0.235], color="#d1d5db", linewidth=1.0)
    ax.text(0.5, 0.165, "EFFECT ON ANGLE", fontsize=7.5, color=_MUTED, ha="center", va="center")
    ax.text(0.5, 0.075, card["effect"], fontsize=11, fontweight="bold", color=_INK,
            ha="center", va="center")

    mine, theirs = card["mine"], card["theirs"]
    if pd.isna(mine) or pd.isna(theirs):
        return                                   # missing data: show "n/a", draw nothing
    picture = ax.inset_axes([0.14, 0.27, 0.72, 0.33])
    picture.axis("off")

    if card["key"] == "arm_angle":
        picture.set_aspect("equal")              # equal scales: the arm angle is TRUE
        direction = -1 if hand == "R" else 1     # which side the throwing arm is on
        picture.set_xlim(*sorted((direction * 3.2, -direction * 1.4)))
        picture.set_ylim(2.72, 6.9)
        start = direction * vaa_figures.FRONT_SHOULDER_HALF
        level = vaa_figures.FRONT_SHOULDER_Z
        picture.plot([start, start + direction * 2.5], [level, level], color="#9ca3af",
                     linewidth=1.0, linestyle=(0, (3, 3)), zorder=0.5)
        vaa_figures.draw_pitcher_front(picture, hand,
                                       [(theirs, _LEAGUE_COLOR), (mine, _PITCHER_COLOR)],
                                       _FIGURE_FILL)
    else:
        # Two bars from a shared zero: pitcher on top, league below.
        picture.barh([1, 0], [mine, theirs], height=0.55, color=[_PITCHER_COLOR, _LEAGUE_COLOR])
        picture.axvline(0, color=_INK, linewidth=1.2)
        picture.set_xlim(min(0.0, mine, theirs), max(mine, theirs, 0.0) * 1.04)
        picture.set_ylim(-0.7, 1.7)


def _draw_footnotes(fig, c: Comparison, stretch: float) -> None:
    """Source, sample, method and any warnings, in one quiet block."""
    p, lg = c.pitcher, c.league
    dates = f", {c.first_date} to {c.last_date}" if c.first_date else ""
    lines = [
        f"Source: Baseball Savant Statcast via pybaseball  ·  {c.season} regular season{dates}  ·  "
        f"{c.name}: {p.pitches:,} four-seamers  ·  {_hand_word(c.hand)}-handed MLB: "
        f"{lg.pitches:,} from {c.league_pitchers} pitchers",
        f"Paths are modeled from average measured release point, speed and acceleration "
        f"(gravity, drag, spin).  Side view height stretched {stretch:.1f}×.  Figures illustrative.",
        f"VAA = atan2(vz, |vy|) at the front of the plate.  VAA AA = observed VAA minus the MLB "
        f"fit at the same plate height (+ is flatter).",
        "Card effects split the modeled difference between the four inputs; extension is "
        "measured with release height and speed held equal.  Arm angle: 0° sidearm, 90° over the top.",
    ]
    for row, line in enumerate(lines):
        fig.text(0.5, 0.070 - row * 0.017, line, fontsize=8.5, color=_MUTED, ha="center")
    if c.notes:
        estimated = any("estimated" in n or "derived" in n for n in c.notes)
        lead = "APPROXIMATION" if estimated else "Note"
        fig.text(0.5, 0.070 - len(lines) * 0.017, f"{lead}: {'; '.join(c.notes)}.",
                 fontsize=8.5, fontweight="bold", color=_PITCHER_COLOR, ha="center")
