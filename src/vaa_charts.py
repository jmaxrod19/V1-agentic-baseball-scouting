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
acceleration, tangent lines at the plate, five trait cards, and source/method
notes. The numbers come from src/vaa_model.py.
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
from src.vaa_model import SMALL_SAMPLE, ZONE_PARTS, Comparison, not_primary_words

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
# Prototype A6: modeled flight paths and five cards
# All numbers come from src/vaa_model.py; this section only draws them.
# ===========================================================================

RUBBER_Y = 60.5            # feet from the plate to the front of the pitching rubber
_GROUND = "#e7e5e4"
_TILE_FILL = "#f3f4f6"
_MUTED = "#6b7280"
_DEFAULT_HEIGHT_IN = 74    # figurine size when the listed height is unknown
_FIGURE_FILL = "#dbdee3"   # light gray for people who are only context
# The smallest type on the chart. The page shows the chart at about half size,
# where 12.5 pt comes out near 13 screen pixels.
_TEXT = 12.5
_ZONE_COLOR = "#2563eb"       # strike-zone outline and label
_GHOST_COLOR = "#bcc1c9"      # the MLB average pitcher
_BEND_FROM = 26.0             # flight paths are exact from here to the plate (ft)
_BEHIND_RUBBER = 2.5          # room left of the rubber for the back foot (ft, as drawn)
# Arm angles (degrees) below which the drawing names the slot, because a side view
# cannot show it. These cut-offs are this page's own, not a Statcast category.
# A few degrees under zero is still a flat, sidearm arm; submarine is clearly under.
_SIDEARM_BELOW, _SUBMARINE_BELOW = 20, -15
# Chart layout, in inches.
_WIDTH_IN = 13.0
_TOP_IN, _GAP_IN, _BOTTOM_IN = 0.10, 0.15, 0.95
_HEADER_IN, _HEADER_WITH_VERDICT_IN = 0.80, 1.90
_SIDE_IN, _CARDS_IN = 4.15, 2.75
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
    # Adding 0.0 turns a "-0" (from something like -0.4) into a plain "0".
    return f"{round(value, decimals) + 0.0:.{decimals}f}{unit}"


# A trait that moves the angle less than this (degrees) is not worth naming.
_BIG_EFFECT = 0.15
_TRAIT_NAMES = {"release_height": "release height", "extension": "extension",
                "velocity": "velocity", "ride": "ride"}


def _length(feet: float) -> str:
    """A distance the way a broadcaster says it: "1 ft", "1.5 ft" or "5 in"."""
    if abs(feet) >= 0.875:                        # 10.5 inches and up reads as feet
        return f"{round(abs(feet) * 2) / 2:g} ft"
    return f"{max(1, round(abs(feet) * 12))} in"


def _trait_sentence(name: str, c: Comparison) -> str | None:
    """"His release height is about 1 ft lower than most pitchers." or None."""
    p, lg = c.pitcher, c.league
    attribute = "ivb" if name == "ride" else name
    gap = getattr(p, attribute) - getattr(lg, attribute)
    if pd.isna(gap) or gap == 0:
        return None                               # no data for this trait
    if name == "release_height":
        word = "higher" if gap > 0 else "lower"
        return f"His release height is about {_length(gap)} {word} than most pitchers."
    if name == "extension":
        word = "longer" if gap > 0 else "shorter"
        return f"His extension is about {_length(gap)} {word} than most pitchers."
    if name == "velocity":
        word = "higher" if gap > 0 else "lower"
        return f"His velocity is about {max(1, round(abs(gap)))} mph {word} than most pitchers."
    # The model uses acceleration, the card shows break. If the two disagree about
    # more or less, say nothing rather than something confusing.
    if (gap > 0) != (p.state["az"] > lg.state["az"]):
        return None
    word = "more" if gap > 0 else "less"
    return f"His fastball has about {max(1, round(abs(gap)))} in {word} ride than most pitchers."


def _why(c: Comparison) -> str:
    """The one main reason for a flatter or steeper verdict; "" when about average."""
    if c.classification == "About average":
        return ""                      # nothing stands out, so offer no explanation

    # Only traits that push the SAME way as the verdict can explain it.
    direction = 1 if c.classification.startswith("Flatter") else -1
    helpers = sorted(((share * direction, name) for name, share in c.shares.items()
                      if share * direction >= _BIG_EFFECT), reverse=True)
    for _, name in helpers:
        sentence = _trait_sentence(name, c)
        if sentence:
            return f"Why: {sentence}"
    return "Why: Several small differences add up, not one big one."


def _ordinal(n: int) -> str:
    """1 -> "1st", 22 -> "22nd", 13 -> "13th"."""
    if 11 <= n % 100 <= 13:                      # the teens all take "th"
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _percentile_words(share_below: float | None) -> str:
    """"90th percentile"; "" when there is no ranking.

    Ranks the number from low to high, not good to bad.
    """
    if share_below is None:
        return ""
    # Round down and stay inside 1-99. (1e-9 absorbs 0.58*100 = 57.999...)
    percentile = min(99, max(1, int(share_below * 100 + 1e-9)))
    return f"{_ordinal(percentile)} percentile"


def _cards(c: Comparison) -> list[dict]:
    """The five trait cards as plain dicts (separate from drawing, so testable)."""
    p, lg = c.pitcher, c.league
    specs = (
        # title, attribute, decimals, unit
        ("RELEASE HEIGHT", "release_height", 1, " ft"),
        ("EXTENSION", "extension", 1, " ft"),
        ("ARM ANGLE", "arm_angle", 0, "°"),
        ("VELOCITY", "velocity", 1, " mph"),
        ("RIDE (IVB)", "ivb", 1, " in"),
    )
    cards = []
    for title, attribute, decimals, unit in specs:
        mine, theirs = getattr(p, attribute), getattr(lg, attribute)
        if attribute == "extension":             # feet and inches, as on the drawing
            shown, average = _format_height(mine * 12), _format_height(theirs * 12)
        else:
            shown, average = _fmt(mine, decimals, unit), _fmt(theirs, decimals, unit)
        rank = None if attribute == "arm_angle" else c.ranks.get(attribute)
        cards.append({
            "key": attribute,
            "title": title,
            "mine": mine,
            "theirs": theirs,
            "rank": rank,
            "percentile": _percentile_words(rank),
            "value": shown,
            "league": f"MLB avg {average}",
        })
    return cards


def fastball_vaa_chart(
    c: Comparison,
    pitcher_height_in: float | None = None,
    league_height_in: float | None = None,
) -> Figure:
    """Draw one pitcher's four-seamer comparison (build `c` with compare_fastball).

    Top: who, and for a fastball that stands out, the verdict and reason.
    Middle: both modeled flight paths with the two approach angles tagged in the sky.
    Bottom: five trait cards and the source notes.
    """
    # Heights in inches. The header is only as tall as what it has to say.
    header = _HEADER_IN if c.classification == "About average" else _HEADER_WITH_VERDICT_IN
    height = _TOP_IN + header + _SIDE_IN + _GAP_IN + _CARDS_IN + _BOTTOM_IN
    fig = plt.figure(figsize=(_WIDTH_IN, height))

    def box(bottom_in: float, tall_in: float) -> list[float]:
        """[left, bottom, width, height] in the figure fractions matplotlib wants."""
        return [0.04, bottom_in / height, 0.93, tall_in / height]

    cards_bottom = _BOTTOM_IN
    side_bottom = cards_bottom + _CARDS_IN + _GAP_IN
    _draw_header(fig.add_axes(box(side_bottom + _SIDE_IN, header)), c,
                 pitcher_height_in, league_height_in)

    side = fig.add_axes(box(side_bottom, _SIDE_IN))
    def feet(inches: float | None) -> float:
        known = inches is not None and not pd.isna(inches)
        return (inches if known else _DEFAULT_HEIGHT_IN) / 12

    _draw_side_view(side, c, feet(pitcher_height_in), feet(league_height_in))

    grid = fig.add_gridspec(1, 5, left=0.04, right=0.97, wspace=0.07,
                            bottom=cards_bottom / height,
                            top=(cards_bottom + _CARDS_IN) / height)
    for column, card in enumerate(_cards(c)):
        _draw_card(fig.add_subplot(grid[0, column]), card, c.hand)

    _draw_footnotes(fig, c, _vertical_stretch(side))
    return fig


def _full_season(c: Comparison) -> bool:
    """True when the data runs from opening month to closing month."""
    return c.first_date.startswith(("Mar", "Apr")) and c.last_date.startswith(("Sep", "Oct"))


def _sample_warning(c: Comparison) -> str:
    """Say how few fastballs there were, and over what: it is not a short date range."""
    where = " in this third" if c.zone_part != "all" else ""
    when = " all season" if _full_season(c) else ""
    return f"SMALL SAMPLE: {c.pitcher.pitches} four-seamers{where}{when}"


def _draw_header(ax, c: Comparison, height_in: float | None,
                 league_height_in: float | None) -> None:
    """Who, the warnings, and for a fastball that stands out the verdict and why."""
    tall = ax.get_position().height * ax.figure.get_figheight()      # inches
    ax.set_xlim(0, 1)
    ax.set_ylim(tall, 0)             # y now means "inches below the top of the header"
    ax.axis("off")

    ax.text(0.5, 0.20, c.name, fontsize=22, color=_INK, ha="center", va="center")
    context = f"{c.season or ''} four-seamer vs. {_hand_word(c.hand)}-handed MLB".strip()
    if c.zone_part != "all":
        context = f"{ZONE_PARTS[c.zone_part].capitalize()} only  ·  {context}"
    if c.first_date:
        context += ", full season" if _full_season(c) else f", {c.first_date}–{c.last_date}"
    if height_in is not None and not pd.isna(height_in):
        average = ""
        if league_height_in is not None and not pd.isna(league_height_in):
            average = f" (avg {_format_height(league_height_in)})"
        context += f"  ·  {_format_height(height_in)}{average}"
    ax.text(0.5, 0.60, context, fontsize=_TEXT, color=_MUTED, ha="center", va="center")
    # The two warnings sit in opposite corners, each on two short lines.
    if c.pitcher.pitches < SMALL_SAMPLE:
        title, detail = _sample_warning(c).split(": ", 1)
        ax.text(1.0, 0.10, title, fontsize=_TEXT, fontweight="bold", color=_PITCHER_COLOR,
                ha="right", va="center")
        ax.text(1.0, 0.34, detail, fontsize=_TEXT, color=_INK, ha="right", va="center")
    flag = not_primary_words(c.ff_pct, c.main_pitch, c.main_pct)
    if flag:
        main, four_seam, most = flag
        ax.text(0.0, 0.10, "NOT HIS PRIMARY PITCH", fontsize=_TEXT, fontweight="bold",
                color=_PITCHER_COLOR, va="center")
        ax.text(0.0, 0.34, f"Four-seamer {four_seam}  ·  {main.capitalize()} {most}",
                fontsize=_TEXT, color=_INK, va="center")
    # Only a fastball that stands out gets a heading and a reason.
    if c.classification != "About average":
        ax.text(0.5, 1.13, c.classification, fontsize=34, fontweight="bold", color=_INK,
                ha="center", va="center")
        ax.text(0.5, 1.62, _why(c), fontsize=16, color="#374151", ha="center", va="center")


def _draw_angle_tags(ax, c: Comparison, y_top: float) -> None:
    """The two headline numbers, as colored tags in the sky above the pitch paths."""
    # Two numbers to compare by eye: his angle, and what MLB does at that height.
    his = round(c.pitcher.observed_vaa, 1)
    # Built from the rounded gap, so what a reader subtracts matches the verdict.
    typical = his - round(c.pitcher.vaa_aa, 1)
    number_z, label_z = y_top - 0.75, y_top - 1.85
    # x is feet from the plate; his tag is on the mound side, MLB's toward the plate.
    for x, label, value, color in ((40.0, "HIS VAA", his, _PITCHER_COLOR),
                                   (14.0, "MLB VAA, SAME HEIGHT", typical, _LEAGUE_COLOR)):
        ax.text(x, number_z, f"{value:.1f}°", fontsize=30, fontweight="bold", color="white",
                ha="center", va="center", zorder=6, gid="angle-tag",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "none"})
        ax.text(x, label_z, label, fontsize=_TEXT, fontweight="bold", color=color,
                ha="center", va="center", zorder=6)
    ax.text(27.0, number_z, "vertical approach\nangle (VAA)\ncloser to 0 = flatter",
            fontsize=_TEXT, color=_MUTED, ha="center", va="center", zorder=6)


def _hand_word(hand: str) -> str:
    return "right" if hand == "R" else "left"


def _mound_x(distance: float, anchor: float, stretch: float) -> float:
    """Where a spot on or near the mound is drawn.

    The pitchers are drawn to true proportions, so distances there are widened by the
    same factor as height, measured from `anchor`: the MLB average release point. That
    keeps the gray pitcher's hand exactly where his real flight path starts.
    """
    return anchor + (distance - anchor) * stretch


def _bend_to_hand(y: np.ndarray, release_y: float, hand_x: float) -> np.ndarray:
    """Slide the mound end of a flight path so it starts in the drawn pitcher's hand.

    Heights are untouched, and nothing closer to the plate than _BEND_FROM moves, so
    the arrival (the part VAA is about) is exactly as modeled. The slide is small:
    it only covers how far his release differs from the average one.
    """
    t = np.clip((y - _BEND_FROM) / (release_y - _BEND_FROM), 0.0, 1.0)
    return y + (hand_x - release_y) * t * t * (3 - 2 * t)      # 0 to 1 with no corner


def _draw_side_view(ax, c: Comparison, figure_height_ft: float,
                    average_height_ft: float) -> None:
    """Both pitchers and their modeled flight paths, with the arrival tangents."""
    height = c.comparison_height
    # The extension measurement sits above both balls and both caps.
    measure_z = max(c.pitcher.state["release_z"], c.league.state["release_z"],
                    0.9 * max(figure_height_ft, average_height_ft) + 0.4) + 1.1
    x_near, y_bottom = -2.5, -0.45
    # Each path gets its name this far from the plate.
    label_y = 40.0
    at_label = {}
    for key, s in (("p", c.pitcher), ("lg", c.league)):
        # np.interp needs increasing x, and the path runs from far to near: reverse it.
        at_label[key] = float(np.interp(label_y, s.modeled.y[::-1], s.modeled.z[::-1]))
    # Leave sky above the path names for the two angle tags.
    y_top = max(7.6, height + 4.0, measure_z + 1.1, max(at_label.values()) + 0.83 + 2.5)
    ax.set_ylim(y_bottom, y_top)
    ax.axis("off")
    # The left edge depends on how wide the pitchers are, which depends on the left
    # edge. stretch = k * (far - near), so the two can be solved for directly.
    anchor = c.league.state["release_y"]
    ax.set_xlim(1.0, 0.0)
    k = _vertical_stretch(ax)             # the stretch for a view exactly 1 ft wide
    rubber_back = RUBBER_Y - anchor       # rubber behind the average release (ft)
    x_far = ((anchor + _BEHIND_RUBBER - rubber_back * k * x_near)
             / (1 - rubber_back * k))
    # Seen from his throwing-arm side: a righty from the third-base side (mound on the
    # left), a lefty from the first-base side (mound on the right). Swapping the two
    # limits mirrors every shape and position, while the words still read normally.
    lefty = c.hand == "L"
    ax.set_xlim(*((x_near, x_far) if lefty else (x_far, x_near)))
    stretch = _vertical_stretch(ax)       # limits are final now
    # The mound is widened like the pitchers who stand on it.
    ground_x = np.linspace(x_near, x_far, 400)
    ground_z = vaa_figures.mound_height(anchor + (ground_x - anchor) / stretch)
    ax.fill_between(ground_x, y_bottom, ground_z, color=_GROUND, linewidth=0, zorder=0.5)
    if c.zone is not None:
        bottom, top = c.zone
        ax.add_patch(Rectangle((0, bottom), vaa.PLATE_FRONT_Y, top - bottom,
                               facecolor="#bfdbfe", alpha=0.6, edgecolor="none", zorder=1))
        if c.zone_part != "all":                     # darken the chosen third
            third = (top - bottom) / 3
            low = bottom + third * ("bottom", "middle", "top").index(c.zone_part)
            ax.add_patch(Rectangle((0, low), vaa.PLATE_FRONT_Y, third, facecolor="#3b82f6",
                                   alpha=0.55, edgecolor="none", zorder=1.1, gid="zone-third"))

    # The MLB average pitcher in gray first, then him in red on top.
    for side, height_ft, color, extra in (
            (c.league, average_height_ft, _GHOST_COLOR, {"behind": -1.4, "ball_gid": "average-ball"}),
            (c.pitcher, figure_height_ft, _PITCHER_COLOR, {})):
        vaa_figures.draw_pitcher_side(
            ax, (side.state["release_y"], side.state["release_z"]), RUBBER_Y, height_ft,
            color, stretch, anchor=anchor, **extra)
    _draw_plate_end(ax, c)
    hand_x = _mound_x(c.pitcher.state["release_y"], anchor, stretch)
    _draw_extension(ax, c, measure_z, _mound_x(RUBBER_Y, anchor, stretch), hand_x)
    _draw_slot_tag(ax, c, hand_x)

    front, back = vaa.PLATE_FRONT_Y, vaa.PLATE_FRONT_Y + _TANGENT_RUN
    # A level line at the plate-crossing height, to read the tangents against.
    ax.plot([back, front], [height, height], color="#9ca3af", linewidth=1.1,
            linestyle=(0, (1, 2)), zorder=2.9, gid="level-line")
    ax.text(back, height - 0.12, "level", fontsize=_TEXT, color=_MUTED,
            ha="right" if lefty else "left", va="top", zorder=5)   # along its line

    drawn = {}
    for key, s, color in (("lg", c.league, _LEAGUE_COLOR), ("p", c.pitcher, _PITCHER_COLOR)):
        path = s.modeled
        release_y = s.state["release_y"]
        drawn[key] = (_bend_to_hand(path.y, release_y, _mound_x(release_y, anchor, stretch)),
                      path.z)
        ax.plot(*drawn[key], color=color, linewidth=3.0, zorder=3, gid="pitch-path")
        # Tangent: carry the arrival direction straight back from the plate.
        slope = path.vz_final / path.vy_final          # feet up per foot toward the mound
        ax.plot([front, back], [height, height + slope * _TANGENT_RUN], color=color,
                linewidth=1.5, linestyle=(0, (5, 3)), zorder=3.1, gid="tangent-line")
    ax.scatter([front], [height], color=_INK, s=34, zorder=4.2,
               edgecolors="white", linewidths=1.0)
    # Under the paths, a little below the level line.
    ax.text(22.0, height - 1.45, "dashed line = the ball's direction\nas it reaches the plate",
            fontsize=_TEXT, color=_MUTED, ha="center", va="center", zorder=5)

    # Name each path right on it; the higher one gets its name above.
    for key, (x, z) in drawn.items():                     # where the line really is now
        at_label[key] = float(np.interp(label_y, x[::-1], z[::-1]))
    higher = "p" if at_label["p"] >= at_label["lg"] else "lg"
    for key, text, color in (("p", c.name, _PITCHER_COLOR), ("lg", "MLB average", _LEAGUE_COLOR)):
        above = key == higher
        ax.text(label_y, at_label[key] + (0.28 if above else -0.28), text, fontsize=15,
                fontweight="bold", color=color, ha="center",
                va="bottom" if above else "top", zorder=5)

    # Written on the ground itself.
    ax.text(30.0, y_bottom / 2,
            f"His typical fastball and MLB's, both crossing the plate {height:g} ft high",
            fontsize=_TEXT, color=_MUTED, ha="center", va="center", zorder=5)
    _draw_angle_tags(ax, c, y_top)


def _draw_extension(ax, c: Comparison, level: float, rubber_x: float,
                    hand_x: float) -> None:
    """Measure how far in front of the rubber he lets go, the way a pitching lab does:

    one upright line from the rubber, one from the ball, joined above his head with
    the length written in the middle. `rubber_x` and `hand_x` are where the widened
    mound drawing puts the rubber and the hand.
    """
    if pd.isna(c.pitcher.extension):
        return                                    # nothing measured: draw nothing
    release_z = c.pitcher.state["release_z"]
    rubber_top = float(vaa_figures.mound_height(RUBBER_Y))
    # The pitching rubber, where every extension is measured from.
    ax.add_patch(Rectangle((rubber_x - 0.7, rubber_top), 1.4, 0.12,
                           facecolor="white", edgecolor=_INK, linewidth=1.0, zorder=2.9))
    # The two uprights, each running a little past the measuring line. They sit
    # behind the figure (low zorder), so a low release does not cross out his face.
    for y, bottom in ((rubber_x, rubber_top), (hand_x, release_z)):
        ax.plot([y, y], [bottom, level + 0.22], color=_PITCHER_COLOR, linewidth=1.5,
                zorder=1.9, gid="extension-upright")
    ax.plot([rubber_x, hand_x], [level, level], color=_PITCHER_COLOR, linewidth=1.5,
            zorder=3.2, gid="extension-line")
    middle = (rubber_x + hand_x) / 2
    # A white box behind the number breaks the line, as in the reference drawing.
    ax.text(middle, level, _format_height(c.pitcher.extension * 12), fontsize=14,
            fontweight="bold", color=_PITCHER_COLOR, ha="center", va="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0})
    # Above the tops of the uprights: a short extension is narrower than the word.
    ax.text(middle, level + 0.34, "extension", fontsize=_TEXT, color=_PITCHER_COLOR,
            ha="center", va="bottom", zorder=5)


def _slot_name(arm_angle: float) -> str | None:
    """"SIDEARM" or "SUBMARINE" for a low slot; None for everyone else or no data."""
    if pd.isna(arm_angle):
        return None
    angle = round(arm_angle)              # judge the number the card shows
    if angle < _SUBMARINE_BELOW:
        return "SUBMARINE"
    return "SIDEARM" if angle < _SIDEARM_BELOW else None


def _draw_slot_tag(ax, c: Comparison, hand_x: float) -> None:
    """Say it under his hand: from the side, a low arm slot just looks like a short arm."""
    name = _slot_name(c.pitcher.arm_angle)
    if name is None:
        return
    # A little toward the plate and below the hand; never down on the ground.
    height = max(1.2, c.pitcher.state["release_z"] - 2.25)
    ax.text(hand_x - 7.0, height, f"{name}\n{_fmt(c.pitcher.arm_angle, 0, '°')} arm angle",
            fontsize=15, fontweight="bold", color="white", ha="center", va="center",
            zorder=7, gid="slot-tag",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": _PITCHER_COLOR, "edgecolor": "none"})


def _draw_plate_end(ax, c: Comparison) -> None:
    """What the pitch is thrown to: home plate, and the strike zone with its name."""
    front = vaa.PLATE_FRONT_Y                 # the plate runs from 0 (back tip) to here
    ax.add_patch(Rectangle((0, 0), front, 0.14, facecolor="white", edgecolor=_INK,
                           linewidth=1.2, zorder=1.5, gid="home-plate"))
    if c.zone is None:                        # no strike-zone data: just the plate
        return
    bottom, top = c.zone
    ax.add_patch(Rectangle((0, bottom), front, top - bottom, facecolor="none",
                           edgecolor=_ZONE_COLOR, linewidth=2.0, zorder=2.0, gid="zone-outline"))
    ax.text(front / 2, bottom - 0.25, "strike zone", fontsize=_TEXT, color=_ZONE_COLOR,
            ha="center", va="top", zorder=5)


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
    """One trait: value, league value, and a ranking line (or the arm drawing)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98, boxstyle="round,pad=0,rounding_size=0.06",
        facecolor=_TILE_FILL, edgecolor="none",
    ))
    ax.text(0.5, 0.88, card["title"], fontsize=_TEXT, fontweight="bold", color=_MUTED,
            ha="center", va="center")
    ax.text(0.5, 0.70, card["value"], fontsize=25, fontweight="bold", color=_INK,
            ha="center", va="center")
    ax.text(0.5, 0.545, card["league"], fontsize=_TEXT, color=_MUTED, ha="center", va="center")
    ax.text(0.5, 0.14, card["percentile"], fontsize=14, color=_INK, ha="center", va="center")
    if card["key"] == "arm_angle":
        _draw_arm_picture(ax, card, hand)
    elif card["rank"] is not None:
        _draw_rank_line(ax, card["rank"])


def _draw_rank_line(ax, share_below: float) -> None:
    """A low-to-high line: the middle pitcher marked, and a dot where this one sits."""
    left, right, y = 0.12, 0.88, 0.33
    ax.plot([left, right], [y, y], color="#d1d5db", linewidth=7, solid_capstyle="round")
    ax.plot([0.5, 0.5], [y - 0.06, y + 0.06], color=_LEAGUE_COLOR, linewidth=2)
    # share_below runs from 0 (lowest) to 1 (highest), so it places the dot directly.
    ax.scatter([left + (right - left) * share_below], [y], s=190, color=_PITCHER_COLOR,
               edgecolors="white", linewidths=2, zorder=3, gid="rank-dot")


def _draw_arm_picture(ax, card: dict, hand: str) -> None:
    """A small front view with both arm angles, drawn true to scale."""
    mine, theirs = card["mine"], card["theirs"]
    if pd.isna(mine) or pd.isna(theirs):
        return                                   # missing data: show "n/a", draw nothing
    picture = ax.inset_axes([0.06, 0.05, 0.88, 0.43])     # the whole lower half
    picture.axis("off")
    picture.set_aspect("equal")                  # equal scales: the arm angle is TRUE
    direction = -1 if hand == "R" else 1         # which side the throwing arm is on
    picture.set_xlim(*sorted((direction * 3.2, -direction * 1.4)))
    picture.set_ylim(2.72, 6.9)
    start = direction * vaa_figures.FRONT_SHOULDER_HALF
    level = vaa_figures.FRONT_SHOULDER_Z
    picture.plot([start, start + direction * 2.5], [level, level], color="#9ca3af",
                 linewidth=1.0, linestyle=(0, (3, 3)), zorder=0.5)
    vaa_figures.draw_pitcher_front(picture, hand,
                                   [(theirs, _LEAGUE_COLOR), (mine, _PITCHER_COLOR)],
                                   _FIGURE_FILL)


def _season_words(c: Comparison) -> str:
    if _full_season(c):
        return f"full {c.season} regular season"
    dates = f", {c.first_date} to {c.last_date}" if c.first_date else ""
    return f"{c.season} regular season{dates}"


def method_notes(c: Comparison) -> list[str]:
    """How the chart's numbers are made, for the page to show on request."""
    lg = c.league
    return [
        f"Source: Baseball Savant Statcast via pybaseball, {_season_words(c)}. League: "
        f"{lg.pitches:,} {_hand_word(c.hand)}-handed four-seamers from {c.league_pitchers} pitchers.",
        "VAA is the downward angle of the ball at the front of the plate: atan2(vz, |vy|).",
        "MLB VAA, same height: the league average for fastballs crossing the plate where his did.",
        "Pitch paths are modeled from average measured release point, speed and acceleration "
        "(gravity, drag, spin).",
        "Red is him; gray is an MLB average pitcher with the average height and release. "
        "The pitchers are drawn to true proportions, much larger than the distance to the "
        "plate. Release height and extension are measured; the stride and lean are drawn "
        "from extension, not tracked.",
        "The reason given for a flatter or steeper fastball comes from a model of four inputs. "
        "Extension is judged with release height and speed held equal.",
        "Percentiles run from lowest to highest number, not worst to best. The 50th is the "
        "middle pitcher, which can differ a little from the MLB average of all pitches.",
        "Arm angle is Statcast's line from the throwing shoulder to the ball at release, "
        "against flat ground: 0° is sidearm, 90° is over the top.",
        f"The drawing tags a slot below {_SIDEARM_BELOW}° as sidearm and below "
        f"{_SUBMARINE_BELOW}° as submarine. Those cut-offs are this page's own, not "
        "Statcast categories.",
    ]


def _draw_footnotes(fig, c: Comparison, stretch: float) -> None:
    """Source and sample in two lines, plus any warning about the data."""
    third = "" if c.zone_part == "all" else f" in the {c.zone_part} third"
    lines = [f"Statcast, {_season_words(c)}  ·  {c.pitcher.pitches:,} four-seamers{third}  ·  "
             f"ball flight drawn {stretch:.1f}× taller than true"]
    if c.ranks:
        lines.append(f"Percentiles: low to high, vs. {c.rank_pool} {_hand_word(c.hand)}-handers "
                     f"with {SMALL_SAMPLE}+ four-seamers{third}")
    tall = fig.get_figheight()
    for row, line in enumerate(lines):
        fig.text(0.5, (0.62 - row * 0.27) / tall, line, fontsize=_TEXT, color=_MUTED, ha="center")
    # The small-sample warning is already at the top of the chart; do not repeat it.
    notes = [n for n in c.notes if not n.startswith("small sample")]
    if notes:
        estimated = any("estimated" in n or "derived" in n for n in notes)
        lead = "APPROXIMATION" if estimated else "Note"
        fig.text(0.5, (0.62 - len(lines) * 0.27) / tall, f"{lead}: {'; '.join(notes)}.",
                 fontsize=11, fontweight="bold", color=_PITCHER_COLOR, ha="center")
