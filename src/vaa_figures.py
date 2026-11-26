"""
Figurines for the VAA chart prototypes: shaped baseball-player silhouettes.

Each body part is a filled "capsule": two circles (one per joint) with straight
sides between them, so a thigh can be thick at the hip and thin at the knee.
Stacking capsules gives a smooth silhouette with a cap, glove, shoes and ball.

The figures are SCHEMATIC. Only a few things come from real data:
    - the pitcher's ball sits at the measured average release point,
    - the body is sized from his listed height,
    - the arm in the front view is drawn at the measured Savant arm angle.
The rest of the pose (knees, hips, lead foot) is a believable guess.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon

# Official mound, simplified to straight segments: 10 inches high at the rubber,
# falling 1 inch per foot toward the plate, then down to field level.
# Pairs are (feet from the plate, height in feet), in increasing distance.
_MOUND_Y = [50.0, 54.0, 60.0, 62.0, 68.0]
_MOUND_Z = [0.0, 4 / 12, 10 / 12, 10 / 12, 0.0]

_LEATHER = "#8b5e3c"      # gloves
_WOOD = "#c8a27a"         # bat


def mound_height(y):
    """Ground height in feet at distance y from the plate (0 away from the mound)."""
    # np.interp joins the points with straight lines and holds the end values (0) outside.
    return np.interp(y, _MOUND_Y, _MOUND_Z)


def _mix(color: str, other: str, amount: float) -> tuple[float, float, float]:
    """Blend `color` toward `other`: amount 0 keeps color, 1 gives other."""
    a, b = np.array(to_rgb(color)), np.array(to_rgb(other))
    return tuple(a + (b - a) * amount)


# ---------------------------------------------------------------------------
# Shapes. These are pure geometry: they return outline points, and know nothing
# about charts.
# ---------------------------------------------------------------------------

def _arc(center, radius: float, start: float, stop: float, n: int = 24) -> np.ndarray:
    """Points along part of a circle, angles in radians."""
    t = np.linspace(start, stop, n)
    return np.column_stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)])


def capsule(p0, p1, r0: float, r1: float) -> np.ndarray:
    """Outline of a tapered limb: radius r0 around p0, r1 around p1, straight sides.

    Returns an (n, 2) array of points going once around the shape.
    """
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    delta = p1 - p0
    length = float(np.hypot(*delta))
    # If one circle swallows the other there are no straight sides to draw.
    if length <= abs(r0 - r1):
        center, radius = (p0, r0) if r0 >= r1 else (p1, r1)
        return _arc(center, radius, 0, 2 * np.pi)
    axis = np.arctan2(delta[1], delta[0])        # direction from p0 to p1
    # Angle at which a straight side just touches both circles.
    touch = np.arccos((r0 - r1) / length)
    back = _arc(p0, r0, axis + touch, axis + 2 * np.pi - touch)   # rounded end at p0
    front = _arc(p1, r1, axis - touch, axis + touch)              # rounded end at p1
    return np.vstack([back, front])


class _Pen:
    """Draws shapes so they keep their proportions on a vertically stretched chart.

    The side view draws height `stretch` times bigger than distance. A circle
    drawn directly in feet would look like a tall oval. So shapes are built in
    "screen" coordinates (distance, height * stretch), where a circle is round,
    and converted back to feet just before drawing. Joints stay at their true
    positions in feet.
    """

    def __init__(self, ax, stretch: float = 1.0):
        self.ax = ax
        self.stretch = stretch

    def screen(self, point) -> np.ndarray:
        """(distance, height) in feet -> screen-proportional coordinates."""
        return np.array([point[0], point[1] * self.stretch], dtype=float)

    def fill(self, outline: np.ndarray, color, zorder: float, edge="none",
             edge_width: float = 0.0, gid: str | None = None) -> None:
        """Draw a screen-coordinate outline as a filled shape."""
        points = np.array(outline, dtype=float)
        points[:, 1] /= self.stretch                      # back to feet
        self.ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=edge,
                                  linewidth=edge_width, zorder=zorder, gid=gid,
                                  joinstyle="round"))

    def limb(self, joints: list, radii: list[float], color, zorder: float,
             halo: bool = False) -> None:
        """A chain of capsules through `joints` (in feet), one radius per joint.

        halo=True first draws a white rim around the whole chain, which separates
        a near arm or leg from the body behind it.
        """
        points = [self.screen(j) for j in joints]
        pieces = [capsule(points[i], points[i + 1], radii[i], radii[i + 1])
                  for i in range(len(points) - 1)]
        if halo:
            for piece in pieces:
                self.fill(piece, "white", zorder, edge="white", edge_width=2.2)
        for piece in pieces:
            self.fill(piece, color, zorder + 0.005)

    def disc(self, center, radius: float, color, zorder: float, **kwargs) -> None:
        """A circle centred at `center` (feet) that looks round on screen."""
        self.fill(_arc(self.screen(center), radius, 0, 2 * np.pi, n=40), color, zorder, **kwargs)

    def mitt(self, center, size: float, color, zorder: float) -> None:
        """A glove: a rounded pocket that widens toward the fingers (upward)."""
        c = self.screen(center)
        self.fill(capsule(c + [0, -size * 0.45], c + [0, size * 0.55], size * 0.65, size),
                  color, zorder, edge="white", edge_width=1.0)

    def cap(self, center, radius: float, facing: int, color, zorder: float) -> None:
        """A cap sitting on a head: a dome plus a brim.

        facing  +1 / -1 for a brim toward larger / smaller x, 0 for facing the viewer.
        """
        c = self.screen(center)
        dome = _arc(c + [0, radius * 0.15], radius * 1.08, 0, np.pi)
        self.fill(dome, color, zorder)
        brim_z = c[1] + radius * 0.28
        if facing == 0:
            ends = ([c[0] - radius * 1.25, brim_z], [c[0] + radius * 1.25, brim_z])
        else:
            ends = ([c[0] + facing * radius * 0.5, brim_z], [c[0] + facing * radius * 1.8, brim_z])
        self.fill(capsule(ends[0], ends[1], radius * 0.17, radius * 0.13), color, zorder)


# ---------------------------------------------------------------------------
# The people.
# ---------------------------------------------------------------------------

def draw_pitcher_side(ax, release: tuple[float, float], rubber_y: float,
                      height_ft: float, color: str, stretch: float = 1.0) -> None:
    """A pitcher at the moment of release, seen from the side.

    release    (distance from plate, height) of the ball: the hand goes here.
    rubber_y   distance of the pitching rubber from the plate.
    height_ft  listed height; body segments are fractions of it.
    stretch    how much taller than true scale the chart draws things.
    The plate is at SMALLER y, so "forward" means subtracting from y.
    """
    rel_y, rel_z = release
    h = height_ft
    pen = _Pen(ax, stretch)
    s = h * stretch                       # body height as it appears on screen

    far = _mix(color, "white", 0.38)      # limbs on the far side are lighter
    dark = _mix(color, "black", 0.32)     # cap and shoes

    # Joints. The lead foot lands about 85% of body height in front of the rubber.
    lead_y = rubber_y - 0.85 * h
    ankle = (lead_y, float(mound_height(lead_y)) + 0.14)
    back_y = rubber_y - 0.7
    back_ankle = (back_y, float(mound_height(back_y)) + 0.22)   # dragging off the rubber

    hip = (lead_y + 0.15 * h, ankle[1] + 0.39 * h)
    shoulder = (hip[0] - 0.12 * h, hip[1] + 0.28 * h)           # torso tilted forward
    lead_knee = (lead_y - 0.03 * h, ankle[1] + 0.22 * h)
    back_knee = (hip[0] + 0.22 * h, hip[1] - 0.20 * h)
    head = (shoulder[0] - 0.05 * h, shoulder[1] + 0.135 * h)
    elbow = ((shoulder[0] + rel_y) / 2 + 0.25, (shoulder[1] + rel_z) / 2 - 0.15)
    glove_elbow = (shoulder[0] - 0.09 * h, shoulder[1] - 0.17 * h)
    glove = (shoulder[0] - 0.23 * h, shoulder[1] - 0.13 * h)

    # Far side first, so everything nearer is painted over it.
    pen.limb([hip, back_knee, back_ankle], [0.050 * s, 0.031 * s, 0.018 * s], far, 2.0)
    pen.limb([back_ankle, (back_ankle[0] + 0.045 * h, back_ankle[1] - 0.12)],
             [0.020 * s, 0.014 * s], _mix(dark, "white", 0.3), 2.02)
    pen.limb([shoulder, glove_elbow, glove], [0.030 * s, 0.022 * s, 0.017 * s], far, 2.04)
    pen.mitt(glove, 0.034 * s, _LEATHER, 2.06)

    # Lead leg, torso, belt, head.
    pen.limb([hip, lead_knee, ankle], [0.055 * s, 0.034 * s, 0.020 * s], color, 2.2)
    toe = (ankle[0] - 0.075 * h, ankle[1] - 0.04)
    pen.limb([ankle, toe], [0.022 * s, 0.014 * s], dark, 2.22)
    pen.limb([hip, shoulder], [0.058 * s, 0.074 * s], color, 2.3)
    waist = (hip[0] - 0.012 * h, hip[1] + 0.035 * h)
    pen.limb([(waist[0] - 0.088 * h, waist[1] + 0.007 * h), (waist[0] + 0.088 * h, waist[1] - 0.007 * h)],
             [0.008 * s, 0.008 * s], dark, 2.33)
    pen.limb([shoulder, head], [0.030 * s, 0.025 * s], color, 2.35)            # neck
    pen.disc(head, 0.058 * s, color, 2.7, edge="white", edge_width=1.0)
    pen.cap(head, 0.058 * s, facing=-1, color=dark, zorder=2.72)               # looking at the plate

    # Throwing arm on top, ending in the ball exactly at the release point.
    pen.limb([shoulder, elbow, release], [0.034 * s, 0.022 * s, 0.014 * s], color, 2.6, halo=True)
    pen.disc(release, 0.024 * s, "white", 3.5, edge=dark, edge_width=1.3, gid="ball")


def draw_batter_side(ax, color: str, stretch: float = 1.0) -> None:
    """A generic right-sized batter at the plate, faint, for context only."""
    pen = _Pen(ax, stretch)
    s = 6.1 * stretch
    far = _mix(color, "white", 0.45)
    dark = _mix(color, "black", 0.12)

    hip, shoulder = (0.9, 3.0), (0.8, 4.55)
    head, hands = (0.95, 5.35), (-0.15, 4.55)

    pen.limb([hip, (-0.35, 1.6), (-0.6, 0.18)], [0.050 * s, 0.032 * s, 0.019 * s], far, 0.60)
    pen.limb([(-0.6, 0.16), (-0.05, 0.1)], [0.020 * s, 0.013 * s], far, 0.61)
    pen.limb([hip, (2.2, 1.6), (2.8, 0.18)], [0.055 * s, 0.034 * s, 0.020 * s], color, 0.65)
    pen.limb([(2.8, 0.16), (3.35, 0.1)], [0.022 * s, 0.014 * s], dark, 0.66)
    pen.limb([hip, shoulder], [0.060 * s, 0.075 * s], color, 0.70)
    pen.limb([shoulder, head], [0.028 * s, 0.025 * s], color, 0.71)
    pen.disc(head, 0.058 * s, color, 0.72)
    pen.cap(head, 0.058 * s, facing=1, color=dark, zorder=0.73)    # looking at the pitcher
    pen.limb([hands, (-1.05, 6.2)], [0.008 * s, 0.024 * s], _mix(_WOOD, "white", 0.35), 0.78)
    pen.limb([shoulder, (0.55, 3.85), hands], [0.030 * s, 0.022 * s, 0.016 * s], dark, 0.80)


# Front view geometry, in arbitrary body units (the axes use equal scales).
FRONT_SHOULDER_HALF = 0.55
FRONT_SHOULDER_Z = 4.4
FRONT_ARM_LENGTH = 2.2


def arm_end(hand: str, arm_angle: float) -> tuple[float, float]:
    """Where the throwing hand is in the front view (viewed from the catcher).

    A right-hander's throwing arm appears on the viewer's LEFT. Savant's arm
    angle is 0 for straight out sideways and 90 for straight up.
    """
    side = -1 if hand == "R" else 1
    angle = np.radians(arm_angle)
    x = side * (FRONT_SHOULDER_HALF + FRONT_ARM_LENGTH * np.cos(angle))
    z = FRONT_SHOULDER_Z + FRONT_ARM_LENGTH * np.sin(angle)
    return float(x), float(z)


def draw_pitcher_front(ax, hand: str, arms: list[tuple[float, str]], body_color: str) -> None:
    """A pitcher facing the viewer, with one throwing arm per (angle, color) in `arms`.

    Arms are drawn in list order, so put the one that should be on top LAST.
    """
    pen = _Pen(ax)                                   # equal scales: no stretch
    side = -1 if hand == "R" else 1
    throwing_shoulder = (side * FRONT_SHOULDER_HALF, FRONT_SHOULDER_Z)
    glove_shoulder = (-side * FRONT_SHOULDER_HALF, FRONT_SHOULDER_Z)
    dark = _mix(body_color, "black", 0.18)

    # Legs and shoes, mirrored left and right.
    for x in (-1, 1):
        pen.limb([(x * 0.27, 2.65), (x * 0.52, 1.45), (x * 0.72, 0.24)],
                 [0.31, 0.20, 0.12], body_color, 1.0, halo=True)
        pen.limb([(x * 0.72, 0.16), (x * 1.02, 0.12)], [0.15, 0.11], dark, 1.02)

    # Torso: broad at the shoulders, narrower at the waist.
    torso = np.array([(-0.60, 4.50), (0.60, 4.50), (0.40, 2.75), (-0.40, 2.75)])
    pen.fill(torso, body_color, 1.1)
    pen.limb([throwing_shoulder, glove_shoulder], [0.22, 0.22], body_color, 1.12)
    pen.limb([(-0.34, 2.75), (0.34, 2.75)], [0.20, 0.20], body_color, 1.13)
    pen.limb([(-0.40, 2.88), (0.40, 2.88)], [0.05, 0.05], dark, 1.15)            # belt
    pen.limb([(0.0, 4.55), (0.0, 5.0)], [0.17, 0.17], body_color, 1.16)          # neck

    pen.disc((0.0, 5.32), 0.40, body_color, 1.4)
    pen.cap((0.0, 5.32), 0.40, facing=0, color=dark, zorder=1.42)

    # Glove arm tucked in, glove at the chest.
    glove = (-side * 0.12, 3.72)
    pen.limb([glove_shoulder, (-side * 0.98, 3.45), glove], [0.18, 0.14, 0.11],
             body_color, 1.5, halo=True)
    pen.mitt(glove, 0.24, _LEATHER, 1.55)

    for order, (angle, color) in enumerate(arms):
        end = arm_end(hand, angle)
        z = 2.0 + order * 0.1
        pen.limb([throwing_shoulder, end], [0.17, 0.09], color, z, halo=True)
        pen.disc(end, 0.17, "white", z + 0.05, edge=color, edge_width=2.0)       # the ball
