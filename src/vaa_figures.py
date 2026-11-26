"""
Figurines for the VAA chart prototypes: simple stick-style people.

Each figure is a few thick, round-ended lines (the limbs) plus a dot (the head).
They are SCHEMATIC. Only a few things come from real data:
    - the pitcher's hand sits at the measured average release point,
    - the body is sized from his listed height,
    - the arm in the front view is drawn at the measured Savant arm angle.
The rest of the pose (knees, hips, lead foot) is a believable guess.
"""

from __future__ import annotations

import numpy as np

# Official mound, simplified to straight segments: 10 inches high at the rubber,
# falling 1 inch per foot toward the plate, then down to field level.
# Pairs are (feet from the plate, height in feet), in increasing distance.
_MOUND_Y = [50.0, 54.0, 60.0, 62.0, 68.0]
_MOUND_Z = [0.0, 4 / 12, 10 / 12, 10 / 12, 0.0]


def mound_height(y):
    """Ground height in feet at distance y from the plate (0 away from the mound)."""
    # np.interp joins the points with straight lines and holds the end values (0) outside.
    return np.interp(y, _MOUND_Y, _MOUND_Z)


def _limb(ax, points: list[tuple[float, float]], color: str, width: float,
          zorder: float, alpha: float = 1.0) -> None:
    """One body part: a thick line through the joints, with rounded ends and corners.

    `width` is in screen points, so limbs keep their thickness even though the
    side view stretches height more than distance.
    """
    xs, ys = zip(*points)       # [(x1,y1),(x2,y2)] -> (x1,x2), (y1,y2)
    ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, zorder=zorder,
            solid_capstyle="round", solid_joinstyle="round")


def _head(ax, center: tuple[float, float], color: str, size: float,
          zorder: float, alpha: float = 1.0) -> None:
    """A round head. scatter() draws a true circle whatever the axis scales are."""
    ax.scatter([center[0]], [center[1]], s=size, color=color, alpha=alpha,
               zorder=zorder, linewidths=0)


def draw_pitcher_side(ax, release: tuple[float, float], rubber_y: float,
                      height_ft: float, color: str) -> None:
    """A pitcher at the moment of release, seen from the side.

    release    (distance from plate, height) of the ball: the hand goes here.
    rubber_y   distance of the pitching rubber from the plate.
    height_ft  listed height; body segments are fractions of it.
    The plate is at SMALLER y, so "forward" means subtracting from y.
    """
    rel_y, rel_z = release
    h = height_ft

    # Feet: the lead foot lands about 85% of body height in front of the rubber.
    lead_foot_y = rubber_y - 0.85 * h
    lead_foot = (lead_foot_y, float(mound_height(lead_foot_y)))
    # The back foot is just dragging off the rubber, slightly off the ground.
    back_y = rubber_y - 0.7
    back_foot = (back_y, float(mound_height(back_y)) + 0.15)

    hip = (lead_foot_y + 0.15 * h, lead_foot[1] + 0.40 * h)
    shoulder = (hip[0] - 0.12 * h, hip[1] + 0.28 * h)     # torso tilted toward the plate
    lead_knee = (lead_foot_y - 0.02 * h, lead_foot[1] + 0.22 * h)
    back_knee = (hip[0] + 0.22 * h, hip[1] - 0.20 * h)
    head = (shoulder[0] - 0.03 * h, shoulder[1] + 0.13 * h)

    # Throwing arm ends exactly at the release point; the elbow is bent slightly.
    elbow = ((shoulder[0] + rel_y) / 2 + 0.25, (shoulder[1] + rel_z) / 2 - 0.15)
    glove = (shoulder[0] - 0.14 * h, shoulder[1] - 0.17 * h)

    _limb(ax, [hip, back_knee, back_foot], color, 6.5, zorder=2.0, alpha=0.65)  # far leg, faded
    _limb(ax, [shoulder, glove], color, 4.5, zorder=2.0, alpha=0.65)            # glove arm, faded
    _limb(ax, [hip, lead_knee, lead_foot], color, 7.0, zorder=2.2)
    _limb(ax, [hip, shoulder], color, 11.0, zorder=2.3)
    _limb(ax, [shoulder, elbow, (rel_y, rel_z)], color, 5.0, zorder=2.4)
    _head(ax, head, color, 230, zorder=2.5)


def draw_batter_side(ax, color: str) -> None:
    """A generic batter at the plate, drawn faintly behind the pitches for context."""
    hip, shoulder = (0.9, 3.0), (0.8, 4.6)
    hands = (-0.1, 4.5)
    _limb(ax, [(-0.6, 0.0), (-0.3, 1.6), hip], color, 6.5, zorder=0.6)   # back leg
    _limb(ax, [(2.9, 0.0), (2.2, 1.6), hip], color, 6.5, zorder=0.6)     # front leg
    _limb(ax, [hip, shoulder], color, 11.0, zorder=0.7)
    _limb(ax, [shoulder, (0.7, 3.9), hands], color, 4.5, zorder=0.8)     # arms
    _limb(ax, [hands, (-1.0, 6.2)], color, 3.0, zorder=0.8)              # bat
    _head(ax, (0.8, 5.45), color, 230, zorder=0.9)


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
    side = -1 if hand == "R" else 1
    throwing_shoulder = (side * FRONT_SHOULDER_HALF, FRONT_SHOULDER_Z)
    glove_shoulder = (-side * FRONT_SHOULDER_HALF, FRONT_SHOULDER_Z)
    hip = (0.0, 2.7)

    _limb(ax, [(-0.8, 0.0), (-0.45, 1.4), hip], body_color, 7.0, zorder=1)
    _limb(ax, [(0.8, 0.0), (0.45, 1.4), hip], body_color, 7.0, zorder=1)
    _limb(ax, [hip, (0.0, FRONT_SHOULDER_Z)], body_color, 13.0, zorder=1.1)
    _limb(ax, [throwing_shoulder, glove_shoulder], body_color, 7.0, zorder=1.2)
    _limb(ax, [glove_shoulder, (-side * 1.05, 3.2), (-side * 0.2, 3.5)], body_color, 5.0, zorder=1.3)
    _head(ax, (0.0, 5.35), body_color, 260, zorder=1.4)

    for order, (angle, color) in enumerate(arms):
        end = arm_end(hand, angle)
        _limb(ax, [throwing_shoulder, end], color, 4.5, zorder=2 + order * 0.1)
        _head(ax, end, color, 55, zorder=2.05 + order * 0.1)             # the ball
