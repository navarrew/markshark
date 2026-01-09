from __future__ import annotations

from typing import Iterable, List, Tuple, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    # Only for type hints, avoids runtime import cost/cycles
    from ..bublmap_io import GridLayout


def grid_centers_axis_mode(
    x_tl: float,
    y_tl: float,
    x_br: float,
    y_br: float,
    questions: int,
    choices: int,
) -> List[Tuple[float, float]]:
    """
    Return normalized (x, y) centers for a questions x choices grid, by linear
    interpolation between the top-left and bottom-right bubble centers.

    Coordinates are normalized fractions (0..1) in the bublmap coordinate system.
    """
    centers: List[Tuple[float, float]] = []
    q_den = max(1, questions - 1)
    c_den = max(1, choices - 1)

    for q in range(questions):
        t_q = q / q_den
        y = y_tl + t_q * (y_br - y_tl)
        for c in range(choices):
            t_c = c / c_den
            x = x_tl + t_c * (x_br - x_tl)
            centers.append((x, y))
    return centers


def centers_to_radius_px(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int,
    img_h: int,
    radius_pct: float,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Convert normalized centers to pixel centers, and return a pixel radius.

    radius_pct is interpreted as a fraction of image width (consistent with bublmap).
    """
    r_px = max(1, int(round(radius_pct * img_w)))
    pts_px: List[Tuple[int, int]] = []
    for (x, y) in centers_pct:
        cx = int(round(x * img_w))
        cy = int(round(y * img_h))
        pts_px.append((cx, cy))
    return pts_px, r_px


def draw_layout_circles(
    img_bgr: np.ndarray,
    layout: "GridLayout",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Draw circles in-place for one GridLayout using axis-mode geometry.
    """
    h, w = img_bgr.shape[:2]
    centers = grid_centers_axis_mode(
        layout.x_topleft,
        layout.y_topleft,
        layout.x_bottomright,
        layout.y_bottomright,
        layout.questions,
        layout.choices,
    )
    pts_px, r_px = centers_to_radius_px(centers, w, h, layout.radius_pct)
    for (cx, cy) in pts_px:
        cv2.circle(img_bgr, (cx, cy), r_px, color, thickness)
