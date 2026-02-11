from __future__ import annotations

from typing import Iterable, List, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .score_tools import grid_centers_axis_mode  # canonical implementation

if TYPE_CHECKING:
    # Only for type hints, avoids runtime import cost/cycles
    from ..bubblemap_io import GridLayout


def centers_to_radius_px(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int,
    img_h: int,
    radius_pct: float,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Convert normalized centers to pixel centers, and return a pixel radius.

    radius_pct is interpreted as a fraction of image width (consistent with bubblemap).
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
    """Draw bubble shapes in-place for one GridLayout using axis-mode geometry.

    If the grid cells are not square the bubbles are drawn as ellipses so the
    overlay matches the actual oval bubbles on the template.
    """
    h, w = img_bgr.shape[:2]
    numrows = layout.numrows
    numcols = layout.numcols

    centers = grid_centers_axis_mode(
        layout.x_topleft,
        layout.y_topleft,
        layout.x_bottomright,
        layout.y_bottomright,
        numrows,
        numcols,
    )
    pts_px, r_px = centers_to_radius_px(centers, w, h, layout.radius_pct)

    # Compute Y-radius by matching the aspect ratio of the grid cells.
    # radius_pct is relative to image width; derive the Y-radius from the
    # cell spacing ratio so oval bubbles are drawn correctly.
    x_extent = abs(layout.x_bottomright - layout.x_topleft)
    y_extent = abs(layout.y_bottomright - layout.y_topleft)
    col_span = max(1, numcols - 1)
    row_span = max(1, numrows - 1)

    if x_extent > 0 and y_extent > 0 and numrows > 1 and numcols > 1:
        # Cell spacing in normalised coordinates
        dx = x_extent / col_span   # horizontal spacing (fraction of width)
        dy = y_extent / row_span   # vertical spacing   (fraction of height)
        # Convert both to pixel units so they're comparable
        dx_px = dx * w
        dy_px = dy * h
        aspect = dy_px / dx_px     # >1 → tall cells, <1 → wide cells
        ry_px = max(1, int(round(r_px * aspect)))
    else:
        ry_px = r_px

    if ry_px == r_px:
        # Perfect circles — use the faster cv2.circle path
        for (cx, cy) in pts_px:
            cv2.circle(img_bgr, (cx, cy), r_px, color, thickness)
    else:
        # Ellipses — (rx, ry), angle 0, full arc
        for (cx, cy) in pts_px:
            cv2.ellipse(
                img_bgr, (cx, cy), (r_px, ry_px), 0, 0, 360, color, thickness,
            )
