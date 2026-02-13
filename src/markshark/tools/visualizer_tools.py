from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .score_tools import grid_centers_axis_mode, centers_to_circle_rois

if TYPE_CHECKING:
    # Only for type hints, avoids runtime import cost/cycles
    from ..bubblemap_io import GridLayout


# Must match the default used by decode_layout() in score_tools.py
SCORING_INNER_RADIUS_RATIO = 0.85


def draw_layout_circles(
    img_bgr: np.ndarray,
    layout: "GridLayout",
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    inner_color: Tuple[int, int, int] = (0, 0, 255),
    inner_thickness: int = 1,
    inner_radius_ratio: float = SCORING_INNER_RADIUS_RATIO,
) -> None:
    """Draw bubble overlay using the same ROI geometry as the scoring engine.

    Draws two concentric shapes per bubble:

    - **Outer** (*color*, default green): full ROI bounding circle produced by
      :func:`centers_to_circle_rois` — the same function the scorer calls.
    - **Inner** (*inner_color*, default red, thin): the actual scoring mask,
      shrunk by *inner_radius_ratio* exactly as :func:`measure_fill_ratio`
      does at scoring time.

    For ``bubble_shape == "oval"`` layouts the shapes become concentric
    ellipses whose radii are derived from the grid-cell dimensions (matching
    the mock-dataset renderer).
    """
    h, w = img_bgr.shape[:2]

    centers = grid_centers_axis_mode(
        layout.x_topleft,
        layout.y_topleft,
        layout.x_bottomright,
        layout.y_bottomright,
        layout.numrows,
        layout.numcols,
    )

    # ---- Oval path (cell-dimension based) --------------------------------
    if getattr(layout, "bubble_shape", "circle") == "oval":
        cell_w = abs(layout.x_bottomright - layout.x_topleft) / max(layout.numcols, 1)
        cell_h = abs(layout.y_bottomright - layout.y_topleft) / max(layout.numrows, 1)
        fill_frac = 0.70
        rx = max(1, int(cell_w * w * fill_frac / 2))
        ry = max(1, int(cell_h * h * fill_frac / 2))
        inner_rx = max(1, int(rx * inner_radius_ratio))
        inner_ry = max(1, int(ry * inner_radius_ratio))

        # Still need pixel centers — derive from ROIs for consistency
        rois = centers_to_circle_rois(centers, w, h, layout.radius_pct)
        for (x, y, rw, rh) in rois:
            cx = x + rw // 2
            cy = y + rh // 2
            cv2.ellipse(img_bgr, (cx, cy), (rx, ry), 0, 0, 360, color, thickness)
            cv2.ellipse(img_bgr, (cx, cy), (inner_rx, inner_ry), 0, 0, 360,
                        inner_color, inner_thickness)
        return

    # ---- Circle path (standard) ------------------------------------------
    # Use the scoring engine's canonical ROI function so geometry can never
    # drift between the map viewer and the scorer.
    rois = centers_to_circle_rois(centers, w, h, layout.radius_pct)

    for (x, y, rw, rh) in rois:
        cx = x + rw // 2
        cy = y + rh // 2
        outer_r = min(rw, rh) // 2

        # Outer circle — full ROI boundary
        cv2.circle(img_bgr, (cx, cy), outer_r, color, thickness)

        # Inner circle — actual scoring mask
        # Replicates measure_fill_ratio():
        #   r = int(0.5 * min(W, H) * inner_radius_ratio)
        inner_r = max(1, int(0.5 * min(rw, rh) * inner_radius_ratio))
        cv2.circle(img_bgr, (cx, cy), inner_r, inner_color, inner_thickness)
