# src/markshark/visualize_core.py
#!/usr/bin/env python3
"""
MarkShark
visualize_core.py
Visualize OMR bubble positions from an axis-based config.

Exports:
  - overlay_config(config_path, input_path, out_image, dpi=300, color=(0,255,0), thickness=2) -> str
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .config_io import load_config, Config, GridLayout
from .tools.zone_visualizer import (
    grid_centers_axis_mode,   # centers (normalized) from (x_tl,y_tl)->(x_br,y_br)
    centers_to_radius_px,     # centers -> (pixel centers, pixel radius)
)

def _load_input_image(path: str, dpi: int = 300) -> np.ndarray:
    """Return a BGR image for either a PDF (first page) or a raster image."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except Exception as e:
            raise RuntimeError("pdf2image is required to read PDFs. Install with `pip install pdf2image`.") from e
        pages = convert_from_path(str(p), dpi=dpi)
        if not pages:
            raise ValueError(f"No pages in PDF: {path}")
        rgb = np.array(pages[0])  # first page as RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def _draw_layout_circles(
    img_bgr: np.ndarray,
    layout: GridLayout,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> None:
    """Draw circles in-place for one layout using axis-mode geometry."""
    h, w = img_bgr.shape[:2]
    centers = grid_centers_axis_mode(
        layout.x_topleft, layout.y_topleft,
        layout.x_bottomright, layout.y_bottomright,
        layout.questions, layout.choices
    )
    pts_px, r_px = centers_to_radius_px(centers, w, h, layout.radius_pct)
    for (cx, cy) in pts_px:
        cv2.circle(img_bgr, (cx, cy), r_px, color, thickness, lineType=cv2.LINE_AA)

def overlay_config(
    config_path: str,
    input_path: str,
    out_image: str,
    dpi: int = 300,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> str:
    """
    Load config + first page of input, draw axis-based circles for all layouts,
    write PNG to out_image. Returns the path written.
    """
    cfg: Config = load_config(config_path)
    img = _load_input_image(input_path, dpi=dpi)

    # Answer blocks
    for layout in cfg.answer_layouts:
        _draw_layout_circles(img, layout, color=color, thickness=thickness)

    # Optional blocks: last/first name, ID, version
    for opt in ("last_name_layout", "first_name_layout", "id_layout", "version_layout"):
        lay = getattr(cfg, opt, None)
        if lay is not None:
            _draw_layout_circles(img, lay, color=color, thickness=thickness)

    if not cv2.imwrite(out_image, img):
        raise IOError(f"Failed to write {out_image}")
    return out_image