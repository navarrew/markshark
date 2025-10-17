#!/usr/bin/env python3
"""
zone_visualizer.py
------------------
Axis-based overlay visualizer for MarkShark sheets.

It reads the same config used by the scorer (see config_io.py), computes bubble
centers from top-left and bottom-right reference centers, and draws *circles only*
at those centers with the configured radius.

CLI examples:
  python zone_visualizer.py --config 64_question_config.yaml 64_question_template.pdf
  python zone_visualizer.py --config 64_question_config.yaml page1.png page2.png --out-dir overlays

Notes:
- Requires: opencv-python, numpy, pyyaml, pdf2image (if you input PDFs).
- This is strictly a visualization helper; it does not grade.
"""

from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Iterable

import numpy as np
import cv2

# Local import within the tools package
from ..config_io import load_config, Config, GridLayout

# ------------------------------------------------------------------------------
# Geometry (identical logic to scorer)
# ------------------------------------------------------------------------------

def grid_centers_axis_mode(
    x_tl: float, y_tl: float, x_br: float, y_br: float,
    questions: int, choices: int
) -> List[Tuple[float, float]]:
    """
    Return normalized (x%, y%) centers for a questions x choices grid
    by interpolating between top-left and bottom-right bubble centers.
    """
    centers: List[Tuple[float, float]] = []
    q_den = max(1, questions - 1)
    c_den = max(1, choices - 1)
    for r in range(questions):
        v = r / q_den
        y = y_tl + (y_br - y_tl) * v
        for c in range(choices):
            u = c / c_den
            x = x_tl + (x_br - x_tl) * u
            centers.append((x, y))
    return centers


def centers_to_radius_px(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int, img_h: int,
    radius_pct: float
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Convert normalized centers to pixel centers, and return a pixel radius
    computed from radius_pct (fraction of image width).
    """
    r_px = max(1, int(round(radius_pct * img_w)))
    pts_px: List[Tuple[int, int]] = []
    for (cxp, cyp) in centers_pct:
        cx = int(round(float(cxp) * img_w))
        cy = int(round(float(cyp) * img_h))
        # clamp to image
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        pts_px.append((cx, cy))
    return pts_px, r_px


# ------------------------------------------------------------------------------
# I/O helpers: PDF/images
# ------------------------------------------------------------------------------

def load_pages(paths: List[str], dpi: int = 300) -> List[np.ndarray]:
    """
    Load pages from input files. For PDFs, uses pdf2image; for images, read with OpenCV.
    Returns a list of BGR np.ndarrays.
    """
    images: List[np.ndarray] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in (".pdf",):
            try:
                from pdf2image import convert_from_path
            except Exception as e:
                raise RuntimeError("pdf2image is required to read PDFs. Install with `pip install pdf2image`") from e
            pil_pages = convert_from_path(p, dpi=dpi)
            for pg in pil_pages:
                rgb = np.array(pg)  # HxWx3 RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                images.append(bgr)
        else:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p}")
            images.append(img)
    return images


# ------------------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------------------

def draw_layout_circles(
    img_bgr: np.ndarray,
    layout: GridLayout,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw circles at all bubble centers for a single layout.
    Returns the modified image (in-place safe).
    """
    h, w = img_bgr.shape[:2]
    centers = grid_centers_axis_mode(
        layout.x_topleft, layout.y_topleft,
        layout.x_bottomright, layout.y_bottomright,
        layout.questions, layout.choices
    )
    pts_px, r_px = centers_to_radius_px(centers, w, h, layout.radius_pct)

    for (cx, cy) in pts_px:
        cv2.circle(img_bgr, (cx, cy), r_px, color, thickness, lineType=cv2.LINE_AA)

    return img_bgr


def visualize_page(
    img_bgr: np.ndarray,
    cfg: Config,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw circles for all answer_layouts onto a copy of the page image and return it.
    """
    canvas = img_bgr.copy()
    for layout in cfg.answer_layouts:
        canvas = draw_layout_circles(canvas, layout, color=color, thickness=thickness)
    # Optional: add others (id_layout, names, version) once defined in new format.
    for opt_name in ("id_layout", "last_name_layout", "first_name_layout", "version_layout"):
        lay = getattr(cfg, opt_name, None)
        if lay is not None:
            canvas = draw_layout_circles(canvas, lay, color=color, thickness=thickness)
    return canvas


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize bubble centers (circles only) using axis-based geometry.")
    ap.add_argument("--config", required=True, help="YAML config path (axis-mode fields).")
    ap.add_argument("--out-dir", default="overlays", help="Directory to write overlay PNGs.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI when rasterizing PDFs.")
    ap.add_argument("--thickness", type=int, default=2, help="Circle outline thickness (pixels).")
    ap.add_argument("--color", default="0,255,0",
                    help="B,G,R color for circles (e.g., '0,255,0' for green).")
    ap.add_argument("inputs", nargs="+", help="PDF(s) or image file(s) to visualize.")
    args = ap.parse_args()

    # Parse color
    try:
        bgr = tuple(int(v) for v in args.color.split(","))
        if len(bgr) != 3:
            raise ValueError
        color = (bgr[0], bgr[1], bgr[2])
    except Exception:
        raise ValueError("--color must be 'B,G,R' like '0,255,0'")

    cfg = load_config(args.config)
    pages = load_pages(args.inputs, dpi=args.dpi)

    os.makedirs(args.out_dir, exist_ok=True)

    for i, img in enumerate(pages):
        overlay = visualize_page(img, cfg, color=color, thickness=args.thickness)
        out_path = os.path.join(args.out_dir, f"overlay_page_{i+1}.png")
        ok = cv2.imwrite(out_path, overlay)
        if not ok:
            raise IOError(f"Failed to write {out_path}")
        print(f"[OK] Wrote {out_path}")

    print(f"[DONE] {len(pages)} page(s) processed. Overlays in: {args.out_dir}")


if __name__ == "__main__":
    main()