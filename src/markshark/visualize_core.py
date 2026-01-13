# src/markshark/visualize_core.py
#!/usr/bin/env python3
"""
MarkShark
visualize_core.py
Visualize OMR bubble positions from an axis-based bubblemap.

Exports:
  - overlay_bublmap(bublmap_path, input_path, out_image, dpi=300, color=(0,255,0), thickness=2, pdf_renderer="auto") -> str

Notes:
  - If input_path is a PDF, the first page is used.
  - If out_image ends with .pdf, a single-page PDF is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .tools.bubblemap_io import load_bublmap, Bubblemap
from .tools.visualizer_tools import draw_layout_circles
from .tools import io_pages as IO


def _load_input_image(path: str, dpi: int = 300, pdf_renderer: str = "auto") -> np.ndarray:
    """Return a BGR image for either a PDF (first page) or a raster image."""
    pages = IO.load_pages(path, dpi=dpi, renderer=pdf_renderer)
    if not pages:
        raise ValueError(f"No pages found in input: {path}")
    return pages[0]


def overlay_bublmap(
    bublmap_path: str,
    input_path: str,
    out_image: str,
    dpi: int = 300,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    pdf_renderer: str = "auto",  #options: 'auto', 'fitz', or 'pdf2image'
) -> str:
    """
    Load an axis-mode YAML bublmap and draw all bubble circles on the input image/PDF.

    Returns:
        out_image (string path), after writing.
        
    
    """
    bmap: Config = load_bublmap(bublmap_path)
    img = _load_input_image(input_path, dpi=dpi, pdf_renderer=pdf_renderer)

    # Answer blocks
    for layout in bmap.answer_layouts:
        draw_layout_circles(img, layout, color=color, thickness=thickness)

    # Optional blocks: last/first name, ID, version
    for opt in ("last_name_layout", "first_name_layout", "id_layout", "version_layout"):
        lay = getattr(bmap, opt, None)
        if lay is not None:
            draw_layout_circles(img, lay, color=color, thickness=thickness)

    out_p = Path(out_image)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if out_p.suffix.lower() == ".pdf":
        IO.save_images_as_pdf([img], str(out_p), dpi=dpi)
        return str(out_p)

    if not cv2.imwrite(str(out_p), img):
        raise IOError(f"Failed to write {out_p}")
    return str(out_p)
