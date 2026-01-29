# src/markshark/mapviewer_core.py
#!/usr/bin/env python3
"""
MarkShark
mapviewer_core.py
Visualize OMR bubble positions from an axis-based bubblemap (Map Viewer).
NOW WITH MULTI-PAGE SUPPORT!

Exports:
  - overlay_bublmap(bublmap_path, input_path, out_image, dpi=300, color=(0,255,0), thickness=2, pdf_renderer="auto") -> str

Notes:
  - If input_path is a PDF, all pages matching the bubblemap are used.
  - For multi-page bubblemaps, overlays each page's layouts on the corresponding template page.
  - If out_image ends with .pdf, a multi-page PDF is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .tools.bubblemap_io import load_bublmap, Bubblemap
from .tools.visualizer_tools import draw_layout_circles
from .tools import io_pages as IO
from .defaults import RENDER_DEFAULTS


def _load_input_image(path: str, dpi: int = 300, pdf_renderer: str = "auto") -> np.ndarray:
    """Return a BGR image for either a PDF (first page) or a raster image."""
    pages = IO.load_pages(path, dpi=dpi, renderer=pdf_renderer)
    if not pages:
        raise ValueError(f"No pages found in input: {path}")
    return pages[0]


def _load_input_images(path: str, dpi: int = 300, pdf_renderer: str = "auto") -> list[np.ndarray]:
    """Return all BGR images from a PDF or a single raster image."""
    pages = IO.load_pages(path, dpi=dpi, renderer=pdf_renderer)
    if not pages:
        raise ValueError(f"No pages found in input: {path}")
    return pages


def overlay_bublmap(
    bublmap_path: str,
    input_path: str,
    out_image: str,
    dpi: int = RENDER_DEFAULTS.dpi,
    pdf_quality: int = RENDER_DEFAULTS.pdf_quality,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    pdf_renderer: str = "auto",  # options: 'auto', 'fitz', or 'pdf2image'
) -> str:
    """
    Load an axis-mode YAML bublmap and draw all bubble circles on the input image/PDF.
    
    NEW: Multi-page support!
    - For single-page bubblemaps: overlays on first page of input (legacy behavior)
    - For multi-page bubblemaps: overlays each page's layouts on corresponding template page

    Returns:
        out_image (string path), after writing.
    """
    bmap: Bubblemap = load_bublmap(bublmap_path)
    
    # Check if this is a multi-page bubblemap
    num_pages = bmap.num_pages
    
    if num_pages == 1:
        # Legacy single-page behavior
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
            IO.save_images_as_pdf([img], str(out_p), dpi=dpi, quality=pdf_quality)
            return str(out_p)

        if not cv2.imwrite(str(out_p), img):
            raise IOError(f"Failed to write {out_p}")
        return str(out_p)
    
    else:
        # Multi-page mode
        print(f"[info] Multi-page bubblemap detected: {num_pages} pages")
        
        # Load all template pages
        template_pages = _load_input_images(input_path, dpi=dpi, pdf_renderer=pdf_renderer)
        
        if len(template_pages) < num_pages:
            raise ValueError(
                f"Bubblemap has {num_pages} pages but template PDF only has {len(template_pages)} page(s). "
                f"Template must have at least {num_pages} page(s)."
            )
        
        # Process each page
        overlaid_pages = []
        for page_num in range(1, num_pages + 1):
            print(f"[info] Overlaying bubblemap page {page_num} on template page {page_num}")
            
            # Get the corresponding template page
            img = template_pages[page_num - 1].copy()  # 0-indexed
            
            # Get layouts for this page
            page_layout = bmap.get_page(page_num)
            
            if page_layout is None:
                raise ValueError(f"Could not get layout for page {page_num}")
            
            # Draw answer layouts for this page
            for layout in page_layout.answer_layouts:
                draw_layout_circles(img, layout, color=color, thickness=thickness)
            
            # Draw optional blocks if present on this page
            if page_layout.last_name_layout is not None:
                draw_layout_circles(img, page_layout.last_name_layout, color=color, thickness=thickness)
            if page_layout.first_name_layout is not None:
                draw_layout_circles(img, page_layout.first_name_layout, color=color, thickness=thickness)
            if page_layout.id_layout is not None:
                draw_layout_circles(img, page_layout.id_layout, color=color, thickness=thickness)
            if page_layout.version_layout is not None:
                draw_layout_circles(img, page_layout.version_layout, color=color, thickness=thickness)
            
            overlaid_pages.append(img)
        
        # Save output
        out_p = Path(out_image)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if out_p.suffix.lower() == ".pdf":
            IO.save_images_as_pdf(overlaid_pages, str(out_p), dpi=dpi, quality=pdf_quality)
            print(f"[info] Saved {num_pages}-page visualization to {out_p}")
            return str(out_p)
        
        # For non-PDF output with multi-page, save only the first page (with warning)
        print(f"[warning] Multi-page bubblemap but output is not PDF. Saving only page 1 to {out_p}")
        if not cv2.imwrite(str(out_p), overlaid_pages[0]):
            raise IOError(f"Failed to write {out_p}")
        return str(out_p)
