#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarkShark
io_pages.py
------------
Single source of truth for page IO:

- Render PDF pages to OpenCV-friendly BGR arrays, with a single rendering path
- Load a PDF or image file into a list of BGR pages
- Save a list of BGR pages as a multi-page PDF

Design goals:
- Keep all PDF rendering logic in one module to avoid drift
- Support PyMuPDF (fitz) and pdf2image fallback
- Allow forcing a renderer to keep template and scans rasterized the same way
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Union
import os

import numpy as np  # type: ignore
import cv2  # type: ignore


PdfRenderer = str  # "auto" | "fitz" | "pdf2image"


def _render_pdf_pages_fitz(pdf_path: str, dpi: int) -> List[np.ndarray]:
    import fitz  # type: ignore

    zoom = float(dpi) / 72.0
    doc = fitz.open(pdf_path)
    pages: List[np.ndarray] = []
    try:
        for p in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = img[:, :, ::-1].copy()  # RGB -> BGR
            pages.append(bgr)
    finally:
        doc.close()
    return pages


def _render_pdf_pages_pdf2image(pdf_path: str, dpi: int) -> List[np.ndarray]:
    from pdf2image import convert_from_path  # type: ignore

    pil_pages = convert_from_path(pdf_path, dpi=int(dpi))
    pages: List[np.ndarray] = []
    for pil in pil_pages:
        rgb = np.array(pil.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        pages.append(bgr)
    return pages


def choose_common_pdf_renderer(
    pdf_paths: Sequence[str],
    dpi: int,
    prefer: str = "fitz",
) -> str:
    """
    Choose a single PDF renderer that is likely to work for all PDFs in this run.

    Behavior:
    - Try the preferred renderer (default: fitz) on the first page of every PDF.
    - If any PDF fails with fitz, fall back to pdf2image for all PDFs.
    - If pdf2image is not available either, raise a RuntimeError.

    This reduces template versus scans drift caused by mixing renderers.
    """
    pdf_paths = [p for p in pdf_paths if p]
    if not pdf_paths:
        return "fitz"  # should not happen in normal use

    if prefer not in ("fitz", "pdf2image"):
        raise ValueError(f"Unknown renderer '{prefer}'. Use 'fitz' or 'pdf2image'.")

    if prefer == "pdf2image":
        # Validate availability early
        try:
            _ = _render_pdf_pages_pdf2image(pdf_paths[0], dpi)
            return "pdf2image"
        except Exception as e:
            raise RuntimeError(
                f"pdf2image rendering failed for '{pdf_paths[0]}'. Original error: {e}"
            )

    # prefer == "fitz"
    try:
        import fitz  # type: ignore
        _ = fitz  # quiet type checkers
    except Exception:
        # fitz missing, fall back
        try:
            _ = _render_pdf_pages_pdf2image(pdf_paths[0], dpi)
            return "pdf2image"
        except Exception as e:
            raise RuntimeError(
                "Cannot render PDFs. Install PyMuPDF (fitz) or pdf2image plus Poppler. "
                f"Original error: {e}"
            )

    # Try rendering first page of each PDF with fitz
    try:
        import fitz  # type: ignore
        zoom = float(dpi) / 72.0
        for pdf_path in pdf_paths:
            doc = fitz.open(pdf_path)
            try:
                if doc.page_count < 1:
                    raise RuntimeError(f"PDF has no pages: {path}")
                page0 = doc.load_page(0)
                pix = page0.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                _ = pix.samples  # force materialization
            finally:
                doc.close()
        return "fitz"
    except Exception:
        # Fall back to pdf2image for all PDFs
        try:
            _ = _render_pdf_pages_pdf2image(pdf_paths[0], dpi)
            return "pdf2image"
        except Exception as e:
            raise RuntimeError(
                "Cannot render PDFs. PyMuPDF (fitz) failed for at least one file, and pdf2image failed too. "
                f"Original error: {e}"
            )


def render_pdf_to_bgr_pages(
    pdf_path: str,
    dpi: int,
    renderer: PdfRenderer = "auto",
) -> List[np.ndarray]:
    """Render a PDF to a list of BGR pages."""
    if renderer == "auto":
        renderer = choose_common_pdf_renderer([pdf_path], dpi=dpi, prefer="fitz")

    if renderer == "fitz":
        try:
            return _render_pdf_pages_fitz(pdf_path, dpi)
        except Exception as e:
            raise RuntimeError(
                f"Cannot render PDF '{pdf_path}' with fitz. Original error: {e}"
            )
    if renderer == "pdf2image":
        try:
            return _render_pdf_pages_pdf2image(pdf_path, dpi)
        except Exception as e:
            raise RuntimeError(
                f"Cannot render PDF '{pdf_path}' with pdf2image. Original error: {e}"
            )

    raise ValueError(f"Unknown renderer '{renderer}'. Use 'auto', 'fitz', or 'pdf2image'.")


def iter_pdf_bgr_tuples(
    pdf_path: str,
    dpi: int,
    renderer: PdfRenderer = "auto",
) -> Iterator[Tuple[str, int, np.ndarray]]:
    """
    Yield (pdf_path, 1-based page_index, bgr_image) for each page in a PDF.
    """
    pages = render_pdf_to_bgr_pages(pdf_path, dpi=dpi, renderer=renderer)
    for idx, bgr in enumerate(pages, 1):
        yield (pdf_path, idx, bgr)


def convert_pdf_pages_to_bgr_tuples(
    inputs: Union[str, Sequence[str]],
    dpi: int,
    renderer: PdfRenderer = "auto",
) -> List[Tuple[str, int, np.ndarray]]:
    """
    Normalize inputs (single path or list of paths), enforce PDF only,
    and return a list of (source_path, page_number_1_based, bgr_image).
    """
    if isinstance(inputs, str):
        input_files = [inputs]
    else:
        input_files = list(inputs)

    out: List[Tuple[str, int, np.ndarray]] = []
    for path in input_files:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".pdf":
            raise ValueError(f"Only PDF input is supported here, got: {path}")
        for triple in iter_pdf_bgr_tuples(path, dpi=dpi, renderer=renderer):
            out.append(triple)

    if not out:
        raise RuntimeError("No input pages found.")
    return out


def _load_image_to_bgr(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def load_pages(path: str, dpi: int, renderer: PdfRenderer = "auto") -> List[np.ndarray]:
    """
    If path is a PDF, returns a list of BGR pages.
    If path is an image, returns [single BGR image].
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return render_pdf_to_bgr_pages(path, dpi=dpi, renderer=renderer)

    img = _load_image_to_bgr(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return [img]


class PdfPageWriter:
    """
    Stream pages into a PDF using PyMuPDF (fitz), embedding each page image as a JPEG.

    This keeps memory usage low (no need to hold all pages) and produces much smaller
    PDFs than embedding lossless PNGs.
    """

    def __init__(self, out_path: str, dpi: int, jpeg_quality: int = 85) -> None:
        self.out_path = out_path
        self.dpi = int(dpi)
        self.jpeg_quality = int(jpeg_quality)
        import fitz  # type: ignore
        self._fitz = fitz
        self._doc = fitz.open()
        self._closed = False

    def add_page(self, page_bgr: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("PdfPageWriter is closed.")
        if page_bgr is None or page_bgr.size == 0:
            raise ValueError("Empty page image.")

        # Encode as JPEG to reduce PDF size
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        ok, buf = cv2.imencode(".jpg", page_bgr, encode_params)
        if not ok:
            raise RuntimeError("cv2.imencode('.jpg', page_bgr) failed")

        h, w = page_bgr.shape[:2]
        page_w_pt = w * 72.0 / float(self.dpi)
        page_h_pt = h * 72.0 / float(self.dpi)
        page = self._doc.new_page(width=page_w_pt, height=page_h_pt)
        page.insert_image(page.rect, stream=buf.tobytes())

    def close(self, save: bool = True) -> None:
        if self._closed:
            return
        try:
            if save:
                # deflate + garbage cleanup can shrink the output further
                self._doc.save(self.out_path, deflate=True, garbage=4)
        finally:
            self._doc.close()
            self._closed = True

    def __enter__(self) -> "PdfPageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If an exception occurred, avoid writing a partial PDF.
        self.close(save=(exc_type is None))


def save_images_as_pdf(
    pages_bgr: List[np.ndarray],
    out_path: str,
    dpi: int,
    quality: int = 85
) -> None:
    """
    Save a list of BGR arrays to a single multi-page PDF using PyMuPDF with compression.

    Args:
        pages_bgr: List of BGR numpy arrays (one per page)
        out_path: Output PDF path
        dpi: Resolution for PDF metadata
        quality: JPEG quality for compression (1-100, higher=better quality, larger file)
                 Default 85 provides good balance. Use 95 for high quality, 75 for smaller files.

    Note: This replaces the old Pillow-based implementation with PyMuPDF for better compression.
          Typical file size reduction: 50-70% at quality=85 compared to uncompressed.
    """
    import fitz  # type: ignore
    import tempfile

    if not pages_bgr:
        raise ValueError("No pages to save.")

    # Create a new PDF document
    doc = fitz.open()

    try:
        for page_idx, bgr_img in enumerate(pages_bgr):
            # Convert BGR to RGB
            rgb_img = bgr_img[:, :, ::-1].astype("uint8")

            # Get image dimensions
            height, width = rgb_img.shape[:2]

            # Calculate page size in points (72 points = 1 inch)
            page_width_pts = (width / dpi) * 72
            page_height_pts = (height / dpi) * 72

            # Create a new page with correct dimensions
            page = doc.new_page(width=page_width_pts, height=page_height_pts)

            # Save image to temporary file for PyMuPDF insertion
            # (PyMuPDF can insert from bytes, but file is more reliable for compression)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                # Use cv2 to write JPEG with specified quality
                cv2.imwrite(tmp_path, bgr_img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            try:
                # Insert image into page (fills entire page)
                page.insert_image(
                    page.rect,
                    filename=tmp_path,
                    keep_proportion=True,
                    overlay=True
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        # Save with compression options
        # deflate=True enables general compression, garbage=4 is aggressive cleanup
        doc.save(
            out_path,
            garbage=4,       # Remove unused objects (0-4, higher=more aggressive)
            deflate=True,    # Compress streams
            clean=True,      # Clean up content streams
        )
    finally:
        doc.close()
