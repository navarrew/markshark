import os
import csv
import sys
from types import SimpleNamespace
from typing import Optional, List, Tuple, Sequence, Union, Iterator

import numpy as np  # type: ignore
import cv2  # type: ignore

from .defaults import (
    FEAT_DEFAULTS, MATCH_DEFAULTS, EST_DEFAULTS, ALIGN_DEFAULTS, RENDER_DEFAULTS,
    apply_feat_overrides, apply_est_overrides,
)
from .tools import align_tools as SA 

"""
Convert a set of PDFs to a list of BGR images; align against a template.
"""

# ---------- low-level: PDF → BGR pages (generator) ----------

"""
_iter_pdf_bgr_pages
A generator that rasterizes a PDF into BGR images page-by-page. 
It first tries PyMuPDF (fitz) with a zoom set by dpi/72, and if that fails,
falls back to pdf2image (which requires Poppler). 
It yields (source_path, 1-based page_index, bgr_image) for each page 
and raises a RuntimeError if neither renderer works.
"""

def _iter_pdf_bgr_pages(pdf_path: str, dpi: int) -> Iterator[Tuple[str, int, np.ndarray]]:
    """
    Yield (pdf_path, 1-based page_index, bgr_image) for each page in the PDF.
    Tries PyMuPDF first; falls back to pdf2image if needed.
    """
    # Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore
        zoom = dpi / 72.0
        doc = fitz.open(pdf_path)
        try:
            for idx, p in enumerate(doc, 1):
                mat = fitz.Matrix(zoom, zoom)
                pix = p.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                bgr = img[:, :, ::-1].copy()  # RGB -> BGR
                yield (pdf_path, idx, bgr)
            return
        finally:
            doc.close()
    except Exception:
        pass

    # Fallback: pdf2image
    try:
        from pdf2image import convert_from_path  # type: ignore
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        for idx, pil in enumerate(pil_pages, 1):
            rgb = np.array(pil.convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
            yield (pdf_path, idx, bgr)
    except Exception as e:
        raise RuntimeError(
            f"Cannot render PDF '{pdf_path}'. Install PyMuPDF (fitz) or pdf2image+poppler. Original error: {e}"
        )

# ---------- high-level: inputs → list of (path, page#, bgr) ----------

"""
convert_pdf_pages_to_bgr_tuples
(inputs: Union[str, Sequence[str]], dpi: int) -> List[Tuple[str, int, np.ndarray]]
Normalizes inputs (single path or list of paths), enforces “PDF only,” 
and uses _iter_pdf_bgr_pages to collect all pages into 
a list of (source_path, page_number, bgr_image) tuples. 
If nothing is produced, it raises RuntimeError("No input pages found.").
"""

def convert_pdf_pages_to_bgr_tuples(
    inputs: Union[str, Sequence[str]],
    dpi: int,
) -> List[Tuple[str, int, np.ndarray]]:
    """
    Accepts a single path or a list of paths. For now, only PDFs are supported.
    Returns a list of (source_path, page_number, bgr_image) tuples.
    """
    if isinstance(inputs, str):
        input_files = [inputs]
    else:
        input_files = list(inputs)

    out: List[Tuple[str, int, np.ndarray]] = []
    for path in input_files:
        ext = os.path.splitext(path)[1].lower()
        if ext != ".pdf":
            raise ValueError(f"Only PDFs supported for now (got: {path})")
        for triple in _iter_pdf_bgr_pages(path, dpi=dpi):
            out.append(triple)
    if not out:
        raise RuntimeError("No input pages found.")
    return out

"""
save_images_as_pdf
Writes a list of BGR numpy images to a single multi-page PDF using PIL. 
It converts BGR→RGB for each image, saves the first page, appends the rest, 
sets the PDF “resolution” metadata to dpi, and prints a confirmation line. 
Raises ValueError if the list is empty.
"""

def save_images_as_pdf(pages_bgr: List[np.ndarray], out_path: str, dpi: int = 150) -> None:
    """
    Save a list of BGR NumPy arrays to a single PDF at the requested DPI.
    """
    from PIL import Image
    if not pages_bgr:
        raise ValueError("No pages to save.")
    pil_pages = [Image.fromarray(p[:, :, ::-1].astype("uint8")).convert("RGB") for p in pages_bgr]  # BGR->RGB
    first, rest = pil_pages[0], pil_pages[1:]
    first.save(out_path, save_all=True, append_images=rest, format="PDF", resolution=dpi)
    print(f"[ok] wrote PDF: {out_path}")



"""
align_pdf_scans
a high-level convenience wrapper
    1.  Builds an args SimpleNamespace that bundles rendering params 
        (DPI, page range, outputs), ArUco options, feature detection/matching settings 
        (tiles, ORB, ratio test, FLANN), and geometric estimation/ECC settings 
        (method, RANSAC, iters, confidence, ECC details).
    2.  Loads the template PDF at the requested dpi, 
        selects template_page (1-based), and converts 
        the input_pdf to raw page images.
    3.  Calls align_raw_bgr_scans(...) to align each scan to the 
        template and collect metrics.
    4.  If alignment produced pages and an output filename is set, writes 
        an aligned PDF via save_images_as_pdf.
Returns the output PDF path. (Note: fallback_original is forwarded in args for
downstream use; this function itself doesn’t branch on it.)
"""

def align_pdf_scans(
    input_pdf: str,
    template: str,
    out_pdf: str = "aligned_scans.pdf",
    dpi: int = RENDER_DEFAULTS.dpi,
    template_page: int = 1,
    method: str = EST_DEFAULTS.method,
    dict_name: str = ALIGN_DEFAULTS.dict_name,
    min_markers: int = ALIGN_DEFAULTS.min_aruco,
    ransac: float = EST_DEFAULTS.ransac_thresh,
    orb_nfeatures: int = FEAT_DEFAULTS.orb_nfeatures,
    match_ratio: float = MATCH_DEFAULTS.ratio_test,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    save_debug: Optional[str] = None,
    fallback_original: bool = True,
) -> str:
    # Build args expected by the lower-level aligner
    args = SimpleNamespace(
        # IO / high-level
        dpi=int(dpi) if dpi else RENDER_DEFAULTS.dpi,
        first_page=first_page, last_page=last_page,
        out=out_pdf, out_dir=None, prefix="aligned",
        # ArUco / align toggles
        dict_name=dict_name, min_markers=min_markers, use_aruco=ALIGN_DEFAULTS.use_aruco,
        # Feature detection
        tiles_x=FEAT_DEFAULTS.tiles_x, tiles_y=FEAT_DEFAULTS.tiles_y,
        topk_per_tile=FEAT_DEFAULTS.topk_per_tile,
        orb_nfeatures=orb_nfeatures or FEAT_DEFAULTS.orb_nfeatures,
        orb_fast_threshold=FEAT_DEFAULTS.orb_fast_threshold,
        # Matching
        match_ratio=match_ratio or MATCH_DEFAULTS.ratio_test,
        mutual_check=MATCH_DEFAULTS.mutual_check,
        max_matches=MATCH_DEFAULTS.max_matches,
        use_flann=MATCH_DEFAULTS.use_flann,
        # Estimation / ECC
        method=method or EST_DEFAULTS.method,
        ransac=ransac or EST_DEFAULTS.ransac_thresh,
        ransac_thresh=ransac or EST_DEFAULTS.ransac_thresh,  # alias for downstream
        max_iters=EST_DEFAULTS.max_iters,
        confidence=EST_DEFAULTS.confidence,
        use_ecc=getattr(EST_DEFAULTS, 'use_ecc', True),
        ecc_levels=getattr(EST_DEFAULTS, 'ecc_levels', 4),
        ecc_max_iters=getattr(EST_DEFAULTS, 'ecc_max_iters', 50),
        ecc_eps=getattr(EST_DEFAULTS, 'ecc_eps', 1e-6),
        # Guard thresholds
        fail_med=ALIGN_DEFAULTS.fail_med, fail_p95=ALIGN_DEFAULTS.fail_p95, fail_br=ALIGN_DEFAULTS.fail_br,
        # Misc
        save_debug=save_debug, fallback_original=fallback_original,
    )

    # Load template page as BGR image (1-based page index)
    tpl_tuples = convert_pdf_pages_to_bgr_tuples(template, dpi=dpi)
    if not (1 <= template_page <= len(tpl_tuples)):
        template_page = 1
    template_bgr = tpl_tuples[template_page - 1][2]  # (path, idx, bgr) → bgr

    # Convert raw scans to (src_path, page_idx, bgr)
    raw_scans_bgr = convert_pdf_pages_to_bgr_tuples(input_pdf, dpi=dpi)

    # Align
    aligned_pages, metrics_rows = align_raw_bgr_scans(raw_scans_bgr, template_bgr, args)

    # Write outputs
    if args.out and aligned_pages:
        save_images_as_pdf(aligned_pages, args.out, dpi=args.dpi)

    # Optional: write metrics CSV here if desired (similar to below)
    return out_pdf


"""
align_raw_bgr_scans
Core alignment routine. 
It prepares feature (fpar) and estimation (epar) parameter packs
via apply_feat_overrides/apply_est_overrides. 
Then it runs in up to two passes:
    (1) Optional ArUco pass (if args.use_aruco): for each page, 
        tries SA.align_with_aruco(...) using the specified dictionary 
        and minimum marker count. Successful pages are added to outputs
        immediately (optionally written as PNGs if out_dir is set) and
        logged with “aruco” mode and inlier counts; unsuccessful ones 
        fall back to the next pass.
    (2) Feature-based pass with guardrails: for remaining pages, calls 
        SA.align_with_guardrails(template_bgr, scan_bgr, fpar, epar,
        args.match_ratio, args.fail_med, args.fail_p95, args.fail_br).
        On success, it appends the warped image, optionally writes a PNG,
        and records per-page metrics (median residual, p95, border-region p95, inliers)
        formatted to three decimals (or “nan”). Errors are caught and
        logged without aborting the whole batch.
Returns the list of aligned page images and a list of per-page metrics dictionaries.
"""

def align_raw_bgr_scans(
    raw_scans_bgr: List[Tuple[str, int, np.ndarray]],
    template_bgr: np.ndarray,
    args: SimpleNamespace,
) -> Tuple[List[np.ndarray], List[dict]]:
    aligned_pages: List[np.ndarray] = []
    metrics_rows: List[dict] = []

    # Pre-build base params for feature pass
    fpar = apply_feat_overrides(
        tiles_x=args.tiles_x, tiles_y=args.tiles_y,
        topk_per_tile=args.topk_per_tile,
        orb_nfeatures=args.orb_nfeatures,
        orb_fast_threshold=args.orb_fast_threshold,
    )
    epar = apply_est_overrides(
        method=args.method, ransac_thresh=args.ransac_thresh,
        max_iters=args.max_iters, confidence=args.confidence,
        use_ecc=args.use_ecc, ecc_levels=args.ecc_levels,
        ecc_max_iters=args.ecc_max_iters, ecc_eps=args.ecc_eps,
    )

    # 1) Optional ArUco pass
    remaining_pages = raw_scans_bgr
    if getattr(args, "use_aruco", False):
        print("[info] ArUco-based alignment enabled.", file=sys.stderr)
        non_aruco_pages = []
        for (src_path, page_idx, scan_bgr) in raw_scans_bgr:
            try:
                warped_a, H_a, detected_ids = SA.align_with_aruco(
                    scan_bgr, template_bgr,
                    dict_name=args.dict_name,
                    min_markers=args.min_markers,
                    ransac=args.ransac_thresh,
                )
                if warped_a is not None and H_a is not None and len(detected_ids) >= args.min_markers:
                    metrics = {
                        "mode": "aruco", "n_inliers": len(detected_ids),
                        "res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"),
                    }
                    aligned_pages.append(warped_a)
                    if getattr(args, "out_dir", None):
                        base = os.path.splitext(os.path.basename(src_path))[0]
                        fn = os.path.join(args.out_dir, f"{args.prefix}_{base}_p{page_idx:03d}.png")
                        cv2.imwrite(fn, warped_a)
                    metrics_rows.append({
                        "source": os.path.basename(src_path),
                        "page": page_idx,
                        "mode": metrics["mode"],
                        "res_median": "nan", "res_p95": "nan", "res_br_p95": "nan",
                        "n_inliers": metrics["n_inliers"],
                    })
                    print(f"[ok] {src_path} p{page_idx}: ArUco mode with {len(detected_ids)} markers detected.")
                else:
                    print(f"[info] {src_path} p{page_idx}: ArUco alignment insufficient. Falling back.", file=sys.stderr)
                    non_aruco_pages.append((src_path, page_idx, scan_bgr))
            except Exception as e:
                print(f"[info] {src_path} p{page_idx}: ArUco alignment error: {e}. Falling back.", file=sys.stderr)
                non_aruco_pages.append((src_path, page_idx, scan_bgr))
        remaining_pages = non_aruco_pages
        if not remaining_pages:
            print("[info] All pages processed with ArUco alignment.", file=sys.stderr)
    else:
        print("[info] ArUco-based alignment not enabled.", file=sys.stderr)

    # 2) Feature-based pass with guardrails
    for (src_path, page_idx, scan_bgr) in remaining_pages:
        try:
            # Depending on your SA.align_with_guardrails signature, adjust the args list:
            warped, metrics = SA.align_with_guardrails(
                template_bgr, scan_bgr, fpar, epar, args.match_ratio,
                args.fail_med, args.fail_p95, args.fail_br
            )
            aligned_pages.append(warped)
            if getattr(args, "out_dir", None):
                base = os.path.splitext(os.path.basename(src_path))[0]
                fn = os.path.join(args.out_dir, f"{args.prefix}_{base}_p{page_idx:03d}.png")
                cv2.imwrite(fn, warped)
            row = {
                "source": os.path.basename(src_path),
                "page": page_idx,
                "mode": metrics.get("mode", "base"),
                "res_median": f"{metrics['res_median']:.3f}" if metrics.get('res_median') == metrics.get('res_median') else "nan",
                "res_p95": f"{metrics['res_p95']:.3f}" if metrics.get('res_p95') == metrics.get('res_p95') else "nan",
                "res_br_p95": f"{metrics['res_br_p95']:.3f}" if metrics.get('res_br_p95') == metrics.get('res_br_p95') else "nan",
                "n_inliers": metrics.get("n_inliers", ""),
            }
            metrics_rows.append(row)
            print(f"[ok] {src_path} p{page_idx}: mode={row['mode']} median={row['res_median']} p95={row['res_p95']} br_p95={row['res_br_p95']}")
        except Exception as e:
            print(f"[error] {src_path} p{page_idx}: {e}", file=sys.stderr)

    return aligned_pages, metrics_rows