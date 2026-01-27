#!/usr/bin/env python3
"""
MarkShark
align_core.py  — MarkShark bubblesheet alignment engine with MULTI-PAGE support
"""

import os
import sys
from types import SimpleNamespace
from typing import Optional, List, Tuple, Any

import numpy as np  # type: ignore
import cv2  # type: ignore

from .defaults import (
    FEAT_DEFAULTS, MATCH_DEFAULTS, EST_DEFAULTS, ALIGN_DEFAULTS, RENDER_DEFAULTS,
    apply_feat_overrides, apply_est_overrides,
)
from .tools import align_tools as SA
from .tools import io_pages as IO

"""
Convert a set of PDFs to a list of BGR images; align against a template.
Now supports multi-page templates!
"""

def align_pdf_scans(
    input_pdf: str,
    template: str,
    out_pdf: str = "aligned_scans.pdf",
    dpi: int = RENDER_DEFAULTS.dpi,
    pdf_quality: int = RENDER_DEFAULTS.pdf_quality,
    pdf_renderer: str = "auto",
    template_page: int = 1,  # Deprecated for multi-page, kept for backward compat
    estimator_method: str = EST_DEFAULTS.estimator_method,
    align_method: str = "auto",
    dict_name: str = ALIGN_DEFAULTS.dict_name,
    min_markers: int = ALIGN_DEFAULTS.min_aruco,
    ransac: float = EST_DEFAULTS.ransac_thresh,
    use_ecc: bool = EST_DEFAULTS.use_ecc,
    ecc_max_iters: int = EST_DEFAULTS.ecc_max_iters,
    ecc_eps: float = EST_DEFAULTS.ecc_eps,
    orb_nfeatures: int = FEAT_DEFAULTS.orb_nfeatures,
    match_ratio: float = MATCH_DEFAULTS.ratio_test,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
    save_debug: Optional[str] = None,
    fallback_original: bool = True,
    bubblemap: Any = None,  # NEW: Optional Bubblemap object for bubble grid alignment
) -> str:
    """
    Align scanned bubble sheets to a template.
    
    NEW: Multi-page template support!
    - Template can be 1-page (legacy) or multi-page PDF
    - For multi-page: scans are aligned in groups
      * Pages 1,3,5... align to template page 1
      * Pages 2,4,6... align to template page 2
      * etc.
    - Input scan pages must be a multiple of template pages
    
    NEW: Bubble grid alignment fallback!
    - If bubblemap is provided, uses known bubble positions for alignment
    - More robust than ORB features for sheets without ArUco markers
    """
    # Build args expected by the lower-level aligner
    args = SimpleNamespace(
        # IO / high-level
        dpi=int(dpi) if dpi else RENDER_DEFAULTS.dpi,
        first_page=first_page, last_page=last_page,
        out=out_pdf, out_dir=None, prefix="aligned",
        # ArUco / align toggles
        dict_name=dict_name, min_markers=min_markers,
        align_method=align_method,
        use_aruco=(align_method in ("auto", "aruco")) and ALIGN_DEFAULTS.use_aruco,
        # Feature detection
        tiles_x=FEAT_DEFAULTS.tiles_x, tiles_y=FEAT_DEFAULTS.tiles_y,
        topk_per_tile=FEAT_DEFAULTS.topk_per_tile,
        orb_nfeatures=orb_nfeatures or FEAT_DEFAULTS.orb_nfeatures,
        orb_fast_threshold=FEAT_DEFAULTS.orb_fast_threshold,
        # Matching
        match_ratio=match_ratio or MATCH_DEFAULTS.ratio_test,
        # Estimation / ECC
        estimator_method=estimator_method or EST_DEFAULTS.estimator_method,
        ransac=ransac or EST_DEFAULTS.ransac_thresh,
        ransac_thresh=ransac or EST_DEFAULTS.ransac_thresh,  # alias for downstream
        max_iters=EST_DEFAULTS.max_iters,
        confidence=EST_DEFAULTS.confidence,
        use_ecc=bool(use_ecc),
        ecc_levels=getattr(EST_DEFAULTS, 'ecc_levels', 4),
        ecc_max_iters=int(ecc_max_iters),
        ecc_eps=float(ecc_eps),
        # Guard thresholds
        fail_med=ALIGN_DEFAULTS.fail_med, fail_p95=ALIGN_DEFAULTS.fail_p95, fail_br=ALIGN_DEFAULTS.fail_br,
        # Misc
        save_debug=save_debug, fallback_original=fallback_original,
        # NEW: Bubblemap for bubble grid alignment
        bubblemap=bubblemap,
    )

    # Choose renderer
    renderer = pdf_renderer
    if renderer == "auto":
        renderer = IO.choose_common_pdf_renderer([template, input_pdf], dpi=int(dpi), prefer="fitz")

    # Load ALL template pages as BGR images
    template_pages_tuples = IO.convert_pdf_pages_to_bgr_tuples(template, dpi=dpi, renderer=renderer)
    num_template_pages = len(template_pages_tuples)
    
    if num_template_pages == 0:
        raise ValueError(f"Template PDF '{template}' has no pages!")
    
    # Extract just the BGR arrays (discard path and index)
    template_bgr_list = [bgr for (path, idx, bgr) in template_pages_tuples]
    
    print(f"[info] Loaded template with {num_template_pages} page(s)", file=sys.stderr)
    if bubblemap is not None:
        print("[info] Bubblemap provided - bubble grid alignment available as fallback", file=sys.stderr)

    # Convert raw scans to (src_path, page_idx, bgr)
    raw_scans_bgr = IO.convert_pdf_pages_to_bgr_tuples(input_pdf, dpi=dpi, renderer=renderer)
    num_scan_pages = len(raw_scans_bgr)
    
    # Multi-page validation: Check that scan pages are a multiple of template pages
    if num_scan_pages % num_template_pages != 0:
        raise ValueError(
            f"ERROR: Input PDF has {num_scan_pages} pages, but template has {num_template_pages} page(s). "
            f"For multi-page templates, the input must have a multiple of {num_template_pages} pages "
            f"(e.g., {num_template_pages}, {num_template_pages*2}, {num_template_pages*3}...). "
            f"You have {num_scan_pages % num_template_pages} extra page(s)."
        )
    
    num_students = num_scan_pages // num_template_pages
    print(f"[info] Processing {num_scan_pages} scan pages as {num_students} student(s) × {num_template_pages} page(s)", file=sys.stderr)

    # Align with multi-page support
    aligned_pages, metrics_rows = align_raw_bgr_scans_multipage(
        raw_scans_bgr, 
        template_bgr_list, 
        args
    )

    # Write outputs
    if args.out and aligned_pages:
        IO.save_images_as_pdf(aligned_pages, args.out, dpi=args.dpi, quality=pdf_quality)

    return out_pdf


"""
align_raw_bgr_scans_multipage
NEW: Multi-page aware alignment routine.

For each scan page, determines which template page to use:
- Scan page 1 → Template page 1
- Scan page 2 → Template page 2
- Scan page 3 → Template page 1
- Scan page 4 → Template page 2
- etc.

Then calls the existing alignment engine (ArUco or feature-based) with the
appropriate template page. Returns aligned pages IN ORIGINAL ORDER.
"""

def align_raw_bgr_scans_multipage(
    raw_scans_bgr: List[Tuple[str, int, np.ndarray]],
    template_bgr_list: List[np.ndarray],
    args: SimpleNamespace,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Align scans using multi-page template support.
    
    Args:
        raw_scans_bgr: List of (src_path, page_idx, bgr_image) tuples
        template_bgr_list: List of template page images [page1_bgr, page2_bgr, ...]
        args: Alignment parameters (now includes optional bubblemap)
        
    Returns:
        (aligned_pages, metrics_rows) in original scan order
    """
    num_template_pages = len(template_bgr_list)
    aligned_pages: List[np.ndarray] = []
    metrics_rows: List[dict] = []

    # Get bubblemap if provided
    bubblemap = getattr(args, 'bubblemap', None)

    # Pre-build base params for feature pass
    fpar = apply_feat_overrides(
        tiles_x=args.tiles_x, tiles_y=args.tiles_y,
        topk_per_tile=args.topk_per_tile,
        orb_nfeatures=args.orb_nfeatures,
        orb_fast_threshold=args.orb_fast_threshold,
    )
    epar = apply_est_overrides(
        estimator_method=args.estimator_method, ransac_thresh=args.ransac_thresh,
        max_iters=args.max_iters, confidence=args.confidence,
        use_ecc=args.use_ecc, ecc_levels=args.ecc_levels,
        ecc_max_iters=args.ecc_max_iters, ecc_eps=args.ecc_eps,
    )

    # Process each scan page with its corresponding template page
    for scan_idx, (src_path, page_idx, scan_bgr) in enumerate(raw_scans_bgr):
        # Determine which template page to use
        # scan_idx 0,1,2,3,4,5... maps to template page 0,1,0,1,0,1... (for 2-page template)
        template_page_idx = scan_idx % num_template_pages
        template_bgr = template_bgr_list[template_page_idx]
        
        # Handle first_page/last_page filters
        skip_this_page = False
        if getattr(args, "first_page", None) is not None and page_idx < int(args.first_page):
            skip_this_page = True
        if getattr(args, "last_page", None) is not None and page_idx > int(args.last_page):
            skip_this_page = True
        
        if skip_this_page:
            # Keep original page if skipping due to filters
            aligned_pages.append(scan_bgr)
            metrics_rows.append({
                "source": os.path.basename(src_path),
                "page": page_idx,
                "template_page": template_page_idx + 1,
                "mode": "skipped",
                "res_median": "nan",
                "res_p95": "nan",
                "res_br_p95": "nan",
                "n_inliers": "",
            })
            print(f"[info] Skipping page {page_idx} (outside first_page/last_page range)", file=sys.stderr)
            continue
        
        print(f"[info] Aligning scan page {page_idx} to template page {template_page_idx + 1}", file=sys.stderr)
        
        # Try ArUco alignment first (if enabled)
        aligned_with_aruco = False
        if getattr(args, "use_aruco", False):
            try:
                warped_a, H_a, detected_ids = SA.align_with_aruco(
                    scan_bgr, template_bgr,
                    dict_name=args.dict_name,
                    min_markers=args.min_markers,
                    ransac=args.ransac_thresh,
                )
                if warped_a is not None and H_a is not None and len(detected_ids) >= args.min_markers:
                    aligned_pages.append(warped_a)
                    if getattr(args, "out_dir", None):
                        base = os.path.splitext(os.path.basename(src_path))[0]
                        fn = os.path.join(args.out_dir, f"{args.prefix}_{base}_p{page_idx:03d}.png")
                        cv2.imwrite(fn, warped_a)
                    metrics_rows.append({
                        "source": os.path.basename(src_path),
                        "page": page_idx,
                        "template_page": template_page_idx + 1,
                        "mode": "aruco",
                        "res_median": "nan", "res_p95": "nan", "res_br_p95": "nan",
                        "n_inliers": len(detected_ids),
                    })
                    print(f"[ok] {src_path} p{page_idx}: ArUco mode with {len(detected_ids)} markers detected.", file=sys.stderr)
                    aligned_with_aruco = True
            except Exception as e:
                print(f"[info] {src_path} p{page_idx}: ArUco alignment error: {e}. Falling back to features.", file=sys.stderr)
        
        # Feature-based alignment (if ArUco didn't work)
        if not aligned_with_aruco:
            # If user requested ArUco-only, use fallback
            if getattr(args, "align_method", "auto") == "aruco":
                if getattr(args, "fallback_original", True):
                    aligned_pages.append(scan_bgr)
                    metrics_rows.append({
                        "source": os.path.basename(src_path),
                        "page": page_idx,
                        "template_page": template_page_idx + 1,
                        "mode": "fallback_original",
                        "res_median": "nan",
                        "res_p95": "nan",
                        "res_br_p95": "nan",
                        "n_inliers": "",
                    })
                    print(f"[info] {src_path} p{page_idx}: kept original (ArUco-only mode).", file=sys.stderr)
            else:
                # Feature-based alignment with guardrails
                # NOW: Pass bubblemap for bubble grid fallback
                # Determine align_mode: "fast" uses coarse-to-fine, "slow" uses full-res ORB
                align_mode = getattr(args, 'align_method', 'auto')
                if align_mode == 'feature':
                    align_mode = 'slow'  # "feature" is legacy name for slow mode
                elif align_mode not in ('fast', 'slow', 'auto'):
                    align_mode = 'auto'
                
                try:
                    warped, metrics = SA.align_with_guardrails(
                        template_bgr, scan_bgr, fpar, epar, args.match_ratio,
                        args.fail_med, args.fail_p95, args.fail_br,
                        bubblemap=bubblemap,
                        page_num=template_page_idx + 1,  # 1-indexed page number for bubblemap
                        full_dpi=float(getattr(args, 'dpi', 300)),
                        align_mode=align_mode
                    )
                    aligned_pages.append(warped)
                    if getattr(args, "out_dir", None):
                        base = os.path.splitext(os.path.basename(src_path))[0]
                        fn = os.path.join(args.out_dir, f"{args.prefix}_{base}_p{page_idx:03d}.png")
                        cv2.imwrite(fn, warped)
                    row = {
                        "source": os.path.basename(src_path),
                        "page": page_idx,
                        "template_page": template_page_idx + 1,
                        "mode": metrics.get("mode", "base"),
                        "res_median": f"{metrics['res_median']:.3f}" if metrics.get('res_median') == metrics.get('res_median') else "nan",
                        "res_p95": f"{metrics['res_p95']:.3f}" if metrics.get('res_p95') == metrics.get('res_p95') else "nan",
                        "res_br_p95": f"{metrics['res_br_p95']:.3f}" if metrics.get('res_br_p95') == metrics.get('res_br_p95') else "nan",
                        "n_inliers": metrics.get("n_inliers", ""),
                    }
                    metrics_rows.append(row)
                    print(f"[ok] {src_path} p{page_idx}: mode={row['mode']} median={row['res_median']} p95={row['res_p95']} br_p95={row['res_br_p95']}", file=sys.stderr)
                except Exception as e:
                    print(f"[error] {src_path} p{page_idx}: {e}", file=sys.stderr)
                    # CRITICAL: Even if alignment fails, we must append a page to maintain count
                    if getattr(args, "fallback_original", True):
                        print(f"[info] {src_path} p{page_idx}: Using original unaligned page as fallback", file=sys.stderr)
                        aligned_pages.append(scan_bgr)
                    else:
                        print(f"[info] {src_path} p{page_idx}: Using black page (fallback disabled)", file=sys.stderr)
                        # Create a black page of the same size
                        aligned_pages.append(np.zeros_like(scan_bgr))
                    metrics_rows.append({
                        "source": os.path.basename(src_path),
                        "page": page_idx,
                        "template_page": template_page_idx + 1,
                        "mode": "error",
                        "res_median": "nan",
                        "res_p95": "nan",
                        "res_br_p95": "nan",
                        "n_inliers": "",
                    })


    return aligned_pages, metrics_rows
