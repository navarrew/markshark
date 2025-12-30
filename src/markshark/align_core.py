import os
import sys
from types import SimpleNamespace
from typing import Optional, List, Tuple

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
"""

def align_pdf_scans(
    input_pdf: str,
    template: str,
    out_pdf: str = "aligned_scans.pdf",
    dpi: int = RENDER_DEFAULTS.dpi,
    pdf_renderer: str = "auto",
    template_page: int = 1,
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
) -> str:
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
    )

    # Load template page as BGR image (1-based page index)
    renderer = pdf_renderer
    if renderer == "auto":
        renderer = IO.choose_common_pdf_renderer([template, input_pdf], dpi=int(dpi), prefer="fitz")

    tpl_tuples = IO.convert_pdf_pages_to_bgr_tuples(template, dpi=dpi, renderer=renderer)
    if not (1 <= template_page <= len(tpl_tuples)):
        template_page = 1
    template_bgr = tpl_tuples[template_page - 1][2]  # (path, idx, bgr) → bgr

    # Convert raw scans to (src_path, page_idx, bgr)
    raw_scans_bgr = IO.convert_pdf_pages_to_bgr_tuples(input_pdf, dpi=dpi, renderer=renderer)

    # Align
    aligned_pages, metrics_rows = align_raw_bgr_scans(raw_scans_bgr, template_bgr, args)

    # Write outputs
    if args.out and aligned_pages:
        IO.save_images_as_pdf(aligned_pages, args.out, dpi=args.dpi)

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
        estimator_method=args.estimator_method, ransac_thresh=args.ransac_thresh,
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
            if getattr(args, "first_page", None) is not None and page_idx < int(args.first_page):
                continue
            if getattr(args, "last_page", None) is not None and page_idx > int(args.last_page):
                continue
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

    # If user requested ArUco-only, do not run feature alignment.
    if getattr(args, "align_method", "auto") == "aruco":
        if getattr(args, "fallback_original", True):
            for (src_path, page_idx, scan_bgr) in remaining_pages:
                aligned_pages.append(scan_bgr)
                metrics_rows.append({
                    "source": os.path.basename(src_path),
                    "page": page_idx,
                    "mode": "fallback_original",
                    "res_median": "nan",
                    "res_p95": "nan",
                    "res_br_p95": "nan",
                    "n_inliers": "",
                })
                print(f"[info] {src_path} p{page_idx}: kept original (ArUco-only mode).", file=sys.stderr)
        return aligned_pages, metrics_rows

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