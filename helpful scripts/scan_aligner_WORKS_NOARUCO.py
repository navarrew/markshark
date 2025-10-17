#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_aligner.py
-----------------------
Adds catastrophic-failure guardrails:
- Auto-retries with stricter settings if residuals are too high
- Last-resort fallback using detected page corners (quadrilateral) to compute scan->template warp
- Re-measures metrics after fallback

Includes:
- USAC/MAGSAC (if available) with fallback to RANSAC
- Grid-balanced ORB keypoints
- Optional ECC refinement (fixed to operate in template frame)
- Residual metrics including bottom-right 95th percentile
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import img2pdf, tempfile, os
from PIL import Image

def _ensure_dir(p: str) -> None:
    if not p: return
    os.makedirs(p, exist_ok=True)

def _has_cv_usac() -> bool:
    try:
        import cv2  # type: ignore
        return hasattr(cv2, "USAC_MAGSAC")
    except Exception:
        return False

def _imwrite(path: str, img) -> None:
    import cv2  # type: ignore
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


#---- WRITES OUT TO PDF 

def _save_as_pdf(pages_bgr, out_path: str, dpi: int = 150) -> None:
    """
    Save a list of BGR NumPy arrays to a single PDF at the requested DPI.
    """
    from PIL import Image

    if not pages_bgr:
        raise ValueError("No pages to save.")

    pil_pages = []
    for bgr in pages_bgr:
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        pil_pages.append(Image.fromarray(rgb.astype("uint8")).convert("RGB"))

    # Pillow's PDF writer embeds images; `resolution` sets DPI metadata.
    first, rest = pil_pages[0], pil_pages[1:]
    first.save(out_path, save_all=True, append_images=rest, format="PDF", resolution=dpi)
    print(f"[ok] wrote PDF: {out_path}")


#---- READS INPUT PDF TO BGR NUMPY OBJECT
 
def _render_pdf_to_bgr_pages(pdf_path: str, dpi: int) -> List:
    try:
        import fitz  # type: ignore
        import numpy as np  # type: ignore
        zoom = dpi / 72.0
        doc = fitz.open(pdf_path)
        pages = []
        for p in doc:
            mat = fitz.Matrix(zoom, zoom)
            pix = p.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = img[:, :, ::-1].copy()
            pages.append(bgr)
        doc.close()
        return pages
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path  # type: ignore
        import numpy as np  # type: ignore
        pil_pages = convert_from_path(pdf_path, dpi=dpi)
        pages = []
        for pil in pil_pages:
            rgb = np.array(pil.convert("RGB"))
            bgr = rgb[:, :, ::-1].copy()
            pages.append(bgr)
        return pages
    except Exception as e:
        raise RuntimeError(
            f"Cannot render PDF '{pdf_path}'. Install PyMuPDF (fitz) or pdf2image+poppler. Original error: {e}"
        )

def _load_image_to_bgr(path: str) -> Optional:
    try:
        import cv2  # type: ignore
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def load_pages(path: str, dpi: int) -> List:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pdf"]:
        return _render_pdf_to_bgr_pages(path, dpi)
    else:
        img = _load_image_to_bgr(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return [img]

# --------------------------- Features & Matching ---------------------------

@dataclass
class FeatureParams:
    tiles_x: int = 6
    tiles_y: int = 8
    topk_per_tile: int = 150
    orb_nfeatures: int = 3000
    orb_fast_threshold: int = 12

def _to_gray(img_bgr):
    import cv2  # type: ignore
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def detect_orb_grid(gray, params: FeatureParams):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    H, W = gray.shape[:2]
    orb = cv2.ORB_create(nfeatures=params.orb_nfeatures, fastThreshold=params.orb_fast_threshold)
    kpts, desc = [], []
    tile_w = W / params.tiles_x
    tile_h = H / params.tiles_y
    for ty in range(params.tiles_y):
        for tx in range(params.tiles_x):
            x0 = int(tx * tile_w); y0 = int(ty * tile_h)
            x1 = int(min(W, (tx + 1) * tile_w)); y1 = int(min(H, (ty + 1) * tile_h))
            roi = gray[y0:y1, x0:x1]
            kps = orb.detect(roi, None)
            if not kps: continue
            kps, des = orb.compute(roi, kps)
            if des is None or len(kps) == 0: continue
            idxs = np.argsort([-kp.response for kp in kps])[: params.topk_per_tile]
            kps_sel = [kps[i] for i in idxs]
            des_sel = des[idxs]
            for kp in kps_sel:
                kp.pt = (kp.pt[0] + x0, kp.pt[1] + y0)
            kpts.extend(kps_sel)
            desc.append(des_sel)
    if len(desc) == 0:
        return [], None
    import numpy as np  # type: ignore
    desc = np.vstack(desc)
    return kpts, desc

def match_descriptors(desc1, desc2, ratio=0.75, cross_check=True):
    import cv2  # type: ignore
    if desc1 is None or desc2 is None: return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    if cross_check and good:
        bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn2 = bf2.knnMatch(desc2, desc1, k=1)
        rev = {m.queryIdx: m.trainIdx for [m] in knn2}
        good = [m for m in good if rev.get(m.trainIdx, -1) == m.queryIdx]
    return good

# --------------------------- Estimation & ECC ---------------------------

@dataclass
class EstParams:
    method: str = "auto"
    ransac_thresh: float = 3.0
    max_iters: int = 10000
    confidence: float = 0.999
    ecc_levels: int = 4
    ecc_iters: int = 50
    ecc_eps: float = 1e-6
    use_ecc: bool = True

def estimate_homography(src_pts, dst_pts, est: EstParams):
    import cv2  # type: ignore
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None, None
    method_flag = None
    if est.method == "usac" or (est.method == "auto" and _has_cv_usac()):
        method_flag = getattr(cv2, "USAC_MAGSAC", None)
    if method_flag is None:
        method_flag = cv2.RANSAC
    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=method_flag,
        ransacReprojThreshold=est.ransac_thresh,
        maxIters=est.max_iters,
        confidence=est.confidence
    )
    return H, mask

def make_content_mask(gray):
    import cv2  # type: ignore
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    dil = cv2.dilate(opened, kernel, iterations=1)
    mask = (dil > 0).astype('uint8')
    return mask

def cv_project_points(src_pts_1x2, H):
    import numpy as np  # type: ignore
    N = src_pts_1x2.shape[0]
    ones = np.ones((N,1,1), dtype=np.float64)
    pts = np.concatenate([src_pts_1x2.astype(np.float64), ones], axis=2)
    Ht = H.T
    proj = pts @ Ht
    w = proj[:,:,2:3]
    proj = proj[:,:,:2] / np.maximum(w, 1e-8)
    return proj

def compute_residuals(H, src_pts, dst_pts, img_shape=None):
    import numpy as np  # type: ignore
    if H is None or src_pts is None or dst_pts is None or len(src_pts) == 0:
        return {"res_median": float("inf"), "res_p95": float("inf"),
                "res_br_p95": float("inf"), "n_inliers": 0}
    src = src_pts.reshape(-1,1,2)
    dst = dst_pts.reshape(-1,1,2)
    proj = cv_project_points(src, H)
    diff = proj - dst
    dists = np.sqrt((diff[:,:,0]**2 + diff[:,:,1]**2)).reshape(-1)
    import numpy as np  # type: ignore
    p95 = float(np.percentile(dists, 95)) if len(dists) else float("inf")
    med = float(np.median(dists)) if len(dists) else float("inf")
    br_p95 = p95
    if img_shape is not None:
        Hh, Hw = img_shape[:2]
        dst_flat = dst.reshape(-1,2)
        import numpy as np  # type: ignore
        br_mask = (dst_flat[:,0] > (Hw/2)) & (dst_flat[:,1] > (Hh/2))
        if np.any(br_mask):
            br_d = dists[br_mask]
            br_p95 = float(np.percentile(br_d, 95)) if len(br_d) else float("inf")
    return {"res_median": med, "res_p95": p95, "res_br_p95": br_p95, "n_inliers": int(len(dists))}

def ecc_refine(template_gray, scan_gray, H_init, mask, est: EstParams):
    if not est.use_ecc:
        return H_init
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    try:
        h, w = template_gray.shape[:2]
        tpl = template_gray.astype('float32') / 255.0
        try:
            H_inv_init = np.linalg.inv(H_init)
        except Exception:
            H_inv_init = np.linalg.pinv(H_init)
        scan_warped = cv2.warpPerspective(scan_gray, H_inv_init, (w, h), flags=cv2.INTER_LINEAR)
        scn = scan_warped.astype('float32') / 255.0
        warp = np.eye(3, dtype='float32')
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, est.ecc_iters, est.ecc_eps)
        inputMask = (mask.astype('uint8') * 255) if mask.dtype != 'uint8' else (mask * 255)
        inputMask = (inputMask > 0).astype('uint8')
        cc, warp_delta = cv2.findTransformECC(
            tpl, scn, warp, cv2.MOTION_HOMOGRAPHY, criteria,
            inputMask=inputMask, gaussFiltSize=5
        )
        H_inv_refined = warp_delta.astype('float64') @ H_inv_init
        H_refined = np.linalg.inv(H_inv_refined)
        return H_refined
    except Exception as e:
        print(f"[warn] ECC refinement skipped due to: {e}", file=sys.stderr)
        return H_init

# --------------------------- Fallback: Page Quadrilateral ---------------------------

def detect_page_quad(gray) -> Optional[List[Tuple[float,float]]]:
    """
    Detect the page contour as a 4-point polygon (tl, tr, br, bl) in SCAN frame.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    H, W = gray.shape[:2]
    # Strong threshold to isolate white page
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Invert if necessary (we want page as largest contour)
    if np.mean(thr) > 127:
        thr = 255 - thr
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        # Fallback to minAreaRect
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1,1,2).astype('int32')
    pts = approx.reshape(-1,2).astype('float32')
    # Order corners tl,tr,br,bl
    s = pts.sum(axis=1); diff = (pts[:,0]-pts[:,1])
    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

def warp_by_page_quad(template_bgr, scan_bgr):
    """
    Compute scan->template warp from detected page corners.
    Returns warped scan in template size and the H_inv used.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    tpl_h, tpl_w = _to_gray(template_bgr).shape[:2]
    quad = detect_page_quad(_to_gray(scan_bgr))
    if quad is None:
        raise RuntimeError("Page quadrilateral not found.")
    src = np.float32(quad)  # tl,tr,br,bl in scan
    dst = np.float32([[0,0],[tpl_w-1,0],[tpl_w-1,tpl_h-1],[0,tpl_h-1]])  # template corners
    H_inv = cv2.getPerspectiveTransform(src, dst)  # scan->template
    warped = cv2.warpPerspective(scan_bgr, H_inv, (tpl_w, tpl_h), flags=cv2.INTER_LINEAR)
    return warped, H_inv

# --------------------------- Main Alignment with Guardrails ---------------------------

def align_page_once(template_bgr, scan_bgr, fpar, epar, ransac_ratio):
    import numpy as np  # type: ignore
    import cv2  # type: ignore
    tpl_g = _to_gray(template_bgr); scn_g = _to_gray(scan_bgr)
    k1, d1 = detect_orb_grid(tpl_g, fpar)
    k2, d2 = detect_orb_grid(scn_g, fpar)
    if len(k1) < 4 or len(k2) < 4:
        raise RuntimeError("Not enough keypoints.")
    matches = match_descriptors(d1, d2, ratio=ransac_ratio, cross_check=True)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches after filtering.")
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    H, inliers = estimate_homography(src, dst, epar)
    if H is None or inliers is None or inliers.sum() < 4:
        raise RuntimeError("Homography estimation failed.")
    inlier_mask = inliers.ravel().astype(bool)
    src_in = src[inlier_mask]; dst_in = dst[inlier_mask]
    mask = make_content_mask(tpl_g)
    H_ref = ecc_refine(tpl_g, scn_g, H, mask, epar)
    # Warp scan into template canvas
    try:
        H_inv = np.linalg.inv(H_ref)
    except Exception:
        H_inv = np.linalg.pinv(H_ref)
    h, w = tpl_g.shape[:2]
    warped = cv2.warpPerspective(scan_bgr, H_inv, (w, h), flags=cv2.INTER_LINEAR)
    metrics = compute_residuals(H_ref, src_in, dst_in, img_shape=scn_g.shape)
    return warped, metrics

def need_retry(metrics, fail_med, fail_p95, fail_br):
    return (metrics['res_median'] > fail_med) or (metrics['res_p95'] > fail_p95) or (metrics['res_br_p95'] > fail_br)

def align_with_guardrails(template_bgr, scan_bgr, base_fpar, base_epar, base_ratio, fail_med, fail_p95, fail_br):
    """
    Try once with base params; if bad, retry with stricter grid/matching; if still bad, fallback to page-quad warp.
    Returns warped image and a dictionary of metrics + 'mode' ('base'|'retry'|'quad_fallback').
    """
    # Attempt 1: base
    try:
        warped, metrics = align_page_once(template_bgr, scan_bgr, base_fpar, base_epar, base_ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'base'
            return warped, metrics
        print(f"[retry] metrics high (med={metrics['res_median']:.1f}, p95={metrics['res_p95']:.1f}, br={metrics['res_br_p95']:.1f}) -> tightening", file=sys.stderr)
    except Exception as e:
        print(f"[retry] base failed: {e}", file=sys.stderr)

    # Attempt 2: stricter params
    from copy import deepcopy
    fpar = deepcopy(base_fpar)
    epar = deepcopy(base_epar)
    ratio = max(0.68, base_ratio - 0.05)
    fpar.tiles_x = max(fpar.tiles_x, 8)
    fpar.tiles_y = max(fpar.tiles_y, 10)
    fpar.orb_fast_threshold = max(6, fpar.orb_fast_threshold - 4)
    epar.ransac_thresh = min(epar.ransac_thresh, 2.0)
    epar.max_iters = max(epar.max_iters, 20000)

    try:
        warped, metrics = align_page_once(template_bgr, scan_bgr, fpar, epar, ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'retry'
            return warped, metrics
        print(f"[retry] still high after tighten (med={metrics['res_median']:.1f}, p95={metrics['res_p95']:.1f}, br={metrics['res_br_p95']:.1f}) -> quad fallback", file=sys.stderr)
    except Exception as e:
        print(f"[retry] tighten failed: {e}", file=sys.stderr)

    # Attempt 3: page-quad fallback
    try:
        warped_q, H_inv_q = warp_by_page_quad(template_bgr, scan_bgr)
        # Optional ECC micro-refinement in template frame: align warped_q to template via ECC identity delta
        # (reuse ecc_refine by passing template and warped_q with H_init=identity)
        import numpy as np
        import cv2
        H_init = np.eye(3, dtype='float64')
        mask = make_content_mask(_to_gray(template_bgr))
        H_ref = ecc_refine(_to_gray(template_bgr), _to_gray(warped_q), H_init, mask, base_epar)
        try:
            H_inv_ref = np.linalg.inv(H_ref)
        except Exception:
            H_inv_ref = np.linalg.pinv(H_ref)
        warped = cv2.warpPerspective(warped_q, H_inv_ref, (_to_gray(template_bgr).shape[1], _to_gray(template_bgr).shape[0]))

        # Re-measure by running feature alignment between template and warped (identity-ish); compute residuals by new H2
        # If this fails, at least return the quad-warp.
        try:
            # Use mild params to compute diagnostic H2
            fpar2 = FeatureParams(tiles_x=6, tiles_y=8, topk_per_tile=120, orb_nfeatures=2500, orb_fast_threshold=12)
            epar2 = EstParams(method=base_epar.method, ransac_thresh=3.0, max_iters=10000, confidence=0.999, use_ecc=False,
                              ecc_levels=base_epar.ecc_levels, ecc_iters=base_epar.ecc_iters, ecc_eps=base_epar.ecc_eps)
            # Align template -> warped to get H2 (should be near identity)
            import numpy as np
            import cv2
            tpl_g = _to_gray(template_bgr); war_g = _to_gray(warped)
            k1, d1 = detect_orb_grid(tpl_g, fpar2); k2, d2 = detect_orb_grid(war_g, fpar2)
            matches = match_descriptors(d1, d2, ratio=0.75, cross_check=True)
            if len(matches) >= 4:
                src = np.float32([k1[m.queryIdx].pt for m in matches]); dst = np.float32([k2[m.trainIdx].pt for m in matches])
                H2, inl2 = estimate_homography(src, dst, epar2)
                if H2 is not None and inl2 is not None and inl2.sum() >= 4:
                    metrics = compute_residuals(H2, src[inl2.ravel()>0], dst[inl2.ravel()>0], img_shape=war_g.shape)
                else:
                    metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}
            else:
                metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}
        except Exception as _:
            metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}
        metrics['mode'] = 'quad_fallback'
        return warped, metrics
    except Exception as e:
        raise RuntimeError(f"Quad fallback failed: {e}")

# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Align scanned OMR pages to a template with robust homography, ECC, and guardrails.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-pdf", type=str, help="Single input PDF of scanned pages.")
    src.add_argument("--inputs", nargs="+", help="One or more inputs (PDF or image files).")
    ap.add_argument("--template", required=True, help="Template PDF or image to align to (first page used if PDF).")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering. Default: 300")
    ap.add_argument("--out", type=str, default=None, help="Output PDF path (multipage) if provided.")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory for per-page aligned PNGs.")
    ap.add_argument("--prefix", type=str, default="aligned", help="Filename prefix for per-page PNGs. Default: aligned")
    ap.add_argument("--metrics-csv", type=str, default=None, help="CSV file to append per-page residual metrics.")
    # ArUco approaches
    ap.add_argument("--aruco", action="store_true", help="Use ArUco-based alignment.")
    # Feature params
    ap.add_argument("--tiles-x", type=int, default=6, help="Grid tiles horizontally for feature coverage.")
    ap.add_argument("--tiles-y", type=int, default=8, help="Grid tiles vertically for feature coverage.")
    ap.add_argument("--topk-per-tile", type=int, default=150, help="Top ORB features per tile.")
    ap.add_argument("--orb-nfeatures", type=int, default=3000, help="ORB nfeatures.")
    ap.add_argument("--orb-fast-threshold", type=int, default=12, help="ORB FAST threshold. Lower for more features.")
    # Matching
    ap.add_argument("--match-ratio", type=float, default=0.75, help="Lowe ratio for descriptor matching.")
    # Estimation & refinement
    ap.add_argument("--method", type=str, default="auto", choices=["auto","usac","ransac"], help="Homography estimator.")
    ap.add_argument("--ransac", type=float, dest="ransac_thresh", default=3.0, help="RANSAC/USAC reprojection threshold (px).")
    ap.add_argument("--max-iters", type=int, default=10000, help="Max iterations for robust estimator.")
    ap.add_argument("--confidence", type=float, default=0.999, help="Confidence for robust estimator.")
    ap.add_argument("--ecc", action="store_true", help="Enable ECC refinement.")
    ap.add_argument("--ecc-levels", type=int, default=4, help="(Hint) pyramid levels (if used by OpenCV build).")
    ap.add_argument("--ecc-iters", type=int, default=50, help="ECC iterations.")
    ap.add_argument("--ecc-eps", type=float, default=1e-6, help="ECC epsilon.")
    # Guardrail thresholds
    ap.add_argument("--fail-med", type=float, default=3.0, help="If median error exceeds this, trigger retry. Default 3 px.")
    ap.add_argument("--fail-p95", type=float, default=8.0, help="If 95th error exceeds this, trigger retry. Default 8 px.")
    ap.add_argument("--fail-br", type=float, default=8.0, help="If bottom-right 95th exceeds this, trigger retry. Default 8 px.")
    return ap.parse_args()


# @dataclass
# class FeatureParams:
#     tiles_x: int = 6
#     tiles_y: int = 8
#     topk_per_tile: int = 150
#     orb_nfeatures: int = 3000
#     orb_fast_threshold: int = 12


def main():
    args = parse_args()
    fpar = FeatureParams(tiles_x=args.tiles_x, tiles_y=args.tiles_y, topk_per_tile=args.topk_per_tile,
                         orb_nfeatures=args.orb_nfeatures, orb_fast_threshold=args.orb_fast_threshold)
    epar = EstParams(method=args.method, ransac_thresh=args.ransac_thresh, max_iters=args.max_iters,
                     confidence=args.confidence, ecc_levels=args.ecc_levels, ecc_iters=args.ecc_iters,
                     ecc_eps=args.ecc_eps, use_ecc=(args.ecc))
    dpi = int(args.dpi)
    _ensure_dir(args.out_dir or ".")
 
    if not args.out and not args.out_dir:
        print("[info] No output paths specified. Use --out for PDF or --out-dir for PNGs.", file=sys.stderr)
   
#import the template and raise an error if it fails
    tpl_pages = load_pages(args.template, dpi=dpi)
    if not tpl_pages:
        raise RuntimeError(f"No pages loaded from template: {args.template}")
    template_bgr = tpl_pages[0]

#import the unaligned pdf scans by passing in either a single pdf or a list of images/pdfs to a list called input_files
    input_files = [args.input_pdf] if args.input_pdf else (args.inputs or [])
    all_scan_pages = []
    for path in input_files:
        pages = load_pages(path, dpi=dpi) # returns list of bgr images from each input file
        all_scan_pages.extend([(path, i, p) for i, p in enumerate(pages, 1)])
    if not all_scan_pages:
        raise RuntimeError("No input pages found.")
    aligned_pages = []
    metrics_rows = []
#at this point we could split into ARUCO and non-ARUCO pages, but for now we just do all the same way
    for (src_path, page_idx, scan_bgr) in all_scan_pages:
        try:
            warped, metrics = align_with_guardrails(template_bgr, scan_bgr, fpar, epar, args.match_ratio,
                                                    args.fail_med, args.fail_p95, args.fail_br)
            aligned_pages.append(warped)
            if args.out_dir:
                base = os.path.splitext(os.path.basename(src_path))[0]
                fn = os.path.join(args.out_dir, f"{args.prefix}_{base}_p{page_idx:03d}.png")
                _imwrite(fn, warped)
            row = {
                "source": os.path.basename(src_path),
                "page": page_idx,
                "mode": metrics.get("mode","base"),
                "res_median": f"{metrics['res_median']:.3f}" if metrics['res_median']==metrics['res_median'] else "nan",
                "res_p95": f"{metrics['res_p95']:.3f}" if metrics['res_p95']==metrics['res_p95'] else "nan",
                "res_br_p95": f"{metrics['res_br_p95']:.3f}" if metrics['res_br_p95']==metrics['res_br_p95'] else "nan",
                "n_inliers": metrics.get("n_inliers",""),
            }
            metrics_rows.append(row)
            print(f"[ok] {src_path} p{page_idx}: mode={row['mode']} median={row['res_median']} p95={row['res_p95']} br_p95={row['res_br_p95']}")
        except Exception as e:
            print(f"[error] {src_path} p{page_idx}: {e}", file=sys.stderr)

#-----------if you want to write out a csv file of your alignment metrics------
    if args.metrics_csv and metrics_rows:
        write_header = (not os.path.exists(args.metrics_csv))
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            if write_header:
                w.writeheader()
            for r in metrics_rows:
                w.writerow(r)
        print(f"[ok] metrics → {args.metrics_csv}")

#----------------
    if args.out and aligned_pages:
        _save_as_pdf(aligned_pages, args.out, dpi=args.dpi)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr); sys.exit(130)
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr); sys.exit(1)
