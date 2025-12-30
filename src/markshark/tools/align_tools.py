#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MarkShark
align_tools.py (helpers only)
------------------------------
Reusable alignment primitives:

- ArUco detection + homography: detect_aruco_centers, align_with_aruco
- Grid-balanced ORB keypoints + matching: detect_orb_grid, match_descriptors
- Robust homography estimation (USAC/MAGSAC if available) + ECC refinement:
  estimate_homography, ecc_refine
- Residual metrics: compute_residuals
- Fallback via page quadrilateral: detect_page_quad, warp_by_page_quad
- Guardrailed alignment pipeline: align_page_once, align_with_guardrails

"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import os
import numpy as np  # type: ignore
import cv2 as cv    # type: ignore

from ..defaults import FeatureParams, EstParams, apply_feat_overrides, apply_est_overrides

# ---------------------------- Utilities & I/O ----------------------------

def _ensure_dir(p: str) -> None:
    if not p:
        return
    os.makedirs(p, exist_ok=True)

def _has_cv_usac() -> bool:
    """Return True if this OpenCV build exposes the USAC constants we want."""
    return hasattr(cv, "USAC_MAGSAC")

def detect_aruco_centers(img_bgr: np.ndarray, dict_name: str = "DICT_4X4_50") -> Dict[int, Tuple[float, float]]:
    """Return dict id -> (cx, cy) for detected ArUco markers (in image coords)."""
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    aruco = cv.aruco
    name = dict_name.upper()
    if not name.startswith("DICT_"):
        name = "DICT_4X4_50"
    DICT_CONST = getattr(aruco, name, aruco.DICT_4X4_50)
    adict = aruco.getPredefinedDictionary(DICT_CONST)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(adict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    centers: Dict[int, Tuple[float, float]] = {}
    if ids is not None:
        ids = ids.flatten()
        for i, cid in enumerate(ids):
            c = corners[i][0]  # (4,2)
            cx = float(np.mean(c[:, 0])); cy = float(np.mean(c[:, 1]))
            centers[int(cid)] = (cx, cy)
    return centers

def homography_from_points(src_pts: np.ndarray, dst_pts: np.ndarray, ransac: float = 3.0) -> Optional[np.ndarray]:
    if len(src_pts) >= 4 and len(dst_pts) == len(src_pts):
        H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac)
        return H
    return None

def align_with_aruco(page_bgr: np.ndarray,
                     template_bgr: np.ndarray,
                     dict_name: str = "DICT_4X4_50",
                     min_markers: int = 4,
                     ransac: float = 3.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int]]:
    """
    Return (aligned_scan_in_template_canvas, H_scan_to_template, detected_ids)
    """
    t_centers = detect_aruco_centers(template_bgr, dict_name)
    p_centers = detect_aruco_centers(page_bgr, dict_name)
    common = sorted(set(t_centers.keys()) & set(p_centers.keys()))
    if len(common) < max(3, min_markers):
        return None, None, common
    src = np.float32([p_centers[i] for i in common]).reshape(-1, 1, 2)
    dst = np.float32([t_centers[i] for i in common]).reshape(-1, 1, 2)
    H = homography_from_points(src, dst, ransac=ransac)
    if H is None:
        return None, None, common
    Ht, Wt = template_bgr.shape[:2]
    aligned = cv.warpPerspective(page_bgr, H, (Wt, Ht))
    return aligned, H, common

# --------------------------- Features & Matching ---------------------------

def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

def detect_orb_grid(gray: np.ndarray, params: FeatureParams):
    H, W = gray.shape[:2]
    orb = cv.ORB_create(nfeatures=params.orb_nfeatures, fastThreshold=params.orb_fast_threshold)
    kpts, desc = [], []
    tile_w = W / params.tiles_x
    tile_h = H / params.tiles_y
    for ty in range(params.tiles_y):
        for tx in range(params.tiles_x):
            x0 = int(tx * tile_w); y0 = int(ty * tile_h)
            x1 = int(min(W, (tx + 1) * tile_w)); y1 = int(min(H, (ty + 1) * tile_h))
            roi = gray[y0:y1, x0:x1]
            kps = orb.detect(roi, None)
            if not kps:
                continue
            kps, des = orb.compute(roi, kps)
            if des is None or len(kps) == 0:
                continue
            idxs = np.argsort([-kp.response for kp in kps])[: params.topk_per_tile]
            kps_sel = [kps[i] for i in idxs]
            des_sel = des[idxs]
            for kp in kps_sel:
                kp.pt = (kp.pt[0] + x0, kp.pt[1] + y0)
            kpts.extend(kps_sel)
            desc.append(des_sel)
    if len(desc) == 0:
        return [], None
    desc = np.vstack(desc)
    return kpts, desc

def match_descriptors(desc1, desc2, ratio=0.75, cross_check=True):
    if desc1 is None or desc2 is None:
        return []
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    if cross_check and good:
        bf2 = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        knn2 = bf2.knnMatch(desc2, desc1, k=1)
        rev = {m.queryIdx: m.trainIdx for [m] in knn2}
        good = [m for m in good if rev.get(m.trainIdx, -1) == m.queryIdx]
    return good

# --------------------------- Estimation & ECC ---------------------------

def estimate_homography(src_pts: np.ndarray, dst_pts: np.ndarray, est: EstParams):
    if len(src_pts) < 4 or len(dst_pts) < 4:
        return None, None
    method_flag = None
    if est.estimator_method == "usac" or (est.estimator_method == "auto" and _has_cv_usac()):
        import cv2  # type: ignore
        method_flag = getattr(cv2, "USAC_MAGSAC", None)
    if method_flag is None:
        method_flag = cv.RANSAC
    H, mask = cv.findHomography(
        src_pts, dst_pts,
        method=method_flag,
        ransacReprojThreshold=est.ransac_thresh,
        maxIters=est.max_iters,
        confidence=est.confidence,
    )
    return H, mask

def make_content_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thr = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY_INV, 35, 10)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opened = cv.morphologyEx(thr, cv.MORPH_OPEN, kernel, iterations=1)
    dil = cv.dilate(opened, kernel, iterations=1)
    mask = (dil > 0).astype('uint8')
    return mask

def cv_project_points(src_pts_1x2: np.ndarray, H: np.ndarray) -> np.ndarray:
    N = src_pts_1x2.shape[0]
    ones = np.ones((N, 1, 1), dtype=np.float64)
    pts = np.concatenate([src_pts_1x2.astype(np.float64), ones], axis=2)
    Ht = H.T
    proj = pts @ Ht
    w = proj[:, :, 2:3]
    proj = proj[:, :, :2] / np.maximum(w, 1e-8)
    return proj

def compute_residuals(H: Optional[np.ndarray],
                      src_pts: Optional[np.ndarray],
                      dst_pts: Optional[np.ndarray],
                      img_shape: Optional[Tuple[int, int]] = None) -> Dict[str, float]:
    if H is None or src_pts is None or dst_pts is None or len(src_pts) == 0:
        return {"res_median": float("inf"), "res_p95": float("inf"),
                "res_br_p95": float("inf"), "n_inliers": 0}
    src = src_pts.reshape(-1, 1, 2)
    dst = dst_pts.reshape(-1, 1, 2)
    proj = cv_project_points(src, H)
    diff = proj - dst
    dists = np.sqrt((diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)).reshape(-1)
    p95 = float(np.percentile(dists, 95)) if len(dists) else float("inf")
    med = float(np.median(dists)) if len(dists) else float("inf")
    br_p95 = p95
    if img_shape is not None:
        Hh, Hw = img_shape[:2]
        dst_flat = dst.reshape(-1, 2)
        br_mask = (dst_flat[:, 0] > (Hw / 2)) & (dst_flat[:, 1] > (Hh / 2))
        if np.any(br_mask):
            br_d = dists[br_mask]
            br_p95 = float(np.percentile(br_d, 95)) if len(br_d) else float("inf")
    return {"res_median": med, "res_p95": p95, "res_br_p95": br_p95, "n_inliers": int(len(dists))}

def ecc_refine(template_gray: np.ndarray,
               scan_gray: np.ndarray,
               H_init: np.ndarray,
               mask: np.ndarray,
               est: EstParams) -> np.ndarray:
    if not est.use_ecc:
        return H_init
    try:
        h, w = template_gray.shape[:2]
        tpl = template_gray.astype('float32') / 255.0
        try:
            H_inv_init = np.linalg.inv(H_init)
        except Exception:
            H_inv_init = np.linalg.pinv(H_init)
        scan_warped = cv.warpPerspective(scan_gray, H_inv_init, (w, h), flags=cv.INTER_LINEAR)
        scn = scan_warped.astype('float32') / 255.0
        warp = np.eye(3, dtype='float32')
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, est.ecc_max_iters, est.ecc_eps)
        inputMask = (mask.astype('uint8') * 255) if mask.dtype != 'uint8' else (mask * 255)
        inputMask = (inputMask > 0).astype('uint8')
        cc, warp_delta = cv.findTransformECC(
            tpl, scn, warp, cv.MOTION_HOMOGRAPHY, criteria,
            inputMask=inputMask, gaussFiltSize=5
        )
        H_inv_refined = warp_delta.astype('float64') @ H_inv_init
        H_refined = np.linalg.inv(H_inv_refined)
        return H_refined
    except Exception:
        # If ECC fails for any reason, just return the initial H
        return H_init

# --------------------------- Fallback: Page Quadrilateral ---------------------------

def detect_page_quad(gray: np.ndarray) -> Optional[List[Tuple[float, float]]]:
    """Detect page contour as a 4-point polygon (tl, tr, br, bl) in SCAN frame."""
    H, W = gray.shape[:2]
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Invert if necessary (we want page as largest contour)
    if np.mean(thr) > 127:
        thr = 255 - thr
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    thr = cv.morphologyEx(thr, cv.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv.contourArea)
    peri = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype('int32')
    pts = approx.reshape(-1, 2).astype('float32')
    # Order corners tl,tr,br,bl
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]

def warp_by_page_quad(template_bgr: np.ndarray, scan_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scan->template warp from detected page corners.
    Returns (warped_scan_in_template_size, H_inv).
    """
    tpl_h, tpl_w = _to_gray(template_bgr).shape[:2]
    quad = detect_page_quad(_to_gray(scan_bgr))
    if quad is None:
        raise RuntimeError("Page quadrilateral not found.")
    src = np.float32(quad)  # tl,tr,br,bl in scan
    dst = np.float32([[0, 0], [tpl_w - 1, 0], [tpl_w - 1, tpl_h - 1], [0, tpl_h - 1]])  # template corners
    H_inv = cv.getPerspectiveTransform(src, dst)  # scan->template
    warped = cv.warpPerspective(scan_bgr, H_inv, (tpl_w, tpl_h), flags=cv.INTER_LINEAR)
    return warped, H_inv

# --------------------------- Guardrailed Alignment ---------------------------

def align_page_once(template_bgr: np.ndarray,
                    scan_bgr: np.ndarray,
                    fpar: FeatureParams,
                    epar: EstParams,
                    ratio: float):
    tpl_g = _to_gray(template_bgr)
    scn_g = _to_gray(scan_bgr)
    k1, d1 = detect_orb_grid(tpl_g, fpar)
    k2, d2 = detect_orb_grid(scn_g, fpar)
    if len(k1) < 4 or len(k2) < 4:
        raise RuntimeError("Not enough keypoints.")
    matches = match_descriptors(d1, d2, ratio=ratio, cross_check=True)
    if len(matches) < 4:
        raise RuntimeError("Not enough matches after filtering.")
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    H, inliers = estimate_homography(src, dst, epar)
    if H is None or inliers is None or inliers.sum() < 4:
        raise RuntimeError("Homography estimation failed.")
    inlier_mask = inliers.ravel().astype(bool)
    src_in = src[inlier_mask]
    dst_in = dst[inlier_mask]
    mask = make_content_mask(tpl_g)
    H_ref = ecc_refine(tpl_g, scn_g, H, mask, epar)
    # Warp scan into template canvas
    try:
        H_inv = np.linalg.inv(H_ref)
    except Exception:
        H_inv = np.linalg.pinv(H_ref)
    h, w = tpl_g.shape[:2]
    warped = cv.warpPerspective(scan_bgr, H_inv, (w, h), flags=cv.INTER_LINEAR)
    metrics = compute_residuals(H_ref, src_in, dst_in, img_shape=scn_g.shape)
    return warped, metrics

def need_retry(metrics: Dict[str, float], fail_med: float, fail_p95: float, fail_br: float) -> bool:
    return (metrics['res_median'] > fail_med) or (metrics['res_p95'] > fail_p95) or (metrics['res_br_p95'] > fail_br)

def align_with_guardrails(template_bgr: np.ndarray,
                          scan_bgr: np.ndarray,
                          base_fpar: FeatureParams,
                          base_epar: EstParams,
                          base_ratio: float,
                          fail_med: float,
                          fail_p95: float,
                          fail_br: float):
    """
    Try once with base params; if metrics too high, retry with stricter settings; if still bad, fallback via page quad.
    Returns (warped, metrics_dict_with_mode).
    """
    # Attempt 1: base
    try:
        warped, metrics = align_page_once(template_bgr, scan_bgr, base_fpar, base_epar, base_ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'base'
            return warped, metrics
    except Exception:
        pass

    # Attempt 2: stricter params
    ratio = max(0.68, base_ratio - 0.05)
    
    fpar = apply_feat_overrides(
        tiles_x=max(base_fpar.tiles_x, 8),
        tiles_y=max(base_fpar.tiles_y, 10),
        orb_fast_threshold=max(6, base_fpar.orb_fast_threshold - 4),
        # carry forward unchanged fields
        topk_per_tile=base_fpar.topk_per_tile,
        orb_nfeatures=base_fpar.orb_nfeatures,
    )
    
    epar = apply_est_overrides(
        method=base_epar.method,
        ransac_thresh=min(base_epar.ransac_thresh, 2.0),
        max_iters=max(base_epar.max_iters, 20000),
        confidence=base_epar.confidence,
        use_ecc=base_epar.use_ecc,
        ecc_levels=base_epar.ecc_levels,
        # if your EstParams uses 'ecc_max_iters'/'ecc_eps', include them; adjust names to your dataclass
        ecc_max_iters=getattr(base_epar, "ecc_max_iters", 50),
        ecc_eps=base_epar.ecc_eps,
    )

    try:
        warped, metrics = align_page_once(template_bgr, scan_bgr, fpar, epar, ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'retry'
            return warped, metrics
    except Exception:
        pass

    # Attempt 3: page-quad fallback (+ optional ECC micro-refine)
    warped_q, H_inv_q = warp_by_page_quad(template_bgr, scan_bgr)
    try:
        H_init = np.eye(3, dtype='float64')
        mask = make_content_mask(_to_gray(template_bgr))
        H_ref = ecc_refine(_to_gray(template_bgr), _to_gray(warped_q), H_init, mask, base_epar)
        try:
            H_inv_ref = np.linalg.inv(H_ref)
        except Exception:
            H_inv_ref = np.linalg.pinv(H_ref)
        warped = cv.warpPerspective(warped_q, H_inv_ref,
                                    (_to_gray(template_bgr).shape[1], _to_gray(template_bgr).shape[0]))
    except Exception:
        warped = warped_q

    # Optional: compute diagnostic residuals after fallback (best-effort)
    try:
        fpar2 = apply_feat_overrides(tiles_x=6, tiles_y=8, topk_per_tile=120,
                                     orb_nfeatures=2500, orb_fast_threshold=12)
        epar2 = apply_est_overrides(method=base_epar.method, ransac_thresh=3.0,
                                    max_iters=10000, confidence=0.999,
                                    use_ecc=False, ecc_levels=base_epar.ecc_levels,
                                    ecc_max_iters=base_epar.ecc_iters, ecc_eps=base_epar.ecc_eps)
        tpl_g = _to_gray(template_bgr); war_g = _to_gray(warped)
        k1, d1 = detect_orb_grid(tpl_g, fpar2); k2, d2 = detect_orb_grid(war_g, fpar2)
        matches = match_descriptors(d1, d2, ratio=0.75, cross_check=True)
        if len(matches) >= 4:
            src = np.float32([k1[m.queryIdx].pt for m in matches]); dst = np.float32([k2[m.trainIdx].pt for m in matches])
            H2, inl2 = estimate_homography(src, dst, epar2)
            if H2 is not None and inl2 is not None and inl2.sum() >= 4:
                metrics = compute_residuals(H2, src[inl2.ravel() > 0], dst[inl2.ravel() > 0], img_shape=war_g.shape)
            else:
                metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}
        else:
            metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}
    except Exception:
        metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}

    metrics['mode'] = 'quad_fallback'
    return warped, metrics
