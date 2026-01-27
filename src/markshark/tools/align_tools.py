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
- Bubble grid alignment: align_with_bubble_grid (NEW)
- Guardrailed alignment pipeline: align_page_once, align_with_guardrails

"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import os
import sys
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


# ==================== NEW: Bubble Grid Alignment ====================
#
# Uses known bubble positions from bubblemap to detect circles in scans
# and compute a robust homography. This is more reliable than ORB features
# for bubble sheets because it leverages the structured grid layout.
# ====================================================================

def _get_bubble_centers_from_layout(
    layout: Any,
    img_width: int,
    img_height: int
) -> List[Tuple[float, float]]:
    """
    Extract bubble center coordinates from a single GridLayout or dict.
    Works with both GridLayout dataclass objects and raw dictionaries.
    
    Args:
        layout: GridLayout object or dict with x_topleft, y_topleft, etc.
        img_width: Template image width in pixels
        img_height: Template image height in pixels
        
    Returns:
        List of (x, y) pixel coordinates for each bubble center
    """
    # Handle both GridLayout objects and dictionaries
    if hasattr(layout, 'x_topleft'):
        # GridLayout dataclass
        x1 = layout.x_topleft * img_width
        y1 = layout.y_topleft * img_height
        x2 = layout.x_bottomright * img_width
        y2 = layout.y_bottomright * img_height
        numrows = layout.numrows
        numcols = layout.numcols
    else:
        # Dictionary
        x1 = layout['x_topleft'] * img_width
        y1 = layout['y_topleft'] * img_height
        x2 = layout['x_bottomright'] * img_width
        y2 = layout['y_bottomright'] * img_height
        numrows = layout['numrows']
        numcols = layout['numcols']
    
    # Calculate spacing between bubble centers
    if numcols > 1:
        x_step = (x2 - x1) / (numcols - 1)
    else:
        x_step = 0
        x1 = (x1 + x2) / 2  # Center single column
        
    if numrows > 1:
        y_step = (y2 - y1) / (numrows - 1)
    else:
        y_step = 0
        y1 = (y1 + y2) / 2  # Center single row
    
    centers = []
    for row in range(numrows):
        for col in range(numcols):
            cx = x1 + col * x_step
            cy = y1 + row * y_step
            centers.append((cx, cy))
    
    return centers


def _get_all_bubble_centers_from_bubblemap(
    bubblemap: Any,
    img_width: int,
    img_height: int,
    page_num: int = 1
) -> Tuple[np.ndarray, float]:
    """
    Extract all bubble center coordinates from a Bubblemap object.
    
    Args:
        bubblemap: Bubblemap object (from bubblemap_io)
        img_width: Template image width in pixels
        img_height: Template image height in pixels
        page_num: Which page to extract (1-indexed)
        
    Returns:
        (centers_array, radius_pct) - numpy array of shape (N, 2) and radius as fraction
    """
    all_centers = []
    radius_pct = 0.008  # Default
    
    # Get the page layout
    if hasattr(bubblemap, 'get_page'):
        page = bubblemap.get_page(page_num)
        if page is None:
            raise ValueError(f"Page {page_num} not found in bubblemap")
        
        # Extract from standard layouts
        for layout_attr in ['last_name_layout', 'first_name_layout', 'id_layout', 'version_layout']:
            layout = getattr(page, layout_attr, None)
            if layout is not None:
                centers = _get_bubble_centers_from_layout(layout, img_width, img_height)
                all_centers.extend(centers)
                if hasattr(layout, 'radius_pct'):
                    radius_pct = layout.radius_pct
        
        # Extract from answer layouts
        if hasattr(page, 'answer_layouts') and page.answer_layouts:
            for layout in page.answer_layouts:
                centers = _get_bubble_centers_from_layout(layout, img_width, img_height)
                all_centers.extend(centers)
                if hasattr(layout, 'radius_pct'):
                    radius_pct = layout.radius_pct
    else:
        # Fallback: treat as raw dict (backward compatibility)
        page_key = f"page_{page_num}"
        if page_key not in bubblemap:
            raise ValueError(f"Page key '{page_key}' not found in bubblemap")
        
        page_data = bubblemap[page_key]
        
        for layout_name in ['last_name_layout', 'first_name_layout', 'id_layout', 'version_layout']:
            if layout_name in page_data:
                centers = _get_bubble_centers_from_layout(page_data[layout_name], img_width, img_height)
                all_centers.extend(centers)
                radius_pct = page_data[layout_name].get('radius_pct', radius_pct)
        
        if 'answer_layouts' in page_data:
            for layout in page_data['answer_layouts']:
                centers = _get_bubble_centers_from_layout(layout, img_width, img_height)
                all_centers.extend(centers)
                radius_pct = layout.get('radius_pct', radius_pct)
    
    return np.array(all_centers, dtype=np.float32), radius_pct


def _detect_circles_hough(
    gray: np.ndarray,
    min_radius: float,
    max_radius: float,
    param1: int = 50,
    param2: int = 25,
    min_dist_factor: float = 1.5
) -> np.ndarray:
    """
    Detect circles using Hough Circle Transform.
    
    Returns:
        Array of shape (N, 3) with (x, y, radius) for each detected circle,
        or empty array if no circles found
    """
    # Apply slight blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 1.5)
    
    # Detect circles
    circles = cv.HoughCircles(
        blurred,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=int(max(min_radius * min_dist_factor, 5)),
        param1=param1,
        param2=param2,
        minRadius=int(max(min_radius * 0.5, 3)),
        maxRadius=int(max_radius * 1.5)
    )
    
    if circles is None:
        return np.array([]).reshape(0, 3)
    
    return circles[0]  # Shape: (N, 3)


def _detect_circles_contour(
    gray: np.ndarray,
    min_radius: float,
    max_radius: float,
    circularity_thresh: float = 0.65
) -> np.ndarray:
    """
    Detect circles using contour analysis (backup method).
    More robust to partially filled bubbles than Hough.
    
    Returns:
        Array of shape (N, 3) with (x, y, radius) for each detected circle
    """
    # Adaptive threshold to handle varying lighting
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV, 15, 5
    )
    
    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    circles = []
    min_area = np.pi * (min_radius * 0.5) ** 2
    max_area = np.pi * (max_radius * 1.5) ** 2
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < circularity_thresh:
            continue
        
        # Get enclosing circle
        (x, y), radius = cv.minEnclosingCircle(cnt)
        if min_radius * 0.5 <= radius <= max_radius * 1.5:
            circles.append([x, y, radius])
    
    return np.array(circles, dtype=np.float32).reshape(-1, 3) if circles else np.array([]).reshape(0, 3)


def _match_circles_to_expected(
    detected_circles: np.ndarray,
    expected_centers: np.ndarray,
    max_dist_ratio: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match detected circles to expected bubble positions.
    
    Returns:
        (matched_detected, matched_expected) - corresponding point pairs
    """
    if len(detected_circles) == 0 or len(expected_centers) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    detected_xy = detected_circles[:, :2]  # Just x, y
    
    # Estimate typical bubble spacing from expected centers
    if len(expected_centers) > 10:
        diffs = np.diff(expected_centers[:50], axis=0)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        small_dists = dists[dists < np.median(dists) * 2]
        if len(small_dists) > 0:
            typical_spacing = np.median(small_dists)
        else:
            typical_spacing = np.median(dists)
    else:
        typical_spacing = 30  # Fallback
    
    max_match_dist = typical_spacing * max_dist_ratio
    
    matched_det = []
    matched_exp = []
    used_expected = set()
    
    # For each detected circle, find nearest expected bubble
    for det in detected_xy:
        dists = np.sqrt(np.sum((expected_centers - det) ** 2, axis=1))
        nearest_idx = np.argmin(dists)
        
        if dists[nearest_idx] < max_match_dist and nearest_idx not in used_expected:
            matched_det.append(det)
            matched_exp.append(expected_centers[nearest_idx])
            used_expected.add(nearest_idx)
    
    return (
        np.array(matched_det, dtype=np.float32).reshape(-1, 2),
        np.array(matched_exp, dtype=np.float32).reshape(-1, 2)
    )


def _validate_homography(
    H: np.ndarray,
    img_shape: Tuple[int, int],
    max_scale_change: float = 0.35,
    max_shear: float = 0.25
) -> bool:
    """
    Validate that a homography is reasonable (not a crazy warp).
    """
    if H is None:
        return False
    
    h, w = img_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    try:
        warped_corners = cv.perspectiveTransform(corners, H).reshape(-1, 2)
    except:
        return False
    
    # Check corners stay in roughly same order
    sums = warped_corners.sum(axis=1)
    if np.argmin(sums) != 0:
        return False
    
    # Check area doesn't change drastically
    original_area = w * h
    warped_area = cv.contourArea(warped_corners.astype(np.float32))
    area_ratio = warped_area / original_area
    if area_ratio < (1 - max_scale_change) or area_ratio > (1 + max_scale_change):
        return False
    
    # Check angles stay roughly 90 degrees
    def angle_at_corner(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cos_angle, -1, 1))
    
    for i in range(4):
        p1 = warped_corners[(i - 1) % 4]
        p2 = warped_corners[i]
        p3 = warped_corners[(i + 1) % 4]
        angle = angle_at_corner(p1, p2, p3)
        if abs(angle - np.pi/2) > max_shear * np.pi:
            return False
    
    return True


def align_with_bubble_grid(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    bubblemap: Any,
    page_num: int = 1,
    ransac_thresh: float = 5.0,
    min_inliers: int = 30,
    hough_param1: int = 50,
    hough_param2: int = 25
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Align a scan to template using bubble grid positions from bubblemap.
    
    This function:
    1. Gets expected bubble positions from bubblemap (percentage-based, scale-independent)
    2. Detects circles in the scan using Hough transform (with contour fallback)
    3. Matches detected circles to expected positions
    4. Computes homography with RANSAC
    5. Validates transform including bounding box check to prevent row-shift errors
    
    Args:
        scan_bgr: Scan image (BGR)
        template_bgr: Template image (BGR)
        bubblemap: Bubblemap object (from bubblemap_io) or raw dict
        page_num: Which page in bubblemap to use (1-indexed)
        ransac_thresh: RANSAC reprojection threshold
        min_inliers: Minimum inliers required for valid alignment
        hough_param1: Canny threshold for Hough circles
        hough_param2: Accumulator threshold for Hough circles
        
    Returns:
        (aligned_image, homography_matrix, metrics_dict)
        If alignment fails, returns (None, None, metrics_dict_with_error)
    """
    metrics: Dict[str, Any] = {
        "method": "bubble_grid",
        "detected_circles": 0,
        "expected_bubbles": 0,
        "matched_pairs": 0,
        "inliers": 0,
        "success": False,
        "error": None
    }
    
    # Convert to grayscale
    scan_gray = _to_gray(scan_bgr)
    template_gray = _to_gray(template_bgr)
    
    tpl_h, tpl_w = template_gray.shape[:2]
    scan_h, scan_w = scan_gray.shape[:2]
    
    # Get expected bubble positions (in template coordinates)
    try:
        expected_centers, radius_pct = _get_all_bubble_centers_from_bubblemap(
            bubblemap, tpl_w, tpl_h, page_num
        )
        metrics["expected_bubbles"] = len(expected_centers)
    except Exception as e:
        metrics["error"] = f"Failed to parse bubblemap: {e}"
        return None, None, metrics
    
    if len(expected_centers) < 20:
        metrics["error"] = f"Too few bubbles in bubblemap ({len(expected_centers)})"
        return None, None, metrics
    
    # Compute expected bounding box (in template coordinates)
    exp_bbox = {
        'x_min': expected_centers[:, 0].min(),
        'x_max': expected_centers[:, 0].max(),
        'y_min': expected_centers[:, 1].min(),
        'y_max': expected_centers[:, 1].max(),
    }
    exp_bbox['width'] = exp_bbox['x_max'] - exp_bbox['x_min']
    exp_bbox['height'] = exp_bbox['y_max'] - exp_bbox['y_min']
    
    # Estimate typical row spacing from expected centers (for validation later)
    y_coords = np.sort(expected_centers[:, 1])
    y_diffs = np.diff(y_coords)
    # Filter to get actual row spacing (not within-row spacing)
    significant_diffs = y_diffs[y_diffs > radius_pct * tpl_h * 2]
    if len(significant_diffs) > 5:
        typical_row_spacing = np.median(significant_diffs)
    else:
        typical_row_spacing = exp_bbox['height'] / 50  # Fallback estimate
    
    # Estimate bubble radius in pixels
    expected_radius = radius_pct * tpl_w
    
    # Scale for scan dimensions (scan might be different size than template)
    scale_x = scan_w / tpl_w
    scale_y = scan_h / tpl_h
    avg_scale = (scale_x + scale_y) / 2
    scan_radius = expected_radius * avg_scale
    
    # Detect circles in scan - try Hough first
    detected = _detect_circles_hough(
        scan_gray,
        min_radius=scan_radius * 0.5,
        max_radius=scan_radius * 2.5,
        param1=hough_param1,
        param2=hough_param2
    )
    
    # If Hough doesn't find enough, try contour method
    if len(detected) < 100:
        detected_contour = _detect_circles_contour(
            scan_gray,
            min_radius=scan_radius * 0.5,
            max_radius=scan_radius * 2.5
        )
        # Combine both methods
        if len(detected_contour) > 0:
            if len(detected) > 0:
                detected = np.vstack([detected, detected_contour])
            else:
                detected = detected_contour
    
    metrics["detected_circles"] = len(detected)
    
    if len(detected) < 30:
        metrics["error"] = f"Only detected {len(detected)} circles (need 30+)"
        return None, None, metrics
    
    # Scale expected centers to scan coordinates for matching
    expected_scaled = expected_centers.copy()
    expected_scaled[:, 0] *= scale_x
    expected_scaled[:, 1] *= scale_y
    
    # Match detected circles to expected positions
    matched_det, matched_exp_scaled = _match_circles_to_expected(
        detected, expected_scaled, max_dist_ratio=0.45
    )
    
    metrics["matched_pairs"] = len(matched_det)
    
    if len(matched_det) < min_inliers:
        metrics["error"] = f"Only matched {len(matched_det)} pairs (need {min_inliers}+)"
        return None, None, metrics
    
    # Scale matched expected points back to template coordinates
    matched_exp_template = matched_exp_scaled.copy()
    matched_exp_template[:, 0] /= scale_x
    matched_exp_template[:, 1] /= scale_y
    
    # Compute homography: scan -> template
    H, mask = cv.findHomography(
        matched_det,             # Source points (scan coordinates)
        matched_exp_template,    # Destination points (template coordinates)
        cv.RANSAC,
        ransac_thresh
    )
    
    if H is None:
        metrics["error"] = "Homography estimation failed"
        return None, None, metrics
    
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    metrics["inliers"] = inliers
    
    if inliers < min_inliers:
        metrics["error"] = f"Only {inliers} inliers (need {min_inliers}+)"
        return None, None, metrics
    
    # Validate homography is reasonable
    if not _validate_homography(H, (scan_h, scan_w)):
        metrics["error"] = "Homography validation failed (unreasonable transform)"
        return None, None, metrics
    
    # ==========================================================================
    # BOUNDING BOX VALIDATION - Catch row/column shift errors
    # ==========================================================================
    # Transform detected circle centers to template space and check bounding box
    inlier_mask = mask.ravel().astype(bool)
    inlier_det = matched_det[inlier_mask]
    
    # Transform inlier points to template space
    inlier_det_h = np.hstack([inlier_det, np.ones((len(inlier_det), 1))])  # Homogeneous
    transformed = (H @ inlier_det_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]  # Dehomogenize
    
    # Compute bounding box of transformed points
    trans_bbox = {
        'x_min': transformed[:, 0].min(),
        'x_max': transformed[:, 0].max(),
        'y_min': transformed[:, 1].min(),
        'y_max': transformed[:, 1].max(),
    }
    
    # Check if bounding boxes roughly match
    # Allow some tolerance but catch shifts of ~1 row
    bbox_tolerance = typical_row_spacing * 0.7  # Less than one row spacing
    
    y_min_diff = abs(trans_bbox['y_min'] - exp_bbox['y_min'])
    y_max_diff = abs(trans_bbox['y_max'] - exp_bbox['y_max'])
    x_min_diff = abs(trans_bbox['x_min'] - exp_bbox['x_min'])
    x_max_diff = abs(trans_bbox['x_max'] - exp_bbox['x_max'])
    
    metrics["bbox_y_min_diff"] = y_min_diff
    metrics["bbox_y_max_diff"] = y_max_diff
    metrics["bbox_tolerance"] = bbox_tolerance
    
    # If top or bottom edge is off by more than tolerance, likely a row shift
    if y_min_diff > bbox_tolerance or y_max_diff > bbox_tolerance:
        metrics["error"] = (f"Bounding box mismatch suggests row shift: "
                          f"y_min_diff={y_min_diff:.1f}, y_max_diff={y_max_diff:.1f}, "
                          f"tolerance={bbox_tolerance:.1f}")
        return None, None, metrics
    
    # Also check x bounds (column shift)
    x_tolerance = bbox_tolerance * 1.5  # Be a bit more lenient on x
    if x_min_diff > x_tolerance or x_max_diff > x_tolerance:
        metrics["error"] = (f"Bounding box mismatch suggests column shift: "
                          f"x_min_diff={x_min_diff:.1f}, x_max_diff={x_max_diff:.1f}")
        return None, None, metrics
    
    # ==========================================================================
    # All validation passed - apply homography
    # ==========================================================================
    aligned = cv.warpPerspective(scan_bgr, H, (tpl_w, tpl_h))
    
    metrics["success"] = True
    return aligned, H, metrics


# --------------------------- Coarse-to-Fine Alignment ---------------------------

def _scale_homography(H: np.ndarray, from_scale: float, to_scale: float) -> np.ndarray:
    """
    Scale a homography matrix from one resolution to another.
    
    If H was computed at 72 DPI and we want to apply it at 300 DPI:
        H_300 = _scale_homography(H_72, from_scale=72, to_scale=300)
    
    The math: H' = S @ H @ S^-1 where S is the scaling matrix.
    """
    ratio = to_scale / from_scale
    S = np.array([
        [ratio, 0,     0],
        [0,     ratio, 0],
        [0,     0,     1]
    ], dtype=np.float64)
    S_inv = np.array([
        [1/ratio, 0,       0],
        [0,       1/ratio, 0],
        [0,       0,       1]
    ], dtype=np.float64)
    return S @ H @ S_inv


def _resize_for_dpi(img: np.ndarray, current_dpi: float, target_dpi: float) -> np.ndarray:
    """Resize image from current_dpi to target_dpi."""
    if abs(current_dpi - target_dpi) < 1:
        return img
    scale = target_dpi / current_dpi
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    interp = cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR
    return cv.resize(img, (new_w, new_h), interpolation=interp)


def align_coarse_to_fine(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    bubblemap: Any,
    page_num: int = 1,
    full_dpi: float = 300.0,
    coarse_dpi: float = 72.0,
    base_fpar: Optional[FeatureParams] = None,
    base_epar: Optional[EstParams] = None,
    base_ratio: float = 0.75,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Two-stage coarse-to-fine alignment:
    
    Stage 1 (Coarse): Fast ORB alignment at low resolution (72 DPI)
        - Gets us "in the ballpark" - handles rotation, translation, scale
        - Very fast due to small image size
    
    Stage 2 (Fine): Bubble grid refinement at full resolution
        - Now that we're close, bubble matching is unambiguous
        - Fine-tunes the alignment using known bubble positions
    
    This approach is both faster AND more reliable than either method alone:
    - ORB at low res is fast but imprecise
    - Bubble grid is precise but can match to wrong row if starting point is far off
    - Combined: fast coarse alignment makes bubble grid unambiguous
    
    Args:
        scan_bgr: Full resolution scan image (BGR)
        template_bgr: Full resolution template image (BGR)
        bubblemap: Bubblemap object for bubble grid refinement
        page_num: Which page in bubblemap (1-indexed)
        full_dpi: DPI of the input images (default 300)
        coarse_dpi: DPI for coarse alignment (default 72)
        base_fpar: FeatureParams for ORB (or None for defaults)
        base_epar: EstParams for homography (or None for defaults)
        base_ratio: Lowe's ratio test threshold
        
    Returns:
        (aligned_image, metrics_dict)
        If alignment fails, returns (None, metrics_dict_with_error)
    """
    from ..defaults import FEAT_DEFAULTS, EST_DEFAULTS
    
    if base_fpar is None:
        base_fpar = FEAT_DEFAULTS
    if base_epar is None:
        base_epar = EST_DEFAULTS
    
    metrics: Dict[str, Any] = {
        "method": "coarse_to_fine",
        "coarse_dpi": coarse_dpi,
        "full_dpi": full_dpi,
        "coarse_success": False,
        "fine_success": False,
        "success": False,
        "error": None,
    }
    
    tpl_h, tpl_w = template_bgr.shape[:2]
    scan_h, scan_w = scan_bgr.shape[:2]
    
    # ==========================================================================
    # Stage 1: Coarse ORB alignment at low resolution
    # ==========================================================================
    print(f"[info] Coarse alignment at {coarse_dpi} DPI...", file=sys.stderr)
    
    # Downsample images
    template_coarse = _resize_for_dpi(template_bgr, full_dpi, coarse_dpi)
    scan_coarse = _resize_for_dpi(scan_bgr, full_dpi, coarse_dpi)
    
    tpl_coarse_g = _to_gray(template_coarse)
    scan_coarse_g = _to_gray(scan_coarse)
    
    # Lighter ORB params for speed at low res
    coarse_fpar = apply_feat_overrides(
        tiles_x=4,
        tiles_y=5,
        topk_per_tile=80,
        orb_nfeatures=1500,
        orb_fast_threshold=15,
    )
    
    coarse_epar = apply_est_overrides(
        estimator_method=base_epar.estimator_method,
        ransac_thresh=3.0,
        max_iters=5000,
        confidence=0.99,
        use_ecc=False,  # Skip ECC at coarse stage
        ecc_levels=base_epar.ecc_levels,
        ecc_max_iters=0,
        ecc_eps=base_epar.ecc_eps,
    )
    
    try:
        # Detect and match features at coarse resolution
        k1, d1 = detect_orb_grid(tpl_coarse_g, coarse_fpar)
        k2, d2 = detect_orb_grid(scan_coarse_g, coarse_fpar)
        
        if len(k1) < 4 or len(k2) < 4:
            metrics["error"] = f"Coarse: Not enough keypoints (template={len(k1)}, scan={len(k2)})"
            return None, metrics
        
        matches = match_descriptors(d1, d2, ratio=base_ratio, cross_check=True)
        
        if len(matches) < 4:
            metrics["error"] = f"Coarse: Not enough matches ({len(matches)})"
            return None, metrics
        
        src = np.float32([k1[m.queryIdx].pt for m in matches])
        dst = np.float32([k2[m.trainIdx].pt for m in matches])
        
        H_coarse, inliers = estimate_homography(src, dst, coarse_epar)
        
        if H_coarse is None or inliers is None or inliers.sum() < 4:
            metrics["error"] = "Coarse: Homography estimation failed"
            return None, metrics
        
        metrics["coarse_inliers"] = int(inliers.sum())
        metrics["coarse_matches"] = len(matches)
        metrics["coarse_success"] = True
        
        print(f"[info] Coarse alignment: {inliers.sum()} inliers from {len(matches)} matches", file=sys.stderr)
        
    except Exception as e:
        metrics["error"] = f"Coarse alignment error: {e}"
        return None, metrics
    
    # ==========================================================================
    # Scale homography to full resolution and apply coarse warp
    # ==========================================================================
    H_coarse_full = _scale_homography(H_coarse, coarse_dpi, full_dpi)
    
    # Warp scan to get coarse-aligned version at full res
    # H_coarse maps template->scan, we need scan->template (inverse)
    try:
        H_coarse_inv = np.linalg.inv(H_coarse_full)
    except:
        H_coarse_inv = np.linalg.pinv(H_coarse_full)
    
    scan_coarse_aligned = cv.warpPerspective(scan_bgr, H_coarse_inv, (tpl_w, tpl_h))
    
    # ==========================================================================
    # Stage 2: Fine bubble grid alignment on coarse-aligned image
    # ==========================================================================
    print("[info] Fine bubble grid alignment...", file=sys.stderr)
    
    try:
        # Now bubble grid should work well - we're already close to correct alignment
        # Use tighter tolerances since we're refining, not doing initial alignment
        aligned_fine, H_fine, fine_metrics = align_with_bubble_grid(
            scan_coarse_aligned,  # Already coarse-aligned
            template_bgr,
            bubblemap,
            page_num=page_num,
            ransac_thresh=3.0,  # Tighter threshold for refinement
            min_inliers=25,
        )
        
        if aligned_fine is not None and fine_metrics.get("success", False):
            metrics["fine_success"] = True
            metrics["fine_inliers"] = fine_metrics.get("inliers", 0)
            metrics["fine_matched_pairs"] = fine_metrics.get("matched_pairs", 0)
            metrics["fine_detected_circles"] = fine_metrics.get("detected_circles", 0)
            metrics["success"] = True

            print(f"[info] Fine alignment: {fine_metrics['inliers']} inliers from {fine_metrics['matched_pairs']} matches", file=sys.stderr)

            # The final result is the fine-aligned image
            # (H_fine was applied to the coarse-aligned image)
            return aligned_fine, metrics
        else:
            # Bubble grid refinement failed - signal failure so guardrails
            # falls through to full-resolution slow-mode ORB alignment
            # (the 72 DPI coarse-only result is too imprecise for scoring)
            print(f"[info] Fine alignment failed: {fine_metrics.get('error', 'unknown')}. Will try slow alignment.", file=sys.stderr)
            metrics["fine_error"] = fine_metrics.get("error", "unknown")
            metrics["success"] = False
            metrics["method"] = "coarse_only"
            return None, metrics

    except Exception as e:
        # Bubble grid threw an exception - signal failure so guardrails
        # falls through to full-resolution slow-mode ORB alignment
        print(f"[info] Fine alignment error: {e}. Will try slow alignment.", file=sys.stderr)
        metrics["fine_error"] = str(e)
        metrics["success"] = False
        metrics["method"] = "coarse_only"
        return None, metrics


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
                          fail_br: float,
                          bubblemap: Any = None,
                          page_num: int = 1,
                          full_dpi: float = 300.0,
                          align_mode: str = "auto"):
    """
    Try alignment with multiple fallback strategies.
    
    align_mode options:
      - "fast": Coarse-to-fine (72 DPI ORB → bubble grid → ORB light → ORB heavy → quad)
                Requires bubblemap. Fast and accurate for bubble sheets.
      - "slow": Full resolution ORB (ORB light → ORB heavy → quad)
                More thorough, works without bubblemap.
      - "auto": Use "fast" if bubblemap provided, else "slow"
    
    Fallback chain for "fast" mode:
      1. Coarse-to-fine (72 DPI ORB → bubble grid refinement)
      2. ORB light at full res
      3. ORB heavy at full res  
      4. Page quadrilateral fallback
    
    Fallback chain for "slow" mode:
      1. ORB light at full res
      2. ORB heavy at full res
      3. Page quadrilateral fallback
    
    Returns (warped, metrics_dict_with_mode).
    """
    
    # Determine effective mode
    if align_mode == "auto":
        effective_mode = "fast" if bubblemap is not None else "slow"
    else:
        effective_mode = align_mode
    
    # Warn if fast mode requested but no bubblemap
    if effective_mode == "fast" and bubblemap is None:
        print("[warning] Fast alignment requested but no bubblemap provided. Falling back to slow mode.", file=sys.stderr)
        effective_mode = "slow"
    
    # ==========================================================================
    # FAST MODE: Coarse-to-fine first
    # ==========================================================================
    if effective_mode == "fast" and bubblemap is not None:
        try:
            warped_ctf, ctf_metrics = align_coarse_to_fine(
                scan_bgr, template_bgr, bubblemap,
                page_num=page_num,
                full_dpi=full_dpi,
                coarse_dpi=72.0,
                base_fpar=base_fpar,
                base_epar=base_epar,
                base_ratio=base_ratio,
            )
            if warped_ctf is not None and ctf_metrics.get("success", False):
                metrics = {
                    "mode": ctf_metrics.get("method", "coarse_to_fine"),
                    "res_median": float("nan"),
                    "res_p95": float("nan"),
                    "res_br_p95": float("nan"),
                    "coarse_inliers": ctf_metrics.get("coarse_inliers", 0),
                    "fine_inliers": ctf_metrics.get("fine_inliers", 0),
                }
                return warped_ctf, metrics
            else:
                print(f"[info] Coarse-to-fine failed: {ctf_metrics.get('error', 'unknown')}. Trying ORB...", file=sys.stderr)
        except Exception as e:
            print(f"[info] Coarse-to-fine error: {e}. Trying ORB...", file=sys.stderr)
    
    # ==========================================================================
    # ORB light (both fast and slow modes)
    # ==========================================================================
    try:
        faster_epar = apply_est_overrides(
            estimator_method=base_epar.estimator_method,
            ransac_thresh=base_epar.ransac_thresh,
            max_iters=base_epar.max_iters,
            confidence=base_epar.confidence,
            use_ecc=base_epar.use_ecc,
            ecc_levels=base_epar.ecc_levels,
            ecc_max_iters=min(getattr(base_epar, "ecc_max_iters", 50), 30),
            ecc_eps=base_epar.ecc_eps,
        )
        warped, metrics = align_page_once(template_bgr, scan_bgr, base_fpar, faster_epar, base_ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'base'
            return warped, metrics
    except Exception:
        pass

    # ==========================================================================
    # ORB heavy (both modes)
    # ==========================================================================
    ratio = max(0.68, base_ratio - 0.05)
    
    fpar = apply_feat_overrides(
        tiles_x=max(base_fpar.tiles_x, 8),
        tiles_y=max(base_fpar.tiles_y, 10),
        orb_fast_threshold=max(6, base_fpar.orb_fast_threshold - 4),
        topk_per_tile=base_fpar.topk_per_tile,
        orb_nfeatures=base_fpar.orb_nfeatures,
    )
    
    epar = apply_est_overrides(
        estimator_method=base_epar.estimator_method,
        ransac_thresh=min(base_epar.ransac_thresh, 2.0),
        max_iters=min(base_epar.max_iters, 15000),
        confidence=base_epar.confidence,
        use_ecc=base_epar.use_ecc,
        ecc_levels=base_epar.ecc_levels,
        ecc_max_iters=min(getattr(base_epar, "ecc_max_iters", 50), 25),
        ecc_eps=base_epar.ecc_eps,
    )

    try:
        warped, metrics = align_page_once(template_bgr, scan_bgr, fpar, epar, ratio)
        if not need_retry(metrics, fail_med, fail_p95, fail_br):
            metrics['mode'] = 'retry'
            return warped, metrics
    except Exception:
        pass

    # ==========================================================================
    # Last resort: Page quadrilateral fallback (+ optional ECC micro-refine)
    # ==========================================================================
    try:
        warped_q, H_inv_q = warp_by_page_quad(template_bgr, scan_bgr)
    except Exception as e:
        print(f"[warning] Page quad detection failed: {e}", file=sys.stderr)
        # Return original image if everything fails
        metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0, "mode": "failed"}
        return scan_bgr, metrics
    
    try:
        H_init = np.eye(3, dtype='float64')
        mask = make_content_mask(_to_gray(template_bgr))
        # Reduced ECC for quad fallback too
        reduced_epar = apply_est_overrides(
            estimator_method=base_epar.estimator_method,
            ransac_thresh=base_epar.ransac_thresh,
            max_iters=base_epar.max_iters,
            confidence=base_epar.confidence,
            use_ecc=base_epar.use_ecc,
            ecc_levels=base_epar.ecc_levels,
            ecc_max_iters=min(getattr(base_epar, "ecc_max_iters", 50), 20),  # Reduced
            ecc_eps=base_epar.ecc_eps,
        )
        H_ref = ecc_refine(_to_gray(template_bgr), _to_gray(warped_q), H_init, mask, reduced_epar)
        try:
            H_inv_ref = np.linalg.inv(H_ref)
        except Exception:
            H_inv_ref = np.linalg.pinv(H_ref)
        warped = cv.warpPerspective(warped_q, H_inv_ref,
                                    (_to_gray(template_bgr).shape[1], _to_gray(template_bgr).shape[0]))
    except Exception:
        warped = warped_q

    # Compute diagnostic residuals after fallback (simplified - skip if bubblemap provided)
    if bubblemap is None:
        try:
            fpar2 = apply_feat_overrides(tiles_x=6, tiles_y=8, topk_per_tile=100,
                                         orb_nfeatures=2000, orb_fast_threshold=12)
            epar2 = apply_est_overrides(estimator_method=base_epar.estimator_method, ransac_thresh=3.0,
                                        max_iters=8000, confidence=0.999,
                                        use_ecc=False, ecc_levels=base_epar.ecc_levels,
                                        ecc_max_iters=20, ecc_eps=base_epar.ecc_eps)
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
    else:
        # Skip expensive residual computation for bubble sheet workflow
        metrics = {"res_median": float("nan"), "res_p95": float("nan"), "res_br_p95": float("nan"), "n_inliers": 0}

    metrics['mode'] = 'quad_fallback'
    return warped, metrics
