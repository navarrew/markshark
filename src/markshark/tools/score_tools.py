"""
MarkShark
score_tools.py - scoring and decoding utilities for MarkShark bubble sheets.

This module implements:
- Grid center generation (in normalized coordinates).
- ROI (region of interest) creation for bubble centers.
- Page binarization and per-ROI fill scoring.
- Simple decision rules to select a single bubble per row or per column.
- Helpers to decode text/ID layouts and answer layouts from an aligned page.

Conventions:
- Layout coordinates are normalized fractions of width/height (0..1).
- A layout grid is always shaped (rows=layout.numrows, cols=layout.numcols).
- `selection_axis == "row"`: select one column per row (typical for answers).
- `selection_axis == "col"`: select one row per column (typical for names/ID).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Dict

import cv2
import numpy as np

from ..bublmap_io import Bubblemap, GridLayout
from ..defaults import SCORING_DEFAULTS


# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------

def grid_centers_axis_mode(*args, **kwargs) -> List[Tuple[float, float]]:
    """Return normalized (x, y) centers for a rows×cols grid.

    Supports two call styles for backward compatibility:

    Positional (legacy):
        grid_centers_axis_mode(x0, y0, x1, y1, rows, cols)

    Keyword (preferred):
        grid_centers_axis_mode(
            w=<int>, h=<int>,                 # accepted but not used
            x0_pct=<float>, y0_pct=<float>,
            x1_pct=<float>, y1_pct=<float>,
            numrows=<int>, numcols=<int>,
            axis=<str>,                       # accepted but not used for geometry
        )

    The returned centers are always normalized fractions (0..1), suitable for
    `centers_to_circle_rois(..., img_w, img_h, ...)`.
    """
    if args and not kwargs:
        if len(args) != 6:
            raise TypeError("grid_centers_axis_mode expects 6 positional args: x0, y0, x1, y1, rows, cols")
        x0, y0, x1, y1, rows, cols = args
    else:
        # Accept and ignore w/h/axis, they are part of the public API in callers.
        x0 = kwargs.get("x0_pct", kwargs.get("x_tl", kwargs.get("x_topleft")))
        y0 = kwargs.get("y0_pct", kwargs.get("y_tl", kwargs.get("y_topleft")))
        x1 = kwargs.get("x1_pct", kwargs.get("x_br", kwargs.get("x_bottomright")))
        y1 = kwargs.get("y1_pct", kwargs.get("y_br", kwargs.get("y_bottomright")))
        rows = kwargs.get("numrows", kwargs.get("rows", kwargs.get("questions")))
        cols = kwargs.get("numcols", kwargs.get("cols", kwargs.get("choices")))
        if x0 is None or y0 is None or x1 is None or y1 is None or rows is None or cols is None:
            raise TypeError(
                "grid_centers_axis_mode missing required args. Provide either 6 positional args or keyword args: "
                "x0_pct,y0_pct,x1_pct,y1_pct,numrows,numcols."
            )

    rows_i = int(rows)
    cols_i = int(cols)
    if rows_i <= 0 or cols_i <= 0:
        return []

    centers: List[Tuple[float, float]] = []
    r_den = max(1, rows_i - 1)
    c_den = max(1, cols_i - 1)

    for r in range(rows_i):
        v = r / r_den
        y = float(y0) + (float(y1) - float(y0)) * v
        for c in range(cols_i):
            u = c / c_den
            x = float(x0) + (float(x1) - float(x0)) * u
            centers.append((x, y))
    return centers


def centers_to_circle_rois(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int,
    img_h: int,
    radius_pct: float,
) -> List[Tuple[int, int, int, int]]:
    """Convert normalized centers into pixel-space square ROIs.

    Each ROI is a square bounding box for a circle of radius
    `radius_pct * img_w` (radius based on width, matching prior behavior).
    """
    rois: List[Tuple[int, int, int, int]] = []
    if img_w <= 0 or img_h <= 0:
        return rois

    r_px = max(1.0, float(radius_pct) * float(img_w))

    for cx_pct, cy_pct in centers_pct:
        cx = float(cx_pct) * img_w
        cy = float(cy_pct) * img_h
        x = int(round(cx - r_px))
        y = int(round(cy - r_px))
        w = h = int(round(2 * r_px))

        # Clamp to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y

        rois.append((x, y, w, h))

    return rois


# ------------------------------------------------------------------------------
# Scoring primitives
# ------------------------------------------------------------------------------

_circle_mask_cache: Dict[Tuple[int, int, int], np.ndarray] = {}


def circle_mask(w: int, h: int, radius_px: Optional[int] = None) -> np.ndarray:
    """Return a boolean mask for a centered circle within an ROI.

    Backward compatible: if radius_px is omitted, uses radius ~= 0.48 * min(w, h),
    matching older MarkShark scoring defaults.
    """
    if radius_px is None:
        radius_px = max(1, int(0.48 * min(w, h)))
    key = (w, h, int(radius_px))
    m = _circle_mask_cache.get(key)
    if m is not None:
        return m

    cx = w // 2
    cy = h // 2
    yy, xx = np.ogrid[:h, :w]
    m = (xx - cx) ** 2 + (yy - cy) ** 2 <= int(radius_px) ** 2
    _circle_mask_cache[key] = m
    return m


def measure_fill_ratio(
    thresh_img: np.ndarray,
    rect: Tuple[int, int, int, int],
    *,
    shape: str = "circle",
    inner_radius_ratio: float = 0.85,
) -> float:
    """Compute fill ratio (0..1) inside an ROI from a binary-inverted page.

    `thresh_img` should be a binary-inverted image where ink is white (255).

    For `shape="circle"`, only pixels inside a centered circle are counted,
    with radius = 0.5 * min(w, h) * inner_radius_ratio.
    """
    x, y, w, h = rect
    roi = thresh_img[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    if shape == "circle":
        H, W = roi.shape[:2]
        r = int(0.5 * min(W, H) * float(inner_radius_ratio))
        r = max(1, r)
        mask = circle_mask(W, H, r)
        return float(np.mean(roi[mask] > 0))

    return float(np.mean(roi > 0))


def roi_fill_scores(
    gray: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    *,
    inner_radius_ratio: float = 0.85,
    blur_ksize: int = 3,
    bin_method: str = "global",          # "adaptive" or "global"
    block_size: int = 35,                # odd; for adaptive threshold
    C: int = 8,                          # subtractive constant for adaptive
    fixed_thresh: Optional[int] = None,  # for global threshold
) -> List[float]:
    """Compute per-ROI fill scores (0..1) for a page."""
    if fixed_thresh is None:
        fixed_thresh = int(SCORING_DEFAULTS.fixed_thresh)

    # Optional denoise
    if blur_ksize and blur_ksize > 1:
        k = int(blur_ksize) | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Binarize once for the whole page (white = ink)
    if str(bin_method).lower().startswith("adap"):
        bs = max(3, int(block_size) | 1)
        thresh_img = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs,
            int(C),
        )
    else:
        _, thresh_img = cv2.threshold(gray, int(fixed_thresh), 255, cv2.THRESH_BINARY_INV)

    scores: List[float] = []
    for rect in rois:
        scores.append(measure_fill_ratio(thresh_img, rect, shape="circle", inner_radius_ratio=inner_radius_ratio))
    return scores




# ------------------------------------------------------------------------------
# Per-page fixed_thresh calibration
# ------------------------------------------------------------------------------

def calibrate_fixed_thresh_for_page(
    img_bgr: np.ndarray,
    bmap: Bubblemap,
    *,
    fixed_thresh_center: Optional[int] = None,
    fixed_thresh_spread: int = 50,
    fixed_thresh_step: int = 5,
    top2_ratio: float = 0.80,
    min_gap: float = 0.07,
    min_rows: int = 10,
    verbose: bool = False,
) -> Tuple[int, Dict[str, float]]:
    """
    Choose a per-page fixed_thresh value that maximizes separation between
    "winner" bubbles and "non-winner" bubbles on that page.

    This is intended to handle students who mark lightly (winner scores drift
    down toward the printed-circle baseline) by increasing fixed_thresh only
    when it improves winner versus non-winner separation.

    Strategy:
      - Use answer layouts with selection_axis == "row" only.
      - For each candidate fixed_thresh, compute per-row best and second-best.
      - Keep rows that look like a clear single selection (second/best <= top2_ratio
        and best-second >= min_gap).
      - Objective Q = P10(best_scores) - P90(nonwinner_scores).
      - Pick fixed_thresh with the largest Q.

    Returns:
      (best_fixed_thresh, stats_dict)

    stats_dict keys:
      - q: separation score used for selection (higher is better)
      - rows_used: number of rows used in calibration
      - winners_p10: P10(best_scores) for chosen threshold
      - nonwinners_p90: P90(nonwinner_scores) for chosen threshold
    """
    import sys

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    if fixed_thresh_center is None:
        fixed_thresh_center = int(SCORING_DEFAULTS.fixed_thresh)

    lo = max(0, int(fixed_thresh_center) - int(fixed_thresh_spread))
    hi = min(255, int(fixed_thresh_center) + int(fixed_thresh_spread))
    step = max(1, int(fixed_thresh_step))

    candidates = list(range(lo, hi + 1, step))
    if fixed_thresh_center not in candidates:
        candidates.append(int(fixed_thresh_center))
        candidates = sorted(set(candidates))

    # Build one combined ROI list for all answer layouts, and per-row slices.
    rois_all: List[Tuple[int, int, int, int]] = []
    row_groups: List[Tuple[int, int]] = []  # (start_index, numcols)

    def _layout_rois(layout: GridLayout) -> List[Tuple[int, int, int, int]]:
        x0 = getattr(layout, "x0_pct", getattr(layout, "x_topleft"))
        y0 = getattr(layout, "y0_pct", getattr(layout, "y_topleft"))
        x1 = getattr(layout, "x1_pct", getattr(layout, "x_bottomright"))
        y1 = getattr(layout, "y1_pct", getattr(layout, "y_bottomright"))
        centers = grid_centers_axis_mode(
            w=W, h=H,
            x0_pct=float(x0), y0_pct=float(y0),
            x1_pct=float(x1), y1_pct=float(y1),
            numrows=int(layout.numrows), numcols=int(layout.numcols),
            axis=str(layout.selection_axis),
        )
        return centers_to_circle_rois(centers, W, H, float(layout.radius_pct))

    for layout in (bmap.answer_layouts or []):
        if str(getattr(layout, "selection_axis", "row")).lower() != "row":
            continue
        base = len(rois_all)
        rois = _layout_rois(layout)
        rois_all.extend(rois)
        # Row-major ordering: each row is a contiguous run of length layout.numcols
        for r in range(int(layout.numrows)):
            row_groups.append((base + r * int(layout.numrows), int(layout.numcols)))

    stats_default: Dict[str, float] = {
        "q": float("nan"),
        "rows_used": 0.0,
        "winners_p10": float("nan"),
        "nonwinners_p90": float("nan"),
    }

    if not rois_all or not row_groups:
        if verbose:
            print("[CALIBRATE] no usable answer layouts for calibration, using center fixed_thresh", file=sys.stderr)
        return int(fixed_thresh_center), stats_default

    best_thresh = int(fixed_thresh_center)
    best_q = -1e9
    best_stats = dict(stats_default)

    for th in candidates:
        scores = roi_fill_scores(gray, rois_all, fixed_thresh=int(th))

        winners: List[float] = []
        nonwinners: List[float] = []

        for start, choices in row_groups:
            row = scores[start : start + choices]
            if not row:
                continue

            best_idx = int(np.argmax(row))
            best_val = float(row[best_idx])

            # Find second best
            if choices >= 2:
                # small list, so sort is fine
                order = np.argsort(row)[::-1]
                second_val = float(row[int(order[1])])
            else:
                second_val = 0.0

            if best_val <= 0.0:
                continue

            ratio = (second_val / best_val) if best_val > 1e-9 else 1.0
            gap = best_val - second_val

            if ratio <= float(top2_ratio) and gap >= float(min_gap):
                winners.append(best_val)
                for j, v in enumerate(row):
                    if j != best_idx:
                        nonwinners.append(float(v))

        if len(winners) < int(min_rows) or len(nonwinners) < int(min_rows):
            continue

        w_p10 = float(np.percentile(winners, 10))
        n_p90 = float(np.percentile(nonwinners, 90))
        q = w_p10 - n_p90

        # Tie-breaker: prefer thresholds closer to the center value.
        if (q > best_q) or (abs(q - best_q) < 1e-12 and abs(int(th) - int(fixed_thresh_center)) < abs(best_thresh - int(fixed_thresh_center))):
            best_q = q
            best_thresh = int(th)
            best_stats = {
                "q": float(q),
                "rows_used": float(len(winners)),
                "winners_p10": float(w_p10),
                "nonwinners_p90": float(n_p90),
            }

    if verbose:
        if best_q <= -1e8:
            print(f"[CALIBRATE] insufficient confident rows, using center fixed_thresh={fixed_thresh_center}", file=sys.stderr)
        else:
            print(
                f"[CALIBRATE] fixed_thresh={best_thresh} q={best_stats['q']:.4f} rows_used={int(best_stats['rows_used'])} "
                f"winners_p10={best_stats['winners_p10']:.3f} nonwinners_p90={best_stats['nonwinners_p90']:.3f}",
                file=sys.stderr,
            )

    if best_q <= -1e8:
        return int(fixed_thresh_center), stats_default

    return int(best_thresh), best_stats


def _pick_single_from_scores(
    best_first: np.ndarray,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
) -> Optional[int]:
    """Pick a single winner index from a 1D vector of scores.

    Returns None for blank or ambiguous (multi-mark) cases.
    """
    if best_first.size == 0:
        return None

    order = np.argsort(best_first)[::-1]
    best_idx = int(order[0])
    top = float(best_first[best_idx])
    second = float(best_first[int(order[1])]) if best_first.size > 1 else 0.0

    # Blank rule: require a minimum absolute fill
    if top < (float(min_fill)):
        return None

    # Separation between best bubble and second best bubble rules
    sep_score = (top - second) * 100.0           # absolute gap in percentage points between top and next best
    sep_ratio_ok = (second <= top * float(top2_ratio))  # ratio separation between top and next best bubbles

    if (sep_score >= float(min_score)) or sep_ratio_ok:
        return best_idx

    return None



def scores_to_labels_row(
    scores: List[float],
    rows: int,
    cols: int,
    choice_labels: List[str],
    *,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
    multi_top_k: int = 2,
    multi_delim: str = ",",
) -> List[Optional[str]]:
    """Convert per-ROI scores into per-row labels.

    Behavior:
      - Blank row (top score < min_fill): returns None
      - Clear single mark: returns a single label (e.g., "A")
      - Ambiguous / multi-mark row: returns the top K labels joined by multi_delim (e.g., "A,C")

    The "clear single" vs "multi" decision matches _pick_single_from_scores():
      a single winner is accepted if either:
        - absolute separation >= min_score (in percentage points), OR
        - ratio separation: second <= top * top2_ratio
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size != int(rows) * int(cols):
        raise ValueError(f"scores length {arr.size} != rows*cols {rows*cols}")

    out: List[Optional[str]] = []
    cols_i = int(cols)
    k = max(2, int(multi_top_k)) if int(cols) > 1 else 1

    for r in range(int(rows)):
        row_slice = arr[r * cols_i : (r + 1) * cols_i]
        if row_slice.size == 0:
            out.append(None)
            continue

        order = np.argsort(row_slice)[::-1]
        best_idx = int(order[0])
        top = float(row_slice[best_idx])

        if top < float(min_fill):
            out.append(None)
            continue

        # If there's only one choice, it's always a single mark when above min_fill.
        if cols_i <= 1:
            out.append(choice_labels[best_idx] if best_idx < len(choice_labels) else None)
            continue

        second_idx = int(order[1])
        second = float(row_slice[second_idx])

        # If the runner-up is below min_fill, we treat as a single mark.
        if second < float(min_fill):
            out.append(choice_labels[best_idx] if best_idx < len(choice_labels) else None)
            continue

        sep_score = (top - second) * 100.0
        sep_ratio_ok = (second <= top * float(top2_ratio))

        if (sep_score >= float(min_score)) or sep_ratio_ok:
            out.append(choice_labels[best_idx] if best_idx < len(choice_labels) else None)
            continue

        # Ambiguous: return top-K labels (default: top 2)
        picks = []
        for j in range(min(k, order.size)):
            idx = int(order[j])
            if 0 <= idx < len(choice_labels):
                picks.append(choice_labels[idx])
        out.append(multi_delim.join(picks) if picks else None)

    return out



def select_per_row(
    scores: List[float],
    rows: int,
    cols: int,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
) -> List[Optional[int]]:
    """For each row, pick one column index or None."""
    arr = np.asarray(scores, dtype=float)
    if arr.size != int(rows) * int(cols):
        raise ValueError(f"scores length {arr.size} != rows*cols {rows*cols}")

    picked: List[Optional[int]] = []
    cols_i = int(cols)
    for r in range(int(rows)):
        row_slice = arr[r * cols_i : (r + 1) * cols_i]
        picked.append(_pick_single_from_scores(row_slice, min_fill, top2_ratio, min_score))
    return picked


def select_per_col(
    scores: List[float],
    rows: int,
    cols: int,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
) -> List[Optional[int]]:
    """For each column, pick one row index or None."""
    arr = np.asarray(scores, dtype=float)
    if arr.size != int(rows) * int(cols):
        raise ValueError(f"scores length {arr.size} != rows*cols {rows*cols}")

    picked: List[Optional[int]] = []
    cols_i = int(cols)
    for c in range(cols_i):
        col_slice = arr[c::cols_i]
        picked.append(_pick_single_from_scores(col_slice, min_fill, top2_ratio, min_score))
    return picked


# ------------------------------------------------------------------------------
# Zone decoding
# ------------------------------------------------------------------------------

def decode_layout(
    gray: np.ndarray,
    layout: GridLayout,
    *,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
    fixed_thresh: Optional[int] = None,
) -> Tuple[List[Optional[int]], List[Tuple[int, int, int, int]], List[float]]:
    """Decode a single GridLayout, returning (picked, rois, scores)."""
    h, w = gray.shape[:2]

    x0 = getattr(layout, "x0_pct", getattr(layout, "x_topleft"))
    y0 = getattr(layout, "y0_pct", getattr(layout, "y_topleft"))
    x1 = getattr(layout, "x1_pct", getattr(layout, "x_bottomright"))
    y1 = getattr(layout, "y1_pct", getattr(layout, "y_bottomright"))

    centers = grid_centers_axis_mode(
        w=w,
        h=h,
        x0_pct=x0,
        y0_pct=y0,
        x1_pct=x1,
        y1_pct=y1,
        numrows=layout.numrows,
        numcols=layout.numcols,
        axis=layout.selection_axis,
    )

    rois = centers_to_circle_rois(centers, w, h, layout.radius_pct)
    scores = roi_fill_scores(gray, rois, fixed_thresh=fixed_thresh)

    if layout.selection_axis == "row":
        picked = select_per_row(scores, layout.numrows, layout.numcols, min_fill, top2_ratio, min_score)
    else:
        picked = select_per_col(scores, layout.numrows, layout.numcols, min_fill, top2_ratio, min_score)

    return picked, rois, scores


def indices_to_labels_row(
    picked: List[Optional[int]],
    choices: int,
    choice_labels: List[str],
) -> List[Optional[str]]:
    """Map per-row picked column index to a label."""
    out: List[Optional[str]] = []
    for idx in picked:
        if idx is None or idx < 0 or idx >= int(choices):
            out.append(None)
        else:
            out.append(choice_labels[int(idx)] if int(idx) < len(choice_labels) else None)
    return out


def indices_to_text_col(picked: List[Optional[int]], row_labels: str) -> str:
    """Map per-column picked row index to a character."""
    blank = " "
    chars: List[str] = []
    for idx in picked:
        if idx is None or idx < 0 or idx >= len(row_labels):
            chars.append(blank)
        else:
            chars.append(row_labels[int(idx)])
    return "".join(chars)


# ------------------------------------------------------------------------------
# Key handling & scoring
# ------------------------------------------------------------------------------

def load_key_txt(path: str) -> List[str]:
    """Load an answer key from a text file, one letter per line."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chars = [c for c in raw if c.isalpha()]
    return [c.upper() for c in chars]


def score_against_key(selections: List[Optional[str]], key: List[str]) -> Tuple[int, int]:
    """Return (correct, total) when comparing selections to key."""
    total = min(len(selections), len(key))
    correct = sum(1 for a, k in zip(selections[:total], key[:total]) if a is not None and a == k)
    return correct, total


# ------------------------------------------------------------------------------
# Page processing
# ------------------------------------------------------------------------------

def process_page_all(
    img_bgr: np.ndarray,
    bmap: Bubblemap,
    *,
    min_fill: float = SCORING_DEFAULTS.min_fill,
    top2_ratio: float = SCORING_DEFAULTS.top2_ratio,
    min_score: float = SCORING_DEFAULTS.min_score,
    fixed_thresh: Optional[int] = None,
) -> Tuple[dict, List[Optional[str]]]:
    """Decode names/ID/version and all answers from an aligned page."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    info = {"last_name": "", "first_name": "", "student_id": "", "version": ""}

    if bmap.last_name_layout:
        picked, _, _ = decode_layout(
            gray,
            bmap.last_name_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["last_name"] = indices_to_text_col(
            picked, bmap.last_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()

    if bmap.first_name_layout:
        picked, _, _ = decode_layout(
            gray,
            bmap.first_name_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["first_name"] = indices_to_text_col(
            picked, bmap.first_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()

    if bmap.id_layout:
        picked, _, _ = decode_layout(
            gray,
            bmap.id_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["student_id"] = indices_to_text_col(picked, bmap.id_layout.labels or "0123456789").strip()

    if bmap.version_layout:
        picked, _, _ = decode_layout(
            gray,
            bmap.version_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        if bmap.version_layout.selection_axis == "row":
            idx = picked[0] if picked else None
            labels = list(bmap.version_layout.labels or "ABCD")
            info["version"] = labels[idx] if idx is not None and 0 <= idx < len(labels) else ""
        else:
            info["version"] = indices_to_text_col(picked, bmap.version_layout.labels or "ABCD").strip()

    answers: List[Optional[str]] = []
    for layout in bmap.answer_layouts:
        picked, _, scores = decode_layout(
            gray,
            layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        choice_labels = list(layout.labels) if layout.labels else [chr(ord("A") + k) for k in range(layout.numcols)]
        if layout.selection_axis == "row":
            answers.extend(
                scores_to_labels_row(
                    scores,
                    layout.numrows,
                    layout.numcols,
                    choice_labels,
                    min_fill=min_fill,
                    top2_ratio=top2_ratio,
                    min_score=min_score,
                )
            )
        else:
            answers.extend(indices_to_labels_row(picked, layout.numcols, choice_labels))

    return info, answers
