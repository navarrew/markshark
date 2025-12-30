#!/usr/bin/env python3
"""
MarkShark
score_tools.py
---------------
Reusable scoring primitives:

Axis-based grading for MarkShark sheets (answers + names + ID + version).

Uses config_io.Config with per-layout axis definitions:
  x_topleft, y_topleft, x_bottomright, y_bottomright, radius_pct,
  questions (rows), choices (cols), selection_axis ("row" or "col"), labels (for rows).

Decoding behavior:
- selection_axis == "row":    pick ONE column (choice) per row  (answers, version-as-row).
- selection_axis == "col":    pick ONE row (label) per column  (last/first name, ID).

Outputs CSV with columns:
  page_index, LastName, FirstName, StudentID, Version, Q1..Qn, [score, total if key provided]
"""


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

from __future__ import annotations
from ..defaults import SCORING_DEFAULTS
from typing import List, Tuple, Iterable, Optional

import numpy as np
import cv2

from ..config_io import GridLayout, Config

# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------

r"""
grid_centers_axis_mode — Computes normalized (x%, y%) centers for a rows×cols grid
by linearly interpolating between the given top-left and bottom-right anchor points.
Returns a list of per-cell center coordinates used to locate bubbles.
"""
def grid_centers_axis_mode(
    x_tl: float, y_tl: float, x_br: float, y_br: float,
    rows: int, cols: int
) -> List[Tuple[float, float]]:
    """
    Return normalized (x%, y%) centers for a rows x cols grid
    by interpolating between top-left and bottom-right bubble centers.
    """
    centers: List[Tuple[float, float]] = []
    r_den = max(1, rows - 1)
    c_den = max(1, cols - 1)
    for r in range(rows):
        v = r / r_den
        y = y_tl + (y_br - y_tl) * v
        for c in range(cols):
            u = c / c_den
            x = x_tl + (x_br - x_tl) * u
            centers.append((x, y))
    return centers



"""
centers_to_circle_rois — Converts normalized centers to pixel-space square ROIs sized 
from radius_pct×image_width, clamping each ROI to image bounds. 
Returns a list of (x, y, w, h) tuples for cropping.
"""
def centers_to_circle_rois(
    centers_pct: Iterable[Tuple[float, float]],
    img_w: int, img_h: int,
    radius_pct: float
) -> List[Tuple[int, int, int, int]]:
    r_px = max(1.0, radius_pct * img_w)
    rois: List[Tuple[int, int, int, int]] = []
    for (cxp, cyp) in centers_pct:
        cx = float(cxp) * img_w
        cy = float(cyp) * img_h
        x = int(round(cx - r_px))
        y = int(round(cy - r_px))
        w = h = int(round(2 * r_px))
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        rois.append((x, y, w, h))
    return rois



"""
circle_mask — Builds a filled circular mask matching a given ROI width/height
centered and radius = half the shorter side. Used to restrict scoring to the 
inner disk of each bubble.
"""
def circle_mask(w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w / 2.0, h / 2.0
    r = min(w, h) / 2.0
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


# ------------------------------------------------------------------------------
# Scoring primitives
# ------------------------------------------------------------------------------
def measure_fill_ratio(thresh_img: np.ndarray, rect: Tuple[int,int,int,int], shape: str="circle") -> float:
    """
    Fraction of white (255) pixels within rect/circle in a binary-inverted image (white=ink).
    """
    x, y, w, h = rect
    roi = thresh_img[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0

    if shape == "circle":
        H, W = roi.shape[:2]
        cx = W // 2
        cy = H // 2
        r = int(min(W, H) * 0.48)
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        filled = np.count_nonzero((roi > 0) & mask)
        total = np.count_nonzero(mask)
    else:
        filled = cv2.countNonZero(roi)
        total = roi.size

    return float(filled) / max(1.0, float(total))


"""
roi_fill_scores — This function converts a grayscale scan into a binary-inverted image 
(white = ink) using either global or adaptive thresholding, optionally smoothing it with
a Gaussian blur first.  It then iterates over each bubble region (ROI) and calls
measure_fill_ratio() to calculate the fraction of white pixels inside a circular mask 
for that region. The result is a list of per-bubble fill scores between 0 and 1,
representing how completely each bubble is marked.

If your scans are of good quality you should use 'global' thresholding.

Adaptive thresholding computes a different threshold value for each small region
(or “block”) of the image instead of one global cutoff for the entire page.
This lets it automatically adjust for uneven lighting, shadows, or shading — so even 
if the left side of a scan is slightly darker than the right, the algorithm still 
correctly distinguishes pencil marks from paper everywhere.

Adaptive process:
It looks at a local square window of size blockSize × blockSize around a pixel.
It computes either the mean or a Gaussian-weighted mean of that neighborhood.
It subtracts a constant C from that mean to get a local threshold value.
If the pixel’s intensity is below that threshold, it becomes white (255)
after inversion (i.e., marked as “ink”).

Adaptive parameters:
blockSize controls how local the decision is (smaller = more sensitive to local shadows).
C adjusts the sensitivity: lower C includes lighter grays as ink; higher C is stricter.
Result: you get a binary-inverted image that adapts to lighting changes across the page.
uneven lighting may be an issue if you used a camera to generate your scan. Images from
any decent scanner, however, should not need these lighting adjustments.
"""

def roi_fill_scores(
    gray: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    inner_radius_ratio: float = 0.85,
    blur_ksize: int = 3,
    bin_method: str = "global",          # "adaptive" or "global"
    block_size: int = 35,                # odd; for adaptive threshold
    C: int = 8,                          # subtractive constant for adaptive
    fixed_thresh: Optional[int] = None,  # for global threshold (bin_method == "global")
) -> List[float]:
    """Compute per-ROI fill scores for a page.

    The page is binarized once (adaptive or global threshold, with inversion so white equals ink),
    then each ROI is scored by measuring the filled fraction inside the ROI.
    """
    if fixed_thresh is None:
        fixed_thresh = SCORING_DEFAULTS.fixed_thresh



    H, W = gray.shape[:2]

    # Optional denoise
    if blur_ksize and blur_ksize > 1:
        k = int(blur_ksize) | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Make one binary-inverted image for the whole page
    if bin_method.lower().startswith("adap"):
        bs = block_size | 1
        thresh_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs, C
        )
    else:
        _, thresh_img = cv2.threshold(gray, fixed_thresh, 255, cv2.THRESH_BINARY_INV)

    scores: List[float] = []
    for rect in rois:
        score = measure_fill_ratio(thresh_img, rect, shape="circle")
        scores.append(score)

    return scores



"""
_pick_single_from_scores — Given candidate scores for one row/column, enforces blank and
separation criteria: minimum absolute fill (min_fill/min_abs) and either an absolute gap
(min_score, in percentage-points) or a ratio gap (top2_ratio). Returns the winning index
or None for blank/ambiguous (multi-mark) cases.
"""
def _pick_single_from_scores(best_first: np.ndarray,
                             min_fill: float = SCORING_DEFAULTS.min_fill, top2_ratio: float = SCORING_DEFAULTS.top2_ratio, min_score: float = SCORING_DEFAULTS.min_score, min_abs: float = SCORING_DEFAULTS.min_abs) -> Optional[int]:
    """
    best_first: 1D array of scores for the candidates (higher is darker).
    Must be length >= 1. We assume it's the set for one row (or one column).
    Returns: winning index or None (blank/multi).
    """
    if best_first.size == 0:
        return None

    # Indices that would sort descending
    order = np.argsort(best_first)[::-1]
    best_idx = int(order[0])
    top = float(best_first[best_idx])

    second = float(best_first[order[1]]) if best_first.size > 1 else 0.0

    # Blank rule: require a minimum absolute fill
    if top < max(min_fill, min_abs):
        return None  # blank

    # Separation rules
    sep_score = (top - second) * 100.0           # absolute gap in percentage points
    sep_ratio_ok = (second <= top * top2_ratio)  # ratio separation

    if (sep_score >= min_score) or sep_ratio_ok:
        return best_idx
    else:
        return None  # multi/ambiguous



"""
select_per_row — Splits a flat score array (row-major) into rows and 
invokes _pick_single_from_scores to choose exactly one column per row. 
Produces a list of selected column indices (or None) for each row.
"""
def select_per_row(scores: List[float],
                   rows: int,
                   cols: int,
                   min_fill: float = SCORING_DEFAULTS.min_fill, top2_ratio: float = SCORING_DEFAULTS.top2_ratio, min_score: float = SCORING_DEFAULTS.min_score, min_abs: float = SCORING_DEFAULTS.min_abs ) -> List[Optional[int]]:
    """
    For each row (length=cols), pick ONE column index or None using the combined rule.
    `scores` is a flat list in row-major order: [r0c0, r0c1, ..., r0c{cols-1}, r1c0, ...]
    """
    arr = np.asarray(scores, dtype=float)
    assert arr.size == rows * cols, f"scores length {arr.size} != rows*cols {rows*cols}"

    picked: List[Optional[int]] = []
    for r in range(rows):
        row_slice = arr[r * cols:(r + 1) * cols]
        win = _pick_single_from_scores(row_slice, min_fill, top2_ratio, min_score, min_abs)
        picked.append(win)
    return picked



"""
select_per_col — Takes the same flat score array and processes by columns (stride slicing)
to choose one row per column. Produces a list of selected row indices (or None)
for each column.
"""
def select_per_col(scores: List[float],
                   rows: int,
                   cols: int,
                   min_fill: float = SCORING_DEFAULTS.min_fill, top2_ratio: float = SCORING_DEFAULTS.top2_ratio, min_score: float = SCORING_DEFAULTS.min_score, min_abs: float = SCORING_DEFAULTS.min_abs ) -> List[Optional[int]]:
    """
    For each column (length=rows), pick ONE row index or None using the combined rule.
    `scores` is a flat list in row-major order.
    """
    arr = np.asarray(scores, dtype=float)
    assert arr.size == rows * cols, f"scores length {arr.size} != rows*cols {rows*cols}"

    picked: List[Optional[int]] = []
    for c in range(cols):
        col_slice = arr[c::cols]  # take every `cols` element starting at c
        win = _pick_single_from_scores(col_slice, min_fill, top2_ratio, min_score, min_abs)
        picked.append(win)
    return picked


# ------------------------------------------------------------------------------
# Zone decoders
# ------------------------------------------------------------------------------

"""
decode_layout — Computes bubble centers and ROIs for a GridLayout, scores them with 
roi_fill_scores, and runs either row-wise or column-wise selection based on 
selection_axis. Returns the picks, the ROIs, and the raw fill scores.
"""
def decode_layout(
    gray: np.ndarray, layout: GridLayout,
    min_fill: float = SCORING_DEFAULTS.min_fill, top2_ratio: float = SCORING_DEFAULTS.top2_ratio, min_score: float = SCORING_DEFAULTS.min_score, min_abs: float = SCORING_DEFAULTS.min_abs
):
    """
    Decode one layout according to its selection_axis.
    Returns (selected_indices, rois, scores).
    - If selection_axis == "row": indices are column indices per row (len = rows)
    - If selection_axis == "col": indices are row indices per column (len = cols)
    """
    h, w = gray.shape[:2]
    centers = grid_centers_axis_mode(
        layout.x_topleft, layout.y_topleft,
        layout.x_bottomright, layout.y_bottomright,
        layout.questions, layout.choices
    )
    rois = centers_to_circle_rois(centers, w, h, layout.radius_pct)
    scores = roi_fill_scores(gray, rois, ...)

    if layout.selection_axis == "row":
        picked = select_per_row(scores, layout.questions, layout.choices, min_fill, top2_ratio, min_score, min_abs)
    else:
        picked = select_per_col(scores, layout.questions, layout.choices, min_fill, top2_ratio, min_score, min_abs)

    return picked, rois, scores



"""
indices_to_labels_row — Maps per-row selected column indices to choice labels
(e.g., A/B/C/…), with bounds checks funneling invalids to None. Returns a list
of label strings or None.
"""
def indices_to_labels_row(picked: List[Optional[int]], choices: int, choice_labels: List[str]) -> List[Optional[str]]:
    """Map per-row selected column index → label (A.. or from provided)."""
    out: List[Optional[str]] = []
    for idx in picked:
        out.append(choice_labels[idx] if idx is not None and 0 <= idx < choices else None)
    return out



"""
indices_to_text_col — Maps per-column selected row indices to characters using a provided
label string (e.g., alphabet/numeric for names/ID). Concatenates into a single text field,
inserting empty strings for blanks/invalids.
"""
def indices_to_text_col(picked: List[Optional[int]], row_labels: str) -> str:
    """Map per-column selected row index → character; None → blank."""
    chars: List[str] = []
    for idx in picked:
        if idx is None or idx < 0 or idx > len(row_labels):
            chars.append("")
        else:
            chars.append(row_labels[idx])
    return "".join(chars)



# ------------------------------------------------------------------------------
# Key handling & scoring
# ------------------------------------------------------------------------------

"""
load_key_txt — Reads a text file, filters to alphabetic characters only, 
uppercases them, and returns as a list. 
This yields the per-question answer key (e.g., ['A','C','B',…]).
"""
def load_key_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    chars = [c for c in raw if c.isalpha()]
    return [c.upper() for c in chars]




"""
score_against_key — Compares a list of selected answers (with None treated as incorrect)
to the key up to the shorter length. Returns (correct, total).
"""
def score_against_key(selections: List[Optional[str]], key: List[str]) -> Tuple[int, int]:
    correct = 0
    total = min(len(selections), len(key))
    for i in range(total):
        if selections[i] is not None and selections[i] == key[i]:
            correct += 1
    return correct, total


# ------------------------------------------------------------------------------
# Page processing
# ------------------------------------------------------------------------------

"""
process_page_all — Orchestrates decoding of last name, first name, student ID, version 
(handling row- vs column-mode layouts), and all answer layouts for one page.
Returns an info dict and a flattened list of per-question answer labels.
"""
def process_page_all(
    img_bgr: np.ndarray,
    cfg: Config,
    min_fill: float = SCORING_DEFAULTS.min_fill, top2_ratio: float = SCORING_DEFAULTS.top2_ratio, min_score: float = SCORING_DEFAULTS.min_score, min_abs: float = SCORING_DEFAULTS.min_abs
):
    """
    Decode ID, names, version (if present) and all answer_layouts.
    Returns:
      info = dict(last_name, first_name, student_id, version)
      answers = list[Optional[str]]  # Q1..Qn
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Names / ID / Version
    info = {"last_name": "", "first_name": "", "student_id": "", "version": ""}

    if cfg.last_name_layout:
        picked, _, _ = decode_layout(gray, cfg.last_name_layout, min_fill, top2_ratio, min_score, min_abs)
        # per-column picks (len = choices); map rows->letters
        info["last_name"] = indices_to_text_col(picked, cfg.last_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()

    if cfg.first_name_layout:
        picked, _, _ = decode_layout(gray, cfg.first_name_layout, min_fill, top2_ratio, min_score, min_abs)
        info["first_name"] = indices_to_text_col(picked, cfg.first_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ").strip()

    if cfg.id_layout:
        picked, _, _ = decode_layout(gray, cfg.id_layout, min_fill, top2_ratio, min_score, min_abs)
        info["student_id"] = indices_to_text_col(picked, cfg.id_layout.labels or "0123456789")

    if cfg.version_layout:
        picked, _, _ = decode_layout(gray, cfg.version_layout, min_fill, top2_ratio, min_score, min_abs)
        # selection_axis likely "row" with one row; picked is per-row list of column indices
        if cfg.version_layout.selection_axis == "row":
            # pick first (only) row
            idx = picked[0] if picked else None
            labels = list(cfg.version_layout.labels or "ABCD")
            info["version"] = labels[idx] if idx is not None and 0 <= idx < len(labels) else ""
        else:
            # column-wise selection (rare); treat as characters-in-columns
            info["version"] = indices_to_text_col(picked, cfg.version_layout.labels or "ABCD")

    # Answers
    answers: List[Optional[str]] = []
    for i, layout in enumerate(cfg.answer_layouts):
        picked, _, _ = decode_layout(gray, layout, min_fill, top2_ratio, min_score, min_abs)  # per-row indices
        choice_labels = list(layout.labels) if layout.labels else [chr(ord('A') + k) for k in range(layout.choices)]
        row_labels = indices_to_labels_row(picked, layout.choices, choice_labels)
        answers.extend(row_labels)

    return info, answers
