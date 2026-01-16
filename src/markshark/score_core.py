#!/usr/bin/env python3
"""
MarkShark
score_core.py  —  Axis-based MarkShark grading engine

Features:
 - Multi-version exam support (NEW)
 - CSV includes: correct, incorrect, blank, multi, percent
 - KEY row(s) written below header (when key(s) provided)
 - Annotated PNGs:
     * Names/ID: blue circles, optional white % text
     * Answers: green=correct, red=incorrect, grey=blank, orange=multi
 - Optional % fill text via --label-density
 - Columns limited to len(key) when a key is provided
"""

from __future__ import annotations
import os
import csv
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2
from .defaults import (
    ANNOTATION_DEFAULTS,
    AnnotationDefaults,
    SCORING_DEFAULTS,
    resolve_scored_pdf_path,
)

from .tools.bubblemap_io import load_bublmap, Bubblemap, GridLayout, PageLayout
from .tools import io_pages as IO
from .tools.score_tools import (
    process_page_all,
    load_key_txt,
    load_multi_version_keys,
    score_against_multi_keys,
    grid_centers_axis_mode,
    centers_to_circle_rois,
    roi_fill_scores,
    calibrate_fixed_thresh_for_page,
)


# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _rowwise_scores(
    gray: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    rows: int,
    cols: int,
    fixed_thresh: Optional[int] = None,
) -> List[List[float]]:
    """Return a rows x cols matrix of fill scores (0-1)."""
    flat = roi_fill_scores(
        gray,
        rois,
        inner_radius_ratio=0.70,
        blur_ksize=5,
        fixed_thresh=fixed_thresh,
    )
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

# ----------------------------
# Multi-page support helper
# ----------------------------

def _process_single_page(
    img_bgr: np.ndarray,
    page_layout: 'PageLayout',
    *,
    min_fill: float,
    top2_ratio: float,
    min_score: float,
    fixed_thresh: Optional[int] = None,
) -> Tuple[dict, List[Optional[str]]]:
    """
    Process a single page using a specific PageLayout from a multi-page bubblemap.
    Returns (info_dict, answers_list) for this page only.
    """
    from .tools.score_tools import decode_layout, indices_to_text_col
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    info = {"last_name": "", "first_name": "", "student_id": "", "version": ""}
    
    # Decode name/ID from this page (if layouts are present)
    if page_layout.last_name_layout:
        picked, _, _ = decode_layout(
            gray,
            page_layout.last_name_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["last_name"] = indices_to_text_col(
            picked, page_layout.last_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()
    
    if page_layout.first_name_layout:
        picked, _, _ = decode_layout(
            gray,
            page_layout.first_name_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["first_name"] = indices_to_text_col(
            picked, page_layout.first_name_layout.labels or " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ).strip()
    
    if page_layout.id_layout:
        picked, _, _ = decode_layout(
            gray,
            page_layout.id_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        info["student_id"] = indices_to_text_col(
            picked, page_layout.id_layout.labels or "0123456789"
        ).strip()
    
    if page_layout.version_layout:
        # Version detection (usually only on page 1)
        picked, _, _ = decode_layout(
            gray,
            page_layout.version_layout,
            min_fill=min_fill,
            top2_ratio=top2_ratio,
            min_score=min_score,
            fixed_thresh=fixed_thresh,
        )
        if picked and len(picked) > 0 and picked[0] is not None:
            labels = page_layout.version_layout.labels or "ABCD"
            info["version"] = labels[picked[0]] if picked[0] < len(labels) else ""
    
    # Decode answers from this page
    answers: List[Optional[str]] = []
    for layout in page_layout.answer_layouts:
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
            from .tools.score_tools import scores_to_labels_row
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
            from .tools.score_tools import indices_to_labels_row
            answers.extend(indices_to_labels_row(picked, layout.numcols, choice_labels))
    
    return info, answers


# ----------------------------
# Annotation helpers
# ----------------------------

def _annotate_names_ids(
    img_bgr: np.ndarray,
    bmap: Bubblemap,
    label_density: bool,
    color_zone=None,
    text_color=None,
    thickness: Optional[int] = None,
    font_scale: Optional[float] = None,
    label_thickness: Optional[int] = None,
    annotation_defaults: Optional[AnnotationDefaults] = None,
) -> np.ndarray:
    """
    Draw blue circles for Last/First Name and Student ID grids.
    If label_density=True, write white % fill text in each bubble.
    Returns a new image with drawings (does not modify input in place).
    """
    out = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    ad = annotation_defaults or ANNOTATION_DEFAULTS
    if color_zone is None:
        color_zone = ad.color_zone
    if text_color is None:
        text_color = ad.percent_text_color
    if thickness is None:
        thickness = ad.thickness_names
    if font_scale is None:
        font_scale = ad.label_font_scale
    if label_thickness is None:
        label_thickness = ad.label_thickness

    def draw_layout(layout: GridLayout) -> None:
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.numrows, layout.numcols
        )
        rois = centers_to_circle_rois(centers, W, H, layout.radius_pct)
        scores = roi_fill_scores(gray, rois, inner_radius_ratio=0.70, blur_ksize=5)
        for idx, (x, y, w, h) in enumerate(rois):
            cx, cy = x + w // 2, y + h // 2
            radius = min(w, h) // 2
            cv2.circle(out, (cx, cy), radius, color_zone, thickness, lineType=cv2.LINE_AA)
            if label_density:
                pct = int(round(100 * scores[idx]))
                cv2.putText(out, f"{pct}", (cx - 8, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), text_color, int(label_thickness), cv2.LINE_AA)

    for attr in ("last_name_layout", "first_name_layout", "id_layout"):
        lay = getattr(bmap, attr, None)
        if isinstance(lay, GridLayout):
            draw_layout(lay)

    return out


def _annotate_answers(
img_bgr: np.ndarray,
bmap: Bubblemap,
key_letters: Optional[List[str]],
label_density: bool,
annotate_all_cells: bool,
min_fill: float,
top2_ratio: float,
min_score: float,
answers_for_annotation: Optional[List[Optional[str]]] = None,
fixed_thresh: Optional[int] = None,
color_correct=None,
color_incorrect=None,
color_blank=None,
color_blank_answer_row=None,
color_multi=None,
thickness: Optional[int] = None,
pct_fill_font_color=None,
pct_fill_font_scale=None,
pct_fill_font_thickness=None,
pct_fill_font_position=None,
annotation_defaults: Optional[AnnotationDefaults] = None,
box_multi: Optional[bool] = None,
box_blank_answer_row: Optional[bool] = None,
box_color_multi=None,
box_color_blank_answer_row=None,
box_thickness: Optional[int] = None,
box_pad: Optional[int] = None,
box_top_extra: Optional[int] = None,
) -> np.ndarray:
    """
    Draw per-bubble overlays for answer blocks:
      - green circle for correct,
      - red for incorrect,
      - grey for blank,
      - orange for multi.
    Optionally put % fill text in each bubble (label_density=True).
    Returns a new image with drawings (does not modify input in place).
    """
    out = img_bgr.copy()
    H, W = out.shape[:2]
    key_seq = [k.upper() for k in key_letters] if key_letters else None

    # Resolve annotation defaults
    ad = annotation_defaults or ANNOTATION_DEFAULTS
    if color_correct is None:
        color_correct = ad.color_correct
    if color_incorrect is None:
        color_incorrect = ad.color_incorrect
    if color_blank is None:
        color_blank = ad.color_blank
    if color_blank_answer_row is None:
        color_blank_answer_row = getattr(ad, "color_blank_answer_row", (255, 0, 255))
    if color_multi is None:
        color_multi = ad.color_multi
    if thickness is None:
        thickness = ad.thickness_answers

    # Resolve annotation defaults
    if pct_fill_font_color is None:
        pct_fill_font_color  = getattr(ad, "pct_fill_font_color", (255, 0, 255))
    if pct_fill_font_scale is None:
        pct_fill_font_scale = getattr(ad, "pct_fill_font_scale", 0.5)
    if pct_fill_font_thickness is None:
        pct_fill_font_thickness = getattr(ad, "pct_fill_font_thickness", 1)
    if pct_fill_font_position is None:
        pct_fill_font_position = getattr(ad, "pct_fill_font_position", 5)

    # Row box defaults
    if box_multi is None:
        box_multi = getattr(ad, "box_multi", False)
    if box_blank_answer_row is None:
        box_blank_answer_row = getattr(ad, "box_blank_answer_row", False)
    if box_color_multi is None:
        box_color_multi = getattr(ad, "box_color_multi", color_multi)
    if box_color_blank_answer_row is None:
        box_color_blank_answer_row = getattr(ad, "box_color_blank_answer_row", color_blank_answer_row)
    if box_thickness is None:
        box_thickness = getattr(ad, "box_thickness", thickness)
    if box_pad is None:
        box_pad = getattr(ad, "box_pad", 6)
    if box_top_extra is None:
        box_top_extra = getattr(ad, "box_top_extra", 0)

    q_global = 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for layout in bmap.answer_layouts:
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.numrows, layout.numcols
        )
        rois = centers_to_circle_rois(centers, W, H, layout.radius_pct)
        M = _rowwise_scores(gray, rois, layout.numrows, layout.numcols, fixed_thresh=fixed_thresh,)
        choice_labels = list(layout.labels) if layout.labels else [chr(ord("A") + i) for i in range(layout.numcols)]
        label_to_idx = {str(lab).strip().upper(): i for i, lab in enumerate(choice_labels)}

        for r in range(layout.numrows):
            # Skip annotation for questions beyond the answer key length
            if answers_for_annotation is not None and q_global >= len(answers_for_annotation):
                q_global += 1
                continue
            
            row_scores = M[r]
            order = np.argsort(row_scores)[::-1]
            best = int(order[0])
            best_val = float(row_scores[best])
            second_val = float(row_scores[order[1]]) if layout.numcols > 1 else 0.0
            is_blank = best_val < min_fill
            is_multi = (not is_blank) and (layout.numcols > 1) and (second_val > top2_ratio * best_val)

            # If available, drive annotation from the same scoring decisions used for the CSV.
            # This prevents borderline cases being labeled blank in the CSV but shown as selected in the PDF overlay.
            selected_labels = []  # type: List[str]
            selected_idx = set()

            if answers_for_annotation is not None and q_global < len(answers_for_annotation):
                ans = answers_for_annotation[q_global]
                if ans is None or ans == "":
                    is_blank, is_multi = True, False
                else:
                    ans_str = str(ans).strip().upper()
                    if "," in ans_str:
                        is_blank, is_multi = False, True
                        selected_labels = [p.strip() for p in ans_str.split(",") if p.strip()]
                    else:
                        is_blank, is_multi = False, False
                        selected_labels = [ans_str]

                for lab in selected_labels:
                    if lab in label_to_idx:
                        selected_idx.add(label_to_idx[lab])

            # Fallback: if we did not map any selected labels, use local score winners.
            if not selected_idx:
                if is_multi and layout.numcols > 1:
                    selected_idx = {best, int(order[1])}
                elif not is_blank:
                    selected_idx = {best}


            key_char = key_seq[q_global] if key_seq and q_global < len(key_seq) else None
            answer_row_blank = bool(is_blank and key_char and (key_char in choice_labels))

            # Optional row-level boxes, draw these before circles and text
            if (is_multi and box_multi) or (answer_row_blank and box_blank_answer_row):
                row_rois = rois[r * layout.numcols:(r + 1) * layout.numcols]
                x0 = min(x for x, y, w, h in row_rois)
                y0 = min(y for x, y, w, h in row_rois)
                x1 = max(x + w for x, y, w, h in row_rois)
                y1 = max(y + h for x, y, w, h in row_rois)

                x0 = max(0, x0 - int(box_pad))
                y0 = max(0, y0 - int(box_pad) - int(box_top_extra))
                x1 = min(W - 1, x1 + int(box_pad))
                y1 = min(H - 1, y1 + int(box_pad))

                if is_multi and box_multi:
                    cv2.rectangle(out, (x0, y0), (x1, y1), box_color_multi, int(box_thickness), lineType=cv2.LINE_AA)
                if answer_row_blank and box_blank_answer_row:
                    cv2.rectangle(out, (x0, y0), (x1, y1), box_color_blank_answer_row, int(box_thickness), lineType=cv2.LINE_AA)


            for c in range(layout.numcols):
                x, y, w, h = rois[r * layout.numcols + c]
                cx, cy = x + w // 2, y + h // 2
                radius = min(w, h) // 2

                draw_this = annotate_all_cells or (c in selected_idx) or is_blank or is_multi
                if not draw_this:
                    continue

                if is_blank:
                    col = (color_blank_answer_row if answer_row_blank else color_blank)
                elif is_multi:
                    col = color_multi
                else:
                    if key_char:
                        is_selected = c in selected_idx

                        selected_single = None
                        if selected_labels and len(selected_labels) == 1:
                            selected_single = selected_labels[0]
                        elif (not selected_labels) and (not is_blank) and (not is_multi) and len(selected_idx) == 1:
                            only_idx = next(iter(selected_idx))
                            selected_single = str(choice_labels[only_idx]).strip().upper()

                        col = color_correct if (is_selected and selected_single == key_char) else (
                            color_incorrect if is_selected else (200, 200, 200)
                        )
                    else:
                        col = (0, 200, 200) if (c in selected_idx) else (200, 200, 200)

                cv2.circle(out, (cx, cy), radius, col, thickness, lineType=cv2.LINE_AA)

                if label_density:
                    pct = int(round(100 * row_scores[c]))
                
                    # Put text above the circle
                    text = f"{pct}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                
                    (tw, th), baseline = cv2.getTextSize(text, font, pct_fill_font_scale, 
                    pct_fill_font_thickness)
                
                    # this formula sets text baseline position above or below circle
                    # pct_fill_font_position values correspond to pixels.
                    # these values should be set in the defaults.py script
                    # a value of zero puts the text right on top of the circle
                    # negative values pull the text into the circle.
                    # positive values push the text farther above the circle
                    tx = cx - tw // 2
                    ty = cy - radius - pct_fill_font_position  
                
                    # Keep text inside the image bounds
                    tx = max(0, min(tx, W - tw - 1))
                    ty = max(th + 1, ty)  # avoid going off the top
                
                    cv2.putText(out, text, (tx, ty),
                                font, pct_fill_font_scale, pct_fill_font_color,
                                pct_fill_font_thickness, cv2.LINE_AA)

            q_global += 1

    return out


# ----------------------------
# Main grading entry point
# ----------------------------

def score_pdf(
    input_path: str,
    bublmap_path: str,
    out_csv: str,
    min_fill: float,
    top2_ratio: float,
    min_score: float,
    fixed_thresh: Optional[int] = None,
    key_txt: Optional[str] = None,
    out_annotated_dir: Optional[str] = None,
    out_pdf: Optional[str] = None,
    dpi: int = 300,
    annotate_all_cells: bool = False,
    label_density: bool = False,
    pdf_renderer: str = "auto",
    auto_calibrate_thresh: bool = True,
    verbose_calibration: bool = False,
) -> str:
    
    """
    Grade a PDF or image stack using axis-based geometry.
    
    NEW: Multi-page bubble sheet support!
    NEW: Multi-version exam support!

    Behavior:
      - Supports single-page or multi-page bubble sheets
      - For multi-page: processes pages in groups (page 1+2 = student 1, page 3+4 = student 2, etc.)
      - Name/ID taken from page 1, answers combined from all pages
      - Detects version from version_layout bubbles
      - Loads multi-version keys if key file contains #Version headers
      - Scores each student against their version's key
      - If key is provided: limit output columns and scoring to first len(key) questions.
      - CSV includes metrics: correct, incorrect, blank, multi, percent.
      - KEY row(s) written under header (one per version).
      - Annotated images combine blue name/ID overlays and answer overlays.
      - Page column shows "1-2" for 2-page sheets, "1" for single-page
    """
    bmap: Bubblemap = load_bublmap(bublmap_path)
    pages = IO.load_pages(input_path, dpi=dpi, renderer=pdf_renderer)
    
    # Try loading multi-version keys, fall back to single version
    keys_dict: Optional[Dict[str, List[str]]] = None
    single_key: Optional[List[str]] = None
    
    if key_txt:
        try:
            keys_dict = load_multi_version_keys(key_txt)
        except Exception:
            # Fall back to single-version key
            single_key = load_key_txt(key_txt)
            if single_key:
                keys_dict = {"A": single_key}  # Default to version A

    # Calculate total questions across all pages
    total_q = sum(
        sum(layout.numrows for layout in page.answer_layouts)
        for page in bmap.pages
    )
    
    if keys_dict:
        # Use length of first version key
        first_version = sorted(keys_dict.keys())[0]
        q_out = len(keys_dict[first_version])
    else:
        q_out = total_q
    
    q_out = max(0, min(q_out, total_q))

    # Make the CSV header
    header = ["Version", "Page", "LastName", "FirstName", "StudentID"]

    if keys_dict:
        header += ["Correct", "Incorrect", "Blank", "Multi", "Percent"]
    else:
        header += ["Blank"]
        
    header += [f"Q{i+1}" for i in range(q_out)]

    _ensure_dir(os.path.dirname(out_csv) or ".")
    if out_annotated_dir:
        _ensure_dir(out_annotated_dir)


    # If requested, write an annotated PDF (default) and optionally a PNG directory.
    out_pdf_path: Optional[str] = None
    pdf_writer = None
    annotated_pages: List[np.ndarray] = []  # fallback only (used if streaming writer is unavailable)


    # out_pdf None means use the default name from defaults.py
    if out_pdf is None:
        out_pdf = SCORING_DEFAULTS.out_pdf

    # out_pdf "" (empty string) means disable PDF output
    if out_pdf:
        out_pdf_path = resolve_scored_pdf_path(out_pdf, out_csv=out_csv, out_pdf_dir=SCORING_DEFAULTS.out_pdf_dir)
        _ensure_dir(os.path.dirname(out_pdf_path) or ".")
        # Prefer a streaming PDF writer (small PDFs via JPEG embedding). Falls back to PIL at end if needed.
        try:
            pdf_writer = IO.PdfPageWriter(out_pdf_path, dpi=dpi)
        except Exception:
            pdf_writer = None
            
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        outputcsv = csv.writer(f)
        outputcsv.writerow(header)
        
        # Write KEY row(s) - one per version if multi-version
        if keys_dict:
            for ver in sorted(keys_dict.keys()):
                key_row = [ver, "0", "KEY", "KEY", "KEY"] + ["", "", "", "", ""] + keys_dict[ver][:q_out]
                outputcsv.writerow(key_row)

        # Determine if multi-page mode
        pages_per_student = bmap.num_pages
        is_multipage = pages_per_student > 1
        
        if is_multipage:
            # MULTI-PAGE MODE: Process pages in groups
            num_students = len(pages) // pages_per_student
            
            # Check for incomplete students (odd number of pages)
            if len(pages) % pages_per_student != 0:
                raise ValueError(
                    f"ERROR: Input PDF has {len(pages)} pages, but template expects {pages_per_student} pages per student. "
                    f"Expected a multiple of {pages_per_student} pages (got {len(pages) % pages_per_student} extra pages). "
                    f"Please check your scans - each student should have exactly {pages_per_student} pages."
                )
            
            for student_idx in range(num_students):
                # Get all pages for this student
                start_page_idx = student_idx * pages_per_student
                student_pages = pages[start_page_idx:start_page_idx + pages_per_student]
                
                # Process each page and collect data
                all_answers = []
                student_info = {}
                page_thresholds = []
                annotated_images = []
                
                for page_num, img_bgr in enumerate(student_pages, start=1):
                    page_layout = bmap.get_page(page_num)
                    
                    # Per-page calibration
                    page_fixed_thresh = fixed_thresh
                    if fixed_thresh is None and auto_calibrate_thresh:
                        # Use page 1 layout for calibration reference
                        calib_page = bmap.get_page(1)
                        if calib_page:
                            # Create temp bmap-like object for calibration
                            temp_bmap = type('TempBmap', (object,), {
                                'answer_layouts': calib_page.answer_layouts,
                                'last_name_layout': calib_page.last_name_layout,
                                'first_name_layout': calib_page.first_name_layout,
                                'id_layout': calib_page.id_layout,
                            })()
                            page_fixed_thresh, _calib_stats = calibrate_fixed_thresh_for_page(
                                img_bgr,
                                temp_bmap,
                                verbose=verbose_calibration,
                            )
                    
                    page_thresholds.append(page_fixed_thresh)
                    
                    # Process this page
                    info, answers = _process_single_page(
                        img_bgr, page_layout,
                        min_fill=min_fill,
                        top2_ratio=top2_ratio,
                        min_score=min_score,
                        fixed_thresh=page_fixed_thresh,
                    )
                    
                    # Take student info from page 1
                    if page_num == 1:
                        student_info = info
                    
                    # Accumulate all answers
                    all_answers.extend(answers)
                
                # Limit answers to key length
                answers_out = all_answers[:q_out]
                answers_csv = [a if a is not None else "" for a in answers_out]
                
                # Get detected version
                student_version = student_info.get("version", "")
                
                # Metrics
                blanks = sum(1 for a in answers_out if (a is None or a == ""))
                multi = sum(1 for a in answers_out if (isinstance(a, str) and "," in a))
                correct = incorrect = 0
                percent = 0.0
                version_used = student_version
                key_out = None
                
                if keys_dict:
                    # Multi-version scoring
                    correct, total_scored, version_used = score_against_multi_keys(
                        answers_out,
                        student_version,
                        keys_dict,
                    )
                    
                    # Count incorrect
                    answered_single = sum(1 for a in answers_out if a and "," not in a)
                    incorrect = answered_single - correct
                    
                    percent = (100.0 * correct / max(1, total_scored))
                    
                    # Get the key for annotation
                    key_version = version_used.rstrip("*")
                    key_out = keys_dict.get(key_version, [])[:q_out]
                
                # Compose CSV row - Page column shows "1-2" for multi-page
                page_range = f"1-{pages_per_student}" if pages_per_student > 1 else "1"
                row = [
                    version_used,
                    page_range,
                    student_info.get("last_name", ""),
                    student_info.get("first_name", ""),
                    student_info.get("student_id", ""),
                ]
                
                if keys_dict:
                    row += [str(correct), str(incorrect), str(blanks), str(multi), f"{percent:.1f}"]
                else:
                    row += [str(blanks)]
                
                row += answers_csv
                
                outputcsv.writerow(row)
                
                # Annotate all pages for this student
                if out_annotated_dir or out_pdf_path:
                    for page_num, img_bgr in enumerate(student_pages, start=1):
                        page_layout = bmap.get_page(page_num)
                        
                        if page_layout is None:
                            raise ValueError(f"Could not get layout for page {page_num}")
                        
                        # DEBUG: Verify we have the right page
                        num_answer_layouts_this_page = len(page_layout.answer_layouts)
                        if verbose_calibration:
                            print(f"  Annotating page {page_num} with {num_answer_layouts_this_page} answer layout(s)")
                            if num_answer_layouts_this_page > 0:
                                first_layout = page_layout.answer_layouts[0]
                                print(f"    First layout coords: ({first_layout.x_topleft:.4f}, {first_layout.y_topleft:.4f}) to ({first_layout.x_bottomright:.4f}, {first_layout.y_bottomright:.4f})")
                                print(f"    First layout: {first_layout.numrows} rows × {first_layout.numcols} cols")
                        
                        page_fixed_thresh = page_thresholds[page_num - 1]
                        
                        # Get answers for this page only
                        # Calculate which answers belong to this page
                        answers_before_page = sum(
                            sum(layout.numrows for layout in bmap.get_page(p).answer_layouts)
                            for p in range(1, page_num)
                        )
                        answers_in_page = sum(layout.numrows for layout in page_layout.answer_layouts)
                        
                        # For annotation, we need the answers specific to THIS page
                        # Slice from all_answers to get just this page's answers
                        page_answers = all_answers[answers_before_page:answers_before_page + answers_in_page]
                        
                        # Limit to key length - only annotate answers that are within the key
                        # But we need to account for the offset
                        if q_out > answers_before_page:
                            # Some or all of this page's questions are within the key
                            page_answers_limit = min(answers_in_page, q_out - answers_before_page)
                            page_answers_out = page_answers[:page_answers_limit]
                        else:
                            # This entire page is beyond the key length
                            page_answers_out = []
                        
                        # Annotate with name/ID zones
                        # Create a proper Bubblemap-like object from PageLayout
                        temp_bmap_names = type('TempBmap', (object,), {
                            'last_name_layout': page_layout.last_name_layout,
                            'first_name_layout': page_layout.first_name_layout,
                            'id_layout': page_layout.id_layout,
                        })()
                        
                        vis = _annotate_names_ids(
                            img_bgr,
                            temp_bmap_names,
                            label_density=label_density,
                            annotation_defaults=ANNOTATION_DEFAULTS,
                        )
                        
                        # Annotate with answer zones
                        # Create a proper Bubblemap-like object from PageLayout
                        temp_bmap_answers = type('TempBmap', (object,), {
                            'answer_layouts': page_layout.answer_layouts,
                        })()
                        
                        vis = _annotate_answers(
                            vis,
                            temp_bmap_answers,
                            key_out[answers_before_page:answers_before_page + answers_in_page] if key_out else None,
                            answers_for_annotation=page_answers_out,
                            label_density=label_density,
                            annotate_all_cells=annotate_all_cells,
                            min_fill=min_fill,
                            top2_ratio=top2_ratio,
                            min_score=min_score,
                            fixed_thresh=page_fixed_thresh,
                            annotation_defaults=ANNOTATION_DEFAULTS,
                        )
                        
                        # Save annotated page
                        if out_annotated_dir:
                            actual_page_num = start_page_idx + page_num
                            out_png = os.path.join(out_annotated_dir, f"page_{actual_page_num:03d}_overlay.png")
                            cv2.imwrite(out_png, vis)
                        
                        if out_pdf_path:
                            if pdf_writer is not None:
                                pdf_writer.add_page(vis)
                            else:
                                annotated_pages.append(vis)
        
        else:
            # SINGLE-PAGE MODE: Original logic (unchanged)
            for page_idx, img_bgr in enumerate(pages, start=1):
                # Per-page calibration for lightly marked sheets
                page_fixed_thresh = fixed_thresh
                if fixed_thresh is None and auto_calibrate_thresh:
                    page_fixed_thresh, _calib_stats = calibrate_fixed_thresh_for_page(
                        img_bgr,
                        bmap,
                        verbose=verbose_calibration,
                    )
                # Decode all fields using the shared axis-mode pipeline
                info, answers = process_page_all(
                    img_bgr, bmap,
                    min_fill=min_fill,
                    top2_ratio=top2_ratio,
                    min_score=min_score,
                    fixed_thresh=page_fixed_thresh,
                )

                # Limit answers to the Qs we output (based on key length if present)
                answers_out = answers[:q_out]
                answers_csv = [a if a is not None else "" for a in answers_out]

                # Get detected version
                student_version = info.get("version", "")
                
                # Metrics
                blanks = sum(1 for a in answers_out if (a is None or a == ""))
                multi = sum(1 for a in answers_out if (isinstance(a, str) and "," in a))
                correct = incorrect = 0
                percent = 0.0
                version_used = student_version
                key_out = None

                if keys_dict:
                    # Multi-version scoring
                    correct, total_scored, version_used = score_against_multi_keys(
                        answers_out,
                        student_version,
                        keys_dict,
                    )
                    
                    # Count incorrect (total scored - blanks - multi - correct)
                    answered_single = sum(1 for a in answers_out if a and "," not in a)
                    incorrect = answered_single - correct
                    
                    percent = (100.0 * correct / max(1, total_scored))
                    
                    # Get the key for annotation
                    key_version = version_used.rstrip("*")
                    key_out = keys_dict.get(key_version, [])[:q_out]
                
                # Compose CSV row
                row = [
                    version_used,
                    str(page_idx),
                    info.get("last_name", ""),
                    info.get("first_name", ""),
                    info.get("student_id", ""),
                ] 

                if keys_dict:
                    row += [str(correct), str(incorrect), str(blanks), str(multi), f"{percent:.1f}"]
                else:
                    row += [str(blanks)]

                row += answers_csv
                
                outputcsv.writerow(row)

                # Annotated image: names/IDs in blue (with optional %), then answers overlay
                if out_annotated_dir or out_pdf_path:
                    vis = _annotate_names_ids(
                        img_bgr,
                        bmap,
                        label_density=label_density,
                        annotation_defaults=ANNOTATION_DEFAULTS,
                    )
                    vis = _annotate_answers(
                        vis,
                        bmap,
                        key_out,
                        answers_for_annotation=answers_out,  # Only annotate questions in the key
                        label_density=label_density,
                        annotate_all_cells=annotate_all_cells,
                        min_fill=min_fill,
                        top2_ratio=top2_ratio,
                        min_score=min_score,
                        fixed_thresh=page_fixed_thresh,
                        annotation_defaults=ANNOTATION_DEFAULTS,
                    )
                    if out_annotated_dir:
                        out_png = os.path.join(out_annotated_dir, f"page_{page_idx:03d}_overlay.png")
                        cv2.imwrite(out_png, vis)
                    if out_pdf_path:
                        if pdf_writer is not None:
                            pdf_writer.add_page(vis)
                        else:
                            annotated_pages.append(vis)
                        
    if out_pdf_path:
        if pdf_writer is not None:
            pdf_writer.close(save=True)
        elif annotated_pages:
            IO.save_images_as_pdf(annotated_pages, out_pdf_path, dpi=dpi)

    return out_csv
