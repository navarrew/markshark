#!/usr/bin/env python3
"""
MarkShark
score_core.py  —  Axis-based MarkShark grading engine

Features:
 - CSV includes: correct, incorrect, blank, multi, percent
 - KEY row written below header (when a key is provided)
 - Annotated PNGs:
     * Names/ID: blue circles, optional white % text
     * Answers: green=correct, red=incorrect, grey=blank, orange=multi
 - Optional % fill text via --label-density
 - Columns limited to len(key) when a key is provided
"""

from __future__ import annotations
import os
import csv
from typing import Optional, List, Tuple

import numpy as np
import cv2
from .defaults import (
    ANNOTATION_DEFAULTS,
    AnnotationDefaults,
    SCORING_DEFAULTS,
    resolve_scored_pdf_path,
)

from .config_io import load_config, Config, GridLayout
from .tools import io_pages as IO
from .tools.score_tools import (
    process_page_all,
    load_key_txt,
    grid_centers_axis_mode,
    centers_to_circle_rois,
    roi_fill_scores,
    calibrate_fixed_thresh_for_page
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
# Annotation helpers
# ----------------------------

def _annotate_names_ids(
    img_bgr: np.ndarray,
    cfg: Config,
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
        lay = getattr(cfg, attr, None)
        if isinstance(lay, GridLayout):
            draw_layout(lay)

    return out


def _annotate_answers(
img_bgr: np.ndarray,
cfg: Config,
key_letters: Optional[List[str]],
label_density: bool,
annotate_all_cells: bool,
min_fill: float,
top2_ratio: float,
min_score: float,
fixed_thresh: Optional[int] = None,
color_correct=None,
color_incorrect=None,
color_blank=None,
color_blank_answer_row=None,
color_multi=None,
thickness: Optional[int] = None,
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

    for layout in cfg.answer_layouts:
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.numrows, layout.numcols
        )
        rois = centers_to_circle_rois(centers, W, H, layout.radius_pct)
        M = _rowwise_scores(gray, rois, layout.numrows, layout.numcols, fixed_thresh=fixed_thresh,)
        choice_labels = list(layout.labels) if layout.labels else [chr(ord("A") + i) for i in range(layout.numcols)]

        for r in range(layout.numrows):
            row_scores = M[r]
            order = np.argsort(row_scores)[::-1]
            best = int(order[0])
            best_val = float(row_scores[best])
            second_val = float(row_scores[order[1]]) if layout.numcols > 1 else 0.0
            is_blank = best_val < min_fill
            is_multi = (not is_blank) and (layout.numcols > 1) and (second_val > top2_ratio * best_val)

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

                draw_this = annotate_all_cells or (c == best) or is_blank or is_multi
                if not draw_this:
                    continue

                if is_blank:
                    col = (color_blank_answer_row if answer_row_blank else color_blank)
                elif is_multi:
                    col = color_multi
                else:
                    if key_char:
                        col = color_correct if (choice_labels[c] == key_char and c == best) else (
                            color_incorrect if c == best else (200, 200, 200)
                        )
                    else:
                        col = (0, 200, 200) if c == best else (200, 200, 200)

                cv2.circle(out, (cx, cy), radius, col, thickness, lineType=cv2.LINE_AA)

                #here we add the text for the density of the pencil marks in the bubble
#                 if label_density:
#                     pct = int(round(100 * row_scores[c]))
#                     cv2.putText(out, f"{pct}", (cx - 8, cy + 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

                if label_density:
                    pct = int(round(100 * row_scores[c]))
                
                    # Put text above the circle
                    text = f"{pct}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.3
                    text_thickness = 1
                
                    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                    gap = 5  # pixels above circle
                    tx = cx - tw // 2
                    ty = cy - radius - gap  # baseline position above circle
                
                    # Keep text inside the image bounds
                    tx = max(0, min(tx, W - tw - 1))
                    ty = max(th + 1, ty)  # avoid going off the top
                
                    cv2.putText(out, text, (tx, ty),
                                font, font_scale, (255, 0, 255),
                                text_thickness, cv2.LINE_AA)

            q_global += 1

    return out


# ----------------------------
# Main grading entry point
# ----------------------------

def score_pdf(
    input_path: str,
    config_path: str,
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

    Behavior:
      - If key is provided: limit output columns and scoring to first len(key) questions.
      - CSV includes metrics: correct, incorrect, blank, multi, percent.
      - If a key is provided, a KEY row is written under the header.
      - Annotated images combine blue name/ID overlays and answer overlays.
    """
    cfg: Config = load_config(config_path)
    pages = IO.load_pages(input_path, dpi=dpi, renderer=pdf_renderer)
    key: Optional[List[str]] = load_key_txt(key_txt) if key_txt else None

    total_q = sum(a.numrows for a in cfg.answer_layouts)
    q_out = len(key) if key else total_q
    q_out = max(0, min(q_out, total_q))

    # Make the CSV header
    header = ["Version", "Page", "LastName", "FirstName", "StudentID"]

    if key:
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
        if key:
            key_row = ["", "0", "KEY", "KEY", "KEY"] + ["", "", "", "", ""] + key[:q_out] 
            outputcsv.writerow(key_row)

        for page_idx, img_bgr in enumerate(pages, start=1):
            # Per-page calibration for lightly marked sheets
            page_fixed_thresh = fixed_thresh
            if fixed_thresh is None and auto_calibrate_thresh:
                page_fixed_thresh, _calib_stats = calibrate_fixed_thresh_for_page(
                    img_bgr,
                    cfg,
                    verbose=verbose_calibration,
                )
            # Decode all fields using the shared axis-mode pipeline
            info, answers = process_page_all(img_bgr, cfg, min_fill=min_fill, top2_ratio=top2_ratio, min_score=min_score, fixed_thresh=page_fixed_thresh,)

            # Limit answers to the Qs we output (based on key length if present)
            answers_out = answers[:q_out]
            answers_csv = [a if a is not None else "" for a in answers_out]

            # Metrics
            blanks = sum(1 for a in answers_out if (a is None or a == ""))
            multi = sum(1 for a in answers_out if (isinstance(a, str) and "," in a))
            correct = incorrect = 0

            if key:
                key_out = key[:q_out]
                for a, k in zip(answers_out, key_out):
                    # Only score single-mark answers against the key.
                    if a is None or a == "":
                        continue
                    if isinstance(a, str) and "," in a:
                        continue
                    if a == k:
                        correct += 1
                    else:
                        incorrect += 1
                percent = (100.0 * correct / max(1, len(key_out)))
            else:
                key_out = None
                percent = 0.0

# Compose CSV row
            row = [
                info.get("version", ""),
                str(page_idx),
                info.get("last_name", ""),
                info.get("first_name", ""),
                info.get("student_id", ""),
            ] 

            if key:
                row += [str(correct), str(incorrect), str(blanks), str(multi), f"{percent:.1f}"]
            else:
                row += [str(blanks)]

            row += answers_csv
            
            outputcsv.writerow(row)

            # Annotated image: names/IDs in blue (with optional %), then answers overlay
            if out_annotated_dir or out_pdf_path:
                vis = _annotate_names_ids(
                    img_bgr,
                    cfg,
                    label_density=label_density,
                    annotation_defaults=ANNOTATION_DEFAULTS,
                )
                vis = _annotate_answers(
                    vis,
                    cfg,
                    key_out,
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
