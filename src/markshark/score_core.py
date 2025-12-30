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

from .config_io import load_config, Config, GridLayout
from .tools import io_pages as IO
from .tools.score_tools import (
    process_page_all,
    load_key_txt,
    grid_centers_axis_mode,
    centers_to_circle_rois,
    roi_fill_scores
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
) -> List[List[float]]:
    """Return a rows×cols matrix of fill scores (0–1)."""
    flat = roi_fill_scores(gray, rois, inner_radius_ratio=0.70, blur_ksize=5)
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

# ----------------------------
# Annotation helpers
# ----------------------------

def _annotate_names_ids(
    img_bgr: np.ndarray,
    cfg: Config,
    label_density: bool,
    color_zone=(255, 0, 0),   # blue (B, G, R)
    text_color=(255, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw blue circles for Last/First Name and Student ID grids.
    If label_density=True, write white % fill text in each bubble.
    Returns a new image with drawings (does not modify input in place).
    """
    out = img_bgr.copy()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    def draw_layout(layout: GridLayout) -> None:
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.questions, layout.choices
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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

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
    min_abs: float,
    color_correct=(0, 200, 0),
    color_incorrect=(0, 0, 255),
    color_blank=(160, 160, 160),
    color_multi=(0, 140, 255),
    thickness: int = 2,
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
    min_score=min_score,
    min_abs=min_abs,
    out = img_bgr.copy()
    H, W = out.shape[:2]
    key_seq = [k.upper() for k in key_letters] if key_letters else None
    q_global = 0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for layout in cfg.answer_layouts:
        centers = grid_centers_axis_mode(
            layout.x_topleft, layout.y_topleft,
            layout.x_bottomright, layout.y_bottomright,
            layout.questions, layout.choices
        )
        rois = centers_to_circle_rois(centers, W, H, layout.radius_pct)
        M = _rowwise_scores(gray, rois, layout.questions, layout.choices)
        choice_labels = [chr(ord("A") + i) for i in range(layout.choices)]

        for r in range(layout.questions):
            row_scores = M[r]
            order = np.argsort(row_scores)[::-1]
            best = int(order[0])
            best_val = float(row_scores[best])
            second_val = float(row_scores[order[1]]) if layout.choices > 1 else 0.0

            is_blank = best_val < min_fill
            is_multi = (not is_blank) and (layout.choices > 1) and (second_val > top2_ratio * best_val)

            key_char = key_seq[q_global] if key_seq and q_global < len(key_seq) else None

            for c in range(layout.choices):
                x, y, w, h = rois[r * layout.choices + c]
                cx, cy = x + w // 2, y + h // 2
                radius = min(w, h) // 2

                draw_this = annotate_all_cells or (c == best) or is_blank or is_multi
                if not draw_this:
                    continue

                if is_blank:
                    col = color_blank
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
                if label_density:
                    pct = int(round(100 * row_scores[c]))
                    cv2.putText(out, f"{pct}", (cx - 8, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

            q_global += 1

    return out


# ----------------------------
# Main grading entry point
# ----------------------------

def grade_pdf(
    input_path: str,
    config_path: str,
    out_csv: str,
    min_fill: float,
    top2_ratio: float,
    min_score: float,
    min_abs: float,
    key_txt: Optional[str] = None,
    out_annotated_dir: Optional[str] = None,
    out_pdf: Optional[str] = "scored_scans.pdf",
    dpi: int = 300,
    annotate_all_cells: bool = False,
    label_density: bool = False,
    pdf_renderer: str = "auto",
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

    total_q = sum(a.questions for a in cfg.answer_layouts)
    q_out = len(key) if key else total_q
    q_out = max(0, min(q_out, total_q))

    # Make the CSV header
    header = ["page_index", "LastName", "FirstName", "StudentID", "Version"] \
             + [f"Q{i+1}" for i in range(q_out)]
    if key:
        header += ["correct", "incorrect", "blank", "multi", "percent"]
    else:
        header += ["blank"]

    _ensure_dir(os.path.dirname(out_csv) or ".")
    if out_annotated_dir:
        _ensure_dir(out_annotated_dir)


    annotated_pages: List[np.ndarray] = []
    out_pdf_path: Optional[str] = None
    if out_annotated_dir and out_pdf:
        out_pdf_path = out_pdf
        if not os.path.isabs(out_pdf_path):
            base_dir = os.path.dirname(out_csv) or "."
            out_pdf_path = os.path.join(base_dir, out_pdf_path)
        _ensure_dir(os.path.dirname(out_pdf_path) or ".")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        if key:
            key_row = ["KEY", "", "", "", "KEY"] + key[:q_out] + ["", "", "", "", ""]
            w.writerow(key_row)

        for page_idx, img_bgr in enumerate(pages, start=1):
            # Decode all fields using the shared axis-mode pipeline
            info, answers = process_page_all(img_bgr, cfg, min_fill=min_fill, top2_ratio=top2_ratio, min_score=min_score, min_abs=min_abs)

            # Limit answers to the Qs we output (based on key length if present)
            answers_out = answers[:q_out]
            answers_csv = [a if a is not None else "" for a in answers_out]

            # Metrics
            blanks = sum(1 for a in answers_out if not a)
            correct = incorrect = 0

            if key:
                key_out = key[:q_out]
                for a, k in zip(answers_out, key_out):
                    if not a:
                        continue
                    if a == k:
                        correct += 1
                    else:
                        incorrect += 1
                percent = (100.0 * correct / max(1, len(key_out)))
            else:
                key_out = None
                percent = 0.0

            # Multi detection limited to first q_out questions
            multi = 0
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape[:2]
            q_idx = 0
            for layout in cfg.answer_layouts:
                centers = grid_centers_axis_mode(
                    layout.x_topleft, layout.y_topleft,
                    layout.x_bottomright, layout.y_bottomright,
                    layout.questions, layout.choices
                )
                rois = centers_to_circle_rois(centers, W, H, layout.radius_pct)
                M = _rowwise_scores(gray, rois, layout.questions, layout.choices)
                for r in range(layout.questions):
                    if q_idx >= q_out:
                        break
                    row_scores = M[r]
                    order = np.argsort(row_scores)[::-1]
                    best_val = float(row_scores[order[0]])
                    second_val = float(row_scores[order[1]]) if layout.choices > 1 else 0.0
                    is_blank = best_val < min_fill
                    is_multi = (not is_blank) and (layout.choices > 1) and (second_val > top2_ratio * best_val)
                    if is_multi:
                        multi += 1
                    q_idx += 1
                if q_idx >= q_out:
                    break

            # Compose CSV row
            row = [
                str(page_idx),
                info.get("last_name", ""),
                info.get("first_name", ""),
                info.get("student_id", ""),
                info.get("version", ""),
            ] + answers_csv

            if key:
                row += [str(correct), str(incorrect), str(blanks), str(multi), f"{percent:.1f}"]
            else:
                row += [str(blanks)]

            w.writerow(row)

            # Annotated image: names/IDs in blue (with optional %), then answers overlay
            if out_annotated_dir:
                vis = _annotate_names_ids(
                    img_bgr,
                    cfg,
                    label_density=label_density,
                    color_zone=(255, 0, 0),      # blue
                    text_color=(255, 0, 0),  # blue
                    thickness=2,
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
                    min_abs=min_abs
                )
                out_png = os.path.join(out_annotated_dir, f"page_{page_idx:03d}_overlay.png")
                cv2.imwrite(out_png, vis)
                annotated_pages.append(vis)
    if out_pdf_path and annotated_pages:
        IO.save_images_as_pdf(annotated_pages, out_pdf_path, dpi=dpi)

    return out_csv
