#!/usr/bin/env python3
"""
generate_mock_exam.py

Generate synthetic exam data for testing MarkShark pipelines.

This script reads a bubblemap YAML and template PDF, auto-detects the format
(answer choices, version field, student ID digits, etc.), generates fake
students with realistic score distributions, and renders filled bubble sheets.

Features:
- Auto-detects answer format (A-E, A-D, 1-5, etc.) from bubblemap
- Auto-detects version field format and generates multiple versions if present
- Generates realistic student score distribution (beta distribution, ~20-100%)
- Includes ~2% blanks and ~2% multi-fills scattered among wrong answers
- Variable bubble darkness to simulate light/dark pencil marks
- Optional random rotation/translation to challenge alignment

Usage:
    python generate_mock_exam.py \\
        --template path/to/master_template.pdf \\
        --bubblemap path/to/bubblemap.yaml \\
        --out-dir output_folder \\
        --num-students 100 \\
        --darkness-min 0.4 \\
        --darkness-max 1.0 \\
        --num-students 100

Output:
    - mock_answer_key.txt: Key file in MarkShark format (#A\\nA,B,C,...\\n#B\\n...)
    - mock_scans.pdf: PDF of all synthesized student sheets
    - mock_student_answers.csv: CSV with student info and expected answers
"""

import argparse
import csv
import os
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("PIL/Pillow is required. Install with: pip install pillow")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF is required. Install with: pip install pymupdf")


# =============================================================================
# Bubblemap parsing (handles current MarkShark schema)
# =============================================================================

def load_bubblemap(path: str) -> Dict[str, Any]:
    """Load and parse a bubblemap YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_format(bubblemap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-detect format from bubblemap.

    Returns dict with:
        - total_questions: int
        - answer_labels: str (e.g., "ABCDE")
        - has_version: bool
        - version_labels: str or None (e.g., "ABCD" or "1234")
        - id_digits: int
        - has_first_name: bool
        - has_last_name: bool
        - answer_layouts: list of layout dicts
        - page_key: str (e.g., "page_1")
    """
    result = {
        "total_questions": 0,
        "answer_labels": "ABCDE",
        "has_version": False,
        "version_labels": None,
        "id_digits": 10,
        "has_first_name": False,
        "has_last_name": False,
        "answer_layouts": [],
        "page_key": "page_1",
    }

    # Find the page key (page_1, page_2, etc.)
    page_key = None
    for key in bubblemap:
        if key.startswith("page_"):
            page_key = key
            break

    if not page_key:
        print("Warning: No page_N key found in bubblemap, assuming flat structure")
        page_data = bubblemap
    else:
        result["page_key"] = page_key
        page_data = bubblemap[page_key]

    # Check metadata for total_questions
    metadata = bubblemap.get("metadata", {})
    if metadata.get("total_questions"):
        result["total_questions"] = int(metadata["total_questions"])

    # Parse answer layouts
    answer_layouts = page_data.get("answer_layouts", [])
    if answer_layouts:
        result["answer_layouts"] = answer_layouts
        # Get labels from first layout
        first_layout = answer_layouts[0]
        result["answer_labels"] = str(first_layout.get("labels", "ABCDE"))

        # Sum up total questions if not in metadata
        if result["total_questions"] == 0:
            result["total_questions"] = sum(
                int(lay.get("numrows", lay.get("questions", 0)))
                for lay in answer_layouts
            )

    # Check for version layout
    version_layout = page_data.get("version_layout")
    if version_layout:
        result["has_version"] = True
        result["version_labels"] = str(version_layout.get("labels", "ABCD"))
        result["version_layout"] = version_layout

    # Check for ID layout
    id_layout = page_data.get("id_layout")
    if id_layout:
        result["id_digits"] = int(id_layout.get("numcols", id_layout.get("choices", 10)))
        result["id_layout"] = id_layout

    # Check for name layouts
    result["has_first_name"] = "first_name_layout" in page_data
    result["has_last_name"] = "last_name_layout" in page_data

    if result["has_first_name"]:
        result["first_name_layout"] = page_data["first_name_layout"]
    if result["has_last_name"]:
        result["last_name_layout"] = page_data["last_name_layout"]

    return result


# =============================================================================
# Answer key generation
# =============================================================================

def generate_answer_key(num_questions: int, labels: str) -> List[str]:
    """Generate a random answer key."""
    label_list = list(labels)
    return [random.choice(label_list) for _ in range(num_questions)]


def generate_versioned_keys(
    num_questions: int,
    labels: str,
    version_labels: str,
    num_versions: int = 2
) -> Dict[str, List[str]]:
    """
    Generate answer keys for multiple versions.

    Each version gets a shuffled variant of the base key to simulate
    different question orderings.
    """
    versions = list(version_labels)[:num_versions]
    keys = {}

    # Generate base key for first version
    base_key = generate_answer_key(num_questions, labels)
    keys[versions[0]] = base_key

    # For subsequent versions, shuffle the key (simulating reordered questions)
    for ver in versions[1:]:
        # Create a permuted version - swap ~30% of answers
        permuted = base_key.copy()
        num_swaps = num_questions // 3
        swap_indices = random.sample(range(num_questions), num_swaps)
        for idx in swap_indices:
            other_labels = [l for l in labels if l != permuted[idx]]
            permuted[idx] = random.choice(other_labels)
        keys[ver] = permuted

    return keys


def write_answer_key(keys: Dict[str, List[str]], output_path: str):
    """
    Write answer key in MarkShark format.

    Format:
        #A
        A,B,C,D,E,...
        #B
        B,A,D,C,E,...
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for version, answers in keys.items():
            f.write(f"#{version}\n")
            f.write(",".join(answers) + "\n")


# =============================================================================
# Fake student generation
# =============================================================================

FIRST_NAMES = [
    'James', 'Mary', 'Robert', 'Patricia', 'Michael', 'Jennifer', 'William', 'Linda',
    'David', 'Barbara', 'Richard', 'Elizabeth', 'Joseph', 'Susan', 'Thomas', 'Jessica',
    'Charles', 'Sarah', 'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa',
    'Anthony', 'Betty', 'Donald', 'Margaret', 'Mark', 'Sandra', 'Steven', 'Ashley',
    'Paul', 'Kimberly', 'Andrew', 'Emily', 'Joshua', 'Donna', 'Kenneth', 'Michelle',
    'Kevin', 'Dorothy', 'Brian', 'Carol', 'George', 'Amanda', 'Edward', 'Melissa',
    'Ronald', 'Deborah', 'Timothy', 'Stephanie', 'Jason', 'Rebecca', 'Jeffrey',
    'Alan', 'Omar', 'Jared', 'Zara', 'Quinn', 'Pearl', 'Blair', 'Fiona', 'Carla',
    'Jenna', 'Tyler', 'Owen', 'Rosie', 'Clara', 'Nina', 'Brent', 'Diana', 'Wade',
    'Molly', 'Anna', 'Zack', 'Mason', 'Derek', 'Adam', 'Abby', 'Yara', 'Nate',
    'Avery', 'Jack', 'Zoey', 'Sharon', 'Wei', 'Mei', 'Raj', 'Priya', 'Hiroshi',
    'Yuki', 'Mohammed', 'Fatima', 'Luis', 'Maria', 'Chen', 'Lin',
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Young',
    'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Peterson', 'Phillips', 'Campbell',
    'Parker', 'Evans', 'Edwards', 'Collins', 'Reyes', 'Stewart', 'Morris', 'Morales',
    'Murphy', 'Cook', 'Rogers', 'Morgan', 'Cooper', 'Reed', 'Bell', 'Berg', 'Kumar',
    'Rahman', 'Klein', 'Novak', 'Silva', 'Popescu', 'Barros', 'Tanaka', 'Li',
    'Voinova', 'Costa', 'Flores', 'Kowalski', 'Blanco', 'Bishop', 'Desai', 'Ford',
    'Hassan', 'Banerjee', 'Andersson', 'Espinoza', 'Ma', 'Ahmed', 'Patel', 'Duran',
    'Ali', 'OBrien', 'Ghosh', 'Singh', 'Bhat', 'Farah', 'Saito', 'Cohen', 'Reddy',
    'Bianchi', 'Black', 'Dubois', 'Dong', 'DelaCruz', 'Lima', 'Chen', 'Kobayashi',
    'Laurent', 'Carter', 'Adams', 'Araya', 'Baker', 'Rossi', 'Bae', 'Le', 'Santos',
    'Mehta', 'Dominguez', 'Hall', 'Delgado', 'Lin', 'Qureshi', 'Becker', 'Rivera',
    'Fernandez', 'Aung', 'Petrov', 'Burke', 'Green', 'Diaz', 'Suzuki', 'Machado',
    'Chang', 'Pereira', 'ElSayed', 'Fischer', 'Gupta', 'Ortega', 'Bennett', 'Moreno',
    'Kruger', 'Kimura', 'Nunez', 'Choi', 'Kuznetsov', 'Cruz', 'Wang', 'Kim', 'Park',
]


def generate_student_id(num_digits: int) -> str:
    """Generate a random student ID (doesn't start with 0)."""
    first = str(random.randint(1, 9))
    rest = ''.join(random.choices(string.digits, k=num_digits - 1))
    return first + rest


def generate_fake_students(
    num_students: int,
    answer_key: List[str],
    answer_labels: str,
    id_digits: int = 10,
    blank_rate: float = 0.02,
    multi_rate: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Generate fake student data with realistic score distribution.

    Returns list of dicts with:
        - student_id, first_name, last_name
        - version (if applicable)
        - answers: list of answer strings (may include blanks "" or multi "A,B")
        - expected_score: float 0-1
    """
    num_questions = len(answer_key)
    label_list = list(answer_labels)

    # Generate accuracy scores: beta distribution centered around 75%
    # with tails from 20% to 100%
    if num_students == 1:
        accuracies = np.array([np.random.beta(3.5, 2.5) * 0.80 + 0.20])
    elif num_students == 2:
        accuracies = np.array([1.00, 0.20])
    else:
        accuracies = np.concatenate([
            [1.00],  # One perfect student
            [0.20],  # One struggling student
            np.random.beta(3.5, 2.5, num_students - 2) * 0.80 + 0.20
        ])
    np.random.shuffle(accuracies)

    # Generate unique student IDs
    used_ids = set()
    students = []

    for i in range(num_students):
        # Generate unique ID
        while True:
            sid = generate_student_id(id_digits)
            if sid not in used_ids:
                used_ids.add(sid)
                break

        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        accuracy = accuracies[i]

        # Determine which questions are correct
        num_correct = round(accuracy * num_questions)
        correct_indices = set(random.sample(range(num_questions), num_correct))

        # Generate answers
        answers = []
        for q_idx in range(num_questions):
            if q_idx in correct_indices:
                # Correct answer
                answers.append(answer_key[q_idx])
            else:
                # Wrong answer - possibly blank or multi
                rand = random.random()
                if rand < blank_rate:
                    answers.append("")  # Blank
                elif rand < blank_rate + multi_rate:
                    # Multi-fill: pick 2 different choices
                    choices = random.sample(label_list, k=2)
                    answers.append(",".join(sorted(choices)))
                else:
                    # Single wrong answer
                    wrong_options = [l for l in label_list if l != answer_key[q_idx]]
                    answers.append(random.choice(wrong_options))

        # Calculate actual score (only single correct answers count)
        actual_correct = sum(
            1 for q_idx, ans in enumerate(answers)
            if ans == answer_key[q_idx]
        )

        students.append({
            "student_id": sid,
            "first_name": first_name,
            "last_name": last_name,
            "answers": answers,
            "expected_score": actual_correct / num_questions,
        })

    return students


# =============================================================================
# Bubble rendering
# =============================================================================

def grid_centers(
    x_tl: float, y_tl: float,
    x_br: float, y_br: float,
    numrows: int, numcols: int
) -> List[Tuple[float, float]]:
    """
    Compute normalized (0-1) center coordinates for a grid of bubbles.
    Returns list of (x, y) tuples in row-major order.
    """
    centers = []
    row_denom = max(1, numrows - 1)
    col_denom = max(1, numcols - 1)

    for r in range(numrows):
        y = y_tl + (y_br - y_tl) * (r / row_denom) if row_denom > 0 else y_tl
        for c in range(numcols):
            x = x_tl + (x_br - x_tl) * (c / col_denom) if col_denom > 0 else x_tl
            centers.append((x, y))

    return centers


def draw_filled_bubble(
    draw: ImageDraw.ImageDraw,
    cx: int, cy: int, radius: int,
    darkness: float = 1.0
):
    """
    Draw a filled bubble with variable darkness.

    Args:
        darkness: 0.0 = very light (barely visible), 1.0 = solid black
    """
    # Map darkness to grayscale and alpha
    # Light marks: high gray value (200), low alpha (100)
    # Dark marks: low gray value (0), high alpha (255)
    gray = int(200 * (1 - darkness))
    alpha = int(100 + 155 * darkness)

    rgba = (gray, gray, gray, alpha)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=rgba)


def fill_layout_by_columns(
    draw: ImageDraw.ImageDraw,
    layout: Dict[str, Any],
    text: str,
    img_w: int, img_h: int,
    darkness: float = 1.0
) -> int:
    """
    Fill bubbles column-by-column (for ID, name fields).
    Each character in text fills one column.
    """
    if not text:
        return 0

    labels = str(layout.get("labels", ""))
    numrows = int(layout.get("numrows", layout.get("questions", 0)))
    numcols = int(layout.get("numcols", layout.get("choices", 0)))
    radius_pct = float(layout.get("radius_pct", 0.008))

    x_tl = float(layout.get("x_topleft", 0))
    y_tl = float(layout.get("y_topleft", 0))
    x_br = float(layout.get("x_bottomright", 0))
    y_br = float(layout.get("y_bottomright", 0))

    centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
    radius_px = max(1, int(radius_pct * img_w))

    # Normalize text for matching
    text = str(text).upper()[:numcols]
    if any(c.isalpha() for c in labels):
        labels_upper = labels.upper()
    else:
        labels_upper = labels

    filled = 0
    for col_idx, char in enumerate(text):
        try:
            row_idx = labels_upper.index(char.upper() if char.isalpha() else char)
        except ValueError:
            continue

        # Index in row-major centers list
        idx = row_idx * numcols + col_idx
        if idx < len(centers):
            cx_pct, cy_pct = centers[idx]
            cx = int(cx_pct * img_w)
            cy = int(cy_pct * img_h)
            draw_filled_bubble(draw, cx, cy, radius_px, darkness)
            filled += 1

    return filled


def fill_layout_by_rows(
    draw: ImageDraw.ImageDraw,
    layout: Dict[str, Any],
    answers: List[str],
    img_w: int, img_h: int,
    darkness_range: Tuple[float, float] = (0.7, 1.0)
) -> int:
    """
    Fill bubbles row-by-row (for answer layouts).
    Each answer in the list fills one row.
    Supports blanks ("") and multi-fills ("A,B").
    """
    if not answers:
        return 0

    labels = str(layout.get("labels", "ABCDE"))
    numrows = int(layout.get("numrows", layout.get("questions", 0)))
    numcols = int(layout.get("numcols", layout.get("choices", 0)))
    radius_pct = float(layout.get("radius_pct", 0.008))

    x_tl = float(layout.get("x_topleft", 0))
    y_tl = float(layout.get("y_topleft", 0))
    x_br = float(layout.get("x_bottomright", 0))
    y_br = float(layout.get("y_bottomright", 0))

    centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
    radius_px = max(1, int(radius_pct * img_w))

    filled = 0
    for row_idx, answer in enumerate(answers[:numrows]):
        if not answer or answer.strip() == "":
            continue  # Blank answer

        # Handle multi-fill (e.g., "A,B")
        choices = [c.strip().upper() for c in answer.split(",")]

        for choice in choices:
            try:
                col_idx = labels.upper().index(choice)
            except ValueError:
                continue

            idx = row_idx * numcols + col_idx
            if idx < len(centers):
                cx_pct, cy_pct = centers[idx]
                cx = int(cx_pct * img_w)
                cy = int(cy_pct * img_h)
                # Random darkness within range
                darkness = random.uniform(*darkness_range)
                draw_filled_bubble(draw, cx, cy, radius_px, darkness)
                filled += 1

    return filled


def apply_random_transform(
    image: Image.Image,
    max_rotation: float = 1.0,
    max_translate_pct: float = 0.01
) -> Image.Image:
    """
    Apply slight random rotation and translation to simulate scan artifacts.

    Args:
        max_rotation: Maximum rotation in degrees (both directions)
        max_translate_pct: Maximum translation as fraction of image size
    """
    w, h = image.size

    # Random rotation
    angle = random.uniform(-max_rotation, max_rotation)

    # Random translation
    tx = int(w * random.uniform(-max_translate_pct, max_translate_pct))
    ty = int(h * random.uniform(-max_translate_pct, max_translate_pct))

    # Rotate around center, then translate
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))

    # Create new image with translation
    result = Image.new("RGB", (w, h), (255, 255, 255))
    result.paste(rotated, (tx, ty))

    return result


def render_student_sheet(
    template_image: Image.Image,
    student: Dict[str, Any],
    format_info: Dict[str, Any],
    version: str,
    darkness_range: Tuple[float, float] = (0.5, 1.0),
    apply_transform: bool = False,
) -> Image.Image:
    """
    Render a filled bubble sheet for one student.
    """
    img_w, img_h = template_image.size

    # Create overlay
    overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Random darkness for this student's marks (consistent within student)
    base_darkness = random.uniform(*darkness_range)
    darkness_var = 0.15  # Variation within a single sheet

    def get_darkness():
        return max(0.3, min(1.0, base_darkness + random.uniform(-darkness_var, darkness_var)))

    # Fill student ID
    if "id_layout" in format_info:
        fill_layout_by_columns(
            draw, format_info["id_layout"],
            student["student_id"],
            img_w, img_h,
            get_darkness()
        )

    # Fill names
    if "first_name_layout" in format_info:
        fill_layout_by_columns(
            draw, format_info["first_name_layout"],
            student["first_name"],
            img_w, img_h,
            get_darkness()
        )

    if "last_name_layout" in format_info:
        fill_layout_by_columns(
            draw, format_info["last_name_layout"],
            student["last_name"],
            img_w, img_h,
            get_darkness()
        )

    # Fill version
    if "version_layout" in format_info:
        layout = format_info["version_layout"]
        # Version is typically a single character in a row layout
        labels = str(layout.get("labels", "ABCD"))
        numrows = int(layout.get("numrows", 1))
        numcols = int(layout.get("numcols", len(labels)))
        radius_pct = float(layout.get("radius_pct", 0.008))

        x_tl = float(layout.get("x_topleft", 0))
        y_tl = float(layout.get("y_topleft", 0))
        x_br = float(layout.get("x_bottomright", 0))
        y_br = float(layout.get("y_bottomright", 0))

        centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
        radius_px = max(1, int(radius_pct * img_w))

        try:
            col_idx = labels.upper().index(version.upper())
            if col_idx < len(centers):
                cx_pct, cy_pct = centers[col_idx]
                cx = int(cx_pct * img_w)
                cy = int(cy_pct * img_h)
                draw_filled_bubble(draw, cx, cy, radius_px, get_darkness())
        except ValueError:
            pass

    # Fill answers
    answer_idx = 0
    for layout in format_info.get("answer_layouts", []):
        numrows = int(layout.get("numrows", layout.get("questions", 0)))
        layout_answers = student["answers"][answer_idx:answer_idx + numrows]

        # Per-bubble darkness variation
        darkness_per_row = [(get_darkness(), get_darkness()) for _ in range(numrows)]
        fill_layout_by_rows(
            draw, layout, layout_answers,
            img_w, img_h,
            darkness_range=(base_darkness - darkness_var, base_darkness + darkness_var)
        )
        answer_idx += numrows

    # Composite overlay onto template
    base = template_image.convert("RGBA")
    result = Image.alpha_composite(base, overlay)
    result = result.convert("RGB")

    # Apply random transform if requested
    if apply_transform:
        result = apply_random_transform(result)

    return result


# =============================================================================
# PDF I/O
# =============================================================================

def load_template_as_image(pdf_path: str, dpi: int = 300) -> Image.Image:
    """Load first page of PDF as PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc[0]

    # Render at specified DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    return img


def save_images_as_pdf(images: List[Image.Image], output_path: str, dpi: int = 300):
    """Save list of PIL Images as a multi-page PDF."""
    if not images:
        return

    # Convert to RGB if necessary
    rgb_images = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb_images.append(img)

    # Save as PDF
    rgb_images[0].save(
        output_path,
        save_all=True,
        append_images=rgb_images[1:],
        resolution=dpi,
    )


def write_students_csv(
    students: List[Dict[str, Any]],
    output_path: str,
    answer_key: List[str],
    num_questions: int
):
    """Write student data to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Header
        q_headers = [f"Q{i+1}" for i in range(num_questions)]
        writer = csv.writer(f)
        writer.writerow(["StudentID", "FirstName", "LastName", "Version", "Score"] + q_headers)

        # Answer key row
        writer.writerow(["ANSWER_KEY", "", "", "", ""] + answer_key)

        # Student rows
        for student in students:
            score_pct = round(student["expected_score"] * 100, 1)
            row = [
                student["student_id"],
                student["first_name"],
                student["last_name"],
                student.get("version", ""),
                score_pct,
            ] + student["answers"]
            writer.writerow(row)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic exam data for testing MarkShark pipelines."
    )
    parser.add_argument(
        "--template", required=True,
        help="Path to template PDF (e.g., master_template.pdf)"
    )
    parser.add_argument(
        "--bubblemap", required=True,
        help="Path to bubblemap YAML file"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--num-students", type=int, default=100,
        help="Number of fake students to generate (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for rendered images (default: 300)"
    )
    parser.add_argument(
        "--darkness-min", type=float, default=0.4,
        help="Minimum bubble darkness 0-1 (default: 0.4 = light marks)"
    )
    parser.add_argument(
        "--darkness-max", type=float, default=1.0,
        help="Maximum bubble darkness 0-1 (default: 1.0 = solid black)"
    )
    parser.add_argument(
        "--apply-transform", action="store_true",
        help="Apply slight random rotation/translation to challenge alignment"
    )
    parser.add_argument(
        "--blank-rate", type=float, default=0.02,
        help="Rate of blank answers among wrong answers (default: 0.02)"
    )
    parser.add_argument(
        "--multi-rate", type=float, default=0.02,
        help="Rate of multi-fill answers among wrong answers (default: 0.02)"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bubblemap and detect format
    print(f"Loading bubblemap: {args.bubblemap}")
    bubblemap = load_bubblemap(args.bubblemap)
    format_info = detect_format(bubblemap)

    print(f"\nDetected format:")
    print(f"  Total questions: {format_info['total_questions']}")
    print(f"  Answer labels: {format_info['answer_labels']}")
    print(f"  Has version field: {format_info['has_version']}")
    if format_info['has_version']:
        print(f"  Version labels: {format_info['version_labels']}")
    print(f"  Student ID digits: {format_info['id_digits']}")
    print(f"  Has first name: {format_info['has_first_name']}")
    print(f"  Has last name: {format_info['has_last_name']}")

    # Load template
    print(f"\nLoading template: {args.template}")
    template_image = load_template_as_image(args.template, args.dpi)
    print(f"  Template size: {template_image.size[0]}x{template_image.size[1]} pixels")

    # Generate answer keys
    print("\nGenerating answer key(s)...")
    if format_info['has_version']:
        # Generate 2 versions
        version_labels = format_info['version_labels']
        keys = generate_versioned_keys(
            format_info['total_questions'],
            format_info['answer_labels'],
            version_labels,
            num_versions=2
        )
        versions_to_use = list(keys.keys())
    else:
        # Single version
        key = generate_answer_key(
            format_info['total_questions'],
            format_info['answer_labels']
        )
        keys = {"": key}
        versions_to_use = [""]

    # Write answer key file
    key_path = out_dir / "mock_answer_key.txt"
    if format_info['has_version']:
        write_answer_key(keys, str(key_path))
        print(f"  Wrote versioned key to {key_path}")
        for ver, ans in keys.items():
            print(f"    Version {ver}: {','.join(ans[:5])}... ({len(ans)} answers)")
    else:
        # Single version format: just comma-separated
        with open(key_path, "w") as f:
            f.write(",".join(keys[""]) + "\n")
        print(f"  Wrote key to {key_path}")

    # Generate students
    print(f"\nGenerating {args.num_students} fake students...")
    all_students = []
    all_images = []

    for i in range(args.num_students):
        # Assign version (alternating if multiple versions)
        version = versions_to_use[i % len(versions_to_use)]
        answer_key = keys[version]

        # Generate one student
        students = generate_fake_students(
            num_students=1,
            answer_key=answer_key,
            answer_labels=format_info['answer_labels'],
            id_digits=format_info['id_digits'],
            blank_rate=args.blank_rate,
            multi_rate=args.multi_rate,
        )
        student = students[0]
        student["version"] = version
        all_students.append(student)

        # Render sheet
        sheet_image = render_student_sheet(
            template_image=template_image,
            student=student,
            format_info=format_info,
            version=version,
            darkness_range=(args.darkness_min, args.darkness_max),
            apply_transform=args.apply_transform,
        )
        all_images.append(sheet_image)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{args.num_students} students...")

    # Save PDF
    pdf_path = out_dir / "mock_scans.pdf"
    print(f"\nSaving PDF: {pdf_path}")
    save_images_as_pdf(all_images, str(pdf_path), args.dpi)

    # Save CSV
    csv_path = out_dir / "mock_student_answers.csv"
    print(f"Saving CSV: {csv_path}")
    # Use first version's key for CSV (or the only key)
    first_key = keys[versions_to_use[0]]
    write_students_csv(all_students, str(csv_path), first_key, format_info['total_questions'])

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    scores = [s["expected_score"] * 100 for s in all_students]
    print(f"  Students generated: {len(all_students)}")
    print(f"  Score range: {min(scores):.1f}% - {max(scores):.1f}%")
    print(f"  Mean score: {np.mean(scores):.1f}%")
    print(f"  Median score: {np.median(scores):.1f}%")

    blanks = sum(1 for s in all_students for a in s["answers"] if a == "")
    multis = sum(1 for s in all_students for a in s["answers"] if "," in a)
    total_answers = len(all_students) * format_info['total_questions']
    print(f"  Blank answers: {blanks} ({100*blanks/total_answers:.1f}%)")
    print(f"  Multi-fill answers: {multis} ({100*multis/total_answers:.1f}%)")

    print(f"\nOutput files:")
    print(f"  {key_path}")
    print(f"  {pdf_path}")
    print(f"  {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
