#!/usr/bin/env python3
"""
MarkShark Mock Dataset Generator

Generate synthetic exam data for testing MarkShark pipelines.

This module reads a bubblemap YAML and template PDF, auto-detects the format
(answer choices, version field, student ID digits, etc.), generates fake
students with realistic score distributions, and renders filled bubble sheets.

Features:
- Auto-detects answer format (A-E, A-D, 1-5, etc.) from bubblemap
- Auto-detects version field format and generates multiple versions if present
- Generates realistic student score distribution (beta distribution, ~20-100%)
- Includes ~2% blanks and ~2% multi-fills scattered among wrong answers
- Variable bubble darkness to simulate light/dark pencil marks
- Optional random rotation/translation to challenge alignment

Usage (as module):
    from markshark.mock_dataset import generate_mock_dataset

    generate_mock_dataset(
        template_path="path/to/master_template.pdf",
        bubblemap_path="path/to/bubblemap.yaml",
        out_dir="output_folder",
        num_students=100
    )

Output:
    - mock_answer_key.txt: Key file in modern MarkShark format (ver:A\\nA\\nB\\nC\\n...)
    - mock_scans.pdf: PDF of all synthesized student sheets
    - mock_student_responses.csv: CSV with student info and expected answers
"""

import csv
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise ImportError("PIL/Pillow is required. Install with: pip install pillow")

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")


# =============================================================================
# Bubblemap parsing (handles current MarkShark schema)
# =============================================================================

def load_bubblemap(path: str) -> Dict[str, Any]:
    """Load and parse a bubblemap YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_page_size_mm(metadata: Dict[str, Any]) -> Tuple[float, float]:
    """Extract page size in mm from metadata. Delegates to canonical implementation."""
    from .tools.bubblemap_io import _get_page_size_mm as _canonical
    return _canonical(metadata)


def detect_format(bubblemap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-detect format from bubblemap, supporting multi-page templates.

    Reads *_zone keys from the bubblemap YAML.

    Returns dict with:
        - total_questions: int
        - answer_labels: str (e.g., "ABCDE")
        - has_version: bool
        - version_labels: str or None (e.g., "ABCD" or "1234")
        - id_digits: int
        - has_first_name: bool
        - has_last_name: bool
        - num_pages: int
        - pages: list of page data dicts (one per page)
        - page_size_mm: tuple (width, height) in mm
    """
    result = {
        "total_questions": 0,
        "answer_labels": "ABCDE",
        "has_version": False,
        "version_labels": None,
        "id_digits": 10,
        "has_first_name": False,
        "has_last_name": False,
        "num_pages": 1,
        "pages": [],
        "page_size_mm": (215.9, 279.4),
        "styles": {},
    }

    # Find all page keys (page_1, page_2, etc.)
    page_keys = sorted([k for k in bubblemap if k.startswith("page_")])

    if not page_keys:
        print("Warning: No page_N key found in bubblemap, assuming flat structure")
        page_keys = ["_flat"]
        pages_data = [bubblemap]
    else:
        pages_data = [bubblemap[k] for k in page_keys]

    result["num_pages"] = len(page_keys)

    # Check metadata for total_questions and page size
    metadata = bubblemap.get("metadata", {})
    if metadata.get("total_questions"):
        result["total_questions"] = int(metadata["total_questions"])
    if metadata.get("pages"):
        result["num_pages"] = int(metadata["pages"])

    # Get page size for v3 coordinate conversion
    result["page_size_mm"] = _get_page_size_mm(metadata)

    # Extract styles (used for bubble_shape resolution, etc.)
    result["styles"] = bubblemap.get("styles", {})

    # Process each page
    all_answer_zones = []
    for page_idx, page_data in enumerate(pages_data):
        answer_zones = page_data.get("answer_zones", [])
        page_info = {
            "page_num": page_idx + 1,
            "answer_zones": answer_zones,
        }

        # Collect answer zones
        all_answer_zones.extend(answer_zones)

        # Get labels from first zone found
        if answer_zones and result["answer_labels"] == "ABCDE":
            first_zone = answer_zones[0]
            result["answer_labels"] = str(first_zone.get("labels", "ABCDE"))

        # Check for version zone (typically on page 1)
        version_zone = page_data.get("version_zone")
        if version_zone:
            result["has_version"] = True
            result["version_labels"] = str(version_zone.get("labels", "ABCD"))
            page_info["version_zone"] = version_zone

        # Check for ID zone (typically on page 1)
        id_zone = page_data.get("id_zone")
        if id_zone:
            result["id_digits"] = int(id_zone.get("numcols", id_zone.get("choices", 10)))
            page_info["id_zone"] = id_zone

        # Check for name zones (typically on page 1)
        first_name_zone = page_data.get("first_name_zone")
        last_name_zone = page_data.get("last_name_zone")

        if first_name_zone:
            result["has_first_name"] = True
            page_info["first_name_zone"] = first_name_zone
        if last_name_zone:
            result["has_last_name"] = True
            page_info["last_name_zone"] = last_name_zone

        result["pages"].append(page_info)

    # Sum up total questions if not in metadata
    if result["total_questions"] == 0:
        result["total_questions"] = sum(
            int(lay.get("numrows", lay.get("questions", 0)))
            for lay in all_answer_zones
        )

    return result


# =============================================================================
# Answer key generation
# =============================================================================

def generate_answer_key(
    num_questions: int,
    labels: str,
    num_and_keys: int = 0,
    num_or_keys: int = 0,
    default_points: int = 1,
    num_double_points: int = 0,
) -> List[str]:
    """Generate a random answer key.

    Args:
        num_questions: Number of questions.
        labels: Available answer labels (e.g. "ABCDE").
        num_and_keys: Number of AND key questions (e.g. "B&C").
        num_or_keys: Number of OR key questions (e.g. "B^C").
        default_points: Default points per question.
        num_double_points: Number of questions worth double points
            (or 1 point if default_points > 1).
    """
    label_list = list(labels)

    # Pre-select which question indices get special key types
    available = list(range(num_questions))
    random.shuffle(available)

    and_indices = set()
    or_indices = set()
    double_indices = set()

    if num_and_keys > 0 and len(label_list) >= 2:
        take = min(num_and_keys, len(available))
        and_indices = set(available[:take])
        available = available[take:]
    if num_or_keys > 0 and len(label_list) >= 2:
        take = min(num_or_keys, len(available))
        or_indices = set(available[:take])
        available = available[take:]
    if num_double_points > 0:
        take = min(num_double_points, len(available))
        double_indices = set(available[:take])

    # Build key
    key = []
    for q in range(num_questions):
        if q in and_indices:
            pair = sorted(random.sample(label_list, k=2))
            entry = "&".join(pair)
        elif q in or_indices:
            pair = sorted(random.sample(label_list, k=2))
            entry = "^".join(pair)
        else:
            entry = random.choice(label_list)

        # Append point override if this question has non-default points
        if q in double_indices:
            pts = 1 if default_points > 1 else default_points * 2
            entry = f"{entry}:{pts}"

        key.append(entry)
    return key


def generate_versioned_keys(
    num_questions: int,
    labels: str,
    version_labels: str,
    num_versions: int = 2,
    num_and_keys: int = 0,
    num_or_keys: int = 0,
    default_points: int = 1,
    num_double_points: int = 0,
) -> Dict[str, List[str]]:
    """
    Generate answer keys for multiple versions.

    Each version gets a shuffled variant of the base key to simulate
    different question orderings.
    """
    versions = list(version_labels)[:num_versions]
    keys = {}

    # Generate base key for first version
    base_key = generate_answer_key(
        num_questions, labels,
        num_and_keys=num_and_keys, num_or_keys=num_or_keys,
        default_points=default_points, num_double_points=num_double_points,
    )
    keys[versions[0]] = base_key

    # For subsequent versions, shuffle the key (simulating reordered questions)
    for ver in versions[1:]:
        # Create a permuted version - swap ~30% of answers
        # Only swap simple (single-letter) answers; keep compound/pointed keys intact
        permuted = base_key.copy()
        simple_indices = [i for i, k in enumerate(base_key)
                          if "&" not in k and "^" not in k and ":" not in k]
        num_swaps = len(simple_indices) // 3
        if num_swaps > 0:
            swap_indices = random.sample(simple_indices, num_swaps)
            for idx in swap_indices:
                other_labels = [l for l in labels if l != permuted[idx]]
                permuted[idx] = random.choice(other_labels)
        keys[ver] = permuted

    return keys


def write_answer_key(
    keys: Dict[str, List[str]],
    output_path: str,
    default_points: int = 1,
):
    """
    Write answer key in modern MarkShark format.

    Format:
        ver:A default:2
        A
        B&C
        D:4
        ...
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (version, answers) in enumerate(keys.items()):
            if i > 0:
                f.write("\n")  # Blank line between versions
            header = f"ver:{version}"
            if default_points != 1:
                header += f" default:{default_points}"
            f.write(header + "\n")
            for answer in answers:
                f.write(f"{answer}\n")


# =============================================================================
# Fake student generation
# =============================================================================

FIRST_NAMES = [
    'Abby', 'Adam', 'Alan', 'Amanda', 'Andrew', 'Anna', 'Anthony', 'Ashley', 'Avery',
    'Barbara', 'Betty', 'Blair', 'Brent', 'Brian', 'Carla', 'Carol', 'Charles', 'Chen',
    'Christopher', 'Clara', 'Daniel', 'David', 'Deborah', 'Derek', 'Diana', 'Donald',
    'Donna', 'Dorothy', 'Edward', 'Elizabeth', 'Emily', 'Fatima', 'Fiona', 'George',
    'Hiroshi', 'Jack', 'James', 'Jared', 'Jason', 'Jeffrey', 'Jenna', 'Jennifer',
    'Jessica', 'Joseph', 'Joshua', 'Karen', 'Kenneth', 'Kevin', 'Kimberly', 'Lin',
    'Linda', 'Lisa', 'Luis', 'Margaret', 'Maria', 'Mark', 'Mary', 'Mason', 'Matthew',
    'Mei', 'Melissa', 'Michael', 'Michelle', 'Mohammed', 'Molly', 'Nancy', 'Nate',
    'Nina', 'Omar', 'Owen', 'Patricia', 'Paul', 'Pearl', 'Priya', 'Quinn', 'Raj',
    'Rebecca', 'Richard', 'Robert', 'Ronald', 'Rosie', 'Sandra', 'Sarah', 'Sharon',
    'Stephanie', 'Steven', 'Susan', 'Thomas', 'Timothy', 'Tyler', 'Wade', 'Wei',
    'William', 'Yara', 'Yuki', 'Zack', 'Zara', 'Zoey',
]

LAST_NAMES = [
    'Abdi', 'Adams', 'Ahmed', 'Alexander', 'Ali', 'Ali-Hassan', 'Allen', 'Anderson',
    'Andersson', 'Araya', 'Aung', 'Bae', 'Bailey', 'Baker', 'Bakker', 'Banerjee',
    'Barker', 'Barnes', 'Barros', 'Becker', 'Bell', 'Bennett', 'Berg', 'Bhat',
    'Bianchi', 'Bishop', 'Black', 'Blanco', 'Boyd', 'Brooks', 'Brown', 'Bryant',
    'Burke', 'Burns', 'Butler', 'Campbell', 'Carter', 'Castillo', 'Chang', 'Chen',
    'Chen-Wang', 'Choi', 'Clark', 'Cohen', 'Cohen-Levy', 'Cole', 'Coleman', 'Collins',
    'Cook', 'Cooper', 'Costa', 'Cox', 'Crawford', 'Cruz', 'Davis', 'Davis-Clark',
    'De La Rosa', 'De Leon', 'Del Toro', 'Delgado', 'Demir', 'Desai', 'Di Marco',
    'Diallo', 'Diaz', 'Dixon', 'Dominguez', 'Dong', 'Drake', 'Dubois',
    'Dubois-Laurent', 'Duran', 'Edwards', 'Ellis', 'ElSayed', 'Espinoza', 'Evans',
    'Farah', 'Fernandez', 'Fischer', 'Fischer-Klein', 'Fisher', 'Flores', 'Ford',
    'Foster', 'Freeman', 'Frost', 'Garcia', 'Garcia-Lopez', 'Ghosh', 'Gibson', 'Gomez',
    'Gonzalez', 'Gordon', 'Graham', 'Gray', 'Green', 'Griffin', 'Guerrero', 'Gupta',
    'Hall', 'Hamilton', 'Harris', 'Harrison', 'Hassan', 'Hayes', 'Henderson', 'Henry',
    'Hernandez', 'Hicks', 'Holmes', 'Holt', 'Horvat', 'Howard', 'Hughes', 'Hunt',
    'Hunter', 'Inoue', 'Ito', 'Jackson', 'James', 'Janssen', 'Jenkins', 'Johansson',
    'Johnson', 'Johnson-Williams', 'Jones', 'Jordan', 'Kamara', 'Kaya', 'Kennedy',
    'Kim', 'Kim-Park', 'Kimura', 'King', 'Klein', 'Kobayashi', 'Kowalski', 'Kruger',
    'Kumar', 'Kuznetsov', 'La Fontaine', 'Larsson', 'Laurent', 'Le', 'Lee',
    'Lee-Martin', 'Lewis', 'Li', 'Lima', 'Lin', 'Long', 'Lopez', 'Lowe', 'Ma',
    'Mac Alister', 'MacArthur', 'MacDonald', 'MacGregor', 'Machado', 'Marshall',
    'Martin', 'Martinez', 'Mason', 'McBride', 'McCarthy', 'McConnell', 'McDonald',
    'McGraw', 'Medina', 'Mehta', 'Mendoza', 'Mensah', 'Miller', 'Mohammed', 'Moore',
    'Morales', 'Moreno', 'Morgan', 'Morris', 'Muller', 'Murphy', 'Murray', 'Myers',
    'Nakamura', 'Nilsson', 'Nkosi', 'Novak', 'Nunez', "O'Brien", "O'Connor",
    "O'Malley", "O'Neill", 'Okafor', 'Ortega', 'Ortiz', 'Owens', 'Palmer', 'Park',
    'Parker', 'Patel', 'Patel-Shah', 'Patterson', 'Pereira', 'Perez', 'Perry',
    'Peterson', 'Petrov', 'Petrov-Ivanov', 'Phillips', 'Popescu', 'Porter', 'Powell',
    'Price', 'Qureshi', 'Rahman', 'Ramirez', 'Ramos', 'Reddy', 'Reed', 'Reyes',
    'Reynolds', 'Rice', 'Rivera', 'Rivera-Diaz', 'Robinson', 'Rodriguez', 'Rogers',
    'Rojas', 'Ross', 'Rossi', 'Rossi-Bianchi', 'Russell', 'Saito', 'Sanchez', 'Santos',
    'Santos-Cruz', 'Schneider', 'Scott', 'Shaw', 'Silva', 'Simmons', 'Simpson', 'Singh',
    'Smith', 'Smith-Jones', 'Spencer', 'St. Claire', 'St. James', 'Stevens', 'Stewart',
    'Stone', 'Sullivan', 'Suzuki', 'Tanaka', 'Tanaka-Suzuki', 'Taylor', 'Taylor-Brown',
    'Thomas', 'Thompson', 'Torres', 'Torres-Reyes', 'Toure', 'Traore', 'Tucker',
    'Van Der Berg', 'Van Dyke', 'Vargas', 'Voinova', 'Von Trapp', 'Wallace', 'Wang',
    'Ward', 'Warren', 'Washington', 'Watanabe', 'Watson', 'Webb', 'Weber', 'Wells',
    'West', 'White', 'Williams', 'Wilson', 'Wilson-Moore', 'Wood', 'Woods', 'Wright',
    'Yamamoto', 'Yilmaz', 'Young',
]


def generate_student_id(num_digits: int) -> str:
    """Generate a random student ID (doesn't start with 0)."""
    first = str(random.randint(1, 9))
    rest = ''.join(random.choices(string.digits, k=num_digits - 1))
    return first + rest


def corrupt_student_id(sid: str) -> str:
    """Corrupt a student ID to simulate a mis-entry on the bubble sheet.

    Four equally-weighted error types:
    - single-digit typo (change one digit to a different digit)
    - adjacent transposition (swap two adjacent digits)
    - extra digit (insert a random digit)
    - missing digit (delete one digit)
    """
    error_type = random.choice(["typo", "transpose", "extra", "missing"])

    if error_type == "typo":
        pos = random.randrange(len(sid))
        original = sid[pos]
        replacement = random.choice([d for d in string.digits if d != original])
        return sid[:pos] + replacement + sid[pos + 1:]

    elif error_type == "transpose":
        if len(sid) < 2:
            return sid
        pos = random.randrange(len(sid) - 1)
        lst = list(sid)
        lst[pos], lst[pos + 1] = lst[pos + 1], lst[pos]
        # Ensure we actually changed something (adjacent digits might be same)
        if "".join(lst) == sid:
            return corrupt_student_id(sid)
        return "".join(lst)

    elif error_type == "extra":
        pos = random.randrange(len(sid) + 1)
        return sid[:pos] + random.choice(string.digits) + sid[pos:]

    else:  # missing
        if len(sid) < 2:
            return sid
        pos = random.randrange(len(sid))
        return sid[:pos] + sid[pos + 1:]


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
            raw_key = answer_key[q_idx]
            # Strip point suffix (e.g. "A:2" -> "A", "B&C:4" -> "B&C")
            key_ans = raw_key.split(":")[0] if ":" in raw_key else raw_key
            is_and = "&" in key_ans
            is_or = "^" in key_ans

            if q_idx in correct_indices:
                # Correct answer
                if is_and:
                    # AND key "B&C" -> fill all required bubbles
                    parts = sorted(key_ans.split("&"))
                    answers.append(",".join(parts))
                elif is_or:
                    # OR key "B^C" -> pick one of the acceptable answers
                    parts = key_ans.split("^")
                    answers.append(random.choice(parts))
                else:
                    answers.append(key_ans)
            else:
                # Wrong answer - possibly blank or multi
                rand = random.random()
                if rand < blank_rate:
                    answers.append("")  # Blank
                elif rand < blank_rate + multi_rate:
                    # Multi-fill: pick 2 different choices
                    choices = random.sample(label_list, k=2)
                    answers.append(",".join(sorted(choices)))
                elif is_and:
                    # Wrong for AND key: partial fill (one of the required
                    # bubbles) or a completely wrong single bubble
                    parts = key_ans.split("&")
                    if random.random() < 0.6:
                        # Partial fill — pick only one required bubble
                        answers.append(random.choice(parts))
                    else:
                        # Completely wrong single bubble
                        wrong = [l for l in label_list if l not in parts]
                        answers.append(random.choice(wrong) if wrong else "")
                elif is_or:
                    # Wrong for OR key: pick a label NOT in the OR set
                    parts = key_ans.split("^")
                    wrong = [l for l in label_list if l not in parts]
                    answers.append(random.choice(wrong) if wrong else "")
                else:
                    # Single wrong answer
                    wrong_options = [l for l in label_list if l != key_ans]
                    answers.append(random.choice(wrong_options))

        # Calculate actual score — check each answer against its key
        actual_correct = 0
        for q_idx, ans in enumerate(answers):
            raw_key = answer_key[q_idx]
            key_ans = raw_key.split(":")[0] if ":" in raw_key else raw_key
            if "&" in key_ans:
                # AND: student must fill exactly the required set
                required = set(key_ans.split("&"))
                filled = set(a.strip() for a in ans.split(",")) if ans else set()
                if filled == required:
                    actual_correct += 1
            elif "^" in key_ans:
                # OR: student fills one bubble, it must be in the accepted set
                accepted = set(key_ans.split("^"))
                if ans in accepted and "," not in ans:
                    actual_correct += 1
            else:
                if ans == key_ans:
                    actual_correct += 1

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
    numrows: int, numcols: int,
) -> List[Tuple[float, float]]:
    """Compute normalized (0-1) center coordinates for a grid of bubbles.

    Delegates to the canonical implementation in score_tools.
    """
    from .tools.score_tools import grid_centers_axis_mode
    return grid_centers_axis_mode(x_tl, y_tl, x_br, y_br, numrows, numcols)


def draw_filled_bubble(
    draw: ImageDraw.ImageDraw,
    cx: int, cy: int, rx: int,
    darkness: float = 1.0,
    ry: int = None,
):
    """
    Draw a filled bubble (circle or oval) with variable darkness.

    Args:
        cx, cy: Centre pixel coordinates.
        rx: Horizontal radius in pixels.
        darkness: 0.0 = very light (barely visible), 1.0 = solid black.
        ry: Vertical radius in pixels.  If *None*, uses *rx* (circle).
    """
    if ry is None:
        ry = rx
    # Map darkness to grayscale and alpha
    # Light marks: high gray value (200), low alpha (100)
    # Dark marks: low gray value (0), high alpha (255)
    gray = int(200 * (1 - darkness))
    alpha = int(100 + 155 * darkness)

    rgba = (gray, gray, gray, alpha)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rgba)


def _get_layout_coords(layout: Dict[str, Any], page_size_mm: Tuple[float, float] = (215.9, 279.4)) -> Tuple[float, float, float, float, float]:
    """
    Extract normalized coordinates from a layout, supporting both v1 and v3 formats.

    v1: x_topleft, y_topleft, x_bottomright, y_bottomright, radius_pct (0.0-1.0)
    v3: x_mm, y_mm, width_mm, height_mm, bubble_diameter_mm (millimeters)

    Returns: (x_tl, y_tl, x_br, y_br, radius_pct) all as normalized 0.0-1.0 values
    """
    # Check which format
    has_v3 = "x_mm" in layout

    if has_v3:
        page_w, page_h = page_size_mm
        x_mm = float(layout.get("x_mm", 0))
        y_mm = float(layout.get("y_mm", 0))
        width_mm = float(layout.get("width_mm", 0))
        height_mm = float(layout.get("height_mm", 0))
        diameter_mm = float(layout.get("bubble_diameter_mm", 4.0))

        x_tl = x_mm / page_w
        y_tl = y_mm / page_h
        x_br = (x_mm + width_mm) / page_w
        y_br = (y_mm + height_mm) / page_h
        radius_pct = (diameter_mm / 2) / page_w
    else:
        x_tl = float(layout.get("x_topleft", 0))
        y_tl = float(layout.get("y_topleft", 0))
        x_br = float(layout.get("x_bottomright", 0))
        y_br = float(layout.get("y_bottomright", 0))
        radius_pct = float(layout.get("radius_pct", 0.008))

    return x_tl, y_tl, x_br, y_br, radius_pct


def _resolve_bubble_shape(layout: Dict[str, Any],
                          styles: Dict[str, Any] = None) -> str:
    """Resolve the bubble_shape for a layout, checking style references.

    Precedence:
      1. ``layout["bubble_shape"]`` (explicit on the zone)
      2. style referenced by ``layout["style"]`` → ``bubble_shape``
      3. style's ``extends`` chain
      4. ``"circle"`` (default)
    """
    # Direct on layout
    if "bubble_shape" in layout:
        return str(layout["bubble_shape"]).lower()

    if styles and "style" in layout:
        style_name = layout["style"]
        visited: set = set()
        while style_name and style_name not in visited:
            visited.add(style_name)
            style = styles.get(style_name, {})
            if "bubble_shape" in style:
                return str(style["bubble_shape"]).lower()
            style_name = style.get("extends")

    return "circle"


def _get_bubble_radii(
    layout: Dict[str, Any],
    img_w: int, img_h: int,
    page_size_mm: Tuple[float, float] = (215.9, 279.4),
    styles: Dict[str, Any] = None,
) -> Tuple[int, int]:
    """Return (rx, ry) pixel radii for a bubble in *layout*.

    For ``bubble_shape == "oval"``, the radii are derived from the grid
    cell dimensions so the fill matches the printed oval proportions.
    For circles, rx == ry.
    """
    x_tl, y_tl, x_br, y_br, radius_pct = _get_layout_coords(layout, page_size_mm)
    rx = max(1, int(radius_pct * img_w))

    shape = _resolve_bubble_shape(layout, styles)
    if shape == "oval":
        numrows = int(layout.get("numrows", layout.get("questions", 1)))
        numcols = int(layout.get("numcols", layout.get("choices", 1)))
        # Cell dimensions as fractions of image
        cell_w = (x_br - x_tl) / max(numcols, 1)
        cell_h = (y_br - y_tl) / max(numrows, 1)
        # Convert to pixels
        cell_w_px = cell_w * img_w
        cell_h_px = cell_h * img_h
        # Oval radii: fill ~70% of cell in each direction
        fill_frac = 0.70
        rx = max(1, int(cell_w_px * fill_frac / 2))
        ry = max(1, int(cell_h_px * fill_frac / 2))
    else:
        ry = rx

    return rx, ry


def fill_layout_by_columns(
    draw: ImageDraw.ImageDraw,
    layout: Dict[str, Any],
    text: str,
    img_w: int, img_h: int,
    darkness: float = 1.0,
    page_size_mm: Tuple[float, float] = (215.9, 279.4),
    styles: Dict[str, Any] = None,
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

    x_tl, y_tl, x_br, y_br, _radius_pct = _get_layout_coords(layout, page_size_mm)

    centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
    rx, ry = _get_bubble_radii(layout, img_w, img_h, page_size_mm, styles)

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
            draw_filled_bubble(draw, cx, cy, rx, darkness, ry)
            filled += 1

    return filled


def fill_layout_by_rows(
    draw: ImageDraw.ImageDraw,
    layout: Dict[str, Any],
    answers: List[str],
    img_w: int, img_h: int,
    darkness_range: Tuple[float, float] = (0.7, 1.0),
    page_size_mm: Tuple[float, float] = (215.9, 279.4),
    styles: Dict[str, Any] = None,
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

    x_tl, y_tl, x_br, y_br, _radius_pct = _get_layout_coords(layout, page_size_mm)

    centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
    rx, ry = _get_bubble_radii(layout, img_w, img_h, page_size_mm, styles)

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
                draw_filled_bubble(draw, cx, cy, rx, darkness, ry)
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


def render_student_sheets(
    template_images: List[Image.Image],
    student: Dict[str, Any],
    format_info: Dict[str, Any],
    version: str,
    darkness_range: Tuple[float, float] = (0.5, 1.0),
    apply_transform: bool = False,
    page_size_mm: Tuple[float, float] = (215.9, 279.4),
    skip_version: bool = False,
) -> List[Image.Image]:
    """
    Render filled bubble sheets for one student (one image per template page).
    """
    # Random darkness for this student's marks (consistent within student)
    base_darkness = random.uniform(*darkness_range)
    darkness_var = 0.15  # Variation within a single sheet

    def get_darkness():
        return max(0.3, min(1.0, base_darkness + random.uniform(-darkness_var, darkness_var)))

    # Styles dict for bubble shape resolution (oval vs circle)
    styles = format_info.get("styles", {})

    result_images = []
    answer_idx = 0  # Track position across all pages

    for page_idx, template_image in enumerate(template_images):
        img_w, img_h = template_image.size

        # Create overlay for this page
        overlay = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Get page-specific info
        if page_idx < len(format_info.get("pages", [])):
            page_info = format_info["pages"][page_idx]
        else:
            page_info = {}

        # Fill student ID (typically page 1 only)
        if "id_zone" in page_info:
            fill_layout_by_columns(
                draw, page_info["id_zone"],
                student["student_id"],
                img_w, img_h,
                get_darkness(),
                page_size_mm,
                styles,
            )

        # Fill names (typically page 1 only)
        if "first_name_zone" in page_info:
            fill_layout_by_columns(
                draw, page_info["first_name_zone"],
                student["first_name"],
                img_w, img_h,
                get_darkness(),
                page_size_mm,
                styles,
            )

        if "last_name_zone" in page_info:
            fill_layout_by_columns(
                draw, page_info["last_name_zone"],
                student["last_name"],
                img_w, img_h,
                get_darkness(),
                page_size_mm,
                styles,
            )

        # Fill version (typically page 1 only)
        if "version_zone" in page_info and not skip_version:
            layout = page_info["version_zone"]
            labels = str(layout.get("labels", "ABCD"))
            numrows = int(layout.get("numrows", 1))
            numcols = int(layout.get("numcols", len(labels)))

            x_tl, y_tl, x_br, y_br, _radius_pct = _get_layout_coords(layout, page_size_mm)

            centers = grid_centers(x_tl, y_tl, x_br, y_br, numrows, numcols)
            rx, ry = _get_bubble_radii(layout, img_w, img_h, page_size_mm, styles)

            try:
                col_idx = labels.upper().index(version.upper())
                if col_idx < len(centers):
                    cx_pct, cy_pct = centers[col_idx]
                    cx = int(cx_pct * img_w)
                    cy = int(cy_pct * img_h)
                    draw_filled_bubble(draw, cx, cy, rx, get_darkness(), ry)
            except ValueError:
                pass

        # Fill answers for this page
        for layout in page_info.get("answer_zones", []):
            numrows = int(layout.get("numrows", layout.get("questions", 0)))
            layout_answers = student["answers"][answer_idx:answer_idx + numrows]

            fill_layout_by_rows(
                draw, layout, layout_answers,
                img_w, img_h,
                darkness_range=(base_darkness - darkness_var, base_darkness + darkness_var),
                page_size_mm=page_size_mm,
                styles=styles,
            )
            answer_idx += numrows

        # Composite overlay onto template
        base = template_image.convert("RGBA")
        result = Image.alpha_composite(base, overlay)
        result = result.convert("RGB")

        # Apply random transform if requested
        if apply_transform:
            result = apply_random_transform(result)

        result_images.append(result)

    return result_images


# =============================================================================
# PDF I/O
# =============================================================================

def load_template_pages(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """Load all pages of PDF as PIL Images."""
    doc = fitz.open(pdf_path)
    images = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


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


def _write_student_row(writer, student: Dict[str, Any]):
    """Write a single student row to the CSV writer."""
    score_pct = round(student["expected_score"] * 100, 1)
    original_id = student.get("original_id", student["student_id"])
    bubbled_id = student["student_id"]
    id_error = "Y" if student.get("id_error") else ""
    missing_ver = "Y" if student.get("missing_version") else ""
    # If student forgot to bubble version, the CSV should show blank
    # (the scoring engine won't detect a version from the sheet)
    version_val = "" if student.get("missing_version") else student.get("version", "")
    row = [
        original_id,
        bubbled_id,
        student["first_name"],
        student["last_name"],
        version_val,
        score_pct,
        id_error,
        missing_ver,
    ] + student["answers"]
    writer.writerow(row)


def write_students_csv(
    students: List[Dict[str, Any]],
    output_path: str,
    keys: Dict[str, List[str]],
    num_questions: int
):
    """Write student data to CSV.

    Args:
        students: list of student dicts (from generate loop)
        output_path: CSV file path
        keys: dict mapping version letter -> answer key list
        num_questions: total questions
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # Header
        q_headers = [f"Q{i+1}" for i in range(num_questions)]
        writer = csv.writer(f)
        writer.writerow([
            "OriginalID", "BubbledID", "FirstName", "LastName", "Version",
            "Score", "IDError", "MissingVersion",
        ] + q_headers)

        # Group by version: key row then its students, for each version
        sorted_versions = sorted(keys.keys()) or [""]
        # Index students by version
        by_version: Dict[str, List[Dict[str, Any]]] = {v: [] for v in sorted_versions}
        no_version: List[Dict[str, Any]] = []
        for student in students:
            sv = student.get("version", "")
            # Missing-version students go under their assigned version
            # but with blank version column (handled below)
            if sv in by_version:
                by_version[sv].append(student)
            else:
                no_version.append(student)

        for ver in sorted_versions:
            ver_label = ver if ver else "A"
            writer.writerow(
                ["ANSWER_KEY", "", "", "", ver_label, "", "", ""] + keys[ver]
            )
            for student in by_version[ver]:
                _write_student_row(writer, student)

        # Any students whose version wasn't in keys (shouldn't happen)
        for student in no_version:
            _write_student_row(writer, student)


def generate_absent_students(
    num_absent: int,
    id_digits: int = 10,
    used_ids: Optional[set] = None
) -> List[Dict[str, Any]]:
    """
    Generate absent students (name and ID only, no answers).

    Args:
        num_absent: Number of absent students to generate
        id_digits: Number of digits in student ID
        used_ids: Set of already-used student IDs to avoid duplicates

    Returns:
        List of dicts with: student_id, first_name, last_name, absent=True
    """
    if used_ids is None:
        used_ids = set()

    absent_students = []
    for _ in range(num_absent):
        # Generate unique ID
        while True:
            sid = generate_student_id(id_digits)
            if sid not in used_ids:
                used_ids.add(sid)
                break

        absent_students.append({
            "student_id": sid,
            "first_name": random.choice(FIRST_NAMES),
            "last_name": random.choice(LAST_NAMES),
            "absent": True,
        })

    return absent_students


def write_roster_csv(
    students: List[Dict[str, Any]],
    absent_students: List[Dict[str, Any]],
    output_path: str
):
    """
    Write class roster to CSV (all students including absent ones).

    The roster contains StudentID, FirstName, LastName columns only.
    This can be used as input for scoring to identify absent students.
    """
    # Combine and shuffle all students
    all_students = students + absent_students
    random.shuffle(all_students)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["StudentID", "FirstName", "LastName"])

        for student in all_students:
            # Use original (real) ID for the roster — the roster is the
            # teacher's authoritative record, not what the student bubbled.
            roster_id = student.get("original_id", student["student_id"])
            writer.writerow([
                roster_id,
                student["first_name"],
                student["last_name"],
            ])


# =============================================================================
# Main API function
# =============================================================================

def generate_mock_dataset(
    template_path: str,
    bubblemap_path: str,
    out_dir: str,
    num_students: int = 100,
    num_absent: int = 2,
    num_versions: int = 2,
    seed: int = 42,
    dpi: int = 150,
    darkness_min: float = 0.4,
    darkness_max: float = 1.0,
    apply_transform: bool = False,
    blank_rate: float = 0.01,
    multi_rate: float = 0.01,
    num_id_errors: int = 2,
    num_missing_version: int = 2,
    num_and_keys: int = 0,
    num_or_keys: int = 0,
    default_points: int = 1,
    num_double_points: int = 0,
    verbose: bool = True,
) -> Dict[str, Path]:
    """
    Generate a complete mock dataset from a template.

    Args:
        template_path: Path to template PDF (master_template.pdf)
        bubblemap_path: Path to bubblemap YAML file
        out_dir: Output directory for generated files
        num_students: Number of fake students to generate (default: 100)
        num_absent: Number of absent students to add to roster (default: 2)
        num_versions: Number of exam versions to generate (default: 2).
            Only used when the template has a version field.
            Set to 1 for a single version, up to the number of version
            labels defined in the bubblemap (e.g. 4 for "ABCD").
        seed: Random seed for reproducibility (default: 42)
        dpi: DPI for rendered images (default: 150)
        darkness_min: Minimum bubble darkness 0-1 (default: 0.4)
        darkness_max: Maximum bubble darkness 0-1 (default: 1.0)
        apply_transform: Apply slight random rotation/translation (default: False)
        blank_rate: Rate of blank answers among wrong answers (default: 0.01)
        multi_rate: Rate of multi-fill answers among wrong answers (default: 0.01)
        num_id_errors: Number of students with corrupted IDs (default: 2)
        num_missing_version: Number of students with blank version field (default: 2)
        num_and_keys: Number of AND key questions in answer key (default: 0)
        num_or_keys: Number of OR key questions in answer key (default: 0)
        default_points: Default points per question (default: 1)
        num_double_points: Number of questions worth double points, or
            1 point if default_points > 1 (default: 0)
        verbose: Print progress messages (default: True)

    Returns:
        Dictionary with paths to generated files:
        - 'answer_key': Path to mock_answer_key.txt
        - 'scans': Path to mock_scans.pdf
        - 'responses': Path to mock_student_responses.csv
        - 'roster': Path to mock_roster.csv (includes absent students)
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Create output directory
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Load bubblemap and detect format
    if verbose:
        print(f"Loading bubblemap: {bubblemap_path}")
    bubblemap = load_bubblemap(bubblemap_path)
    format_info = detect_format(bubblemap)

    if verbose:
        print(f"\nDetected format:")
        print(f"  Total questions: {format_info['total_questions']}")
        print(f"  Answer labels: {format_info['answer_labels']}")
        print(f"  Has version field: {format_info['has_version']}")
        if format_info['has_version']:
            print(f"  Version labels: {format_info['version_labels']}")
        print(f"  Student ID digits: {format_info['id_digits']}")
        print(f"  Has first name: {format_info['has_first_name']}")
        print(f"  Has last name: {format_info['has_last_name']}")
        print(f"  Number of pages: {format_info['num_pages']}")

    # Load template
    if verbose:
        print(f"\nLoading template: {template_path}")
    template_images = load_template_pages(template_path, dpi)
    if verbose:
        print(f"  Template pages: {len(template_images)}")
        print(f"  Page size: {template_images[0].size[0]}x{template_images[0].size[1]} pixels")

    # Generate answer keys
    if verbose:
        print("\nGenerating answer key(s)...")
    if format_info['has_version']:
        version_labels = format_info['version_labels']
        # Clamp num_versions to available labels
        actual_num_versions = max(1, min(num_versions, len(version_labels)))
        keys = generate_versioned_keys(
            format_info['total_questions'],
            format_info['answer_labels'],
            version_labels,
            num_versions=actual_num_versions,
            num_and_keys=num_and_keys,
            num_or_keys=num_or_keys,
            default_points=default_points,
            num_double_points=num_double_points,
        )
        versions_to_use = list(keys.keys())
    else:
        # Single version
        key = generate_answer_key(
            format_info['total_questions'],
            format_info['answer_labels'],
            num_and_keys=num_and_keys,
            num_or_keys=num_or_keys,
            default_points=default_points,
            num_double_points=num_double_points,
        )
        keys = {"": key}
        versions_to_use = [""]

    # Write answer key file
    key_path = out_dir_path / "mock_answer_key.txt"
    if format_info['has_version']:
        write_answer_key(keys, str(key_path), default_points=default_points)
        if verbose:
            print(f"  Wrote versioned key to {key_path}")
            for ver, ans in keys.items():
                print(f"    Version {ver}: {','.join(ans[:5])}... ({len(ans)} answers)")
    else:
        # Single version format: modern format with ver:A header
        with open(key_path, "w") as f:
            header = "ver:A"
            if default_points != 1:
                header += f" default:{default_points}"
            f.write(header + "\n")
            for answer in keys[""]:
                f.write(f"{answer}\n")
        if verbose:
            print(f"  Wrote key to {key_path}")

    # Generate students
    if verbose:
        print(f"\nGenerating {num_students} fake students...")
    all_students = []
    all_images = []

    # Pre-select which students will have ID errors / missing version
    id_error_indices = set()
    if num_id_errors > 0:
        id_error_indices = set(random.sample(
            range(num_students), min(num_id_errors, num_students)
        ))
    missing_ver_indices = set()
    if num_missing_version > 0:
        missing_ver_indices = set(random.sample(
            range(num_students), min(num_missing_version, num_students)
        ))

    for i in range(num_students):
        # Assign version (alternating if multiple versions)
        version = versions_to_use[i % len(versions_to_use)]
        answer_key = keys[version]

        # Generate one student
        students = generate_fake_students(
            num_students=1,
            answer_key=answer_key,
            answer_labels=format_info['answer_labels'],
            id_digits=format_info['id_digits'],
            blank_rate=blank_rate,
            multi_rate=multi_rate,
        )
        student = students[0]
        student["version"] = version

        # Apply ID corruption if this student was pre-selected
        if i in id_error_indices:
            original_sid = student["student_id"]
            student["original_id"] = original_sid
            student["student_id"] = corrupt_student_id(original_sid)
            student["id_error"] = True

        # Apply missing-version error if pre-selected
        skip_ver = i in missing_ver_indices
        if skip_ver:
            student["missing_version"] = True

        all_students.append(student)

        # Render sheet(s) - one per template page
        sheet_images = render_student_sheets(
            template_images=template_images,
            student=student,
            format_info=format_info,
            version=version,
            darkness_range=(darkness_min, darkness_max),
            apply_transform=apply_transform,
            page_size_mm=format_info.get('page_size_mm', (215.9, 279.4)),
            skip_version=skip_ver,
        )
        all_images.extend(sheet_images)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_students} students...")

    # Save PDF
    pdf_path = out_dir_path / "mock_scans.pdf"
    if verbose:
        print(f"\nSaving PDF: {pdf_path}")
    save_images_as_pdf(all_images, str(pdf_path), dpi)

    # Save CSV
    csv_path = out_dir_path / "mock_student_responses.csv"
    if verbose:
        print(f"Saving CSV: {csv_path}")
    write_students_csv(all_students, str(csv_path), keys, format_info['total_questions'])

    # Generate absent students and save roster
    roster_path = out_dir_path / "mock_roster.csv"
    used_ids = {s["student_id"] for s in all_students}
    absent_students = []
    if num_absent > 0:
        if verbose:
            print(f"\nGenerating {num_absent} absent students...")
        absent_students = generate_absent_students(
            num_absent=num_absent,
            id_digits=format_info['id_digits'],
            used_ids=used_ids
        )
    if verbose:
        print(f"Saving roster: {roster_path}")
    write_roster_csv(all_students, absent_students, str(roster_path))

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        scores = [s["expected_score"] * 100 for s in all_students]
        print(f"  Students generated: {len(all_students)}")
        if num_absent > 0:
            print(f"  Absent students: {len(absent_students)}")
            print(f"  Total roster size: {len(all_students) + len(absent_students)}")
        print(f"  Score range: {min(scores):.1f}% - {max(scores):.1f}%")
        print(f"  Mean score: {np.mean(scores):.1f}%")
        print(f"  Median score: {np.median(scores):.1f}%")

        blanks = sum(1 for s in all_students for a in s["answers"] if a == "")
        multis = sum(1 for s in all_students for a in s["answers"] if "," in a)
        total_answers = len(all_students) * format_info['total_questions']
        print(f"  Blank answers: {blanks} ({100*blanks/total_answers:.1f}%)")
        print(f"  Multi-fill answers: {multis} ({100*multis/total_answers:.1f}%)")

        id_errors = sum(1 for s in all_students if s.get("id_error"))
        missing_vers = sum(1 for s in all_students if s.get("missing_version"))
        if id_errors:
            print(f"  ID mis-entries: {id_errors} ({100*id_errors/len(all_students):.1f}%)")
        if missing_vers:
            print(f"  Missing version: {missing_vers} ({100*missing_vers/len(all_students):.1f}%)")

        # Report compound key and points stats from first version's key
        first_key = keys[versions_to_use[0]]
        and_keys = sum(1 for k in first_key if "&" in k)
        or_keys = sum(1 for k in first_key if "^" in k)
        double_pts = sum(1 for k in first_key if ":" in k)
        if and_keys:
            print(f"  AND keys: {and_keys} ({100*and_keys/len(first_key):.1f}%)")
        if or_keys:
            print(f"  OR keys: {or_keys} ({100*or_keys/len(first_key):.1f}%)")
        if default_points != 1:
            print(f"  Default points: {default_points}")
        if double_pts:
            print(f"  Weighted questions: {double_pts}")

        print(f"\nOutput files:")
        print(f"  {key_path}")
        print(f"  {pdf_path}")
        print(f"  {csv_path}")
        print(f"  {roster_path}")
        print("\nDone!")

    return {
        'answer_key': key_path,
        'scans': pdf_path,
        'responses': csv_path,
        'roster': roster_path,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line entry point."""
    import argparse

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
        "--dpi", type=int, default=150,
        help="DPI for rendered images (default: 150)"
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
        "--blank-rate", type=float, default=0.01,
        help="Rate of blank answers among wrong answers (default: 0.01)"
    )
    parser.add_argument(
        "--multi-rate", type=float, default=0.01,
        help="Rate of multi-fill answers among wrong answers (default: 0.01)"
    )
    parser.add_argument(
        "--num-absent", type=int, default=2,
        help="Number of absent students to add to roster (default: 2)"
    )
    parser.add_argument(
        "--num-id-errors", type=int, default=2,
        help="Number of students with corrupted IDs (default: 2)"
    )
    parser.add_argument(
        "--num-missing-version", type=int, default=2,
        help="Number of students with blank version field (default: 2)"
    )
    parser.add_argument(
        "--num-and-keys", type=int, default=0,
        help="Number of AND key questions in answer key, e.g. B&C (default: 0)"
    )
    parser.add_argument(
        "--num-or-keys", type=int, default=0,
        help="Number of OR key questions in answer key, e.g. B^C (default: 0)"
    )
    parser.add_argument(
        "--default-points", type=int, default=1,
        help="Default points per question (default: 1)"
    )
    parser.add_argument(
        "--num-double-points", type=int, default=0,
        help="Number of questions worth double points (or 1pt if default>1) (default: 0)"
    )

    args = parser.parse_args()

    generate_mock_dataset(
        template_path=args.template,
        bubblemap_path=args.bubblemap,
        out_dir=args.out_dir,
        num_students=args.num_students,
        num_absent=args.num_absent,
        seed=args.seed,
        dpi=args.dpi,
        darkness_min=args.darkness_min,
        darkness_max=args.darkness_max,
        apply_transform=args.apply_transform,
        blank_rate=args.blank_rate,
        multi_rate=args.multi_rate,
        num_id_errors=args.num_id_errors,
        num_missing_version=args.num_missing_version,
        num_and_keys=args.num_and_keys,
        num_or_keys=args.num_or_keys,
        default_points=args.default_points,
        num_double_points=args.num_double_points,
    )


if __name__ == "__main__":
    main()
