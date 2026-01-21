#!/usr/bin/env python3
"""
MarkShark
report_tools.py
Generate teacher-friendly Excel reports from scored CSV results

Features:
- Multi-version support: separate tab per version
- Roster matching: flags absent students and orphan scans
- Color-coded item quality indicators
- Summary statistics and item analysis
"""

from __future__ import annotations
import re
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils.dataframe import dataframe_to_rows
except ImportError:
    raise ImportError(
        "openpyxl is required for Excel report generation. "
        "Install it with: pip install openpyxl"
    )

try:
    from rapidfuzz import fuzz
except ImportError:
    raise ImportError(
        "rapidfuzz is required for student roster matching. "
        "Install it with: pip install rapidfuzz"
    )

from .stats_tools import (
    detect_item_columns,
    detect_key_row_index,
    prepare_correctness_matrix,
    point_biserial,
    kr20,
    kr21,
)


# ==================== ROSTER MATCHING ====================

def load_roster(roster_path: str) -> pd.DataFrame:
    """
    Load and normalize a class roster CSV.

    Expected columns (case-insensitive, auto-detected):
    - StudentID / ID / Student_ID
    - LastName / Last / Surname
    - FirstName / First (optional)

    Returns DataFrame with standardized columns: StudentID, LastName, FirstName
    """
    df = pd.read_csv(roster_path)

    # Normalize column names
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('studentid', 'id', 'student_id', 'sid'):
            col_map[col] = 'StudentID'
        elif col_lower in ('lastname', 'last', 'surname', 'last_name'):
            col_map[col] = 'LastName'
        elif col_lower in ('firstname', 'first', 'first_name'):
            col_map[col] = 'FirstName'

    if 'StudentID' not in col_map.values():
        raise ValueError(
            f"Roster CSV must have a student ID column. "
            f"Expected: StudentID/ID/Student_ID. Found: {list(df.columns)}"
        )

    if 'LastName' not in col_map.values():
        raise ValueError(
            f"Roster CSV must have a last name column. "
            f"Expected: LastName/Last/Surname. Found: {list(df.columns)}"
        )

    df = df.rename(columns=col_map)

    # Fill missing FirstName with empty string
    if 'FirstName' not in df.columns:
        df['FirstName'] = ''

    # Convert StudentID to string and strip whitespace
    df['StudentID'] = df['StudentID'].astype(str).str.strip()
    df['LastName'] = df['LastName'].astype(str).str.strip()
    df['FirstName'] = df['FirstName'].astype(str).str.strip()

    return df[['StudentID', 'LastName', 'FirstName']]


def fuzzy_match_student(
    scanned_id: str,
    scanned_last: str,
    scanned_first: str,
    roster: pd.DataFrame,
    id_threshold: float = 85.0,
    name_threshold: float = 85.0,
) -> Tuple[Optional[str], float, str]:
    """
    Attempt to match a scanned student to the roster using fuzzy matching.

    Returns:
        (matched_roster_id, confidence, match_type)

        match_type can be:
        - "exact": Exact StudentID match
        - "high_confidence": High ID similarity or ID + name match
        - "probable": Moderate confidence match
        - "no_match": No good match found
    """
    if roster.empty:
        return None, 0.0, "no_match"

    # Clean inputs
    scanned_id = str(scanned_id).strip()
    scanned_last = str(scanned_last).strip().upper()
    scanned_first = str(scanned_first).strip().upper()

    best_match = None
    best_score = 0.0
    match_type = "no_match"

    for _, row in roster.iterrows():
        roster_id = str(row['StudentID']).strip()
        roster_last = str(row['LastName']).strip().upper()
        roster_first = str(row['FirstName']).strip().upper()

        # Exact ID match
        if scanned_id == roster_id:
            return roster_id, 100.0, "exact"

        # Fuzzy ID match
        id_score = fuzz.ratio(scanned_id, roster_id)

        # Name matching
        last_score = fuzz.ratio(scanned_last, roster_last) if scanned_last else 0
        first_score = fuzz.ratio(scanned_first, roster_first) if scanned_first and roster_first else 0

        # Combined scoring strategies
        # Strategy 1: Very high ID match (typo in one digit)
        if id_score >= 95:
            confidence = id_score
            if confidence > best_score:
                best_match = roster_id
                best_score = confidence
                match_type = "high_confidence"

        # Strategy 2: Good ID match + exact last name
        elif id_score >= id_threshold and last_score == 100:
            confidence = (id_score + last_score) / 2
            if confidence > best_score:
                best_match = roster_id
                best_score = confidence
                match_type = "high_confidence"

        # Strategy 3: Perfect name match (both first and last)
        elif last_score == 100 and first_score == 100 and scanned_first:
            confidence = 100.0
            if confidence > best_score:
                best_match = roster_id
                best_score = confidence
                match_type = "probable"

        # Strategy 4: Good overall match
        elif id_score >= id_threshold or (last_score >= name_threshold and id_score >= 70):
            confidence = max(id_score, (id_score + last_score) / 2)
            if confidence > best_score:
                best_match = roster_id
                best_score = confidence
                match_type = "probable"

    return best_match, best_score, match_type


def match_students_to_roster(
    students_df: pd.DataFrame,
    roster_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Match scanned students to roster.

    Returns:
        (students_df_with_matches, orphan_scans, absent_students)

        students_df_with_matches: Original DataFrame with added columns:
            - RosterID: Matched roster ID (or None)
            - MatchConfidence: 0-100
            - MatchType: exact/high_confidence/probable/no_match

        orphan_scans: List of dicts for students who couldn't be matched confidently
        absent_students: List of dicts for roster students not found in scans
    """
    # Add matching columns
    students_df['RosterID'] = None
    students_df['MatchConfidence'] = 0.0
    students_df['MatchType'] = 'no_match'

    matched_roster_ids = set()

    for idx, row in students_df.iterrows():
        scanned_id = str(row.get('StudentID', '')).strip()
        scanned_last = str(row.get('LastName', '')).strip()
        scanned_first = str(row.get('FirstName', '')).strip()

        if not scanned_id and not scanned_last:
            # Can't match without any identifying info
            continue

        matched_id, confidence, match_type = fuzzy_match_student(
            scanned_id, scanned_last, scanned_first, roster_df
        )

        if matched_id:
            students_df.at[idx, 'RosterID'] = matched_id
            students_df.at[idx, 'MatchConfidence'] = confidence
            students_df.at[idx, 'MatchType'] = match_type
            matched_roster_ids.add(matched_id)

    # Find orphan scans (low confidence or no match)
    orphan_scans = []
    for idx, row in students_df.iterrows():
        if row['MatchType'] in ('no_match', 'probable'):
            orphan_scans.append({
                'ScannedID': row.get('StudentID', ''),
                'LastName': row.get('LastName', ''),
                'FirstName': row.get('FirstName', ''),
                'MatchType': row['MatchType'],
                'PossibleMatch': row['RosterID'] if row['MatchType'] == 'probable' else None,
                'Confidence': row['MatchConfidence'],
            })

    # Find absent students
    absent_students = []
    for _, row in roster_df.iterrows():
        if row['StudentID'] not in matched_roster_ids:
            absent_students.append({
                'StudentID': row['StudentID'],
                'LastName': row['LastName'],
                'FirstName': row['FirstName'],
            })

    return students_df, orphan_scans, absent_students


# ==================== EXCEL FORMATTING ====================

# Color scheme
COLOR_HEADER = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
COLOR_KEY_ROW = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
COLOR_GOOD = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
COLOR_WARNING = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
COLOR_PROBLEM = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
COLOR_ORPHAN = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")

FONT_HEADER = Font(bold=True, color="FFFFFF")
FONT_BOLD = Font(bold=True)
BORDER_THIN = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)


def format_header_row(ws, row_num: int):
    """Apply header formatting to a row."""
    for cell in ws[row_num]:
        cell.fill = COLOR_HEADER
        cell.font = FONT_HEADER
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = BORDER_THIN


def format_key_row(ws, row_num: int):
    """Apply KEY row formatting."""
    for cell in ws[row_num]:
        cell.fill = COLOR_KEY_ROW
        cell.font = FONT_BOLD
        cell.border = BORDER_THIN


def auto_size_columns(ws):
    """Auto-size all columns based on content."""
    for column in ws.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        adjusted_width = min(max_length + 2, 50)
        ws.column_dimensions[column_letter].width = adjusted_width


# ==================== CSV LOADING ====================

def _load_score_csv_robust(input_csv: str) -> pd.DataFrame:
    """
    Load CSV from score command, handling messy format with include_stats.

    The score command with include_stats can create CSVs with:
    - Section headers like "=== VERSION A (112 students) ==="
    - Multiple header rows (one per version)
    - Stats rows at the bottom

    This function:
    1. Finds the first valid header row
    2. Reads only the student data rows (skipping section headers and stats)
    3. Returns a clean DataFrame
    """
    import csv

    # Read the raw CSV to find structure
    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Find the first valid header row (has question columns like Q1, Q2, etc.)
    header_idx = None
    header = None

    for idx, line in enumerate(lines):
        row = next(csv.reader([line]))
        # Valid header should have StudentID and Q1
        if 'StudentID' in row or 'StudentID' in str(row):
            # Check if this looks like a real header with questions
            if any(col.startswith('Q') and col[1:].isdigit() for col in row if isinstance(col, str)):
                header_idx = idx
                header = row
                break

    if header_idx is None:
        # Fallback: just try reading normally
        return pd.read_csv(input_csv)

    # Collect all data rows (skip section headers and stats rows)
    data_rows = []
    reader = csv.reader(lines[header_idx + 1:])

    for row in reader:
        # Skip empty rows
        if not row or all(cell.strip() == '' for cell in row):
            continue

        # Skip section headers (=== VERSION A === etc.)
        if row and row[0].strip().startswith('==='):
            continue

        # Skip stats rows (start with specific labels)
        if row and row[0].strip() in ['PCT_CORRECT', 'POINT_BISERIAL', 'N_STUDENTS',
                                        'MEAN_SCORE', 'MEAN_PERCENT', 'STD_DEV',
                                        'HIGH_SCORE', 'LOW_SCORE', 'KR20_OVERALL'] or \
           (row and 'PCT_CORRECT' in row[0]) or \
           (row and 'POINT_BISERIAL' in row[0]) or \
           (row and 'KR20_VERSION' in row[0]) or \
           (row and '--- ITEM STATISTICS' in row[0]):
            continue

        # This looks like a valid data row
        # Ensure row has same length as header
        if len(row) < len(header):
            row.extend([''] * (len(header) - len(row)))
        elif len(row) > len(header):
            row = row[:len(header)]

        data_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=header)

    # Normalize column names to lowercase for consistency
    # This handles both 'Correct' and 'correct', 'Version' and 'version', etc.
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        # Map common variations to standard names
        if col_lower in ('correct', 'incorrect', 'blank', 'multi', 'percent',
                         'version', 'studentid', 'lastname', 'firstname',
                         'page', 'page_index'):
            column_mapping[col] = col_lower
        # Keep question columns as-is (Q1, Q2, etc.)
        elif col.startswith('Q') and col[1:].isdigit():
            column_mapping[col] = col
        else:
            column_mapping[col] = col

    df = df.rename(columns=column_mapping)

    return df


# ==================== REPORT GENERATION ====================

def generate_report(
    input_csv: str,
    out_xlsx: str,
    roster_csv: Optional[str] = None,
    item_pattern: str = r"Q\d+",
):
    r"""
    Generate comprehensive Excel report from scored CSV.

    Args:
        input_csv: Path to scored CSV from markshark score
        out_xlsx: Path to output Excel file
        roster_csv: Optional path to class roster CSV
        item_pattern: Regex pattern for item columns (default: Q\d+)
    """
    # Load scored results - handle messy CSV from score with include_stats
    df = _load_score_csv_robust(input_csv)

    # Detect item columns and key row
    item_cols = detect_item_columns(df, item_pattern)
    if not item_cols:
        raise ValueError(
            f"No item columns found matching pattern '{item_pattern}'. "
            f"Available columns: {list(df.columns)}"
        )

    key_row_idx = detect_key_row_index(df, item_cols, key_label="KEY")

    # Load roster if provided
    roster_df = None
    orphan_scans = []
    absent_students = []

    if roster_csv:
        roster_df = load_roster(roster_csv)
        # Remove KEY row before matching
        students_only = df.drop(index=df.index[key_row_idx]).reset_index(drop=True)
        students_only, orphan_scans, absent_students = match_students_to_roster(
            students_only, roster_df
        )

    # Prepare correctness matrix and stats
    items_num, total_scores, students_df, key_series = prepare_correctness_matrix(
        df, item_cols, key_row_idx, answers_mode="auto"
    )

    # Compute overall stats
    k = len(item_cols)
    mean_total = float(total_scores.mean())
    std_total = float(total_scores.std(ddof=1))
    kr20_val = kr20(items_num, total_scores)
    kr21_val = kr21(items_num, total_scores)

    # Group by version (handle both 'Version' and 'version' column names)
    version_col = None
    if 'version' in df.columns:
        version_col = 'version'
    elif 'Version' in df.columns:
        version_col = 'Version'

    if version_col:
        # Get unique versions, filtering out invalid values
        all_versions = df[version_col].dropna().astype(str).str.strip().unique()
        # Filter out "VERSION" and other non-single-letter values
        versions = sorted([v for v in all_versions
                          if v and len(v) <= 2 and v.upper() != 'VERSION' and v != 'KEY'])
    else:
        versions = ['A']  # Default single version

    # Compute per-version difficulty and point-biserial statistics
    # This is critical for multi-version exams where each version has different answer keys
    version_stats = {}
    for version in versions:
        # Filter to only students who took this version
        if version_col:
            version_mask = df[version_col].astype(str).str.strip() == str(version).strip()
            df_version = df[version_mask].copy()
        else:
            df_version = df.copy()

        # Find KEY row for this version
        key_row_idx_version = detect_key_row_index(df_version, item_cols, key_label="KEY")

        # Prepare correctness matrix for this version only
        items_num_v, total_scores_v, students_df_v, key_series_v = prepare_correctness_matrix(
            df_version, item_cols, key_row_idx_version, answers_mode="auto"
        )

        # Compute difficulty (% correct) for this version
        difficulty_v = items_num_v.mean(axis=0)

        # Compute point-biserial for this version
        pb_vals_v = {}
        for col in item_cols:
            item_series = items_num_v[col]
            total_minus = total_scores_v - item_series.fillna(0)
            pb_vals_v[col] = point_biserial(item_series, total_minus)

        # Store version-specific stats
        version_stats[version] = {
            'difficulty': difficulty_v,
            'pb_vals': pb_vals_v,
            'key_series': key_series_v
        }

    # Create Excel workbook
    wb = Workbook()
    wb.remove(wb.active)  # Remove default sheet

    # ========== SUMMARY TAB ==========
    create_summary_tab(
        wb, k, len(students_df), mean_total, std_total, kr20_val, kr21_val,
        len(versions), orphan_scans, absent_students
    )

    # ========== PER-VERSION TABS ==========
    for version in versions:
        # Get version-specific stats
        vstats = version_stats[version]
        create_version_tab(
            wb, df, students_df, version, item_cols, vstats['key_series'],
            vstats['difficulty'], vstats['pb_vals'], roster_df, orphan_scans if roster_csv else None
        )

    # Save workbook
    wb.save(out_xlsx)
    print(f"Excel report generated: {out_xlsx}")


def create_summary_tab(
    wb, k, n_students, mean_total, std_total, kr20_val, kr21_val,
    n_versions, orphan_scans, absent_students
):
    """Create summary tab with overall exam statistics."""
    ws = wb.create_sheet("Summary", 0)

    # Title
    ws['A1'] = "MarkShark Exam Report"
    ws['A1'].font = Font(size=16, bold=True)

    # Overall stats
    row = 3
    ws[f'A{row}'] = "Overall Exam Statistics"
    ws[f'A{row}'].font = FONT_BOLD
    row += 1

    stats = [
        ("Number of Students", n_students),
        ("Number of Questions", k),
        ("Number of Versions", n_versions),
        ("Mean Score", f"{mean_total:.2f}"),
        ("Mean Percentage", f"{mean_total/k*100:.1f}%"),
        ("Standard Deviation", f"{std_total:.2f}"),
        ("KR-20 Reliability", f"{kr20_val:.3f}" if not np.isnan(kr20_val) else "N/A"),
        ("KR-21 Reliability", f"{kr21_val:.3f}" if not np.isnan(kr21_val) else "N/A"),
    ]

    for stat_name, stat_value in stats:
        ws[f'A{row}'] = stat_name
        ws[f'B{row}'] = stat_value
        row += 1

    # Reliability interpretation
    row += 1
    ws[f'A{row}'] = "Reliability Interpretation"
    ws[f'A{row}'].font = FONT_BOLD
    row += 1

    if not np.isnan(kr20_val):
        if kr20_val >= 0.80:
            ws[f'A{row}'] = "Excellent reliability (≥0.80)"
            ws[f'A{row}'].fill = COLOR_GOOD
        elif kr20_val >= 0.70:
            ws[f'A{row}'] = "Good reliability (0.70-0.80)"
            ws[f'A{row}'].fill = COLOR_GOOD
        elif kr20_val >= 0.60:
            ws[f'A{row}'] = "Acceptable reliability (0.60-0.70)"
            ws[f'A{row}'].fill = COLOR_WARNING
        else:
            ws[f'A{row}'] = "Poor reliability (<0.60) - exam needs work"
            ws[f'A{row}'].fill = COLOR_PROBLEM

    # Roster issues
    if orphan_scans or absent_students:
        row += 2
        ws[f'A{row}'] = "Roster Issues"
        ws[f'A{row}'].font = Font(size=14, bold=True, color="FF0000")
        row += 1

        if orphan_scans:
            ws[f'A{row}'] = f"⚠ {len(orphan_scans)} orphan scan(s)"
            ws[f'A{row}'].fill = COLOR_ORPHAN
            row += 1

            # List orphan scans
            ws[f'A{row}'] = "Orphan Scans (ID mismatch or fuzzy match):"
            ws[f'A{row}'].font = FONT_BOLD
            row += 1
            ws[f'A{row}'] = "Scanned ID"
            ws[f'B{row}'] = "Last Name"
            ws[f'C{row}'] = "First Name"
            ws[f'D{row}'] = "Match Type"
            ws[f'E{row}'] = "Possible Match"
            format_header_row(ws, row)
            row += 1

            for orphan in orphan_scans:
                ws[f'A{row}'] = orphan.get('ScannedID', '')
                ws[f'B{row}'] = orphan.get('LastName', '')
                ws[f'C{row}'] = orphan.get('FirstName', '')
                ws[f'D{row}'] = orphan.get('MatchType', 'no_match')
                ws[f'E{row}'] = orphan.get('PossibleMatch', '')
                row += 1
            row += 1

        if absent_students:
            ws[f'A{row}'] = f"⚠ {len(absent_students)} absent student(s)"
            ws[f'A{row}'].fill = COLOR_WARNING
            row += 1

            # List absent students
            ws[f'A{row}'] = "Absent Students:"
            ws[f'A{row}'].font = FONT_BOLD
            row += 1
            ws[f'A{row}'] = "Student ID"
            ws[f'B{row}'] = "Last Name"
            ws[f'C{row}'] = "First Name"
            format_header_row(ws, row)
            row += 1

            for student in absent_students:
                ws[f'A{row}'] = student['StudentID']
                ws[f'B{row}'] = student['LastName']
                ws[f'C{row}'] = student['FirstName']
                row += 1

    auto_size_columns(ws)


def create_version_tab(
    wb, df_full, students_df, version, item_cols, key_series,
    difficulty, pb_vals, roster_df, orphan_scans
):
    """Create a tab for a specific exam version."""
    ws = wb.create_sheet(f"Version {version}")

    # Filter for this version (handle both 'Version' and 'version')
    version_col = 'version' if 'version' in df_full.columns else 'Version' if 'Version' in df_full.columns else None

    if version_col:
        version_mask = df_full[version_col].astype(str).str.strip() == str(version).strip()
        df_version = df_full[version_mask].copy()
    else:
        df_version = df_full.copy()

    # Get KEY row for this version
    key_row_data = df_version[df_version.apply(
        lambda row: any(str(cell).strip().upper() == 'KEY' for cell in row), axis=1
    )]

    # Get student rows
    student_rows = df_version[~df_version.apply(
        lambda row: any(str(cell).strip().upper() == 'KEY' for cell in row), axis=1
    )]

    # Determine columns to display in the desired order:
    # LastName, FirstName, StudentID, Issue, correct, incorrect, blank, multi, percent, Version, Q1, Q2, ...
    display_cols = []

    # Identity columns first (check both cases)
    for col_variants in [('lastname', 'LastName'), ('firstname', 'FirstName'), ('studentid', 'StudentID')]:
        for variant in col_variants:
            if variant in df_version.columns:
                display_cols.append(variant)
                break

    # Add Issue column (will be computed)
    display_cols.append('Issue')

    # Score columns (check both cases)
    for col in ['correct', 'incorrect', 'blank', 'multi', 'percent']:
        if col in df_version.columns:
            display_cols.append(col)
        elif col.capitalize() in df_version.columns:
            display_cols.append(col.capitalize())

    # Version column (check both cases)
    if 'version' in df_version.columns:
        display_cols.append('version')
    elif 'Version' in df_version.columns:
        display_cols.append('Version')

    # Question columns
    display_cols.extend(item_cols)

    # Build column index map for quick lookup
    col_idx_map = {col: idx + 1 for idx, col in enumerate(display_cols)}

    # Write header
    for col_idx, col_name in enumerate(display_cols, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
    format_header_row(ws, 1)

    # Get KEY answers for this version
    key_answers = {}
    if not key_row_data.empty:
        key_row = key_row_data.iloc[0]
        for col in item_cols:
            key_answers[col] = str(key_row.get(col, '')).strip().upper()

    # Write student rows
    row_num = 2
    for _, student_row in student_rows.iterrows():
        # Determine issues for this student
        issues = []

        # Check for blank/multi answers
        blank_count = int(student_row.get('blank', 0))
        multi_count = int(student_row.get('multi', 0))
        if blank_count > 0:
            issues.append(f"{blank_count} blank")
        if multi_count > 0:
            issues.append(f"{multi_count} multi")

        # Check roster matching (if available)
        if roster_df is not None:
            student_id = str(student_row.get('StudentID', '')).strip()
            if orphan_scans:
                # Check if this student is an orphan
                for orphan in orphan_scans:
                    if str(orphan.get('ScannedID', '')).strip() == student_id:
                        match_type = orphan.get('MatchType', 'no_match')
                        if match_type == 'no_match':
                            issues.append("ID mismatch")
                        elif match_type == 'probable':
                            issues.append("Fuzzy match")
                        break

        issue_text = "; ".join(issues) if issues else ""

        # Write all column values
        for col_idx, col_name in enumerate(display_cols, start=1):
            if col_name == 'Issue':
                cell = ws.cell(row=row_num, column=col_idx, value=issue_text)
                if issue_text:
                    cell.fill = COLOR_WARNING
            else:
                value = student_row.get(col_name, '')
                cell = ws.cell(row=row_num, column=col_idx, value=value)

                # Highlight incorrect answers in light red
                if col_name in item_cols and key_answers.get(col_name):
                    student_answer = str(value).strip().upper()
                    correct_answer = key_answers[col_name]
                    if student_answer and student_answer != correct_answer:
                        # Light red fill for incorrect answers
                        cell.fill = PatternFill(start_color="FFD7D7", end_color="FFD7D7", fill_type="solid")

        row_num += 1

    # Add KEY answer row before statistics
    row_num += 1
    ws.cell(row=row_num, column=1, value="KEY")
    ws.cell(row=row_num, column=1).font = FONT_BOLD
    for col_name in item_cols:
        col_idx = col_idx_map.get(col_name)
        if col_idx and col_name in key_answers:
            ws.cell(row=row_num, column=col_idx, value=key_answers[col_name])
    format_key_row(ws, row_num)
    row_num += 1

    # Add item statistics rows
    ws.cell(row=row_num, column=1, value="% Correct")
    ws.cell(row=row_num, column=1).font = FONT_BOLD
    for col_name in item_cols:
        col_idx = col_idx_map.get(col_name)
        if col_idx:
            pct = difficulty[col_name] * 100 if not np.isnan(difficulty[col_name]) else 0
            ws.cell(row=row_num, column=col_idx, value=f"{pct:.1f}%")
    row_num += 1

    ws.cell(row=row_num, column=1, value="Point-Biserial")
    ws.cell(row=row_num, column=1).font = FONT_BOLD
    for col_name in item_cols:
        col_idx = col_idx_map.get(col_name)
        if col_idx:
            pb = pb_vals[col_name]
            if not np.isnan(pb):
                cell = ws.cell(row=row_num, column=col_idx, value=f"{pb:.3f}")
                # Color code based on quality
                if pb >= 0.20:
                    cell.fill = COLOR_GOOD
                elif pb >= 0.10:
                    cell.fill = COLOR_WARNING
                else:
                    cell.fill = COLOR_PROBLEM
    row_num += 1

    ws.cell(row=row_num, column=1, value="Item Quality")
    ws.cell(row=row_num, column=1).font = FONT_BOLD
    for col_name in item_cols:
        col_idx = col_idx_map.get(col_name)
        if col_idx:
            pb = pb_vals[col_name]
            if not np.isnan(pb):
                if pb >= 0.20:
                    quality = "✓ Good"
                    fill = COLOR_GOOD
                elif pb >= 0.10:
                    quality = "⚠ Review"
                    fill = COLOR_WARNING
                else:
                    quality = "✗ Problem"
                    fill = COLOR_PROBLEM
                cell = ws.cell(row=row_num, column=col_idx, value=quality)
                cell.fill = fill

    auto_size_columns(ws)
