#!/usr/bin/env python3
"""
Project-based file management utilities for MarkShark.
Provides structured directory organization for exam grading projects.
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List


def sanitize_project_name(name: str) -> str:
    """
    Sanitize project name for filesystem safety.

    Converts spaces to underscores, removes special characters,
    and ensures the name is filesystem-safe.

    Args:
        name: Raw project name from user input

    Returns:
        Sanitized project name suitable for directory names

    Examples:
        >>> sanitize_project_name("FINAL EXAM BIO101 2025")
        'FINAL_EXAM_BIO101_2025'
        >>> sanitize_project_name("Test: Spring/Fall")
        'Test_Spring_Fall'
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove or replace problematic characters
    # Keep: letters, numbers, underscores, hyphens, periods
    name = re.sub(r'[^\w\-.]', '_', name)

    # Remove leading/trailing underscores or periods
    name = name.strip("_.")

    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)

    return name


def get_next_scored_number(project_dir: Path) -> int:
    """
    Find the next available scored run number in a project's scored directory.

    Scans the scored/ subdirectory for existing scored_XXX_* folders
    and returns the next sequential number.

    Args:
        project_dir: Path to the project directory

    Returns:
        Next scored number (1-based)

    Examples:
        If scored/ contains: scored_001_..., scored_002_...
        Returns: 3
    """
    scored_dir = project_dir / "scored"

    if not scored_dir.exists():
        return 1

    # Find all scored_XXX_* directories
    scored_pattern = re.compile(r'^scored_(\d+)_')
    max_num = 0

    for item in scored_dir.iterdir():
        if item.is_dir():
            match = scored_pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

    return max_num + 1


# Keep old function name for backward compatibility
def get_next_run_number(project_dir: Path) -> int:
    """Deprecated: Use get_next_scored_number instead."""
    return get_next_scored_number(project_dir)


def create_scored_directory(project_dir: Path, timestamp: Optional[datetime] = None) -> Tuple[Path, str]:
    """
    Create a new versioned scored directory within a project.

    Creates a directory like: scored/scored_001_2025-01-21_1430/

    Args:
        project_dir: Path to the project directory
        timestamp: Optional datetime to use (defaults to now)

    Returns:
        Tuple of (scored_directory_path, scored_label)
        where scored_label is like "scored_001_2025-01-21_1430"
    """
    if timestamp is None:
        timestamp = datetime.now()

    scored_num = get_next_scored_number(project_dir)
    date_str = timestamp.strftime("%Y-%m-%d_%H%M")
    scored_label = f"scored_{scored_num:03d}_{date_str}"

    scored_dir = project_dir / "scored" / scored_label
    scored_dir.mkdir(parents=True, exist_ok=True)

    return scored_dir, scored_label


# Keep old function name for backward compatibility
def create_run_directory(project_dir: Path, timestamp: Optional[datetime] = None) -> Tuple[Path, str]:
    """Deprecated: Use create_scored_directory instead."""
    return create_scored_directory(project_dir, timestamp)


def create_project_structure(base_dir: Path, project_name: str) -> Path:
    """
    Create the complete project directory structure.

    Creates:
        {base_dir}/{project_name}/
        ├── input/       (raw scans and aligned scans)
        ├── scored/
        ├── reports/
        └── logs/

    The input directory contains:
        - raw_scans.pdf: Original uploaded scans
        - aligned_scan_YYYY-MM-DD_HHMM.pdf: Aligned scans (one per alignment run)

    Args:
        base_dir: Base working directory
        project_name: Sanitized project name

    Returns:
        Path to the project directory
    """
    project_dir = base_dir / project_name

    # Create subdirectories
    (project_dir / "input").mkdir(parents=True, exist_ok=True)
    (project_dir / "scored").mkdir(parents=True, exist_ok=True)
    (project_dir / "reports").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    return project_dir


def get_project_info(project_dir: Path) -> dict:
    """
    Get information about a project directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        Dictionary with project metadata:
        - name: project name (directory name)
        - num_runs: number of completed scoring runs
        - last_run: path to most recent scored run (or None)
        - created: creation time (or None if unavailable)
    """
    info = {
        "name": project_dir.name,
        "num_runs": 0,
        "last_run": None,
        "created": None,
    }

    scored_dir = project_dir / "scored"

    if scored_dir.exists():
        scored_dirs = sorted(
            [d for d in scored_dir.iterdir() if d.is_dir() and d.name.startswith("scored_")],
            key=lambda x: x.name
        )
        info["num_runs"] = len(scored_dirs)
        if scored_dirs:
            info["last_run"] = scored_dirs[-1]

    # Also check legacy 'results' folder for backward compatibility
    results_dir = project_dir / "results"
    if results_dir.exists():
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        info["num_runs"] += len(run_dirs)
        if run_dirs and info["last_run"] is None:
            info["last_run"] = sorted(run_dirs, key=lambda x: x.name)[-1]

    try:
        info["created"] = datetime.fromtimestamp(project_dir.stat().st_ctime)
    except Exception:
        pass

    return info


def find_projects(base_dir: Path) -> list[dict]:
    """
    Find all project directories in the base directory.

    A project directory is identified by having the expected subdirectory
    structure (input/, scored/ or results/, logs/).

    Args:
        base_dir: Base working directory to scan

    Returns:
        List of project info dictionaries (see get_project_info)
    """
    if not base_dir.exists():
        return []

    projects = []

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        # Check if it looks like a project (has expected subdirs)
        # Support both new (scored) and legacy (results/aligned) structures
        has_input = (item / "input").exists()
        has_scored = (item / "scored").exists() or (item / "results").exists()
        has_logs = (item / "logs").exists()
        # Legacy support: check for aligned/ but don't require it
        has_aligned_legacy = (item / "aligned").exists()

        # Accept projects with either the new structure (input + scored + logs)
        # or legacy structure that also had aligned/
        if has_input and has_scored and has_logs:
            projects.append(get_project_info(item))
        elif has_input and has_aligned_legacy and has_scored and has_logs:
            projects.append(get_project_info(item))

    return sorted(projects, key=lambda x: x.get("created") or datetime.min, reverse=True)


def _find_results_csv(directory: Path) -> Optional[Path]:
    """
    Find a results CSV file in a scored run directory.

    Looks for common results CSV names:
    - results.csv (from Score tab)
    - quick_grade_results.csv (from Quick Grade tab)
    - Any other *results*.csv file

    Args:
        directory: Path to the scored run directory

    Returns:
        Path to the results CSV, or None if not found
    """
    # Check common names first
    for name in ["results.csv", "quick_grade_results.csv"]:
        csv_path = directory / name
        if csv_path.exists():
            return csv_path

    # Fall back to any file matching *results*.csv
    for csv_file in directory.glob("*results*.csv"):
        return csv_file

    # Last resort: any CSV file
    csv_files = list(directory.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    return None


def find_scored_runs(project_dir: Path) -> List[dict]:
    """
    Find all scored run directories in a project, sorted by most recent first.

    Searches both the new 'scored/' directory and legacy 'results/' directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        List of dictionaries with run info:
        - path: Path to the scored directory
        - label: Directory name (e.g., "scored_001_2025-01-21_1430")
        - number: Run number (e.g., 1)
        - timestamp: Extracted datetime or None
        - has_csv: Whether a results CSV exists in this directory
        - csv_path: Path to the results CSV (or None)
    """
    runs = []

    # Check new 'scored/' directory
    scored_dir = project_dir / "scored"
    if scored_dir.exists():
        scored_pattern = re.compile(r'^scored_(\d+)_(\d{4}-\d{2}-\d{2}_\d{4})$')
        for item in scored_dir.iterdir():
            if item.is_dir():
                match = scored_pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    date_str = match.group(2)
                    try:
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H%M")
                    except ValueError:
                        timestamp = None

                    csv_path = _find_results_csv(item)
                    runs.append({
                        "path": item,
                        "label": item.name,
                        "number": num,
                        "timestamp": timestamp,
                        "has_csv": csv_path is not None,
                        "csv_path": csv_path,
                    })

    # Check legacy 'results/' directory for backward compatibility
    results_dir = project_dir / "results"
    if results_dir.exists():
        run_pattern = re.compile(r'^run_(\d+)_(\d{4}-\d{2}-\d{2}_\d{4})$')
        for item in results_dir.iterdir():
            if item.is_dir():
                match = run_pattern.match(item.name)
                if match:
                    num = int(match.group(1))
                    date_str = match.group(2)
                    try:
                        timestamp = datetime.strptime(date_str, "%Y-%m-%d_%H%M")
                    except ValueError:
                        timestamp = None

                    csv_path = _find_results_csv(item)
                    runs.append({
                        "path": item,
                        "label": item.name,
                        "number": num,
                        "timestamp": timestamp,
                        "has_csv": csv_path is not None,
                        "csv_path": csv_path,
                    })

    # Sort by timestamp (most recent first), then by number
    runs.sort(key=lambda x: (x["timestamp"] or datetime.min, x["number"]), reverse=True)

    return runs


def get_report_path(project_dir: Path, timestamp: Optional[datetime] = None) -> Path:
    """
    Get the path for a new report file.

    Creates path like: reports/report_2025-01-21.xlsx

    Args:
        project_dir: Path to the project directory
        timestamp: Optional datetime to use (defaults to now)

    Returns:
        Path for the report file
    """
    if timestamp is None:
        timestamp = datetime.now()

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    date_str = timestamp.strftime("%Y-%m-%d")
    report_name = f"report_{date_str}.xlsx"

    # If file already exists, add a counter
    report_path = reports_dir / report_name
    if report_path.exists():
        counter = 2
        while True:
            report_name = f"report_{date_str}_{counter}.xlsx"
            report_path = reports_dir / report_name
            if not report_path.exists():
                break
            counter += 1

    return report_path
