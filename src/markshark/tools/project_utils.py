#!/usr/bin/env python3
"""
Project-based file management utilities for MarkShark.

Flat project structure — one working copy of each file type,
with optional timestamped archiving on re-run.

Project layout:
    project_name/
    ├── input_files/           # User inputs + aligned scans
    │   ├── scans.pdf
    │   ├── aligned_scans.pdf
    │   ├── key.txt
    │   └── roster.csv
    ├── score_data/            # Scoring artifacts
    │   ├── results.csv
    │   ├── corrections.csv
    │   └── result_params.json
    ├── exam_report.xlsx       # Top-level outputs
    ├── scored_scans.pdf
    ├── logs/
    └── archive/               # Timestamped snapshots (user opts in)
        └── 2025-01-21_143022/
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List


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


def create_project_structure(base_dir: Path, project_name: str) -> Path:
    """
    Create the complete project directory structure.

    Creates:
        {base_dir}/{project_name}/
        ├── input_files/    (scans, key, roster, aligned scans)
        ├── score_data/     (results, corrections, params)
        └── logs/

    Args:
        base_dir: Base working directory
        project_name: Sanitized project name

    Returns:
        Path to the project directory
    """
    project_dir = base_dir / project_name

    # Create subdirectories
    (project_dir / "input_files").mkdir(parents=True, exist_ok=True)
    (project_dir / "score_data").mkdir(parents=True, exist_ok=True)
    (project_dir / "logs").mkdir(parents=True, exist_ok=True)

    return project_dir


def get_project_paths(project_dir: Path) -> Dict[str, Path]:
    """
    Return canonical file paths for a flat-structure project.

    For key and roster, globs input_files/ for key.* and roster.*
    and returns the first match (or the .txt / .csv default).

    Args:
        project_dir: Path to the project directory

    Returns:
        Dictionary of canonical path names to Path objects.
    """
    input_files = project_dir / "input_files"
    score_data = project_dir / "score_data"

    # Glob for key and roster (any extension)
    key_matches = sorted(input_files.glob("key.*")) if input_files.exists() else []
    roster_matches = sorted(input_files.glob("roster.*")) if input_files.exists() else []

    return {
        # Directories
        "input_files": input_files,
        "score_data": score_data,
        "logs": project_dir / "logs",
        "archive": project_dir / "archive",
        # Input files
        "scans": input_files / "scans.pdf",
        "aligned": input_files / "aligned_scans.pdf",
        "key": key_matches[0] if key_matches else input_files / "key.txt",
        "roster": roster_matches[0] if roster_matches else input_files / "roster.csv",
        # Score data
        "results_csv": score_data / "results.csv",
        "corrections_csv": score_data / "corrections.csv",
        "result_params": score_data / "result_params.json",
        # Top-level outputs
        "report": project_dir / "exam_report.xlsx",
        "scored_pdf": project_dir / "scored_scans.pdf",
    }


def has_existing_results(project_dir: Path) -> bool:
    """Check whether this project already has scoring results."""
    return (project_dir / "score_data" / "results.csv").exists()


def archive_current_results(project_dir: Path) -> Path:
    """
    Move current project outputs to a timestamped archive folder.

    Moves:
      - input_files/aligned_scans.pdf  (if exists)
      - score_data/*                   (if exists)
      - scored_scans.pdf                     (if exists)
      - exam_report.xlsx               (if exists)

    Returns:
        Path to the archive folder created.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_dir = project_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive score_data/
    src_score = project_dir / "score_data"
    if src_score.exists() and any(src_score.iterdir()):
        dst_score = archive_dir / "score_data"
        dst_score.mkdir(exist_ok=True)
        for f in src_score.iterdir():
            if f.is_file():
                shutil.move(str(f), str(dst_score / f.name))

    # Archive aligned scans
    aligned = project_dir / "input_files" / "aligned_scans.pdf"
    if aligned.exists():
        dst_input = archive_dir / "input_files"
        dst_input.mkdir(exist_ok=True)
        shutil.move(str(aligned), str(dst_input / "aligned_scans.pdf"))

    # Archive top-level outputs
    for name in ("scored_scans.pdf", "exam_report.xlsx"):
        src = project_dir / name
        if src.exists():
            shutil.move(str(src), str(archive_dir / name))

    return archive_dir


def get_project_info(project_dir: Path) -> dict:
    """
    Get information about a project directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        Dictionary with project metadata:
        - name: project name (directory name)
        - has_results: whether score_data/results.csv exists
        - last_scored: modification time of results.csv (or None)
        - num_archives: count of archive/ subdirectories
        - created: creation time (or None if unavailable)
    """
    info = {
        "name": project_dir.name,
        "has_results": False,
        "last_scored": None,
        "num_archives": 0,
        "created": None,
        "template_name": None,
    }

    results_csv = project_dir / "score_data" / "results.csv"
    if results_csv.exists():
        info["has_results"] = True
        try:
            info["last_scored"] = datetime.fromtimestamp(results_csv.stat().st_mtime)
        except Exception:
            pass

    # Read template info from the scoring params JSON (if present)
    params_json = project_dir / "score_data" / "results_params.json"
    if params_json.exists():
        try:
            import json as _json
            with open(params_json, "r", encoding="utf-8") as f:
                params = _json.load(f)
            tpl = params.get("template", {})
            if tpl.get("name"):
                info["template_name"] = tpl["name"]
        except Exception:
            pass

    archive_dir = project_dir / "archive"
    if archive_dir.exists():
        info["num_archives"] = len([
            d for d in archive_dir.iterdir() if d.is_dir()
        ])

    try:
        info["created"] = datetime.fromtimestamp(project_dir.stat().st_ctime)
    except Exception:
        pass

    return info


def find_projects(base_dir: Path) -> list[dict]:
    """
    Find all project directories in the base directory.

    A project directory is identified by having input_files/ and score_data/ subdirs.

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

        has_input_files = (item / "input_files").exists()
        has_score_data = (item / "score_data").exists()

        if has_input_files and has_score_data:
            projects.append(get_project_info(item))

    return sorted(projects, key=lambda x: x.get("created") or datetime.min, reverse=True)


def _find_results_csv(directory: Path) -> Optional[Path]:
    """
    Find a results CSV file in a directory.

    Looks for common results CSV names:
    - results.csv (from Score tab)
    - quick_grade_results.csv (from Quick Grade tab)
    - Any other *results*.csv file

    Args:
        directory: Path to search

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


def get_report_path(project_dir: Path, timestamp: Optional[datetime] = None) -> Path:
    """
    Get the path for the project report file.

    In the flat structure, the report always lives at the project root
    as exam_report.xlsx.

    Args:
        project_dir: Path to the project directory
        timestamp: Ignored (kept for API compatibility)

    Returns:
        Path for the report file
    """
    return project_dir / "exam_report.xlsx"
