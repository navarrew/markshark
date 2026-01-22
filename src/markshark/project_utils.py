#!/usr/bin/env python3
"""
Project-based file management utilities for MarkShark.
Provides structured directory organization for exam grading projects.
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


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


def get_next_run_number(project_dir: Path) -> int:
    """
    Find the next available run number in a project's results directory.

    Scans the results/ subdirectory for existing run_XXX_* folders
    and returns the next sequential number.

    Args:
        project_dir: Path to the project directory

    Returns:
        Next run number (1-based)

    Examples:
        If results/ contains: run_001_..., run_002_...
        Returns: 3
    """
    results_dir = project_dir / "results"

    if not results_dir.exists():
        return 1

    # Find all run_XXX_* directories
    run_pattern = re.compile(r'^run_(\d+)_')
    max_num = 0

    for item in results_dir.iterdir():
        if item.is_dir():
            match = run_pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

    return max_num + 1


def create_run_directory(project_dir: Path, timestamp: Optional[datetime] = None) -> Tuple[Path, str]:
    """
    Create a new versioned run directory within a project.

    Creates a directory like: results/run_001_2025-01-21_1430/

    Args:
        project_dir: Path to the project directory
        timestamp: Optional datetime to use (defaults to now)

    Returns:
        Tuple of (run_directory_path, run_label)
        where run_label is like "run_001_2025-01-21_1430"
    """
    if timestamp is None:
        timestamp = datetime.now()

    run_num = get_next_run_number(project_dir)
    date_str = timestamp.strftime("%Y-%m-%d_%H%M")
    run_label = f"run_{run_num:03d}_{date_str}"

    run_dir = project_dir / "results" / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, run_label


def create_project_structure(base_dir: Path, project_name: str) -> Path:
    """
    Create the complete project directory structure.

    Creates:
        {base_dir}/{project_name}/
        ├── input/
        ├── aligned/
        ├── results/
        └── config/

    Args:
        base_dir: Base working directory
        project_name: Sanitized project name

    Returns:
        Path to the project directory
    """
    project_dir = base_dir / project_name

    # Create subdirectories
    (project_dir / "input").mkdir(parents=True, exist_ok=True)
    (project_dir / "aligned").mkdir(parents=True, exist_ok=True)
    (project_dir / "results").mkdir(parents=True, exist_ok=True)
    (project_dir / "config").mkdir(parents=True, exist_ok=True)

    return project_dir


def get_project_info(project_dir: Path) -> dict:
    """
    Get information about a project directory.

    Args:
        project_dir: Path to the project directory

    Returns:
        Dictionary with project metadata:
        - name: project name (directory name)
        - num_runs: number of completed runs
        - last_run: path to most recent run (or None)
        - created: creation time (or None if unavailable)
    """
    info = {
        "name": project_dir.name,
        "num_runs": 0,
        "last_run": None,
        "created": None,
    }

    results_dir = project_dir / "results"

    if results_dir.exists():
        run_dirs = sorted(
            [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda x: x.name
        )
        info["num_runs"] = len(run_dirs)
        if run_dirs:
            info["last_run"] = run_dirs[-1]

    try:
        info["created"] = datetime.fromtimestamp(project_dir.stat().st_ctime)
    except Exception:
        pass

    return info


def find_projects(base_dir: Path) -> list[dict]:
    """
    Find all project directories in the base directory.

    A project directory is identified by having the expected subdirectory
    structure (input/, aligned/, results/, config/).

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
        required_subdirs = ["input", "aligned", "results", "config"]
        has_all = all((item / subdir).exists() for subdir in required_subdirs)

        if has_all:
            projects.append(get_project_info(item))

    return sorted(projects, key=lambda x: x.get("created") or datetime.min, reverse=True)
