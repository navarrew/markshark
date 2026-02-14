"""
Shared GUI utility functions.

Centralises platform-specific helpers, style constants, and common
import patterns so individual pages stay DRY.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Colour palette — mirrors the brand values used across the GUI
# ---------------------------------------------------------------------------
TEAL = "#0E817E"
TEAL_HOVER = "#0a6b68"
BLUE = "#0d6efd"
BLUE_HOVER = "#0b5ed7"
GRAY_DISABLED = "#6c757d"

# ---------------------------------------------------------------------------
# Reusable stylesheet for prominent "Run" / action buttons
# ---------------------------------------------------------------------------
RUN_BUTTON_STYLE = (
    "QPushButton { background-color: #0d6efd; color: white; "
    "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 20px; }"
    "QPushButton:hover { background-color: #0b5ed7; }"
    "QPushButton:disabled { background-color: #6c757d; }"
)


# ---------------------------------------------------------------------------
# Platform file / folder opener
# ---------------------------------------------------------------------------
def open_file_or_folder(path: str | Path) -> None:
    """
    Open a file or folder with the system default handler.

    Works cross-platform:
    - macOS: ``open``
    - Windows: ``os.startfile``
    - Linux/other: ``xdg-open``
    """
    path_str = str(path)
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path_str])
        elif system == "Windows":
            os.startfile(path_str)
        else:
            subprocess.Popen(["xdg-open", path_str])
    except Exception as exc:
        print(f"Could not open {path_str}: {exc}")


# ---------------------------------------------------------------------------
# Safe file copy (macOS hidden-flag workaround)
# ---------------------------------------------------------------------------
def safe_copy_file(src: str | Path, dest: str | Path) -> None:
    """
    Copy a file to a destination, stripping macOS hidden flags.

    Uses shutil.copy (not copy2) to avoid propagating extended attributes
    like the hidden flag from bundled assets. Then explicitly clears macOS
    hidden, provenance, and quarantine markers so the file appears in Finder.
    """
    shutil.copy(str(src), str(dest))

    # Clear macOS-specific flags that can hide files in Finder
    try:
        subprocess.run(["chflags", "nohidden", str(dest)], check=False, capture_output=True)
        subprocess.run(["xattr", "-d", "com.apple.provenance", str(dest)], check=False, capture_output=True)
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(dest)], check=False, capture_output=True)
    except FileNotFoundError:
        pass  # Not macOS — chflags/xattr don't exist


# ---------------------------------------------------------------------------
# Template display helper
# ---------------------------------------------------------------------------
def template_display_label(template, tm=None) -> str:
    """
    Return a display string for a template, prefixed with ``★`` if favorited.

    Args:
        template: A ``BubbleSheetTemplate`` with ``.display_name`` and
                  ``.template_id`` attributes.
        tm:       An optional ``TemplateManager`` instance.  When provided,
                  ``tm.is_favorite(template.template_id)`` is checked to
                  decide whether to prepend the star.

    Returns:
        e.g. ``"★ 80Q Standard"`` or ``"80Q Standard"``
    """
    name = template.display_name
    if tm is not None:
        try:
            if tm.is_favorite(template.template_id):
                return f"★ {name}"
        except Exception:
            pass
    return name


# ---------------------------------------------------------------------------
# New-project helper (shared by Welcome page, Project Manager, etc.)
# ---------------------------------------------------------------------------
def create_new_project(parent_widget=None) -> Path | None:
    """
    Prompt the user to name a new project and create it on disk.

    Mirrors the flow in ``ProjectSelector._new_project()``:

    1. Reads the last-used working directory from ``SettingsStore``.
       If none is saved, asks the user to pick one first.
    2. Opens a ``QInputDialog`` for the project name.
    3. Creates the project folder with ``input_files/``, ``score_data/``,
       and ``logs/`` sub-directories.
    4. Registers the new project in ``ProjectRegistry``.

    Returns:
        The ``Path`` to the newly created project, or ``None`` if the user
        cancelled or an error occurred.
    """
    from PySide6.QtWidgets import (
        QFileDialog, QInputDialog, QMessageBox,
    )
    from .models.settings_store import SettingsStore
    from .models.project_registry import ProjectRegistry

    settings = SettingsStore()
    workdir = settings.value("project/working_dir", "")

    # If no working directory is saved, ask the user to pick one
    if not workdir or not Path(workdir).is_dir():
        workdir = QFileDialog.getExistingDirectory(
            parent_widget,
            "Select Course Folder",
            str(Path.home()),
        )
        if not workdir:
            return None
        # Persist for next time
        settings.setValue("project/working_dir", workdir)

    # Prompt for project name
    name, ok = QInputDialog.getText(
        parent_widget,
        "New Assessment",
        f"Enter assessment name:\n(will be created in {workdir})",
    )
    if not ok or not name:
        return None

    # Sanitize
    safe_name = "".join(
        c if c.isalnum() or c in "-_ " else "_" for c in name
    ).strip()
    if not safe_name:
        QMessageBox.warning(
            parent_widget, "Invalid Name",
            "Please enter a valid assessment name.",
        )
        return None

    # Create directory structure
    project_path = Path(workdir) / safe_name
    try:
        project_path.mkdir(exist_ok=True)
        (project_path / "input_files").mkdir(exist_ok=True)
        (project_path / "score_data").mkdir(exist_ok=True)
        (project_path / "logs").mkdir(exist_ok=True)
    except Exception as e:
        QMessageBox.warning(
            parent_widget, "Error",
            f"Could not create assessment: {e}",
        )
        return None

    # Register in global registry
    try:
        ProjectRegistry().register(project_path)
    except Exception:
        pass  # Registration is best-effort

    return project_path


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------
def get_app_version() -> str:
    """Return the installed MarkShark version string, or ``'development'``."""
    try:
        from markshark import __version__
        return __version__
    except ImportError:
        return "development"
