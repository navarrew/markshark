"""
Project and working directory selector widget.

Provides a compact bar for selecting:
- Course (display name) — backed by a remembered folder path
- Current assessment (subdirectory within the course folder)

Settings are persisted via SettingsStore (~/.markshark/settings.json).
Known course folders are persisted via ProjectRegistry.
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Signal, Qt

from ..models.project_registry import ProjectRegistry
from ..models.settings_store import SettingsStore
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
    QMessageBox,
)


class ProjectSelector(QWidget):
    """
    Compact project/directory selector bar.

    Emits signals when the working directory or project changes.
    Persists selections via SettingsStore.
    """

    working_dir_changed = Signal(Path)
    project_changed = Signal(str)  # project name or empty string

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = SettingsStore()
        self._registry = ProjectRegistry()
        self._setup_ui()
        self._load_settings()

    def showEvent(self, event):
        """Re-populate the course combo every time the widget becomes visible.

        The ProjectRegistry is a singleton, so courses added by the
        Course Manager (or another page) are already in memory — we just
        need to refresh the QComboBox items.
        """
        super().showEvent(event)
        self._populate_course_combo()

    def _setup_ui(self):
        """Build the UI."""
        # Use a frame for visual grouping
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { background-color: #0E817E; border-radius: 3px; }")

        frame_layout = QHBoxLayout(frame)
        frame_layout.setContentsMargins(8, 6, 8, 6)
        frame_layout.setSpacing(12)

        # Project selector (left side)
        frame_layout.addWidget(QLabel("Assessment:"))

        _BAR_BTN = (
            "QPushButton { background-color: #0d6efd; color: white;"
            "              border: none; border-radius: 3px; padding: 3px 10px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
        )

        _COMBO_STYLE = (
            "QComboBox { background-color: #1a1a1a; color: white;"
            "            border: 1px solid #444; border-radius: 3px;"
            "            padding: 3px 8px; }"
            "QComboBox QAbstractItemView { background-color: #1a1a1a;"
            "            color: white; selection-background-color: #0d6efd; }"
        )

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(210)
        self.project_combo.setEditable(False)
        self.project_combo.setPlaceholderText("(none)")
        self.project_combo.setStyleSheet(_COMBO_STYLE)
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        frame_layout.addWidget(self.project_combo, 1)

        new_project_btn = QPushButton("New")
        new_project_btn.setMaximumWidth(50)
        new_project_btn.setStyleSheet(_BAR_BTN)
        new_project_btn.clicked.connect(self._new_project)
        frame_layout.addWidget(new_project_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(70)
        refresh_btn.setToolTip("Refresh assessment list")
        refresh_btn.setStyleSheet(_BAR_BTN)
        refresh_btn.clicked.connect(self._refresh_projects)
        frame_layout.addWidget(refresh_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        frame_layout.addWidget(sep)

        # Course selector (right side) — shows display names, stores paths
        frame_layout.addWidget(QLabel("Course:"))

        self.workdir_combo = QComboBox()
        self.workdir_combo.setMinimumWidth(200)
        self.workdir_combo.setEditable(False)
        self.workdir_combo.setPlaceholderText("Select a course...")
        self.workdir_combo.setStyleSheet(_COMBO_STYLE)
        self.workdir_combo.currentIndexChanged.connect(self._on_workdir_index_changed)
        frame_layout.addWidget(self.workdir_combo)

        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(70)
        browse_btn.setStyleSheet(_BAR_BTN)
        browse_btn.clicked.connect(self._browse_workdir)
        frame_layout.addWidget(browse_btn)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(frame)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self):
        """Load saved settings and populate course combo."""
        saved_workdir = self.settings.value("project/working_dir", "")
        saved_project = self.settings.value("project/last_project", "")

        # Populate course combo from registry
        self._populate_course_combo()

        if saved_workdir:
            # Find the combo entry whose path (UserRole data) matches
            self._select_course_by_path(saved_workdir)
            self._refresh_projects()

        if saved_project:
            idx = self.project_combo.findText(saved_project)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
            else:
                # Project no longer exists — show placeholder
                self.project_combo.setCurrentIndex(-1)

    def _save_settings(self):
        """Save current settings (stores the path, not the display name)."""
        path = self._current_workdir_path() or ""
        self.settings.setValue("project/working_dir", path)
        self.settings.setValue("project/last_project", self.project_combo.currentText())
        self.settings.sync()

    # ------------------------------------------------------------------
    # Course combo (display name / path split)
    # ------------------------------------------------------------------

    def _populate_course_combo(self):
        """Populate the course combo: display names visible, paths in UserRole."""
        self.workdir_combo.blockSignals(True)
        current_path = self._current_workdir_path()
        self.workdir_combo.clear()

        courses = self._registry.list_courses()
        for course in courses:
            display = course["name"]
            path = course["path"]
            self.workdir_combo.addItem(display, userData=path)

        # Restore previous selection by path
        if current_path:
            self._select_course_by_path(current_path)

        self.workdir_combo.blockSignals(False)

    def _select_course_by_path(self, path_str: str):
        """Select the combo entry whose UserRole data matches *path_str*."""
        for i in range(self.workdir_combo.count()):
            if self.workdir_combo.itemData(i, Qt.ItemDataRole.UserRole) == path_str:
                self.workdir_combo.setCurrentIndex(i)
                return
        # Not found — leave at placeholder
        self.workdir_combo.setCurrentIndex(-1)

    def _current_workdir_path(self) -> str:
        """Return the path stored in the currently selected combo item."""
        idx = self.workdir_combo.currentIndex()
        if idx < 0:
            return ""
        return self.workdir_combo.itemData(idx, Qt.ItemDataRole.UserRole) or ""

    def _browse_workdir(self):
        """Open the course dialog to register a new course folder."""
        from ..dialogs import CourseDialog

        dlg = CourseDialog(
            self,
            title="Add Course",
            confirm_label="Add Course",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        course_path = Path(data["course_path"])
        course_path.mkdir(parents=True, exist_ok=True)
        self._registry.register_course(course_path, data["name"])

        self._populate_course_combo()
        self._select_course_by_path(str(course_path))
        # Trigger the change handler manually since blockSignals may have eaten it
        self._on_workdir_index_changed(self.workdir_combo.currentIndex())

    def _on_workdir_index_changed(self, index: int):
        """Handle course selection change (by index, not text)."""
        path_str = self._current_workdir_path()
        if not path_str:
            return
        path = Path(path_str)
        if path.is_dir():
            self._registry.update_course_last_opened(path)
            self._refresh_projects()
            self.working_dir_changed.emit(path)
            self._save_settings()

    def _on_project_changed(self, text: str):
        """Handle project selection change."""
        self.project_changed.emit(text)
        self._save_settings()

        # Auto-register project in the global registry
        project_path = self.project_dir()
        if project_path and project_path.exists():
            self._registry.register(project_path)

    def _refresh_projects(self):
        """Scan working directory for existing projects."""
        workdir = self._current_workdir_path()
        if not workdir or not Path(workdir).is_dir():
            return

        current_text = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        # Find project directories (those containing expected structure)
        workdir_path = Path(workdir)
        projects = []

        for item in sorted(workdir_path.iterdir()):
            if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("mock_"):
                # Project structure
                if (item / "input_files").exists() or (item / "score_data").exists():
                    projects.append(item.name)
                # Non-empty directory without project structure - still list it
                elif any(item.iterdir()):
                    projects.append(item.name)

        for proj in projects:
            self.project_combo.addItem(proj)

        # Restore previous selection if still valid, otherwise show placeholder
        restored = False
        if current_text:
            idx = self.project_combo.findText(current_text)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
                restored = True
        if not restored:
            self.project_combo.setCurrentIndex(-1)  # show "(none selected)" placeholder

        self.project_combo.blockSignals(False)

    def _new_project(self):
        """Create a new project."""
        workdir = self._current_workdir_path()
        if not workdir or not Path(workdir).is_dir():
            QMessageBox.warning(
                self,
                "No Course Selected",
                "Please select a course first."
            )
            return

        # Prompt for project name
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "New Assessment",
            "Enter assessment name:",
        )

        if ok and name:
            # Sanitize name
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
            safe_name = safe_name.strip()

            if not safe_name:
                QMessageBox.warning(self, "Invalid Name", "Please enter a valid assessment name.")
                return

            # Create project directory with flat structure
            project_path = Path(workdir) / safe_name
            try:
                project_path.mkdir(exist_ok=True)
                (project_path / "input_files").mkdir(exist_ok=True)
                (project_path / "score_data").mkdir(exist_ok=True)
                (project_path / "logs").mkdir(exist_ok=True)

                # Register in global registry
                self._registry.register(project_path)

                # Refresh and select new project
                self._refresh_projects()
                idx = self.project_combo.findText(safe_name)
                if idx >= 0:
                    self.project_combo.setCurrentIndex(idx)

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not create assessment: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def working_dir(self) -> Optional[Path]:
        """Get the current working directory as a Path."""
        path_str = self._current_workdir_path()
        if path_str and Path(path_str).is_dir():
            return Path(path_str)
        return None

    def project_name(self) -> str:
        """Get the current project name (empty string if none selected)."""
        if self.project_combo.currentIndex() < 0:
            return ""
        return self.project_combo.currentText()

    def project_dir(self) -> Optional[Path]:
        """Get the full path to the current project directory."""
        workdir = self.working_dir()
        project = self.project_name()
        if workdir and project:
            project_path = workdir / project
            if project_path.is_dir():
                return project_path
        return None

    def output_dir(self) -> Path:
        """
        Get the appropriate output directory.

        Returns the project directory directly (flat structure).
        Falls back to working_dir or a temp directory.
        """
        import tempfile

        project_path = self.project_dir()
        if project_path:
            # Ensure subdirs exist
            (project_path / "input_files").mkdir(exist_ok=True)
            (project_path / "score_data").mkdir(exist_ok=True)
            (project_path / "logs").mkdir(exist_ok=True)
            return project_path

        workdir = self.working_dir()
        if workdir:
            return workdir

        return Path(tempfile.mkdtemp(prefix="markshark_"))

    def set_working_dir(self, path: str):
        """Set the working directory programmatically (by path)."""
        # If the path isn't in the combo, register it first
        if path and Path(path).is_dir():
            found = False
            for i in range(self.workdir_combo.count()):
                if self.workdir_combo.itemData(i, Qt.ItemDataRole.UserRole) == path:
                    found = True
                    break
            if not found:
                self._registry.register_course(Path(path))
                self._populate_course_combo()
        self._select_course_by_path(path)

    def set_project(self, name: str):
        """Set the project name programmatically."""
        idx = self.project_combo.findText(name)
        if idx < 0:
            # Project not in combo yet — rescan directory and retry
            self._refresh_projects()
            idx = self.project_combo.findText(name)
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)
        else:
            self.project_combo.setCurrentText(name)
