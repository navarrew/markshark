"""
Project and working directory selector widget.

Provides a compact bar for selecting:
- Working directory (where projects live)
- Current project (subdirectory within working dir)

Settings are persisted via SettingsStore (~/.markshark/settings.json).
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Signal

from ..models.project_registry import ProjectRegistry
from ..models.settings_store import SettingsStore
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFileDialog,
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
        frame_layout.addWidget(QLabel("Project:"))

        _BAR_BTN = (
            "QPushButton { background-color: #0d6efd; color: white;"
            "              border: none; border-radius: 3px; padding: 3px 10px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
        )

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(140)
        self.project_combo.setEditable(False)
        self.project_combo.setPlaceholderText("(none)")
        self.project_combo.setStyleSheet(
            "QComboBox { background-color: #1a1a1a; color: white;"
            "            border: 1px solid #444; border-radius: 3px;"
            "            padding: 3px 8px; }"
            "QComboBox QAbstractItemView { background-color: #1a1a1a;"
            "            color: white; selection-background-color: #0d6efd; }"
        )
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        frame_layout.addWidget(self.project_combo)

        new_project_btn = QPushButton("New")
        new_project_btn.setMaximumWidth(50)
        new_project_btn.setStyleSheet(_BAR_BTN)
        new_project_btn.clicked.connect(self._new_project)
        frame_layout.addWidget(new_project_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setMaximumWidth(70)
        refresh_btn.setToolTip("Refresh project list")
        refresh_btn.setStyleSheet(_BAR_BTN)
        refresh_btn.clicked.connect(self._refresh_projects)
        frame_layout.addWidget(refresh_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        frame_layout.addWidget(sep)

        # Working directory (right side)
        frame_layout.addWidget(QLabel("Working Directory:"))

        self.workdir_edit = QLineEdit()
        self.workdir_edit.setPlaceholderText("Select a folder for your grading projects...")
        self.workdir_edit.setMinimumWidth(250)
        self.workdir_edit.textChanged.connect(self._on_workdir_changed)
        frame_layout.addWidget(self.workdir_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.setStyleSheet(_BAR_BTN)
        browse_btn.clicked.connect(self._browse_workdir)
        frame_layout.addWidget(browse_btn)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(frame)

    def _load_settings(self):
        """Load saved settings."""
        saved_workdir = self.settings.value("project/working_dir", "")
        saved_project = self.settings.value("project/last_project", "")

        if saved_workdir:
            self.workdir_edit.setText(saved_workdir)
            self._refresh_projects()

        if saved_project:
            idx = self.project_combo.findText(saved_project)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)
            else:
                # Project no longer exists — show placeholder
                self.project_combo.setCurrentIndex(-1)

    def _save_settings(self):
        """Save current settings."""
        self.settings.setValue("project/working_dir", self.workdir_edit.text())
        self.settings.setValue("project/last_project", self.project_combo.currentText())
        self.settings.sync()

    def _browse_workdir(self):
        """Open folder browser for working directory."""
        current = self.workdir_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            current,
        )
        if path:
            self.workdir_edit.setText(path)

    def _on_workdir_changed(self, text: str):
        """Handle working directory change."""
        path = Path(text) if text else None
        if path and path.is_dir():
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
        workdir = self.workdir_edit.text()
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
        workdir = self.workdir_edit.text()
        if not workdir or not Path(workdir).is_dir():
            QMessageBox.warning(
                self,
                "No Working Directory",
                "Please select a working directory first."
            )
            return

        # Prompt for project name
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Enter project name:",
        )

        if ok and name:
            # Sanitize name
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
            safe_name = safe_name.strip()

            if not safe_name:
                QMessageBox.warning(self, "Invalid Name", "Please enter a valid project name.")
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
                QMessageBox.warning(self, "Error", f"Could not create project: {e}")

    # Public API

    def working_dir(self) -> Optional[Path]:
        """Get the current working directory as a Path."""
        text = self.workdir_edit.text()
        if text and Path(text).is_dir():
            return Path(text)
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
        """Set the working directory programmatically."""
        self.workdir_edit.setText(path)

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
