"""
Project and working directory selector widget.

Provides a compact bar for selecting:
- Working directory (where projects live)
- Current project (subdirectory within working dir)

Settings are persisted via QSettings.
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Signal, QSettings
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
    Persists selections via QSettings.
    """

    working_dir_changed = Signal(Path)
    project_changed = Signal(str)  # project name or empty string

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
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

        # Working directory
        frame_layout.addWidget(QLabel("Working Directory:"))

        self.workdir_edit = QLineEdit()
        self.workdir_edit.setPlaceholderText("Select a folder for your grading projects...")
        self.workdir_edit.setMinimumWidth(250)
        self.workdir_edit.textChanged.connect(self._on_workdir_changed)
        frame_layout.addWidget(self.workdir_edit, 1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_workdir)
        frame_layout.addWidget(browse_btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        frame_layout.addWidget(sep)

        # Project selector
        frame_layout.addWidget(QLabel("Project:"))

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(150)
        self.project_combo.setEditable(True)
        self.project_combo.setPlaceholderText("(none)")
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        frame_layout.addWidget(self.project_combo)

        new_project_btn = QPushButton("New...")
        new_project_btn.setMaximumWidth(60)
        new_project_btn.clicked.connect(self._new_project)
        frame_layout.addWidget(new_project_btn)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMaximumWidth(30)
        refresh_btn.setToolTip("Refresh project list")
        refresh_btn.clicked.connect(self._refresh_projects)
        frame_layout.addWidget(refresh_btn)

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
                self.project_combo.setCurrentText(saved_project)

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

    def _refresh_projects(self):
        """Scan working directory for existing projects."""
        workdir = self.workdir_edit.text()
        if not workdir or not Path(workdir).is_dir():
            return

        current_text = self.project_combo.currentText()
        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        # Add empty option
        self.project_combo.addItem("")

        # Find project directories (those containing expected structure)
        workdir_path = Path(workdir)
        projects = []

        for item in sorted(workdir_path.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                # Check if it looks like a project (has input/ or runs/ subfolder)
                if (item / "input").exists() or (item / "runs").exists():
                    projects.append(item.name)
                # Or just list all directories as potential projects
                elif not any(item.iterdir()):
                    # Empty directory - could be new project
                    pass
                else:
                    # Non-empty directory without project structure - still list it
                    projects.append(item.name)

        for proj in projects:
            self.project_combo.addItem(proj)

        # Restore previous selection if still valid
        if current_text:
            idx = self.project_combo.findText(current_text)
            if idx >= 0:
                self.project_combo.setCurrentIndex(idx)

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

            # Create project directory
            project_path = Path(workdir) / safe_name
            try:
                project_path.mkdir(exist_ok=True)
                (project_path / "input").mkdir(exist_ok=True)
                (project_path / "runs").mkdir(exist_ok=True)
                (project_path / "logs").mkdir(exist_ok=True)

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
        """Get the current project name."""
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

        Returns project/runs/<next_run> if project is selected,
        otherwise returns working_dir or a temp directory.
        """
        import tempfile

        project_path = self.project_dir()
        if project_path:
            runs_dir = project_path / "runs"
            runs_dir.mkdir(exist_ok=True)

            # Find next run number
            existing = [d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if existing:
                nums = [int(n.split("_")[1]) for n in existing if n.split("_")[1].isdigit()]
                next_num = max(nums) + 1 if nums else 1
            else:
                next_num = 1

            run_dir = runs_dir / f"run_{next_num:03d}"
            run_dir.mkdir(exist_ok=True)
            return run_dir

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
        if idx >= 0:
            self.project_combo.setCurrentIndex(idx)
        else:
            self.project_combo.setCurrentText(name)
