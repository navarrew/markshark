"""
Project Manager page — browse, inspect, and manage registered projects.

Maintains a persistent JSON registry of known project directories and
displays their run history, disk usage, and status.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QFormLayout,
    QSplitter,
    QFileDialog,
    QMessageBox,
    QAbstractItemView,
)

from ..widgets import PageHeader
from ..models.project_registry import ProjectRegistry

# Best-effort import of project_utils
try:
    from markshark.project_utils import get_project_info
except ImportError:
    get_project_info = None


# ── Column indices ──
_COL_NAME = 0
_COL_DESCRIPTION = 1
_COL_PATH = 2
_COL_RUNS = 3
_COL_REGISTERED = 4
_COL_STATUS = 5
_COLUMNS = ["Name", "Description", "Path", "Archives", "Registered", "Status"]


def _human_size(nbytes: int) -> str:
    """Format bytes as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _folder_size(path: Path) -> int:
    """Total size in bytes of all files under *path*."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except Exception:
        pass
    return total


def _open_folder(folder: str):
    """Open a folder in the system file manager."""
    from ..utils import open_file_or_folder
    open_file_or_folder(folder)


class ProjectManagerPage(QWidget):
    """Project Manager page — registry-backed project browser."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._registry = ProjectRegistry()
        self._selected_path: str | None = None
        self._setup_ui()

    def showEvent(self, event):
        """Reload projects every time the page becomes visible."""
        super().showEvent(event)
        self._registry._load()          # re-read JSON in case another widget wrote it
        self._load_projects()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark Project Manager",
            "View and manage your grading projects across all directories.",
        )
        layout.addWidget(header)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        new_btn = QPushButton("+ New Project")
        new_btn.setToolTip("Create a new project in your working directory")
        new_btn.clicked.connect(self._on_new_project)
        toolbar.addWidget(new_btn)

        add_btn = QPushButton("Add Existing...")
        add_btn.setToolTip("Register an existing project folder")
        add_btn.clicked.connect(self._on_add_project)
        toolbar.addWidget(add_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-scan all registered projects")
        refresh_btn.clicked.connect(self._load_projects)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()

        remove_missing_btn = QPushButton("Remove Missing")
        remove_missing_btn.setToolTip("Unregister projects whose folders no longer exist")
        remove_missing_btn.clicked.connect(self._on_remove_missing)
        toolbar.addWidget(remove_missing_btn)

        layout.addLayout(toolbar)

        # ── Splitter: table on top, detail on bottom ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Project table ──
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_COL_DESCRIPTION, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(_COL_PATH, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(_COL_RUNS, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_REGISTERED, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        hdr.resizeSection(_COL_NAME, 150)
        hdr.resizeSection(_COL_PATH, 250)

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)

        splitter.addWidget(self.table)

        # ── Detail panel ──
        self.detail_box = QGroupBox("Project Details")
        detail_layout = QVBoxLayout(self.detail_box)

        # Two-column layout: info on left, status checklist on right
        columns_layout = QHBoxLayout()

        # ── Left column: project info ──
        info_form = QFormLayout()
        info_form.setHorizontalSpacing(16)
        self.detail_name = QLabel("—")
        self.detail_path = QLabel("—")
        self.detail_path.setWordWrap(False)
        self.detail_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.detail_description = QLineEdit()
        self.detail_description.setPlaceholderText(
            "E.g., 'Final exam from 2025', 'BIO101 midterm'..."
        )
        self.detail_description.textChanged.connect(self._on_description_changed)
        self.detail_template = QLabel("—")
        self.detail_runs = QLabel("—")
        self.detail_size = QLabel("—")
        self.detail_last_scored = QLabel("—")
        self.detail_last_opened = QLabel("—")

        info_form.addRow("Name:", self.detail_name)
        info_form.addRow("Template:", self.detail_template)
        info_form.addRow("Description:", self.detail_description)
        info_form.addRow("Path:", self.detail_path)
        info_form.addRow("Last Scored:", self.detail_last_scored)
        info_form.addRow("Last Opened:", self.detail_last_opened)
        info_form.addRow("Archives:", self.detail_runs)
        info_form.addRow("Disk usage:", self.detail_size)

        columns_layout.addLayout(info_form, 1)

        # ── Right column: project status checklist ──
        status_box = QGroupBox("Project Status")
        status_layout = QVBoxLayout(status_box)
        status_layout.setSpacing(6)

        self.status_scans = QLabel()
        self.status_key = QLabel()
        self.status_roster = QLabel()
        self.status_aligned = QLabel()
        self.status_scored = QLabel()
        self.status_report = QLabel()

        for lbl in (
            self.status_scans, self.status_key, self.status_roster,
            self.status_aligned, self.status_scored, self.status_report,
        ):
            status_layout.addWidget(lbl)

        status_layout.addStretch()
        columns_layout.addWidget(status_box)

        detail_layout.addLayout(columns_layout)

        # Action buttons
        btn_row = QHBoxLayout()

        self.open_grader_btn = QPushButton("Open in Grader")
        self.open_grader_btn.clicked.connect(self._on_open_in_grader)
        btn_row.addWidget(self.open_grader_btn)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._on_open_folder)
        btn_row.addWidget(self.open_folder_btn)

        btn_row.addStretch()

        self.remove_btn = QPushButton("Remove from Registry")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_btn)

        detail_layout.addLayout(btn_row)

        self.detail_box.setVisible(False)
        splitter.addWidget(self.detail_box)

        splitter.setSizes([400, 200])
        layout.addWidget(splitter, 1)

        # ── Empty-state label ──
        self.empty_label = QLabel(
            "No projects registered yet.\n\n"
            "Projects are automatically added when you use the Grader,\n"
            "or click \"+ New Project\" to create one, or \"Add Existing...\" to register a folder."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.empty_label)

        # Initial load
        self._load_projects()

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _load_projects(self):
        """Populate the table from the registry."""
        entries = self._registry.list_all()

        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        for entry in entries:
            path = Path(entry["path"])
            exists = path.is_dir()

            # Gather info from disk
            if exists and get_project_info is not None:
                info = get_project_info(path)
                num_archives = info.get("num_archives", 0)
                last_scored = info.get("last_scored")
                last_scored_label = last_scored.strftime("%Y-%m-%d %H:%M") if last_scored else "—"
                template_name = info.get("template_name") or "—"
            else:
                num_archives = 0
                last_scored_label = "—"
                template_name = "—"

            # Format dates nicely (ISO -> human-friendly)
            reg_date = entry.get("registered_at", "")
            opened_date = entry.get("last_opened", "")
            try:
                from datetime import datetime as _dt
                if reg_date:
                    reg_date = _dt.fromisoformat(reg_date).strftime("%Y-%m-%d %H:%M")
                if opened_date:
                    opened_date = _dt.fromisoformat(opened_date).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

            row = self.table.rowCount()
            self.table.insertRow(row)

            # Name item stores path + date metadata for the detail panel
            name_item = QTableWidgetItem(entry.get("name", path.name))
            name_item.setData(Qt.ItemDataRole.UserRole, entry["path"])
            name_item.setData(Qt.ItemDataRole.UserRole + 1, last_scored_label)
            name_item.setData(Qt.ItemDataRole.UserRole + 2, opened_date or "—")
            name_item.setData(Qt.ItemDataRole.UserRole + 3, template_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_NAME, name_item)

            desc_item = QTableWidgetItem(entry.get("description", ""))
            self.table.setItem(row, _COL_DESCRIPTION, desc_item)

            path_item = QTableWidgetItem(entry["path"])
            path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_PATH, path_item)

            runs_item = QTableWidgetItem()
            runs_item.setData(Qt.ItemDataRole.DisplayRole, num_archives)
            runs_item.setFlags(runs_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_RUNS, runs_item)

            reg_item = QTableWidgetItem(reg_date)
            reg_item.setFlags(reg_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_REGISTERED, reg_item)

            status_text = "OK" if exists else "Missing"
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, _COL_STATUS, status_item)

            # Gray out missing projects
            if not exists:
                gray = QColor("#999999")
                for col in range(len(_COLUMNS)):
                    item = self.table.item(row, col)
                    if item:
                        item.setForeground(gray)

        self.table.setSortingEnabled(True)

        has_projects = self.table.rowCount() > 0
        self.table.setVisible(has_projects)
        self.empty_label.setVisible(not has_projects)
        if not has_projects:
            self.detail_box.setVisible(False)

    # ------------------------------------------------------------------
    # Selection / detail panel
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        """Update the detail panel for the selected project."""
        items = self.table.selectedItems()
        if not items:
            self.detail_box.setVisible(False)
            self._selected_path = None
            return

        row = items[0].row()
        name_item = self.table.item(row, _COL_NAME)
        if not name_item:
            return

        path_str = name_item.data(Qt.ItemDataRole.UserRole)
        self._selected_path = path_str
        path = Path(path_str)
        exists = path.is_dir()

        self.detail_name.setText(name_item.text())
        self.detail_template.setText(
            name_item.data(Qt.ItemDataRole.UserRole + 3) or "—"
        )
        self.detail_path.setText(path_str)

        # Description — sync from table cell
        desc_item = self.table.item(row, _COL_DESCRIPTION)
        self.detail_description.blockSignals(True)
        self.detail_description.setText(desc_item.text() if desc_item else "")
        self.detail_description.blockSignals(False)

        # Dates (stored on the name item as extra UserRole data)
        self.detail_last_scored.setText(
            name_item.data(Qt.ItemDataRole.UserRole + 1) or "—"
        )
        self.detail_last_opened.setText(
            name_item.data(Qt.ItemDataRole.UserRole + 2) or "—"
        )

        # Archives
        if exists and get_project_info is not None:
            info = get_project_info(path)
            self.detail_runs.setText(str(info.get("num_archives", 0)))
        else:
            self.detail_runs.setText("—" if not exists else "0")

        # Disk usage
        if exists:
            size = _folder_size(path)
            self.detail_size.setText(_human_size(size))
        else:
            self.detail_size.setText("—")

        # Project status checklist
        self._update_status_checklist(path, exists)

        # Enable/disable buttons
        self.open_grader_btn.setEnabled(exists)
        self.open_folder_btn.setEnabled(exists)

        self.detail_box.setVisible(True)

    def _update_status_checklist(self, path: Path, exists: bool):
        """Populate the right-side status checklist for the selected project."""
        _CHECK = "\u2705"   # green check
        _CROSS = "\u274c"   # red cross

        def _mark(label_text: str, present: bool) -> str:
            icon = _CHECK if present else _CROSS
            return f"{icon}  {label_text}"

        if not exists:
            for lbl, text in (
                (self.status_scans, "Scans"),
                (self.status_key, "Answer Key"),
                (self.status_roster, "Roster"),
                (self.status_aligned, "Aligned"),
                (self.status_scored, "Scored"),
                (self.status_report, "Report"),
            ):
                lbl.setText(_mark(text, False))
            return

        inp = path / "input_files"

        has_scans = any(inp.glob("*.pdf")) if inp.exists() else False
        has_key = (
            any(inp.glob("key.*")) or any(inp.glob("answer_key.*"))
        ) if inp.exists() else False
        has_roster = any(inp.glob("roster.*")) if inp.exists() else False
        has_aligned = (inp / "aligned_scans.pdf").exists() if inp.exists() else False
        has_scored = (path / "scored_scans.pdf").exists()
        has_report = (path / "exam_report.xlsx").exists()

        self.status_scans.setText(_mark("Scans", has_scans))
        self.status_key.setText(_mark("Answer Key", has_key))
        self.status_roster.setText(_mark("Roster", has_roster))
        self.status_aligned.setText(_mark("Aligned", has_aligned))
        self.status_scored.setText(_mark("Scored", has_scored))
        self.status_report.setText(_mark("Report", has_report))

    def _on_description_changed(self):
        """Save description when the user edits via the detail panel."""
        if not self._selected_path:
            return
        desc = self.detail_description.text().strip()
        self._registry.set_description(Path(self._selected_path), desc)
        # Sync to the table cell (block signals to avoid loop)
        items = self.table.selectedItems()
        if items:
            row = items[0].row()
            desc_item = self.table.item(row, _COL_DESCRIPTION)
            if desc_item:
                self.table.blockSignals(True)
                desc_item.setText(desc)
                self.table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        """Save description when the user edits directly in the table."""
        if item.column() != _COL_DESCRIPTION:
            return
        name_item = self.table.item(item.row(), _COL_NAME)
        if not name_item:
            return
        path_str = name_item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        desc = item.text().strip()
        self._registry.set_description(Path(path_str), desc)
        # Sync to the detail panel if this row is selected
        if path_str == self._selected_path:
            self.detail_description.blockSignals(True)
            self.detail_description.setText(desc)
            self.detail_description.blockSignals(False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_new_project(self):
        """Create a new project via the shared helper and refresh the list."""
        from ..utils import create_new_project

        project_path = create_new_project(parent_widget=self)
        if project_path:
            self._load_projects()

    def _on_add_project(self):
        """Browse for an existing project folder and register it."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Project Folder", str(Path.home())
        )
        if not path:
            return

        p = Path(path)
        # Validate it looks like a project
        has_input_files = (p / "input_files").exists()
        has_score_data = (p / "score_data").exists()

        if not (has_input_files or has_score_data):
            reply = QMessageBox.question(
                self,
                "Not a Project Folder?",
                f"'{p.name}' doesn't look like a MarkShark project "
                "(no input_files/ or score_data/ folder found).\n\n"
                "Register it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._registry.register(p)
        self._load_projects()

    def _on_remove_missing(self):
        """Remove all registry entries whose folders no longer exist."""
        entries = self._registry.list_all()
        missing = [e for e in entries if not Path(e["path"]).is_dir()]

        if not missing:
            QMessageBox.information(
                self, "No Missing Projects",
                "All registered projects still exist on disk."
            )
            return

        names = "\n".join(f"  - {e['name']}" for e in missing)
        reply = QMessageBox.question(
            self,
            "Remove Missing Projects",
            f"Remove {len(missing)} missing project(s) from the registry?\n\n"
            f"{names}\n\n"
            "(No files will be deleted.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for e in missing:
                self._registry.unregister(Path(e["path"]))
            self._load_projects()

    def _on_open_in_grader(self):
        """Navigate to the Grader with this project loaded."""
        if not self._selected_path or not self._main_window:
            return
        path = Path(self._selected_path)
        if path.is_dir() and hasattr(self._main_window, "navigate_to_grader"):
            self._main_window.navigate_to_grader(path)

    def _on_open_folder(self):
        """Open the project folder in the system file manager."""
        if not self._selected_path:
            return
        path = Path(self._selected_path)
        if path.is_dir():
            _open_folder(str(path))

    def _on_remove(self):
        """Remove the selected project from the registry."""
        if not self._selected_path:
            return

        name = Path(self._selected_path).name
        reply = QMessageBox.question(
            self,
            "Remove from Registry",
            f"Remove '{name}' from the project registry?\n\n"
            "(The project folder and files will NOT be deleted.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._registry.unregister(Path(self._selected_path))
            self._selected_path = None
            self.detail_box.setVisible(False)
            self._load_projects()
