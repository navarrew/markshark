"""
Course and Assessment Manager page — browse, inspect, and manage
courses and their assessments.

Uses the persistent ProjectRegistry (v2 schema) which stores both
course folders and their child assessments.
"""

from pathlib import Path

from PySide6.QtCore import Qt
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
    QScrollArea,
    QFrame,
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
    """Course and Assessment Manager — registry-backed, grouped by course."""

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._registry = ProjectRegistry()
        self._selected_path: str | None = None
        self._course_tables: list[QTableWidget] = []
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
            "MarkShark Course and Assessment Manager",
            "View and manage your courses and assessments.",
        )
        layout.addWidget(header)

        # ── Toolbar ──
        toolbar = QHBoxLayout()

        new_btn = QPushButton("+ New Assessment")
        new_btn.setToolTip("Create a new assessment in your course folder")
        new_btn.clicked.connect(self._on_new_project)
        toolbar.addWidget(new_btn)

        add_btn = QPushButton("Add Existing Assessment")
        add_btn.setToolTip("Register an existing assessment folder")
        add_btn.clicked.connect(self._on_add_project)
        toolbar.addWidget(add_btn)

        create_course_btn = QPushButton("Create New Course")
        create_course_btn.setToolTip("Create a new course folder on disk and register it")
        create_course_btn.clicked.connect(self._on_create_course)
        toolbar.addWidget(create_course_btn)

        add_course_btn = QPushButton("Add Existing Course")
        add_course_btn.setToolTip("Register an existing folder as a course")
        add_course_btn.clicked.connect(self._on_add_course)
        toolbar.addWidget(add_course_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Re-scan all registered assessments")
        refresh_btn.clicked.connect(self._load_projects)
        toolbar.addWidget(refresh_btn)

        toolbar.addStretch()

        remove_missing_btn = QPushButton("Remove Missing")
        remove_missing_btn.setToolTip("Unregister assessments whose folders no longer exist")
        remove_missing_btn.clicked.connect(self._on_remove_missing)
        toolbar.addWidget(remove_missing_btn)

        layout.addLayout(toolbar)

        # ── Splitter: grouped tables on top, detail on bottom ──
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Scroll area for per-course groups ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self.courses_container = QWidget()
        self.courses_layout = QVBoxLayout(self.courses_container)
        self.courses_layout.setSpacing(12)
        self.courses_layout.setContentsMargins(4, 4, 4, 4)
        self.courses_layout.addStretch()

        self.scroll_area.setWidget(self.courses_container)
        splitter.addWidget(self.scroll_area)

        # ── Detail panel ──
        self.detail_box = QGroupBox("Assessment Details")
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
        status_box = QGroupBox("Assessment Status")
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
            "No assessments registered yet.\n\n"
            "Assessments are automatically added when you use the Grader,\n"
            "or click \"+ New Assessment\" to create one, or \"Add Existing...\" to register a folder."
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.empty_label)

        # Initial load
        self._load_projects()

    # ------------------------------------------------------------------
    # Table population (per-course groups)
    # ------------------------------------------------------------------

    def _load_projects(self):
        """Populate per-course groups from the registry."""
        # Clear existing course groups (keep the stretch at the end)
        while self.courses_layout.count() > 1:
            item = self.courses_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._course_tables.clear()

        courses = self._registry.list_courses()
        grouped = self._registry.list_by_course()

        has_any = False

        for course in courses:
            course_path = course["path"]
            projects = grouped.get(course_path, [])
            has_any = True
            group = self._make_course_group(course, projects)
            # Insert before the stretch
            self.courses_layout.insertWidget(
                self.courses_layout.count() - 1, group
            )

        # Handle orphan projects
        orphans = grouped.get("__orphan__", [])
        if orphans:
            has_any = True
            orphan_course = {
                "path": "",
                "name": "Other",
                "description": "Assessments not in a registered course folder",
            }
            group = self._make_course_group(orphan_course, orphans)
            self.courses_layout.insertWidget(
                self.courses_layout.count() - 1, group
            )

        self.scroll_area.setVisible(has_any)
        self.empty_label.setVisible(not has_any)
        if not has_any:
            self.detail_box.setVisible(False)

    def _make_course_group(
        self, course: dict, projects: list[dict]
    ) -> QGroupBox:
        """Build a collapsible group box for one course."""
        name = course.get("name", "Unknown")
        count = len(projects)
        group = QGroupBox(
            f"{name}  ({count} assessment{'s' if count != 1 else ''})"
        )
        group.setCheckable(True)   # toggling collapses/expands
        group.setChecked(True)     # expanded by default

        group_layout = QVBoxLayout(group)

        # Course info row: path + action buttons
        course_path = course.get("path", "")
        if course_path:
            info_row = QHBoxLayout()
            info_row.setSpacing(8)

            path_label = QLabel(course_path)
            path_label.setStyleSheet("color: #888; font-size: 11px;")
            path_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            info_row.addWidget(path_label, 1)

            # Missing-folder warning
            course_exists = Path(course_path).is_dir()
            if not course_exists:
                warn_label = QLabel("\u26a0 Folder not found")
                warn_label.setStyleSheet(
                    "color: #ff6b6b; font-size: 11px; font-weight: bold;"
                )
                info_row.addWidget(warn_label)

            # Inline action buttons for this course
            _SMALL_BTN = (
                "QPushButton { font-size: 10px; padding: 2px 6px; "
                "border: 1px solid #555; border-radius: 2px; }"
                "QPushButton:hover { background-color: #444; }"
            )

            edit_btn = QPushButton("Edit")
            edit_btn.setStyleSheet(_SMALL_BTN)
            edit_btn.setToolTip(
                "Edit course name, parent folder, or subfolder"
            )
            edit_btn.clicked.connect(
                lambda _checked, cp=course_path, cn=name: self._on_rename_course(cp, cn)
            )
            info_row.addWidget(edit_btn)

            if not course_exists:
                relocate_btn = QPushButton("Relocate Folder")
                relocate_btn.setStyleSheet(
                    "QPushButton { font-size: 10px; padding: 2px 6px; "
                    "border: 1px solid #ff6b6b; border-radius: 2px; color: #ff6b6b; }"
                    "QPushButton:hover { background-color: #442222; }"
                )
                relocate_btn.setToolTip(
                    "This folder has moved or been renamed — click to point "
                    "MarkShark to the new location"
                )
                relocate_btn.clicked.connect(
                    lambda _checked, cp=course_path, cn=name: self._on_relocate_course(cp, cn)
                )
                info_row.addWidget(relocate_btn)

            remove_course_btn = QPushButton("Remove Course")
            remove_course_btn.setStyleSheet(_SMALL_BTN)
            remove_course_btn.setToolTip(
                "Remove this course from the registry (no files are deleted)"
            )
            remove_course_btn.clicked.connect(
                lambda _checked, cp=course_path, cn=name: self._on_remove_course(cp, cn)
            )
            info_row.addWidget(remove_course_btn)

            group_layout.addLayout(info_row)

        if not projects:
            empty = QLabel("No assessments in this course folder yet.")
            empty.setStyleSheet("color: #999; font-size: 12px; padding: 8px;")
            group_layout.addWidget(empty)
            return group

        # Table for assessments in this course
        table = QTableWidget(0, len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        table.setSortingEnabled(True)
        table.verticalHeader().setVisible(False)

        hdr = table.horizontalHeader()
        hdr.setSectionResizeMode(_COL_NAME, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(
            _COL_DESCRIPTION, QHeaderView.ResizeMode.Stretch
        )
        hdr.setSectionResizeMode(_COL_PATH, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(
            _COL_RUNS, QHeaderView.ResizeMode.ResizeToContents
        )
        hdr.setSectionResizeMode(
            _COL_REGISTERED, QHeaderView.ResizeMode.ResizeToContents
        )
        hdr.setSectionResizeMode(
            _COL_STATUS, QHeaderView.ResizeMode.ResizeToContents
        )
        hdr.resizeSection(_COL_NAME, 150)
        hdr.resizeSection(_COL_PATH, 250)

        table.itemSelectionChanged.connect(self._on_selection_changed)
        table.itemChanged.connect(self._on_table_item_changed)

        self._populate_table(table, projects)

        # Size table to content height (avoid scroll bars inside each group)
        row_height = 30
        table.setMinimumHeight(
            min(len(projects) * row_height + 35, 300)
        )

        group_layout.addWidget(table)
        self._course_tables.append(table)

        return group

    def _populate_table(
        self, table: QTableWidget, entries: list[dict]
    ):
        """Fill a QTableWidget with assessment rows."""
        table.setSortingEnabled(False)
        table.setRowCount(0)

        for entry in entries:
            path = Path(entry["path"])
            exists = path.is_dir()

            # Gather info from disk
            if exists and get_project_info is not None:
                info = get_project_info(path)
                num_archives = info.get("num_archives", 0)
                last_scored = info.get("last_scored")
                last_scored_label = (
                    last_scored.strftime("%Y-%m-%d %H:%M")
                    if last_scored
                    else "—"
                )
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
                    reg_date = _dt.fromisoformat(reg_date).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                if opened_date:
                    opened_date = _dt.fromisoformat(opened_date).strftime(
                        "%Y-%m-%d %H:%M"
                    )
            except Exception:
                pass

            row = table.rowCount()
            table.insertRow(row)

            # Name item stores path + date metadata for the detail panel
            name_item = QTableWidgetItem(entry.get("name", path.name))
            name_item.setData(Qt.ItemDataRole.UserRole, entry["path"])
            name_item.setData(
                Qt.ItemDataRole.UserRole + 1, last_scored_label
            )
            name_item.setData(
                Qt.ItemDataRole.UserRole + 2, opened_date or "—"
            )
            name_item.setData(
                Qt.ItemDataRole.UserRole + 3, template_name
            )
            name_item.setFlags(
                name_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, _COL_NAME, name_item)

            desc_item = QTableWidgetItem(entry.get("description", ""))
            table.setItem(row, _COL_DESCRIPTION, desc_item)

            path_item = QTableWidgetItem(entry["path"])
            path_item.setFlags(
                path_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, _COL_PATH, path_item)

            runs_item = QTableWidgetItem()
            runs_item.setData(Qt.ItemDataRole.DisplayRole, num_archives)
            runs_item.setFlags(
                runs_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, _COL_RUNS, runs_item)

            reg_item = QTableWidgetItem(reg_date)
            reg_item.setFlags(
                reg_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, _COL_REGISTERED, reg_item)

            status_text = "OK" if exists else "Missing"
            status_item = QTableWidgetItem(status_text)
            status_item.setFlags(
                status_item.flags() & ~Qt.ItemFlag.ItemIsEditable
            )
            table.setItem(row, _COL_STATUS, status_item)

            # Gray out missing projects
            if not exists:
                gray = QColor("#999999")
                for col in range(len(_COLUMNS)):
                    item = table.item(row, col)
                    if item:
                        item.setForeground(gray)

        table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Selection / detail panel
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        """Update the detail panel for the selected project."""
        sender_table = self.sender()
        if not isinstance(sender_table, QTableWidget):
            return

        # Clear selection in all other course tables
        for t in self._course_tables:
            if t is not sender_table:
                t.clearSelection()

        items = sender_table.selectedItems()
        if not items:
            self.detail_box.setVisible(False)
            self._selected_path = None
            return

        row = items[0].row()
        name_item = sender_table.item(row, _COL_NAME)
        if not name_item:
            return

        path_str = name_item.data(Qt.ItemDataRole.UserRole)
        self._selected_path = path_str
        self._selected_table = sender_table
        path = Path(path_str)
        exists = path.is_dir()

        self.detail_name.setText(name_item.text())
        self.detail_template.setText(
            name_item.data(Qt.ItemDataRole.UserRole + 3) or "—"
        )
        self.detail_path.setText(path_str)

        # Description — sync from table cell
        desc_item = sender_table.item(row, _COL_DESCRIPTION)
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
        table = getattr(self, "_selected_table", None)
        if table is None:
            return
        items = table.selectedItems()
        if items:
            row = items[0].row()
            desc_item = table.item(row, _COL_DESCRIPTION)
            if desc_item:
                table.blockSignals(True)
                desc_item.setText(desc)
                table.blockSignals(False)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        """Save description when the user edits directly in the table."""
        if item.column() != _COL_DESCRIPTION:
            return
        table = item.tableWidget()
        if table is None:
            return
        name_item = table.item(item.row(), _COL_NAME)
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
            self, "Select Assessment Folder", str(Path.home())
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
                "Not an Assessment Folder?",
                f"'{p.name}' doesn't look like a MarkShark assessment "
                "(no input_files/ or score_data/ folder found).\n\n"
                "Register it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._registry.register(p)
        self._load_projects()

    def _on_create_course(self):
        """Create a new course folder on disk and register it."""
        from ..dialogs import CourseDialog

        dlg = CourseDialog(
            self,
            title="Create New MarkShark Course Folder",
            confirm_label="Create Course Folder",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        course_path = Path(data["course_path"])
        try:
            course_path.mkdir(parents=True, exist_ok=True)
            self._registry.register_course(course_path, data["name"])
            self._load_projects()
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Could not create course folder: {e}"
            )

    def _on_add_course(self):
        """Register an existing folder as a course via the course dialog."""
        from ..dialogs import CourseDialog

        dlg = CourseDialog(
            self,
            title="Add Existing Course",
            confirm_label="Add Course",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        course_path = Path(data["course_path"])
        if not course_path.is_dir():
            reply = QMessageBox.question(
                self,
                "Folder Doesn't Exist",
                f"The folder does not exist yet:\n"
                f"  {course_path}\n\n"
                "Create it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    course_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Error", f"Could not create folder: {e}"
                    )
                    return
            else:
                return

        self._registry.register_course(course_path, data["name"])
        self._load_projects()

    def _on_rename_course(self, course_path: str, current_name: str):
        """Edit a course's name and/or folder location."""
        from ..dialogs import CourseDialog

        # Split existing path into parent + subfolder for the dialog
        cp = Path(course_path)
        dlg = CourseDialog(
            self,
            title=f"Edit Course \u2014 {current_name}",
            course_name=current_name,
            parent_folder=str(cp.parent),
            subfolder_name=cp.name,
            confirm_label="Save Changes",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        new_name = data["name"]
        new_path = data["course_path"]

        # Apply name change if different
        if new_name != current_name:
            self._registry.set_course_name(Path(course_path), new_name)

        # Apply path change if different
        if new_path != course_path:
            self._registry.update_course_path(Path(course_path), Path(new_path))

        self._load_projects()

    def _on_relocate_course(self, old_path: str, course_name: str):
        """Re-point a course to a new (or moved) folder on disk.

        Opens the same course dialog pre-filled with current values so the
        teacher can change the parent folder or subfolder name.
        """
        from ..dialogs import CourseDialog

        cp = Path(old_path)
        dlg = CourseDialog(
            self,
            title=f"Relocate Course \u2014 {course_name}",
            course_name=course_name,
            parent_folder=str(cp.parent),
            subfolder_name=cp.name,
            confirm_label="Save Changes",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        new_name = data["name"]
        new_path = data["course_path"]

        # Apply name change if different
        if new_name != course_name:
            self._registry.set_course_name(Path(old_path), new_name)

        # Apply path change if different
        if new_path != old_path:
            self._registry.update_course_path(Path(old_path), Path(new_path))

        self._load_projects()

    def _on_remove_course(self, course_path: str, course_name: str):
        """Remove a course from the registry (no files deleted)."""
        reply = QMessageBox.question(
            self,
            "Remove Course",
            f"Remove \"{course_name}\" from the course registry?\n\n"
            f"Folder: {course_path}\n\n"
            "(No files or assessments will be deleted.\n"
            "Assessments will appear under \"Other\" until re-assigned.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._registry.unregister_course(Path(course_path))
        self._load_projects()

    def _on_remove_missing(self):
        """Remove all registry entries whose folders no longer exist."""
        entries = self._registry.list_all()
        missing = [e for e in entries if not Path(e["path"]).is_dir()]

        if not missing:
            QMessageBox.information(
                self, "No Missing Assessments",
                "All registered assessments still exist on disk."
            )
            return

        names = "\n".join(f"  - {e['name']}" for e in missing)
        reply = QMessageBox.question(
            self,
            "Remove Missing Assessments",
            f"Remove {len(missing)} missing assessment(s) from the registry?\n\n"
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
            f"Remove '{name}' from the assessment registry?\n\n"
            "(The assessment folder and files will NOT be deleted.)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._registry.unregister(Path(self._selected_path))
            self._selected_path = None
            self.detail_box.setVisible(False)
            self._load_projects()
