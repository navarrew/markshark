"""
Review & Correct page - review grading results and apply corrections.

Features:
- Full spreadsheet view of scored CSV data (all columns)
- Inline editing: dropdown for Q cells, text edit for name/ID cells
- PDF scan preview with zoom toggle and multi-page support
- Flag info panel showing issues for selected student
- Corrections saved to append-only log
"""

import csv
import shutil
from pathlib import Path
from typing import Optional, List, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QStyledItemDelegate,
    QMenu,
    QFrame,
)

from ..widgets import PageHeader, PDFPreview, ProjectSelector
from ..models import CorrectionLog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_field(row: Dict[str, str], *field_names, default: str = "") -> str:
    """Get a field value trying multiple possible column names."""
    for name in field_names:
        if name in row and row[name]:
            return row[name]
    return default


# Columns that should not be user-editable
_READ_ONLY_COLUMNS = frozenset({
    "Page", "Version", "Score", "Correct", "Incorrect", "Blank", "Multi", "Percent",
    "Flagged", "FlagDetails",
})

# Columns to hide from the user (internal data)
_HIDDEN_COLUMNS = frozenset({"Flagged", "FlagDetails"})

# Columns that should sort numerically
_NUMERIC_COLUMNS = frozenset({
    "Page", "Score", "Correct", "Incorrect", "Blank", "Multi", "Percent",
})

# Columns that may be numeric or alphabetic (e.g. Version: "A"/"B" or "1"/"2")
_SMART_SORT_COLUMNS = frozenset({"Version"})

# Cell background colours
_COLOR_CORRECTED = QColor("#BBDEFB")  # blue — has a correction
_COLOR_BLANK_Q = QColor("#F9C8F5")   # pink — blank answer
_COLOR_MULTI_Q = QColor("#FFB366")   # orange — multi-mark


# ---------------------------------------------------------------------------
# Delegate: dropdown editor for Q cells
# ---------------------------------------------------------------------------

class AnswerChoiceDelegate(QStyledItemDelegate):
    """
    Item delegate that presents a QComboBox with answer choices
    when the user double-clicks a Q column cell.

    The choices are set dynamically from the loaded CSV data (KEY rows
    + student answers) so they match the template — e.g. A-E for a
    five-choice sheet, T/F for true-false, A-J for ten-choice, etc.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._choices: List[str] = ["", "A", "B", "C", "D", "E"]  # fallback

    def set_choices(self, choices: List[str]):
        """Set the answer choices (blank is always prepended)."""
        self._choices = [""] + [c for c in choices if c]

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self._choices)
        # Commit data immediately when user picks from the dropdown,
        # rather than waiting for focus loss.
        combo.activated.connect(self._on_activated)
        return combo

    def _on_activated(self, _idx):
        """Called when user picks a value. Commit and close the editor."""
        editor = self.sender()
        if editor:
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)

    def setEditorData(self, editor: QComboBox, index):
        value = index.data(Qt.ItemDataRole.DisplayRole) or ""
        idx = editor.findText(value)
        editor.setCurrentIndex(max(0, idx))

    def setModelData(self, editor: QComboBox, model, index):
        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically instead of alphabetically.

    Values like page ranges ("1-2") sort by the first number.
    Non-numeric values sort after all numeric ones.
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        def _num(text: str) -> float:
            try:
                # Handle page ranges like "1-2" — sort by first number
                return float(text.split("-")[0])
            except (ValueError, IndexError):
                return float("inf")

        return _num(self.text()) < _num(other.text())


class _SmartSortItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when values are numbers,
    alphabetically when they are letters (e.g. Version: "A"/"B" or "1"/"2").
    """

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a, b = self.text().strip(), other.text().strip()
        try:
            return float(a) < float(b)
        except ValueError:
            return a.lower() < b.lower()


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class ReviewPanelPage(QWidget):
    """
    Review and correct grading results.

    Auto-loads the project's score_data/results.csv when a project is selected.

    Layout:
        +------------------------------------------------------------------+
        | Header                                                           |
        +------------------------------------------------------------------+
        | ProjectSelector                                                  |
        +------------------------------------------------------------------+
        | [Show All] [Flagged Only]                                        |
        +------------------------------+-----------------------------------+
        | SPREADSHEET (full CSV)       |  [Fit Page] [Zoom In]             |
        | Pg|Ver|Last|First|ID|C|I|..  |  +-----------------------------+  |
        | --|---|----|----|--|--|--|..  |  | PDF page(s)                 |  |
        | 1 |A  |Doe |Jane|10|..|..|  |  | (scrollable when zoomed)    |  |
        |                              |  +-----------------------------+  |
        | FLAG INFO: blank at Q10, ... |                                   |
        +------------------------------+-----------------------------------+
        | Status: 45 students | 8 flagged | 3 corrections                  |
        +------------------------------------------------------------------+
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._correction_log: Optional[CorrectionLog] = None
        self._scored_data: List[Dict[str, str]] = []
        self._current_student_idx: int = -1
        self._show_flagged_only: bool = True
        self._scored_pdf_path: Optional[Path] = None
        self._columns: List[str] = []           # current column names in display order
        self._programmatic_update: bool = False  # suppress cellChanged during population
        self._zoom_mode: str = "fit"             # "fit" or "scroll"
        self._current_dpi: int = 72

        self._answer_delegate = AnswerChoiceDelegate(self)
        self._last_correction_key: str = ""  # debounce: "row:col:value"

        self._setup_ui()

        # Kick-start from whatever the ProjectSelector restored from settings
        # (signals fired during __init__ before connections were wired up)
        project_dir = self.project_selector.project_dir()
        if project_dir:
            self._scan_for_scored_files(project_dir)

    def showEvent(self, event):
        """Re-scan for scored files every time the page becomes visible.

        This ensures newly-created results (e.g. from Grader) appear in the
        Results dropdown.  If newer results exist than what is currently loaded,
        a subtle notification bar is shown so the user can choose to load them.
        """
        super().showEvent(event)
        self._new_results_bar.hide()

        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return

        prev_label = self.results_combo.currentText()
        prev_count = self.results_combo.count()
        self._scan_for_scored_files(project_dir)
        new_count = self.results_combo.count()

        if new_count == 0:
            return

        newest_label = self.results_combo.itemText(new_count - 1)

        if not prev_label or prev_count == 0:
            # Nothing was loaded before — just show the latest results
            self.results_combo.setCurrentIndex(new_count - 1)
            return

        # Restore the previously-selected results file
        idx = self.results_combo.findText(prev_label)
        if idx >= 0:
            self.results_combo.blockSignals(True)
            self.results_combo.setCurrentIndex(idx)
            self.results_combo.blockSignals(False)

        # If there are newer results than what was selected, show the bar
        if newest_label != prev_label and new_count > prev_count:
            self._new_results_label.setText(
                f"New results available: {newest_label}"
            )
            self._pending_newest_idx = new_count - 1
            self._new_results_bar.show()

    def _load_newest_results(self):
        """Switch to the newest results file (from the notification bar)."""
        self._new_results_bar.hide()
        idx = getattr(self, '_pending_newest_idx', -1)
        if idx >= 0 and idx < self.results_combo.count():
            self.results_combo.setCurrentIndex(idx)

    # =====================================================================
    # UI Construction
    # =====================================================================

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header
        header = PageHeader(
            "MarkShark - Review & Correct",
            "Review grading results and manually apply corrections."
        )
        layout.addWidget(header)

        # Project / directory selector bar
        self.project_selector = ProjectSelector()
        self.project_selector.working_dir_changed.connect(self._on_directory_changed)
        self.project_selector.project_changed.connect(self._on_project_changed)
        layout.addWidget(self.project_selector)

        # Hidden combo used internally for results tracking (not shown in UI)
        self.results_combo = QComboBox()
        self.results_combo.setVisible(False)
        self.results_combo.currentTextChanged.connect(self._on_results_changed)
        layout.addWidget(self.results_combo)

        # Hidden state for export/log (kept so internal methods still work)
        self.export_btn = QPushButton()
        self.export_btn.setVisible(False)
        self.view_log_btn = QPushButton()
        self.view_log_btn.setVisible(False)

        # "Newer results available" notification bar (hidden by default)
        self._new_results_bar = QFrame()
        self._new_results_bar.setStyleSheet(
            "QFrame { background: #FFF3CD; border: 1px solid #FFCA2C; "
            "border-radius: 4px; padding: 4px 8px; }"
            " QFrame QLabel { color: #664D03; }"
            " QFrame QPushButton { color: #664D03; }"
        )
        bar_layout = QHBoxLayout(self._new_results_bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        self._new_results_label = QLabel()
        self._new_results_label.setStyleSheet("color: #664D03; font-weight: bold;")
        bar_layout.addWidget(self._new_results_label)
        bar_layout.addStretch()
        self._new_results_load_btn = QPushButton("Load")
        self._new_results_load_btn.setFixedWidth(60)
        self._new_results_load_btn.clicked.connect(self._load_newest_results)
        bar_layout.addWidget(self._new_results_load_btn)
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFixedWidth(28)
        dismiss_btn.setFlat(True)
        dismiss_btn.clicked.connect(self._new_results_bar.hide)
        bar_layout.addWidget(dismiss_btn)
        self._new_results_bar.hide()
        layout.addWidget(self._new_results_bar)

        # Filter bar
        filter_layout = QHBoxLayout()
        self.show_all_btn = QPushButton("Show All")
        self.show_all_btn.setCheckable(True)
        self.show_all_btn.clicked.connect(lambda: self._set_filter(False))
        filter_layout.addWidget(self.show_all_btn)

        self.show_flagged_btn = QPushButton("Flagged Only")
        self.show_flagged_btn.setCheckable(True)
        self.show_flagged_btn.setChecked(True)
        self.show_flagged_btn.clicked.connect(lambda: self._set_filter(True))
        filter_layout.addWidget(self.show_flagged_btn)

        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # ----- Main content splitter -----
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_splitter, 1)

        # ---- Left panel: spreadsheet + flag info ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Spreadsheet
        self.spreadsheet = QTableWidget()
        self.spreadsheet.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.spreadsheet.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.spreadsheet.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.AnyKeyPressed
        )
        self.spreadsheet.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.spreadsheet.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.spreadsheet.horizontalHeader().setMinimumSectionSize(36)
        self.spreadsheet.horizontalHeader().setSortIndicatorShown(True)
        self.spreadsheet.verticalHeader().setVisible(False)
        self.spreadsheet.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.spreadsheet.customContextMenuRequested.connect(self._on_context_menu)
        self.spreadsheet.currentCellChanged.connect(self._on_row_selected)
        self.spreadsheet.cellChanged.connect(self._on_cell_changed)
        left_layout.addWidget(self.spreadsheet, 1)

        # Flag info label
        self.flag_info_label = QLabel("Select a student to see flags.")
        self.flag_info_label.setWordWrap(True)
        self.flag_info_label.setStyleSheet(
            "padding: 6px; background-color: #FFF8E1; color: #333333; "
            "border: 1px solid #FFE082; border-radius: 3px;"
        )
        self.flag_info_label.setMaximumHeight(60)
        left_layout.addWidget(self.flag_info_label)

        main_splitter.addWidget(left_widget)

        # ---- Right panel: zoom buttons + PDF preview ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Zoom toggle
        zoom_layout = QHBoxLayout()
        self.fit_btn = QPushButton("Fit Page")
        self.fit_btn.setCheckable(True)
        self.fit_btn.setChecked(True)
        self.fit_btn.clicked.connect(lambda: self._toggle_zoom("fit"))
        zoom_layout.addWidget(self.fit_btn)

        self.zoom_btn = QPushButton("Zoom In")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.clicked.connect(lambda: self._toggle_zoom("scroll"))
        zoom_layout.addWidget(self.zoom_btn)

        zoom_layout.addStretch()
        right_layout.addLayout(zoom_layout)

        # PDF preview in scroll area
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: #f0f0f0; }"
        )

        self.scan_preview = PDFPreview(width=500, height=650, scale_to_fit=True)
        self.preview_scroll.setWidget(self.scan_preview)
        right_layout.addWidget(self.preview_scroll, 1)

        main_splitter.addWidget(right_widget)

        # Splitter proportions
        main_splitter.setSizes([550, 450])

        # Status bar
        self.status_label = QLabel("No results loaded. Click 'Load Results' to begin.")
        layout.addWidget(self.status_label)

    # =====================================================================
    # Column helpers
    # =====================================================================

    def _build_column_list(self) -> List[str]:
        """
        Build ordered column list from the CSV headers.

        Returns columns in a sensible order:
        Page, Version, LastName, FirstName, StudentID,
        Correct, Incorrect, Blank, Multi,
        Flagged, FlagDetails (hidden),
        Q1, Q2, ... Qn
        """
        if not self._scored_data:
            return []
        first_row = self._scored_data[0]

        ordered: List[str] = []
        # Fixed identity / metrics columns
        for name in ["Page", "Version", "LastName", "FirstName", "StudentID",
                      "Score", "Correct", "Incorrect", "Blank", "Multi", "Percent"]:
            if name in first_row:
                ordered.append(name)

        # Hidden columns (needed internally for flag filtering)
        for name in ["Flagged", "FlagDetails"]:
            if name in first_row:
                ordered.append(name)

        # Question columns in numeric order
        q_cols = sorted(
            [k for k in first_row.keys()
             if k and k.startswith("Q") and k[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        ordered.extend(q_cols)

        return ordered

    @staticmethod
    def _is_q_column(col_name: str) -> bool:
        return bool(col_name and col_name.startswith("Q") and col_name[1:].isdigit())

    def _discover_answer_choices(self, key_rows: List[Dict[str, str]]):
        """
        Discover the valid answer choices from KEY rows and student data,
        then configure the AnswerChoiceDelegate.

        Scans all Q-column values to find every distinct single-letter answer
        that appears, and sorts them alphabetically.  This means the dropdown
        automatically matches the template: A-E for 5-choice, T/F for
        true-false, A-J for 10-choice, etc.
        """
        found: set = set()

        # Scan KEY rows first (most reliable source of valid choices)
        for row in key_rows:
            for key, val in row.items():
                if self._is_q_column(key) and val:
                    v = val.strip()
                    if v and len(v) == 1 and v.isalpha():
                        found.add(v.upper())

        # Scan student data too (picks up edge cases, e.g. no-key scoring)
        for row in self._scored_data:
            for key, val in row.items():
                if self._is_q_column(key) and val:
                    v = val.strip()
                    if v and len(v) == 1 and v.isalpha():
                        found.add(v.upper())

        if found:
            choices = sorted(found)
        else:
            # Fallback to A-E if we couldn't discover any
            choices = ["A", "B", "C", "D", "E"]

        self._answer_delegate.set_choices(choices)

    # =====================================================================
    # Loading Data
    # =====================================================================

    def _on_load_results(self):
        """Open file dialog to load scored results."""
        start_dir = str(Path.home())
        project_dir = self.project_selector.project_dir()
        working_dir = self.project_selector.working_dir()

        if project_dir:
            score_data_dir = project_dir / "score_data"
            if score_data_dir.exists():
                start_dir = str(score_data_dir)
            else:
                start_dir = str(project_dir)
        elif working_dir:
            start_dir = str(working_dir)

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Scored Results", start_dir, "CSV Files (*.csv)"
        )
        if file_path:
            self._load_scored_csv(Path(file_path))

    def _load_scored_csv(self, csv_path: Path):
        """Load scored results from CSV file."""
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                lines = f.readlines()

            # Filter out comments, empty lines, and version-marker rows
            data_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("==="):
                    continue
                data_lines.append(line)

            if not data_lines:
                QMessageBox.warning(self, "Error", "No data found in CSV file.")
                return

            reader = csv.DictReader(data_lines)

            # Separate KEY rows from student rows
            self._scored_data = []
            key_rows: List[Dict[str, str]] = []
            for row in reader:
                page_val = row.get("Page", row.get("page", ""))
                version_val = row.get("Version", row.get("version", ""))
                student_id = row.get("StudentID", row.get("student_id", row.get("ID", "")))

                is_key = (
                    (page_val and page_val.upper() == "KEY")
                    or (version_val and version_val.upper() == "KEY")
                    or (student_id and student_id.upper() == "KEY")
                )
                if is_key:
                    key_rows.append(row)
                    continue

                if not student_id:
                    continue
                if student_id.lower() in (
                    "mean", "stdev", "high", "low", "n", "kr-20", "item stats",
                ):
                    continue
                self._scored_data.append(row)

            if not self._scored_data:
                QMessageBox.warning(self, "Error", "No student data rows found in CSV file.")
                return

            # Discover valid answer choices from KEY rows + student data
            self._discover_answer_choices(key_rows)

            # Corrections file — always "corrections.csv" next to the scored CSV
            corrections_path = csv_path.parent / "corrections.csv"
            self._correction_log = CorrectionLog(corrections_path, str(csv_path))

            # Find associated annotated PDF
            # In flat structure: scored_scans.pdf lives at project root (parent of score_data/)
            self._scored_pdf_path = None
            project_root = csv_path.parent.parent if csv_path.parent.name == "score_data" else csv_path.parent
            for candidate in [
                project_root / "scored_scans.pdf",
                csv_path.parent / "scored_scans.pdf",
            ]:
                if candidate.exists():
                    self._scored_pdf_path = candidate
                    break

            # Update results combo
            csv_path_str = str(csv_path)
            existing_idx = -1
            for i in range(self.results_combo.count()):
                if self.results_combo.itemData(i) == csv_path_str:
                    existing_idx = i
                    break

            self.results_combo.blockSignals(True)
            if existing_idx < 0:
                self.results_combo.clear()
                label = self._results_display_label(csv_path)
                self.results_combo.addItem(label, csv_path_str)
            else:
                self.results_combo.setCurrentIndex(existing_idx)
            self.results_combo.blockSignals(False)

            # Populate spreadsheet
            self._populate_spreadsheet()

            # Enable buttons
            self.export_btn.setEnabled(True)
            self.view_log_btn.setEnabled(True)
            self._update_status()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to load CSV: {e}")

    # =====================================================================
    # Spreadsheet Population
    # =====================================================================

    def _populate_spreadsheet(self):
        """Fill the spreadsheet with CSV data, applying corrections and colours."""
        self._programmatic_update = True

        # Disable sorting while populating (avoids partial-insert sort bugs)
        self.spreadsheet.setSortingEnabled(False)
        self.spreadsheet.setRowCount(0)
        self.spreadsheet.setColumnCount(0)

        data = self._scored_data
        if self._show_flagged_only:
            data = [r for r in self._scored_data if _get_field(r, "Flagged", "flagged")]

        if not data:
            self._columns = []
            self._programmatic_update = False
            self._update_status()
            return

        # Build columns
        self._columns = self._build_column_list()
        self.spreadsheet.setColumnCount(len(self._columns))
        self.spreadsheet.setHorizontalHeaderLabels(self._columns)

        # Install answer-choice delegate on Q columns
        for col_idx, col_name in enumerate(self._columns):
            if self._is_q_column(col_name):
                self.spreadsheet.setItemDelegateForColumn(col_idx, self._answer_delegate)

        # Column widths
        for col_idx, col_name in enumerate(self._columns):
            if self._is_q_column(col_name):
                self.spreadsheet.setColumnWidth(col_idx, 40)
            elif col_name in ("Page", "Version"):
                self.spreadsheet.setColumnWidth(col_idx, 50)
            elif col_name in ("Score", "Correct", "Incorrect", "Blank", "Multi", "Percent"):
                self.spreadsheet.setColumnWidth(col_idx, 55)
            elif col_name in ("StudentID",):
                self.spreadsheet.setColumnWidth(col_idx, 80)
            elif col_name in ("LastName", "FirstName"):
                self.spreadsheet.setColumnWidth(col_idx, 100)

        # Hide internal columns
        for col_idx, col_name in enumerate(self._columns):
            if col_name in _HIDDEN_COLUMNS:
                self.spreadsheet.setColumnHidden(col_idx, True)

        # Get effective corrections
        effective = (
            self._correction_log.get_effective_corrections()
            if self._correction_log is not None else {}
        )

        # Populate rows
        self.spreadsheet.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            student_id = _get_field(row_data, "StudentID", "student_id", "ID")
            student_corrections = effective.get(student_id, {})

            for col_idx, col_name in enumerate(self._columns):
                original_value = row_data.get(col_name, "") or ""
                display_value = student_corrections.get(col_name, original_value)

                # Use sort-aware items for columns that need special ordering
                if col_name in _NUMERIC_COLUMNS:
                    item = _NumericTableItem(display_value)
                elif col_name in _SMART_SORT_COLUMNS:
                    item = _SmartSortItem(display_value)
                else:
                    item = QTableWidgetItem(display_value)

                # Store full row_data on the first column for later retrieval
                if col_idx == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row_data)

                # Cell colouring
                if col_name in student_corrections:
                    item.setBackground(QBrush(_COLOR_CORRECTED))
                elif self._is_q_column(col_name):
                    if not display_value.strip():
                        item.setBackground(QBrush(_COLOR_BLANK_Q))
                    elif "," in display_value:
                        item.setBackground(QBrush(_COLOR_MULTI_Q))

                # Read-only?
                if col_name in _READ_ONLY_COLUMNS:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                self.spreadsheet.setItem(row_idx, col_idx, item)

        # Enable sorting now that all items are in place
        self.spreadsheet.setSortingEnabled(True)
        self._programmatic_update = False
        self._update_status()

    # =====================================================================
    # Cell Editing → CorrectionLog
    # =====================================================================

    def _on_cell_changed(self, row: int, col: int):
        """Handle an inline cell edit and route to CorrectionLog."""
        if self._programmatic_update:
            return
        if self._correction_log is None:
            return
        if col < 0 or col >= len(self._columns):
            return

        col_name = self._columns[col]
        cell_item = self.spreadsheet.item(row, col)
        if cell_item is None:
            return
        new_value = cell_item.text()

        # Debounce: Qt can fire cellChanged twice when a delegate emits
        # commitData + closeEditor in quick succession.  Skip if this is
        # the exact same (row, col, value) as the last correction we wrote.
        correction_key = f"{row}:{col}:{new_value}"
        if correction_key == self._last_correction_key:
            return
        self._last_correction_key = correction_key

        # Retrieve original row_data from the first cell
        first_item = self.spreadsheet.item(row, 0)
        if not first_item:
            return
        row_data = first_item.data(Qt.ItemDataRole.UserRole)
        if not row_data:
            return

        student_id = _get_field(row_data, "StudentID", "student_id", "ID")
        original_value = row_data.get(col_name, "") or ""

        if self._is_q_column(col_name):
            # Answer correction
            if new_value == original_value:
                # Value matches original — revert any existing correction
                if self._correction_log.has_correction(student_id, col_name):
                    self._correction_log.revert(student_id, col_name, "Reverted via GUI")
            else:
                self._correction_log.add_answer_correction(
                    student_id=student_id,
                    question=col_name,
                    original=original_value,
                    corrected=new_value,
                    reason="Manual correction via GUI",
                )
        elif col_name == "StudentID":
            if new_value != original_value:
                self._correction_log.add_student_id_correction(
                    original_value, new_value, "Manual correction via GUI",
                )
        elif col_name in ("LastName", "FirstName"):
            if new_value != original_value:
                self._correction_log.add_answer_correction(
                    student_id=student_id,
                    question=col_name,
                    original=original_value,
                    corrected=new_value,
                    reason="Manual correction via GUI",
                )

        # Update cell colour and sync
        self._update_cell_style(row, col, new_value, original_value, col_name, student_id)
        self._sync_corrections_to_logs()
        self._update_status()

    def _update_cell_style(self, row: int, col: int, value: str,
                           original: str, col_name: str, student_id: str):
        """Refresh a single cell's background after an edit."""
        item = self.spreadsheet.item(row, col)
        if item is None:
            return

        has_correction = (
            self._correction_log is not None
            and self._correction_log.has_correction(student_id, col_name)
        )

        if has_correction:
            item.setBackground(QBrush(_COLOR_CORRECTED))
        elif self._is_q_column(col_name):
            if not value.strip():
                item.setBackground(QBrush(_COLOR_BLANK_Q))
            elif "," in value:
                item.setBackground(QBrush(_COLOR_MULTI_Q))
            else:
                item.setBackground(QBrush(QColor("white")))
        else:
            item.setBackground(QBrush(QColor("white")))

    def _sync_corrections_to_logs(self):
        """Copy the corrections log file to the project's logs/ folder.

        In the flat structure, corrections.csv already lives in score_data/
        which is the canonical location. We still sync a backup to logs/.
        """
        if self._correction_log is None or not self._correction_log.path.exists():
            return
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return
        logs_dir = project_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        dst = logs_dir / "corrections_backup.csv"
        try:
            shutil.copy2(str(self._correction_log.path), str(dst))
        except Exception as e:
            print(f"[warn] Could not sync corrections to logs/: {e}")

    # =====================================================================
    # Context menu (right-click → Revert)
    # =====================================================================

    def _on_context_menu(self, pos):
        """Show context menu with Revert option on corrected cells."""
        item = self.spreadsheet.itemAt(pos)
        if item is None or self._correction_log is None:
            return

        row = item.row()
        col = item.column()
        if col >= len(self._columns):
            return

        col_name = self._columns[col]
        first_item = self.spreadsheet.item(row, 0)
        if not first_item:
            return
        row_data = first_item.data(Qt.ItemDataRole.UserRole)
        if not row_data:
            return

        student_id = _get_field(row_data, "StudentID", "student_id", "ID")

        if not self._correction_log.has_correction(student_id, col_name):
            return

        menu = QMenu(self)
        revert_action = menu.addAction("Revert to Original")
        action = menu.exec(self.spreadsheet.viewport().mapToGlobal(pos))

        if action == revert_action:
            self._correction_log.revert(student_id, col_name, "Reverted via GUI")
            # Refresh the cell to show original value
            original_value = row_data.get(col_name, "") or ""
            self._programmatic_update = True
            item.setText(original_value)
            self._programmatic_update = False
            self._update_cell_style(row, col, original_value, original_value, col_name, student_id)
            self._sync_corrections_to_logs()
            self._update_status()

    # =====================================================================
    # Row Selection → PDF Preview + Flag Info
    # =====================================================================

    def _on_row_selected(self, row: int, col: int, prev_row: int, prev_col: int):
        """Handle spreadsheet row selection: update PDF preview and flag info."""
        if row < 0:
            return

        first_item = self.spreadsheet.item(row, 0)
        if not first_item:
            return
        row_data = first_item.data(Qt.ItemDataRole.UserRole)
        if not row_data:
            return

        self._current_student_idx = row

        # Update flag info
        flag_details = _get_field(row_data, "FlagDetails", "flagdetails")
        self.flag_info_label.setText(self._parse_flag_details(flag_details))

        # Update PDF preview
        self._load_pdf_page(row_data)

    @staticmethod
    def _parse_flag_details(flag_details: str) -> str:
        """
        Convert coded flag string to human-readable text.

        "Q5:blank|Q10:multi|ID:orphan" → "Flags: blank at Q5, multi at Q10, orphan ID"
        """
        if not flag_details:
            return "No flags for this student."

        parts = flag_details.split("|")
        readable = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                field, flag_type = part.split(":", 1)
                if field.startswith("Q"):
                    readable.append(f"{flag_type} at {field}")
                elif field == "ID":
                    readable.append("orphan ID")
                else:
                    readable.append(part)
            else:
                readable.append(part)

        if not readable:
            return "No flags for this student."
        return "Flags: " + ", ".join(readable)

    # =====================================================================
    # PDF Preview
    # =====================================================================

    def _load_pdf_page(self, row_data: Dict[str, str]):
        """Load the PDF page(s) for the student into the preview."""
        if not self._scored_pdf_path or not self._scored_pdf_path.exists():
            self.scan_preview.clear()
            return

        page_str = _get_field(row_data, "Page", "page")
        if not page_str:
            self.scan_preview.clear()
            return

        try:
            if "-" in page_str:
                # Multi-page: "1-2" → pages [0, 1]
                parts = page_str.split("-")
                start = int(parts[0]) - 1
                end = int(parts[1])  # end is inclusive in the CSV, range() is exclusive
                pages = list(range(start, end))
                self.scan_preview.load_pdf_pages(
                    self._scored_pdf_path, pages, dpi=self._current_dpi
                )
            else:
                page_num = int(page_str)
                self.scan_preview.load_pdf(
                    self._scored_pdf_path, page=page_num - 1, dpi=self._current_dpi
                )
        except (ValueError, Exception) as e:
            print(f"[warn] Failed to load PDF page '{page_str}': {e}")
            self.scan_preview.clear()

    def _toggle_zoom(self, mode: str):
        """Switch between fit-to-page and zoomed-in scrollable modes."""
        self._zoom_mode = mode
        self.fit_btn.setChecked(mode == "fit")
        self.zoom_btn.setChecked(mode == "scroll")

        if mode == "fit":
            self._current_dpi = 72
            self.scan_preview.set_scale_to_fit(True)
        else:
            self._current_dpi = 150
            self.scan_preview.set_scale_to_fit(False)

        # Reload current page at new DPI
        if self._current_student_idx >= 0:
            first_item = self.spreadsheet.item(self._current_student_idx, 0)
            if first_item:
                row_data = first_item.data(Qt.ItemDataRole.UserRole)
                if row_data:
                    # Invalidate cache so it re-renders
                    self.scan_preview._cached_pixmap = None
                    self._load_pdf_page(row_data)

    # =====================================================================
    # Results Selection
    # =====================================================================

    def _on_results_changed(self, label: str):
        """Handle results file selection change from the combo box."""
        if not label:
            return

        # Check stored file path first
        idx = self.results_combo.currentIndex()
        if idx >= 0:
            stored_path = self.results_combo.itemData(idx)
            if stored_path:
                csv_path = Path(stored_path)
                if csv_path.exists():
                    self._load_scored_csv(csv_path)
                    return

        # Fallback: search by name in project structure
        project_dir = self.project_selector.project_dir()
        working_dir = self.project_selector.working_dir()

        search_dirs: List[Path] = []
        if project_dir:
            search_dirs += [project_dir / "score_data", project_dir]
        if working_dir:
            search_dirs += [working_dir]

        for dir_path in search_dirs:
            if not dir_path.exists():
                continue
            csv_path = dir_path / f"{label}.csv"
            if csv_path.exists():
                self._load_scored_csv(csv_path)
                return

    # =====================================================================
    # Filter
    # =====================================================================

    def _set_filter(self, flagged_only: bool):
        """Set the student filter and repopulate."""
        self._show_flagged_only = flagged_only
        self.show_all_btn.setChecked(not flagged_only)
        self.show_flagged_btn.setChecked(flagged_only)
        self._populate_spreadsheet()

    # =====================================================================
    # Export & Log
    # =====================================================================

    def _on_export(self):
        """Export final grades with corrections applied."""
        if not self._scored_data:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Final Grades",
            str(Path.home() / "final_grades.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            if self._correction_log is not None:
                final_data = self._correction_log.apply_to_data(
                    self._scored_data, student_id_field="StudentID"
                )
            else:
                final_data = self._scored_data

            if final_data:
                fieldnames = list(final_data[0].keys())
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    f.write("# Final grades with corrections applied\n")
                    f.write(f"# Exported: {Path(file_path).name}\n")
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(final_data)

                self.status_label.setText(f"Exported to: {file_path}")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export: {e}")

    def _on_view_log(self):
        """View the corrections log in a dialog."""
        if self._correction_log is None:
            return

        log_text = [
            f"Corrections Log: {self._correction_log.path.name}",
            f"Total entries: {len(self._correction_log)}",
            f"Effective corrections: {self._correction_log.effective_count()}",
            "",
            "Recent corrections:",
            "-" * 50,
        ]

        for correction in list(self._correction_log)[-10:]:
            log_text.append(
                f"{correction.timestamp[:19]} | {correction.student_id} | "
                f"{correction.field} | {correction.original_value} → {correction.corrected_value}"
            )

        QMessageBox.information(self, "Corrections Log", "\n".join(log_text))

    # =====================================================================
    # Status
    # =====================================================================

    def _update_status(self):
        """Update the status bar."""
        parts = []
        if self._scored_data:
            total = len(self._scored_data)
            flagged = len([r for r in self._scored_data if _get_field(r, "Flagged", "flagged")])
            parts.append(f"{total} students")
            parts.append(f"{flagged} flagged")

        if self._correction_log is not None:
            corrections = self._correction_log.effective_count()
            if corrections > 0:
                parts.append(f"{corrections} corrections")

        self.status_label.setText(" | ".join(parts) if parts else "No results loaded.")

    # =====================================================================
    # External Interface
    # =====================================================================

    def load_results(self, results_data: dict):
        """Load results data from Quick Grade."""
        if "results_csv" in results_data:
            csv_path = Path(results_data["results_csv"])
            if csv_path.exists():
                self._load_scored_csv(csv_path)

    def _on_project_changed(self, project_name: str):
        """Handle project selection change — auto-load results if available."""
        project_dir = self.project_selector.project_dir()
        if project_dir:
            self._scan_for_scored_files(project_dir)
        else:
            # Project has no directory (or no results yet) — clear stale state
            self._clear_results()

    def _on_directory_changed(self, directory: Path):
        """Handle working directory change from project selector."""
        if not directory:
            return
        dir_path = Path(directory) if isinstance(directory, str) else directory
        self._scan_for_scored_files(dir_path)

    # =====================================================================
    # Results Discovery
    # =====================================================================

    @staticmethod
    def _results_display_label(csv_path: Path, base_dir: Path = None) -> str:
        """
        Build a human-friendly label for a scored CSV file.

        For files in score_data/, uses "results" (the canonical location).
        Otherwise uses the filename stem.
        """
        parent = csv_path.parent
        if parent.name == "score_data":
            return "results"
        return csv_path.stem

    def _scan_for_scored_files(self, dir_path: Path):
        """Scan a project directory for scored CSV files and auto-load results."""
        scored_files: List[Path] = []

        def find_result_csvs(folder: Path):
            found = []
            if not folder.exists():
                return found
            for f in folder.glob("*.csv"):
                name_lower = f.name.lower()
                if (name_lower.startswith("scored")
                        or name_lower.startswith("results")
                        or "_results" in name_lower
                        or "_scored" in name_lower):
                    if not name_lower.startswith("corrections"):
                        found.append(f)
            return found

        # Flat structure: check score_data/ first
        score_data = dir_path / "score_data"
        scored_files.extend(find_result_csvs(score_data))

        # Also check project root and direct CSVs
        scored_files.extend(find_result_csvs(dir_path))

        scored_files = sorted(set(scored_files))

        if scored_files:
            self.results_combo.blockSignals(True)
            self.results_combo.clear()
            for f in scored_files:
                label = self._results_display_label(f, dir_path)
                self.results_combo.addItem(label, str(f))
            self.results_combo.blockSignals(False)

            # Auto-load the results file (prefer score_data/results.csv)
            self._load_scored_csv(scored_files[-1])
            self.status_label.setText(f"Loaded: {scored_files[-1].name}")
        else:
            self._clear_results()
            self.status_label.setText(f"No scored results found in {dir_path}")

    def _clear_results(self):
        """Clear loaded results and reset UI when switching to a project with no results."""
        self.results_combo.clear()
        self._scored_data = None
        self.spreadsheet.setRowCount(0)
        self.spreadsheet.setColumnCount(0)
        self.export_btn.setEnabled(False)
        self.view_log_btn.setEnabled(False)
