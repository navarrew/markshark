"""
Review & Correct page - review grading results and apply corrections.

Features:
- Full spreadsheet view of scored CSV data (all columns)
- Inline editing: free-text entry for Q cells (supports multi-answer like A,C),
  text edit for name/ID cells.  Warns on unexpected characters.
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
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QScrollArea,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QMenu,
    QFrame,
)

from ..widgets import PageHeader, PDFPreview, ProjectSelector, FlagInfoPanel
from ..models import CorrectionLog
from ..utils import RUN_BUTTON_STYLE

# Lazy-loaded roster helpers (avoid import errors if score_core unavailable)
try:
    from markshark.score_core import load_roster, suggest_matches
except ImportError:
    load_roster = None
    suggest_matches = None


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
_COLOR_CORRECTED = QColor("#42A5F5")  # medium-bright blue — has a correction
_COLOR_BLANK_Q = QColor("#F9C8F5")   # pink — blank answer
_COLOR_MULTI_Q = QColor("#FFB366")   # orange — multi-mark
_COLOR_ORPHAN_ID = QColor("#FFCDD2") # light red — orphan ID warning

# Custom data role: stores the flag QColor on flagged cells so the selection
# delegate can preserve it instead of painting solid blue.
_ROLE_FLAG_COLOR = Qt.ItemDataRole.UserRole + 10

# Selection highlight for non-flagged cells in a selected row
_COLOR_ROW_SELECTED = QColor("#90CAF9")  # medium blue (Material Blue 200)


def _flag_cell(item: QTableWidgetItem, color: QColor):
    """Set background AND store flag role so selection delegate can see it."""
    item.setBackground(QBrush(color))
    item.setData(_ROLE_FLAG_COLOR, color)


def _unflag_cell(item: QTableWidgetItem):
    """Clear flag colour — restore to plain white."""
    item.setBackground(QBrush(QColor("white")))
    item.setData(_ROLE_FLAG_COLOR, None)


# ---------------------------------------------------------------------------
# Delegate: text editor for Q cells
# ---------------------------------------------------------------------------

_COLOR_WARNING_Q = QColor("#FFF9C4")  # light yellow — unexpected input


class _FlagPreservingDelegate(QStyledItemDelegate):
    """Delegate that keeps flagged-cell backgrounds visible when the row
    is selected, instead of painting them solid blue.

    Non-flagged cells in a selected row get a light blue tint.
    Flagged cells keep their flag colour with a slightly darkened tint
    so the user can still tell the row is selected.
    """

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        flag_color = index.data(_ROLE_FLAG_COLOR)
        sel_flag = QStyle.StateFlag.State_Selected
        if option.state & sel_flag:
            if flag_color and isinstance(flag_color, QColor):
                # Darken the flag colour slightly to hint at selection
                darker = flag_color.darker(115)
                option.backgroundBrush = QBrush(darker)
                # Remove the Selected state so Qt doesn't paint over it
                option.state &= ~sel_flag
            else:
                # Normal (non-flagged) cell in a selected row
                option.backgroundBrush = QBrush(_COLOR_ROW_SELECTED)
                option.state &= ~sel_flag


class AnswerTextDelegate(_FlagPreservingDelegate):
    """
    Item delegate that presents a QLineEdit for answer cells.

    Teachers can type anything: single letters (A), multi-answers (A,C or AC),
    or blank.  Input is normalized on commit (uppercased, sorted, comma-
    separated).  If the entry contains characters not in the known answer
    labels the cell gets a yellow warning background.

    Inherits from _FlagPreservingDelegate so flagged Q-cells keep their
    background colour when the row is selected.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._known_labels: set = set("ABCDE")  # fallback

    def set_known_labels(self, labels: List[str]):
        """Set the known answer labels for validation warnings."""
        self._known_labels = set(l.upper() for l in labels if l)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setMaxLength(20)
        return editor

    def setEditorData(self, editor: QLineEdit, index):
        value = index.data(Qt.ItemDataRole.DisplayRole) or ""
        editor.setText(value)
        editor.selectAll()

    def setModelData(self, editor: QLineEdit, model, index):
        raw = editor.text().strip().upper()
        normalized = self._normalize_answer(raw)
        model.setData(index, normalized, Qt.ItemDataRole.EditRole)

    @staticmethod
    def _normalize_answer(raw: str) -> str:
        """Normalize free-text answer input.

        - Strips whitespace, uppercases
        - 'AC' or 'A,C' or 'A, C' all become 'A,C'
        - Single letter stays as-is
        - Blank stays blank
        """
        if not raw:
            return ""
        # Split on commas first, then split remaining multi-char tokens
        # into individual characters
        parts = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if len(chunk) <= 1:
                if chunk:
                    parts.append(chunk)
            else:
                # "AC" -> ["A", "C"]
                parts.extend(list(chunk))
        # Deduplicate and sort
        unique = sorted(set(parts))
        return ",".join(unique)


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

        self._answer_delegate = AnswerTextDelegate(self)
        self._known_answer_labels: set = set("ABCDE")  # updated after CSV load
        self._last_correction_key: str = ""  # debounce: "row:col:value"

        # Compound-key lookup: {version: {Q-col-name: key_value}}
        # Used to suppress orange multi-mark colouring for expected compound answers.
        self._compound_keys: Dict[str, Dict[str, str]] = {}

        # Roster for orphan ID resolution
        self._roster: Optional[Dict[str, Dict[str, str]]] = None
        self._absent_roster: Optional[Dict[str, Dict[str, str]]] = None

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

        # Navigate back to the Grader page
        self.return_grader_btn = QPushButton("← Return to Grader")
        self.return_grader_btn.setFixedWidth(160)
        self.return_grader_btn.setStyleSheet(
            "QPushButton { color: #0E817E; border: 1px solid #0E817E; "
            "border-radius: 4px; padding: 4px 8px; background: transparent; }"
            "QPushButton:hover { background: #e0f2f1; }"
        )
        self.return_grader_btn.clicked.connect(self._on_return_to_grader)
        filter_layout.addWidget(self.return_grader_btn)

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

        # Re-annotate button — regenerates annotated PDF + CSV with corrections applied
        self.reannotate_btn = QPushButton("Apply Corrections && Re-annotate")
        self.reannotate_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.reannotate_btn.setFixedWidth(280)
        self.reannotate_btn.setEnabled(False)
        self.reannotate_btn.clicked.connect(self._on_reannotate)
        filter_layout.addWidget(self.reannotate_btn)

        # Clear corrections button — removes all pending corrections
        self.clear_corrections_btn = QPushButton("Clear Corrections")
        self.clear_corrections_btn.setFixedWidth(140)
        self.clear_corrections_btn.setEnabled(False)
        self.clear_corrections_btn.clicked.connect(self._on_clear_corrections)
        filter_layout.addWidget(self.clear_corrections_btn)

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
        self._flag_delegate = _FlagPreservingDelegate(self.spreadsheet)
        self.spreadsheet.setItemDelegate(self._flag_delegate)
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

        # Flag info / orphan suggestions panel
        self.flag_panel = FlagInfoPanel()
        self.flag_panel.suggestion_accepted.connect(self._accept_suggestion)
        self.flag_panel.roster_requested.connect(self._on_roster_requested)
        self.flag_panel.undo_correction.connect(self._undo_id_correction)
        left_layout.addWidget(self.flag_panel)

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
        Discover the valid answer labels from KEY rows and student data,
        then configure the AnswerTextDelegate for validation warnings.

        Scans all Q-column values to find every distinct single-letter answer
        that appears.  These are stored so the delegate can warn (yellow
        background) if a teacher types something unexpected.
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
            labels = sorted(found)
        else:
            # Fallback to A-E if we couldn't discover any
            labels = ["A", "B", "C", "D", "E"]

        self._known_answer_labels = set(labels)
        self._answer_delegate.set_known_labels(labels)

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

                # Skip VALUE rows (per-question point values, one per version)
                is_value = (
                    (page_val and page_val.upper() == "VALUE")
                    or (version_val and version_val.upper() == "VALUE")
                    or (student_id and student_id.upper() == "VALUE")
                )
                if is_value:
                    continue

                # Skip stats/summary rows
                if student_id and student_id.lower() in (
                    "mean", "stdev", "high", "low", "n", "kr-20", "item stats",
                ):
                    continue
                # Accept the row if it has a StudentID OR a Page number
                # (mock data / un-rostered scans may lack StudentID)
                if not student_id and not page_val:
                    continue
                self._scored_data.append(row)

            if not self._scored_data:
                QMessageBox.warning(self, "Error", "No student data rows found in CSV file.")
                return

            # Discover valid answer choices from KEY rows + student data
            self._discover_answer_choices(key_rows)

            # Build compound-key lookup so cell colouring can suppress
            # orange multi-mark warnings for AND / partial keys.
            self._compound_keys = {}
            for kr in key_rows:
                ver = kr.get("Version", kr.get("version", "A")).strip().upper()
                keys_for_ver: Dict[str, str] = {}
                for col_name, val in kr.items():
                    if self._is_q_column(col_name) and val:
                        v = val.strip()
                        if any(op in v for op in ("&", "@", "~")):
                            keys_for_ver[col_name] = v
                self._compound_keys[ver] = keys_for_ver

            # Corrections file — always "corrections.csv" next to the scored CSV
            corrections_path = csv_path.parent / "corrections.csv"
            self._correction_log = CorrectionLog(corrections_path, str(csv_path))

            # Detect Simple Grade mode for this project — corrections are
            # keyed by page number instead of student ID in simple mode.
            self._simple_mode = False
            project_dir = self.project_selector.project_dir()
            if project_dir:
                from ..models.project_registry import ProjectRegistry
                self._simple_mode = ProjectRegistry().get_simple_mode(project_dir)

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
            self._update_reannotate_enabled()
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
                self.spreadsheet.setColumnWidth(col_idx, 100)
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

            # Orphan detection for visual indicator
            flag_details = _get_field(row_data, "FlagDetails", "flagdetails")
            is_orphan = "ID:orphan" in (flag_details or "")
            orphan_corrected = (
                is_orphan
                and "student_id" in student_corrections
            )

            for col_idx, col_name in enumerate(self._columns):
                original_value = row_data.get(col_name, "") or ""
                # CorrectionLog stores ID corrections under "student_id"
                # but the CSV column is "StudentID" — check both keys.
                correction_key = "student_id" if col_name == "StudentID" else col_name
                display_value = student_corrections.get(correction_key, original_value)

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
                if correction_key in student_corrections:
                    _flag_cell(item, _COLOR_CORRECTED)
                elif col_name == "StudentID" and is_orphan and not orphan_corrected:
                    _flag_cell(item, _COLOR_ORPHAN_ID)
                    item.setForeground(QBrush(QColor("#B71C1C")))  # dark red text
                    item.setToolTip(
                        "Orphan ID \u2014 not found in roster. "
                        "Select row to see suggested matches."
                    )
                elif col_name == "Version" and display_value.strip().endswith("*"):
                    _flag_cell(item, _COLOR_MULTI_Q)  # orange
                    item.setToolTip(
                        "Version bubble was blank \u2014 auto-detected "
                        "from best key match"
                    )
                elif self._is_q_column(col_name):
                    if not display_value.strip():
                        _flag_cell(item, _COLOR_BLANK_Q)
                    elif "," in display_value:
                        # Only flag as multi if this column does NOT expect
                        # a compound answer (AND/partial key).
                        # Strip trailing * from auto-detected versions (e.g. "A*" → "A")
                        # so the lookup matches the KEY row's clean version letter.
                        student_ver = row_data.get(
                            "Version", row_data.get("version", "")
                        ).strip().upper().rstrip("*")
                        compound_cols = self._compound_keys.get(student_ver, {})
                        if col_name not in compound_cols:
                            _flag_cell(item, _COLOR_MULTI_Q)

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

        # In Simple Grade mode, corrections are keyed by page number (always
        # unique and stable) instead of student ID (which may be blank).
        if getattr(self, "_simple_mode", False):
            student_id = _get_field(row_data, "Page", "page")
        else:
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

        correction_key = "student_id" if col_name == "StudentID" else col_name
        has_correction = (
            self._correction_log is not None
            and self._correction_log.has_correction(student_id, correction_key)
        )

        if has_correction:
            _flag_cell(item, _COLOR_CORRECTED)
            item.setForeground(QBrush(QColor("black")))
            # Clear orphan tooltip if this was a corrected orphan
            if col_name == "StudentID":
                item.setToolTip("")
        elif self._is_q_column(col_name):
            # Check for unexpected characters (validation warning)
            known = getattr(self, "_known_answer_labels", set("ABCDE"))
            entered_chars = set(c for c in value.upper().replace(",", "") if c.strip())
            unexpected = entered_chars - known
            if unexpected and value.strip():
                _flag_cell(item, _COLOR_WARNING_Q)
                item.setToolTip(
                    f"Unexpected: {', '.join(sorted(unexpected))}  "
                    f"(expected {', '.join(sorted(known))})"
                )
            elif not value.strip():
                _flag_cell(item, _COLOR_BLANK_Q)
                item.setToolTip("")
            elif "," in value:
                # Check if compound answer expected for this student's version.
                # Strip trailing * from auto-detected versions (e.g. "A*" → "A")
                # so the lookup matches the KEY row's clean version letter.
                version = ""
                first_item = self.spreadsheet.item(row, 0)
                if first_item:
                    rd = first_item.data(Qt.ItemDataRole.UserRole)
                    if rd:
                        version = rd.get(
                            "Version", rd.get("version", "")
                        ).strip().upper().rstrip("*")
                compound_cols = self._compound_keys.get(version, {})
                if col_name not in compound_cols:
                    _flag_cell(item, _COLOR_MULTI_Q)
                    item.setToolTip("")
                else:
                    _unflag_cell(item)
                    item.setToolTip("")
            else:
                _unflag_cell(item)
                item.setToolTip("")
        else:
            _unflag_cell(item)

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

        # Enable Re-annotate button when corrections exist
        self._update_reannotate_enabled()

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

        # Update flag info / orphan suggestions
        flag_details = _get_field(row_data, "FlagDetails", "flagdetails")
        if "ID:orphan" in (flag_details or ""):
            # Check if the orphan ID has already been corrected
            student_id = _get_field(row_data, "StudentID", "student_id", "ID") or ""
            orphan_corrected = (
                self._correction_log is not None
                and self._correction_log.has_correction(student_id, "student_id")
            )
            if orphan_corrected:
                # Look up what the ID was corrected to, so the teacher can
                # see the mapping and undo it if needed.
                effective = self._correction_log.get_effective_corrections()
                corrected_id = effective.get(student_id, {}).get("student_id", "")
                self.flag_panel.show_corrected(
                    "CORRECTED",
                    student_id=student_id,
                    original_id=student_id,
                    corrected_id=corrected_id,
                )
            else:
                self._show_orphan_suggestions(row_data)
        else:
            self.flag_panel.show_regular_flags(self._parse_flag_details(flag_details))

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
                elif field == "Version":
                    # flag_type is e.g. "blank, inferred B" or just "blank"
                    if "inferred" in flag_type:
                        readable.append(f"version bubble {flag_type}")
                    else:
                        readable.append("version bubble blank (auto-detected)")
                else:
                    readable.append(part)
            else:
                readable.append(part)

        if not readable:
            return "No flags for this student."
        return "Flags: " + ", ".join(readable)

    # =====================================================================
    # Orphan ID Resolution
    # =====================================================================

    def _load_roster_if_needed(self) -> Optional[Dict[str, Dict[str, str]]]:
        """Load roster from the project's input_files/ directory (cached)."""
        if self._roster is not None:
            return self._roster
        if load_roster is None:
            return None  # score_core not available

        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return None

        input_files = project_dir / "input_files"
        if not input_files.exists():
            return None

        candidates = sorted(input_files.glob("roster.*"))
        if not candidates:
            return None

        try:
            self._roster = load_roster(str(candidates[0]))
            self._absent_roster = None  # force recompute
            return self._roster
        except Exception:
            return None

    def _compute_absent_roster(
        self, roster: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Return roster entries whose IDs are not matched by any non-orphan scan."""
        if self._absent_roster is not None:
            return self._absent_roster

        # IDs that are accounted for (non-orphan scans + corrected orphans)
        found_ids: set = set()
        for row in self._scored_data:
            sid = _get_field(row, "StudentID", "student_id", "ID")
            fd = _get_field(row, "FlagDetails", "flagdetails")
            if sid and "ID:orphan" not in fd:
                found_ids.add(sid)

        # Also count corrections that map orphan → roster ID
        if self._correction_log is not None:
            for corrections in self._correction_log.get_effective_corrections().values():
                corrected_id = corrections.get("student_id")
                if corrected_id:
                    found_ids.add(corrected_id)

        self._absent_roster = {
            rid: info for rid, info in roster.items() if rid not in found_ids
        }
        return self._absent_roster

    def _show_orphan_suggestions(self, row_data: Dict[str, str]):
        """Show orphan resolution panel for the selected row."""
        roster = self._load_roster_if_needed()
        if not roster:
            self.flag_panel.show_no_roster()
            return

        if suggest_matches is None:
            self.flag_panel.show_no_roster()
            return

        orphan_id = _get_field(row_data, "StudentID", "student_id", "ID")
        orphan_last = _get_field(row_data, "LastName", "last_name", "Last")
        orphan_first = _get_field(row_data, "FirstName", "first_name", "First")
        orphan_name = f"{orphan_last}, {orphan_first}".strip(", ")

        absent = self._compute_absent_roster(roster)
        suggestions = suggest_matches(
            orphan_id=orphan_id,
            orphan_name=orphan_name,
            absent_roster=absent,
            max_suggestions=3,
        )

        self.flag_panel.show_orphan_suggestions(orphan_id, orphan_name, suggestions)

    def _accept_suggestion(self, original_id: str, suggested_id: str, reason: str):
        """Handle accepting a roster match suggestion for an orphan."""
        if self._correction_log is None:
            QMessageBox.warning(
                self, "No Results Loaded",
                "Cannot save correction — no scored results are loaded.",
            )
            return

        self._correction_log.add_student_id_correction(
            original_id=original_id,
            corrected_id=suggested_id,
            reason=f"Orphan match: {reason}",
        )

        # Remove accepted ID from the absent pool
        if self._absent_roster and suggested_id in self._absent_roster:
            del self._absent_roster[suggested_id]

        self._sync_corrections_to_logs()
        self._populate_spreadsheet()
        self._update_status()

    def _undo_id_correction(self, student_id: str):
        """Revert an orphan ID correction and return to the suggestions view."""
        if self._correction_log is None:
            return

        # Revert the correction in the log (appends a REVERT entry)
        self._correction_log.revert(student_id, "student_id", "Undone via GUI")
        self._sync_corrections_to_logs()

        # Rebuild the spreadsheet so the cell reverts to original styling
        self._populate_spreadsheet()
        self._update_status()

        # Re-select the current row to refresh the flag panel — this will
        # now show the orphan suggestions again since the correction is gone.
        if self._current_student_idx >= 0:
            self.spreadsheet.setCurrentCell(self._current_student_idx, 0)

    def _on_roster_requested(self):
        """Teacher clicked 'Load Roster...' — open a file dialog."""
        if load_roster is None:
            QMessageBox.warning(
                self, "Unavailable",
                "Roster loading is not available (score_core not found).",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Class Roster",
            str(self.project_selector.project_dir() or Path.home()),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                self._roster = load_roster(path)
                self._absent_roster = None  # force recompute
                # Re-trigger suggestions for current row
                row = self.spreadsheet.currentRow()
                if row >= 0:
                    first_item = self.spreadsheet.item(row, 0)
                    if first_item:
                        row_data = first_item.data(Qt.ItemDataRole.UserRole)
                        if row_data:
                            flag_details = _get_field(row_data, "FlagDetails", "flagdetails")
                            if "ID:orphan" in (flag_details or ""):
                                self._show_orphan_suggestions(row_data)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load roster: {e}")

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
    # Re-annotation
    # =====================================================================

    def _update_reannotate_enabled(self):
        """Enable/disable the Re-annotate and Clear Corrections buttons."""
        has_corrections = (
            self._correction_log is not None
            and bool(self._correction_log.get_effective_corrections())
        )
        self.reannotate_btn.setEnabled(has_corrections)
        self.clear_corrections_btn.setEnabled(has_corrections)

    def _on_clear_corrections(self):
        """Delete the corrections log so the teacher starts fresh."""
        if self._correction_log is None:
            return

        effective = self._correction_log.get_effective_corrections()
        n = sum(len(v) for v in effective.values())
        reply = QMessageBox.question(
            self,
            "Clear Corrections",
            f"This will permanently delete {n} correction(s).\n\n"
            "The scored CSV and annotated PDF will NOT be changed — "
            "only the pending corrections are removed.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._correction_log.clear()
        # Invalidate the absent-roster cache so orphan suggestions are
        # recomputed without the now-deleted corrections.
        self._absent_roster = None
        self._populate_spreadsheet()
        self._update_reannotate_enabled()
        self._update_status()
        self.status_label.setText("Corrections cleared.")

    def _on_reannotate(self):
        """Re-run scoring with corrections applied to regenerate annotated PDF."""
        if not self._correction_log or not self._scored_data:
            return

        # Load results_params.json to get original scoring parameters
        csv_path = Path(self.results_combo.currentData())
        params_path = csv_path.with_name(csv_path.stem + "_params.json")
        if not params_path.exists():
            QMessageBox.warning(
                self,
                "Missing Parameters",
                "Cannot find scoring parameters file.\n\n"
                "Re-annotation requires the original scoring parameters "
                f"({csv_path.stem}_params.json) which is created during scoring.\n\n"
                "Try re-scoring the original scans first.",
            )
            return

        import json
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)

        # Determine project root (parent of score_data/) for path discovery
        project_root = (
            csv_path.parent.parent
            if csv_path.parent.name == "score_data"
            else csv_path.parent
        )
        input_files = project_root / "input_files"

        # Resolve aligned PDF path — try params first, fall back to standard
        # project structure.  Older results_params.json files (pre-re-annotation
        # feature) don't store input_path, so the fallback is essential.
        input_path = params.get("input_path")
        if not input_path or not Path(input_path).exists():
            # Standard project structure: <project>/input_files/aligned_scans.pdf
            fallback = input_files / "aligned_scans.pdf"
            if fallback.exists():
                input_path = str(fallback)
            else:
                QMessageBox.warning(
                    self,
                    "Missing Input",
                    "Aligned PDF not found.\n\n"
                    "Looked in:\n"
                    f"  • {input_path or '(not recorded in params)'}\n"
                    f"  • {fallback}\n\n"
                    "The aligned scans file may have been moved or deleted.",
                )
                return

        # Resolve bubblemap path
        bubblemap_path = params.get("template", {}).get("bubblemap_path")
        if not bubblemap_path or not Path(bubblemap_path).exists():
            QMessageBox.warning(
                self,
                "Missing Template",
                f"Bubblemap not found:\n{bubblemap_path}\n\n"
                "The template may have been removed.",
            )
            return

        # Get corrections in the format score_pdf() expects:
        # {student_id: {field: corrected_value, ...}, ...}
        corrections = self._correction_log.get_effective_corrections()

        # Output paths (overwrite existing scored outputs)
        out_pdf = str(project_root / "scored_scans.pdf")
        out_csv = str(csv_path)

        # Key file — try params, fall back to input_files/key.*
        key_txt = params.get("key_txt")
        if key_txt and not Path(key_txt).exists():
            key_txt = None
        if not key_txt and input_files.exists():
            for k in sorted(input_files.glob("key.*")):
                key_txt = str(k)
                break

        # Roster — try params, fall back to input_files/roster.*
        roster_csv = params.get("roster_csv")
        if roster_csv and not Path(roster_csv).exists():
            roster_csv = None
        if not roster_csv and input_files.exists():
            for r in sorted(input_files.glob("roster.*")):
                roster_csv = str(r)
                break

        # Archive the original results CSV before overwriting — only on the
        # first re-annotation so the teacher always has the raw scanner output.
        original_backup = Path(out_csv).with_name("results_original.csv")
        will_archive = Path(out_csv).exists() and not original_backup.exists()

        # Confirm with user before overwriting
        n_corrections = sum(len(v) for v in corrections.values())
        archive_note = (
            f"\nOriginal results will be saved as {original_backup.name}."
            if will_archive else ""
        )
        reply = QMessageBox.question(
            self,
            "Apply Corrections",
            f"This will re-score all sheets with {n_corrections} correction(s) applied "
            f"and overwrite:\n"
            f"  • {Path(out_csv).name}\n"
            f"  • {Path(out_pdf).name}\n"
            f"{archive_note}\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Save original results before overwriting (first time only)
        if will_archive:
            try:
                shutil.copy2(out_csv, str(original_backup))
            except Exception as e:
                print(f"[warn] Could not archive original results: {e}")

        # Run score_pdf() in a worker thread so the GUI stays responsive
        self.reannotate_btn.setEnabled(False)
        self.status_label.setText("Re-annotating with corrections...")

        from ..workers.reannotate_worker import ReAnnotateWorker

        self._reannotate_worker = ReAnnotateWorker(
            input_path=input_path,
            bubblemap_path=bubblemap_path,
            out_csv=out_csv,
            out_pdf=out_pdf,
            key_txt=key_txt,
            roster_csv=roster_csv,
            corrections=corrections,
            params=params,
        )
        self._reannotate_worker.finished.connect(self._on_reannotate_done)
        self._reannotate_worker.error.connect(self._on_reannotate_error)
        self._reannotate_worker.start()

    def _on_reannotate_done(self, out_csv: str):
        """Handle successful re-annotation completion."""
        self.status_label.setText("Re-annotation complete! PDF and CSV updated.")
        # Clear the PDF preview cache so the new annotated pages load fresh
        self.scan_preview.clear()
        # Reload the CSV so the spreadsheet shows corrected scores
        self._load_scored_csv(Path(out_csv))
        self.reannotate_btn.setEnabled(True)

    def _on_reannotate_error(self, error_msg: str):
        """Handle re-annotation failure."""
        self.status_label.setText("Re-annotation failed.")
        QMessageBox.warning(self, "Re-annotation Error", error_msg)
        self.reannotate_btn.setEnabled(True)

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
        """Update the status bar with student, orphan, flagged, and correction counts."""
        parts = []
        if self._scored_data:
            total = len(self._scored_data)
            flagged = 0
            orphans = 0
            for r in self._scored_data:
                if _get_field(r, "Flagged", "flagged"):
                    flagged += 1
                fd = _get_field(r, "FlagDetails", "flagdetails")
                if "ID:orphan" in fd:
                    orphans += 1

            # Subtract corrected orphans (they have an ID correction)
            if orphans and self._correction_log is not None:
                effective = self._correction_log.get_effective_corrections()
                for r in self._scored_data:
                    fd = _get_field(r, "FlagDetails", "flagdetails")
                    if "ID:orphan" in fd:
                        sid = _get_field(r, "StudentID", "student_id", "ID")
                        if sid in effective and "student_id" in effective[sid]:
                            orphans -= 1

            parts.append(f"{total} students")
            if orphans > 0:
                parts.append(f"{orphans} orphans")
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

    def _on_return_to_grader(self):
        """Navigate back to the Grader page via the main window."""
        main_win = self.window()
        if hasattr(main_win, "_navigate_to_key"):
            main_win._navigate_to_key("quick_grade")

    def _on_project_changed(self, project_name: str):
        """Handle project selection change — auto-load results if available."""
        # Clear roster cache so it reloads from the new project
        self._roster = None
        self._absent_roster = None

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
        self.reannotate_btn.setEnabled(False)
        self.clear_corrections_btn.setEnabled(False)
