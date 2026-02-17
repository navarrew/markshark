"""
Answer Key Utility — create, import, edit, and export answer keys.

Workflow:
1. Paste a comma-separated list of answers into the text box, OR
   import an existing key file (.txt, .csv, .xlsx).
2. Click "Add to Key" — a wizard asks for version, test code,
   default points.
3. The table on the right auto-populates.  Repeat for more versions.
4. Edit answers / points in the table as needed.
5. Export to the current assessment or a custom location.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QGroupBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QLineEdit,
    QFrame,
    QScrollArea,
    QStyledItemDelegate,
    QMenu,
)

from ..widgets import PageHeader, ProjectSelector
from ..utils import RUN_BUTTON_STYLE, TEAL, BLUE

# Lazy-imported at use sites:
#   from markshark.key_parser import (
#       load_key_file, parse_answer, answer_spec_to_text,
#       write_key_file, AnswerKeySet, VersionKey, AnswerSpec, ScoringMode,
#   )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_VERSIONS = 8
_VERSION_LETTERS = "ABCDEFGH"

_COLOR_FREEBIE = QColor("#81C784")      # medium green
_COLOR_DISCARD = QColor("#BDBDBD")      # medium gray
_COLOR_PARTIAL = QColor("#FFD54F")      # medium amber
_COLOR_NORMAL = QColor("#E3F2FD")       # soft blue for regular answers
_COLOR_TEXT = QColor("#212121")          # near-black text for all cells
_COLOR_VERSION_BAR = "#B2DFDB"          # medium teal (string for QSS)

# Consistent dark-text style for help panel labels
_HELP_LABEL_STYLE = "color: #212121; font-size: 12px;"


# ============================================================================
# Custom table with keyboard navigation
# ============================================================================

class _KeyTable(QTableWidget):
    """QTableWidget with spreadsheet-style keyboard navigation.

    - Enter / Return -> commit edit and move down one row
    - Tab -> commit edit and move to next column
    - Shift+Tab -> commit edit and move to previous column
    """

    def closeEditor(self, editor, hint):
        from PySide6.QtWidgets import QAbstractItemDelegate

        if hint == QAbstractItemDelegate.EndEditHint.SubmitModelCache:
            super().closeEditor(editor, hint)
            row, col = self.currentRow(), self.currentColumn()
            if row + 1 < self.rowCount():
                self.setCurrentCell(row + 1, col)
                self.editItem(self.currentItem())
            return

        if hint == QAbstractItemDelegate.EndEditHint.EditNextItem:
            super().closeEditor(editor, hint)
            row, col = self.currentRow(), self.currentColumn()
            next_col = col + 1
            if next_col >= self.columnCount():
                next_col, row = 1, row + 1
            if row < self.rowCount():
                self.setCurrentCell(row, next_col)
                self.editItem(self.currentItem())
            return

        if hint == QAbstractItemDelegate.EndEditHint.EditPreviousItem:
            super().closeEditor(editor, hint)
            row, col = self.currentRow(), self.currentColumn()
            prev_col = col - 1
            if prev_col < 1:
                prev_col, row = self.columnCount() - 1, row - 1
            if row >= 0:
                self.setCurrentCell(row, prev_col)
                self.editItem(self.currentItem())
            return

        super().closeEditor(editor, hint)


# ============================================================================
# Delegates
# ============================================================================

class _AnswerCellDelegate(QStyledItemDelegate):
    """Editor for answer cells — accepts letters + key operators."""

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setMaxLength(20)
        editor.installEventFilter(self)
        return editor

    def setEditorData(self, editor, index):
        editor.setText(index.data(Qt.ItemDataRole.EditRole) or "")

    def setModelData(self, editor, model, index):
        model.setData(index, editor.text().strip().upper(), Qt.ItemDataRole.EditRole)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.commitData.emit(obj)
                self.closeEditor.emit(
                    obj, QStyledItemDelegate.EndEditHint.SubmitModelCache
                )
                return True
        return super().eventFilter(obj, event)


class _PointsCellDelegate(QStyledItemDelegate):
    """Editor for point-value cells — numeric with 0.5 step."""

    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        spin.setRange(0.0, 100.0)
        spin.setSingleStep(0.5)
        spin.setDecimals(1)
        spin.installEventFilter(self)
        return spin

    def setEditorData(self, editor, index):
        try:
            editor.setValue(float(index.data(Qt.ItemDataRole.EditRole)))
        except (TypeError, ValueError):
            editor.setValue(1.0)

    def setModelData(self, editor, model, index):
        model.setData(index, str(editor.value()), Qt.ItemDataRole.EditRole)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.commitData.emit(obj)
                self.closeEditor.emit(
                    obj, QStyledItemDelegate.EndEditHint.SubmitModelCache
                )
                return True
        return super().eventFilter(obj, event)


# ============================================================================
# "Add to Key" wizard dialog
# ============================================================================

class _AddVersionDialog(QDialog):
    """Wizard that asks for version letter, test code, default points.

    In *edit mode* the version letter is read-only and pre-filled, and
    the dialog title changes to "Edit Version".
    """

    def __init__(
        self,
        existing_versions: List[str],
        parent=None,
        *,
        edit_version: Optional[str] = None,
        edit_code: str = "",
        edit_points: float = 1.0,
    ):
        super().__init__(parent)
        self._edit_mode = edit_version is not None
        self.setWindowTitle(
            "Edit Version" if self._edit_mode else "Add Version to Key"
        )
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        if self._edit_mode:
            suggested = edit_version
        else:
            # Suggest the next unused version letter
            used = set(v.upper() for v in existing_versions)
            suggested = "A"
            for ch in _VERSION_LETTERS:
                if ch not in used:
                    suggested = ch
                    break

        form = QFormLayout()

        self.version_edit = QLineEdit(suggested)
        self.version_edit.setMaxLength(2)
        self.version_edit.setToolTip("Version letter (A, B, C …)")
        if self._edit_mode:
            self.version_edit.setReadOnly(True)
            # Dark background with white text so the read-only version letter
            # is clearly visible but obviously not editable.
            self.version_edit.setStyleSheet(
                "background: #333333; color: #FFFFFF;"
            )
        form.addRow("Version:", self.version_edit)

        self.code_edit = QLineEdit(edit_code if self._edit_mode else "")
        self.code_edit.setPlaceholderText("(optional)")
        self.code_edit.setToolTip("Machine-readable test code, e.g. 101")
        form.addRow("Test code:", self.code_edit)

        self.pts_spin = QDoubleSpinBox()
        self.pts_spin.setRange(0.1, 100.0)
        self.pts_spin.setSingleStep(0.5)
        self.pts_spin.setDecimals(1)
        self.pts_spin.setValue(edit_points if self._edit_mode else 1.0)
        self.pts_spin.setToolTip("Default points per question")
        form.addRow("Default points:", self.pts_spin)

        layout.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._validate_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _validate_and_accept(self):
        v = self.version_edit.text().strip().upper()
        if not v or not v[0].isalpha():
            self.error_label.setText("Version must start with a letter.")
            self.error_label.show()
            return
        self.accept()

    @property
    def version(self) -> str:
        return self.version_edit.text().strip().upper()

    @property
    def code(self) -> str:
        return self.code_edit.text().strip()

    @property
    def default_points(self) -> float:
        return self.pts_spin.value()


# ============================================================================
# Main page
# ============================================================================

class KeyBuilderPage(QWidget):
    """MarkShark Answer Key Utility — create, edit, and export answer keys."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Tracked state: per-version metadata
        self._versions: List[str] = []
        self._version_meta: Dict[str, dict] = {}   # {ver: {dp, code}}

        self._setup_ui()

        # Wire signals
        self.project_selector.project_changed.connect(self._on_project_changed)
        self.project_selector.working_dir_changed.connect(
            lambda _: self._on_project_changed()
        )

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        root = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark Answer Key Utility",
            "Create and edit your answer keys.",
        )
        root.addWidget(header)

        self.project_selector = ProjectSelector()
        root.addWidget(self.project_selector)

        # -- Action bar --
        action_bar = QHBoxLayout()
        self.save_btn = QPushButton("Save to Assessment")
        self.save_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.save_btn.setMinimumHeight(34)
        self.save_btn.clicked.connect(self._on_export_to_project)
        action_bar.addWidget(self.save_btn)

        self.saveas_btn = QPushButton("Save As\u2026")
        self.saveas_btn.setMinimumHeight(34)
        self.saveas_btn.clicked.connect(self._on_export_as)
        action_bar.addWidget(self.saveas_btn)

        action_bar.addStretch()
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #555;")
        action_bar.addWidget(self.status_label)
        root.addLayout(action_bar)

        # -- Existing-key notification bar --
        self._key_found_bar = QFrame()
        self._key_found_bar.setStyleSheet(
            "QFrame { background: #FFF3CD; border: 1px solid #FFCA2C; "
            "border-radius: 4px; padding: 4px 8px; }"
            " QLabel { color: #664D03; }"
            " QPushButton { color: #664D03; }"
        )
        bar_layout = QHBoxLayout(self._key_found_bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        self._key_found_label = QLabel("")
        bar_layout.addWidget(self._key_found_label, 1)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_detected_key)
        bar_layout.addWidget(load_btn)
        dismiss_btn = QPushButton("\u2715")
        dismiss_btn.setFixedWidth(24)
        dismiss_btn.setFlat(True)
        dismiss_btn.clicked.connect(self._key_found_bar.hide)
        bar_layout.addWidget(dismiss_btn)
        self._key_found_bar.hide()
        root.addWidget(self._key_found_bar)

        # -- Main splitter: left input | centre table | right help --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left panel ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setMinimumWidth(220)
        left_scroll.setMaximumWidth(320)
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)

        # Paste box
        paste_group = QGroupBox("Enter Answers Into Your Key")
        paste_group.setStyleSheet(
            "QGroupBox { background: #FFFFFF; }"
            " QLabel { background: transparent; }"
        )
        paste_layout = QVBoxLayout(paste_group)
        paste_hint = QLabel(
            "Type or paste answers separated by commas, "
            "tabs, spaces, semicolons, or newlines.  "
            "You can cut and paste directly from Microsoft Word, "
            "Google Docs, or a column or row from Excel, or Google Sheets."
        )
        paste_hint.setWordWrap(True)
        paste_hint.setStyleSheet("color: #424242; font-size: 12px;")
        paste_layout.addWidget(paste_hint)

        self.paste_edit = QPlainTextEdit()
        self.paste_edit.setPlaceholderText("Type or paste text here.\n(e.g. D, B, A, A, E\u2026)")
        self.paste_edit.setMinimumHeight(100)
        self.paste_edit.setMaximumHeight(200)
        paste_layout.addWidget(self.paste_edit)

        self.add_btn = QPushButton("Add to Key \u2026")
        self.add_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.add_btn.setMinimumHeight(32)
        self.add_btn.clicked.connect(self._on_add_to_key)
        paste_layout.addWidget(self.add_btn)

        left_layout.addWidget(paste_group)

        # Import group
        import_group = QGroupBox("Edit An Existing Key")
        import_layout = QVBoxLayout(import_group)

        self.import_btn = QPushButton("Import Key\u2026")
        self.import_btn.clicked.connect(self._on_import_file)
        import_layout.addWidget(self.import_btn)

        left_layout.addWidget(import_group)

        # Summary
        self.summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(self.summary_group)
        self.summary_label = QLabel("No key loaded.")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.summary_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # ---- Centre panel: table ----
        centre_widget = QWidget()
        centre_layout = QVBoxLayout(centre_widget)
        centre_layout.setContentsMargins(4, 0, 0, 0)

        # Table
        self.table = _KeyTable()
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectItems
        )
        self.table.setSelectionMode(
            QTableWidget.SelectionMode.ExtendedSelection
        )
        self.table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.AnyKeyPressed
        )
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        hdr = self.table.horizontalHeader()
        hdr.setMinimumSectionSize(36)
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizeAdjustPolicy(
            QTableWidget.SizeAdjustPolicy.AdjustToContents
        )
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        centre_layout.addWidget(self.table, 1)

        splitter.addWidget(centre_widget)

        # ---- Right panel: help reference ----
        help_scroll = QScrollArea()
        help_scroll.setWidgetResizable(True)
        help_scroll.setFrameShape(QFrame.Shape.NoFrame)
        help_scroll.setMinimumWidth(180)
        help_scroll.setMaximumWidth(260)

        help_widget = QWidget()
        help_widget.setStyleSheet(
            "QWidget { background: #FFFFFF; }"
            " QLabel { background: transparent; }"
        )
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(8, 8, 8, 8)

        ref_label = QLabel("Answer Format Reference")
        ref_label.setStyleSheet(f"font-weight: bold; color: {TEAL}; font-size: 13px;")
        help_layout.addWidget(ref_label)

        ref_text = QLabel(
            "<b>Single answer</b><br>"
            "<code>A</code> \u2014 student must answer A<br><br>"
            "<b>Custom points</b><br>"
            "<code>A:3</code> \u2014 worth 3 points<br>"
            "<i>(the 3 goes in the Pts column)</i><br><br>"
            "<b>Either / Or</b><br>"
            "<code>A^C</code> \u2014 A <i>or</i> C for full credit<br><br>"
            "<b>Must answer both</b><br>"
            "<code>C&amp;E</code> \u2014 student must select "
            "<i>both</i> C and E<br><br>"
            "<b>Partial credit (lenient)</b><br>"
            "<code>A@B</code> \u2014 +pts per correct, wrong ignored<br><br>"
            "<b>Partial credit (strict)</b><br>"
            "<code>A~B</code> \u2014 +pts per correct, \u2212pts per wrong<br><br>"
            "<b>Freebie</b><br>"
            "<code>*</code> \u2014 everyone gets points<br><br>"
            "<b>Discard</b><br>"
            "<i>(leave blank)</i> \u2014 removed from scoring"
        )
        ref_text.setWordWrap(True)
        ref_text.setTextFormat(Qt.TextFormat.RichText)
        ref_text.setStyleSheet(_HELP_LABEL_STYLE)
        help_layout.addWidget(ref_text)

        help_layout.addSpacing(12)

        legend_label = QLabel("Cell Colors")
        legend_label.setStyleSheet(f"font-weight: bold; color: {TEAL}; font-size: 13px;")
        help_layout.addWidget(legend_label)

        legend_text = QLabel(
            f'<span style="background:{_COLOR_NORMAL.name()}; color:#212121;'
            ' padding:1px 6px;">&nbsp;Blue&nbsp;</span>'
            " \u2014 single answer<br>"
            f'<span style="background:{_COLOR_PARTIAL.name()}; color:#212121;'
            ' padding:1px 6px;">&nbsp;Amber&nbsp;</span>'
            " \u2014 multi-answer (^, &amp;, @, ~)<br>"
            f'<span style="background:{_COLOR_FREEBIE.name()}; color:#212121;'
            ' padding:1px 6px;">&nbsp;Green&nbsp;</span>'
            " \u2014 freebie (*)<br>"
            f'<span style="background:{_COLOR_DISCARD.name()}; color:#212121;'
            ' padding:1px 6px;">&nbsp;Gray&nbsp;</span>'
            " \u2014 empty / discard"
        )
        legend_text.setWordWrap(True)
        legend_text.setTextFormat(Qt.TextFormat.RichText)
        legend_text.setStyleSheet(_HELP_LABEL_STYLE)
        help_layout.addWidget(legend_text)

        help_layout.addSpacing(12)

        tips_label = QLabel("Tips")
        tips_label.setStyleSheet(f"font-weight: bold; color: {TEAL}; font-size: 13px;")
        help_layout.addWidget(tips_label)

        tips_text = QLabel(
            "\u2022 Double-click a cell to edit\n"
            "\u2022 Enter moves down, Tab moves right\n"
            "\u2022 Points default to the version\u2019s\n"
            "  default \u2014 override per-question as needed\n"
            "\u2022 Paste answers from Word, Notepad,\n"
            "  or email into the box on the left"
        )
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet(_HELP_LABEL_STYLE)
        help_layout.addWidget(tips_text)

        help_layout.addStretch()
        help_scroll.setWidget(help_widget)
        splitter.addWidget(help_scroll)

        splitter.setSizes([240, 480, 220])

    # ------------------------------------------------------------------ #
    # Add to Key (paste -> wizard -> table)
    # ------------------------------------------------------------------ #

    def _on_add_to_key(self):
        """Parse the paste box, open the wizard, add version to table."""
        text = self.paste_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(
                self, "Nothing to Add",
                "Type or paste a comma-separated list of answers first."
            )
            return

        # Parse answers
        from markshark.key_parser import parse_answer

        parts = self._split_answer_text(text)
        if not parts:
            QMessageBox.warning(
                self, "Parse Error",
                "Could not find any answers in the text.\n\n"
                "Expected comma-separated letters like:  A, B, C, D, A \u2026"
            )
            return

        # Open wizard
        dlg = _AddVersionDialog(self._versions, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        ver = dlg.version
        code = dlg.code
        dp = dlg.default_points

        # Check for duplicate version
        if ver in self._versions:
            reply = QMessageBox.question(
                self, "Replace Version?",
                f"Version {ver} already exists.\nReplace it with the new answers?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            self._versions.append(ver)

        # Store metadata
        self._version_meta[ver] = {"dp": dp, "code": code}

        # Parse each answer into (letter_text, points)
        parsed = []
        for p in parts:
            spec = parse_answer(p, dp)
            ans_display, pts_display = self._spec_to_display(spec, dp)
            parsed.append((ans_display, pts_display))

        # Ensure table has enough data rows (+ 1 for header)
        num_q = len(parsed)
        self._grow_table_rows(num_q)

        # Rebuild columns for all versions, then fill this version
        self._rebuild_table_structure()

        vi = self._versions.index(ver)
        ans_col = 1 + vi * 2
        pts_col = 2 + vi * 2

        self.table.blockSignals(True)
        for r, (ans, pts) in enumerate(parsed):
            self.table.setItem(r + 1, ans_col, QTableWidgetItem(ans))
            self.table.setItem(r + 1, pts_col, QTableWidgetItem(pts))
        self.table.blockSignals(False)

        self._colorize_all()
        self._update_summary()
        self.paste_edit.clear()
        self.status_label.setText(
            f"Added version {ver} \u2014 {num_q} questions."
        )

    def _split_answer_text(self, text: str) -> List[str]:
        """Split pasted text into individual answer tokens.

        Handles commas, semicolons, spaces, tabs, and any line endings
        (``\\n``, ``\\r``, ``\\r\\n``).  Multi-character answer specs
        like ``A^B`` or ``A:3`` are kept intact.
        """
        import re

        # Normalize all line endings to \n, then split on any delimiter
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on comma, semicolon, whitespace (space/tab/newline)
        tokens = re.split(r"[,;\s]+", text)
        return [t for t in tokens if t]

    @staticmethod
    def _spec_to_display(spec, default_points: float):
        """Convert an AnswerSpec into (answer_text, points_text) for the table.

        Strips point annotations from the answer text and puts them
        in the points column instead.  E.g. ``E:3`` -> (``E``, ``3``).
        """
        from markshark.key_parser import ScoringMode

        pts = str(spec.total_points)

        if spec.mode == ScoringMode.DISCARD:
            return ("", "0.0")
        if spec.mode == ScoringMode.FREEBIE:
            return ("*", pts)

        keys_sorted = sorted(spec.correct_answers.keys())
        if not keys_sorted:
            return ("", pts)

        op_map = {
            ScoringMode.OR: "^",
            ScoringMode.AND: "&",
            ScoringMode.PARTIAL_LENIENT: "@",
            ScoringMode.PARTIAL_STRICT: "~",
        }
        op = op_map.get(spec.mode)
        if op is not None:
            return (op.join(keys_sorted), pts)

        # SINGLE
        return (keys_sorted[0], pts)

    # ------------------------------------------------------------------ #
    # Import file
    # ------------------------------------------------------------------ #

    def _on_import_file(self):
        start_dir = ""
        project_dir = self.project_selector.project_dir()
        if project_dir:
            inp = project_dir / "input_files"
            start_dir = str(inp) if inp.is_dir() else str(project_dir)

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Answer Key", start_dir,
            "Key files (*.txt *.csv *.tsv *.xlsx);;All files (*)",
        )
        if not path:
            return
        self._import_from_path(Path(path))

    def _import_from_path(self, path: Path):
        from markshark.key_parser import load_key_file

        try:
            key_set = load_key_file(path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Import Error",
                f"Could not parse key file:\n\n{exc}",
            )
            return

        self._load_key_set(key_set)
        self.status_label.setText(f"Imported: {path.name}")

    # ------------------------------------------------------------------ #
    # Load AnswerKeySet into the table
    # ------------------------------------------------------------------ #

    def _load_key_set(self, key_set):
        """Replace entire table contents from an AnswerKeySet."""
        self._versions = sorted(key_set.keys)
        if not self._versions:
            self._versions = ["A"]

        max_q = max(
            (vk.num_questions for vk in key_set.keys.values()), default=0
        )

        # Store metadata
        self._version_meta.clear()
        for vid in self._versions:
            vk = key_set.keys[vid]
            self._version_meta[vid] = {
                "dp": vk.default_points,
                "code": vk.code or "",
            }

        # Build table
        self._grow_table_rows(max_q)
        self._rebuild_table_structure()

        self.table.blockSignals(True)
        for vi, vid in enumerate(self._versions):
            vk = key_set.keys.get(vid)
            if not vk:
                continue
            dp = vk.default_points
            ans_col = 1 + vi * 2
            pts_col = 2 + vi * 2
            for q_idx, spec in enumerate(vk.answers):
                if q_idx + 1 >= self.table.rowCount():
                    break
                ans_text, pts_text = self._spec_to_display(spec, dp)
                self.table.setItem(q_idx + 1, ans_col, QTableWidgetItem(ans_text))
                self.table.setItem(q_idx + 1, pts_col, QTableWidgetItem(pts_text))
        self.table.blockSignals(False)

        self._colorize_all()
        self._update_summary()

    # ------------------------------------------------------------------ #
    # Table construction
    # ------------------------------------------------------------------ #

    def _grow_table_rows(self, needed_data_rows: int):
        """Ensure the table has at least *needed_data_rows* + 1 (header)."""
        total = needed_data_rows + 1  # row 0 is the version header
        if self.table.rowCount() < total:
            self.table.setRowCount(total)

    def _rebuild_table_structure(self):
        """Rebuild columns, merged version header row, and Q# items."""
        self.table.blockSignals(True)
        # data_rows = rows of actual answers (excludes header row)
        data_rows = max(self.table.rowCount() - 1, 0) if self._versions else 0

        # Save existing data (reads from row offsets that skip row 0)
        old_data = self._read_table_data()

        # Column layout: Q# | Answer | Pts | Answer | Pts | …
        cols = ["Q#"]
        for _v in self._versions:
            cols.append("Answer")
            cols.append("Pts")

        total_rows = data_rows + 1  # +1 for the merged header row
        self.table.setRowCount(total_rows)
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        # --- Row 0: merged version header cells ---
        # Clear any previous spans (only reset cells that are actually merged)
        for c in range(self.table.columnCount()):
            if self.table.columnSpan(0, c) > 1 or self.table.rowSpan(0, c) > 1:
                self.table.setSpan(0, c, 1, 1)

        # Q# cell in row 0 — blank, non-editable
        q0 = QTableWidgetItem("")
        q0.setFlags(Qt.ItemFlag.ItemIsEnabled)
        q0.setBackground(QBrush(QColor(_COLOR_VERSION_BAR)))
        self.table.setItem(0, 0, q0)

        for vi, v in enumerate(self._versions):
            start_col = 1 + vi * 2
            self.table.setSpan(0, start_col, 1, 2)  # merge Answer+Pts

            meta = self._version_meta.get(v, {})
            dp = meta.get("dp", 1.0)
            code = meta.get("code", "")
            line2 = f"{dp:g} pts"
            if code:
                line2 += f" · {code}"
            label = f"VER {v}\n{line2}"

            cell = QTableWidgetItem(label)
            cell.setFlags(Qt.ItemFlag.ItemIsEnabled)
            cell.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            cell.setBackground(QBrush(QColor(_COLOR_VERSION_BAR)))
            cell.setForeground(QBrush(_COLOR_TEXT))
            font = cell.font()
            font.setBold(True)
            cell.setFont(font)
            self.table.setItem(0, start_col, cell)

        # Make header row tall enough for two lines
        self.table.setRowHeight(0, 40)

        # --- Delegates ---
        ans_del = _AnswerCellDelegate(self.table)
        pts_del = _PointsCellDelegate(self.table)
        for vi in range(len(self._versions)):
            self.table.setItemDelegateForColumn(1 + vi * 2, ans_del)
            self.table.setItemDelegateForColumn(2 + vi * 2, pts_del)

        # --- Q# column (rows 1+) ---
        for row in range(1, total_rows):
            q_item = QTableWidgetItem(str(row))  # row 1 = Q1
            q_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            )
            q_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, q_item)

        self.table.horizontalHeader().setStretchLastSection(False)

        # Restore old data
        self._write_table_data(old_data)
        self.table.blockSignals(False)

    def _read_table_data(self) -> Dict[str, List[tuple]]:
        """Read answer/pts data from data rows (skipping header row 0)."""
        data: Dict[str, List[tuple]] = {}
        for vi, v in enumerate(self._versions):
            ans_col = 1 + vi * 2
            pts_col = 2 + vi * 2
            if ans_col >= self.table.columnCount():
                continue
            rows = []
            for r in range(1, self.table.rowCount()):  # skip row 0
                ai = self.table.item(r, ans_col)
                pi = self.table.item(r, pts_col)
                rows.append((ai.text() if ai else "", pi.text() if pi else ""))
            data[v] = rows
        return data

    def _write_table_data(self, data: Dict[str, List[tuple]]):
        """Write answer/pts data into data rows (skipping header row 0)."""
        for vi, v in enumerate(self._versions):
            ans_col = 1 + vi * 2
            pts_col = 2 + vi * 2
            if ans_col >= self.table.columnCount():
                continue
            rows = data.get(v, [])
            for r in range(min(len(rows), self.table.rowCount() - 1)):
                self.table.setItem(r + 1, ans_col, QTableWidgetItem(rows[r][0]))
                self.table.setItem(r + 1, pts_col, QTableWidgetItem(rows[r][1]))

    # ------------------------------------------------------------------ #
    # Table -> AnswerKeySet
    # ------------------------------------------------------------------ #

    def _table_to_key_set(self):
        from markshark.key_parser import (
            AnswerKeySet, VersionKey, AnswerSpec, parse_answer,
        )

        keys = {}
        code_to_version = {}

        for vi, v in enumerate(self._versions):
            meta = self._version_meta.get(v, {})
            dp = meta.get("dp", 1.0)
            code = meta.get("code", "")

            answers = []
            ans_col = 1 + vi * 2
            pts_col = 2 + vi * 2

            for r in range(1, self.table.rowCount()):  # skip header row
                ai = self.table.item(r, ans_col)
                pi = self.table.item(r, pts_col)
                ans_text = ai.text().strip() if ai else ""
                pts_text = pi.text().strip() if pi else ""

                spec = parse_answer(ans_text, dp)

                # Override points from the Pts column
                try:
                    custom_pts = float(pts_text)
                    if spec.total_points != custom_pts:
                        spec.total_points = custom_pts
                except (ValueError, TypeError):
                    pass

                answers.append(spec)

            vk = VersionKey(
                version=v, code=code or None,
                default_points=dp, answers=answers,
            )
            keys[v] = vk
            if code:
                code_to_version[code] = v

        return AnswerKeySet(keys=keys, code_to_version=code_to_version)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def _on_export_to_project(self):
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            QMessageBox.warning(
                self, "No Assessment Selected",
                "Please select a course and assessment first,\n\n"
                "or use \u2018Save As\u2026\u2019 to pick a location.",
            )
            return

        if not self._versions or self.table.rowCount() <= 1:
            QMessageBox.information(
                self, "Nothing to Save",
                "Add at least one version to the key first.",
            )
            return

        ok, warnings = self._validate_table()
        if warnings:
            reply = QMessageBox.question(
                self, "Validation Warnings",
                "Some questions have issues:\n\n"
                + "\n".join(warnings[:10])
                + ("\n\u2026" if len(warnings) > 10 else "")
                + "\n\nSave anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        inp_dir = project_dir / "input_files"
        inp_dir.mkdir(exist_ok=True)
        dest = inp_dir / "key.txt"

        if dest.exists():
            reply = QMessageBox.question(
                self, "Overwrite?",
                f"{dest.name} already exists.\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self._do_export(dest, "txt")

    def _on_export_as(self):
        if not self._versions or self.table.rowCount() <= 1:
            QMessageBox.information(
                self, "Nothing to Save",
                "Add at least one version to the key first.",
            )
            return

        start_dir = ""
        project_dir = self.project_selector.project_dir()
        if project_dir:
            inp = project_dir / "input_files"
            start_dir = str(inp) if inp.is_dir() else str(project_dir)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Answer Key", start_dir,
            "Text Key (*.txt);;Excel Key (*.xlsx)",
        )
        if not path:
            return

        ok, warnings = self._validate_table()
        if warnings:
            reply = QMessageBox.question(
                self, "Validation Warnings",
                "Some questions have issues:\n\n"
                + "\n".join(warnings[:10])
                + ("\n\u2026" if len(warnings) > 10 else "")
                + "\n\nSave anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        fmt = "xlsx" if path.endswith(".xlsx") else "txt"
        self._do_export(Path(path), fmt)

    def _do_export(self, path: Path, fmt: str):
        from markshark.key_parser import write_key_file

        key_set = self._table_to_key_set()
        try:
            write_key_file(key_set, path, fmt=fmt)
        except Exception as exc:
            QMessageBox.warning(
                self, "Export Error", f"Could not save key file:\n\n{exc}"
            )
            return
        self.status_label.setText(f"Saved: {path.name}")

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate_table(self) -> tuple:
        warnings = []
        for vi, v in enumerate(self._versions):
            ans_col = 1 + vi * 2
            for r in range(1, self.table.rowCount()):  # skip header row
                item = self.table.item(r, ans_col)
                text = item.text().strip() if item else ""
                if not text:
                    warnings.append(
                        f"Q{r} Version {v}: empty (will be discarded)"
                    )
        return (len(warnings) == 0, warnings)

    # ------------------------------------------------------------------ #
    # Cell colouring
    # ------------------------------------------------------------------ #

    def _on_cell_clicked(self, row: int, col: int):
        """If the user clicks on the merged header row, open an edit dialog."""
        if row != 0 or col == 0 or not self._versions:
            return
        # Determine which version this column belongs to
        vi = (col - 1) // 2
        if vi < 0 or vi >= len(self._versions):
            return
        self._edit_version_meta(self._versions[vi])

    def _edit_version_meta(self, ver: str):
        """Open the version dialog pre-filled for editing metadata."""
        meta = self._version_meta.get(ver, {})
        old_dp = meta.get("dp", 1.0)

        dlg = _AddVersionDialog(
            self._versions,
            self,
            edit_version=ver,
            edit_code=meta.get("code", ""),
            edit_points=old_dp,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        new_dp = dlg.default_points

        # If default points changed, update every cell that still holds the
        # old default so the teacher doesn't have to fix them one by one.
        # Cells that were manually overridden (different from old default)
        # are left untouched.
        if new_dp != old_dp and ver in self._versions:
            vi = self._versions.index(ver)
            pts_col = 2 + vi * 2          # points column for this version
            new_pts_str = str(new_dp)

            self.table.blockSignals(True)
            for r in range(1, self.table.rowCount()):
                cell = self.table.item(r, pts_col)
                if cell is None:
                    continue
                try:
                    cell_val = float(cell.text())
                except ValueError:
                    continue
                # Only swap cells still at the old default value
                if cell_val == old_dp:
                    cell.setText(new_pts_str)
            self.table.blockSignals(False)

        # Update stored metadata
        self._version_meta[ver] = {
            "dp": new_dp,
            "code": dlg.code,
        }

        # Refresh the header row to show the new values
        self._rebuild_table_structure()
        self._colorize_all()
        self._update_summary()
        self.status_label.setText(
            f"Updated version {ver} metadata."
        )

    # ------------------------------------------------------------------ #
    # Right-click context menu
    # ------------------------------------------------------------------ #

    def _on_table_context_menu(self, pos):
        """Show a context menu with row/column operations."""
        item = self.table.itemAt(pos)
        row = self.table.rowAt(pos.y())
        col = self.table.columnAt(pos.x())

        menu = QMenu(self.table)

        # --- Row operations (only for data rows) ---
        if row >= 1:
            q_num = row  # row 1 = Q1

            # Per-version shift (only when clicking in a version column)
            if col >= 1 and self._versions:
                vi = (col - 1) // 2
                if 0 <= vi < len(self._versions):
                    ver = self._versions[vi]
                    menu.addAction(
                        f"Shift {ver} answers down from Q{q_num}",
                        lambda r=row, v=ver: self._shift_version_down(r, v),
                    )
                    menu.addSeparator()

            menu.addAction(
                f"Insert row above Q{q_num} (all versions)",
                lambda r=row: self._insert_row(r),
            )
            menu.addAction(
                f"Insert row below Q{q_num} (all versions)",
                lambda r=row: self._insert_row(r + 1),
            )
            menu.addSeparator()
            menu.addAction(
                f"Delete Q{q_num} (all versions)",
                lambda r=row: self._delete_row(r),
            )
        elif row == 0 and self._versions:
            # Clicked on the header row
            vi = (col - 1) // 2 if col >= 1 else -1
            if 0 <= vi < len(self._versions):
                ver = self._versions[vi]
                menu.addAction(
                    f"Edit Version {ver}\u2026",
                    lambda v=ver: self._edit_version_meta(v),
                )
                menu.addAction(
                    f"Delete Version {ver}",
                    lambda v=ver: self._delete_version(v),
                )

        # --- General row operations ---
        if self._versions:
            menu.addSeparator()
            menu.addAction(
                "Add row at end",
                self._add_row_at_end,
            )

        if not menu.isEmpty():
            menu.exec(self.table.viewport().mapToGlobal(pos))

    def _insert_row(self, at_row: int):
        """Insert a blank data row at the given table row index."""
        self.table.insertRow(at_row)
        # Set the Q# cell for the new row
        q_item = QTableWidgetItem("")
        q_item.setFlags(
            Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        )
        q_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(at_row, 0, q_item)
        self._renumber_questions()
        self._update_summary()

    def _delete_row(self, row: int):
        """Delete a data row (never the header row 0)."""
        if row < 1 or row >= self.table.rowCount():
            return
        num_data = self.table.rowCount() - 1
        if num_data <= 1:
            QMessageBox.information(
                self, "Cannot Delete",
                "The key must have at least one question.",
            )
            return
        self.table.removeRow(row)
        self._renumber_questions()
        self._colorize_all()
        self._update_summary()

    def _add_row_at_end(self):
        """Append a blank data row at the bottom."""
        new_row = self.table.rowCount()
        self.table.insertRow(new_row)
        q_item = QTableWidgetItem(str(new_row))
        q_item.setFlags(
            Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        )
        q_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(new_row, 0, q_item)
        self._update_summary()

    def _shift_version_down(self, from_row: int, ver: str):
        """Shift one version's answers down from *from_row*, inserting a blank.

        If the last data row for this version is non-empty, a new row is
        appended to the table first so nothing is lost.  Other versions'
        columns are left untouched.
        """
        if ver not in self._versions:
            return
        vi = self._versions.index(ver)
        ans_col = 1 + vi * 2
        pts_col = 2 + vi * 2
        last_row = self.table.rowCount() - 1

        # If the last row for this version has data, grow the table
        last_ans = self.table.item(last_row, ans_col)
        if last_ans and last_ans.text().strip():
            self._add_row_at_end()
            last_row = self.table.rowCount() - 1

        # Shift cells down (bottom-up to avoid overwriting)
        self.table.blockSignals(True)
        for r in range(last_row, from_row, -1):
            src_ans = self.table.item(r - 1, ans_col)
            src_pts = self.table.item(r - 1, pts_col)
            self.table.setItem(
                r, ans_col,
                QTableWidgetItem(src_ans.text() if src_ans else ""),
            )
            self.table.setItem(
                r, pts_col,
                QTableWidgetItem(src_pts.text() if src_pts else ""),
            )

        # Blank out the insertion row
        self.table.setItem(from_row, ans_col, QTableWidgetItem(""))
        self.table.setItem(from_row, pts_col, QTableWidgetItem(""))
        self.table.blockSignals(False)

        self._colorize_all()
        self._update_summary()
        self.status_label.setText(
            f"Shifted Version {ver} answers down from Q{from_row}."
        )

    def _delete_version(self, ver: str):
        """Remove a version (two columns) from the table after confirmation."""
        if ver not in self._versions:
            return
        if len(self._versions) <= 1:
            QMessageBox.information(
                self, "Cannot Delete",
                "The key must have at least one version.",
            )
            return
        reply = QMessageBox.question(
            self, "Delete Version?",
            f"Delete Version {ver} and all its answers?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._versions.remove(ver)
        self._version_meta.pop(ver, None)
        self._rebuild_table_structure()
        self._colorize_all()
        self._update_summary()
        self.status_label.setText(f"Deleted version {ver}.")

    def _renumber_questions(self):
        """Renumber the Q# column after row insert/delete."""
        for row in range(1, self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                item.setText(str(row))
            else:
                q_item = QTableWidgetItem(str(row))
                q_item.setFlags(
                    Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                )
                q_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 0, q_item)

    def _on_cell_changed(self, row: int, col: int):
        if row == 0 or col == 0:
            return  # header row or Q# column — ignore
        if (col - 1) % 2 == 0:
            self._colorize_answer_cell(row, col)
        self._update_summary()

    def _colorize_all(self):
        for vi in range(len(self._versions)):
            ans_col = 1 + vi * 2
            for r in range(1, self.table.rowCount()):  # skip header row
                self._colorize_answer_cell(r, ans_col)

    def _colorize_answer_cell(self, row: int, col: int):
        item = self.table.item(row, col)
        if not item:
            return
        text = item.text().strip().upper()
        dark = QBrush(_COLOR_TEXT)

        if not text:
            item.setBackground(QBrush(_COLOR_DISCARD))
        elif text == "*" or text.startswith("*:"):
            item.setBackground(QBrush(_COLOR_FREEBIE))
        elif any(op in text for op in ("^", "&", "@", "~")):
            item.setBackground(QBrush(_COLOR_PARTIAL))
        else:
            item.setBackground(QBrush(_COLOR_NORMAL))
        item.setForeground(dark)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def _update_summary(self):
        num_data_rows = max(self.table.rowCount() - 1, 0)
        if not self._versions or num_data_rows == 0:
            self.summary_label.setText("No key loaded.")
            return

        lines = [
            f"Versions: {len(self._versions)} "
            f"({', '.join(self._versions)})"
        ]
        lines.append(f"Questions: {num_data_rows}")

        for vi, v in enumerate(self._versions):
            meta = self._version_meta.get(v, {})
            dp = meta.get("dp", 1.0)
            ans_col = 1 + vi * 2
            pts_col = 2 + vi * 2
            total = 0.0
            filled = 0
            for r in range(1, self.table.rowCount()):  # skip header row
                ai = self.table.item(r, ans_col)
                pi = self.table.item(r, pts_col)
                if ai and ai.text().strip():
                    filled += 1
                    try:
                        total += float(pi.text()) if pi else dp
                    except (ValueError, TypeError):
                        total += dp
            lines.append(
                f"  {v}: {filled}/{num_data_rows} filled, "
                f"{total:g} pts"
            )

        self.summary_label.setText("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Project integration
    # ------------------------------------------------------------------ #

    def _on_project_changed(self, _name: str = ""):
        self._auto_detect_existing_key()

    def _auto_detect_existing_key(self):
        self._key_found_bar.hide()
        self._detected_key_path = None

        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return

        inp = project_dir / "input_files"
        if not inp.is_dir():
            return

        for name in ("key.txt", "key.csv", "key.xlsx", "key.tsv"):
            candidate = inp / name
            if candidate.is_file():
                self._detected_key_path = candidate
                self._key_found_label.setText(
                    f"Found existing key: {candidate.name}"
                )
                self._key_found_bar.show()
                return

        for f in inp.iterdir():
            if (
                f.is_file()
                and "key" in f.stem.lower()
                and f.suffix.lower() in (".txt", ".csv", ".tsv", ".xlsx")
            ):
                self._detected_key_path = f
                self._key_found_label.setText(
                    f"Found existing key: {f.name}"
                )
                self._key_found_bar.show()
                return

    def _load_detected_key(self):
        if self._detected_key_path:
            self._import_from_path(self._detected_key_path)
            self._key_found_bar.hide()

    # ------------------------------------------------------------------ #
    # Show event
    # ------------------------------------------------------------------ #

    def showEvent(self, event):
        super().showEvent(event)
        self._auto_detect_existing_key()
