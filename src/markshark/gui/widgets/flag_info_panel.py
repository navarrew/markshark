"""
Flag information panel for the Review & Correct page.

Two modes:
  - **Normal**: shows flag text like "Flags: blank at Q5, multi at Q10"
  - **Orphan**: shows the scanned ID, suggested roster matches, and
    [Accept] buttons so the teacher can correct the ID in one click.
"""

from typing import Dict, List, Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSizePolicy,
)


# Style constants — use #FlagContainer selector so styles don't cascade
# to child QPushButtons / QLabels unintentionally.
_NORMAL_STYLE = (
    "#FlagContainer { padding: 6px; background-color: #FFF8E1; "
    "border: 1px solid #FFE082; border-radius: 3px; }"
    " #FlagContainer QLabel { color: #333333; }"
)
_ORPHAN_STYLE = (
    "#FlagContainer { padding: 6px; background-color: #C62828; "
    "border: 1px solid #B71C1C; border-radius: 3px; }"
    " #FlagContainer QLabel { color: white; }"
)
_CORRECTED_STYLE = (
    "#FlagContainer { padding: 6px; background-color: #1565C0; "
    "border: 1px solid #0D47A1; border-radius: 3px; }"
    " #FlagContainer QLabel { color: white; font-weight: bold; }"
)
_ACCEPT_BTN = (
    "QPushButton { background-color: #28a745; color: white; "
    "font-weight: bold; border: none; border-radius: 3px; padding: 4px 12px; }"
    "QPushButton:hover { background-color: #218838; }"
)
_LOAD_BTN = (
    "QPushButton { background-color: #0d6efd; color: white; "
    "border: none; border-radius: 3px; padding: 4px 12px; }"
    "QPushButton:hover { background-color: #0b5ed7; }"
)
_UNDO_BTN = (
    "QPushButton { background-color: #e0e0e0; color: #333; "
    "border: none; border-radius: 3px; padding: 4px 12px; }"
    "QPushButton:hover { background-color: #bdbdbd; }"
)


class FlagInfoPanel(QWidget):
    """Interactive flag information panel.

    Emits:
        suggestion_accepted(original_id, suggested_id, reason)
        roster_requested()  — teacher clicked "Load Roster..."
        undo_correction(student_id)  — teacher clicked "Undo" on a corrected orphan
    """

    suggestion_accepted = Signal(str, str, str)  # original_id, suggested_id, reason
    roster_requested = Signal()
    undo_correction = Signal(str)  # student_id whose correction should be reverted

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Container frame — styled per mode
        self._container = QWidget()
        self._container.setObjectName("FlagContainer")
        self._container.setStyleSheet(_NORMAL_STYLE)
        self._inner = QVBoxLayout(self._container)
        self._inner.setContentsMargins(6, 4, 6, 4)
        self._inner.setSpacing(4)

        # Normal-mode label (always exists, shown/hidden)
        self._flags_label = QLabel("Select a student to see flags.")
        self._flags_label.setWordWrap(True)
        self._inner.addWidget(self._flags_label)

        # Orphan header (hidden by default)
        self._orphan_header = QLabel()
        self._orphan_header.setWordWrap(True)
        self._orphan_header.setStyleSheet("font-weight: bold;")
        self._orphan_header.hide()
        self._inner.addWidget(self._orphan_header)

        # Suggestions container (holds rows of [Accept] id — name (reason))
        self._suggestions_widget = QWidget()
        self._suggestions_layout = QVBoxLayout(self._suggestions_widget)
        self._suggestions_layout.setContentsMargins(0, 0, 0, 0)
        self._suggestions_layout.setSpacing(3)
        self._suggestions_widget.hide()
        self._inner.addWidget(self._suggestions_widget)

        # "Load Roster..." button (hidden by default)
        self._load_roster_btn = QPushButton("Load Roster...")
        self._load_roster_btn.setStyleSheet(_LOAD_BTN)
        self._load_roster_btn.setFixedWidth(120)
        self._load_roster_btn.clicked.connect(self.roster_requested.emit)
        self._load_roster_btn.hide()
        self._inner.addWidget(self._load_roster_btn)

        # Undo button — shown only in corrected state
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setStyleSheet(_UNDO_BTN)
        self._undo_btn.setFixedWidth(70)
        self._undo_btn.hide()
        self._inner.addWidget(self._undo_btn)

        layout.addWidget(self._container)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_regular_flags(self, flag_text: str):
        """Show normal flag info (non-orphan row selected)."""
        self._container.setStyleSheet(_NORMAL_STYLE)
        self._flags_label.setText(flag_text)
        self._flags_label.show()
        self._orphan_header.hide()
        self._suggestions_widget.hide()
        self._load_roster_btn.hide()
        self._undo_btn.hide()

    def show_corrected(
        self,
        message: str = "CORRECTED",
        student_id: str = "",
        original_id: str = "",
        corrected_id: str = "",
    ):
        """Show a blue 'CORRECTED' banner with undo capability.

        Args:
            message:      Primary label text.
            student_id:   The *original* student ID (used to revert the correction).
            original_id:  Display string for the original scanned ID.
            corrected_id: Display string for the corrected (roster) ID.
        """
        self._container.setStyleSheet(_CORRECTED_STYLE)

        # Show what changed so the teacher can see original → corrected
        if original_id and corrected_id:
            self._flags_label.setText(f"{message}  ({original_id} → {corrected_id})")
        else:
            self._flags_label.setText(message)
        self._flags_label.show()
        self._orphan_header.hide()
        self._suggestions_widget.hide()
        self._load_roster_btn.hide()

        # Undo button — only show if we have a student_id to revert
        if student_id:
            # Disconnect any previous connection to avoid stale closures
            try:
                self._undo_btn.clicked.disconnect()
            except RuntimeError:
                pass  # no previous connection
            self._undo_btn.clicked.connect(
                lambda _checked=False, sid=student_id: self.undo_correction.emit(sid)
            )
            self._undo_btn.show()
        else:
            self._undo_btn.hide()

    def show_orphan_suggestions(
        self,
        orphan_id: str,
        orphan_name: str,
        suggestions: List[Dict[str, str]],
    ):
        """Show the orphan resolution panel with roster suggestions.

        Args:
            orphan_id:   The student ID read from the scan.
            orphan_name: "Last, First" or similar display name.
            suggestions: Output of ``score_core.suggest_matches()``
                         — list of ``{student_id, name, score, reason}``.
        """
        self._container.setStyleSheet(_ORPHAN_STYLE)
        self._flags_label.hide()

        # Header
        header = f"Orphan ID: {orphan_id}"
        if orphan_name:
            header += f"  ({orphan_name})"
        self._orphan_header.setText(header)
        self._orphan_header.show()

        # Clear old suggestion rows
        self._clear_suggestions()

        if suggestions:
            for sugg in suggestions:
                self._add_suggestion_row(
                    orphan_id,
                    sugg["student_id"],
                    sugg.get("name", ""),
                    sugg.get("reason", ""),
                )
            self._suggestions_widget.show()
        else:
            no_match = QLabel("No close roster matches found. Edit the ID manually.")
            no_match.setStyleSheet("color: #666; font-style: italic;")
            self._suggestions_layout.addWidget(no_match)
            self._suggestions_widget.show()

        self._load_roster_btn.hide()
        self._undo_btn.hide()

    def show_no_roster(self):
        """Show a message when no roster is available."""
        self._container.setStyleSheet(_ORPHAN_STYLE)
        self._flags_label.setText("Orphan ID — load a roster to see suggested matches.")
        self._flags_label.show()
        self._orphan_header.hide()
        self._suggestions_widget.hide()
        self._load_roster_btn.show()
        self._undo_btn.hide()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_suggestion_row(
        self, original_id: str, suggested_id: str, name: str, reason: str
    ):
        """Add one clickable suggestion row."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        btn = QPushButton("Accept")
        btn.setMinimumWidth(70)
        btn.setStyleSheet(_ACCEPT_BTN)

        def _on_accept(_checked=False, oid=original_id, sid=suggested_id, r=reason):
            self.suggestion_accepted.emit(oid, sid, r)

        btn.clicked.connect(_on_accept)
        row.addWidget(btn)

        label = QLabel(f"<b>{suggested_id}</b> — {name}  <i>({reason})</i>")
        label.setTextFormat(Qt.TextFormat.RichText)
        row.addWidget(label, 1)

        wrapper = QWidget()
        wrapper.setLayout(row)
        self._suggestions_layout.addWidget(wrapper)

    def _clear_suggestions(self):
        """Remove all suggestion rows."""
        while self._suggestions_layout.count():
            item = self._suggestions_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
