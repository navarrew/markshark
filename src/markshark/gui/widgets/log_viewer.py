"""
Log viewer widget for displaying CLI output and process logs.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QGroupBox,
)


class LogViewer(QWidget):
    """
    A text area for displaying log output with controls.

    Features:
    - Monospace font for log readability
    - Auto-scroll to bottom on new content
    - Clear button
    - Copy button
    """

    def __init__(self, title: str = "Log", parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Group box with title
        group = QGroupBox(title)
        group_layout = QVBoxLayout(group)

        # Text area
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = QFont("Courier New", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.text_edit.setFont(font)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        group_layout.addWidget(self.text_edit)

        # Control buttons
        btn_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        btn_layout.addWidget(clear_btn)

        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(self._copy_all)
        btn_layout.addWidget(copy_btn)

        btn_layout.addStretch()
        group_layout.addLayout(btn_layout)

        layout.addWidget(group)

    def append(self, text: str):
        """Append text to the log and scroll to bottom."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()

    def append_line(self, text: str):
        """Append a line of text (adds newline)."""
        self.append(text + "\n")

    def append_header(self, text: str, char: str = "="):
        """Append a formatted header."""
        line = char * 50
        self.append(f"\n{line}\n{text}\n{line}\n\n")

    def clear(self):
        """Clear all log content."""
        self.text_edit.clear()

    def _copy_all(self):
        """Copy all log content to clipboard."""
        from PySide6.QtWidgets import QApplication

        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_edit.toPlainText())

    def get_text(self) -> str:
        """Get all log text."""
        return self.text_edit.toPlainText()

    def save_to_file(self, path) -> bool:
        """
        Save all log content to a text file.

        Args:
            path: File path (str or Path) to write the log to.

        Returns:
            True if saved successfully, False otherwise.
        """
        from pathlib import Path as _Path

        try:
            p = _Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.get_text(), encoding="utf-8")
            return True
        except Exception as e:
            print(f"[warn] Could not save log to {path}: {e}")
            return False
