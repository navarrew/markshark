"""
Page header widget with title and description.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
)


class PageHeader(QWidget):
    """
    Page header with title on top row and description below.

    Layout:
        +----------------------------------+
        | Title (large, bold)              |
        | Description (smaller, gray)      |
        +----------------------------------+

    Usage:
        header = PageHeader("Quick Grade", "Upload scans and grade them.")
        layout.addWidget(header)
    """

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self._setup_ui(title, description)

    def _setup_ui(self, title: str, description: str):
        """Build the header UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 5)

        # Title and description in a vertical stack
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        # Title (top row)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        text_layout.addWidget(title_label)

        # Description (bottom row, if provided)
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: white; font-size: 14pt;")
            text_layout.addWidget(desc_label)

        layout.addLayout(text_layout, 1)
