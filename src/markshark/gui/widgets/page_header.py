"""
Page header widget with title and MarkShark icon.
"""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
)


# Path to the icon
ICON_PATH = Path(__file__).parent.parent / "resources" / "icons" / "SHARKICON.png"


class PageHeader(QWidget):
    """
    Page header with title on top row, description below, and MarkShark icon on the right.

    Layout:
        +----------------------------------+--------+
        | Title (large, bold)              |        |
        | Description (smaller, gray)      |  ICON  |
        +----------------------------------+--------+

    Usage:
        header = PageHeader("Quick Grade", "Upload scans and grade them.")
        layout.addWidget(header)
    """

    def __init__(self, title: str, description: str = "", parent=None):
        super().__init__(parent)
        self._setup_ui(title, description)

    def _setup_ui(self, title: str, description: str):
        """Build the header UI."""
        # Main horizontal layout: icon on left, text on right
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 5)

        # Left side: icon (vertically centered)
        if ICON_PATH.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(ICON_PATH))
            # Scale to 80x80 pixels
            scaled = pixmap.scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            icon_label.setPixmap(scaled)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(icon_label)

        # Right side: title on top, description below (vertical stack)
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
