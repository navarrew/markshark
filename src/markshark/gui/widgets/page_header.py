"""
Page header widget with title and MarkShark icon.
"""

from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
)


# Path to the icon
ICON_PATH = Path(__file__).parent.parent / "resources" / "icons" / "SHARKICON.png"


class PageHeader(QWidget):
    """
    Page header with title on the left and MarkShark icon on the right.

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

        # Left side: title and description
        left_layout = QHBoxLayout()
        left_layout.setSpacing(10)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        left_layout.addWidget(title_label)

        # Description (if provided)
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("color: #666;")
            left_layout.addWidget(desc_label, 1)

        layout.addLayout(left_layout, 1)

        # Right side: icon
        if ICON_PATH.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(ICON_PATH))
            # Scale to approximately 2cm (~75 pixels at 96 DPI)
            scaled = pixmap.scaled(
                80, 80,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            icon_label.setPixmap(scaled)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(icon_label)
