"""
About dialog showing application info.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
)


class AboutDialog(QDialog):
    """About MarkShark dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About MarkShark")
        self.setFixedSize(400, 250)
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("MarkShark")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Version
        from ..utils import get_app_version
        version_label = QLabel(f"Version {get_app_version()}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        # Description
        desc = QLabel(
            "Optical Mark Recognition for bubble sheet grading.\n\n"
            "Fast, accurate, and teacher-friendly."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()

        # Links / info
        info = QLabel(
            '<a href="https://github.com/navarrew/markshark">GitHub Repository</a>'
        )
        info.setOpenExternalLinks(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
