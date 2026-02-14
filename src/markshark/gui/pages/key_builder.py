"""
Key Build Utility - guided answer key creation.

Placeholder page for upcoming key-building workflow.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from ..widgets import PageHeader


class KeyBuilderPage(QWidget):
    """Placeholder page for the Key Build Utility."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        header = PageHeader(
            "Key Build Utility",
            "Build and manage answer keys for your bubble sheets.",
        )
        layout.addWidget(header)

        coming = QLabel("Coming soon")
        coming.setAlignment(Qt.AlignmentFlag.AlignCenter)
        coming.setStyleSheet("color: #888; font-size: 18px; margin-top: 60px;")
        layout.addWidget(coming)

        layout.addStretch()
