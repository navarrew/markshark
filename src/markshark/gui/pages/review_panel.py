"""
Review Panel page - review flagged items and enter corrections.

This is a placeholder for the future review functionality.
See docs/issues/pyside-gui-roadmap.md for planned features.
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QFrame,
    QGroupBox,
)


class ReviewPanelPage(QWidget):
    """
    Review and correct flagged items.

    TODO: Implement full review functionality:
    - PDF page viewer with zoom/pan
    - Flagged item list with inline corrections
    - Navigation between flagged pages
    - Save corrections to apply to results
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results_data: Optional[dict] = None
        self._setup_ui()

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Review & Correct")
        header.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(header)

        desc = QLabel(
            "Review flagged items (blank, ambiguous, low confidence) and enter corrections."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Placeholder content
        placeholder = QFrame()
        placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px dashed #999;")
        placeholder_layout = QVBoxLayout(placeholder)

        coming_soon = QLabel("Review Panel - Coming Soon")
        coming_soon.setStyleSheet("font-size: 16px; color: #666;")
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(coming_soon)

        features = QLabel(
            "Planned features:\n\n"
            "• PDF page viewer with zoom/pan\n"
            "• Highlighted flagged bubble regions\n"
            "• Inline correction entry (radio buttons)\n"
            "• Navigation through flagged pages only\n"
            "• Apply corrections without Excel round-trip\n"
            "• Keyboard shortcuts for fast review"
        )
        features.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_layout.addWidget(features)

        layout.addWidget(placeholder, 1)

        # Status bar
        self.status_label = QLabel("No results loaded. Run Quick Grade first.")
        layout.addWidget(self.status_label)

    def load_results(self, results_data: dict):
        """
        Load results data from Quick Grade.

        Args:
            results_data: Dict with work_dir, results_csv, scored_pdf, etc.
        """
        self._results_data = results_data
        self.status_label.setText(
            f"Results loaded from: {results_data.get('work_dir', 'unknown')}"
        )

        # TODO: Parse flagged_for_review.xlsx or results.csv to find flagged items
        # TODO: Load scored PDF for page viewing
        # TODO: Populate the review UI
