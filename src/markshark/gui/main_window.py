"""
Main application window with navigation sidebar and stacked pages.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QLabel,
    QMenuBar,
    QMenu,
    QMessageBox,
    QSplitter,
)

from .pages.quick_grade import QuickGradePage
from .pages.review_panel import ReviewPanelPage
from .pages.settings import SettingsPage
from .dialogs.about import AboutDialog


class MainWindow(QMainWindow):
    """Main application window with sidebar navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarkShark")
        self.setMinimumSize(1000, 700)

        self._setup_menu_bar()
        self._setup_ui()

    def _setup_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_scans_action = QAction("&Open Scans...", self)
        open_scans_action.setShortcut(QKeySequence.StandardKey.Open)
        open_scans_action.triggered.connect(self._on_open_scans)
        file_menu.addAction(open_scans_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About MarkShark", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self):
        """Build the main UI with sidebar and content area."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Splitter for resizable sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Sidebar navigation
        self.nav_list = QListWidget()
        self.nav_list.setMaximumWidth(200)
        self.nav_list.setMinimumWidth(150)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        # Add navigation items
        nav_items = [
            ("Quick Grade", "quick_grade"),
            ("Review & Correct", "review"),
            ("Settings", "settings"),
        ]

        for label, key in nav_items:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.nav_list.addItem(item)

        splitter.addWidget(self.nav_list)

        # Stacked widget for pages
        self.pages = QStackedWidget()

        # Create pages
        self.quick_grade_page = QuickGradePage()
        self.review_page = ReviewPanelPage()
        self.settings_page = SettingsPage()

        self.pages.addWidget(self.quick_grade_page)
        self.pages.addWidget(self.review_page)
        self.pages.addWidget(self.settings_page)

        splitter.addWidget(self.pages)

        # Set splitter proportions
        splitter.setSizes([180, 820])

        # Select first item
        self.nav_list.setCurrentRow(0)

    def _on_nav_changed(self, row: int):
        """Handle navigation selection change."""
        self.pages.setCurrentIndex(row)

    def _on_open_scans(self):
        """Handle File > Open Scans action."""
        # Navigate to Quick Grade and trigger file open
        self.nav_list.setCurrentRow(0)
        self.quick_grade_page.browse_scans()

    def _show_about(self):
        """Show the About dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def navigate_to_review(self, results_data: dict = None):
        """
        Navigate to the Review panel, optionally with data.

        Called from Quick Grade after scoring completes.
        """
        if results_data:
            self.review_page.load_results(results_data)
        self.nav_list.setCurrentRow(1)
