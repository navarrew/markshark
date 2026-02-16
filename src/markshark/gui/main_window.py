"""
MarkShark
Main application window with navigation sidebar and stacked pages.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QFont, QColor
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
    QApplication,
)

#import the many pages that will be framed in this main window
from .pages.quick_grade import QuickGradePage
from .pages.review_panel import ReviewPanelPage
from .pages.settings import SettingsPage
from .pages.template_manager import TemplateManagerPage
from .pages.mock_data_utility import MockDataPage
from .pages.map_viewer import MapViewerPage
from .pages.help_page import HelpPage
from .pages.project_manager_page import ProjectManagerPage
from .pages.align_only import AlignOnlyPage
from .pages.score_only import ScoreOnlyPage
from .pages.report_only import ReportOnlyPage
from .pages.pdf_tools import PdfToolsPage
from .pages.lms_integration import LmsIntegrationPage
from .pages.key_builder import KeyBuilderPage
from .pages.welcome_page import WelcomePage

#import the dialog that will be used in the main window and the menu bar
from .dialogs.about import AboutDialog


# Sentinel used to mark divider items in the sidebar
_DIVIDER = "__divider__"


def _make_placeholder(title: str) -> QWidget:
    """Create a simple placeholder page for not-yet-implemented features."""
    page = QWidget()
    layout = QVBoxLayout(page)
    lbl = QLabel(f"{title}\n\n(Coming soon)")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setStyleSheet("color: #888; font-size: 16px;")
    layout.addWidget(lbl)
    return page


class MainWindow(QMainWindow):
    """Main application window with sidebar navigation."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarkShark - easy bubblesheet scoring")
        self.setMinimumSize(1000, 800)
        self.resize(1200, 900)

        # Maps nav-list row → stacked-widget index (skipping dividers)
        self._row_to_page: dict[int, int] = {}
        self._syncing_project = False  # re-entrancy guard for project sync

        self._setup_menu_bar()
        self._setup_ui()
        self._connect_project_selectors()

    def _setup_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # ── File menu ──
        file_menu = menubar.addMenu("&File")

        open_scans_action = QAction("&Open Scans...", self)
        open_scans_action.setShortcut(QKeySequence.StandardKey.Open)
        open_scans_action.triggered.connect(self._on_open_scans)
        file_menu.addAction(open_scans_action)

        file_menu.addSeparator()

        settings_action = QAction("&Settings...", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self._on_open_settings)
        file_menu.addAction(settings_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # ── Grade menu ──
        grade_menu = menubar.addMenu("&Grade")

        grader_action = QAction("&Grader", self)
        grader_action.triggered.connect(lambda: self._navigate_to_key("quick_grade"))
        grade_menu.addAction(grader_action)

        review_action = QAction("&Review && Correct", self)
        review_action.triggered.connect(lambda: self._navigate_to_key("review"))
        grade_menu.addAction(review_action)

        grade_menu.addSeparator()

        align_action = QAction("&Align Only", self)
        align_action.triggered.connect(lambda: self._navigate_to_key("align_only"))
        grade_menu.addAction(align_action)

        score_action = QAction("&Score Only", self)
        score_action.triggered.connect(lambda: self._navigate_to_key("score_only"))
        grade_menu.addAction(score_action)

        report_action = QAction("Report &Only", self)
        report_action.triggered.connect(lambda: self._navigate_to_key("report_only"))
        grade_menu.addAction(report_action)

        # ── Utilities menu ──
        util_menu = menubar.addMenu("&Utilities")

        template_action = QAction("&Template Manager", self)
        template_action.triggered.connect(lambda: self._navigate_to_key("template_manager"))
        util_menu.addAction(template_action)

        project_action = QAction("&Course Manager", self)
        project_action.triggered.connect(lambda: self._navigate_to_key("project_manager"))
        util_menu.addAction(project_action)

        key_builder_action = QAction("Answer &Key Utility", self)
        key_builder_action.triggered.connect(lambda: self._navigate_to_key("key_builder"))
        util_menu.addAction(key_builder_action)

        util_menu.addSeparator()

        mock_action = QAction("&Mock Data Utility", self)
        mock_action.triggered.connect(lambda: self._navigate_to_key("mock_data"))
        util_menu.addAction(mock_action)

        map_action = QAction("&Bubblemap Utility", self)
        map_action.triggered.connect(lambda: self._navigate_to_key("map_viewer"))
        util_menu.addAction(map_action)

        lms_action = QAction("&LMS Integration", self)
        lms_action.triggered.connect(lambda: self._navigate_to_key("lms_integration"))
        util_menu.addAction(lms_action)

        # ── Window menu ──
        window_menu = menubar.addMenu("&Window")

        reset_window_action = QAction("&Reset Window Size && Center", self)
        reset_window_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        reset_window_action.triggered.connect(self._reset_window)
        window_menu.addAction(reset_window_action)

        # ── Help menu ──
        help_menu = menubar.addMenu("&Help")

        help_page_action = QAction("&Help && Documentation", self)
        help_page_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_page_action.triggered.connect(lambda: self._navigate_to_key("help"))
        help_menu.addAction(help_page_action)

        help_menu.addSeparator()

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
        self.nav_list.setMaximumWidth(220)
        self.nav_list.setMinimumWidth(170)
        nav_font = self.nav_list.font()
        nav_font.setPointSize(16)
        self.nav_list.setFont(nav_font)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)

        # Stacked widget for pages
        self.pages = QStackedWidget()

        # Create pages
        self.welcome_page = WelcomePage(main_window=self)
        self.quick_grade_page = QuickGradePage()
        self.review_page = ReviewPanelPage()
        self.template_manager_page = TemplateManagerPage()
        self.mock_data_page = MockDataPage()
        self.map_viewer_page = MapViewerPage()
        self.settings_page = SettingsPage()
        self.help_page = HelpPage()
        self.project_manager_page = ProjectManagerPage(main_window=self)
        self.align_only_page = AlignOnlyPage()
        self.score_only_page = ScoreOnlyPage()
        self.report_only_page = ReportOnlyPage()
        self.pdf_tools_page = PdfToolsPage()
        self.key_builder_page = KeyBuilderPage()
        self.lms_integration_page = LmsIntegrationPage()

        # Define sidebar structure: (label, key, widget)
        # Use _DIVIDER as a separator between groups.
        nav_entries = [
            # --- Welcome ---
            ("Welcome",           "welcome",          self.welcome_page),
            # --- Divider ---
            _DIVIDER,
            # --- Main ---
            ("Grader",            "quick_grade",      self.quick_grade_page),
            ("Review & Correct",  "review",           self.review_page),
            # --- Divider ---
            _DIVIDER,
            # --- Standalone functions ---
            ("Align Only",        "align_only",       self.align_only_page),
            ("Score Only",        "score_only",       self.score_only_page),
            ("Report Only",       "report_only",      self.report_only_page),
            # --- Divider ---
            _DIVIDER,
            # --- Utilities ---
            ("Course Manager",     "project_manager", self.project_manager_page),
            ("Template Manager",  "template_manager", self.template_manager_page),
            ("LMS Integration",   "lms_integration",  self.lms_integration_page),
            # --- Divider ---
            _DIVIDER,
            # --- Utilities ---
            ("Answer Key Utility", "key_builder",     self.key_builder_page),
            ("PDF Tools",         "pdf_tools",        self.pdf_tools_page),
            ("Mock Data Utility", "mock_data",        self.mock_data_page),
            ("Bubblemap Utility", "map_viewer",       self.map_viewer_page),
            # --- Divider ---
            _DIVIDER,
            # --- Bottom ---
            ("Help",              "help",             self.help_page),
            ("Settings",          "settings",         self.settings_page),
        ]

        page_index = 0
        for entry in nav_entries:
            if entry is _DIVIDER:
                # Add a visual divider row
                divider_item = QListWidgetItem()
                divider_item.setText("─" * 20)
                divider_item.setFlags(Qt.ItemFlag.NoItemFlags)  # not selectable
                divider_item.setForeground(QColor("#999999"))
                font = divider_item.font()
                font.setPointSize(8)
                divider_item.setFont(font)
                divider_item.setSizeHint(divider_item.sizeHint().__class__(
                    divider_item.sizeHint().width(), 20
                ))
                self.nav_list.addItem(divider_item)
                # no mapping for this row
            else:
                label, key, widget = entry
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.nav_list.addItem(item)
                self.pages.addWidget(widget)
                self._row_to_page[self.nav_list.count() - 1] = page_index
                page_index += 1

        splitter.addWidget(self.nav_list)
        splitter.addWidget(self.pages)

        # Set splitter proportions
        splitter.setSizes([180, 820])

        # Select first item
        self.nav_list.setCurrentRow(0)

    # ---- Project selector synchronization ----

    def _connect_project_selectors(self):
        """Wire up all ProjectSelector instances so they stay in sync.

        When a teacher changes the project on *any* page, all other pages
        update to match, preventing accidental work in the wrong project.
        """
        self._project_selector_pages = [
            page for page in (
                self.quick_grade_page,
                self.review_page,
                self.align_only_page,
                self.score_only_page,
                self.report_only_page,
                self.mock_data_page,
                self.key_builder_page,
            )
            if hasattr(page, 'project_selector')
        ]

        for page in self._project_selector_pages:
            ps = page.project_selector
            ps.working_dir_changed.connect(
                lambda _path, src=ps: self._sync_project_from(src)
            )
            ps.project_changed.connect(
                lambda _name, src=ps: self._sync_project_from(src)
            )

    def _sync_project_from(self, source_selector):
        """Propagate a project change from *source_selector* to all others."""
        if self._syncing_project:
            return
        self._syncing_project = True
        try:
            workdir = source_selector.working_dir()
            project = source_selector.project_name()

            for page in self._project_selector_pages:
                ps = page.project_selector
                if ps is source_selector:
                    continue
                # Only update if actually different to avoid spurious signals
                if workdir and ps.working_dir() != workdir:
                    ps.set_working_dir(str(workdir))
                if ps.project_name() != project:
                    ps.set_project(project)
        finally:
            self._syncing_project = False

    def _sync_visible_page(self):
        """Ensure the newly-visible page's ProjectSelector matches the others.

        Called on every page switch so the project header bar is always
        up-to-date, even if a signal-based sync was missed.
        """
        if not hasattr(self, "_project_selector_pages"):
            return  # called before _connect_project_selectors()

        current_widget = self.pages.currentWidget()
        if not hasattr(current_widget, "project_selector"):
            return

        # Find a reference selector (the first OTHER page with a project set)
        ref = None
        for page in self._project_selector_pages:
            if page is current_widget:
                continue
            ps = page.project_selector
            if ps.working_dir():
                ref = ps
                break

        if ref is None:
            return

        target = current_widget.project_selector
        workdir = ref.working_dir()
        project = ref.project_name()

        self._syncing_project = True
        try:
            if workdir and target.working_dir() != workdir:
                target.set_working_dir(str(workdir))
            if target.project_name() != project:
                target.set_project(project)
        finally:
            self._syncing_project = False

    # ---- Navigation ----

    def _on_nav_changed(self, row: int):
        """Handle navigation selection change."""
        page_idx = self._row_to_page.get(row)
        if page_idx is not None:
            self.pages.setCurrentIndex(page_idx)
            self._sync_visible_page()

    def _on_open_scans(self):
        """Handle File > Open Scans action."""
        # Navigate to Quick Grade and trigger file open
        self.nav_list.setCurrentRow(0)
        self.quick_grade_page.browse_scans()

    def _navigate_to_key(self, key: str):
        """Navigate to a sidebar page by its key (e.g. 'settings', 'review')."""
        for row in range(self.nav_list.count()):
            item = self.nav_list.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == key:
                self.nav_list.setCurrentRow(row)
                return

    def _on_open_settings(self):
        """Handle File > Settings action."""
        self._navigate_to_key("settings")

    def _reset_window(self):
        """Reset the window to its default size and centre it on screen."""
        default_w, default_h = 1200, 800
        self.resize(default_w, default_h)

        screen = self.screen()
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - default_w) // 2
            y = geo.y() + (geo.height() - default_h) // 2
            self.move(x, y)

    def _show_about(self):
        """Show the About dialog."""
        dialog = AboutDialog(self)
        dialog.exec()

    def navigate_to_grader(self, project_path):
        """
        Navigate to the Grader page with a specific project loaded.

        Called from the Project Manager's "Open in Grader" button.
        """
        from pathlib import Path

        project_path = Path(project_path)
        self.quick_grade_page.project_selector.set_working_dir(
            str(project_path.parent)
        )
        self.quick_grade_page.project_selector.set_project(project_path.name)
        self._navigate_to_key("quick_grade")

    def navigate_to_review(self, results_data: dict = None):
        """
        Navigate to the Review panel, optionally with data.

        Called from Grader after scoring completes.
        """
        if results_data:
            self.review_page.load_results(results_data)
        self._navigate_to_key("review")
