"""
Welcome page - the default landing page for MarkShark.

Guides new users with quick-start navigation cards and shows
recently opened projects for easy access.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QDesktopServices, QCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QSizePolicy,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QDialogButtonBox,
    QGroupBox,
    QFileDialog,
)

from ..widgets import PageHeader

# Try to import MarkShark modules
try:
    from markshark.template_manager import TemplateManager
except ImportError:
    TemplateManager = None

try:
    from ..models.project_registry import ProjectRegistry
except ImportError:
    ProjectRegistry = None


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

_TEAL = "#0E817E"
_BLUE = "#0d6efd"
_CARD_STYLE = """
    QFrame#card {{
        background-color: {bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px;
    }}
    QFrame#card:hover {{
        border: 2px solid {hover};
        background-color: {hover_bg};
    }}
"""

_CARD_CONFIGS = {
    "bubblesheet": {
        "bg": "#f0fdfa",
        "border": "#99e0db",
        "hover": _TEAL,
        "hover_bg": "#e6faf8",
    },
    "grade": {
        "bg": "#eff6ff",
        "border": "#93c5fd",
        "hover": _BLUE,
        "hover_bg": "#dbeafe",
    },
    "review": {
        "bg": "#fef9ee",
        "border": "#fcd481",
        "hover": "#d97706",
        "hover_bg": "#fef3c7",
    },
    "projects": {
        "bg": "#f5f3ff",
        "border": "#c4b5fd",
        "hover": "#7c3aed",
        "hover_bg": "#ede9fe",
    },
}


# ---------------------------------------------------------------------------
# Template picker dialog
# ---------------------------------------------------------------------------

class _TemplatePicker(QDialog):
    """Dialog to browse available bubble sheet templates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Your Bubble Sheet")
        self.setMinimumSize(520, 420)
        self._selected_template = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Intro text
        intro = QLabel(
            "MarkShark includes several ready-to-print bubble sheet templates.\n"
            "Select one below to view details or download the PDF."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: #444; margin-bottom: 8px;")
        layout.addWidget(intro)

        # Template list
        self.template_list = QListWidget()
        self.template_list.setStyleSheet(
            "QListWidget { font-size: 14px; }"
            "QListWidget::item { padding: 8px; }"
            "QListWidget::item:selected { background-color: #e6faf8; color: #000; }"
        )
        self.template_list.currentRowChanged.connect(self._on_selection_changed)
        layout.addWidget(self.template_list, 1)

        # Details panel
        self.details_label = QLabel("Select a template to see details.")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "font-size: 12px; color: #555; padding: 8px; "
            "background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        layout.addWidget(self.details_label)

        # Buttons
        btn_layout = QHBoxLayout()

        self.download_btn = QPushButton("Save Template PDF...")
        self.download_btn.setEnabled(False)
        self.download_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_TEAL}; color: white; "
            f"padding: 8px 16px; border-radius: 6px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: #0a6b68; }}"
            f"QPushButton:disabled {{ background-color: #ccc; color: #888; }}"
        )
        self.download_btn.clicked.connect(self._on_download)
        btn_layout.addWidget(self.download_btn)

        self.open_manager_btn = QPushButton("Open Template Manager")
        self.open_manager_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_BLUE}; color: white; "
            f"padding: 8px 16px; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: #0b5ed7; }}"
        )
        btn_layout.addWidget(self.open_manager_btn)

        btn_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

        # Populate templates
        self._templates = []
        self._load_templates()

    def _load_templates(self):
        if TemplateManager is None:
            self.template_list.addItem("(Template system not available)")
            return

        try:
            tm = TemplateManager()
            self._templates = tm.scan_templates()
        except Exception as e:
            self.template_list.addItem(f"(Error loading templates: {e})")
            return

        if not self._templates:
            self.template_list.addItem("(No templates found)")
            return

        for t in self._templates:
            item = QListWidgetItem(t.display_name)
            item.setData(Qt.ItemDataRole.UserRole, t)
            self.template_list.addItem(item)

    def _on_selection_changed(self, row: int):
        if row < 0 or row >= len(self._templates):
            self.details_label.setText("Select a template to see details.")
            self.download_btn.setEnabled(False)
            self._selected_template = None
            return

        t = self._templates[row]
        self._selected_template = t

        parts = [f"<b>{t.display_name}</b>"]
        if t.description:
            parts.append(t.description)
        if t.num_questions:
            parts.append(f"Questions: {t.num_questions}")
        if t.choices_label:
            parts.append(f"Choices: {t.choices_label}")
        if t.num_pages:
            parts.append(f"Pages: {t.num_pages}")

        self.details_label.setText("<br>".join(parts))
        self.download_btn.setEnabled(True)

    def _on_download(self):
        if self._selected_template is None:
            return

        pdf_path = self._selected_template.template_pdf_path
        if not pdf_path or not pdf_path.exists():
            self.details_label.setText(
                self.details_label.text()
                + "<br><span style='color: red;'>PDF file not found.</span>"
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Template PDF",
            str(Path.home() / f"{self._selected_template.template_id}.pdf"),
            "PDF Files (*.pdf)",
        )

        if save_path:
            import shutil
            try:
                shutil.copy2(str(pdf_path), save_path)
                self.details_label.setText(
                    self.details_label.text()
                    + f"<br><span style='color: green;'>Saved to {save_path}</span>"
                )
            except Exception as e:
                self.details_label.setText(
                    self.details_label.text()
                    + f"<br><span style='color: red;'>Error: {e}</span>"
                )


# ---------------------------------------------------------------------------
# Clickable card widget
# ---------------------------------------------------------------------------

class _ActionCard(QFrame):
    """A clickable card with icon, title, and description."""

    clicked = Signal()

    def __init__(self, icon_text: str, title: str, description: str,
                 color_key: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMinimumSize(200, 140)
        self.setMaximumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        colors = _CARD_CONFIGS.get(color_key, _CARD_CONFIGS["grade"])
        self.setStyleSheet(_CARD_STYLE.format(**colors))

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Icon (text-based, large)
        icon_label = QLabel(icon_text)
        icon_label.setStyleSheet("font-size: 32px; background: transparent; border: none;")
        layout.addWidget(icon_label)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #1a1a1a; "
            "background: transparent; border: none;"
        )
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        layout.addWidget(desc_label)

        layout.addStretch()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Welcome page
# ---------------------------------------------------------------------------

class WelcomePage(QWidget):
    """
    Welcome & Quick Start page.

    Shows quick-start action cards and a list of recent projects.
    Designed to guide new users through the application.
    """

    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Header ---
        header = PageHeader(
            "Welcome to MarkShark",
            "Fast, accurate bubble sheet grading for teachers.",
        )
        layout.addWidget(header)

        # Version
        try:
            from markshark import __version__
            version = __version__
        except ImportError:
            version = "development"

        version_label = QLabel(f"Version {version}")
        version_label.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(version_label)

        # --- Scroll area for content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 0, 8, 8)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # --- Quick Start section ---
        qs_label = QLabel("Quick Start")
        qs_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #333; margin-top: 8px;"
        )
        content_layout.addWidget(qs_label)

        qs_desc = QLabel(
            "Choose an action below to get started. New to MarkShark? "
            "Start by finding your bubble sheet template."
        )
        qs_desc.setWordWrap(True)
        qs_desc.setStyleSheet("font-size: 13px; color: #666; margin-bottom: 8px;")
        content_layout.addWidget(qs_desc)

        # Card row
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(16)

        # Card 1: Find Your Bubblesheet
        self.card_bubblesheet = _ActionCard(
            icon_text="\U0001F4C4",  # page facing up emoji
            title="Find Your Bubble Sheet",
            description="Browse and download ready-to-print bubble sheet templates.",
            color_key="bubblesheet",
        )
        self.card_bubblesheet.clicked.connect(self._on_find_bubblesheet)
        cards_layout.addWidget(self.card_bubblesheet)

        # Card 2: Start Grading
        self.card_grade = _ActionCard(
            icon_text="\U00002705",  # check mark emoji
            title="Start Grading",
            description="Upload scans, select a template, and grade your exams.",
            color_key="grade",
        )
        self.card_grade.clicked.connect(self._on_start_grading)
        cards_layout.addWidget(self.card_grade)

        # Card 3: Review Results
        self.card_review = _ActionCard(
            icon_text="\U0001F50D",  # magnifying glass emoji
            title="Review Results",
            description="Review flagged answers and make corrections.",
            color_key="review",
        )
        self.card_review.clicked.connect(self._on_review_results)
        cards_layout.addWidget(self.card_review)

        # Card 4: Manage Projects
        self.card_projects = _ActionCard(
            icon_text="\U0001F4C1",  # folder emoji
            title="Manage Projects",
            description="Organize your grading projects and archives.",
            color_key="projects",
        )
        self.card_projects.clicked.connect(self._on_manage_projects)
        cards_layout.addWidget(self.card_projects)

        cards_layout.addStretch()
        content_layout.addLayout(cards_layout)

        # --- How It Works section ---
        content_layout.addSpacing(16)

        how_label = QLabel("How It Works")
        how_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #333; margin-top: 8px;"
        )
        content_layout.addWidget(how_label)

        steps = [
            ("1. Print", "Download a bubble sheet template and print copies for your class."),
            ("2. Scan", "After your exam, scan the completed sheets into a single PDF."),
            ("3. Grade", "Open the Grader, select your template and answer key, and click Score."),
            ("4. Review", "Check flagged items, make corrections, and export your final grades."),
        ]

        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(12)
        for step_title, step_desc in steps:
            step_frame = QFrame()
            step_frame.setStyleSheet(
                "QFrame { background-color: #f8f9fa; border: 1px solid #e9ecef; "
                "border-radius: 8px; padding: 12px; }"
            )
            step_layout = QVBoxLayout(step_frame)
            step_layout.setSpacing(4)

            st = QLabel(step_title)
            st.setStyleSheet(
                f"font-size: 14px; font-weight: bold; color: {_TEAL}; "
                "background: transparent; border: none;"
            )
            step_layout.addWidget(st)

            sd = QLabel(step_desc)
            sd.setWordWrap(True)
            sd.setStyleSheet(
                "font-size: 12px; color: #555; background: transparent; border: none;"
            )
            step_layout.addWidget(sd)

            step_layout.addStretch()
            steps_layout.addWidget(step_frame)

        content_layout.addLayout(steps_layout)

        # --- Recent Projects section ---
        content_layout.addSpacing(16)

        rp_label = QLabel("Recent Projects")
        rp_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #333; margin-top: 8px;"
        )
        content_layout.addWidget(rp_label)

        self.projects_container = QVBoxLayout()
        content_layout.addLayout(self.projects_container)

        self._populate_recent_projects()

        content_layout.addStretch()

    # ----- Recent projects -----

    def _populate_recent_projects(self):
        """Load and display recent projects from the registry."""
        # Clear existing
        while self.projects_container.count():
            item = self.projects_container.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        if ProjectRegistry is None:
            empty = QLabel("Project registry not available.")
            empty.setStyleSheet("color: #888; font-size: 13px; padding: 12px;")
            self.projects_container.addWidget(empty)
            return

        try:
            registry = ProjectRegistry()
            projects = registry.list_all()
        except Exception:
            projects = []

        if not projects:
            empty = QLabel(
                "No recent projects yet. Use the Grader to create your first project, "
                "or open the Project Manager to set up a new one."
            )
            empty.setWordWrap(True)
            empty.setStyleSheet(
                "color: #888; font-size: 13px; padding: 16px; "
                "background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 8px;"
            )
            self.projects_container.addWidget(empty)
            return

        # Sort by last_opened descending (most recent first)
        projects.sort(
            key=lambda p: p.get("last_opened", p.get("registered_at", "")),
            reverse=True,
        )

        # Show up to 8 recent projects
        for proj in projects[:8]:
            row = self._make_project_row(proj)
            self.projects_container.addWidget(row)

    def _make_project_row(self, proj: dict) -> QFrame:
        """Create a clickable row widget for a recent project."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #e0e0e0; "
            "border-radius: 8px; padding: 10px; }"
            "QFrame:hover { border-color: #0E817E; background-color: #f0fdfa; }"
        )
        frame.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        row_layout = QHBoxLayout(frame)
        row_layout.setContentsMargins(8, 4, 8, 4)

        # Project info
        info_layout = QVBoxLayout()

        name = proj.get("name", "Unnamed")
        name_label = QLabel(name)
        name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #1a1a1a;")
        info_layout.addWidget(name_label)

        desc = proj.get("description", "")
        path_str = proj.get("path", "")
        sub_parts = []
        if desc:
            sub_parts.append(desc)
        if path_str:
            sub_parts.append(path_str)
        if sub_parts:
            sub_label = QLabel(" \u2022 ".join(sub_parts))
            sub_label.setStyleSheet("font-size: 11px; color: #777;")
            sub_label.setWordWrap(True)
            info_layout.addWidget(sub_label)

        last_opened = proj.get("last_opened", "")
        if last_opened:
            # Show just the date portion
            date_str = last_opened[:10] if len(last_opened) >= 10 else last_opened
            date_label = QLabel(f"Last opened: {date_str}")
            date_label.setStyleSheet("font-size: 10px; color: #999;")
            info_layout.addWidget(date_label)

        row_layout.addLayout(info_layout, 1)

        # Open button
        open_btn = QPushButton("Open")
        open_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_TEAL}; color: white; "
            f"padding: 6px 16px; border-radius: 6px; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #0a6b68; }}"
        )
        project_path = proj.get("path", "")
        open_btn.clicked.connect(lambda checked, p=project_path: self._open_project(p))
        row_layout.addWidget(open_btn, 0, Qt.AlignmentFlag.AlignRight)

        # Also make the whole frame clickable
        frame.mousePressEvent = lambda event, p=project_path: self._open_project(p)

        return frame

    def _open_project(self, project_path: str):
        """Open a project in the Grader."""
        if self._main_window and project_path:
            self._main_window.navigate_to_grader(project_path)

    # ----- Card actions -----

    def _on_find_bubblesheet(self):
        """Open the template picker dialog."""
        dialog = _TemplatePicker(self)
        dialog.open_manager_btn.clicked.connect(
            lambda: self._go_to_page_and_close(dialog, "template_manager")
        )
        dialog.exec()

    def _go_to_page_and_close(self, dialog: QDialog, page_key: str):
        """Navigate to a page and close the dialog."""
        dialog.accept()
        if self._main_window:
            self._main_window._navigate_to_key(page_key)

    def _on_start_grading(self):
        """Navigate to the Grader page."""
        if self._main_window:
            self._main_window._navigate_to_key("quick_grade")

    def _on_review_results(self):
        """Navigate to the Review & Correct page."""
        if self._main_window:
            self._main_window._navigate_to_key("review")

    def _on_manage_projects(self):
        """Navigate to the Project Manager page."""
        if self._main_window:
            self._main_window._navigate_to_key("project_manager")

    def showEvent(self, event):
        """Refresh recent projects when the page becomes visible."""
        super().showEvent(event)
        self._populate_recent_projects()
