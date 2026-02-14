"""
Welcome page - the default landing page for MarkShark.

Guides new users with quick-start navigation cards and shows
recently opened projects for easy access.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QFontDatabase
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
)

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
            from ..utils import safe_copy_file
            try:
                safe_copy_file(pdf_path, save_path)
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

        # --- Header: branded "MarkShark" wordmark ---
        # Load Poppins from the project's bundled assets
        _FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
        for variant in ("Poppins-Medium.ttf", "Poppins-Regular.ttf"):
            font_file = _FONT_DIR / variant
            if font_file.exists():
                QFontDatabase.addApplicationFont(str(font_file))

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)

        wordmark = QLabel(
            "<span style=\"color: white;\">Mark</span>"
            "<span style=\"color: #0d6efd;\">Shark</span>"
        )
        wordmark.setStyleSheet(
            "font-family: 'Poppins', sans-serif; "
            "font-size: 48px; font-weight: 500;"
        )
        header_row.addWidget(wordmark)

        # Subtitle + version on the right, bottom-aligned
        from ..utils import get_app_version
        subtitle_col = QVBoxLayout()
        subtitle_col.addStretch()
        subtitle = QLabel("Fast, accurate bubble sheet grading for teachers.")
        subtitle.setStyleSheet("color: #aaa; font-size: 13px;")
        subtitle_col.addWidget(subtitle)
        version_label = QLabel(f"v{get_app_version()}")
        version_label.setStyleSheet("color: #666; font-size: 11px;")
        subtitle_col.addWidget(version_label)
        header_row.addLayout(subtitle_col)

        header_row.addStretch()
        layout.addLayout(header_row)

        # --- Scroll area for content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 0, 8, 8)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        # --- Top tiles: two side-by-side cards ---
        top_tiles = QHBoxLayout()
        top_tiles.setSpacing(16)

        # ── Left column: two stacked tiles ──
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # ── Top-left tile: Bubble Sheet Templates ──
        tpl_tile = QFrame()
        tpl_tile.setObjectName("tpl_tile")
        tpl_tile.setStyleSheet(
            "QFrame#tpl_tile { background-color: #f0fdfa; "
            "border: 1px solid #99e0db; border-radius: 12px; }"
        )
        tpl_tile.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        tpl_layout = QVBoxLayout(tpl_tile)
        tpl_layout.setContentsMargins(20, 14, 20, 14)
        tpl_layout.setSpacing(6)

        tpl_title = QLabel("Bubble Sheet Templates")
        tpl_title.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {_TEAL}; "
            "background: transparent; border: none;"
        )
        tpl_layout.addWidget(tpl_title)

        tpl_desc = QLabel(
            "Browse ready-to-print bubble sheets, pick a favorite, "
            "and download the PDF."
        )
        tpl_desc.setWordWrap(True)
        tpl_desc.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        tpl_layout.addWidget(tpl_desc)

        tpl_layout.addStretch()

        tpl_btn_row = QHBoxLayout()
        tpl_browse_btn = QPushButton("Browse Templates")
        tpl_browse_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_TEAL}; color: white; "
            f"padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #0a6b68; }}"
        )
        tpl_browse_btn.clicked.connect(self._on_find_bubblesheet)
        tpl_btn_row.addWidget(tpl_browse_btn)
        tpl_btn_row.addStretch()
        tpl_layout.addLayout(tpl_btn_row)

        left_col.addWidget(tpl_tile, 1)

        # ── Bottom-left tile: Build Your Answer Key ──
        key_tile = QFrame()
        key_tile.setObjectName("key_tile")
        key_tile.setStyleSheet(
            "QFrame#key_tile { background-color: #fef9ef; "
            "border: 1px solid #f0d89a; border-radius: 12px; }"
        )
        key_tile.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        key_layout = QVBoxLayout(key_tile)
        key_layout.setContentsMargins(20, 14, 20, 14)
        key_layout.setSpacing(6)

        key_title = QLabel("Build Your Answer Key")
        key_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #b8860b; "
            "background: transparent; border: none;"
        )
        key_layout.addWidget(key_title)

        key_desc = QLabel(
            "Create and manage answer keys for your bubble sheets. "
            "Coming soon!"
        )
        key_desc.setWordWrap(True)
        key_desc.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        key_layout.addWidget(key_desc)

        key_layout.addStretch()

        key_btn_row = QHBoxLayout()
        key_build_btn = QPushButton("Key Builder")
        key_build_btn.setStyleSheet(
            "QPushButton { background-color: #b8860b; color: white; "
            "padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background-color: #9a7209; }"
        )
        key_build_btn.clicked.connect(self._on_key_builder)
        key_btn_row.addWidget(key_build_btn)
        key_btn_row.addStretch()
        key_layout.addLayout(key_btn_row)

        left_col.addWidget(key_tile, 1)

        top_tiles.addLayout(left_col, 1)

        # ── Right tile: Getting Started ──
        gs_tile = QFrame()
        gs_tile.setObjectName("gs_tile")
        gs_tile.setStyleSheet(
            "QFrame#gs_tile { background-color: #eff6ff; "
            "border: 1px solid #93c5fd; border-radius: 12px; }"
        )
        gs_layout = QVBoxLayout(gs_tile)
        gs_layout.setContentsMargins(20, 18, 20, 18)
        gs_layout.setSpacing(10)

        gs_title = QLabel("Getting Started")
        gs_title.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {_BLUE}; "
            "background: transparent; border: none;"
        )
        gs_layout.addWidget(gs_title)

        steps = [
            ("\U0001F5A8  Print", "Download a bubble sheet template and print copies for your class."),
            ("\U0001F4E0  Scan", "After your exam, scan the completed sheets into a single PDF."),
            ("\u2705  Grade", "Open the Grader, select your template and answer key, and click Score."),
            ("\U0001F50D  Review", "Check flagged items, make corrections, and export your final grades."),
            ("\U0001F4CA  Report", "Generate a summary report to see class performance, question difficulty, and score distributions."),
        ]

        for step_title, step_desc in steps:
            step_row = QHBoxLayout()
            step_row.setSpacing(8)

            st = QLabel(f"<b>{step_title}</b>")
            st.setStyleSheet(
                "font-size: 13px; color: #1a1a1a; background: transparent; border: none;"
            )
            st.setFixedWidth(110)
            step_row.addWidget(st)

            sd = QLabel(step_desc)
            sd.setWordWrap(True)
            sd.setStyleSheet(
                "font-size: 12px; color: #555; background: transparent; border: none;"
            )
            step_row.addWidget(sd, 1)

            gs_layout.addLayout(step_row)

        gs_layout.addStretch()

        top_tiles.addWidget(gs_tile, 1)

        content_layout.addLayout(top_tiles)

        # --- "Where MarkShark keeps your files" tile ---
        content_layout.addSpacing(16)

        workdir_tile = QFrame()
        workdir_tile.setObjectName("workdir_tile")
        workdir_tile.setStyleSheet(
            "QFrame#workdir_tile { background-color: #f8f8f8; "
            "border: 1px solid #ddd; border-radius: 10px; }"
        )
        wd_layout = QVBoxLayout(workdir_tile)
        wd_layout.setContentsMargins(20, 16, 20, 16)
        wd_layout.setSpacing(8)

        wd_title = QLabel("Where MarkShark Keeps Your Files")
        wd_title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #1a1a1a; "
            "background: transparent; border: none;"
        )
        wd_layout.addWidget(wd_title)

        wd_desc = QLabel(
            "Set a <b>working directory</b> for each course you teach "
            "(e.g. <i>Biology 101</i> or <i>AP History</i>). "
            "Inside that folder, each <b>project</b> is one test — "
            "like <i>Midterm 1</i> or <i>Final Exam 2025</i>. "
            "MarkShark creates the sub-folders automatically."
        )
        wd_desc.setWordWrap(True)
        wd_desc.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        wd_layout.addWidget(wd_desc)

        # Current working directory readout
        wd_current_row = QHBoxLayout()
        wd_current_label = QLabel("Your current working directory:")
        wd_current_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #333; "
            "background: transparent; border: none;"
        )
        wd_current_row.addWidget(wd_current_label)

        self.wd_path_label = QLabel()
        self.wd_path_label.setStyleSheet(
            "font-size: 12px; color: #777; font-style: italic; "
            "background: transparent; border: none;"
        )
        wd_current_row.addWidget(self.wd_path_label, 1)
        wd_layout.addLayout(wd_current_row)

        content_layout.addWidget(workdir_tile)

        # --- Recent Projects section ---
        content_layout.addSpacing(16)

        self.projects_container = QVBoxLayout()
        content_layout.addLayout(self.projects_container)

        self._populate_recent_projects()
        self._refresh_working_dir_label()

        content_layout.addStretch()

    # ----- Working directory -----

    def _refresh_working_dir_label(self):
        """Update the working-directory readout from SettingsStore."""
        from ..models.settings_store import SettingsStore
        settings = SettingsStore()
        workdir = settings.value("project/working_dir", "")
        if workdir and Path(workdir).is_dir():
            self.wd_path_label.setText(str(workdir))
            self.wd_path_label.setStyleSheet(
                "font-size: 12px; color: #1a1a1a; font-style: normal; "
                "background: transparent; border: none;"
            )
        else:
            self.wd_path_label.setText("(none currently selected)")
            self.wd_path_label.setStyleSheet(
                "font-size: 12px; color: #999; font-style: italic; "
                "background: transparent; border: none;"
            )

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

        # Single tile containing all project rows
        tile = QFrame()
        tile.setStyleSheet(
            "QFrame#projects_tile { background-color: #ffffff; "
            "border: 1px solid #e0e0e0; border-radius: 10px; }"
        )
        tile.setObjectName("projects_tile")
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        tile_layout.setSpacing(0)

        # Header row inside the tile
        header_widget = QWidget()
        header_widget.setStyleSheet(
            f"background-color: {_TEAL}; "
            f"border-top-left-radius: 10px; border-top-right-radius: 10px;"
        )
        header_row = QHBoxLayout(header_widget)
        header_row.setContentsMargins(14, 8, 10, 8)

        header_label = QLabel("Recent Projects")
        header_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: white; background: transparent;"
        )
        header_row.addWidget(header_label)

        header_row.addStretch()

        new_proj_btn = QPushButton("+ New Project")
        new_proj_btn.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,0.2); color: white; "
            "padding: 4px 12px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(255,255,255,0.4); }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.35); }"
        )
        new_proj_btn.clicked.connect(self._on_new_project)
        header_row.addWidget(new_proj_btn)

        manage_btn = QPushButton("Manage All Projects")
        manage_btn.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,0.2); color: white; "
            "padding: 4px 12px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(255,255,255,0.4); }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.35); }"
        )
        manage_btn.clicked.connect(self._on_manage_projects)
        header_row.addWidget(manage_btn)

        tile_layout.addWidget(header_widget)

        # Show up to 8 recent projects as compact rows inside the tile
        display = projects[:8]
        for i, proj in enumerate(display):
            row = self._make_project_row(proj)
            tile_layout.addWidget(row)
            # Add a thin separator between rows (not after the last)
            if i < len(display) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("color: #e9ecef;")
                sep.setFixedHeight(1)
                tile_layout.addWidget(sep)

        self.projects_container.addWidget(tile)

    def _make_project_row(self, proj: dict) -> QWidget:
        """Create a compact single-line row for a project inside the tile."""
        project_path = proj.get("path", "")
        missing = not Path(project_path).is_dir() if project_path else True

        row = QWidget()
        row.setStyleSheet(
            "QWidget { background: transparent; }"
            "QWidget:hover { background-color: #f0fdfa; }"
        )
        if not missing:
            row.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(14, 8, 14, 8)
        row_layout.setSpacing(16)

        # Project name (bold), with warning flag if missing
        name = proj.get("name", "Unnamed")
        if missing:
            name_label = QLabel(f"\u26A0 <b>{name}</b>")
            name_label.setToolTip("Project folder not found — it may have been moved or deleted.")
            name_label.setStyleSheet("font-size: 13px; color: #b91c1c;")
        else:
            name_label = QLabel(f"<b>{name}</b>")
            name_label.setStyleSheet("font-size: 13px; color: #1a1a1a;")
        name_label.setFixedWidth(170)
        row_layout.addWidget(name_label)

        # Directory path
        path_label = QLabel(project_path)
        if missing:
            path_label.setStyleSheet("font-size: 11px; color: #b91c1c;")
        else:
            path_label.setStyleSheet("font-size: 11px; color: #777;")
        path_label.setWordWrap(False)
        row_layout.addWidget(path_label, 1)

        # Last opened date
        last_opened = proj.get("last_opened", "")
        if last_opened:
            date_str = last_opened[:10] if len(last_opened) >= 10 else last_opened
        else:
            date_str = ""
        date_label = QLabel(date_str)
        date_label.setStyleSheet("font-size: 11px; color: #999;")
        date_label.setFixedWidth(80)
        row_layout.addWidget(date_label)

        # Open Folder button
        folder_btn = QPushButton("Open Folder")
        folder_btn.setStyleSheet(
            "QPushButton { background-color: #e9ecef; color: #333; "
            "padding: 4px 14px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background-color: #dee2e6; }"
            "QPushButton:disabled { background-color: #f5f5f5; color: #bbb; }"
        )
        folder_btn.setEnabled(not missing)
        folder_btn.clicked.connect(lambda checked, p=project_path: self._open_folder(p))
        row_layout.addWidget(folder_btn)

        # Load button
        load_btn = QPushButton("Load")
        load_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_TEAL}; color: white; "
            f"padding: 4px 14px; border-radius: 4px; font-size: 11px; }}"
            f"QPushButton:hover {{ background-color: #0a6b68; }}"
            f"QPushButton:disabled {{ background-color: #ccc; color: #888; }}"
        )
        load_btn.setEnabled(not missing)
        load_btn.clicked.connect(lambda checked, p=project_path: self._open_project(p))
        row_layout.addWidget(load_btn)

        # Make the whole row clickable (only if path exists)
        if not missing:
            row.mousePressEvent = lambda event, p=project_path: self._open_project(p)

        return row

    def _open_folder(self, project_path: str):
        """Open the project folder in the system file manager."""
        if project_path:
            from ..utils import open_file_or_folder
            open_file_or_folder(project_path)

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

    def _on_key_builder(self):
        """Navigate to the Key Build Utility page."""
        if self._main_window:
            self._main_window._navigate_to_key("key_builder")

    def _go_to_page_and_close(self, dialog: QDialog, page_key: str):
        """Navigate to a page and close the dialog."""
        dialog.accept()
        if self._main_window:
            self._main_window._navigate_to_key(page_key)

    def _on_new_project(self):
        """Create a new project and open it in the Grader."""
        from ..utils import create_new_project

        project_path = create_new_project(parent_widget=self)
        if project_path and self._main_window:
            self._main_window.navigate_to_grader(str(project_path))
            self._populate_recent_projects()

    def _on_manage_projects(self):
        """Navigate to the Project Manager page."""
        if self._main_window:
            self._main_window._navigate_to_key("project_manager")

    def showEvent(self, event):
        """Refresh recent projects and working dir when the page becomes visible."""
        super().showEvent(event)
        self._populate_recent_projects()
        self._refresh_working_dir_label()
