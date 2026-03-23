"""
Welcome page - the default landing page for MarkShark.

Guides new users with quick-start navigation cards and shows
recently opened projects for easy access.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QFontDatabase, QPixmap
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
    QMessageBox,
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

    # Max dimensions for the preview thumbnail inside the dialog
    _PREVIEW_MAX_W = 200
    _PREVIEW_MAX_H = 280

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Your Bubble Sheet")
        self.setMinimumSize(640, 480)
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
        intro.setStyleSheet("font-size: 13px; color: white; margin-bottom: 8px;")
        layout.addWidget(intro)

        # ── Middle area: template list (left) + details/preview (right) ──
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # Template list (left side)
        self.template_list = QListWidget()
        self.template_list.setStyleSheet(
            "QListWidget { font-size: 14px; }"
            "QListWidget::item { padding: 8px; }"
            "QListWidget::item:selected { background-color: #e6faf8; color: #000; }"
        )
        self.template_list.currentRowChanged.connect(self._on_selection_changed)
        middle.addWidget(self.template_list, 3)

        # Right side: details text + preview image (stacked vertically)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Details text
        self.details_label = QLabel("Select a template to see details.")
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.details_label.setStyleSheet(
            "font-size: 12px; color: #555; padding: 8px; "
            "background-color: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 6px;"
        )
        right_panel.addWidget(self.details_label)

        # Preview image
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 6px;"
        )
        self.preview_label.setFixedSize(self._PREVIEW_MAX_W, self._PREVIEW_MAX_H)
        self.preview_label.hide()  # hidden until a template with a preview is selected
        right_panel.addWidget(
            self.preview_label, 0, Qt.AlignmentFlag.AlignHCenter
        )

        right_panel.addStretch()
        middle.addLayout(right_panel, 2)

        layout.addLayout(middle, 1)

        # Buttons
        btn_layout = QHBoxLayout()

        self.download_btn = QPushButton("Save Bubble Sheet PDF...")
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
        close_btn.setStyleSheet(
            "QPushButton { background-color: #e0e0e0; color: #333; "
            "padding: 8px 16px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
        )
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
            self.preview_label.hide()
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

        # Show preview image if available
        self._update_preview(t)

    def _update_preview(self, template):
        """Load and display the template preview image, or hide the label."""
        img_path = template.preview_image_path
        if img_path and img_path.exists():
            pixmap = QPixmap(str(img_path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self._PREVIEW_MAX_W,
                    self._PREVIEW_MAX_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.preview_label.setPixmap(scaled)
                self.preview_label.show()
                return
        # No preview available — hide the label entirely
        self.preview_label.clear()
        self.preview_label.hide()

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
# Tutorial dialog
# ---------------------------------------------------------------------------

class _TutorialDialog(QDialog):
    """Dialog offering tutorial PDF and sample dataset downloads.

    Tutorial assets are expected in assets/tutorial/:
      - tutorial.pdf           — walkthrough guide
      - sample_scans.pdf       — scanned bubble sheets for practice
      - sample_answer_key.txt  — answer key that matches the sample scans
      - sample_roster.csv      — class roster with some deliberate mismatches
    Each button auto-disables if its file is not yet present, so the
    dialog still works while assets are being prepared.
    """

    _ASSETS = Path(__file__).resolve().parent.parent.parent / "assets" / "tutorial"

    # Each sample file: (asset_filename, button_label, save_name, file_filter)
    _SAMPLE_FILES = [
        ("sample_scans.pdf", "Sample Scans", "sample_scans.pdf", "PDF Files (*.pdf)"),
        ("sample_answer_key.txt", "Sample Answer Key", "sample_answer_key.txt", "Text Files (*.txt)"),
        ("sample_roster.csv", "Sample Roster", "sample_roster.csv", "CSV Files (*.csv)"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MarkShark Tutorial")
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Intro
        intro = QLabel(
            "New to MarkShark?  Download the tutorial PDF for a step-by-step "
            "walkthrough, and grab the sample files to follow along."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: white; margin-bottom: 4px;")
        layout.addWidget(intro)

        # ── Tutorial PDF row ──
        pdf_frame = QFrame()
        pdf_frame.setStyleSheet(
            "QFrame { background-color: #eff6ff; border: 1px solid #93c5fd; "
            "border-radius: 8px; }"
        )
        pdf_layout = QHBoxLayout(pdf_frame)
        pdf_layout.setContentsMargins(14, 10, 14, 10)

        pdf_label = QLabel(
            "<b>Tutorial PDF</b><br>"
            "<span style='font-size: 11px; color: #555;'>"
            "A printable guide covering setup, scoring, and reports.</span>"
        )
        pdf_label.setWordWrap(True)
        pdf_label.setStyleSheet("color: #1a1a1a; background: transparent; border: none;")
        pdf_layout.addWidget(pdf_label, 1)

        self._pdf_path = self._ASSETS / "tutorial.pdf"
        pdf_btn = QPushButton("Download PDF")
        pdf_btn.setEnabled(self._pdf_path.exists())
        pdf_btn.setToolTip(
            str(self._pdf_path) if self._pdf_path.exists()
            else "tutorial.pdf not yet available"
        )
        pdf_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_BLUE}; color: white; "
            f"padding: 6px 14px; border-radius: 6px; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: #0b5ed7; }}"
            f"QPushButton:disabled {{ background-color: #ccc; color: #888; }}"
        )
        pdf_btn.clicked.connect(self._on_download_pdf)
        pdf_layout.addWidget(pdf_btn)

        layout.addWidget(pdf_frame)

        # ── Sample data files ──
        # One teal-themed frame with a description and a row of download
        # buttons — one per file.  Each button is independently enabled
        # based on whether the asset exists on disk.
        data_frame = QFrame()
        data_frame.setStyleSheet(
            "QFrame { background-color: #f0fdfa; border: 1px solid #99e0db; "
            "border-radius: 8px; }"
        )
        data_layout = QVBoxLayout(data_frame)
        data_layout.setContentsMargins(14, 10, 14, 10)
        data_layout.setSpacing(8)

        data_label = QLabel(
            "<b>Sample Data Files</b><br>"
            "<span style='font-size: 11px; color: #555;'>"
            "Practice scans, answer key, and class roster so you can try "
            "MarkShark right away.  The roster includes deliberate ID "
            "mismatches so you can see how absent students and orphan "
            "scans are flagged.</span>"
        )
        data_label.setWordWrap(True)
        data_label.setStyleSheet("color: #1a1a1a; background: transparent; border: none;")
        data_layout.addWidget(data_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._sample_paths: dict[str, Path] = {}
        for asset_name, btn_label, save_name, file_filter in self._SAMPLE_FILES:
            asset_path = self._ASSETS / asset_name
            self._sample_paths[asset_name] = asset_path

            btn = QPushButton(btn_label)
            btn.setEnabled(asset_path.exists())
            btn.setToolTip(
                str(asset_path) if asset_path.exists()
                else f"{asset_name} not yet available"
            )
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {_TEAL}; color: white; "
                f"padding: 6px 14px; border-radius: 6px; font-weight: bold; }}"
                f"QPushButton:hover {{ background-color: #0a6b68; }}"
                f"QPushButton:disabled {{ background-color: #ccc; color: #888; }}"
            )
            # Capture loop variables with default args so each lambda
            # binds to the correct path / name / filter.
            btn.clicked.connect(
                lambda checked, p=asset_path, s=save_name, f=file_filter:
                    self._save_asset(p, s, f)
            )
            btn_row.addWidget(btn)

        btn_row.addStretch()
        data_layout.addLayout(btn_row)

        layout.addWidget(data_frame)

        # ── Status label (shows save confirmations / errors) ──
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.status_label)

        # ── Close button ──
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #e0e0e0; color: #333; "
            "padding: 8px 16px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
        )
        close_btn.clicked.connect(self.reject)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

    # ── Download helpers ──

    def _save_asset(self, source: Path, suggested_name: str, file_filter: str):
        """Prompt user for a save location and copy *source* there."""
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save File",
            str(Path.home() / suggested_name),
            file_filter,
        )
        if not save_path:
            return  # user cancelled

        from ..utils import safe_copy_file
        try:
            safe_copy_file(source, save_path)
            self.status_label.setText(
                f"<span style='color: green;'>Saved to {save_path}</span>"
            )
        except Exception as e:
            self.status_label.setText(
                f"<span style='color: red;'>Error: {e}</span>"
            )

    def _on_download_pdf(self):
        self._save_asset(
            self._pdf_path, "MarkShark_Tutorial.pdf", "PDF Files (*.pdf)"
        )


# ---------------------------------------------------------------------------
# First-run welcome dialog
# ---------------------------------------------------------------------------

class _FirstRunDialog(QDialog):
    """One-time onboarding dialog shown when MarkShark is opened for the first time.

    Detected via the ``app/onboarding_dismissed`` flag in SettingsStore.

    Three outcomes (checked by the caller via ``result``):
      - **Don't show again** → ``accept()`` — caller persists the flag
      - **Download Tutorial** → ``accept()`` with ``_open_tutorial`` set
        — caller persists the flag and opens the tutorial dialog
      - **Maybe Later** / window close → ``reject()`` — caller does NOT
        persist the flag, so the dialog reappears next launch

    The dialog uses the ``Qt.WindowType.Window`` flag so it appears as
    its own top-level window.  This lets the user see the MarkShark
    main window behind it rather than blocking the entire view.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to MarkShark!")
        self.setMinimumWidth(460)
        # Non-modal so the user can see and interact with the main
        # window behind this dialog.  We avoid Qt.WindowType.Window
        # because detaching a dialog from its parent's widget tree
        # causes a use-after-free crash on macOS when Python GC's the
        # dialog while Qt still references it in showChildren().
        self.setModal(False)
        self._open_tutorial = False  # set True if user clicks the tutorial button
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # ── Greeting ──
        greeting = QLabel("Welcome!")
        greeting.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {_BLUE};"
        )
        greeting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(greeting)

        body = QLabel(
            "It looks like you might be new to MarkShark!\n\n"
            "We have lots of help to get you started, including a "
            "tutorial with sample student data that walks you through "
            "the entire grading process — from scanning to reports."
        )
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 13px; color: white;")
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(body)

        layout.addSpacing(4)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        later_btn = QPushButton("Maybe Later")
        later_btn.setStyleSheet(
            "QPushButton { background-color: #e0e0e0; color: #333; "
            "padding: 8px 16px; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
        )
        later_btn.clicked.connect(self.reject)
        btn_layout.addWidget(later_btn)

        dismiss_btn = QPushButton("Don't show this again")
        dismiss_btn.setStyleSheet(
            "QPushButton { background-color: #6c757d; color: white; "
            "padding: 8px 16px; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #565e64; }"
        )
        dismiss_btn.clicked.connect(self.accept)
        btn_layout.addWidget(dismiss_btn)

        tutorial_btn = QPushButton("Download Tutorial")
        tutorial_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_BLUE}; color: white; "
            f"padding: 8px 16px; border-radius: 6px; font-weight: bold; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #0b5ed7; }}"
        )
        tutorial_btn.clicked.connect(self._on_tutorial)
        btn_layout.addWidget(tutorial_btn)

        layout.addLayout(btn_layout)

    def _on_tutorial(self):
        """Signal that the tutorial dialog should open, then close this dialog."""
        self._open_tutorial = True
        self.accept()


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
        self._onboarding_checked = False  # guard so we only check once per session
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
        subtitle = QLabel("Fast and accurate bubble sheet grading for teachers.")
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
            "Browse one of our many ready-to-use bubble sheets, pick a favorite, "
            "and download the PDF for printing."
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

        key_title = QLabel("Create Your Answer Keys")
        key_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #b8860b; "
            "background: transparent; border: none;"
        )
        key_layout.addWidget(key_title)

        key_desc = QLabel(
            "It's easy to make an answer key for MarkShark."
            "  You can use a word processor or spreadsheet "
            "or our utility to create and edit answer keys for your tests."
        )
        key_desc.setWordWrap(True)
        key_desc.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        key_layout.addWidget(key_desc)

        key_layout.addStretch()

        key_btn_row = QHBoxLayout()
        key_build_btn = QPushButton("Answer Key Utility")
        key_build_btn.setStyleSheet(
            "QPushButton { background-color: #b8860b; color: white; "
            "padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background-color: #9a7209; }"
        )
        key_build_btn.clicked.connect(self._on_key_builder)
        key_btn_row.addWidget(key_build_btn)

        sample_key_btn = QPushButton("Download Sample Key")
        sample_key_btn.setStyleSheet(
            "QPushButton { background-color: #e8d5a3; color: #6b4f0a; "
            "padding: 6px 14px; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #dcc68e; }"
        )
        sample_key_btn.clicked.connect(self._on_download_sample_key)
        key_btn_row.addWidget(sample_key_btn)

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

        gs_title = QLabel("Steps to working with MarkShark")
        gs_title.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {_BLUE}; "
            "background: transparent; border: none;"
        )
        gs_layout.addWidget(gs_title)

        steps = [
            ("\U0001F5A8  Print", "Download a bubble sheet. Print copies for your class."),
            ("\U0001F4E0  Scan", "Scan the completed test sheets into a PDF."),
            ("\U0001F5C2\ufe0f Set Folder", "Set a MarkShark folder for your class or section."),
            ("\u2705  Grade", "Upload your scans, answer key, and click Score."),
            ("\U0001F50D  Review", "Review student answers and correct if needed."),
            ("\U0001F4CA  Report", "Get a summary report of student performance, question difficulty, and score distributions."),
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
                "font-size: 11px; color: #555; background: transparent; border: none;"
            )
            step_row.addWidget(sd, 1)

            gs_layout.addLayout(step_row)

        # ── Help & Tutorial buttons ──
        gs_btn_row = QHBoxLayout()
        gs_btn_row.setSpacing(8)

        help_btn = QPushButton("Help && Documentation")
        help_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_BLUE}; color: white; "
            f"padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 12px; }}"
            f"QPushButton:hover {{ background-color: #0b5ed7; }}"
        )
        help_btn.clicked.connect(self._on_open_help)
        gs_btn_row.addWidget(help_btn)

        tutorial_btn = QPushButton("Tutorial && Sample Data")
        tutorial_btn.setStyleSheet(
            "QPushButton { background-color: #6c757d; color: white; "
            "padding: 6px 14px; border-radius: 6px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background-color: #565e64; }"
        )
        tutorial_btn.clicked.connect(self._on_open_tutorial)
        gs_btn_row.addWidget(tutorial_btn)

        gs_btn_row.addStretch()
        gs_layout.addLayout(gs_btn_row)

        gs_layout.addStretch()

        top_tiles.addWidget(gs_tile, 1)

        content_layout.addLayout(top_tiles)

        # --- "Set Up Your Courses!" tile ---
        content_layout.addSpacing(16)

        self.courses_tile_container = QVBoxLayout()
        self.courses_tile_container.setSpacing(0)
        content_layout.addLayout(self.courses_tile_container)

        # --- Recent Projects section ---
        content_layout.addSpacing(16)

        self.projects_container = QVBoxLayout()
        content_layout.addLayout(self.projects_container)

        self._populate_recent_projects()
        self._refresh_course_list()

        content_layout.addStretch()

    # ----- Course folders -----

    def _refresh_course_list(self):
        """Build the courses tile from scratch each time."""
        # Clear existing tile
        while self.courses_tile_container.count():
            item = self.courses_tile_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if ProjectRegistry is None:
            return

        try:
            registry = ProjectRegistry()
            courses = registry.list_courses()
        except Exception:
            courses = []

        # Build the tile frame (same pattern as Recent Assessments)
        tile = QFrame()
        tile.setObjectName("courses_tile")
        tile.setStyleSheet(
            "QFrame#courses_tile { background-color: #ffffff; "
            "border: 1px solid #e0e0e0; border-radius: 10px; }"
        )
        tile_layout = QVBoxLayout(tile)
        tile_layout.setContentsMargins(0, 0, 0, 0)
        tile_layout.setSpacing(0)

        # ── Header bar ──
        header_widget = QWidget()
        header_widget.setStyleSheet(
            f"background-color: {_BLUE}; "
            "border-top-left-radius: 10px; border-top-right-radius: 10px;"
        )
        header_row = QHBoxLayout(header_widget)
        header_row.setContentsMargins(14, 8, 10, 8)

        header_label = QLabel("Set up a MarkShark folder for each of your courses (or sections).")
        header_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: white; background: transparent;"
        )
        header_row.addWidget(header_label)

        header_row.addStretch()

        _GHOST_BTN = (
            "QPushButton { background-color: rgba(255,255,255,0.2); color: white; "
            "padding: 4px 12px; border-radius: 4px; font-size: 11px; "
            "border: 1px solid rgba(255,255,255,0.4); }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.35); }"
        )

        new_course_btn = QPushButton("+ New Course")
        new_course_btn.setStyleSheet(_GHOST_BTN)
        new_course_btn.clicked.connect(self._on_new_course)
        header_row.addWidget(new_course_btn)

        manage_btn = QPushButton("Course Manager")
        manage_btn.setStyleSheet(_GHOST_BTN)
        manage_btn.clicked.connect(self._on_manage_projects)
        header_row.addWidget(manage_btn)

        tile_layout.addWidget(header_widget)

        # ── Helpful description text ──
        desc_widget = QWidget()
        desc_widget.setStyleSheet("background: transparent;")
        desc_layout = QVBoxLayout(desc_widget)
        desc_layout.setContentsMargins(14, 10, 14, 6)

        desc = QLabel(
            "Make a different MarkShark folder for each course or section you teach "
            "inside the folder you're already using for your class.  "
            "MarkShark will create subfolders for each assessment "
            "to store all associated files and data "
            "(e.g., Midterm 1 or Final Exam 2025)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet(
            "font-size: 12px; color: #555; background: transparent; border: none;"
        )
        desc_layout.addWidget(desc)

        tile_layout.addWidget(desc_widget)

        # ── Course rows ──
        if not courses:
            empty_widget = QWidget()
            empty_widget.setStyleSheet("background: transparent;")
            empty_layout = QHBoxLayout(empty_widget)
            empty_layout.setContentsMargins(14, 8, 14, 12)

            empty_label = QLabel(
                "No courses yet — click <b>+ New Course</b> above to get started!"
            )
            empty_label.setWordWrap(True)
            empty_label.setStyleSheet(
                "font-size: 12px; color: #999; font-style: italic; "
                "background: transparent; border: none;"
            )
            empty_layout.addWidget(empty_label)
            tile_layout.addWidget(empty_widget)
        else:
            # Separator between description and first row
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet("color: #e9ecef;")
            sep.setFixedHeight(1)
            tile_layout.addWidget(sep)

            display = courses[:6]
            for i, course in enumerate(display):
                row = self._make_course_row(course)
                tile_layout.addWidget(row)
                if i < len(display) - 1:
                    sep = QFrame()
                    sep.setFrameShape(QFrame.Shape.HLine)
                    sep.setStyleSheet("color: #e9ecef;")
                    sep.setFixedHeight(1)
                    tile_layout.addWidget(sep)

            if len(courses) > 6:
                more_widget = QWidget()
                more_widget.setStyleSheet("background: transparent;")
                more_layout = QHBoxLayout(more_widget)
                more_layout.setContentsMargins(14, 4, 14, 8)
                more_label = QLabel(
                    f"…and {len(courses) - 6} more — open the "
                    "<b>Course Manager</b> to see all."
                )
                more_label.setStyleSheet(
                    "font-size: 11px; color: #999; font-style: italic; "
                    "background: transparent; border: none;"
                )
                more_layout.addWidget(more_label)
                tile_layout.addWidget(more_widget)

        self.courses_tile_container.addWidget(tile)

    def _make_course_row(self, course: dict) -> QWidget:
        """Create a compact row for one course inside the courses tile."""
        course_path = course.get("path", "")
        missing = not Path(course_path).is_dir() if course_path else True

        row = QWidget()
        row.setStyleSheet(
            "QWidget { background: transparent; }"
            "QWidget:hover { background-color: #eff6ff; }"
        )
        if not missing:
            row.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(14, 8, 14, 8)
        row_layout.setSpacing(16)

        # Course name
        name = course.get("name", "Unnamed")
        if missing:
            name_label = QLabel(f"\u26A0 <b>{name}</b>")
            name_label.setToolTip(
                "Course folder not found — it may have been moved or renamed."
            )
            name_label.setStyleSheet("font-size: 13px; color: #b91c1c;")
        else:
            name_label = QLabel(f"<b>{name}</b>")
            name_label.setStyleSheet("font-size: 13px; color: #1a1a1a;")
        name_label.setFixedWidth(170)
        row_layout.addWidget(name_label)

        # Path
        path_label = QLabel(course_path)
        if missing:
            path_label.setStyleSheet("font-size: 11px; color: #b91c1c;")
        else:
            path_label.setStyleSheet("font-size: 11px; color: #777;")
        path_label.setWordWrap(False)
        row_layout.addWidget(path_label, 1)

        # Assessment count
        try:
            registry = ProjectRegistry()
            grouped = registry.list_by_course()
            count = len(grouped.get(course_path, []))
        except Exception:
            count = 0
        count_label = QLabel(
            f"{count} assessment{'s' if count != 1 else ''}"
        )
        count_label.setStyleSheet("font-size: 11px; color: #999;")
        count_label.setFixedWidth(100)
        row_layout.addWidget(count_label)

        # Open Folder button
        folder_btn = QPushButton("Open Folder")
        folder_btn.setStyleSheet(
            "QPushButton { background-color: #e9ecef; color: #333; "
            "padding: 4px 14px; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background-color: #dee2e6; }"
            "QPushButton:disabled { background-color: #f5f5f5; color: #bbb; }"
        )
        folder_btn.setEnabled(not missing)
        folder_btn.clicked.connect(
            lambda checked, p=course_path: self._open_folder(p)
        )
        row_layout.addWidget(folder_btn)

        # Make the whole row clickable → go to Course Manager
        if not missing:
            row.mousePressEvent = lambda event: self._on_manage_projects()

        return row

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
            empty = QLabel("Assessment registry not available.")
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
                "No recent assessments yet. Use the Grader to create your first assessment, "
                "or open the Course Manager to set up a new one."
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

        header_label = QLabel("Recent Assessments")
        header_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: white; background: transparent;"
        )
        header_row.addWidget(header_label)

        header_row.addStretch()

        new_proj_btn = QPushButton("+ New Assessment")
        new_proj_btn.setStyleSheet(
            "QPushButton { background-color: rgba(255,255,255,0.2); color: white; "
            "padding: 4px 12px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(255,255,255,0.4); }"
            "QPushButton:hover { background-color: rgba(255,255,255,0.35); }"
        )
        new_proj_btn.clicked.connect(self._on_new_project)
        header_row.addWidget(new_proj_btn)

        manage_btn = QPushButton("Course Manager")
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
            name_label.setToolTip("Assessment folder not found — it may have been moved or deleted.")
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

    def _on_open_help(self):
        """Navigate to the Help & Documentation page."""
        if self._main_window:
            self._main_window._navigate_to_key("help")

    def _on_open_tutorial(self):
        """Open the tutorial dialog with download options."""
        dialog = _TutorialDialog(self)
        dialog.exec()

    def _on_find_bubblesheet(self):
        """Open the template picker dialog."""
        dialog = _TemplatePicker(self)
        dialog.open_manager_btn.clicked.connect(
            lambda: self._go_to_page_and_close(dialog, "template_manager")
        )
        dialog.exec()

    def _on_key_builder(self):
        """Navigate to the Answer Key Utility page."""
        if self._main_window:
            self._main_window._navigate_to_key("key_builder")

    def _on_download_sample_key(self):
        """Let the user choose Text or Excel, then save a sample answer key.

        Uses a small custom QDialog instead of QMessageBox so we can
        control button order exactly (QMessageBox reorders buttons by
        role, which varies across macOS / Windows).
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Download Sample Answer Key")
        dlg.setMinimumWidth(360)
        layout = QVBoxLayout(dlg)

        label = QLabel(
            "Which format would you like?\n\n"
            "Text (.txt) \u2014 simple, one answer per line\n"
            "Excel (.xlsx) \u2014 spreadsheet with instructions tab"
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        btn_row = QHBoxLayout()
        txt_btn = QPushButton("Text (.txt)")
        xlsx_btn = QPushButton("Excel (.xlsx)")
        cancel_btn = QPushButton("Cancel")

        for btn in (txt_btn, xlsx_btn):
            btn.setStyleSheet(
                "QPushButton { background-color: #b8860b; color: white; "
                "padding: 6px 16px; border-radius: 6px; font-weight: bold; }"
                "QPushButton:hover { background-color: #9a7209; }"
            )
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #e0e0e0; color: #333; "
            "padding: 6px 16px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #d0d0d0; }"
        )

        btn_row.addWidget(txt_btn)
        btn_row.addWidget(xlsx_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Wire buttons — store the choice, then close
        choice = {}
        txt_btn.clicked.connect(lambda: (choice.update(fmt="txt"), dlg.accept()))
        xlsx_btn.clicked.connect(lambda: (choice.update(fmt="xlsx"), dlg.accept()))
        cancel_btn.clicked.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            if choice.get("fmt") == "txt":
                self._save_sample_key_txt()
            elif choice.get("fmt") == "xlsx":
                self._save_sample_key_xlsx()

    def _save_sample_key_txt(self):
        """Copy the bundled sample_answer_key.txt to a user-chosen location."""
        # The sample lives alongside the other bundled assets
        sample = (
            Path(__file__).resolve().parent.parent.parent
            / "assets" / "sample_answer_key.txt"
        )
        if not sample.exists():
            QMessageBox.warning(
                self, "File Not Found",
                "Sample text key not found in application assets.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Sample Answer Key",
            str(Path.home() / "sample_answer_key.txt"),
            "Text Files (*.txt)",
        )
        if save_path:
            from ..utils import safe_copy_file
            try:
                safe_copy_file(sample, save_path)
                QMessageBox.information(
                    self, "Saved",
                    f"Sample answer key saved to:\n{save_path}",
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save file:\n{e}")

    def _save_sample_key_xlsx(self):
        """Copy the bundled answer_key_template.xlsx to a user-chosen location."""
        template = (
            Path(__file__).resolve().parent.parent.parent
            / "assets" / "answer_key_template.xlsx"
        )
        if not template.exists():
            QMessageBox.warning(
                self, "File Not Found",
                "Excel key template not found in application assets.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Sample Answer Key",
            str(Path.home() / "answer_key_template.xlsx"),
            "Excel Files (*.xlsx)",
        )
        if save_path:
            from ..utils import safe_copy_file
            try:
                safe_copy_file(template, save_path)
                QMessageBox.information(
                    self, "Saved",
                    f"Sample answer key saved to:\n{save_path}",
                )
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save file:\n{e}")

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

    def _on_new_course(self):
        """Create a new course via the CourseDialog and refresh the tile."""
        from ..dialogs import CourseDialog

        dlg = CourseDialog(
            self,
            title="Create New MarkShark Course Folder",
            confirm_label="Create Course",
        )
        if dlg.exec() != CourseDialog.DialogCode.Accepted:
            return

        data = dlg.result_data()
        if not data:
            return

        course_path = Path(data["course_path"])
        try:
            course_path.mkdir(parents=True, exist_ok=True)
            registry = ProjectRegistry()
            registry.register_course(course_path, data["name"])
            self._refresh_course_list()
        except Exception:
            pass

    def _on_manage_projects(self):
        """Navigate to the Project Manager page."""
        if self._main_window:
            self._main_window._navigate_to_key("project_manager")

    def showEvent(self, event):
        """Refresh recent projects and working dir when the page becomes visible.

        On the very first show after launch, checks SettingsStore for the
        ``app/onboarding_dismissed`` flag.  If it's unset (i.e. the user
        has never seen the app before), shows a one-time welcome dialog.
        The guard ``_onboarding_checked`` prevents re-checking on every
        subsequent navigation back to this page within the same session.
        """
        super().showEvent(event)
        self._populate_recent_projects()
        self._refresh_course_list()
        self._maybe_show_onboarding()

    def _maybe_show_onboarding(self):
        """Show the first-run welcome dialog if the user hasn't dismissed it.

        Uses ``app/onboarding_dismissed`` in SettingsStore as the
        persistent flag.  Only runs once per application session
        (guarded by ``_onboarding_checked``).

        The dialog is opened with ``show()`` (non-modal) rather than
        ``exec()`` so the user can see and interact with the main
        window behind it.  Using ``exec()`` with custom window flags
        causes a segfault on macOS because Python may GC the dialog
        while Qt's widget tree still references it.
        """
        if self._onboarding_checked:
            return
        self._onboarding_checked = True

        try:
            from ..models.settings_store import SettingsStore
            settings = SettingsStore()
        except ImportError:
            return

        if settings.value("app/onboarding_dismissed", False, type=bool):
            return  # user has seen the dialog before

        # Defer showing the dialog until the next event-loop pass.
        # showEvent fires while the main window is still being
        # activated, so any dialog shown now gets buried behind it.
        # QTimer.singleShot(0, ...) queues the call for after the
        # window manager has finished placing the main window.
        QTimer.singleShot(0, self._show_onboarding_dlg)

    def _show_onboarding_dlg(self):
        """Create and display the first-run dialog (called via deferred timer).

        Stored on ``self`` so Python doesn't garbage-collect the dialog
        while it's still visible.
        """
        self._onboarding_dlg = _FirstRunDialog(self)
        self._onboarding_dlg.finished.connect(self._on_onboarding_finished)
        self._onboarding_dlg.show()
        self._onboarding_dlg.raise_()
        self._onboarding_dlg.activateWindow()

    def _on_onboarding_finished(self, result: int):
        """Handle the first-run dialog closing.

        Called via the ``finished`` signal after the user clicks one of
        the three buttons or closes the window.
        """
        dlg = self._onboarding_dlg

        # Only persist the flag if the user explicitly chose "Don't show
        # again" or "Download Tutorial" (both call accept()).  "Maybe
        # Later" and the window close button call reject(), leaving the
        # flag unset so the dialog reappears on the next launch.
        if result == QDialog.DialogCode.Accepted:
            try:
                from ..models.settings_store import SettingsStore
                SettingsStore().setValue("app/onboarding_dismissed", True)
            except ImportError:
                pass

        # If user chose "Download Tutorial", open the tutorial dialog
        if dlg._open_tutorial:
            self._on_open_tutorial()

        # Clean up
        dlg.deleteLater()
        self._onboarding_dlg = None
