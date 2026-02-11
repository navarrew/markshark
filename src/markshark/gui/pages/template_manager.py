"""
Template Manager page - browse, manage, and preview bubble sheet templates.

Features:
- List all available templates
- PDF preview of selected template
- Favorite/unfavorite templates
- Reorder templates (move up/down)
- Archive/unarchive templates
- Validate templates
"""

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QSplitter,
    QFrame,
    QMessageBox,
    QScrollArea,
    QFileDialog,
)

from ..widgets import PageHeader, PDFPreview

# Try to import MarkShark template manager
try:
    from markshark.template_manager import TemplateManager, BubbleSheetTemplate
except ImportError:
    TemplateManager = None
    BubbleSheetTemplate = None


class TemplateManagerPage(QWidget):
    """
    Template management page with list and preview.

    Signals:
        template_selected: Emitted when a template is selected
    """

    template_selected = Signal(object)  # Emits BubbleSheetTemplate or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._template_manager: Optional[TemplateManager] = None
        self._templates: List = []
        self._archived_templates: List = []
        self._selected_template = None

        self._init_template_manager()
        self._setup_ui()
        self._refresh_templates()

    def _init_template_manager(self):
        """Initialize the template manager."""
        if TemplateManager is None:
            return

        try:
            self._template_manager = TemplateManager()
        except Exception as e:
            print(f"Could not initialize TemplateManager: {e}")

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header with icon
        header = PageHeader(
            "MarkShark Template Manager",
            "Browse, preview, and manage your bubble sheet templates."
        )
        layout.addWidget(header)

        # Templates directory info
        if self._template_manager:
            dir_label = QLabel(f"Templates directory: {self._template_manager.templates_dir}")
            dir_label.setStyleSheet("color: #666; font-size: 11px;")
            layout.addWidget(dir_label)

        # Top controls
        controls_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_templates)
        controls_layout.addWidget(refresh_btn)

        open_folder_btn = QPushButton("Open Templates Folder")
        open_folder_btn.clicked.connect(self._open_templates_folder)
        controls_layout.addWidget(open_folder_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Main content: splitter with list and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        # Left side: template list
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)

        # Active templates
        active_group = QGroupBox("Active Templates")
        active_layout = QVBoxLayout(active_group)

        self.template_list = QListWidget()
        self.template_list.currentItemChanged.connect(self._on_template_selected)
        active_layout.addWidget(self.template_list)

        # Template action buttons
        btn_layout = QHBoxLayout()

        self.fav_btn = QPushButton("Favorite")
        self.fav_btn.clicked.connect(self._toggle_favorite)
        self.fav_btn.setEnabled(False)
        btn_layout.addWidget(self.fav_btn)

        self.up_btn = QPushButton("Up")
        self.up_btn.clicked.connect(self._move_up)
        self.up_btn.setEnabled(False)
        btn_layout.addWidget(self.up_btn)

        self.down_btn = QPushButton("Down")
        self.down_btn.clicked.connect(self._move_down)
        self.down_btn.setEnabled(False)
        btn_layout.addWidget(self.down_btn)

        self.archive_btn = QPushButton("Archive")
        self.archive_btn.clicked.connect(self._archive_template)
        self.archive_btn.setEnabled(False)
        btn_layout.addWidget(self.archive_btn)

        active_layout.addLayout(btn_layout)
        list_layout.addWidget(active_group)

        # Archived templates
        archived_group = QGroupBox("Archived Templates")
        archived_layout = QVBoxLayout(archived_group)

        self.archived_list = QListWidget()
        self.archived_list.currentItemChanged.connect(self._on_archived_selected)
        archived_layout.addWidget(self.archived_list)

        self.unarchive_btn = QPushButton("Unarchive")
        self.unarchive_btn.clicked.connect(self._unarchive_template)
        self.unarchive_btn.setEnabled(False)
        archived_layout.addWidget(self.unarchive_btn)

        list_layout.addWidget(archived_group)

        splitter.addWidget(list_widget)

        # Right side: details first, then preview below
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Template details (compact, at top)
        details_group = QGroupBox("Template Details")
        details_layout = QVBoxLayout(details_group)
        details_layout.setSpacing(4)

        self.details_label = QLabel("Select a template to see details.")
        self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        details_layout.addWidget(self.details_label)

        # Download buttons row
        download_layout = QHBoxLayout()
        download_layout.setSpacing(8)

        self.download_pdf_btn = QPushButton("Download PDF")
        self.download_pdf_btn.clicked.connect(self._download_pdf)
        self.download_pdf_btn.setEnabled(False)
        download_layout.addWidget(self.download_pdf_btn)

        self.download_yaml_btn = QPushButton("Download Bubblemap")
        self.download_yaml_btn.clicked.connect(self._download_yaml)
        self.download_yaml_btn.setEnabled(False)
        download_layout.addWidget(self.download_yaml_btn)

        download_layout.addStretch()
        details_layout.addLayout(download_layout)

        right_layout.addWidget(details_group)

        # Preview (scrollable, takes remaining space)
        preview_group = QGroupBox("Preview")
        preview_group_layout = QVBoxLayout(preview_group)

        # Zoom toggle buttons
        self._preview_zoom_mode: str = "fit"
        self._preview_dpi: int = 96

        zoom_layout = QHBoxLayout()
        self.preview_fit_btn = QPushButton("Fit Page")
        self.preview_fit_btn.setCheckable(True)
        self.preview_fit_btn.setChecked(True)
        self.preview_fit_btn.clicked.connect(lambda: self._toggle_preview_zoom("fit"))
        zoom_layout.addWidget(self.preview_fit_btn)

        self.preview_zoom_btn = QPushButton("Zoom In")
        self.preview_zoom_btn.setCheckable(True)
        self.preview_zoom_btn.clicked.connect(lambda: self._toggle_preview_zoom("scroll"))
        zoom_layout.addWidget(self.preview_zoom_btn)

        zoom_layout.addStretch()
        preview_group_layout.addLayout(zoom_layout)

        # Create scroll area for the PDF preview
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview_scroll.setStyleSheet("QScrollArea { border: none; background-color: #808080; }")

        # PDF preview - starts in fit mode
        self.pdf_preview = PDFPreview(width=500, height=650, scale_to_fit=True)
        self.preview_scroll.setWidget(self.pdf_preview)

        preview_group_layout.addWidget(self.preview_scroll, 1)
        right_layout.addWidget(preview_group, 1)  # Give preview the stretch

        splitter.addWidget(right_widget)

        # Set splitter proportions
        splitter.setSizes([350, 400])

        # Status bar
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def _refresh_templates(self):
        """Refresh the template lists."""
        self.template_list.clear()
        self.archived_list.clear()
        self._templates = []
        self._archived_templates = []

        if not self._template_manager:
            self.status_label.setText("Template manager not available")
            return

        try:
            # Clear caches to force refresh
            self._template_manager._templates_cache = None
            self._template_manager._archived_templates_cache = None

            # Get templates
            self._templates = self._template_manager.scan_templates(force_refresh=True)
            self._archived_templates = self._template_manager.scan_archived_templates(force_refresh=True)

            # Populate active list
            for template in self._templates:
                is_fav = self._template_manager.is_favorite(template.template_id)
                icon = "* " if is_fav else "  "
                item = QListWidgetItem(f"{icon}{template.display_name}")
                item.setData(Qt.ItemDataRole.UserRole, template)
                self.template_list.addItem(item)

            # Populate archived list
            for template in self._archived_templates:
                item = QListWidgetItem(f"  {template.display_name}")
                item.setData(Qt.ItemDataRole.UserRole, template)
                self.archived_list.addItem(item)

            self.status_label.setText(
                f"{len(self._templates)} active, {len(self._archived_templates)} archived"
            )

        except Exception as e:
            self.status_label.setText(f"Error loading templates: {e}")

    def _open_templates_folder(self):
        """Open the templates directory in the system file manager."""
        if not self._template_manager:
            QMessageBox.warning(
                self, "Not Available",
                "Template manager is not initialised.",
            )
            return

        folder = self._template_manager.templates_dir
        if not folder or not Path(folder).exists():
            QMessageBox.warning(
                self, "Not Found",
                "The templates directory does not exist.",
            )
            return

        # Warn the user before opening
        reply = QMessageBox.information(
            self,
            "Open Templates Folder",
            "You are about to open the templates directory in your file manager.\n\n"
            "You can add your own template folders here. Each template folder "
            "should contain a bubblemap YAML file and a template PDF.\n\n"
            "Back up your templates before making any changes.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Ok:
            return

        folder_str = str(folder)
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", folder_str])
            elif system == "Windows":
                subprocess.Popen(["explorer", folder_str])
            else:
                subprocess.Popen(["xdg-open", folder_str])
            self.status_label.setText(f"Opened: {folder_str}")
        except Exception as e:
            QMessageBox.warning(
                self, "Error", f"Could not open folder:\n{e}",
            )

    def _on_template_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle template selection in active list."""
        # Deselect archived list
        self.archived_list.clearSelection()

        if current is None:
            self._clear_preview()
            return

        template = current.data(Qt.ItemDataRole.UserRole)
        self._selected_template = template
        self._show_template(template)

        # Update button states
        is_fav = self._template_manager.is_favorite(template.template_id)
        self.fav_btn.setText("Unfavorite" if is_fav else "Favorite")
        self.fav_btn.setEnabled(True)

        idx = self.template_list.currentRow()
        self.up_btn.setEnabled(idx > 0)
        self.down_btn.setEnabled(idx < len(self._templates) - 1)
        self.archive_btn.setEnabled(True)
        self.unarchive_btn.setEnabled(False)

        self.template_selected.emit(template)

    def _on_archived_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle template selection in archived list."""
        # Deselect active list
        self.template_list.clearSelection()

        if current is None:
            self._clear_preview()
            return

        template = current.data(Qt.ItemDataRole.UserRole)
        self._selected_template = template
        self._show_template(template)

        # Update button states
        self.fav_btn.setEnabled(False)
        self.up_btn.setEnabled(False)
        self.down_btn.setEnabled(False)
        self.archive_btn.setEnabled(False)
        self.unarchive_btn.setEnabled(True)

    def _show_template(self, template):
        """Display template preview and details."""
        if template.template_pdf_path and template.template_pdf_path.exists():
            self._load_template_pdf(template)
        else:
            self.pdf_preview.clear()

        # Validate template first to get status
        is_valid = True
        errors = []
        if self._template_manager:
            is_valid, errors = self._template_manager.validate_template(template)

        # Build compact details text
        details = []

        # Line 1: ID + validation status
        valid_text = "<span style='color: green;'>Valid</span>" if is_valid else "<span style='color: red;'>Invalid</span>"
        details.append(f"<b>ID:</b> {template.template_id} &nbsp;&nbsp;&nbsp; {valid_text}")

        # Show errors if invalid
        if not is_valid and errors:
            details.append("<span style='color: red; font-size: 10px;'>" + " | ".join(errors) + "</span>")

        # Line 2: Description (if present)
        if template.description:
            details.append(f"<b>Description:</b> {template.description}")

        # Line 3: Pages, Questions, Choices on same line
        pqc = []
        if template.num_pages:
            pqc.append(f"<b>Pages:</b> {template.num_pages}")
        if template.num_questions:
            pqc.append(f"<b>Questions:</b> {template.num_questions}")
        if template.choices_label:
            pqc.append(f"<b>Choices:</b> {template.choices_label}")
        elif template.num_choices:
            pqc.append(f"<b>Choices:</b> {template.num_choices}")
        if pqc:
            details.append(" &nbsp;&nbsp; ".join(pqc))

        # Line 4: File names on same line
        pdf_name = template.template_pdf_path.name if template.template_pdf_path else 'N/A'
        yaml_name = template.bubblemap_yaml_path.name if template.bubblemap_yaml_path else 'N/A'
        details.append(f"<b>PDF:</b> {pdf_name} &nbsp;&nbsp; <b>Bubblemap:</b> {yaml_name}")

        self.details_label.setText("<br>".join(details))

        # Enable/disable download buttons based on file availability
        self.download_pdf_btn.setEnabled(
            template.template_pdf_path and template.template_pdf_path.exists()
        )
        self.download_yaml_btn.setEnabled(
            template.bubblemap_yaml_path and template.bubblemap_yaml_path.exists()
        )

    def _load_template_pdf(self, template=None):
        """Load the template PDF with multi-page support."""
        if template is None:
            template = self._selected_template
        if not template or not template.template_pdf_path or not template.template_pdf_path.exists():
            self.pdf_preview.clear()
            return

        # Invalidate cache so it re-renders at the new DPI
        self.pdf_preview._cached_pixmap = None

        num_pages = getattr(template, "num_pages", 1) or 1
        if num_pages > 1:
            pages = list(range(num_pages))
            self.pdf_preview.load_pdf_pages(
                template.template_pdf_path, pages, dpi=self._preview_dpi
            )
        else:
            self.pdf_preview.load_pdf(
                template.template_pdf_path, page=0, dpi=self._preview_dpi
            )

    def _toggle_preview_zoom(self, mode: str):
        """Switch between fit-to-page and zoomed-in scrollable modes."""
        self._preview_zoom_mode = mode
        self.preview_fit_btn.setChecked(mode == "fit")
        self.preview_zoom_btn.setChecked(mode == "scroll")

        if mode == "fit":
            self._preview_dpi = 96
            self.pdf_preview.set_scale_to_fit(True)
        else:
            self._preview_dpi = 150
            self.pdf_preview.set_scale_to_fit(False)

        # Reload current template at new DPI
        self._load_template_pdf()

    def _clear_preview(self):
        """Clear the preview and details."""
        self.pdf_preview.clear()
        self.details_label.setText("Select a template to see details.")
        self._selected_template = None

        self.fav_btn.setEnabled(False)
        self.up_btn.setEnabled(False)
        self.down_btn.setEnabled(False)
        self.archive_btn.setEnabled(False)
        self.unarchive_btn.setEnabled(False)
        self.download_pdf_btn.setEnabled(False)
        self.download_yaml_btn.setEnabled(False)

    def _toggle_favorite(self):
        """Toggle favorite status of selected template."""
        if not self._selected_template or not self._template_manager:
            return

        # Save ID before refresh (which clears self._selected_template)
        selected_id = self._selected_template.template_id
        self._template_manager.toggle_favorite(selected_id)
        self._refresh_templates()

        # Re-select the template
        for i in range(self.template_list.count()):
            item = self.template_list.item(i)
            template = item.data(Qt.ItemDataRole.UserRole)
            if template and template.template_id == selected_id:
                self.template_list.setCurrentRow(i)
                break

    def _move_up(self):
        """Move selected template up in the list."""
        if not self._selected_template or not self._template_manager:
            return

        self._template_manager.move_template_up(self._selected_template.template_id)
        current_row = self.template_list.currentRow()
        self._refresh_templates()

        # Select the moved template
        if current_row > 0:
            self.template_list.setCurrentRow(current_row - 1)

    def _move_down(self):
        """Move selected template down in the list."""
        if not self._selected_template or not self._template_manager:
            return

        self._template_manager.move_template_down(self._selected_template.template_id)
        current_row = self.template_list.currentRow()
        self._refresh_templates()

        # Select the moved template
        if current_row < self.template_list.count() - 1:
            self.template_list.setCurrentRow(current_row + 1)

    def _archive_template(self):
        """Archive the selected template."""
        if not self._selected_template or not self._template_manager:
            return

        reply = QMessageBox.question(
            self,
            "Archive Template",
            f"Archive '{self._selected_template.display_name}'?\n\n"
            "It will be hidden from dropdown menus but can be restored later.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self._template_manager.archive_template(self._selected_template.template_id):
                self._refresh_templates()
                self._clear_preview()
            else:
                QMessageBox.warning(self, "Error", "Failed to archive template.")

    def _unarchive_template(self):
        """Unarchive the selected template."""
        if not self._selected_template or not self._template_manager:
            return

        if self._template_manager.unarchive_template(self._selected_template.template_id):
            self._refresh_templates()
            self._clear_preview()
        else:
            QMessageBox.warning(self, "Error", "Failed to unarchive template.")

    def _download_pdf(self):
        """Download (copy) the template PDF to a user-selected location."""
        if not self._selected_template or not self._selected_template.template_pdf_path:
            return

        src_path = self._selected_template.template_pdf_path
        if not src_path.exists():
            QMessageBox.warning(self, "Error", "PDF file not found.")
            return

        # Open save dialog
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF As",
            str(Path.home() / src_path.name),
            "PDF Files (*.pdf)"
        )

        if dest_path:
            try:
                shutil.copy2(src_path, dest_path)
                self.status_label.setText(f"PDF saved to: {dest_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save PDF: {e}")

    def _download_yaml(self):
        """Download (copy) the template YAML to a user-selected location."""
        if not self._selected_template or not self._selected_template.bubblemap_yaml_path:
            return

        src_path = self._selected_template.bubblemap_yaml_path
        if not src_path.exists():
            QMessageBox.warning(self, "Error", "YAML file not found.")
            return

        # Open save dialog
        dest_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save YAML As",
            str(Path.home() / src_path.name),
            "YAML Files (*.yaml *.yml)"
        )

        if dest_path:
            try:
                shutil.copy2(src_path, dest_path)
                self.status_label.setText(f"YAML saved to: {dest_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save YAML: {e}")
