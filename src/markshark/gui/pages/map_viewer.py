"""
Bubblemap Viewer — overlay bubblemap circles on any PDF to verify placement.

Separates bubblemap (YAML) and PDF selection so users can apply any
bubblemap to any PDF (template, aligned scans, raw scans, etc.).
Supports multi-page PDFs with page-by-page navigation.
"""

import platform
import subprocess
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QSpinBox,
    QComboBox,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
)

from ..widgets import PageHeader

# Template manager (best-effort import)
try:
    from markshark.template_manager import TemplateManager
except ImportError:
    TemplateManager = None

# Overlay function (best-effort import)
try:
    from markshark.mapviewer_core import overlay_bublmap_pages
except ImportError:
    overlay_bublmap_pages = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _numpy_to_pixmap(img_bgr: np.ndarray) -> QPixmap:
    """Convert a BGR numpy array to a QPixmap."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class _OverlayWorker(QThread):
    """Generate all overlaid pages in a background thread."""

    finished = Signal(list)   # List[np.ndarray]
    errored = Signal(str)

    def __init__(self, bublmap_path: str, pdf_path: str, dpi: int, parent=None):
        super().__init__(parent)
        self._bublmap_path = bublmap_path
        self._pdf_path = pdf_path
        self._dpi = dpi

    def run(self):
        try:
            pages = overlay_bublmap_pages(
                bublmap_path=self._bublmap_path,
                input_path=self._pdf_path,
                dpi=self._dpi,
            )
            self.finished.emit(pages)
        except Exception as e:
            self.errored.emit(str(e))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
class MapViewerPage(QWidget):
    """
    Bubblemap Viewer — overlay bubble positions on any PDF.

    Features:
      - Separate bubblemap YAML and PDF selectors (independent)
      - Multi-page overlay with page-by-page navigation
      - Fit / Full-size zoom
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._template_manager: Optional[TemplateManager] = None
        self._templates: List = []
        self._worker: Optional[_OverlayWorker] = None

        # Page state
        self._overlaid_pages: List[np.ndarray] = []
        self._current_page_idx: int = 0
        self._zoom_mode: str = "fit"
        self._full_pixmap: Optional[QPixmap] = None
        self._result_path: Optional[str] = None

        self._init_template_manager()
        self._setup_ui()

    def _init_template_manager(self):
        if TemplateManager is None:
            return
        try:
            self._template_manager = TemplateManager()
        except Exception as e:
            print(f"Could not initialize TemplateManager: {e}")

    # -------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark Bubblemap Viewer",
            "Overlay bubblemap circles on any PDF to verify placement and debug alignment.",
        )
        layout.addWidget(header)

        # === Input section ===
        input_group = QGroupBox("Select Files")
        input_layout = QVBoxLayout(input_group)

        # Bubblemap YAML selector
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(QLabel("Select Bubblemap:"))
        self.yaml_combo = QComboBox()
        self.yaml_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.yaml_combo.currentIndexChanged.connect(self._on_yaml_combo_changed)
        yaml_row.addWidget(self.yaml_combo, 1)

        yaml_browse_btn = QPushButton("Browse...")
        yaml_browse_btn.setMaximumWidth(80)
        yaml_browse_btn.clicked.connect(self._browse_yaml)
        yaml_row.addWidget(yaml_browse_btn)
        input_layout.addLayout(yaml_row)

        # PDF selector
        pdf_row = QHBoxLayout()
        pdf_row.addWidget(QLabel("Select PDF:"))
        self.pdf_combo = QComboBox()
        self.pdf_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.pdf_combo.currentIndexChanged.connect(self._on_pdf_combo_changed)
        pdf_row.addWidget(self.pdf_combo, 1)

        pdf_browse_btn = QPushButton("Browse...")
        pdf_browse_btn.setMaximumWidth(80)
        pdf_browse_btn.clicked.connect(self._browse_pdf)
        pdf_row.addWidget(pdf_browse_btn)
        input_layout.addLayout(pdf_row)

        # Info label
        self.info_label = QLabel("Select a bubblemap YAML and a PDF, then click Visualize.")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        input_layout.addWidget(self.info_label)

        # Refresh button
        refresh_btn = QPushButton("Refresh Templates")
        refresh_btn.setMaximumWidth(130)
        refresh_btn.clicked.connect(self._refresh_templates)
        input_layout.addWidget(refresh_btn)

        layout.addWidget(input_group)

        # === Settings row ===
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setToolTip("Render resolution (150 recommended)")
        settings_layout.addWidget(self.dpi_spin)

        settings_layout.addSpacing(20)

        # Visualize button
        self.viz_btn = QPushButton("Visualize Bubblemap")
        self.viz_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self.viz_btn.clicked.connect(self._on_visualize)
        settings_layout.addWidget(self.viz_btn)

        settings_layout.addStretch()

        # Save As / Open Folder (hidden until generated)
        self.save_btn = QPushButton("Save Page As...")
        self.save_btn.clicked.connect(self._on_save_as)
        self.save_btn.setVisible(False)
        settings_layout.addWidget(self.save_btn)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        self.open_folder_btn.setVisible(False)
        settings_layout.addWidget(self.open_folder_btn)

        layout.addLayout(settings_layout)

        # === Status ===
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # === Result image (scrollable) ===
        self.result_scroll = QScrollArea()
        self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #ccc; background-color: #808080; }"
        )

        self.result_image = QLabel()
        self.result_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_image.setText("Overlay image will appear here.")
        self.result_image.setStyleSheet("color: #ccc; font-size: 14px; padding: 40px;")
        self.result_scroll.setWidget(self.result_image)

        layout.addWidget(self.result_scroll, 1)

        # === Bottom bar: zoom + page navigation ===
        bottom_layout = QHBoxLayout()

        # Zoom controls
        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setCheckable(True)
        self.fit_btn.setChecked(True)
        self.fit_btn.clicked.connect(lambda: self._set_zoom("fit"))
        bottom_layout.addWidget(self.fit_btn)

        self.zoom_btn = QPushButton("Full Size")
        self.zoom_btn.setCheckable(True)
        self.zoom_btn.clicked.connect(lambda: self._set_zoom("full"))
        bottom_layout.addWidget(self.zoom_btn)

        bottom_layout.addStretch()

        # Page navigation
        self.prev_btn = QPushButton("\u25c0 Prev")
        self.prev_btn.clicked.connect(self._on_prev_page)
        self.prev_btn.setEnabled(False)
        bottom_layout.addWidget(self.prev_btn)

        self.page_label = QLabel("")
        self.page_label.setStyleSheet("font-weight: bold; padding: 0 12px;")
        bottom_layout.addWidget(self.page_label)

        self.next_btn = QPushButton("Next \u25b6")
        self.next_btn.clicked.connect(self._on_next_page)
        self.next_btn.setEnabled(False)
        bottom_layout.addWidget(self.next_btn)

        layout.addLayout(bottom_layout)

        # Populate template dropdowns
        self._refresh_templates()

    # -------------------------------------------------------------------
    # Template management
    # -------------------------------------------------------------------
    def _refresh_templates(self):
        """Populate both combo boxes from registered templates."""
        self.yaml_combo.blockSignals(True)
        self.pdf_combo.blockSignals(True)
        self.yaml_combo.clear()
        self.pdf_combo.clear()
        self._templates = []

        # Placeholder items
        self.yaml_combo.addItem("(Select a bubblemap YAML)", None)
        self.pdf_combo.addItem("(Select a PDF)", None)

        if self._template_manager:
            try:
                self._template_manager._templates_cache = None
                self._templates = self._template_manager.scan_templates(
                    force_refresh=True
                )
            except Exception as e:
                print(f"Error scanning templates: {e}")

            for t in self._templates:
                # YAML dropdown
                yaml_path = t.bubblemap_yaml_path
                if yaml_path and yaml_path.exists():
                    self.yaml_combo.addItem(
                        t.display_name, str(yaml_path)
                    )
                # PDF dropdown
                pdf_path = t.template_pdf_path
                if pdf_path and pdf_path.exists():
                    self.pdf_combo.addItem(
                        f"{t.display_name} (template PDF)", str(pdf_path)
                    )

        self.yaml_combo.blockSignals(False)
        self.pdf_combo.blockSignals(False)

    def _on_yaml_combo_changed(self, index: int):
        """Update info label when a YAML is selected."""
        yaml_path = self.yaml_combo.currentData()
        if not yaml_path:
            self.info_label.setText(
                "Select a bubblemap YAML and a PDF, then click Visualize."
            )
            return
        # Find the matching template by YAML path
        for t in self._templates:
            if t.bubblemap_yaml_path and str(t.bubblemap_yaml_path) == yaml_path:
                parts = []
                if t.num_questions:
                    parts.append(f"Questions: {t.num_questions}")
                if t.num_choices:
                    parts.append(f"Choices: {t.num_choices}")
                if t.num_pages:
                    parts.append(f"Pages: {t.num_pages}")
                self.info_label.setText(
                    " | ".join(parts) if parts else t.template_id
                )
                return
        # Custom file — just show the filename
        self.info_label.setText(f"Custom: {Path(yaml_path).name}")

    def _on_pdf_combo_changed(self, index: int):
        """No-op for now — selection is read at visualize time."""
        pass

    # -------------------------------------------------------------------
    # Browse for custom files
    # -------------------------------------------------------------------
    def _browse_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Bubblemap YAML", "",
            "YAML files (*.yaml *.yml);;All Files (*)",
        )
        if not path:
            return
        name = Path(path).name
        # Add as custom entry and select it
        label = f"(Custom: {name})"
        self.yaml_combo.blockSignals(True)
        self.yaml_combo.addItem(label, path)
        self.yaml_combo.setCurrentIndex(self.yaml_combo.count() - 1)
        self.yaml_combo.blockSignals(False)
        self.info_label.setText(f"Custom bubblemap: {name}")

    def _browse_pdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "",
            "PDF files (*.pdf);;All Files (*)",
        )
        if not path:
            return
        name = Path(path).name
        label = f"(Custom: {name})"
        self.pdf_combo.blockSignals(True)
        self.pdf_combo.addItem(label, path)
        self.pdf_combo.setCurrentIndex(self.pdf_combo.count() - 1)
        self.pdf_combo.blockSignals(False)

    # -------------------------------------------------------------------
    # Resolve selected paths
    # -------------------------------------------------------------------
    def _resolve_yaml(self) -> Optional[str]:
        data = self.yaml_combo.currentData()
        return data if data else None

    def _resolve_pdf(self) -> Optional[str]:
        data = self.pdf_combo.currentData()
        return data if data else None

    # -------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------
    def _on_visualize(self):
        yaml_path = self._resolve_yaml()
        pdf_path = self._resolve_pdf()

        if not yaml_path or not Path(yaml_path).exists():
            QMessageBox.warning(
                self, "Missing Bubblemap",
                "Please select a bubblemap YAML file.",
            )
            return
        if not pdf_path or not Path(pdf_path).exists():
            QMessageBox.warning(
                self, "Missing PDF",
                "Please select a PDF file.",
            )
            return
        if overlay_bublmap_pages is None:
            QMessageBox.warning(
                self, "Not Available",
                "markshark.mapviewer_core module is not installed.",
            )
            return

        # Disable UI
        self.viz_btn.setEnabled(False)
        self.viz_btn.setText("Generating...")
        self.status_label.setText("Loading pages and generating overlays...")
        self.status_label.setStyleSheet("color: #1565C0;")

        self._worker = _OverlayWorker(yaml_path, pdf_path, self.dpi_spin.value(), self)
        self._worker.finished.connect(self._on_overlay_finished)
        self._worker.errored.connect(self._on_overlay_error)
        self._worker.start()

    def _on_overlay_finished(self, pages: list):
        self.viz_btn.setEnabled(True)
        self.viz_btn.setText("Visualize Bubblemap")

        if not pages:
            self.status_label.setText("No pages generated.")
            self.status_label.setStyleSheet("color: red;")
            self._worker = None
            return

        self._overlaid_pages = pages
        self._current_page_idx = 0

        n = len(pages)
        self.status_label.setText(f"Overlay generated: {n} page{'s' if n != 1 else ''}.")
        self.status_label.setStyleSheet("color: green;")

        self._update_page_display()

        # Show action buttons
        self.save_btn.setVisible(True)
        self.open_folder_btn.setVisible(False)  # no file on disk yet

        self._worker = None

    def _on_overlay_error(self, error_msg: str):
        self.viz_btn.setEnabled(True)
        self.viz_btn.setText("Visualize Bubblemap")
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: red;")

        QMessageBox.warning(
            self, "Overlay Error",
            f"Failed to generate overlay:\n\n{error_msg}",
        )
        self._worker = None

    # -------------------------------------------------------------------
    # Page display & navigation
    # -------------------------------------------------------------------
    def _update_page_display(self):
        """Render the current page to the QLabel."""
        if not self._overlaid_pages:
            return

        idx = self._current_page_idx
        img = self._overlaid_pages[idx]
        self._full_pixmap = _numpy_to_pixmap(img)
        self._apply_zoom()

        n = len(self._overlaid_pages)
        self.page_label.setText(f"Page {idx + 1} / {n}")
        self.prev_btn.setEnabled(idx > 0)
        self.next_btn.setEnabled(idx < n - 1)

    def _on_prev_page(self):
        if self._current_page_idx > 0:
            self._current_page_idx -= 1
            self._update_page_display()

    def _on_next_page(self):
        if self._current_page_idx < len(self._overlaid_pages) - 1:
            self._current_page_idx += 1
            self._update_page_display()

    # -------------------------------------------------------------------
    # Zoom
    # -------------------------------------------------------------------
    def _set_zoom(self, mode: str):
        self._zoom_mode = mode
        self.fit_btn.setChecked(mode == "fit")
        self.zoom_btn.setChecked(mode == "full")
        self._apply_zoom()

    def _apply_zoom(self):
        if self._full_pixmap is None:
            return

        if self._zoom_mode == "fit":
            scroll_w = self.result_scroll.viewport().width() - 20
            if scroll_w > 50:
                scaled = self._full_pixmap.scaledToWidth(
                    scroll_w, Qt.TransformationMode.SmoothTransformation
                )
            else:
                scaled = self._full_pixmap
            self.result_image.setPixmap(scaled)
        else:
            self.result_image.setPixmap(self._full_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._zoom_mode == "fit" and self._full_pixmap is not None:
            self._apply_zoom()

    # -------------------------------------------------------------------
    # Save / Open
    # -------------------------------------------------------------------
    def _on_save_as(self):
        """Save the currently displayed page as an image file."""
        if not self._overlaid_pages:
            return

        idx = self._current_page_idx
        default_name = f"overlay_page_{idx + 1}.png"

        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Overlay Image",
            default_name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)",
        )
        if not dest:
            return

        try:
            img = self._overlaid_pages[idx]
            if not cv2.imwrite(dest, img):
                raise IOError(f"cv2.imwrite failed for {dest}")
            self._result_path = dest
            self.open_folder_btn.setVisible(True)
            self.status_label.setText(f"Saved page {idx + 1} to: {dest}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save: {e}")

    def _open_output_folder(self):
        if not self._result_path:
            return
        folder = str(Path(self._result_path).parent)
        if not Path(folder).exists():
            return

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.Popen(["open", folder])
            elif system == "Windows":
                subprocess.Popen(["explorer", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            print(f"Could not open folder: {e}")
