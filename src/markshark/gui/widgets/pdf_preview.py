"""
PDF preview widget for displaying PDF pages as images.

Uses pdf2image (poppler) to render PDF pages to QPixmap.
Supports single-page and multi-page (stacked vertically) display,
with fit-to-container or full-size (scrollable) modes.
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QScrollArea,
    QFrame,
    QSizePolicy,
)


class PDFPreview(QWidget):
    """
    Widget to display a preview of one or more PDF pages.

    Features:
    - Renders PDF page(s) at specified DPI
    - Scales to fit within given dimensions (or renders at full size)
    - Multi-page stacking (vertical) for multi-page tests
    - Shows placeholder when no PDF loaded
    - Caches rendered image
    """

    # Signal emitted when PDF is clicked
    clicked = Signal()

    def __init__(
        self,
        width: int = 216,  # ~3 inches at 72 DPI
        height: int = 360,  # ~5 inches at 72 DPI
        scale_to_fit: bool = True,  # If False, renders at actual DPI size
        parent=None,
    ):
        super().__init__(parent)
        self.preview_width = width
        self.preview_height = height
        self.scale_to_fit = scale_to_fit
        self._current_path: Optional[Path] = None
        self._current_page: int = -1
        self._current_pages: Optional[List[int]] = None  # For multi-page
        self._current_dpi: int = 72
        self._cached_pixmap: Optional[QPixmap] = None

        self._setup_ui()

    def _setup_ui(self):
        """Build the preview UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frame for border
        self.frame = QFrame()
        self.frame.setFrameShape(QFrame.Shape.Box)
        self.frame.setStyleSheet(
            "QFrame { border: 1px solid #ccc; background-color: white; }"
        )

        if self.scale_to_fit:
            # Fixed size for scaled preview
            self.frame.setFixedSize(self.preview_width, self.preview_height)
        else:
            # Allow frame to grow to fit content (for scrollable preview)
            self.frame.setMinimumSize(self.preview_width, self.preview_height)

        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(2, 2, 2, 2)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._show_placeholder()

        frame_layout.addWidget(self.image_label)
        layout.addWidget(self.frame)

    def _show_placeholder(self):
        """Show placeholder text when no PDF is loaded."""
        self.image_label.setText("No preview\navailable")
        self.image_label.setStyleSheet("color: #999; font-size: 12px;")

    def load_pdf(self, pdf_path: Path, page: int = 0, dpi: int = 72) -> bool:
        """
        Load and display a PDF page.

        Args:
            pdf_path: Path to the PDF file
            page: Page number (0-indexed)
            dpi: Render DPI (higher = better quality but slower)

        Returns:
            True if successful, False otherwise
        """
        if not pdf_path or not pdf_path.exists():
            self._show_placeholder()
            self._current_path = None
            self._current_page = -1
            self._current_pages = None
            return False

        # Check if we have this cached (same path AND same page AND same dpi)
        if (self._current_path == pdf_path and self._current_page == page
                and self._current_dpi == dpi and self._cached_pixmap):
            self.image_label.setPixmap(self._cached_pixmap)
            return True

        try:
            # Try pdf2image first (requires poppler)
            pixmap = self._render_with_pdf2image(pdf_path, page, dpi)

            if pixmap is None:
                # Fallback to PyMuPDF if available
                pixmap = self._render_with_pymupdf(pdf_path, page, dpi)

            if pixmap is None:
                self._show_placeholder()
                self._current_path = None
                return False

            scaled = self._apply_scaling(pixmap)

            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet("")  # Clear placeholder style
            self._current_path = pdf_path
            self._current_page = page
            self._current_pages = None
            self._current_dpi = dpi
            self._cached_pixmap = scaled
            return True

        except Exception as e:
            print(f"PDF preview error: {e}")
            self._show_placeholder()
            self._current_path = None
            self._current_page = -1
            return False

    def load_pdf_pages(self, pdf_path: Path, pages: List[int], dpi: int = 72) -> bool:
        """
        Load and display multiple PDF pages stacked vertically.

        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers (0-indexed)
            dpi: Render DPI

        Returns:
            True if successful, False otherwise
        """
        if not pdf_path or not pdf_path.exists() or not pages:
            self._show_placeholder()
            self._current_path = None
            self._current_pages = None
            return False

        # Check cache
        if (self._current_path == pdf_path and self._current_pages == pages
                and self._current_dpi == dpi and self._cached_pixmap):
            self.image_label.setPixmap(self._cached_pixmap)
            return True

        try:
            # Render each page
            pixmaps = []
            for page in pages:
                pixmap = self._render_with_pdf2image(pdf_path, page, dpi)
                if pixmap is None:
                    pixmap = self._render_with_pymupdf(pdf_path, page, dpi)
                if pixmap:
                    pixmaps.append(pixmap)

            if not pixmaps:
                self._show_placeholder()
                self._current_path = None
                self._current_pages = None
                return False

            # Stack pages vertically with a small gap
            gap = 4
            total_height = sum(p.height() for p in pixmaps) + gap * (len(pixmaps) - 1)
            max_width = max(p.width() for p in pixmaps)

            combined = QPixmap(max_width, total_height)
            combined.fill(QColor("#f0f0f0"))

            painter = QPainter(combined)
            y_offset = 0
            for pixmap in pixmaps:
                # Center horizontally if pages have different widths
                x_offset = (max_width - pixmap.width()) // 2
                painter.drawPixmap(x_offset, y_offset, pixmap)
                y_offset += pixmap.height() + gap
            painter.end()

            scaled = self._apply_scaling(combined)

            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet("")
            self._current_path = pdf_path
            self._current_page = pages[0]
            self._current_pages = list(pages)
            self._current_dpi = dpi
            self._cached_pixmap = scaled
            return True

        except Exception as e:
            print(f"PDF multi-page preview error: {e}")
            self._show_placeholder()
            self._current_path = None
            self._current_pages = None
            return False

    def set_scale_to_fit(self, enabled: bool):
        """
        Toggle between fit-to-container and full-size modes.

        Reloads the current page(s) if any are loaded.
        """
        if self.scale_to_fit == enabled:
            return

        self.scale_to_fit = enabled

        if enabled:
            self.frame.setFixedSize(self.preview_width, self.preview_height)
        else:
            self.frame.setMinimumSize(self.preview_width, self.preview_height)
            self.frame.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX

        # Invalidate cache and reload if we have a current page
        self._cached_pixmap = None
        if self._current_path:
            if self._current_pages:
                self.load_pdf_pages(self._current_path, self._current_pages, self._current_dpi)
            elif self._current_page >= 0:
                self.load_pdf(self._current_path, self._current_page, self._current_dpi)

    def _apply_scaling(self, pixmap: QPixmap) -> QPixmap:
        """Apply scaling based on current mode."""
        if self.scale_to_fit:
            return pixmap.scaled(
                self.preview_width - 4,
                self.preview_height - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            # Use rendered pixmap at full size (for scrollable preview)
            self.frame.setFixedSize(pixmap.width() + 4, pixmap.height() + 4)
            return pixmap

    def _render_with_pdf2image(
        self, pdf_path: Path, page: int, dpi: int
    ) -> Optional[QPixmap]:
        """Render PDF using pdf2image (poppler)."""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(
                str(pdf_path),
                first_page=page + 1,
                last_page=page + 1,
                dpi=dpi,
            )

            if not images:
                return None

            # Convert PIL image to QPixmap
            pil_image = images[0]

            # Convert to RGB if necessary
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            data = pil_image.tobytes("raw", "RGB")
            qimage = QImage(
                data,
                pil_image.width,
                pil_image.height,
                pil_image.width * 3,
                QImage.Format.Format_RGB888,
            )

            return QPixmap.fromImage(qimage)

        except ImportError:
            return None
        except Exception as e:
            print(f"pdf2image error: {e}")
            return None

    def _render_with_pymupdf(
        self, pdf_path: Path, page: int, dpi: int
    ) -> Optional[QPixmap]:
        """Render PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            if page >= len(doc):
                return None

            pdf_page = doc[page]

            # Calculate zoom factor for desired DPI
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            pix = pdf_page.get_pixmap(matrix=mat)

            # Convert to QImage
            if pix.alpha:
                qimage = QImage(
                    pix.samples,
                    pix.width,
                    pix.height,
                    pix.stride,
                    QImage.Format.Format_RGBA8888,
                )
            else:
                qimage = QImage(
                    pix.samples,
                    pix.width,
                    pix.height,
                    pix.stride,
                    QImage.Format.Format_RGB888,
                )

            doc.close()
            return QPixmap.fromImage(qimage)

        except ImportError:
            return None
        except Exception as e:
            print(f"PyMuPDF error: {e}")
            return None

    def clear(self):
        """Clear the preview."""
        self._show_placeholder()
        self._current_path = None
        self._current_page = -1
        self._current_pages = None
        self._cached_pixmap = None

    def mousePressEvent(self, event):
        """Emit clicked signal on mouse press."""
        self.clicked.emit()
        super().mousePressEvent(event)
