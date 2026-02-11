"""
File and folder selection widget with label, text field, and browse button.

Supports both Browse dialog and drag-and-drop of files/folders from the
system file manager.
"""

from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)


class _DropLineEdit(QLineEdit):
    """QLineEdit that accepts file/folder drops from the system file manager."""

    def __init__(self, directory_mode: bool = False, parent=None):
        super().__init__(parent)
        self._directory_mode = directory_mode
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return

        path = urls[0].toLocalFile()
        if not path:
            return

        # In directory mode, if a file is dropped use its parent folder
        if self._directory_mode and Path(path).is_file():
            path = str(Path(path).parent)

        self.setText(path)
        self.textChanged.emit(path)
        event.acceptProposedAction()


class FileSelector(QWidget):
    """
    Reusable file/folder selection widget.

    Emits file_selected signal when a file is chosen.
    """

    file_selected = Signal(str)

    def __init__(
        self,
        label: str,
        file_filter: str = "",
        placeholder: str = "",
        save_mode: bool = False,
        directory_mode: bool = False,
        label_width: int = 180,
        parent=None,
    ):
        """
        Initialize the file selector.

        Args:
            label: Label text displayed before the text field
            file_filter: File filter for dialog (e.g., "PDF files (*.pdf)")
            placeholder: Placeholder text in the text field
            save_mode: If True, show "Save As" dialog instead of "Open"
            directory_mode: If True, select directories instead of files
            label_width: Fixed width for the label
            parent: Parent widget
        """
        super().__init__(parent)
        self.file_filter = file_filter
        self.save_mode = save_mode
        self.directory_mode = directory_mode
        self._start_dir: str = ""  # fallback start directory for browse dialog

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(label_width)

        self.path_edit = _DropLineEdit(directory_mode=self.directory_mode)
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.textChanged.connect(lambda t: self.file_selected.emit(t))

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setMaximumWidth(100)
        self.browse_btn.clicked.connect(self._browse)

        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        """Open the appropriate file dialog."""
        # Start from the current path if set, otherwise the project start dir
        start = self.path_edit.text().strip() or self._start_dir or ""
        if self.directory_mode:
            path = QFileDialog.getExistingDirectory(
                self,
                f"Select {self.label.text().rstrip(':')}",
                start,
            )
        elif self.save_mode:
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Select {self.label.text().rstrip(':')}",
                start,
                self.file_filter,
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {self.label.text().rstrip(':')}",
                start,
                self.file_filter,
            )

        if path:
            self.path_edit.setText(path)

    def path(self) -> str:
        """Get the current path as a string."""
        return self.path_edit.text().strip()

    def path_obj(self) -> Path:
        """Get the current path as a Path object."""
        return Path(self.path()) if self.path() else None

    def set_path(self, path: str):
        """Set the path programmatically."""
        self.path_edit.setText(path)

    def clear(self):
        """Clear the path."""
        self.path_edit.clear()

    def exists(self) -> bool:
        """Check if the current path exists."""
        p = self.path()
        return bool(p) and Path(p).exists()

    def set_start_dir(self, directory: str):
        """Set the fallback start directory for the browse dialog.

        Used when the text field is empty so the dialog opens in the
        project folder instead of the system default.
        """
        self._start_dir = directory

    def trigger_browse(self):
        """Programmatically trigger the browse dialog."""
        self._browse()
