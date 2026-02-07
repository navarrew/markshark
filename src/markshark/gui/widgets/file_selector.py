"""
File and folder selection widget with label, text field, and browse button.
"""

from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)


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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(label_width)

        self.path_edit = QLineEdit()
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
        if self.directory_mode:
            path = QFileDialog.getExistingDirectory(
                self,
                f"Select {self.label.text().rstrip(':')}",
                self.path_edit.text() or "",
            )
        elif self.save_mode:
            path, _ = QFileDialog.getSaveFileName(
                self,
                f"Select {self.label.text().rstrip(':')}",
                self.path_edit.text() or "",
                self.file_filter,
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {self.label.text().rstrip(':')}",
                self.path_edit.text() or "",
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

    def trigger_browse(self):
        """Programmatically trigger the browse dialog."""
        self._browse()
