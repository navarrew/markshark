"""
Course creation / editing dialog.

A single panel with three rows:
  1. Course name  — teacher-visible display name (e.g. "Biology 101")
  2. Parent folder — the existing folder on disk (e.g. /Users/teacher/BIO101)
  3. MarkShark folder name — subfolder inside the parent (e.g. "MarkShark")

Used for:
  - Creating a new course (all three fields editable, blank)
  - Editing an existing course (rename / relocate — fields pre-filled)
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
)


class CourseDialog(QDialog):
    """Dialog for creating or editing a course.

    After ``exec()``, call :meth:`result_data` to retrieve the final
    values (or *None* if cancelled).
    """

    def __init__(
        self,
        parent=None,
        *,
        title: str = "New Course",
        course_name: str = "",
        parent_folder: str = "",
        subfolder_name: str = "MarkShark",
        confirm_label: str = "Create Course",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(550)
        self._accepted = False

        self._setup_ui(course_name, parent_folder, subfolder_name, confirm_label)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(
        self,
        course_name: str,
        parent_folder: str,
        subfolder_name: str,
        confirm_label: str,
    ):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Explanation text
        hint = QLabel(
            "Set the course display name, the parent folder on disk, and the "
            "name of the subfolder MarkShark uses for its data."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(hint)

        # ── Form rows ──
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Row 1: Course name
        self.name_edit = QLineEdit(course_name)
        self.name_edit.setPlaceholderText('e.g. "Biology 101", "AP History"')
        form.addRow("Course name:", self.name_edit)

        # Row 2: Parent folder (text + browse button)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self.folder_edit = QLineEdit(parent_folder)
        self.folder_edit.setPlaceholderText("Select the folder where this course lives...")
        folder_row.addWidget(self.folder_edit, 1)

        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)

        form.addRow("Parent folder:", folder_row)

        # Row 3: Subfolder name
        self.subfolder_edit = QLineEdit(subfolder_name)
        self.subfolder_edit.setPlaceholderText('e.g. "MarkShark", "Test Results"')
        form.addRow("MarkShark folder:", self.subfolder_edit)

        # Preview label showing the resolved path
        self._preview = QLabel()
        self._preview.setStyleSheet("color: #888; font-size: 11px;")
        self._preview.setWordWrap(True)
        form.addRow("", self._preview)

        # Update preview whenever any field changes
        self.folder_edit.textChanged.connect(self._update_preview)
        self.subfolder_edit.textChanged.connect(self._update_preview)
        self._update_preview()

        layout.addLayout(form)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._confirm_btn = QPushButton(confirm_label)
        self._confirm_btn.setDefault(True)
        self._confirm_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "border: none; border-radius: 3px; padding: 6px 18px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
        )
        self._confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(self._confirm_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse_folder(self):
        """Open a folder browser for the parent folder."""
        start = self.folder_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Parent Folder", start
        )
        if path:
            self.folder_edit.setText(path)

    def _update_preview(self):
        """Show the full resolved path in the preview label."""
        parent = self.folder_edit.text().strip()
        sub = self.subfolder_edit.text().strip()
        if parent and sub:
            self._preview.setText(f"Data path:  {parent}/{sub}")
        elif parent:
            self._preview.setText(f"Data path:  {parent}")
        else:
            self._preview.setText("")

    def _on_confirm(self):
        """Validate and accept the dialog."""
        name = self.name_edit.text().strip()
        parent = self.folder_edit.text().strip()
        subfolder = self.subfolder_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Missing Field", "Please enter a course name.")
            self.name_edit.setFocus()
            return
        if not parent:
            QMessageBox.warning(self, "Missing Field", "Please select a parent folder.")
            self.folder_edit.setFocus()
            return
        if not subfolder:
            QMessageBox.warning(self, "Missing Field", "Please enter a subfolder name.")
            self.subfolder_edit.setFocus()
            return

        # Sanitize the subfolder name
        safe = "".join(
            c if c.isalnum() or c in "-_ " else "_" for c in subfolder
        ).strip()
        if not safe:
            QMessageBox.warning(
                self, "Invalid Name",
                "The subfolder name contains no valid characters."
            )
            self.subfolder_edit.setFocus()
            return

        self.subfolder_edit.setText(safe)
        self._accepted = True
        self.accept()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def result_data(self) -> Optional[dict]:
        """Return the dialog values, or None if cancelled.

        Keys:
          - ``name``: display name
          - ``parent_folder``: absolute parent path (str)
          - ``subfolder``: sanitized subfolder name
          - ``course_path``: full path (parent_folder / subfolder)
        """
        if not self._accepted:
            return None

        parent = self.folder_edit.text().strip()
        subfolder = self.subfolder_edit.text().strip()

        return {
            "name": self.name_edit.text().strip(),
            "parent_folder": parent,
            "subfolder": subfolder,
            "course_path": str(Path(parent) / subfolder),
        }
