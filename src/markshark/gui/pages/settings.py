"""
Settings page - application preferences and configuration.
"""

from pathlib import Path
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
)

from ..widgets import FileSelector, PageHeader


class SettingsPage(QWidget):
    """Application settings and preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings()
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header with icon
        header = PageHeader("Settings", "Configure application preferences.")
        layout.addWidget(header)

        # Default paths
        paths_group = QGroupBox("Default Paths")
        paths_layout = QVBoxLayout(paths_group)

        self.templates_dir = FileSelector(
            "Templates directory:",
            "",
            "Where to find bubble sheet templates...",
            directory_mode=True,
        )
        paths_layout.addWidget(self.templates_dir)

        self.output_dir = FileSelector(
            "Default output directory:",
            "",
            "Where to save results...",
            directory_mode=True,
        )
        paths_layout.addWidget(self.output_dir)

        layout.addWidget(paths_group)

        # Default options
        defaults_group = QGroupBox("Default Options")
        defaults_layout = QFormLayout(defaults_group)

        self.default_dpi = QSpinBox()
        self.default_dpi.setRange(72, 600)
        self.default_dpi.setValue(150)
        defaults_layout.addRow("Render DPI:", self.default_dpi)

        self.default_align_method = QComboBox()
        self.default_align_method.addItems(["auto", "fast", "slow", "aruco"])
        defaults_layout.addRow("Alignment method:", self.default_align_method)

        self.auto_open_results = QCheckBox("Automatically open results after grading")
        self.auto_open_results.setChecked(True)
        defaults_layout.addRow("", self.auto_open_results)

        layout.addWidget(defaults_group)

        # Buttons
        btn_layout = QHBoxLayout()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_settings)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

    def _load_settings(self):
        """Load settings from QSettings."""
        self.templates_dir.set_path(self.settings.value("paths/templates_dir", ""))
        self.output_dir.set_path(self.settings.value("paths/output_dir", ""))
        self.default_dpi.setValue(int(self.settings.value("defaults/dpi", 150)))
        self.default_align_method.setCurrentText(
            self.settings.value("defaults/align_method", "auto")
        )
        self.auto_open_results.setChecked(
            self.settings.value("defaults/auto_open", True, type=bool)
        )

    def _save_settings(self):
        """Save settings to QSettings."""
        self.settings.setValue("paths/templates_dir", self.templates_dir.path())
        self.settings.setValue("paths/output_dir", self.output_dir.path())
        self.settings.setValue("defaults/dpi", self.default_dpi.value())
        self.settings.setValue("defaults/align_method", self.default_align_method.currentText())
        self.settings.setValue("defaults/auto_open", self.auto_open_results.isChecked())
        self.settings.sync()

        QMessageBox.information(self, "Settings Saved", "Your settings have been saved.")

    def _reset_settings(self):
        """Reset to default values."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.templates_dir.clear()
            self.output_dir.clear()
            self.default_dpi.setValue(150)
            self.default_align_method.setCurrentText("auto")
            self.auto_open_results.setChecked(True)
            self._save_settings()
