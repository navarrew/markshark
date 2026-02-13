"""
Mock Data Utility - generate synthetic student datasets for any template.

Wraps markshark.mock_dataset.generate_mock_dataset() with a PySide6 GUI
that mirrors the Streamlit version's capabilities: template selection,
configurable parameters (students, DPI, darkness, blank/multi rates),
threaded generation, and results display.
"""

import platform
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QScrollArea,
)

from ..widgets import PageHeader, FileSelector, ProjectSelector

# Template manager (best-effort import)
try:
    from markshark.template_manager import TemplateManager, BubbleSheetTemplate
except ImportError:
    TemplateManager = None
    BubbleSheetTemplate = None

# Mock dataset generator (best-effort import)
try:
    from markshark.mock_dataset import generate_mock_dataset
except ImportError:
    generate_mock_dataset = None


# ---------------------------------------------------------------------------
# Worker thread for generation
# ---------------------------------------------------------------------------
class _GenerateWorker(QThread):
    """Run generate_mock_dataset in a background thread."""

    finished = Signal(dict)   # emits results dict on success
    errored = Signal(str)     # emits error message on failure

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self._kwargs = kwargs

    def run(self):
        try:
            results = generate_mock_dataset(**self._kwargs)
            # Convert Path values to str for safe cross-thread transfer
            self.finished.emit({k: str(v) for k, v in results.items()})
        except Exception as e:
            self.errored.emit(str(e))


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
class MockDataPage(QWidget):
    """
    Mock Data Generator - create synthetic student datasets for any template.

    Layout:
        Template selector + info
        Settings (single box, two columns: left=generation, right=quality)
        Output directory
        Generate button
        Results panel (shown after generation)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._template_manager: Optional[TemplateManager] = None
        self._templates: List = []
        self._worker: Optional[_GenerateWorker] = None

        self._init_template_manager()
        self._setup_ui()

    def _init_template_manager(self):
        """Initialize the template manager."""
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
        """Build the page UI."""
        outer = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark Mock Data Generation Utility",
            "Create a mock student dataset for any bubblesheet template.  "
            "Provides unaligned PDFs, a key, class roster, and a table of "
            "mock answers provided by students."
        )
        outer.addWidget(header)

        # Project selector (synced with other pages via MainWindow)
        self.project_selector = ProjectSelector()
        self.project_selector.project_changed.connect(self._on_project_changed)
        outer.addWidget(self.project_selector)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)

        # === Template selection ===
        tmpl_group = QGroupBox("Template")
        tmpl_layout = QVBoxLayout(tmpl_group)

        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        tmpl_row.addWidget(self.template_combo, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_templates)
        tmpl_row.addWidget(refresh_btn)
        tmpl_layout.addLayout(tmpl_row)

        self.template_info_label = QLabel("Select a template above.")
        self.template_info_label.setStyleSheet("color: #666; font-size: 11px;")
        tmpl_layout.addWidget(self.template_info_label)

        layout.addWidget(tmpl_group)
        layout.addSpacing(8)

        # === Settings — single box, two columns ===
        settings_group = QGroupBox("Settings")
        grid = QGridLayout(settings_group)
        grid.setColumnStretch(0, 0)  # left labels
        grid.setColumnStretch(1, 1)  # left controls
        grid.setColumnStretch(2, 0)  # right labels
        grid.setColumnStretch(3, 1)  # right controls

        # ── Left column: Dataset ──
        row = 0
        left_header = QLabel("Dataset")
        left_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(left_header, row, 0, 1, 2)

        row += 1
        grid.addWidget(QLabel("Students:"), row, 0)
        self.num_students_spin = QSpinBox()
        self.num_students_spin.setRange(1, 500)
        self.num_students_spin.setValue(40)
        self.num_students_spin.setToolTip("How many fake student sheets to generate")
        grid.addWidget(self.num_students_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Versions:"), row, 0)
        self.num_versions_spin = QSpinBox()
        self.num_versions_spin.setRange(1, 10)
        self.num_versions_spin.setValue(2)
        self.num_versions_spin.setToolTip(
            "Number of exam versions (e.g. 1=single version, 2=A/B, 3=A/B/C).\n"
            "Only used when the template has a version field."
        )
        grid.addWidget(self.num_versions_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Absent students:"), row, 0)
        self.absent_spin = QSpinBox()
        self.absent_spin.setRange(0, 20)
        self.absent_spin.setValue(2)
        self.absent_spin.setToolTip(
            "Absent students appear in the roster but have no scantron"
        )
        grid.addWidget(self.absent_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Random seed:"), row, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(42)
        self.seed_spin.setToolTip("Use the same seed to reproduce identical results")
        grid.addWidget(self.seed_spin, row, 1)

        row += 1
        sep_left = QFrame()
        sep_left.setFrameShape(QFrame.Shape.HLine)
        sep_left.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep_left, row, 0, 1, 2)

        row += 1
        rate_header = QLabel("Error Rates")
        rate_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(rate_header, row, 0, 1, 2)

        row += 1
        grid.addWidget(QLabel("Blank answer rate:"), row, 0)
        self.blank_rate_spin = QDoubleSpinBox()
        self.blank_rate_spin.setRange(0.0, 0.10)
        self.blank_rate_spin.setDecimals(3)
        self.blank_rate_spin.setSingleStep(0.005)
        self.blank_rate_spin.setValue(0.01)
        self.blank_rate_spin.setToolTip("Fraction of wrong answers left blank")
        grid.addWidget(self.blank_rate_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Multi-fill rate:"), row, 0)
        self.multi_rate_spin = QDoubleSpinBox()
        self.multi_rate_spin.setRange(0.0, 0.10)
        self.multi_rate_spin.setDecimals(3)
        self.multi_rate_spin.setSingleStep(0.005)
        self.multi_rate_spin.setValue(0.01)
        self.multi_rate_spin.setToolTip("Fraction of wrong answers with multiple marks")
        grid.addWidget(self.multi_rate_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("ID mis-entries:"), row, 0)
        self.num_id_errors_spin = QSpinBox()
        self.num_id_errors_spin.setRange(0, 50)
        self.num_id_errors_spin.setValue(2)
        self.num_id_errors_spin.setToolTip(
            "Number of students with corrupted IDs "
            "(typo, transposition, extra/missing digit)"
        )
        grid.addWidget(self.num_id_errors_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Missing version:"), row, 0)
        self.num_missing_version_spin = QSpinBox()
        self.num_missing_version_spin.setRange(0, 50)
        self.num_missing_version_spin.setValue(2)
        self.num_missing_version_spin.setToolTip(
            "Number of students with blank version field"
        )
        grid.addWidget(self.num_missing_version_spin, row, 1)

        # ── Right column: Scan Quality ──
        row_r = 0
        right_header = QLabel("Scan Quality")
        right_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(right_header, row_r, 2, 1, 2)

        row_r += 1
        grid.addWidget(QLabel("DPI:"), row_r, 2)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 600)
        self.dpi_spin.setValue(150)
        self.dpi_spin.setToolTip("Image resolution (150 recommended to start)")
        grid.addWidget(self.dpi_spin, row_r, 3)

        row_r += 1
        grid.addWidget(QLabel("Min darkness:"), row_r, 2)
        self.darkness_spin = QDoubleSpinBox()
        self.darkness_spin.setRange(0.2, 1.0)
        self.darkness_spin.setDecimals(2)
        self.darkness_spin.setSingleStep(0.05)
        self.darkness_spin.setValue(0.50)
        self.darkness_spin.setToolTip(
            "Minimum bubble darkness (lower = lighter pencil marks)"
        )
        grid.addWidget(self.darkness_spin, row_r, 3)

        row_r += 1
        self.transform_cb = QCheckBox("Random rotation/translation")
        self.transform_cb.setChecked(True)
        self.transform_cb.setToolTip("Simulate slightly misaligned scans")
        grid.addWidget(self.transform_cb, row_r, 2, 1, 2)

        row_r += 1
        sep_right = QFrame()
        sep_right.setFrameShape(QFrame.Shape.HLine)
        sep_right.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep_right, row_r, 2, 1, 2)

        row_r += 1
        key_header = QLabel("Answer Key")
        key_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(key_header, row_r, 2, 1, 2)

        row_r += 1
        grid.addWidget(QLabel("AND keys:"), row_r, 2)
        self.num_and_keys_spin = QSpinBox()
        self.num_and_keys_spin.setRange(0, 50)
        self.num_and_keys_spin.setValue(0)
        self.num_and_keys_spin.setToolTip(
            "Number of questions with AND keys (e.g. B&C).\n"
            "Student must fill ALL specified bubbles to get credit."
        )
        grid.addWidget(self.num_and_keys_spin, row_r, 3)

        row_r += 1
        grid.addWidget(QLabel("OR keys:"), row_r, 2)
        self.num_or_keys_spin = QSpinBox()
        self.num_or_keys_spin.setRange(0, 50)
        self.num_or_keys_spin.setValue(0)
        self.num_or_keys_spin.setToolTip(
            "Number of questions with OR keys (e.g. B^C).\n"
            "Student can fill ANY one of the specified bubbles for credit."
        )
        grid.addWidget(self.num_or_keys_spin, row_r, 3)

        row_r += 1
        grid.addWidget(QLabel("Default points:"), row_r, 2)
        self.default_points_spin = QSpinBox()
        self.default_points_spin.setRange(1, 10)
        self.default_points_spin.setValue(1)
        self.default_points_spin.setToolTip(
            "Default points per question (adds 'default:N' to key header)"
        )
        grid.addWidget(self.default_points_spin, row_r, 3)

        row_r += 1
        grid.addWidget(QLabel("Weighted questions:"), row_r, 2)
        self.num_double_points_spin = QSpinBox()
        self.num_double_points_spin.setRange(0, 50)
        self.num_double_points_spin.setValue(0)
        self.num_double_points_spin.setToolTip(
            "Number of questions worth double the default points\n"
            "(or 1 point if default > 1)"
        )
        grid.addWidget(self.num_double_points_spin, row_r, 3)

        layout.addWidget(settings_group)
        layout.addSpacing(8)

        # === Output directory ===
        self.output_dir_selector = FileSelector(
            "Output directory:",
            "",
            "Where to save generated files...",
            directory_mode=True,
        )
        layout.addWidget(self.output_dir_selector)

        # === Generate button ===
        gen_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Mock Dataset")
        self.generate_btn.setStyleSheet(
            "QPushButton { padding: 8px 24px; font-weight: bold; }"
        )
        self.generate_btn.clicked.connect(self._on_generate)
        self.generate_btn.setEnabled(False)  # enabled when template selected
        gen_layout.addWidget(self.generate_btn)
        gen_layout.addStretch()
        layout.addLayout(gen_layout)

        # === Status ===
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # === Results (hidden until generation completes) ===
        self.results_group = QGroupBox("Generated Files")
        results_layout = QVBoxLayout(self.results_group)

        self.results_label = QLabel("")
        self.results_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        open_btn_layout = QHBoxLayout()
        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(self._open_output_folder)
        open_btn_layout.addWidget(self.open_folder_btn)
        open_btn_layout.addStretch()
        results_layout.addLayout(open_btn_layout)

        self.results_group.setVisible(False)
        layout.addWidget(self.results_group)

        # Stretch at bottom
        layout.addStretch()

        # Populate templates
        self._refresh_templates()

    # -------------------------------------------------------------------
    # Template management
    # -------------------------------------------------------------------
    def _refresh_templates(self):
        """Refresh the template dropdown."""
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        self._templates = []

        if not self._template_manager:
            self.template_combo.addItem("(template manager not available)")
            self.template_combo.blockSignals(False)
            return

        try:
            self._template_manager._templates_cache = None
            self._templates = self._template_manager.scan_templates(force_refresh=True)
        except Exception as e:
            print(f"Error scanning templates: {e}")

        self.template_combo.addItem("(select a template)")
        for t in self._templates:
            self.template_combo.addItem(t.display_name)

        self.template_combo.blockSignals(False)
        self.template_combo.setCurrentIndex(0)
        self._on_template_changed(0)

    def _on_template_changed(self, index: int):
        """Update info label and enable/disable generate button."""
        if index <= 0 or index > len(self._templates):
            self.template_info_label.setText("Select a template above.")
            self.generate_btn.setEnabled(False)
            return

        t = self._templates[index - 1]
        parts = []
        if t.num_questions:
            parts.append(f"Questions: {t.num_questions}")
        if t.num_choices:
            parts.append(f"Choices: {t.num_choices}")
        elif t.choices_label:
            parts.append(f"Choices: {t.choices_label}")
        if t.num_pages:
            parts.append(f"Pages: {t.num_pages}")
        if t.description:
            parts.append(t.description)

        self.template_info_label.setText(" | ".join(parts) if parts else t.template_id)

        # Set default output dir based on template
        if not self.output_dir_selector.path():
            default_dir = str(Path.cwd() / f"mock_{t.template_id}")
            self.output_dir_selector.set_path(default_dir)

        self.generate_btn.setEnabled(generate_mock_dataset is not None)

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------
    def _on_generate(self):
        """Launch mock dataset generation in a background thread."""
        idx = self.template_combo.currentIndex()
        if idx <= 0 or idx > len(self._templates):
            QMessageBox.warning(self, "No Template", "Please select a template first.")
            return

        template = self._templates[idx - 1]

        if not template.template_pdf_path or not template.template_pdf_path.exists():
            QMessageBox.warning(self, "Missing PDF", "Template PDF file not found.")
            return
        if not template.bubblemap_yaml_path or not template.bubblemap_yaml_path.exists():
            QMessageBox.warning(self, "Missing YAML", "Bubblemap YAML file not found.")
            return

        out_dir = self.output_dir_selector.path()
        if not out_dir:
            QMessageBox.warning(self, "No Output", "Please specify an output directory.")
            return

        if generate_mock_dataset is None:
            QMessageBox.warning(
                self, "Not Available",
                "markshark.mock_dataset module is not installed."
            )
            return

        # Build kwargs
        kwargs = {
            "template_path": str(template.template_pdf_path),
            "bubblemap_path": str(template.bubblemap_yaml_path),
            "out_dir": out_dir,
            "num_students": self.num_students_spin.value(),
            "num_absent": self.absent_spin.value(),
            "num_versions": self.num_versions_spin.value(),
            "seed": self.seed_spin.value(),
            "dpi": self.dpi_spin.value(),
            "darkness_min": self.darkness_spin.value(),
            "darkness_max": 1.0,
            "apply_transform": self.transform_cb.isChecked(),
            "blank_rate": self.blank_rate_spin.value(),
            "multi_rate": self.multi_rate_spin.value(),
            "num_id_errors": self.num_id_errors_spin.value(),
            "num_missing_version": self.num_missing_version_spin.value(),
            "num_and_keys": self.num_and_keys_spin.value(),
            "num_or_keys": self.num_or_keys_spin.value(),
            "default_points": self.default_points_spin.value(),
            "num_double_points": self.num_double_points_spin.value(),
            "verbose": False,
        }

        # Disable UI during generation
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        self.results_group.setVisible(False)
        num = self.num_students_spin.value()
        self.status_label.setText(f"Generating {num} mock students... please wait.")
        self.status_label.setStyleSheet("color: #1565C0;")

        # Launch worker thread
        self._worker = _GenerateWorker(kwargs, self)
        self._worker.finished.connect(self._on_generate_finished)
        self._worker.errored.connect(self._on_generate_error)
        self._worker.start()

    def _on_generate_finished(self, results: dict):
        """Handle successful generation."""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Mock Dataset")

        self.status_label.setText("Mock dataset generated successfully!")
        self.status_label.setStyleSheet("color: green;")

        # Show results
        lines = []
        for label, key in [
            ("Answer Key", "answer_key"),
            ("Scans PDF", "scans"),
            ("Responses CSV", "responses"),
            ("Roster CSV", "roster"),
        ]:
            path = results.get(key, "")
            if path:
                lines.append(f"<b>{label}:</b> {Path(path).name}")
        lines.append(f"<br><b>Location:</b> {Path(results.get('scans', '')).parent}")

        absent = self.absent_spin.value()
        if absent > 0:
            lines.append(f"<i>Roster includes {absent} absent student(s)</i>")

        num_ver = self.num_versions_spin.value()
        lines.append(f"<i>Generated with {num_ver} version(s)</i>")

        self.results_label.setText("<br>".join(lines))
        self._last_output_dir = str(Path(results.get("scans", "")).parent)
        self.results_group.setVisible(True)

        self._worker = None

    def _on_generate_error(self, error_msg: str):
        """Handle generation error."""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Mock Dataset")

        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: red;")

        QMessageBox.warning(
            self, "Generation Error",
            f"Failed to generate mock dataset:\n\n{error_msg}"
        )
        self._worker = None

    def _open_output_folder(self):
        """Open the output folder in the system file manager."""
        folder = getattr(self, "_last_output_dir", None)
        if not folder or not Path(folder).exists():
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

    def _on_project_changed(self, name: str):
        """Update output directory when the active project changes."""
        project_dir = self.project_selector.project_dir()
        if project_dir and project_dir.exists():
            self.output_dir_selector.set_path(str(project_dir / "input_files"))
