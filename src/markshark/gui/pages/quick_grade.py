"""
Grader page - the main grading workflow.

Mirrors the Streamlit "Quick Grade" functionality:
1. Select template
2. Upload scans, key, roster
3. Configure options
4. Run align & score
5. Generate report
"""

import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QTabWidget,
    QProgressBar,
    QSplitter,
    QMessageBox,
    QFrame,
)

from ..widgets import FileSelector, LogViewer, ProjectSelector, PageHeader
from ..workers import CLIRunner


# Try to import MarkShark defaults
try:
    from markshark.defaults import (
        SCORING_DEFAULTS,
        ALIGN_DEFAULTS,
        RENDER_DEFAULTS,
    )
    from markshark.template_manager import TemplateManager
except ImportError:
    SCORING_DEFAULTS = ALIGN_DEFAULTS = RENDER_DEFAULTS = None
    TemplateManager = None


def _dflt(obj, attr: str, fallback):
    """Best-effort defaults helper."""
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)


class QuickGradePage(QWidget):
    """
    Quick Grade workflow page.

    Signals:
        grading_complete: Emitted when align+score finishes successfully
    """

    grading_complete = Signal(dict)  # Emits results data dict

    def __init__(self, parent=None):
        super().__init__(parent)

        # CLI runner for async operations
        self.runner = CLIRunner(self)
        self.runner.output_received.connect(self._on_output)
        self.runner.error_received.connect(self._on_output)
        self.runner.finished.connect(self._on_step_finished)

        # State
        self.work_dir: Optional[Path] = None
        self.aligned_pdf: Optional[Path] = None
        self.results_csv: Optional[Path] = None
        self.scored_pdf: Optional[Path] = None
        self._pending_score = False
        self._current_bubblemap: Optional[str] = None

        self._setup_ui()
        self._load_templates()

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header with icon
        header = PageHeader(
            "Quick Grade",
            "Upload scans, answer key, and roster (optional), then align, score, and generate reports."
        )
        layout.addWidget(header)

        # Project/directory selector bar
        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Action buttons
        btn_layout = QHBoxLayout()

        self.align_score_btn = QPushButton("1. Align && Score")
        self.align_score_btn.setMinimumHeight(36)
        self.align_score_btn.clicked.connect(self._run_align_and_score)
        btn_layout.addWidget(self.align_score_btn)

        self.report_btn = QPushButton("2. Generate Report")
        self.report_btn.setMinimumHeight(36)
        self.report_btn.clicked.connect(self._run_report)
        self.report_btn.setEnabled(False)
        btn_layout.addWidget(self.report_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Splitter for inputs and log
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        # Tabs for inputs
        tabs = QTabWidget()
        tabs.addTab(self._create_inputs_tab(), "Input Files")
        tabs.addTab(self._create_after_tab(), "Upload Corrections")
        tabs.addTab(self._create_options_tab(), "Options")
        splitter.addWidget(tabs)

        # Log viewer
        self.log = LogViewer("Processing Log")
        splitter.addWidget(self.log)

        splitter.setSizes([350, 200])

        # Output buttons (hidden initially)
        self.output_frame = QFrame()
        output_layout = QHBoxLayout(self.output_frame)
        output_layout.setContentsMargins(0, 5, 0, 0)

        self.open_csv_btn = QPushButton("Open results.csv")
        self.open_csv_btn.clicked.connect(lambda: self._open_file(self.results_csv))
        output_layout.addWidget(self.open_csv_btn)

        self.open_scored_btn = QPushButton("Open scored.pdf")
        self.open_scored_btn.clicked.connect(lambda: self._open_file(self.scored_pdf))
        output_layout.addWidget(self.open_scored_btn)

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(lambda: self._open_file(self.work_dir))
        output_layout.addWidget(self.open_folder_btn)

        output_layout.addStretch()
        self.output_frame.setVisible(False)
        layout.addWidget(self.output_frame)

    def _create_inputs_tab(self) -> QWidget:
        """Create the Inputs tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Template selection
        template_group = QGroupBox("Choose your template bubblesheet")
        template_layout = QHBoxLayout(template_group)
        template_layout.addWidget(QLabel("Select template:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(300)
        template_layout.addWidget(self.template_combo, 1)
        layout.addWidget(template_group)

        # File inputs
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)

        self.scans_selector = FileSelector(
            "Scanned answer sheets:",
            "PDF files (*.pdf)",
            "Select scanned PDF...",
        )
        files_layout.addWidget(self.scans_selector)

        self.key_selector = FileSelector(
            "Answer key:",
            "Key files (*.txt *.csv *.tsv *.xlsx)",
            "Select answer key...",
        )
        files_layout.addWidget(self.key_selector)

        self.roster_selector = FileSelector(
            "Class roster (optional):",
            "CSV files (*.csv)",
            "Optional roster CSV...",
        )
        files_layout.addWidget(self.roster_selector)

        layout.addWidget(files_group)

        layout.addStretch()
        return widget

    def _create_options_tab(self) -> QWidget:
        """Create the Options tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scoring options
        scoring_group = QGroupBox("Scoring Options")
        scoring_layout = QGridLayout(scoring_group)

        self.annotate_all_cb = QCheckBox("Annotate all bubbles")
        self.annotate_all_cb.setChecked(True)
        scoring_layout.addWidget(self.annotate_all_cb, 0, 0)

        self.label_density_cb = QCheckBox("Show % fill labels")
        self.label_density_cb.setChecked(True)
        scoring_layout.addWidget(self.label_density_cb, 0, 1)

        self.auto_thresh_cb = QCheckBox("Auto-calibrate threshold")
        self.auto_thresh_cb.setChecked(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
        scoring_layout.addWidget(self.auto_thresh_cb, 1, 0)

        self.verbose_thresh_cb = QCheckBox("Verbose threshold")
        self.verbose_thresh_cb.setChecked(True)
        scoring_layout.addWidget(self.verbose_thresh_cb, 1, 1)

        self.review_pdf_cb = QCheckBox("Generate review PDF")
        self.review_pdf_cb.setChecked(True)
        scoring_layout.addWidget(self.review_pdf_cb, 2, 0)

        self.flagged_xlsx_cb = QCheckBox("Generate flagged XLSX")
        self.flagged_xlsx_cb.setChecked(True)
        scoring_layout.addWidget(self.flagged_xlsx_cb, 2, 1)

        self.include_stats_cb = QCheckBox("Include statistics")
        self.include_stats_cb.setChecked(True)
        scoring_layout.addWidget(self.include_stats_cb, 3, 0)

        layout.addWidget(scoring_group)

        # Alignment options
        align_group = QGroupBox("Alignment Options")
        align_layout = QGridLayout(align_group)

        align_layout.addWidget(QLabel("Method:"), 0, 0)
        self.align_method_combo = QComboBox()
        self.align_method_combo.addItems(["auto", "fast", "slow", "aruco"])
        align_layout.addWidget(self.align_method_combo, 0, 1)

        align_layout.addWidget(QLabel("Min ArUco markers:"), 1, 0)
        self.min_markers_spin = QSpinBox()
        self.min_markers_spin.setRange(0, 32)
        self.min_markers_spin.setValue(_dflt(ALIGN_DEFAULTS, "min_aruco", 4))
        align_layout.addWidget(self.min_markers_spin, 1, 1)

        align_layout.addWidget(QLabel("Render DPI:"), 2, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(_dflt(RENDER_DEFAULTS, "dpi", 150))
        align_layout.addWidget(self.dpi_spin, 2, 1)

        layout.addWidget(align_group)

        layout.addStretch()
        return widget

    def _create_after_tab(self) -> QWidget:
        """Create the After Scoring tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Corrections
        corrections_group = QGroupBox("Apply Corrections")
        corrections_layout = QVBoxLayout(corrections_group)
        corrections_layout.addWidget(QLabel(
            "After reviewing flagged items, upload the filled corrections file:"
        ))
        self.corrections_selector = FileSelector(
            "Corrections XLSX:",
            "Excel files (*.xlsx)",
            "Optional corrections file...",
        )
        corrections_layout.addWidget(self.corrections_selector)
        layout.addWidget(corrections_group)

        layout.addStretch()
        return widget

    def _load_templates(self):
        """Load templates into the combo box."""
        self.template_combo.clear()
        self.template_combo.addItem("(Select a template)", None)

        if TemplateManager is None:
            self.template_combo.addItem("(TemplateManager not available)", None)
            return

        try:
            tm = TemplateManager()
            for t in tm.scan_templates():
                self.template_combo.addItem(t.display_name, t)
        except Exception as e:
            self.template_combo.addItem(f"(Error: {e})", None)

    def browse_scans(self):
        """Trigger the scans file browser (called from main window menu)."""
        self.scans_selector.trigger_browse()

    def _validate_inputs(self) -> bool:
        """Validate required inputs."""
        if not self.scans_selector.exists():
            QMessageBox.warning(self, "Missing Input", "Please select a scanned PDF file.")
            return False

        template = self.template_combo.currentData()
        if template is None:
            QMessageBox.warning(self, "Missing Template", "Please select a template.")
            return False

        return True

    def _run_align_and_score(self):
        """Run the align and score workflow."""
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return

        if not self._validate_inputs():
            return

        # Setup work directory from project selector
        self.work_dir = self.project_selector.output_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Update status with project info
        project_name = self.project_selector.project_name()
        if project_name:
            self.log.append_line(f"Project: {project_name}")
        self.log.append_line(f"Output directory: {self.work_dir}\n")

        # Get template
        template = self.template_combo.currentData()
        template_pdf = str(template.template_pdf_path)
        bubblemap = str(template.bubblemap_yaml_path)
        self._current_bubblemap = bubblemap

        # Output paths
        self.aligned_pdf = self.work_dir / "aligned_scans.pdf"
        self.results_csv = self.work_dir / "results.csv"
        self.scored_pdf = self.work_dir / "scored.pdf"

        # UI updates
        self.log.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.align_score_btn.setEnabled(False)
        self.status_label.setText("Step 1/2: Aligning scans...")

        self.log.append_header("STEP 1: ALIGNMENT")

        # Build align command
        args = [
            "align",
            self.scans_selector.path(),
            "--template", template_pdf,
            "--out-pdf", str(self.aligned_pdf),
            "--dpi", str(self.dpi_spin.value()),
            "--align-method", self.align_method_combo.currentText(),
            "--min-markers", str(self.min_markers_spin.value()),
            "--bubblemap", bubblemap,
        ]

        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self._pending_score = True
        self.runner.run(args, "align")

    def _run_score(self):
        """Run the score step after alignment."""
        self.status_label.setText("Step 2/2: Scoring sheets...")
        self.log.append_header("STEP 2: SCORING")

        args = [
            "score",
            str(self.aligned_pdf),
            "--bublmap", self._current_bubblemap,
            "--out-csv", str(self.results_csv),
            "--out-pdf", "scored.pdf",
            "--dpi", str(self.dpi_spin.value()),
        ]

        # Key file
        if self.key_selector.exists():
            args += ["--key-txt", self.key_selector.path()]

        # Options
        if self.annotate_all_cb.isChecked():
            args += ["--annotate-all-cells"]
        if self.label_density_cb.isChecked():
            args += ["--label-density"]
        if not self.auto_thresh_cb.isChecked():
            args += ["--no-auto-thresh"]
        if self.verbose_thresh_cb.isChecked():
            args += ["--verbose-thresh"]
        if self.include_stats_cb.isChecked():
            args += ["--include-stats"]
        else:
            args += ["--no-include-stats"]

        # Flagging
        if self.review_pdf_cb.isChecked():
            args += ["--review-pdf", str(self.work_dir / "flagged_for_review.pdf")]
        if self.flagged_xlsx_cb.isChecked():
            args += ["--flagged-xlsx", str(self.work_dir / "flagged_for_review.xlsx")]

        # Roster
        if self.roster_selector.exists():
            args += ["--roster-csv", self.roster_selector.path()]

        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "score")

    def _run_report(self):
        """Generate the Excel report."""
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return

        if not self.results_csv or not self.results_csv.exists():
            QMessageBox.warning(self, "No Results", "Run 'Align & Score' first.")
            return

        self.status_label.setText("Generating report...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.report_btn.setEnabled(False)

        self.log.append_header("GENERATING REPORT")

        report_path = self.work_dir / "exam_report.xlsx"
        args = [
            "report",
            str(self.results_csv),
            "--out-xlsx", str(report_path),
        ]

        if self.corrections_selector.exists():
            args += ["--corrections", self.corrections_selector.path()]
            self.log.append_line(f"Applying corrections from: {self.corrections_selector.path()}")

        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "report")

    def _on_output(self, text: str):
        """Handle CLI output."""
        self.log.append(text)

    def _on_step_finished(self, exit_code: int, step_name: str):
        """Handle step completion."""
        self.log.append_line(f"\n[{step_name}] Finished with exit code {exit_code}")

        if exit_code == 0:
            if step_name == "align" and self._pending_score:
                self._pending_score = False
                self._run_score()
                return

            elif step_name == "score":
                self.status_label.setText("Quick Grade complete!")
                self.output_frame.setVisible(True)
                self.report_btn.setEnabled(True)
                self.grading_complete.emit({
                    "work_dir": str(self.work_dir),
                    "results_csv": str(self.results_csv),
                    "scored_pdf": str(self.scored_pdf),
                    "aligned_pdf": str(self.aligned_pdf),
                })
                QMessageBox.information(self, "Complete", "Alignment and scoring complete!")

            elif step_name == "report":
                self.status_label.setText("Report generated!")
                report_path = self.work_dir / "exam_report.xlsx"
                if report_path.exists():
                    QMessageBox.information(self, "Report Ready", f"Report saved to:\n{report_path}")
                    self._open_file(report_path)
        else:
            self.status_label.setText(f"Error in {step_name}")
            self._pending_score = False
            QMessageBox.warning(self, "Error", f"{step_name} failed. Check the log.")

        self.progress.setVisible(False)
        self.align_score_btn.setEnabled(True)
        self.report_btn.setEnabled(self.results_csv and self.results_csv.exists())

    def _open_file(self, path: Optional[Path]):
        """Open a file with the system default application."""
        if path is None or not path.exists():
            QMessageBox.warning(self, "Not Found", f"File not found: {path}")
            return

        import subprocess
        import platform
        import os

        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", str(path)])
            elif platform.system() == "Windows":
                os.startfile(str(path))
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open: {e}")
