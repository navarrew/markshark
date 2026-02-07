#!/usr/bin/env python3
"""
PySide6 GUI for MarkShark Quick Grade workflow.

A native desktop alternative to the Streamlit app, designed for
packaging as a standalone application for teachers.

Runs the same CLI commands as the Streamlit app:
  - markshark align
  - markshark score
  - markshark report

Usage:
  python -m markshark.quickgrade_gui
"""

import sys
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QProcess, Signal, QObject
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QMessageBox,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QGroupBox,
    QTabWidget,
    QProgressBar,
    QSplitter,
    QScrollArea,
    QFrame,
    QSizePolicy,
)


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


class ProcessRunner(QObject):
    """Helper to run CLI commands asynchronously with progress updates."""

    output_received = Signal(str)
    error_received = Signal(str)
    finished = Signal(int, str)  # exit_code, step_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._on_finished)
        self.current_step = ""

    def run(self, args: list, step_name: str = ""):
        """Run a markshark CLI command."""
        self.current_step = step_name
        program = sys.executable
        full_args = ["-m", "markshark.cli"] + args
        self.process.start(program, full_args)

    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _read_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.output_received.emit(data)

    def _read_stderr(self):
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data:
            self.error_received.emit(data)

    def _on_finished(self, exit_code: int, _exit_status):
        self.finished.emit(exit_code, self.current_step)


class FileSelector(QWidget):
    """Reusable file selection widget with label, text field, and browse button."""

    file_selected = Signal(str)

    def __init__(self, label: str, file_filter: str = "",
                 placeholder: str = "", save_mode: bool = False, parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self.save_mode = save_mode

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(180)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(placeholder)
        self.path_edit.textChanged.connect(lambda t: self.file_selected.emit(t))

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)

        layout.addWidget(self.label)
        layout.addWidget(self.path_edit, 1)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        if self.save_mode:
            path, _ = QFileDialog.getSaveFileName(
                self, f"Select {self.label.text()}", "", self.file_filter
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, f"Select {self.label.text()}", "", self.file_filter
            )
        if path:
            self.path_edit.setText(path)

    def path(self) -> str:
        return self.path_edit.text().strip()

    def set_path(self, path: str):
        self.path_edit.setText(path)

    def clear(self):
        self.path_edit.clear()


class QuickGradeWindow(QMainWindow):
    """Main window for Quick Grade workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MarkShark Quick Grade")
        self.setMinimumSize(900, 700)

        # Process runner for CLI commands
        self.runner = ProcessRunner(self)
        self.runner.output_received.connect(self._append_log)
        self.runner.error_received.connect(self._append_log)
        self.runner.finished.connect(self._on_step_finished)

        # State
        self.aligned_pdf_path: Optional[Path] = None
        self.results_csv_path: Optional[Path] = None
        self.scored_pdf_path: Optional[Path] = None
        self.work_dir: Optional[Path] = None
        self.pending_score = False  # Flag to run score after align

        self._setup_ui()
        self._load_templates()

    def _setup_ui(self):
        """Build the UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Header
        header = QLabel("Quick Grade")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        main_layout.addWidget(header)

        description = QLabel(
            "Complete grading workflow: Upload scans, answer key, and roster (optional), "
            "then align, score, and generate reports."
        )
        description.setWordWrap(True)
        main_layout.addWidget(description)

        # Action buttons at top
        btn_layout = QHBoxLayout()
        self.align_score_btn = QPushButton("1. Align && Score")
        self.align_score_btn.setMinimumHeight(40)
        self.align_score_btn.clicked.connect(self._run_align_and_score)

        self.report_btn = QPushButton("2. Generate Report")
        self.report_btn.setMinimumHeight(40)
        self.report_btn.clicked.connect(self._run_report)
        self.report_btn.setEnabled(False)

        btn_layout.addWidget(self.align_score_btn)
        btn_layout.addWidget(self.report_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        main_layout.addWidget(self.progress)

        # Status label
        self.status_label = QLabel("")
        main_layout.addWidget(self.status_label)

        # Splitter for inputs and log
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)

        # Top section: inputs in tabs
        input_tabs = QTabWidget()
        splitter.addWidget(input_tabs)

        # Tab 1: Required Inputs
        inputs_tab = self._create_inputs_tab()
        input_tabs.addTab(inputs_tab, "Required Inputs")

        # Tab 2: Options
        options_tab = self._create_options_tab()
        input_tabs.addTab(options_tab, "Options")

        # Tab 3: After Scoring
        after_tab = self._create_after_scoring_tab()
        input_tabs.addTab(after_tab, "After Scoring")

        # Bottom section: Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 10))
        log_layout.addWidget(self.log_text)

        # Log control buttons
        log_btn_layout = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_btn_layout.addWidget(clear_log_btn)
        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)

        splitter.addWidget(log_group)
        splitter.setSizes([400, 200])

        # Bottom: Download buttons (initially hidden)
        self.download_frame = QFrame()
        download_layout = QHBoxLayout(self.download_frame)

        self.open_csv_btn = QPushButton("Open results.csv")
        self.open_csv_btn.clicked.connect(lambda: self._open_file(self.results_csv_path))

        self.open_scored_btn = QPushButton("Open scored.pdf")
        self.open_scored_btn.clicked.connect(lambda: self._open_file(self.scored_pdf_path))

        self.open_aligned_btn = QPushButton("Open aligned.pdf")
        self.open_aligned_btn.clicked.connect(lambda: self._open_file(self.aligned_pdf_path))

        self.open_folder_btn = QPushButton("Open Output Folder")
        self.open_folder_btn.clicked.connect(lambda: self._open_file(self.work_dir))

        download_layout.addWidget(self.open_csv_btn)
        download_layout.addWidget(self.open_scored_btn)
        download_layout.addWidget(self.open_aligned_btn)
        download_layout.addWidget(self.open_folder_btn)
        download_layout.addStretch()

        self.download_frame.setVisible(False)
        main_layout.addWidget(self.download_frame)

    def _create_inputs_tab(self) -> QWidget:
        """Create the Required Inputs tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Template selection
        template_group = QGroupBox("Template")
        template_layout = QVBoxLayout(template_group)

        template_row = QHBoxLayout()
        template_row.addWidget(QLabel("Select template:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(300)
        template_row.addWidget(self.template_combo, 1)
        template_layout.addLayout(template_row)

        layout.addWidget(template_group)

        # File inputs
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)

        self.scans_selector = FileSelector(
            "Scanned answer sheets:",
            "PDF files (*.pdf)",
            "Select your scanned PDF..."
        )
        files_layout.addWidget(self.scans_selector)

        self.key_selector = FileSelector(
            "Answer key:",
            "Key files (*.txt *.csv *.tsv *.xlsx)",
            "Select answer key file..."
        )
        files_layout.addWidget(self.key_selector)

        self.roster_selector = FileSelector(
            "Class roster (optional):",
            "CSV files (*.csv)",
            "Optional: Select roster CSV..."
        )
        files_layout.addWidget(self.roster_selector)

        layout.addWidget(files_group)

        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self.output_selector = FileSelector(
            "Output directory:",
            "",
            "Select output folder..."
        )
        self.output_selector.browse_btn.clicked.disconnect()
        self.output_selector.browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(self.output_selector)

        layout.addWidget(output_group)

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

        self.verbose_thresh_cb = QCheckBox("Verbose threshold calibration")
        self.verbose_thresh_cb.setChecked(True)
        scoring_layout.addWidget(self.verbose_thresh_cb, 1, 1)

        # Min fill
        scoring_layout.addWidget(QLabel("Min fill score (0-100):"), 2, 0)
        self.min_fill_edit = QLineEdit()
        self.min_fill_edit.setPlaceholderText(str(_dflt(SCORING_DEFAULTS, "min_fill", 45)))
        scoring_layout.addWidget(self.min_fill_edit, 2, 1)

        # Top2 ratio
        scoring_layout.addWidget(QLabel("Top2 ratio (0-100):"), 3, 0)
        self.top2_ratio_edit = QLineEdit()
        self.top2_ratio_edit.setPlaceholderText(str(_dflt(SCORING_DEFAULTS, "top2_ratio", 80)))
        scoring_layout.addWidget(self.top2_ratio_edit, 3, 1)

        layout.addWidget(scoring_group)

        # Alignment options
        align_group = QGroupBox("Alignment Options")
        align_layout = QGridLayout(align_group)

        align_layout.addWidget(QLabel("Alignment method:"), 0, 0)
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

        # Flagging options
        flag_group = QGroupBox("Flagging && Review")
        flag_layout = QVBoxLayout(flag_group)

        self.review_pdf_cb = QCheckBox("Generate review PDF (flagged pages only)")
        self.review_pdf_cb.setChecked(True)
        flag_layout.addWidget(self.review_pdf_cb)

        self.flagged_xlsx_cb = QCheckBox("Generate flagged items XLSX")
        self.flagged_xlsx_cb.setChecked(True)
        flag_layout.addWidget(self.flagged_xlsx_cb)

        self.include_stats_cb = QCheckBox("Include basic statistics in CSV")
        self.include_stats_cb.setChecked(True)
        flag_layout.addWidget(self.include_stats_cb)

        layout.addWidget(flag_group)

        layout.addStretch()
        return widget

    def _create_after_scoring_tab(self) -> QWidget:
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
            "Optional: Select filled flagged_for_review.xlsx..."
        )
        corrections_layout.addWidget(self.corrections_selector)

        layout.addWidget(corrections_group)

        # Report options
        report_group = QGroupBox("Report Generation")
        report_layout = QVBoxLayout(report_group)

        report_layout.addWidget(QLabel(
            "Click 'Generate Report' after scoring to create the final Excel report."
        ))

        layout.addWidget(report_group)

        layout.addStretch()
        return widget

    def _browse_output_dir(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_selector.set_path(path)

    def _load_templates(self):
        """Load available templates into the combo box."""
        self.template_combo.clear()
        self.template_combo.addItem("(Select a template)", None)

        if TemplateManager is None:
            self.template_combo.addItem("(TemplateManager not available)", None)
            return

        try:
            tm = TemplateManager()
            templates = tm.scan_templates()
            for t in templates:
                self.template_combo.addItem(t.display_name, t)
        except Exception as e:
            self.template_combo.addItem(f"(Error loading templates: {e})", None)

    def _get_selected_template(self):
        """Get the currently selected template object."""
        return self.template_combo.currentData()

    def _validate_inputs(self) -> bool:
        """Validate required inputs before running."""
        if not self.scans_selector.path():
            QMessageBox.warning(self, "Missing Input", "Please select a scanned PDF file.")
            return False

        if not Path(self.scans_selector.path()).exists():
            QMessageBox.warning(self, "File Not Found", "Scanned PDF file does not exist.")
            return False

        template = self._get_selected_template()
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

        # Setup work directory
        output_dir = self.output_selector.path()
        if output_dir:
            self.work_dir = Path(output_dir)
        else:
            import tempfile
            self.work_dir = Path(tempfile.mkdtemp(prefix="quickgrade_"))
            self.output_selector.set_path(str(self.work_dir))

        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Get template paths
        template = self._get_selected_template()
        template_pdf = str(template.template_pdf_path)
        bubblemap = str(template.bubblemap_yaml_path)

        # Output paths
        self.aligned_pdf_path = self.work_dir / "aligned_scans.pdf"
        self.results_csv_path = self.work_dir / "results.csv"
        self.scored_pdf_path = self.work_dir / "scored.pdf"

        # Clear log and show progress
        self.log_text.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate
        self.align_score_btn.setEnabled(False)
        self.status_label.setText("Step 1/2: Aligning scans to template...")

        self._append_log("=" * 50 + "\n")
        self._append_log("STEP 1: ALIGNMENT\n")
        self._append_log("=" * 50 + "\n\n")

        # Build align command
        align_args = [
            "align",
            self.scans_selector.path(),
            "--template", template_pdf,
            "--out-pdf", str(self.aligned_pdf_path),
            "--dpi", str(self.dpi_spin.value()),
            "--align-method", self.align_method_combo.currentText(),
            "--min-markers", str(self.min_markers_spin.value()),
            "--bubblemap", bubblemap,
        ]

        self._append_log(f"Command: markshark {' '.join(align_args)}\n\n")

        # Store bubblemap for score step
        self._current_bubblemap = bubblemap
        self.pending_score = True

        # Run align
        self.runner.run(align_args, "align")

    def _run_score(self):
        """Run the score step after alignment."""
        template = self._get_selected_template()
        bubblemap = self._current_bubblemap

        self.status_label.setText("Step 2/2: Scoring aligned sheets...")
        self._append_log("\n" + "=" * 50 + "\n")
        self._append_log("STEP 2: SCORING\n")
        self._append_log("=" * 50 + "\n\n")

        # Build score command
        score_args = [
            "score",
            str(self.aligned_pdf_path),
            "--bublmap", bubblemap,
            "--out-csv", str(self.results_csv_path),
            "--out-pdf", "scored.pdf",
            "--dpi", str(self.dpi_spin.value()),
        ]

        # Add key if provided
        key_path = self.key_selector.path()
        if key_path and Path(key_path).exists():
            score_args += ["--key-txt", key_path]

        # Add options
        if self.annotate_all_cb.isChecked():
            score_args += ["--annotate-all-cells"]
        if self.label_density_cb.isChecked():
            score_args += ["--label-density"]
        if not self.auto_thresh_cb.isChecked():
            score_args += ["--no-auto-thresh"]
        if self.verbose_thresh_cb.isChecked():
            score_args += ["--verbose-thresh"]

        min_fill = self.min_fill_edit.text().strip()
        if min_fill:
            score_args += ["--min-fill", min_fill]

        top2_ratio = self.top2_ratio_edit.text().strip()
        if top2_ratio:
            score_args += ["--top2-ratio", top2_ratio]

        if self.include_stats_cb.isChecked():
            score_args += ["--include-stats"]
        else:
            score_args += ["--no-include-stats"]

        # Flagging options
        if self.review_pdf_cb.isChecked():
            score_args += ["--review-pdf", str(self.work_dir / "flagged_for_review.pdf")]
        if self.flagged_xlsx_cb.isChecked():
            score_args += ["--flagged-xlsx", str(self.work_dir / "flagged_for_review.xlsx")]

        # Roster
        roster_path = self.roster_selector.path()
        if roster_path and Path(roster_path).exists():
            score_args += ["--roster-csv", roster_path]

        self._append_log(f"Command: markshark {' '.join(score_args)}\n\n")

        self.runner.run(score_args, "score")

    def _run_report(self):
        """Generate the Excel report."""
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return

        if not self.results_csv_path or not self.results_csv_path.exists():
            QMessageBox.warning(self, "No Results",
                "Please run 'Align & Score' first before generating a report.")
            return

        self.status_label.setText("Generating Excel report...")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.report_btn.setEnabled(False)

        self._append_log("\n" + "=" * 50 + "\n")
        self._append_log("GENERATING REPORT\n")
        self._append_log("=" * 50 + "\n\n")

        report_path = self.work_dir / "exam_report.xlsx"

        report_args = [
            "report",
            str(self.results_csv_path),
            "--out-xlsx", str(report_path),
        ]

        # Add corrections if provided
        corrections_path = self.corrections_selector.path()
        if corrections_path and Path(corrections_path).exists():
            report_args += ["--corrections", corrections_path]
            self._append_log(f"Applying corrections from: {corrections_path}\n")

        self._append_log(f"Command: markshark {' '.join(report_args)}\n\n")

        self.runner.run(report_args, "report")

    def _on_step_finished(self, exit_code: int, step_name: str):
        """Handle completion of a CLI step."""
        self._append_log(f"\n[{step_name}] Process finished with exit code {exit_code}\n")

        if exit_code == 0:
            if step_name == "align" and self.pending_score:
                self.pending_score = False
                self._run_score()
                return

            elif step_name == "score":
                self.status_label.setText("Quick Grade complete!")
                self.download_frame.setVisible(True)
                self.report_btn.setEnabled(True)
                QMessageBox.information(self, "Complete",
                    "Alignment and scoring complete!\n\n"
                    "You can now:\n"
                    "- Open the output files using the buttons below\n"
                    "- Review flagged items and add corrections\n"
                    "- Generate the final Excel report"
                )

            elif step_name == "report":
                self.status_label.setText("Report generated!")
                report_path = self.work_dir / "exam_report.xlsx"
                if report_path.exists():
                    QMessageBox.information(self, "Report Ready",
                        f"Excel report generated:\n{report_path}")
                    self._open_file(report_path)
        else:
            self.status_label.setText(f"Error in {step_name} (exit code {exit_code})")
            self.pending_score = False
            QMessageBox.warning(self, "Error",
                f"The {step_name} step failed with exit code {exit_code}.\n"
                "Check the log for details.")

        self.progress.setVisible(False)
        self.align_score_btn.setEnabled(True)
        self.report_btn.setEnabled(self.results_csv_path and self.results_csv_path.exists())

    def _append_log(self, text: str):
        """Append text to the log."""
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()

    def _open_file(self, path: Optional[Path]):
        """Open a file or folder with the system default application."""
        if path is None or not path.exists():
            QMessageBox.warning(self, "Not Found", f"File or folder not found: {path}")
            return

        import subprocess
        import platform

        try:
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(path)])
            elif platform.system() == "Windows":
                os.startfile(str(path))
            else:  # Linux
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open file: {e}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MarkShark Quick Grade")

    window = QuickGradeWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
