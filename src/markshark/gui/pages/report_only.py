"""
Report Only page - generates Excel reports from scored CSV files.

Allows generating reports from existing scored CSV files without
re-running alignment or scoring.
"""

from datetime import datetime
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
    QLineEdit,
    QGroupBox,
    QProgressBar,
    QSplitter,
    QScrollArea,
    QMessageBox,
    QFrame,
)

from ..widgets import FileSelector, LogViewer, ProjectSelector, PageHeader
from ..workers import CLIRunner


class ReportOnlyPage(QWidget):
    """
    Report Only workflow page.

    Generates Excel reports from an existing results.csv produced by
    existing scored results. Supports optional roster and corrections files.
    """

    report_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.runner = CLIRunner(self)
        self.runner.output_received.connect(self._on_output)
        self.runner.error_received.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)

        self.work_dir: Optional[Path] = None
        self.report_path: Optional[Path] = None

        self._setup_ui()

        self.project_selector.project_changed.connect(self._update_browse_dirs)
        self.project_selector.working_dir_changed.connect(
            lambda _: self._update_browse_dirs()
        )
        self._update_browse_dirs()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = PageHeader(
            "MarkShark - Report Only",
            "Generate an Excel report from an existing results CSV. "
            "Useful for re-generating reports with corrections or a new roster."
        )
        layout.addWidget(header)

        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Generate Report")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self.run_btn.clicked.connect(self._run_report)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Splitter: top = scrollable inputs, bottom = log + output buttons
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        # Scrollable input area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        scroll.setWidget(input_container)

        # --- Input Files ---
        files_group = QGroupBox("Input Files")
        files_layout = QVBoxLayout(files_group)

        results_row = QHBoxLayout()
        self.results_selector = FileSelector(
            "Results CSV:",
            "CSV files (*.csv)",
            "Select results.csv from score_data/...",
        )
        results_row.addWidget(self.results_selector, 1)

        load_recent_btn = QPushButton("Load Recent")
        load_recent_btn.setMaximumWidth(90)
        load_recent_btn.setToolTip(
            "Load results.csv from this project's score_data/ folder."
        )
        load_recent_btn.clicked.connect(self._load_most_recent_results)
        results_row.addWidget(load_recent_btn)
        files_layout.addLayout(results_row)

        self.roster_selector = FileSelector(
            "Class roster (optional):",
            "CSV files (*.csv)",
            "Optional roster CSV (StudentID, LastName, FirstName)...",
        )
        files_layout.addWidget(self.roster_selector)

        corrections_row = QHBoxLayout()
        self.corrections_selector = FileSelector(
            "Corrections (optional):",
            "CSV files (*.csv);;Excel files (*.xlsx)",
            "Optional corrections file...",
        )
        corrections_row.addWidget(self.corrections_selector, 1)

        load_recent_corrections_btn = QPushButton("Load Recent")
        load_recent_corrections_btn.setMaximumWidth(90)
        load_recent_corrections_btn.setToolTip(
            "Load corrections.csv from this project's score_data/ folder."
        )
        load_recent_corrections_btn.clicked.connect(self._load_most_recent_corrections)
        corrections_row.addWidget(load_recent_corrections_btn)
        files_layout.addLayout(corrections_row)

        input_layout.addWidget(files_group)
        input_layout.addSpacing(8)

        # --- Report Options ---
        options_group = QGroupBox("Report Options")
        options_layout = QGridLayout(options_group)

        options_layout.addWidget(QLabel("Project name (optional):"), 0, 0)
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("Included in report header...")
        self.project_name_edit.setToolTip("Optional label included at the top of the Excel report.")
        options_layout.addWidget(self.project_name_edit, 0, 1)

        options_layout.addWidget(QLabel("Run label (optional):"), 1, 0)
        self.run_label_edit = QLineEdit()
        self.run_label_edit.setPlaceholderText("e.g. 2025-01-21_final")
        self.run_label_edit.setToolTip("Optional run identifier included in report header.")
        options_layout.addWidget(self.run_label_edit, 1, 1)

        input_layout.addWidget(options_group)

        input_layout.addStretch()
        splitter.addWidget(scroll)

        # Log viewer + output buttons side-by-side
        log_row = QHBoxLayout()

        self.log = LogViewer("Report Log")
        log_row.addWidget(self.log, 1)

        # Output buttons — always visible, stacked, greyed out until report exists
        output_btn_panel = QVBoxLayout()
        output_btn_panel.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Outputs")
        output_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #6c757d;")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_btn_panel.addWidget(output_label)

        self.open_report_btn = QPushButton("Open\nReport")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.setFixedWidth(100)
        self.open_report_btn.clicked.connect(lambda: self._open_file(self.report_path))
        output_btn_panel.addWidget(self.open_report_btn)

        self.open_folder_btn = QPushButton("Open\nFolder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.setFixedWidth(100)
        self.open_folder_btn.clicked.connect(lambda: self._open_file(self.work_dir))
        output_btn_panel.addWidget(self.open_folder_btn)

        output_btn_panel.addStretch()
        log_row.addLayout(output_btn_panel)

        log_container = QWidget()
        log_container.setLayout(log_row)
        splitter.addWidget(log_container)

        splitter.setSizes([300, 200])

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _update_browse_dirs(self, _name: str = ""):
        project_dir = self.project_selector.project_dir()
        working_dir = self.project_selector.working_dir()
        start = str(project_dir) if project_dir else (str(working_dir) if working_dir else "")
        if not start:
            return
        self.results_selector.set_start_dir(start)
        self.roster_selector.set_start_dir(start)
        self.corrections_selector.set_start_dir(start)

    def _load_most_recent_results(self):
        """Find and load the results.csv from the project's score_data/ folder."""
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            QMessageBox.warning(
                self, "No Project",
                "Please select a project first."
            )
            return

        results_csv = project_dir / "score_data" / "results.csv"
        if not results_csv.exists():
            QMessageBox.information(
                self, "No Results Found",
                "No results.csv found in this project's score_data/ folder."
            )
            return

        self.results_selector.set_path(str(results_csv))
        self.log.append_line(f"Loaded results: {results_csv}")

    def _load_most_recent_corrections(self):
        """Find and load the corrections.csv from the project's score_data/ folder."""
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            QMessageBox.warning(
                self, "No Project",
                "Please select a project first."
            )
            return

        corrections_csv = project_dir / "score_data" / "corrections.csv"
        if not corrections_csv.exists():
            QMessageBox.information(
                self, "No Corrections Found",
                "No corrections.csv found in this project's score_data/ folder."
            )
            return

        self.corrections_selector.set_path(str(corrections_csv))
        self.log.append_line(f"Loaded corrections: {corrections_csv}")

    def _validate_inputs(self) -> bool:
        if not self.results_selector.exists():
            QMessageBox.warning(self, "Missing Input", "Please select a results CSV file.")
            return False
        return True

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    def _run_report(self):
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return
        if not self._validate_inputs():
            return

        self.work_dir = self.project_selector.output_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.report_path = self.work_dir / "exam_report.xlsx"  # top-level

        self.log.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self.status_label.setText("Generating report...")

        self.log.append_header("GENERATING REPORT")

        args = [
            "report",
            self.results_selector.path(),
            "--out-xlsx", str(self.report_path),
        ]

        # Optional roster
        if self.roster_selector.exists():
            args += ["--roster", self.roster_selector.path()]

        # Optional corrections
        if self.corrections_selector.exists():
            args += ["--corrections", self.corrections_selector.path()]

        # Optional metadata
        pname = self.project_name_edit.text().strip()
        if pname:
            args += ["--project-name", pname]
        elif self.project_selector.project_name():
            args += ["--project-name", self.project_selector.project_name()]

        rlabel = self.run_label_edit.text().strip()
        if rlabel:
            args += ["--run-label", rlabel]

        project_name = self.project_selector.project_name()
        if project_name:
            self.log.append_line(f"Project: {project_name}")
        self.log.append_line(f"Output directory: {self.work_dir}")
        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "report")

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------
    def _on_output(self, text: str):
        self.log.append(text)

    def _save_log(self):
        """Save the current log to the project's logs/ directory."""
        if not self.work_dir:
            return
        project_dir = self.project_selector.project_dir()
        if project_dir:
            logs_dir = project_dir / "logs"
        else:
            logs_dir = self.work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_path = logs_dir / f"report_log_{timestamp}.txt"
        if self.log.save_to_file(log_path):
            self.log.append_line(f"\nLog saved to: {log_path}")

    def _on_finished(self, exit_code: int, step_name: str):
        self.log.append_line(f"\n[{step_name}] Finished with exit code {exit_code}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._save_log()

        if exit_code == 0:
            self.status_label.setText("Report generated!")
            self._enable_output_buttons()
            self.report_complete.emit({
                "work_dir": str(self.work_dir),
                "report_path": str(self.report_path),
            })
            QMessageBox.information(
                self, "Report Ready", f"Report saved to:\n{self.report_path}"
            )
        else:
            self.status_label.setText(f"Error in {step_name}")
            QMessageBox.warning(self, "Error", f"{step_name} failed. Check the log.")

    def _enable_output_buttons(self):
        """Enable output buttons based on which files actually exist."""
        self.open_report_btn.setEnabled(
            self.report_path is not None and self.report_path.exists()
        )
        self.open_folder_btn.setEnabled(
            self.work_dir is not None and self.work_dir.exists()
        )

    def _open_file(self, path: Optional[Path]):
        if path is None or not path.exists():
            QMessageBox.warning(self, "Not Found", f"File not found: {path}")
            return
        import subprocess, platform, os
        try:
            if platform.system() == "Darwin":
                subprocess.run(["open", str(path)])
            elif platform.system() == "Windows":
                os.startfile(str(path))
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open: {e}")
