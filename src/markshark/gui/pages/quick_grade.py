"""
Grader page - the main grading workflow.

Mirrors the Streamlit "Quick Grade" functionality:
1. Select template
2. Upload scans, key, roster
3. Configure options
4. Run align & score
5. Generate report
"""

import shutil
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
    from markshark.defaults import SCORING_DEFAULTS
    from markshark.template_manager import TemplateManager
    from markshark.project_utils import (
        get_project_paths,
        has_existing_results,
        archive_current_results,
    )
except ImportError:
    SCORING_DEFAULTS = None
    TemplateManager = None
    get_project_paths = None
    has_existing_results = None
    archive_current_results = None


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
        self.report_xlsx: Optional[Path] = None
        self._pending_score = False
        self._current_bubblemap: Optional[str] = None

        self._setup_ui()
        self._load_templates()
        self._restore_template_for_current_project()

        # Save template choice whenever the user changes it
        self.template_combo.currentIndexChanged.connect(self._save_template_choice)

        # When the project changes, clear stale paths and re-populate
        self.project_selector.project_changed.connect(self._on_project_changed)
        self.project_selector.working_dir_changed.connect(
            lambda _: self._update_browse_dirs()
        )
        # Seed browse dirs from whatever the ProjectSelector restored from settings
        self._update_browse_dirs()

    # ------------------------------------------------------------------
    # Re-scan for project files every time the page becomes visible
    # ------------------------------------------------------------------

    def showEvent(self, event):
        """Re-populate file selectors and template list when the page becomes visible.

        Files created by other pages (e.g. corrections from Review & Correct,
        results from a previous run) need to appear in the Report tab selectors.
        Templates are reloaded to reflect any reorder/add/remove in the Template Manager.
        """
        super().showEvent(event)
        self._reload_templates_preserving_selection()
        self._auto_populate_input_files()
        self._auto_populate_report_files()

    def _reload_templates_preserving_selection(self):
        """Reload template dropdown while keeping the current selection."""
        current = self.template_combo.currentData()
        current_id = getattr(current, "template_id", None) if current else None
        self._load_templates()
        if current_id:
            self._select_template_by_id(current_id)

    def _save_template_choice(self):
        """Persist the current template selection to the project registry."""
        t = self.template_combo.currentData()
        tid = getattr(t, "template_id", None) if t else None
        project_dir = self.project_selector.project_dir()
        if tid and project_dir:
            from ..models.project_registry import ProjectRegistry
            ProjectRegistry().set_template_id(project_dir, tid)

    def _restore_template_for_current_project(self):
        """Restore the saved template selection for the active project."""
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return
        from ..models.project_registry import ProjectRegistry
        tid = ProjectRegistry().get_template_id(project_dir)
        if tid:
            self._select_template_by_id(tid)

    def _select_template_by_id(self, template_id: str):
        """Set the template combo to the item matching *template_id*."""
        for i in range(self.template_combo.count()):
            t = self.template_combo.itemData(i)
            if t and getattr(t, "template_id", None) == template_id:
                self.template_combo.setCurrentIndex(i)
                return

    def _setup_ui(self):
        """Build the page UI."""
        layout = QVBoxLayout(self)

        # Header with icon
        header = PageHeader(
            "MarkShark Quick Grader",
            "Upload scans, answer key, and roster (optional), then align, score, and generate reports."
        )
        layout.addWidget(header)

        # Project/directory selector bar
        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

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

        # Tabs for inputs — minimum height prevents file selectors from squishing
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_inputs_tab(), "Align && Score")
        self.tabs.addTab(self._create_after_tab(), "Generate Report")
        self.tabs.addTab(self._create_options_tab(), "Grader Settings")
        self.tabs.setMinimumHeight(320)
        splitter.addWidget(self.tabs)

        # Log viewer + output buttons side-by-side
        log_row = QHBoxLayout()

        self.log = LogViewer("Processing Log")
        log_row.addWidget(self.log, 1)

        # Output buttons — always visible, stacked vertically, greyed out until results exist
        output_btn_panel = QVBoxLayout()
        output_btn_panel.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Outputs")
        output_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #6c757d;")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_btn_panel.addWidget(output_label)

        self.open_folder_btn = QPushButton("Open\n Project Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.setFixedWidth(100)
        self.open_folder_btn.clicked.connect(lambda: self._open_file(self.work_dir))
        output_btn_panel.addWidget(self.open_folder_btn)

        self.open_report_btn = QPushButton("Open\nReport")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.setFixedWidth(100)
        self.open_report_btn.clicked.connect(lambda: self._open_file(self.report_xlsx))
        output_btn_panel.addWidget(self.open_report_btn)

        self.open_scored_btn = QPushButton("Open\nScored Scans")
        self.open_scored_btn.setEnabled(False)
        self.open_scored_btn.setFixedWidth(100)
        self.open_scored_btn.clicked.connect(lambda: self._open_file(self.scored_pdf))
        output_btn_panel.addWidget(self.open_scored_btn)

        output_btn_panel.addStretch()
        log_row.addLayout(output_btn_panel)

        # Wrap log_row in a widget so QSplitter can manage it
        log_container = QWidget()
        log_container.setLayout(log_row)
        splitter.addWidget(log_container)

        splitter.setSizes([350, 200])

    def _create_inputs_tab(self) -> QWidget:
        """Create the Inputs tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Template selection
        layout.addSpacing(20)
        template_group = QGroupBox("Choose your template bubblesheet")
        template_layout = QHBoxLayout(template_group)
        template_layout.addWidget(QLabel("Select template:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(300)
        template_layout.addWidget(self.template_combo, 1)
        layout.addWidget(template_group)
        layout.addSpacing(20)

        # File inputs
        files_group = QGroupBox("Upload Scanned Bubblesheets, Key, and Class Roster Files")
        files_layout = QVBoxLayout(files_group)

        self.scans_selector = FileSelector(
            "Scanned answer sheets (PDF):",
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

        # Run button at the bottom of this tab
        self.align_score_btn = QPushButton("Run")
        self.align_score_btn.setMinimumHeight(36)
        from ..utils import RUN_BUTTON_STYLE
        self.align_score_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.align_score_btn.clicked.connect(self._run_align_and_score)
        layout.addWidget(self.align_score_btn)

        return widget

    def _create_options_tab(self) -> QWidget:
        """Create the Grader Settings tab — single box, two columns."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addSpacing(20)
        settings_group = QGroupBox("Grader Settings")
        grid = QGridLayout(settings_group)
        grid.setColumnStretch(0, 0)  # labels
        grid.setColumnStretch(1, 1)  # left controls
        grid.setColumnStretch(2, 0)  # labels
        grid.setColumnStretch(3, 1)  # right controls

        # ── Left column: Alignment + Scoring ──
        row = 0
        left_header = QLabel("Alignment")
        left_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(left_header, row, 0, 1, 2)

        row += 1
        grid.addWidget(QLabel("Align method:"), row, 0)
        self.align_method_combo = QComboBox()
        self.align_method_combo.addItems(["auto", "fast", "slow", "aruco"])
        grid.addWidget(self.align_method_combo, row, 1)

        row += 1
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        grid.addWidget(sep1, row, 0, 1, 2)

        row += 1
        left_header2 = QLabel("Scoring")
        left_header2.setStyleSheet("font-weight: bold;")
        grid.addWidget(left_header2, row, 0, 1, 2)

        row += 1
        grid.addWidget(QLabel("Min % filled:"), row, 0)
        self.min_fill_spin = QSpinBox()
        self.min_fill_spin.setRange(0, 100)
        self.min_fill_spin.setSuffix("%")
        self.min_fill_spin.setValue(_dflt(SCORING_DEFAULTS, "min_fill", 45))
        self.min_fill_spin.setToolTip(
            "Minimum fill score (0-100%) to consider a bubble filled.\n"
            "Increase to require darker marks; decrease to accept lighter marks."
        )
        grid.addWidget(self.min_fill_spin, row, 1)

        row += 1
        grid.addWidget(QLabel("Fixed threshold (gray):"), row, 0)
        self.fixed_thresh_spin = QSpinBox()
        self.fixed_thresh_spin.setRange(0, 255)
        self.fixed_thresh_spin.setValue(_dflt(SCORING_DEFAULTS, "fixed_thresh", 180))
        self.fixed_thresh_spin.setToolTip(
            "Global binarization threshold (0-255) for gray pixels.\n"
            "Pixels darker than this value are treated as ink."
        )
        grid.addWidget(self.fixed_thresh_spin, row, 1)

        # ── Right column: Annotation options ──
        row_r = 0
        right_header = QLabel("Annotation")
        right_header.setStyleSheet("font-weight: bold;")
        grid.addWidget(right_header, row_r, 2, 1, 2)

        row_r += 1
        self.annotate_all_cb = QCheckBox("Annotate all bubbles")
        self.annotate_all_cb.setChecked(True)
        grid.addWidget(self.annotate_all_cb, row_r, 2, 1, 2)

        row_r += 1
        self.label_density_cb = QCheckBox("Show % fill labels")
        self.label_density_cb.setChecked(True)
        grid.addWidget(self.label_density_cb, row_r, 2, 1, 2)

        row_r += 1
        self.auto_thresh_cb = QCheckBox("Auto-calibrate threshold")
        self.auto_thresh_cb.setChecked(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
        grid.addWidget(self.auto_thresh_cb, row_r, 2, 1, 2)

        row_r += 1
        self.verbose_thresh_cb = QCheckBox("Verbose threshold")
        self.verbose_thresh_cb.setChecked(True)
        grid.addWidget(self.verbose_thresh_cb, row_r, 2, 1, 2)

        layout.addWidget(settings_group)

        layout.addStretch()
        return widget

    def _create_after_tab(self) -> QWidget:
        """Create the Generate Report tab.

        All three file selectors (results, corrections, roster) live in a
        single group box.  Files auto-populate from the project's flat
        structure when a project is selected.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Single group box for all report input files
        layout.addSpacing(20)
        files_group = QGroupBox("Upload Files to Generate Report")
        files_layout = QVBoxLayout(files_group)

        self.results_selector = FileSelector(
            "Results CSV:",
            "CSV files (*.csv)",
            "Auto-populated from project score_data/...",
        )
        self.results_selector.file_selected.connect(self._on_results_loaded)
        files_layout.addWidget(self.results_selector)

        self.corrections_selector = FileSelector(
            "Corrections (optional):",
            "CSV files (*.csv);;Excel files (*.xlsx)",
            "Auto-populated from project score_data/...",
        )
        files_layout.addWidget(self.corrections_selector)

        self.report_roster_selector = FileSelector(
            "Class roster (optional):",
            "CSV files (*.csv)",
            "Auto-populated from project input_files/...",
        )
        files_layout.addWidget(self.report_roster_selector)

        layout.addWidget(files_group)
        layout.addSpacing(20)

        layout.addStretch()

        # Create Report button at the bottom of this tab
        self.report_btn = QPushButton("Create Report")
        self.report_btn.setMinimumHeight(36)
        self.report_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self.report_btn.setEnabled(False)
        self.report_btn.clicked.connect(self._run_report)
        layout.addWidget(self.report_btn)

        return widget

    def _auto_populate_input_files(self):
        """Auto-populate the Align & Score file selectors from input_files/.

        Looks for:
          - input_files/scans.pdf       → Scans selector
          - input_files/key.*           → Key selector
          - input_files/roster.*        → Roster selector  (Align & Score tab)
          - input_files/aligned_scans.pdf → (sets aligned_pdf path)
        Only sets a path if the file exists and the selector is currently empty.
        """
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return

        input_files = project_dir / "input_files"
        if not input_files.exists():
            return

        # Scans
        if not self.scans_selector.exists():
            scans_matches = sorted(input_files.glob("scans.*"))
            if scans_matches:
                self.scans_selector.set_path(str(scans_matches[0]))

        # Key
        if not self.key_selector.exists():
            key_matches = sorted(input_files.glob("key.*"))
            if key_matches:
                self.key_selector.set_path(str(key_matches[0]))

        # Roster (Align & Score tab)
        if not self.roster_selector.exists():
            roster_matches = sorted(input_files.glob("roster.*"))
            if roster_matches:
                self.roster_selector.set_path(str(roster_matches[0]))

    def _auto_populate_report_files(self, _name: str = ""):
        """Auto-populate report file selectors from the project's flat structure.

        Called when the project changes.  Fills in:
          - score_data/results.csv   → Results CSV
          - score_data/corrections.csv → Corrections
          - input_files/roster.*     → Roster
        Only sets a path if the file exists and the selector is currently empty
        (so we don't overwrite something the user manually chose).
        """
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return

        # Results
        results_csv = project_dir / "score_data" / "results.csv"
        if results_csv.exists() and not self.results_selector.exists():
            self.results_selector.set_path(str(results_csv))

        # Corrections
        corrections_csv = project_dir / "score_data" / "corrections.csv"
        if corrections_csv.exists() and not self.corrections_selector.exists():
            self.corrections_selector.set_path(str(corrections_csv))

        # Roster — glob for roster.* in input_files/
        input_files = project_dir / "input_files"
        if input_files.exists() and not self.report_roster_selector.exists():
            roster_matches = sorted(input_files.glob("roster.*"))
            if roster_matches:
                self.report_roster_selector.set_path(str(roster_matches[0]))

        # Check for existing report
        report_xlsx = project_dir / "exam_report.xlsx"
        if report_xlsx.exists():
            self.report_xlsx = report_xlsx

        # Enable report button if we have results
        if self.results_selector.exists():
            self.report_btn.setEnabled(True)

        self._enable_output_buttons()

    def _load_templates(self):
        """Load templates into the combo box."""
        self.template_combo.clear()
        self.template_combo.addItem("(Select a template)", None)

        if TemplateManager is None:
            self.template_combo.addItem("(TemplateManager not available)", None)
            return

        try:
            from ..utils import template_display_label
            tm = TemplateManager()
            for t in tm.scan_templates():
                self.template_combo.addItem(template_display_label(t, tm), t)
        except Exception as e:
            self.template_combo.addItem(f"(Error: {e})", None)

    def browse_scans(self):
        """Trigger the scans file browser (called from main window menu)."""
        self.scans_selector.trigger_browse()

    def _on_results_loaded(self, path_str: str):
        """Handle user loading a previous results CSV via the Generate Report tab."""
        csv_path = Path(path_str)
        if not csv_path.exists():
            return
        self.results_csv = csv_path

        # Determine work_dir: if CSV is in score_data/, use the project root
        if csv_path.parent.name == "score_data":
            self.work_dir = csv_path.parent.parent
        else:
            self.work_dir = csv_path.parent

        # Look for scored PDF — check project root first, then next to CSV
        scored_candidates = [
            self.work_dir / "scored_scans.pdf",
            csv_path.parent / "scored_scans.pdf",
        ]
        for p in scored_candidates:
            if p.exists():
                self.scored_pdf = p
                break

        self.report_btn.setEnabled(True)
        self._enable_output_buttons()
        self.status_label.setText(f"Loaded results: {csv_path.name}")
        self.log.append_line(f"Loaded previous results: {csv_path}")

    def _on_project_changed(self, _name: str = ""):
        """Handle project change: clear all file selectors and re-populate.

        When switching projects (or creating a new one), stale file paths
        from the previous project must not linger.  We clear every selector
        first, then auto-populate from the new project's flat structure.
        """
        # Clear all file selectors
        self.scans_selector.clear()
        self.key_selector.clear()
        self.roster_selector.clear()
        self.results_selector.clear()
        self.corrections_selector.clear()
        self.report_roster_selector.clear()

        # Reset state
        self.results_csv = None
        self.scored_pdf = None
        self.aligned_pdf = None
        self.report_xlsx = None
        self.report_btn.setEnabled(False)
        self._enable_output_buttons()

        # Update browse dirs and auto-populate from new project
        self._update_browse_dirs()

        # Restore saved template for the newly selected project
        self._restore_template_for_current_project()

    def _update_browse_dirs(self, _name: str = ""):
        """Point all file-browse dialogs at the current project folder,
        and auto-populate the report selectors from the flat structure."""
        project_dir = self.project_selector.project_dir()
        working_dir = self.project_selector.working_dir()
        start = str(project_dir) if project_dir else (str(working_dir) if working_dir else "")
        if not start:
            return
        self.scans_selector.set_start_dir(start)
        self.key_selector.set_start_dir(start)
        self.roster_selector.set_start_dir(start)
        self.results_selector.set_start_dir(start)
        self.corrections_selector.set_start_dir(start)
        self.report_roster_selector.set_start_dir(start)

        # Auto-populate file selectors from the project's flat structure
        self._auto_populate_input_files()
        self._auto_populate_report_files()

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

    def _copy_inputs_to_project(self):
        """
        Copy input files (scans, key, roster) to the project's input_files/ folder.

        Files are renamed to standard names so the project is self-contained:
            input_files/scans.pdf
            input_files/key.csv  (or key.txt, key.xlsx — preserves extension)
            input_files/roster.csv
        Existing files are overwritten (the user is re-running with new inputs).
        """
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return

        input_dir = project_dir / "input_files"
        input_dir.mkdir(exist_ok=True)

        # Copy scans
        if self.scans_selector.exists():
            src = Path(self.scans_selector.path())
            dst = input_dir / f"scans{src.suffix}"
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"[warn] Could not copy scans to input_files/: {e}")

        # Copy key
        if self.key_selector.exists():
            src = Path(self.key_selector.path())
            dst = input_dir / f"key{src.suffix}"
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"[warn] Could not copy key to input_files/: {e}")

        # Copy roster
        if self.roster_selector.exists():
            src = Path(self.roster_selector.path())
            dst = input_dir / f"roster{src.suffix}"
            try:
                shutil.copy2(str(src), str(dst))
            except Exception as e:
                print(f"[warn] Could not copy roster to input_files/: {e}")

    def _run_align_and_score(self):
        """Run the align and score workflow."""
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return

        if not self._validate_inputs():
            return

        # Setup work directory from project selector (now returns project root)
        self.work_dir = self.project_selector.output_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Archive-or-overwrite dialog if results already exist
        project_dir = self.project_selector.project_dir()
        if project_dir and has_existing_results and has_existing_results(project_dir):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Existing Results Found")
            msg.setText(
                "This project already has scoring results.\n"
                "Would you like to archive them before re-running?"
            )
            archive_btn = msg.addButton("Archive && Continue", QMessageBox.ButtonRole.AcceptRole)
            overwrite_btn = msg.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(archive_btn)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == cancel_btn:
                return
            elif clicked == archive_btn:
                archive_path = archive_current_results(project_dir)
                self.log.append_line(f"Archived previous results to: {archive_path}\n")

        # Copy input files to project input_files/ folder for archival
        self._copy_inputs_to_project()

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

        # Output paths — flat structure
        self.aligned_pdf = self.work_dir / "input_files" / "aligned_scans.pdf"
        self.results_csv = self.work_dir / "score_data" / "results.csv"
        self.scored_pdf = self.work_dir / "scored_scans.pdf"

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
            "--align-method", self.align_method_combo.currentText(),
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
            "--out-pdf", str(self.scored_pdf),
        ]

        # Key file
        if self.key_selector.exists():
            args += ["--key-txt", self.key_selector.path()]

        # Scoring parameters
        args += ["--min-fill", str(self.min_fill_spin.value())]
        args += ["--fixed-thresh", str(self.fixed_thresh_spin.value())]

        # Options
        if self.annotate_all_cb.isChecked():
            args += ["--annotate-all-cells"]
        if self.label_density_cb.isChecked():
            args += ["--label-density"]
        if not self.auto_thresh_cb.isChecked():
            args += ["--no-auto-thresh"]
        if self.verbose_thresh_cb.isChecked():
            args += ["--verbose-thresh"]

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

        report_path = self.work_dir / "exam_report.xlsx"  # top-level
        args = [
            "report",
            str(self.results_csv),
            "--out-xlsx", str(report_path),
        ]

        # Roster — prefer the one on this tab, fall back to the Align & Score tab
        if self.report_roster_selector.exists():
            args += ["--roster", self.report_roster_selector.path()]
        elif self.roster_selector.exists():
            args += ["--roster", self.roster_selector.path()]

        # Corrections
        if self.corrections_selector.exists():
            args += ["--corrections", self.corrections_selector.path()]
            self.log.append_line(f"Applying corrections from: {self.corrections_selector.path()}")

        # Optional project name
        project_name = self.project_selector.project_name()
        if project_name:
            args += ["--project-name", project_name]

        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "report")

    def _on_output(self, text: str):
        """Handle CLI output."""
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
        log_path = logs_dir / f"log_{timestamp}.txt"
        if self.log.save_to_file(log_path):
            self.log.append_line(f"\nLog saved to: {log_path}")

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
                self._enable_output_buttons()
                self.report_btn.setEnabled(True)
                # Force-populate report tab with the results we just created
                if self.results_csv and self.results_csv.exists():
                    self.results_selector.set_path(str(self.results_csv))
                self._auto_populate_report_files()
                self._save_log()
                self.grading_complete.emit({
                    "work_dir": str(self.work_dir),
                    "results_csv": str(self.results_csv),
                    "scored_pdf": str(self.scored_pdf),
                    "aligned_pdf": str(self.aligned_pdf),
                })
                project_label = self.project_selector.project_name() or "Quick Grade"
                self._show_scoring_complete_dialog(project_label)

            elif step_name == "report":
                self.status_label.setText("Report generated!")
                self._save_log()
                self.report_xlsx = self.work_dir / "exam_report.xlsx"
                self._enable_output_buttons()
                if self.report_xlsx.exists():
                    QMessageBox.information(self, "Report Ready", f"Report saved to:\n{self.report_xlsx}")
                    self._open_file(self.report_xlsx)
        else:
            self.status_label.setText(f"Error in {step_name}")
            self._pending_score = False
            self._save_log()
            QMessageBox.warning(self, "Error", f"{step_name} failed. Check the log.")

        self.progress.setVisible(False)
        self.align_score_btn.setEnabled(True)
        self.report_btn.setEnabled(self.results_csv and self.results_csv.exists())

    def _show_scoring_complete_dialog(self, project_label: str):
        """Show a dialog after scoring completes with navigation options."""
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Scoring Complete")
        dlg.setText(f"Scoring complete!  ({project_label})")
        dlg.setIcon(QMessageBox.Icon.Information)

        # Add custom buttons
        review_btn = dlg.addButton("Review && Correct", QMessageBox.ButtonRole.ActionRole)
        report_btn = dlg.addButton("Generate Report", QMessageBox.ButtonRole.ActionRole)
        ok_btn = dlg.addButton(QMessageBox.StandardButton.Ok)
        dlg.setDefaultButton(ok_btn)

        dlg.exec()

        clicked = dlg.clickedButton()
        if clicked is review_btn:
            # Navigate to Review & Correct via the MainWindow
            main_win = self.window()
            if hasattr(main_win, "navigate_to_review"):
                main_win.navigate_to_review({
                    "work_dir": str(self.work_dir),
                    "results_csv": str(self.results_csv),
                    "scored_pdf": str(self.scored_pdf),
                    "aligned_pdf": str(self.aligned_pdf),
                })
        elif clicked is report_btn:
            # Switch to the Generate Report tab
            self.tabs.setCurrentIndex(1)

    def _enable_output_buttons(self):
        """Enable the output buttons based on which files actually exist."""
        self.open_report_btn.setEnabled(
            self.report_xlsx is not None and self.report_xlsx.exists()
        )
        self.open_scored_btn.setEnabled(
            self.scored_pdf is not None and self.scored_pdf.exists()
        )
        self.open_folder_btn.setEnabled(
            self.work_dir is not None and self.work_dir.exists()
        )

    def _open_file(self, path: Optional[Path]):
        """Open a file with the system default application."""
        if path is None or not path.exists():
            QMessageBox.warning(self, "Not Found", f"File not found: {path}")
            return
        from ..utils import open_file_or_folder
        try:
            open_file_or_folder(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open: {e}")
