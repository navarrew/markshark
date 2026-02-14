"""
Score Only page - runs scoring on already-aligned PDFs.

Exposes the full set of scoring parameters so users
can re-score with different thresholds without re-aligning.
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
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QProgressBar,
    QSplitter,
    QMessageBox,
    QFrame,
    QScrollArea,
)

from ..widgets import FileSelector, LogViewer, ProjectSelector, PageHeader
from ..workers import CLIRunner

# Best-effort import of defaults
try:
    from markshark.defaults import (
        SCORING_DEFAULTS,
        RENDER_DEFAULTS,
    )
    from markshark.template_manager import TemplateManager
except ImportError:
    SCORING_DEFAULTS = RENDER_DEFAULTS = None
    TemplateManager = None


def _dflt(obj, attr: str, fallback):
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)


class ScoreOnlyPage(QWidget):
    """
    Score Only workflow page.

    Scores already-aligned PDFs using a bubblemap, without running alignment.
    Exposes detailed scoring parameters beyond what QuickGrade offers.
    """

    scoring_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.runner = CLIRunner(self)
        self.runner.output_received.connect(self._on_output)
        self.runner.error_received.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)

        self.work_dir: Optional[Path] = None
        self.results_csv: Optional[Path] = None
        self.scored_pdf: Optional[Path] = None

        self._setup_ui()
        self._load_templates()
        self._restore_template_for_current_project()

        self.template_combo.currentIndexChanged.connect(self._save_template_choice)
        self.project_selector.project_changed.connect(self._on_project_changed)
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
            "MarkShark - Score Only",
            "Score already-aligned answer sheets without re-running alignment. "
            "Useful for re-scoring with different thresholds."
        )
        layout.addWidget(header)

        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Scoring")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #0d6efd; color: white; "
            "font-weight: bold; font-size: 14px; border-radius: 4px; padding: 6px 20px; }"
            "QPushButton:hover { background-color: #0b5ed7; }"
            "QPushButton:disabled { background-color: #6c757d; }"
        )
        self.run_btn.clicked.connect(self._run_score)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Splitter: top = scroll with inputs/options, bottom = log
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll.setWidget(scroll_container)

        # --- Input Files ---
        files_group = QGroupBox("Input Files")
        files_layout = QVBoxLayout(files_group)

        # Template (for bubblemap)
        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(300)
        tmpl_row.addWidget(self.template_combo, 1)
        files_layout.addLayout(tmpl_row)

        self.aligned_selector = FileSelector(
            "Aligned PDF:",
            "PDF files (*.pdf)",
            "Select already-aligned scans PDF...",
        )
        files_layout.addWidget(self.aligned_selector)

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

        scroll_layout.addWidget(files_group)
        scroll_layout.addSpacing(8)

        # --- General Settings ---
        gen_group = QGroupBox("General")
        gen_layout = QGridLayout(gen_group)

        gen_layout.addWidget(QLabel("Render DPI:"), 0, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(_dflt(RENDER_DEFAULTS, "dpi", 150))
        self.dpi_spin.setToolTip("DPI for rendering aligned PDF pages.")
        gen_layout.addWidget(self.dpi_spin, 0, 1)

        scroll_layout.addWidget(gen_group)
        scroll_layout.addSpacing(8)

        # --- Scoring Thresholds ---
        thresh_group = QGroupBox("Scoring Thresholds")
        thresh_layout = QGridLayout(thresh_group)

        thresh_layout.addWidget(QLabel("Min fill (%):"), 0, 0)
        self.min_fill_spin = QSpinBox()
        self.min_fill_spin.setRange(0, 100)
        self.min_fill_spin.setSuffix("%")
        self.min_fill_spin.setValue(_dflt(SCORING_DEFAULTS, "min_fill", 45))
        self.min_fill_spin.setToolTip(
            "Minimum fill score (0-100) for a bubble to count as filled.\n"
            "Increase to require darker marks; decrease to accept lighter marks."
        )
        thresh_layout.addWidget(self.min_fill_spin, 0, 1)

        thresh_layout.addWidget(QLabel("Fixed threshold (gray):"), 1, 0)
        self.fixed_thresh_spin = QSpinBox()
        self.fixed_thresh_spin.setRange(0, 255)
        self.fixed_thresh_spin.setValue(_dflt(SCORING_DEFAULTS, "fixed_thresh", 180))
        self.fixed_thresh_spin.setToolTip("Global binarization threshold for gray pixels.")
        thresh_layout.addWidget(self.fixed_thresh_spin, 1, 1)

        scroll_layout.addWidget(thresh_group)
        scroll_layout.addSpacing(8)

        # --- Calibration & Adaptive ---
        cal_group = QGroupBox("Calibration & Adaptive Scoring")
        cal_layout = QGridLayout(cal_group)

        self.auto_thresh_cb = QCheckBox("Auto-calibrate threshold")
        self.auto_thresh_cb.setChecked(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
        self.auto_thresh_cb.setToolTip("Automatically tune the fill threshold per page.")
        cal_layout.addWidget(self.auto_thresh_cb, 0, 0)

        self.verbose_thresh_cb = QCheckBox("Verbose threshold diagnostics")
        self.verbose_thresh_cb.setChecked(False)
        self.verbose_thresh_cb.setToolTip("Print per-page threshold calibration info to the log.")
        cal_layout.addWidget(self.verbose_thresh_cb, 0, 1)

        self.calibrate_bg_cb = QCheckBox("Calibrate background")
        self.calibrate_bg_cb.setChecked(_dflt(SCORING_DEFAULTS, "calibrate_background", True))
        self.calibrate_bg_cb.setToolTip("Subtract per-column background to remove letter printing bias.")
        cal_layout.addWidget(self.calibrate_bg_cb, 1, 0)

        cal_layout.addWidget(QLabel("Background percentile:"), 2, 0)
        self.bg_percentile_spin = QDoubleSpinBox()
        self.bg_percentile_spin.setRange(0, 100)
        self.bg_percentile_spin.setValue(_dflt(SCORING_DEFAULTS, "background_percentile", 10.0))
        self.bg_percentile_spin.setToolTip("Percentile for background calculation (10th = robust to noise).")
        cal_layout.addWidget(self.bg_percentile_spin, 2, 1)

        self.adaptive_cb = QCheckBox("Adaptive rescoring")
        self.adaptive_cb.setChecked(_dflt(SCORING_DEFAULTS, "adaptive_rescoring", True))
        self.adaptive_cb.setToolTip("Re-score blank rows with progressively lower thresholds.")
        cal_layout.addWidget(self.adaptive_cb, 3, 0)

        cal_layout.addWidget(QLabel("Adaptive max adjustment:"), 4, 0)
        self.adaptive_max_spin = QSpinBox()
        self.adaptive_max_spin.setRange(0, 100)
        self.adaptive_max_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_max_adjustment", 40))
        self.adaptive_max_spin.setToolTip("Maximum threshold reduction to try (in steps of 10).")
        cal_layout.addWidget(self.adaptive_max_spin, 4, 1)

        cal_layout.addWidget(QLabel("Adaptive min above floor:"), 5, 0)
        self.adaptive_floor_spin = QDoubleSpinBox()
        self.adaptive_floor_spin.setRange(0, 100)
        self.adaptive_floor_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_min_above_floor", 30))
        self.adaptive_floor_spin.setToolTip("Winner must be this many points above lowest bubble.")
        cal_layout.addWidget(self.adaptive_floor_spin, 5, 1)

        scroll_layout.addWidget(cal_group)
        scroll_layout.addSpacing(8)

        # --- Annotation Options ---
        annot_group = QGroupBox("Annotation Options")
        annot_layout = QGridLayout(annot_group)

        self.annotate_all_cb = QCheckBox("Annotate all bubbles")
        self.annotate_all_cb.setChecked(True)
        self.annotate_all_cb.setToolTip("Draw circles on every bubble, not just the filled ones.")
        annot_layout.addWidget(self.annotate_all_cb, 0, 0)

        self.label_density_cb = QCheckBox("Show % fill labels")
        self.label_density_cb.setChecked(True)
        self.label_density_cb.setToolTip("Overlay percentage fill text at each bubble.")
        annot_layout.addWidget(self.label_density_cb, 0, 1)

        scroll_layout.addWidget(annot_group)

        scroll_layout.addStretch()
        splitter.addWidget(scroll)

        # Log viewer + output buttons side-by-side
        log_row = QHBoxLayout()

        self.log = LogViewer("Scoring Log")
        log_row.addWidget(self.log, 1)

        # Output buttons — always visible, stacked, greyed out until results exist
        output_btn_panel = QVBoxLayout()
        output_btn_panel.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Outputs")
        output_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #6c757d;")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_btn_panel.addWidget(output_label)

        self.open_csv_btn = QPushButton("Open\nresults.csv")
        self.open_csv_btn.setEnabled(False)
        self.open_csv_btn.setFixedWidth(100)
        self.open_csv_btn.clicked.connect(lambda: self._open_file(self.results_csv))
        output_btn_panel.addWidget(self.open_csv_btn)

        self.open_pdf_btn = QPushButton("Open\nScored Scans")
        self.open_pdf_btn.setEnabled(False)
        self.open_pdf_btn.setFixedWidth(100)
        self.open_pdf_btn.clicked.connect(lambda: self._open_file(self.scored_pdf))
        output_btn_panel.addWidget(self.open_pdf_btn)

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

        splitter.setSizes([400, 200])

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _load_templates(self):
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

    def showEvent(self, event):
        """Re-populate file selectors and template list when the page becomes visible."""
        super().showEvent(event)
        self._reload_templates_preserving_selection()
        self._auto_populate_from_project()

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

    def _on_project_changed(self, _name: str = ""):
        """Handle project change: update browse dirs and restore template."""
        self._update_browse_dirs()
        self._restore_template_for_current_project()

    def _update_browse_dirs(self, _name: str = ""):
        project_dir = self.project_selector.project_dir()
        working_dir = self.project_selector.working_dir()
        start = str(project_dir) if project_dir else (str(working_dir) if working_dir else "")
        if not start:
            return
        self.aligned_selector.set_start_dir(start)
        self.key_selector.set_start_dir(start)
        self.roster_selector.set_start_dir(start)
        self._auto_populate_from_project()

    def _auto_populate_from_project(self):
        """Auto-fill selectors from project's flat structure if files exist."""
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return
        input_files = project_dir / "input_files"
        if not input_files.exists():
            return

        # Aligned PDF
        aligned = input_files / "aligned_scans.pdf"
        if aligned.exists() and not self.aligned_selector.exists():
            self.aligned_selector.set_path(str(aligned))

        # Key
        if not self.key_selector.exists():
            for k in sorted(input_files.glob("key.*")):
                self.key_selector.set_path(str(k))
                break

        # Roster
        if not self.roster_selector.exists():
            for r in sorted(input_files.glob("roster.*")):
                self.roster_selector.set_path(str(r))
                break

    def _validate_inputs(self) -> bool:
        if not self.aligned_selector.exists():
            QMessageBox.warning(self, "Missing Input", "Please select an aligned PDF file.")
            return False
        template = self.template_combo.currentData()
        if template is None:
            QMessageBox.warning(self, "Missing Template", "Please select a template (for its bubblemap).")
            return False
        return True

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    def _run_score(self):
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return
        if not self._validate_inputs():
            return

        self.work_dir = self.project_selector.output_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        template = self.template_combo.currentData()
        bubblemap = str(template.bubblemap_yaml_path)

        # Flat structure: results in score_data/, scored PDF at root
        score_data = self.work_dir / "score_data"
        score_data.mkdir(exist_ok=True)
        self.results_csv = score_data / "results.csv"
        self.scored_pdf = self.work_dir / "scored_scans.pdf"

        self.log.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self.status_label.setText("Scoring sheets...")

        self.log.append_header("SCORING")

        args = [
            "score",
            self.aligned_selector.path(),
            "--bublmap", bubblemap,
            "--out-csv", str(self.results_csv),
            "--out-pdf", str(self.scored_pdf),
            "--dpi", str(self.dpi_spin.value()),
            "--min-fill", str(self.min_fill_spin.value()),
            "--fixed-thresh", str(self.fixed_thresh_spin.value()),
        ]

        # Boolean flags
        if self.auto_thresh_cb.isChecked():
            args += ["--auto-thresh"]
        else:
            args += ["--no-auto-thresh"]

        if self.verbose_thresh_cb.isChecked():
            args += ["--verbose-thresh"]

        if self.annotate_all_cb.isChecked():
            args += ["--annotate-all-cells"]
        if self.label_density_cb.isChecked():
            args += ["--label-density"]

        # Key file
        if self.key_selector.exists():
            args += ["--key-txt", self.key_selector.path()]

        # Roster
        if self.roster_selector.exists():
            args += ["--roster-csv", self.roster_selector.path()]

        project_name = self.project_selector.project_name()
        if project_name:
            self.log.append_line(f"Project: {project_name}")
        self.log.append_line(f"Output directory: {self.work_dir}")
        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "score")

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
        log_path = logs_dir / f"score_log_{timestamp}.txt"
        if self.log.save_to_file(log_path):
            self.log.append_line(f"\nLog saved to: {log_path}")

    def _on_finished(self, exit_code: int, step_name: str):
        self.log.append_line(f"\n[{step_name}] Finished with exit code {exit_code}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._save_log()

        if exit_code == 0:
            self.status_label.setText("Scoring complete!")
            self._enable_output_buttons()
            self.scoring_complete.emit({
                "work_dir": str(self.work_dir),
                "results_csv": str(self.results_csv),
                "scored_pdf": str(self.scored_pdf),
            })
            QMessageBox.information(self, "Complete", "Scoring complete!")
        else:
            self.status_label.setText(f"Error in {step_name}")
            QMessageBox.warning(self, "Error", f"{step_name} failed. Check the log.")

    def _enable_output_buttons(self):
        """Enable output buttons based on which files actually exist."""
        self.open_csv_btn.setEnabled(
            self.results_csv is not None and self.results_csv.exists()
        )
        self.open_pdf_btn.setEnabled(
            self.scored_pdf is not None and self.scored_pdf.exists()
        )
        self.open_folder_btn.setEnabled(
            self.work_dir is not None and self.work_dir.exists()
        )

    def _open_file(self, path: Optional[Path]):
        if path is None or not path.exists():
            QMessageBox.warning(self, "Not Found", f"File not found: {path}")
            return
        from ..utils import open_file_or_folder
        try:
            open_file_or_folder(path)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open: {e}")
