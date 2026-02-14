"""
Align Only page - runs alignment without scoring.

Exposes the full set of alignment pipeline parameters so users
can troubleshoot or fine-tune alignment independently.
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
        ALIGN_DEFAULTS,
        EST_DEFAULTS,
        FEAT_DEFAULTS,
        MATCH_DEFAULTS,
        RENDER_DEFAULTS,
    )
    from markshark.template_manager import TemplateManager
except ImportError:
    ALIGN_DEFAULTS = EST_DEFAULTS = FEAT_DEFAULTS = MATCH_DEFAULTS = RENDER_DEFAULTS = None
    TemplateManager = None


def _dflt(obj, attr: str, fallback):
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)


class AlignOnlyPage(QWidget):
    """
    Align Only workflow page.

    Runs alignment on scanned PDFs against a template without scoring.
    Exposes detailed pipeline parameters beyond what QuickGrade offers.
    """

    alignment_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.runner = CLIRunner(self)
        self.runner.output_received.connect(self._on_output)
        self.runner.error_received.connect(self._on_output)
        self.runner.finished.connect(self._on_finished)

        self.work_dir: Optional[Path] = None
        self.aligned_pdf: Optional[Path] = None

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
            "MarkShark - Align Scans",
            "Align scanned answer sheets to a template without scoring. "
            "Useful for troubleshooting alignment issues."
        )
        layout.addWidget(header)

        self.project_selector = ProjectSelector()
        layout.addWidget(self.project_selector)

        # Run button
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Alignment")
        self.run_btn.setMinimumHeight(36)
        from ..utils import RUN_BUTTON_STYLE
        self.run_btn.setStyleSheet(RUN_BUTTON_STYLE)
        self.run_btn.clicked.connect(self._run_align)
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

        # Scrollable options area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_container = QWidget()
        scroll_layout = QVBoxLayout(scroll_container)
        scroll.setWidget(scroll_container)

        # --- Input Files ---
        files_group = QGroupBox("Input Files")
        files_layout = QVBoxLayout(files_group)

        # Template
        tmpl_row = QHBoxLayout()
        tmpl_row.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(300)
        tmpl_row.addWidget(self.template_combo, 1)
        files_layout.addLayout(tmpl_row)

        self.scans_selector = FileSelector(
            "Scanned PDF:", "PDF files (*.pdf)", "Select scanned answer sheets..."
        )
        files_layout.addWidget(self.scans_selector)

        scroll_layout.addWidget(files_group)
        scroll_layout.addSpacing(8)

        # --- General Settings ---
        gen_group = QGroupBox("General")
        gen_layout = QGridLayout(gen_group)

        gen_layout.addWidget(QLabel("Alignment method:"), 0, 0)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["auto", "fast", "slow", "aruco"])
        self.method_combo.setToolTip(
            "auto: tries ArUco first, falls back to feature matching.\n"
            "fast: coarse-to-fine (72 DPI ORB + bubble grid). Uses template bubblemap.\n"
            "slow: full-res ORB only.\n"
            "aruco: ArUco markers only."
        )
        gen_layout.addWidget(self.method_combo, 0, 1)

        gen_layout.addWidget(QLabel("Render DPI:"), 1, 0)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(_dflt(RENDER_DEFAULTS, "dpi", 150))
        self.dpi_spin.setToolTip("DPI used when rendering PDF pages to images.")
        gen_layout.addWidget(self.dpi_spin, 1, 1)

        gen_layout.addWidget(QLabel("First page (0=all):"), 2, 0)
        self.first_page_spin = QSpinBox()
        self.first_page_spin.setRange(0, 9999)
        self.first_page_spin.setValue(0)
        self.first_page_spin.setToolTip("1-based. Set to 0 to process all pages.")
        gen_layout.addWidget(self.first_page_spin, 2, 1)

        gen_layout.addWidget(QLabel("Last page (0=all):"), 3, 0)
        self.last_page_spin = QSpinBox()
        self.last_page_spin.setRange(0, 9999)
        self.last_page_spin.setValue(0)
        self.last_page_spin.setToolTip("1-based inclusive. Set to 0 to process all pages.")
        gen_layout.addWidget(self.last_page_spin, 3, 1)

        scroll_layout.addWidget(gen_group)
        scroll_layout.addSpacing(8)

        # --- ArUco Detection ---
        aruco_group = QGroupBox("Step 1 — ArUco Marker Detection")
        aruco_layout = QGridLayout(aruco_group)

        aruco_layout.addWidget(QLabel("Min markers required:"), 0, 0)
        self.min_markers_spin = QSpinBox()
        self.min_markers_spin.setRange(0, 32)
        self.min_markers_spin.setValue(_dflt(ALIGN_DEFAULTS, "min_aruco", 4))
        self.min_markers_spin.setToolTip(
            "Minimum ArUco markers that must be found.\n"
            "If fewer are detected, falls back to feature matching."
        )
        aruco_layout.addWidget(self.min_markers_spin, 0, 1)

        aruco_layout.addWidget(QLabel("ArUco dictionary:"), 1, 0)
        self.dict_combo = QComboBox()
        self.dict_combo.addItems([
            "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
            "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
            "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
        ])
        self.dict_combo.setCurrentText(_dflt(ALIGN_DEFAULTS, "dict_name", "DICT_4X4_50"))
        self.dict_combo.setToolTip("ArUco marker dictionary to look for on the sheet.")
        aruco_layout.addWidget(self.dict_combo, 1, 1)

        scroll_layout.addWidget(aruco_group)
        scroll_layout.addSpacing(8)

        # --- Feature Detection ---
        feat_group = QGroupBox("Step 2 — Feature Detection (fallback)")
        feat_layout = QGridLayout(feat_group)

        feat_layout.addWidget(QLabel("Tile grid X:"), 0, 0)
        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setRange(1, 32)
        self.tiles_x_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_x", 8))
        self.tiles_x_spin.setToolTip("Image is divided into a grid for uniform feature extraction.")
        feat_layout.addWidget(self.tiles_x_spin, 0, 1)

        feat_layout.addWidget(QLabel("Tile grid Y:"), 1, 0)
        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setRange(1, 32)
        self.tiles_y_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_y", 10))
        feat_layout.addWidget(self.tiles_y_spin, 1, 1)

        feat_layout.addWidget(QLabel("Top-K per tile:"), 2, 0)
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(10, 1000)
        self.topk_spin.setValue(_dflt(FEAT_DEFAULTS, "topk_per_tile", 150))
        self.topk_spin.setToolTip("Best features kept from each tile. Higher = more data, slower.")
        feat_layout.addWidget(self.topk_spin, 2, 1)

        feat_layout.addWidget(QLabel("ORB max features:"), 3, 0)
        self.orb_nfeatures_spin = QSpinBox()
        self.orb_nfeatures_spin.setRange(100, 20000)
        self.orb_nfeatures_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_nfeatures", 3000))
        self.orb_nfeatures_spin.setToolTip("Total ORB features to extract across the whole image.")
        feat_layout.addWidget(self.orb_nfeatures_spin, 3, 1)

        feat_layout.addWidget(QLabel("ORB FAST threshold:"), 4, 0)
        self.orb_fast_spin = QSpinBox()
        self.orb_fast_spin.setRange(1, 100)
        self.orb_fast_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_fast_threshold", 12))
        self.orb_fast_spin.setToolTip("Lower = more features detected (including noise).")
        feat_layout.addWidget(self.orb_fast_spin, 4, 1)

        scroll_layout.addWidget(feat_group)
        scroll_layout.addSpacing(8)

        # --- Feature Matching ---
        match_group = QGroupBox("Step 3 — Feature Matching")
        match_layout = QGridLayout(match_group)

        match_layout.addWidget(QLabel("Lowe ratio test:"), 0, 0)
        self.ratio_test_spin = QDoubleSpinBox()
        self.ratio_test_spin.setRange(0.1, 1.0)
        self.ratio_test_spin.setDecimals(2)
        self.ratio_test_spin.setSingleStep(0.05)
        self.ratio_test_spin.setValue(_dflt(MATCH_DEFAULTS, "ratio_test", 0.75))
        self.ratio_test_spin.setToolTip("Rejects ambiguous matches. Lower = stricter.")
        match_layout.addWidget(self.ratio_test_spin, 0, 1)

        self.mutual_check_cb = QCheckBox("Mutual check")
        self.mutual_check_cb.setChecked(_dflt(MATCH_DEFAULTS, "mutual_check", True))
        self.mutual_check_cb.setToolTip("Only keep matches where both images agree.")
        match_layout.addWidget(self.mutual_check_cb, 1, 0)

        self.use_flann_cb = QCheckBox("Use FLANN matcher")
        self.use_flann_cb.setChecked(_dflt(MATCH_DEFAULTS, "use_flann", False))
        self.use_flann_cb.setToolTip("FLANN is faster for large feature sets but less precise.")
        match_layout.addWidget(self.use_flann_cb, 1, 1)

        match_layout.addWidget(QLabel("Max matches:"), 2, 0)
        self.max_matches_spin = QSpinBox()
        self.max_matches_spin.setRange(100, 50000)
        self.max_matches_spin.setValue(_dflt(MATCH_DEFAULTS, "max_matches", 5000))
        match_layout.addWidget(self.max_matches_spin, 2, 1)

        scroll_layout.addWidget(match_group)
        scroll_layout.addSpacing(8)

        # --- RANSAC & Homography ---
        ransac_group = QGroupBox("Step 4 — RANSAC & Homography")
        ransac_layout = QGridLayout(ransac_group)

        ransac_layout.addWidget(QLabel("Estimator method:"), 0, 0)
        self.estimator_combo = QComboBox()
        self.estimator_combo.addItems(["auto", "ransac", "usac"])
        self.estimator_combo.setCurrentText(_dflt(EST_DEFAULTS, "estimator_method", "auto"))
        self.estimator_combo.setToolTip("RANSAC filters bad matches; USAC is a newer adaptive variant.")
        ransac_layout.addWidget(self.estimator_combo, 0, 1)

        ransac_layout.addWidget(QLabel("RANSAC threshold:"), 1, 0)
        self.ransac_thresh_spin = QDoubleSpinBox()
        self.ransac_thresh_spin.setRange(0.1, 20)
        self.ransac_thresh_spin.setDecimals(1)
        self.ransac_thresh_spin.setValue(_dflt(EST_DEFAULTS, "ransac_thresh", 3.0))
        self.ransac_thresh_spin.setSuffix(" px")
        self.ransac_thresh_spin.setToolTip("Max pixel error for an inlier. Lower = stricter.")
        ransac_layout.addWidget(self.ransac_thresh_spin, 1, 1)

        ransac_layout.addWidget(QLabel("Max iterations:"), 2, 0)
        self.max_iters_spin = QSpinBox()
        self.max_iters_spin.setRange(100, 100000)
        self.max_iters_spin.setValue(_dflt(EST_DEFAULTS, "max_iters", 10000))
        self.max_iters_spin.setToolTip("More iterations = more likely to find best fit, but slower.")
        ransac_layout.addWidget(self.max_iters_spin, 2, 1)

        self.use_ecc_cb = QCheckBox("ECC refinement")
        self.use_ecc_cb.setChecked(_dflt(EST_DEFAULTS, "use_ecc", True))
        self.use_ecc_cb.setToolTip("Fine-tunes alignment by comparing pixel intensities.")
        ransac_layout.addWidget(self.use_ecc_cb, 3, 0)

        ransac_layout.addWidget(QLabel("ECC pyramid levels:"), 4, 0)
        self.ecc_levels_spin = QSpinBox()
        self.ecc_levels_spin.setRange(1, 8)
        self.ecc_levels_spin.setValue(_dflt(EST_DEFAULTS, "ecc_levels", 4))
        self.ecc_levels_spin.setToolTip("Multi-scale pyramid levels for ECC.")
        ransac_layout.addWidget(self.ecc_levels_spin, 4, 1)

        scroll_layout.addWidget(ransac_group)
        scroll_layout.addSpacing(8)

        # --- Quality Checks ---
        quality_group = QGroupBox("Step 5 — Alignment Quality Checks")
        quality_layout = QGridLayout(quality_group)

        quality_layout.addWidget(QLabel("Fail median residual:"), 0, 0)
        self.fail_med_spin = QDoubleSpinBox()
        self.fail_med_spin.setRange(0, 50)
        self.fail_med_spin.setDecimals(1)
        self.fail_med_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_med", 3.0))
        self.fail_med_spin.setSuffix(" px")
        self.fail_med_spin.setToolTip("Page flagged as failed if median error exceeds this.")
        quality_layout.addWidget(self.fail_med_spin, 0, 1)

        quality_layout.addWidget(QLabel("Fail P95 residual:"), 1, 0)
        self.fail_p95_spin = QDoubleSpinBox()
        self.fail_p95_spin.setRange(0, 50)
        self.fail_p95_spin.setDecimals(1)
        self.fail_p95_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_p95", 8.0))
        self.fail_p95_spin.setSuffix(" px")
        self.fail_p95_spin.setToolTip("Page flagged if the 95th-percentile error exceeds this.")
        quality_layout.addWidget(self.fail_p95_spin, 1, 1)

        quality_layout.addWidget(QLabel("Fail BR residual:"), 2, 0)
        self.fail_br_spin = QDoubleSpinBox()
        self.fail_br_spin.setRange(0, 50)
        self.fail_br_spin.setDecimals(1)
        self.fail_br_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_br", 8.0))
        self.fail_br_spin.setSuffix(" px")
        self.fail_br_spin.setToolTip("Page flagged if the bottom-right corner error exceeds this.")
        quality_layout.addWidget(self.fail_br_spin, 2, 1)

        scroll_layout.addWidget(quality_group)

        scroll_layout.addStretch()
        splitter.addWidget(scroll)

        # Log viewer + output buttons side-by-side
        log_row = QHBoxLayout()

        self.log = LogViewer("Alignment Log")
        log_row.addWidget(self.log, 1)

        # Output buttons — always visible, stacked, greyed out until results exist
        output_btn_panel = QVBoxLayout()
        output_btn_panel.setContentsMargins(0, 0, 0, 0)

        output_label = QLabel("Outputs")
        output_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #6c757d;")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        output_btn_panel.addWidget(output_label)

        self.open_pdf_btn = QPushButton("Open\nAligned PDF")
        self.open_pdf_btn.setEnabled(False)
        self.open_pdf_btn.setFixedWidth(100)
        self.open_pdf_btn.clicked.connect(lambda: self._open_file(self.aligned_pdf))
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
        self.scans_selector.set_start_dir(start)
        self._auto_populate_from_project()

    def _auto_populate_from_project(self):
        """Auto-fill selectors from project's flat structure if files exist."""
        project_dir = self.project_selector.project_dir()
        if not project_dir:
            return
        input_files = project_dir / "input_files"
        if not input_files.exists():
            return

        # Scans
        if not self.scans_selector.exists():
            for s in sorted(input_files.glob("scans.*")):
                self.scans_selector.set_path(str(s))
                break

    def _validate_inputs(self) -> bool:
        if not self.scans_selector.exists():
            QMessageBox.warning(self, "Missing Input", "Please select a scanned PDF file.")
            return False
        template = self.template_combo.currentData()
        if template is None:
            QMessageBox.warning(self, "Missing Template", "Please select a template.")
            return False
        return True

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------
    def _run_align(self):
        if self.runner.is_running():
            QMessageBox.information(self, "Running", "A process is already running.")
            return
        if not self._validate_inputs():
            return

        self.work_dir = self.project_selector.output_dir()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        template = self.template_combo.currentData()
        template_pdf = str(template.template_pdf_path)

        # Flat structure: aligned scans go in input_files/
        input_files = self.work_dir / "input_files"
        input_files.mkdir(exist_ok=True)
        self.aligned_pdf = input_files / "aligned_scans.pdf"

        self.log.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.run_btn.setEnabled(False)
        self.status_label.setText("Aligning scans...")

        self.log.append_header("ALIGNMENT")

        args = [
            "align",
            self.scans_selector.path(),
            "--template", template_pdf,
            "--out-pdf", str(self.aligned_pdf),
            "--dpi", str(self.dpi_spin.value()),
            "--align-method", self.method_combo.currentText(),
            "--min-markers", str(self.min_markers_spin.value()),
            "--dict-name", self.dict_combo.currentText(),
            "--orb-nfeatures", str(self.orb_nfeatures_spin.value()),
            "--match-ratio", str(self.ratio_test_spin.value()),
            "--estimator-method", self.estimator_combo.currentText(),
            "--ransac", str(self.ransac_thresh_spin.value()),
        ]

        # ECC toggle
        if self.use_ecc_cb.isChecked():
            args += ["--use-ecc"]
        else:
            args += ["--no-use-ecc"]

        # Bubblemap from template
        args += ["--bubblemap", str(template.bubblemap_yaml_path)]

        # Page range
        first = self.first_page_spin.value()
        last = self.last_page_spin.value()
        if first > 0:
            args += ["--first-page", str(first)]
        if last > 0:
            args += ["--last-page", str(last)]

        project_name = self.project_selector.project_name()
        if project_name:
            self.log.append_line(f"Project: {project_name}")
        self.log.append_line(f"Output directory: {self.work_dir}")
        self.log.append_line(f"Command: markshark {' '.join(args)}\n")
        self.runner.run(args, "align")

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
        log_path = logs_dir / f"align_log_{timestamp}.txt"
        if self.log.save_to_file(log_path):
            self.log.append_line(f"\nLog saved to: {log_path}")

    def _on_finished(self, exit_code: int, step_name: str):
        self.log.append_line(f"\n[{step_name}] Finished with exit code {exit_code}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._save_log()

        if exit_code == 0:
            self.status_label.setText("Alignment complete!")
            self._enable_output_buttons()
            self.alignment_complete.emit({
                "work_dir": str(self.work_dir),
                "aligned_pdf": str(self.aligned_pdf),
            })
            QMessageBox.information(self, "Complete", "Alignment complete!")
        else:
            self.status_label.setText(f"Error in {step_name}")
            QMessageBox.warning(self, "Error", f"{step_name} failed. Check the log.")

    def _enable_output_buttons(self):
        """Enable output buttons based on which files actually exist."""
        self.open_pdf_btn.setEnabled(
            self.aligned_pdf is not None and self.aligned_pdf.exists()
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
