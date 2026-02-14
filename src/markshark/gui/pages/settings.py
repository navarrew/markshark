"""
Settings page - application preferences and advanced configuration.

Collapsible sections expose every tuning knob from markshark.defaults,
pre-populated with the compiled-in default values.  Users can tweak,
save (to ~/.markshark/settings.json), and reset back to factory defaults.
"""

from pathlib import Path
from PySide6.QtCore import Qt

from ..models.settings_store import SettingsStore
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
    QScrollArea,
    QFrame,
    QToolButton,
    QSizePolicy,
)

from ..widgets import PageHeader

# Best-effort import of defaults ------------------------------------------------
try:
    from markshark.defaults import (
        SCORING_DEFAULTS,
        ALIGN_DEFAULTS,
        EST_DEFAULTS,
        FEAT_DEFAULTS,
        MATCH_DEFAULTS,
        RENDER_DEFAULTS,
        ANNOTATION_DEFAULTS,
    )
except ImportError:
    SCORING_DEFAULTS = ALIGN_DEFAULTS = None
    EST_DEFAULTS = FEAT_DEFAULTS = MATCH_DEFAULTS = RENDER_DEFAULTS = None
    ANNOTATION_DEFAULTS = None


def _dflt(obj, attr: str, fallback):
    if obj is None:
        return fallback
    return getattr(obj, attr, fallback)


def _make_bgr_row(bgr_tuple):
    """Create a row of B, G, R spin boxes for a BGR color tuple."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    b_spin = QSpinBox()
    b_spin.setRange(0, 255)
    b_spin.setPrefix("B:")
    b_spin.setValue(bgr_tuple[0])

    g_spin = QSpinBox()
    g_spin.setRange(0, 255)
    g_spin.setPrefix("G:")
    g_spin.setValue(bgr_tuple[1])

    r_spin = QSpinBox()
    r_spin.setRange(0, 255)
    r_spin.setPrefix("R:")
    r_spin.setValue(bgr_tuple[2])

    layout.addWidget(b_spin)
    layout.addWidget(g_spin)
    layout.addWidget(r_spin)
    layout.addStretch()

    widget.b_spin = b_spin
    widget.g_spin = g_spin
    widget.r_spin = r_spin
    return widget


def _get_bgr(widget):
    """Read (B, G, R) tuple from a widget created by _make_bgr_row."""
    return (widget.b_spin.value(), widget.g_spin.value(), widget.r_spin.value())


def _set_bgr(widget, bgr_tuple):
    """Set (B, G, R) values on a widget created by _make_bgr_row."""
    widget.b_spin.setValue(bgr_tuple[0])
    widget.g_spin.setValue(bgr_tuple[1])
    widget.r_spin.setValue(bgr_tuple[2])


# ---------------------------------------------------------------------------
# Collapsible group helper
# ---------------------------------------------------------------------------
class _CollapsibleSection(QWidget):
    """A section with a clickable header that expands / collapses its body."""

    def __init__(self, title: str, parent=None, start_collapsed: bool = True):
        super().__init__(parent)
        self._body_visible = not start_collapsed

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header button
        self._toggle = QToolButton()
        self._toggle.setStyleSheet(
            "QToolButton { font-weight: bold; border: none; padding: 6px; }"
        )
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(self._body_visible)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._body_visible else Qt.ArrowType.RightArrow
        )
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle)

        # Body frame
        self._body = QFrame()
        self._body.setFrameShape(QFrame.Shape.StyledPanel)
        self._body_layout = QFormLayout(self._body)
        self._body_layout.setContentsMargins(16, 6, 6, 6)
        self._body.setVisible(self._body_visible)
        layout.addWidget(self._body)

    @property
    def form(self) -> QFormLayout:
        return self._body_layout

    def _on_toggle(self, checked: bool):
        self._body_visible = checked
        self._body.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------
class SettingsPage(QWidget):
    """Application settings and preferences with collapsible advanced sections."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = SettingsStore()
        self._setup_ui()
        self._load_settings()

    # ---------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------
    def _setup_ui(self):
        """Build the page UI."""
        outer = QVBoxLayout(self)

        header = PageHeader("MarkShark Settings", "Configure application preferences and advanced options.")
        outer.addWidget(header)

        # Scrollable area for all sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, 1)

        container = QWidget()
        layout = QVBoxLayout(container)
        scroll.setWidget(container)

        # === General Preferences (always visible) ===
        prefs_group = QGroupBox("General Preferences")
        prefs_layout = QVBoxLayout(prefs_group)

        self.auto_open_results = QCheckBox("Automatically open results after grading")
        self.auto_open_results.setChecked(True)
        prefs_layout.addWidget(self.auto_open_results)

        prefs_layout.addSpacing(8)

        # Read-only display of key paths
        paths_header = QLabel("<b>Application Paths</b>")
        paths_header.setTextFormat(Qt.TextFormat.RichText)
        prefs_layout.addWidget(paths_header)

        try:
            from markshark.template_manager import TemplateManager
            templates_path = str(TemplateManager.get_default_templates_dir())
        except Exception:
            templates_path = "(could not detect)"
        templates_label = QLabel(f"Templates directory:  <code>{templates_path}</code>")
        templates_label.setTextFormat(Qt.TextFormat.RichText)
        templates_label.setStyleSheet("color: #fff; font-size: 11px; padding: 2px 0;")
        prefs_layout.addWidget(templates_label)

        config_dir = Path.home() / ".markshark"
        config_label = QLabel(f"Config data location:  <code>{config_dir}</code>")
        config_label.setTextFormat(Qt.TextFormat.RichText)
        config_label.setStyleSheet("color: #fff; font-size: 11px; padding: 2px 0;")
        prefs_layout.addWidget(config_label)

        layout.addWidget(prefs_group)
        layout.addSpacing(8)

        # === Scoring Settings (collapsible) ===
        self._scoring_section = _CollapsibleSection("Scoring Settings")
        f = self._scoring_section.form

        self.min_fill_spin = QSpinBox()
        self.min_fill_spin.setRange(0, 100)
        self.min_fill_spin.setSuffix("%")
        self.min_fill_spin.setValue(_dflt(SCORING_DEFAULTS, "min_fill", 45))
        f.addRow("Min fill threshold:", self.min_fill_spin)

        self.calibrate_bg_cb = QCheckBox("Calibrate background")
        self.calibrate_bg_cb.setChecked(_dflt(SCORING_DEFAULTS, "calibrate_background", True))
        f.addRow("", self.calibrate_bg_cb)

        self.bg_percentile_spin = QDoubleSpinBox()
        self.bg_percentile_spin.setRange(0, 100)
        self.bg_percentile_spin.setValue(_dflt(SCORING_DEFAULTS, "background_percentile", 10.0))
        f.addRow("Background percentile:", self.bg_percentile_spin)

        self.adaptive_cb = QCheckBox("Adaptive rescoring")
        self.adaptive_cb.setChecked(_dflt(SCORING_DEFAULTS, "adaptive_rescoring", True))
        f.addRow("", self.adaptive_cb)

        self.adaptive_max_spin = QSpinBox()
        self.adaptive_max_spin.setRange(0, 100)
        self.adaptive_max_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_max_adjustment", 40))
        f.addRow("Adaptive max adjustment:", self.adaptive_max_spin)

        self.adaptive_floor_spin = QDoubleSpinBox()
        self.adaptive_floor_spin.setRange(0, 100)
        self.adaptive_floor_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_min_above_floor", 30))
        f.addRow("Adaptive min above floor:", self.adaptive_floor_spin)

        self.auto_calibrate_cb = QCheckBox("Auto-calibrate threshold")
        self.auto_calibrate_cb.setChecked(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
        f.addRow("", self.auto_calibrate_cb)

        self.fixed_thresh_spin = QSpinBox()
        self.fixed_thresh_spin.setRange(0, 255)
        self.fixed_thresh_spin.setValue(_dflt(SCORING_DEFAULTS, "fixed_thresh", 180))
        f.addRow("Fixed threshold (gray):", self.fixed_thresh_spin)

        layout.addWidget(self._scoring_section)
        layout.addSpacing(8)

        # === Alignment Pipeline (collapsible) ===
        # All alignment settings grouped by pipeline stage:
        #   1. General → 2. ArUco markers → 3. Feature detection (fallback)
        #   → 4. Feature matching → 5. RANSAC / homography → 6. Quality checks
        self._align_section = _CollapsibleSection("Alignment Pipeline")
        f = self._align_section.form

        # ── 1. General ──
        f.addRow(QLabel("<b>General</b>"))

        self.default_dpi = QSpinBox()
        self.default_dpi.setRange(72, 600)
        self.default_dpi.setValue(_dflt(ALIGN_DEFAULTS, "dpi", 150))
        self.default_dpi.setToolTip("DPI used when rendering PDF pages to images for alignment.")
        f.addRow("Render DPI:", self.default_dpi)

        self.default_align_method = QComboBox()
        self.default_align_method.addItems(["auto", "fast", "slow", "aruco"])
        self.default_align_method.setToolTip(
            "auto: tries ArUco first, falls back to feature matching.\n"
            "aruco: ArUco markers only (fails if too few found).\n"
            "fast/slow: feature-based with different quality/speed trade-offs.")
        f.addRow("Alignment method:", self.default_align_method)

        # ── 2. ArUco marker detection ──
        f.addRow(QLabel("<b>Step 1 — ArUco Marker Detection</b>"))

        self.min_aruco_spin = QSpinBox()
        self.min_aruco_spin.setRange(0, 32)
        self.min_aruco_spin.setValue(_dflt(ALIGN_DEFAULTS, "min_aruco", 4))
        self.min_aruco_spin.setToolTip(
            "Minimum ArUco markers that must be found for marker-based alignment.\n"
            "If fewer are detected, falls back to feature matching (in auto mode).")
        f.addRow("Min markers required:", self.min_aruco_spin)

        # ── 3. Feature detection (fallback when ArUco fails) ──
        f.addRow(QLabel("<b>Step 2 — Feature Detection (fallback)</b>"))

        self.tiles_x_spin = QSpinBox()
        self.tiles_x_spin.setRange(1, 32)
        self.tiles_x_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_x", 8))
        self.tiles_x_spin.setToolTip("Image is divided into a grid for uniform feature extraction.")
        f.addRow("Tile grid X:", self.tiles_x_spin)

        self.tiles_y_spin = QSpinBox()
        self.tiles_y_spin.setRange(1, 32)
        self.tiles_y_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_y", 10))
        f.addRow("Tile grid Y:", self.tiles_y_spin)

        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(10, 1000)
        self.topk_spin.setValue(_dflt(FEAT_DEFAULTS, "topk_per_tile", 150))
        self.topk_spin.setToolTip("Best features kept from each tile. Higher = more data, slower.")
        f.addRow("Top-K per tile:", self.topk_spin)

        self.orb_nfeatures_spin = QSpinBox()
        self.orb_nfeatures_spin.setRange(100, 20000)
        self.orb_nfeatures_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_nfeatures", 3000))
        self.orb_nfeatures_spin.setToolTip("Total ORB features to extract across the whole image.")
        f.addRow("ORB max features:", self.orb_nfeatures_spin)

        self.orb_fast_spin = QSpinBox()
        self.orb_fast_spin.setRange(1, 100)
        self.orb_fast_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_fast_threshold", 12))
        self.orb_fast_spin.setToolTip("Lower = more features detected (including noise).")
        f.addRow("ORB FAST threshold:", self.orb_fast_spin)

        # ── 4. Feature matching ──
        f.addRow(QLabel("<b>Step 3 — Feature Matching</b>"))

        self.ratio_test_spin = QDoubleSpinBox()
        self.ratio_test_spin.setRange(0.1, 1.0)
        self.ratio_test_spin.setDecimals(2)
        self.ratio_test_spin.setSingleStep(0.05)
        self.ratio_test_spin.setValue(_dflt(MATCH_DEFAULTS, "ratio_test", 0.75))
        self.ratio_test_spin.setToolTip(
            "Lowe's ratio test: rejects ambiguous matches.\n"
            "Lower = stricter (fewer but better matches).")
        f.addRow("Lowe ratio test:", self.ratio_test_spin)

        self.mutual_check_cb = QCheckBox("Mutual check")
        self.mutual_check_cb.setChecked(_dflt(MATCH_DEFAULTS, "mutual_check", True))
        self.mutual_check_cb.setToolTip("Only keep matches where both images agree on the pairing.")
        f.addRow("", self.mutual_check_cb)

        self.max_matches_spin = QSpinBox()
        self.max_matches_spin.setRange(100, 50000)
        self.max_matches_spin.setValue(_dflt(MATCH_DEFAULTS, "max_matches", 5000))
        f.addRow("Max matches:", self.max_matches_spin)

        self.use_flann_cb = QCheckBox("Use FLANN matcher")
        self.use_flann_cb.setChecked(_dflt(MATCH_DEFAULTS, "use_flann", False))
        self.use_flann_cb.setToolTip("FLANN is faster for large feature sets but less precise.")
        f.addRow("", self.use_flann_cb)

        # ── 5. RANSAC / homography estimation ──
        f.addRow(QLabel("<b>Step 4 — RANSAC & Homography</b>"))

        self.estimator_combo = QComboBox()
        self.estimator_combo.addItems(["auto", "ransac", "usac"])
        self.estimator_combo.setCurrentText(_dflt(EST_DEFAULTS, "estimator_method", "auto"))
        self.estimator_combo.setToolTip(
            "RANSAC filters out bad matches to compute a robust transformation.\n"
            "USAC is a newer variant with adaptive features.")
        f.addRow("Estimator method:", self.estimator_combo)

        self.ransac_thresh_spin = QDoubleSpinBox()
        self.ransac_thresh_spin.setRange(0.1, 20)
        self.ransac_thresh_spin.setDecimals(1)
        self.ransac_thresh_spin.setValue(_dflt(EST_DEFAULTS, "ransac_thresh", 3.0))
        self.ransac_thresh_spin.setSuffix(" px")
        self.ransac_thresh_spin.setToolTip(
            "Max pixel error for a match to be considered an inlier.\n"
            "Lower = stricter filtering of bad matches.")
        f.addRow("RANSAC threshold:", self.ransac_thresh_spin)

        self.max_iters_spin = QSpinBox()
        self.max_iters_spin.setRange(100, 100000)
        self.max_iters_spin.setValue(_dflt(EST_DEFAULTS, "max_iters", 10000))
        self.max_iters_spin.setToolTip("More iterations = more likely to find the best fit, but slower.")
        f.addRow("Max iterations:", self.max_iters_spin)

        self.use_ecc_cb = QCheckBox("ECC refinement")
        self.use_ecc_cb.setChecked(_dflt(EST_DEFAULTS, "use_ecc", True))
        self.use_ecc_cb.setToolTip(
            "Enhanced Correlation Coefficient: fine-tunes the alignment\n"
            "after RANSAC by directly comparing pixel intensities.")
        f.addRow("", self.use_ecc_cb)

        self.ecc_levels_spin = QSpinBox()
        self.ecc_levels_spin.setRange(1, 8)
        self.ecc_levels_spin.setValue(_dflt(EST_DEFAULTS, "ecc_levels", 4))
        self.ecc_levels_spin.setToolTip("Multi-scale pyramid levels for ECC. More levels = coarser-to-fine.")
        f.addRow("ECC pyramid levels:", self.ecc_levels_spin)

        # ── 6. Quality checks ──
        f.addRow(QLabel("<b>Step 5 — Alignment Quality Checks</b>"))

        self.fail_med_spin = QDoubleSpinBox()
        self.fail_med_spin.setRange(0, 50)
        self.fail_med_spin.setDecimals(1)
        self.fail_med_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_med", 3.0))
        self.fail_med_spin.setSuffix(" px")
        self.fail_med_spin.setToolTip(
            "If the median alignment error exceeds this, the page is flagged as failed.\n"
            "Higher = more tolerant of imperfect alignments.")
        f.addRow("Fail median residual:", self.fail_med_spin)

        self.fail_p95_spin = QDoubleSpinBox()
        self.fail_p95_spin.setRange(0, 50)
        self.fail_p95_spin.setDecimals(1)
        self.fail_p95_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_p95", 8.0))
        self.fail_p95_spin.setSuffix(" px")
        self.fail_p95_spin.setToolTip("If the 95th-percentile error exceeds this, the page is flagged.")
        f.addRow("Fail P95 residual:", self.fail_p95_spin)

        self.fail_br_spin = QDoubleSpinBox()
        self.fail_br_spin.setRange(0, 50)
        self.fail_br_spin.setDecimals(1)
        self.fail_br_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_br", 8.0))
        self.fail_br_spin.setSuffix(" px")
        self.fail_br_spin.setToolTip("If the bottom-right corner error exceeds this, the page is flagged.")
        f.addRow("Fail BR residual:", self.fail_br_spin)

        layout.addWidget(self._align_section)
        layout.addSpacing(8)

        # === Render Settings (collapsible) ===
        self._render_section = _CollapsibleSection("Image/PDF Rendering Settings")
        f = self._render_section.form

        self.render_dpi_spin = QSpinBox()
        self.render_dpi_spin.setRange(72, 600)
        self.render_dpi_spin.setValue(_dflt(RENDER_DEFAULTS, "dpi", 150))
        f.addRow("Render DPI:", self.render_dpi_spin)

        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["png", "jpg"])
        self.image_format_combo.setCurrentText(_dflt(RENDER_DEFAULTS, "image_format", "png"))
        f.addRow("Image format:", self.image_format_combo)

        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(1, 100)
        self.jpeg_quality_spin.setValue(_dflt(RENDER_DEFAULTS, "jpeg_quality", 85))
        f.addRow("JPEG quality:", self.jpeg_quality_spin)

        self.pdf_quality_spin = QSpinBox()
        self.pdf_quality_spin.setRange(1, 100)
        self.pdf_quality_spin.setValue(_dflt(RENDER_DEFAULTS, "pdf_quality", 85))
        f.addRow("PDF quality:", self.pdf_quality_spin)

        layout.addWidget(self._render_section)
        layout.addSpacing(8)

        # === Annotation Settings (collapsible) ===
        self._annot_section = _CollapsibleSection("Annotated PDF Settings")
        f = self._annot_section.form

        # Circle colors
        self.annot_color_correct = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "color_correct", (0, 200, 0)))
        f.addRow("Color of correct questions (BGR):", self.annot_color_correct)

        self.annot_color_incorrect = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "color_incorrect", (0, 0, 255)))
        f.addRow("Color of incorrect questions (BGR):", self.annot_color_incorrect)

        self.annot_color_blank = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "color_blank", (160, 160, 160)))
        f.addRow("Color of unanswered questions (BGR):", self.annot_color_blank)

        self.annot_color_multi = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "color_multi", (0, 140, 255)))
        f.addRow("Color of multi-answer rows (BGR):", self.annot_color_multi)

        self.annot_color_blank_row = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "color_blank_answer_row", (255, 0, 255)))
        f.addRow("Color of blank answer rows: (BGR)", self.annot_color_blank_row)

        # Circle/line thickness
        self.annot_thickness_answers = QSpinBox()
        self.annot_thickness_answers.setRange(1, 10)
        self.annot_thickness_answers.setValue(
            _dflt(ANNOTATION_DEFAULTS, "thickness_answers", 2))
        f.addRow("Answer circle thickness:", self.annot_thickness_answers)

        self.annot_thickness_names = QSpinBox()
        self.annot_thickness_names.setRange(1, 10)
        self.annot_thickness_names.setValue(
            _dflt(ANNOTATION_DEFAULTS, "thickness_names", 2))
        f.addRow("Name zone circle thickness:", self.annot_thickness_names)

        # % fill label settings
        self.annot_pct_color = _make_bgr_row(
            _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_color", (255, 0, 0)))
        f.addRow("% fill font color (BGR):", self.annot_pct_color)

        self.annot_pct_scale = QDoubleSpinBox()
        self.annot_pct_scale.setRange(0.1, 3.0)
        self.annot_pct_scale.setDecimals(2)
        self.annot_pct_scale.setSingleStep(0.1)
        self.annot_pct_scale.setValue(
            _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_scale", 0.4))
        f.addRow("% fill font scale:", self.annot_pct_scale)

        self.annot_pct_thickness = QSpinBox()
        self.annot_pct_thickness.setRange(1, 5)
        self.annot_pct_thickness.setValue(
            _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_thickness", 1))
        f.addRow("% fill font thickness:", self.annot_pct_thickness)

        self.annot_pct_position = QSpinBox()
        self.annot_pct_position.setRange(-50, 50)
        self.annot_pct_position.setSuffix(" px")
        self.annot_pct_position.setValue(
            _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_position", 3))
        self.annot_pct_position.setToolTip(
            "Vertical offset for % label. 0 = on bubble, positive = above, negative = inside.")
        f.addRow("fill label vertical offset (px):", self.annot_pct_position)

        # General label font
        self.annot_label_scale = QDoubleSpinBox()
        self.annot_label_scale.setRange(0.1, 3.0)
        self.annot_label_scale.setDecimals(2)
        self.annot_label_scale.setSingleStep(0.1)
        self.annot_label_scale.setValue(
            _dflt(ANNOTATION_DEFAULTS, "label_font_scale", 0.5))
        f.addRow("Label font scale:", self.annot_label_scale)

        self.annot_label_thickness = QSpinBox()
        self.annot_label_thickness.setRange(1, 5)
        self.annot_label_thickness.setValue(
            _dflt(ANNOTATION_DEFAULTS, "label_thickness", 1))
        f.addRow("Label font thickness:", self.annot_label_thickness)

        # Box drawing
        self.annot_box_multi_cb = QCheckBox("Draw box around multi-answer rows")
        self.annot_box_multi_cb.setChecked(
            _dflt(ANNOTATION_DEFAULTS, "box_multi", True))
        f.addRow("", self.annot_box_multi_cb)

        self.annot_box_blank_cb = QCheckBox("Draw box around blank-answer rows")
        self.annot_box_blank_cb.setChecked(
            _dflt(ANNOTATION_DEFAULTS, "box_blank_answer_row", True))
        f.addRow("", self.annot_box_blank_cb)

        self.annot_box_thickness = QSpinBox()
        self.annot_box_thickness.setRange(1, 10)
        self.annot_box_thickness.setValue(
            _dflt(ANNOTATION_DEFAULTS, "box_thickness", 2))
        f.addRow("Box thickness:", self.annot_box_thickness)

        self.annot_box_pad = QSpinBox()
        self.annot_box_pad.setRange(0, 20)
        self.annot_box_pad.setSuffix(" px")
        self.annot_box_pad.setValue(
            _dflt(ANNOTATION_DEFAULTS, "box_pad", 4))
        f.addRow("Box padding:", self.annot_box_pad)

        layout.addWidget(self._annot_section)

        # Stretch at bottom
        layout.addStretch()

        # === Buttons (outside scroll area) ===
        btn_layout = QHBoxLayout()

        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_settings)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()
        outer.addLayout(btn_layout)

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------
    def _load_bgr(self, key: str, widget, default):
        """Load a BGR tuple from settings into a _make_bgr_row widget."""
        b = int(self.settings.value(f"{key}/b", default[0]))
        g = int(self.settings.value(f"{key}/g", default[1]))
        r = int(self.settings.value(f"{key}/r", default[2]))
        _set_bgr(widget, (b, g, r))

    def _save_bgr(self, key: str, widget):
        """Save a BGR tuple from a _make_bgr_row widget to settings."""
        bgr = _get_bgr(widget)
        self.settings.setValue(f"{key}/b", bgr[0])
        self.settings.setValue(f"{key}/g", bgr[1])
        self.settings.setValue(f"{key}/r", bgr[2])

    def _load_settings(self):
        """Load settings from SettingsStore."""
        # General
        self.auto_open_results.setChecked(
            self.settings.value("defaults/auto_open", True, type=bool)
        )

        # Scoring
        self.min_fill_spin.setValue(int(self.settings.value(
            "scoring/min_fill", _dflt(SCORING_DEFAULTS, "min_fill", 45))))
        self.calibrate_bg_cb.setChecked(
            self.settings.value("scoring/calibrate_background",
                                _dflt(SCORING_DEFAULTS, "calibrate_background", True), type=bool))
        self.bg_percentile_spin.setValue(float(self.settings.value(
            "scoring/background_percentile", _dflt(SCORING_DEFAULTS, "background_percentile", 10.0))))
        self.adaptive_cb.setChecked(
            self.settings.value("scoring/adaptive_rescoring",
                                _dflt(SCORING_DEFAULTS, "adaptive_rescoring", True), type=bool))
        self.adaptive_max_spin.setValue(int(self.settings.value(
            "scoring/adaptive_max_adjustment", _dflt(SCORING_DEFAULTS, "adaptive_max_adjustment", 40))))
        self.adaptive_floor_spin.setValue(float(self.settings.value(
            "scoring/adaptive_min_above_floor", _dflt(SCORING_DEFAULTS, "adaptive_min_above_floor", 30))))
        self.auto_calibrate_cb.setChecked(
            self.settings.value("scoring/auto_calibrate_thresh",
                                _dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True), type=bool))
        self.fixed_thresh_spin.setValue(int(self.settings.value(
            "scoring/fixed_thresh", _dflt(SCORING_DEFAULTS, "fixed_thresh", 180))))

        # Alignment
        self.default_dpi.setValue(int(self.settings.value(
            "align/dpi", _dflt(ALIGN_DEFAULTS, "dpi", 150))))
        self.default_align_method.setCurrentText(
            self.settings.value("align/method", "auto"))
        self.min_aruco_spin.setValue(int(self.settings.value(
            "align/min_aruco", _dflt(ALIGN_DEFAULTS, "min_aruco", 4))))
        self.fail_med_spin.setValue(float(self.settings.value(
            "align/fail_med", _dflt(ALIGN_DEFAULTS, "fail_med", 3.0))))
        self.fail_p95_spin.setValue(float(self.settings.value(
            "align/fail_p95", _dflt(ALIGN_DEFAULTS, "fail_p95", 8.0))))
        self.fail_br_spin.setValue(float(self.settings.value(
            "align/fail_br", _dflt(ALIGN_DEFAULTS, "fail_br", 8.0))))

        # Estimation
        self.estimator_combo.setCurrentText(
            self.settings.value("est/estimator_method", _dflt(EST_DEFAULTS, "estimator_method", "auto")))
        self.ransac_thresh_spin.setValue(float(self.settings.value(
            "est/ransac_thresh", _dflt(EST_DEFAULTS, "ransac_thresh", 3.0))))
        self.max_iters_spin.setValue(int(self.settings.value(
            "est/max_iters", _dflt(EST_DEFAULTS, "max_iters", 10000))))
        self.use_ecc_cb.setChecked(
            self.settings.value("est/use_ecc", _dflt(EST_DEFAULTS, "use_ecc", True), type=bool))
        self.ecc_levels_spin.setValue(int(self.settings.value(
            "est/ecc_levels", _dflt(EST_DEFAULTS, "ecc_levels", 4))))

        # Features
        self.tiles_x_spin.setValue(int(self.settings.value(
            "feat/tiles_x", _dflt(FEAT_DEFAULTS, "tiles_x", 8))))
        self.tiles_y_spin.setValue(int(self.settings.value(
            "feat/tiles_y", _dflt(FEAT_DEFAULTS, "tiles_y", 10))))
        self.topk_spin.setValue(int(self.settings.value(
            "feat/topk_per_tile", _dflt(FEAT_DEFAULTS, "topk_per_tile", 150))))
        self.orb_nfeatures_spin.setValue(int(self.settings.value(
            "feat/orb_nfeatures", _dflt(FEAT_DEFAULTS, "orb_nfeatures", 3000))))
        self.orb_fast_spin.setValue(int(self.settings.value(
            "feat/orb_fast_threshold", _dflt(FEAT_DEFAULTS, "orb_fast_threshold", 12))))

        # Matching
        self.ratio_test_spin.setValue(float(self.settings.value(
            "match/ratio_test", _dflt(MATCH_DEFAULTS, "ratio_test", 0.75))))
        self.mutual_check_cb.setChecked(
            self.settings.value("match/mutual_check",
                                _dflt(MATCH_DEFAULTS, "mutual_check", True), type=bool))
        self.max_matches_spin.setValue(int(self.settings.value(
            "match/max_matches", _dflt(MATCH_DEFAULTS, "max_matches", 5000))))
        self.use_flann_cb.setChecked(
            self.settings.value("match/use_flann",
                                _dflt(MATCH_DEFAULTS, "use_flann", False), type=bool))

        # Rendering
        self.render_dpi_spin.setValue(int(self.settings.value(
            "render/dpi", _dflt(RENDER_DEFAULTS, "dpi", 150))))
        self.image_format_combo.setCurrentText(
            self.settings.value("render/image_format", _dflt(RENDER_DEFAULTS, "image_format", "png")))
        self.jpeg_quality_spin.setValue(int(self.settings.value(
            "render/jpeg_quality", _dflt(RENDER_DEFAULTS, "jpeg_quality", 85))))
        self.pdf_quality_spin.setValue(int(self.settings.value(
            "render/pdf_quality", _dflt(RENDER_DEFAULTS, "pdf_quality", 85))))

        # Annotations
        self._load_bgr("annot/color_correct", self.annot_color_correct,
                        _dflt(ANNOTATION_DEFAULTS, "color_correct", (0, 200, 0)))
        self._load_bgr("annot/color_incorrect", self.annot_color_incorrect,
                        _dflt(ANNOTATION_DEFAULTS, "color_incorrect", (0, 0, 255)))
        self._load_bgr("annot/color_blank", self.annot_color_blank,
                        _dflt(ANNOTATION_DEFAULTS, "color_blank", (160, 160, 160)))
        self._load_bgr("annot/color_multi", self.annot_color_multi,
                        _dflt(ANNOTATION_DEFAULTS, "color_multi", (0, 140, 255)))
        self._load_bgr("annot/color_blank_answer_row", self.annot_color_blank_row,
                        _dflt(ANNOTATION_DEFAULTS, "color_blank_answer_row", (255, 0, 255)))
        self.annot_thickness_answers.setValue(int(self.settings.value(
            "annot/thickness_answers", _dflt(ANNOTATION_DEFAULTS, "thickness_answers", 2))))
        self.annot_thickness_names.setValue(int(self.settings.value(
            "annot/thickness_names", _dflt(ANNOTATION_DEFAULTS, "thickness_names", 2))))
        self._load_bgr("annot/pct_fill_font_color", self.annot_pct_color,
                        _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_color", (255, 0, 0)))
        self.annot_pct_scale.setValue(float(self.settings.value(
            "annot/pct_fill_font_scale", _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_scale", 0.4))))
        self.annot_pct_thickness.setValue(int(self.settings.value(
            "annot/pct_fill_font_thickness", _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_thickness", 1))))
        self.annot_pct_position.setValue(int(self.settings.value(
            "annot/pct_fill_font_position", _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_position", 3))))
        self.annot_label_scale.setValue(float(self.settings.value(
            "annot/label_font_scale", _dflt(ANNOTATION_DEFAULTS, "label_font_scale", 0.5))))
        self.annot_label_thickness.setValue(int(self.settings.value(
            "annot/label_thickness", _dflt(ANNOTATION_DEFAULTS, "label_thickness", 1))))
        self.annot_box_multi_cb.setChecked(
            self.settings.value("annot/box_multi",
                                _dflt(ANNOTATION_DEFAULTS, "box_multi", True), type=bool))
        self.annot_box_blank_cb.setChecked(
            self.settings.value("annot/box_blank_answer_row",
                                _dflt(ANNOTATION_DEFAULTS, "box_blank_answer_row", True), type=bool))
        self.annot_box_thickness.setValue(int(self.settings.value(
            "annot/box_thickness", _dflt(ANNOTATION_DEFAULTS, "box_thickness", 2))))
        self.annot_box_pad.setValue(int(self.settings.value(
            "annot/box_pad", _dflt(ANNOTATION_DEFAULTS, "box_pad", 4))))

    def _save_settings(self):
        """Save all settings to SettingsStore."""
        s = self.settings

        # General
        s.setValue("defaults/auto_open", self.auto_open_results.isChecked())

        # Scoring
        s.setValue("scoring/min_fill", self.min_fill_spin.value())
        s.setValue("scoring/calibrate_background", self.calibrate_bg_cb.isChecked())
        s.setValue("scoring/background_percentile", self.bg_percentile_spin.value())
        s.setValue("scoring/adaptive_rescoring", self.adaptive_cb.isChecked())
        s.setValue("scoring/adaptive_max_adjustment", self.adaptive_max_spin.value())
        s.setValue("scoring/adaptive_min_above_floor", self.adaptive_floor_spin.value())
        s.setValue("scoring/auto_calibrate_thresh", self.auto_calibrate_cb.isChecked())
        s.setValue("scoring/fixed_thresh", self.fixed_thresh_spin.value())

        # Alignment
        s.setValue("align/dpi", self.default_dpi.value())
        s.setValue("align/method", self.default_align_method.currentText())
        s.setValue("align/min_aruco", self.min_aruco_spin.value())
        s.setValue("align/fail_med", self.fail_med_spin.value())
        s.setValue("align/fail_p95", self.fail_p95_spin.value())
        s.setValue("align/fail_br", self.fail_br_spin.value())

        # Estimation
        s.setValue("est/estimator_method", self.estimator_combo.currentText())
        s.setValue("est/ransac_thresh", self.ransac_thresh_spin.value())
        s.setValue("est/max_iters", self.max_iters_spin.value())
        s.setValue("est/use_ecc", self.use_ecc_cb.isChecked())
        s.setValue("est/ecc_levels", self.ecc_levels_spin.value())

        # Features
        s.setValue("feat/tiles_x", self.tiles_x_spin.value())
        s.setValue("feat/tiles_y", self.tiles_y_spin.value())
        s.setValue("feat/topk_per_tile", self.topk_spin.value())
        s.setValue("feat/orb_nfeatures", self.orb_nfeatures_spin.value())
        s.setValue("feat/orb_fast_threshold", self.orb_fast_spin.value())

        # Matching
        s.setValue("match/ratio_test", self.ratio_test_spin.value())
        s.setValue("match/mutual_check", self.mutual_check_cb.isChecked())
        s.setValue("match/max_matches", self.max_matches_spin.value())
        s.setValue("match/use_flann", self.use_flann_cb.isChecked())

        # Rendering
        s.setValue("render/dpi", self.render_dpi_spin.value())
        s.setValue("render/image_format", self.image_format_combo.currentText())
        s.setValue("render/jpeg_quality", self.jpeg_quality_spin.value())
        s.setValue("render/pdf_quality", self.pdf_quality_spin.value())

        # Annotations
        self._save_bgr("annot/color_correct", self.annot_color_correct)
        self._save_bgr("annot/color_incorrect", self.annot_color_incorrect)
        self._save_bgr("annot/color_blank", self.annot_color_blank)
        self._save_bgr("annot/color_multi", self.annot_color_multi)
        self._save_bgr("annot/color_blank_answer_row", self.annot_color_blank_row)
        s.setValue("annot/thickness_answers", self.annot_thickness_answers.value())
        s.setValue("annot/thickness_names", self.annot_thickness_names.value())
        self._save_bgr("annot/pct_fill_font_color", self.annot_pct_color)
        s.setValue("annot/pct_fill_font_scale", self.annot_pct_scale.value())
        s.setValue("annot/pct_fill_font_thickness", self.annot_pct_thickness.value())
        s.setValue("annot/pct_fill_font_position", self.annot_pct_position.value())
        s.setValue("annot/label_font_scale", self.annot_label_scale.value())
        s.setValue("annot/label_thickness", self.annot_label_thickness.value())
        s.setValue("annot/box_multi", self.annot_box_multi_cb.isChecked())
        s.setValue("annot/box_blank_answer_row", self.annot_box_blank_cb.isChecked())
        s.setValue("annot/box_thickness", self.annot_box_thickness.value())
        s.setValue("annot/box_pad", self.annot_box_pad.value())

        s.sync()
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved.")

    def _reset_settings(self):
        """Reset all values to compiled-in defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset all settings to factory defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # General
        self.auto_open_results.setChecked(True)

        # Scoring
        self.min_fill_spin.setValue(_dflt(SCORING_DEFAULTS, "min_fill", 45))
        self.calibrate_bg_cb.setChecked(_dflt(SCORING_DEFAULTS, "calibrate_background", True))
        self.bg_percentile_spin.setValue(_dflt(SCORING_DEFAULTS, "background_percentile", 10.0))
        self.adaptive_cb.setChecked(_dflt(SCORING_DEFAULTS, "adaptive_rescoring", True))
        self.adaptive_max_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_max_adjustment", 40))
        self.adaptive_floor_spin.setValue(_dflt(SCORING_DEFAULTS, "adaptive_min_above_floor", 30))
        self.auto_calibrate_cb.setChecked(_dflt(SCORING_DEFAULTS, "auto_calibrate_thresh", True))
        self.fixed_thresh_spin.setValue(_dflt(SCORING_DEFAULTS, "fixed_thresh", 180))

        # Alignment
        self.default_dpi.setValue(_dflt(ALIGN_DEFAULTS, "dpi", 150))
        self.default_align_method.setCurrentText("auto")
        self.min_aruco_spin.setValue(_dflt(ALIGN_DEFAULTS, "min_aruco", 4))
        self.fail_med_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_med", 3.0))
        self.fail_p95_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_p95", 8.0))
        self.fail_br_spin.setValue(_dflt(ALIGN_DEFAULTS, "fail_br", 8.0))

        # Estimation
        self.estimator_combo.setCurrentText(_dflt(EST_DEFAULTS, "estimator_method", "auto"))
        self.ransac_thresh_spin.setValue(_dflt(EST_DEFAULTS, "ransac_thresh", 3.0))
        self.max_iters_spin.setValue(_dflt(EST_DEFAULTS, "max_iters", 10000))
        self.use_ecc_cb.setChecked(_dflt(EST_DEFAULTS, "use_ecc", True))
        self.ecc_levels_spin.setValue(_dflt(EST_DEFAULTS, "ecc_levels", 4))

        # Features
        self.tiles_x_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_x", 8))
        self.tiles_y_spin.setValue(_dflt(FEAT_DEFAULTS, "tiles_y", 10))
        self.topk_spin.setValue(_dflt(FEAT_DEFAULTS, "topk_per_tile", 150))
        self.orb_nfeatures_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_nfeatures", 3000))
        self.orb_fast_spin.setValue(_dflt(FEAT_DEFAULTS, "orb_fast_threshold", 12))

        # Matching
        self.ratio_test_spin.setValue(_dflt(MATCH_DEFAULTS, "ratio_test", 0.75))
        self.mutual_check_cb.setChecked(_dflt(MATCH_DEFAULTS, "mutual_check", True))
        self.max_matches_spin.setValue(_dflt(MATCH_DEFAULTS, "max_matches", 5000))
        self.use_flann_cb.setChecked(_dflt(MATCH_DEFAULTS, "use_flann", False))

        # Rendering
        self.render_dpi_spin.setValue(_dflt(RENDER_DEFAULTS, "dpi", 150))
        self.image_format_combo.setCurrentText(_dflt(RENDER_DEFAULTS, "image_format", "png"))
        self.jpeg_quality_spin.setValue(_dflt(RENDER_DEFAULTS, "jpeg_quality", 85))
        self.pdf_quality_spin.setValue(_dflt(RENDER_DEFAULTS, "pdf_quality", 85))

        # Annotations
        _set_bgr(self.annot_color_correct, _dflt(ANNOTATION_DEFAULTS, "color_correct", (0, 200, 0)))
        _set_bgr(self.annot_color_incorrect, _dflt(ANNOTATION_DEFAULTS, "color_incorrect", (0, 0, 255)))
        _set_bgr(self.annot_color_blank, _dflt(ANNOTATION_DEFAULTS, "color_blank", (160, 160, 160)))
        _set_bgr(self.annot_color_multi, _dflt(ANNOTATION_DEFAULTS, "color_multi", (0, 140, 255)))
        _set_bgr(self.annot_color_blank_row, _dflt(ANNOTATION_DEFAULTS, "color_blank_answer_row", (255, 0, 255)))
        self.annot_thickness_answers.setValue(_dflt(ANNOTATION_DEFAULTS, "thickness_answers", 2))
        self.annot_thickness_names.setValue(_dflt(ANNOTATION_DEFAULTS, "thickness_names", 2))
        _set_bgr(self.annot_pct_color, _dflt(ANNOTATION_DEFAULTS, "pct_fill_font_color", (255, 0, 0)))
        self.annot_pct_scale.setValue(_dflt(ANNOTATION_DEFAULTS, "pct_fill_font_scale", 0.4))
        self.annot_pct_thickness.setValue(_dflt(ANNOTATION_DEFAULTS, "pct_fill_font_thickness", 1))
        self.annot_pct_position.setValue(_dflt(ANNOTATION_DEFAULTS, "pct_fill_font_position", 3))
        self.annot_label_scale.setValue(_dflt(ANNOTATION_DEFAULTS, "label_font_scale", 0.5))
        self.annot_label_thickness.setValue(_dflt(ANNOTATION_DEFAULTS, "label_thickness", 1))
        self.annot_box_multi_cb.setChecked(_dflt(ANNOTATION_DEFAULTS, "box_multi", True))
        self.annot_box_blank_cb.setChecked(_dflt(ANNOTATION_DEFAULTS, "box_blank_answer_row", True))
        self.annot_box_thickness.setValue(_dflt(ANNOTATION_DEFAULTS, "box_thickness", 2))
        self.annot_box_pad.setValue(_dflt(ANNOTATION_DEFAULTS, "box_pad", 4))

        self._save_settings()
