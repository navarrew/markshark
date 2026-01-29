# MarkShark
# defaults.py
# Unified "single source of truth" for scoring + alignment tuning knobs.

# This is not a script you run separately.  It serves as a source of default
# values (as ScoringDefault objects, etc) and defines the functions needed to replace 
# specific values without needing to redefine all default values.  

# Having a single source of truth prevents all sorts of mismatches and 
# screwy things from happening across all the different markshark scripts.

# Import the defaults and functions from your tools like:
#   from markshark.defaults import SCORING_DEFAULTS, ALIGN_DEFAULTS, EST_DEFAULTS, FEAT_DEFAULTS, MATCH_DEFAULTS, RENDER_DEFAULTS
#   from markshark.defaults import apply_scoring_overrides, apply_align_overrides, apply_est_overrides, apply_feat_overrides, apply_match_overrides, apply_render_overrides

from dataclasses import dataclass, replace as _dc_replace
from typing import Optional, Literal

# ---------------------------
# Scoring thresholds
# ---------------------------
@dataclass(frozen=True)
class ScoringDefaults:
    # Thresholds for deciding filled area bubbles and resolving ties.
    # min_fill and top2_ratio are stored as integers (0-100) matching the UI.
    # They are converted to fractions (0-1) at the point of use in the scoring engine.
    min_fill: int = 45             # minimum fill score (0-100) to accept non-blank (matches annotated PDF scores)
    top2_ratio: int = 80           # second-best must be <= top2_ratio% of best (0-100)
    min_top2_diff: float = 10    # minimum difference between top 2 bubbles in percentage points to not score as multi

    # Background subtraction calibration
    calibrate_background: bool = True     # subtract per-column background to remove letter printing bias
    background_percentile: float = 10.0   # percentile to use for background calculation (10th = robust to noise)

    # Adaptive rescoring for light pencil marks
    adaptive_rescoring: bool = True               # enable adaptive threshold reduction for blank rows
    adaptive_max_adjustment: int = 40             # maximum threshold reduction to try (in steps of 10)
    adaptive_min_above_floor: float = 30          # winner must be this many points above lowest bubble (after calibration)

    # Global binarization threshold (only used when bin_method == "global") for gray pixels
    fixed_thresh: int = 180
    # Per-page auto calibration for lightly marked sheets
    auto_calibrate_thresh: bool = True
    verbose_calibration: bool = False
    #Default output names and directories for scoring.
    out_pdf: str = "scored_scans.pdf"
    out_pdf_dir: Optional[str] = None  # None means "same directory as out_csv"

    
SCORING_DEFAULTS = ScoringDefaults()

def apply_scoring_overrides(**kwargs) -> ScoringDefaults:
    """Return a copy of SCORING_DEFAULTS with the provided fields overridden."""
    return _dc_replace(SCORING_DEFAULTS, **kwargs)

def resolve_scored_pdf_path(out_pdf: str, out_csv: str, out_pdf_dir: Optional[str] = None) -> str:
    """Resolve the scored PDF path.

    Rules:
      - If out_pdf is absolute, return as-is.
      - If out_pdf is relative:
          - If out_pdf_dir is provided (not None), join out_pdf_dir/out_pdf.
          - Else, use SCORING_DEFAULTS.out_pdf_dir.
          - If that is None, default to dirname(out_csv).
    """
    import os
    if os.path.isabs(out_pdf):
        return out_pdf
    base_dir = out_pdf_dir if out_pdf_dir is not None else SCORING_DEFAULTS.out_pdf_dir
    if base_dir is None:
        base_dir = os.path.dirname(out_csv) or "."
    return os.path.normpath(os.path.join(base_dir, out_pdf))


# ---------------------------
# High-level alignment outputs / IO
# ---------------------------
@dataclass(frozen=True)
class AlignDefaults:
    """Top-level IO and run-mode toggles for alignment pipelines."""
    out: str = "aligned_scans.pdf"              # Output aligned PDF
    out_dir: str = "aligned_pngs"               # Where per-page aligned PNGs go (if exported)
    prefix: str = "aligned_"                    # Prefix for per-page PNG filenames
    metrics_csv: str = "alignment_metrics.csv"  # CSV to append per-page residual metrics
    dpi: int = 150                              # DPI for PDF rendering
    use_aruco: bool = True                      # Try ArUco-based coarse align first
    dict_name: str = "DICT_4X4_50"              # ArUco dictionary
    min_aruco: int = 4                          # Min detected markers to accept ArUco pose
    export_pngs: bool = False                   # Also export aligned PNGs
    overwrite: bool = True                      # Overwrite existing outputs if present
    verbose: bool = True                        # Print per-page status lines
    # Guardrails for non-ArUco feature alignment residuals (pixels)
    fail_med: float = 3.0
    fail_p95: float = 8.0
    fail_br: float = 8.0
    
ALIGN_DEFAULTS = AlignDefaults()

def apply_align_overrides(**kwargs) -> AlignDefaults:
    return _dc_replace(ALIGN_DEFAULTS, **kwargs)


# ---------------------------
# Geometry estimation / refinement
# ---------------------------
@dataclass(frozen=True)
class EstParams:
    """Homography estimation knobs (RANSAC/USAC) and optional ECC refinement."""
    estimator_method: Literal["auto", "ransac", "usac"] = "auto"  # 'auto' picks USAC if available
    ransac_thresh: float = 3.0              # px reprojection threshold
    max_iters: int = 10000                  # robust estimator iterations
    confidence: float = 0.999               # success probability for RANSAC/USAC
    # ECC (optional refinement after feature-based homography)
    use_ecc: bool = True
    ecc_levels: int = 4                     # pyramid levels (depends on OpenCV build)
    ecc_max_iters: int = 50                 # iterations per level
    ecc_eps: float = 1e-6                   # termination epsilon

EST_DEFAULTS = EstParams()

def apply_est_overrides(**kwargs) -> EstParams:
    return _dc_replace(EST_DEFAULTS, **kwargs)


# ---------------------------
# Feature detection / tiling
# ---------------------------
@dataclass(frozen=True)
class FeatureParams:
    """ORB + tiling parameters to spread keypoints across the page."""
    tiles_x: int = 8
    tiles_y: int = 10
    topk_per_tile: int = 150                # keep N strongest per tile
    orb_nfeatures: int = 3000               # global ORB budget (guard-rail)
    orb_fast_threshold: int = 12            # lower -> more keypoints, more noise
    orb_edge_threshold: int = 31            # OpenCV default
    clahe_clip_limit: Optional[float] = None  # e.g., 2.0 to enable contrast boosting
    clahe_tile_grid: int = 8                # square grid size for CLAHE if enabled

FEAT_DEFAULTS = FeatureParams()

def apply_feat_overrides(**kwargs) -> FeatureParams:
    return _dc_replace(FEAT_DEFAULTS, **kwargs)


# ---------------------------
# Matching
# ---------------------------
@dataclass(frozen=True)
class MatchParams:
    """Descriptor matching & filtering."""
    ratio_test: float = 0.75                # Lowe ratio test
    mutual_check: bool = True               # require A->B and B->A consistency
    max_matches: int = 5000                 # hard cap to keep compute in check
    use_flann: bool = False                 # BF by default; FLANN can help on large pages

MATCH_DEFAULTS = MatchParams()

def apply_match_overrides(**kwargs) -> MatchParams:
    return _dc_replace(MATCH_DEFAULTS, **kwargs)


# ---------------------------
# Rendering / export
# ---------------------------
@dataclass(frozen=True)
class RenderParams:
    """Rendering controls for intermediate images and PDF output."""
    dpi: int = 150                          # DPI for rendering (150 is good balance of quality/size)
    image_format: Literal["png", "jpg"] = "png"
    jpeg_quality: int = 85                  # if image_format='jpg'
    pdf_quality: int = 85                   # JPEG quality for PDF compression (1-100, higher=better quality)
    keep_intermediates: bool = False        # store pre/post align debug layers

RENDER_DEFAULTS = RenderParams()

def apply_render_overrides(**kwargs) -> RenderParams:
    return _dc_replace(RENDER_DEFAULTS, **kwargs)



# ---------------------------
# Annotation / drawing
# ---------------------------
from typing import Tuple

@dataclass(frozen=True)
class AnnotationDefaults:
    """Colors/thickness/font for drawn overlays (BGR order)."""
    # Name/ID zones
    color_zone: Tuple[int, int, int] = (255, 0, 0)        # blue circles for name/ID zones
    percent_text_color: Tuple[int, int, int] = (255, 0, 255)  # purple for % fill labels
    text_color: Tuple[int, int, int] = (255, 0, 0)      # alias used by older code paths
    thickness_names: int = 2
    label_font_scale: float = 0.5
    label_thickness: int = 1

    # Answer bubbles circles
    color_correct: Tuple[int, int, int] = (0, 200, 0)     # green
    color_incorrect: Tuple[int, int, int] = (0, 0, 255)   # red
    color_blank: Tuple[int, int, int] = (160, 160, 160)   # grey
    color_blank_answer_row: Tuple[int, int, int] = (255, 0, 255)   # purple
    color_multi: Tuple[int, int, int] = (0, 140, 255)     # orange
    thickness_answers: int = 2
    
    # Answer bubble percent filled annotations    
    pct_fill_font_color: Tuple[int, int, int] = (255, 0, 0)  # blue for %fill labels
    pct_fill_font_scale: float = 0.4
    pct_fill_font_thickness: int = 1
    
    #this will set the position of the %filled value on the annotated
    #pdf files.  0 puts the text directly on top of the bubble.  10 would
    #be 10 pixels above the bubble.  -20 would put the text within the bubble.
    pct_fill_font_position: int = 3

    # Answer row boxes and required-blank highlighting
    box_multi: bool = True
    box_blank_answer_row: bool = True
    box_color_multi: Tuple[int, int, int] = (0, 140, 255)        # orange
    box_color_blank_answer_row: Tuple[int, int, int] = (255, 0, 255)  # purple
    box_thickness: int = 2
    box_pad: int = 4
    box_top_extra: int = 7  # raise the top edge to avoid overwriting % labels

ANNOTATION_DEFAULTS = AnnotationDefaults()

def apply_annotation_overrides(**kwargs) -> AnnotationDefaults:
    return _dc_replace(ANNOTATION_DEFAULTS, **kwargs)
    
    
# ---------------------------
# Template management
# ---------------------------
@dataclass(frozen=True)
class TemplateDefaults:
    """Template directory and discovery settings."""
    # Default templates directory (can be overridden by MARKSHARK_TEMPLATES_DIR env var)
    templates_dir: Optional[str] = None  # None means auto-detect (package dir or cwd/templates)
    # Whether to auto-discover templates on startup
    auto_discover: bool = True

TEMPLATE_DEFAULTS = TemplateDefaults()

def apply_template_overrides(**kwargs) -> TemplateDefaults:
    return _dc_replace(TEMPLATE_DEFAULTS, **kwargs)


# ---------------------------
# Convenience: compile all knobs into a single object if desired
# ---------------------------
@dataclass(frozen=True)
class AllDefaults:
    annotation: AnnotationDefaults = ANNOTATION_DEFAULTS
    scoring: ScoringDefaults = SCORING_DEFAULTS
    align: AlignDefaults = ALIGN_DEFAULTS
    est: EstParams = EST_DEFAULTS
    feat: FeatureParams = FEAT_DEFAULTS
    matching: MatchParams = MATCH_DEFAULTS
    render: RenderParams = RENDER_DEFAULTS
    template: TemplateDefaults = TEMPLATE_DEFAULTS

ALL_DEFAULTS = AllDefaults()


__all__ = [
    "ScoringDefaults", "AlignDefaults", "EstParams", "FeatureParams", "MatchParams", "RenderParams", "AllDefaults", "AnnotationDefaults", "TemplateDefaults",
    "SCORING_DEFAULTS", "ALIGN_DEFAULTS", "EST_DEFAULTS", "FEAT_DEFAULTS", "MATCH_DEFAULTS", "RENDER_DEFAULTS", "ALL_DEFAULTS", "ANNOTATION_DEFAULTS", "TEMPLATE_DEFAULTS",
    "apply_scoring_overrides", "apply_align_overrides", "apply_est_overrides", "apply_feat_overrides", "apply_match_overrides", "apply_render_overrides",
    "apply_annotation_overrides", "apply_template_overrides", "resolve_scored_pdf_path",
]
