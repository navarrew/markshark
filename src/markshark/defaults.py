
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
    """Thresholds for deciding filled bubbles and resolving ties."""
    min_fill: float = 0.20         # minimum filled fraction to accept non-blank
    top2_ratio: float = 0.80       # second-best must be <= top2_ratio * best
    min_score: float = 10.0        # absolute gap in percentage points (100*(best-second))
    min_abs: float = 0.10          # absolute minimum fill guard in _pick_single_from_scores

SCORING_DEFAULTS = ScoringDefaults()

def apply_scoring_overrides(**kwargs) -> ScoringDefaults:
    """Return a copy of SCORING_DEFAULTS with the provided fields overridden."""
    return _dc_replace(SCORING_DEFAULTS, **kwargs)

"""
_dc_replace() (aliased from dataclasses.replace) creates a copy of SCORING_DEFAULTS 
    with specific fields replaced.  The two asterisks in the function definition signal
    the program to 'pack' keyword arguments into a dictionary.  In the function call
    the two asterisks in front of kwargs serve to 'unpack' the dictionary.
    
kwargs is a standard convential notation for 'keyword arguments'.  
"""

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
    dpi: int = 300                              # DPI for PDF rendering
    use_aruco: bool = True                      # Try ArUco-based coarse align first
    dict_name: str = "DICT_4X4_50"              # ArUco dictionary
    min_aruco: int = 4                          # Min detected markers to accept ArUco pose
    export_pngs: bool = False                   # Also export aligned PNGs
    overwrite: bool = True                      # Overwrite existing outputs if present
    verbose: bool = True                        # Print per-page status lines

ALIGN_DEFAULTS = AlignDefaults()

def apply_align_overrides(**kwargs) -> AlignDefaults:
    return _dc_replace(ALIGN_DEFAULTS, **kwargs)

"""
_dc_replace() (aliased from dataclasses.replace) creates a copy of ALIGN_DEFAULTS 
    with specific fields replaced.
"""

# ---------------------------
# Geometry estimation / refinement
# ---------------------------
@dataclass(frozen=True)
class EstParams:
    """Homography estimation knobs (RANSAC/USAC) and optional ECC refinement."""
    method: Literal["auto", "ransac", "usac"] = "auto"  # 'auto' picks USAC if available
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

"""
_dc_replace() (aliased from dataclasses.replace) creates a copy of EST_DEFAULTS 
    with specific variables replaced.  
"""

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
    """Rendering controls for intermediate images."""
    dpi: int = 300
    image_format: Literal["png", "jpg"] = "png"
    jpeg_quality: int = 85                  # if image_format='jpg'
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
    percent_text_color: Tuple[int, int, int] = (255, 0, 255)  # magenta for % labels
    text_color: Tuple[int, int, int] = (255, 0, 255)      # alias used by older code paths
    thickness_names: int = 2
    label_font_scale: float = 0.4
    label_thickness: int = 1

    # Answer bubbles
    color_correct: Tuple[int, int, int] = (0, 200, 0)     # green
    color_incorrect: Tuple[int, int, int] = (0, 0, 255)   # red
    color_blank: Tuple[int, int, int] = (160, 160, 160)   # grey
    color_multi: Tuple[int, int, int] = (0, 140, 255)     # orange
    thickness_answers: int = 2

ANNOTATION_DEFAULTS = AnnotationDefaults()

def apply_annotation_overrides(**kwargs) -> AnnotationDefaults:
    return _dc_replace(ANNOTATION_DEFAULTS, **kwargs)
    
    
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

ALL_DEFAULTS = AllDefaults()


__all__ = [
    "ScoringDefaults", "AlignDefaults", "EstParams", "FeatureParams", "MatchParams", "RenderParams", "AllDefaults", "AnnotationDefaults",
    "SCORING_DEFAULTS", "ALIGN_DEFAULTS", "EST_DEFAULTS", "FEAT_DEFAULTS", "MATCH_DEFAULTS", "RENDER_DEFAULTS", "ALL_DEFAULTS", "ANNOTATION_DEFAULTS",
    "apply_scoring_overrides", "apply_align_overrides", "apply_est_overrides", "apply_feat_overrides", "apply_match_overrides", "apply_render_overrides", "apply_annotation_overrides",
]

"""
USAGE:
__all__ is a special module-level variable (a plain Python list of strings) that tells Python:

“When someone does from module import *, only export these names.”

This allows you to keep some functions 'hidden' from general importing.  Such functions 
or objects are typically internal to the script and not meant for use outside of it.
"""