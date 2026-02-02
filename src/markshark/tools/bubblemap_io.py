"""
MarkShark
bubblemap_io.py
------------
Axis-based YAML bubblemap loader for MarkShark OMR with multi-page support.

Each bubble block in the bubblemap defines:
  x_topleft:      normalized X of the top-left bubble center   (0..1)
  y_topleft:      normalized Y of the top-left bubble center   (0..1)
  x_bottomright:  normalized X of the bottom-right bubble center
  y_bottomright:  normalized Y of the bottom-right bubble center
  radius_pct:     bubble radius as fraction of image width
  numrows:      number of rows (vertical count)
  numcols:        number of columns (horizontal count)
  bubble_shape:   "circle" (optional, default "circle")
  labels:         optional string giving the symbols in the ROWS (length == numrows)
                  e.g., " ABCDEFGHIJKLMNOPQRSTUVWXYZ" for name rows,
                        "0123456789" for ID rows,
                        "ABCD" for version rows (if numrows==4).
  selection_axis: "row" or "col"
                  - "row": select one column per row (answers, version-as-row)
                  - "col": select one row per column (names, ID)

Multi-page YAML structure:

metadata:
  display_name: "Template Name"
  description: "Description"
  pages: 2  # Number of pages
  total_questions: 128

registration:  # NEW: Alignment configuration
  primary_method: "aruco"
  fallback_methods: ["bubble_grid", "orb_features"]
  aruco:
    enabled: true
    dictionary: "DICT_4X4_50"
    min_markers: 4
  bubble_grid:
    enabled: true
    ransac_threshold: 5.0
    min_inliers: 30

page_1:
  last_name_zone: { ... }
  first_name_zone: { ... }
  id_zone: { ... }
  test_id_zone: { ... }  # Optional: numerical test ID (distinct from version)
  version_zone: { ... }  # Test version (A, B, C, D, etc.)
  answer_zones:
    - { ... }

page_2:
  # Optional zones for validation/redundancy
  last_name_zone: { ... }  # Optional
  answer_zones:
    - { ... }

Note: For backward compatibility, *_layout keys are still accepted (e.g., last_name_layout).
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class GridLayout:
    """Defines a single bubble grid (answers, ID, names, version)."""
    name: str
    x_topleft: float
    y_topleft: float
    x_bottomright: float
    y_bottomright: float
    radius_pct: float
    numrows: int
    numcols: int
    bubble_shape: str = "circle"
    labels: Optional[str] = None       # symbols across ROWS (length == numrows), optional
    selection_axis: str = "row"        # "row" or "col"


@dataclass
class ArUcoConfig:
    """ArUco marker alignment configuration."""
    enabled: bool = False
    dictionary: str = "DICT_4X4_50"
    min_markers: int = 4
    ransac_threshold: float = 3.0
    markers: Optional[List[Dict[str, Any]]] = None  # Optional explicit marker positions


@dataclass
class BubbleGridConfig:
    """Bubble grid alignment configuration."""
    enabled: bool = True
    alignment_zones: Optional[List[str]] = None  # Which zones to use for alignment
    hough_param1: int = 50
    hough_param2: int = 25
    ransac_threshold: float = 5.0
    min_inliers: int = 30


@dataclass
class CornerMarksConfig:
    """Corner registration marks configuration."""
    enabled: bool = False
    type: str = "filled_square"  # "filled_square", "L_bracket", "crosshair", "circle"
    size_pct: float = 0.015
    positions: Optional[List[Dict[str, Any]]] = None


@dataclass
class OrbFeaturesConfig:
    """ORB feature alignment configuration."""
    enabled: bool = True
    exclude_zones: Optional[List[str]] = None  # Zones to exclude from feature detection
    orb_nfeatures: int = 2000
    match_ratio: float = 0.75


@dataclass
class RegistrationConfig:
    """Alignment/registration configuration."""
    primary_method: str = "aruco"
    fallback_methods: List[str] = field(default_factory=lambda: ["bubble_grid", "orb_features"])
    aruco: ArUcoConfig = field(default_factory=ArUcoConfig)
    bubble_grid: BubbleGridConfig = field(default_factory=BubbleGridConfig)
    corner_marks: CornerMarksConfig = field(default_factory=CornerMarksConfig)
    orb_features: OrbFeaturesConfig = field(default_factory=OrbFeaturesConfig)


@dataclass
class PageLayout:
    """Layouts for a single page of the bubble sheet."""
    page_number: int
    answer_zones: List[GridLayout]
    last_name_zone: GridLayout | None = None
    first_name_zone: GridLayout | None = None
    id_zone: GridLayout | None = None
    test_id_zone: GridLayout | None = None  # Numerical test ID (distinct from version)
    version_zone: GridLayout | None = None  # Test version (A, B, C, D, etc.)

    # Backward compatibility aliases
    @property
    def answer_layouts(self) -> List[GridLayout]:
        """Alias for answer_zones (backward compatibility)."""
        return self.answer_zones

    @property
    def last_name_layout(self) -> GridLayout | None:
        """Alias for last_name_zone (backward compatibility)."""
        return self.last_name_zone

    @property
    def first_name_layout(self) -> GridLayout | None:
        """Alias for first_name_zone (backward compatibility)."""
        return self.first_name_zone

    @property
    def id_layout(self) -> GridLayout | None:
        """Alias for id_zone (backward compatibility)."""
        return self.id_zone

    @property
    def version_layout(self) -> GridLayout | None:
        """Alias for version_zone (backward compatibility)."""
        return self.version_zone

    @property
    def version_zones(self) -> List[GridLayout]:
        """Backward compatibility: return version_zone as a list."""
        return [self.version_zone] if self.version_zone else []


@dataclass
class Bubblemap:
    """Top-level bubblemap configuration object with multi-page support."""
    pages: List[PageLayout]  # One PageLayout per page
    metadata: Dict[str, Any] | None = None
    total_questions: int | None = None
    registration: RegistrationConfig | None = None  # NEW: Registration config
    
    @property
    def num_pages(self) -> int:
        """Number of pages in this bubble sheet."""
        return len(self.pages)
    
    def get_page(self, page_num: int) -> PageLayout | None:
        """Get layouts for a specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_num:
                return page
        return None
    
    def get_aruco_config(self) -> ArUcoConfig:
        """Get ArUco configuration, with fallback to metadata for backward compat."""
        if self.registration is not None:
            return self.registration.aruco
        
        # Backward compatibility: check metadata.alignment.aruco
        if self.metadata and 'alignment' in self.metadata:
            align = self.metadata['alignment']
            if 'aruco' in align:
                aruco_data = align['aruco']
                return ArUcoConfig(
                    enabled=aruco_data.get('enabled', False),
                    dictionary=aruco_data.get('dictionary', 'DICT_4X4_50'),
                    min_markers=aruco_data.get('min_markers', 4),
                )
        
        return ArUcoConfig()  # Default
    
    def get_bubble_grid_config(self) -> BubbleGridConfig:
        """Get bubble grid alignment configuration."""
        if self.registration is not None:
            return self.registration.bubble_grid
        return BubbleGridConfig()  # Default
    
    def should_use_aruco(self) -> bool:
        """Check if ArUco alignment should be attempted."""
        config = self.get_aruco_config()
        return config.enabled and config.dictionary.upper() not in ("NONE", "")
    
    def should_use_bubble_grid(self) -> bool:
        """Check if bubble grid alignment should be attempted."""
        config = self.get_bubble_grid_config()
        return config.enabled
    
    # Preferred accessors using *_zone naming
    @property
    def answer_zones(self) -> List[GridLayout]:
        """Get answer zones from page 1."""
        if self.pages:
            return self.pages[0].answer_zones
        return []

    @property
    def last_name_zone(self) -> GridLayout | None:
        """Get last_name_zone from page 1."""
        if self.pages:
            return self.pages[0].last_name_zone
        return None

    @property
    def first_name_zone(self) -> GridLayout | None:
        """Get first_name_zone from page 1."""
        if self.pages:
            return self.pages[0].first_name_zone
        return None

    @property
    def id_zone(self) -> GridLayout | None:
        """Get id_zone from page 1."""
        if self.pages:
            return self.pages[0].id_zone
        return None

    @property
    def test_id_zone(self) -> GridLayout | None:
        """Get test_id_zone from page 1."""
        if self.pages:
            return self.pages[0].test_id_zone
        return None

    @property
    def version_zone(self) -> GridLayout | None:
        """Get version_zone from page 1."""
        if self.pages:
            return self.pages[0].version_zone
        return None

    @property
    def version_zones(self) -> List[GridLayout]:
        """Get version_zone as list from page 1 (backward compatibility)."""
        if self.pages:
            return self.pages[0].version_zones
        return []

    # Backward compatibility properties for single-page sheets
    @property
    def answer_layouts(self) -> List[GridLayout]:
        """Get answer zones from page 1 (backward compatibility alias)."""
        return self.answer_zones

    @property
    def last_name_layout(self) -> GridLayout | None:
        """Get last_name_zone from page 1 (backward compatibility alias)."""
        return self.last_name_zone

    @property
    def first_name_layout(self) -> GridLayout | None:
        """Get first_name_zone from page 1 (backward compatibility alias)."""
        return self.first_name_zone

    @property
    def id_layout(self) -> GridLayout | None:
        """Get id_zone from page 1 (backward compatibility alias)."""
        return self.id_zone

    @property
    def version_layout(self) -> GridLayout | None:
        """Get version_zone from page 1 (backward compatibility alias)."""
        return self.version_zone


# ---------------------------------------------------------------------------

def _parse_layout(name: str, section: Dict[str, Any], page_size_mm: tuple = (215.9, 279.4)) -> GridLayout:
    """
    Parse a layout section, supporting both v1 (percentage) and v3 (mm) coordinate formats.

    v1 format: x_topleft, y_topleft, x_bottomright, y_bottomright, radius_pct (0.0-1.0)
    v3 format: x_mm, y_mm, width_mm, height_mm, bubble_diameter_mm (millimeters)

    Internally converts v3 to v1 percentages for consistent processing.
    """
    # Check which format we have
    has_v1 = "x_topleft" in section
    has_v3 = "x_mm" in section

    if has_v3:
        # Convert v3 mm coordinates to v1 percentages
        page_width_mm, page_height_mm = page_size_mm

        x_mm = float(section["x_mm"])
        y_mm = float(section["y_mm"])
        width_mm = float(section.get("width_mm", 0))
        height_mm = float(section.get("height_mm", 0))
        diameter_mm = float(section.get("bubble_diameter_mm", 4.0))

        # Convert to percentages (0.0-1.0)
        x_topleft = x_mm / page_width_mm
        y_topleft = y_mm / page_height_mm
        x_bottomright = (x_mm + width_mm) / page_width_mm
        y_bottomright = (y_mm + height_mm) / page_height_mm
        radius_pct = (diameter_mm / 2) / page_width_mm

    elif has_v1:
        # Use v1 format directly
        required = [
            "x_topleft", "y_topleft",
            "x_bottomright", "y_bottomright",
            "radius_pct", "numrows", "numcols",
        ]
        missing = [k for k in required if k not in section]
        if missing:
            raise ValueError(f"Layout '{name}' missing required fields: {missing}")

        x_topleft = float(section["x_topleft"])
        y_topleft = float(section["y_topleft"])
        x_bottomright = float(section["x_bottomright"])
        y_bottomright = float(section["y_bottomright"])
        radius_pct = float(section["radius_pct"])
    else:
        raise ValueError(
            f"Layout '{name}' must have either v1 fields (x_topleft, y_topleft, etc.) "
            f"or v3 fields (x_mm, y_mm, width_mm, height_mm)"
        )

    # Common validation
    if "numrows" not in section or "numcols" not in section:
        raise ValueError(f"Layout '{name}' missing required fields: numrows, numcols")

    selection_axis = section.get("selection_axis", "row").lower()
    if selection_axis not in ("row", "col"):
        raise ValueError(f"Layout '{name}': selection_axis must be 'row' or 'col'.")

    labels = section.get("labels")
    # Validate labels length against the appropriate axis
    if labels is not None:
        numrows = int(section["numrows"])
        numcols = int(section["numcols"])
        expected = numcols if selection_axis == "row" else numrows
        if len(labels) != expected:
            raise ValueError(
                f"Layout '{name}': labels length ({len(labels)}) must equal "
                f"{'numcols' if selection_axis=='row' else 'numrows'} ({expected})."
            )

    return GridLayout(
        name=name,
        x_topleft=x_topleft,
        y_topleft=y_topleft,
        x_bottomright=x_bottomright,
        y_bottomright=y_bottomright,
        radius_pct=radius_pct,
        numrows=int(section["numrows"]),
        numcols=int(section["numcols"]),
        bubble_shape=section.get("bubble_shape", "circle"),
        labels=labels,
        selection_axis=selection_axis,
    )


def _parse_page_layouts(page_num: int, page_data: Dict[str, Any],
                        page_size_mm: tuple = (215.9, 279.4)) -> PageLayout:
    """Parse layouts for a single page.

    Accepts both *_zone (preferred) and *_layout (backward compatibility) keys.
    """
    # Parse answer zones (accept both answer_zones and answer_layouts)
    answer_zones_data = page_data.get("answer_zones") or page_data.get("answer_layouts", [])
    answer_zones: List[GridLayout] = []
    for i, block in enumerate(answer_zones_data):
        # Default labels for answers if omitted
        if "labels" not in block and "numcols" in block:
            ch = int(block["numcols"])
            block["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        if "selection_axis" not in block:
            block["selection_axis"] = "row"
        answer_zones.append(_parse_layout(f"page{page_num}_answers_{i+1}", block, page_size_mm))

    # Parse version zone (accept version_zone, version_zones list, or version_layout)
    version_zone: GridLayout | None = None
    version_zone_data = page_data.get("version_zone")
    version_zones_data = page_data.get("version_zones")  # Legacy: list format
    version_layout_data = page_data.get("version_layout")  # Legacy: old naming

    if version_zone_data:
        # Preferred format: single version_zone dict
        layout_dict = dict(version_zone_data)
        layout_dict.setdefault("selection_axis", "row")
        if "labels" not in layout_dict and "numcols" in layout_dict:
            ch = int(layout_dict["numcols"])
            layout_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        version_zone = _parse_layout(f"page{page_num}_version", layout_dict, page_size_mm)
    elif version_zones_data:
        # Legacy format: list of version zones (use first one)
        if version_zones_data:
            layout_dict = dict(version_zones_data[0])
            layout_dict.setdefault("selection_axis", "row")
            if "labels" not in layout_dict and "numcols" in layout_dict:
                ch = int(layout_dict["numcols"])
                layout_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
            version_zone = _parse_layout(f"page{page_num}_version", layout_dict, page_size_mm)
    elif version_layout_data:
        # Backward compatibility: single version layout (old naming)
        layout_dict = dict(version_layout_data)
        layout_dict.setdefault("selection_axis", "row")
        if "labels" not in layout_dict and "numcols" in layout_dict:
            ch = int(layout_dict["numcols"])
            layout_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        version_zone = _parse_layout(f"page{page_num}_version", layout_dict, page_size_mm)

    page_layout = PageLayout(
        page_number=page_num,
        answer_zones=answer_zones,
        version_zone=version_zone
    )

    # Optional other zones (accept both *_zone and *_layout for backward compatibility)
    zone_mappings = [
        ("last_name_zone", "last_name_layout"),
        ("first_name_zone", "first_name_layout"),
        ("id_zone", "id_layout"),
        ("test_id_zone", None),  # New zone type, no legacy name
    ]

    for zone_name, layout_name in zone_mappings:
        # Prefer *_zone, fall back to *_layout if available
        zone_data = page_data.get(zone_name)
        if not zone_data and layout_name:
            zone_data = page_data.get(layout_name)
        if zone_data:
            layout_dict = dict(zone_data)  # Shallow copy
            # Sensible defaults if omitted
            if zone_name in ("last_name_zone", "first_name_zone"):
                layout_dict.setdefault("selection_axis", "col")
                layout_dict.setdefault("labels", " ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif zone_name in ("id_zone", "test_id_zone"):
                layout_dict.setdefault("selection_axis", "col")
                layout_dict.setdefault("labels", "0123456789")
            setattr(page_layout, zone_name, _parse_layout(f"page{page_num}_{zone_name}", layout_dict, page_size_mm))

    return page_layout


def _parse_registration_config(data: Dict[str, Any]) -> Optional[RegistrationConfig]:
    """Parse the registration configuration section."""
    if 'registration' not in data:
        return None
    
    reg_data = data['registration']
    
    # Parse ArUco config
    aruco_config = ArUcoConfig()
    if 'aruco' in reg_data:
        aruco_data = reg_data['aruco']
        aruco_config = ArUcoConfig(
            enabled=aruco_data.get('enabled', False),
            dictionary=aruco_data.get('dictionary', 'DICT_4X4_50'),
            min_markers=aruco_data.get('min_markers', 4),
            ransac_threshold=aruco_data.get('ransac_threshold', 3.0),
            markers=aruco_data.get('markers'),
        )
    
    # Parse bubble grid config
    bubble_grid_config = BubbleGridConfig()
    if 'bubble_grid' in reg_data:
        bg_data = reg_data['bubble_grid']
        bubble_grid_config = BubbleGridConfig(
            enabled=bg_data.get('enabled', True),
            alignment_zones=bg_data.get('alignment_zones'),
            hough_param1=bg_data.get('hough_param1', 50),
            hough_param2=bg_data.get('hough_param2', 25),
            ransac_threshold=bg_data.get('ransac_threshold', 5.0),
            min_inliers=bg_data.get('min_inliers', 30),
        )
    
    # Parse corner marks config
    corner_marks_config = CornerMarksConfig()
    if 'corner_marks' in reg_data:
        cm_data = reg_data['corner_marks']
        corner_marks_config = CornerMarksConfig(
            enabled=cm_data.get('enabled', False),
            type=cm_data.get('type', 'filled_square'),
            size_pct=cm_data.get('size_pct', 0.015),
            positions=cm_data.get('positions'),
        )
    
    # Parse ORB features config
    orb_config = OrbFeaturesConfig()
    if 'orb_features' in reg_data:
        orb_data = reg_data['orb_features']
        orb_config = OrbFeaturesConfig(
            enabled=orb_data.get('enabled', True),
            exclude_zones=orb_data.get('exclude_zones'),
            orb_nfeatures=orb_data.get('orb_nfeatures', 2000),
            match_ratio=orb_data.get('match_ratio', 0.75),
        )
    
    return RegistrationConfig(
        primary_method=reg_data.get('primary_method', 'aruco'),
        fallback_methods=reg_data.get('fallback_methods', ['bubble_grid', 'orb_features']),
        aruco=aruco_config,
        bubble_grid=bubble_grid_config,
        corner_marks=corner_marks_config,
        orb_features=orb_config,
    )


def _get_page_size_mm(metadata: Dict[str, Any]) -> tuple:
    """Extract page size in mm from metadata, with defaults."""
    # Standard page sizes in mm
    PAGE_SIZES_MM = {
        "letter": (215.9, 279.4),
        "a4": (210.0, 297.0),
        "legal": (215.9, 355.6),
        "a3": (297.0, 420.0),
    }

    page_size = metadata.get("page_size", "letter")

    if isinstance(page_size, dict):
        # Custom size specified as dict
        return (
            page_size.get("width_mm", 215.9),
            page_size.get("height_mm", 279.4)
        )
    elif isinstance(page_size, str):
        return PAGE_SIZES_MM.get(page_size.lower(), PAGE_SIZES_MM["letter"])
    else:
        return PAGE_SIZES_MM["letter"]


def load_bublmap(path: str) -> Bubblemap:
    """Load and validate a Bubblemap YAML file with multi-page support."""
    import io
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)

    # Extract metadata
    metadata = data.get("metadata", {})
    num_pages = metadata.get("pages", 1)
    total_questions = metadata.get("total_questions")

    # Get page size for v3 coordinate conversion
    page_size_mm = _get_page_size_mm(metadata)

    # Parse registration config (NEW)
    registration = _parse_registration_config(data)

    # Parse pages
    pages: List[PageLayout] = []

    for page_num in range(1, num_pages + 1):
        page_key = f"page_{page_num}"

        if page_key not in data:
            raise ValueError(f"Missing '{page_key}' section in YAML (metadata says {num_pages} pages)")

        page_data = data[page_key]
        page_layout = _parse_page_layouts(page_num, page_data, page_size_mm)
        pages.append(page_layout)

    bmap = Bubblemap(
        pages=pages,
        metadata=metadata,
        total_questions=total_questions,
        registration=registration,
    )

    return bmap
