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

page_1:
  last_name_layout: { ... }
  first_name_layout: { ... }
  id_layout: { ... }
  version_layout: { ... }
  answer_layouts:
    - { ... }

page_2:
  # Optional layouts for validation/redundancy
  last_name_layout: { ... }  # Optional
  answer_layouts:
    - { ... }
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass
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
class PageLayout:
    """Layouts for a single page of the bubble sheet."""
    page_number: int
    answer_layouts: List[GridLayout]
    last_name_layout: GridLayout | None = None
    first_name_layout: GridLayout | None = None
    id_layout: GridLayout | None = None
    version_layout: GridLayout | None = None


@dataclass
class Bubblemap:
    """Top-level bubblemap configuration object with multi-page support."""
    pages: List[PageLayout]  # One PageLayout per page
    metadata: Dict[str, Any] | None = None
    total_questions: int | None = None
    
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
    
    # Backward compatibility properties for single-page sheets
    @property
    def answer_layouts(self) -> List[GridLayout]:
        """Get answer layouts from page 1 (for backward compatibility)."""
        if self.pages:
            return self.pages[0].answer_layouts
        return []
    
    @property
    def last_name_layout(self) -> GridLayout | None:
        """Get last_name_layout from page 1 (for backward compatibility)."""
        if self.pages:
            return self.pages[0].last_name_layout
        return None
    
    @property
    def first_name_layout(self) -> GridLayout | None:
        """Get first_name_layout from page 1 (for backward compatibility)."""
        if self.pages:
            return self.pages[0].first_name_layout
        return None
    
    @property
    def id_layout(self) -> GridLayout | None:
        """Get id_layout from page 1 (for backward compatibility)."""
        if self.pages:
            return self.pages[0].id_layout
        return None
    
    @property
    def version_layout(self) -> GridLayout | None:
        """Get version_layout from page 1 (for backward compatibility)."""
        if self.pages:
            return self.pages[0].version_layout
        return None


# ---------------------------------------------------------------------------

def _parse_layout(name: str, section: Dict[str, Any]) -> GridLayout:
    required = [
        "x_topleft", "y_topleft",
        "x_bottomright", "y_bottomright",
        "radius_pct", "numrows", "numcols",
    ]
    missing = [k for k in required if k not in section]
    if missing:
        raise ValueError(f"Layout '{name}' missing required fields: {missing}")

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
        x_topleft=float(section["x_topleft"]),
        y_topleft=float(section["y_topleft"]),
        x_bottomright=float(section["x_bottomright"]),
        y_bottomright=float(section["y_bottomright"]),
        radius_pct=float(section["radius_pct"]),
        numrows=int(section["numrows"]),
        numcols=int(section["numcols"]),
        bubble_shape=section.get("bubble_shape", "circle"),
        labels=labels,
        selection_axis=selection_axis,
    )


def _parse_page_layouts(page_num: int, page_data: Dict[str, Any]) -> PageLayout:
    """Parse layouts for a single page."""
    # Parse answer layouts
    answer_layouts_data = page_data.get("answer_layouts", [])
    answer_layouts: List[GridLayout] = []
    for i, block in enumerate(answer_layouts_data):
        # Default labels for answers if omitted
        if "labels" not in block and "numcols" in block:
            ch = int(block["numcols"])
            block["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        if "selection_axis" not in block:
            block["selection_axis"] = "row"
        answer_layouts.append(_parse_layout(f"page{page_num}_answers_{i+1}", block))

    page_layout = PageLayout(
        page_number=page_num,
        answer_layouts=answer_layouts
    )

    # Optional other layouts
    for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
        if opt_name in page_data:
            layout_dict = dict(page_data[opt_name])  # Shallow copy
            # Sensible defaults if omitted
            if opt_name in ("last_name_layout", "first_name_layout"):
                layout_dict.setdefault("selection_axis", "col")
                layout_dict.setdefault("labels", " ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif opt_name == "id_layout":
                layout_dict.setdefault("selection_axis", "col")
                layout_dict.setdefault("labels", "0123456789")
            elif opt_name == "version_layout":
                layout_dict.setdefault("selection_axis", "row")
                # If numcols present and labels omitted, auto ABCD...
                if "labels" not in layout_dict and "numcols" in layout_dict:
                    ch = int(layout_dict["numcols"])
                    layout_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
            setattr(page_layout, opt_name, _parse_layout(f"page{page_num}_{opt_name}", layout_dict))

    return page_layout


def load_bublmap(path: str) -> Bubblemap:
    """Load and validate a Bubblemap YAML file with multi-page support."""
    import io
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    
    # Extract metadata
    metadata = data.get("metadata", {})
    num_pages = metadata.get("pages", 1)
    total_questions = metadata.get("total_questions")
    
    # Parse pages
    pages: List[PageLayout] = []
    
    for page_num in range(1, num_pages + 1):
        page_key = f"page_{page_num}"
        
        if page_key not in data:
            raise ValueError(f"Missing '{page_key}' section in YAML (metadata says {num_pages} pages)")
        
        page_data = data[page_key]
        page_layout = _parse_page_layouts(page_num, page_data)
        pages.append(page_layout)
    
    bmap = Bubblemap(
        pages=pages,
        metadata=metadata,
        total_questions=total_questions
    )
    
    return bmap


# ---------------------------------------------------------------------------

def dump_bublmap(bmap: Bubblemap, path: str) -> None:
    """Write a Bubblemap back to YAML (useful for debugging)."""
    def layout_to_dict(gl: GridLayout) -> Dict[str, Any]:
        out = {
            "x_topleft": gl.x_topleft,
            "y_topleft": gl.y_topleft,
            "x_bottomright": gl.x_bottomright,
            "y_bottomright": gl.y_bottomright,
            "radius_pct": gl.radius_pct,
            "numrows": gl.numrows,
            "numcols": gl.numcols,
            "bubble_shape": gl.bubble_shape,
            "selection_axis": gl.selection_axis,
        }
        if gl.labels is not None:
            out["labels"] = gl.labels
        return out

    data: Dict[str, Any] = {}
    
    # Add metadata
    if bmap.metadata:
        data["metadata"] = bmap.metadata
    elif bmap.num_pages > 1:
        data["metadata"] = {
            "pages": bmap.num_pages,
            "total_questions": bmap.total_questions
        }
    
    # Add page layouts
    for page in bmap.pages:
        page_key = f"page_{page.page_number}"
        page_data: Dict[str, Any] = {
            "answer_layouts": [layout_to_dict(a) for a in page.answer_layouts]
        }
        
        for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
            layout = getattr(page, opt_name)
            if layout is not None:
                page_data[opt_name] = layout_to_dict(layout)
        
        data[page_key] = page_data
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
