"""
MarkShark
bublmap_io.py
------------
Axis-based YAML bubblemap loader for MarkShark OMR.

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

Example YAML snippets:

answer_layouts:
  - x_topleft: 0.5931
    y_topleft: 0.0626
    x_bottomright: 0.9227
    y_bottomright: 0.2458
    radius_pct: 0.008
    numrows: 16
    numcols: 5
    bubble_shape: circle
    selection_axis: row      # pick A..E per question row
    labels: ABCDE            # optional (auto A..E if omitted)

last_name_layout:
  x_topleft: 0.1062
  y_topleft: 0.3591
  x_bottomright: 0.5690
  y_bottomright: 0.7693
  radius_pct: 0.008
  numrows: 27
  numcols: 14
  selection_axis: col       # pick a letter (row) per column (position)
  labels: " ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # rows = 27

id_layout:
  x_topleft: 0.3978
  y_topleft: 0.0370
  x_bottomright: 0.5690
  y_bottomright: 0.3255
  radius_pct: 0.008
  numrows: 10
  numcols: 10
  selection_axis: col
  labels: "0123456789"

version_layout:               # one row, 4 columns horizontally (A-D)
  x_topleft: 0.3080
  y_topleft: 0.2030
  x_bottomright: 0.3286
  y_bottomright: 0.3092
  radius_pct: 0.008
  numrows: 1
  numcols: 4
  selection_axis: row        # one row → pick one column
  labels: "ABCD"
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
class Bubblemap:
    """Top-level bubblemap configuration object."""
    answer_layouts: List[GridLayout]
    last_name_layout: GridLayout | None = None
    first_name_layout: GridLayout | None = None
    id_layout: GridLayout | None = None
    version_layout: GridLayout | None = None
    total_numrows: int | None = None


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
    # NEW: validate against the axis we select on
    if labels is not None:
        numrows = int(section["numrows"])
        numcols   = int(section["numcols"])
        expected  = numcols if selection_axis == "row" else numrows
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

def load_bublmap(path: str) -> Bubblemap:
    """Load and validate a Bubblemap YAML file."""
    import io
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    # Parse answer layouts
    answer_layouts_data = data.get("answer_layouts", [])
    answer_layouts: List[GridLayout] = []
    for i, block in enumerate(answer_layouts_data):
        # default labels for answers if omitted
        if "labels" not in block and "numcols" in block:
            ch = int(block["numcols"])
            block["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        if "selection_axis" not in block:
            block["selection_axis"] = "row"
        answer_layouts.append(_parse_layout(f"answers_{i+1}", block))

    bmap = Bubblemap(answer_layouts=answer_layouts)

    # Optional other layouts
    for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
        if opt_name in data:
            bmap_dict = dict(data[opt_name])  # shallow copy
            # sensible defaults if omitted
            if opt_name in ("last_name_layout", "first_name_layout"):
                bmap_dict.setdefault("selection_axis", "col")
                bmap_dict.setdefault("labels", " ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif opt_name == "id_layout":
                bmap_dict.setdefault("selection_axis", "col")
                bmap_dict.setdefault("labels", "0123456789")
            elif opt_name == "version_layout":
                bmap_dict.setdefault("selection_axis", "row")
                # if numcols present and labels omitted, auto ABCD...
                if "labels" not in bmap_dict and "numcols" in bmap_dict:
                    ch = int(bmap_dict["numcols"])
                    bmap_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
            setattr(bmap, opt_name, _parse_layout(opt_name, bmap_dict))

    bmap.total_numrows = data.get("total_numrows")
    return bmap


# ---------------------------------------------------------------------------

def dump_bublmap(bmap: Config, path: str) -> None:
    """Write a Config back to YAML (useful for debugging)."""
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

    data: Dict[str, Any] = {
        "answer_layouts": [layout_to_dict(a) for a in bmap.answer_layouts]
    }

    for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
        layout = getattr(bmap, opt_name)
        if layout is not None:
            data[opt_name] = layout_to_dict(layout)

    if bmap.total_numrows is not None:
        data["total_numrows"] = bmap.total_numrows

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)