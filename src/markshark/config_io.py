"""
MarkShark
config_io.py
------------
Axis-based YAML configuration loader for MarkShark OMR.

Each bubble block in the config defines:
  x_topleft:      normalized X of the top-left bubble center   (0..1)
  y_topleft:      normalized Y of the top-left bubble center   (0..1)
  x_bottomright:  normalized X of the bottom-right bubble center
  y_bottomright:  normalized Y of the bottom-right bubble center
  radius_pct:     bubble radius as fraction of image width
  questions:      number of rows (vertical count)
  choices:        number of columns (horizontal count)
  bubble_shape:   "circle" (optional, default "circle")
  labels:         optional string giving the symbols in the ROWS (length == questions)
                  e.g., " ABCDEFGHIJKLMNOPQRSTUVWXYZ" for name rows,
                        "0123456789" for ID rows,
                        "ABCD" for version rows (if questions==4).
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
    questions: 16
    choices: 5
    bubble_shape: circle
    selection_axis: row      # pick A..E per question row
    labels: ABCDE            # optional (auto A..E if omitted)

last_name_layout:
  x_topleft: 0.1062
  y_topleft: 0.3591
  x_bottomright: 0.5690
  y_bottomright: 0.7693
  radius_pct: 0.008
  questions: 27
  choices: 14
  selection_axis: col       # pick a letter (row) per column (position)
  labels: " ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # rows = 27

id_layout:
  x_topleft: 0.3978
  y_topleft: 0.0370
  x_bottomright: 0.5690
  y_bottomright: 0.3255
  radius_pct: 0.008
  questions: 10
  choices: 10
  selection_axis: col
  labels: "0123456789"

version_layout:               # one row, 4 columns horizontally (A-D)
  x_topleft: 0.3080
  y_topleft: 0.2030
  x_bottomright: 0.3286
  y_bottomright: 0.3092
  radius_pct: 0.008
  questions: 1
  choices: 4
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
    questions: int
    choices: int
    bubble_shape: str = "circle"
    labels: Optional[str] = None       # symbols across ROWS (length == questions), optional
    selection_axis: str = "row"        # "row" or "col"


@dataclass
class Config:
    """Top-level configuration object."""
    answer_layouts: List[GridLayout]
    last_name_layout: GridLayout | None = None
    first_name_layout: GridLayout | None = None
    id_layout: GridLayout | None = None
    version_layout: GridLayout | None = None
    total_questions: int | None = None


# ---------------------------------------------------------------------------

def _parse_layout(name: str, section: Dict[str, Any]) -> GridLayout:
    required = [
        "x_topleft", "y_topleft",
        "x_bottomright", "y_bottomright",
        "radius_pct", "questions", "choices",
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
        questions = int(section["questions"])
        choices   = int(section["choices"])
        expected  = choices if selection_axis == "row" else questions
        if len(labels) != expected:
            raise ValueError(
                f"Layout '{name}': labels length ({len(labels)}) must equal "
                f"{'choices' if selection_axis=='row' else 'questions'} ({expected})."
            )

    return GridLayout(
        name=name,
        x_topleft=float(section["x_topleft"]),
        y_topleft=float(section["y_topleft"]),
        x_bottomright=float(section["x_bottomright"]),
        y_bottomright=float(section["y_bottomright"]),
        radius_pct=float(section["radius_pct"]),
        questions=int(section["questions"]),
        choices=int(section["choices"]),
        bubble_shape=section.get("bubble_shape", "circle"),
        labels=labels,
        selection_axis=selection_axis,
    )

def load_config(path: str) -> Config:
    """Load and validate a bubble-OMR configuration YAML file."""
    import io
    with io.open(path, "r", encoding="utf-8", errors="replace") as f:
        data = yaml.safe_load(f)
    # Parse answer layouts
    answer_layouts_data = data.get("answer_layouts", [])
    answer_layouts: List[GridLayout] = []
    for i, block in enumerate(answer_layouts_data):
        # default labels for answers if omitted
        if "labels" not in block and "choices" in block:
            ch = int(block["choices"])
            block["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
        if "selection_axis" not in block:
            block["selection_axis"] = "row"
        answer_layouts.append(_parse_layout(f"answers_{i+1}", block))

    cfg = Config(answer_layouts=answer_layouts)

    # Optional other layouts
    for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
        if opt_name in data:
            cfg_dict = dict(data[opt_name])  # shallow copy
            # sensible defaults if omitted
            if opt_name in ("last_name_layout", "first_name_layout"):
                cfg_dict.setdefault("selection_axis", "col")
                cfg_dict.setdefault("labels", " ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            elif opt_name == "id_layout":
                cfg_dict.setdefault("selection_axis", "col")
                cfg_dict.setdefault("labels", "0123456789")
            elif opt_name == "version_layout":
                cfg_dict.setdefault("selection_axis", "row")
                # if choices present and labels omitted, auto ABCD...
                if "labels" not in cfg_dict and "choices" in cfg_dict:
                    ch = int(cfg_dict["choices"])
                    cfg_dict["labels"] = "".join(chr(ord("A") + k) for k in range(ch))
            setattr(cfg, opt_name, _parse_layout(opt_name, cfg_dict))

    cfg.total_questions = data.get("total_questions")
    return cfg


# ---------------------------------------------------------------------------

def dump_config(cfg: Config, path: str) -> None:
    """Write a Config back to YAML (useful for debugging)."""
    def layout_to_dict(gl: GridLayout) -> Dict[str, Any]:
        out = {
            "x_topleft": gl.x_topleft,
            "y_topleft": gl.y_topleft,
            "x_bottomright": gl.x_bottomright,
            "y_bottomright": gl.y_bottomright,
            "radius_pct": gl.radius_pct,
            "questions": gl.questions,
            "choices": gl.choices,
            "bubble_shape": gl.bubble_shape,
            "selection_axis": gl.selection_axis,
        }
        if gl.labels is not None:
            out["labels"] = gl.labels
        return out

    data: Dict[str, Any] = {
        "answer_layouts": [layout_to_dict(a) for a in cfg.answer_layouts]
    }

    for opt_name in ["last_name_layout", "first_name_layout", "id_layout", "version_layout"]:
        layout = getattr(cfg, opt_name)
        if layout is not None:
            data[opt_name] = layout_to_dict(layout)

    if cfg.total_questions is not None:
        data["total_questions"] = cfg.total_questions

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)