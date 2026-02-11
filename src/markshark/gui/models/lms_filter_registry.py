"""
Persistent registry of saved LMS column-mapping filters.

Stores filter definitions in ~/.markshark/lms_filters.json so teachers
can reuse column mappings across classes and semesters.

Each filter records which columns in an LMS gradebook export correspond
to MarkShark properties (Student ID, Last Name, First Name).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


_DEFAULT_PATH = Path.home() / ".markshark" / "lms_filters.json"

_instances: dict[str, "LmsFilterRegistry"] = {}


class LmsFilterRegistry:
    """
    JSON-backed LMS filter registry (singleton per file path).

    Schema:
    {
        "version": 1,
        "filters": [
            {
                "name": "Canvas - BIO101",
                "student_id_col": "SIS User ID",
                "last_name_col": "Last Name",
                "first_name_col": "First Name",
                "delimiter": ",",
                "skip_rows": 0,
                "created_at": "2025-06-01T10:00:00",
                "last_used": "2025-06-15T08:30:00"
            }
        ]
    }
    """

    def __new__(cls, registry_path: Optional[Path] = None):
        resolved = str((registry_path or _DEFAULT_PATH).resolve())
        if resolved in _instances:
            return _instances[resolved]
        inst = super().__new__(cls)
        _instances[resolved] = inst
        return inst

    def __init__(self, registry_path: Optional[Path] = None):
        if hasattr(self, "_path"):
            return
        self._path = registry_path or _DEFAULT_PATH
        self._data: dict = {"version": 1, "filters": []}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_filter(
        self,
        name: str,
        student_id_col: str,
        last_name_col: str,
        first_name_col: str,
        delimiter: str = ",",
        skip_rows: int = 0,
        combined_name_col: str = "",
    ) -> bool:
        """Save or update a filter by name. Returns True on success."""
        if not name.strip():
            return False

        now = datetime.now().isoformat()

        # Update existing
        for entry in self._data["filters"]:
            if entry["name"] == name:
                entry["student_id_col"] = student_id_col
                entry["last_name_col"] = last_name_col
                entry["first_name_col"] = first_name_col
                entry["combined_name_col"] = combined_name_col
                entry["delimiter"] = delimiter
                entry["skip_rows"] = skip_rows
                entry["last_used"] = now
                self._save()
                return True

        # New entry
        self._data["filters"].append({
            "name": name,
            "student_id_col": student_id_col,
            "last_name_col": last_name_col,
            "first_name_col": first_name_col,
            "combined_name_col": combined_name_col,
            "delimiter": delimiter,
            "skip_rows": skip_rows,
            "created_at": now,
            "last_used": now,
        })
        self._save()
        return True

    def delete_filter(self, name: str) -> bool:
        """Remove a filter by name. Returns True if found."""
        before = len(self._data["filters"])
        self._data["filters"] = [
            f for f in self._data["filters"] if f["name"] != name
        ]
        if len(self._data["filters"]) < before:
            self._save()
            return True
        return False

    def get_filter(self, name: str) -> Optional[dict]:
        """Return a single filter by name, or None."""
        for entry in self._data["filters"]:
            if entry["name"] == name:
                return dict(entry)
        return None

    def list_all(self) -> list[dict]:
        """Return all saved filters (shallow copies)."""
        return [dict(f) for f in self._data["filters"]]

    def list_names(self) -> list[str]:
        """Return just the filter names, sorted alphabetically."""
        return sorted(f["name"] for f in self._data["filters"])

    def touch_last_used(self, name: str):
        """Update the last_used timestamp for a filter."""
        now = datetime.now().isoformat()
        for entry in self._data["filters"]:
            if entry["name"] == name:
                entry["last_used"] = now
                self._save()
                return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and "filters" in data:
                self._data = data
        except Exception:
            pass

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
