"""
Persistent registry of known MarkShark projects.

Stores a list of project directory paths (with timestamps) in
~/.markshark/projects.json so the Project Manager page can display
projects across different working directories.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


_DEFAULT_REGISTRY_PATH = Path.home() / ".markshark" / "projects.json"

# Singleton instances keyed by resolved registry path, so every
# ProjectRegistry() that points at the same file shares one in-memory
# copy.  This prevents stale writes from clobbering data saved by
# another instance (e.g. a ProjectSelector overwriting a description
# set by the ProjectManager).
_instances: dict[str, "ProjectRegistry"] = {}


class ProjectRegistry:
    """
    JSON-backed project registry (singleton per file path).

    Schema (version-stamped for future migration):
    {
        "version": 1,
        "projects": [
            {
                "path": "/absolute/path/to/project",
                "name": "BIO101_Final",
                "registered_at": "2025-01-21T14:30:00",
                "last_opened": "2025-01-21T15:45:00"
            }
        ]
    }
    """

    def __new__(cls, registry_path: Optional[Path] = None):
        resolved = str((registry_path or _DEFAULT_REGISTRY_PATH).resolve())
        if resolved in _instances:
            return _instances[resolved]
        inst = super().__new__(cls)
        _instances[resolved] = inst
        return inst

    def __init__(self, registry_path: Optional[Path] = None):
        # Only initialise once (singleton)
        if hasattr(self, '_path'):
            return
        self._path = registry_path or _DEFAULT_REGISTRY_PATH
        self._data: dict = {"version": 1, "projects": []}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, project_path: Path, description: str = "") -> bool:
        """
        Register a project (or update its last_opened if already known).

        Returns True if successfully added/updated, False if path is invalid.
        """
        resolved = self._normalize(project_path)
        if resolved is None:
            return False

        now = datetime.now().isoformat()

        for entry in self._data["projects"]:
            if entry["path"] == str(resolved):
                entry["last_opened"] = now
                entry["name"] = resolved.name
                # Only update description if a non-empty one is provided
                if description:
                    entry["description"] = description
                # Ensure description key exists
                entry.setdefault("description", "")
                self._save()
                return True

        self._data["projects"].append({
            "path": str(resolved),
            "name": resolved.name,
            "description": description,
            "registered_at": now,
            "last_opened": now,
        })
        self._save()
        return True

    def unregister(self, project_path: Path) -> bool:
        """Remove a project from the registry. Returns True if found."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return False

        before = len(self._data["projects"])
        self._data["projects"] = [
            e for e in self._data["projects"]
            if e["path"] != str(resolved)
        ]
        if len(self._data["projects"]) < before:
            self._save()
            return True
        return False

    def list_all(self) -> list[dict]:
        """Return all registered projects (shallow copies)."""
        return [dict(e) for e in self._data["projects"]]

    def set_description(self, project_path: Path, description: str):
        """Update the description for a registered project."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return
        for entry in self._data["projects"]:
            if entry["path"] == str(resolved):
                entry["description"] = description
                self._save()
                return

    def update_last_opened(self, project_path: Path):
        """Touch the last_opened timestamp for a project."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return
        now = datetime.now().isoformat()
        for entry in self._data["projects"]:
            if entry["path"] == str(resolved):
                entry["last_opened"] = now
                self._save()
                return

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load the registry from disk."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and "projects" in data:
                self._data = data
        except Exception:
            # Corrupted file — start fresh
            pass

    def _save(self):
        """Write the registry to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(path: Path) -> Optional[Path]:
        """Resolve to an absolute path. Returns None for empty/invalid."""
        try:
            return Path(path).resolve()
        except Exception:
            return None
