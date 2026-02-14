"""
Persistent registry of known MarkShark projects and course folders.

Stores projects and course folders in ~/.markshark/projects.json so
the Course Manager can display assessments grouped by course.

Schema v2 adds a ``courses`` list alongside the existing ``projects``
list.  v1 files are auto-migrated on first load by deriving courses
from unique project parent directories.
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
    JSON-backed project and course registry (singleton per file path).

    Schema v2 (version-stamped for future migration):
    {
        "version": 2,
        "courses": [
            {
                "path": "/absolute/path/to/course_folder",
                "name": "Biology 101",
                "description": "",
                "registered_at": "2025-01-15T10:00:00",
                "last_opened": "2025-01-21T15:45:00"
            }
        ],
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
        self._data: dict = {"version": 2, "projects": [], "courses": []}
        self._load()

    # ------------------------------------------------------------------
    # Public API — projects (assessment directories)
    # ------------------------------------------------------------------

    def register(self, project_path: Path, description: str = "") -> bool:
        """
        Register a project (or update its last_opened if already known).

        Also auto-registers the parent directory as a course folder
        if it isn't already known.

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
                self._auto_register_course(resolved.parent, now)
                return True

        self._data["projects"].append({
            "path": str(resolved),
            "name": resolved.name,
            "description": description,
            "registered_at": now,
            "last_opened": now,
        })
        self._save()
        self._auto_register_course(resolved.parent, now)
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

    def set_template_id(self, project_path: Path, template_id: str):
        """Save the last-used template for a project."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return
        for entry in self._data["projects"]:
            if entry["path"] == str(resolved):
                entry["template_id"] = template_id
                self._save()
                return

    def get_template_id(self, project_path: Path) -> str:
        """Return the last-used template_id for a project, or empty string."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return ""
        for entry in self._data["projects"]:
            if entry["path"] == str(resolved):
                return entry.get("template_id", "")
        return ""

    # ------------------------------------------------------------------
    # Public API — courses (course folders / working directories)
    # ------------------------------------------------------------------

    def register_course(self, course_path: Path, name: str = "") -> bool:
        """
        Register a course folder (or update its last_opened).

        If *name* is empty the directory name is used as the display name.
        Returns True if successfully added/updated, False if path is invalid.
        """
        resolved = self._normalize(course_path)
        if resolved is None:
            return False

        now = datetime.now().isoformat()
        display_name = name or resolved.name

        for entry in self._data["courses"]:
            if entry["path"] == str(resolved):
                entry["last_opened"] = now
                if name:
                    entry["name"] = name
                self._save()
                return True

        self._data["courses"].append({
            "path": str(resolved),
            "name": display_name,
            "description": "",
            "registered_at": now,
            "last_opened": now,
        })
        self._save()
        return True

    def unregister_course(self, course_path: Path) -> bool:
        """
        Remove a course from the registry.  Returns True if found.

        Child assessments are NOT removed — they become "orphans" and
        are grouped under an "Other" heading in the UI.
        """
        resolved = self._normalize(course_path)
        if resolved is None:
            return False

        before = len(self._data["courses"])
        self._data["courses"] = [
            c for c in self._data["courses"]
            if c["path"] != str(resolved)
        ]
        if len(self._data["courses"]) < before:
            self._save()
            return True
        return False

    def list_courses(self) -> list[dict]:
        """Return all registered courses, sorted by most-recently-active first.

        "Active" is the latest ``last_opened`` timestamp among the course
        itself *and* any of its child assessments.  This way a course
        bubbles to the top when the teacher grades something inside it,
        even if the course entry itself hasn't been touched.
        """
        # Build a map: course_path -> max(last_opened) across children
        child_latest: dict[str, str] = {}
        for proj in self._data.get("projects", []):
            parent = str(Path(proj["path"]).parent)
            ts = proj.get("last_opened", "")
            if ts > child_latest.get(parent, ""):
                child_latest[parent] = ts

        courses = []
        for c in self._data.get("courses", []):
            entry = dict(c)
            course_ts = entry.get("last_opened", "")
            child_ts = child_latest.get(entry["path"], "")
            # Effective recency: whichever is later
            entry["_effective_last_active"] = max(course_ts, child_ts)
            courses.append(entry)

        # Most recent first; tie-break alphabetically by name
        courses.sort(
            key=lambda c: (c["_effective_last_active"], c.get("name", "").lower()),
            reverse=True,
        )
        return courses

    def set_course_name(self, course_path: Path, name: str):
        """Set a display name for a course (overrides the directory name)."""
        resolved = self._normalize(course_path)
        if resolved is None:
            return
        for entry in self._data["courses"]:
            if entry["path"] == str(resolved):
                entry["name"] = name
                self._save()
                return

    def update_course_last_opened(self, course_path: Path):
        """Touch the last_opened timestamp for a course."""
        resolved = self._normalize(course_path)
        if resolved is None:
            return
        now = datetime.now().isoformat()
        for entry in self._data["courses"]:
            if entry["path"] == str(resolved):
                entry["last_opened"] = now
                self._save()
                return

    def set_course_description(self, course_path: Path, description: str):
        """Set a description for a course."""
        resolved = self._normalize(course_path)
        if resolved is None:
            return
        for entry in self._data["courses"]:
            if entry["path"] == str(resolved):
                entry["description"] = description
                self._save()
                return

    def update_course_path(self, old_path: Path, new_path: Path) -> bool:
        """Re-point a course (and its child assessments) to a new folder.

        Useful when a teacher moves or renames a course folder on disk.
        Updates the course entry and migrates any project paths that were
        children of the old path.

        Returns True if the course was found and updated.
        """
        old_resolved = self._normalize(old_path)
        new_resolved = self._normalize(new_path)
        if old_resolved is None or new_resolved is None:
            return False

        old_str = str(old_resolved)
        new_str = str(new_resolved)

        # Update the course entry
        found = False
        for entry in self._data["courses"]:
            if entry["path"] == old_str:
                entry["path"] = new_str
                found = True
                break

        if not found:
            return False

        # Migrate child project paths: /old/course/Assessment -> /new/course/Assessment
        for project in self._data["projects"]:
            proj_path = project["path"]
            if proj_path.startswith(old_str + "/") or proj_path.startswith(old_str + "\\"):
                # Replace the old course prefix with the new one
                relative = proj_path[len(old_str):]
                project["path"] = new_str + relative

        self._save()
        return True

    def get_course_for_project(self, project_path: Path) -> Optional[dict]:
        """Return the course entry that owns a project, or None."""
        resolved = self._normalize(project_path)
        if resolved is None:
            return None
        parent = str(resolved.parent)
        for entry in self._data.get("courses", []):
            if entry["path"] == parent:
                return dict(entry)
        return None

    def list_by_course(self) -> dict[str, list[dict]]:
        """
        Group registered projects by their parent course folder.

        Returns a dict mapping ``course_path`` (str) to a list of
        project entries.  Projects whose parent is not a registered
        course are grouped under the key ``"__orphan__"``.
        """
        course_paths = {c["path"] for c in self._data.get("courses", [])}
        grouped: dict[str, list[dict]] = {}

        for project in self._data.get("projects", []):
            parent = str(Path(project["path"]).parent)
            if parent in course_paths:
                grouped.setdefault(parent, []).append(dict(project))
            else:
                grouped.setdefault("__orphan__", []).append(dict(project))

        return grouped

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

        # Ensure courses list exists and migrate if needed
        self._data.setdefault("courses", [])
        self._migrate_if_needed()

    def _migrate_if_needed(self):
        """Migrate v1 schema to v2 by deriving courses from project paths."""
        version = self._data.get("version", 1)
        if version >= 2:
            return  # already current

        # Derive unique parent directories from existing projects
        seen_paths: set[str] = set()
        now = datetime.now().isoformat()

        for project in self._data.get("projects", []):
            parent = str(Path(project["path"]).parent)
            if parent not in seen_paths:
                seen_paths.add(parent)
                self._data["courses"].append({
                    "path": parent,
                    "name": Path(parent).name,
                    "description": "",
                    "registered_at": project.get("registered_at", now),
                    "last_opened": project.get("last_opened", now),
                })

        self._data["version"] = 2
        self._save()

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

    def _auto_register_course(self, parent: Path, now: str):
        """Ensure a project's parent directory is registered as a course."""
        parent_str = str(parent)
        if any(c["path"] == parent_str for c in self._data.get("courses", [])):
            return  # already known
        self._data["courses"].append({
            "path": parent_str,
            "name": parent.name,
            "description": "",
            "registered_at": now,
            "last_opened": now,
        })
        self._save()

    @staticmethod
    def _normalize(path: Path) -> Optional[Path]:
        """Resolve to an absolute path. Returns None for empty/invalid."""
        try:
            return Path(path).resolve()
        except Exception:
            return None
