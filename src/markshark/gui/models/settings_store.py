"""
Persistent application settings stored in ~/.markshark/settings.json.

Replaces QSettings with a cross-platform JSON file in the same
~/.markshark/ directory used by ProjectRegistry and LmsFilterRegistry.
One folder to delete for a clean uninstall.
"""

import json
from pathlib import Path
from typing import Any, Optional, Type


_DEFAULT_PATH = Path.home() / ".markshark" / "settings.json"

# Singleton instances keyed by resolved file path, so every
# SettingsStore() that points at the same file shares one in-memory
# copy — same pattern as ProjectRegistry and LmsFilterRegistry.
_instances: dict[str, "SettingsStore"] = {}


class SettingsStore:
    """
    JSON-backed settings store (singleton per file path).

    API mirrors QSettings so existing callers need only swap the import:
        value(key, default, type=)  →  get a setting
        setValue(key, value)         →  set a setting
        sync()                       →  flush to disk (also done on every set)

    Keys use slash-delimited paths matching the existing QSettings
    namespace: "scoring/min_fill" maps to settings["scoring"]["min_fill"]
    in the JSON.

    Schema:
    {
        "version": 1,
        "settings": { ... nested dict ... }
    }
    """

    def __new__(cls, path: Optional[Path] = None):
        resolved = str((path or _DEFAULT_PATH).resolve())
        if resolved in _instances:
            return _instances[resolved]
        inst = super().__new__(cls)
        _instances[resolved] = inst
        return inst

    def __init__(self, path: Optional[Path] = None):
        # Only initialise once (singleton)
        if hasattr(self, "_path"):
            return
        self._path = path or _DEFAULT_PATH
        self._data: dict = {"version": 1, "settings": {}}
        self._load()

    # ------------------------------------------------------------------
    # Public API (QSettings-compatible signatures)
    # ------------------------------------------------------------------

    def value(
        self,
        key: str,
        defaultValue: Any = None,
        type: Optional[Type] = None,
    ) -> Any:
        """
        Get a setting by slash-delimited key.

        Matches QSettings.value(key, defaultValue, type=) so callers
        don't need any changes beyond swapping the import.
        """
        parts = key.split("/")
        node = self._data.get("settings", {})
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return defaultValue
        if type is not None and node is not None:
            try:
                if type is bool:
                    # JSON preserves bools natively, but guard against
                    # strings "true"/"false" from hand-edited files.
                    if isinstance(node, bool):
                        return node
                    if isinstance(node, str):
                        return node.lower() in ("true", "1", "yes")
                    return bool(node)
                return type(node)
            except (ValueError, TypeError):
                return defaultValue
        return node

    def setValue(self, key: str, value: Any) -> None:
        """
        Set a setting by slash-delimited key and persist to disk.

        Matches QSettings.setValue(key, value).
        """
        parts = key.split("/")
        node = self._data["settings"]
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
        self._save()

    def sync(self) -> None:
        """Flush to disk. Compatibility with QSettings.sync()."""
        self._save()

    def clear(self) -> None:
        """Remove all settings (used by reset-to-defaults)."""
        self._data["settings"] = {}
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        """Load settings from disk."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and "settings" in data:
                self._data = data
        except Exception:
            # Corrupted file — start fresh
            pass

    def _save(self):
        """Write settings to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
