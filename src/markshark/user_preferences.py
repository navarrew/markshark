#!/usr/bin/env python3
"""
Persistent user preferences for MarkShark GUI.

Stores user session state in a JSON file in the user's standard config directory:
- macOS: ~/Library/Application Support/MarkShark/preferences.json
- Windows: %APPDATA%/MarkShark/preferences.json
- Linux: ~/.config/MarkShark/preferences.json

This module handles USER preferences (session state that persists across runs).
For algorithm/rendering defaults, see defaults.py.
For template ordering/favorites, see template_manager.py.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Optional, List, Any

# Maximum number of recent items to keep
MAX_RECENT_PROJECTS = 10
MAX_RECENT_TEMPLATES = 10


@dataclass
class UserPreferences:
    """
    Schema for persistent user preferences.

    These are user-specific settings that persist across sessions,
    such as recently used projects and templates.
    """
    # Directory/project memory
    default_workdir: Optional[str] = None
    last_project: Optional[str] = None
    recent_projects: List[str] = field(default_factory=list)

    # Template memory
    last_template_id: Optional[str] = None
    recent_templates: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "UserPreferences":
        """
        Load from dict, ignoring unknown keys for forward compatibility.

        Args:
            data: Dictionary loaded from JSON

        Returns:
            UserPreferences instance with validated fields
        """
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def get_preferences_dir() -> Path:
    """Get the platform-appropriate config directory for MarkShark."""
    system = platform.system()

    if system == "Darwin":  # macOS
        config_dir = Path.home() / "Library" / "Application Support" / "MarkShark"
    elif system == "Windows":
        appdata = Path.home() / "AppData" / "Roaming"
        config_dir = appdata / "MarkShark"
    else:  # Linux and others
        config_dir = Path.home() / ".config" / "MarkShark"

    return config_dir


def get_preferences_path() -> Path:
    """Get the full path to the preferences file."""
    return get_preferences_dir() / "preferences.json"


def load_preferences() -> UserPreferences:
    """
    Load preferences from disk.

    Returns:
        UserPreferences instance (with defaults if file doesn't exist or is invalid).
    """
    prefs_path = get_preferences_path()

    if not prefs_path.exists():
        return UserPreferences()

    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return UserPreferences.from_dict(data)
    except (json.JSONDecodeError, IOError, TypeError):
        return UserPreferences()


def save_preferences(prefs: UserPreferences) -> bool:
    """
    Save preferences to disk.

    Args:
        prefs: UserPreferences instance to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs_path = get_preferences_path()

    try:
        # Ensure directory exists
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prefs_path, "w", encoding="utf-8") as f:
            json.dump(prefs.to_dict(), f, indent=2)
        return True
    except IOError:
        return False


def get_preference(key: str, default: Any = None) -> Any:
    """
    Get a single preference value.

    Args:
        key: Preference key (e.g., "default_workdir")
        default: Default value if key doesn't exist

    Returns:
        The preference value, or default if not found.
    """
    prefs = load_preferences()
    return getattr(prefs, key, default)


def set_preference(key: str, value: Any) -> bool:
    """
    Set a single preference value and save to disk.

    Args:
        key: Preference key
        value: Value to store

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs = load_preferences()
    if hasattr(prefs, key):
        # Create a new instance with the updated value
        prefs_dict = prefs.to_dict()
        prefs_dict[key] = value
        updated_prefs = UserPreferences.from_dict(prefs_dict)
        return save_preferences(updated_prefs)
    return False


def add_recent_project(project_path: str) -> bool:
    """
    Add a project to the recent projects list.

    Moves to front if already present, maintains max size.

    Args:
        project_path: Path or name of the project

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs = load_preferences()
    recent = prefs.recent_projects.copy()

    # Remove if already present (will re-add at front)
    if project_path in recent:
        recent.remove(project_path)

    # Add to front
    recent.insert(0, project_path)

    # Trim to max size
    recent = recent[:MAX_RECENT_PROJECTS]

    prefs_dict = prefs.to_dict()
    prefs_dict["recent_projects"] = recent
    prefs_dict["last_project"] = project_path

    return save_preferences(UserPreferences.from_dict(prefs_dict))


def add_recent_template(template_id: str) -> bool:
    """
    Add a template to the recent templates list.

    Moves to front if already present, maintains max size.

    Args:
        template_id: Template identifier

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs = load_preferences()
    recent = prefs.recent_templates.copy()

    # Remove if already present (will re-add at front)
    if template_id in recent:
        recent.remove(template_id)

    # Add to front
    recent.insert(0, template_id)

    # Trim to max size
    recent = recent[:MAX_RECENT_TEMPLATES]

    prefs_dict = prefs.to_dict()
    prefs_dict["recent_templates"] = recent
    prefs_dict["last_template_id"] = template_id

    return save_preferences(UserPreferences.from_dict(prefs_dict))


def get_recent_projects() -> List[str]:
    """Get list of recent projects (most recent first)."""
    prefs = load_preferences()
    return prefs.recent_projects.copy()


def get_recent_templates() -> List[str]:
    """Get list of recent template IDs (most recent first)."""
    prefs = load_preferences()
    return prefs.recent_templates.copy()


def clear_preferences() -> bool:
    """
    Clear all preferences (delete the file).

    Returns:
        True if cleared successfully, False otherwise.
    """
    prefs_path = get_preferences_path()
    try:
        if prefs_path.exists():
            prefs_path.unlink()
        return True
    except IOError:
        return False


__all__ = [
    "UserPreferences",
    "MAX_RECENT_PROJECTS",
    "MAX_RECENT_TEMPLATES",
    "get_preferences_dir",
    "get_preferences_path",
    "load_preferences",
    "save_preferences",
    "get_preference",
    "set_preference",
    "add_recent_project",
    "add_recent_template",
    "get_recent_projects",
    "get_recent_templates",
    "clear_preferences",
]
