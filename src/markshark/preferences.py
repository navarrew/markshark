#!/usr/bin/env python3
"""
Persistent user preferences for MarkShark GUI.

Stores preferences in a JSON file in the user's standard config directory:
- macOS: ~/Library/Application Support/MarkShark/preferences.json
- Windows: %APPDATA%/MarkShark/preferences.json
- Linux: ~/.config/MarkShark/preferences.json
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any


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


def load_preferences() -> dict[str, Any]:
    """
    Load preferences from disk.

    Returns:
        Dictionary of preferences, or empty dict if file doesn't exist.
    """
    prefs_path = get_preferences_path()

    if not prefs_path.exists():
        return {}

    try:
        with open(prefs_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_preferences(prefs: dict[str, Any]) -> bool:
    """
    Save preferences to disk.

    Args:
        prefs: Dictionary of preferences to save.

    Returns:
        True if saved successfully, False otherwise.
    """
    prefs_path = get_preferences_path()

    try:
        # Ensure directory exists
        prefs_path.parent.mkdir(parents=True, exist_ok=True)

        with open(prefs_path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
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
    return prefs.get(key, default)


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
    prefs[key] = value
    return save_preferences(prefs)


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
    "get_preferences_dir",
    "get_preferences_path",
    "load_preferences",
    "save_preferences",
    "get_preference",
    "set_preference",
    "clear_preferences",
]
