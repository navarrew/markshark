"""
Data models for MarkShark GUI.

Includes:
- CorrectionLog: Append-only log for tracking review corrections
- ProjectRegistry: Persistent JSON registry of known project directories
- LmsFilterRegistry: Persistent JSON registry of saved LMS column-mapping filters
- SettingsStore: JSON-backed application settings (~/.markshark/settings.json)
"""

from .corrections import (
    CorrectionLog,
    Correction,
)

from .project_registry import ProjectRegistry
from .lms_filter_registry import LmsFilterRegistry
from .settings_store import SettingsStore

__all__ = [
    "CorrectionLog",
    "Correction",
    "ProjectRegistry",
    "LmsFilterRegistry",
    "SettingsStore",
]
