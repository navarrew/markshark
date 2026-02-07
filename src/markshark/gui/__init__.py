"""
MarkShark GUI - Native desktop interface built with PySide6.

Usage:
    python -m markshark.gui

Structure:
    - app.py: Application setup and entry point
    - main_window.py: Main window with navigation
    - pages/: Major workflow screens (quick_grade, review, etc.)
    - widgets/: Reusable UI components
    - dialogs/: Popup dialogs
    - models/: Qt data models for tables
    - workers/: Background task runners
"""

from .app import main

__all__ = ["main"]
