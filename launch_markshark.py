#!/usr/bin/env python3
"""
PyInstaller entry point for MarkShark GUI.

This script uses absolute imports so PyInstaller can resolve
the full package tree. It should NOT be used for normal development
— use `markshark-gui` or `python -m markshark.gui` instead.
"""

from markshark.gui.app import main

if __name__ == "__main__":
    main()
