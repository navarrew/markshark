# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MarkShark GUI — Windows build.

Build with:
    pyinstaller MarkShark-win.spec

Produces:  dist/MarkShark/MarkShark.exe
"""

import os
from pathlib import Path

SRC = Path("src/markshark")

# Read version from MARKSHARK_VERSION env var (set by CI), or fall back
VERSION = os.environ.get("MARKSHARK_VERSION", "0.0.0")

a = Analysis(
    ["launch_markshark.py"],
    pathex=["src"],
    binaries=[],
    datas=[
        (str(SRC / "templates"), "markshark/templates"),
        (str(SRC / "assets"), "markshark/assets"),
        (str(SRC / "gui" / "resources"), "markshark/gui/resources"),
    ],
    hiddenimports=[
        # PySide6 extras
        "PySide6.QtSvg",
        "PySide6.QtPrintSupport",
        # GUI pages
        "markshark.gui.pages.quick_grade",
        "markshark.gui.pages.review_panel",
        "markshark.gui.pages.template_manager",
        "markshark.gui.pages.settings",
        "markshark.gui.pages.align_only",
        "markshark.gui.pages.score_only",
        "markshark.gui.pages.report_only",
        "markshark.gui.pages.pdf_tools",
        "markshark.gui.pages.map_viewer",
        "markshark.gui.pages.lms_integration",
        "markshark.gui.pages.mock_data_utility",
        "markshark.gui.pages.help_page",
        "markshark.gui.pages.key_builder",
        "markshark.gui.pages.welcome_page",
        "markshark.gui.pages.project_manager_page",
        # GUI widgets
        "markshark.gui.widgets.file_selector",
        "markshark.gui.widgets.log_viewer",
        "markshark.gui.widgets.page_header",
        "markshark.gui.widgets.pdf_preview",
        "markshark.gui.widgets.project_selector",
        # GUI dialogs
        "markshark.gui.dialogs.about",
        "markshark.gui.dialogs.course_dialog",
        # GUI flag panel
        "markshark.gui.widgets.flag_info_panel",
        # GUI models
        "markshark.gui.models.corrections",
        "markshark.gui.models.project_registry",
        "markshark.gui.models.lms_filter_registry",
        "markshark.gui.models.settings_store",
        # GUI workers
        "markshark.gui.workers.cli_runner",
        # GUI main window
        "markshark.gui.main_window",
        # Core modules
        "markshark.tools.bubblemap_io",
        "markshark.tools.align_tools",
        "markshark.tools.score_tools",
        "markshark.tools.report_tools",
        "markshark.tools.stats_tools",
        "markshark.tools.visualizer_tools",
        "markshark.tools.io_pages",
        "markshark.template_manager",
        "markshark.align_core",
        "markshark.score_core",
        "markshark.mapviewer_core",
        "markshark.tools.key_parser",
        "markshark.defaults",
        "markshark.tools.project_utils",
        "markshark.mock_dataset",
        # Third-party
        "yaml",
        "openpyxl",
        "rapidfuzz",
        "cv2",
        "fitz",
        "pdf2image",
        "numpy",
        "pandas",
        "matplotlib",
        "matplotlib.backends.backend_agg",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "streamlit",
        "tkinter",
        "unittest",
        "test",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    icon="markshark.ico",
    exclude_binaries=True,
    name="MarkShark",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX disabled — can cause false positives with Windows Defender
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MarkShark",
)
