"""
MarkShark PySide6 GUI
Application entry point and QApplication setup.
"""

import sys
import platform

# Set application name BEFORE QApplication is created
# This affects the macOS menu bar application name
if platform.system() == "Darwin":
    # macOS: Set the process name for the menu bar
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info:
            info["CFBundleName"] = "MarkShark"
    except ImportError:
        pass  # PyObjC not installed, menu will show "python"

from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette

# Import the layout and functions of the main window from the script
# 'main_window.py' within the markshark GUI directory.
# It is from that script you can edit the application layouts, etc.
from .main_window import MainWindow


def _load_bundled_fonts():
    """Register Poppins font files bundled in assets/fonts/."""
    fonts_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    if not fonts_dir.is_dir():
        return
    for ttf in sorted(fonts_dir.glob("Poppins*.ttf")):
        QFontDatabase.addApplicationFont(str(ttf))


def main():
    """Launch the MarkShark GUI application."""
    # Set argv[0] to help Qt identify the app
    sys.argv[0] = "MarkShark"

    app = QApplication(sys.argv)

    # Application metadata (used by Qt for window geometry, dialogs, etc.)
    app.setApplicationName("MarkShark")
    app.setApplicationDisplayName("MarkShark")
    app.setOrganizationName("MarkShark")
    app.setOrganizationDomain("markshark.io")

    # ── Force dark mode ──────────────────────────────────────────────
    # Use Qt's "Fusion" style so the app draws its own widgets instead
    # of inheriting the OS native theme.  This makes the look identical
    # on macOS (light or dark), Windows, and Linux regardless of system
    # appearance settings.  Every color role is set explicitly so nothing
    # leaks in from the OS palette.
    app.setStyle("Fusion")

    dark = QPalette()
    dark.setColor(QPalette.ColorRole.Window,          QColor(43, 43, 43))
    dark.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    dark.setColor(QPalette.ColorRole.Base,            QColor(35, 35, 35))
    dark.setColor(QPalette.ColorRole.AlternateBase,   QColor(50, 50, 50))
    dark.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    dark.setColor(QPalette.ColorRole.Button,          QColor(53, 53, 53))
    dark.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    dark.setColor(QPalette.ColorRole.BrightText,      QColor(255, 255, 255))
    dark.setColor(QPalette.ColorRole.Link,            QColor(42, 130, 218))
    dark.setColor(QPalette.ColorRole.Highlight,       QColor(14, 129, 126))  # teal brand
    dark.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    dark.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 50))
    dark.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    dark.setColor(QPalette.ColorRole.PlaceholderText, QColor(128, 128, 128))

    # Disabled-state colours so greyed-out widgets are still readable
    dark.setColor(QPalette.ColorGroup.Disabled,
                  QPalette.ColorRole.WindowText, QColor(128, 128, 128))
    dark.setColor(QPalette.ColorGroup.Disabled,
                  QPalette.ColorRole.Text,       QColor(128, 128, 128))
    dark.setColor(QPalette.ColorGroup.Disabled,
                  QPalette.ColorRole.ButtonText, QColor(128, 128, 128))

    app.setPalette(dark)
    # ─────────────────────────────────────────────────────────────────

    # Load bundled Poppins font and set as app-wide default
    _load_bundled_fonts()
    font = QFont("Poppins", 12)
    app.setFont(font)

    # App-wide stylesheet: QGroupBox "card" look with visible contrast
    app.setStyleSheet(
        "QGroupBox { font-family: 'Poppins'; font-weight: 500; font-size: 15px;"
        "            padding-top: 22px;"
        "            background-color: rgba(255, 255, 255, 0.06);"
        "            border: 1px solid rgba(255, 255, 255, 0.12);"
        "            border-radius: 6px;"
        "            margin-top: 6px; }"
        "QGroupBox::title { subcontrol-origin: margin;"
        "                    subcontrol-position: top left;"
        "                    padding: 4px 8px;"
        "                    background-color: #0E817E;"
        "                    color: white;"
        "                    border-top-left-radius: 4px;"
        "                    border-top-right-radius: 4px;"
        "                    border-bottom-left-radius: 0px;"
        "                    border-bottom-right-radius: 4px; }"
    )

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
