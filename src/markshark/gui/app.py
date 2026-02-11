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
from PySide6.QtGui import QFont, QFontDatabase

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

    # Load bundled Poppins font and set as app-wide default
    _load_bundled_fonts()
    font = QFont("Poppins", 12)
    app.setFont(font)

    # App-wide stylesheet: larger Poppins Medium for QGroupBox section titles
    app.setStyleSheet(
        "QGroupBox { font-family: 'Poppins'; font-weight: 500; font-size: 15px;"
        "            padding-top: 22px; }"
        "QGroupBox::title { subcontrol-origin: margin;"
        "                    subcontrol-position: top left;"
        "                    padding: 4px 8px;"
        "                    background-color: #0E817E;"
        "                    color: white; }"
    )

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
