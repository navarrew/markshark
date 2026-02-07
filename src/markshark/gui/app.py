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

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

# Import the layout and functions of the main window from the script
# 'main_window.py' within the markshark GUI directory.
# It is from that script you can edit the application layouts, etc.
from .main_window import MainWindow


def main():
    """Launch the MarkShark GUI application."""
    # Set argv[0] to help Qt identify the app
    sys.argv[0] = "MarkShark"

    app = QApplication(sys.argv)

    # Application metadata
    app.setApplicationName("MarkShark")
    app.setApplicationDisplayName("MarkShark")
    app.setOrganizationName("MarkShark")
    app.setOrganizationDomain("markshark.io")

    # Optional: Set default font
    # font = QFont("Segoe UI", 10)  # Windows
    # font = QFont("SF Pro", 13)    # macOS
    # app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
