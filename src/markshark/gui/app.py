"""
Application entry point and QApplication setup.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from .main_window import MainWindow


def main():
    """Launch the MarkShark GUI application."""
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
