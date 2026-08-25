#!/usr/bin/env python3
"""Sage Model Chain Studio — entry point."""

import sys
from PyQt6.QtWidgets import QApplication
from sage_studio.constants import APP_NAME
from sage_studio.ui_style import DARK_QSS
from sage_studio.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
