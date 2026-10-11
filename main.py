#!/usr/bin/env python3
"""Sage Model Chain Studio — entry point."""
import os
for _dll_dir in [
    r"C:\Users\SageDesk\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\SageDesk\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin",
]:
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)

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
