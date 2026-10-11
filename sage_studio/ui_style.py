"""Sage Model Chain Studio — Ui Style"""



DARK_QSS = """
QWidget { background-color: #14161c; color: #e6e6ea; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QMainWindow { background-color: #14161c; }
QGroupBox { border: 1px solid #262a35; border-radius: 10px; margin-top: 14px; padding-top: 12px; font-weight: 600; color: #a9b1c3; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #8b7cf6; }
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background-color: #1c1f28; border: 1px solid #2a2e3a; border-radius: 8px; padding: 6px 10px; selection-background-color: #6c4de6; }
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #8b7cf6; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background-color: #1c1f28; border: 1px solid #2a2e3a; selection-background-color: #6c4de6; }
QSpinBox, QDoubleSpinBox { color: #e6e6ea; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { background: #333848; border: none; width: 18px; }
QPushButton { background-color: #262a35; border: 1px solid #333848; border-radius: 8px; padding: 7px 14px; color: #e6e6ea; font-weight: 600; }
QPushButton:hover { background-color: #313644; border-color: #8b7cf6; }
QPushButton:pressed { background-color: #23262f; }
QPushButton:disabled { color: #5c6070; background-color: #1c1f28; }
QPushButton#accent { background-color: #6c4de6; border: none; color: white; }
QPushButton#accent:hover { background-color: #7c5cf0; }
QPushButton#danger { background-color: #3a2030; border: 1px solid #5c2a44; color: #ff9bb3; }
QPushButton#danger:hover { background-color: #4a2740; }
QPushButton#stop { background-color: #4a1f1f; border: 1px solid #6b2b2b; color: #ffb4b4; }
QPushButton#stop:hover { background-color: #5c2626; }
QListWidget, QTableWidget { background-color: #1a1c24; border: 1px solid #262a35; border-radius: 8px; outline: none; }
QListWidget::item, QTableWidget::item { padding: 6px; border-bottom: 1px solid #21242e; }
QListWidget::item:selected, QTableWidget::item:selected { background-color: #322a55; color: #ffffff; }
QHeaderView::section { background-color: #1c1f28; color: #a9b1c3; padding: 6px; border: none; border-bottom: 1px solid #262a35; font-weight: 600; }
QScrollBar:vertical { background: #14161c; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #333848; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #454b5f; }
QScrollBar:horizontal { background: #14161c; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: #333848; border-radius: 5px; min-width: 24px; }
QStatusBar { background-color: #1a1c24; color: #8b90a0; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #3a3f4d; background: #1c1f28; }
QCheckBox::indicator:checked { background: #6c4de6; border: 1px solid #6c4de6; }
QSplitter::handle { background-color: #14161c; }
QMenu { background-color: #1c1f28; border: 1px solid #2a2e3a; color: #e6e6ea; }
QMenu::item:selected { background-color: #6c4de6; }
QTabWidget::pane { border: 1px solid #262a35; border-radius: 10px; top: -1px; }
QTabBar::tab { background: #1a1c24; color: #a9b1c3; padding: 8px 18px; border: 1px solid #262a35; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px; }
QTabBar::tab:selected { background: #262a35; color: #ffffff; border-bottom: 2px solid #8b7cf6; }
QTabBar::tab:hover { background: #232733; }
QSlider::groove:horizontal { height: 6px; background: #2a2e3a; border-radius: 3px; }
QSlider::handle:horizontal { background: #6c4de6; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: #7c5cf0; }
QScrollArea { border: none; }
QLabel#badgeFree { color: #4ade80; font-weight: 700; }
QLabel#badgePaid { color: #f87171; font-weight: 700; }
QLabel#badgeUnknown { color: #facc15; font-weight: 700; }
QLabel#sectionHint { color: #757c8f; font-style: italic; }
QLabel#chatTitle { color: #e6e6ea; font-weight: 700; }
"""


