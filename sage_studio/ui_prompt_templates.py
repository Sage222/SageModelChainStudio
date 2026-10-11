"""Sage Model Chain Studio — Prompt Templates management tab.

Full CRUD for saved prompt templates (sage_data/prompt_templates.json).
The same file is read by the StepEditDialog's template picker, so anything
created/edited/deleted here is immediately available there too, and
vice versa.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QPlainTextEdit, QLineEdit, QMessageBox, QInputDialog,
    QGroupBox,
)


class PromptTemplatesPanel(QWidget):
    def __init__(self, persistence):
        super().__init__()
        self.persistence = persistence
        self.templates = self.persistence.load_prompt_templates()
        self._current_name = None

        root = QVBoxLayout(self)

        title = QLabel("Prompt Templates")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        root.addWidget(title)

        hint = QLabel(
            "Create and manage reusable prompt templates (personas, rules, instructions). "
            "Anything saved here is available in every step's 'Edit Prompt' dialog."
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        body = QHBoxLayout()
        root.addLayout(body, 1)

        # ── Left: list of saved templates ───────────────────────────────────
        left_group = QGroupBox("Saved Templates")
        left_layout = QVBoxLayout(left_group)
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.list_widget, 1)

        left_btn_row = QHBoxLayout()
        new_btn = QPushButton("+ New")
        new_btn.setObjectName("accent")
        new_btn.clicked.connect(self._new_template)
        left_btn_row.addWidget(new_btn)
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self._delete_template)
        left_btn_row.addWidget(delete_btn)
        left_layout.addLayout(left_btn_row)

        body.addWidget(left_group, 1)

        # ── Right: editor ────────────────────────────────────────────────────
        right_group = QGroupBox("Editor")
        right_layout = QVBoxLayout(right_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Dungeon Master")
        name_row.addWidget(self.name_edit, 1)
        right_layout.addLayout(name_row)

        right_layout.addWidget(QLabel("Template text (use {input} where the previous step's output should go):"))
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "You are a Dungeon Master running a dark fantasy campaign. Follow these rules "
            "strictly:\n- Never break character.\n- Keep responses under 200 words.\n\n"
            "Current player action: {input}"
        )
        right_layout.addWidget(self.text_edit, 1)

        save_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.setObjectName("accent")
        save_btn.clicked.connect(self._save_current)
        save_row.addWidget(save_btn)
        save_row.addStretch(1)
        right_layout.addLayout(save_row)

        body.addWidget(right_group, 2)

        self.status = QLabel("")
        self.status.setObjectName("sectionHint")
        root.addWidget(self.status)

        self._refresh_list()

    def _refresh_list(self, select_name=None):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for name in sorted(self.templates.keys()):
            item = QListWidgetItem(name)
            self.list_widget.addItem(item)
            if name == select_name:
                self.list_widget.setCurrentItem(item)
        self.list_widget.blockSignals(False)

    def _on_selection_changed(self, current, previous):
        if current is None:
            return
        name = current.text()
        self._current_name = name
        self.name_edit.setText(name)
        self.text_edit.setPlainText(self.templates.get(name, ""))
        self.status.setText(f"Editing '{name}'")

    def _new_template(self):
        name, ok = QInputDialog.getText(self, "New Prompt Template", "Template name:")
        if ok and name.strip():
            name = name.strip()
            if name in self.templates:
                QMessageBox.warning(self, "Already exists", f"A template named '{name}' already exists.")
                return
            self.templates[name] = ""
            self.persistence.save_prompt_templates(self.templates)
            self._refresh_list(select_name=name)
            self.name_edit.setText(name)
            self.text_edit.clear()
            self.text_edit.setFocus()
            self.status.setText(f"Created '{name}' \u2014 write the template text and click Save")

    def _save_current(self):
        new_name = self.name_edit.text().strip()
        text = self.text_edit.toPlainText()
        if not new_name:
            QMessageBox.warning(self, "Missing name", "Give the template a name first.")
            return
        if self._current_name and self._current_name != new_name:
            # Renaming: remove the old key
            self.templates.pop(self._current_name, None)
        self.templates[new_name] = text
        self.persistence.save_prompt_templates(self.templates)
        self._current_name = new_name
        self._refresh_list(select_name=new_name)
        self.status.setText(f"Saved '{new_name}'")

    def _delete_template(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a template to delete first.")
            return
        name = item.text()
        confirm = QMessageBox.question(
            self, "Delete template", f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.templates.pop(name, None)
            self.persistence.save_prompt_templates(self.templates)
            self._current_name = None
            self.name_edit.clear()
            self.text_edit.clear()
            self._refresh_list()
            self.status.setText(f"Deleted '{name}'")
