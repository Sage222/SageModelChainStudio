"""Sage Model Chain Studio — Ui Chain Builder"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMessageBox,
    QInputDialog,
    QComboBox,
    QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QDialogButtonBox,
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from .models import ChainStep
from .utils import step_display


class StepEditDialog(QDialog):
    def __init__(self, step, persistence, parent=None):
        super().__init__(parent)
        self.persistence = persistence
        self.saved_templates = self.persistence.load_prompt_templates() if persistence else {}
        self.setWindowTitle(f"Edit Step \u2014 {step_display(step)}")
        self.resize(520, 380)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Provider: {step.provider_label}  Model: {step_display(step)}"))
        layout.addWidget(QLabel("Prompt template. Use {input} for the previous step's output."))

        template_row = QHBoxLayout()
        self.template_combo = QComboBox()
        self.template_combo.addItem("(Load a saved template...)")
        for name in sorted(self.saved_templates.keys()):
            self.template_combo.addItem(name)
        self.template_combo.currentIndexChanged.connect(self._on_template_selected)
        template_row.addWidget(self.template_combo, 1)
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self._save_as_template)
        template_row.addWidget(save_as_btn)
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self._delete_template)
        template_row.addWidget(delete_btn)
        layout.addLayout(template_row)

        self.template_edit = QPlainTextEdit(step.prompt_template)
        layout.addWidget(self.template_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_template_selected(self, idx):
        if idx <= 0:
            return
        name = self.template_combo.currentText()
        text = self.saved_templates.get(name, "")
        self.template_edit.setPlainText(text)

    def _save_as_template(self):
        name, ok = QInputDialog.getText(self, "Save Prompt Template", "Template name:")
        if ok and name.strip():
            name = name.strip()
            self.saved_templates[name] = self.template_edit.toPlainText()
            if self.persistence:
                self.persistence.save_prompt_templates(self.saved_templates)
            if self.template_combo.findText(name) == -1:
                self.template_combo.addItem(name)
            self.template_combo.setCurrentText(name)
            QMessageBox.information(self, "Saved", f"Template '{name}' saved.")

    def _delete_template(self):
        name = self.template_combo.currentText()
        if name not in self.saved_templates:
            return
        confirm = QMessageBox.question(
            self, "Delete Template", f"Delete '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            del self.saved_templates[name]
            if self.persistence:
                self.persistence.save_prompt_templates(self.saved_templates)
            idx = self.template_combo.findText(name)
            if idx >= 0:
                self.template_combo.removeItem(idx)

    def get_template(self):
        return self.template_edit.toPlainText()



class ChainBuilderPanel(QGroupBox):
    def __init__(self, persistence=None):
        super().__init__("Loaded Models in Chain")
        self.persistence = persistence
        self.steps = []
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "Provider / Model", "Prompt Template"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.cellDoubleClicked.connect(self._on_double_click_remove)
        self.table.setMaximumHeight(180)
        layout.addWidget(self.table)

        hint = QLabel("Double-click a row to remove it. Add models from the Models tab or Favorites.")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QHBoxLayout()
        up_btn = QPushButton("\u2191 Move Up")
        up_btn.clicked.connect(self._move_up)
        down_btn = QPushButton("\u2193 Move Down")
        down_btn.clicked.connect(self._move_down)
        edit_btn = QPushButton("Edit Prompt")
        edit_btn.clicked.connect(self._edit_step)
        remove_btn = QPushButton("Remove Step")
        remove_btn.setObjectName("danger")
        remove_btn.clicked.connect(self._remove_step)
        for b in (up_btn, down_btn, edit_btn, remove_btn):
            btn_row.addWidget(b)
        layout.addLayout(btn_row)
        self.on_change = None

    def set_steps(self, steps):
        self.steps = steps
        self._refresh_table()

    def add_step(self, step):
        self.steps.append(step)
        self._refresh_table()
        if self.on_change:
            self.on_change()

    def _refresh_table(self):
        self.table.setRowCount(len(self.steps))
        for i, s in enumerate(self.steps):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(f"{s.provider_label} \u2192 {step_display(s)}"))
            preview = s.prompt_template.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57] + "..."
            self.table.setItem(i, 2, QTableWidgetItem(preview))

    def _current_row(self):
        row = self.table.currentRow()
        return row if row is not None and row >= 0 else None

    def _on_double_click_remove(self, row, col):
        if 0 <= row < len(self.steps):
            del self.steps[row]
            self._refresh_table()
            if self.on_change:
                self.on_change()


    def edit_step_by_model_id(self, model_id):
        row = None
        for i, s in enumerate(self.steps):
            if s.model_id == model_id:
                row = i
                break
        if row is None:
            return False
        dlg = StepEditDialog(self.steps[row], self.persistence, self)
        if dlg.exec():
            self.steps[row].prompt_template = dlg.get_template()
            self._refresh_table()
            if self.on_change:
                self.on_change()
        return True

    def _edit_step(self):
        row = self._current_row()
        if row is None:
            return
        dlg = StepEditDialog(self.steps[row], self.persistence, self)
        if dlg.exec():
            self.steps[row].prompt_template = dlg.get_template()
            self._refresh_table()
            if self.on_change:
                self.on_change()


    def remove_step_by_model_id(self, model_id):
        row = None
        for i, s in enumerate(self.steps):
            if s.model_id == model_id:
                row = i
                break
        if row is None:
            return False
        del self.steps[row]
        self._refresh_table()
        if self.on_change:
            self.on_change()
        return True

    def _remove_step(self):
        row = self._current_row()
        if row is None:
            return
        del self.steps[row]
        self._refresh_table()
        if self.on_change:
            self.on_change()

    def _move_up(self):
        row = self._current_row()
        if row is None or row == 0:
            return
        self.steps[row - 1], self.steps[row] = self.steps[row], self.steps[row - 1]
        self._refresh_table()
        self.table.selectRow(row - 1)
        if self.on_change:
            self.on_change()

    def _move_down(self):
        row = self._current_row()
        if row is None or row >= len(self.steps) - 1:
            return
        self.steps[row + 1], self.steps[row] = self.steps[row], self.steps[row + 1]
        self._refresh_table()
        self.table.selectRow(row + 1)
        if self.on_change:
            self.on_change()


