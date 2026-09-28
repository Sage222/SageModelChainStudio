"""Sage Model Chain Studio — small GitHub token widget for the Settings tab.

Deliberately tiny: just the token field, remember checkbox, and save button.
Both this widget and the GitHub tab read/write the same persisted key
("github" scope in sage_data/api_keys.json), so saving here is immediately
usable by the GitHub Vibe Coding tab with no extra wiring needed.
"""

from PyQt6.QtWidgets import (
    QGroupBox, QFormLayout, QHBoxLayout, QLineEdit, QPushButton, QCheckBox,
)


class GitHubSettingsPanel(QGroupBox):
    def __init__(self, persistence):
        super().__init__("GitHub Token")
        self.persistence = persistence

        form = QFormLayout(self)

        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("ghp_xxxxxxxx (needs 'repo' scope)")
        saved_token = persistence.get_key("github")
        if saved_token:
            self.token_edit.setText(saved_token)
        form.addRow("Token:", self.token_edit)

        row = QHBoxLayout()
        self.remember_check = QCheckBox("Remember token")
        self.remember_check.setChecked(bool(saved_token))
        row.addWidget(self.remember_check)
        save_btn = QPushButton("Save Token")
        save_btn.clicked.connect(self._save_token)
        row.addWidget(save_btn)
        form.addRow("", row)

    def _save_token(self):
        token = self.token_edit.text().strip()
        if token and self.remember_check.isChecked():
            self.persistence.save_key("github", token)
        elif not token:
            self.persistence.forget_key("github")
