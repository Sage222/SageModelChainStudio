"""Sage Model Chain Studio — Ui Model Browser"""

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QCheckBox, QGroupBox,
    QFormLayout, QMessageBox, QMenu, QAbstractItemView,
)
from .constants import PROVIDER_PRESETS
from .models import ModelInfo, ChainStep, FavoriteModel
from .utils import _scope_for, _is_openrouter_free, _is_nim_free, _is_gemini_free
from .workers import FetchModelsWorker, OpenRouterKeyInfoWorker


class ModelListItemWidget(QWidget):
    def __init__(self, model):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        nl = QLabel(model.display_name)
        nl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        layout.addWidget(nl, 1)
        bm = {"free": ("FREE", "badgeFree"), "paid": ("PAID", "badgePaid"), "unknown": ("?", "badgeUnknown")}
        t, on = bm.get(model.free_status, ("?", "badgeUnknown"))
        b = QLabel(t)
        b.setObjectName(on)
        layout.addWidget(b)


class ModelBrowserPanel(QGroupBox):
    def __init__(self, persistence, add_to_chain_callback, add_to_favorites_callback):
        super().__init__("\U0001F310 Remote API \u2014 Provider, Key & Model Selection")
        self.persistence = persistence
        self.add_to_chain_callback = add_to_chain_callback
        self.add_to_favorites_callback = add_to_favorites_callback
        self.all_models = []
        layout = QVBoxLayout(self)

        conn_form = QFormLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_PRESETS.keys()))
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        conn_form.addRow("Provider:", self.provider_combo)

        self.base_url_edit = QLineEdit(PROVIDER_PRESETS["OpenRouter"]["base_url"])
        self.base_url_edit.editingFinished.connect(self._load_saved_key_for_current_scope)
        conn_form.addRow("Base URL:", self.base_url_edit)

        key_row = QHBoxLayout()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("API key")
        key_row.addWidget(self.api_key_edit, 1)
        self.show_key_btn = QPushButton("\U0001F441")
        self.show_key_btn.setFixedWidth(36)
        self.show_key_btn.setCheckable(True)
        self.show_key_btn.toggled.connect(self._toggle_key_visibility)
        key_row.addWidget(self.show_key_btn)
        conn_form.addRow("API Key:", key_row)
        layout.addLayout(conn_form)

        remember_row = QHBoxLayout()
        self.remember_key_check = QCheckBox("Remember this key")
        self.remember_key_check.setChecked(True)
        remember_row.addWidget(self.remember_key_check)
        forget_btn = QPushButton("Forget Saved Key")
        forget_btn.setObjectName("danger")
        forget_btn.clicked.connect(self._forget_key)
        remember_row.addWidget(forget_btn)
        layout.addLayout(remember_row)

        action_row = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Models")
        self.fetch_btn.setObjectName("accent")
        self.fetch_btn.clicked.connect(self.fetch_models)
        action_row.addWidget(self.fetch_btn, 1)
        self.check_limits_btn = QPushButton("Check Key Limits")
        self.check_limits_btn.clicked.connect(self._check_openrouter_limits)
        action_row.addWidget(self.check_limits_btn)
        layout.addLayout(action_row)

        filter_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search models by name...")
        self.search_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search_edit, 1)
        self.free_only_check = QCheckBox("Free only")
        self.free_only_check.stateChanged.connect(self._apply_filter)
        filter_row.addWidget(self.free_only_check)
        layout.addLayout(filter_row)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.model_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.model_list, 1)

        hint = QLabel("Double-click a model to add it to the chain, or right-click to favorite it.")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        add_btn = QPushButton("Add Selected Model to Chain \u2192")
        add_btn.clicked.connect(self._add_selected_to_chain)
        layout.addWidget(add_btn)

        self.worker = None
        self.limits_worker = None
        self._load_saved_key_for_current_scope()

    def _toggle_key_visibility(self, checked):
        self.api_key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )

    def _current_scope(self):
        return _scope_for(self.provider_combo.currentText(), self.base_url_edit.text().strip())

    def _load_saved_key_for_current_scope(self):
        saved = self.persistence.get_key(self._current_scope())
        if saved:
            self.api_key_edit.setText(saved)

    def _forget_key(self):
        self.persistence.forget_key(self._current_scope())
        self.api_key_edit.clear()

    def _on_provider_changed(self, label):
        self.base_url_edit.setText(PROVIDER_PRESETS[label]["base_url"])
        self._load_saved_key_for_current_scope()

    def fetch_models(self):
        label = self.provider_combo.currentText()
        preset = PROVIDER_PRESETS[label]
        base_url = self.base_url_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        if not base_url:
            QMessageBox.warning(self, "Missing Base URL", "Please enter a base URL.")
            return
        if self.remember_key_check.isChecked() and api_key:
            self.persistence.save_key(self._current_scope(), api_key)
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("Fetching...")
        self.worker = FetchModelsWorker(label, preset["kind"], base_url, api_key)
        self.worker.finished_ok.connect(self._on_fetch_ok)
        self.worker.finished_err.connect(self._on_fetch_err)
        self.worker.start()

    def _on_fetch_ok(self, models):
        self.all_models = models
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch Models")
        self._apply_filter()

    def _on_fetch_err(self, msg):
        self.fetch_btn.setEnabled(True)
        self.fetch_btn.setText("Fetch Models")
        QMessageBox.critical(self, "Fetch failed", msg)

    def _check_openrouter_limits(self):
        if self.provider_combo.currentText() != "OpenRouter":
            QMessageBox.information(self, "OpenRouter only", "Key-limit checking supports OpenRouter only.")
            return
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Missing API Key", "Enter your OpenRouter API key first.")
            return
        self.check_limits_btn.setEnabled(False)
        self.check_limits_btn.setText("Checking...")
        self.limits_worker = OpenRouterKeyInfoWorker(api_key)
        self.limits_worker.finished_ok.connect(self._on_limits_ok)
        self.limits_worker.finished_err.connect(self._on_limits_err)
        self.limits_worker.start()

    def _on_limits_ok(self, data):
        self.check_limits_btn.setEnabled(True)
        self.check_limits_btn.setText("Check Key Limits")
        ft = data.get("is_free_tier", True)
        limit = data.get("limit")
        remaining = data.get("limit_remaining")
        usage = data.get("usage")
        rpd = OR_FREE_RPD_NO_CREDITS if ft else OR_FREE_RPD_WITH_CREDITS
        msg = (
            f"<b>Account:</b> {'free tier' if ft else 'has credits'}<br>"
            f"<b>Usage:</b> {usage}<br>"
            f"<b>Per-key limit:</b> {limit if limit is not None else 'unlimited'}<br>"
            f"<b>Remaining:</b> {remaining if remaining is not None else 'unlimited'}<br><br>"
            f"<b>Free model caps:</b> {OR_FREE_RPM} req/min, {rpd} req/day"
            + (" (buy 10+ credits for 1000/day)" if ft else "")
        )
        QMessageBox.information(self, "OpenRouter Key Limits", msg)

    def _on_limits_err(self, msg):
        self.check_limits_btn.setEnabled(True)
        self.check_limits_btn.setText("Check Key Limits")
        QMessageBox.critical(self, "Could not check limits", msg)

    def _apply_filter(self):
        query = self.search_edit.text().strip().lower()
        free_only = self.free_only_check.isChecked()
        self.model_list.clear()
        for m in self.all_models:
            if query and query not in m.display_name.lower():
                continue
            if free_only and m.free_status != "free":
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, m)
            item.setSizeHint(QSize(0, 34))
            self.model_list.addItem(item)
            self.model_list.setItemWidget(item, ModelListItemWidget(m))

    def _build_step_from_model(self, model):
        label = self.provider_combo.currentText()
        preset = PROVIDER_PRESETS[label]
        return ChainStep(
            provider_label=label, kind=preset["kind"],
            base_url=self.base_url_edit.text().strip(),
            api_key=self.api_key_edit.text().strip(),
            model_id=model.id, prompt_template="{input}",
            display_name=model.display_name,
        )

    def _add_selected_to_chain(self):
        item = self.model_list.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a model first.")
            return
        self.add_to_chain_callback(self._build_step_from_model(item.data(Qt.ItemDataRole.UserRole)))

    def _on_item_double_clicked(self, item):
        self.add_to_chain_callback(self._build_step_from_model(item.data(Qt.ItemDataRole.UserRole)))

    def _show_context_menu(self, pos):
        item = self.model_list.itemAt(pos)
        if not item:
            return
        model = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        fa = menu.addAction("\u2605 Add to Favorites")
        aa = menu.addAction("Add to Chain")
        chosen = menu.exec(self.model_list.mapToGlobal(pos))
        if chosen == fa:
            label = self.provider_combo.currentText()
            preset = PROVIDER_PRESETS[label]
            self.add_to_favorites_callback(FavoriteModel(
                provider_label=label, kind=preset["kind"],
                base_url=self.base_url_edit.text().strip(),
                model_id=model.id, display_name=model.display_name,
            ))
        elif chosen == aa:
            self.add_to_chain_callback(self._build_step_from_model(model))


