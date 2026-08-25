"""Sage Model Chain Studio — Ui Chats Favorites"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QLabel, QMessageBox, QInputDialog, QMenu,
    QAbstractItemView,
)
from .models import ChainStep, FavoriteModel
from .utils import step_display, _scope_for
from .persistence import PersistenceManager


class ChatsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Chats")
        self.on_select = None
        self.on_new = None
        self.on_rename = None
        self.on_delete = None
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, 1)

        btn_row = QHBoxLayout()
        new_btn = QPushButton("+ New")
        new_btn.setObjectName("accent")
        new_btn.clicked.connect(self._new_chat)
        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_chat)
        delete_btn = QPushButton("Delete")
        delete_btn.setObjectName("danger")
        delete_btn.clicked.connect(self._delete_chat)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(rename_btn)
        btn_row.addWidget(delete_btn)
        layout.addLayout(btn_row)
        self._suppress_signal = False

    def populate(self, chats, select_id=None):
        self._suppress_signal = True
        self.list_widget.clear()
        for chat in chats:
            item = QListWidgetItem(chat.name)
            item.setData(Qt.ItemDataRole.UserRole, chat.id)
            self.list_widget.addItem(item)
            if chat.id == select_id:
                self.list_widget.setCurrentItem(item)
        self._suppress_signal = False

    def _on_selection_changed(self, current, previous):
        if self._suppress_signal or current is None:
            return
        chat_id = current.data(Qt.ItemDataRole.UserRole)
        if self.on_select:
            self.on_select(chat_id)

    def _new_chat(self):
        if self.on_new:
            self.on_new()

    def _rename_chat(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        new_name, ok = QInputDialog.getText(self, "Rename Chat", "Chat name:", text=item.text())
        if ok and new_name.strip() and self.on_rename:
            self.on_rename(chat_id, new_name.strip())

    def _delete_chat(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        chat_id = item.data(Qt.ItemDataRole.UserRole)
        confirm = QMessageBox.question(
            self, "Delete chat", f"Delete '{item.text()}' and all of its history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes and self.on_delete:
            self.on_delete(chat_id)


class FavoritesPanel(QGroupBox):
    def __init__(self, persistence, add_to_chain_callback):
        super().__init__("Favorites")
        self.persistence = persistence
        self.add_to_chain_callback = add_to_chain_callback
        self.favorites = self.persistence.load_favorites()
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget, 1)
        hint = QLabel("Right-click a model on the Models tab to add. Double-click to add to chain.")
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for fav in self.favorites:
            item = QListWidgetItem(f"\u2605 {fav.display_name}  ({fav.provider_label})")
            item.setData(Qt.ItemDataRole.UserRole, fav)
            self.list_widget.addItem(item)

    def add_favorite(self, fav):
        for existing in self.favorites:
            if existing.provider_label == fav.provider_label and existing.base_url == fav.base_url and existing.model_id == fav.model_id:
                return
        self.favorites.append(fav)
        self.persistence.save_favorites(self.favorites)
        self._refresh_list()

    def _remove_favorite(self, fav):
        self.favorites = [
            f for f in self.favorites
            if not (f.provider_label == fav.provider_label and f.base_url == fav.base_url and f.model_id == fav.model_id)
        ]
        self.persistence.save_favorites(self.favorites)
        self._refresh_list()

    def _build_step(self, fav):
        api_key = self.persistence.get_key(_scope_for(fav.provider_label, fav.base_url))
        return ChainStep(
            provider_label=fav.provider_label, kind=fav.kind, base_url=fav.base_url,
            api_key=api_key, model_id=fav.model_id, prompt_template="{input}",
            display_name=fav.display_name,
        )

    def _on_double_click(self, item):
        self.add_to_chain_callback(self._build_step(item.data(Qt.ItemDataRole.UserRole)))

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        fav = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        aa = menu.addAction("Add to Chain")
        ra = menu.addAction("Remove from Favorites")
        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen == aa:
            self.add_to_chain_callback(self._build_step(fav))
        elif chosen == ra:
            self._remove_favorite(fav)


