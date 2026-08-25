"""Sage Model Chain Studio — Main Window"""

import sys, os
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QSplitter, QTabWidget, QStatusBar,
)
from .constants import APP_NAME, DATA_DIR
from .persistence import PersistenceManager
from .models import ChatSession
from .ui_style import DARK_QSS
from .ui_model_browser import ModelBrowserPanel
from .ui_local_models import LocalModelsPanel
from .ui_chain_builder import ChainBuilderPanel
from .ui_run_panel import RunPanel
from .ui_voice_panel import VoicePanel
from .ui_chats_favorites import ChatsPanel, FavoritesPanel

from PyQt6.QtWidgets import QHBoxLayout
from datetime import datetime
import uuid


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 880)
        self.persistence = PersistenceManager()
        self.chats = self.persistence.load_chats()
        self.current_chat = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        title = QLabel(APP_NAME)
        title.setObjectName("chatTitle")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        subtitle = QLabel(
            "Chat: chats, favorites, chain, run. "
            "Models: API keys, model browsing & GPU-aware parameters. "
            "Voice: Kokoro TTS with GPU support."
        )
        subtitle.setObjectName("sectionHint")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        self.chain_panel = ChainBuilderPanel()
        self.chain_panel.on_change = self._save_current_chat

        # ---- Chat tab ----
        chat_tab = QWidget()
        chat_layout = QHBoxLayout(chat_tab)
        chat_splitter = QSplitter(Qt.Orientation.Horizontal)
        chat_layout.addWidget(chat_splitter)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setMaximumWidth(300)
        self.chats_panel = ChatsPanel()
        self.chats_panel.on_select = self._select_chat
        self.chats_panel.on_new = self._new_chat
        self.chats_panel.on_rename = self._rename_chat
        self.chats_panel.on_delete = self._delete_chat
        self.favorites_panel = FavoritesPanel(self.persistence, self.chain_panel.add_step)
        left_splitter.addWidget(self.chats_panel)
        left_splitter.addWidget(self.favorites_panel)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([1, 1])
        chat_splitter.addWidget(left_splitter)

        self.run_panel = RunPanel(lambda: self.chain_panel.steps)
        self.run_panel.on_run_finished = self._on_run_finished
        self.run_panel.on_chain_complete = self._on_chain_complete
        self.run_panel.on_auto_tts_toggled = self._on_auto_tts_toggled

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_split.addWidget(self.chain_panel)
        right_split.addWidget(self.run_panel)
        right_split.setStretchFactor(0, 0)
        right_split.setStretchFactor(1, 1)
        right_split.setSizes([220, 600])
        chat_splitter.addWidget(right_split)
        chat_splitter.setStretchFactor(0, 0)
        chat_splitter.setStretchFactor(1, 1)
        self.tabs.addTab(chat_tab, "\U0001F4AC Chat")

        # ---- Models tab ----
        models_tab = QWidget()
        models_layout = QHBoxLayout(models_tab)
        models_splitter = QSplitter(Qt.Orientation.Horizontal)
        models_layout.addWidget(models_splitter)
        self.browser_panel = ModelBrowserPanel(
            self.persistence, self.chain_panel.add_step, self.favorites_panel.add_favorite
        )
        self.local_models_panel = LocalModelsPanel(
            self.persistence, self.chain_panel.add_step, self.favorites_panel.add_favorite
        )
        models_splitter.addWidget(self.browser_panel)
        models_splitter.addWidget(self.local_models_panel)
        models_splitter.setStretchFactor(0, 1)
        models_splitter.setStretchFactor(1, 1)
        self.tabs.addTab(models_tab, "\U0001F9E0 Models")

        # ---- Voice tab ----
        self.voice_panel = VoicePanel(self.persistence, self._latest_chain_output)
        self.tabs.addTab(self.voice_panel, "\U0001F50A Voice")

        self._sync_local_params()
        voice_settings = self.persistence.load_voice_settings()
        self.run_panel.auto_tts_check.setChecked(bool(voice_settings.get("enabled", False)))

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(f"Data stored at {DATA_DIR}")

        if not self.chats:
            self._new_chat()
        else:
            self.chats_panel.populate(self.chats, select_id=self.chats[0].id)
            self._select_chat(self.chats[0].id)

    def _sync_local_params(self):
        params = {}
        for lm in self.local_models_panel.local_models:
            params[lm.path] = {
                "n_ctx": lm.n_ctx, "n_gpu_layers": lm.n_gpu_layers,
                "temperature": lm.temperature, "max_tokens": lm.max_tokens,
                "top_p": lm.top_p, "repeat_penalty": lm.repeat_penalty,
            }
        self.run_panel.set_local_model_params(params)

    def _on_auto_tts_toggled(self, checked):
        self.voice_panel.enabled_check.setChecked(checked)
        self.voice_panel.settings["enabled"] = checked
        self.persistence.save_voice_settings(self.voice_panel.settings)

    def _find_chat(self, chat_id):
        return next((c for c in self.chats if c.id == chat_id), None)

    def _new_chat(self):
        n = len(self.chats) + 1
        chat = ChatSession(
            id=str(uuid.uuid4()), name=f"Chat {n}",
            created_at=datetime.now().isoformat(),
        )
        self.chats.append(chat)
        self.persistence.save_chats(self.chats)
        self.chats_panel.populate(self.chats, select_id=chat.id)
        self._select_chat(chat.id)

    def _select_chat(self, chat_id):
        chat = self._find_chat(chat_id)
        if not chat:
            return
        self.current_chat = chat
        self.chain_panel.set_steps(chat.steps)
        self.run_panel.load_history(chat.history, chat.last_input)
        self.statusBar().showMessage(f"Editing '{chat.name}'")

    def _rename_chat(self, chat_id, new_name):
        chat = self._find_chat(chat_id)
        if not chat:
            return
        chat.name = new_name
        self.persistence.save_chats(self.chats)
        self.chats_panel.populate(self.chats, select_id=chat_id)

    def _delete_chat(self, chat_id):
        self.chats = [c for c in self.chats if c.id != chat_id]
        self.persistence.save_chats(self.chats)
        if not self.chats:
            self._new_chat()
        else:
            self.chats_panel.populate(self.chats, select_id=self.chats[0].id)
            self._select_chat(self.chats[0].id)

    def _save_current_chat(self):
        if self.current_chat is not None:
            self.current_chat.last_input = self.run_panel.input_edit.toPlainText()
            self.persistence.save_chats(self.chats)

    def _on_run_finished(self, record):
        if self.current_chat is not None:
            self.current_chat.history.append(record)
            self.current_chat.last_input = self.run_panel.input_edit.toPlainText()
            self.persistence.save_chats(self.chats)

    def _latest_chain_output(self):
        if self.current_chat and self.current_chat.history:
            record = self.current_chat.history[-1]
            for result in reversed(record.step_results):
                if result.get("output"):
                    return result["output"]
        return ""

    def _on_chain_complete(self, final_text):
        if hasattr(self, "voice_panel") and self.voice_panel.enabled_check.isChecked():
            self.voice_panel.speak_text(final_text)

    def closeEvent(self, event):
        self._save_current_chat()
        self.persistence.save_chats(self.chats)
        super().closeEvent(event)


