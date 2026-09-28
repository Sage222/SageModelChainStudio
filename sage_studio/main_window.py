import time
import json
"""Sage Model Chain Studio — Main Window"""

import sys, os
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QScrollArea,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QSplitter, QTabWidget, QStatusBar,
)
from .utils import get_gpu_memory
from .utils import get_gpu_utilization, get_ram_usage
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
from .ui_github_panel import GitHubPanel
from .ui_prompt_templates import PromptTemplatesPanel
from .ui_image_panel import ImagePanel
from .ui_video_panel import VideoPanel
from .ui_music_panel import MusicStemPanel
from .ui_speech_panel import SpeechTextPanel

from PyQt6.QtWidgets import QHBoxLayout
from datetime import datetime
import uuid


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 880)
        self.setMaximumHeight(1440)
        self.persistence = PersistenceManager()
        self.chats = self.persistence.load_chats()
        self.current_chat = None
        self._transient_status = None
        self._transient_status_until = 0.0

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        self.chain_panel = ChainBuilderPanel(self.persistence)
        self.chain_panel.on_change = self._save_current_chat

        # ---- Chat tab ----
        chat_tab = QWidget()
        chat_layout = QHBoxLayout(chat_tab)
        chat_splitter = QSplitter(Qt.Orientation.Horizontal)
        chat_splitter.setObjectName("chat_splitter")
        chat_layout.addWidget(chat_splitter)

        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setObjectName("chat_left_splitter")
        left_splitter.setMaximumWidth(300)
        self.chats_panel = ChatsPanel()
        self.chats_panel.on_select = self._select_chat
        self.chats_panel.on_new = self._new_chat
        self.chats_panel.on_rename = self._rename_chat
        self.chats_panel.on_delete = self._delete_chat
        self.favorites_panel = FavoritesPanel(
            self.persistence, self.chain_panel.add_step,
            get_local_models_callback=lambda: self.local_models_panel.local_models,
            get_chain_steps_callback=lambda: self.chain_panel.steps,
            edit_step_callback=self.chain_panel.edit_step_by_model_id,
            remove_step_by_model_id_callback=self.chain_panel.remove_step_by_model_id,
        )
        left_splitter.addWidget(self.chats_panel)
        chat_splitter.addWidget(left_splitter)

        self.run_panel = RunPanel(lambda: self.chain_panel.steps)
        self.run_panel.on_run_finished = self._on_run_finished
        self.run_panel.on_chain_complete = self._on_chain_complete
        self.run_panel.on_auto_tts_toggled = self._on_auto_tts_toggled

        chat_splitter.addWidget(self.run_panel)
        chat_splitter.setStretchFactor(0, 0)
        chat_splitter.setStretchFactor(1, 1)
        self.tabs.addTab(chat_tab, "\U0001F4AC Chat")

        # ---- Models tab ----
        models_tab = QWidget()
        models_layout = QVBoxLayout(models_tab)
        models_splitter = QSplitter(Qt.Orientation.Horizontal)
        models_splitter.setObjectName("models_splitter")
        models_layout.addWidget(models_splitter)
        self.browser_panel = ModelBrowserPanel(
            self.persistence, self.chain_panel.add_step, self.favorites_panel.add_favorite
        )
        self.local_models_panel = LocalModelsPanel(
            self.persistence, self.chain_panel.add_step, self.favorites_panel.add_favorite
        )

        self.local_models_panel.on_params_saved = self._sync_local_params
        self.local_models_panel.on_model_unloading = self._on_model_unloading

        
        

        models_splitter.addWidget(self.browser_panel)
        models_splitter.addWidget(self.local_models_panel)
        models_splitter.addWidget(self.favorites_panel)
        models_splitter.setStretchFactor(0, 1)
        models_splitter.setStretchFactor(1, 1)
        models_splitter.setStretchFactor(2, 1)
        models_splitter.setSizes([1, 1, 1])
        models_layout.addWidget(self.chain_panel)
        models_tab_scroll = QScrollArea()

        models_tab_scroll.setWidgetResizable(True)

        models_tab_scroll.setWidget(models_tab)

        self.tabs.addTab(models_tab_scroll, "\u2699\ufe0f Settings")

        # ---- Voice tab ----
        self.voice_panel = VoicePanel(self.persistence, self._latest_chain_output)
        self.tabs.addTab(self.voice_panel, "\U0001F50A Voice")
        # ---- GitHub tab ----
        self.github_panel = GitHubPanel(
            self.persistence,
            get_steps_callback=lambda: self.chain_panel.steps,
            get_local_model_params_callback=lambda: self.run_panel.local_model_params,
        )
        self.github_panel_scroll = QScrollArea()

        self.github_panel_scroll.setWidgetResizable(True)

        self.github_panel_scroll.setWidget(self.github_panel)

        self.tabs.addTab(self.github_panel_scroll, "\U0001F4C8 GitHub")

        self.templates_panel = PromptTemplatesPanel(self.persistence)
        self.tabs.addTab(self.templates_panel, "\U0001F4DD Templates")
        self.image_panel = ImagePanel(self.persistence)
        self.tabs.addTab(self.image_panel, "\U0001F5BC\uFE0F Image")
        self.video_panel = VideoPanel(self.persistence)
        self.tabs.addTab(self.video_panel, "🎬 Video")
        self.music_panel = MusicStemPanel(self.persistence)
        self.tabs.addTab(self.music_panel, "🎵 MusicStem")
        self.speech_panel = SpeechTextPanel(self.persistence, set_chain_input_callback=self._set_chain_input)
        self.tabs.addTab(self.speech_panel, "🎤 SpeechText")
        self._sync_local_params()
        voice_settings = self.persistence.load_voice_settings()
        self.run_panel.auto_tts_check.setChecked(bool(voice_settings.get("enabled", False)))

        self.setStatusBar(QStatusBar())
        

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_live_status)
        self._status_timer.start(1000)
        self._update_live_status()
        if not self.chats:
            self._new_chat()
        else:
            self.chats_panel.populate(self.chats, select_id=self.chats[0].id)
            self._select_chat(self.chats[0].id)
        self._restore_window_state()

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
        if hasattr(self, "favorites_panel"):
            self.favorites_panel.refresh_loaded_list()
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

    def _on_model_unloading(self, label):
        self._transient_status = f"Unloading {label}..."
        self._transient_status_until = time.time() + 2.5
        self.statusBar().showMessage(self._transient_status)

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


    def _set_chain_input(self, text):
        """Load text into the Run panel's input box (used by GitHub panel)."""
        self.run_panel.input_edit.setPlainText(text)


    def _window_state_file(self):
        return os.path.join(DATA_DIR, "window_state.json")


    def _update_live_status(self):
        if self._transient_status and time.time() < self._transient_status_until:
            self.statusBar().showMessage(self._transient_status)
            return
        parts = []
        try:
            import psutil
            cpu_pct = psutil.cpu_percent()
            parts.append(f"CPU: {cpu_pct:.0f}%")
        except ImportError:
            parts.append("CPU: n/a (pip install psutil)")

        mem = get_gpu_memory()
        if mem is not None:
            used, total = mem
            parts.append(f"VRAM: {used/1024:.1f}/{total/1024:.1f} GB")
        else:
            parts.append("VRAM: n/a")
        gpu_util = get_gpu_utilization()
        if gpu_util is not None:
            parts.append(f"GPU: {gpu_util:.0f}%")
        else:
            parts.append("GPU: n/a")

        ram = get_ram_usage()
        if ram is not None:
            used_ram, total_ram, ram_pct = ram
            parts.append(f"RAM: {used_ram:.1f}/{total_ram:.1f} GB ({ram_pct:.0f}%)")
        else:
            parts.append("RAM: n/a")

        candidates = []
        if hasattr(self, "run_panel"):
            candidates.append(self.run_panel)
        if hasattr(self, "github_panel"):
            candidates.append(self.github_panel)
        active = max(
            (p for p in candidates if getattr(p, "_last_update", 0)),
            key=lambda p: p._last_update,
            default=None,
        )
        if active is not None and (time.time() - active._last_update) < 10:
            parts.append(f"{active._last_tps:.1f} tok/s \u00b7 {active._last_tokens} tokens")

        model_status = None
        if hasattr(self, "run_panel") and hasattr(self.run_panel, "get_model_status"):
            model_status = self.run_panel.get_model_status()
        parts.append(f"Model: {model_status or 'Idle'}")

        network_status = None
        
        parts.append(f"Network: {network_status or 'idle'}")

        self.statusBar().showMessage("   |   ".join(parts))

    def _save_window_state(self):
        state = {
            "geometry": [self.x(), self.y(), self.width(), self.height()],
            "splitters": {
                s.objectName(): list(s.sizes())
                for s in self.findChildren(QSplitter)
                if s.objectName()
            },
        }
        try:
            with open(self._window_state_file(), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _restore_window_state(self):
        path = self._window_state_file()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return
        geo = state.get("geometry")
        if geo and len(geo) == 4:
            x, y, w, h = geo
            w = min(w, self.maximumHeight() and 99999 or w)
            h = min(h, self.maximumHeight()) if self.maximumHeight() else h
            self.setGeometry(x, y, w, h)
        saved_sizes = state.get("splitters", {})
        if isinstance(saved_sizes, dict):
            for splitter in self.findChildren(QSplitter):
                name = splitter.objectName()
                sizes = saved_sizes.get(name)
                if name and sizes:
                    splitter.setSizes(sizes)
        # else: old positional-list format from before this fix -- ignore it
        # rather than risk applying sizes to the wrong splitter.

    def closeEvent(self, event):
        self.speech_panel.stop_recording_for_close()
        self._save_window_state()
        self._save_current_chat()
        self.persistence.save_chats(self.chats)
        super().closeEvent(event)


