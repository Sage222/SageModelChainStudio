"""Sage Model Chain Studio — Persistence"""

import os, json
from typing import Dict
from .constants import *
from .models import ChatSession, FavoriteModel, LocalModel

from dataclasses import asdict


class PersistenceManager:
    def __init__(self):
        self.keys: Dict[str, str] = {}
        self._load_keys()

    def _load_keys(self):
        if os.path.exists(KEYS_FILE):
            try:
                with open(KEYS_FILE, "r", encoding="utf-8") as f:
                    self.keys = json.load(f)
            except Exception:
                self.keys = {}

    def save_key(self, scope, api_key):
        self.keys[scope] = api_key
        try:
            with open(KEYS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.keys, f, indent=2)
        except Exception:
            pass

    def get_key(self, scope):
        return self.keys.get(scope, "")

    def forget_key(self, scope):
        if scope in self.keys:
            del self.keys[scope]
            try:
                with open(KEYS_FILE, "w", encoding="utf-8") as f:
                    json.dump(self.keys, f, indent=2)
            except Exception:
                pass

    def load_chats(self):
        if not os.path.exists(CHATS_FILE):
            return []
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [ChatSession.from_dict(d) for d in raw]
        except Exception:
            return []

    def save_chats(self, chats):
        try:
            with open(CHATS_FILE, "w", encoding="utf-8") as f:
                json.dump([c.to_dict() for c in chats], f, indent=2)
        except Exception:
            pass

    def load_favorites(self):
        if not os.path.exists(FAVORITES_FILE):
            return []
        try:
            with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [FavoriteModel(**d) for d in raw]
        except Exception:
            return []

    def save_favorites(self, favorites):
        try:
            with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
                json.dump([asdict(fv) for fv in favorites], f, indent=2)
        except Exception:
            pass

    def load_local_models(self):
        if not os.path.exists(LOCAL_MODELS_FILE):
            return []
        try:
            with open(LOCAL_MODELS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return [LocalModel.from_dict(d) for d in raw]
        except Exception:
            return []

    def save_local_models(self, models):
        try:
            with open(LOCAL_MODELS_FILE, "w", encoding="utf-8") as f:
                json.dump([m.to_dict() for m in models], f, indent=2)
        except Exception:
            pass

    def load_voice_settings(self):
        defaults = {
            "enabled": False, "engine": "Kokoro (local)", "voice": "af_heart",
            "speed": 1.0, "auto_play": True, "last_output": "",
            "use_gpu": True, "clean_text": True,
        }
        if not os.path.exists(VOICE_FILE):
            return defaults
        try:
            with open(VOICE_FILE, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
        except Exception:
            pass
        return defaults

    def save_voice_settings(self, settings):
        try:
            with open(VOICE_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass


