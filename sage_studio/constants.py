"""Sage Model Chain Studio — Constants"""

import os
from typing import Dict, Any


APP_NAME = "Sage Model Chain Studio"


PROVIDER_PRESETS = {
    "OpenRouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "auth_style": "bearer"},
    "NVIDIA NIM": {"kind": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "auth_style": "bearer"},
    "Google Gemini": {"kind": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta", "auth_style": "query_key"},
    "Custom (OpenAI-compatible)": {"kind": "openai", "base_url": "", "auth_style": "bearer"},
}


LOCAL_GGUF_KIND = "local_gguf"


LOCAL_TRANSFORMERS_KIND = "local_transformers"


OR_FREE_RPM = 20


OR_FREE_RPD_NO_CREDITS = 50


OR_FREE_RPD_WITH_CREDITS = 1000


OR_FREE_CREDITS_THRESHOLD = 10


_GGUF_CACHE: Dict[str, Any] = {}


_TRANSFORMERS_CACHE: Dict[str, Any] = {}


CUDA_WHEEL_HINT = (
    "pip uninstall -y llama-cpp-python\n"
    "pip install llama-cpp-python --extra-index-url "
    "https://abetlen.github.io/llama-cpp-python/whl/cu124 "
    "--force-reinstall --no-cache-dir"
)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(APP_ROOT, "sage_data")


AUDIO_DIR = os.path.join(DATA_DIR, "audio")


KOKORO_MODEL_DIR = os.path.join(DATA_DIR, "kokoro_model")


CHATS_FILE = os.path.join(DATA_DIR, "chats.json")


KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")


FAVORITES_FILE = os.path.join(DATA_DIR, "favorites.json")


LOCAL_MODELS_FILE = os.path.join(DATA_DIR, "local_models.json")


VOICE_FILE = os.path.join(DATA_DIR, "voice_settings.json")


os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(KOKORO_MODEL_DIR, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", KOKORO_MODEL_DIR)

__all__ = [
    'APP_NAME', 'PROVIDER_PRESETS', 'LOCAL_GGUF_KIND',
    'LOCAL_TRANSFORMERS_KIND', 'OR_FREE_RPM', 'OR_FREE_RPD_NO_CREDITS',
    'OR_FREE_RPD_WITH_CREDITS', 'OR_FREE_CREDITS_THRESHOLD',
    '_GGUF_CACHE', '_TRANSFORMERS_CACHE', 'CUDA_WHEEL_HINT',
    'APP_ROOT', 'DATA_DIR', 'AUDIO_DIR', 'KOKORO_MODEL_DIR',
    'CHATS_FILE', 'KEYS_FILE', 'FAVORITES_FILE',
    'LOCAL_MODELS_FILE', 'VOICE_FILE',
]
