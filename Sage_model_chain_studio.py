"""
Sage Model Chain Studio
A PyQt6 desktop app to browse models across OpenRouter, Gemini, NVIDIA NIM,
custom OpenAI-compatible endpoints, or local GGUF/safetensors models, filter
for free models, chain multiple models together, manage multiple persistent
chat sessions, keep favorites, run local Kokoro TTS (GPU-accelerated), and
gracefully handle OpenRouter's credit/rate limits.

All settings, JSON stores, and generated audio live in a portable sage_data/
folder beside this script.

Requirements (core):
    pip install PyQt6 requests

Optional, for local inference:
    pip install llama-cpp-python                        # GGUF (CUDA build: CMAKE_ARGS="-DGGML_CUDA=on" pip install ...)
    pip install transformers torch accelerate safetensors  # safetensors

Optional, for Voice tab (Kokoro TTS):
    pip install kokoro soundfile numpy
    (GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121)
    (Linux: also apt install espeak-ng)

Run:
    python sage_model_chain_studio.py
"""

import sys, os, json, time, uuid, random, re, subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QListWidget, QListWidgetItem,
    QTextEdit, QSplitter, QCheckBox, QGroupBox, QFormLayout, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPlainTextEdit, QStatusBar, QDialog, QDialogButtonBox, QInputDialog,
    QMenu, QTabWidget, QFileDialog, QSlider, QSpinBox, QDoubleSpinBox
)

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

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_ROOT, "sage_data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

CHATS_FILE = os.path.join(DATA_DIR, "chats.json")
KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")
FAVORITES_FILE = os.path.join(DATA_DIR, "favorites.json")
LOCAL_MODELS_FILE = os.path.join(DATA_DIR, "local_models.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_settings.json")


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def unload_all_local_models():
    """Clear all cached local models from VRAM/RAM."""
    global _GGUF_CACHE, _TRANSFORMERS_CACHE
    cleared = []
    for mid, llm in _GGUF_CACHE.items():
        try:
            del llm
        except Exception:
            pass
        cleared.append(f"GGUF: {os.path.basename(mid)}")
    for mid, cached in _TRANSFORMERS_CACHE.items():
        try:
            del cached
        except Exception:
            pass
        cleared.append(f"Safetensors: {os.path.basename(mid.rstrip('/'))}")
    _GGUF_CACHE.clear()
    _TRANSFORMERS_CACHE.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return cleared


def clean_text_for_speech(text: str) -> str:
    text = re.sub(r'\*\*[^*]+\*\*', '', text)
    text = re.sub(r'\*[^*]+\*', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    emoticon_pat = r'[:;=8xX][-^]?[)DdPp/\\|Oo3]|<3|>:[(D]|[Bb][oO]|[xX][dD]'
    text = re.sub(emoticon_pat, '', text)
    emoji_pat = re.compile('['
        '\U0001F600-\U0001F64F''\U0001F300-\U0001F5FF''\U0001F680-\U0001F6FF'
        '\U0001F700-\U0001F77F''\U0001F780-\U0001F7FF''\U0001F800-\U0001F8FF'
        '\U0001F900-\U0001F9FF''\U0001FA00-\U0001FA6F''\U0001FA70-\U0001FAFF'
        '\U00002600-\U000027BF''\U0001F1E0-\U0001F1FF''\U00002B00-\U00002BFF'
        ']+', flags=re.UNICODE)
    text = emoji_pat.sub('', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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


@dataclass
class ModelInfo:
    id: str
    display_name: str
    provider_label: str
    free_status: str
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChainStep:
    provider_label: str
    kind: str
    base_url: str
    api_key: str
    model_id: str
    prompt_template: str
    display_name: str = ""

@dataclass
class FavoriteModel:
    provider_label: str
    kind: str
    base_url: str
    model_id: str
    display_name: str

@dataclass
class LocalModel:
    kind: str
    path: str
    display_name: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    repeat_penalty: float = 1.1

    def to_dict(self):
        return {
            "kind": self.kind, "path": self.path, "display_name": self.display_name,
            "n_ctx": self.n_ctx, "n_gpu_layers": self.n_gpu_layers,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "top_p": self.top_p, "repeat_penalty": self.repeat_penalty,
        }

    @staticmethod
    def from_dict(d):
        return LocalModel(
            kind=d.get("kind", "local_gguf"), path=d.get("path", ""),
            display_name=d.get("display_name", ""),
            n_ctx=d.get("n_ctx", 4096), n_gpu_layers=d.get("n_gpu_layers", -1),
            temperature=d.get("temperature", 0.7), max_tokens=d.get("max_tokens", 1024),
            top_p=d.get("top_p", 0.9), repeat_penalty=d.get("repeat_penalty", 1.1),
        )

@dataclass
class RunRecord:
    timestamp: str
    initial_input: str
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    stopped: bool = False

@dataclass
class ChatSession:
    id: str
    name: str
    created_at: str
    steps: List[ChainStep] = field(default_factory=list)
    history: List[RunRecord] = field(default_factory=list)
    last_input: str = ""

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "created_at": self.created_at,
            "steps": [asdict(s) for s in self.steps],
            "history": [asdict(h) for h in self.history],
            "last_input": self.last_input,
        }

    @staticmethod
    def from_dict(d):
        return ChatSession(
            id=d.get("id", str(uuid.uuid4())),
            name=d.get("name", "Untitled"),
            created_at=d.get("created_at", datetime.now().isoformat()),
            steps=[ChainStep(**s) for s in d.get("steps", [])],
            history=[RunRecord(**h) for h in d.get("history", [])],
            last_input=d.get("last_input", ""),
        )


def _is_openrouter_free(model):
    mid = model.get("id", "")
    pricing = model.get("pricing", {}) or {}
    try:
        pp = float(pricing.get("prompt", "1") or "1")
        cp = float(pricing.get("completion", "1") or "1")
    except (TypeError, ValueError):
        pp, cp = 1.0, 1.0
    if mid.endswith(":free") or (pp == 0 and cp == 0):
        return "free"
    return "paid"

def _is_nim_free(model):
    return "free"

def _is_gemini_free(name):
    n = name.lower()
    if "flash" in n or "gemma" in n:
        return "free"
    if "pro" in n or "ultra" in n or "deep-research" in n:
        return "paid"
    return "unknown"

def _scope_for(label, base_url):
    if label == "Custom (OpenAI-compatible)":
        return f"custom::{base_url}"
    return label

def _is_local_kind(kind):
    return kind in (LOCAL_GGUF_KIND, LOCAL_TRANSFORMERS_KIND)

def step_display(step):
    if step.display_name:
        return step.display_name
    if _is_local_kind(step.kind):
        return os.path.basename(step.model_id.rstrip("/\\"))
    return step.model_id


class FetchModelsWorker(QThread):
    finished_ok = pyqtSignal(list)
    finished_err = pyqtSignal(str)

    def __init__(self, provider_label, kind, base_url, api_key):
        super().__init__()
        self.provider_label = provider_label
        self.kind = kind
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def run(self):
        try:
            self.finished_ok.emit(self._fetch())
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")

    def _fetch(self):
        if self.kind == "gemini":
            return self._fetch_gemini()
        return self._fetch_openai()

    def _fetch_openai(self):
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.get(f"{self.base_url}/models", headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("data", data if isinstance(data, list) else [])
        results = []
        for m in items:
            mid = m.get("id") or m.get("name") or "unknown"
            if self.provider_label == "OpenRouter":
                fs = _is_openrouter_free(m)
            elif self.provider_label == "NVIDIA NIM":
                fs = _is_nim_free(m)
            else:
                fs = "unknown"
            results.append(ModelInfo(id=mid, display_name=mid, provider_label=self.provider_label, free_status=fs, raw=m))
        results.sort(key=lambda x: x.id.lower())
        return results

    def _fetch_gemini(self):
        url = f"{self.base_url}/models"
        params = {"key": self.api_key} if self.api_key else {}
        results = []
        page_token = None
        while True:
            p = dict(params)
            p["pageSize"] = 200
            if page_token:
                p["pageToken"] = page_token
            resp = requests.get(url, params=p, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("models", []):
                name = m.get("name", "unknown")
                short = name.split("/")[-1]
                if "generateContent" not in m.get("supportedGenerationMethods", []):
                    continue
                results.append(ModelInfo(id=short, display_name=short, provider_label="Google Gemini", free_status=_is_gemini_free(short), raw=m))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        results.sort(key=lambda x: x.id.lower())
        return results


class OpenRouterKeyInfoWorker(QThread):
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key

    def run(self):
        try:
            resp = requests.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=20)
            resp.raise_for_status()
            self.finished_ok.emit(resp.json().get("data", {}))
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


class ChatWorker(QThread):
    step_done = pyqtSignal(int, str)
    step_retry = pyqtSignal(int, int, float, str)
    step_error = pyqtSignal(int, str)
    step_stopped = pyqtSignal(int)
    step_loading = pyqtSignal(int, str)
    chain_finished = pyqtSignal()
    MAX_RETRIES = 6
    MAX_BACKOFF_SECS = 30
    SLEEP_QUANTUM = 0.2

    def __init__(self, steps, initial_input, retry_on_429, local_model_params=None):
        super().__init__()
        self.steps = steps
        self.initial_input = initial_input
        self.retry_on_429 = retry_on_429
        self.step_results = []
        self.local_model_params = local_model_params or {}
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _interruptible_sleep(self, seconds):
        slept = 0.0
        while slept < seconds:
            if self._stop_requested:
                return True
            chunk = min(self.SLEEP_QUANTUM, seconds - slept)
            time.sleep(chunk)
            slept += chunk
        return False

    def run(self):
        current_input = self.initial_input
        for idx, step in enumerate(self.steps):
            if self._stop_requested:
                self.step_stopped.emit(idx)
                return
            attempt = 0
            while True:
                if self._stop_requested:
                    self.step_stopped.emit(idx)
                    return
                try:
                    prompt = step.prompt_template.replace("{input}", current_input) \
                        if "{input}" in step.prompt_template \
                        else f"{step.prompt_template}\n\n{current_input}".strip()
                    if step.kind == "gemini":
                        output = self._call_gemini(step, prompt)
                    elif _is_local_kind(step.kind):
                        self.step_loading.emit(idx, step_display(step))
                        output = self._call_local(step, prompt)
                    else:
                        output = self._call_openai(step, prompt)
                    current_input = output
                    self.step_results.append({
                        "step_index": idx, "provider_label": step.provider_label,
                        "model_id": step.model_id, "display_name": step_display(step),
                        "output": output, "error": None, "retries": attempt,
                    })
                    self.step_done.emit(idx, output)
                    break
                except requests.exceptions.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status == 429 and self.retry_on_429 and attempt < self.MAX_RETRIES:
                        attempt += 1
                        wait, reason = self._retry_wait(e.response, attempt)
                        self.step_retry.emit(idx, attempt, wait, reason)
                        if self._interruptible_sleep(wait):
                            self.step_stopped.emit(idx)
                            return
                        continue
                    err_msg = self._friendly_http_error(e, status)
                    self.step_results.append({
                        "step_index": idx, "provider_label": step.provider_label,
                        "model_id": step.model_id, "display_name": step_display(step),
                        "output": None, "error": err_msg, "retries": attempt,
                    })
                    self.step_error.emit(idx, err_msg)
                    return
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    self.step_results.append({
                        "step_index": idx, "provider_label": step.provider_label,
                        "model_id": step.model_id, "display_name": step_display(step),
                        "output": None, "error": err_msg, "retries": attempt,
                    })
                    self.step_error.emit(idx, err_msg)
                    return
        self.chain_finished.emit()

    def _friendly_http_error(self, e, status):
        if status == 402:
            return f"HTTPError 402: Payment required. Raw: {e}"
        if status == 429:
            return f"HTTPError 429: Rate limited after retries. Raw: {e}"
        return f"HTTPError: {e}"

    def _retry_wait(self, resp, attempt):
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra), self.MAX_BACKOFF_SECS), "429 rate limited (Retry-After)"
                except ValueError:
                    pass
            rh = resp.headers.get("X-RateLimit-Reset")
            if rh:
                try:
                    re_epoch = float(rh) / 1000.0 if float(rh) > 1e12 else float(rh)
                    wait = max(0.0, re_epoch - time.time())
                    if wait > 0:
                        return min(wait, self.MAX_BACKOFF_SECS), "429 rate limited (X-RateLimit-Reset)"
                except ValueError:
                    pass
        base = min(2 ** attempt, self.MAX_BACKOFF_SECS)
        return base + random.uniform(0, 1.0), "429 rate limited (backoff)"

    def _call_openai(self, step, prompt):
        headers = {"Content-Type": "application/json"}
        if step.api_key:
            headers["Authorization"] = f"Bearer {step.api_key}"
        body = {"model": step.model_id, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(f"{step.base_url.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(self, step, prompt):
        url = f"{step.base_url.rstrip('/')}/models/{step.model_id}:generateContent"
        params = {"key": step.api_key} if step.api_key else {}
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, params=params, json=body, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates returned (possibly blocked by safety filters).")
        return "".join(p.get("text", "") for p in candidates[0]["content"]["parts"])

    def _call_local(self, step, prompt):
        params = self.local_model_params.get(step.model_id, {})
        if step.kind == LOCAL_GGUF_KIND:
            return self._call_local_gguf(step, prompt, params)
        return self._call_local_transformers(step, prompt, params)

    def _call_local_gguf(self, step, prompt, params):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python\n"
                "For GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
            )
        llm = _GGUF_CACHE.get(step.model_id)
        if llm is None:
            if not os.path.exists(step.model_id):
                raise RuntimeError(f"GGUF file not found: {step.model_id}")
            n_ctx = params.get("n_ctx", 4096)
            n_gpu = params.get("n_gpu_layers", -1)
            llm = Llama(model_path=step.model_id, n_ctx=n_ctx, n_gpu_layers=n_gpu, verbose=False)
            _GGUF_CACHE[step.model_id] = llm
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=params.get("max_tokens", 1024),
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            repeat_penalty=params.get("repeat_penalty", 1.1),
        )
        return result["choices"][0]["message"]["content"]

    def _call_local_transformers(self, step, prompt, params):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise RuntimeError(
                "transformers/torch are not installed. Run: "
                "pip install transformers torch accelerate safetensors"
            )
        cached = _TRANSFORMERS_CACHE.get(step.model_id)
        if cached is None:
            if not os.path.isdir(step.model_id):
                raise RuntimeError(f"Model folder not found: {step.model_id}")
            tokenizer = AutoTokenizer.from_pretrained(step.model_id)
            device_map = "auto"
            torch_dtype = torch.float16
            if _cuda_available():
                device_map = "cuda"
                torch_dtype = torch.float16
            model = AutoModelForCausalLM.from_pretrained(
                step.model_id, torch_dtype=torch_dtype, device_map=device_map
            )
            cached = (tokenizer, model)
            _TRANSFORMERS_CACHE[step.model_id] = cached
        tokenizer, model = cached
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model.generate(
            input_ids,
            max_new_tokens=params.get("max_tokens", 512),
            do_sample=True,
            temperature=params.get("temperature", 0.7),
            top_p=params.get("top_p", 0.9),
            repetition_penalty=params.get("repeat_penalty", 1.1),
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)


class VoiceWorker(QThread):
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, text, voice, speed, output_path, use_gpu, clean_text):
        super().__init__()
        self.text = text
        self.voice = voice
        self.speed = speed
        self.output_path = output_path
        self.use_gpu = use_gpu
        self.clean_text = clean_text

    def run(self):
        try:
            try:
                import numpy as np
                import soundfile as sf
                from kokoro import KPipeline
            except ImportError:
                raise RuntimeError(
                    "Kokoro not installed. Run: pip install kokoro soundfile numpy "
                    "(Linux: also apt install espeak-ng)"
                )
            text = self.text
            if self.clean_text:
                text = clean_text_for_speech(text)
            if not text.strip():
                raise RuntimeError("After cleaning, there is no text left to synthesize.")
            device = "cuda" if (self.use_gpu and _cuda_available()) else None
            lang = "a" if self.voice.startswith("a") else "b"
            pipeline = KPipeline(lang_code=lang, device=device)
            chunks = []
            for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
                chunks.append(audio)
            if not chunks:
                raise RuntimeError("Kokoro returned no audio.")
            sf.write(self.output_path, np.concatenate(chunks), 24000)
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


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


class LocalModelsPanel(QGroupBox):
    def __init__(self, persistence, add_to_chain_callback, add_to_favorites_callback):
        super().__init__("\U0001F4BE Local Models \u2014 GGUF / Safetensors")
        self.persistence = persistence
        self.add_to_chain_callback = add_to_chain_callback
        self.add_to_favorites_callback = add_to_favorites_callback
        self.local_models = self.persistence.load_local_models()
        layout = QVBoxLayout(self)

        gpu_status = "GPU (CUDA) available" if _cuda_available() else "CPU only"
        gpu_label = QLabel(f"<b>Device:</b> {gpu_status}")
        gpu_label.setObjectName("sectionHint")
        layout.addWidget(gpu_label)

        btn_row = QHBoxLayout()
        gguf_btn = QPushButton("+ Load GGUF File...")
        gguf_btn.setObjectName("accent")
        gguf_btn.clicked.connect(self._load_gguf)
        btn_row.addWidget(gguf_btn)
        st_btn = QPushButton("+ Load Safetensors Folder...")
        st_btn.setObjectName("accent")
        st_btn.clicked.connect(self._load_safetensors)
        btn_row.addWidget(st_btn)
        unload_btn = QPushButton("\u274C Unload All from VRAM/RAM")
        unload_btn.setObjectName("danger")
        unload_btn.clicked.connect(self._unload_all)
        btn_row.addWidget(unload_btn)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget, 1)

        params_group = QGroupBox("Model Parameters (applies to selected local model)")
        params_layout = QFormLayout(params_group)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 131072)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setToolTip(
            "Maximum context window in tokens.\n"
            "Larger = more text memory but more VRAM.\n"
            "KV cache scales linearly with this value.\n"
            "Typical: 2048-8192. Reduce if you get OOM errors."
        )
        params_layout.addRow("Context Length:", self.ctx_spin)

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 999)
        self.gpu_layers_spin.setValue(-1)
        self.gpu_layers_spin.setToolTip(
            "GGUF only: Number of transformer layers to offload to GPU VRAM.\n"
            "-1 = all layers (fastest, needs enough VRAM).\n"
            "0 = CPU only.\n"
            "For 7B Q4 model: ~33 layers fits in 6GB VRAM.\n"
            "If you get CUDA OOM, reduce this number.\n"
            "Safetensors models always use device_map='cuda' if available."
        )
        params_layout.addRow("GPU Layers:", self.gpu_layers_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setToolTip(
            "Controls randomness of output.\n"
            "0.0 = deterministic/greedy decoding.\n"
            "0.7 = balanced creativity (recommended).\n"
            "1.0+ = more random/creative.\n"
            ">1.2 = increasingly chaotic."
        )
        params_layout.addRow("Temperature:", self.temp_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 32768)
        self.max_tokens_spin.setValue(1024)
        self.max_tokens_spin.setToolTip(
            "Maximum number of tokens to generate per response.\n"
            "Higher = longer outputs but slower and more VRAM for KV cache.\n"
            "512 = concise, 1024 = standard, 2048+ = detailed.\n"
            "GGUF: also affects generation time linearly."
        )
        params_layout.addRow("Max Tokens:", self.max_tokens_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setToolTip(
            "Nucleus sampling: only consider tokens comprising the top P probability mass.\n"
            "0.9 = consider top 90% of probability mass.\n"
            "1.0 = disabled (all tokens considered).\n"
            "Lower = more focused/deterministic.\n"
            "Use temperature OR top_p, not both aggressively."
        )
        params_layout.addRow("Top P:", self.top_p_spin)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(0.5, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setValue(1.1)
        self.repeat_penalty_spin.setToolTip(
            "Penalizes repeated tokens to prevent loops.\n"
            "1.0 = no penalty.\n"
            "1.1 = mild (recommended default).\n"
            "1.3+ = strong (may reduce coherence).\n"
            "Increase if outputs are repetitive."
        )
        params_layout.addRow("Repeat Penalty:", self.repeat_penalty_spin)

        save_params_btn = QPushButton("Save Parameters to Selected Model")
        save_params_btn.clicked.connect(self._save_params)
        params_layout.addRow(save_params_btn)
        layout.addWidget(params_group)

        hint = QLabel(
            "GGUF loads to GPU VRAM via n_gpu_layers=-1 (all layers).\n"
            "Safetensors loads via device_map='cuda' with float16.\n"
            "Models load lazily on first use and stay cached.\n"
            "Click Unload to free VRAM/RAM.\n"
            "Double-click a model to add to chain.\n"
            "Install GGUF with GPU: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall"
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._refresh_list()

    def _load_gguf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a GGUF Model File", "", "GGUF Models (*.gguf)")
        if not path:
            return
        self._add_local_model(LocalModel(kind=LOCAL_GGUF_KIND, path=path, display_name=os.path.basename(path)))

    def _load_safetensors(self):
        path = QFileDialog.getExistingDirectory(self, "Select a Model Folder")
        if not path:
            return
        try:
            has_st = any(f.endswith(".safetensors") for f in os.listdir(path))
        except OSError:
            has_st = False
        if not has_st:
            QMessageBox.warning(self, "No .safetensors found", "That folder doesn't appear to contain .safetensors files. Added anyway.")
        name = os.path.basename(os.path.normpath(path)) or path
        self._add_local_model(LocalModel(kind=LOCAL_TRANSFORMERS_KIND, path=path, display_name=name))

    def _add_local_model(self, lm):
        for existing in self.local_models:
            if existing.path == lm.path:
                QMessageBox.information(self, "Already added", "This model is already in your local list.")
                return
        self.local_models.append(lm)
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for lm in self.local_models:
            kl = "GGUF" if lm.kind == LOCAL_GGUF_KIND else "Safetensors"
            gpu_info = f" GPU layers={lm.n_gpu_layers}" if lm.kind == LOCAL_GGUF_KIND else " GPU=auto"
            item = QListWidgetItem(f"[{kl}] {lm.display_name}  (ctx={lm.n_ctx}, temp={lm.temperature}{gpu_info})")
            item.setData(Qt.ItemDataRole.UserRole, lm)
            item.setToolTip(lm.path)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        self.ctx_spin.setValue(lm.n_ctx)
        self.gpu_layers_spin.setValue(lm.n_gpu_layers)
        self.temp_spin.setValue(lm.temperature)
        self.max_tokens_spin.setValue(lm.max_tokens)
        self.top_p_spin.setValue(lm.top_p)
        self.repeat_penalty_spin.setValue(lm.repeat_penalty)

    def _save_params(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a model first.")
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        lm.n_ctx = self.ctx_spin.value()
        lm.n_gpu_layers = self.gpu_layers_spin.value()
        lm.temperature = self.temp_spin.value()
        lm.max_tokens = self.max_tokens_spin.value()
        lm.top_p = self.top_p_spin.value()
        lm.repeat_penalty = self.repeat_penalty_spin.value()
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()
        self.list_widget.setCurrentRow(self.list_widget.currentRow())
        if lm.model_id in _GGUF_CACHE:
            del _GGUF_CACHE[lm.model_id]
        if lm.model_id in _TRANSFORMERS_CACHE:
            del _TRANSFORMERS_CACHE[lm.model_id]
        QMessageBox.information(self, "Saved", f"Parameters saved for {lm.display_name}.\nModel will reload with new parameters on next use.")

    def _unload_all(self):
        count = len(_GGUF_CACHE) + len(_TRANSFORMERS_CACHE)
        if count == 0:
            QMessageBox.information(self, "Nothing to unload", "No local models are currently loaded in VRAM/RAM.")
            return
        cleared = unload_all_local_models()
        QMessageBox.information(self, "Unloaded", f"Cleared {len(cleared)} model(s) from VRAM/RAM:\n" + "\n".join(cleared))

    def _favorite_label(self, lm):
        return "Local (GGUF)" if lm.kind == LOCAL_GGUF_KIND else "Local (Safetensors)"

    def _build_step(self, lm):
        return ChainStep(
            provider_label=self._favorite_label(lm), kind=lm.kind, base_url="", api_key="",
            model_id=lm.path, prompt_template="{input}", display_name=lm.display_name,
        )

    def _on_double_click(self, item):
        self.add_to_chain_callback(self._build_step(item.data(Qt.ItemDataRole.UserRole)))

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        aa = menu.addAction("Add to Chain")
        fa = menu.addAction("\u2605 Add to Favorites")
        ra = menu.addAction("Remove from List")
        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen == aa:
            self.add_to_chain_callback(self._build_step(lm))
        elif chosen == fa:
            self.add_to_favorites_callback(FavoriteModel(
                provider_label=self._favorite_label(lm), kind=lm.kind, base_url="",
                model_id=lm.path, display_name=lm.display_name,
            ))
        elif chosen == ra:
            if lm.path in _GGUF_CACHE:
                del _GGUF_CACHE[lm.path]
            if lm.path in _TRANSFORMERS_CACHE:
                del _TRANSFORMERS_CACHE[lm.path]
            self.local_models = [m for m in self.local_models if m.path != lm.path]
            self.persistence.save_local_models(self.local_models)
            self._refresh_list()


class StepEditDialog(QDialog):
    def __init__(self, step, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Step \u2014 {step_display(step)}")
        self.resize(480, 260)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Provider: {step.provider_label}    Model: {step_display(step)}"))
        layout.addWidget(QLabel("Prompt template. Use {input} for the previous step's output."))
        self.template_edit = QPlainTextEdit(step.prompt_template)
        layout.addWidget(self.template_edit, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_template(self):
        return self.template_edit.toPlainText()


class ChainBuilderPanel(QGroupBox):
    def __init__(self):
        super().__init__("Model Chain")
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

    def _edit_step(self):
        row = self._current_row()
        if row is None:
            return
        dlg = StepEditDialog(self.steps[row], self)
        if dlg.exec():
            self.steps[row].prompt_template = dlg.get_template()
            self._refresh_table()
            if self.on_change:
                self.on_change()

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


class RunPanel(QGroupBox):
    def __init__(self, get_steps_callback):
        super().__init__("Run")
        self.get_steps_callback = get_steps_callback
        self.worker = None
        self.on_run_finished = None
        self.on_chain_complete = None
        self.on_auto_tts_toggled = None
        self.local_model_params = {}
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Initial input:"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type the starting prompt for step 1...")
        self.input_edit.setFixedHeight(120)
        layout.addWidget(self.input_edit)

        options_row = QHBoxLayout()
        self.retry_check = QCheckBox("Auto-retry on 429")
        self.retry_check.setChecked(True)
        options_row.addWidget(self.retry_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6  Run Chain")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self.run_chain)
        run_row.addWidget(self.run_btn, 1)
        self.stop_btn = QPushButton("\u25a0  Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_chain)
        run_row.addWidget(self.stop_btn)
        self.auto_tts_check = QCheckBox("\U0001F50A Auto TTS")
        self.auto_tts_check.setToolTip("When enabled, the final chain output is automatically sent to Kokoro TTS.")
        self.auto_tts_check.toggled.connect(self._on_auto_tts_toggled)
        run_row.addWidget(self.auto_tts_check)
        layout.addLayout(run_row)

        layout.addWidget(QLabel("Output / history log:"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        layout.addWidget(self.output_log, 1)
        self._pending_record = None

    def set_local_model_params(self, params):
        self.local_model_params = params

    def _on_auto_tts_toggled(self, checked):
        if self.on_auto_tts_toggled:
            self.on_auto_tts_toggled(checked)

    def load_history(self, history, last_input=""):
        self.output_log.clear()
        for record in history:
            self._render_record(record)
        if last_input:
            self.input_edit.setPlainText(last_input)
        else:
            self.input_edit.clear()

    def _render_record(self, record):
        self.output_log.append(f"<span style='color:#757c8f'>[{record.timestamp}]</span> <b>Input:</b> {record.initial_input}")
        for res in record.step_results:
            idx = res["step_index"]
            label = res.get("display_name") or res.get("model_id")
            if res.get("error"):
                self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} ({label}) failed:</b> {res['error']}")
            else:
                retries = res.get("retries", 0)
                rn = f" <i style='color:#facc15'>(after {retries} retr{'y' if retries==1 else 'ies'})</i>" if retries else ""
                self.output_log.append(f"<b>\u2500\u2500 Step {idx + 1} \u2014 {label} \u2500\u2500</b>{rn}")
                self.output_log.append(res["output"])
        if record.stopped:
            self.output_log.append("<i style='color:#facc15'>Chain was stopped by the user.</i>")
        self.output_log.append("<hr>")

    def run_chain(self):
        steps = self.get_steps_callback()
        if not steps:
            QMessageBox.information(self, "No steps", "Add at least one model to the chain first.")
            return
        initial_input = self.input_edit.toPlainText().strip()
        if not initial_input:
            QMessageBox.information(self, "No input", "Type an initial prompt first.")
            return
        self._pending_record = RunRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            initial_input=initial_input, step_results=[],
        )
        self.output_log.append(f"<span style='color:#757c8f'>[{self._pending_record.timestamp}]</span> <b>Input:</b> {initial_input}")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("\u25a0  Stop")
        self.worker = ChatWorker(steps, initial_input, self.retry_check.isChecked(), self.local_model_params)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.step_retry.connect(self._on_step_retry)
        self.worker.step_error.connect(self._on_step_error)
        self.worker.step_stopped.connect(self._on_step_stopped)
        self.worker.step_loading.connect(self._on_step_loading)
        self.worker.chain_finished.connect(self._on_finished)
        self.worker.start()

    def stop_chain(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

    def _on_step_loading(self, idx, label):
        self.output_log.append(f"<i style='color:#60a5fa'>Step {idx + 1}: loading local model '{label}' into VRAM...</i>")

    def _on_step_retry(self, idx, attempt, wait, reason):
        self.output_log.append(f"<i style='color:#facc15'>Step {idx + 1}: {reason}, retry {attempt} in {wait:.1f}s...</i>")

    def _on_step_done(self, idx, output):
        self.output_log.append(f"<b>\u2500\u2500 Step {idx + 1} output \u2500\u2500</b>")
        self.output_log.append(output)
        self.output_log.append("")

    def _reset_run_controls(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("\u25b6  Run Chain")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("\u25a0  Stop")

    def _finalize_record(self, stopped=False):
        if self._pending_record is not None and self.worker is not None:
            self._pending_record.step_results = self.worker.step_results
            self._pending_record.stopped = stopped
            if self.on_run_finished:
                self.on_run_finished(self._pending_record)
            self._pending_record = None

    def _on_step_error(self, idx, err):
        self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} failed:</b> {err}")
        self._reset_run_controls()
        self._finalize_record(stopped=False)

    def _on_step_stopped(self, idx):
        self.output_log.append(f"<i style='color:#facc15'>Stopped before/at step {idx + 1} by user request.</i>")
        self.output_log.append("<hr>")
        self._reset_run_controls()
        self._finalize_record(stopped=True)

    def _on_finished(self):
        self.output_log.append("<b style='color:#4ade80'>Chain complete.</b>")
        self.output_log.append("<hr>")
        final_text = ""
        if self.worker is not None and self.worker.step_results:
            final_text = self.worker.step_results[-1].get("output") or ""
        self._reset_run_controls()
        self._finalize_record(stopped=False)
        if final_text and self.on_chain_complete:
            self.on_chain_complete(final_text)


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


class VoicePanel(QWidget):
    KOKORO_VOICES = [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
    ]

    def __init__(self, persistence, get_output_callback):
        super().__init__()
        self.persistence = persistence
        self.get_output_callback = get_output_callback
        self.settings = self.persistence.load_voice_settings()
        self.worker = None
        root = QVBoxLayout(self)

        title = QLabel("Voice Output")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        root.addWidget(title)

        gpu_status = "GPU (CUDA) available" if _cuda_available() else "CPU only"
        hint = QLabel(
            f"Turn chain output into speech with Kokoro TTS (82M). "
            f"All audio saved in sage_data/audio/. "
            f"Install: pip install kokoro soundfile numpy. <b>{gpu_status}</b>"
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        config = QGroupBox("TTS Settings")
        form = QFormLayout(config)

        self.enabled_check = QCheckBox("Auto-synthesize final chain output")
        self.enabled_check.setChecked(bool(self.settings.get("enabled", False)))
        self.enabled_check.toggled.connect(self._save_settings)
        form.addRow("Auto TTS:", self.enabled_check)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Kokoro (local)")
        form.addRow("Engine:", self.engine_combo)

        self.voice_combo = QComboBox()
        self.voice_combo.addItems(self.KOKORO_VOICES)
        saved_voice = self.settings.get("voice", "af_heart")
        if saved_voice in self.KOKORO_VOICES:
            self.voice_combo.setCurrentText(saved_voice)
        self.voice_combo.currentTextChanged.connect(self._save_settings)
        form.addRow("Voice:", self.voice_combo)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(int(float(self.settings.get("speed", 1.0)) * 100))
        self.speed_label = QLabel()
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_label)
        form.addRow("Speed:", speed_row)

        gpu_row = QHBoxLayout()
        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(bool(self.settings.get("use_gpu", True)))
        self.gpu_check.setToolTip("Uses CUDA if available. Falls back to CPU if not.")
        self.gpu_check.toggled.connect(self._save_settings)
        gpu_row.addWidget(self.gpu_check)
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setObjectName("sectionHint")
        gpu_row.addStretch(1)
        gpu_row.addWidget(self.gpu_status_label)
        form.addRow("Device:", gpu_row)

        self.auto_play_check = QCheckBox("Play WAV automatically after synthesis")
        self.auto_play_check.setChecked(bool(self.settings.get("auto_play", True)))
        self.auto_play_check.toggled.connect(self._save_settings)
        form.addRow("Playback:", self.auto_play_check)

        self.clean_check = QCheckBox("Clean text for speech (remove emoji, *actions*, **bold**, [narration])")
        self.clean_check.setChecked(bool(self.settings.get("clean_text", True)))
        self.clean_check.setToolTip(
            "Strips emoticons, Unicode emoji, markdown formatting (*text*, **text**), "
            "and stage directions like [laughs] or (whispers) before synthesis."
        )
        self.clean_check.toggled.connect(self._save_settings)
        form.addRow("Text cleaning:", self.clean_check)
        root.addWidget(config)

        self._update_gpu_status()
        self._on_speed_changed(self.speed_slider.value())

        text_group = QGroupBox("Text to Speak")
        text_layout = QVBoxLayout(text_group)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Paste text here, or click Use Latest Chain Output...")
        text_layout.addWidget(self.text_edit, 1)

        text_btns = QHBoxLayout()
        latest_btn = QPushButton("Use Latest Chain Output")
        latest_btn.clicked.connect(self._use_latest_output)
        text_btns.addWidget(latest_btn)
        clean_btn = QPushButton("Preview Cleaned Text")
        clean_btn.clicked.connect(self._preview_cleaned)
        text_btns.addWidget(clean_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.text_edit.clear)
        text_btns.addWidget(clear_btn)
        text_layout.addLayout(text_btns)
        root.addWidget(text_group, 1)

        action_row = QHBoxLayout()
        self.speak_btn = QPushButton("\U0001F50A Speak / Save WAV")
        self.speak_btn.setObjectName("accent")
        self.speak_btn.clicked.connect(self.synthesize)
        action_row.addWidget(self.speak_btn, 1)
        play_btn = QPushButton("Play Last WAV")
        play_btn.clicked.connect(self.play_last)
        action_row.addWidget(play_btn)
        open_btn = QPushButton("Open Audio Folder")
        open_btn.clicked.connect(self.open_audio_folder)
        action_row.addWidget(open_btn)
        root.addLayout(action_row)

        self.status = QLabel("Ready.")
        self.status.setObjectName("sectionHint")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _update_gpu_status(self):
        if _cuda_available():
            self.gpu_status_label.setText("CUDA detected \u2705")
            self.gpu_status_label.setStyleSheet("color: #4ade80;")
        else:
            self.gpu_status_label.setText("CUDA not detected \u274c")
            self.gpu_status_label.setStyleSheet("color: #f87171;")

    def _on_speed_changed(self, value):
        self.speed_label.setText(f"{value / 100:.2f}x")
        self._save_settings()

    def _save_settings(self, *args):
        self.settings.update({
            "enabled": self.enabled_check.isChecked(), "engine": "Kokoro (local)",
            "voice": self.voice_combo.currentText(), "speed": self.speed_slider.value() / 100,
            "auto_play": self.auto_play_check.isChecked(),
            "use_gpu": self.gpu_check.isChecked(), "clean_text": self.clean_check.isChecked(),
        })
        self.persistence.save_voice_settings(self.settings)

    def _use_latest_output(self):
        text = self.get_output_callback()
        if not text:
            QMessageBox.information(self, "No chain output", "Run a chain first, or paste text into the box.")
            return
        self.text_edit.setPlainText(text)

    def _preview_cleaned(self):
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No text", "Paste some text first.")
            return
        cleaned = clean_text_for_speech(text)
        self.text_edit.setPlainText(cleaned)
        self.status.setText(f"Cleaned: {len(text)} chars \u2192 {len(cleaned)} chars")

    def speak_text(self, text):
        if not text.strip():
            return
        self.text_edit.setPlainText(text)
        self.synthesize()

    def synthesize(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "No text", "Paste text or use the latest chain output first.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(AUDIO_DIR, f"sage_tts_{timestamp}.wav")
        self.speak_btn.setEnabled(False)
        self.speak_btn.setText("Synthesizing...")
        device_label = "GPU (CUDA)" if (self.gpu_check.isChecked() and _cuda_available()) else "CPU"
        self.status.setText(f"Loading Kokoro on {device_label} and generating audio...")
        self.worker = VoiceWorker(
            text, self.voice_combo.currentText(),
            self.speed_slider.value() / 100, output_path,
            self.gpu_check.isChecked(), self.clean_check.isChecked(),
        )
        self.worker.finished_ok.connect(self._on_voice_ok)
        self.worker.finished_err.connect(self._on_voice_err)
        self.worker.start()

    def _on_voice_ok(self, path):
        self.speak_btn.setEnabled(True)
        self.speak_btn.setText("\U0001F50A Speak / Save WAV")
        self.settings["last_output"] = path
        self.persistence.save_voice_settings(self.settings)
        self.status.setText(f"Saved: {os.path.basename(path)}")
        if self.auto_play_check.isChecked():
            self._play_path(path)

    def _on_voice_err(self, error):
        self.speak_btn.setEnabled(True)
        self.speak_btn.setText("\U0001F50A Speak / Save WAV")
        self.status.setText("Voice generation failed.")
        QMessageBox.critical(self, "Kokoro TTS failed", error)

    def _play_path(self, path):
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "No audio file", "No generated WAV file is available yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "Could not play audio", str(e))

    def play_last(self):
        self._play_path(self.settings.get("last_output", ""))

    def open_audio_folder(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(AUDIO_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", AUDIO_DIR])
            else:
                subprocess.Popen(["xdg-open", AUDIO_DIR])
        except Exception as e:
            QMessageBox.warning(self, "Could not open folder", str(e))


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
            "Models: API keys, model browsing & parameters. "
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


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
