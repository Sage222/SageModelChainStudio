"""Sage Model Chain Studio — Utils"""

import os, re, warnings
from typing import Dict, Any
from .constants import *


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def llama_cpp_installed() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def llama_cpp_gpu_supported():
    """Returns (installed: bool, gpu_supported: bool, detail: str)."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False, False, "llama-cpp-python is not installed."
    try:
        from llama_cpp.llama_cpp import llama_supports_gpu_offload
        supported = bool(llama_supports_gpu_offload())
        return True, supported, "Checked via llama_supports_gpu_offload()."
    except Exception:
        pass
    try:
        from llama_cpp import Llama
        supported = bool(getattr(Llama, "_supports_gpu_offload", lambda: False)())
        return True, supported, "Checked via Llama._supports_gpu_offload()."
    except Exception as e:
        return True, False, f"llama-cpp-python is installed but GPU support could not be verified ({e}). Assume CPU-only unless you compiled it with CUDA yourself."


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


