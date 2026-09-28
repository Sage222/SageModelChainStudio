"""Sage Model Chain Studio — Constants"""

import os
from typing import Dict, Any

APP_NAME = "Sage LLM Studio"

PROVIDER_PRESETS = {
    "OpenRouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "auth_style": "bearer"},
    "NVIDIA NIM": {"kind": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "auth_style": "bearer"},
    "Google Gemini": {"kind": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta", "auth_style": "query_key"},
    "Custom (OpenAI-compatible)": {"kind": "openai", "base_url": "", "auth_style": "bearer"},
}

LOCAL_GGUF_KIND = "local_gguf"

LOCAL_TRANSFORMERS_KIND = "local_transformers"

# Image generation model kinds -- diffusers-format folder (model_index.json +
# unet/vae/text_encoder subfolders) vs a single packed .safetensors checkpoint.
IMAGE_DIFFUSERS_FOLDER_KIND = "diffusers_folder"

IMAGE_DIFFUSERS_SINGLE_FILE_KIND = "diffusers_single_file"

IMAGE_PIPELINE_FAMILIES = ["sd15", "sdxl", "sd3", "flux"]

OR_FREE_RPM = 20

OR_FREE_RPD_NO_CREDITS = 50

OR_FREE_RPD_WITH_CREDITS = 1000

OR_FREE_CREDITS_THRESHOLD = 10

_GGUF_CACHE: Dict[str, Any] = {}

_TRANSFORMERS_CACHE: Dict[str, Any] = {}

# Cache of loaded diffusers pipelines, keyed by model path -- mirrors
# _GGUF_CACHE / _TRANSFORMERS_CACHE so repeated generations don't reload.
_DIFFUSERS_CACHE: Dict[str, Any] = {}

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

# Generated images (text-to-image outputs) live here, mirroring AUDIO_DIR.
IMAGE_OUTPUT_DIR = os.path.join(DATA_DIR, "generated_images")

CHATS_FILE = os.path.join(DATA_DIR, "chats.json")

KEYS_FILE = os.path.join(DATA_DIR, "api_keys.json")

FAVORITES_FILE = os.path.join(DATA_DIR, "favorites.json")

LOCAL_MODELS_FILE = os.path.join(DATA_DIR, "local_models.json")

IMAGE_MODELS_FILE = os.path.join(DATA_DIR, "image_models.json")

VOICE_FILE = os.path.join(DATA_DIR, "voice_settings.json")
PROMPT_TEMPLATES_FILE = os.path.join(DATA_DIR, "prompt_templates.json")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(KOKORO_MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", KOKORO_MODEL_DIR)
# Windows requires Developer Mode or admin rights to create symlinks;
# without this, huggingface_hub's cache system (used by diffusers for
# image models, and by Kokoro for voice) fails with:
#   OSError: [WinError 1314] A required privilege is not held by the client
# Disabling symlinks makes it copy files instead -- more disk space, no
# functional downside.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

__all__ = [
    'APP_NAME', 'PROVIDER_PRESETS', 'LOCAL_GGUF_KIND',
    'LOCAL_TRANSFORMERS_KIND', 'IMAGE_DIFFUSERS_FOLDER_KIND',
    'IMAGE_DIFFUSERS_SINGLE_FILE_KIND', 'IMAGE_PIPELINE_FAMILIES',
    'OR_FREE_RPM', 'OR_FREE_RPD_NO_CREDITS',
    'OR_FREE_RPD_WITH_CREDITS', 'OR_FREE_CREDITS_THRESHOLD',
    '_GGUF_CACHE', '_TRANSFORMERS_CACHE', '_DIFFUSERS_CACHE', 'CUDA_WHEEL_HINT',
    'APP_ROOT', 'DATA_DIR', 'AUDIO_DIR', 'KOKORO_MODEL_DIR', 'IMAGE_OUTPUT_DIR',
    'CHATS_FILE', 'KEYS_FILE', 'FAVORITES_FILE',
    'LOCAL_MODELS_FILE', 'IMAGE_MODELS_FILE', 'VOICE_FILE',
    'PROMPT_TEMPLATES_FILE',
    'REALESRGAN_MODEL_DIR', 'REALESRGAN_WEIGHTS', 'GFPGAN_WEIGHT_URL', 'REALESRGAN_PIP_HINT',
    'DEMUCS_MODEL_DIR', 'DEMUCS_PIP_HINT',
    'WHISPER_MODEL_DIR', 'SPEECH_PIP_HINT'
]


REALESRGAN_MODEL_DIR = os.path.join(DATA_DIR, "realesrgan_weights")
os.makedirs(REALESRGAN_MODEL_DIR, exist_ok=True)

REALESRGAN_WEIGHTS = {
    "realesr-general-x4v3": {
        "arch": "srvgg",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "wdn_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    },
    "RealESRGAN_x4plus": {
        "arch": "rrdbnet",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "RealESRNet_x4plus": {
        "arch": "rrdbnet",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    },
    "RealESRGAN_x4plus_anime_6B": {
        "arch": "rrdbnet_anime",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
}

GFPGAN_WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

REALESRGAN_PIP_HINT = (
    "pip install opencv-python basicsr facexlib gfpgan\n"
    "pip install realesrgan\n"
    "(also requires ffmpeg on your PATH for audio muxing -- "
    "https://ffmpeg.org/download.html)"
)


DEMUCS_MODEL_DIR = os.path.join(DATA_DIR, "demucs_models")
os.makedirs(DEMUCS_MODEL_DIR, exist_ok=True)
os.environ.setdefault("TORCH_HOME", DEMUCS_MODEL_DIR)

DEMUCS_PIP_HINT = (
    "pip install demucs\n"
    "(this also pulls in torch/torchaudio -- if you already have a CUDA build of torch "
    "installed for other features, demucs will reuse it and run on GPU automatically)"
)


WHISPER_MODEL_DIR = os.path.join(DATA_DIR, "whisper_models")
os.makedirs(WHISPER_MODEL_DIR, exist_ok=True)

SPEECH_PIP_HINT = (
    "pip install faster-whisper soundfile numpy\n"
    "pip install git+https://github.com/huggingface/transformers   # needed for Parakeet-TDT support\n"
    "(also requires ffmpeg on your PATH to decode input files -- https://ffmpeg.org/download.html)"
)
