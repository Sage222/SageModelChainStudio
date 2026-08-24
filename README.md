# Sage Model Chain Studio

A PyQt6 desktop app for browsing, chaining, and running LLMs from multiple providers — remote APIs (OpenRouter, Google Gemini, NVIDIA NIM, or any OpenAI-compatible endpoint) and local models (GGUF via llama.cpp, safetensors via Transformers) — with GPU-aware inference, persistent multi-chat history, and local text-to-speech output via Kokoro. All data is stored portably next to the script itself.

---

## Features

### Chat tab
- **Multiple persistent chats** — create, rename, and delete chat sessions; each keeps its own model chain and full run history.
- **Favorites** — star any remote or local model for one-click reuse across chats (double-click to add to the current chain, right-click for more options).
- **Model Chain builder** — stack multiple models into a sequential pipeline where each step's output feeds into the next step's prompt (via a `{input}` placeholder in a custom prompt template). Reorder, edit, or remove steps; the chain list is height-capped (~5 rows visible) with a scrollbar for longer chains.
- **Run panel** — large input box (~8 lines), a big output/history log, a **Stop** button to cancel an in-flight chain, auto-retry on HTTP 429 (rate limits) with exponential backoff honoring `Retry-After`/`X-RateLimit-Reset` headers, and a one-click **Auto TTS** toggle that speaks the final output automatically.

### Models tab
- **Remote API browser** — pick a provider (OpenRouter, Google Gemini, NVIDIA NIM, or a custom OpenAI-compatible base URL), enter and optionally remember an API key, fetch the live model list, search/filter it, and flag models as FREE / PAID / unknown (OpenRouter pricing is checked exactly via its API; Gemini/NIM use best-effort heuristics).
- **OpenRouter key-limit checker** — shows your credit usage, remaining balance, and the exact free-tier rate limits (20 req/min; 50 or 1000 req/day depending on whether you've bought 10+ credits).
- **Local Models panel** — load `.gguf` files or Hugging Face-style safetensors folders, with per-model parameters:
  - Context Length, GPU Layers, Temperature, Max Tokens, Top P, Repeat Penalty — each with a tooltip explaining what it does and typical values.
  - **GPU verification** — a live status line shows whether your installed `llama-cpp-python` actually supports GPU offload (a plain `pip install` gives a CPU-only build that silently ignores GPU settings), with a **Check GGUF GPU Support** button that gives exact fix commands.
  - **Unload All from VRAM/RAM** — evicts every cached local model and calls `torch.cuda.empty_cache()` to free memory without restarting the app.
  - Runtime warnings appear in the chat log if you request GPU offload but your build can't actually do it.

### Voice tab
- **Kokoro TTS** (local, 82M-parameter, GPU-capable) — turn any chain output or pasted text into speech.
- 11 built-in voices, adjustable speed, GPU/CPU toggle, auto-play after synthesis.
- **Text cleaning** — strips emoji, `**bold**`, `*actions*`, `[stage directions]`, and `(narration)` before synthesis so only spoken dialogue is read aloud; a preview button shows the cleaned text before you commit.
- Generated WAV files are saved to `sage_data/audio/` with a one-click folder opener.

### Portability
Everything — chat history, API keys, favorites, local model configs, voice settings, and generated audio — lives in a `sage_data/` folder created **beside the script**, not in an OS-specific app-data directory. Move the `.py` file and its `sage_data/` folder together and nothing is lost.

---

## Prerequisites

### Core (required)
```bash
pip install PyQt6 requests
```

### Local GGUF models (optional)
```bash
pip install llama-cpp-python
```
⚠️ **Important:** a plain `pip install` above gives a **CPU-only** build. GPU offload (`n_gpu_layers`) will be silently ignored and models will load into system RAM instead of VRAM. To get real GPU acceleration:

```bash
# Easiest — prebuilt CUDA wheel (match the tag to your CUDA version)
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

Or compile from source (requires the CUDA Toolkit + a C++ compiler already installed):
```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```
```bash
# Linux/macOS
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Use the in-app **"Check GGUF GPU Support"** button (Models tab) to confirm which kind of build you actually have.

### Local safetensors models (optional)
```bash
pip install transformers torch accelerate safetensors
```
GPU acceleration works automatically if CUDA is available (`torch.cuda.is_available()`); the app loads the model with `device_map="cuda"` and `float16` precision when possible.

### Voice / Kokoro TTS (optional)
```bash
pip install kokoro soundfile numpy
```
- **Linux:** also install the system package `espeak-ng` (e.g. `sudo apt install espeak-ng`) — Kokoro uses it for phonemization.
- **GPU acceleration:** install a CUDA build of PyTorch, e.g.:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

---

## Running the App

```bash
python sage_model_chain_studio.py
```

On first launch it creates a `sage_data/` folder next to the script containing:

```
sage_model_chain_studio.py
sage_data/
├── chats.json           # all chat sessions + full run history
├── api_keys.json        # remembered API keys, keyed by provider/base URL
├── favorites.json        # starred models for quick reuse
├── local_models.json    # registered GGUF/safetensors models + their parameters
├── voice_settings.json  # Kokoro TTS preferences
└── audio/               # generated WAV files, timestamped
```

---

## Supported Providers

| Provider | Type | Notes |
|---|---|---|
| OpenRouter | Remote | Free-tier models auto-detected via `:free` suffix + zero pricing; live key-limit checking |
| Google Gemini | Remote | Free tier limited to Flash/Flash-Lite/Gemma models (Pro requires billing as of April 2026) |
| NVIDIA NIM | Remote | Free rate-limited catalog access via `build.nvidia.com` API keys |
| Custom (OpenAI-compatible) | Remote | Point at any `/v1/chat/completions`-compatible endpoint (local llama.cpp server, vLLM, etc.) |
| GGUF | Local | Via `llama-cpp-python`; runs fully offline |
| Safetensors | Local | Hugging Face-format folders via `transformers`; runs fully offline |

---

## Known Limitations

- API keys are stored as **plaintext JSON** in `sage_data/api_keys.json` — fine for a personal machine, but not suitable if the folder is shared or synced to untrusted locations. Swapping in the `keyring` package for OS-level secret storage would be the natural upgrade.
- Stopping a running chain takes effect between steps or during a retry wait, not mid-HTTP-request — an in-flight API call will finish before the stop is honored.
- Free/Paid model badges for Gemini and NVIDIA NIM are best-effort heuristics (Gemini has no pricing field in its API; NIM's catalog has no per-model price at all), while OpenRouter's badge is verified against actual API pricing data.
- Local safetensors inference loads the full model into memory each time it's first used in a session — very large models may need significant VRAM/RAM even with float16.
