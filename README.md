# Sage Model Chain Studio

A PyQt6 desktop application for browsing, chaining, and running AI models across multiple providers — OpenRouter, Google Gemini, NVIDIA NIM, custom OpenAI-compatible endpoints, and local GGUF/safetensors models — with GPU-aware local inference, persistent chat sessions, model favorites, and built-in Kokoro TTS for speech synthesis.

All settings, API keys, chat history, and generated audio live in a portable `sage_data/` folder beside the application. Nothing touches your global config or cache directories.

## Features

### Multi-Provider Model Browsing
- **OpenRouter** — browse hundreds of models, filter for free-tier, check per-key rate limits and remaining quota
- **Google Gemini** — list available Gemini models with free/paid detection (Flash/Gemma = free, Pro/Ultra = paid)
- **NVIDIA NIM** — browse NVIDIA's hosted inference models
- **Custom OpenAI-compatible** — connect to any endpoint that speaks the OpenAI API format (vLLM, Ollama, LM Studio, etc.)
- **Local GGUF** — load `.gguf` files with configurable context length, GPU layer offload, temperature, top-p, max tokens, and repeat penalty
- **Local Safetensors** — load HuggingFace model folders via `transformers` with automatic CUDA detection and float16

### Model Chaining
- Build a chain of multiple models that run sequentially — each step's output feeds into the next step's prompt
- Edit the prompt template for each step independently using `{input}` as a placeholder for the previous step's output
- Reorder, remove, and edit steps via drag-and-drop-style controls
- Auto-retry on HTTP 429 rate-limit errors with exponential backoff and `Retry-After` header support (up to 6 retries)

### Persistent Chat Sessions
- Multiple named chat sessions, each with its own chain configuration and full run history
- Sessions persist across restarts as JSON in `sage_data/chats.json`
- Each session remembers its chain steps, input history, and last input

### Favorites
- Right-click any model to add it to your favorites for quick access
- Favorites persist across sessions in `sage_data/favorites.json`

### GPU-Aware Local Inference
- **GGUF**: Offloads transformer layers to GPU VRAM via `llama-cpp-python`'s `n_gpu_layers` setting. A built-in **Check GGUF GPU Support** button verifies whether your installed build actually has CUDA support — a plain `pip install` is CPU-only and silently ignores the GPU setting
- **Safetensors**: Automatically uses `device_map='cuda'` with `torch.float16` when CUDA is available
- Models load lazily on first use and stay cached in VRAM/RAM — click **Unload All** to free memory
- Unloading also calls `torch.cuda.empty_cache()` to release fragmented VRAM

### Kokoro TTS (Voice Tab)
- Convert chain output (or any pasted text) to speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- GPU-accelerated synthesis when CUDA is available
- 11 built-in voices (American and British English)
- Adjustable speech speed (0.5x–2.0x)
- Text cleaning option that strips emoji, markdown formatting, emoticons, and stage directions before synthesis
- Auto-TTS mode that automatically synthesizes the final chain output after each run
- All generated WAV files saved to `sage_data/audio/`

### Dark Theme UI
- Custom dark stylesheet with a purple accent color scheme
- Color-coded output log: blue for input text, white for model output, red for errors, yellow for warnings/retries, green for completion
- Clean single-model mode: when only one model is in the chain, step headers and "Chain complete" messages are suppressed for a cleaner output

## Requirements

### Core (required)
```
pip install PyQt6 requests
```

### Local Inference (optional)
```
pip install llama-cpp-python          # GGUF models — see GPU note below
pip install transformers torch accelerate safetensors  # Safetensors models
```

### Voice / TTS (optional)
```
pip install kokoro soundfile numpy
pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU
```
Linux also requires: `sudo apt install espeak-ng`

## GGUF GPU Support

A plain `pip install llama-cpp-python` downloads a **CPU-only** wheel. The `n_gpu_layers` setting is silently ignored on that build — models will always run on CPU/system RAM regardless of what you set.

To get real GPU/VRAM offload, install a CUDA-enabled build:

```bash
# Easiest: prebuilt CUDA wheel (pick the tag matching your CUDA version)
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

Or compile from source with the CUDA Toolkit installed:
```bash
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Linux/macOS
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Use the **Check GGUF GPU Support** button on the Models tab to verify which kind of build you have installed.

## Installation

```bash
git clone https://github.com/Sage222/SageModelChainStudio.git
cd SageModelChainStudio
pip install PyQt6 requests

# Optional dependencies (see above)
pip install llama-cpp-python
pip install kokoro soundfile numpy

python main.py
```

## Usage

### Quick Start
1. Open the **Models** tab
2. Select a provider (OpenRouter, Gemini, NVIDIA NIM, or Custom), enter your API key, and click **Fetch Models**
3. Double-click a model to add it to the chain (or right-click to favorite it)
4. For local models, click **Load GGUF File** or **Load Safetensors Folder** and configure GPU parameters
5. Switch to the **Chat** tab
6. Type your prompt in the **Initial input** box
7. Click **Run Chain**
8. Each model runs sequentially, with output feeding into the next step
9. Optionally enable **Auto TTS** to automatically synthesize the final output

### Chain Example
1. Add `gpt-4o-mini` as Step 1 with prompt template: `Rewrite this as a poem: {input}`
2. Add `gemma-2-9b:free` as Step 2 with prompt template: `Translate into French: {input}`
3. Run with input "The weather is nice today"
4. Step 1 output (a poem) feeds into Step 2 (French translation)

### Voice Tab
1. Paste text or click **Use Latest Chain Output**
2. Select a voice and adjust speed
3. Enable **Clean text for speech** to strip emoji, markdown, and stage directions
4. Click **Speak & Save WAV**
5. Use **Play Last WAV** to replay, or **Delete Audio Files** to clear all generated WAVs

## Project Structure

```
SageModelChainStudio/
├── main.py                        # Entry point
├── sage_studio/
│   ├── __init__.py
│   ├── constants.py               # App config, paths, provider presets
│   ├── utils.py                   # CUDA detection, text cleaning, helpers
│   ├── models.py                  # Dataclasses (ChainStep, ChatSession, etc.)
│   ├── persistence.py             # JSON load/save for all data
│   ├── workers.py                 # QThread workers (fetch, chat, voice)
│   ├── ui_style.py                # Dark theme QSS stylesheet
│   ├── ui_model_browser.py        # Remote API model browser panel
│   ├── ui_local_models.py         # Local GGUF/safetensors panel
│   ├── ui_chain_builder.py        # Chain builder + step editor
│   ├── ui_run_panel.py            # Run controls + output log
│   ├── ui_voice_panel.py          # Kokoro TTS panel
│   ├── ui_chats_favorites.py      # Chat session + favorites panels
│   └── main_window.py             # Main window assembly
├── sage_model_chain_studio.py     # Original single-file version (preserved)
└── sage_data/                     # Created on first run (portable storage)
    ├── chats.json                 # Chat sessions + history
    ├── api_keys.json              # Saved API keys
    ├── favorites.json             # Favorite models
    ├── local_models.json          # Local model configs
    ├── voice_settings.json        # TTS settings
    ├── audio/                     # Generated WAV files
    └── kokoro_model/              # Kokoro TTS weights cache
```

## Data Storage

All application data is stored in a portable `sage_data/` folder beside the script:

| File | Contents |
|------|----------|
| `chats.json` | Chat sessions, chain steps, run history, last input |
| `api_keys.json` | Saved API keys (per provider) |
| `favorites.json` | Favorited models |
| `local_models.json` | Local model paths and parameters |
| `voice_settings.json` | TTS voice, speed, GPU, cleaning preferences |
| `audio/` | Generated WAV files from Kokoro TTS |
| `kokoro_model/` | Kokoro-82M weights (downloaded on first use via HuggingFace hub) |

Kokoro's model cache is scoped to `sage_data/kokoro_model/` via `HF_HUB_CACHE` — this only affects the Kokoro repo download, not your global HuggingFace cache.

## Kokoro Voices

| Voice | Gender | Accent |
|-------|--------|--------|
| `af_heart` | Female | American |
| `af_bella` | Female | American |
| `af_nicole` | Female | American |
| `af_sarah` | Female | American |
| `af_sky` | Female | American |
| `am_adam` | Male | American |
| `am_michael` | Male | American |
| `bf_emma` | Female | British |
| `bf_isabella` | Female | British |
| `bm_george` | Male | British |
| `bm_lewis` | Male | British |

## OpenRouter Free Tier

OpenRouter's free tier has the following limits:

- **20 requests/minute** on free models
- **50 requests/day** on free models (no credits)
- **1,000 requests/day** with 10+ credits purchased

The **Check Key Limits** button queries your OpenRouter key and displays your tier, usage, and remaining quota.

## Tech Stack

- **PyQt6** — GUI framework
- **requests** — HTTP client for API calls
- **llama-cpp-python** — Local GGUF inference (CUDA optional)
- **transformers + torch** — Local safetensors inference (CUDA optional)
- **Kokoro-82M** — Text-to-speech synthesis
- **soundfile + numpy** — Audio file writing

## License

This project is open source. See the repository for license details.
