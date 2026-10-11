# Sage Model Chain Studio

A PyQt6 desktop application for chaining language models together and running them locally or through hosted APIs. It combines a model-chain runner with persistent chats, GPU-aware local inference (GGUF and Safetensors), automatic memory configuration, local text-to-speech, speech-to-text, music stem separation, image generation, video upscaling, GitHub commits, and a wrapper around the REA reverse-engineering tool, all in one dark-themed tabbed window.

This README is written so a fresh Windows PC with an NVIDIA GPU can be set up from scratch. Most of the troubleshooting sections come from problems that actually occurred while building this setup on an RTX 4070 Ti (12 GB), so they are specific rather than generic.

---

## Table of Contents

1. [Feature overview](#1-feature-overview)
2. [Quick start (new Windows PC)](#2-quick-start-new-windows-pc)
3. [Requirements](#3-requirements)
4. [Installation in detail](#4-installation-in-detail)
5. [Project structure](#5-project-structure)
6. [Patch scripts](#6-patch-scripts)
7. [Tabs and UI guide](#7-tabs-and-ui-guide)
8. [Local models and Auto-configuration](#8-local-models-and-auto-configuration)
9. [Choosing models for a 12 GB GPU](#9-choosing-models-for-a-12-gb-gpu)
10. [Persistent Session (long chats)](#10-persistent-session-long-chats)
11. [Data storage and security](#11-data-storage-and-security)
12. [Troubleshooting](#12-troubleshooting)
13. [Updating packages safely](#13-updating-packages-safely)
14. [Reading the terminal log](#14-reading-the-terminal-log)
15. [Development notes](#15-development-notes)
16. [Third-party components](#16-third-party-components)

---

## 1. Feature overview

- **Model chaining.** Any number of steps, each a remote or local model. Each step's output is fed into the next step's prompt template (use `{input}` in a template to place the previous output; without it the output is appended after the template).
- **Remote providers.** OpenRouter, NVIDIA NIM, Google Gemini, and any OpenAI-compatible custom endpoint. Free/paid detection per model, automatic retry with backoff on HTTP 429 (honouring `Retry-After` and `X-RateLimit-Reset` headers), and clear messages when a reply is cut off by the token limit.
- **Local models.**
  - GGUF through `llama-cpp-python` (CUDA build strongly recommended).
  - Safetensors through `transformers` and `torch`.
  - **Auto-configuration** of context length, GPU layers and KV-cache type from your free VRAM (section 8).
  - Automatic prompt-fit guard, automatic Alpaca prompt format for GGUF files that ship without a chat template, automatic unloading of previously loaded models.
- **Persistent chats and favourites.** Multiple named chats with full history, a favourites list for re-adding models to a chain, and a Persistent Session system (rolling summary plus Keep First) for very long conversations.
- **Voice (TTS).** Kokoro-82M text-to-speech, GPU or CPU, with text cleaning before synthesis and an Auto TTS option that speaks each final chain output.
- **SpeechText (STT).** NVIDIA Parakeet-TDT (fast, English) or faster-whisper (99 languages) with chunked processing so memory stays bounded.
- **MusicStem.** Demucs stem separation (vocals, drums, bass, other, plus guitar and piano with the 6-stem model).
- **Image.** Local image generation through `diffusers` (SD 1.5, SDXL, SD3, FLUX single-file or folder models), with text-to-image and image-to-image.
- **Video.** Real-ESRGAN denoise, sharpen and upscale with GFPGAN face restoration, a batch queue, per-file trim, scheduled start and live ETA.
- **GitHub.** Connect a repository, select files, describe a change, run it through your chain, and push the result as one atomic multi-file commit (optionally to a new branch).
- **REA.** A tab that installs, updates and runs the REA reverse-engineering CLI and can hand its output to your chat.
- **Prompt templates.** Save and reuse prompt templates.
- **Live status footer.** CPU %, VRAM used/total, GPU %, RAM used/total, current model activity and network state, refreshed every second.

---

## 2. Quick start (new Windows PC)

Follow these steps in order. Each is explained in detail in section 4.

1. Install **Python 3.11** (64-bit) and make sure `python` and `pip` work in a terminal.
2. Install the **Microsoft Visual C++ Redistributable** (x64).
3. Copy the project folder to the new PC.
4. `pip install PyQt6 requests psutil`
5. Install **PyTorch with the CUDA index** (never a bare `pip install torch`):
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`
6. Install **llama-cpp-python from the CUDA wheel index**:
   `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir`
7. `pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12`, then copy `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` next to `llama.dll` (section 4.3).
8. Install optional feature packages you need (section 4.4).
9. Run `python main.py`.
10. In **Settings > Local Models**, load a GGUF, leave **Auto-configure** ticked and check the status line.

Do **not** run a blanket "update all packages" command afterwards (section 13).

---

## 3. Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10/11 (primary target). Linux and macOS are possible but the GPU and DLL steps below are Windows-specific. |
| Python | 3.11, 64-bit |
| GPU | NVIDIA GPU with a recent driver for CUDA features. Developed on an RTX 4070 Ti, 12 GB. |
| Core packages | `PyQt6`, `requests` |
| Strongly recommended | `psutil` (footer RAM/CPU and process-RAM logging) |
| Local GGUF models | `llama-cpp-python` (CUDA build) |
| Torch features | CUDA build of `torch` (Image, Video, MusicStem, Voice on GPU, Safetensors) |
| ffmpeg | On `PATH` for Video (audio muxing) and SpeechText (decoding) |
| Node.js | 22.19+, 24.11+ or 26+ for the REA tab only |

Disk space: GGUF models are 4 to 14 GB each; Video, Demucs, Whisper, Kokoro and diffusers models download their own weights into `sage_studio/sage_data/` caches.

---

## 4. Installation in detail

### 4.1 Python and core dependencies

```
pip install PyQt6 requests psutil
python main.py
```

Everything else is optional and the app degrades gracefully, with in-app messages, when a package is missing. On a machine with an NVIDIA GPU, complete section 4.2 and 4.3 **before** installing the optional packages, because installing GPU packages with the wrong index or flags is the most common source of trouble.

### 4.2 GPU stack

The app has two independent GPU stacks that can each be silently downgraded to CPU-only builds:

- `llama-cpp-python` (local GGUF chat models)
- `torch` (Image, Video, MusicStem, Kokoro on GPU, Safetensors)

A plain `pip install` or `pip install --upgrade` of either package fetches the **CPU-only** build from PyPI, with no warning.

**1. Check the driver.**

```
nvidia-smi
```

The "CUDA Version" at the top right is the **maximum** your driver supports. Wheels tagged with that version or lower will work.

**2. Install the Visual C++ Redistributable.** Without it you can get generic "module or one of its dependencies could not be found" errors unrelated to CUDA.

```
winget install --id Microsoft.VCRedist.2015+.x64 --exact --source winget --accept-source-agreements --accept-package-agreements
```

**3. Install torch with the CUDA index.** Use `--index-url` (not `--extra-index-url`), so pip cannot fall back to the CPU build:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Replace `cu124` with the current tag from pytorch.org/get-started/locally that is at or below your driver's CUDA version. Verify:

```
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You want a version ending in `+cu124` (or similar), `True`, and your GPU name. A version ending in `+cpu` means the wrong build is installed.

**4. Install llama-cpp-python from the CUDA wheel index.**

```
pip uninstall -y llama-cpp-python
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

The CUDA and CPU wheels have the **same file name**; only the index they come from differs. `--force-reinstall --no-cache-dir` prevents pip from reusing a cached CPU wheel. The Local Models panel has a **Check GGUF GPU Support** button that reports which build you really have.

### 4.3 CUDA runtime DLLs for llama-cpp-python (Windows)

The `cu124` wheel's CUDA backend (`ggml-cuda.dll`) does not bundle `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll`. If they are not discoverable, `import llama_cpp` fails with:

```
FileNotFoundError: Could not find module '...\llama_cpp\lib\llama.dll' (or one of its dependencies)
```

The full NVIDIA CUDA Toolkit installer provides these DLLs, but it is large and can fail without a useful error. The reliable alternative is the pip-packaged redistributables:

```
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

Then copy the three DLLs next to `llama.dll` (adjust the Python folder if yours differs):

```
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cuda_runtime\bin\cudart64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin\cublasLt64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
```

Windows always searches a DLL's own folder first, so this works however the app is launched. `main.py` also registers the two `nvidia\...\bin` folders at startup with `os.add_dll_directory()` (before any other import) as a fallback:

```python
import os
for _dll_dir in [
    r"C:\Users\<you>\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cuda_runtime\bin",
    r"C:\Users\<you>\AppData\Local\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin",
]:
    if os.path.isdir(_dll_dir):
        os.add_dll_directory(_dll_dir)
```

Update the paths for the new PC's user name and Python version. This fallback only helps the Python process itself, not external tools.

Verify:

```
python -c "import llama_cpp; print('llama_cpp OK')"
```

Note: `namespace package` quirk. `import nvidia.cuda_runtime` followed by `.__file__` returns `None` because these are namespace packages. That is normal; locate the DLLs with `dir /s /b "...\site-packages\nvidia\cuda_runtime\bin\*.dll"` instead.

### 4.4 Optional feature installs

| Feature | Install |
|---|---|
| Local GGUF models (GPU) | Sections 4.2 and 4.3 in full |
| Local GGUF models (CPU only) | `pip install llama-cpp-python` (no GPU offload) |
| Safetensors local models | `pip install transformers accelerate safetensors` (plus CUDA `torch` for GPU) |
| Voice (Kokoro TTS) | `pip install kokoro soundfile numpy` (Linux also needs `apt install espeak-ng`); GPU needs CUDA `torch` |
| Image generation | `pip install diffusers accelerate` plus CUDA `torch` |
| Video cleanup / upscaling | `pip install opencv-python basicsr facexlib gfpgan realesrgan` plus ffmpeg on `PATH`; GPU needs CUDA `torch` |
| MusicStem | `pip install demucs`; GPU needs CUDA `torch` |
| SpeechText (faster-whisper) | `pip install faster-whisper soundfile numpy` plus ffmpeg on `PATH` |
| SpeechText (Parakeet) | The faster-whisper packages above plus `pip install git+https://github.com/huggingface/transformers` (needed until Parakeet-TDT ships in a stable transformers release) |
| Footer CPU/RAM readings and process-RAM logging | `pip install psutil` |
| Footer GPU/VRAM readings and Auto-configuration | NVIDIA driver with `nvidia-smi` on `PATH` (no pip package) |
| GitHub tab | A GitHub personal access token with `repo` scope, entered in the app |
| REA tab | Node.js 22.19+, 24.11+ or 26+, then install REA from inside the tab (section 7.10) |

After installing the Video packages, if you see the `torchvision.transforms.functional_tensor` error, run `fix_basicsr_torchvision.py` (section 12).

### 4.5 Hugging Face notices

The warning `You are sending unauthenticated requests to the HF Hub` is harmless. Set an `HF_TOKEN` environment variable if you want higher rate limits and faster downloads. The app redirects the Hugging Face cache for Kokoro into `sage_data/kokoro_model/`.

---

## 5. Project structure

```
SageModel/
├── main.py                              # Entry point. Registers Windows DLL search paths for the
│                                        #   NVIDIA runtime DLLs BEFORE other imports, then starts the app.
├── sage_studio/
│   ├── __init__.py
│   ├── constants.py                     # App constants, provider presets, data paths, model caches
│   │                                    #   (_GGUF_CACHE, _TRANSFORMERS_CACHE), CUDA wheel hint
│   ├── models.py                        # Dataclasses: ChatSession, ChainStep, LocalModel, FavoriteModel,
│   │                                    #   ModelInfo, RunRecord
│   ├── persistence.py                   # JSON load/save for chats, favourites, local models, API keys,
│   │                                    #   voice settings, prompt templates
│   ├── utils.py                         # CUDA / GGUF GPU-support detection, nvidia-smi readers, psutil
│   │                                    #   readers, TTS text cleaning, unload_all_local_models()
│   ├── auto_config.py                   # NEW: automatic GGUF configuration (section 8)
│   ├── workers.py                       # QThread workers: model fetching, ChatWorker (runs the chain,
│   │                                    #   local GGUF / Safetensors calls), VoiceWorker, ImageGenWorker
│   ├── ui_style.py                      # DARK_QSS stylesheet
│   ├── main_window.py                   # MainWindow: builds all tabs, footer, window-state persistence
│   ├── ui_model_browser.py              # Remote provider/model browser and API-key management
│   ├── ui_local_models.py               # Local Models panel (GGUF/Safetensors, parameters, Auto checkbox)
│   ├── ui_chain_builder.py              # Ordered chain steps: add, edit, reorder, remove
│   ├── ui_run_panel.py                  # Input box, Run/Stop, output log, Persistent Session controls,
│   │                                    #   context meter, tokens/s and VRAM readout, Auto TTS
│   ├── ui_chats_favorites.py            # Chat session list and favourites list
│   ├── ui_voice_panel.py                # Kokoro TTS panel
│   ├── ui_github_panel.py               # GitHub tab: repo connect, file select, AI-assisted commits
│   ├── ui_github_settings.py            # GitHub connection settings panel
│   ├── ui_prompt_templates.py           # Prompt template manager
│   ├── ui_hf_search.py                  # Hugging Face model search panel
│   ├── ui_image_panel.py                # Image generation tab (diffusers)
│   ├── ui_video_panel.py                # Video cleanup / upscaling tab (Real-ESRGAN, GFPGAN)
│   ├── ui_music_panel.py                # MusicStem tab (Demucs)
│   ├── ui_speech_panel.py               # SpeechText tab (Parakeet / faster-whisper)
│   ├── ui_rea_panel.py                  # NEW: REA tab (section 7.10)
│   └── sage_data/                       # Created at runtime (section 11)
├── add_*.py / fix_*.py / replace_*.py / remove_*.py / clear_*.py    # One-time patch scripts (section 6)
└── patch_*.py                                                       # Newer patch scripts (section 6)
```

Anything you copy to a new PC as a whole folder already contains the effects of the patch scripts. Run the scripts only against an unpatched copy of the source.

---

## 6. Patch scripts

Features were added to the project by small patch scripts that edit specific files in place. They are idempotent (safe to re-run), report what they changed, and most keep a backup.

| Script | Purpose | Backup / undo |
|---|---|---|
| `add_video_tab.py` | Registers the Video tab | - |
| `add_gpu_ram_footer.py` | Adds GPU % and RAM to the footer | - |
| `add_video_trim.py` | Adds trim start/end to the Video tab | - |
| `add_video_trim_hhmmss.py` | Converts trim fields to HH:MM:SS | - |
| `add_video_eta.py` | Adds ETA and fps readout to Video | - |
| `add_realesrnet_and_precision_log.py` | Adds RealESRNet_x4plus model and FP16/FP32 precision logging | - |
| `replace_video_panel_with_batch.py` | Replaces the Video panel with the batch-queue version (scheduled start) | - |
| `add_music_stem_tab.py` | Registers the MusicStem tab | - |
| `add_speech_text_tab.py` | Registers the SpeechText tab | - |
| `fix_output_log_formatting.py` | Word-wrap and follow-bottom autoscroll for the output log | - |
| `remove_output_log_hr_lines.py` | Removes divider lines from the output log | - |
| `clear_input_on_run.py` | Clears the input box on Run/Enter | - |
| `add_keep_first.py` | Adds "Keep First N turns" to Persistent Session | - |
| `fix_basicsr_torchvision.py` | Patches the installed `basicsr` package for newer `torchvision` | Re-run after reinstalling `torchvision` |
| `remove_local_models_hint.py` | Removes a hint label (example utility) | - |
| `patch_debug_local_gguf.py` | Adds a temporary `[DEBUG]` line after every local GGUF generation (prompt tokens, `n_ctx`, tokens out, finish reason, prompt tail) | `workers.py.bak`; `--undo` |
| `patch_gguf_alpaca_fallback_v2.py` | Uses the Alpaca prompt format for GGUF files with no chat template | `workers.py.alpacabak`; `--undo` |
| `patch_auto_config.py` | Wires `auto_config.py` into `workers.py`, `ui_run_panel.py` and `ui_local_models.py` | `*.autocfgbak`; `--undo` |
| `patch_use_mmap.py` | Turns memory-mapping off when every layer is on the GPU | `auto_config.py.mmapbak`; `--undo` |
| `patch_rea_tab.py` | Registers the REA tab | `main_window.py.reabak`; `--undo` |

Notes on the patch scripts:

- Run them from the `SageModel` folder (the folder that contains `sage_studio/`).
- Each sub-patch reports `APPLIED`, `SKIPPED (already applied)` or `FAILED` with the reason. A `FAILED - ... not found` message means the file has a different layout than the script expects; nothing is written for that step.
- `patch_auto_config.py` prints `patch_auto_config v2` on its first line. If it does not, you are running an older draft that looked for the wrong function names.
- `patch_auto_config.py` requires `auto_config.py` (v3 or newer) to be copied into `sage_studio/` first. `patch_use_mmap.py` requires the v3 or v4 `auto_config.py`.
- The source code is formatted in a canonical style (single-line statements, double or single quotes consistently), which is why the scripts can anchor on regular expressions. If you reformat the files, anchors may stop matching.
- `patch_debug_local_gguf.py` is a diagnostic. Remove it with `--undo` when you no longer need the `[DEBUG]` lines.
- Backup files (`*.bak`, `*.autocfgbak`, `*.alpacabak`, `*.mmapbak`, `*.reabak`) can be deleted once you are satisfied.

---

## 7. Tabs and UI guide

### 7.1 Chat

The main working area: a chat list on the left, the chain builder, and the Run panel.

- **Chats.** Create, rename, select and delete chats. Each chat stores its chain and full history in `chats.json`.
- **Chain builder.** Add steps from Settings (remote models, local models, favourites). Each step has a prompt template; `{input}` marks where the previous output goes. Reorder, edit and remove steps.
- **Run panel.**
  - *Initial input* box. Enter runs the chain; Shift+Enter inserts a newline. The box clears after a run starts.
  - *Auto-retry on 429* for remote providers.
  - *Persistent Session*, *Budget*, *Auto* and *Keep First* (section 10).
  - *Auto TTS*: speaks the final chain output with Kokoro.
  - *Run Chain* and *Stop*. Stop is cooperative, including during retry waits.
  - *Context meter*: an estimate of the prompt tokens (including session history) and the percentage of the first local model's saved Context Length.
  - *Output history log*: word-wrapped, auto-scrolling, Markdown rendered.
  - *Speed / VRAM line*: live tokens per second, token count and GPU memory.

### 7.2 Settings

- **Model browser** for remote providers with API-key storage, free/paid filtering and OpenRouter free-tier limit hints.
- **Local Models** panel (section 8): load GGUF files or Safetensors folders, per-model parameters, **Auto-configure** checkbox with a status line, **Check GGUF GPU Support**, **Unload All from VRAM/RAM**, remove from list, delete from disk.
- **Favourites** for quick re-adding.

Local model parameters:

| Parameter | Meaning | Default |
|---|---|---|
| Context Length | Context window in tokens. Disabled while Auto is on. | 4096 (manual default) |
| GPU Layers | Layers offloaded to the GPU; `-1` = all, `0` = CPU only. Disabled while Auto is on. | -1 |
| Temperature | Randomness | 0.7 |
| Max Tokens | Maximum reply length | 1024 |
| Top P | Nucleus sampling | 0.9 |
| Repeat Penalty | Penalises repetition | 1.1 |

Parameters apply the next time the model loads; saving them unloads the cached model automatically.

### 7.3 Voice

Kokoro-82M text-to-speech. Choose a voice (voices starting with `a` use American English, `b` British), speed, GPU on/off, and synthesise the latest chain output or pasted text. Text is cleaned first (Markdown, emoji, emoticons and stage directions are removed). WAV files are saved in `sage_data/audio/`.

Tip: with a large local model resident, turn **Kokoro's GPU option off** or keep a large VRAM reserve; Kokoro loads after the language model and takes VRAM that the model's sizing did not count (section 12).

### 7.4 GitHub

Connect a repository with a personal access token (`repo` scope), select files, describe a change, run it through your chain, review, and push as one atomic multi-file commit, optionally to a new branch. It uses its own chain run, independent of the Chat tab.

### 7.5 Templates

Save, edit and reuse prompt templates.

### 7.6 Image

Local image generation through `diffusers`: text-to-image and image-to-image, with SD 1.5, SDXL, SD3 and FLUX families, from single files or model folders. Pipelines stay cached in VRAM until unloaded. Needs CUDA `torch` for GPU speed.

### 7.7 Video

Real-ESRGAN denoise, sharpen and upscale with ready-made presets, GFPGAN face restoration, a batch queue, per-file HH:MM:SS trim, scheduled or overnight start, live ETA and a cancel-safe partial output. Needs CUDA `torch` for GPU speed.

### 7.8 MusicStem

Demucs stem separation. Models: `htdemucs_ft` (highest quality, default), `htdemucs` (fast), `htdemucs_6s` (adds guitar and piano). Vocals-only mode or pick individual stems.

### 7.9 SpeechText

Transcribe audio or video files with NVIDIA Parakeet-TDT (fast, English) or faster-whisper (tiny to large-v3, 99 languages). Input is split into uniform chunks (default 60 seconds) so VRAM stays bounded. Save as `.txt` or send straight to the Chat input.

### 7.10 REA

Wraps the REA reverse-engineering tool (github.com/morluto/rea, npm package `rea-agents`). REA inspects native binaries, JavaScript and Electron apps, .NET assemblies and websites and returns evidence-based findings. It runs locally and needs Node.js 22.19+, 24.11+ or 26+.

REA's headline workflow is asking an MCP-capable agent (Claude Code, Cursor and similar) to investigate a target. This app is not an MCP client, so the tab uses REA's **command-line** commands and lets you pass the results to your chain.

What the tab provides:

- **Setup status:** checks Node.js, npm and whether the `rea` command is installed. If it is not installed, commands run through `npx -y rea-agents@latest` (slower on first use).
- **Install REA** (`npm install --global rea-agents`) and **Update REA** (`rea update`) inside the tab.
- **Agent setup** opens `npx rea-agents setup` in its own terminal window (it is interactive). It is only needed if you also want REA inside another agent.
- **Analyze:** a preset for JavaScript and Electron apps (`rea analyze-javascript-application <path> --json`), plus a **Custom command** option where you type any arguments after `rea`. **List commands** runs `rea --help`. Other analysis types (native binaries, .NET and so on) go through the Custom option; see the REA docs for their command names.
- **Output** streams live. Copy, save, clear, or **Send to Chat input**, which truncates to a character limit (default 6000) because local models have limited context.
- A reminder to analyse only software you are authorised to examine. Static inspection reads files without running them; runtime-capture features run the target with your user permissions.

Registration: copy `ui_rea_panel.py` into `sage_studio/`, then run `python patch_rea_tab.py`. The tab label (`REA`, with a magnifier emoji) can be edited in `main_window.py`.

### 7.11 Footer

`CPU % | VRAM used/total | GPU % | RAM used/total (%) | Model: <state> | Network: <state>`. VRAM and GPU % need `nvidia-smi`; CPU and RAM need `psutil`. While a model loads or runs, the Model field shows the activity.

---

## 8. Local models and Auto-configuration

### 8.1 Why this exists

Local models fail in confusing ways when their settings do not match the machine: a context length that is too large makes the KV cache spill out of VRAM into system RAM (slow, high RAM), and a prompt that is too long for the context makes generation end silently. Auto-configuration (`sage_studio/auto_config.py`) picks settings from the model's file and your free VRAM, and guards against the common failures.

### 8.2 What it does, step by step

On every GGUF load (Auto on):

1. **Unloads other GGUF models.** Unless disabled, any other model in the cache is closed first, so the free-VRAM reading is honest and old models stop using VRAM and RAM.
2. **Reads the model header** with a metadata-only load (`vocab_only`): layer count, KV heads, head size and trained context. (llama.cpp may print `n_ctx_seq (512) > n_ctx_train (0)` during this read; it is harmless.)
3. **Reads free VRAM** from `nvidia-smi`.
4. **Chooses the largest context that fits** while keeping a free-VRAM reserve (default **2 GiB**), using this preference order:
   1. f16 KV cache, if that gives at least 4096 tokens (or the model's trained maximum, if smaller).
   2. q8_0 KV cache with flash attention, if that gives at least 2048 tokens.
   3. The same two options with the reserve **relaxed** down to `min_free_mib` (default 1 GiB) before any layer moves to the CPU, with the context capped at 4096 so more headroom is kept.
   4. Partial GPU offload as a last resort (layers are reduced until the estimate fits). This is slow and uses system RAM.
5. **Rounds context** down to a multiple of 256 and caps it at the model's trained context and 16384.
6. **Loads with automatic retries.** If loading fails, it retries with 75% of the context (down to 2048), then a q8 KV cache, then 75% of the GPU layers, and reports each retry as a warning in the log.
7. **Disables memory-mapping** when every layer is on the GPU (see 8.6).
8. **Logs the result** (section 14), including estimated versus actual VRAM and process RAM.

The sizing arithmetic:

```
free budget = free VRAM - reserve - model file size - 450 MiB compute buffer
KV per token (MiB) = layers x KV heads x (key size + value size) x bytes per value / 1 MiB
                     (2 bytes per value for f16, about 1.06 for q8_0)
context = free budget / KV per token, rounded down to a multiple of 256
```

Example, MythoMax-L2-13B Q4_K_S: 40 layers, 40 KV heads, 128-wide keys and values give 0.78 MiB per token at f16, which matches a measured 3200 MiB KV cache at 4096 tokens. With about 11 GB free and the 7.4 GB file, the f16 option only reaches about 1,800 tokens, so Auto selects a q8 cache with about 3,584 tokens and every layer on the GPU, leaving about 2 GB free.

### 8.3 The Local Models panel

- **Auto-configure Context Length and GPU Layers (recommended)** checkbox per GGUF model, on by default. While on, the Context Length and GPU Layers boxes are disabled and show the computed values, and a status line explains the choice, for example:
  `Auto: 3584 ctx · all layers on GPU · q8 KV (1.5 GB) · est. 8.8 GB VRAM, about 2.0 GB left (target 2.0 GB).`
- Untick Auto to type your own values. Manual mode uses the saved Context Length and GPU Layers with an f16 cache and no automatic retries.
- Changing the Auto setting unloads the cached model so the next run reloads with the new configuration.
- The manual default Context Length is **4096** (it used to be 16000, which cannot fit beside a 13B model on a 12 GB card).

### 8.4 Settings file

`sage_studio/sage_data/auto_config.json`:

```json
{
  "settings": {
    "target_free_mib": 2048,
    "min_free_mib": 1024,
    "single_model": true,
    "use_mmap": false
  },
  "models": {
    "c:\\users\\you\\desktop\\sagemodel\\models\\model.gguf": { "auto": true }
  }
}
```

| Setting | Default | Meaning |
|---|---|---|
| `target_free_mib` | 2048 | Free VRAM to leave unused. Lower it to be more aggressive. |
| `min_free_mib` | 1024 | Lowest reserve the sizing may relax to before splitting a model between GPU and CPU. |
| `single_model` | true | Unload other cached GGUF models before loading a new one. |
| `use_mmap` | (unset) | If unset: memory-map only when part of the model runs on the CPU. Set `true` or `false` to force it. |

Per-model entries are written when you tick or untick Auto. Edit the file while the app is closed.

### 8.5 Prompt guard and token budget

Before every local GGUF generation the prompt is checked against the loaded context:

- **Max Tokens** is limited to half the context.
- About **160 tokens** are reserved for the chat-template wrapper.
- If the prompt still does not fit, the **middle** is trimmed (the first 25% and the most recent text are kept), a warning appears in the log, and generation proceeds.

The Persistent Session budget (section 10) is derived from the real loaded context when Auto budget is on:

```
budget = (context - max_tokens - template tokens - 160) x 0.85, minimum 300
```

### 8.6 Memory-mapping and system RAM

llama.cpp memory-maps the model file by default. On Windows the mapped pages can stay counted in the process's RAM even after the weights have been copied to VRAM, so a model that fits entirely on the GPU can still show as much system RAM as its file size. `patch_use_mmap.py` makes the loader pass `use_mmap=False` when all layers are on the GPU, which removed this effect in testing. With partial offload, memory-mapping stays on.

The log line after each load reports process RAM before and after, so you can verify the effect.

### 8.7 Prompt formats and stop sequences

- If the GGUF contains a chat template, llama.cpp uses it.
- If it does **not**, llama-cpp-python would fall back to the Llama-2 `[INST]` format. Alpaca-style models such as MythoMax can then return empty replies. `patch_gguf_alpaca_fallback_v2.py` switches such models to the Alpaca format after loading and prints `[INFO] No chat template in GGUF; using Alpaca format.`
- Generation stops when any of these strings appears: `</s>`, `<|im_end|>`, `<|eot_id|>`, `<|end_of_turn|>`, `[INST]`, `[/INST]`, `<|user|>`, `<|assistant|>`, `\nUser:`, `\nUSER:`, `\nHuman:`. A model that begins its reply with one of these ends immediately with zero tokens. Persona instructions in the chain step's prompt template help keep the model in the intended role.
- A bare prompt such as "hi" gives an Alpaca model almost nothing to work with and it may invent its own instruction. Put a short persona or instruction in the step's prompt template, for example: `You are a friendly storyteller. Reply only as the Assistant and never write the User's lines.\n\n{input}`.

### 8.8 Model caching

Loaded models stay cached in memory between runs. With `single_model` on (default), loading a different GGUF unloads the previous one. **Unload All from VRAM/RAM** in the Local Models panel frees everything, including the CUDA cache.

---

## 9. Choosing models for a 12 GB GPU

These figures are estimates from published file sizes. The app measures real values when it loads a model and prints them to the terminal.

### 9.1 Rules of thumb

- Aim for the whole model, its KV cache, a 450 MiB compute buffer and a free-VRAM reserve of 1 to 2 GB to fit in about 11 GB of free VRAM (Windows and a browser typically use around 1 GB).
- **Q4_K_S or Q4_K_M** is the sweet spot for 12 GB with 8B to 13B models. Q5 and above only fit smaller models.
- A model that "mostly" fits is slower than a smaller quant that fits fully, and fills system RAM. Prefer a smaller quant over partial offload.
- Quantizing the KV cache to q8_0 roughly halves its size. It needs flash attention and can degrade some models (one Mistral Nemo finetune's author warns about special-token leakage). Prefer an f16 cache when it fits.
- Keep Kokoro on the CPU, or leave a larger reserve, when the language model is large.

### 9.2 MythoMax-L2-13B (file sizes from the TheBloke GGUF repository)

| File | Size | Fit with a 2 GiB reserve |
|---|---|---|
| Q3_K_M | 6.34 GB | Full GPU, 4096 context (q8 cache), about 2.8 GB left |
| Q3_K_L | 6.93 GB | Full GPU, 4096 context (q8 cache), about 2.3 GB left |
| Q4_K_S | 7.41 GB | Full GPU, about 3,500 context (q8 cache), about 2.0 GB left |
| Q4_K_M | 7.87 GB | Full GPU, about 2,300 context (q8 cache), about 2.1 GB left |
| Q5_K_S | 8.97 GB | Only with the reserve relaxed to about 1 GB; about 2,300 context |
| Q5_K_M | 9.23 GB | Does not fit fully; part runs on the CPU |
| Q6_K and Q8_0 | 10.7 and 13.8 GB | Too large |

MythoMax is a Llama 2 model with a 4096-token trained context, so no quant can give genuinely longer context. Q4_K_S is the recommended balance.

### 9.3 Newer roleplay-oriented options that fit

| Model | Quant | Approx. size | Notes |
|---|---|---|---|
| MN-12B-Mag-Mell-R1 (Mistral Nemo merge) | Q4_K_M | 7.5 GB | Roleplay and worldbuilding merge; long native context; official Q4_K_M and Q6_K GGUFs exist |
| Rocinante 12B (Mistral Nemo finetune) | Q4_K_M or Q4_XS | roughly 6.7 GB at Q4_XS | Roleplay and storytelling finetune |
| Hermes 3 Llama 3.1 8B | Q4_K_M or Q6_K | about 4.9 GB / 6.6 GB | Smaller, lightly aligned, lots of headroom |

For Mistral Nemo 12B at Q4_K_M, the arithmetic gives roughly 8K tokens with an f16 cache (more with q8) while keeping a 2 GiB reserve.

### 9.4 Qwen3.5-9B

The Q8_0 file is about 9.5 GB, which does not fit fully beside a 2 GiB reserve on a 12 GB card, so Auto either relaxes the reserve to 1 GiB or splits the model between GPU and CPU (slow, RAM-heavy). **Q6_K** (roughly 7 to 8 GB) fits comfortably with room for context and Kokoro.

---

## 10. Persistent Session (long chats)

Local and remote models receive each run as a single fresh request. The only reason a later turn can refer to an earlier one is that the Run panel builds a transcript of earlier turns and prepends it to the prompt. The **Persistent Session** checkbox (on by default) controls this.

### 10.1 Controls

- **Persistent Session** on: each run prepends a rolling transcript (`User: ...` and the reply, per turn). Off: every message is sent alone with no memory of earlier turns.
- **Budget** and **Auto**: the number of tokens the transcript may use.
  - **Auto on (default):** for a first-step local GGUF model the budget follows the real loaded context (section 8.5). For local Safetensors models it is `Context Length - Max Tokens - 400` (minimum 500). For remote providers it is 8000. The Budget box is greyed out and ignored while Auto is on.
  - **Auto off:** the number in the Budget box is used as typed.
- **Keep First** (default 0 = off): protects the first N turns from ever being summarised or dropped, for example campaign set-up turns in a long story.

### 10.2 What happens when the budget is exceeded

1. The oldest turns that no longer fit are condensed into a **rolling summary** by an extra model call with the first step's model (you will see "Summarizing earlier turns..." in the log), then dropped from the transcript.
2. The summary is prepended as "Summary of earlier events".
3. The persona or rules text in a chain step's prompt template is **never** summarised; it is sent in full every turn.

Limitations:

- The rolling summary is **not saved to disk**. It resets when you restart the app or switch chats and is regenerated from the saved raw history the next time the budget is exceeded.
- `chats.json` keeps the full raw history indefinitely; the budget only controls what is sent to the model.
- A very small budget (for example, a 2000-token context) triggers a summary almost every turn, which doubles the model calls. Keep a context of at least 3,000 to 4,000 tokens for chat use.
- The estimate uses roughly four characters per token; the prompt guard (section 8.5) catches overruns.

---

## 11. Data storage and security

Everything the app stores lives in a portable folder next to the code, `sage_studio/sage_data/`, so nothing touches your system profile:

```
sage_data/
├── chats.json                 # All chats and full turn-by-turn history
├── api_keys.json              # API keys and tokens, stored in PLAIN TEXT
├── favorites.json             # Favourite models
├── local_models.json          # Registered local models and their parameters
├── voice_settings.json        # Kokoro settings
├── prompt_templates.json      # Saved prompt templates
├── window_state.json          # Window geometry and splitter sizes
├── auto_config.json           # Auto-configuration settings and per-model Auto flag
├── audio/                     # Generated TTS .wav files
├── kokoro_model/              # Kokoro / Hugging Face cache (HF_HUB_CACHE redirected here)
├── realesrgan_weights/        # Real-ESRGAN and GFPGAN weights
├── demucs_models/             # Demucs / torch cache (TORCH_HOME redirected here)
├── whisper_models/            # faster-whisper model cache
└── (image outputs and diffusers cache; see IMAGE_OUTPUT_DIR and DIFFUSERS_CACHE in constants.py)
```

Moving to a new PC: copy `sage_data/` to restore chats, favourites, local-model registrations and settings. Re-verify the GPU setup separately; it does not transfer. If a registered GGUF path differs on the new PC, re-add the model; per-model Auto flags are keyed by file path.

Security notes:

- `api_keys.json` stores API keys and the GitHub token **unencrypted**. Never commit `sage_data/` to a repository or share it. A suitable `.gitignore`:

```
sage_studio/sage_data/
*.bak
*.autocfgbak
*.alpacabak
*.mmapbak
*.reabak
__pycache__/
models/
```

- The REA tab runs external commands (`npm`, `npx`, `rea`) with your user permissions. Review what you run, and only analyse software you are authorised to examine.
- Anything sent from the REA or SpeechText tab to the Chat input goes to whichever model the chain uses, including remote providers.

---

## 12. Troubleshooting

### 12.1 Startup and installation

**`FileNotFoundError: Could not find module '...\llama_cpp\lib\llama.dll' (or one of its dependencies)` and the app will not start.**
`llama.dll` exists but a DLL it depends on is missing. This crashes the whole app, because the Local Models panel checks GPU support at startup.
1. Install the VC++ Redistributable (section 4.2).
2. Open the free **Dependencies** tool (github.com/lucasg/Dependencies), load `...\llama_cpp\lib\llama.dll`, and expand the tree under `ggml-cuda.dll` (CUDA support is a runtime plugin, so the top-level list can look fine). Entries shown in red are the missing DLLs, normally `cudart64_12.dll` and `cublas64_12.dll`.
3. Install `nvidia-cuda-runtime-cu12 nvidia-cublas-cu12` and copy the DLLs into `llama_cpp\lib\` (section 4.3). Note that Dependencies does not know about `os.add_dll_directory`, so it may still show them as missing after the app works.

**`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'` in the Video tab.**
`basicsr` uses an internal module removed in `torchvision` 0.17+. Run `fix_basicsr_torchvision.py`. It can resurface whenever `torchvision` is reinstalled; just run it again.

**Patch script says `FAILED - ... not found`.**
The file layout differs from what the script expects, or you are running an older draft. For `patch_auto_config.py`, the first output line must be `patch_auto_config v2`.

### 12.2 GPU is not being used

**Image, Video or MusicStem run on the CPU.**
`torch` was reinstalled as the CPU build, usually by an upgrade. Check `torch.cuda.is_available()` and the version suffix (`+cpu`). Reinstall with `--index-url https://download.pytorch.org/whl/cuXXX` (section 4.2).

**Local GGUF model runs on the CPU.**
`llama-cpp-python` is the CPU build. Use **Check GGUF GPU Support** and reinstall from the CUDA wheel index (section 4.2). The `GPU Layers` setting is silently ignored on CPU builds.

### 12.3 Memory problems

**About 3 tokens per second, very high system RAM from the first message.**
The context is too large, so the KV cache did not fit in VRAM. At roughly 0.78 MiB per token (13B-class, f16), a 16,000-token context needs about 12 GB for the cache alone. KV cache memory is reserved for the **full** context length when the model loads, even in a brand-new chat. Use Auto-configuration (section 8), or lower Context Length.

**System RAM is high even though the model fits in VRAM.**
Memory-mapping. Apply `patch_use_mmap.py`, or set `"use_mmap": false` in `auto_config.json`. Verify with the process-RAM numbers in the `[auto_config] loaded:` log line. Part of the figure can also be Kokoro and PyTorch, which load into the same process; compare Task Manager's "private working set" for `python.exe` before and after the first speech.

**The log says "Model too large for full offload: part of it runs on CPU".**
The model file plus cache plus reserve does not fit in free VRAM. Part of the model runs on the CPU and uses system RAM, which is much slower. Use a smaller quant (for example Q6_K instead of Q8_0), reduce other VRAM users, or lower `target_free_mib` / `min_free_mib`.

**Free VRAM looks lower than expected when a model is chosen.**
Another cached model, Kokoro, an image pipeline, or other apps hold VRAM. Click **Unload All**, close GPU-heavy apps, and turn Kokoro's GPU option off. With `single_model` on, the previous GGUF is unloaded automatically before a new one loads.

**VRAM is nearly full and everything is slow.**
On Windows the NVIDIA driver can silently spill allocations into system RAM instead of failing, so a too-large workload runs very slowly. Set **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback** for `python.exe` in the NVIDIA Control Panel (Manage 3D settings > Program Settings; driver 546.01 or newer). You then get a clear out-of-memory error instead of a silent slowdown. Also lower the context or choose a smaller quant.

**Load fails or the app aborts with an out-of-memory error.**
Auto mode retries with a smaller context, a q8 cache and fewer layers. If it still fails, raise `target_free_mib`, free VRAM, or choose a smaller quant. A hard out-of-memory abort inside the CUDA library cannot be caught; a larger reserve prevents it.

### 12.4 Replies are empty, stop early, or ramble

**The speed readout flashes 0 tokens per second and nothing appears; no error.**
Generation ended with zero tokens. Enable `patch_debug_local_gguf.py` and look at the `[DEBUG]` line: `tokens_out=0` with `finish='stop'` means the model emitted an end token or a stop string immediately. For GGUF files with no chat template, apply `patch_gguf_alpaca_fallback_v2.py` (section 8.7).

**The reply invents `User:` lines or answers questions you never asked.**
The model has no persona and little context, or it is writing the next turn itself. Put an instruction in the chain step's prompt template (section 8.7). Do not add `Instruction:` to the stop list: if the model starts its reply with that word, it would bring back empty replies.

**The chat stops answering after two or three messages.**
The Persistent Session budget was larger than the model's real context, so the prompt overflowed. Turn **Auto** on next to Budget (default after `patch_auto_config.py`) so the budget follows the real context. Turning Persistent Session off also avoids it, but removes conversational memory.

**A reply is cut off.**
It reached Max Tokens. Raise Max Tokens (and context if needed), or ask for shorter output.

### 12.5 Other

**GFPGAN face restoration is slow (about 1 to 3 fps).** It runs a second model per frame on top of Real-ESRGAN. Use it selectively or trim to a short test range.

**Real-ESRGAN "no resize" presets still cost full 4x compute.** The scale option only controls a cheap resize after the model; the network always computes at its native scale.

**Parakeet VRAM grows with clip length.** The SpeechText tab chunks all input (default 60 seconds) to keep it bounded.

**Footer shows no GPU or VRAM figures.** `nvidia-smi` is not on `PATH`, or the driver is not installed. CPU and RAM figures need `psutil`.

**REA tab: "Node.js is NOT supported".** Install Node.js 22.19+, 24.11+ or 26+. **"REA CLI: not installed"** is fine: commands run through `npx`, or use **Install REA**. `npx` downloads the package on first use, so the first run is slow.

---

## 13. Updating packages safely

**Do not run a blanket "update all packages" command in this environment.** `llama-cpp-python`, `torch`, `torchvision` and (for Parakeet) `transformers` are pinned to specific CUDA-matched or feature-matched builds. A generic `pip install --upgrade <package>` or a loop over `pip list --outdated`:

- replaces `llama-cpp-python` with the CPU-only PyPI build, or breaks its CUDA DLL chain;
- replaces `torch` with the CPU-only build;
- can reintroduce the `basicsr` / `torchvision` incompatibility;
- can swap the Parakeet-enabled `transformers` development install for a stable release without Parakeet.

To update something:

1. `pip freeze > requirements-backup.txt` first, for rollback.
2. Update packages outside the GPU stack individually (`requests`, `psutil`, and so on).
3. For `llama-cpp-python`, `torch` and `torchvision`, re-run the full install command from section 4.2 with the correct index URL instead of a bare `--upgrade`.
4. Verify with `torch.cuda.is_available()`, `import llama_cpp`, and the **Check GGUF GPU Support** button. If the Video tab breaks, run `fix_basicsr_torchvision.py` again.

---

## 14. Reading the terminal log

Run `python main.py` from a terminal to see these messages.

| Message | Meaning |
|---|---|
| `ggml_cuda_init: found 1 CUDA devices` | The CUDA backend loaded; your GPU was detected. |
| `[auto_config] Auto: <ctx> ctx · <layers> · <kv> KV (...) · est. ... VRAM, about ... left` | The configuration chosen for this load and the estimated free VRAM afterwards. |
| `[auto_config] manual: n_ctx=... n_gpu_layers=...` | Auto is off for this model; your values are used. |
| `[auto_config] loaded: ... VRAM used by load ... (estimated ...) | VRAM free now ... | process RAM a -> b GB` | Result after loading: actual versus estimated VRAM, and process RAM before and after. |
| `[auto_config] could not read GGUF metadata: ...` | The header could not be read; a conservative 2048-token context is used. |
| `Unloaded N other GGUF model(s) ...` | `single_model` unloaded cached models before loading. |
| `Model load failed (...); retrying with ...` | Automatic fallback in progress. |
| `Prompt was N tokens but only M fit ...` | The prompt guard trimmed the middle of the prompt. |
| `Max Tokens lowered from X to Y ...` | Max Tokens exceeded half the context and was reduced. |
| `[INFO] No chat template in GGUF; using Alpaca format.` | The Alpaca fallback is active for this model. |
| `[INFO] GGUF has its own chat template; leaving it as is.` | The model's built-in template is used. |
| `[DEBUG] prompt_tokens=... n_ctx=... max_tokens=... tokens_out=... finish='...' text_len=...` | Diagnostic from `patch_debug_local_gguf.py`. `finish='stop'` with `tokens_out=0` means an immediate stop. `finish='length'` means Max Tokens was reached. |
| `[DEBUG] prompt tail: '...'` | The last 300 characters of the prompt that was sent. |
| `llama_context: n_ctx_seq (512) > n_ctx_train (0) ...` | Harmless; produced by the metadata-only header read. |
| `You are sending unauthenticated requests to the HF Hub` | Harmless; set `HF_TOKEN` to remove it. Also marks Kokoro loading. |

---

## 15. Development notes

- **Architecture.** A `QThread` worker (`ChatWorker`) runs the chain; local GGUF calls go through `_call_local_gguf` in `workers.py`, which streams tokens with `create_chat_completion` and reports progress through signals. Remote calls use `requests`.
- **Model caches.** `_GGUF_CACHE` and `_TRANSFORMERS_CACHE` in `constants.py` hold loaded models keyed by file path. `auto_config.py` and the Local Models panel manage evictions.
- **Auto-configuration entry points.** `load_gguf_auto()` (loading with fallback), `suggest()` (sizing), `guard_prompt()` (prompt fitting), `auto_budget()` (session budget), `describe()` (status text). The Run panel and Local Models panel call these through small patches.
- **Adding a tab.**
  1. Create `sage_studio/ui_<name>_panel.py` with a `QWidget` subclass. Accept optional callbacks (for example `send_to_chat`).
  2. Register it in `MainWindow.__init__` with `self.tabs.addTab(self.<name>_panel, "<label>")`. To hand text to the chat, set `self.run_panel.input_edit.setPlainText(text)` and `self.tabs.setCurrentIndex(0)`.
  3. Ship a small patch script that inserts the registration after the last existing `self.tabs.addTab(...)` line, keeps a backup, supports `--undo`, and is idempotent through a marker comment (see `patch_rea_tab.py`).
- **Long-running external tools.** `ui_rea_panel.py` shows the pattern: `QProcess` with merged output channels, ANSI stripping, `taskkill /T` on stop for Windows process trees, and `cmd.exe /c` wrapping for `.cmd` shims such as `npm`.
- **Styling.** Buttons use object names from `ui_style.py` (`accent`, `stop`, `danger`); hint text uses `sectionHint`.
- **Testing without a GPU.** The sizing logic in `auto_config.py` is pure Python once `read_gguf_info` and `get_free_vram_mib` are replaced with stubs.

---

## 16. Third-party components

| Component | Used for |
|---|---|
| PyQt6 | User interface |
| llama.cpp and llama-cpp-python | Local GGUF inference |
| PyTorch, Hugging Face `transformers`, `diffusers`, `accelerate`, `safetensors` | Safetensors models, image generation, GPU compute |
| Kokoro-82M | Text-to-speech |
| NVIDIA Parakeet-TDT, faster-whisper | Speech-to-text |
| Demucs | Music stem separation |
| Real-ESRGAN, GFPGAN, basicsr, facexlib, OpenCV | Video upscaling and face restoration |
| ffmpeg | Media decoding and muxing |
| REA (`rea-agents` by morluto) | Reverse-engineering CLI used by the REA tab |
| Dependencies (lucasg) | Optional DLL diagnostic tool referenced in troubleshooting |

Each component has its own licence and terms; check them before redistributing models or outputs.
