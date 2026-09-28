# Sage Model Chain Studio

A PyQt6 desktop app for chaining LLMs together, running local GGUF/Safetensors
models with GPU-aware controls, managing persistent chats and favorites,
pushing AI-assisted code changes straight to GitHub, generating speech with
local TTS, transcribing speech to text, separating music into stems,
generating images, and cleaning up / upscaling video with Real-ESRGAN — all
from one dark-themed tabbed interface.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [GPU Setup (Windows) — Read This First](#gpu-setup-windows--read-this-first)
- [Optional Feature Installs](#optional-feature-installs)
- [Data Storage](#data-storage)
- [Tabs Overview](#tabs-overview)
- [Persistent Session (Long-Running Chats)](#persistent-session-long-running-chats)
- [Known Issues & Fixes](#known-issues--fixes)
- [Updating Packages Safely](#updating-packages-safely)
- [Running the App](#running-the-app)

## Features

- **Model chaining** — string together multiple models (remote API or local)
  where each step's output feeds the next step's prompt.
- **Remote providers** — OpenRouter, NVIDIA NIM, Google Gemini, or any
  OpenAI-compatible custom endpoint, with free/paid model detection.
- **Local models** — GGUF (via `llama-cpp-python`) and Safetensors (via
  `transformers`), with GPU-layer control, live GPU-support detection, and
  per-model parameter presets (context length, temperature, top-p, etc.).
- **Persistent chats & favorites** — multiple named chat sessions with full
  history, a rolling-summary long-context system for very long chats (see
  below), and a favorites list for quickly re-adding models to a chain.
- **GitHub integration** — connect a repo, select files, describe a change,
  run it through your model chain, and push the result as an atomic
  multi-file commit (optionally to a new branch).
- **Voice (TTS)** — local, GPU-accelerated speech synthesis via Kokoro-82M,
  with text cleaning (strips markdown/emoji/stage directions) before
  synthesis.
- **SpeechText (STT)** — local speech-to-text via NVIDIA Parakeet-TDT (fast,
  English-only) or faster-whisper (99 languages, tiny→large-v3), with
  chunked processing so VRAM use stays bounded regardless of file length.
  Transcribes audio or video files directly.
- **MusicStem** — local stem separation via Demucs (vocals/drums/bass/other,
  or 6 stems including guitar/piano), high-quality `htdemucs_ft` model by
  default. Runs on PyTorch — requires a CUDA-enabled `torch` build for GPU use.
- **Image generation** — self-contained Image tab for local image-generation
  models (no Hugging Face token required). Runs on PyTorch — requires a
  CUDA-enabled `torch` build for GPU use.
- **Video cleanup / upscaling** — Real-ESRGAN-based denoise, sharpen, and
  optional resolution upscaling, with ready-made presets, GFPGAN face
  restoration, batch queue with per-file HH:MM:SS trim, scheduled/overnight
  start, and live ETA. Runs on PyTorch — requires a CUDA-enabled `torch`
  build for GPU use.
- **Live status footer** — CPU%, GPU%, VRAM, and RAM usage updated every
  second.

## Project Structure

```
SageModel/
├── main.py                        # Entry point — launches QApplication + MainWindow.
│                                   #   Also registers Windows DLL search directories for
│                                   #   the pip-installed NVIDIA CUDA runtime DLLs
│                                   #   (nvidia-cuda-runtime-cu12 / nvidia-cublas-cu12) via
│                                   #   os.add_dll_directory(), BEFORE any other imports —
│                                   #   this is a fallback safety net for llama-cpp-python's
│                                   #   CUDA backend; see GPU Setup section below.
├── sage_studio/
│   ├── __init__.py
│   ├── constants.py                # App-wide constants, provider presets, data-file paths,
│   │                                #   Real-ESRGAN/Demucs/Whisper model dirs & pip hints
│   ├── models.py                   # Dataclasses: ChatSession, ChainStep, LocalModel,
│   │                                #   FavoriteModel, ModelInfo, RunRecord
│   ├── persistence.py               # PersistenceManager — JSON load/save for chats,
│   │                                #   favorites, local models, API keys, voice/prompt settings
│   ├── utils.py                     # CUDA/GPU detection, GGUF GPU-support checks,
│   │                                #   nvidia-smi VRAM/GPU% readers, psutil RAM reader,
│   │                                #   TTS text cleaning, model-cache unloading
│   ├── workers.py                   # QThread workers: FetchModelsWorker, OpenRouterKeyInfoWorker,
│   │                                #   ChatWorker (runs the model chain), VoiceWorker
│   ├── ui_style.py                  # DARK_QSS — the app's dark Qt stylesheet
│   ├── main_window.py               # MainWindow — builds all tabs, wires panels together,
│   │                                #   live status footer, window-state persistence
│   ├── ui_model_browser.py          # ModelBrowserPanel — remote provider/model browsing,
│   │                                #   API key management, free/paid filtering
│   ├── ui_local_models.py           # LocalModelsPanel — load GGUF/Safetensors, per-model
│   │                                #   parameters, GPU-support checks, remove/delete from disk.
│   │                                #   Calls llama_cpp_gpu_supported() on startup — an
│   │                                #   unhandled import/DLL-load failure here crashes the
│   │                                #   whole app (see Known Issues)
│   ├── ui_chain_builder.py          # ChainBuilderPanel — the ordered list of chain steps,
│   │                                #   reorder/edit/remove
│   ├── ui_run_panel.py              # RunPanel — initial input box (clears on Run/Enter),
│   │                                #   Run/Stop chain execution, output history log
│   │                                #   (word-wrapped, auto-scrolling, no divider clutter),
│   │                                #   Persistent Session rolling-summary + Keep First
│   ├── ui_chats_favorites.py        # ChatsPanel + FavoritesPanel — chat session list and
│   │                                #   favorited models list
│   ├── ui_voice_panel.py            # VoicePanel — Kokoro TTS settings, voice/speed selection,
│   │                                #   synthesize + play/save WAV
│   ├── ui_github_panel.py           # GitHubPanel — repo connect, file selection, prepare/run/push
│   │                                #   AI-assisted commits (self-contained, own chain run)
│   ├── ui_prompt_templates.py       # PromptTemplatesPanel — save/reuse prompt templates
│   ├── ui_image_panel.py            # ImagePanel — local image-generation tooling (PyTorch)
│   ├── ui_video_panel.py            # VideoPanel — Real-ESRGAN video cleanup/upscaling (PyTorch),
│   │                                #   batch queue, per-file HH:MM:SS trim, schedule/overnight
│   │                                #   start, live ETA, FP16/FP32 precision log
│   ├── ui_music_panel.py            # MusicStemPanel — Demucs stem separation (PyTorch)
│   │                                #   (vocals/drums/bass/other/guitar/piano)
│   ├── ui_speech_panel.py           # SpeechTextPanel — Parakeet-TDT / faster-whisper
│   │                                #   speech-to-text, chunked for bounded VRAM use
│   └── sage_data/                   # Auto-created at runtime — see Data Storage below
├── add_video_tab.py                 # One-off patch script: registers the Video tab
├── add_gpu_ram_footer.py            # One-off patch script: adds GPU%/RAM to the footer
├── add_video_trim.py                # One-off patch script: adds trim start/end to Video tab
├── add_video_trim_hhmmss.py         # One-off patch script: converts trim fields to HH:MM:SS
├── add_video_eta.py                 # One-off patch script: adds ETA/fps readout to Video tab
├── add_realesrnet_and_precision_log.py  # One-off patch script: adds RealESRNet_x4plus model +
│                                         #   FP16/FP32 precision logging
├── replace_video_panel_with_batch.py    # Full replacement: batch queue + scheduled start for Video
├── add_music_stem_tab.py            # One-off patch script: registers the MusicStem tab
├── add_speech_text_tab.py           # One-off patch script: registers the SpeechText tab
├── fix_output_log_formatting.py     # One-off patch script: word-wrap + follow-bottom autoscroll
├── remove_output_log_hr_lines.py    # One-off patch script: removes all <hr> divider lines
├── clear_input_on_run.py            # One-off patch script: clears input box on Run/Enter
├── add_keep_first.py                # One-off patch script: adds "Keep First N turns" to
│                                     #   Persistent Session (protects early turns from summarization)
├── fix_basicsr_torchvision.py       # One-off patch script: fixes basicsr/torchvision
│                                     #   functional_tensor incompatibility (see Known Issues)
└── remove_local_models_hint.py      # One-off patch script: removes a hint label (example utility)
```

> The `add_*.py` / `fix_*.py` / `replace_*.py` scripts at the project root are
> one-time migration helpers used to safely patch specific files without
> hand-editing them. They're idempotent (safe to re-run) and can be deleted
> once applied, or kept as a record of how each feature was added.

## Installation

Core dependencies:

```
pip install PyQt6 requests
```

Then run:

```
python main.py
```

All other dependencies are **optional** and only required for specific
features — the app degrades gracefully (with clear in-app messaging) if
they're missing. **However, on a fresh Windows PC with an NVIDIA GPU, follow
the GPU Setup section below before installing anything else** — installing
GPU packages in the wrong order or with the wrong pip flags is the single
biggest source of setup pain with this app.

## GPU Setup (Windows) — Read This First

This app depends on two *independent* GPU stacks that are easy to
accidentally break: `llama-cpp-python` (for local GGUF chat models) and
`torch` (for Image, Video, and MusicStem). Both default to silently
installing **CPU-only** builds if you don't force the CUDA variant
explicitly. Set both up correctly the first time to avoid the
troubleshooting below.

### 1. Check your driver and CUDA support

```cmd
nvidia-smi
```

Note the "CUDA Version" shown top-right — that's the *maximum* CUDA version
your driver supports. Any `cuXXX`-tagged wheel at or below that number will
work.

### 2. Install the Microsoft Visual C++ Redistributable

Required for both `llama-cpp-python`'s and `torch`'s compiled CUDA
extensions to load at all. Skip this and you can hit generic
"DLL or one of its dependencies" errors that have nothing to do with CUDA.

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 --exact --source winget --accept-source-agreements --accept-package-agreements
```

### 3. Install `torch` with the CUDA index explicitly

Always use `--index-url` (not `--extra-index-url`, not a bare
`pip install torch`) — this fully replaces the package source so pip cannot
fall back to the CPU-only PyPI build:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Swap `cu124` for whatever tag is current and ≤ your driver's max CUDA
version per [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).
Verify:

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

You want `True` and a version string ending in `+cuXXX`, not `+cpu`.

### 4. Install `llama-cpp-python` with the CUDA wheel index

```cmd
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --force-reinstall --no-cache-dir
```

Unlike `torch`, this package's wheel filenames are identical between the CPU
and CUDA builds (only the source index/URL differs), so `--extra-index-url`
is fine here — but `--force-reinstall --no-cache-dir` is important to avoid
pip reusing a stale cached CPU wheel.

### 5. Supply the CUDA runtime DLLs `llama-cpp-python` needs

The `cu124` wheel's CUDA backend (`ggml-cuda.dll`) does **not** bundle its
own copies of `cudart64_12.dll` / `cublas64_12.dll` / `cublasLt64_12.dll` —
it expects them to be discoverable by Windows' DLL loader. The full NVIDIA
CUDA Toolkit installer is one way to get these, but it's heavyweight and can
fail to install with no useful error message on some systems. The reliable,
lightweight alternative is the pip-packaged redistributable DLLs:

```cmd
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

Then copy the three DLLs directly next to `llama.dll` (Windows always checks
a DLL's own folder first, so this works regardless of how the app is
launched):

```cmd
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cuda_runtime\bin\cudart64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin\cublas64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
copy "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\nvidia\cublas\bin\cublasLt64_12.dll" "%LOCALAPPDATA%\Programs\Python\Python311\Lib\site-packages\llama_cpp\lib\"
```

Adjust the Python version folder (`Python311`) if different on the new
machine. `main.py` also registers these same two source folders via
`os.add_dll_directory()` at startup as a fallback, but the direct-copy above
is what actually guarantees it works, since that fallback only helps the
Python process itself, not any external tooling.

### 6. Verify

```cmd
python -c "import llama_cpp" && echo llama_cpp OK
python main.py
```

If `python -c "import llama_cpp"` fails with
`FileNotFoundError: ... llama.dll (or one of its dependencies)`, see
[Known Issues & Fixes](#known-issues--fixes) for how to pinpoint the exact
missing DLL with a dependency walker rather than guessing.

## Optional Feature Installs

| Feature | Install |
|---|---|
| GGUF local models (GPU/CUDA, Windows) | See [GPU Setup](#gpu-setup-windows--read-this-first) steps 2–5 in full — a bare `pip install llama-cpp-python` or a blanket `pip install --upgrade` gets you the CPU-only build |
| GGUF local models (CPU only) | `pip install llama-cpp-python` |
| Safetensors local models | `pip install transformers torch accelerate safetensors` — for GPU, install `torch` per [GPU Setup](#gpu-setup-windows--read-this-first) step 3 first |
| Voice / Kokoro TTS | `pip install kokoro soundfile numpy` (Linux also needs `apt install espeak-ng`); GPU: install a CUDA build of `torch` per step 3 above |
| Image generation (GPU) | Requires a CUDA build of `torch` — see [GPU Setup](#gpu-setup-windows--read-this-first) step 3 |
| Video cleanup/upscaling | `pip install opencv-python basicsr facexlib gfpgan realesrgan` plus [ffmpeg](https://ffmpeg.org/download.html) on your PATH (for audio muxing); GPU requires a CUDA `torch` build (step 3) |
| MusicStem (stem separation) | `pip install demucs`; GPU requires a CUDA `torch` build (step 3) |
| SpeechText (Parakeet) | `pip install faster-whisper soundfile numpy` plus `pip install git+https://github.com/huggingface/transformers` (needed until Parakeet-TDT ships in a stable transformers release) |
| SpeechText (faster-whisper) | `pip install faster-whisper soundfile numpy` |
| SpeechText (either engine) | [ffmpeg](https://ffmpeg.org/download.html) on your PATH (decodes input to 16kHz mono) |
| Live status footer CPU%/RAM% | `pip install psutil` |
| Live status footer GPU%/VRAM | NVIDIA driver with `nvidia-smi` on PATH (no extra pip package) |
| GitHub tab | A GitHub personal access token with repo scope, entered in-app |

Use the Local Models panel's "Check GGUF GPU Support" button, and check
`torch.cuda.is_available()` (see GPU Setup step 3), to verify which kind of
build you actually have before assuming something's broken — a plain `pip
install` or a blanket `pip install --upgrade` on either `llama-cpp-python`
or `torch` silently reverts to CPU-only, with no error or warning.

## Data Storage

Everything the app stores lives in a portable `sage_studio/sage_data/`
folder next to the code, so nothing touches your system profile or global
caches:

```
sage_data/
├── chats.json                 # All chat sessions + full turn-by-turn history
├── api_keys.json              # Saved API keys / tokens per provider scope
├── favorites.json             # Favorited models
├── local_models.json          # Registered local GGUF/Safetensors models + their params
├── voice_settings.json        # Kokoro TTS settings
├── prompt_templates.json      # Saved prompt templates
├── window_state.json          # Window geometry + splitter sizes
├── audio/                     # Generated TTS .wav files
├── kokoro_model/               # Kokoro model cache (HF_HUB_CACHE redirected here)
├── realesrgan_weights/        # Downloaded Real-ESRGAN / GFPGAN model weights
├── demucs_models/             # Demucs/torch cache (TORCH_HOME redirected here)
└── whisper_models/            # faster-whisper model cache (download_root)
```

Note: `chats.json` retains full raw conversation history indefinitely with
no automatic pruning — the Persistent Session budget/summary settings only
affect what's sent to the model each turn, not what's kept on disk.

On a new PC, copying this folder over from the old machine restores all
chats, favorites, local model registrations, and settings — just re-verify
GPU setup separately, since it's environment-specific and won't transfer.

## Tabs Overview

- **💬 Chat** — chat session list, chain builder, run panel with streaming
  output log.
- **⚙️ Settings** — remote model browser, local models manager, chain builder,
  favorites.
- **🔊 Voice** — Kokoro TTS: pick a voice, speed, GPU on/off, synthesize the
  latest chain output or pasted text.
- **📈 GitHub** — connect a repo, select files, describe a change, run it
  through your chain, push as a commit.
- **📝 Templates** — save and reuse prompt templates.
- **🖼️ Image** — local image-generation tooling.
- **🎬 Video** — Real-ESRGAN cleanup/upscale presets, batch queue with
  per-file HH:MM:SS trim, scheduled/overnight start, live ETA, cancel-safe
  (produces a valid partial file if stopped early).
- **🎵 MusicStem** — Demucs stem separation: choose `htdemucs_ft` (quality),
  `htdemucs` (fast), or `htdemucs_6s` (adds guitar/piano); vocals-only mode
  or pick individual stems to save.
- **🎤 SpeechText** — Parakeet-TDT or faster-whisper transcription of audio
  or video files, chunked processing, save as `.txt` or send straight to the
  Chat tab's input box.

## Persistent Session (Long-Running Chats)

The Run panel's **Persistent Session** checkbox (on by default) keeps very
long chats coherent without re-sending the entire history every turn:

- **Rolling summary** — once the conversation exceeds the token **Budget**
  (default 12,000 tokens, or auto-sized from the active local model's
  context length), the oldest turns are automatically condensed into a
  running summary by an extra model call, rather than being sent verbatim
  forever. This is fully automatic — no manual trigger needed.
- **Keep First** (default 0 = off) — set this to protect the first N turns
  of a chat from ever being summarized or dropped, regardless of how long
  the budget-based pruning runs. Useful for locking in campaign
  setup/scene-establishing turns in a long-running story so they're never
  paraphrased away.
- **What this does *not* protect**: a persona/rules prompt baked into a
  Chain Step's prompt template (e.g. a Dungeon Master persona) is *never*
  at risk of being forgotten in the first place — it's reinjected in full on
  every single turn regardless of Persistent Session settings, since it
  lives outside the summarized conversation history entirely.
- The rolling summary itself is **not persisted to disk** — it resets on
  app restart or when switching chats, and regenerates from the full raw
  history (which *is* persisted) the next time the budget is exceeded.

## Known Issues & Fixes

**`FileNotFoundError: Could not find module '...\llama_cpp\lib\llama.dll'
(or one of its dependencies)`** — this means `llama.dll` itself is present,
but a DLL *it* depends on can't be resolved. This crashes the entire app on
startup, not just the Local Models tab, because `LocalModelsPanel.__init__`
calls `llama_cpp_gpu_supported()` unconditionally. Diagnose the exact
missing DLL rather than guessing:

1. Confirm the VC++ Redistributable is installed (see GPU Setup step 2).
2. Open [Dependencies](https://github.com/lucasg/Dependencies) (a modern,
   free dependency walker) and load
   `...\site-packages\llama_cpp\lib\llama.dll`. Expand the tree under
   `ggml-cuda.dll` specifically — CUDA support loads as a runtime backend
   plugin, not a static import, so top-level entries can look fine while
   `ggml-cuda.dll`'s own children (`cudart64_12.dll`, `cublas64_12.dll`) show
   red/missing.
3. Whatever's flagged missing there is what GPU Setup step 5 above resolves
   — copy the matching DLL from the `nvidia-cuda-runtime-cu12` /
   `nvidia-cublas-cu12` pip packages directly into `llama_cpp\lib\`.

**Image/Video/MusicStem tabs silently running on CPU instead of GPU** — these
three all depend on PyTorch, independently of `llama-cpp-python`. Check
`torch.cuda.is_available()` (GPU Setup step 3). If it's `False` and the
version string ends in `+cpu`, `torch` got reinstalled as CPU-only — usually
from a bare/blanket `pip install --upgrade` — and needs reinstalling with
`--index-url https://download.pytorch.org/whl/cuXXX` explicitly.

**`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`**
when using the Video tab — this is a known `basicsr` incompatibility with
newer `torchvision` releases (0.17+ removed that internal module). Run
`fix_basicsr_torchvision.py` to patch the installed `basicsr` package in
place (redirects the import to `torchvision.transforms.functional`, which
still has the needed functions). This does not touch your `torch`/CUDA setup,
and can resurface any time `torchvision` is reinstalled or upgraded — just
re-run the patch script again if so.

**GFPGAN face restoration is slow** — roughly 1–3 fps even on capable GPUs is
expected; face enhancement runs a second model per frame on top of
Real-ESRGAN. Use it selectively rather than on full-length footage, or trim
to a short test range first via the Video tab's trim fields.

**Real-ESRGAN "no resize" presets still cost full 4x-upscale compute** —
`outscale` only controls a cheap resize *after* the model runs; the network
always computes at its native 4x scale internally regardless of the
requested output size. There's no setting to reduce this; it's an
architectural property of the model family.

**Parakeet's VRAM use scales with clip duration** — unlike Whisper's
fixed-window chunking, Parakeet's encoder holds the whole clip's features in
memory. The SpeechText tab chunks all input uniformly (default 60s,
adjustable) specifically to keep this bounded regardless of file length.

## Updating Packages Safely

**Do not run a blanket "update all packages" command in this project's
environment.** `llama-cpp-python`, `torch`, `torchvision`, and `transformers`
are all pinned to specific CUDA-matched or feature-matched builds, not just
"whatever's newest." A generic `pip install --upgrade <package>` or a loop
over `pip list --outdated` silently reverts `llama-cpp-python` and `torch` to
CPU-only PyPI builds, and can swap `transformers` from the Parakeet-enabled
dev install back to a stable release that's missing that model support.

If you do need to update something, do it one package at a time, and for the
four packages above, always re-specify the correct index URL from
[GPU Setup](#gpu-setup-windows--read-this-first) rather than a bare
`--upgrade`. Re-verify with `torch.cuda.is_available()` and the "Check GGUF
GPU Support" button afterward before assuming it worked.

## Running the App

```
python main.py
```

The window remembers its size, position, and panel-splitter layout between
runs (`sage_data/window_state.json`).
