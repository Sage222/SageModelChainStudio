"""Automatic configuration for local GGUF models (v4).

Drop this file into sage_studio/ (next to workers.py). It replaces v2/v3.

What it does
------------
* Reads the GGUF header (layers, KV heads, trained context) without loading weights.
* Reads free VRAM from nvidia-smi.
* Unloads any OTHER cached GGUF model before loading a new one, so the free-VRAM
  reading is honest and old models do not keep eating VRAM/RAM.
* Picks the largest context that fits while leaving TARGET_FREE_MIB of VRAM free.
  If the model cannot fit fully on the GPU with that reserve, the reserve is
  relaxed down to MIN_FREE_MIB before any layers are moved to the CPU, because
  partial offload is much slower and fills system RAM.
  Order of preference: f16 KV -> q8_0 KV (+flash attention) -> relaxed reserve
  -> partial GPU offload.
* Loads the model with automatic retry (smaller context, q8 KV, fewer GPU layers).
* Prints estimated vs actual VRAM and process RAM after every load.
* Guards every prompt so it always fits the context window.
* Derives the Persistent Session budget from the real loaded context.

Settings live in sage_data/auto_config.json, for example:
  {"settings": {"target_free_mib": 2048, "min_free_mib": 1024, "single_model": true}}
"""
import gc
import json
import os
import subprocess
import time
from pathlib import Path

AUTO_CONFIG_VERSION = 4
MIB = 1024 * 1024
TARGET_FREE_MIB = 2048
MIN_FREE_MIB = 1024
COMPUTE_BUFFER_MIB = 450
MIN_CTX = 2048
COMFORT_CTX = 4096
MAX_CTX = 16384
CTX_STEP = 256
PROMPT_OVERHEAD_TOKENS = 160
_SUGGEST_TTL = 20.0

_DATA_DIR = Path(__file__).resolve().parent / "sage_data"
_CFG_FILE = _DATA_DIR / "auto_config.json"

_info_cache = {}
_suggest_cache = {}
LAST_CONFIG = {}


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _floor_step(x):
    return int(x // CTX_STEP) * CTX_STEP


def _load_cfg():
    try:
        return json.loads(_CFG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cfg(data):
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _CFG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        print("[auto_config] could not save settings:", e, flush=True)


def is_auto(path):
    models = _load_cfg().get("models", {})
    return bool(models.get(_norm(path), {}).get("auto", True))


def set_auto(path, value):
    data = _load_cfg()
    data.setdefault("models", {}).setdefault(_norm(path), {})["auto"] = bool(value)
    _save_cfg(data)
    _suggest_cache.clear()


def _setting(name, default):
    try:
        return _load_cfg().get("settings", {}).get(name, default)
    except Exception:
        return default


def target_free_mib():
    try:
        return int(_setting("target_free_mib", TARGET_FREE_MIB))
    except Exception:
        return TARGET_FREE_MIB


def min_free_mib():
    try:
        return int(_setting("min_free_mib", MIN_FREE_MIB))
    except Exception:
        return MIN_FREE_MIB


def get_free_vram_mib():
    """Returns (free_mib, total_mib) from nvidia-smi, or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if r.returncode != 0:
            return None
        free, total = r.stdout.strip().splitlines()[0].split(",")
        return float(free), float(total)
    except Exception:
        return None


def _process_rss_mib():
    try:
        import psutil
        return psutil.Process().memory_info().rss / MIB
    except Exception:
        return None


def read_gguf_info(path):
    """Reads GGUF metadata only (vocab_only load). Returns a dict or None.

    llama.cpp may print 'n_ctx_seq (512) > n_ctx_train (0)' for this metadata-only
    read. It is harmless.
    """
    try:
        key = (_norm(path), os.path.getmtime(path))
    except OSError:
        return None
    if key in _info_cache:
        return _info_cache[key]
    info = None
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=path, vocab_only=True, verbose=False)
        md = dict(llm.metadata or {})
        try:
            del llm
        except Exception:
            pass
        arch = md.get("general.architecture", "llama")

        def geti(suffix, default=None):
            try:
                return int(float(md[f"{arch}.{suffix}"]))
            except Exception:
                return default

        n_layer = geti("block_count")
        n_head = geti("attention.head_count")
        n_head_kv = geti("attention.head_count_kv", n_head)
        n_embd = geti("embedding_length")
        ctx_train = geti("context_length", 4096)
        key_len = geti("attention.key_length")
        val_len = geti("attention.value_length", key_len)
        if not key_len and n_head and n_embd:
            key_len = n_embd // n_head
            val_len = key_len
        if n_layer and n_head_kv and key_len and val_len:
            info = dict(arch=arch, n_layer=n_layer, n_head_kv=n_head_kv,
                        key_len=key_len, val_len=val_len, ctx_train=ctx_train,
                        has_template="tokenizer.chat_template" in md)
    except Exception as e:
        print("[auto_config] could not read GGUF metadata:", e, flush=True)
    _info_cache[key] = info
    return info


def kv_mib_per_token(info, kv_type="f16"):
    bytes_per_value = 2.0 if kv_type == "f16" else 34.0 / 32.0  # q8_0 = 8.5 bits
    per_token = info["n_layer"] * info["n_head_kv"] * (info["key_len"] + info["val_len"])
    return per_token * bytes_per_value / MIB


def _result(n_ctx, n_gpu, kv_type, kv_mib, weights_mib, free_mib, target, note, ctx_train):
    est = weights_mib + kv_mib + (COMPUTE_BUFFER_MIB if n_gpu != 0 else 0)
    return dict(n_ctx=int(n_ctx), n_gpu_layers=int(n_gpu), kv_type=kv_type,
                flash_attn=(kv_type != "f16"), kv_mib=float(kv_mib),
                weights_mib=float(weights_mib), est_used_mib=float(est),
                free_mib=float(free_mib), left_mib=float(free_mib - est),
                target_mib=float(target), note=note, ctx_train=ctx_train)


def _gguf_cache():
    try:
        from .constants import _GGUF_CACHE
        return _GGUF_CACHE
    except Exception:
        return {}


def _is_loaded(path):
    return path in _gguf_cache()


def suggest(path):
    """Best-guess parameters for this model on this machine, right now."""
    loaded = _is_loaded(path)
    ckey = (_norm(path), loaded)
    hit = _suggest_cache.get(ckey)
    if hit and time.time() - hit[0] < _SUGGEST_TTL:
        return hit[1]

    info = read_gguf_info(path)
    ctx_train = info["ctx_train"] if info else 4096
    top = _floor_step(min(ctx_train, MAX_CTX)) or CTX_STEP
    comfort = min(top, COMFORT_CTX)
    min_ctx = min(top, MIN_CTX)
    target = target_free_mib()
    floor = min(target, min_free_mib())
    try:
        file_mib = os.path.getsize(path) / MIB
    except OSError:
        file_mib = 0.0

    try:
        from .utils import llama_cpp_gpu_supported
        _, gpu_ok, _ = llama_cpp_gpu_supported()
    except Exception:
        gpu_ok = True
    mem = get_free_vram_mib() if gpu_ok else None

    if not gpu_ok:
        res = _result(comfort, 0, "f16", 0, 0, 0, target,
                      " CPU-only llama-cpp build: GPU offload disabled.", ctx_train)
    elif mem is None or info is None:
        res = _result(min_ctx, -1, "f16", 0, file_mib, mem[0] if mem else 0, target,
                      " Could not read VRAM or model metadata; using a conservative context.",
                      ctx_train)
    else:
        free, _total = mem
        prev = LAST_CONFIG.get(_norm(path))
        if loaded and prev and prev.get("est_used_mib"):
            free += prev["est_used_mib"]
        per_f16 = kv_mib_per_token(info, "f16")
        per_q8 = kv_mib_per_token(info, "q8_0")

        res = None
        reserves = [target] if floor >= target else [target, floor]
        for reserve in reserves:
            budget = free - reserve - file_mib - COMPUTE_BUFFER_MIB
            ctx_f16 = _floor_step(budget / per_f16) if budget > 0 else 0
            ctx_q8 = _floor_step(budget / per_q8) if budget > 0 else 0
            relaxed = ""
            if reserve < target:
                relaxed = (f" Free-VRAM reserve relaxed from {target / 1024:.1f} to "
                           f"{reserve / 1024:.1f} GB to keep every layer on the GPU "
                           f"(partial offload is far slower and fills system RAM). "
                           f"Keep Voice (Kokoro) on CPU.")
            if ctx_f16 >= comfort:
                n_ctx = min(ctx_f16, top)
                res = _result(n_ctx, -1, "f16", n_ctx * per_f16, file_mib, free, target,
                              relaxed, ctx_train)
                break
            if ctx_q8 >= min_ctx:
                n_ctx = min(ctx_q8, top)
                res = _result(n_ctx, -1, "q8_0", n_ctx * per_q8, file_mib, free, target,
                              " q8 KV cache + flash attention used to fit more context." + relaxed,
                              ctx_train)
                break
        if res is None:
            total_layers = info["n_layer"] + 1
            kv_full = comfort * per_q8
            gl = total_layers
            while gl > 0:
                frac = gl / total_layers
                need = file_mib * frac + kv_full * frac + COMPUTE_BUFFER_MIB
                if need <= free - target:
                    break
                gl -= 1
            n_gpu = -1 if gl == total_layers else gl
            frac = gl / total_layers
            res = _result(comfort, n_gpu, "q8_0", kv_full * frac, file_mib * frac, free, target,
                          " Model too large for full offload: part of it runs on CPU"
                          " (uses system RAM). A smaller quant (e.g. Q6_K) would fit fully.",
                          ctx_train)
    _suggest_cache[ckey] = (time.time(), res)
    return res


def describe(s):
    if s["n_gpu_layers"] == -1:
        gl = "all layers on GPU"
    elif s["n_gpu_layers"] > 0:
        gl = f"{s['n_gpu_layers']} layers on GPU"
    else:
        gl = "CPU only"
    kv = "q8" if s["kv_type"] != "f16" else "f16"
    text = f"Auto: {s['n_ctx']} ctx \u00b7 {gl} \u00b7 {kv} KV ({s['kv_mib'] / 1024:.1f} GB)"
    if s["free_mib"]:
        text += (f" \u00b7 est. {s['est_used_mib'] / 1024:.1f} GB VRAM,"
                 f" about {s['left_mib'] / 1024:.1f} GB left (target {s['target_mib'] / 1024:.1f} GB)")
    return text + "." + (s["note"] or "")


def effective_ctx(path, params):
    params = params or {}
    if not is_auto(path):
        return int(params.get("n_ctx", 4096))
    cfg = LAST_CONFIG.get(_norm(path))
    if cfg and _is_loaded(path):
        return int(cfg["n_ctx"])
    try:
        return int(suggest(path)["n_ctx"])
    except Exception:
        return int(params.get("n_ctx", 4096))


def auto_budget(step, params_map):
    """Persistent Session budget (tokens) for a local GGUF first step, else None."""
    kind = str(getattr(step, "kind", "")).lower()
    if "gguf" not in kind:
        return None
    p = (params_map or {}).get(step.model_id) or {}
    n_ctx = effective_ctx(step.model_id, p)
    max_tokens = min(int(p.get("max_tokens", 1024)), n_ctx // 2)
    template = len(getattr(step, "prompt_template", "") or "") // 3
    budget = int((n_ctx - max_tokens - template - PROMPT_OVERHEAD_TOKENS) * 0.85)
    return max(budget, 300)


def _evict_other_models(path, warn):
    """Unload every other cached GGUF model so VRAM/RAM readings are honest."""
    if not _setting("single_model", True):
        return
    cache = _gguf_cache()
    others = [k for k in list(cache) if _norm(k) != _norm(path)]
    if not others:
        return
    for k in others:
        obj = cache.pop(k, None)
        LAST_CONFIG.pop(_norm(k), None)
        try:
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        except Exception:
            pass
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    time.sleep(0.5)
    _suggest_cache.clear()
    warn(f"Unloaded {len(others)} other GGUF model(s) to free VRAM and RAM before loading this one.")


def load_gguf_auto(path, params, warn=None):
    """Loads a GGUF model with automatic settings and automatic fallback."""
    warn = warn or (lambda m: print(m, flush=True))
    import llama_cpp
    from llama_cpp import Llama
    params = params or {}
    _evict_other_models(path, warn)
    auto = is_auto(path)
    est_used = None
    est_total = None
    if auto:
        s = suggest(path)
        n_ctx, n_gpu, kv = s["n_ctx"], s["n_gpu_layers"], s["kv_type"]
        est_used = s["est_used_mib"] or None
        est_total = est_used
        print("[auto_config] " + describe(s), flush=True)
    else:
        n_ctx = int(params.get("n_ctx", 4096))
        n_gpu = int(params.get("n_gpu_layers", -1))
        kv = "f16"
        print(f"[auto_config] manual: n_ctx={n_ctx} n_gpu_layers={n_gpu}", flush=True)

    info = read_gguf_info(path)
    total_layers = (info["n_layer"] + 1) if info else 41
    q8 = getattr(llama_cpp, "GGML_TYPE_Q8_0", 8)
    last_err = None
    for _attempt in range(6):
        kwargs = dict(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu, verbose=False)
        # [SAGE-NOMMAP]
        kwargs["use_mmap"] = bool(_load_cfg().get("settings", {}).get("use_mmap", n_gpu != -1))
        if kv != "f16":
            kwargs.update(flash_attn=True, type_k=q8, type_v=q8)
        before = get_free_vram_mib()
        rss_before = _process_rss_mib()
        try:
            llm = Llama(**kwargs)
            after = get_free_vram_mib()
            rss_after = _process_rss_mib()
            actual = (before[0] - after[0]) if (before and after) else None
            LAST_CONFIG[_norm(path)] = dict(n_ctx=int(llm.n_ctx()), n_gpu_layers=n_gpu,
                                            kv_type=kv,
                                            est_used_mib=(actual if actual else est_used))
            _suggest_cache.clear()
            parts = [f"loaded: n_ctx={llm.n_ctx()} n_gpu_layers={n_gpu} kv={kv}"]
            if actual is not None:
                parts.append(f"VRAM used by load {actual / 1024:.1f} GB"
                             + (f" (estimated {est_total / 1024:.1f} GB)" if est_total else ""))
            if after:
                parts.append(f"VRAM free now {after[0] / 1024:.1f} GB")
            if rss_before is not None and rss_after is not None:
                parts.append(f"process RAM {rss_before / 1024:.1f} -> {rss_after / 1024:.1f} GB")
            print("[auto_config] " + " | ".join(parts), flush=True)
            return llm
        except Exception as e:
            last_err = e
            if not auto:
                raise
            est_used = None
            est_total = None
            if n_ctx > MIN_CTX:
                n_ctx = max(MIN_CTX, _floor_step(n_ctx * 0.75))
                what = f"smaller context ({n_ctx})"
            elif kv == "f16":
                kv = "q8_0"
                what = "q8 KV cache"
            elif n_gpu != 0:
                cur = total_layers if n_gpu == -1 else n_gpu
                n_gpu = max(0, int(cur * 0.75))
                what = f"fewer GPU layers ({n_gpu})"
            else:
                break
            warn(f"Model load failed ({type(e).__name__}); retrying with {what}.")
    raise RuntimeError(f"Could not load {os.path.basename(path)} even after automatic "
                       f"fallbacks: {last_err}")


def guard_prompt(llm, prompt, max_tokens, warn=None):
    """Makes sure prompt + reply fit in the context window. Returns (prompt, max_tokens)."""
    warn = warn or (lambda m: None)
    n_ctx = int(llm.n_ctx())
    mt = int(max_tokens)
    cap = max(256, n_ctx // 2)
    if mt > cap:
        warn(f"Max Tokens lowered from {mt} to {cap} to leave room for the prompt.")
        mt = cap
    avail = n_ctx - mt - PROMPT_OVERHEAD_TOKENS
    if avail < 128:
        mt = max(64, n_ctx - PROMPT_OVERHEAD_TOKENS - 128)
        avail = 128
    toks = llm.tokenize(prompt.encode("utf-8", "ignore"), add_bos=False)
    if len(toks) <= avail:
        return prompt, mt
    head_n = max(32, int(avail * 0.25))
    tail_n = max(32, avail - head_n - 16)
    head = llm.detokenize(toks[:head_n]).decode("utf-8", "ignore")
    tail = llm.detokenize(toks[-tail_n:]).decode("utf-8", "ignore")
    warn(f"Prompt was {len(toks)} tokens but only {avail} fit in the {n_ctx}-token context; "
         f"the middle was trimmed (start and most recent text kept).")
    return head + "\n\n[... earlier text trimmed to fit the context window ...]\n\n" + tail, mt
