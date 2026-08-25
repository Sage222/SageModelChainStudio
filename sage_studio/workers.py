"""Sage Model Chain Studio — Workers"""

import os, time, random, requests, warnings
from typing import List, Dict, Any, Optional
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from .constants import *
from .utils import (
    _cuda_available, llama_cpp_gpu_supported, _is_local_kind,
    step_display, _is_openrouter_free, _is_nim_free,
    _is_gemini_free, clean_text_for_speech,
)
from .models import ModelInfo, ChainStep


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
    step_warning = pyqtSignal(int, str)
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
                        output = self._call_local(step, prompt, idx)
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

    def _call_local(self, step, prompt, idx):
        params = self.local_model_params.get(step.model_id, {})
        if step.kind == LOCAL_GGUF_KIND:
            return self._call_local_gguf(step, prompt, params, idx)
        return self._call_local_transformers(step, prompt, params)

    def _call_local_gguf(self, step, prompt, params, idx):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python\n"
                "For GPU support use the prebuilt CUDA wheel:\n" + CUDA_WHEEL_HINT
            )
        llm = _GGUF_CACHE.get(step.model_id)
        if llm is None:
            if not os.path.exists(step.model_id):
                raise RuntimeError(f"GGUF file not found: {step.model_id}")
            n_ctx = params.get("n_ctx", 4096)
            n_gpu = params.get("n_gpu_layers", -1)
            if n_gpu != 0:
                installed, gpu_ok, detail = llama_cpp_gpu_supported()
                if installed and not gpu_ok:
                    self.step_warning.emit(
                        idx,
                        "GPU Layers is set to " + str(n_gpu) + " but your installed llama-cpp-python "
                        "has NO GPU backend compiled in, so this model will load into SYSTEM RAM "
                        "and run on CPU regardless of that setting. Fix with:\n" + CUDA_WHEEL_HINT
                    )
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
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                pipeline = KPipeline(lang_code=lang, device=device, repo_id="hexgrad/Kokoro-82M")
                chunks = []
                for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
                    chunks.append(audio)
            if not chunks:
                raise RuntimeError("Kokoro returned no audio.")
            sf.write(self.output_path, np.concatenate(chunks), 24000)
            self.finished_ok.emit(self.output_path)
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")


