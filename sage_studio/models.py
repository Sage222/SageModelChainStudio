"""Sage Model Chain Studio — Models"""

import os, uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any
from .constants import *


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
    n_ctx: int = 16000
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
            n_ctx=d.get("n_ctx", 16000), n_gpu_layers=d.get("n_gpu_layers", -1),
            temperature=d.get("temperature", 0.7), max_tokens=d.get("max_tokens", 1024),
            top_p=d.get("top_p", 0.9), repeat_penalty=d.get("repeat_penalty", 1.1),
        )


@dataclass
class ImageModel:
    """A local diffusers-compatible image generation model.

    kind: IMAGE_DIFFUSERS_FOLDER_KIND (a diffusers-format folder containing
          model_index.json + unet/vae/text_encoder subfolders) or
          IMAGE_DIFFUSERS_SINGLE_FILE_KIND (a single packed .safetensors
          checkpoint, loaded via from_single_file).
    family: one of IMAGE_PIPELINE_FAMILIES ("sd15", "sdxl", "sd3", "flux") --
          determines which diffusers pipeline class is used to load it.
    """
    kind: str
    path: str
    display_name: str
    family: str = "sdxl"
    steps: int = 30
    guidance_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    negative_prompt: str = ""

    def to_dict(self):
        return {
            "kind": self.kind, "path": self.path, "display_name": self.display_name,
            "family": self.family, "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "width": self.width, "height": self.height,
            "negative_prompt": self.negative_prompt,
        }

    @staticmethod
    def from_dict(d):
        return ImageModel(
            kind=d.get("kind", "diffusers_folder"), path=d.get("path", ""),
            display_name=d.get("display_name", ""), family=d.get("family", "sdxl"),
            steps=d.get("steps", 30), guidance_scale=d.get("guidance_scale", 7.0),
            width=d.get("width", 1024), height=d.get("height", 1024),
            negative_prompt=d.get("negative_prompt", ""),
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
