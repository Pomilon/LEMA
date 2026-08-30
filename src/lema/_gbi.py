from __future__ import annotations

import json

import torch
from safetensors import safe_open
import os
import glob


class GlobalBinaryIndex:
    #: Optional safetensors metadata key mapping tensor name -> logical shape.
    #: Pre-quantized int4 checkpoints store packed bytes, so the storage shape
    #: differs from the logical weight shape; writers record it here.
    LOGICAL_SHAPES_KEY = "lema_logical_shapes"

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        self.files: list[str] = []
        if os.path.isdir(model_path):
            self.files = glob.glob(os.path.join(model_path, "*.safetensors"))
            if not self.files:
                self.files = [model_path] if model_path.endswith(".safetensors") else []
        else:
            self.files = [model_path]
        if not self.files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        self.handles: list = []
        self.param_map: dict[str, any] = {}
        self.logical_shapes: dict[str, tuple] = {}
        for f in self.files:
            handle = safe_open(f, framework="pt", device="cpu")
            self.handles.append(handle)
            meta = handle.metadata() or {}
            raw = meta.get(self.LOGICAL_SHAPES_KEY)
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            self.logical_shapes.setdefault(k, tuple(v))
                except (ValueError, TypeError):
                    pass
            for key in handle.keys():
                if key not in self.param_map:
                    self.param_map[key] = handle

    def load_tensors(self, param_names: list[str], device: str = "cpu") -> dict[str, torch.Tensor]:
        tensors = {}
        for name in param_names:
            if name in self.param_map:
                tensors[name] = self.param_map[name].get_tensor(name)
        return tensors

    def get_keys(self) -> list[str]:
        return list(self.param_map.keys())

    def close(self):
        for handle in self.handles:
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()
        self.param_map.clear()

    def get_tensor_shape(self, name: str):
        if name in self.logical_shapes:
            return self.logical_shapes[name]
        if name in self.param_map:
            try:
                return self.param_map[name].get_slice(name).get_shape()
            except Exception:
                return None
        return None
