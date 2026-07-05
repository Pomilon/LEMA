import torch
from safetensors import safe_open
from typing import Dict, List, Any, Optional
import os
import glob

class GlobalBinaryIndex:
    """
    GBI v0.5: Multi-file Support.
    Handles split safetensors by mapping parameter names to their respective files.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.files = []
        if os.path.isdir(model_path):
            self.files = glob.glob(os.path.join(model_path, "*.safetensors"))
            if not self.files:
                # Fallback to checking the parent dir or if it's already a monolith
                self.files = [model_path] if model_path.endswith(".safetensors") else []
        else:
            self.files = [model_path]

        if not self.files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        self.handles = []
        self.param_map = {}
        
        for f in self.files:
            handle = safe_open(f, framework="pt", device="cpu")
            self.handles.append(handle)
            for key in handle.keys():
                if key in self.param_map:
                    # Parameter collision is rare in valid split models
                    continue
                self.param_map[key] = handle

    def load_tensors(self, param_names: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        tensors = {}
        for name in param_names:
            if name not in self.param_map:
                # Some adapters might look for optional params, warn instead of crash?
                # For now, let's be strict or return None?
                continue
            tensors[name] = self.param_map[name].get_tensor(name)
        return tensors

    def get_keys(self) -> List[str]:
        return list(self.param_map.keys())

    def get_tensor_shape(self, name: str):
        if name in self.param_map:
            # safe_open handles don't have a direct shape check without get_slice
            return self.param_map[name].get_slice(name).get_shape()
        return None