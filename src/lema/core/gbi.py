import torch
from safetensors import safe_open
from typing import Dict, List, Any, Optional
import os

class GlobalBinaryIndex:
    """
    GBI v0.4: Contiguous Block Access.
    Allows fetching a whole layer as a single byte-range if possible,
    but here we focus on providing tensors for contiguous packing.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.handle = safe_open(self.model_path, framework="pt", device="cpu")
        self.keys = list(self.handle.keys())

    def load_tensors(self, param_names: List[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        tensors = {}
        for name in param_names:
            tensors[name] = self.handle.get_tensor(name)
        return tensors