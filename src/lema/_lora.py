from __future__ import annotations

import torch
import torch.nn as nn
import math

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None


class LoRAWrapper(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, lora_A: nn.Parameter, lora_B: nn.Parameter):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = lora_A
        self.lora_B = lora_B

    def forward(self, x):
        result = self.base_layer(x)
        lora_out = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result + lora_out


class LoRAManager:
    def __init__(self, config: dict, device: str = "cuda", dtype: torch.dtype | None = None):
        self.rank = config.get("r", 8)
        self.alpha = config.get("alpha", 16)
        self.target_modules = config.get("target_modules", ["c_attn", "c_proj", "c_fc"])
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32
        self.params: dict[str, dict[str, nn.Parameter]] = {}

    def get_or_init_params(self, layer_id: int, module_name: str, in_features: int, out_features: int) -> dict[str, nn.Parameter]:
        key = f"{layer_id}.{module_name}"
        if key not in self.params:
            lora_A = torch.zeros((self.rank, in_features), device=self.device, dtype=self.dtype)
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            lora_B = torch.zeros((out_features, self.rank), device=self.device, dtype=self.dtype)
            nn.init.zeros_(lora_B)
            self.params[key] = {
                'A': nn.Parameter(lora_A, requires_grad=True),
                'B': nn.Parameter(lora_B, requires_grad=True)
            }
        return self.params[key]

    @staticmethod
    def _get_features(module: nn.Module) -> tuple[int, int] | None:
        if isinstance(module, nn.Linear):
            return (module.in_features, module.out_features)
        if Conv1D is not None and isinstance(module, Conv1D):
            return (module.weight.shape[0], module.weight.shape[1])
        return None

    def _apply_or_update(
        self, layer_id: int, module: nn.Module, prefix: str = "", cache: list | None = None
    ):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = any(name == target or name.endswith(target) for target in self.target_modules)

            if isinstance(child, LoRAWrapper):
                if is_target:
                    features = self._get_features(child.base_layer)
                    if features:
                        params = self.get_or_init_params(layer_id, full_name, features[0], features[1])
                        child.lora_A = params['A']
                        child.lora_B = params['B']
                        if cache is not None:
                            cache.append((child, full_name, features[0], features[1]))
                continue

            features = self._get_features(child) if is_target else None
            if features:
                params = self.get_or_init_params(layer_id, full_name, features[0], features[1])
                lora_layer = LoRAWrapper(
                    base_layer=child, rank=self.rank, alpha=self.alpha,
                    lora_A=params['A'], lora_B=params['B']
                )
                setattr(module, name, lora_layer)
                if cache is not None:
                    cache.append((lora_layer, full_name, features[0], features[1]))
            else:
                self._apply_or_update(layer_id, child, full_name, cache)

    def apply_lora(self, layer_id: int, module: nn.Module, module_name_prefix: str = ""):
        self._apply_or_update(layer_id, module, module_name_prefix, cache=None)

    def update_lora_params(self, layer_id: int, module: nn.Module):
        if not hasattr(module, "_lora_cache"):
            module._lora_cache = []
            self._apply_or_update(layer_id, module, "", module._lora_cache)
        else:
            for wrapper, name, in_f, out_f in module._lora_cache:
                params = self.get_or_init_params(layer_id, name, in_f, out_f)
                wrapper.lora_A = params['A']
                wrapper.lora_B = params['B']

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        all_params = []
        for p_dict in self.params.values():
            all_params.append(p_dict['A'])
            all_params.append(p_dict['B'])
        return all_params

    def save_pretrained(self, save_directory: str):
        import os
        os.makedirs(save_directory, exist_ok=True)
        state_dict = {}
        for key, p_dict in self.params.items():
            state_dict[f"{key}.lora_A"] = p_dict['A'].data.cpu()
            state_dict[f"{key}.lora_B"] = p_dict['B'].data.cpu()
        torch.save(state_dict, os.path.join(save_directory, "adapter_model.bin"))

    def load_pretrained(self, load_directory: str):
        import os
        weight_path = os.path.join(load_directory, "adapter_model.bin")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Adapter weights not found in {load_directory}")
        state_dict = torch.load(weight_path, map_location="cpu")
        for full_key, tensor in state_dict.items():
            parts = full_key.split(".")
            param_type = parts[-1]
            key = ".".join(parts[:-1])
            if key not in self.params:
                self.params[key] = {}
            p_dict = self.params[key]
            if param_type == "lora_A":
                if 'A' not in p_dict:
                    p_dict['A'] = nn.Parameter(tensor.to(self.device), requires_grad=True)
                else:
                    p_dict['A'].data.copy_(tensor.to(self.device))
            elif param_type == "lora_B":
                if 'B' not in p_dict:
                    p_dict['B'] = nn.Parameter(tensor.to(self.device), requires_grad=True)
                else:
                    p_dict['B'].data.copy_(tensor.to(self.device))
