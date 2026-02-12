import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, List
try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

class LoRAWrapper(nn.Module):
    """
    Wraps a Linear or Conv1D layer with LoRA adapters.
    """
    def __init__(self, base_layer: nn.Module, rank: int, alpha: float, lora_A: nn.Parameter, lora_B: nn.Parameter):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = lora_A
        self.lora_B = lora_B
        
    def forward(self, x):
        # Base forward
        result = self.base_layer(x)
        
        # LoRA forward
        # Calculation: (x @ A.T @ B.T) * scaling
        lora_out = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result + lora_out

class LoRAManager:
    """
    Manages the lifecycle and storage of LoRA parameters.
    """
    def __init__(self, config: Dict, device="cuda"):
        self.rank = config.get("r", 8)
        self.alpha = config.get("alpha", 16)
        self.target_modules = config.get("target_modules", ["c_attn", "c_proj", "c_fc"])
        self.device = device
        
        # Store parameters: key -> {'A': Param, 'B': Param}
        self.params: Dict[str, Dict[str, nn.Parameter]] = {}
        
    def get_or_init_params(self, layer_id: int, module_name: str, in_features: int, out_features: int) -> Dict[str, nn.Parameter]:
        key = f"{layer_id}.{module_name}"
        
        if key not in self.params:
            lora_A = torch.zeros((self.rank, in_features), device=self.device)
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            
            lora_B = torch.zeros((out_features, self.rank), device=self.device)
            nn.init.zeros_(lora_B)
            
            self.params[key] = {
                'A': nn.Parameter(lora_A, requires_grad=True),
                'B': nn.Parameter(lora_B, requires_grad=True)
            }
            
        return self.params[key]

    def apply_lora(self, layer_id: int, module: nn.Module, module_name_prefix: str = ""):
        """
        Recursively replaces Linear/Conv1D layers with LoRAWrapper if they match target_modules.
        """
        for name, child in module.named_children():
            full_name = f"{module_name_prefix}.{name}" if module_name_prefix else name
            
            # Check if this is a target module
            is_target = any(name == target or name.endswith(target) for target in self.target_modules)
            
            if isinstance(child, LoRAWrapper) and is_target:
                # Already wrapped, just swap parameters for the new layer
                if isinstance(child.base_layer, nn.Linear):
                    in_features = child.base_layer.in_features
                    out_features = child.base_layer.out_features
                elif Conv1D is not None and isinstance(child.base_layer, Conv1D):
                    in_features = child.base_layer.weight.shape[0]
                    out_features = child.base_layer.weight.shape[1]
                else:
                    # Generic fallback if weight exists
                    in_features = child.base_layer.weight.shape[1] if hasattr(child.base_layer, "weight") else 0
                    out_features = child.base_layer.weight.shape[0] if hasattr(child.base_layer, "weight") else 0

                params = self.get_or_init_params(layer_id, full_name, in_features, out_features)
                child.lora_A = params['A']
                child.lora_B = params['B']
                continue

            in_features = None
            out_features = None
            
            if isinstance(child, nn.Linear) and is_target:
                in_features = child.in_features
                out_features = child.out_features
            elif Conv1D is not None and isinstance(child, Conv1D) and is_target:
                in_features = child.weight.shape[0]
                out_features = child.weight.shape[1]
                
            if in_features is not None and out_features is not None:
                params = self.get_or_init_params(layer_id, full_name, in_features, out_features)
                
                lora_layer = LoRAWrapper(
                    base_layer=child,
                    rank=self.rank,
                    alpha=self.alpha,
                    lora_A=params['A'],
                    lora_B=params['B']
                )
                setattr(module, name, lora_layer)
            else:
                self.apply_lora(layer_id, child, full_name)

    def update_lora_params(self, layer_id: int, module: nn.Module):
        """
        Efficiently updates LoRA parameters for a reused module.
        Uses cached wrapper list if available, otherwise traverses and builds cache.
        """
        if not hasattr(module, "_lora_cache"):
            module._lora_cache = []
            # First time: Traverse and collect wrappers
            # We reuse apply_lora logic but adapted for collection
            self._collect_and_update_wrappers(layer_id, module, "", module._lora_cache)
        else:
            # Fast path: Update parameters from cache
            for wrapper, name, in_f, out_f in module._lora_cache:
                params = self.get_or_init_params(layer_id, name, in_f, out_f)
                wrapper.lora_A = params['A']
                wrapper.lora_B = params['B']

    def _collect_and_update_wrappers(self, layer_id: int, module: nn.Module, prefix: str, cache: List):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRAWrapper):
                # Already wrapped (from previous usage or just now)
                in_f = child.base_layer.in_features if hasattr(child.base_layer, "in_features") else child.base_layer.weight.shape[1]
                out_f = child.base_layer.out_features if hasattr(child.base_layer, "out_features") else child.base_layer.weight.shape[0]
                
                params = self.get_or_init_params(layer_id, full_name, in_f, out_f)
                child.lora_A = params['A']
                child.lora_B = params['B']
                
                cache.append((child, full_name, in_f, out_f))
                continue
            
            # Check if this is a target module to wrap
            is_target = any(name == target or name.endswith(target) for target in self.target_modules)
            
            if is_target and (isinstance(child, nn.Linear) or (Conv1D is not None and isinstance(child, Conv1D))):
                in_features = child.in_features if isinstance(child, nn.Linear) else child.weight.shape[0]
                out_features = child.out_features if isinstance(child, nn.Linear) else child.weight.shape[1]
                
                params = self.get_or_init_params(layer_id, full_name, in_features, out_features)
                
                lora_layer = LoRAWrapper(
                    base_layer=child,
                    rank=self.rank,
                    alpha=self.alpha,
                    lora_A=params['A'],
                    lora_B=params['B']
                )
                setattr(module, name, lora_layer)
                cache.append((lora_layer, full_name, in_features, out_features))
            else:
                self._collect_and_update_wrappers(layer_id, child, full_name, cache)

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Returns a list of all nn.Parameter objects managed by this manager.
        """
        all_params = []
        for p_dict in self.params.values():
            all_params.append(p_dict['A'])
            all_params.append(p_dict['B'])
        return all_params

    def save_pretrained(self, save_directory: str):
        import os
        os.makedirs(save_directory, exist_ok=True)
        # Filter for only LoRA weights
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
            # full_key is e.g. "1.self_attn.q_proj.lora_A"
            parts = full_key.split(".")
            param_type = parts[-1] # lora_A or lora_B
            key = ".".join(parts[:-1]) # e.g. "1.self_attn.q_proj"
            
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