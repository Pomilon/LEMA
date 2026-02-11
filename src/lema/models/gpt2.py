import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config
from typing import List, Dict, Any, Optional

from .base import LemaModelAdapter

class GPT2Adapter(LemaModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hf_config = GPT2Config(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")
        self.layer_pool: List[nn.Module] = []
        self.param_mappings: Dict[int, List[tuple]] = {}
        
    def get_layer_metadata(self) -> List[Dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.n_layer):
            layers.append({'id': i + 1, 'name': f'h.{i}', 'type': 'block', 'block_index': i})
        layers.append({'id': self.hf_config.n_layer + 1, 'name': 'head', 'type': 'head'})
        return layers

    def get_param_names_for_layer(self, layer_id: int) -> List[str]:
        if layer_id == 0:
            return ['transformer.wte.weight', 'transformer.wpe.weight']
        elif 1 <= layer_id <= self.hf_config.n_layer:
            idx = layer_id - 1
            prefix = f"transformer.h.{idx}"
            return [
                f"{prefix}.attn.c_attn.weight", f"{prefix}.attn.c_attn.bias",
                f"{prefix}.attn.c_proj.weight", f"{prefix}.attn.c_proj.bias",
                f"{prefix}.ln_1.weight", f"{prefix}.ln_1.bias",
                f"{prefix}.ln_2.weight", f"{prefix}.ln_2.bias",
                f"{prefix}.mlp.c_fc.weight", f"{prefix}.mlp.c_fc.bias",
                f"{prefix}.mlp.c_proj.weight", f"{prefix}.mlp.c_proj.bias",
            ]
        elif layer_id == self.hf_config.n_layer + 1:
            return ['transformer.ln_f.weight', 'transformer.ln_f.bias', 'lm_head.weight']
        return []

    def construct_layer_module(self, layer_id: int, flat_buffer: Optional[torch.Tensor] = None, lora_manager: Optional[Any] = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")
        module = None
        for i, m in enumerate(self.layer_pool):
            if layer_id == 0 and isinstance(m, GPT2EmbeddingsLayer):
                module = self.layer_pool.pop(i); break
            elif layer_id == self.hf_config.n_layer + 1 and isinstance(m, GPT2HeadLayer):
                module = self.layer_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.n_layer and isinstance(m, GPT2Block):
                module = self.layer_pool.pop(i); break
        
        if module is None:
            if layer_id == 0: module = GPT2EmbeddingsLayer(self.hf_config)
            elif layer_id == self.hf_config.n_layer + 1: module = GPT2HeadLayer(self.hf_config)
            else:
                module = GPT2Block(self.hf_config)
            
            # Initialization only
            module.to(device)

        if lora_manager and 1 <= layer_id <= self.hf_config.n_layer:
            lora_manager.update_lora_params(layer_id, module)

        if id(module) not in self.param_mappings:
            self.param_mappings[id(module)] = self._create_mapping(layer_id, module)

        if flat_buffer is not None:
            mapping = self.param_mappings[id(module)]
            offset = 0
            with torch.no_grad():
                for param, numel, shape in mapping:
                    param.data.copy_(flat_buffer[offset : offset + numel].view(shape), non_blocking=True)
                    offset += numel
            
        return module

    def _create_mapping(self, layer_id: int, module: nn.Module) -> List[tuple]:
        names = self.get_param_names_for_layer(layer_id)
        idx = layer_id - 1
        module_params = dict(module.named_parameters())
        mapping = []
        for full_name in names:
            if layer_id == 0:
                clean_k = "wte.weight" if "wte" in full_name else "wpe.weight"
            elif layer_id == self.hf_config.n_layer + 1:
                if "ln_f" in full_name: clean_k = "ln_f.weight" if "weight" in full_name else "ln_f.bias"
                else: clean_k = "head.weight"
            else:
                prefix = f"transformer.h.{idx}."
                clean_k = full_name[len(prefix):]
                if clean_k not in module_params: clean_k = clean_k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            param = module_params[clean_k]
            mapping.append((param, param.numel(), param.shape))
        return mapping

    def release_layer_module(self, module: nn.Module):
        if len(self.layer_pool) < 5:
            self.layer_pool.append(module)

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(layer_module, GPT2Block):
            return layer_module(hidden_states)[0]
        return layer_module(hidden_states)

    @property
    def hidden_size(self) -> int:
        return self.hf_config.n_embd

class GPT2EmbeddingsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
    def forward(self, input_ids):
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device).unsqueeze(0)
        return self.wte(input_ids) + self.wpe(position_ids)

class GPT2HeadLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, x): return self.head(self.ln_f(x))