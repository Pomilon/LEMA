import torch
import torch.nn as nn
from transformers.models.lfm2_moe.modeling_lfm2_moe import (
    Lfm2MoeDecoderLayer, Lfm2MoeConfig, Lfm2MoeRotaryEmbedding, Lfm2MoeRMSNorm,
)
try:
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
except ImportError:
    create_causal_mask = create_recurrent_attention_mask = None
from typing import List, Dict, Any, Optional

from .base import LemaModelAdapter

class Lfm2Adapter(LemaModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hf_config = Lfm2MoeConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")

        self.rotary_emb = Lfm2MoeRotaryEmbedding(self.hf_config)

        self.module_pool: List[nn.Module] = []
        self._max_pool_size = 3
        self.param_mappings: Dict[int, List[tuple]] = {}

    def get_layer_metadata(self) -> List[Dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.num_hidden_layers):
            layer_type = self.hf_config.layer_types[i]
            layers.append({'id': i + 1, 'name': f'layers.{i}', 'type': 'block',
                           'block_index': i, 'sub_type': layer_type})
        layers.append({'id': self.hf_config.num_hidden_layers + 1, 'name': 'head', 'type': 'head'})
        return layers

    def _layer_param_names(self, idx: int) -> List[str]:
        prefix = f"model.layers.{idx}"
        names = [f"{prefix}.operator_norm.weight", f"{prefix}.ffn_norm.weight"]

        if self.hf_config.layer_types[idx] == "full_attention":
            names += [
                f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.q_layernorm.weight", f"{prefix}.self_attn.k_layernorm.weight",
            ]
        else:
            names += [
                f"{prefix}.conv.conv.weight", f"{prefix}.conv.in_proj.weight",
                f"{prefix}.conv.out_proj.weight",
            ]

        if idx < self.hf_config.num_dense_layers:
            names += [
                f"{prefix}.feed_forward.w1.weight", f"{prefix}.feed_forward.w2.weight",
                f"{prefix}.feed_forward.w3.weight",
            ]
        else:
            names.append(f"{prefix}.feed_forward.gate.weight")
            for ei in range(self.hf_config.num_experts):
                names += [
                    f"{prefix}.feed_forward.experts.{ei}.w1.weight",
                    f"{prefix}.feed_forward.experts.{ei}.w2.weight",
                    f"{prefix}.feed_forward.experts.{ei}.w3.weight",
                ]
        return names

    def get_param_names_for_layer(self, layer_id: int) -> List[str]:
        if layer_id == 0:
            return ['model.embed_tokens.weight']
        elif 1 <= layer_id <= self.hf_config.num_hidden_layers:
            return self._layer_param_names(layer_id - 1)
        elif layer_id == self.hf_config.num_hidden_layers + 1:
            names = ['model.embedding_norm.weight']
            if not self.hf_config.tie_word_embeddings:
                names.append('lm_head.weight')
            return names
        return []

    def construct_layer_module(self, layer_id: int, flat_buffer: Optional[torch.Tensor] = None, lora_manager: Optional[Any] = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")

        module = None
        for i, m in enumerate(self.module_pool):
            if layer_id == 0 and isinstance(m, Lfm2EmbeddingsLayer):
                module = self.module_pool.pop(i); break
            elif layer_id == self.hf_config.num_hidden_layers + 1 and isinstance(m, Lfm2HeadLayer):
                module = self.module_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.num_hidden_layers and isinstance(m, Lfm2MoeDecoderLayer):
                module = self.module_pool.pop(i); break

        if module is None:
            dtype_str = self.config.get("dtype", "float32")
            target_dtype = getattr(torch, dtype_str) if dtype_str else torch.float32
            if layer_id == 0:
                module = Lfm2EmbeddingsLayer(self.hf_config)
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                module = Lfm2HeadLayer(self.hf_config)
            else:
                module = Lfm2MoeDecoderLayer(self.hf_config, layer_idx=layer_id - 1)
            module.to(device=device, dtype=target_dtype)

            if lora_manager and 1 <= layer_id <= self.hf_config.num_hidden_layers:
                lora_manager.update_lora_params(layer_id, module)

            self.param_mappings[id(module)] = self._create_mapping(layer_id, module)

        if flat_buffer is not None and next(module.parameters()).device != flat_buffer.device:
            module.to(flat_buffer.device)

        if hasattr(module, "layer_idx") and 1 <= layer_id <= self.hf_config.num_hidden_layers:
            module.layer_idx = layer_id - 1

        if flat_buffer is not None:
            mapping = self.param_mappings[id(module)]
            with torch.no_grad():
                for param, offset, numel, shape in mapping:
                    param.data.copy_(flat_buffer[offset:offset+numel].view(shape), non_blocking=True)

        return module

    def _create_mapping(self, layer_id: int, module: nn.Module) -> List[tuple]:
        names = self.get_param_names_for_layer(layer_id)
        module_params = dict(module.named_parameters())
        mapping = []
        offset = 0
        for full_name in names:
            if layer_id == 0:
                clean_k = "embed_tokens.weight"
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                clean_k = "embedding_norm.weight" if "embedding_norm" in full_name else "lm_head.weight"
            else:
                idx = layer_id - 1
                prefix = f"model.layers.{idx}."
                clean_k = full_name[len(prefix):]

            if clean_k not in module_params:
                lora_k = clean_k.replace(".weight", ".base_layer.weight")
                if lora_k in module_params:
                    clean_k = lora_k
                else:
                    found = False
                    for mk in module_params.keys():
                        if mk == clean_k or mk.endswith("." + clean_k) or mk.replace(".base_layer", "") == clean_k:
                            clean_k = mk
                            found = True
                            break
                    if not found:
                        raise KeyError(f"Could not find parameter {clean_k} (from {full_name}) in module. Available: {list(module_params.keys())}")

            param = module_params[clean_k]
            numel = param.numel()
            mapping.append((param, offset, numel, param.shape))
            offset += numel
        return mapping

    def release_layer_module(self, module: nn.Module):
        if len(self.module_pool) < self._max_pool_size:
            self.module_pool.append(module)
        else:
            for p in module.parameters():
                del p.data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        if not isinstance(layer_module, Lfm2MoeDecoderLayer):
            if isinstance(layer_module, Lfm2EmbeddingsLayer):
                return layer_module(hidden_states)
            if isinstance(layer_module, Lfm2HeadLayer):
                return layer_module(hidden_states)
            return layer_module(hidden_states)

        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
        elif not hasattr(self, "_cache_seq") or self._cache_seq != seq_len:
            self._cache_seq = seq_len
            self._cache_pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            self._cache_rope = None
            position_ids = self._cache_pos
        else:
            position_ids = self._cache_pos

        if self._cache_rope is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            if cos.ndim == 2:
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            self._cache_rope = (cos, sin)
        else:
            cos, sin = self._cache_rope

        idx = layer_module.layer_idx if hasattr(layer_module, "layer_idx") else 0
        layer_type = self.hf_config.layer_types[idx]

        if not hasattr(self, "_cache_masks") or self._cache_seq != seq_len:
            self._cache_masks = {}
        mask_key = f"{layer_type}_{seq_len}"
        if mask_key in self._cache_masks:
            attention_mask = self._cache_masks[mask_key]
        elif create_causal_mask is not None:
            mask_kwargs = {
                "config": self.hf_config,
                "inputs_embeds": hidden_states,
                "attention_mask": None,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            if layer_type == "full_attention":
                mask = create_causal_mask(**mask_kwargs)
            else:
                mask = create_recurrent_attention_mask(**mask_kwargs)
            self._cache_masks[mask_key] = mask
            attention_mask = mask
        else:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            attention_mask = mask[None, None, :, :]

        return layer_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
        )

    @property
    def hidden_size(self) -> int:
        return self.hf_config.hidden_size


class Lfm2EmbeddingsLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    def forward(self, x):
        return self.embed_tokens(x)


class Lfm2HeadLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, x):
        return self.lm_head(self.embedding_norm(x))
