from __future__ import annotations

import torch
import torch.nn as nn
try:
    from transformers.models.lfm2_moe.modeling_lfm2_moe import (
        Lfm2MoeDecoderLayer, Lfm2MoeConfig, Lfm2MoeRotaryEmbedding, Lfm2MoeRMSNorm,
    )
except ImportError:
    Lfm2MoeDecoderLayer = Lfm2MoeConfig = Lfm2MoeRotaryEmbedding = Lfm2MoeRMSNorm = None

try:
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
except ImportError:
    create_causal_mask = create_recurrent_attention_mask = None
from typing import Any

from ._base import LemaModelAdapter
from .._quantized_linear import QuantizedLinear
import torch.nn.functional as F
from transformers.activations import ACT2FN


class Lfm2Adapter(LemaModelAdapter):
    MODEL_TYPE = "lfm2_moe"
    MAX_POOL_SIZE = 3
    supports_quantized = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.hf_config = Lfm2MoeConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")
        if getattr(self.hf_config, "_experts_implementation", None) is None:
            self.hf_config._experts_implementation = "grouped_mm"

        self.rotary_emb = Lfm2MoeRotaryEmbedding(self.hf_config)

        self.module_pool: list[nn.Module] = []
        self.param_mappings: dict[int, list[tuple]] = {}

    def get_layer_metadata(self) -> list[dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.num_hidden_layers):
            layer_type = self.hf_config.layer_types[i]
            layers.append({'id': i + 1, 'name': f'layers.{i}', 'type': 'block',
                           'block_index': i, 'sub_type': layer_type})
        layers.append({'id': self.hf_config.num_hidden_layers + 1, 'name': 'head', 'type': 'head'})
        return layers

    def _layer_param_names(self, idx: int) -> list[str]:
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
            names += [
                f"{prefix}.feed_forward.gate.weight",
                f"{prefix}.feed_forward.experts.gate_up_proj",
                f"{prefix}.feed_forward.experts.down_proj",
            ]
        return names

    def get_param_names_for_layer(self, layer_id: int) -> list[str]:
        if layer_id == 0:
            return ['model.embed_tokens.weight']
        elif 1 <= layer_id <= self.hf_config.num_hidden_layers:
            return self._layer_param_names(layer_id - 1)
        elif layer_id == self.hf_config.num_hidden_layers + 1:
            names = ['model.embedding_norm.weight']
            names.append('lm_head.weight')
            return names
        return []

    def load_tensor(self, gbi: Any, name: str) -> torch.Tensor:
        if name in set(gbi.get_keys()):
            return super().load_tensor(gbi, name)
        if ".feed_forward.experts.gate_up_proj" in name:
            prefix = name.rsplit(".experts.gate_up_proj", 1)[0]
            per_expert = []
            for e in range(self.hf_config.num_experts):
                w1 = gbi.load_tensors([f"{prefix}.experts.{e}.w1.weight"], device="cpu")
                w3 = gbi.load_tensors([f"{prefix}.experts.{e}.w3.weight"], device="cpu")
                per_expert.append(torch.cat([w1[f"{prefix}.experts.{e}.w1.weight"],
                                             w3[f"{prefix}.experts.{e}.w3.weight"]], dim=0))
            return torch.stack(per_expert, dim=0)
        if ".feed_forward.experts.down_proj" in name:
            prefix = name.rsplit(".experts.down_proj", 1)[0]
            per_expert = []
            for e in range(self.hf_config.num_experts):
                w2 = gbi.load_tensors([f"{prefix}.experts.{e}.w2.weight"], device="cpu")
                per_expert.append(w2[f"{prefix}.experts.{e}.w2.weight"])
            return torch.stack(per_expert, dim=0)
        if name == "lm_head.weight" and name not in set(gbi.get_keys()):
            embed = gbi.load_tensors(["model.embed_tokens.weight"], device="cpu")
            return embed["model.embed_tokens.weight"]
        return super().load_tensor(gbi, name)

    def get_tensor_shape(self, gbi: Any, name: str) -> tuple | None:
        if name in set(gbi.get_keys()):
            return super().get_tensor_shape(gbi, name)
        if ".feed_forward.experts.gate_up_proj" in name:
            prefix = name.rsplit(".experts.gate_up_proj", 1)[0]
            w1_shape = gbi.get_tensor_shape(f"{prefix}.experts.0.w1.weight")
            if w1_shape is not None:
                return (self.hf_config.num_experts, 2 * w1_shape[0], w1_shape[1])
        if ".feed_forward.experts.down_proj" in name:
            prefix = name.rsplit(".experts.down_proj", 1)[0]
            w2_shape = gbi.get_tensor_shape(f"{prefix}.experts.0.w2.weight")
            if w2_shape is not None:
                return (self.hf_config.num_experts, w2_shape[0], w2_shape[1])
        if name == "lm_head.weight" and name not in set(gbi.get_keys()):
            return gbi.get_tensor_shape("model.embed_tokens.weight")
        return super().get_tensor_shape(gbi, name)

    def supports_quantized_layer(self, layer_id: int) -> bool:
        if not (1 <= layer_id <= self.hf_config.num_hidden_layers):
            return False
        return self.hf_config.layer_types[layer_id - 1] == "full_attention"

    def _construct_quantized_layer(self, layer_id: int, flat: torch.Tensor, full_ft_manager: Any = None) -> nn.Module:
        module = QuantizedLfm2Layer(self.hf_config, layer_id - 1)
        module.to(device=flat.device)
        transfer = getattr(self, "transfer_engine", None)
        if transfer is None:
            raise RuntimeError("Quantized layer construction requires the transfer engine scales")
        scale_all = transfer.get_layer_scale(layer_id, 0)
        if scale_all is None:
            raise RuntimeError(f"Missing quantization scale for layer {layer_id}")
        scale_all = scale_all.to(flat.device)
        idx = layer_id - 1
        prefix = f"model.layers.{idx}."
        is_moe = idx >= self.hf_config.num_dense_layers
        offset = 0
        s_off = 0
        with torch.no_grad():
            for full_name in self.get_param_names_for_layer(layer_id):
                clean = full_name[len(prefix):]
                if clean == "operator_norm.weight":
                    n = module.operator_norm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.operator_norm.weight.shape)
                    module.operator_norm.weight.data.copy_(
                        q_slice.float() * scale_all[s_off:s_off + 1].view(-1), non_blocking=True)
                    offset += n; s_off += 1
                    continue
                if clean == "ffn_norm.weight":
                    n = module.ffn_norm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.ffn_norm.weight.shape)
                    module.ffn_norm.weight.data.copy_(
                        q_slice.float() * scale_all[s_off:s_off + 1].view(-1), non_blocking=True)
                    offset += n; s_off += 1
                    continue
                if clean == "self_attn.q_proj.weight":
                    shape = (module.self_attn.q_proj.out_features, module.self_attn.q_proj.in_features)
                    numel = shape[0] * shape[1]
                    module.self_attn.q_proj.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "self_attn.k_proj.weight":
                    shape = (module.self_attn.k_proj.out_features, module.self_attn.k_proj.in_features)
                    numel = shape[0] * shape[1]
                    module.self_attn.k_proj.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "self_attn.v_proj.weight":
                    shape = (module.self_attn.v_proj.out_features, module.self_attn.v_proj.in_features)
                    numel = shape[0] * shape[1]
                    module.self_attn.v_proj.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "self_attn.out_proj.weight":
                    shape = (module.self_attn.out_proj.out_features, module.self_attn.out_proj.in_features)
                    numel = shape[0] * shape[1]
                    module.self_attn.out_proj.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "self_attn.q_layernorm.weight":
                    n = module.self_attn.q_layernorm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.self_attn.q_layernorm.weight.shape)
                    module.self_attn.q_layernorm.weight.data.copy_(
                        q_slice.float() * scale_all[s_off:s_off + 1].view(-1), non_blocking=True)
                    offset += n; s_off += 1
                    continue
                if clean == "self_attn.k_layernorm.weight":
                    n = module.self_attn.k_layernorm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.self_attn.k_layernorm.weight.shape)
                    module.self_attn.k_layernorm.weight.data.copy_(
                        q_slice.float() * scale_all[s_off:s_off + 1].view(-1), non_blocking=True)
                    offset += n; s_off += 1
                    continue
                if clean == "conv.conv.weight":
                    continue
                if clean == "conv.in_proj.weight":
                    continue
                if clean == "conv.out_proj.weight":
                    continue
                if clean == "feed_forward.w1.weight":
                    shape = (module.feed_forward.w1.out_features, module.feed_forward.w1.in_features)
                    numel = shape[0] * shape[1]
                    module.feed_forward.w1.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "feed_forward.w2.weight":
                    shape = (module.feed_forward.w2.out_features, module.feed_forward.w2.in_features)
                    numel = shape[0] * shape[1]
                    module.feed_forward.w2.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "feed_forward.w3.weight":
                    shape = (module.feed_forward.w3.out_features, module.feed_forward.w3.in_features)
                    numel = shape[0] * shape[1]
                    module.feed_forward.w3.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "feed_forward.gate.weight":
                    shape = (module.feed_forward.gate.weight.shape[0], module.feed_forward.gate.weight.shape[1])
                    numel = shape[0] * shape[1]
                    module.feed_forward.gate.weight.data.copy_(
                        flat[offset:offset+numel].view(shape).float() * scale_all[s_off:s_off+1].view(-1, 1), non_blocking=True)
                    offset += numel; s_off += 1
                    continue
                if clean == "feed_forward.experts.gate_up_proj":
                    shape = tuple(module.feed_forward.experts.gate_up_proj.shape)
                    numel = 1
                    for d in shape: numel *= d
                    q_slice = flat[offset:offset+numel].view(shape)
                    s = scale_all[s_off:s_off+1].view(-1).item() if scale_all.numel() > s_off else 1.0
                    module.feed_forward.experts.gate_up_proj.data.copy_(q_slice.float() * s, non_blocking=True)
                    offset += numel; s_off += 1
                    continue
                if clean == "feed_forward.experts.down_proj":
                    shape = tuple(module.feed_forward.experts.down_proj.shape)
                    numel = 1
                    for d in shape: numel *= d
                    q_slice = flat[offset:offset+numel].view(shape)
                    s = scale_all[s_off:s_off+1].view(-1).item() if scale_all.numel() > s_off else 1.0
                    module.feed_forward.experts.down_proj.data.copy_(q_slice.float() * s, non_blocking=True)
                    offset += numel; s_off += 1
                    continue
                raise KeyError(f"Unmapped quantized param {full_name}")
        if full_ft_manager is not None:
            full_ft_manager.apply_to_module(layer_id, module)
        if self._is_generation_mode():
            module.eval()
        return module

    def construct_layer_module(self, layer_id: int, flat_buffer: torch.Tensor | None = None, lora_manager: Any = None, full_ft_manager: Any = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")

        if (1 <= layer_id <= self.hf_config.num_hidden_layers
                and flat_buffer is not None and flat_buffer.dtype == torch.int8
                and self.hf_config.layer_types[layer_id - 1] == "full_attention"):
            if lora_manager is not None:
                raise RuntimeError(
                    "LoRA adapters are not supported on W8A8 int8 buffers; "
                    "use allow_quantized=False to receive dequantized weights"
                )
            return self._construct_quantized_layer(layer_id, flat_buffer, full_ft_manager)

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

        if full_ft_manager is not None:
            full_ft_manager.apply_to_module(layer_id, module)

        if self._is_generation_mode():
            module.eval()
        return module

    def get_module_param_name(self, layer_id: int, full_param_name: str) -> str:
        if layer_id == 0:
            return "embed_tokens.weight"
        elif layer_id == self.hf_config.num_hidden_layers + 1:
            return "embedding_norm.weight" if "embedding_norm" in full_param_name else "lm_head.weight"
        else:
            prefix = f"model.layers.{layer_id - 1}."
            return full_param_name[len(prefix):]

    def _create_mapping(self, layer_id: int, module: nn.Module) -> list[tuple]:
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
        if isinstance(module, QuantizedLfm2Layer):
            return
        if len(self.module_pool) < self.MAX_POOL_SIZE:
            self.module_pool.append(module)
        else:
            del module
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def chunked_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                              kv_store, layer_id: int, kv_chunk_size: int = 8192) -> torch.Tensor:
        from ._chunked_rope import lfm2_chunked_forward_layer, compute_rope
        idx = layer_module.layer_idx if hasattr(layer_module, "layer_idx") else 0
        if self.hf_config.layer_types[idx] != "full_attention":
            return self.forward_layer(layer_module, hidden_states)
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if cos.ndim == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        return lfm2_chunked_forward_layer(layer_module, hidden_states, kv_store, layer_id,
                                          kv_chunk_size, cos, sin)

    def decode_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                             kv_store, layer_id: int, kv_chunk_size: int = 8192,
                             position: int = 0, is_new_token: bool = True) -> torch.Tensor:
        from ._chunked_rope import lfm2_decode_forward_layer
        idx = layer_module.layer_idx if hasattr(layer_module, "layer_idx") else 0
        if self.hf_config.layer_types[idx] != "full_attention":
            return self.forward_layer(layer_module, hidden_states)
        pos_ids = torch.tensor([[position]], dtype=torch.long, device=hidden_states.device)
        cos, sin = self.rotary_emb(hidden_states, pos_ids)
        if cos.ndim == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        return lfm2_decode_forward_layer(layer_module, hidden_states, kv_store, layer_id,
                                         kv_chunk_size, cos, sin, is_new_token=is_new_token)

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        if isinstance(layer_module, QuantizedLfm2Layer):
            kv_store = kwargs.get("kv_store")
            layer_id = kwargs.get("layer_id")
            kv_chunk_size = kwargs.get("kv_chunk_size", 0)
            if kv_store is not None and kv_chunk_size > 0:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store, layer_id, kv_chunk_size)
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
                mask_kwargs = {"config": self.hf_config, "inputs_embeds": hidden_states, "attention_mask": None, "past_key_values": None, "position_ids": position_ids}
                if layer_type == "full_attention":
                    mask = create_causal_mask(**mask_kwargs)
                else:
                    mask = create_recurrent_attention_mask(**mask_kwargs)
                self._cache_masks[mask_key] = mask
                attention_mask = mask
            else:
                mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
                attention_mask = mask[None, None, :, :]
            return layer_module(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=(cos, sin))
        kv_store = kwargs.get("kv_store")
        layer_id = kwargs.get("layer_id")
        kv_chunk_size = kwargs.get("kv_chunk_size", 0)
        if isinstance(layer_module, Lfm2MoeDecoderLayer):
            if kv_store is not None and kv_chunk_size > 0:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store,
                                                   layer_id, kv_chunk_size)
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


class QuantizedLfm2Layer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = _QuantizedLfm2Attention(config, layer_idx)
        else:
            self.self_attn = None
            self.conv = None
        is_moe = layer_idx >= config.num_dense_layers
        if is_moe:
            self.feed_forward = _QuantizedLfm2MoE(config)
        else:
            self.feed_forward = _QuantizedLfm2MLP(config)
        self.operator_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, position_embeddings=None) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            hidden_states = hidden_states + torch.zeros_like(hidden_states)
            hidden_states = hidden_states + residual * 0
            residual = hidden_states
            hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
            return hidden_states
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


class _QuantizedLfm2Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = QuantizedLinear(config.hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = QuantizedLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = QuantizedLinear(config.hidden_size, config.num_key_value_heads * self.head_dim)
        self.out_proj = QuantizedLinear(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.q_layernorm = Lfm2MoeRMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = Lfm2MoeRMSNorm(self.head_dim, eps=config.norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, position_ids=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        key_states = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        from transformers.models.lfm2_moe.modeling_lfm2_moe import apply_rotary_pos_emb as _rope
        query_states, key_states = _rope(query_states, key_states, cos, sin)
        from transformers.models.lfm2_moe.modeling_lfm2_moe import eager_attention_forward as _eager
        attn_output, _ = _eager(self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=self.scaling)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn_output), None


class _QuantizedLfm2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = QuantizedLinear(config.hidden_size, config.intermediate_size)
        self.w3 = QuantizedLinear(config.hidden_size, config.intermediate_size)
        self.w2 = QuantizedLinear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _QuantizedLfm2MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * config.moe_intermediate_size, config.hidden_size))
        self.experts.down_proj = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.moe_intermediate_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        flat = x.view(-1, hidden_dim)
        router_logits = F.linear(flat, self.gate.weight)
        router_logits = torch.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        final_hidden = torch.zeros_like(flat)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            cur = flat[token_idx]
            gate, up = F.linear(cur, self.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            cur = F.silu(gate) * up
            cur = F.linear(cur, self.experts.down_proj[expert_idx])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, cur.to(final_hidden.dtype))
        return final_hidden.view(batch_size, seq_len, hidden_dim)
