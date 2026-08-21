from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig, LlamaRotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv,
)
from typing import Any

from ._base import LemaModelAdapter
from .._quantized_linear import QuantizedLinear


class LlamaAdapter(LemaModelAdapter):
    MODEL_TYPE = "llama"
    MAX_POOL_SIZE = 3
    supports_quantized = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.hf_config = LlamaConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")

        try:
            self.rotary_emb = LlamaRotaryEmbedding(self.hf_config)
        except TypeError:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.hf_config.hidden_size // self.hf_config.num_attention_heads,
                max_position_embeddings=self.hf_config.max_position_embeddings
            )

        self.module_pool: list[nn.Module] = []  # Sliding window pool
        self.param_mappings: dict[int, list[tuple]] = {}

    def get_layer_metadata(self) -> list[dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.num_hidden_layers):
            layers.append({'id': i + 1, 'name': f'layers.{i}', 'type': 'block', 'block_index': i})
        layers.append({'id': self.hf_config.num_hidden_layers + 1, 'name': 'head', 'type': 'head'})
        return layers

    def get_param_names_for_layer(self, layer_id: int) -> list[str]:
        if layer_id == 0:
            return ['model.embed_tokens.weight']
        elif 1 <= layer_id <= self.hf_config.num_hidden_layers:
            idx = layer_id - 1
            prefix = f"model.layers.{idx}"
            return [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        elif layer_id == self.hf_config.num_hidden_layers + 1:
            return ['model.norm.weight', 'lm_head.weight']
        return []

    def construct_layer_module(self, layer_id: int, flat_buffer: torch.Tensor | None = None, lora_manager: Any = None, full_ft_manager: Any = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")

        if (1 <= layer_id <= self.hf_config.num_hidden_layers
                and flat_buffer is not None and flat_buffer.dtype == torch.int8):
            return self._construct_quantized_layer(layer_id, flat_buffer)

        # Pop matching module from sliding-window pool, or create on CPU
        module = None
        for i, m in enumerate(self.module_pool):
            if layer_id == 0 and isinstance(m, LlamaEmbeddingsLayer):
                module = self.module_pool.pop(i); break
            elif layer_id == self.hf_config.num_hidden_layers + 1 and isinstance(m, LlamaHeadLayer):
                module = self.module_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.num_hidden_layers and isinstance(m, LlamaDecoderLayer):
                module = self.module_pool.pop(i); break

        if module is None:
            dtype_str = self.config.get("dtype", "float32")
            target_dtype = getattr(torch, dtype_str) if dtype_str else torch.float32
            if layer_id == 0:
                module = LlamaEmbeddingsLayer(self.hf_config, None)
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                module = LlamaHeadLayer(self.hf_config, None)
            else:
                module = LlamaDecoderLayer(self.hf_config, layer_idx=0)
            module.to(device=device, dtype=target_dtype)

            if lora_manager and 1 <= layer_id <= self.hf_config.num_hidden_layers:
                lora_manager.update_lora_params(layer_id, module)

            self.param_mappings[id(module)] = self._create_mapping(layer_id, module)

        # Move to GPU if needed (sliding window: module created on CPU, moved to GPU for compute)
        if flat_buffer is not None and next(module.parameters()).device != flat_buffer.device:
            module.to(flat_buffer.device)

        if hasattr(module, "layer_idx") and 1 <= layer_id <= self.hf_config.num_hidden_layers:
            module.layer_idx = layer_id - 1

        # Copy weights from flat buffer
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
            return "norm.weight" if "model.norm" in full_param_name else "lm_head.weight"
        else:
            prefix = f"model.layers.{layer_id - 1}."
            return full_param_name[len(prefix):]

    def supports_quantized_layer(self, layer_id: int) -> bool:
        return 1 <= layer_id <= self.hf_config.num_hidden_layers

    def _construct_quantized_layer(self, layer_id: int, flat: torch.Tensor) -> nn.Module:
        module = QuantizedLlamaLayer(self.hf_config)
        module.to(device=flat.device)
        transfer = getattr(self, "transfer_engine", None)
        if transfer is None:
            raise RuntimeError("Quantized layer construction requires the transfer engine scales")
        scale_all = transfer.get_layer_scale(layer_id, 0)
        if scale_all is None:
            raise RuntimeError(f"Missing quantization scale for layer {layer_id}")
        scale_all = scale_all.to(flat.device)
        lin_by_name = {
            "self_attn.q_proj": module.self_attn.q_proj,
            "self_attn.k_proj": module.self_attn.k_proj,
            "self_attn.v_proj": module.self_attn.v_proj,
            "self_attn.o_proj": module.self_attn.o_proj,
            "mlp.gate_proj": module.mlp.gate_proj,
            "mlp.up_proj": module.mlp.up_proj,
            "mlp.down_proj": module.mlp.down_proj,
        }
        norm_by_name = {
            "input_layernorm": module.input_layernorm,
            "post_attention_layernorm": module.post_attention_layernorm,
        }
        prefix = f"model.layers.{layer_id - 1}."
        offset = 0
        s_off = 0
        with torch.no_grad():
            for full_name in self.get_param_names_for_layer(layer_id):
                clean = full_name[len(prefix):]
                if clean.endswith(".weight"):
                    base = clean[:-len(".weight")]
                    lin = lin_by_name.get(base)
                    if lin is not None:
                        shape = (lin.out_features, lin.in_features)
                        numel = shape[0] * shape[1]
                        lin.set_weight_from_int8(
                            flat[offset:offset + numel].view(shape),
                            scale_all[s_off:s_off + shape[0]],
                        )
                        offset += numel
                        s_off += shape[0]
                        continue
                    norm = norm_by_name.get(base)
                    if norm is not None:
                        n = norm.weight.numel()
                        q_slice = flat[offset:offset + n].view(norm.weight.shape)
                        norm.weight.data.copy_(
                            q_slice.float() * scale_all[s_off:s_off + 1].view(-1),
                            non_blocking=True,
                        )
                        offset += n
                        s_off += 1
                        continue
                raise KeyError(f"Unmapped quantized param {full_name}")
        if self._is_generation_mode():
            module.eval()
        return module

    def _create_mapping(self, layer_id: int, module: nn.Module) -> list[tuple]:
        names = self.get_param_names_for_layer(layer_id)
        idx = layer_id - 1
        module_params = dict(module.named_parameters())
        mapping = []
        offset = 0
        for full_name in names:
            if layer_id == 0:
                clean_k = "embed_tokens.weight"
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                clean_k = "norm.weight" if "model.norm" in full_name else "lm_head.weight"
            else:
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
        """Return module to sliding-window pool. Discarded modules free their GPU memory."""
        if isinstance(module, QuantizedLlamaLayer):
            return
        if len(self.module_pool) < self.MAX_POOL_SIZE:
            self.module_pool.append(module)
        else:
            del module
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def chunked_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                              kv_store, layer_id: int, kv_chunk_size: int = 8192) -> torch.Tensor:
        from ._chunked_rope import rope_chunked_forward_layer, compute_rope
        seq_len = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cos, sin = compute_rope(self, layer_module.self_attn, hidden_states, position_ids)
        return rope_chunked_forward_layer(layer_module, hidden_states, kv_store, layer_id,
                                          kv_chunk_size, cos, sin)

    def decode_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                             kv_store, layer_id: int, kv_chunk_size: int = 8192,
                             position: int = 0, is_new_token: bool = True) -> torch.Tensor:
        from ._chunked_rope import rope_decode_forward_layer, compute_rope
        pos_ids = torch.tensor([[position]], dtype=torch.long, device=hidden_states.device)
        cos, sin = compute_rope(self, layer_module.self_attn, hidden_states, pos_ids)
        return rope_decode_forward_layer(layer_module, hidden_states, kv_store, layer_id,
                                         kv_chunk_size, cos, sin, is_new_token=is_new_token)

    def _attn_context(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                      kwargs: dict[str, Any]):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            attention_mask = getattr(self, "_cache_mask", None)
            if attention_mask is None:
                mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)
        elif not hasattr(self, "_cache_seq") or self._cache_seq != seq_len:
            self._cache_seq = seq_len
            self._cache_pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._cache_mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)
            self._cache_rope = None
            position_ids = self._cache_pos
            attention_mask = self._cache_mask
        else:
            position_ids = self._cache_pos
            attention_mask = self._cache_mask

        if not hasattr(self, "_cache_rope") or self._cache_rope is None:
            attn = layer_module.self_attn
            try:
                if hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
                    try: cos, sin = attn.rotary_emb(hidden_states, position_ids)
                    except: cos, sin = attn.rotary_emb(position_ids)
                else:
                    cos, sin = self.rotary_emb(hidden_states, position_ids)
            except Exception:
                head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
                cos, sin = self.rotary_emb(torch.zeros(batch_size, seq_len, head_dim, device=device), position_ids)
            if cos.ndim == 2:
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            elif cos.ndim == 4:
                cos, sin = cos.squeeze(1), sin.squeeze(1)
            if cos.shape[0] != batch_size and cos.shape[0] == 1:
                cos, sin = cos.expand(batch_size, -1, -1), sin.expand(batch_size, -1, -1)
            self._cache_rope = (cos, sin)
        else:
            cos, sin = self._cache_rope

        return position_ids, attention_mask, cos, sin

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        kv_store = kwargs.get("kv_store")
        layer_id = kwargs.get("layer_id")
        kv_chunk_size = kwargs.get("kv_chunk_size", 0)
        if isinstance(layer_module, QuantizedLlamaLayer):
            if kv_store is not None and kv_chunk_size > 0:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store,
                                                  layer_id, kv_chunk_size)
            position_ids, attention_mask, cos, sin = self._attn_context(layer_module, hidden_states, kwargs)
            return layer_module(hidden_states, attention_mask=attention_mask, cos=cos, sin=sin)
        if not isinstance(layer_module, LlamaDecoderLayer):
            return layer_module(hidden_states)

        if kv_store is not None and kv_chunk_size > 0:
            return self.chunked_forward_layer(layer_module, hidden_states, kv_store,
                                              layer_id, kv_chunk_size)

        position_ids, attention_mask, cos, sin = self._attn_context(layer_module, hidden_states, kwargs)

        # Forward
        residual = hidden_states
        hidden_states = layer_module.input_layernorm(hidden_states)
        if torch.is_grad_enabled() and kwargs.get("gradient_checkpointing", False):
            attn_output = checkpoint(
                lambda x: layer_module.self_attn(hidden_states=x, attention_mask=attention_mask,
                    position_ids=position_ids, position_embeddings=(cos, sin))[0],
                hidden_states, use_reentrant=False)
        else:
            attn_output = layer_module.self_attn(hidden_states=hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, position_embeddings=(cos, sin))[0]
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = layer_module.post_attention_layernorm(hidden_states)
        hidden_states = layer_module.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @property
    def hidden_size(self) -> int:
        return self.hf_config.hidden_size


class LlamaEmbeddingsLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    def forward(self, x): return self.embed_tokens(x)


class LlamaHeadLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, x): return self.lm_head(self.norm(x))


class QuantizedLlamaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _QuantizedLlamaAttention(config)
        self.mlp = _QuantizedLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None,
                cos=None, sin=None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask, cos, sin)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _QuantizedLlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.q_proj = QuantizedLinear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = QuantizedLinear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = QuantizedLinear(config.hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask,
                cos, sin) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(attn_weights.dtype)
        if self.attention_dropout > 0 and self.training:
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _QuantizedLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = QuantizedLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = QuantizedLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = QuantizedLinear(config.intermediate_size, config.hidden_size)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
