from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer, MixtralRMSNorm, MixtralConfig, MixtralRotaryEmbedding,
    apply_rotary_pos_emb, repeat_kv,
)
from typing import Any

from ._base import LemaModelAdapter
from .._quantized_linear import QuantizedLinear
import torch.nn.functional as F
from transformers.activations import ACT2FN


class MixtralAdapter(LemaModelAdapter):
    MODEL_TYPE = "mixtral"
    MAX_POOL_SIZE = 3
    supports_quantized = True

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.hf_config = MixtralConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")
        if getattr(self.hf_config, "_experts_implementation", None) is None:
            self.hf_config._experts_implementation = "grouped_mm"

        try:
            self.rotary_emb = MixtralRotaryEmbedding(self.hf_config)
        except TypeError:
            self.rotary_emb = MixtralRotaryEmbedding(
                self.hf_config.hidden_size // self.hf_config.num_attention_heads,
                max_position_embeddings=self.hf_config.max_position_embeddings
            )

        self.module_pool: list[nn.Module] = []
        self.param_mappings: dict[int, list[tuple]] = {}

        # Detect Mixtral parameter naming (transformers v4 vs v5 compatibility)
        dummy = MixtralDecoderLayer(self.hf_config, layer_idx=0)
        self._moe_prefix = "mlp" if hasattr(dummy, "mlp") else "block_sparse_moe"
        self._expert_params = [k for k in dict(dummy.named_parameters()).keys()
                               if f"{self._moe_prefix}." in k]
        del dummy

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
            names = [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight", f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight", f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
            ]
            # Add moe params matching the actual module structure
            for mk in self._expert_params:
                names.append(f"{prefix}.{mk}")
            return names
        elif layer_id == self.hf_config.num_hidden_layers + 1:
            return ['model.norm.weight', 'lm_head.weight']
        return []

    def load_tensor(self, gbi: Any, name: str) -> torch.Tensor:
        if name in set(gbi.get_keys()):
            return super().load_tensor(gbi, name)
        if ".mlp.experts.gate_up_proj" in name:
            prefix = name.rsplit(".mlp.experts.gate_up_proj", 1)[0]
            prefix = prefix.replace(".mlp", ".block_sparse_moe")
            per_expert = []
            for e in range(self.hf_config.num_local_experts):
                w1 = gbi.load_tensors([f"{prefix}.experts.{e}.w1.weight"], device="cpu")
                w3 = gbi.load_tensors([f"{prefix}.experts.{e}.w3.weight"], device="cpu")
                per_expert.append(torch.cat([w1[f"{prefix}.experts.{e}.w1.weight"],
                                             w3[f"{prefix}.experts.{e}.w3.weight"]], dim=0))
            return torch.stack(per_expert, dim=0)
        if ".mlp.experts.down_proj" in name:
            prefix = name.rsplit(".mlp.experts.down_proj", 1)[0]
            prefix = prefix.replace(".mlp", ".block_sparse_moe")
            per_expert = []
            for e in range(self.hf_config.num_local_experts):
                w2 = gbi.load_tensors([f"{prefix}.experts.{e}.w2.weight"], device="cpu")
                per_expert.append(w2[f"{prefix}.experts.{e}.w2.weight"])
            return torch.stack(per_expert, dim=0)
        if ".mlp.gate.weight" in name:
            alt = name.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
            return gbi.load_tensors([alt], device="cpu")[alt]
        return super().load_tensor(gbi, name)

    def get_tensor_shape(self, gbi: Any, name: str) -> tuple | None:
        if name in set(gbi.get_keys()):
            return super().get_tensor_shape(gbi, name)
        if ".mlp.experts.gate_up_proj" in name:
            prefix = name.rsplit(".mlp.experts.gate_up_proj", 1)[0]
            prefix = prefix.replace(".mlp", ".block_sparse_moe")
            w1_shape = gbi.get_tensor_shape(f"{prefix}.experts.0.w1.weight")
            if w1_shape is not None:
                return (self.hf_config.num_local_experts, 2 * w1_shape[0], w1_shape[1])
        if ".mlp.experts.down_proj" in name:
            prefix = name.rsplit(".mlp.experts.down_proj", 1)[0]
            prefix = prefix.replace(".mlp", ".block_sparse_moe")
            w2_shape = gbi.get_tensor_shape(f"{prefix}.experts.0.w2.weight")
            if w2_shape is not None:
                return (self.hf_config.num_local_experts, w2_shape[0], w2_shape[1])
        if ".mlp.gate.weight" in name:
            return gbi.get_tensor_shape(name.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight"))
        return super().get_tensor_shape(gbi, name)

    def supports_quantized_layer(self, layer_id: int) -> bool:
        return 1 <= layer_id <= self.hf_config.num_hidden_layers

    def _construct_quantized_layer(self, layer_id: int, flat: torch.Tensor, full_ft_manager: Any = None) -> nn.Module:
        module = QuantizedMixtralLayer(self.hf_config)
        module.to(device=flat.device)
        transfer = getattr(self, "transfer_engine", None)
        if transfer is None:
            raise RuntimeError("Quantized layer construction requires the transfer engine scales")
        scale_all = transfer.get_layer_scale(layer_id, 0)
        if scale_all is None:
            raise RuntimeError(f"Missing quantization scale for layer {layer_id}")
        scale_all = scale_all.to(flat.device)
        prefix = f"model.layers.{layer_id - 1}."
        offset = 0
        s_off = 0
        with torch.no_grad():
            for full_name in self.get_param_names_for_layer(layer_id):
                clean = full_name[len(prefix):]
                if clean == "input_layernorm.weight":
                    n = module.input_layernorm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.input_layernorm.weight.shape)
                    module.input_layernorm.weight.data.copy_(
                        q_slice.float() * scale_all[s_off:s_off + 1].view(-1), non_blocking=True)
                    offset += n; s_off += 1
                    continue
                if clean == "post_attention_layernorm.weight":
                    n = module.post_attention_layernorm.weight.numel()
                    q_slice = flat[offset:offset + n].view(module.post_attention_layernorm.weight.shape)
                    module.post_attention_layernorm.weight.data.copy_(
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
                if clean == "self_attn.o_proj.weight":
                    shape = (module.self_attn.o_proj.out_features, module.self_attn.o_proj.in_features)
                    numel = shape[0] * shape[1]
                    module.self_attn.o_proj.set_weight_from_int8(flat[offset:offset+numel].view(shape), scale_all[s_off:s_off+shape[0]])
                    offset += numel; s_off += shape[0]
                    continue
                if clean == "mlp.gate.weight":
                    shape = (module.mlp.gate.weight.shape[0], module.mlp.gate.weight.shape[1])
                    numel = shape[0] * shape[1]
                    n_sc = shape[0]
                    module.mlp.gate.weight.data.copy_(
                        flat[offset:offset+numel].view(shape).float() * scale_all[s_off:s_off+n_sc].view(-1, 1), non_blocking=True)
                    offset += numel; s_off += n_sc
                    continue
                if clean == "mlp.experts.gate_up_proj":
                    shape = tuple(module.mlp.experts.gate_up_proj.shape)
                    numel = 1
                    for d in shape: numel *= d
                    q_slice = flat[offset:offset+numel].view(shape)
                    s = scale_all[s_off:s_off+1].view(-1).item() if scale_all.numel() > s_off else 1.0
                    module.mlp.experts.gate_up_proj.data.copy_(q_slice.float() * s, non_blocking=True)
                    offset += numel; s_off += 1
                    continue
                if clean == "mlp.experts.down_proj":
                    shape = tuple(module.mlp.experts.down_proj.shape)
                    numel = 1
                    for d in shape: numel *= d
                    q_slice = flat[offset:offset+numel].view(shape)
                    s = scale_all[s_off:s_off+1].view(-1).item() if scale_all.numel() > s_off else 1.0
                    module.mlp.experts.down_proj.data.copy_(q_slice.float() * s, non_blocking=True)
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
                and flat_buffer is not None and flat_buffer.dtype == torch.int8):
            if lora_manager is not None:
                raise RuntimeError(
                    "LoRA adapters are not supported on W8A8 int8 buffers; "
                    "use allow_quantized=False to receive dequantized weights"
                )
            return self._construct_quantized_layer(layer_id, flat_buffer, full_ft_manager)

        module = None
        for i, m in enumerate(self.module_pool):
            if layer_id == 0 and isinstance(m, MixtralEmbeddingsLayer):
                module = self.module_pool.pop(i); break
            elif layer_id == self.hf_config.num_hidden_layers + 1 and isinstance(m, MixtralHeadLayer):
                module = self.module_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.num_hidden_layers and isinstance(m, MixtralDecoderLayer):
                module = self.module_pool.pop(i); break

        if module is None:
            dtype_str = self.config.get("dtype", "float32")
            target_dtype = getattr(torch, dtype_str) if dtype_str else torch.float32
            if layer_id == 0:
                module = MixtralEmbeddingsLayer(self.hf_config, None)
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                module = MixtralHeadLayer(self.hf_config, None)
            else:
                module = MixtralDecoderLayer(self.hf_config, layer_idx=0)
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
            return "norm.weight" if "model.norm" in full_param_name else "lm_head.weight"
        else:
            prefix = f"model.layers.{layer_id - 1}."
            return full_param_name[len(prefix):]

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
        if isinstance(module, QuantizedMixtralLayer):
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

    def _attn_context(self, layer_module: nn.Module, hidden_states: torch.Tensor, kwargs: dict[str, Any]):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device
        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            attention_mask = kwargs.get("attention_mask")
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
        if isinstance(layer_module, QuantizedMixtralLayer):
            kv_store = kwargs.get("kv_store")
            layer_id = kwargs.get("layer_id")
            kv_chunk_size = kwargs.get("kv_chunk_size", 0)
            if kv_store is not None and kv_chunk_size > 0:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store, layer_id, kv_chunk_size)
            position_ids, attention_mask, cos, sin = self._attn_context(layer_module, hidden_states, kwargs)
            return layer_module(hidden_states, attention_mask=attention_mask, cos=cos, sin=sin)
        if isinstance(layer_module, MixtralDecoderLayer):
            kv_store = kwargs.get("kv_store")
            layer_id = kwargs.get("layer_id")
            kv_chunk_size = kwargs.get("kv_chunk_size", 0)
            if kv_store is not None and kv_chunk_size > 0:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store,
                                                  layer_id, kv_chunk_size)
            batch_size, seq_len = hidden_states.shape[:2]
            device = hidden_states.device

            # Cached per-step constants (same across all layers)
            if "position_ids" in kwargs:
                position_ids = kwargs["position_ids"]
                attention_mask = kwargs["attention_mask"]
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

            # Compute RoPE (once — same for all layers)
            if self._cache_rope is None:
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

            residual = hidden_states
            hidden_states = layer_module.input_layernorm(hidden_states)

            def attn_block(x, mask, pids, cos_sin):
                return layer_module.self_attn(
                    hidden_states=x,
                    attention_mask=mask,
                    position_ids=pids,
                    position_embeddings=cos_sin,
                    past_key_value=kwargs.get("past_key_value"),
                    output_attentions=kwargs.get("output_attentions", False),
                    use_cache=kwargs.get("use_cache", False),
                    cache_position=kwargs.get("cache_position")
                )[0]

            if torch.is_grad_enabled() and kwargs.get("gradient_checkpointing", False):
                attn_output = checkpoint(attn_block, hidden_states, attention_mask, position_ids, (cos, sin), use_reentrant=False)
            else:
                attn_output = attn_block(hidden_states, attention_mask, position_ids, (cos, sin))

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer_module.post_attention_layernorm(hidden_states)
            moe = getattr(layer_module, "mlp", None) or getattr(layer_module, "block_sparse_moe")
            out = moe(hidden_states)
            hidden_states = out[0] if isinstance(out, tuple) else out
            hidden_states = residual + hidden_states

            return hidden_states

        return layer_module(hidden_states)

    @property
    def hidden_size(self) -> int:
        return self.hf_config.hidden_size


class MixtralEmbeddingsLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    def forward(self, x): return self.embed_tokens(x)


class MixtralHeadLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, x): return self.lm_head(self.norm(x))


class QuantizedMixtralLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _QuantizedMixtralAttention(config)
        self.mlp = _QuantizedMixtralMoE(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, cos=None, sin=None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states, attention_mask, cos, sin)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _QuantizedMixtralAttention(nn.Module):
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

    def forward(self, hidden_states: torch.Tensor, attention_mask, cos, sin) -> torch.Tensor:
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


class _QuantizedMixtralMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * config.intermediate_size, config.hidden_size))
        self.experts.down_proj = nn.Parameter(torch.empty(self.num_experts, config.hidden_size, config.intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = F.linear(hidden_states_flat, self.gate.weight)
        router_logits = torch.softmax(router_logits.float(), dim=-1)
        top_k_weights, top_k_index = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        final_hidden = torch.zeros_like(hidden_states_flat)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            cur = hidden_states_flat[token_idx]
            gate, up = F.linear(cur, self.experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            cur = self.act_fn(gate) * up
            cur = F.linear(cur, self.experts.down_proj[expert_idx])
            cur = cur * top_k_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, cur.to(final_hidden.dtype))
        return final_hidden.view(batch_size, seq_len, hidden_dim)
