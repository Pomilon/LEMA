import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm, MistralConfig, MistralRotaryEmbedding
from typing import List, Dict, Any, Optional

from .base import LemaModelAdapter

class MistralAdapter(LemaModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hf_config = MistralConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")

        try:
            self.rotary_emb = MistralRotaryEmbedding(self.hf_config)
        except TypeError:
            self.rotary_emb = MistralRotaryEmbedding(
                self.hf_config.hidden_size // self.hf_config.num_attention_heads,
                max_position_embeddings=self.hf_config.max_position_embeddings
            )

        self.permanent_modules: Dict[int, nn.Module] = {}
        self.param_mappings: Dict[int, List[tuple]] = {}

    def get_layer_metadata(self) -> List[Dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.num_hidden_layers):
            layers.append({'id': i + 1, 'name': f'layers.{i}', 'type': 'block', 'block_index': i})
        layers.append({'id': self.hf_config.num_hidden_layers + 1, 'name': 'head', 'type': 'head'})
        return layers

    def get_param_names_for_layer(self, layer_id: int) -> List[str]:
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

    def construct_layer_module(self, layer_id: int, flat_buffer: Optional[torch.Tensor] = None, lora_manager: Optional[Any] = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")

        if layer_id not in self.permanent_modules:
            dtype_str = self.config.get("dtype", "float32")
            target_dtype = getattr(torch, dtype_str) if dtype_str else torch.float32
            if layer_id == 0:
                module = MistralEmbeddingsLayer(self.hf_config, None)
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                module = MistralHeadLayer(self.hf_config, None)
            else:
                module = MistralDecoderLayer(self.hf_config, layer_idx=0)
            module.to(device=device, dtype=target_dtype)

            if lora_manager and 1 <= layer_id <= self.hf_config.num_hidden_layers:
                lora_manager.update_lora_params(layer_id, module)

            self.param_mappings[id(module)] = self._create_mapping(layer_id, module)
            self.permanent_modules[layer_id] = module

        module = self.permanent_modules[layer_id]

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
        pass

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs

        if isinstance(layer_module, MistralDecoderLayer):
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
            hidden_states = layer_module.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states

        return layer_module(hidden_states)

    @property
    def hidden_size(self) -> int:
        return self.hf_config.hidden_size

class MistralEmbeddingsLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
    def forward(self, x): return self.embed_tokens(x)

class MistralHeadLayer(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    def forward(self, x): return self.lm_head(self.norm(x))
