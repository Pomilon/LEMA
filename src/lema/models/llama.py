import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig, LlamaRotaryEmbedding
from typing import List, Dict, Any, Optional

from .base import LemaModelAdapter

class LlamaAdapter(LemaModelAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hf_config = LlamaConfig(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")
        self.rotary_emb = LlamaRotaryEmbedding(self.hf_config)
        self.layer_pool: List[nn.Module] = []
        self.param_mappings: Dict[int, List[tuple]] = {}
        self._max_pool_size = 8
        
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
        if flat_buffer is not None:
            device = flat_buffer.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        module = None
        for i, m in enumerate(self.layer_pool):
            if layer_id == 0 and isinstance(m, LlamaEmbeddingsLayer):
                module = self.layer_pool.pop(i); break
            elif layer_id == self.hf_config.num_hidden_layers + 1 and isinstance(m, LlamaHeadLayer):
                module = self.layer_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.num_hidden_layers and isinstance(m, LlamaDecoderLayer):
                module = self.layer_pool.pop(i); break
        
        if module is None:
            if layer_id == 0: module = LlamaEmbeddingsLayer(self.hf_config, None)
            elif layer_id == self.hf_config.num_hidden_layers + 1: module = LlamaHeadLayer(self.hf_config, None)
            else:
                module = LlamaDecoderLayer(self.hf_config, layer_idx=0)
            
            # Initialization only: move to target device
            module.to(device)

        if lora_manager and 1 <= layer_id <= self.hf_config.num_hidden_layers:
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
            
        if hasattr(module, "layer_idx") and 1 <= layer_id <= self.hf_config.num_hidden_layers:
            module.layer_idx = layer_id - 1
        return module

    def _create_mapping(self, layer_id: int, module: nn.Module) -> List[tuple]:
        names = self.get_param_names_for_layer(layer_id)
        idx = layer_id - 1
        module_params = dict(module.named_parameters())
        mapping = []
        for full_name in names:
            if layer_id == 0: clean_k = "embed_tokens.weight"
            elif layer_id == self.hf_config.num_hidden_layers + 1:
                clean_k = "norm.weight" if "model.norm" in full_name else "lm_head.weight"
            else:
                prefix = f"model.layers.{idx}."
                clean_k = full_name[len(prefix):]
                if clean_k not in module_params: clean_k = clean_k.replace(".weight", ".base_layer.weight")
            param = module_params[clean_k]
            mapping.append((param, param.numel(), param.shape))
        return mapping

    def release_layer_module(self, module: nn.Module):
        if len(self.layer_pool) < self._max_pool_size:
            self.layer_pool.append(module)
        else:
            del module
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        
        if isinstance(layer_module, LlamaDecoderLayer):
            batch_size, seq_len = hidden_states.shape[:2]
            device = hidden_states.device
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            
            # Compute RoPE (Should be [bs, seq, dim])
            attn = layer_module.self_attn
            try:
                if hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
                    try: cos, sin = attn.rotary_emb(hidden_states, position_ids)
                    except: cos, sin = attn.rotary_emb(position_ids)
                else:
                    cos, sin = self.rotary_emb(hidden_states, position_ids)
            except Exception:
                head_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads
                dummy = torch.zeros(batch_size, seq_len, head_dim, device=device)
                cos, sin = self.rotary_emb(dummy, position_ids)

            # Ensure 3D [bs, seq, dim] for transformers broadcasting
            if cos.ndim == 2:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            elif cos.ndim == 4:
                cos = cos.squeeze(1)
                sin = sin.squeeze(1)
            
            # If batch size mismatch (e.g. rotary_emb returned [1, seq, dim] but bs > 1)
            if cos.shape[0] != batch_size and cos.shape[0] == 1:
                cos = cos.expand(batch_size, -1, -1)
                sin = sin.expand(batch_size, -1, -1)

            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
                mask = torch.triu(mask, diagonal=1)
                attention_mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

            # 2. Manual Forward with Checkpointing
            residual = hidden_states
            hidden_states = layer_module.input_layernorm(hidden_states)
            
            # Wrap Self-Attention for memory efficiency
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
                from torch.utils.checkpoint import checkpoint
                attn_output = checkpoint(attn_block, hidden_states, attention_mask, position_ids, (cos, sin), use_reentrant=False)
            else:
                attn_output = attn_block(hidden_states, attention_mask, position_ids, (cos, sin))
            
            hidden_states = residual + attn_output
            
            # MLP block
            residual = hidden_states
            hidden_states = layer_module.post_attention_layernorm(hidden_states)
            hidden_states = layer_module.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            return hidden_states
                
        return layer_module(hidden_states)

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
