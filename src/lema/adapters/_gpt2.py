from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config
from typing import Any

from ._base import LemaModelAdapter
from .._tensorstore import chunked_attention


class GPT2Adapter(LemaModelAdapter):
    MODEL_TYPE = "gpt2"
    MAX_POOL_SIZE = 3

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.hf_config = GPT2Config(**config)
        if getattr(self.hf_config, "_attn_implementation", None) is None:
            self.hf_config._attn_implementation = config.get("attn_implementation", "eager")
        # Permanent module cache: created once, reused across all steps
        self.module_pool: list[nn.Module] = []
        self.param_mappings: dict[int, list[tuple]] = {}

    def get_layer_metadata(self) -> list[dict[str, Any]]:
        layers = []
        layers.append({'id': 0, 'name': 'embeddings', 'type': 'embedding'})
        for i in range(self.hf_config.n_layer):
            layers.append({'id': i + 1, 'name': f'h.{i}', 'type': 'block', 'block_index': i})
        layers.append({'id': self.hf_config.n_layer + 1, 'name': 'head', 'type': 'head'})
        return layers

    def get_param_names_for_layer(self, layer_id: int) -> list[str]:
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

    def construct_layer_module(self, layer_id: int, flat_buffer: torch.Tensor | None = None, lora_manager: Any = None, full_ft_manager: Any = None) -> nn.Module:
        device = flat_buffer.device if flat_buffer is not None else torch.device("cpu")

        module = None
        for i, m in enumerate(self.module_pool):
            if layer_id == 0 and isinstance(m, GPT2EmbeddingsLayer):
                module = self.module_pool.pop(i); break
            elif layer_id == self.hf_config.n_layer + 1 and isinstance(m, GPT2HeadLayer):
                module = self.module_pool.pop(i); break
            elif 1 <= layer_id <= self.hf_config.n_layer and isinstance(m, GPT2Block):
                module = self.module_pool.pop(i); break

        if module is None:
            dtype_str = self.config.get("dtype", "float32")
            target_dtype = getattr(torch, dtype_str) if dtype_str else torch.float32
            if layer_id == 0:
                module = GPT2EmbeddingsLayer(self.hf_config)
            elif layer_id == self.hf_config.n_layer + 1:
                module = GPT2HeadLayer(self.hf_config)
            else:
                module = GPT2Block(self.hf_config)
            module.to(device=device, dtype=target_dtype)

            if lora_manager and 1 <= layer_id <= self.hf_config.n_layer:
                lora_manager.update_lora_params(layer_id, module)

            self.param_mappings[id(module)] = self._create_mapping(layer_id, module)

        if flat_buffer is not None and next(module.parameters()).device != flat_buffer.device:
            module.to(flat_buffer.device)

        if flat_buffer is not None:
            mapping = self.param_mappings[id(module)]
            with torch.no_grad():
                for param, offset, numel, shape in mapping:
                    param.data.copy_(flat_buffer[offset:offset+numel].view(shape), non_blocking=True)

        if full_ft_manager is not None:
            full_ft_manager.apply_to_module(layer_id, module)

        return module

    def get_module_param_name(self, layer_id: int, full_param_name: str) -> str:
        if layer_id == 0:
            return "wte.weight" if "wte" in full_param_name else "wpe.weight"
        elif layer_id == self.hf_config.n_layer + 1:
            if "ln_f" in full_param_name:
                return "ln_f.weight" if "weight" in full_param_name else "ln_f.bias"
            return "head.weight"
        else:
            prefix = f"transformer.h.{layer_id - 1}."
            return full_param_name[len(prefix):]

    def _create_mapping(self, layer_id: int, module: nn.Module) -> list[tuple]:
        names = self.get_param_names_for_layer(layer_id)
        idx = layer_id - 1
        module_params = dict(module.named_parameters())
        mapping = []
        offset = 0
        for full_name in names:
            if layer_id == 0:
                clean_k = "wte.weight" if "wte" in full_name else "wpe.weight"
            elif layer_id == self.hf_config.n_layer + 1:
                if "ln_f" in full_name: clean_k = "ln_f.weight" if "weight" in full_name else "ln_f.bias"
                else: clean_k = "head.weight"
            else:
                prefix = f"transformer.h.{idx}."
                clean_k = full_name[len(prefix):]
                if clean_k not in module_params:
                    clean_k = clean_k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
            param = module_params[clean_k]
            numel = param.numel()
            mapping.append((param, offset, numel, param.shape))
            offset += numel
        return mapping

    def release_layer_module(self, module: nn.Module):
        self.param_mappings.pop(id(module), None)

    def chunked_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                              kv_store, layer_id: int, kv_chunk_size: int = 8192) -> torch.Tensor:
        """Forward a GPT2Block using chunked causal attention over a KV store.

        Reuses the module's own projections (ln_1, c_attn, c_proj, ln_2, mlp) so
        weights are shared with the resident module; only the attention softmax
        is computed chunk-by-chunk over stashed KV. Produces output that matches
        the module's full forward within fp32 tolerance (mathematically identical).
        """
        block = layer_module
        head_dim = block.attn.head_dim
        n_head = block.attn.num_heads

        residual = hidden_states
        h = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(h)
        split = block.attn.split_size
        q, k, v = qkv.split(split, dim=-1)
        batch, seq, _ = q.shape

        q = q.view(batch, seq, n_head, head_dim).transpose(1, 2)
        k = k.view(batch, seq, n_head, head_dim).transpose(1, 2)
        v = v.view(batch, seq, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim ** 0.5) if block.attn.scale_attn_weights else 1.0

        # stash KV chunk-by-chunk
        chunks_in_seq = (seq + kv_chunk_size - 1) // kv_chunk_size
        attn_parts = []
        for c in range(chunks_in_seq):
            s = c * kv_chunk_size
            e = min(s + kv_chunk_size, seq)
            kv_store.stash(layer_id, c, k[:, :, s:e].contiguous(), v[:, :, s:e].contiguous())

        dropout = block.attn.attn_dropout if block.attn.attn_dropout.p > 0 else None

        # query chunk by chunk, attend over all prior + current chunks (causal)
        for c in range(chunks_in_seq):
            s = c * kv_chunk_size
            e = min(s + kv_chunk_size, seq)
            q_chunk = q[:, :, s:e]
            kv_chunks = [kv_store.load(layer_id, cc) for cc in range(c + 1)]
            attn_part = chunked_attention(q_chunk, kv_chunks, scale,
                                          query_start=s, kv_chunk_size=kv_chunk_size,
                                          causal=True, dropout=dropout)
            attn_parts.append(attn_part)
        attn_out = torch.cat(attn_parts, dim=2).transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_out = block.attn.c_proj(attn_out)
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = block.ln_2(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def decode_forward_layer(self, layer_module: nn.Module, hidden_states: torch.Tensor,
                             kv_store, layer_id: int, kv_chunk_size: int = 8192,
                             is_new_token: bool = True) -> torch.Tensor:
        """Forward a single (new) token's hidden state through a GPT2Block using
        the cached KV store. Attends over all cached chunks (causal trivially —
        all cached keys precede the new token), then appends the new K/V."""
        block = layer_module
        head_dim = block.attn.head_dim
        n_head = block.attn.num_heads

        residual = hidden_states
        h = block.ln_1(hidden_states)
        qkv = block.attn.c_attn(h)
        split = block.attn.split_size
        q, k, v = qkv.split(split, dim=-1)
        batch, tok, _ = q.shape

        q = q.view(batch, tok, n_head, head_dim).transpose(1, 2)
        k = k.view(batch, tok, n_head, head_dim).transpose(1, 2)
        v = v.view(batch, tok, n_head, head_dim).transpose(1, 2)

        scale = 1.0 / (head_dim ** 0.5) if block.attn.scale_attn_weights else 1.0

        # append new K/V to the cache
        if is_new_token:
            kv_store.append(layer_id, k, v)

        # attend over all cached chunks
        num_c = kv_store.num_chunks(layer_id)
        kv_chunks = [kv_store.load(layer_id, c) for c in range(num_c)]
        dropout = block.attn.attn_dropout if block.attn.attn_dropout.p > 0 else None
        attn_out = chunked_attention(q, kv_chunks, scale, causal=False, dropout=dropout)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, tok, -1)
        attn_out = block.attn.c_proj(attn_out)
        hidden_states = attn_out + residual

        residual = hidden_states
        hidden_states = block.ln_2(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any:
        hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
        kv_store = kwargs.get("kv_store")
        layer_id = kwargs.get("layer_id")
        kv_chunk_size = kwargs.get("kv_chunk_size", 0)
        position_offset = kwargs.get("position_offset", 0)
        if isinstance(layer_module, GPT2EmbeddingsLayer):
            return layer_module(hidden_states, position_offset=position_offset)
        if isinstance(layer_module, GPT2Block) and kv_store is not None and kv_chunk_size > 0:
            if hidden_states.size(1) > kv_chunk_size:
                return self.chunked_forward_layer(layer_module, hidden_states, kv_store,
                                                  layer_id, kv_chunk_size)
            return layer_module(hidden_states)[0]
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
    def forward(self, input_ids, position_offset: int = 0):
        position_ids = torch.arange(position_offset, position_offset + input_ids.size(-1),
                                    dtype=torch.long, device=input_ids.device).unsqueeze(0)
        return self.wte(input_ids) + self.wpe(position_ids)


class GPT2HeadLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, x): return self.head(self.ln_f(x))
