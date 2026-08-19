from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .._tensorstore import KVChunkStore, chunked_attention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, num_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, num_kv_heads, n_rep, slen, head_dim)
        .reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    )


def _qkv_proj(layer_module: nn.Module, hidden_states: torch.Tensor):
    """Project hidden states to Q/K/V (heads transposed), RoPE applied."""
    attn = layer_module.self_attn
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    return q, k, v


def compute_rope(adapter, attn, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    """Compute RoPE cos/sin at the given positions, matching the adapter's forward_layer."""
    try:
        if hasattr(attn, "rotary_emb") and attn.rotary_emb is not None:
            try:
                cos, sin = attn.rotary_emb(hidden_states, position_ids)
            except Exception:
                cos, sin = attn.rotary_emb(position_ids)
        else:
            cos, sin = adapter.rotary_emb(hidden_states, position_ids)
    except Exception:
        head_dim = adapter.hf_config.hidden_size // adapter.hf_config.num_attention_heads
        cos, sin = adapter.rotary_emb(
            torch.zeros(*hidden_states.shape[:-1], head_dim, device=hidden_states.device),
            position_ids,
        )
    if cos.ndim == 2:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    elif cos.ndim == 4:
        cos, sin = cos.squeeze(1), sin.squeeze(1)
    if cos.shape[0] != hidden_states.shape[0] and cos.shape[0] == 1:
        cos, sin = cos.expand(hidden_states.shape[0], -1, -1), sin.expand(hidden_states.shape[0], -1, -1)
    return cos, sin


def _rope_apply(q: torch.Tensor, k: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rotary_pos_emb(q, k, cos, sin)


def rope_chunked_forward_layer(layer_module: nn.Module, hidden_states: torch.Tensor,
                               kv_store: KVChunkStore, layer_id: int,
                               kv_chunk_size: int, cos, sin) -> torch.Tensor:
    """Chunked causal forward for RoPE-based decoder layers (llama/mistral/mixtral).

    Computes Q/K/V projections + RoPE for the whole sequence, stashes K/V chunks
    into the store (GQA-expanded so the cache is full-head), then attends query
    chunk-by-chunk over prior + current chunks with a causal mask. Output matches
    the module's full forward within fp32 tolerance.
    """
    block = layer_module
    attn = block.self_attn
    n_rep = attn.num_key_value_groups

    residual = hidden_states
    h = block.input_layernorm(hidden_states)
    q, k, v = _qkv_proj(block, h)
    q, k = _rope_apply(q, k, cos, sin)

    # GQA: expand K/V to full heads so the store holds full-head chunks
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    batch, _, seq, _ = q.shape
    scale = attn.scaling
    chunks_in_seq = (seq + kv_chunk_size - 1) // kv_chunk_size

    for c in range(chunks_in_seq):
        s = c * kv_chunk_size
        e = min(s + kv_chunk_size, seq)
        kv_store.stash(layer_id, c, k[:, :, s:e].contiguous(), v[:, :, s:e].contiguous())

    dropout = None
    if getattr(attn, "attention_dropout", 0) > 0 and block.training:
        dropout = nn.Dropout(attn.attention_dropout)

    attn_parts = []
    for c in range(chunks_in_seq):
        s = c * kv_chunk_size
        e = min(s + kv_chunk_size, seq)
        q_chunk = q[:, :, s:e]
        kv_chunks = [kv_store.load(layer_id, cc) for cc in range(c + 1)]
        attn_parts.append(chunked_attention(q_chunk, kv_chunks, scale,
                                            query_start=s, kv_chunk_size=kv_chunk_size,
                                            causal=True, dropout=dropout))
    attn_out = torch.cat(attn_parts, dim=2).transpose(1, 2).contiguous()
    attn_out = attn_out.view(batch, seq, -1)
    attn_out = attn.o_proj(attn_out)

    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = block.post_attention_layernorm(hidden_states)
    hidden_states = block.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def rope_decode_forward_layer(layer_module: nn.Module, hidden_states: torch.Tensor,
                              kv_store: KVChunkStore, layer_id: int,
                              kv_chunk_size: int, cos, sin,
                              is_new_token: bool = True) -> torch.Tensor:
    """Decode one new token against the cached KV for RoPE-based layers.

    Computes Q/K/V + RoPE for the new token, appends its K/V to the store
    (GQA-expanded), attends over all cached chunks (all keys precede the new
    token, so no mask), and applies the MLP. Supports both the chunked-decode
    pattern and simple per-token decode.
    """
    block = layer_module
    attn = block.self_attn
    n_rep = attn.num_key_value_groups

    residual = hidden_states
    h = block.input_layernorm(hidden_states)
    q, k, v = _qkv_proj(block, h)
    q, k = _rope_apply(q, k, cos, sin)

    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    batch, _, tok, _ = q.shape
    scale = attn.scaling

    if is_new_token:
        kv_store.append(layer_id, k, v)

    num_c = kv_store.num_chunks(layer_id)
    kv_chunks = [kv_store.load(layer_id, c) for c in range(num_c)]

    dropout = None
    if getattr(attn, "attention_dropout", 0) > 0 and block.training:
        dropout = nn.Dropout(attn.attention_dropout)

    attn_out = chunked_attention(q, kv_chunks, scale, causal=False, dropout=dropout)
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, tok, -1)
    attn_out = attn.o_proj(attn_out)

    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = block.post_attention_layernorm(hidden_states)
    hidden_states = block.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
