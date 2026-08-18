import torch
import pytest
from lema._tensorstore import chunked_attention, KVChunkStore


def _make(batch=2, heads=4, seq=16, head_dim=8, dtype=torch.float32):
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, dtype=dtype)
    return q, k, v


def _split(k, v, chunk_size):
    return [(k[:, :, i:i+chunk_size], v[:, :, i:i+chunk_size])
            for i in range(0, k.size(2), chunk_size)]


def _reference(q, kv_chunks, scale):
    """Reference built the SAME way as chunked_attention (per-chunk matmul + concat)."""
    qf = q.float()
    scores = torch.cat([qf @ k.float().transpose(-2, -1) for k, _ in kv_chunks], dim=-1) * scale
    m = scores.max(dim=-1, keepdim=True).values
    p = (scores - m).exp()
    v_full = torch.cat([v for _, v in kv_chunks], dim=2).float()
    out = p @ v_full
    return (out / p.sum(dim=-1, keepdim=True)).to(q.dtype)


def test_chunked_attention_bit_exact_vs_same_way_reference():
    q, k, v = _make()
    scale = 1.0 / (q.size(-1) ** 0.5)
    for chunk in (4, 8, 16):
        chunks = _split(k, v, chunk)
        out = chunked_attention(q, chunks, scale)
        ref = _reference(q, chunks, scale)
        assert torch.equal(out, ref)


def test_chunked_attention_close_to_full():
    q, k, v = _make()
    scale = 1.0 / (q.size(-1) ** 0.5)
    chunks = _split(k, v, 4)
    out = chunked_attention(q, chunks, scale)
    # plain full attention: single matmul
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale
    m = scores.max(dim=-1, keepdim=True).values
    p = (scores - m).exp()
    ref = (p @ v.float()) / p.sum(dim=-1, keepdim=True)
    assert torch.allclose(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


def test_chunked_attention_fp16_close_to_reference():
    q, k, v = _make(dtype=torch.float16)
    scale = 1.0 / (q.size(-1) ** 0.5)
    chunks = _split(k, v, 4)
    out = chunked_attention(q, chunks, scale)
    ref = _reference(q, chunks, scale)
    assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=1e-3)


def test_chunked_attention_via_kv_store():
    q, k, v = _make()
    scale = 1.0 / (q.size(-1) ** 0.5)
    store = KVChunkStore(kv_chunk_size=4)
    for i in range(0, k.size(2), 4):
        store.stash(0, i // 4, k[:, :, i:i+4], v[:, :, i:i+4])
    chunks = [store.load(0, c) for c in range(store.num_chunks(0))]
    out = chunked_attention(q, chunks, scale)
    ref = _reference(q, chunks, scale)
    assert torch.equal(out, ref)
