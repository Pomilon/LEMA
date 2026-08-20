# tests/test_w8a8_kernel.py
import torch
import pytest
from lema import _w8a8


def test_native_int8_gemm_matches_fp32_reference():
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    a = (torch.randn(M, K) * 3).clamp(-127, 127).round().to(torch.int8)
    b = (torch.randn(K, N) * 3).clamp(-127, 127).round().to(torch.int8)
    ref = (a.float() @ b.float()).to(torch.int32)
    out = _w8a8.native_int8_gemm(a, b)
    assert out.dtype == torch.int32
    assert torch.equal(out, ref)


def test_native_int8_gemm_medium_shape():
    torch.manual_seed(1)
    M, K, N = 2048, 4096, 4096
    a = (torch.randn(M, K) * 2).round().to(torch.int8)
    b = (torch.randn(K, N) * 2).round().to(torch.int8)
    out = _w8a8.native_int8_gemm(a, b)
    ref = (a.float() @ b.float()).to(torch.int32)
    assert (out.float() - ref.float()).abs().max().item() == 0


def test_quantize_act_roundtrip():
    torch.manual_seed(2)
    x = torch.randn(32, 64) * 2.0
    q, s = _w8a8.quantize_act(x)
    assert q.dtype == torch.int8
    assert q.shape == x.shape
    assert s.ndim == 0 or s.numel() == 1
    recon = q.float() * s
    rel = (recon - x).abs().max() / x.abs().max()
    assert rel.item() < 1e-1


def test_apply_scale_matches_manual():
    torch.manual_seed(3)
    acc = torch.randint(-1000, 1000, (8, 16), dtype=torch.int32)
    scale_w = torch.rand(16) * 0.01
    scale_a = torch.tensor(0.02)
    out = _w8a8.apply_scale(acc, scale_w, scale_a)
    ref = acc.float() * scale_w * scale_a
    assert torch.allclose(out, ref, atol=1e-5)


def test_fallback_when_ext_missing(monkeypatch):
    monkeypatch.setattr(_w8a8, "HAS_NATIVE", False)
    with pytest.raises(RuntimeError):
        _w8a8.native_int8_gemm(torch.zeros(4, 4, dtype=torch.int8),
                               torch.zeros(4, 4, dtype=torch.int8))
