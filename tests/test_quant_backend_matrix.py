"""Backend x bits matrix tests: every quant_backend must produce engine-compatible
output (or loudly fall back to custom), never silently corrupt."""
import logging

import pytest
import torch

import lema._quant_backend as _qb
from lema._quant_backend import dequantize_with_backend, quantize_tensor_with_backend


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(_qb, "_WARNED", set())

_ALL = ("custom", "torchao", "quanto", "bitsandbytes")
_HAS = {
    "custom": True,
    "torchao": pytest.importorskip("torchao", reason="torchao not installed") is not None,
    "quanto": pytest.importorskip("optimum.quanto", reason="quanto not installed") is not None,
    "bitsandbytes": pytest.importorskip("bitsandbytes", reason="bitsandbytes not installed")
    is not None,
}


def _matrix() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(64, 128, generator=g)


def test_all_backend_bits_combos_roundtrip():
    t = _matrix()
    for backend in _ALL:
        if not _HAS[backend]:
            continue
        for bits in (8, 4):
            q, scale = quantize_tensor_with_backend(t, bits, backend=backend)
            assert q.dtype in (torch.int8, torch.uint8), f"{backend}/b{bits}: dtype {q.dtype}"
            d = dequantize_with_backend(q, scale, bits=bits, backend=backend)
            assert d.shape == t.shape, f"{backend}/b{bits}: shape {d.shape}"
            rel = (d - t).abs().max() / t.abs().max()
            tol = 0.05 if bits == 8 else 0.12
            assert rel.item() < tol, f"{backend}/b{bits}: rel err {rel.item():.4f}"


@pytest.mark.parametrize("backend", ["bitsandbytes", "quanto"])
def test_incompatible_backend_bits4_warns_and_falls_back(backend, caplog):
    if not _HAS[backend]:
        pytest.skip(f"{backend} not installed")
    t = _matrix()
    with caplog.at_level(logging.WARNING, logger="lema"):
        q, scale = quantize_tensor_with_backend(t, 4, backend=backend)
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "custom" in msgs.lower(), f"{backend}/b4: no fallback warning in {msgs!r}"
    d = dequantize_with_backend(q, scale, bits=4, backend=backend)
    rel = (d - t).abs().max() / t.abs().max()
    assert rel.item() < 0.12, f"{backend}/b4 fallback inaccurate: {rel.item():.4f}"


def test_unexpected_exception_fallback_warns(caplog, monkeypatch):
    pytest.importorskip("torchao")

    def boom(*a, **k):
        raise RuntimeError("simulated torchao failure")

    monkeypatch.setattr(_qb, "_WARNED", set())
    import torchao.quantization.utils as tu

    def boom(*a, **k):
        raise RuntimeError("simulated torchao failure")

    monkeypatch.setattr(tu, "choose_qparams_affine", boom)
    t = _matrix()
    with caplog.at_level(logging.WARNING, logger="lema"):
        q, scale = quantize_tensor_with_backend(t, 8, backend="torchao")
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "torchao" in msgs.lower(), f"no warning on exception fallback: {msgs!r}"
    d = dequantize_with_backend(q, scale, bits=8, backend="torchao")
    rel = (d - t).abs().max() / t.abs().max()
    assert rel.item() < 0.05
