from __future__ import annotations

import torch

try:
    from lema._csrc import _w8a8_cpp as _cpp
    _HAS_NATIVE = hasattr(_cpp, "int8_gemm_avx2")
except Exception:
    _cpp = None
    _HAS_NATIVE = False

HAS_NATIVE = _HAS_NATIVE


def _native_int8_gemm_cpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.contiguous().to(torch.int8)
    b = b.contiguous().to(torch.int8)
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"K mismatch {K} vs {Kb}"
    out = torch.empty(M, N, dtype=torch.int32)
    _cpp.int8_gemm_avx2(a, b, M, K, N, out)
    return out


def _native_int8_gemm_cuda(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        from lema._csrc import _lema_cpp
    except Exception as e:
        raise RuntimeError(
            "CUDA int8 GEMM not available (CUDA extension not compiled)"
        ) from e
    a = a.contiguous().to(torch.int8)
    b = b.contiguous().to(torch.int8)
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"K mismatch {K} vs {Kb}"
    out = torch.empty(M, N, dtype=torch.int32, device=a.device)
    _lema_cpp.int8_gemm_cuda(a, b, M, K, N, out)
    return out


def _native_int8_gemm_fallback(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.float()).to(torch.int32)


def native_int8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not HAS_NATIVE:
        raise RuntimeError("native int8 GEMM not available (extension not compiled)")
    if a.is_cuda:
        return _native_int8_gemm_cuda(a, b)
    return _native_int8_gemm_cpu(a, b)


def _native_quantize_act_cuda(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
    try:
        from lema._csrc import _lema_cpp
    except Exception as e:
        raise RuntimeError(
            "CUDA quantize not available (CUDA extension not compiled)"
        ) from e
    _lema_cpp.quantize_act_cuda(x, x.numel(), q, scale)


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float().contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    if x.is_cuda:
        scale = torch.zeros(1, dtype=torch.float32, device=x.device)
        _native_quantize_act_cuda(x, q, scale)
        return q, scale[0]
    scale = torch.zeros(1, dtype=torch.float32)
    if HAS_NATIVE:
        _cpp.quantize_act_avx2(x, x.numel(), q, scale)
        return q, scale[0]
    amax = x.abs().max()
    s = (amax / 127.0) if amax > 0 else torch.tensor(1.0)
    q = (x / s).round().clamp(-128, 127).to(torch.int8)
    return q, s


def apply_scale(acc: torch.Tensor, scale_w: torch.Tensor, scale_a: torch.Tensor) -> torch.Tensor:
    acc = acc.contiguous()
    M, N = acc.shape
    out = torch.empty_like(acc, dtype=torch.float32)
    if acc.is_cuda or not HAS_NATIVE:
        out.copy_(acc.float() * scale_w.float().view(1, -1) * float(scale_a))
        return out
    _cpp.dequant_scale_gemm_avx2(acc, M, N, scale_w.float().contiguous(), float(scale_a), out)
    return out
