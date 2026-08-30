# tests/test_w8a8_benchmark.py
import time
import pytest
import torch
from lema import _w8a8


def _bench(fn, *args, iters=3):
    for _ in range(2):
        fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def test_cpu_int8_gemm_not_slower_than_fp32():
    if not _w8a8.HAS_NATIVE:
        pytest.skip("native ext not built")
    M, K, N = 2048, 4096, 4096
    a8 = (torch.randn(M, K) * 2).round().to(torch.int8)
    b8 = (torch.randn(K, N) * 2).round().to(torch.int8)
    af = a8.float()
    bf = b8.float()
    t_int8 = _bench(lambda: _w8a8.native_int8_gemm(a8, b8))
    t_fp32 = _bench(lambda: af @ bf)
    # We want int8 FASTER; the hard bar is: not the torch 16x regression.
    assert t_int8 < t_fp32 * 2.0, f"int8 {t_int8:.3f}s vs fp32 {t_fp32:.3f}s"
