import torch
import pytest
from lema._quant import quantize_tensor, dequantize, pack_int4, unpack_int4


def test_int8_roundtrip_error_bounded():
    torch.manual_seed(0)
    t = torch.randn(16, 32) * 3.0
    q, s = quantize_tensor(t, 8)
    assert q.dtype == torch.int8
    assert q.shape == t.shape
    assert s.shape == (16, 1)
    out = dequantize(q, s)
    rel = (out - t).abs().max() / t.abs().max()
    assert rel.item() < 1e-2


def test_int8_scale_math_exact():
    t = torch.tensor([[3.0, -3.0, 1.0, 0.5]], dtype=torch.float32)
    q, s = quantize_tensor(t, 8)
    assert s[0, 0].item() == pytest.approx(3.0 / 127.0)
    assert torch.equal(q[0], torch.tensor([127, -127, 42, 21], dtype=torch.int8))


def test_int4_pack_unpack_roundtrip():
    torch.manual_seed(1)
    t = torch.randint(-7, 8, (10,), dtype=torch.int32)
    packed = pack_int4(t)
    assert packed.dtype == torch.uint8
    assert packed.numel() == (t.numel() + 1) // 2
    assert torch.equal(unpack_int4(packed)[: t.numel()], t)


def test_int4_quantize_roundtrip_error_bounded():
    torch.manual_seed(2)
    t = torch.randn(8, 16) * 2.0
    q, s = quantize_tensor(t, 4)
    assert q.dtype == torch.uint8
    assert q.numel() * 2 == t.numel()
    out = dequantize(q, s, bits=4)
    rel = (out - t).abs().max() / t.abs().max()
    assert rel.item() < 1e-1


def test_group_size_scales():
    torch.manual_seed(3)
    t = torch.randn(4, 128)
    q, s = quantize_tensor(t, 8, group_size=32)
    assert s.shape == (4, 4)  # 128 / 32 groups per row
    out = dequantize(q, s)
    rel = (out - t).abs().max() / t.abs().max()
    assert rel.item() < 1e-2


def test_per_group_reconstruction_correct():
    torch.manual_seed(4)
    t = torch.randn(1, 64)
    q, s = quantize_tensor(t, 8, group_size=16)
    out = dequantize(q, s)
    # each group scaled by its own absmax → reconstruct each group exactly vs its scale
    g = out[:, :16]
    assert torch.allclose(g, (q[:, :16].float() * s[0, 0]), atol=1e-6)
