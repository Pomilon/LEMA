from __future__ import annotations

import torch


def quantize_tensor(t: torch.Tensor, bits: int, group_size: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bits: {bits!r}; choose 4 or 8")
    t = t.float().contiguous()
    if t.ndim >= 2 and group_size > 0:
        orig = t.shape
        flat = t.view(t.shape[0], -1)
        n_groups = flat.shape[1] // group_size
        groups = flat.view(flat.shape[0], n_groups, group_size)
        scale = groups.abs().amax(dim=2, keepdim=True) / (2 ** (bits - 1) - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = (groups / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        q = q.view(orig)
        scale = scale.view(t.shape[0], n_groups)
    elif t.ndim == 2:
        scale = t.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = (t / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    else:
        scale = t.abs().amax() / (2 ** (bits - 1) - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = (t / scale).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
        scale = scale.reshape(1)
    if bits == 4:
        return pack_int4(q.to(torch.int32).view(-1)), scale
    return q.to(torch.int8), scale


def dequantize(q_int: torch.Tensor, scale: torch.Tensor, bits: int = 8) -> torch.Tensor:
    if bits == 4:
        q = unpack_int4(q_int).view(scale.shape[0], -1) if scale.ndim == 2 else unpack_int4(q_int)
    else:
        q = q_int.float()
    if scale.ndim == 2 and scale.shape[1] > 1:
        n = q.numel() // scale.numel()
        return (q.view(scale.shape[0], scale.shape[1], n) * scale.view(scale.shape[0], scale.shape[1], 1)).view(scale.shape[0], -1)
    if q.shape[0] != scale.numel():
        q = q.view(scale.shape[0], -1)
    if q.ndim == 1:
        return q * scale
    return q * scale.view(-1, 1)


def pack_int4(t: torch.Tensor) -> torch.Tensor:
    t = t.to(torch.int32).view(-1)
    n = t.numel()
    if n % 2:
        t = torch.cat([t, torch.zeros(1, dtype=t.dtype, device=t.device)])
    lo = (t[0::2] & 0x0F).to(torch.uint8)
    hi = ((t[1::2] & 0x0F) << 4).to(torch.uint8)
    return lo | hi


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.int32).view(-1)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi).to(torch.int8)
    out = torch.empty(p.numel() * 2, dtype=torch.int8, device=packed.device)
    out[0::2] = lo
    out[1::2] = hi
    return out
