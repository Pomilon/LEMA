from __future__ import annotations

import torch
import torch.nn as nn

from . import _w8a8
from ._quant import quantize_tensor


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_int8 = nn.Parameter(
            torch.zeros(in_features, out_features, dtype=torch.int8), requires_grad=False
        )
        self.register_buffer("scale_w", torch.zeros(out_features, dtype=torch.float32))
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)
            if bias else None
        )

    def set_int8_weight(self, w: torch.Tensor) -> None:
        q, s = quantize_tensor(w, 8)
        self.weight_int8.data.copy_(q.t().contiguous())
        self.scale_w.data.copy_(s.view(-1))

    def set_weight_from_int8(self, q: torch.Tensor, scale: torch.Tensor) -> None:
        self.weight_int8.data.copy_(q.to(torch.int8).t().contiguous())
        self.scale_w.data.copy_(scale.float().to(self.scale_w.device).view(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weight_int8.device)
        if _w8a8.HAS_NATIVE and x.device.type in ("cpu", "cuda") and not torch.is_grad_enabled():
            q, scale_a = _w8a8.quantize_act(x)
            acc = _w8a8.native_int8_gemm(q.reshape(-1, q.shape[-1]), self.weight_int8)
            out = _w8a8.apply_scale(acc, self.scale_w, scale_a)
            out = out.reshape(*x.shape[:-1], self.out_features)
            if self.bias is not None:
                out = out + self.bias
            return out
        w = self.weight_int8.float() * self.scale_w.view(1, -1)
        out = x @ w
        if self.bias is not None:
            out = out + self.bias
        return out