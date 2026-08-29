from __future__ import annotations

import torch
import torch.nn as nn

from . import _w8a8
from ._quant import quantize_tensor


class _W8A8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_int8, scale_w):
        ctx.save_for_backward(weight_int8, scale_w)
        ctx.x_shape = x.shape
        ctx.orig_dtype = x.dtype
        q, scale_a = _w8a8.quantize_act(x)
        ctx.scale_a = scale_a
        acc = _w8a8.native_int8_gemm(q.reshape(-1, q.shape[-1]), weight_int8)
        out = _w8a8.apply_scale(acc, scale_w, scale_a)
        out = out.reshape(*x.shape[:-1], scale_w.shape[0])
        if out.dtype != ctx.orig_dtype:
            out = out.to(ctx.orig_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        weight_int8, scale_w = ctx.saved_tensors
        w_fp = weight_int8.float() * scale_w.view(1, -1)
        if w_fp.dtype != grad_out.dtype:
            w_fp = w_fp.to(grad_out.dtype)
        grad_out_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x_2d = grad_out_2d @ w_fp.t()
        grad_x = grad_x_2d.reshape(ctx.x_shape)
        if grad_x.dtype != ctx.orig_dtype:
            grad_x = grad_x.to(ctx.orig_dtype)
        return grad_x, None, None


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
        orig_dtype = x.dtype
        x = x.to(self.weight_int8.device)
        if _w8a8.HAS_NATIVE and x.device.type in ("cpu", "cuda"):
            if torch.is_grad_enabled() and x.requires_grad:
                out = _W8A8LinearFn.apply(x, self.weight_int8, self.scale_w)
                if self.bias is not None:
                    out = out + self.bias
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)
                return out
            if torch.is_grad_enabled() and not x.requires_grad:
                q, scale_a = _w8a8.quantize_act(x)
                acc = _w8a8.native_int8_gemm(q.reshape(-1, q.shape[-1]), self.weight_int8)
                out = _w8a8.apply_scale(acc, self.scale_w, scale_a)
                out = out.reshape(*x.shape[:-1], self.out_features)
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)
                if self.bias is not None:
                    out = out + self.bias
                return out
            if not torch.is_grad_enabled():
                q, scale_a = _w8a8.quantize_act(x)
                acc = _w8a8.native_int8_gemm(q.reshape(-1, q.shape[-1]), self.weight_int8)
                out = _w8a8.apply_scale(acc, self.scale_w, scale_a)
                out = out.reshape(*x.shape[:-1], self.out_features)
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)
                if self.bias is not None:
                    out = out + self.bias
                return out
        w = self.weight_int8.float() * self.scale_w.view(1, -1)
        out = x @ w
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        if self.bias is not None:
            out = out + self.bias
        return out