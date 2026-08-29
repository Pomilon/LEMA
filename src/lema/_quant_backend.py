import importlib.util

import torch


_BACKENDS = ("custom", "torchao", "quanto", "bitsandbytes")


def is_backend_available(backend: str) -> bool:
    if backend == "custom":
        return True
    try:
        if backend == "torchao":
            return importlib.util.find_spec("torchao") is not None
        if backend == "quanto":
            if importlib.util.find_spec("quanto") is not None:
                return True
            return importlib.util.find_spec("optimum.quanto") is not None
        if backend == "bitsandbytes":
            return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        return False
    return False


def list_available_backends() -> list[str]:
    return [b for b in _BACKENDS if is_backend_available(b)]


def resolve_backend(requested: str | None) -> str:
    if requested is None or requested == "auto":
        for name in ("torchao", "quanto", "bitsandbytes"):
            if is_backend_available(name):
                if name == "bitsandbytes":
                    try:
                        if not torch.cuda.is_available():
                            continue
                    except Exception:
                        continue
                return name
        return "custom"
    if requested not in _BACKENDS:
        raise ValueError(f"Unknown quant backend '{requested}'. Choose from {_BACKENDS} or 'auto'")
    if not is_backend_available(requested):
        raise ImportError(
            f"Requested quant backend '{requested}' is not installed. "
            f"Install it with: pip install lema[{requested}]  "
            f"(or pip install lema[all-quant] for all backends)"
        )
    if requested == "bitsandbytes":
        pass
    return requested


def quantize_tensor_with_backend(
    t: torch.Tensor, bits: int, backend: str | None = "auto", group_size: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    name = resolve_backend(backend)
    if name == "custom":
        from ._quant import quantize_tensor as _custom_q

        return _custom_q(t, bits, group_size=group_size)
    if name == "torchao":
        try:
            import torch as _torch

            _t = t.detach().float()
            if bits == 8:
                from torchao.quantization.utils import choose_qparams_affine, quantize_affine
                from torchao.quantization.quant_primitives import MappingType

                if _t.ndim == 1:
                    block_size = (_t.shape[0],)
                elif _t.ndim == 2:
                    block_size = (1, _t.shape[1])
                else:
                    block_size = tuple([1] * (_t.ndim - 1) + [_t.shape[-1]])
                scale, zp = choose_qparams_affine(
                    _t, MappingType.SYMMETRIC, block_size, _torch.int8, eps=1e-6
                )
                q = quantize_affine(_t, block_size, scale, zp, _torch.int8)
                if q.ndim == 2 and scale.ndim == 1:
                    scale = scale.view(-1, 1)
                return q.to(_torch.int8), scale.to(_torch.float32)
            if bits == 4:
                from ._quant import quantize_tensor as _custom_q

                return _custom_q(t, bits, group_size=group_size)
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    if name == "quanto":
        try:
            import torch as _torch
            from optimum.quanto import qint8, qint4, absmax_scale, quantize_weight

            _t = t.detach().float()
            if bits == 8:
                scale = absmax_scale(_t, qint8, axis=0)
                q_tensor = quantize_weight(_t, qint8, axis=0, scale=scale)
                q_data = q_tensor._data if hasattr(q_tensor, "_data") else q_tensor._qdata
                return q_data.to(_torch.int8), scale.to(_torch.float32)
            if bits == 4:
                scale = absmax_scale(_t, qint4, axis=0)
                q_tensor = quantize_weight(_t, qint4, axis=0, scale=scale)
                q_data = q_tensor._data if hasattr(q_tensor, "_data") else q_tensor._qdata
                return q_data.to(_torch.int8), scale.to(_torch.float32)
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    if name == "bitsandbytes":
        try:
            import torch as _torch
            import bitsandbytes as bnb

            _t = t.detach().float()
            if bits == 8:
                result = bnb.functional.int8_vectorwise_quant(_t)
                if isinstance(result, (tuple, list)):
                    q = result[0]
                    scale = result[1] if len(result) > 1 else None
                    if scale is None:
                        from ._quant import quantize_tensor as _custom_q

                        return _custom_q(t, bits, group_size=group_size)
                    if scale.ndim == 1:
                        scale = scale.view(-1, 1)
                    scale = (scale.float() / 127.0).to(_torch.float32)
                    return q.to(_torch.int8), scale
                return result.to(_torch.int8), _torch.ones((result.shape[0], 1), dtype=_torch.float32)
            if bits == 4:
                q4, state = bnb.functional.quantize_4bit(_t, blocksize=group_size or 64, quant_type="nf4")
                scale = state.absmax if hasattr(state, "absmax") else _torch.ones((1,), dtype=_torch.float32)
                return q4, scale.to(_torch.float32)
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    from ._quant import quantize_tensor as _custom_q

    return _custom_q(t, bits, group_size=group_size)


def dequantize_with_backend(
    q: torch.Tensor, scale: torch.Tensor, bits: int = 8, backend: str | None = "auto"
) -> torch.Tensor:
    from ._quant import dequantize as _custom_dq

    return _custom_dq(q, scale, bits=bits)
