import importlib.util

import torch

from ._utils._logger import logger


_BACKENDS = ("custom", "torchao", "quanto", "bitsandbytes")

# Engine contract for streamed weights (dequantize_with_backend is custom-format):
#   bits=8 -> (int8 q, per-row fp32 scale)
#   bits=4 -> custom pack_int4 uint8 (half numel) + scale
# Backends that cannot honor this contract must fall back to custom LOUDLY.
_WARNED: set[tuple[str, int, str]] = set()


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
    return requested


def _warn_once(key: tuple[str, int, str], msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(f"LEMA: {msg}")


def _fallback_custom(
    t: torch.Tensor, bits: int, group_size: int, backend: str, reason: str
) -> tuple[torch.Tensor, torch.Tensor]:
    _warn_once(
        (backend, bits, "fallback"),
        f"quant backend '{backend}' cannot serve bits={bits} ({reason}); "
        f"falling back to the custom quantizer.",
    )
    from ._quant import quantize_tensor as _custom_q

    return _custom_q(t, bits, group_size=group_size)


def quantize_tensor_with_backend(
    t: torch.Tensor, bits: int, backend: str | None = "auto", group_size: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    name = resolve_backend(backend)
    if name == "custom":
        from ._quant import quantize_tensor as _custom_q

        return _custom_q(t, bits, group_size=group_size)
    if name == "torchao":
        if bits != 8:
            return _fallback_custom(t, bits, group_size, name, "only 8-bit is engine-compatible")
        try:
            import torch as _torch

            _t = t.detach().float()
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
        except Exception as e:
            return _fallback_custom(t, bits, group_size, name, f"{type(e).__name__}: {e}")
    if name == "quanto":
        if bits != 8:
            return _fallback_custom(t, bits, group_size, name, "only 8-bit is engine-compatible")
        try:
            import torch as _torch
            from optimum.quanto import qint8, absmax_scale, quantize_weight

            _t = t.detach().float()
            scale = absmax_scale(_t, qint8, axis=0)
            q_tensor = quantize_weight(_t, qint8, axis=0, scale=scale)
            q_data = q_tensor._data if hasattr(q_tensor, "_data") else q_tensor._qdata
            return q_data.to(_torch.int8), scale.to(_torch.float32)
        except Exception as e:
            return _fallback_custom(t, bits, group_size, name, f"{type(e).__name__}: {e}")
    if name == "bitsandbytes":
        if bits != 8:
            return _fallback_custom(
                t, bits, group_size, name,
                "NF4 blockwise format is incompatible with the engine dequantizer",
            )
        try:
            import torch as _torch
            import bitsandbytes as bnb

            _t = t.detach().float()
            result = bnb.functional.int8_vectorwise_quant(_t)
            if isinstance(result, (tuple, list)):
                q = result[0]
                scale = result[1] if len(result) > 1 else None
                if scale is None:
                    return _fallback_custom(t, bits, group_size, name, "no scale returned")
                if scale.ndim == 1:
                    scale = scale.view(-1, 1)
                scale = (scale.float() / 127.0).to(_torch.float32)
                return q.to(_torch.int8), scale
            return result.to(_torch.int8), _torch.ones(
                (result.shape[0], 1), dtype=_torch.float32
            )
        except Exception as e:
            return _fallback_custom(t, bits, group_size, name, f"{type(e).__name__}: {e}")
    from ._quant import quantize_tensor as _custom_q

    return _custom_q(t, bits, group_size=group_size)


def dequantize_with_backend(
    q: torch.Tensor, scale: torch.Tensor, bits: int = 8, backend: str | None = "auto"
) -> torch.Tensor:
    # All backends emit the custom wire format (enforced in quantize_tensor_with_backend),
    # so dequantization is always the custom path.
    from ._quant import dequantize as _custom_dq

    return _custom_dq(q, scale, bits=bits)
