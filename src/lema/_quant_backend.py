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
        try:
            if not torch.cuda.is_available():
                raise ImportError(
                    "bitsandbytes backend requires CUDA but torch.cuda.is_available() is False. "
                    "Use quant_backend='quanto' or 'torchao' for CPU."
                )
        except ImportError:
            raise
        except Exception:
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
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    if name == "quanto":
        try:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    if name == "bitsandbytes":
        try:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
        except Exception:
            from ._quant import quantize_tensor as _custom_q

            return _custom_q(t, bits, group_size=group_size)
    from ._quant import quantize_tensor as _custom_q

    return _custom_q(t, bits, group_size=group_size)
