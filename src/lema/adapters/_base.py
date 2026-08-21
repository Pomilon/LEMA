from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class LemaModelAdapter(ABC):
    MODEL_TYPE: str = ""
    supports_quantized: bool = False

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_layer_metadata(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def construct_layer_module(
        self, layer_id: int, weights: dict[str, torch.Tensor] | torch.Tensor | None,
        lora_manager: Any = None, full_ft_manager: Any = None
    ) -> nn.Module: ...

    @abstractmethod
    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any: ...

    @abstractmethod
    def get_param_names_for_layer(self, layer_id: int) -> list[str]: ...

    @abstractmethod
    def get_module_param_name(self, layer_id: int, full_param_name: str) -> str: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    def load_tensor(self, gbi: Any, name: str) -> torch.Tensor:
        """Load a single weight from disk. Adapters with fused module params
        (e.g. grouped-MoE) override this to assemble tensors from per-expert
        checkpoint keys."""
        return gbi.load_tensors([name], device="cpu")[name]

    def get_tensor_shape(self, gbi: Any, name: str) -> tuple | None:
        """Resolve a weight's shape. Default: query the GBI (no bytes loaded).
        Fused-param adapters override to report the fused shape."""
        return gbi.get_tensor_shape(name)

    def supports_quantized_layer(self, layer_id: int) -> bool:
        """Whether W8A8 compute applies to this layer (vs. dequantized weights)."""
        return False

    def _set_generation_mode(self) -> None:
        """Put pooled layer modules into eval mode (disables dropout) for
        deterministic generation. Adapters that create fresh modules per call
        also check `self._generation_mode` in construct_layer_module."""
        self._generation_mode = True
        pool = getattr(self, "module_pool", None)
        if pool:
            for m in pool:
                m.eval()

    def _is_generation_mode(self) -> bool:
        return bool(getattr(self, "_generation_mode", False))
