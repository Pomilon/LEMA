from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class LemaModelAdapter(ABC):
    MODEL_TYPE: str = ""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def get_layer_metadata(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def construct_layer_module(
        self, layer_id: int, weights: dict[str, torch.Tensor] | torch.Tensor | None,
        lora_manager: Any = None
    ) -> nn.Module: ...

    @abstractmethod
    def forward_layer(self, layer_module: nn.Module, inputs: Any, **kwargs) -> Any: ...

    @abstractmethod
    def get_param_names_for_layer(self, layer_id: int) -> list[str]: ...

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...
