from __future__ import annotations

import math
from typing import Any

from ._config import LemaConfig, TrainingMode
from ._utils._logger import logger


class FullFTManager:
    """Resolves and holds the trainable-weight selection for selective full FT."""

    def __init__(self, gbi: Any, adapter: Any, config: LemaConfig):
        self.gbi = gbi
        self.adapter = adapter
        self.config = config
        if config.training_mode != TrainingMode.SELECTIVE_FULL:
            raise ValueError("FullFTManager requires training_mode='selective_full'")
        self.selected: dict[int, list[str]] = {}
        self.selected_layer_keys: dict[int, list[tuple[int, str]]] = {}
        self.resolve_selection()

    def _resolve_layers(self) -> list[int]:
        meta = self.adapter.get_layer_metadata()
        block_ids = [l["id"] for l in meta if l.get("type") == "block"]
        emb_id = next(l["id"] for l in meta if l.get("type") == "embedding")
        head_id = next(l["id"] for l in meta if l.get("type") == "head")
        specs = self.config.trainable_layers
        if not specs:
            return [emb_id] + block_ids + [head_id]
        ids: set[int] = set()
        for spec in specs:
            if spec == "emb":
                ids.add(emb_id)
            elif spec == "head":
                ids.add(head_id)
            elif spec.startswith("last:"):
                k = int(spec.split(":", 1)[1])
                ids.update(block_ids[-k:])
            elif spec.startswith("first:"):
                k = int(spec.split(":", 1)[1])
                ids.update(block_ids[:k])
            else:
                ids.add(int(spec))
        return sorted(ids)

    def _match_modules(self, param_name: str) -> bool:
        patterns = self.config.trainable_modules
        if not patterns:
            return True
        core = param_name
        for suffix in (".weight", ".bias"):
            if core.endswith(suffix):
                core = core[: -len(suffix)]
                break
        return any(p == core or core.endswith(p) for p in patterns)

    def resolve_selection(self) -> None:
        layer_ids = self._resolve_layers()
        for layer_id in layer_ids:
            names = [
                n for n in self.adapter.get_param_names_for_layer(layer_id)
                if self.gbi.get_tensor_shape(n) is not None and self._match_modules(n)
            ]
            if names:
                self.selected[layer_id] = names
                self.selected_layer_keys[layer_id] = [(layer_id, n) for n in names]
        if not self.selected:
            raise ValueError(
                "Selective full FT selection resolved to zero parameters. "
                f"trainable_modules={self.config.trainable_modules}, "
                f"trainable_layers={self.config.trainable_layers}"
            )
        logger.info(
            f"LEMA: Selective full FT selected {self.total_selected_params():,} params "
            f"across layers {sorted(self.selected.keys())}"
        )

    def total_selected_params(self) -> int:
        total = 0
        for layer_id, keys in self.selected_layer_keys.items():
            for _, name in keys:
                shape = self.gbi.get_tensor_shape(name)
                if shape is not None:
                    total += math.prod(shape)
        return total
