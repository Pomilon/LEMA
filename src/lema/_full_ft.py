from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch

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
        self.module_name_to_key: dict[int, dict[str, tuple[int, str]]] = {}
        self.true_weights: dict[tuple[int, str], torch.Tensor] = {}
        self.original: dict[tuple[int, str], torch.Tensor] = {}
        self.opt_states: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self.accumulators: dict[tuple[int, str], torch.Tensor] = {}
        self.layer_steps: dict[int, int] = {}
        self._init_weights()
        self._init_module_name_map()
        self.accumulator_backend = self._choose_accum_backend()
        self._memmaps: dict[tuple[int, str], np.ndarray] = {}
        self._memmap_files: dict[int, object] = {}
        if self.accumulator_backend == "disk":
            self._init_disk_accumulators()
        else:
            for acc in self.accumulators.values():
                acc.zero_()

    def _choose_accum_backend(self) -> str:
        requested = self.config.grad_accum_backend
        if requested == "disk":
            return "disk"
        if requested == "ram":
            return "ram"
        # auto: estimate fp32 accumulator bytes vs half the RAM budget
        bytes_needed = self.total_selected_params() * 4
        ram_budget = self.config.max_ram_gb
        if ram_budget <= 0:
            import psutil
            ram_budget = psutil.virtual_memory().total / (1024**3) * 0.7
        if bytes_needed > ram_budget * 0.5 * 1e9:
            logger.info(f"LEMA: Accumulators ({bytes_needed/1e9:.1f} GB) exceed RAM budget — using disk backend")
            return "disk"
        return "ram"

    def _init_disk_accumulators(self) -> None:
        import os
        dirpath = os.path.join(self.config.output_dir, "grad_accum")
        os.makedirs(dirpath, exist_ok=True)
        self._memmap_files = {}
        self._memmaps = {}
        for layer_id, keys in self.selected_layer_keys.items():
            total = sum(self.get_accumulator(k).numel() for k in keys)
            path = os.path.join(dirpath, f"grad_acc_{layer_id}.bin")
            is_new = not os.path.exists(path)
            f = open(path, "a+b")
            if is_new:
                f.truncate(total * 4)
            self._memmap_files[layer_id] = f
            arr = np.memmap(path, dtype="float32", mode="r+", shape=(total,))
            if is_new:
                arr.fill(0)
            offset = 0
            for key in keys:
                n = self.get_accumulator(key).numel()
                view = torch.from_numpy(arr[offset:offset + n]).view(self.get_accumulator(key).shape)
                self.accumulators[key] = view
                self._memmaps[key] = arr
                offset += n

    def close(self) -> None:
        for f in self._memmap_files.values():
            try:
                f.flush()
            except Exception:
                pass
        for layer_id, f in self._memmap_files.items():
            f.close()
        self._memmap_files = {}
        self._memmaps = {}

    def _init_weights(self) -> None:
        dtype = self.config.dtype if isinstance(self.config.dtype, torch.dtype) else getattr(torch, self.config.dtype, torch.float32)
        for layer_id, keys in self.selected_layer_keys.items():
            for key in keys:
                _, name = key
                w = self.gbi.load_tensors([name], device="cpu")[name]
                w = w.to(dtype).contiguous()
                self.true_weights[key] = w
                self.original[key] = w.clone()
                self.opt_states[key] = {
                    "exp_avg": torch.zeros_like(w, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(w, dtype=torch.float32),
                }
                self.accumulators[key] = torch.zeros_like(w, dtype=torch.float32)

    def _init_module_name_map(self) -> None:
        for layer_id, keys in self.selected_layer_keys.items():
            self.module_name_to_key[layer_id] = {}
            for key in keys:
                _, name = key
                module_name = self.adapter.get_module_param_name(layer_id, name)
                self.module_name_to_key[layer_id][module_name] = key

    def get_opt_state(self, key: tuple[int, str]) -> dict[str, torch.Tensor]:
        return self.opt_states[key]

    def get_accumulator(self, key: tuple[int, str]) -> torch.Tensor:
        return self.accumulators[key]

    def apply_to_module(self, layer_id: int, module) -> None:
        name_to_key = self.module_name_to_key.get(layer_id, {})
        for name, param in module.named_parameters():
            key = name_to_key.get(name)
            if key is not None:
                param.requires_grad_(True)
                param.data.copy_(self.true_weights[key], non_blocking=True)
            else:
                param.requires_grad_(False)

    def accumulate_grads(self, layer_id: int, module) -> None:
        name_to_key = self.module_name_to_key.get(layer_id, {})
        for name, param in module.named_parameters():
            key = name_to_key.get(name)
            if key is not None and param.grad is not None:
                self.get_accumulator(key).add_(param.grad.float().to(self.accumulators[key].device))
                param.grad = None

    def clip_grad_norm_(self, layer_id: int, max_norm: float = 1.0) -> float:
        keys = self.selected_layer_keys.get(layer_id, [])
        total = sum(self.get_accumulator(k).float().pow(2).sum().item() for k in keys)
        norm = math.sqrt(total)
        if norm > max_norm and norm > 0:
            coeff = max_norm / norm
            for k in keys:
                self.get_accumulator(k).mul_(coeff)
        return norm

    def step_layer(self, layer_id: int) -> None:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        step = self.layer_steps.get(layer_id, 0) + 1
        self.layer_steps[layer_id] = step
        b1 = 1 - beta1 ** step
        b2 = 1 - beta2 ** step
        for name, key in self.module_name_to_key.get(layer_id, {}).items():
            w = self.true_weights[key]
            grad = self.get_accumulator(key)
            state = self.get_opt_state(key)
            w_dev = w.to(self.config.device).float()
            g_dev = grad.to(self.config.device)
            m = state["exp_avg"].to(self.config.device)
            v = state["exp_avg_sq"].to(self.config.device)
            if wd:
                w_dev.mul_(1 - lr * wd)
            m.mul_(beta1).add_(g_dev, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g_dev, g_dev, value=1 - beta2)
            denom = v.sqrt().div_(math.sqrt(b2)).add_(eps)
            w_dev.addcdiv_(m, denom, value=-(lr / b1))
            w.copy_(w_dev.to(w.dtype))
            state["exp_avg"].copy_(m)
            state["exp_avg_sq"].copy_(v)
            self.get_accumulator(key).zero_()

    def get_trainable_parameters(self) -> list[torch.Tensor]:
        return list(self.true_weights.values())

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
