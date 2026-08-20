from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch

from ._config import LemaConfig, TrainingMode
from ._quant import quantize_tensor, dequantize
from ._tensorstore import Stream, StreamKind
from ._utils._logger import logger


class FullFTManager:
    """Resolves and holds the trainable-weight selection for selective full FT."""

    def __init__(self, gbi: Any, adapter: Any, config: LemaConfig, store: Any = None):
        self.gbi = gbi
        self.adapter = adapter
        self.config = config
        self.store = store
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
        self.accumulation_step = 0
        self._init_weights()
        self._init_module_name_map()
        self.accumulator_backend = self._choose_accum_backend()
        self._memmaps: dict[tuple[int, str], np.ndarray] = {}
        self._memmap_files: dict[int, object] = {}
        if self.accumulator_backend == "disk":
            self._init_disk_accumulators()
        else:
            for acc in self.accumulators.values():
                if isinstance(acc, tuple):
                    acc[0].zero_()
                else:
                    acc.zero_()
        if self.store is not None:
            self._register_streams()

    def _q(self, v):
        return v[0] if isinstance(v, tuple) else v

    def _s(self, v):
        return v[1] if isinstance(v, tuple) else None

    def _deq(self, x, shape, bits):
        if isinstance(x, tuple):
            return dequantize(x[0], x[1], bits=bits).reshape(shape)
        return x

    def _deq_acc(self, x, key):
        if isinstance(x, tuple):
            return self._deq(x, self.true_weights[key].shape, self.config.grad_acc_bits or 8)
        return x

    def _restore_state_val(self, s, name, shape):
        scale = s.get(f"{name}.scale")
        if scale is not None:
            return dequantize(s[name], scale, bits=8).reshape(shape)
        return s[name]

    def _register_streams(self) -> None:
        for key in self.true_weights:
            layer_id, name = key
            self.store.register(Stream(
                StreamKind.OPT_STATE, layer_id, f"{name}::exp_avg",
                tuple(self._q(self.opt_states[key]["exp_avg"]).shape),
                self._q(self.opt_states[key]["exp_avg"]).dtype,
                source=lambda k=key: self._q(self.opt_states[k]["exp_avg"]),
            ))
            self.store.register(Stream(
                StreamKind.OPT_STATE, layer_id, f"{name}::exp_avg_sq",
                tuple(self._q(self.opt_states[key]["exp_avg_sq"]).shape),
                self._q(self.opt_states[key]["exp_avg_sq"]).dtype,
                source=lambda k=key: self._q(self.opt_states[k]["exp_avg_sq"]),
            ))
            self.store.register(Stream(
                StreamKind.GRAD_ACC, layer_id, name,
                tuple(self._q(self.accumulators[key]).shape),
                self._q(self.accumulators[key]).dtype,
                source=lambda k=key: self._q(self.accumulators[k]),
            ))

    def _choose_accum_backend(self) -> str:
        requested = self.config.grad_accum_backend
        if requested == "disk":
            return "disk"
        if requested == "ram":
            return "ram"
        # auto: estimate fp32 accumulator bytes vs half the RAM budget
        bytes_needed = self.total_selected_params() * (self.config.grad_acc_bits or 32) // 8
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
        import json
        dirpath = os.path.join(self.config.output_dir, "grad_accum")
        os.makedirs(dirpath, exist_ok=True)
        self._memmap_files = {}
        self._memmaps = {}

        # Sidecar: records the selection signature so reopening with a different
        # selection does not silently reuse stale accumulator files.
        sidecar_path = os.path.join(dirpath, "selection.json")
        signature = {
            str(layer_id): sorted(names)
            for layer_id, names in self.selected.items()
        }
        signature["grad_acc_bits"] = self.config.grad_acc_bits
        fresh = False
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path) as f:
                    old = json.load(f)
                fresh = (old != signature)
            except Exception:
                fresh = True
        else:
            fresh = True
        if fresh:
            with open(sidecar_path, "w") as f:
                json.dump(signature, f, indent=2)

        acc_bits = self.config.grad_acc_bits if self.config.grad_acc_bits else None
        for layer_id, keys in self.selected_layer_keys.items():
            total = sum(self._q(self.get_accumulator(k)).numel() for k in keys)
            path = os.path.join(dirpath, f"grad_acc_{layer_id}.bin")
            is_new = (not os.path.exists(path)) or fresh
            f = open(path, "a+b")
            if is_new:
                f.truncate(total * (1 if acc_bits else 4))
            self._memmap_files[layer_id] = f
            arr = np.memmap(path, dtype="int8" if acc_bits else "float32", mode="r+", shape=(total,))
            if is_new:
                arr.fill(0)
                arr.flush()
            if acc_bits:
                total_scale = sum(self._s(self.get_accumulator(k)).numel() for k in keys)
                scale_path = os.path.join(dirpath, f"grad_acc_{layer_id}_scale.bin")
                sf = open(scale_path, "a+b")
                if is_new:
                    sf.truncate(total_scale * 4)
                self._memmap_files[f"{layer_id}.scale"] = sf
                sarr = np.memmap(scale_path, dtype="float32", mode="r+", shape=(total_scale,))
                if is_new:
                    sarr.fill(1.0)
                    sarr.flush()
            offset = 0
            s_offset = 0
            for key in keys:
                n = self._q(self.get_accumulator(key)).numel()
                view = torch.from_numpy(arr[offset:offset + n]).view(self._q(self.get_accumulator(key)).shape)
                if acc_bits:
                    s_n = self._s(self.get_accumulator(key)).numel()
                    s_view = torch.from_numpy(sarr[s_offset:s_offset + s_n]).view(self._s(self.get_accumulator(key)).shape)
                    self.accumulators[key] = (view, s_view)
                    s_offset += s_n
                else:
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
        opt_bits = self.config.opt_state_bits if self.config.opt_state_bits else None
        acc_bits = self.config.grad_acc_bits if self.config.grad_acc_bits else None
        for layer_id, keys in self.selected_layer_keys.items():
            for key in keys:
                _, name = key
                w = self.adapter.load_tensor(self.gbi, name)
                w = w.to(dtype).contiguous()
                self.true_weights[key] = w
                self.original[key] = w.clone()
                if opt_bits:
                    self.opt_states[key] = {
                        "exp_avg": quantize_tensor(torch.zeros_like(w, dtype=torch.float32), opt_bits),
                        "exp_avg_sq": quantize_tensor(torch.zeros_like(w, dtype=torch.float32), opt_bits),
                    }
                else:
                    self.opt_states[key] = {
                        "exp_avg": torch.zeros_like(w, dtype=torch.float32),
                        "exp_avg_sq": torch.zeros_like(w, dtype=torch.float32),
                    }
                if acc_bits:
                    self.accumulators[key] = quantize_tensor(torch.zeros_like(w, dtype=torch.float32), acc_bits)
                else:
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
        acc_bits = self.config.grad_acc_bits if self.config.grad_acc_bits else None
        name_to_key = self.module_name_to_key.get(layer_id, {})
        for name, param in module.named_parameters():
            key = name_to_key.get(name)
            if key is not None and param.grad is not None:
                acc = self.get_accumulator(key)
                if isinstance(acc, tuple):
                    g = param.grad.float().to(acc[0].device)
                    deq = self._deq_acc(acc, key).to(acc[0].device)
                    q, s = quantize_tensor(deq + g, acc_bits)
                    acc[0].copy_(q)
                    acc[1].copy_(s)
                else:
                    self.get_accumulator(key).add_(param.grad.float().to(self.accumulators[key].device))
                param.grad = None

    def clip_grad_norm_(self, layer_id: int, max_norm: float = 1.0) -> float:
        acc_bits = self.config.grad_acc_bits if self.config.grad_acc_bits else None
        keys = self.selected_layer_keys.get(layer_id, [])
        total = sum(self._deq_acc(self.get_accumulator(k), k).float().pow(2).sum().item() for k in keys)
        norm = math.sqrt(total)
        if norm > max_norm and norm > 0:
            coeff = max_norm / norm
            for k in keys:
                acc = self.get_accumulator(k)
                if isinstance(acc, tuple):
                    q, s = quantize_tensor(self._deq_acc(acc, k) * coeff, acc_bits)
                    acc[0].copy_(q)
                    acc[1].copy_(s)
                else:
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
        opt_bits = self.config.opt_state_bits if self.config.opt_state_bits else None
        acc_bits = self.config.grad_acc_bits if self.config.grad_acc_bits else None
        for name, key in self.module_name_to_key.get(layer_id, {}).items():
            w = self.true_weights[key]
            grad = self._deq_acc(self.get_accumulator(key), key)
            state = self.get_opt_state(key)
            w_dev = w.to(self.config.device).float()
            g_dev = grad.to(self.config.device)
            m = self._deq(state["exp_avg"], w.shape, opt_bits or 8).to(self.config.device)
            v = self._deq(state["exp_avg_sq"], w.shape, opt_bits or 8).to(self.config.device)
            if wd:
                w_dev.mul_(1 - lr * wd)
            m.mul_(beta1).add_(g_dev, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(g_dev, g_dev, value=1 - beta2)
            denom = v.sqrt().div_(math.sqrt(b2)).add_(eps)
            w_dev.addcdiv_(m, denom, value=-(lr / b1))
            w.copy_(w_dev.to(w.dtype))
            if opt_bits:
                self.opt_states[key]["exp_avg"] = quantize_tensor(m.cpu(), opt_bits)
                self.opt_states[key]["exp_avg_sq"] = quantize_tensor(v.cpu(), opt_bits)
            else:
                state["exp_avg"].copy_(m)
                state["exp_avg_sq"].copy_(v)
            if acc_bits:
                q, s = quantize_tensor(torch.zeros_like(w, dtype=torch.float32), acc_bits)
                acc_q, acc_s = self.get_accumulator(key)
                acc_q.copy_(q)
                acc_s.copy_(s)
            else:
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
        # Tied word embeddings: lm_head.weight is the same logical weight as the
        # embedding (wte/embed_tokens), so it must not be selected as a separate
        # trainable tensor (avoids gradient divergence and doubled embedding RAM).
        tied = bool(getattr(getattr(self.adapter, "hf_config", None), "tie_word_embeddings", False))
        for layer_id in layer_ids:
            names = [
                n for n in self.adapter.get_param_names_for_layer(layer_id)
                if self.adapter.get_tensor_shape(self.gbi, n) is not None
                and self._match_modules(n)
                and not (tied and n == "lm_head.weight")
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
                shape = self.adapter.get_tensor_shape(self.gbi, name)
                if shape is not None:
                    total += math.prod(shape)
        return total

    def save_delta(self, save_directory: str) -> None:
        import os
        from safetensors.torch import save_file as st_save
        os.makedirs(save_directory, exist_ok=True)
        tensors = {}
        index_weights = {}
        for key, w in self.true_weights.items():
            _, name = key
            tensors[name] = (w.float() - self.original[key].float()).contiguous()
            index_weights[name] = {"layer_id": key[0]}
        st_save(tensors, os.path.join(save_directory, "delta.safetensors"))
        index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                 "weights": index_weights}
        with open(os.path.join(save_directory, "delta.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    def load_delta(self, load_directory: str) -> None:
        import os
        from safetensors import safe_open
        delta_path = os.path.join(load_directory, "delta.safetensors")
        if not os.path.exists(delta_path):
            logger.warning(f"LEMA: no delta.safetensors found in {load_directory} — loading base weights only")
            return
        name_to_key = {key[1]: key for key in self.true_weights}
        with safe_open(delta_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                key = name_to_key.get(name)
                if key is not None:
                    # Add in fp32 then cast once: adding in model dtype would round
                    # the fp32 delta before the add and silently lose precision
                    # (original + delta != w for fp16/bf16 weights).
                    restored = self.original[key].float() + f.get_tensor(name).float()
                    self.true_weights[key].copy_(restored.to(self.true_weights[key].dtype))

    def save_optimizer(self, save_directory: str) -> None:
        import os
        os.makedirs(save_directory, exist_ok=True)
        states = {}
        for key, s in self.opt_states.items():
            ref = f"{key[0]}.{key[1]}"
            if isinstance(s["exp_avg"], tuple):
                states[ref] = {
                    "exp_avg": s["exp_avg"][0],
                    "exp_avg_sq": s["exp_avg_sq"][0],
                    "exp_avg.scale": s["exp_avg"][1],
                    "exp_avg_sq.scale": s["exp_avg_sq"][1],
                }
            else:
                states[ref] = {
                    "exp_avg": s["exp_avg"],
                    "exp_avg_sq": s["exp_avg_sq"],
                }
        torch.save({"layer_steps": self.layer_steps, "states": states},
                   os.path.join(save_directory, "optimizer_fullft.bin"))

    def load_optimizer(self, load_directory: str) -> None:
        import os
        path = os.path.join(load_directory, "optimizer_fullft.bin")
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location="cpu", weights_only=True)
        self.layer_steps = data["layer_steps"]
        key_by_ref = {f"{key[0]}.{key[1]}": key for key in self.opt_states}
        opt_bits = self.config.opt_state_bits if self.config.opt_state_bits else None
        for ref, s in data["states"].items():
            key = key_by_ref.get(ref)
            if key is None:
                continue
            target = self.opt_states[key]
            shape = self._q(target["exp_avg"]).shape
            m = self._restore_state_val(s, "exp_avg", shape)
            v = self._restore_state_val(s, "exp_avg_sq", shape)
            if opt_bits:
                target["exp_avg"] = quantize_tensor(m, opt_bits)
                target["exp_avg_sq"] = quantize_tensor(v, opt_bits)
            else:
                target["exp_avg"].copy_(m)
                target["exp_avg_sq"].copy_(v)
