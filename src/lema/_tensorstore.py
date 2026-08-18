from __future__ import annotations

import torch
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class StreamKind(Enum):
    WEIGHTS = "weights"
    OPT_STATE = "opt_state"
    GRAD_ACC = "grad_acc"
    KV_CHUNK = "kv_chunk"


@dataclass
class Stream:
    kind: StreamKind
    layer_id: int
    key: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    source: Callable[[], torch.Tensor] | None = None


def parse_vram_setting(value: str, max_vram_gb: float) -> float:
    v = value.strip().lower()
    if v in ("", "auto"):
        return max_vram_gb / 4.0
    if v.endswith("gb"):
        return float(v[:-2])
    return float(v) * max_vram_gb


class TensorStore:
    _KINDS = [StreamKind.WEIGHTS, StreamKind.OPT_STATE, StreamKind.GRAD_ACC, StreamKind.KV_CHUNK]

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._streams: dict[tuple[StreamKind, int, str], Stream] = {}
        self._resident: dict[tuple[StreamKind, int, str], torch.Tensor] = {}
        self._lru: list[tuple[StreamKind, int, str]] = []
        self._device = "cpu"
        self._budgets: dict[StreamKind, float] = {}

    @classmethod
    def with_budget(cls, config) -> "TensorStore":
        store = cls()
        store._budgets = {}
        field_map = {
            StreamKind.WEIGHTS: "weights_vram",
            StreamKind.OPT_STATE: "opt_state_vram",
            StreamKind.GRAD_ACC: "grad_acc_vram",
            StreamKind.KV_CHUNK: "kv_vram",
        }
        explicit = {}
        for kind, fname in field_map.items():
            val = getattr(config, fname, "auto")
            if val != "auto":
                explicit[kind] = parse_vram_setting(val, config.max_vram_gb)
        used = sum(explicit.values())
        if used > config.max_vram_gb:
            scale = config.max_vram_gb / used
            explicit = {k: v * scale for k, v in explicit.items()}
        remainder = max(config.max_vram_gb - sum(explicit.values()), 0.0)
        auto_kinds = [k for k in cls._KINDS if k not in explicit]
        share = remainder / max(len(auto_kinds), 1)
        for kind in cls._KINDS:
            store._budgets[kind] = explicit.get(kind, share)
        return store

    def kind_budget(self, kind: StreamKind) -> float:
        return self._budgets.get(kind, 0.0)

    def set_device(self, device: str) -> None:
        self._device = device

    def register(self, stream: Stream) -> None:
        key = (stream.kind, stream.layer_id, stream.key)
        if key in self._streams:
            raise ValueError(f"Stream already registered: {key}")
        self._streams[key] = stream

    def __contains__(self, key) -> bool:
        return key in self._streams

    def ensure(self, *keys, _no_evict: bool = False) -> dict[tuple, torch.Tensor]:
        out = {}
        for key in keys:
            if key not in self._streams:
                raise KeyError(f"Unknown stream: {key}")
            if key not in self._resident:
                if len(self._resident) >= self.capacity:
                    if _no_evict:
                        raise KeyError(f"Stream not resident and at capacity: {key}")
                    self._evict_lru()
                s = self._streams[key]
                t = s.source() if s.source is not None else torch.zeros(*s.shape, dtype=s.dtype)
                self._resident[key] = t.to(self._device)
            self._lru = [k for k in self._lru if k != key] + [key]
            out[key] = self._resident[key]
        return out

    def evict(self, *keys) -> None:
        for key in keys:
            self._resident.pop(key, None)
            self._lru = [k for k in self._lru if k != key]

    def evict_lru(self, *keys) -> None:
        self.evict(*keys)

    def _evict_lru(self) -> None:
        if self._lru:
            victim = self._lru.pop(0)
            self._resident.pop(victim, None)

    def slots(self) -> dict[StreamKind, int]:
        counts: dict[StreamKind, int] = {}
        for k in self._resident:
            counts[k[0]] = counts.get(k[0], 0) + 1
        return counts

    def streams(self) -> dict[tuple[StreamKind, int, str], Stream]:
        return dict(self._streams)

    def __iter__(self):
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._resident)
