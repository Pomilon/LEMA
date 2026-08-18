from __future__ import annotations

import torch
import psutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from ._config import LemaConfig, MemoryStrategy
from ._utils._logger import logger

try:
    from ._csrc import _lema_cpp
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False


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


class _TransferEngine:
    """The triple-buffer transfer pipeline (disk -> pinned RAM -> VRAM slots),
    formerly TripleBufferManager. Owned by the TensorStore as the WEIGHTS policy's
    transport."""

    def __init__(self, gbi, adapter, config: LemaConfig):
        self.gbi = gbi
        self.adapter = adapter
        self.config = config
        self.device = config.device
        self.strategy = config.strategy

        self.is_cuda = self.device.startswith("cuda")
        self.layers_meta = self.adapter.get_layer_metadata()

        # 1. Precision Detection
        self.dtype = getattr(torch, self.config.dtype) if isinstance(self.config.dtype, str) else self.config.dtype
        if self.gbi.get_keys():
            sample_key = self.gbi.get_keys()[0]
            try:
                sample_tensor = self.gbi.load_tensors([sample_key])[sample_key]
                if sample_tensor.dtype != self.dtype:
                     logger.info(f"LEMA: Auto-detected model dtype {sample_tensor.dtype}. Adjusting buffers.")
                     self.dtype = sample_tensor.dtype
            except: pass

        self.itemsize = torch.tensor([], dtype=self.dtype).element_size()
        self.max_params = self._calculate_max_params()

        # 2. Dynamic Resource Detection
        if self.config.max_ram_gb <= 0:
            total_ram = psutil.virtual_memory().total / (1024**3)
            self.config.max_ram_gb = total_ram * 0.7
            logger.info(f"LEMA: Auto-detected RAM. Setting budget to {self.config.max_ram_gb:.2f} GB")

        if self.is_cuda and self.config.max_vram_gb <= 0:
            total_vram = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
            self.config.max_vram_gb = total_vram * 0.6
            logger.info(f"LEMA: Auto-detected VRAM. Setting budget to {self.config.max_vram_gb:.2f} GB")

        if self.is_cuda and self.config.vram_fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(self.config.vram_fraction)

        # 3. Pre-allocated VRAM slots (Double buffering)
        slot_size_gb = (self.max_params * self.itemsize) / (1024**3)
        if 2 * slot_size_gb > self.config.max_vram_gb:
            logger.warning(f"LEMA: VRAM slots ({2 * slot_size_gb:.2f} GB) exceed budget ({self.config.max_vram_gb:.2f} GB)")

        self.vram_flat_buffers = [
            torch.empty(self.max_params, device=self.device, dtype=self.dtype)
            for _ in range(2)
        ]

        # Per-instance CUDA event tracking (avoids full stream.synchronize())
        self._transfer_event_ids: dict[int, int] = {}

        # 4. Initialize C++ Memory Manager
        backend = self.config.backend
        if backend == "auto":
            if not HAS_CPP_BACKEND and self.is_cuda:
                logger.warning("C++ backend not available. Use backend='python' to silence this warning.")
            self.use_cpp = HAS_CPP_BACKEND and self.is_cuda
        elif backend == "cpp":
            if not HAS_CPP_BACKEND:
                raise RuntimeError("C++ backend requested but not available. Install with CUDA extension or use backend='python'.")
            if not self.is_cuda:
                raise RuntimeError("C++ backend requires CUDA device.")
            self.use_cpp = True
        elif backend == "python":
            self.use_cpp = False
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'auto', 'cpp', or 'python'.")

        if self.use_cpp:
            self.cpp_mgr = _lema_cpp.LemaMemoryManager(len(self.layers_meta) + 2, self.max_params)
            for i, buf in enumerate(self.vram_flat_buffers):
                self.cpp_mgr.register_vram_slot(i, buf)
        else:
            self.cpp_mgr = None
            self.transfer_streams = [torch.cuda.Stream() for _ in range(2)] if self.is_cuda else None

        # 5. Python ThreadPoolExecutor for background prefetching
        num_prefetch_workers = min(2, (psutil.cpu_count() or 2))
        self.prefetch_executor = ThreadPoolExecutor(
            max_workers=num_prefetch_workers,
            thread_name_prefix="lema_prefetch"
        )
        self._prefetch_futures: dict[int, Any] = {}

        # 6. RAM Strategy Logic
        self.ram_buffers = {}
        total_model_params = sum(l.get('size', 0) for l in self.layers_meta) or self._calculate_total_params()
        total_model_gb = (total_model_params * self.itemsize) / (1024**3)

        if self.strategy == MemoryStrategy.RESIDENT:
            logger.info(f"LEMA: Initializing RESIDENT strategy (Precision: {self.dtype})...")
            self._initialize_ram_cache()
            return

        # STREAMING initialization
        logger.info(f"LEMA: Initializing STREAMING strategy (Precision: {self.dtype})...")
        for i in range(2):
            buf = torch.empty(self.max_params, device="cpu", dtype=self.dtype)
            if self.is_cuda:
                buf = buf.pin_memory()
            self.ram_buffers[1000 + i] = buf
            if self.use_cpp:
                self.cpp_mgr.register_ram_buffer(1000 + i, buf)
        self.ram_layer_ids = [-1, -1]

    def _sum_layer_params(self, layer_id: int) -> int:
        total = 0
        for name in self.adapter.get_param_names_for_layer(layer_id):
            try:
                shape = self.gbi.get_tensor_shape(name)
            except Exception:
                continue
            if shape is not None:
                total += torch.Size(shape).numel()
        return total

    def _calculate_total_params(self) -> int:
        return sum(self._sum_layer_params(l['id']) for l in self.layers_meta)

    def _calculate_max_params(self) -> int:
        max_p = max(self._sum_layer_params(l['id']) for l in self.layers_meta)
        if max_p == 0:
            raise RuntimeError(
                f"No valid tensor shapes found. Available GBI keys: {list(self.gbi.get_keys())[:20]}..."
            )
        return max_p

    def _initialize_ram_cache(self):
        """Loads as many layers as possible into RAM budget (Greedy)."""
        self.ram_buffers = {}
        processed_gb = 0
        resident_count = 0

        for layer in self.layers_meta:
            names = self.adapter.get_param_names_for_layer(layer['id'])
            layer_params = 0
            for name in names:
                shape = self.gbi.get_tensor_shape(name)
                layer_params += torch.Size(shape).numel()

            layer_gb = (layer_params * self.itemsize) / (1024**3)
            if processed_gb + layer_gb <= self.config.max_ram_gb * 0.9:
                self._pack_layer_to_ram(layer['id'], is_resident=True)
                processed_gb += layer_gb
                resident_count += 1
            else:
                break

        if resident_count > 0:
            logger.info(f"LEMA: {resident_count}/{len(self.layers_meta)} layers are now RESIDENT in RAM ({processed_gb:.2f} GB).")

        # Streaming slots for remaining layers
        for i in range(2):
            buf = torch.empty(self.max_params, device="cpu", dtype=self.dtype)
            if self.is_cuda:
                buf = buf.pin_memory()
            self.ram_buffers[1000 + i] = buf
            if self.use_cpp:
                self.cpp_mgr.register_ram_buffer(1000 + i, buf)
        self.ram_layer_ids = [-1, -1]

    def _pack_layer_to_ram(self, layer_id: int, slot: int = 0, is_resident: bool = False):
        """Load a layer from disk and pack into a flat RAM buffer."""
        param_names = self.adapter.get_param_names_for_layer(layer_id)
        weights = self.gbi.load_tensors(param_names, device="cpu")

        if is_resident:
            total_el = sum(w.numel() for w in weights.values())
            buf = torch.empty(total_el, device="cpu", dtype=self.dtype)
            if self.is_cuda:
                buf = buf.pin_memory()
            self.ram_buffers[layer_id] = buf
            if self.use_cpp:
                self.cpp_mgr.register_ram_buffer(layer_id, buf)
        else:
            buf = self.ram_buffers[1000 + slot]

        # Python packing (equally fast as C++ memcpy, no pybind11 overhead)
        offset = 0
        for name in param_names:
            w = weights[name]
            numel = w.numel()
            buf[offset : offset + numel].copy_(w.view(-1))
            offset += numel

        del weights

        if not is_resident:
            self.ram_layer_ids[slot] = layer_id

    def prefetch_to_ram(self, layer_id: int, slot: int):
        """Stage 1: Load from Disk to RAM Slot (if not already resident)."""
        if layer_id in self.ram_buffers and layer_id < 1000:
            return
        if self.ram_layer_ids[slot] == layer_id:
            return
        self._pack_layer_to_ram(layer_id, slot=slot, is_resident=False)
        self.ram_layer_ids[slot] = layer_id

    def prefetch_to_ram_async(self, layer_id: int, slot: int):
        """Submit a prefetch job to the thread pool (non-blocking)."""
        if layer_id in self.ram_buffers and layer_id < 1000:
            return
        if self.ram_layer_ids[slot] == layer_id:
            return
        if slot in self._prefetch_futures:
            future = self._prefetch_futures[slot]
            if not future.done():
                return
        future = self.prefetch_executor.submit(
            self._pack_layer_to_ram, layer_id, slot, False
        )
        self._prefetch_futures[slot] = future

    def wait_prefetch(self, slot: int):
        """Wait for any outstanding prefetch to complete."""
        if slot in self._prefetch_futures:
            future = self._prefetch_futures[slot]
            if not future.done():
                future.result()
            del self._prefetch_futures[slot]

    def async_transfer_to_vram(self, layer_id: int, vram_slot: int, ram_slot: int | None = None):
        """Stage 2: Async transfer from RAM to GPU VRAM."""
        is_resident = (layer_id in self.ram_buffers and layer_id < 1000)

        if self.use_cpp:
            cpp_layer_id = layer_id if is_resident else (1000 + (ram_slot or 0))
            event_id = self.cpp_mgr.async_transfer_to_vram(cpp_layer_id, vram_slot)
            self._transfer_event_ids[vram_slot] = event_id
        else:
            ram_buf = self.ram_buffers[layer_id] if is_resident else self.ram_buffers[1000 + (ram_slot or 0)]
            vram_buf = self.vram_flat_buffers[vram_slot]

            if self.is_cuda and self.transfer_streams:
                with torch.cuda.stream(self.transfer_streams[vram_slot]):
                    vram_buf[:ram_buf.numel()].copy_(ram_buf, non_blocking=True)
            else:
                vram_buf[:ram_buf.numel()].copy_(ram_buf)

    def get_vram_flat_buffer(self, vram_slot: int) -> torch.Tensor:
        """Stage 3: Wait for transfer to complete and return VRAM buffer."""
        if self.use_cpp:
            event_id = self._transfer_event_ids.pop(vram_slot, -1)
            if event_id >= 0:
                self.cpp_mgr.wait_vram_transfer(event_id)
            return self.vram_flat_buffers[vram_slot]
        else:
            if self.is_cuda and self.transfer_streams:
                self.transfer_streams[vram_slot].synchronize()
            return self.vram_flat_buffers[vram_slot]

    def clear_vram_slot(self, vram_slot: int):
        self.vram_flat_buffers[vram_slot] = torch.empty(
            self.max_params, device=self.device, dtype=self.dtype
        )
        self._transfer_event_ids.pop(vram_slot, None)

    def close(self):
        """Explicit cleanup. Releases GPU memory, shuts down thread pools, destroys C++ backend."""
        if hasattr(self, "prefetch_executor"):
            self.prefetch_executor.shutdown(wait=False)
        if hasattr(self, "cpp_mgr") and self.cpp_mgr is not None:
            del self.cpp_mgr
            self.cpp_mgr = None
        for k in list(self.ram_buffers.keys()):
            self.ram_buffers[k] = None
        for i in range(len(self.vram_flat_buffers)):
            self.vram_flat_buffers[i] = torch.empty(1, device=self.device)
        self._transfer_event_ids.clear()
        self._prefetch_futures.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TensorStore:
    _KINDS = [StreamKind.WEIGHTS, StreamKind.OPT_STATE, StreamKind.GRAD_ACC, StreamKind.KV_CHUNK]

    def __init__(self, capacity: int = 2, gbi=None, adapter=None, config: LemaConfig | None = None):
        self.capacity = capacity
        self._streams: dict[tuple[StreamKind, int, str], Stream] = {}
        self._resident: dict[tuple[StreamKind, int, str], torch.Tensor] = {}
        self._lru: list[tuple[StreamKind, int, str]] = []
        self._device = "cpu"
        self._budgets: dict[StreamKind, float] = {}
        self.transfer: _TransferEngine | None = None
        if gbi is not None and adapter is not None and config is not None:
            self.transfer = _TransferEngine(gbi, adapter, config)
            self._device = config.device

    @classmethod
    def with_budget(cls, config, gbi=None, adapter=None) -> "TensorStore":
        store = cls(gbi=gbi, adapter=adapter, config=config)
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

    @property
    def dtype(self):
        return self.transfer.dtype if self.transfer is not None else torch.float32

    @property
    def device(self):
        return self._device

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

    def close(self) -> None:
        if self.transfer is not None:
            self.transfer.close()

    def __iter__(self):
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._resident)
