from __future__ import annotations

import torch
import numpy as np
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


class KVChunkStore:
    """Stores and streams KV cache chunks, keyed by (layer_id, chunk_idx).

    RAM primary (pinned dict), mmap file fallback when RAM budget is exhausted.
    Supports incremental append for generation (grows the current chunk, rolls
    to a new chunk at the boundary)."""

    def __init__(self, kv_chunk_size: int = 8192, max_ram_gb: float = 0.0,
                 disk_dir: str | None = None, dtype: torch.dtype = torch.float32,
                 device: str = "cpu"):
        self.kv_chunk_size = kv_chunk_size
        self.max_ram_gb = max_ram_gb
        self.disk_dir = disk_dir
        self.dtype = dtype
        self.device = device
        self._ram: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._sizes: dict[tuple[int, int], int] = {}
        self._current: dict[int, int] = {}   # layer -> current chunk_idx
        self._current_size: dict[int, int] = {}  # layer -> tokens in current chunk
        self._memmaps: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._files: dict[tuple[int, int], object] = {}

    def address(self, layer_id: int, chunk_idx: int) -> tuple[StreamKind, int, str]:
        return (StreamKind.KV_CHUNK, layer_id, f"chunk_{chunk_idx}")

    def _use_disk(self) -> bool:
        if self.disk_dir is None:
            return False
        if self.max_ram_gb > 0:
            est_gb = sum(k.numel() * k.element_size() * 2 for k, _ in self._ram.values())
            return est_gb > self.max_ram_gb * 1e9
        return True

    def _disk_paths(self, layer_id: int, chunk_idx: int):
        import os
        base = os.path.join(self.disk_dir, f"kv_{layer_id}_{chunk_idx}")
        return base + "_k.bin", base + "_v.bin"

    def stash(self, layer_id: int, chunk_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        key = (layer_id, chunk_idx)
        self._sizes[key] = k.shape[2]
        # If this chunk is full, the next append starts a fresh chunk (size 0);
        # if partial, append continues growing it.
        is_full = k.shape[2] >= self.kv_chunk_size
        self._current[layer_id] = chunk_idx + 1 if is_full else chunk_idx
        self._current_size[layer_id] = 0 if is_full else k.shape[2]
        if not self._use_disk():
            self._ram[key] = (k.to(self.device), v.to(self.device))
            return
        import numpy as _np
        kp, vp = self._disk_paths(layer_id, chunk_idx)
        km = _np.memmap(kp, dtype="float32", mode="w+", shape=k.shape)
        vm = _np.memmap(vp, dtype="float32", mode="w+", shape=v.shape)
        km[:] = k.float().numpy()
        vm[:] = v.float().numpy()
        km.flush(); vm.flush()
        self._memmaps[key] = (km, vm)
        self._files[key] = None

    def load(self, layer_id: int, chunk_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_id, chunk_idx)
        if key in self._ram:
            return self._ram[key]
        if key in self._memmaps:
            km, vm = self._memmaps[key]
            return (torch.from_numpy(np.array(km)).to(self.dtype).to(self.device),
                    torch.from_numpy(np.array(vm)).to(self.dtype).to(self.device))
        raise KeyError(f"KV chunk not found: {key}")

    def append(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one token's K/V to the current chunk, rolling to a new chunk
        at the boundary. k/v shape: (batch, heads, 1, head_dim)."""
        cur = self._current.get(layer_id, 0)
        size = self._current_size.get(layer_id, 0)
        key = (layer_id, cur)
        if size == 0:
            self.stash(layer_id, cur, k, v)
            self._current_size[layer_id] = 1
            if self.kv_chunk_size == 1:
                self._current[layer_id] = cur + 1
                self._current_size[layer_id] = 0
            return
        # grow current chunk: load, concatenate, re-stash
        if not self._use_disk() and key in self._ram:
            k_cur, v_cur = self._ram[key]
            k_new = torch.cat([k_cur, k], dim=2)
            v_new = torch.cat([v_cur, v], dim=2)
            self._ram[key] = (k_new, v_new)
            self._sizes[key] = k_new.shape[2]
        else:
            k_cur, v_cur = self.load(layer_id, cur)
            k_new = torch.cat([k_cur, k], dim=2)
            v_new = torch.cat([v_cur, v], dim=2)
            self.stash(layer_id, cur, k_new, v_new)
        self._current_size[layer_id] = size + 1
        if self._current_size[layer_id] >= self.kv_chunk_size:
            self._current[layer_id] = cur + 1
            self._current_size[layer_id] = 0

    def num_chunks(self, layer_id: int) -> int:
        tracked = self._current.get(layer_id, 0) + (1 if self._current_size.get(layer_id, 0) > 0 else 0)
        if tracked > 0:
            return tracked
        return max([c for (l, c) in self._sizes if l == layer_id] or [0]) + 1

    def current_size(self, layer_id: int) -> int:
        return self._current_size.get(layer_id, 0)

    def close(self) -> None:
        for f in self._files.values():
            try:
                if f is not None:
                    f.close()
            except Exception:
                pass
        self._files.clear()
        self._memmaps.clear()
        self._ram.clear()


def chunked_attention(q: torch.Tensor, kv_chunks: list[tuple[torch.Tensor, torch.Tensor]],
                      scale: float | None = None, query_start: int = 0,
                      kv_chunk_size: int | None = None, causal: bool = False,
                      dropout=None) -> torch.Tensor:
    """Exact chunked attention over a list of (K, V) chunks.

    q: (batch, heads, q_len, head_dim). Each chunk's K/V: (batch, heads, chunk_len, head_dim).
    Chunk c covers global key positions [c*kv_chunk_size, (c+1)*kv_chunk_size).
    Builds the full score matrix from per-chunk matmuls concatenated along the
    key dimension, then a single fp32 softmax and a single weighted sum against
    the concatenated V. Mathematically identical to full-resident attention;
    bit-exact against a reference built with the same per-chunk decomposition.

    When causal=True, keys at positions strictly after each query position are
    masked to -inf (query positions start at query_start in global key space).
    dropout: an optional nn.Dropout applied to the normalized attention weights
    (replicates the module's attn_dropout in training mode).
    """
    if scale is None:
        head_dim = q.size(-1)
        scale = 1.0 / (head_dim ** 0.5)
    qf = q.float()
    scores_parts = []
    for k, _ in kv_chunks:
        scores_parts.append(qf @ k.float().transpose(-2, -1))
    scores = torch.cat(scores_parts, dim=-1) * scale
    if causal:
        if kv_chunk_size is None:
            raise ValueError("kv_chunk_size is required when causal=True")
        q_pos = torch.arange(query_start, query_start + q.size(-2), device=q.device)
        k_len = scores.size(-1)
        k_pos = torch.arange(0, k_len, device=q.device)
        causal_mask = q_pos[:, None] < k_pos[None, :]  # (q_len, k_len)
        mask_value = torch.finfo(scores.dtype).min
        scores = torch.where(causal_mask[None, None], torch.full_like(scores, mask_value), scores)
    m = scores.max(dim=-1, keepdim=True).values
    p = (scores - m).exp()
    denom = p.sum(dim=-1, keepdim=True)
    p = p / denom
    if dropout is not None:
        p = dropout(p)
    v_full = torch.cat([v for _, v in kv_chunks], dim=2).float()
    out = p @ v_full
    return out.to(q.dtype)
