import torch
import threading
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum

from ..config import LemaConfig, MemoryStrategy
from ..utils.logger import logger

try:
    from ..csrc import _lema_cpp
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

class TripleBufferManager:
    """
    Unified Memory Manager supporting both Disk-Streaming and RAM-Residency.

    Architecture:
      Disk -> RAM (pack)   : Python copy_() via ThreadPoolExecutor
      RAM  -> VRAM (xfer)  : C++ backend via CUDA events (if available)
      VRAM -> Compute      : CUDA event wait (avoids full stream sync)

    The C++ backend provides value via:
      - CUDA event-based transfer tracking (avoids stream.synchronize())
      - Background prefetch thread pool for disk I/O
      - Atomic status flags (no mutex contention)
    """
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
        self._transfer_event_ids: Dict[int, int] = {}

        # 4. Initialize C++ Memory Manager
        self.cpp_mgr = None
        if HAS_CPP_BACKEND and self.is_cuda:
            self.cpp_mgr = _lema_cpp.LemaMemoryManager(len(self.layers_meta) + 2, self.max_params)
            for i, buf in enumerate(self.vram_flat_buffers):
                self.cpp_mgr.register_vram_slot(i, buf)
        else:
            self.transfer_streams = [torch.cuda.Stream() for _ in range(2)] if self.is_cuda else None

        # 5. Python ThreadPoolExecutor for background prefetching
        #    (replaces per-layer threading.Thread creation which had ~50-100us overhead)
        num_prefetch_workers = min(2, (psutil.cpu_count() or 2))
        self.prefetch_executor = ThreadPoolExecutor(
            max_workers=num_prefetch_workers,
            thread_name_prefix="lema_prefetch"
        )
        self._prefetch_futures: Dict[int, Any] = {}

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
            if self.cpp_mgr:
                self.cpp_mgr.register_ram_buffer(1000 + i, buf)
        self.ram_layer_ids = [-1, -1]

    def _calculate_total_params(self) -> int:
        total = 0
        for layer in self.layers_meta:
            names = self.adapter.get_param_names_for_layer(layer['id'])
            for name in names:
                shape = self.gbi.get_tensor_shape(name)
                total += torch.Size(shape).numel()
        return total

    def _calculate_max_params(self) -> int:
        max_p = 0
        for layer in self.layers_meta:
            names = self.adapter.get_param_names_for_layer(layer['id'])
            current_p = 0
            for name in names:
                shape = self.gbi.get_tensor_shape(name)
                current_p += torch.Size(shape).numel()
            max_p = max(max_p, current_p)
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
            if self.cpp_mgr:
                self.cpp_mgr.register_ram_buffer(1000 + i, buf)
        self.ram_layer_ids = [-1, -1]

    def _pack_layer_to_ram(self, layer_id: int, slot: int = 0, is_resident: bool = False):
        """Load a layer from disk and pack into a flat RAM buffer.

        Uses Python copy_() which internally uses memcpy — same as C++ memcpy
        but avoids pybind11 overhead and mutex contention.
        """
        param_names = self.adapter.get_param_names_for_layer(layer_id)
        weights = self.gbi.load_tensors(param_names, device="cpu")

        if is_resident:
            total_el = sum(w.numel() for w in weights.values())
            buf = torch.empty(total_el, device="cpu", dtype=self.dtype)
            if self.is_cuda:
                buf = buf.pin_memory()
            self.ram_buffers[layer_id] = buf
            if self.cpp_mgr:
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
        """Stage 1: Load from Disk to RAM Slot (if not already resident).

        Uses ThreadPoolExecutor to avoid per-call thread creation overhead.
        """
        if layer_id in self.ram_buffers and layer_id < 1000:
            return

        if self.ram_layer_ids[slot] == layer_id:
            return

        self._pack_layer_to_ram(layer_id, slot=slot, is_resident=False)
        self.ram_layer_ids[slot] = layer_id

    def prefetch_to_ram_async(self, layer_id: int, slot: int):
        """Submit a prefetch job to the thread pool (non-blocking).

        Avoids creating a new threading.Thread per layer (~50-100us overhead).
        """
        if layer_id in self.ram_buffers and layer_id < 1000:
            return

        if self.ram_layer_ids[slot] == layer_id:
            return

        # Cancel any pending prefetch for this slot
        if slot in self._prefetch_futures:
            future = self._prefetch_futures[slot]
            if not future.done():
                return  # Already being prefetched

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

    def async_transfer_to_vram(self, layer_id: int, vram_slot: int, ram_slot: Optional[int] = None):
        """Stage 2: Async transfer from RAM to GPU VRAM.

        Uses C++ backend with CUDA events when available (avoids stream.synchronize()).
        Falls back to Python torch.cuda.stream() otherwise.
        """
        is_resident = (layer_id in self.ram_buffers and layer_id < 1000)

        if self.cpp_mgr:
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
        """Stage 3: Wait for transfer to complete and return VRAM buffer.

        With C++: uses cudaEventSynchronize (precise, per-transfer).
        Without C++: uses stream.synchronize() (coarser, per-stream).
        """
        if self.cpp_mgr:
            event_id = self._transfer_event_ids.pop(vram_slot, -1)
            if event_id >= 0:
                self.cpp_mgr.wait_vram_transfer(event_id)
            return self.vram_flat_buffers[vram_slot]
        else:
            if self.is_cuda and self.transfer_streams:
                self.transfer_streams[vram_slot].synchronize()
            return self.vram_flat_buffers[vram_slot]

    def clear_vram_slot(self, vram_slot: int):
        pass
