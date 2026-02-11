import torch
import threading
from typing import Dict, Optional, List, Tuple
from enum import Enum
import gc
from ..config import MemoryStrategy

class TripleBufferManager:
    """
    Unified Memory Manager supporting both Disk-Streaming and RAM-Residency.
    """
    def __init__(self, gbi, adapter, device="cuda", strategy=MemoryStrategy.STREAMING):
        self.gbi = gbi
        self.adapter = adapter
        self.device = device
        self.strategy = strategy
        
        self.is_cuda = self.device.startswith("cuda")
        self.transfer_streams = [torch.cuda.Stream() for _ in range(2)] if self.is_cuda else None
        
        self.layers_meta = self.adapter.get_layer_metadata()
        
        # Calculate max layer size for pre-allocating buffers
        self.max_params = self._calculate_max_params()
        
        # Pre-allocated VRAM slots (Double buffering)
        self.vram_flat_buffers = [
            torch.empty(self.max_params, device=self.device, dtype=torch.float32)
            for _ in range(2)
        ]
        
        # RAM Buffers
        if self.strategy == MemoryStrategy.RESIDENT:
            print(f"LEMA: Initializing RESIDENT strategy (Caching model in RAM)...")
            self.ram_flat_buffers: Dict[int, torch.Tensor] = {}
            self._initialize_full_ram_cache()
        else:
            print(f"LEMA: Initializing STREAMING strategy (Default)...")
            # In streaming mode, we only need 2 RAM slots for the pipeline
            self.ram_flat_buffers: List[torch.Tensor] = [
                torch.empty(self.max_params, device="cpu", dtype=torch.float32).pin_memory() if self.is_cuda else torch.empty(self.max_params, device="cpu", dtype=torch.float32)
                for _ in range(2)
            ]
            self.ram_layer_ids = [-1, -1]

    def _calculate_max_params(self) -> int:
        max_p = 0
        for layer in self.layers_meta:
            names = self.adapter.get_param_names_for_layer(layer['id'])
            current_p = 0
            for name in names:
                meta = self.gbi.handle.get_slice(name)
                current_p += meta.get_shape().numel() if hasattr(meta.get_shape(), 'numel') else torch.Size(meta.get_shape()).numel()
            max_p = max(max_p, current_p)
        return max_p

    def _initialize_full_ram_cache(self):
        """Pre-packs the entire model into pinned RAM."""
        for layer in self.layers_meta:
            layer_id = layer['id']
            self._pack_layer_to_ram(layer_id, is_resident=True)

    def _pack_layer_to_ram(self, layer_id: int, slot: int = 0, is_resident: bool = False):
        """Helper to load a layer from disk and pack it into a flat RAM buffer."""
        param_names = self.adapter.get_param_names_for_layer(layer_id)
        weights = self.gbi.load_tensors(param_names, device="cpu")
        
        if is_resident:
            total_el = sum(w.numel() for w in weights.values())
            buf = torch.empty(total_el, device="cpu", dtype=torch.float32).pin_memory()
        else:
            buf = self.ram_flat_buffers[slot]
            
        offset = 0
        for name in param_names:
            w = weights[name]
            numel = w.numel()
            buf[offset : offset + numel].copy_(w.view(-1))
            offset += numel
            
        if is_resident:
            self.ram_flat_buffers[layer_id] = buf
        else:
            self.ram_layer_ids[slot] = layer_id

    def prefetch_to_ram(self, layer_id: int, ram_slot: int):
        """Stage 1 (Streaming only): Load from Disk to RAM Slot."""
        if self.strategy == MemoryStrategy.RESIDENT:
            return # No-op for resident mode
            
        if self.ram_layer_ids[ram_slot] == layer_id:
            return
        
        self._pack_layer_to_ram(layer_id, ram_slot, is_resident=False)

    def async_transfer_to_vram(self, layer_id: int, vram_slot: int, ram_slot: Optional[int] = None):
        """Stage 2: Async transfer to GPU."""
        if self.strategy == MemoryStrategy.RESIDENT:
            src_buf = self.ram_flat_buffers[layer_id]
        else:
            if ram_slot is None:
                raise ValueError("ram_slot must be provided in streaming mode")
            src_buf = self.ram_flat_buffers[ram_slot]
            
        vram_dest = self.vram_flat_buffers[vram_slot]
        
        if self.is_cuda and self.transfer_streams:
            stream = self.transfer_streams[vram_slot]
            with torch.cuda.stream(stream):
                vram_dest[:src_buf.numel()].copy_(src_buf, non_blocking=True)
        else:
            # CPU or Synchronous copy
            vram_dest[:src_buf.numel()].copy_(src_buf)

    def get_vram_flat_buffer(self, vram_slot: int) -> torch.Tensor:
        """Stage 3: Usage."""
        if self.is_cuda and self.transfer_streams:
            self.transfer_streams[vram_slot].synchronize()
        return self.vram_flat_buffers[vram_slot]

    def clear_vram_slot(self, vram_slot: int):
        pass