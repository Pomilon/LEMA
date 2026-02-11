import torch
import torch.nn.functional as F
import threading
from typing import Any, Optional, List, Union
from ..core.memory import TripleBufferManager
from ..models.base import LemaModelAdapter
from ..config import LemaConfig, MemoryStrategy

class LemaTrainer:
    def __init__(self, 
                 config: LemaConfig,
                 model_adapter: LemaModelAdapter, 
                 gbi, 
                 lora_manager=None, 
                 optimizer=None):
        
        self.config = config
        self.adapter = model_adapter
        self.gbi = gbi
        self.device = config.device
        self.strategy = config.strategy
        
        # Initialize Memory Manager with config strategy
        # Note: TripleBufferManager might need updates to accept config too, 
        # but for now we map config values to its expected args
        self.memory = TripleBufferManager(gbi, model_adapter, self.device, strategy=self.strategy)
        
        self.layers = self.adapter.get_layer_metadata()
        self.lora_manager = lora_manager
        self.optimizer = optimizer
        
    def train_step(self, inputs: Any, labels: Optional[torch.Tensor] = None):
        """
        Executes one forward pass and one backward pass.
        If labels are provided, computes CrossEntropyLoss.
        """
        boundary_activations: List[torch.Tensor] = []
        is_streaming = (self.strategy == MemoryStrategy.STREAMING)
        
        # --- FORWARD PASS ---
        if is_streaming:
            self.memory.prefetch_to_ram(self.layers[0]['id'], 0)
            self.memory.async_transfer_to_vram(self.layers[0]['id'], 0, ram_slot=0)
            if len(self.layers) > 1:
                self.memory.prefetch_to_ram(self.layers[1]['id'], 1)
        else:
            self.memory.async_transfer_to_vram(self.layers[0]['id'], 0)

        hidden_states = inputs
        
        for i, layer_meta in enumerate(self.layers):
            slot = i % 2
            next_slot = (i + 1) % 2
            
            flat_vram = self.memory.get_vram_flat_buffer(slot)
            
            disk_thread = None
            if i + 1 < len(self.layers):
                if is_streaming:
                    self.memory.async_transfer_to_vram(self.layers[i+1]['id'], next_slot, ram_slot=next_slot)
                    if i + 2 < len(self.layers):
                        disk_thread = threading.Thread(target=self.memory.prefetch_to_ram, args=(self.layers[i+2]['id'], slot))
                        disk_thread.start()
                else:
                    self.memory.async_transfer_to_vram(self.layers[i+1]['id'], next_slot)
            
            layer_module = self.adapter.construct_layer_module(layer_meta['id'], flat_vram, self.lora_manager)
            
            # Store input for backward
            if isinstance(hidden_states, tuple): 
                 current_input = hidden_states[0].detach()
            else:
                current_input = hidden_states.detach()
            boundary_activations.append(current_input)
            
            with torch.no_grad():
                hidden_states = self.adapter.forward_layer(layer_module, hidden_states, gradient_checkpointing=False)

            if disk_thread: disk_thread.join()
            if hasattr(self.adapter, "release_layer_module"):
                self.adapter.release_layer_module(layer_module)
            del layer_module

        # Final Logits
        logits = hidden_states
        loss_val = None

        # --- BACKWARD PASS ---
        if not torch.is_grad_enabled():
            return logits, None

        last_idx = len(self.layers) - 1
        if is_streaming:
            self.memory.prefetch_to_ram(self.layers[last_idx]['id'], 0)
            self.memory.async_transfer_to_vram(self.layers[last_idx]['id'], 0, ram_slot=0)
            if last_idx > 0:
                self.memory.prefetch_to_ram(self.layers[last_idx-1]['id'], 1)
        else:
            self.memory.async_transfer_to_vram(self.layers[last_idx]['id'], 0)
        
        grad_output = None
        
        for i in range(last_idx, -1, -1):
            slot = (last_idx - i) % 2
            next_slot = (last_idx - i + 1) % 2
            
            flat_vram = self.memory.get_vram_flat_buffer(slot)
            
            disk_thread = None
            if i - 1 >= 0:
                if is_streaming:
                    self.memory.async_transfer_to_vram(self.layers[i-1]['id'], next_slot, ram_slot=next_slot)
                    if i - 2 >= 0:
                        disk_thread = threading.Thread(target=self.memory.prefetch_to_ram, args=(self.layers[i-2]['id'], slot))
                        disk_thread.start()
                else:
                    self.memory.async_transfer_to_vram(self.layers[i-1]['id'], next_slot)
            
            layer_module = self.adapter.construct_layer_module(self.layers[i]['id'], flat_vram, self.lora_manager)
            layer_input = boundary_activations[i]
            if layer_input.dtype.is_floating_point:
                layer_input.requires_grad_(True)
            
            output = self.adapter.forward_layer(layer_module, layer_input, gradient_checkpointing=self.config.gradient_checkpointing)
            
            if i == last_idx:
                if labels is not None:
                    # Real Causal LM Loss
                    # Shift so that tokens < n predict n
                    shift_logits = output[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_val = loss.item()
                else:
                    loss = output.mean() # Dummy
                
                loss.backward()
                grad_output = layer_input.grad
            else:
                if isinstance(output, tuple):
                    output[0].backward(grad_output)
                else:
                    output.backward(grad_output)
                grad_output = layer_input.grad
            
            if disk_thread: disk_thread.join()
            if hasattr(self.adapter, "release_layer_module"):
                self.adapter.release_layer_module(layer_module)
            del layer_module

        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return logits, loss_val