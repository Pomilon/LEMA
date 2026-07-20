from __future__ import annotations

import torch
import torch.nn.functional as F
import os
from typing import Any
from tqdm import tqdm

from ._memory import TripleBufferManager
from .adapters._base import LemaModelAdapter
from ._config import LemaConfig, MemoryStrategy
from ._utils._logger import logger

PHASE_PREFETCH = 1
PHASE_FORWARD = 2
PHASE_BACKWARD = 3

class LemaTrainer:
    """
    High-level trainer for LEMA models.
    Handles the asynchronous memory orchestration for forward and backward passes.
    """
    def __init__(self, 
                 config: LemaConfig,
                 model_adapter: LemaModelAdapter, 
                 gbi: Any, 
                 lora_manager: Any = None, 
                 optimizer: torch.optim.Optimizer | None = None,
                 memory_manager: TripleBufferManager | None = None):
        
        self.config = config
        self.adapter = model_adapter
        self.gbi = gbi
        self.device = config.device
        self.strategy = config.strategy
        
        # Use provided memory manager or create a new one
        if memory_manager is not None:
            self.memory = memory_manager
        else:
            self.memory = TripleBufferManager(gbi, model_adapter, config=self.config)
        
        self.layers = self.adapter.get_layer_metadata()
        self.lora_manager = lora_manager
        self.optimizer = optimizer
        self.global_step = 0
        self.accumulation_step = 0

    def close(self):
        """Releases the memory manager."""
        if hasattr(self, "memory") and self.memory is not None:
            self.memory.close()

    def save_checkpoint(self, save_directory: str):
        """Saves the model state (config + LoRA) and optionally optimizer state."""
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if self.lora_manager:
            self.lora_manager.save_pretrained(save_directory)
        
        if self.optimizer:
            torch.save(self.optimizer.state_dict(), os.path.join(save_directory, "optimizer.bin"))
        logger.info(f"LEMA: Checkpoint saved to {save_directory}")

    def train_step(self, inputs: Any, labels: torch.Tensor | None = None):
        """
        Executes one training step with the sliding-window pipeline.
        Forward: no_grad (saves inputs, no autograd graph). 
        Backward: re-forwards each layer with grad for autograd.
        Pool recycles 2-3 modules — VRAM stays constant regardless of model size.
        """
        dist = self.config.prefetch_distance
        boundary_activations: list[torch.Tensor] = []

        # Phase 1: Initial Prefetch
        for j in range(min(dist, len(self.layers))):
            self.memory.prefetch_to_ram(self.layers[j]['id'], slot=j % 2)
        self.memory.async_transfer_to_vram(self.layers[0]['id'], vram_slot=0, ram_slot=0)

        hidden_states = inputs

        # Phase 2: Forward Pass
        with torch.no_grad():
            for i, layer_meta in enumerate(self.layers):
                slot = i % 2
                next_slot = (i + 1) % 2

                flat_vram = self.memory.get_vram_flat_buffer(slot)

                if i + 1 < len(self.layers):
                    self.memory.wait_prefetch(next_slot)
                    self.memory.async_transfer_to_vram(self.layers[i+1]['id'], vram_slot=next_slot, ram_slot=next_slot)
                    prefetch_idx = i + dist
                    if prefetch_idx < len(self.layers):
                        self.memory.prefetch_to_ram_async(self.layers[prefetch_idx]['id'], prefetch_idx % 2)

                layer_module = self.adapter.construct_layer_module(layer_meta['id'], flat_vram, self.lora_manager)

                # Save input for backward (detached from graph)
                current_input = hidden_states.detach() if isinstance(hidden_states, torch.Tensor) else hidden_states[0].detach()
                boundary_activations.append(current_input)

                hidden_states = self.adapter.forward_layer(layer_module, hidden_states, gradient_checkpointing=False)

                self.adapter.release_layer_module(layer_module)
                del layer_module

        logits = hidden_states
        loss_val = None

        # Phase 3: Backward Pass
        if not torch.is_grad_enabled():
            return logits, None

        last_idx = len(self.layers) - 1

        for j in range(last_idx, max(-1, last_idx - dist), -1):
            self.memory.prefetch_to_ram(self.layers[j]['id'], slot=j % 2)
        self.memory.async_transfer_to_vram(self.layers[last_idx]['id'], vram_slot=0, ram_slot=last_idx % 2)

        grad_output = None
        for i in range(last_idx, -1, -1):
            slot = i % 2
            prev_slot = (i - 1) % 2

            flat_vram = self.memory.get_vram_flat_buffer(slot)
            layer_module = self.adapter.construct_layer_module(self.layers[i]['id'], flat_vram, self.lora_manager)

            if i - 1 >= 0:
                self.memory.wait_prefetch(prev_slot)
                self.memory.async_transfer_to_vram(self.layers[i-1]['id'], vram_slot=prev_slot, ram_slot=prev_slot)
                prefetch_idx = i - dist
                if prefetch_idx >= 0:
                    self.memory.prefetch_to_ram_async(self.layers[prefetch_idx]['id'], prefetch_idx % 2)

            layer_input = boundary_activations[i]
            if layer_input.dtype.is_floating_point:
                layer_input.requires_grad_(True)
            output = self.adapter.forward_layer(layer_module, layer_input, gradient_checkpointing=self.config.gradient_checkpointing)

            if i == last_idx:
                if labels is not None:
                    shift_logits = output[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss_val = loss.item()
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    loss = output.mean() / self.config.gradient_accumulation_steps
                    loss_val = loss.item()
                loss.backward()
                grad_output = layer_input.grad
            else:
                (output[0] if isinstance(output, tuple) else output).backward(grad_output)
                grad_output = layer_input.grad

            self.adapter.release_layer_module(layer_module)
            del layer_module

        self.accumulation_step += 1
        if self.optimizer and (self.accumulation_step % self.config.gradient_accumulation_steps == 0):
            torch.nn.utils.clip_grad_norm_(self.lora_manager.get_trainable_parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.global_step += 1

        if self.config.save_steps > 0 and self.global_step % (self.config.save_steps * self.config.gradient_accumulation_steps) == 0:
            checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
            self.save_checkpoint(checkpoint_path)

        return logits, loss_val

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader):
        """Executes a validation loop over the dataloader."""
        logger.info(f"LEMA: Starting evaluation on {len(dataloader)} batches...")
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            if isinstance(batch, dict):
                inputs = batch["input_ids"].to(self.device)
                labels = batch.get("labels", None)
                if labels is not None: labels = labels.to(self.device)
            else:
                inputs = batch.to(self.device)
                labels = None
                
            _, loss = self.train_step(inputs, labels=labels)
            if loss is not None:
                total_loss += loss
                
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        logger.info(f"LEMA: Evaluation complete. Avg Loss: {avg_loss:.4f}")
        return avg_loss
