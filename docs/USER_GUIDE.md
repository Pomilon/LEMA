# LEMA User Guide

This guide covers common workflows for fine-tuning Large Language Models using LEMA on memory-constrained hardware.

## 1. Preparing Your Model

LEMA requires model weights in a single, non-sharded `.safetensors` format. We provide a utility to handle conversion and shared-weight breaking automatically.

### Recommended Conversion

```python
from lema.utils.model_utils import prepare_monolithic_safetensors

# This handles downloading, shared-weight cloning, and monolithic saving
prepare_monolithic_safetensors(
    "NousResearch/Llama-2-7b-hf", 
    "llama2_7b.safetensors",
    device="auto" # Use 'auto' to save RAM during conversion if a GPU is available
)
```

## 2. Fine-Tuning Workflow

The standard workflow involves four steps: Configuration, Initialization, Training, and Saving.

### Basic Example

```python
import torch
from lema import LemaConfig, LemaModel, LemaTrainer

# 1. Setup Config
config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    gbi_path="llama2_7b.safetensors",
    lora_rank=16,
    gradient_checkpointing=True
)

# 2. Initialize
model = LemaModel(config)
model.initialize_lora() # Crucial for new models

# 3. Training
optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
trainer = model.get_trainer(optimizer)

for batch in dataloader:
    logits, loss = trainer.train_step(batch['input_ids'], labels=batch['labels'])
    print(f"Loss: {loss}")

# 4. Save
trainer.save_checkpoint("checkpoints/lema-llama-7b-v1")
```

## 3. Architecture Specifics

When using LEMA, ensure your `lora_target_modules` in `LemaConfig` match your model's architecture:
- **Llama**: `["q_proj", "v_proj", ...]` (Default)
- **GPT-2**: `["c_attn"]`

## 4. Memory Strategies

LEMA supports two primary strategies in `LemaConfig`:

- **`MemoryStrategy.STREAMING` (Default)**: 
    - **Path**: Disk -> Pinned RAM -> VRAM.
    - **Pros**: Lowest VRAM usage. Can fit models much larger than System RAM if needed (via `mmap`).
    - **Cons**: Higher latency due to PCIe/Disk bottleneck.
- **`MemoryStrategy.RESIDENT`**:
    - **Path**: RAM -> VRAM.
    - **Pros**: Faster than streaming. Model weights stay in RAM.
    - **Cons**: Requires enough System RAM to hold the full model weights (~14GB for a 7B FP16 model).

## 4. Tips for Maximum Efficiency

1. **Gradient Checkpointing**: Always enable `gradient_checkpointing=True` for 7B+ models. This significantly reduces VRAM usage during the backward pass by not storing intermediate activations.
2. **Pinned Memory**: LEMA automatically uses pinned memory for transfers. Ensure your system has sufficient RAM available for the staging buffers (~2x the size of the largest layer).
3. **NVMe Storage**: When using `STREAMING` mode, placing your `.safetensors` file on an NVMe SSD will greatly reduce the "Streaming Overhead".
