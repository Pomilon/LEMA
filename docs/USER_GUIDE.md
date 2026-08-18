# LEMA User Guide

This guide covers common workflows for fine-tuning Large Language Models using LEMA on memory-constrained hardware.

## 1. Preparing Your Model

LEMA requires model weights in a single, non-sharded `.safetensors` format. We provide a utility to handle conversion and shared-weight breaking automatically.

### Recommended Conversion

```python
from lema._utils._model_utils import prepare_monolithic_safetensors

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

## 5. Selective Full Fine-Tuning

To train real model weights instead of LoRA adapters, set `training_mode="selective_full"` and describe the selection. LoRA and full-FT are exclusive — pick one per run.

```python
import torch
from lema import LemaConfig, LemaModel

# Train attention projections of the last 4 layers of a 7B model
config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    strategy=MemoryStrategy.STREAMING,
    training_mode="selective_full",
    trainable_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    trainable_layers=["last:4"],
    learning_rate=1e-4,
    weight_decay=0.01,
    grad_accum_backend="auto",   # "auto" | "ram" | "disk"
    save_optimizer=True,
    save_steps=500,
    output_dir="checkpoints",
)

model = LemaModel(config)
# No initialize_lora() in this mode. get_trainer() needs no external optimizer —
# a per-layer AdamW step runs internally on the manager's true weights.
trainer = model.get_trainer()

for batch in dataloader:
    logits, loss = trainer.train_step(batch['input_ids'], labels=batch['labels'])
    print(f"Loss: {loss}")

# Save a small delta checkpoint (+ optimizer state)
trainer.save_checkpoint("checkpoints/lema-7b-selective-full-ft")
```

### Choosing a selection

| Goal | `trainable_modules` | `trainable_layers` |
|---|---|---|
| Last-K layers, attention only (cheap, effective) | `["q_proj","k_proj","v_proj","o_proj"]` | `["last:4"]` |
| All decoder blocks | `[]` (all modules) | `["last:22"]` |
| Embeddings + LM head | `[]` | `["emb","head"]` |
| Whole model (LOMO-style) | `[]` | `[]` |

`trainable_modules` entries are suffix matches. `trainable_layers` accepts `"last:K"`, `"first:K"`, explicit IDs (e.g. `"7"`), `"emb"`, and `"head"`. An empty resolved selection raises a `ValueError`.

### Memory footprint

Selected weights, fp32 Adam moments (2×), and gradient accumulators live in RAM. For a selection of N parameters that is roughly **N × 8 bytes** (weights) + **N × 8 bytes** (moments) + **N × 4 bytes** (accumulators). On TinyLlama 1.1B, last-4-layers q/k/v/o (37.7M params ≈ 3.4%) fits in ~0.75 GB VRAM / ~5 GB RAM. When accumulators would exceed the RAM budget, `grad_accum_backend="auto"` switches to the mmap disk backend automatically.

### Serving a merged model

The checkpoint stores deltas (`updated − original`, fp32) — small and cheap. To produce a standalone servable model, merge them back into the base weights:

```python
from lema._utils._conversion import merge_delta

merge_delta(
    "llama2_7b.safetensors",                              # base model file
    "checkpoints/lema-7b-selective-full-ft/delta.safetensors",
    "merged_7b.safetensors",
)
```

To resume training, `LemaModel.from_pretrained("checkpoints/lema-7b-selective-full-ft")` restores both the deltas and the optimizer state.

## 6. Tips for Maximum Efficiency

1. **Gradient Checkpointing**: Always enable `gradient_checkpointing=True` for 7B+ models. This significantly reduces VRAM usage during the backward pass by not storing intermediate activations.
2. **Pinned Memory**: LEMA automatically uses pinned memory for transfers. Ensure your system has sufficient RAM available for the staging buffers (~2x the size of the largest layer).
3. **NVMe Storage**: When using `STREAMING` mode, placing your `.safetensors` file on an NVMe SSD will greatly reduce the "Streaming Overhead".
