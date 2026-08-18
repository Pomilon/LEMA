# LEMA Architecture

This document describes the internal mechanics of the Layer-wise Efficient Memory Abstraction (LEMA) framework.

## The Problem: The VRAM Wall
Standard fine-tuning (even with PEFT/LoRA) requires the entire model weights to be resident in VRAM. For a Llama-2 7B model in FP16, this is ~14GB. Adding optimizer states and activations quickly exceeds the capacity of consumer GPUs (e.g., 16GB).

## The LEMA Solution: Virtualization
LEMA treats GPU VRAM not as a static storage for the model, but as a **dynamic cache** for execution.

### 1. The Triple-Buffer Strategy
LEMA hides data transfer latency by pipelining movements across three memory tiers:

1.  **Storage (NVMe)**: Weights reside in `.safetensors` files. Accessed via `mmap` (Zero-copy).
2.  **System RAM (Pinned)**: Acting as a "Prefetch Buffer". Pinned memory ensures high-speed Host-to-Device (H2D) transfers.
3.  **VRAM (Execution)**: Divided into two "Slots" (Active and Prefetch).

### 2. The Execution Pipeline
While the GPU is computing Layer $N$ in Slot A, LEMA is:
-   Asynchronously transferring Layer $N+1$ from RAM to Slot B (VRAM).
-   Loading Layer $N+2$ from Disk to RAM (Staging).

When Layer $N$ finishes, the slots swap instantly.

### 3. The LEMA-Loop (Training Logic)

#### Forward Pass
-   Model is executed layer-by-layer.
-   Only "Boundary Activations" (the output of each layer) are stored in VRAM.
-   Intermediate activations are discarded.

#### Backward Pass
-   LEMA traverses the layers in reverse.
-   For each layer:
    1.  The weights are swapped back into VRAM.
    2.  The layer's forward pass is **re-executed** (Segmented Gradient Checkpointing) using the stored boundary activations.
    3.  Gradients are calculated for the LoRA adapters.
    4.  Optimizer states for those specific adapters are updated.

### 4. GBI (Global Binary Index)
LEMA uses a specialized indexer to bypass standard PyTorch/Pickle deserialization. By reading the `.safetensors` header, LEMA knows the exact byte offsets for every parameter, allowing it to "slice" the file and load only the parameters needed for the current layer module.

## Selective Full Fine-Tuning

LoRA keeps the base weights frozen and trains small adapters. **Selective full-FT** trains the real weights instead, without giving up LEMA's virtualization: optimizer states and gradient accumulation are fp32 and live in **pinned RAM** (with an mmap disk fallback), so the GPU never holds more than the active layer — the same trick used for weights.

### Selection
A `FullFTManager` resolves a **selection triple** at construction:
- `trainable_modules` — suffix patterns matched against safetensors param names (`"q_proj"` matches `model.layers.7.self_attn.q_proj.weight`).
- `trainable_layers` — `"last:K"`, `"first:K"`, explicit layer IDs, `"emb"`, `"head"`. Empty = all.
- Empty modules + empty layers ⇒ **whole model** (LOMO-style full FT).

The result is `selected: {layer_id: [param_names]}`. When the model config ties word embeddings, `lm_head.weight` is excluded from selection (it aliases the embedding weight — training both as separate copies would diverge and double RAM).

### RAM-resident true weights
For every selected `(layer_id, param_name)` the manager loads one copy into RAM: the **true weight** (model dtype) plus its **original** snapshot. Layer modules are transient (recreated/recycled every step), so a normal PyTorch optimizer cannot hold state across steps. Instead the manager holds stable identities keyed by `(layer_id, name)` and performs a **per-layer custom AdamW step** (`step_layer`):

```
per layer, at the accumulation boundary:
  stream Adam moments (fp32) RAM → VRAM state-slot
  apply the update to the layer's true weight
  stream moments + weight back to RAM
  zero the layer's gradient accumulator
```

### Gradient accumulation
- fp32 accumulators, one per trainable layer.
- RAM backend by default; `grad_accum_backend="disk"` (or `"auto"` when accumulators exceed the RAM budget) stores them as mmap'd `grad_accum/grad_acc_<layer>.bin` files, so multi-GB accumulators don't consume RAM.
- The boundary counter (`accumulation_step`) lives on the manager, not the trainer — trainers are recreated per call but training state persists on the model.

### Delta checkpoints
Checkpoints store `updated − original` for each selected tensor in **fp32** (`delta.safetensors` + `delta.index.json`) plus optional optimizer state (`optimizer_fullft.bin`). Restore computes `original + delta` in fp32 and casts once, so round-trips are exact even when weights are stored fp16/bf16. `merge_delta(base, delta, out)` produces a servable full model by adding deltas back into the base file.

## Performance Trade-offs
-   **VRAM Efficiency**: ~50-70% reduction for 7B+ models.
-   **Compute Overhead**: 1.5x - 3.5x slowdown compared to fully resident training, depending on PCIe bandwidth and disk speed.
-   **System RAM**: 
    -   **STREAMING Mode**: ~2.5 GB (Pinned buffers).
    -   **RESIDENT Mode**: Requires space equal to the model size.
