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

## Performance Trade-offs
-   **VRAM Efficiency**: ~50-70% reduction for 7B+ models.
-   **Compute Overhead**: 1.2x - 1.8x slowdown compared to fully resident training, depending on PCIe bandwidth and disk speed.
-   **System RAM**: Requires space equal to the model size (or less if using aggressive disk streaming).
