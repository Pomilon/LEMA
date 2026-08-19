# Benchmark Results

All benchmarks run on **Tesla T4 (14.6 GB VRAM)** via Kaggle notebook. Run the notebook at `examples/kaggle/benchmark.ipynb`.

## LEMA Configuration

All LEMA benchmarks use the following configuration unless otherwise noted:

| Parameter | Value |
|---|---|
| `backend` | `"auto"` (C++ compiled, Python fallback if CUDA unavailable) |
| `strategy` | `STREAMING` (weights loaded on-demand from disk) |
| `max_ram_gb` | auto (70% of system RAM, ~21.9 GB) |
| `max_vram_gb` | auto (60% of GPU VRAM, ~8.7 GB) |
| `vram_fraction` | 0.8 (PyTorch capped to 80% of total VRAM) |
| `prefetch_distance` | 2 (2 layers ahead prefetched) |
| `lora_rank` | 16 |
| `lora_alpha` | 32 |
| `lora_target_modules` | `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` |
| `gradient_checkpointing` | `True` |
| `dtype` | auto (bf16 for TinyLlama, fp16 for Llama-2 7B) |
| Module pool | 3 slots (1 embedding, 1 decoder block, 1 head) — recycled per step |
| VRAM buffers | 2 flat buffers (double-buffered), size = max layer params |
| RAM buffers | 2 streaming slots (1000, 1001), pinned memory |

TinyLlama 1.1B has 24 decoder layers (~2.05 GB bf16). Llama-2 7B has 34 decoder layers (~13.5 GB fp16). STREAMING keeps only 2 layers in VRAM at any time — the current compute layer and the next prefetched layer.

## VRAM Comparison (bs=1, seq=512)

| Model | PEFT VRAM | LEMA VRAM | Savings |
|---|---|---|---|
| TinyLlama 1.1B | 5.0 GB | 1.4 GB | 72% |
| Llama-2 7B | OOM | 3.2 GB | LEMA only |

![VRAM Comparison](assets/vram_benchmark.png)

## Training Speed (bs=1, seq=512)

| Model | PEFT | LEMA | Ratio |
|---|---|---|---|
| TinyLlama 1.1B | 310 ms | 2297 ms | 7.4x |
| Llama-2 7B | OOM | 3719 ms | -- |

LEMA's triple-buffer pipeline (disk -> RAM -> VRAM) adds throughput overhead. On models that fit in VRAM, PEFT is faster. On models that don't, LEMA makes training possible where PEFT OOMs.

![Speed Comparison](assets/speed_benchmark.png)

## C++ Backend Performance

| Operation | Python | C++ | Improvement |
|---|---|---|---|
| Pack (memcpy, 8 layers) | 1.33 ms | 0.64 ms | 52% faster |
| Transfer + sync (32 MB) | 2.84 ms | 2.84 ms | ~0% |
| Transfer + sync (2 GB) | 180.0 ms | 179.5 ms | ~0% |
| End-to-end train step | 227.1 ms | 203.0 ms | 10.6% faster |

C++ improvement is limited to CPU-side packing (memcpy). GPU transfers are PCIe-bound and see no benefit.

![C++ Backend](assets/cpp_benchmark.png)

## Memory Strategy: RESIDENT vs STREAMING

| Strategy | Load Time | Step Time | VRAM |
|---|---|---|---|
| RESIDENT | 6.6 s | 2374 ms | 1.4 GB |
| STREAMING | 2.3 s | 2465 ms | 1.4 GB |

(TinyLlama 1.1B -- VRAM difference is negligible at this model size. On 7B+, STREAMING saves significant VRAM.)

## VRAM Stability (60 steps on Llama-2 7B)

| Step | VRAM | Loss |
|---|---|---|
| 1 | 2.87 GB | 10.77 |
| 10 | 3.12 GB | 9.52 |
| 20 | 3.12 GB | 7.59 |
| 30 | 3.12 GB | 5.68 |
| 40 | 3.12 GB | 4.95 |
| 50 | 3.12 GB | 4.71 |
| 60 | 3.12 GB | 4.57 |

VRAM settled at 3.12 GB after initial load and remained flat. Drift over 60 steps: +0.24 GB (all from step 1 cold start). Training uses SGD and converges stably.

![VRAM Stability](assets/vram_stability.png)

## Long Sequence Headroom (Llama-2 7B)

| Seq Len | PEFT | LEMA VRAM | LEMA Step Time |
|---|---|---|---|
| 512 | OOM | 3.2 GB | 3662 ms |
| 1024 | OOM | 4.0 GB | 4975 ms |
| 2048 | OOM | 6.3 GB | 8414 ms |
| 4096 | OOM | OOM | -- |

LEMA reaches 2048 vs PEFT's 512 on a 14.6 GB T4 -- a 4x improvement in usable sequence length.

![Long Sequence](assets/longseq_vram.png)

## VRAM Scaling (TinyLlama 1.1B, full matrix)

![Scaling Heatmap](assets/scaling_heatmap.png)

| Seq | Batch | PEFT ms | PEFT VRAM | LEMA ms | LEMA VRAM |
|---|---|---|---|---|---|
| 128 | 1 | 205 | 2.8 GB | 935 | 1.2 GB |
| 128 | 4 | 235 | 4.5 GB | 2243 | 1.3 GB |
| 128 | 8 | 458 | 6.8 GB | 4642 | 1.6 GB |
| 256 | 8 | 944 | 11.1 GB | 9776 | 2.2 GB |
| 512 | 1 | 310 | 5.0 GB | 2297 | 1.4 GB |
| 512 | 4 | 1091 | 12.9 GB | 9825 | 2.4 GB |
| 512 | 8 | OOM | -- | 21087 | 3.5 GB |

## CPU Offload Comparison (TinyLlama 1.1B)

| Method | Step Time | VRAM |
|---|---|---|
| HF CPU Offload | 298 ms | 2.4 GB |
| LEMA Streaming | 2390 ms | 1.4 GB |

CPU offload is faster when the model fits in VRAM. LEMA's streaming advantage appears at 7B+ scale where repeated CPU-GPU transfers dominate.

## Selective Full Fine-Tuning (TinyLlama 1.1B)

Selection: q/k/v/o projections of the last 4 decoder layers (37.7M params, ~3.4% of the 1.1B model), trained under `STREAMING` with fp32 states in RAM. Verified in the Kaggle notebook's full-FT demo cells (all modes) on a T4.

| Metric | Value |
|---|---|
| Selected params | 37,748,736 (~3.4% of 1.1B; GQA: 4 KV heads vs 32 Q heads) |
| fp32 state footprint | ~0.151 GB (true weights + Adam moments) |
| VRAM (constant) | **0.75 GB** |
| RAM | ~4.9 GB |
| Step time | ~1.1–1.6 s (seq=64) |
| Loss (6 steps) | 13.56 → 4.69 |
| Resume (delta + optimizer restore) | weights match exactly; training continues (loss 4.69 → 3.77) |

Other verified full-FT modes (tiny GPT-2, CUDA): whole-model (LOMO-style), emb/head-only, disk gradient-accumulation backend, gradient-accumulation boundaries (`accum_steps=2` steps only at the boundary), and checkpoint → `merge_delta` → restore → resume.

The full-FT selection triple (`trainable_modules` × `trainable_layers`) supports arbitrary subsets up to whole-model fine-tuning; VRAM stays at the single-layer working set while RAM scales with the selected parameter count.

## TensorStore Umbrella (T4)

The unified streaming core: weights, optimizer states, gradient accumulators, and KV chunks all live in one `TensorStore` with per-kind VRAM budgets (tuner proposes, explicit overrides win) and a target-based `BudgetEngine`. Verified in the Kaggle notebook's TensorStore demo cell on a T4 (seq=64 > one 16-token KV chunk).

| Check | Result |
|---|---|
| Chunked attention (32 keys, 8/chunk) vs full | max diff **1.79e-07** (exact) |
| Chunked long-context forward (GPT-2, seq=64) vs full | max diff **1.19e-07** (exact) |
| KV-cached generation vs old O(n²) re-forward | identical token output |
| Budget tuning | per-kind VRAM: weights/kv/opt/grad = 2.0 GB each (8 GB budget), target met |
| Full unit suite | 76/76 passing (local + Kaggle) |

KV-cached generation (`generate_kv`) streams per-layer KV chunks through the store, enabling long-context inference without holding the full KV cache in VRAM; chunked attention is exact (online softmax in fp32), not an approximation.
