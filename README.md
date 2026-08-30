# LEMA: Layer-wise Efficient Memory Abstraction

**Virtualize GPU VRAM for LLM Fine-Tuning**

LEMA is a framework for fine-tuning Large Language Models on GPUs where model size exceeds available VRAM. By treating model weights as addressable binary segments and implementing a **Triple-Buffer Strategy** (Disk → RAM → VRAM) with async prefetching, LEMA allows training 7B+ models on GPUs with as little as 16GB VRAM.

## Key Performance (Tesla T4 — 14.6 GB)

| Model | Config | PEFT VRAM | LEMA VRAM | LEMA Step |
|---|---|---|---|---|
| **TinyLlama 1.1B** | bs=1, seq=512 | 5.0 GB | **1.4 GB** | 2297 ms |
| **TinyLlama 1.1B** | bs=8, seq=512 | OOM | **3.5 GB** | 21087 ms |
| **Llama-2 7B** | bs=1, seq=128 | OOM | **2.9 GB** | 3920 ms |
| **Llama-2 7B** | bs=2, seq=512 | OOM | **3.8 GB** | 4920 ms |
| **Llama-2 7B** | bs=8, seq=512 | OOM | **6.6 GB** | 12816 ms |
| **Llama-2 7B** | seq=2048, bs=1 | OOM | **6.3 GB** | 8414 ms |

PEFT OOMs on Llama-2 7B at every configuration on a 14.6 GB T4. LEMA trains at **2.9–6.6 GB** — under half the VRAM — across all batch sizes and up to 2048 sequence length.

![VRAM](docs/assets/vram_benchmark.png) | ![Speed](docs/assets/speed_benchmark.png)
:---: | :---:
VRAM Usage (bs=1, seq=512) | Training Speed (bs=1, seq=512)

[Full benchmark results](docs/BENCHMARK_RESULTS.md) — VRAM stability, long sequence headroom, C++ backend comparison, and full scaling matrix.

## Fine-tuned Model (PoC)

Successfully fine-tuned `NousResearch/Llama-2-7b-hf` on a custom chat template using an earlier version of LEMA. Available at [huggingface.co/Pomilon/LEMA-llama-2-7b](https://huggingface.co/Pomilon/LEMA-llama-2-7b).

## Features

- **Triple-Buffer Pipeline**: Disk → pinned RAM → VRAM with async prefetching hides PCIe latency.
- **Multi-file Support**: Works directly with HuggingFace sharded `.safetensors` (no longer requires monolithic conversion).
- **C++/Python Backend**: Explicit toggle (`backend="auto" | "cpp" | "python"`).
- **Auto Flight Check**: Benchmarks your hardware and auto-tunes `prefetch_distance` and strategy.
- **5 Model Architectures**: Llama, Mistral, Mixtral (MoE), GPT-2, LFM2 (MoE).
- **Selective Full Fine-Tuning**: Train real model weights (no LoRA) on any selection — attention projections of the last K layers, embeddings, or the entire model — with fp32 optimizer states virtualized into RAM (mmap fallback) the same way weights are.
- **Automatic Checkpointing**: Interval-based saving of LoRA adapters or full-FT delta + optimizer states.
- **Module Pool**: Sliding-window module recycling keeps VRAM constant regardless of model depth.

## Installation

```bash
git clone https://github.com/Pomilon/LEMA.git
cd LEMA
pip install -e .                    # with C++ extension (if CUDA + nvcc available)
pip install -e . --no-cuda-ext     # pure Python only
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, CUDA-capable GPU.

## Quick Start

```python
import torch
from lema import LemaConfig, LemaModel, MemoryStrategy

config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    strategy=MemoryStrategy.STREAMING,
    backend="auto",              # "auto" | "cpp" | "python"
    lora_rank=16,
    gradient_checkpointing=True,
)

model = LemaModel(config)        # auto-downloads from HF Hub if needed
model.initialize_lora()

optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
trainer = model.get_trainer(optimizer)

input_ids = torch.randint(0, 32000, (1, 512)).cuda()
logits, loss = trainer.train_step(input_ids, labels=input_ids)
```

## Selective Full Fine-Tuning

Instead of LoRA adapters, LEMA can train the real model weights directly — a subset of your choosing — using the same VRAM-virtualizing pipeline. Optimizer states and gradient accumulation are fp32 and live in RAM (with an optional mmap disk backend), so even whole-model training stays off-VRAM.

```python
from lema import LemaConfig, LemaModel, MemoryStrategy

config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    strategy=MemoryStrategy.STREAMING,
    training_mode="selective_full",      # instead of LoRA
    trainable_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    trainable_layers=["last:4"],         # last 4 decoder layers
    learning_rate=1e-4,
    save_steps=500,
    output_dir="checkpoints",
)

model = LemaModel(config)                 # no initialize_lora() needed
trainer = model.get_trainer()             # optimizer handled internally (per-layer AdamW)

logits, loss = trainer.train_step(input_ids, labels=input_ids)
```

**Selection syntax** (`trainable_modules` × `trainable_layers`):

| `trainable_modules` | `trainable_layers` | Result |
|---|---|---|
| `["q_proj","k_proj","v_proj","o_proj"]` | `["last:4"]` | Attention projections of the last 4 layers |
| `["c_attn"]` | `["first:2"]` | GPT-2 attention of the first 2 layers |
| `[]` | `["emb","head"]` | Embeddings + LM head only |
| `[]` | `[]` | **Whole model** (all weights, LOMO-style) |

`trainable_modules` entries are suffix matches against parameter names; `trainable_layers` accepts `"last:K"`, `"first:K"`, explicit layer IDs, `"emb"`, and `"head"`.

**Checkpoints & serving:** training saves a small fp32 **delta** (`updated − original`) plus optimizer state. Restore with `LemaModel.from_pretrained` (weights + optimizer), or produce a servable full model with `merge_delta`:

```python
from lema._utils._conversion import merge_delta
merge_delta(base_safetensors, "checkpoints/checkpoint-500/delta.safetensors", "merged.safetensors")
```

## TensorStore: Unified Streaming Core

All streaming — weights, optimizer states, gradient accumulators, and the KV cache — runs through one `TensorStore`: a slot-pool address space of tensor streams, each with a configurable residency policy. VRAM is split per-kind by a target-based `BudgetEngine` (tuner proposes, explicit overrides win), so you control how much of each kind stays in VRAM vs RAM vs disk.

```python
config = LemaConfig(
    model_name_or_path="gpt2",
    strategy=MemoryStrategy.STREAMING,
    weights_vram="auto",      # "auto" | fraction e.g. "0.3" | absolute e.g. "4.0GB"
    kv_vram="2.0GB",
    target_step_time_ms=250,  # budget engine minimizes VRAM to meet this
    kv_chunk_size=8192,       # tokens per KV chunk
)
```

**Long context & KV-cached generation:** when the sequence exceeds one KV chunk, attention runs chunk-by-chunk (exact, fp32 online softmax) with KV streamed per-layer through the store — enabling 128k+ context on consumer VRAM. `generate_kv` uses a real KV cache instead of the O(n²) re-forward loop:

```python
model.generate_kv(prompt, tokenizer, max_new_tokens=200, kv_chunk_size=8192)
```

Chunked attention and KV-cached generation are supported on **all adapters**: GPT-2, Llama, Mistral, Mixtral, and LFM2 (MoE). Generation runs in eval mode so outputs are deterministic.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for the full design and T4 validation.

## Quantization

All four memory classes LEMA moves — streamed weights, full-FT optimizer states, gradient accumulators, and the KV cache — can be quantized per-class via four config fields:

| Field | Allowed values | Savings vs baseline |
|---|---|---|
| `weights_bits` | `None`/`0` (off), `8`, `4` | int8 ≈ 2×; int4 ≈ 4× on weight disk/RAM (fp16 → 8/4-bit) |
| `opt_state_bits` | `None`/`0` (off), `8` | ≈ 4× on Adam moments (fp32 → int8) |
| `grad_acc_bits` | `None`/`0` (off), `8` | ≈ 4× on gradient accumulators (fp32 → int8) |
| `kv_bits` | `None`/`0` (off), `8` | ≈ 2× on the KV cache (fp16 → int8) |

```python
config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    weights_bits=8,        # int8 streamed weights (4 = int4, weights only)
    opt_state_bits=8,      # int8 Adam moments in full-FT
    grad_acc_bits=8,       # int8 gradient accumulators
    kv_bits=8,             # int8 KV cache chunks
    quant_backend="auto",  # auto | custom | torchao | quanto | bitsandbytes
)
```

**Backends (optional installs — pull only what you need):**

| Backend | Install | Hardware | Notes |
|---|---|---|---|
| `custom` | built-in | CPU + CUDA | hand-rolled W8A8 (AVX2 / DP4A) — default fallback |
| `torchao` | `pip install lema[torchao]` | CPU + CUDA + MPS + XPU | PyTorch-native, recommended |
| `quanto` | `pip install lema[quanto]` | CPU + CUDA + MPS | `optimum-quanto`, lightweight |
| `bitsandbytes` | `pip install lema[bitsandbytes]` | CUDA only | `LLM.int8` (bits=8); NF4 is not engine-compatible |
| `auto` | `pip install lema[quant]` → torchao+quanto | auto-detect | picks first available (`torchao` > `quanto` > `bitsandbytes` > `custom`) |

`pip install lema[all-quant]` pulls all three; `pip install lema` alone keeps the `custom` fallback with no extra deps.

`quant_backend` applies to **on-the-fly streamed weight quantization** only (per-row int8 for bits=8 — GPU parity: custom ≡ torchao, rel-err 0.0039). Optimizer states, gradient accumulators, KV cache, and pre-quantized checkpoints always use the built-in `custom` format. All backends emit the engine's int8 + per-row-scale wire format. A backend that cannot serve the requested bit width (e.g. `quanto`/`bitsandbytes` at bits=4) logs a warning and falls back to `custom`; unexpected backend errors also warn instead of failing silently.

With native W8A8 kernels present (built automatically: AVX2 on CPU, DP4A on CUDA), `weights_bits=8` goes further than storage savings — llama decoder layers are served as **raw int8 buffers and consumed directly by int8×int8→int32 GEMM kernels** with fused scale epilogues. No fp32 dequant slot is produced on the hot path, so the streamed weight footprint halves on disk, RAM, PCIe, *and* VRAM simultaneously. Full-FT optimizer states and accumulators stay quantized in RAM (including the mmap disk backend) and are dequantized only inside the per-layer AdamW step. The KV cache stores int8 values with a dynamic per-chunk scale.

**Trade-offs:** on tiny models int8 training output stays within ~1% of fp16 (quantized weight relative error < 1e-2 — see `tests/test_quant_streaming.py`, `tests/test_w8a8_llama_forward.py`); on real models the quantized-weight error shifts the loss by a fraction of a nat at load. Without the native kernels (or for int4, full-FT-selected layers during training, embedding/head layers, and generation-mode KV paths) weights fall back to dequantize-at-consumption: disk/RAM still shrink but compute runs at full precision. LoRA adapters are not yet supported on int8 buffers — such constructions fail loudly rather than silently dropping adapters.

### Pre-quantized checkpoints (recommended for streaming full-FT)

On-the-fly weight quantization runs on the CPU at pack time. On CPU-starved hosts (e.g. 2-vCPU Kaggle) it dominates streaming full-FT training: ~83–85% of step time is spent waiting for the quantizer (`pack_wait`), making W8A8 **15–16× slower than fp16** (15.0 s/step vs 1.3 s/step fp16 on TinyLlama-1.1B / T4).

Pre-quantize the checkpoint offline instead, so packing becomes a raw byte copy with zero CPU quantize:

```bash
python tools/quantize_checkpoint.py <model_dir> --bits 8 --output_dir <out_dir>   # or --bits 4
```

Each bit width gets its own checkpoint (int4 stores packed uint8 + logical shapes in safetensors metadata). The engine detects int8/uint8 + `.scale` siblings per tensor, stages raw bytes into RAM (uint8 staging, exact transfer bytes), and dequantizes at consumption. Measured on T4 / TinyLlama-1.1B:

| Mode | step time | vs fp16 | pack_wait |
|---|---|---|---|
| fp16 | ~0.8–1.3 s | 1.0× | ~15% |
| int8 W8A8 on-the-fly | ~15 s | 0.06× | ~88% |
| int8 W8A8 pre-quantized | ~0.8–1.3 s | ≈1.0× | <1% |
| int4 W4A16 pre-quantized | ~1.2–1.6 s | ~0.7× | <1% |

Pre-quantized int8 full-FT learns at the same rate as fp16 (Δloss 1.06 vs 1.06 over two steps) while saving ~520 MB VRAM. All quantization levels transfer exact payload bytes over PCIe (2×/4× reduction, verified 9.6 GB/s ≈ wire saturation).

## Documentation

- [**Benchmark Results**](docs/BENCHMARK_RESULTS.md): Full VRAM and throughput comparison.
- [**API Reference**](docs/API_REFERENCE.md): Complete class and method specifications.
- [**User Guide**](docs/USER_GUIDE.md): Model preparation, conversion, and tips.
- [**Architecture**](docs/ARCHITECTURE.md): Deep dive into the memory pipeline.

## License

MIT License — Copyright (c) 2026 Pomilon
