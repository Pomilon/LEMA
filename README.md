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
- **Automatic Checkpointing**: Interval-based saving of LoRA adapters and optimizer states.
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

## Documentation

- [**Benchmark Results**](docs/BENCHMARK_RESULTS.md): Full VRAM and throughput comparison.
- [**API Reference**](docs/API_REFERENCE.md): Complete class and method specifications.
- [**User Guide**](docs/USER_GUIDE.md): Model preparation, conversion, and tips.
- [**Architecture**](docs/ARCHITECTURE.md): Deep dive into the memory pipeline.

## License

MIT License — Copyright (c) 2026 Pomilon
