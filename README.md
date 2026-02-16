# LEMA: Layer-wise Efficient Memory Abstraction

## Fine-tuned model (PoC)

I've successfully used LEMA to fine-tune on a custom chat tempalte. While it successfully learned the new vocabulary and special tags, it has not yet mastered the logical structure or grammar of the custom template. You can find the model over [Here](https://huggingface.co/Pomilon/LEMA-llama-2-7b).

**Virtualize GPU VRAM for LLM Fine-Tuning**

LEMA is a specialized framework designed to facilitate the fine-tuning of Large Language Models (LLMs) on hardware where model size exceeds available VRAM. By treating model weights as addressable binary segments and implementing a **Triple-Buffer Strategy**, LEMA allows training 7B+ models on GPUs with as little as 16GB VRAM.

## Key Performance (Tesla P100 - 16GB)

| Model | Standard PEFT VRAM | LEMA VRAM | Savings | Status (Fine-Tuning) |
| :--- | :--- | :--- | :--- | :--- |
| **TinyLlama 1.1B** | 2.67 GB | **2.12 GB** | **20.5%** | **Stable** |
| **SmolLM2 1.7B** | 3.88 GB | **3.20 GB** | **17.6%** | **Stable** |
| **Llama-2 7B** | 13.99 GB* | **5.90 GB** | **~58%** | **LEMA Recommended** |

![VRAM Benchmark](docs/assets/vram_benchmark.png)

![Speed Benchmark](docs/assets/speed_benchmark.png)

*\*Note: At sequence length 128, Standard PEFT narrowly fits in 16GB VRAM. However, increasing the workload to a standard sequence length of 512 causes an immediate **Out-Of-Memory (OOM)** crash. LEMA maintains a consistent ~6GB footprint even as sequence length scales, providing over **10GB of headroom** for activations and larger batches.*

### The Headroom Advantage

The primary value of LEMA is not just "fitting" the model, but providing the **computational headroom** necessary for real-world training. On a 16GB GPU:

- **Standard PEFT**: Operating at ~88% VRAM capacity just to load the model and run a minimal step. Zero room for longer contexts or higher batch sizes.
- **LEMA**: Operating at ~37% VRAM capacity. Allows for significantly larger sequence lengths, higher batch sizes, or even larger models (13B+) on the same hardware

## Core Features

- **Binary Indexed Engagement (GBI)**: Zero-copy mapping of `.safetensors` files using `mmap`.
- **Triple-Buffer Pipeline**: Pipelined data movement (Disk -> RAM -> VRAM) to hide PCIe latency.
- **High-Level API**: Simplified `LemaModel` and `LemaTrainer` interfaces for fast integration.
- **Automatic Checkpointing**: Built-in interval-based saving of LoRA adapters and optimizer states.

## Installation

```bash
git clone https://github.com/Pomilon/LEMA.git
cd LEMA
pip install -e .
```

## Quick Start

```python
import torch
from lema import LemaConfig, LemaModel, MemoryStrategy

# 1. Configuration
config = LemaConfig(
    model_name_or_path="NousResearch/Llama-2-7b-hf",
    gbi_path="llama2_7b.safetensors", # Single monolithic safetensors file
    strategy=MemoryStrategy.STREAMING,
    lora_rank=16,
    gradient_checkpointing=True
)

# 2. Initialize Model & Trainer
model = LemaModel(config)
model.initialize_lora() # Pre-initialize adapters

optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-4)
trainer = model.get_trainer(optimizer)

# 3. Train
input_ids = torch.randint(0, 32000, (1, 512)).cuda()
logits, loss = trainer.train_step(input_ids, labels=input_ids)
```

## Documentation

- [**User Guide**](docs/USER_GUIDE.md): Model preparation, conversion, and tips.
- [**API Reference**](docs/API_REFERENCE.md): Detailed class and method specifications.
- [**Architecture**](docs/ARCHITECTURE.md): Deep dive into the memory pipeline and LEMA-loop.

## Future Roadmap

While LEMA v1.0 is stable and functional for 7B fine-tuning, I aim to significantly reduce the streaming overhead and expand compatibility.

- **C++/CUDA Backend**: I plan to move the `TripleBufferManager` and memory streaming logic from Python to a C++ extension or custom CUDA kernels to bypass the GIL and reduce overhead to the theoretical minimum (~1.1x).
- **Library Integration**: I am working toward deeper integration with Hugging Face `Trainer` and `Accelerate` for seamless usage in existing pipelines.
- **Quantization Support**: I intend to implement native support for 8-bit and 4-bit loading within the streaming pipeline for even lower memory footprints.
- **Model Support**: I am expanding support beyond Llama and GPT-2 to include Mistral, Mixtral (MoE), and other architectures.

## License

MIT License - Copyright (c) 2026 Pomilon
