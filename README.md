# LEMA: Layer-wise Efficient Memory Abstraction

**Architectural Specification for VRAM-Efficient Model Fine-Tuning**

LEMA is a specialized framework designed to facilitate the fine-tuning of Large Language Models (LLMs) on hardware where model size exceeds available VRAM. Unlike standard frameworks that require the full model to be resident in GPU memory, LEMA treats the model as a collection of discrete, addressable binary segments. By implementing a virtualized memory abstraction layer, LEMA performs asynchronous pre-fetching of layers into VRAM, effectively trading PCIe bandwidth for memory headroom.

## Core Features

### 1. Binary Indexed Engagement
LEMA utilizes a **Global Binary Index (GBI)** to map `.safetensors` files directly into the process's virtual address space using `mmap`. This allows for zero-copy mapping and O(1) access to specific layer weights without full model deserialization.

### 2. Layer-wise Execution (Patchwork)
Instead of a monolithic `model.forward()`, LEMA decomposes the computational graph into a sequence of isolated layer blocks.
- **Weight Swapping**: Only the current layer and the next layer occupy VRAM.
- **Persistence**: Model weights remain frozen in System RAM/Disk; only LoRA adapters are maintained in active memory.

### 3. The Triple-Buffer Strategy
LEMA orchestrates data movement across three tiers to hide the latency of PCIe transfers:
- **Storage (NVMe)**: The source of truth (Global Binary File).
- **System RAM**: Pinned Memory Buffers for staging.
- **VRAM**: Active Slot / Prefetch Slot for execution.

This strategy allows for asynchronous prefetching, where the CPU pushes the next layer to VRAM while the GPU computes the current layer.

## Performance

Benchmarks performed on a Tesla P100 (16GB VRAM) comparing Standard PEFT (LoRA) vs LEMA (Streaming).

![VRAM Benchmark](docs/assets/vram_benchmark.png)

| Model | Standard PEFT VRAM | LEMA VRAM | Savings | Status (Fine-Tuning) |
| :--- | :--- | :--- | :--- | :--- |
| **TinyLlama 1.1B** | 2.67 GB | **2.12 GB** | **20.5%** | **Stable** |
| **SmolLM2 1.7B** | 3.88 GB | **3.20 GB** | **17.6%** | **Stable** |
| **Llama-2 7B** | 13.99 GB* | **5.90 GB** | **~58%** | **LEMA Recommended** |

*\*Note: At sequence length 128, Standard PEFT narrowly fits in 16GB VRAM. However, increasing the workload to a standard sequence length of 512 causes an immediate **Out-Of-Memory (OOM)** crash. LEMA maintains a consistent ~6GB footprint even as sequence length scales, providing over **10GB of headroom** for activations and larger batches.*

![Speed Benchmark](docs/assets/speed_benchmark.png)

### The Headroom Advantage
The primary value of LEMA is not just "fitting" the model, but providing the **computational headroom** necessary for real-world training. On a 16GB GPU:
- **Standard PEFT**: Operating at ~88% VRAM capacity just to load the model and run a minimal step. Zero room for longer contexts or higher batch sizes.
- **LEMA**: Operating at ~37% VRAM capacity. Allows for significantly larger sequence lengths, higher batch sizes, or even larger models (13B+) on the same hardware.

## Installation

### From Source
```bash
git clone https://github.com/Pomilon/LEMA.git
cd LEMA
pip install -e .
```

### Requirements
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Safetensors >= 0.3.0
- Accelerate >= 0.20.0
- PEFT >= 0.4.0

## Usage

LEMA uses a configuration-driven approach:

```python
from lema.config import LemaConfig, MemoryStrategy
from lema.engine.trainer import LemaTrainer
from lema.models.llama import LlamaAdapter
from lema.core.gbi import GlobalBinaryIndex
from lema.core.lora import LoRAManager
import torch

# 1. Configuration
config = LemaConfig(
    model_name_or_path="llama2_7b.safetensors",
    device="cuda",
    strategy=MemoryStrategy.STREAMING, # Disk -> RAM -> VRAM
    lora_rank=16,
    learning_rate=1e-4,
    gradient_checkpointing=True # Essential for large models
)

# 2. Components
# Load HF config dict manually or via AutoConfig
from transformers import AutoConfig
hf_config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
adapter = LlamaAdapter(hf_config.to_dict())

gbi = GlobalBinaryIndex(config.gbi_path)

# 3. LoRA Setup
lora_manager = LoRAManager({
    "r": config.lora_rank,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
}, device=config.device)

# Initialize adapter with LoRA
for layer in adapter.get_layer_metadata():
    if layer['type'] == 'block':
        module = adapter.construct_layer_module(layer['id'], None, lora_manager)
        adapter.release_layer_module(module)

# 4. Trainer
optimizer = torch.optim.AdamW(lora_manager.get_trainable_parameters(), lr=config.learning_rate)

trainer = LemaTrainer(
    config=config,
    model_adapter=adapter,
    gbi=gbi,
    lora_manager=lora_manager,
    optimizer=optimizer
)

# 5. Training Step
input_ids = torch.randint(0, 32000, (1, 512)).cuda()
trainer.train_step(input_ids, labels=input_ids)
```

## License
MIT License - Copyright (c) 2026 Pomilon

## Future Roadmap

While LEMA v1.0 is stable and functional for 7B fine-tuning, I aim to significantly reduce the streaming overhead and expand compatibility.

* **C++/CUDA Backend**: I plan to move the `TripleBufferManager` and memory streaming logic from Python to a C++ extension or custom CUDA kernels to bypass the GIL and reduce overhead to the theoretical minimum (~1.1x).
* **Library Integration**: I am working toward deeper integration with Hugging Face `Trainer` and `Accelerate` for seamless usage in existing pipelines.
* **Quantization Support**: I intend to implement native support for 8-bit and 4-bit loading within the streaming pipeline for even lower memory footprints.
* **Model Support**: I am expanding support beyond Llama and GPT-2 to include Mistral, Mixtral (MoE), and other architectures.
