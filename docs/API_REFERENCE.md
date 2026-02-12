# LEMA API Reference

This document provides detailed information about the LEMA (Layer-wise Efficient Memory Abstraction) library API.

## Core API

### `LemaModel`
The primary entry point for the framework. It orchestrates memory management, adapters, and LoRA parameters.

#### `__init__(config: LemaConfig)`
Initializes the model using a `LemaConfig` object.

#### `get_trainer(optimizer: torch.optim.Optimizer)`
Returns a `LemaTrainer` instance pre-configured with this model's components and memory manager.

#### `initialize_lora()`
Pre-initializes all LoRA adapters. Must be called before `get_trainable_parameters()` for new models.

#### `get_trainable_parameters()`
Returns a list of all trainable parameters (LoRA weights) managed by the model.

#### `save_pretrained(save_directory: str)`
Saves the configuration and LoRA adapter weights.

#### `from_pretrained(path: str, **kwargs)` (Class Method)
Loads a LEMA model from a directory containing `lema_config.json` and `adapter_model.bin`.

---

### `LemaConfig`
Configuration dataclass for LEMA.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name_or_path` | `str` | Required | HuggingFace ID or path to model directory. |
| `model_type` | `str` | `None` | `llama` or `gpt2`. Auto-detected if None. |
| `gbi_path` | `str` | `None` | Path to the `.safetensors` file. |
| `device` | `str` | `"cuda"` | Execution device. |
| `strategy` | `MemoryStrategy` | `STREAMING` | `STREAMING` or `RESIDENT`. |
| `save_steps` | `int` | `500` | Steps between automatic checkpoints. |
| `output_dir` | `str` | `"output"` | Directory for automatic checkpoints. |
| `lora_rank` | `int` | `16` | LoRA rank (r). |
| `lora_alpha` | `int` | `32` | LoRA alpha. |
| `learning_rate` | `float` | `1e-4` | Learning rate. |
| `gradient_checkpointing`| `bool` | `False` | Enable to save activation VRAM. |

---

### `LemaTrainer`
Orchestrates the training loop with layer-swapping logic.

#### `__init__(config, model_adapter, gbi, lora_manager=None, optimizer=None, memory_manager=None)`
Low-level constructor. Preferred usage is via `LemaModel.get_trainer()`.

#### `train_step(inputs: torch.Tensor, labels: torch.Tensor = None)`
Executes one forward and backward pass. Tracks `global_step` and triggers auto-checkpointing.
- Returns: `(logits, loss_value)`.

#### `save_checkpoint(save_directory: str)`
Saves the model state, configuration, and optimizer state.