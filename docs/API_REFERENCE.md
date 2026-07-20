# LEMA API Reference

## Public API (`from lema import ...`)

### `LemaConfig`
Configuration dataclass for LEMA. All 21 fields:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name_or_path` | `str` | Required | HF Hub ID or local path. Auto-downloaded if not found. |
| `model_type` | `str \| None` | `None` | Auto-detected from HF config if None. |
| `gbi_path` | `str \| None` | `None` | Path to `.safetensors` file. Auto-resolved if None. |
| `device` | `str` | `"cuda"` | Execution device. |
| `strategy` | `MemoryStrategy` | `STREAMING` | `STREAMING` or `RESIDENT`. |
| `backend` | `str` | `"auto"` | `"auto"` — try cpp, raise if unavailable. `"cpp"` or `"python"` — explicit. |
| `max_ram_gb` | `float` | `0.0` | RAM budget in GB. `0` = auto-detect 70% of system RAM. |
| `max_vram_gb` | `float` | `0.0` | VRAM budget in GB. `0` = auto-detect 60% of GPU VRAM. |
| `vram_fraction` | `float` | `0.8` | Fraction of total VRAM to allow PyTorch (via `set_per_process_memory_fraction`). |
| `prefetch_distance` | `int` | `2` | How many layers ahead to prefetch (auto-tuned by flight check). |
| `lora_rank` | `int` | `16` | LoRA rank (r). |
| `lora_alpha` | `int` | `32` | LoRA alpha scaling. |
| `lora_target_modules` | `list[str]` | `[q_proj,k_proj,...]` | Module names to inject LoRA adapters into. |
| `learning_rate` | `float` | `1e-4` | Learning rate (informational — consumed by user code). |
| `batch_size` | `int` | `1` | Batch size (informational — consumed by user code). |
| `max_seq_length` | `int` | `512` | Max sequence length (informational — consumed by user code). |
| `gradient_accumulation_steps` | `int` | `1` | Steps before optimizer update. Scales loss and checkpoint timing. |
| `gradient_checkpointing` | `bool` | `False` | Enable intra-layer activation checkpointing. |
| `save_steps` | `int` | `500` | Automatic checkpoint interval (0 = disable). |
| `output_dir` | `str` | `"output"` | Directory for automatic checkpoints. |
| `dtype` | `str` | `"float16"` | `"float16"`, `"bfloat16"`, or `"float32"`. |
| `attn_implementation` | `str` | `"eager"` | `"eager"`, `"sdpa"`, or `"flash_attention_2"`. |

Methods:
- `to_dict()` → `dict` — serializes config (handles enums).
- `save_pretrained(path)` — writes `lema_config.json`.
- `from_pretrained(path, **kwargs)` — classmethod, loads config with overrides.

### `MemoryStrategy` (Enum)
- `STREAMING`: Disk → RAM → VRAM. Lower RAM usage, higher latency.
- `RESIDENT`: All weights in pinned RAM. Faster steps, higher RAM usage.

### `LemaModel`
High-level interface. Wraps GBI, adapter, LoRA, and memory manager.

```python
model = LemaModel(config)              # config: LemaConfig or str(path to config)
model.initialize_lora()                # pre-init adapters, warm module pool
model.get_trainable_parameters()       # → list[nn.Parameter]
model.get_trainer(optimizer, **kwargs) # → LemaTrainer
model.simulate_and_optimize()          # flight check, auto-tune strategy/prefetch
model.generate(prompt, tokenizer, ...) # inference (no_grad)
model.save_pretrained(path)            # saves config + adapter_model.bin
model.to(device)                       # move to device
```

Class methods:
- `from_pretrained(path, **kwargs)` — loads config + LoRA adapters from directory.

Transparent internals:
- `model.config` — `LemaConfig`
- `model.adapter` — the model adapter (e.g. `LlamaAdapter`)
- `model.gbi` — `GlobalBinaryIndex`
- `model.lora` — `LoRAManager`
- `model.memory` — `TripleBufferManager`

### `LemaTrainer`
Training loop orchestrator. Handles async memory pipeline.

```python
trainer = LemaTrainer(config, model_adapter, gbi, lora_manager, optimizer,
                      memory_manager=None)
```

Methods:
- `train_step(inputs, labels=None)` → `(logits, loss)` — forward + backward + optimizer step.
- `evaluate(dataloader)` → `avg_loss` — validation loop (no_grad).
- `save_checkpoint(path)` — saves config + LoRA + optimizer state.
- `global_step` / `accumulation_step` — counters.

### Utilities
- `logger` — module-level `logging.Logger` instance.
- `convert_to_monolith(model_path, output_path)` → `str` — merges sharded `.safetensors` into a single file.

---

## Adapter API (`from lema.adapters import ...`)

### `get_adapter(model_type, config)` → `LemaModelAdapter`
Registry lookup. Built-in types: `llama`, `gpt2`, `mistral`, `mixtral`, `lfm2_moe`.

### `register_adapter(model_type, adapter_class)`
Extend LEMA with custom model architectures.

### `LemaModelAdapter` (ABC)
Implement for new architectures:

| Method | Returns | Description |
|---|---|---|
| `get_layer_metadata()` | `list[dict]` | Layer descriptions (id, name, type). |
| `construct_layer_module(id, weights, lora)` | `nn.Module` | Build layer from flat VRAM buffer. |
| `forward_layer(module, inputs, **kwargs)` | `Any` | Execute layer forward. |
| `get_param_names_for_layer(id)` | `list[str]` | Weight keys in safetensors for this layer. |
| `hidden_size` (property) | `int` | Model hidden dimension. |

Each adapter has `MODEL_TYPE: str` — used by the auto-registry.

---

## Private API (available but not in public exports)

| Module | Key Classes |
|---|---|
| `lema._gbi` | `GlobalBinaryIndex` — multi-file safetensors index. |
| `lema._lora` | `LoRAManager`, `LoRAWrapper` — LoRA parameter lifecycle. |
| `lema._memory` | `TripleBufferManager`, `HAS_CPP_BACKEND` — memory pipeline. |
| `lema._utils._model_utils` | `break_shared_weights()`, `prepare_monolithic_safetensors()` |
| `lema._utils._logger` | `setup_logger(name, level)` |
