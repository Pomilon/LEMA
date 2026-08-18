# LEMA API Reference

## Public API (`from lema import ...`)

### `LemaConfig`
Configuration dataclass for LEMA. All 28 fields:

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
| `training_mode` | `TrainingMode` | `LORA` | `LORA` or `SELECTIVE_FULL`. Mutually exclusive — one active per model. |
| `state_strategy` | `StateStrategy` | `STREAMING` | `STREAMING` — states in pinned RAM, streamed into VRAM per layer. `VRAM` is a documented optimization, not yet implemented. |
| `trainable_modules` | `list[str]` | `[]` | Suffix patterns matched against parameter names. `[]` = all modules. |
| `trainable_layers` | `list[str]` | `[]` | `"last:K"`, `"first:K"`, explicit layer IDs, `"emb"`, `"head"`. `[]` = all layers (whole model). |
| `grad_accum_backend` | `str` | `"auto"` | `"auto"` | `"ram"` | `"disk"`. `auto` picks disk when fp32 accumulators exceed the RAM budget. |
| `save_optimizer` | `bool` | `True` | Save fp32 Adam moments + layer steps with each full-FT checkpoint. |
| `weight_decay` | `float` | `0.01` | AdamW weight decay for full-FT per-layer stepping. |

Methods:
- `to_dict()` → `dict` — serializes config (handles enums).
- `save_pretrained(path)` — writes `lema_config.json`.
- `from_pretrained(path, **kwargs)` — classmethod, loads config with overrides.

### `MemoryStrategy` (Enum)
- `STREAMING`: Disk → RAM → VRAM. Lower RAM usage, higher latency.
- `RESIDENT`: All weights in pinned RAM. Faster steps, higher RAM usage.

### `TrainingMode` (Enum)
- `LORA`: Default — injects LoRA adapters; trains adapter params only.
- `SELECTIVE_FULL`: Full fine-tuning of selected real weights (no LoRA). Base weights stay on disk/RAM; selected weights, fp32 optimizer states, and accumulators live in RAM.

### `StateStrategy` (Enum)
- `STREAMING`: Optimizer states/weights in pinned RAM, streamed into VRAM per layer. Default.
- `VRAM`: Documented optimization for small selections (states permanently in VRAM) — **not yet implemented**.

### `LemaModel`
High-level interface. Wraps GBI, adapter, LoRA, and memory manager.

```python
model = LemaModel(config)              # config: LemaConfig or str(path to config)
model.initialize_lora()                # pre-init adapters, warm module pool (LoRA mode only)
model.get_trainable_parameters()       # → list[nn.Parameter] (LoRA params or full-FT true weights)
model.get_trainer(optimizer, **kwargs) # → LemaTrainer; in selective_full mode optimizer may be omitted
model.simulate_and_optimize()          # flight check, auto-tune strategy/prefetch (LoRA mode only)
model.generate(prompt, tokenizer, ...) # inference (no_grad)
model.save_pretrained(path)            # saves config + adapter_model.bin, or + delta/optimizer in full-FT
model.to(device)                       # move to device
```

Mode-specific behavior:
- **LoRA** (`training_mode=LORA`): `full_ft_manager is None`, `lora_manager` is set. Call `initialize_lora()` and pass an `optimizer` to `get_trainer()`.
- **Selective full-FT** (`training_mode=SELECTIVE_FULL`): `lora_manager is None`, `full_ft_manager` is set. `initialize_lora()` is a no-op; `get_trainer()` needs no optimizer (per-layer AdamW is internal). `save_pretrained` writes `delta.safetensors` + `delta.index.json` (+ `optimizer_fullft.bin` if `save_optimizer`).

Class methods:
- `from_pretrained(path, **kwargs)` — loads config, then restores LoRA adapters, or full-FT delta + optimizer state.

Transparent internals:
- `model.config` — `LemaConfig`
- `model.adapter` — the model adapter (e.g. `LlamaAdapter`)
- `model.gbi` — `GlobalBinaryIndex`
- `model.lora` / `model.lora_manager` — `LoRAManager` (LoRA mode)
- `model.full_ft_manager` — `FullFTManager` (selective_full mode)
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
- `save_checkpoint(path)` — saves config + LoRA, or full-FT delta + optimizer state.
- `global_step` / `accumulation_step` — counters (full-FT accumulation counter lives on `full_ft_manager`).

### Utilities
- `logger` — module-level `logging.Logger` instance.
- `convert_to_monolith(model_path, output_path)` → `str` — merges sharded `.safetensors` into a single file.
- `merge_delta(base_path, delta_path, out_path)` — merges a full-FT `delta.safetensors` into a base `.safetensors`, producing a servable model (base tensors + delta, cast back to base dtype).

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
| `lema._full_ft` | `FullFTManager` — full-FT selection, RAM-resident true weights, fp32 accumulators, per-layer AdamW step, disk mmap backend, delta/optimizer save & load. |
| `lema._memory` | `TripleBufferManager`, `HAS_CPP_BACKEND` — memory pipeline. |
| `lema._utils._model_utils` | `break_shared_weights()`, `prepare_monolithic_safetensors()` |
| `lema._utils._logger` | `setup_logger(name, level)` |

### `FullFTManager` (private, selective_full mode)

RAM-resident trainable-weight manager. Constructor: `FullFTManager(gbi, adapter, config)`.

Public state:
- `selected: dict[int, list[str]]` — layer_id → selected safetensors param names.
- `selected_layer_keys: dict[int, list[tuple[int, str]]]` — layer_id → `(layer_id, name)` keys.
- `true_weights: dict[tuple[int, str], torch.Tensor]` — the only persistent trainable copy (model dtype).
- `original: dict[tuple[int, str], torch.Tensor]` — pre-training snapshot, used for delta computation.
- `opt_states` — fp32 Adam moments per key.
- `accumulators` — fp32 gradient accumulators (RAM tensors or disk-backed views).
- `layer_steps` — per-layer step counts (for bias correction).
- `accumulation_step` — cross-micro-batch boundary counter.
- `accumulator_backend` — `"ram"` or `"disk"`.

Methods:
- `resolve_selection()` — applies `trainable_modules` × `trainable_layers`, raises on empty selection, excludes `lm_head.weight` when embeddings are tied.
- `apply_to_module(layer_id, module)` — loads selected true weights into the layer module (fp32 copy), freezes everything else.
- `accumulate_grads(layer_id, module)` — adds fp32 `.grad`s into accumulators, zeroes ephemeral grads.
- `clip_grad_norm_(layer_id, max_norm)` — global-norm clip over the layer's accumulators.
- `step_layer(layer_id)` — per-layer custom AdamW (betas 0.9/0.999, eps 1e-8, weight_decay from config, bias correction); updates true weights, zeroes the accumulator.
- `get_trainable_parameters()` — list of true weight tensors.
- `total_selected_params()` — int count of selected parameters.
- `save_delta(dir)` / `load_delta(dir)` — fp32 `delta.safetensors` + `delta.index.json`; restore computes `original + delta` in fp32 (exact at any model dtype).
- `save_optimizer(dir)` / `load_optimizer(dir)` — `optimizer_fullft.bin` with Adam moments + layer steps.
- `close()` — flushes/closes disk mmap files.
