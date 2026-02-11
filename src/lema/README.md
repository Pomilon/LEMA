# LEMA: Layer-wise Efficient Memory Abstraction

LEMA is a framework for fine-tuning large models on constrained hardware by virtualizing VRAM. It implements a **Triple-Buffer Strategy** (Disk -> RAM -> VRAM) to prefetch model layers asynchronously.

## Project Structure

*   **`core/`**: Core memory management logic.
    *   `gbi.py`: Global Binary Index. Wraps `.safetensors` for O(1) weight access.
    *   `memory.py`: `TripleBufferManager`. Manages the pipeline of moving weights between Disk, Pinned RAM, and VRAM.
    *   `lora.py`: LoRA integration (supports `nn.Linear` and `Conv1D`).
*   **`models/`**: Architecture adapters.
    *   `base.py`: `LemaModelAdapter` abstract base class. All new architectures must implement this.
    *   `gpt2.py`: Example implementation for GPT-2.
*   **`engine/`**: Execution logic.
    *   `trainer.py`: The training loop that orchestrates the memory pipeline and computation.

## Adding a New Model

To support a new architecture (e.g., Llama 3):

1.  Create `src/lema/models/llama.py`.
2.  Inherit from `LemaModelAdapter`.
3.  Implement:
    *   `get_layer_metadata()`: Define the sequence of layers (Embeddings -> Blocks -> Head).
    *   `get_param_names_for_layer(id)`: Map layer IDs to their parameter names in the `.safetensors` file.
    *   `construct_layer_module(id, weights)`: Reconstruct the PyTorch `nn.Module` for that layer using the weights.

## Usage

See `demo_lema.py` in the project root for a complete example of initialization and execution.

## Testing

The project includes a comprehensive test suite in `tests/`.

To run all tests:
```bash
PYTHONPATH=. pytest tests/
```

**Key Tests:**
*   `test_gradient_equivalence.py`: Verifies that LEMA produces identical forward pass results to standard PyTorch.
*   `test_training_loop.py`: Verifies that LoRA parameters are correctly updated during training.
*   `test_core_components.py`: Unit tests for internal logic.