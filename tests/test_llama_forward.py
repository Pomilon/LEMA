import torch
import torch.nn as nn
from src.lema.models.llama import LlamaAdapter
from transformers import LlamaConfig

def test_llama_adapter_forward():
    config = {
        "vocab_size": 32000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
    }
    
    adapter = LlamaAdapter(config)
    
    # We can pass None for flat_buffer to just construct the module (uninitialized)
    module = adapter.construct_layer_module(1, None)
    module.to("cpu")
    
    hidden_states = torch.randn(1, 16, 128)
    
    print("Testing forward_layer...")
    try:
        # Use no_grad for simple forward test
        with torch.no_grad():
            output = adapter.forward_layer(module, hidden_states)
        print(f"Success! Output shape: {output.shape}")
        assert output.shape == (1, 16, 128)
    except Exception as e:
        print(f"Failed! Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_llama_adapter_forward()