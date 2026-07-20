import pytest
import torch
from transformers import MistralConfig, MixtralConfig
from lema.adapters import get_adapter

def test_mistral_adapter_init():
    config = MistralConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2)
    adapter = get_adapter("mistral", config.to_dict())
    
    layers = adapter.get_layer_metadata()
    assert len(layers) == 4  # emb, 2 blocks, head
    
    param_names = adapter.get_param_names_for_layer(1)
    assert "model.layers.0.input_layernorm.weight" in param_names
    assert "model.layers.0.self_attn.q_proj.weight" in param_names
    
    module = adapter.construct_layer_module(1)
    assert module is not None

def test_mixtral_adapter_init():
    config = MixtralConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, num_local_experts=2)
    adapter = get_adapter("mixtral", config.to_dict())
    
    layers = adapter.get_layer_metadata()
    assert len(layers) == 4
    
    param_names = adapter.get_param_names_for_layer(1)
    # Standard attn params (version-independent)
    assert "model.layers.0.input_layernorm.weight" in param_names
    assert "model.layers.0.self_attn.q_proj.weight" in param_names
    # MoE params — exact naming depends on transformers version (block_sparse_moe vs mlp)
    has_moe = any("gate.weight" in n for n in param_names)
    has_experts = any("experts" in n for n in param_names)
    assert has_moe, f"Expected MoE gate param in {param_names}"
    assert has_experts, f"Expected MoE expert params in {param_names}"
    
    module = adapter.construct_layer_module(1)
    assert module is not None
