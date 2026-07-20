import pytest
import torch
from lema.adapters import get_adapter

try:
    from transformers import Lfm2MoeConfig
    HAS_LFM = True
except ImportError:
    HAS_LFM = False

@pytest.mark.skipif(not HAS_LFM, reason="Lfm2MoeConfig not available in this transformers version")
def test_lfm2_adapter_init():
    config = Lfm2MoeConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_experts=4, num_experts_per_tok=2,
        layer_types=["conv", "full_attention"],
        num_dense_layers=1, max_position_embeddings=128,
    )
    adapter = get_adapter("lfm2_moe", config.to_dict())

    layers = adapter.get_layer_metadata()
    assert len(layers) == 4  # emb, 2 blocks, head

    param_names = adapter.get_param_names_for_layer(1)
    assert "model.layers.0.operator_norm.weight" in param_names
    assert "model.layers.0.conv.in_proj.weight" in param_names

    param_names = adapter.get_param_names_for_layer(2)
    assert "model.layers.1.self_attn.q_proj.weight" in param_names
    assert "model.layers.1.feed_forward.gate.weight" in param_names
    assert "model.layers.1.feed_forward.experts.0.w1.weight" in param_names

    module = adapter.construct_layer_module(1)
    assert module is not None
