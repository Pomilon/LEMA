from transformers import GPT2Config, LlamaConfig, MistralConfig, MixtralConfig
import pytest
from lema.adapters import GPT2Adapter, LlamaAdapter, MistralAdapter, MixtralAdapter, Lfm2Adapter

def test_gpt2_block_mapping():
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                     attn_implementation="eager")
    a = GPT2Adapter(cfg.to_dict())
    assert a.get_module_param_name(1, "transformer.h.0.attn.c_attn.weight") == "attn.c_attn.weight"
    assert a.get_module_param_name(3, "transformer.h.2.mlp.c_proj.bias") == "mlp.c_proj.bias"
    assert a.get_module_param_name(0, "transformer.wte.weight") == "wte.weight"
    assert a.get_module_param_name(4, "transformer.ln_f.weight") == "ln_f.weight"
    assert a.get_module_param_name(4, "lm_head.weight") == "head.weight"

def test_llama_block_mapping():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=16, attn_implementation="eager")
    a = LlamaAdapter(cfg.to_dict())
    assert a.get_module_param_name(1, "model.layers.0.self_attn.q_proj.weight") == "self_attn.q_proj.weight"
    assert a.get_module_param_name(2, "model.layers.1.mlp.down_proj.weight") == "mlp.down_proj.weight"
    assert a.get_module_param_name(0, "model.embed_tokens.weight") == "embed_tokens.weight"
    assert a.get_module_param_name(3, "model.norm.weight") == "norm.weight"
    assert a.get_module_param_name(3, "lm_head.weight") == "lm_head.weight"


def _mistral_adapter():
    cfg = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=16, attn_implementation="eager")
    return MistralAdapter(cfg.to_dict())


def _mixtral_adapter():
    cfg = MixtralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        num_local_experts=2, num_experts_per_tok=1,
                        max_position_embeddings=16, attn_implementation="eager")
    return MixtralAdapter(cfg.to_dict())


try:
    from transformers import Lfm2MoeConfig
    HAS_LFM2 = True
except ImportError:
    HAS_LFM2 = False


def _lfm2_adapter():
    cfg = Lfm2MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, num_experts=4, num_experts_per_tok=2,
                        layer_types=["conv", "full_attention"], num_dense_layers=1,
                        max_position_embeddings=16)
    return Lfm2Adapter(cfg.to_dict())


# (adapter factory, head layer id, mapping cases as (layer_id, full name, module name), tied)
_ADAPTER_CASES = [
    (
        _mistral_adapter, 3,
        [
            (1, "model.layers.0.self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            (2, "model.layers.1.mlp.down_proj.weight", "mlp.down_proj.weight"),
            (0, "model.embed_tokens.weight", "embed_tokens.weight"),
            (3, "model.norm.weight", "norm.weight"),
            (3, "lm_head.weight", "lm_head.weight"),
        ],
        False,
    ),
    (
        _mixtral_adapter, 3,
        [
            (1, "model.layers.0.input_layernorm.weight", "input_layernorm.weight"),
            (1, "model.layers.0.self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            (2, "model.layers.1.self_attn.o_proj.weight", "self_attn.o_proj.weight"),
            (0, "model.embed_tokens.weight", "embed_tokens.weight"),
            (3, "model.norm.weight", "norm.weight"),
            (3, "lm_head.weight", "lm_head.weight"),
        ],
        False,
    ),
]


@pytest.mark.parametrize("factory,head_id,mapping_cases,tied", _ADAPTER_CASES)
def test_mistral_and_mixtral_mapping(factory, head_id, mapping_cases, tied):
    a = factory()
    for layer_id, full, module in mapping_cases:
        assert a.get_module_param_name(layer_id, full) == module
    head_names = a.get_param_names_for_layer(head_id)
    assert ("lm_head.weight" in head_names) == (not tied)


@pytest.mark.skipif(not HAS_LFM2, reason="Lfm2MoeConfig not available in this transformers version")
def test_lfm2_mapping():
    a = _lfm2_adapter()
    cases = [
        (1, "model.layers.0.operator_norm.weight", "operator_norm.weight"),
        (1, "model.layers.0.conv.conv.weight", "conv.conv.weight"),
        (2, "model.layers.1.self_attn.q_proj.weight", "self_attn.q_proj.weight"),
        (2, "model.layers.1.feed_forward.gate.weight", "feed_forward.gate.weight"),
        (0, "model.embed_tokens.weight", "embed_tokens.weight"),
        (3, "model.embedding_norm.weight", "embedding_norm.weight"),
        (3, "lm_head.weight", "lm_head.weight"),
    ]
    for layer_id, full, module in cases:
        assert a.get_module_param_name(layer_id, full) == module
    # lfm2 defaults to tie_word_embeddings=True, but lm_head.weight is still
    # listed (the file stores a tied copy) so the head module loads it instead
    # of staying at random init
    assert "lm_head.weight" in a.get_param_names_for_layer(3)
