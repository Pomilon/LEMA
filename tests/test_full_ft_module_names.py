from transformers import GPT2Config, LlamaConfig
from lema.adapters import GPT2Adapter, LlamaAdapter

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
