import os
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file
from lema import LemaModel, LemaConfig, MemoryStrategy


def test_quant_bits_validation_accepts_off_and_valid():
    LemaConfig(model_name_or_path="x", weights_bits=None, opt_state_bits=0,
               grad_acc_bits=8, kv_bits=8)
    LemaConfig(model_name_or_path="x", weights_bits=4)
    LemaConfig(model_name_or_path="x", weights_bits=8)


def test_quant_bits_validation_rejects_invalid():
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", weights_bits=5)
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", opt_state_bits=4)  # 4 only for weights
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", kv_bits=4)  # int8 only for KV
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", grad_acc_bits=16)


def test_quant_bits_serialize_roundtrip(tmp_path):
    c = LemaConfig(model_name_or_path="x", weights_bits=8, kv_bits=8)
    c.save_pretrained(str(tmp_path))
    c2 = LemaConfig.from_pretrained(str(tmp_path))
    assert c2.weights_bits == 8 and c2.kv_bits == 8


def _build_llama(tmp_path, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="llama",
                    gbi_path=str(model_path), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **cfg_kwargs)
    return LemaModel(lc)


def test_quantized_weight_stream_loads_and_matches(tmp_path):
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    assert tr.quant_bits == 8
    layer_id = 1
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = model.adapter.construct_layer_module(layer_id, flat, None)

    # compare against fp16 baseline weights from disk
    q = model.gbi.load_tensors(["model.layers.0.self_attn.q_proj.weight"], device="cpu")
    w_ref = q["model.layers.0.self_attn.q_proj.weight"]
    w_lema = block.self_attn.q_proj.weight.detach()
    rel = (w_lema.float() - w_ref.float()).abs().max() / w_ref.float().abs().max()
    assert rel.item() < 1e-2, f"quantized weight rel error {rel.item()}"


def test_quantized_forward_close_to_fp16(tmp_path):
    model_q = _build_llama(tmp_path, weights_bits=8)
    model_fp = _build_llama(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 16, model_q.adapter.hidden_size)
    layer_id = 1
    outs = []
    for model in (model_q, model_fp):
        tr = model.store.transfer
        tr.prefetch_to_ram(layer_id, slot=0)
        tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(layer_id, flat, None)
        block.eval()
        outs.append(model.adapter.forward_layer(block, hidden))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.1, f"quantized vs fp16 forward max diff {diff}"
