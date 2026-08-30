import os
import torch
import pytest
from transformers import MistralConfig, MixtralConfig, MistralForCausalLM, MixtralForCausalLM
from safetensors.torch import save_file
from lema import LemaModel, LemaConfig, MemoryStrategy, _w8a8


def _build_mistral(tmp_path, **kw):
    torch.manual_seed(0)
    cfg = MistralConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=128, attn_implementation="eager")
    hf = MistralForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "md_mistral"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="mistral",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **kw)
    return LemaModel(lc)


def _build_mixtral(tmp_path, **kw):
    torch.manual_seed(0)
    cfg = MixtralConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        num_local_experts=4, num_experts_per_tok=2, hidden_act="silu",
                        max_position_embeddings=128, attn_implementation="eager")
    cfg._experts_implementation = "grouped_mm"
    hf = MixtralForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "md_mixtral"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="mixtral",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **kw)
    return LemaModel(lc)


def _build_lfm2(tmp_path, **kw):
    torch.manual_seed(0)
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeConfig, Lfm2MoeForCausalLM
    cfg = Lfm2MoeConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                        moe_intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=128,
                        layer_types=["full_attention", "full_attention"],
                        num_dense_layers=1, _experts_implementation="grouped_mm",
                        attn_implementation="eager")
    hf = Lfm2MoeForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "md_lfm2"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="lfm2_moe",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **kw)
    return LemaModel(lc)


def test_mistral_quantized_layer_forward_close_to_fp16(tmp_path):
    q = _build_mistral(tmp_path, weights_bits=8)
    f = _build_mistral(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 4, 64)
    outs = []
    for model in (q, f):
        tr = model.store.transfer
        tr.prefetch_to_ram(1, slot=0)
        tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(1, flat, None)
        block.eval()
        with torch.no_grad():
            outs.append(model.adapter.forward_layer(block, hidden, layer_id=1))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.5


def test_mixtral_quantized_layer_forward_close_to_fp16(tmp_path):
    q = _build_mixtral(tmp_path, weights_bits=8)
    f = _build_mixtral(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 4, 64)
    outs = []
    for model in (q, f):
        tr = model.store.transfer
        tr.prefetch_to_ram(1, slot=0)
        tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(1, flat, None)
        block.eval()
        with torch.no_grad():
            outs.append(model.adapter.forward_layer(block, hidden, layer_id=1))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.5


def test_lfm2_quantized_layer_forward_close_to_fp16(tmp_path):
    q = _build_lfm2(tmp_path, weights_bits=8)
    f = _build_lfm2(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 4, 64)
    outs = []
    for model in (q, f):
        tr = model.store.transfer
        tr.prefetch_to_ram(1, slot=0)
        tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(1, flat, None)
        block.eval()
        with torch.no_grad():
            outs.append(model.adapter.forward_layer(block, hidden, layer_id=1))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.5


def test_lfm2_moe_quantized_layer_forward_close_to_fp16(tmp_path):
    q = _build_lfm2(tmp_path, weights_bits=8)
    f = _build_lfm2(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 4, 64)
    outs = []
    for model in (q, f):
        tr = model.store.transfer
        tr.prefetch_to_ram(2, slot=0)
        tr.async_transfer_to_vram(2, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(2, flat, None)
        block.eval()
        with torch.no_grad():
            outs.append(model.adapter.forward_layer(block, hidden, layer_id=2))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.5


@pytest.mark.skipif(not _w8a8.HAS_NATIVE, reason="native W8A8 ext not built")
def test_w8a8_flat_buffer_is_int8_for_all_adapters(tmp_path):
    for builder in (_build_mistral, _build_mixtral, _build_lfm2):
        m = builder(tmp_path, weights_bits=8)
        tr = m.store.transfer
        tr.prefetch_to_ram(1, slot=0)
        tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        assert flat.dtype == torch.int8


def test_lfm2_short_conv_stays_fp32(tmp_path):
    torch.manual_seed(0)
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeConfig, Lfm2MoeForCausalLM
    cfg = Lfm2MoeConfig(vocab_size=100, hidden_size=64, intermediate_size=128,
                        moe_intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, num_experts=4,
                        num_experts_per_tok=2, max_position_embeddings=128,
                        layer_types=["short_conv", "full_attention"],
                        num_dense_layers=1, _experts_implementation="grouped_mm",
                        attn_implementation="eager")
    hf = Lfm2MoeForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "md_lfm2_conv"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="lfm2_moe",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, weights_bits=8)
    m = LemaModel(lc)
    assert not m.adapter.supports_quantized_layer(1)
    tr = m.store.transfer
    tr.prefetch_to_ram(1, slot=0)
    tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    assert flat.dtype != torch.int8
