import os
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors.torch import save_file
from lema import LemaModel, LemaConfig, MemoryStrategy
from lema import _w8a8
from lema._quantized_linear import QuantizedLinear


def _build_llama(tmp_path, **cfg_kwargs):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128, attn_implementation="eager")
    hf = LlamaForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    save_file(sd, str(model_dir / "model.safetensors"))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="llama",
                    gbi_path=str(model_dir / "model.safetensors"), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0, **cfg_kwargs)
    return LemaModel(lc)


def test_quantized_linear_w8a8_close_to_fp32():
    torch.manual_seed(0)
    lin = QuantizedLinear(64, 64)
    w = torch.randn(64, 64) * 0.1
    lin.set_int8_weight(w)
    x = torch.randn(4, 16, 64)
    out = lin(x)
    ref = x @ w.t()
    rel = (out - ref).abs().max() / ref.abs().max()
    assert rel.item() < 5e-2


def test_quantized_linear_fallback_matches_w8a8(tmp_path):
    torch.manual_seed(0)
    lin = QuantizedLinear(64, 64)
    w = torch.randn(64, 64) * 0.1
    lin.set_int8_weight(w)
    x = torch.randn(4, 16, 64)
    x.requires_grad_(True)
    with torch.enable_grad():
        out_grad = lin(x)
        assert out_grad.requires_grad
        out_grad.sum().backward()
        assert x.grad is not None
    x = x.detach()
    with torch.no_grad():
        out_native = lin(x)
    diff = (out_grad.detach() - out_native).abs().max().item()
    assert diff < 5e-2


def test_quantized_layer_forward_close_to_fp16(tmp_path):
    q = _build_llama(tmp_path, weights_bits=8)
    f = _build_llama(tmp_path, weights_bits=None)
    hidden = torch.randn(1, 8, 32)
    layer_id = 1
    outs = []
    for model in (q, f):
        tr = model.store.transfer
        tr.prefetch_to_ram(layer_id, slot=0)
        tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
        flat = tr.get_vram_flat_buffer(0)
        block = model.adapter.construct_layer_module(layer_id, flat, None)
        block.eval()
        outs.append(model.adapter.forward_layer(block, hidden))
    diff = (outs[0].float() - outs[1].float()).abs().max().item()
    assert diff < 0.5, f"quantized vs fp16 layer forward max diff {diff}"


def test_quantized_layer_scale_w_matches_pack(tmp_path):
    q = _build_llama(tmp_path, weights_bits=8)
    tr = q.store.transfer
    tr.prefetch_to_ram(1, slot=0)
    tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = q.adapter.construct_layer_module(1, flat, None)
    w_ref = q.gbi.load_tensors(["model.layers.0.self_attn.q_proj.weight"], device="cpu")[
        "model.layers.0.self_attn.q_proj.weight"]
    w_int8 = block.self_attn.q_proj.weight_int8.detach()
    w_deq = w_int8.t().float() * block.self_attn.q_proj.scale_w.view(-1, 1)
    rel = (w_deq - w_ref.float()).abs().max() / w_ref.float().abs().max()
    assert rel.item() < 1e-2, f"quantized weight rel error {rel.item()}"


def test_w8a8_flat_buffer_is_int8_not_dequant(tmp_path):
    if not _w8a8.HAS_NATIVE:
        pytest.skip("no native kernel")
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    tr.prefetch_to_ram(1, slot=0)
    tr.async_transfer_to_vram(1, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    assert flat.dtype == torch.int8, f"expected int8 flat buffer, got {flat.dtype}"


def test_w8a8_emb_head_stay_dequantized(tmp_path):
    if not _w8a8.HAS_NATIVE:
        pytest.skip("no native kernel")
    model = _build_llama(tmp_path, weights_bits=8)
    tr = model.store.transfer
    tr.prefetch_to_ram(0, slot=0)
    tr.async_transfer_to_vram(0, vram_slot=0, ram_slot=0)
    emb_flat = tr.get_vram_flat_buffer(0)
    assert emb_flat.dtype == torch.float32, f"emb layer should be dequantized, got {emb_flat.dtype}"
    head_id = 3
    tr.prefetch_to_ram(head_id, slot=1)
    tr.async_transfer_to_vram(head_id, vram_slot=1, ram_slot=1)
    head_flat = tr.get_vram_flat_buffer(1)
    assert head_flat.dtype == torch.float32, f"head layer should be dequantized, got {head_flat.dtype}"
    emb = model.adapter.construct_layer_module(0, emb_flat, None)
    head = model.adapter.construct_layer_module(head_id, head_flat, None)
    ids = torch.randint(0, 50, (1, 8))
    h = emb(ids)
    logits = head(h)
    assert logits.shape == (1, 8, 100)