import os
import torch
import pytest
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._tensorstore import KVChunkStore

try:
    from transformers import Lfm2MoeConfig
    HAS_LFM = True
except ImportError:
    HAS_LFM = False

HAS_LFM = True  # verified available in this env


def _build(tmp_path, layer_types=("full_attention", "full_attention")):
    torch.manual_seed(0)
    cfg = Lfm2MoeConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, layer_types=list(layer_types), num_dense_layers=1,
        max_position_embeddings=128,
    )
    from transformers import Lfm2MoeForCausalLM
    hf = Lfm2MoeForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type="lfm2_moe",
                    gbi_path=str(model_path), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0)
    return LemaModel(lc)


def _block_with_weights(model, layer_id):
    ad = model.adapter
    tr = model.store.transfer
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = ad.construct_layer_module(layer_id, flat, None)
    block.eval()
    return block


def test_lfm2_chunked_forward_matches_full(tmp_path):
    model = _build(tmp_path)
    ad = model.adapter
    seq = 24
    hidden = torch.randn(1, seq, ad.hidden_size)
    layer_id = 2  # full_attention layer
    block = _block_with_weights(model, layer_id)

    full_out = ad.forward_layer(block, hidden)
    kv_store = KVChunkStore(kv_chunk_size=8)
    chunked_out = ad.chunked_forward_layer(block, hidden, kv_store, layer_id, kv_chunk_size=8)

    diff = (full_out.float() - chunked_out.float()).abs().max().item()
    assert diff < 1e-4, f"lfm2 chunked forward max diff {diff}"


def test_lfm2_decode_matches_chunked_last_token(tmp_path):
    model = _build(tmp_path)
    ad = model.adapter
    layer_id = 2
    block = _block_with_weights(model, layer_id)

    hidden33 = torch.randn(1, 33, ad.hidden_size)
    kv_ref = KVChunkStore(kv_chunk_size=8)
    ref = ad.chunked_forward_layer(block, hidden33, kv_ref, layer_id, kv_chunk_size=8)[:, -1:, :]

    kv_decode = KVChunkStore(kv_chunk_size=8)
    ad.chunked_forward_layer(block, hidden33[:, :32], kv_decode, layer_id, kv_chunk_size=8)
    out = ad.decode_forward_layer(block, hidden33[:, -1:, :], kv_decode, layer_id,
                                  kv_chunk_size=8, position=32, is_new_token=True)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 1e-4, f"lfm2 decode last token max diff {diff}"
