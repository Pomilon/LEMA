import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema.adapters._gpt2 import GPT2Adapter
from lema._tensorstore import KVChunkStore


def _build(tmp_path, n_layer=2, n_embd=32, n_head=2, seq=32):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=100, n_positions=128, n_embd=n_embd, n_layer=n_layer,
        n_head=n_head, attn_implementation="eager",
    )
    model_hf = GPT2LMHeadModel(config)
    state_dict = {k: v.clone().detach() for k, v in model_hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(state_dict, str(model_path))
    config.save_pretrained(str(model_dir))
    lema_config = LemaConfig(
        model_name_or_path=str(model_dir), model_type="gpt2", gbi_path=str(model_path),
        device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
    )
    return LemaModel(lema_config)


def test_chunked_forward_matches_full(tmp_path):
    model = _build(tmp_path)
    seq = 32
    hidden = torch.randn(1, seq, model.adapter.hidden_size)
    layer_id = 1
    block = model.adapter.construct_layer_module(layer_id, None, model.lora_manager)
    block.eval()  # disable dropout so the deterministic attention math is compared exactly

    full_out = model.adapter.forward_layer(block, hidden)

    kv_store = KVChunkStore(kv_chunk_size=8)
    chunked_out = model.adapter.chunked_forward_layer(block, hidden, kv_store, layer_id, kv_chunk_size=8)

    assert torch.allclose(full_out.float(), chunked_out.float(), atol=1e-4, rtol=1e-4), (
        f"max diff: {(full_out.float() - chunked_out.float()).abs().max().item()}"
    )


def test_forward_layer_uses_chunked_path_when_long(tmp_path):
    model = _build(tmp_path)
    seq = 32
    hidden = torch.randn(1, seq, model.adapter.hidden_size)
    layer_id = 1
    block = model.adapter.construct_layer_module(layer_id, None, model.lora_manager)

    kv_store = KVChunkStore(kv_chunk_size=8)
    out = model.adapter.forward_layer(block, hidden, kv_store=kv_store, layer_id=layer_id, kv_chunk_size=8)
    assert kv_store.num_chunks(layer_id) == seq // 8
    assert out.shape == hidden.shape
