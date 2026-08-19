import os
import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MixtralConfig
from transformers import MistralForCausalLM, MixtralForCausalLM
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._tensorstore import KVChunkStore


def _build(tmp_path, kind):
    torch.manual_seed(0)
    if kind == "llama":
        cfg = LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=128, attn_implementation="eager")
        hf = LlamaForCausalLM(cfg)
    elif kind == "mistral":
        cfg = MistralConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=128, sliding_window=64,
                            attn_implementation="eager")
        hf = MistralForCausalLM(cfg)
    else:  # mixtral
        cfg = MixtralConfig(vocab_size=100, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            num_local_experts=2, num_experts_per_tok=1,
                            max_position_embeddings=128, attn_implementation="eager")
        hf = MixtralForCausalLM(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type=kind, gbi_path=str(model_path),
                    device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0)
    return LemaModel(lc)


@pytest.mark.parametrize("kind", ["llama", "mistral", "mixtral"])
def test_chunked_forward_matches_full(tmp_path, kind):
    model = _build(tmp_path, kind)
    ad = model.adapter
    seq = 32
    hidden = torch.randn(1, seq, ad.hidden_size)
    layer_id = 1
    # load real weights into the block via the transfer engine (avoids random-init
    # instability, e.g. mixtral MoE producing NaN from random weights)
    tr = model.store.transfer
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = ad.construct_layer_module(layer_id, flat, None)
    block.eval()

    full_out = ad.forward_layer(block, hidden)

    kv_store = KVChunkStore(kv_chunk_size=8)
    chunked_out = ad.chunked_forward_layer(block, hidden, kv_store, layer_id, kv_chunk_size=8)

    diff = (full_out.float() - chunked_out.float()).abs().max().item()
    assert diff < 1e-4, f"{kind} chunked forward max diff {diff}"


@pytest.mark.parametrize("kind", ["llama", "mistral", "mixtral"])
def test_generate_kv_matches_old_greedy(tmp_path, kind):
    model = _build(tmp_path, kind)

    class Tok:
        eos_token_id = 2
        def __call__(self, prompt, return_tensors="pt"):
            ids = torch.tensor([[1, 3, 5, 7]])
            class _W(dict):
                def to(self, device):
                    self["input_ids"] = self["input_ids"].to(device)
                    return self
            return _W(input_ids=ids)
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(str(int(x)) for x in ids.tolist())

    tok = Tok()
    torch.manual_seed(1)
    out_old = model.generate("x", tok, max_new_tokens=8, do_sample=False)
    torch.manual_seed(1)
    out_kv = model.generate_kv("x", tok, max_new_tokens=8, do_sample=False, kv_chunk_size=4)
    assert out_old == out_kv, f"{kind}: old={out_old} kv={out_kv}"


@pytest.mark.parametrize("kind", ["llama", "mistral", "mixtral"])
def test_decode_matches_chunked_last_token(tmp_path, kind):
    model = _build(tmp_path, kind)
    ad = model.adapter
    layer_id = 1
    tr = model.store.transfer
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = ad.construct_layer_module(layer_id, flat, None)
    block.eval()

    # reference: chunked forward of the 33-token sequence, take last token
    hidden33 = torch.randn(1, 33, ad.hidden_size)
    kv_ref = KVChunkStore(kv_chunk_size=8)
    ref = ad.chunked_forward_layer(block, hidden33, kv_ref, layer_id, kv_chunk_size=8)[:, -1:, :]

    # decode: prefill first 32 tokens, then decode the 33rd (position 32)
    kv_decode = KVChunkStore(kv_chunk_size=8)
    ad.chunked_forward_layer(block, hidden33[:, :32], kv_decode, layer_id, kv_chunk_size=8)
    out = ad.decode_forward_layer(block, hidden33[:, -1:, :], kv_decode, layer_id,
                                  kv_chunk_size=8, position=32, is_new_token=True)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 1e-4, f"{kind} decode last token max diff {diff}"
