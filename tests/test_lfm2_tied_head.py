import os
import torch
import pytest
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM


def _build(tmp_path, tied=True):
    torch.manual_seed(0)
    cfg = Lfm2MoeConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, layer_types=["full_attention", "full_attention"],
        num_dense_layers=1, max_position_embeddings=128, tie_word_embeddings=tied,
    )
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
    return LemaModel(lc), hf


def test_lfm2_tied_head_weights_loaded(tmp_path):
    """Tied lfm2: the head's lm_head must be loaded from the file (not random
    init), and must equal the embedding weight (the file stores tied copies)."""
    model, _ = _build(tmp_path, tied=True)
    ad = model.adapter
    tr = model.store.transfer
    head_id = ad.get_layer_metadata()[-1]["id"]

    tr.prefetch_to_ram(head_id, slot=0)
    tr.async_transfer_to_vram(head_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    head = ad.construct_layer_module(head_id, flat, None)

    embed = model.gbi.load_tensors(["model.embed_tokens.weight"])["model.embed_tokens.weight"]
    lm_head = model.gbi.load_tensors(["lm_head.weight"])["lm_head.weight"]

    assert torch.equal(head.lm_head.weight.detach(), lm_head), "lm_head not loaded from file"
    assert torch.equal(lm_head, embed), "file's lm_head copy differs from embedding"


def test_lfm2_head_logits_match_reference(tmp_path):
    """With the lm_head loaded, the head-only output matches the HF model that
    wrote the file (isolates the tied-head fix from unrelated decoder drift)."""
    model, hf = _build(tmp_path, tied=True)
    ad = model.adapter
    tr = model.store.transfer
    head_id = ad.get_layer_metadata()[-1]["id"]

    tr.prefetch_to_ram(head_id, slot=0)
    tr.async_transfer_to_vram(head_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    head = ad.construct_layer_module(head_id, flat, None)

    x = torch.randn(1, 8, 64)
    out_lema = head(x)

    hf.eval()
    with torch.no_grad():
        out_hf = hf.lm_head(hf.model.embedding_norm(x))

    diff = (out_lema.float() - out_hf.float()).abs().max().item()
    assert diff < 1e-4, f"lfm2 tied head logits differ from reference: {diff}"