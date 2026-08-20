import os
import torch
import pytest
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy

try:
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM
    HAS_LFM = True
except ImportError:
    HAS_LFM = False
from transformers import MixtralConfig, MixtralForCausalLM

HAS_LFM = True  # verified available in this env


def _to_real_checkpoint_sd(sd, kind, hf_cfg):
    """Convert a fused state_dict (module names) to the per-expert disk format
    used by real checkpoints (experts.{e}.w1/w2/w3.weight, block_sparse_moe.*,
    no lm_head.weight when tied)."""
    out = {}
    for k, v in sd.items():
        if kind == "lfm2":
            if ".feed_forward.experts.gate_up_proj" in k:
                prefix, _ = k.split(".experts.gate_up_proj")
                layer_idx = int(k.split(".")[2])
                inter = v.shape[1] // 2
                for e in range(hf_cfg.num_experts):
                    out[f"{prefix}.experts.{e}.w1.weight"] = v[e, :inter].clone()
                    out[f"{prefix}.experts.{e}.w3.weight"] = v[e, inter:].clone()
                continue
            if ".feed_forward.experts.down_proj" in k:
                prefix, _ = k.split(".experts.down_proj")
                for e in range(v.shape[0]):
                    out[f"{prefix}.experts.{e}.w2.weight"] = v[e].clone()
                continue
            if k == "lm_head.weight" and getattr(hf_cfg, "tie_word_embeddings", False):
                continue
            out[k] = v.clone()
        elif kind == "mixtral":
            if ".mlp.experts.gate_up_proj" in k:
                prefix = k.replace(".mlp.experts.gate_up_proj", "")
                prefix = prefix.replace("mlp.", "block_sparse_moe.")
                inter = v.shape[1] // 2
                for e in range(hf_cfg.num_local_experts):
                    out[f"{prefix}.experts.{e}.w1.weight"] = v[e, :inter].clone()
                    out[f"{prefix}.experts.{e}.w3.weight"] = v[e, inter:].clone()
                continue
            if ".mlp.experts.down_proj" in k:
                prefix = k.replace(".mlp.experts.down_proj", "")
                prefix = prefix.replace("mlp.", "block_sparse_moe.")
                for e in range(v.shape[0]):
                    out[f"{prefix}.experts.{e}.w2.weight"] = v[e].clone()
                continue
            if ".mlp.gate.weight" in k:
                out[k.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")] = v.clone()
                continue
            out[k] = v.clone()
    return out


def _build_real(tmp_path, kind):
    torch.manual_seed(0)
    if kind == "lfm2":
        cfg = Lfm2MoeConfig(
            vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=4,
            num_experts_per_tok=2, layer_types=["full_attention", "full_attention"],
            num_dense_layers=1, max_position_embeddings=128, tie_word_embeddings=True,
        )
        hf = Lfm2MoeForCausalLM(cfg)
        model_cls = Lfm2MoeForCausalLM
        model_type = "lfm2_moe"
    else:
        cfg = MixtralConfig(
            vocab_size=1000, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
            tie_word_embeddings=False, attn_implementation="eager",
        )
        hf = MixtralForCausalLM(cfg)
        model_cls = MixtralForCausalLM
        model_type = "mixtral"
    sd = _to_real_checkpoint_sd(hf.state_dict(), kind, cfg)
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(model_name_or_path=str(model_dir), model_type=model_type,
                    gbi_path=str(model_path), device="cpu",
                    strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0)
    lema = LemaModel(lc)
    return lema, hf


def _block_with_weights(model, layer_id):
    ad = model.adapter
    tr = model.store.transfer
    tr.prefetch_to_ram(layer_id, slot=0)
    tr.async_transfer_to_vram(layer_id, vram_slot=0, ram_slot=0)
    flat = tr.get_vram_flat_buffer(0)
    block = ad.construct_layer_module(layer_id, flat, None)
    block.eval()
    return block


@pytest.mark.skipif(not HAS_LFM, reason="lfm2_moe unavailable")
def test_lfm2_real_checkpoint_moe_params_match(tmp_path):
    """A real-format LFM2 checkpoint (per-expert w1/w2/w3 keys, tied lm_head
    omitted) must load into the MoE layer exactly like transformers does."""
    lema, ref = _build_real(tmp_path, "lfm2")
    block = _block_with_weights(lema, 2)  # layer idx 1 = MoE (num_dense_layers=1)
    ref_layer = ref.model.layers[1]

    lema_params = dict(block.named_parameters())
    ref_params = dict(ref_layer.named_parameters())
    for name, p in ref_params.items():
        assert name in lema_params, f"missing module param {name}"
        assert torch.allclose(lema_params[name].detach().float(), p.detach().float(),
                              atol=1e-6), f"param {name} differs from reference"


@pytest.mark.skipif(not HAS_LFM, reason="lfm2_moe unavailable")
def test_lfm2_real_checkpoint_dense_layer_params_match(tmp_path):
    lema, ref = _build_real(tmp_path, "lfm2")
    block = _block_with_weights(lema, 1)  # layer idx 0 = dense
    ref_layer = ref.model.layers[0]

    lema_params = dict(block.named_parameters())
    ref_params = dict(ref_layer.named_parameters())
    for name, p in ref_params.items():
        assert name in lema_params, f"missing module param {name}"
        assert torch.allclose(lema_params[name].detach().float(), p.detach().float(),
                              atol=1e-6), f"param {name} differs from reference"


@pytest.mark.skipif(not HAS_LFM, reason="lfm2_moe unavailable")
def test_lfm2_real_checkpoint_head_and_embed(tmp_path):
    lema, ref = _build_real(tmp_path, "lfm2")
    head_id = lema.adapter.get_layer_metadata()[-1]["id"]
    block = _block_with_weights(lema, head_id)
    assert torch.allclose(
        block.lm_head.weight.detach().float(), ref.lm_head.weight.detach().float(), atol=1e-6
    ), "tied lm_head fallback (embed_tokens) differs from reference"
    assert torch.allclose(
        block.lm_head.weight.detach().float(), ref.model.embed_tokens.weight.detach().float(),
        atol=1e-6,
    ), "tied lm_head should equal embedding weight"


def test_mixtral_real_checkpoint_params_match(tmp_path):
    """A real-format Mixtral checkpoint (block_sparse_moe.experts.{e}.w1/w2/w3,
    gate) must load into the MoE layer exactly like transformers does."""
    lema, ref = _build_real(tmp_path, "mixtral")
    block = _block_with_weights(lema, 1)
    ref_layer = ref.model.layers[0]

    lema_params = dict(block.named_parameters())
    ref_params = dict(ref_layer.named_parameters())
    for name, p in ref_params.items():
        assert name in lema_params, f"missing module param {name}"
        assert torch.allclose(lema_params[name].detach().float(), p.detach().float(),
                              atol=1e-6), f"param {name} differs from reference"


def test_mixtral_real_checkpoint_moe_forward(tmp_path):
    lema, hf = _build_real(tmp_path, "mixtral")
    block = _block_with_weights(lema, 1)
    ref_layer = hf.model.layers[0].eval()

    x = torch.randn(1, 8, lema.adapter.hidden_size)
    out_lema = lema.adapter.forward_layer(block, x)
    with torch.no_grad():
        pos_ids = torch.arange(8).unsqueeze(0)
        from lema.adapters._chunked_rope import compute_rope
        cos, sin = compute_rope(lema.adapter, block.self_attn, x, pos_ids)
        if cos.ndim == 2:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        elif cos.ndim == 4:
            cos, sin = cos.squeeze(1), sin.squeeze(1)
        seq = x.shape[1]
        m = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
        mask = m.view(1, 1, seq, seq).expand(1, 1, seq, seq)
        out_ref = ref_layer(
            hidden_states=x, attention_mask=mask, position_ids=pos_ids,
            position_embeddings=(cos, sin), use_cache=False,
        )[0]

    diff = (out_lema.float() - out_ref.float()).abs().max().item()
    assert diff < 1e-4, f"mixtral real-checkpoint MoE forward max diff {diff}"