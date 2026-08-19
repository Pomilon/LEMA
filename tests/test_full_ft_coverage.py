import math
import torch
import pytest
from transformers import GPT2Config
from safetensors.torch import save_file
from lema._gbi import GlobalBinaryIndex
from lema.adapters import GPT2Adapter
from lema._config import LemaConfig
from lema._full_ft import FullFTManager


def make_manager(tmp_path, layer_id=1, **overrides):
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                     attn_implementation="eager")
    import transformers
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path / "m.safetensors"
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(path))
    defaults = dict(model_name_or_path=str(path), model_type="gpt2", gbi_path=str(path),
                    device="cpu", dtype="float32", training_mode="selective_full",
                    trainable_layers=[str(layer_id)], trainable_modules=["c_attn"])
    defaults.update(overrides)
    lema_cfg = LemaConfig(**defaults)
    adapter = GPT2Adapter(cfg.to_dict())
    gbi = GlobalBinaryIndex(str(path))
    return FullFTManager(gbi, adapter, lema_cfg)


def test_clip_grad_norm_scales_when_above_max(tmp_path):
    mgr = make_manager(tmp_path)
    for acc in mgr.accumulators.values():
        acc.fill_(2.0)  # norm well above 1.0
    norm = mgr.clip_grad_norm_(1, max_norm=1.0)
    assert norm > 1.0
    # after clipping, norm should equal max_norm (within fp tolerance)
    clipped = math.sqrt(sum(a.float().pow(2).sum().item() for a in mgr.accumulators.values()))
    assert clipped == pytest.approx(1.0, rel=1e-4)


def test_clip_grad_norm_noop_below_max(tmp_path):
    mgr = make_manager(tmp_path)
    for acc in mgr.accumulators.values():
        acc.fill_(0.01)
    before = {k: v.clone() for k, v in mgr.accumulators.items()}
    norm = mgr.clip_grad_norm_(1, max_norm=1.0)
    assert norm <= 1.0
    for k, v in mgr.accumulators.items():
        assert torch.equal(v, before[k]), "clip must not touch below-max grads"


def test_step_layer_applies_weight_decay(tmp_path):
    mgr = make_manager(tmp_path, weight_decay=0.1, learning_rate=0.01)
    # zero grads so only decay acts
    for acc in mgr.accumulators.values():
        acc.zero_()
    w_before = {k: v.clone() for k, v in mgr.true_weights.items()}
    mgr.step_layer(1)
    for k, w in mgr.true_weights.items():
        # AdamW-style decoupled decay: w *= (1 - lr*wd) on the fp32 update
        expected = w_before[k].float() * (1 - 0.01 * 0.1)
        assert torch.allclose(w.float(), expected, atol=1e-6), f"weight decay not applied for {k}"


def test_auto_backend_picks_disk_when_ram_insufficient(tmp_path):
    mgr = make_manager(tmp_path, max_ram_gb=0.000001)  # tiny RAM budget
    assert mgr.accumulator_backend == "disk"