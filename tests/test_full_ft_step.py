import math
import torch
import pytest
from transformers import GPT2Config
from safetensors.torch import save_file
from lema._gbi import GlobalBinaryIndex
from lema.adapters import GPT2Adapter
from lema._config import LemaConfig
from lema._full_ft import FullFTManager


def make_manager(tmp_path, layer_id=1):
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                     attn_implementation="eager")
    import transformers
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path / "m.safetensors"
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(path))
    lema_cfg = LemaConfig(model_name_or_path=str(path), model_type="gpt2", gbi_path=str(path),
                          device="cpu", dtype="float32", training_mode="selective_full",
                          trainable_layers=[str(layer_id)], trainable_modules=["c_attn"])
    adapter = GPT2Adapter(cfg.to_dict())
    gbi = GlobalBinaryIndex(str(path))
    return FullFTManager(gbi, adapter, lema_cfg)


def test_apply_to_module_marks_frozen_and_loads_true_weights(tmp_path):
    mgr = make_manager(tmp_path)
    module = mgr.adapter.construct_layer_module(1, None)
    mgr.apply_to_module(1, module)
    frozen = [p for n, p in module.named_parameters() if not n.startswith("attn.c_attn")]
    trainable = [p for n, p in module.named_parameters() if n.startswith("attn.c_attn")]
    assert all(p.requires_grad is False for p in frozen)
    assert all(p.requires_grad is True for p in trainable)
    # true weights equal module weights for selected params
    for n, p in module.named_parameters():
        if n.startswith("attn.c_attn"):
            key = mgr.module_name_to_key[1][n]
            assert torch.allclose(p, mgr.true_weights[key])


def test_step_layer_matches_reference_adamw(tmp_path):
    mgr = make_manager(tmp_path)
    lr = 0.01
    wd = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Manual reference on CPU: one AdamW step from zero moments
    refs = {}
    for name, key in mgr.module_name_to_key[1].items():
        w0 = mgr.true_weights[key].clone()
        total = torch.randn_like(w0)
        m = (1 - beta1) * total
        v = (1 - beta2) * total * total
        mhat = m / (1 - beta1 ** 1)
        vhat = v / (1 - beta2 ** 1)
        w_new = w0 - lr * mhat / (torch.sqrt(vhat) + eps)
        refs[name] = (w0, w_new, total)

    # Feed the same grads through the manager
    for name, key in mgr.module_name_to_key[1].items():
        w0, w_new, total = refs[name]
        acc = mgr.get_accumulator(key)
        acc.copy_(total)
    mgr.config.learning_rate = lr
    mgr.config.weight_decay = wd
    mgr.step_layer(1)

    for name, key in mgr.module_name_to_key[1].items():
        w0, w_new, _ = refs[name]
        assert torch.allclose(mgr.true_weights[key], w_new, atol=1e-5), name
        assert mgr.get_accumulator(key).abs().sum() == 0  # zeroed after step


def test_get_trainable_parameters_returns_true_weights(tmp_path):
    mgr = make_manager(tmp_path)
    params = mgr.get_trainable_parameters()
    assert len(params) == len(mgr.true_weights)
    for p, (key, w) in zip(params, mgr.true_weights.items()):
        assert p is w
