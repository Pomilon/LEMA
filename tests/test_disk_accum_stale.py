import os
import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._config import TrainingMode


def _build(tmp_path, trainable_modules, trainable_layers, output_dir):
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2,
                     attn_implementation="eager")
    hf = GPT2LMHeadModel(cfg)
    sd = {k: v.clone().detach() for k, v in hf.state_dict().items()}
    model_dir = tmp_path / "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir / "model.safetensors"
    save_file(sd, str(model_path))
    cfg.save_pretrained(str(model_dir))
    lc = LemaConfig(
        model_name_or_path=str(model_dir), model_type="gpt2", gbi_path=str(model_path),
        device="cpu", strategy=MemoryStrategy.STREAMING, max_vram_gb=4.0,
        training_mode=TrainingMode.SELECTIVE_FULL,
        trainable_modules=trainable_modules, trainable_layers=trainable_layers,
        grad_accum_backend="disk", output_dir=output_dir,
    )
    return LemaModel(lc)


def test_disk_accumulator_reopen_with_different_selection(tmp_path):
    out = str(tmp_path / "run")

    # Run 1: select c_attn of last:1
    m1 = _build(tmp_path, ["c_attn"], ["last:1"], out)
    # accumulate a bit into run1's file
    m1.full_ft_manager.accumulators[list(m1.full_ft_manager.accumulators)[0]].fill_(0.5)
    m1.full_ft_manager.close()

    # Run 2: same output_dir, DIFFERENT selection (c_attn of last:2 -> more params)
    m2 = _build(tmp_path, ["c_attn"], ["last:2"], out)
    # the stale run1 file must not be reused: run2 accumulators must be zeroed
    for acc in m2.full_ft_manager.accumulators.values():
        assert torch.all(acc == 0), "stale disk accumulator reused across selections"
    m2.full_ft_manager.close()


def test_disk_accumulator_reopen_same_selection(tmp_path):
    out = str(tmp_path / "run")
    m1 = _build(tmp_path, ["c_attn"], ["last:1"], out)
    m1.full_ft_manager.accumulators[list(m1.full_ft_manager.accumulators)[0]].fill_(0.5)
    m1.full_ft_manager.close()

    # Same selection, same dir: accumulator should PERSIST (zero only on a fresh start)
    m2 = _build(tmp_path, ["c_attn"], ["last:1"], out)
    accs = list(m2.full_ft_manager.accumulators.values())
    assert any(torch.all(a == 0.5) for a in accs), "same-selection reopen should keep data"
    m2.full_ft_manager.close()