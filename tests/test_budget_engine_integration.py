import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy


def _build(tmp_path):
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=100, n_positions=64, n_embd=32, n_layer=2, n_head=2,
        attn_implementation="eager",
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


def test_tune_budgets_sets_store(tmp_path):
    model = _build(tmp_path)
    model.tune_budgets()
    assert model.store._budgets is not None
    assert len(model.store._budgets) == 4
    assert model.config.prefetch_distance >= 1


def test_tune_budgets_target_respected(tmp_path):
    model = _build(tmp_path)
    model.config.target_step_time_ms = 10000.0
    model.tune_budgets()
    assert model.store._budgets is not None
