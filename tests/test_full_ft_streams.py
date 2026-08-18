import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._config import TrainingMode
from lema._tensorstore import StreamKind


def _build_full_ft_model(tmp_path):
    config = GPT2Config(
        vocab_size=100, n_positions=32, n_embd=32, n_layer=2, n_head=2,
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
        model_name_or_path=str(model_dir),
        model_type="gpt2",
        gbi_path=str(model_path),
        device="cpu",
        strategy=MemoryStrategy.STREAMING,
        max_vram_gb=4.0,
        training_mode=TrainingMode.SELECTIVE_FULL,
        trainable_layers=["last:1"],
        trainable_modules=["c_attn"],
    )
    return LemaModel(lema_config)


def test_fullft_registers_streams(tmp_path):
    model = _build_full_ft_model(tmp_path)
    mgr = model.full_ft_manager
    assert mgr is not None
    assert mgr.store is not None
    weight_keys = [k for k in mgr.store.streams() if k[0] == StreamKind.WEIGHTS]
    assert len(weight_keys) > 0
    k = None
    for cand in weight_keys:
        if cand[1:] in mgr.true_weights:
            k = cand
            break
    assert k is not None, "no store weight stream matches a selected full-FT weight"
    got = mgr.store.ensure(k)[k]
    assert tuple(got.shape) == tuple(mgr.true_weights[k[1:]].shape)


def test_fullft_states_and_acc_registered(tmp_path):
    model = _build_full_ft_model(tmp_path)
    mgr = model.full_ft_manager
    kinds = {k[0] for k in mgr.store.streams()}
    assert StreamKind.OPT_STATE in kinds
    assert StreamKind.GRAD_ACC in kinds
