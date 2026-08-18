import os
import shutil

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from safetensors.torch import save_file

from lema import LemaModel, LemaConfig, MemoryStrategy
from lema._tensorstore import StreamKind


def _build_tiny_model(tmp_path):
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
    )
    return LemaModel(lema_config)


def test_store_holds_weight_streams(tmp_path):
    model = _build_tiny_model(tmp_path)
    assert model.store is not None
    keys = [k for k in model.store.streams() if k[0] == StreamKind.WEIGHTS]
    assert len(keys) > 0
    tensors = model.store.ensure(*keys)
    k = keys[0]
    shape = model.gbi.get_tensor_shape(k[2])
    assert tuple(tensors[k].shape) == tuple(shape)


def test_store_budget_from_config(tmp_path):
    model = _build_tiny_model(tmp_path)
    assert model.store.kind_budget(StreamKind.WEIGHTS) > 0
