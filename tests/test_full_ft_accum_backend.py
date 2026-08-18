import os
import torch
from transformers import GPT2Config
from safetensors.torch import save_file
from lema._gbi import GlobalBinaryIndex
from lema.adapters import GPT2Adapter
from lema._config import LemaConfig
from lema._full_ft import FullFTManager


def build(tmp_path, grad_accum_backend, max_ram_gb=0.0):
    cfg = GPT2Config(vocab_size=64, n_positions=16, n_embd=16, n_layer=3, n_head=2,
                     attn_implementation="eager")
    import transformers
    model = transformers.GPT2LMHeadModel(cfg)
    path = tmp_path / "m.safetensors"
    save_file({k: v.contiguous().clone() for k, v in model.state_dict().items()}, str(path))
    lema_cfg = LemaConfig(model_name_or_path=str(path), model_type="gpt2", gbi_path=str(path),
                          device="cpu", training_mode="selective_full",
                          trainable_modules=["c_attn"], output_dir=str(tmp_path),
                          grad_accum_backend=grad_accum_backend, max_ram_gb=max_ram_gb)
    adapter = GPT2Adapter(cfg.to_dict())
    gbi = GlobalBinaryIndex(str(path))
    return FullFTManager(gbi, adapter, lema_cfg)


def test_ram_backend_forced(tmp_path):
    mgr = build(tmp_path, "ram")
    assert mgr.accumulator_backend == "ram"
    key = mgr.module_name_to_key[1]["attn.c_attn.weight"]
    acc = mgr.get_accumulator(key)
    assert isinstance(acc, torch.Tensor)
    acc.add_(1.0)
    assert mgr.get_accumulator(key).sum().item() == acc.numel()


def test_disk_backend_forced(tmp_path):
    mgr = build(tmp_path, "disk")
    assert mgr.accumulator_backend == "disk"
    assert os.path.isdir(os.path.join(tmp_path, "grad_accum"))
    key = mgr.module_name_to_key[1]["attn.c_attn.weight"]
    acc = mgr.get_accumulator(key)
    assert isinstance(acc, torch.Tensor)
    acc.add_(2.0)
    # write-through persists to the mmap file
    assert mgr.get_accumulator(key).sum().item() == 2.0 * acc.numel()
    mgr.close()
    # reopening the manager re-reads persisted values from disk
    mgr2 = build(tmp_path, "disk")
    key2 = mgr2.module_name_to_key[1]["attn.c_attn.weight"]
    assert mgr2.get_accumulator(key2).sum().item() == 2.0 * acc.numel()
    mgr2.close()


def test_auto_backend_picks_ram_when_fits(tmp_path):
    mgr = build(tmp_path, "auto", max_ram_gb=64.0)
    assert mgr.accumulator_backend == "ram"
