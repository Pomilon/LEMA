import os
import torch
from safetensors.torch import save_file

from lema._utils._conversion import merge_delta


def test_merge_delta_creates_parent_dir(tmp_path):
    base = tmp_path / "base.safetensors"
    save_file({"w": torch.ones(4)}, str(base))
    delta = tmp_path / "delta.safetensors"
    save_file({"w": torch.ones(4)}, str(delta))

    out = tmp_path / "nested" / "deep" / "merged.safetensors"
    merge_delta(str(base), str(delta), str(out))

    assert os.path.exists(out)
    from safetensors import safe_open
    with safe_open(str(out), framework="pt", device="cpu") as f:
        assert torch.equal(f.get_tensor("w"), torch.full((4,), 2.0))