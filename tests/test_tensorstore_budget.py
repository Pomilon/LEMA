import torch
import pytest
from lema._config import LemaConfig
from lema._tensorstore import TensorStore, StreamKind, parse_vram_setting


def test_parse_vram_setting():
    assert parse_vram_setting("auto", 16.0) == 16.0 / 4
    assert parse_vram_setting("0.3", 16.0) == pytest.approx(4.8)
    assert parse_vram_setting("4.0GB", 16.0) == pytest.approx(4.0)


def test_with_budget_splits_kinds():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=16.0,
                   weights_vram="0.25", kv_vram="2.0GB")
    store = TensorStore.with_budget(c)
    assert store.kind_budget(StreamKind.WEIGHTS) == pytest.approx(4.0)
    assert store.kind_budget(StreamKind.KV_CHUNK) == pytest.approx(2.0)
    assert store.kind_budget(StreamKind.OPT_STATE) > 0


def test_caps_bounded():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=16.0,
                   weights_vram="0.9", kv_vram="8.0GB", opt_state_vram="8.0GB",
                   grad_acc_vram="8.0GB")
    store = TensorStore.with_budget(c)
    total = sum(store.kind_budget(k) for k in StreamKind)
    assert total <= 16.0 + 1e-6
