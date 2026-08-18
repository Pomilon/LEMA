import pytest
from lema._config import LemaConfig


def test_new_budget_fields_defaults():
    c = LemaConfig(model_name_or_path="x")
    assert c.weights_vram == "auto"
    assert c.opt_state_vram == "auto"
    assert c.grad_acc_vram == "auto"
    assert c.kv_vram == "auto"
    assert c.target_step_time_ms == 0.0
    assert c.target_tokens_per_sec == 0.0
    assert c.kv_chunk_size == 8192


def test_budget_fields_survive_roundtrip(tmp_path):
    c = LemaConfig(model_name_or_path="x", weights_vram="0.3", kv_vram="4.0GB",
                   target_step_time_ms=250.0, kv_chunk_size=4096)
    c.save_pretrained(str(tmp_path))
    c2 = LemaConfig.from_pretrained(str(tmp_path))
    assert c2.weights_vram == "0.3"
    assert c2.kv_vram == "4.0GB"
    assert c2.target_step_time_ms == 250.0
    assert c2.kv_chunk_size == 4096
