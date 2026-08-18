import pytest
from lema._config import LemaConfig, MemoryStrategy, TrainingMode, StateStrategy

def test_training_mode_default_is_lora():
    cfg = LemaConfig(model_name_or_path="x")
    assert cfg.training_mode == TrainingMode.LORA

def test_training_mode_string_coerced():
    cfg = LemaConfig(model_name_or_path="x", training_mode="selective_full")
    assert cfg.training_mode == TrainingMode.SELECTIVE_FULL

def test_state_strategy_default_streaming():
    cfg = LemaConfig(model_name_or_path="x")
    assert cfg.state_strategy == StateStrategy.STREAMING

def test_full_ft_fields_defaults():
    cfg = LemaConfig(model_name_or_path="x")
    assert cfg.trainable_modules == []
    assert cfg.trainable_layers == []
    assert cfg.grad_accum_backend == "auto"
    assert cfg.save_optimizer is True
    assert cfg.weight_decay == 0.01

def test_to_dict_serializes_enums_as_values():
    cfg = LemaConfig(model_name_or_path="x", training_mode="selective_full", state_strategy="vram")
    d = cfg.to_dict()
    assert d["training_mode"] == "selective_full"
    assert d["state_strategy"] == "vram"
