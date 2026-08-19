import pytest
from lema._config import LemaConfig, TrainingMode


def test_invalid_grad_accum_backend_raises():
    with pytest.raises(ValueError, match="grad_accum_backend"):
        LemaConfig(model_name_or_path="x", grad_accum_backend="ramm")


def test_last_zero_raises():
    with pytest.raises(ValueError, match="last"):
        LemaConfig(model_name_or_path="x", training_mode=TrainingMode.SELECTIVE_FULL,
                   trainable_layers=["last:0"])


def test_first_negative_raises():
    with pytest.raises(ValueError, match="first"):
        LemaConfig(model_name_or_path="x", training_mode=TrainingMode.SELECTIVE_FULL,
                   trainable_layers=["first:-1"])


def test_last_positive_ok():
    c = LemaConfig(model_name_or_path="x", training_mode=TrainingMode.SELECTIVE_FULL,
                   trainable_layers=["last:2"])
    assert c.trainable_layers == ["last:2"]


def test_valid_backends_ok():
    for b in ("auto", "ram", "disk"):
        c = LemaConfig(model_name_or_path="x", grad_accum_backend=b)
        assert c.grad_accum_backend == b