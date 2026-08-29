import pytest
import torch
from lema._quant import quantize_tensor
from lema._quant_backend import is_backend_available, list_available_backends, resolve_backend, quantize_tensor_with_backend
from lema import LemaConfig


def test_custom_always_available():
    assert is_backend_available("custom") is True


def test_list_includes_custom():
    avail = list_available_backends()
    assert "custom" in avail


def test_resolve_auto_falls_back_to_custom_when_no_libs():
    name = resolve_backend("auto")
    assert name in ("custom", "torchao", "quanto", "bitsandbytes")


def test_resolve_explicit_custom_ok():
    assert resolve_backend("custom") == "custom"


def test_resolve_unknown_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend("unknown_backend")


def test_resolve_missing_torchao_raises_import_error():
    if not is_backend_available("torchao"):
        with pytest.raises(ImportError, match="pip install lema"):
            resolve_backend("torchao")


def test_resolve_missing_quanto_raises_import_error():
    if not is_backend_available("quanto"):
        with pytest.raises(ImportError, match="pip install lema"):
            resolve_backend("quanto")


def test_quantize_with_backend_custom_matches_direct():
    t = torch.randn(8, 16)
    q1, s1 = quantize_tensor(t, 8)
    q2, s2 = quantize_tensor_with_backend(t, 8, backend="custom")
    assert torch.equal(q1, q2)
    assert torch.allclose(s1.float(), s2.float())


def test_quantize_with_backend_auto_matches_custom_when_no_libs():
    if resolve_backend("auto") == "custom":
        t = torch.randn(4, 8)
        q1, s1 = quantize_tensor(t, 8)
        q2, s2 = quantize_tensor_with_backend(t, 8, backend="auto")
        assert torch.equal(q1, q2)


def test_config_quant_backend_field():
    cfg = LemaConfig(model_name_or_path="tmp", quant_backend="auto")
    assert cfg.quant_backend == "auto"
    cfg2 = LemaConfig(model_name_or_path="tmp", quant_backend="custom")
    assert cfg2.quant_backend == "custom"


def test_config_invalid_quant_backend_raises():
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="tmp", quant_backend="bad")


def test_config_roundtrip_with_quant_backend(tmp_path):
    cfg = LemaConfig(model_name_or_path="tmp", quant_backend="custom")
    cfg.save_pretrained(str(tmp_path))
    cfg2 = LemaConfig.from_pretrained(str(tmp_path))
    assert cfg2.quant_backend == "custom"
