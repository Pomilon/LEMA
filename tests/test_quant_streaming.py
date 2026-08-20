import pytest
from lema import LemaConfig


def test_quant_bits_validation_accepts_off_and_valid():
    LemaConfig(model_name_or_path="x", weights_bits=None, opt_state_bits=0,
               grad_acc_bits=8, kv_bits=8)
    LemaConfig(model_name_or_path="x", weights_bits=4)
    LemaConfig(model_name_or_path="x", weights_bits=8)


def test_quant_bits_validation_rejects_invalid():
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", weights_bits=5)
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", opt_state_bits=4)  # 4 only for weights
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", kv_bits=4)  # int8 only for KV
    with pytest.raises(ValueError):
        LemaConfig(model_name_or_path="x", grad_acc_bits=16)


def test_quant_bits_serialize_roundtrip(tmp_path):
    c = LemaConfig(model_name_or_path="x", weights_bits=8, kv_bits=8)
    c.save_pretrained(str(tmp_path))
    c2 = LemaConfig.from_pretrained(str(tmp_path))
    assert c2.weights_bits == 8 and c2.kv_bits == 8
