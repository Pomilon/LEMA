import torch
import pytest
from lema._config import LemaConfig
from lema._budget_engine import BudgetEngine, BudgetReport
from lema._tensorstore import StreamKind

MEASURED = {
    "disk_mb_per_s": 2000.0,
    "pcie_mb_per_s": 12000.0,
    "t_comp_layer_ms": 50.0,
    "layer_bytes": 4 * 2**20,
    "kv_bytes_per_chunk": 2 * 2**20,
    "chunks": 16,
}


def test_no_target_maximizes_throughput():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=16.0)
    rep = BudgetEngine().tune(c, MEASURED)
    assert rep.per_kind_gb[StreamKind.WEIGHTS] >= rep.per_kind_gb[StreamKind.KV_CHUNK]


def test_target_reduces_budget():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=16.0, target_step_time_ms=300.0)
    rep = BudgetEngine().tune(c, MEASURED)
    assert rep.predicted_step_ms <= 300.0 + 1e-6


def test_impossible_target_warns_and_maximizes():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=1.0, target_step_time_ms=1.0)
    rep = BudgetEngine().tune(c, MEASURED)
    assert rep.predicted_step_ms > 1.0
    assert rep.per_kind_gb[StreamKind.WEIGHTS] > 0
    assert rep.target_met is False


def test_override_wins():
    c = LemaConfig(model_name_or_path="x", max_vram_gb=16.0, kv_vram="0.5GB")
    rep = BudgetEngine().tune(c, MEASURED)
    assert rep.per_kind_gb[StreamKind.KV_CHUNK] == pytest.approx(0.5)
