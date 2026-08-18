from __future__ import annotations

from dataclasses import dataclass, field

from ._config import LemaConfig
from ._tensorstore import StreamKind, parse_vram_setting

_PRIORITY = [StreamKind.WEIGHTS, StreamKind.KV_CHUNK, StreamKind.OPT_STATE, StreamKind.GRAD_ACC]
_FIELD_MAP = {
    StreamKind.WEIGHTS: "weights_vram",
    StreamKind.OPT_STATE: "opt_state_vram",
    StreamKind.GRAD_ACC: "grad_acc_vram",
    StreamKind.KV_CHUNK: "kv_vram",
}


@dataclass
class BudgetReport:
    per_kind_gb: dict[StreamKind, float] = field(default_factory=dict)
    predicted_step_ms: float = 0.0
    target_met: bool = True


class BudgetEngine:
    def tune(self, config: LemaConfig, measured: dict) -> BudgetReport:
        explicit = {}
        for kind, fname in _FIELD_MAP.items():
            val = getattr(config, fname, "auto")
            if val != "auto":
                explicit[kind] = parse_vram_setting(val, config.max_vram_gb)
        if sum(explicit.values()) > config.max_vram_gb:
            scale = config.max_vram_gb / sum(explicit.values())
            explicit = {k: v * scale for k, v in explicit.items()}
        if config.target_step_time_ms > 0:
            report = self._search_target(config, measured, explicit, config.max_vram_gb)
        else:
            report = self._max_throughput(config, measured, explicit, config.max_vram_gb)
        return report

    def model_step_time(self, report: BudgetReport, measured: dict) -> float:
        disk_mb = measured["disk_mb_per_s"]
        comp_ms = measured["t_comp_layer_ms"]
        layer_bytes = measured["layer_bytes"]
        chunks = measured["chunks"]
        kv_bytes = measured["kv_bytes_per_chunk"]

        w_gb = report.per_kind_gb.get(StreamKind.WEIGHTS, 0.0)
        streamed_weight_mb = layer_bytes / 2**20 if w_gb * 2**30 < layer_bytes else 0.0

        kv_resident_bytes = report.per_kind_gb.get(StreamKind.KV_CHUNK, 0.0) * 2**30
        kv_total_mb = chunks * kv_bytes / 2**20
        kv_streamed_mb = max(kv_total_mb - kv_resident_bytes / 2**20, 0.0)

        io_ms = (streamed_weight_mb + kv_streamed_mb) / disk_mb * 1000
        return comp_ms + io_ms

    def _max_throughput(self, config, measured, explicit, cap):
        budget = dict(explicit)
        remaining = cap - sum(explicit.values())
        for kind in _PRIORITY:
            if kind not in budget and remaining > 0:
                budget[kind] = remaining / max(len(_PRIORITY) - len(budget), 1)
                remaining -= budget[kind]
        rep = BudgetReport(per_kind_gb=budget)
        rep.predicted_step_ms = self.model_step_time(rep, measured)
        return rep

    def _search_target(self, config, measured, explicit, cap):
        lo, hi = 0.0, config.max_vram_gb
        best = self._max_throughput(config, measured, explicit, cap)
        for _ in range(16):
            mid = (lo + hi) / 2
            trial = self._trial_report(measured, explicit, cap, mid)
            t = self.model_step_time(trial, measured)
            if t <= config.target_step_time_ms:
                best = trial
                hi = mid
            else:
                lo = mid
        if best.predicted_step_ms <= config.target_step_time_ms:
            best.target_met = True
        else:
            best = self._max_throughput(config, measured, explicit, cap)
            best.target_met = False
        return best

    def _trial_report(self, measured, explicit, cap, scale):
        budget = dict(explicit)
        auto_kinds = [k for k in _PRIORITY if k not in budget]
        if auto_kinds:
            total = max(cap - sum(explicit.values()), 0.0)
            share = total * scale
            first = auto_kinds[0]
            budget[first] = share
            for k in auto_kinds[1:]:
                budget[k] = share / max(len(auto_kinds) - 1, 1)
        return BudgetReport(per_kind_gb=budget)
