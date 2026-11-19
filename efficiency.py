"""Efficiency measurement (Table 9: average execution time per stage).

A tiny stage-timing accumulator: call `.record(stage, seconds)` as each
pipeline stage runs, then `.summary()` for the mean time and percentage
share of each stage, in the same shape as Table 9 in main.tex.
"""
from __future__ import annotations

from collections import defaultdict


class StageTimer:
    def __init__(self):
        self._times: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, seconds: float) -> None:
        self._times[stage].append(seconds)

    def summary(self) -> dict:
        means = {stage: sum(v) / len(v) for stage, v in self._times.items() if v}
        total = sum(means.values())
        return {
            stage: {
                "time_s": round(mean, 3),
                "pct_total": round(100 * mean / total, 1) if total else 0.0,
            }
            for stage, mean in means.items()
        } | {"_total_s": round(total, 3)}
