"""Intervention-to-conditioning signal mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG, ConditioningSignal, Intervention


class ConditioningModel(Protocol):
    def build(self, intervention: Intervention, dag: BayesianDAG) -> ConditioningSignal:
        """Map intervention into numeric conditioning signal."""


@dataclass
class DenseConditioningModel(ConditioningModel):
    signal_dim: int = 16

    def build(self, intervention: Intervention, dag: BayesianDAG) -> ConditioningSignal:
        # Hash-based deterministic embedding keeps this backend stateless.
        vector = np.zeros((self.signal_dim,), dtype=np.float32)
        text = f"{intervention.variable}:{intervention.value}:{intervention.aggressiveness}"
        for i, ch in enumerate(text.encode("utf-8")):
            idx = i % self.signal_dim
            vector[idx] += (float(ch % 31) / 30.0) - 0.5

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return ConditioningSignal(
            vector=vector,
            metadata={
                "intervention": intervention.description,
                "variable": intervention.variable,
                "value": intervention.value,
            },
        )
