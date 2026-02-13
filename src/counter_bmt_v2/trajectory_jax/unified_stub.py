"""Stub interface for eventual unified LLM+trajectory backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from counter_bmt_v2.contracts import ConditioningSignal, ScenarioInput


@dataclass
class UnifiedBackboneOutput:
    conditioning: ConditioningSignal
    trajectory_latent: np.ndarray
    metadata: Dict[str, str]


class UnifiedLLMTrajectoryBackboneStub:
    """Design placeholder for a future shared language+trajectory model.

    Current stack keeps this explicit so we can migrate from a two-stage setup:
    planner -> conditioning -> trajectory model
    to a unified setup with a shared encoder and multiple heads.
    """

    def forward(self, scene: ScenarioInput, prompt: str) -> UnifiedBackboneOutput:
        # Placeholder behavior: deterministic pseudo-conditioning from prompt text.
        vec = np.zeros((16,), dtype=np.float32)
        for i, b in enumerate(prompt.encode("utf-8")):
            vec[i % 16] += (float(b % 29) / 28.0) - 0.5
        n = np.linalg.norm(vec)
        if n > 0:
            vec = vec / n

        return UnifiedBackboneOutput(
            conditioning=ConditioningSignal(vector=vec, metadata={"source": "unified_stub"}),
            trajectory_latent=vec.copy(),
            metadata={"status": "stub", "note": "replace with NNX unified backbone"},
        )
