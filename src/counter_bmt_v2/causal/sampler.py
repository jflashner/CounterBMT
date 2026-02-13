"""Intervention sampling from Bayesian DAG state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG, Intervention


class InterventionSampler(Protocol):
    def sample(self, dag: BayesianDAG, *, rare: bool = False, seed: int = 0) -> Intervention:
        """Sample one intervention from DAG."""


@dataclass
class SimpleInterventionSampler(InterventionSampler):
    fallback_value: str = "straight"

    def sample(self, dag: BayesianDAG, *, rare: bool = False, seed: int = 0) -> Intervention:
        rng = np.random.default_rng(seed)
        candidates: List[str] = [
            node_id
            for node_id, node in dag.nodes.items()
            if node.node_type in {"maneuver", "decision"}
        ]
        if not candidates:
            return Intervention(variable="maneuver_0", value=self.fallback_value, description="fallback intervention")

        node_id = candidates[int(rng.integers(0, len(candidates)))]
        node = dag.nodes[node_id]

        alternatives = node.metadata.get("alternatives", [])
        if not alternatives:
            alternatives = [node.value]

        options = [x for x in alternatives if x != node.value] or alternatives
        value = options[int(rng.integers(0, len(options)))]

        if rare and node.node_type == "maneuver":
            # Bias toward stronger perturbations for tail exploration.
            for pref in ["stop", "lane_change_left", "lane_change_right"]:
                if pref in options:
                    value = pref
                    break

        return Intervention(
            variable=node_id,
            value=value,
            original_value=node.value,
            timestamp_s=node.timestamp_s,
            aggressiveness=node.metadata.get("aggressiveness", "normal"),
            description=f"sampled intervention on {node_id}: {node.value} -> {value}",
        )
