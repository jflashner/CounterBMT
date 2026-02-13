"""Causal DAG construction for CounterBMT v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG, DAGEdge, DAGNode, ScenarioInput, VLMFeatures


class DAGBuilder(Protocol):
    def build(self, scene: ScenarioInput, features: VLMFeatures) -> BayesianDAG:
        """Build Bayesian DAG from extracted features and scene state."""


@dataclass
class SimpleDAGBuilder(DAGBuilder):
    """Deterministic DAG builder with minimal CPTs for fast iteration."""

    collision_node_id: str = "collision_outcome"

    def build(self, scene: ScenarioInput, features: VLMFeatures) -> BayesianDAG:
        dag = BayesianDAG(scenario_id=scene.scenario_id)

        init_speed = 10.0
        if scene.ego_trajectory_xy is not None and len(scene.ego_trajectory_xy) > 1:
            d = np.linalg.norm(scene.ego_trajectory_xy[1] - scene.ego_trajectory_xy[0])
            init_speed = float(d / 0.1)

        dag.nodes["ego_initial_speed"] = DAGNode(
            node_id="ego_initial_speed",
            node_type="ego_state",
            value=init_speed,
            timestamp_s=0.0,
        )

        for i, m in enumerate(features.maneuvers):
            node_id = f"maneuver_{i}"
            dag.nodes[node_id] = DAGNode(
                node_id=node_id,
                node_type="maneuver",
                value=m.maneuver_type.value,
                timestamp_s=m.start_s,
                metadata={
                    "alternatives": [
                        "straight",
                        "lane_change_left",
                        "lane_change_right",
                        "stop",
                    ]
                },
            )
            dag.edges.append(
                DAGEdge(
                    parent_id="ego_initial_speed",
                    child_id=node_id,
                    confidence=0.7,
                    mechanism="initial speed constrains maneuver choice",
                )
            )

        for i, d in enumerate(features.decisions):
            node_id = f"decision_{i}"
            dag.nodes[node_id] = DAGNode(
                node_id=node_id,
                node_type="decision",
                value=d.choice,
                timestamp_s=d.timestamp_s,
                metadata={"alternatives": d.alternatives or [d.choice]},
            )
            dag.edges.append(
                DAGEdge(
                    parent_id="ego_initial_speed",
                    child_id=node_id,
                    confidence=0.65,
                    mechanism="speed influences decision urgency",
                )
            )

        dag.nodes[self.collision_node_id] = DAGNode(
            node_id=self.collision_node_id,
            node_type="outcome",
            value="collision_avoided",
            metadata={"alternatives": ["collision_avoided", "collision_possible"]},
        )

        for node_id, node in dag.nodes.items():
            if node.node_type in {"maneuver", "decision"}:
                dag.edges.append(
                    DAGEdge(
                        parent_id=node_id,
                        child_id=self.collision_node_id,
                        confidence=0.8,
                        mechanism="event impacts collision risk",
                    )
                )

        # Minimal CPT only on outcome for now.
        dag.cpts[self.collision_node_id] = {
            "values": ["collision_avoided", "collision_possible"],
            "parents": [n.node_id for n in dag.nodes.values() if n.node_type in {"maneuver", "decision"}],
            "cpt": {
                "*": {"collision_avoided": 0.85, "collision_possible": 0.15}
            },
        }
        return dag
