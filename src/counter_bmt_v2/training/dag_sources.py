"""DAG source adapters for supervised DAG-latent training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache import DAGCacheReader
from counter_bmt_v2.training.dag_cache_schema import SCHEMA_VERSION


def _safe_json_scalar(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def _node_entry(node_id: str, node_type: str, value: Any, timestamp_s: float | None, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": str(node_id),
        "node_type": str(node_type),
        "value": _safe_json_scalar(value),
        "timestamp_s": None if timestamp_s is None else float(timestamp_s),
        "metadata": {str(k): _safe_json_scalar(v) for k, v in dict(metadata).items()},
    }


@dataclass
class SceneDerivedDAGBuilder:
    """Deterministic DAG builder from collated ScenarioNet batch slices."""

    dt_default: float = 0.1
    contract: DAGContractConfig = DAGContractConfig(name="compact10", mode="hard")

    def build(self, *, scenario_id: str, batch_slice: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
        pos = np.asarray(batch_slice["agent_position_xy"], dtype=np.float32)  # [T,N,2]
        vel = np.asarray(batch_slice["agent_velocity_xy"], dtype=np.float32)  # [T,N,2]
        valid = np.asarray(batch_slice["agent_valid_mask"], dtype=bool)  # [T,N]
        heading = np.asarray(batch_slice["agent_heading"], dtype=np.float32)  # [T,N]
        dt_s = float(np.asarray(batch_slice.get("dt_s", self.dt_default), dtype=np.float32))
        dt_s = float(max(dt_s, 1e-3))

        if pos.ndim != 3 or pos.shape[1] == 0:
            ego_speed0 = 0.0
            progress = 0.0
            turn_rate = 0.0
            risk_proxy = 0.0
        else:
            ego_valid = valid[:, 0]
            ego_pos = pos[:, 0, :]
            ego_vel = vel[:, 0, :]
            ego_heading = heading[:, 0]
            if np.any(ego_valid):
                idx = np.where(ego_valid)[0]
                first, last = int(idx[0]), int(idx[-1])
                ego_speed0 = float(np.linalg.norm(ego_vel[first]))
                progress = float(ego_pos[last, 0] - ego_pos[first, 0])
                dhead = np.diff(ego_heading[idx]) if idx.size > 1 else np.zeros((0,), dtype=np.float32)
                turn_rate = float(np.mean(np.abs(((dhead + np.pi) % (2.0 * np.pi)) - np.pi))) if dhead.size else 0.0
                acc = np.diff(ego_vel[idx], axis=0) if idx.size > 1 else np.zeros((0, 2), dtype=np.float32)
                jerk = np.diff(acc, axis=0) if acc.shape[0] > 1 else np.zeros((0, 2), dtype=np.float32)
                jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0
                risk_proxy = float(np.clip(0.5 * turn_rate + 0.5 * jerk_mag, 0.0, 1.0))
            else:
                ego_speed0 = 0.0
                progress = 0.0
                turn_rate = 0.0
                risk_proxy = 0.0

        maneuver_class = "straight"
        if turn_rate > 0.2:
            maneuver_class = "left_turn"
        if progress < 0.5:
            maneuver_class = "stop"
        decision_class = "maintain_speed"
        if risk_proxy > 0.65:
            decision_class = "decelerate"
        elif risk_proxy < 0.2 and ego_speed0 < 4.0:
            decision_class = "accelerate"
        outcome_class = "collision_possible" if risk_proxy > 0.75 else "collision_avoided"

        nodes: List[Dict[str, Any]] = [
            _node_entry("ego_initial_speed", "ego_state", ego_speed0, 0.0, {"alternatives": "scalar_speed"}),
            _node_entry(
                "maneuver_0",
                "maneuver",
                maneuver_class,
                dt_s,
                {
                    "alternatives": ",".join(
                        ["straight", "left_turn", "right_turn", "lane_change_left", "lane_change_right", "stop"]
                    )
                },
            ),
            _node_entry(
                "decision_0",
                "decision",
                decision_class,
                2.0 * dt_s,
                {"alternatives": ",".join(["maintain_speed", "accelerate", "decelerate", "yield_or_proceed"])},
            ),
            _node_entry("risk_0", "risk", float(risk_proxy), 3.0 * dt_s, {"alternatives": "scalar_risk_0_1"}),
            _node_entry(
                "collision_outcome",
                "outcome",
                outcome_class,
                None,
                {"alternatives": ",".join(["collision_avoided", "collision_possible"])},
            ),
        ]

        edges = [
            {"parent_id": "ego_initial_speed", "child_id": "maneuver_0", "confidence": 0.7, "mechanism": "speed_to_maneuver"},
            {"parent_id": "maneuver_0", "child_id": "decision_0", "confidence": 0.8, "mechanism": "maneuver_to_decision"},
            {"parent_id": "ego_initial_speed", "child_id": "risk_0", "confidence": 0.7, "mechanism": "context_to_decision"},
            {"parent_id": "risk_0", "child_id": "collision_outcome", "confidence": 0.85, "mechanism": "risk_to_outcome"},
            {"parent_id": "decision_0", "child_id": "collision_outcome", "confidence": 0.8, "mechanism": "decision_to_outcome"},
        ]

        cpts = {
            "collision_outcome": {
                "values": ["collision_avoided", "collision_possible"],
                "parents": ["decision_0", "risk_0"],
                "cpt": {
                    "*": {"collision_avoided": 0.85, "collision_possible": 0.15},
                },
            }
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scenario_id": str(scenario_id),
            "nodes": nodes,
            "edges": edges,
            "cpts": cpts,
            "metadata": {"source": "scene_derived", "sample_index": int(sample_index)},
        }
        ok, canonical, report = enforce_dag_contract(payload, config=self.contract)
        if not ok:
            raise RuntimeError(
                "Scene-derived DAG failed compact contract: "
                f"scenario_id={scenario_id} violations={report.violation_counts}"
            )
        canonical["schema_version"] = SCHEMA_VERSION
        return canonical


@dataclass
class DAGSourceResolver:
    mode: str = "dual"  # dual|cache|scene_derived
    cache_dir: str = ""
    cache_strict: bool = False

    def __post_init__(self) -> None:
        self.mode = str(self.mode)
        self.cache = DAGCacheReader(self.cache_dir) if str(self.cache_dir).strip() else None
        self.scene_builder = SceneDerivedDAGBuilder()

    def resolve_one(
        self,
        *,
        scenario_id: str,
        batch_slice: Dict[str, Any],
        sample_index: int,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        mode = self.mode
        from_cache = None
        if self.cache is not None:
            from_cache = self.cache.get(str(scenario_id))

        if mode == "cache":
            if from_cache is not None:
                return from_cache, "cache"
            if self.cache_strict:
                return None, "cache_miss_strict"
            return None, "cache_miss"

        if mode == "scene_derived":
            return self.scene_builder.build(
                scenario_id=str(scenario_id),
                batch_slice=batch_slice,
                sample_index=int(sample_index),
            ), "scene_derived"

        # dual
        if from_cache is not None:
            return from_cache, "cache"
        return self.scene_builder.build(
            scenario_id=str(scenario_id),
            batch_slice=batch_slice,
            sample_index=int(sample_index),
        ), "scene_derived"
