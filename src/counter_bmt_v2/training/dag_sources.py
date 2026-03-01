"""DAG source adapters for supervised DAG-latent training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache import DAGCacheReader
from counter_bmt_v2.training.dag_cache_schema import schema_version_for_contract


def _safe_json_scalar(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _safe_json_scalar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe_json_scalar(v) for v in x]
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
    contract: DAGContractConfig = DAGContractConfig(name="maneuver_outcome_v1", mode="hard")

    def build(self, *, scenario_id: str, batch_slice: Dict[str, Any], sample_index: int) -> Dict[str, Any]:
        pos = np.asarray(batch_slice["agent_position_xy"], dtype=np.float32)  # [T,N,2]
        vel = np.asarray(batch_slice["agent_velocity_xy"], dtype=np.float32)  # [T,N,2]
        valid = np.asarray(batch_slice["agent_valid_mask"], dtype=bool)  # [T,N]
        heading = np.asarray(batch_slice["agent_heading"], dtype=np.float32)  # [T,N]
        dt_s = float(np.asarray(batch_slice.get("dt_s", self.dt_default), dtype=np.float32))
        dt_s = float(max(dt_s, 1e-3))
        maneuver_classes = [
            "straight",
            "left_turn",
            "right_turn",
            "lane_change_left",
            "lane_change_right",
            "stop",
            "accelerate",
            "decelerate",
            "yield",
            "merge",
            "u_turn",
            "park",
        ]

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        progress = 0.0
        risk_proxy = 0.0
        compliance_proxy = 1.0
        if pos.ndim == 3 and pos.shape[1] > 0 and valid.ndim == 2 and np.any(valid[:, 0]):
            ego_valid = valid[:, 0]
            ego_idx = np.where(ego_valid)[0]
            ego_pos = pos[:, 0, :]
            ego_vel = vel[:, 0, :]
            ego_heading = heading[:, 0]
            first, last = int(ego_idx[0]), int(ego_idx[-1])
            progress = float(np.linalg.norm(ego_pos[last] - ego_pos[first]))

            # Split valid trajectory into up to 3 maneuver segments for deterministic coverage.
            chunks = np.array_split(ego_idx, min(3, int(max(1, ego_idx.size))))
            m_i = 0
            for chunk in chunks:
                if chunk.size <= 0:
                    continue
                c0, c1 = int(chunk[0]), int(chunk[-1])
                seg_vel = ego_vel[chunk]
                speed = np.linalg.norm(seg_vel, axis=1) if seg_vel.ndim == 2 else np.zeros((0,), dtype=np.float32)
                mean_speed = float(np.mean(speed)) if speed.size else 0.0
                dhead = np.diff(ego_heading[chunk]) if chunk.size > 1 else np.zeros((0,), dtype=np.float32)
                turn_mag = (
                    float(np.mean(np.abs(((dhead + np.pi) % (2.0 * np.pi)) - np.pi)))
                    if dhead.size
                    else 0.0
                )
                acc = np.diff(speed) if speed.size > 1 else np.zeros((0,), dtype=np.float32)
                acc_mean = float(np.mean(acc)) if acc.size else 0.0

                if mean_speed < 0.5:
                    m_class = "stop"
                elif turn_mag > 0.55:
                    m_class = "u_turn"
                elif turn_mag > 0.22:
                    m_class = "left_turn" if float(np.mean(dhead)) >= 0.0 else "right_turn"
                elif acc_mean > 0.25:
                    m_class = "accelerate"
                elif acc_mean < -0.25:
                    m_class = "decelerate"
                else:
                    m_class = "straight"

                start_s = float(c0 * dt_s)
                end_s = float(c1 * dt_s)
                duration_s = float(max(0.0, end_s - start_s))
                mid_s = float(0.5 * (start_s + end_s))
                nodes.append(
                    _node_entry(
                        f"maneuver_{m_i}",
                        "maneuver",
                        m_class,
                        mid_s,
                        {
                            "alternatives": maneuver_classes,
                            "start_s": start_s,
                            "end_s": end_s,
                            "duration_s": duration_s,
                            "mid_s": mid_s,
                            "observed": m_i == 0,
                        },
                    )
                )
                m_i += 1

            turn_rate = 0.0
            if ego_idx.size > 1:
                dhead_all = np.diff(ego_heading[ego_idx])
                if dhead_all.size > 0:
                    turn_rate = float(np.mean(np.abs(((dhead_all + np.pi) % (2.0 * np.pi)) - np.pi)))
            jerk = np.diff(np.diff(ego_vel[ego_idx], axis=0), axis=0) if ego_idx.size > 2 else np.zeros((0, 2), dtype=np.float32)
            jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0
            risk_proxy = float(np.clip(0.55 * turn_rate + 0.45 * jerk_mag, 0.0, 1.0))
            compliance_proxy = float(np.clip(1.0 - (0.7 * turn_rate + 0.3 * jerk_mag), 0.0, 1.0))

        if not nodes:
            nodes.append(
                _node_entry(
                    "maneuver_0",
                    "maneuver",
                    "straight",
                    0.0,
                    {
                        "alternatives": maneuver_classes,
                        "start_s": 0.0,
                        "end_s": 0.0,
                        "duration_s": 0.0,
                        "mid_s": 0.0,
                        "observed": True,
                    },
                )
            )

        collision_value = "collision_possible" if risk_proxy > 0.75 else "collision_avoided"
        progress_value = "progress_good" if progress > 8.0 else "progress_limited"
        compliance_value = "compliant" if compliance_proxy >= 0.5 else "violation_possible"

        nodes.extend(
            [
                _node_entry(
                    "collision_outcome",
                    "outcome",
                    collision_value,
                    None,
                    {"alternatives": ["collision_avoided", "collision_possible"], "observed": True},
                ),
                _node_entry(
                    "progress_outcome",
                    "outcome",
                    progress_value,
                    None,
                    {"alternatives": ["progress_good", "progress_limited"], "observed": True},
                ),
                _node_entry(
                    "compliance_outcome",
                    "outcome",
                    compliance_value,
                    None,
                    {"alternatives": ["compliant", "violation_possible"], "observed": True},
                ),
            ]
        )

        maneuver_ids = [str(n["node_id"]) for n in nodes if str(n.get("node_type")) == "maneuver"]
        outcome_ids = ["collision_outcome", "progress_outcome", "compliance_outcome"]
        for m_id in maneuver_ids:
            for o_id in outcome_ids:
                edges.append(
                    {
                        "parent_id": m_id,
                        "child_id": o_id,
                        "confidence": 0.8 if o_id == "collision_outcome" else 0.7,
                        "mechanism": "maneuver_to_outcome",
                    }
                )

        cpts = {
            "collision_outcome": {
                "values": ["collision_avoided", "collision_possible"],
                "parents": maneuver_ids,
                "cpt": {"*": {"collision_avoided": 0.8, "collision_possible": 0.2}},
            },
            "progress_outcome": {
                "values": ["progress_good", "progress_limited"],
                "parents": maneuver_ids,
                "cpt": {"*": {"progress_good": 0.7, "progress_limited": 0.3}},
            },
            "compliance_outcome": {
                "values": ["compliant", "violation_possible"],
                "parents": maneuver_ids,
                "cpt": {"*": {"compliant": 0.85, "violation_possible": 0.15}},
            },
        }
        payload = {
            "schema_version": schema_version_for_contract(str(self.contract.name)),
            "scenario_id": str(scenario_id),
            "nodes": nodes,
            "edges": edges,
            "cpts": cpts,
            "metadata": {"source": "scene_derived", "sample_index": int(sample_index)},
        }
        ok, canonical, report = enforce_dag_contract(payload, config=self.contract)
        if not ok:
            raise RuntimeError(
                "Scene-derived DAG failed contract: "
                f"scenario_id={scenario_id} violations={report.violation_counts}"
            )
        canonical["schema_version"] = schema_version_for_contract(str(self.contract.name))
        return canonical


@dataclass
class DAGSourceResolver:
    mode: str = "dual"  # dual|cache|scene_derived
    cache_dir: str = ""
    cache_strict: bool = False
    expected_schema: str = "any"  # any|counter_bmt_v2_dag_cache_v2_compact10|counter_bmt_v2_dag_cache_v3_maneuver_outcome

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
            if from_cache is not None and str(self.expected_schema) != "any":
                if str(from_cache.get("schema_version", "")) != str(self.expected_schema):
                    from_cache = None
                    if self.mode == "cache":
                        if self.cache_strict:
                            return None, "cache_schema_mismatch_strict"
                        return None, "cache_schema_mismatch"

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
