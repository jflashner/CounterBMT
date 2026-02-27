"""DAG source adapters for supervised DAG-latent training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from counter_bmt_v2.training.dag_cache import DAGCacheReader


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
            outcome = "stable"
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
                outcome = "stable" if risk_proxy < 0.35 else ("caution" if risk_proxy < 0.65 else "unstable")
            else:
                ego_speed0 = 0.0
                progress = 0.0
                turn_rate = 0.0
                risk_proxy = 0.0
                outcome = "stable"

        maneuver_proxy = "straight"
        if turn_rate > 0.15:
            maneuver_proxy = "turning"
        if progress < 0.5:
            maneuver_proxy = "slow_or_stop"

        nodes: List[Dict[str, Any]] = [
            _node_entry("ego_initial_speed", "ego_state", ego_speed0, 0.0, {"alternatives": "scalar_speed"}),
            _node_entry("maneuver_proxy", "maneuver", maneuver_proxy, dt_s, {"alternatives": "straight,turning,slow_or_stop"}),
            _node_entry("risk_proxy", "decision", float(risk_proxy), 2.0 * dt_s, {"alternatives": "scalar_risk"}),
            _node_entry("outcome_proxy", "outcome", outcome, None, {"alternatives": "stable,caution,unstable"}),
        ]

        edges = [
            {"parent_id": "ego_initial_speed", "child_id": "maneuver_proxy", "confidence": 0.7, "mechanism": "speed_to_maneuver"},
            {"parent_id": "maneuver_proxy", "child_id": "risk_proxy", "confidence": 0.8, "mechanism": "maneuver_to_risk"},
            {"parent_id": "risk_proxy", "child_id": "outcome_proxy", "confidence": 0.9, "mechanism": "risk_to_outcome"},
        ]

        cpts = {
            "outcome_proxy": {
                "values": ["stable", "caution", "unstable"],
                "parents": ["risk_proxy"],
                "cpt": {
                    "*": {"stable": 0.6, "caution": 0.3, "unstable": 0.1},
                },
            }
        }

        return {
            "schema_version": "counter_bmt_v2_dag_cache_v1",
            "scenario_id": str(scenario_id),
            "nodes": nodes,
            "edges": edges,
            "cpts": cpts,
            "metadata": {"source": "scene_derived", "sample_index": int(sample_index)},
        }


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

