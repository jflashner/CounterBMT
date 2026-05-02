from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .conflict_analysis import analyze_conflicts
from .dag_adapter import local_intervention_to_bayesian_dag
from .sdc_path_control import normalize_semantic_label
from .signal_qc import (
    is_caution_signal_state,
    is_go_like_signal_state,
    is_stop_like_signal_state,
)
from .types import CanonicalScenario, stable_string_sort_key


_BRANCH_RISK_WEIGHT = {
    "stop": 0.18,
    "right": 0.48,
    "right_lane_change": 0.42,
    "straight": 0.72,
    "left": 0.92,
    "left_lane_change": 0.58,
    "u_turn": 1.00,
}


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _signal_state_category(signal_state: Optional[str]) -> str:
    if signal_state is None:
        return "unknown"
    if is_stop_like_signal_state(signal_state):
        return "stop"
    if is_go_like_signal_state(signal_state):
        return "go"
    if is_caution_signal_state(signal_state):
        return "caution"
    return "unknown"


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _branch_weight(semantic_label: str) -> float:
    return float(_BRANCH_RISK_WEIGHT.get(str(semantic_label), 0.72))


def _min_path_distance_m(path_xy: np.ndarray, point_xy: np.ndarray) -> float:
    if path_xy.size == 0:
        return float("inf")
    deltas = np.asarray(path_xy, dtype=np.float32) - np.asarray(point_xy, dtype=np.float32).reshape(1, 2)
    return float(np.linalg.norm(deltas, axis=-1).min())


def _pick_reference_light(
    canonical: CanonicalScenario,
    *,
    current_time_index: int,
    current_xy: np.ndarray,
    selected_path_xy: np.ndarray,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    t_idx = int(np.clip(int(current_time_index), 0, max(0, canonical.length - 1)))
    for light_id, light in sorted(canonical.traffic_lights.items(), key=lambda item: stable_string_sort_key(item[0])):
        if light.stop_point_xy is None:
            continue
        stop_xy = np.asarray(light.stop_point_xy, dtype=np.float32)
        current_dist = float(np.linalg.norm(stop_xy - np.asarray(current_xy, dtype=np.float32)))
        path_dist = _min_path_distance_m(selected_path_xy, stop_xy)
        # Keep the search local to the maneuver corridor or nearby decision context.
        if path_dist > 30.0 and current_dist > 50.0:
            continue
        state = None
        if t_idx < len(light.object_state):
            state = light.object_state[t_idx]
        signal_category = _signal_state_category(state)
        candidate = {
            "light_id": str(light_id),
            "stop_point_xy": (float(stop_xy[0]), float(stop_xy[1])),
            "signal_state_at_decision": state,
            "signal_state_category": signal_category,
            "path_distance_m": float(path_dist),
            "current_distance_m": float(current_dist),
        }
        key = (
            1 if signal_category == "unknown" else 0,
            min(float(path_dist), float(current_dist)),
            float(path_dist),
            float(current_dist),
            stable_string_sort_key(light_id),
        )
        if best is None or key < best["_sort_key"]:
            best = {**candidate, "_sort_key": key}
    if best is None:
        return {
            "light_id": "",
            "stop_point_xy": None,
            "signal_state_at_decision": None,
            "signal_state_category": "unknown",
            "path_distance_m": float("inf"),
            "current_distance_m": float("inf"),
        }
    best.pop("_sort_key", None)
    return best


def _inferred_compliance_label(*, semantic_label: str, signal_category: str) -> str:
    if semantic_label == "stop":
        return "obey_signal"
    if signal_category in {"stop", "caution"}:
        return "red_light_violation"
    return "obey_signal"


def _inferred_entry_timing(*, semantic_label: str, min_conflict_eta_gap_s: Optional[float], has_conflicts: bool) -> str:
    if semantic_label == "stop":
        return "after_conflict" if has_conflicts else "no_conflict"
    if not has_conflicts or min_conflict_eta_gap_s is None:
        return "no_conflict"
    return "before_conflict" if float(min_conflict_eta_gap_s) <= 1.5 else "after_conflict"


def _risk_components(
    *,
    semantic_label: str,
    signal_category: str,
    min_conflict_eta_gap_s: Optional[float],
    conflict_agent_speed_mps: float,
    num_conflict_agents: int,
) -> Dict[str, float]:
    branch_factor = _branch_weight(semantic_label)
    if min_conflict_eta_gap_s is None:
        gap_closeness = 0.0
    else:
        gap_closeness = math.exp(-max(float(min_conflict_eta_gap_s), 0.0) / 1.75)

    if num_conflict_agents <= 0:
        p_conflict_overlap = 0.015 if semantic_label == "stop" else 0.05 * branch_factor
    else:
        active_factor = 0.22 if semantic_label == "stop" else 1.0
        p_conflict_overlap = active_factor * branch_factor * (0.08 + 0.82 * gap_closeness)
    p_conflict_overlap = _clip01(p_conflict_overlap)

    if semantic_label == "stop":
        if signal_category == "go":
            p_signal_violation = 0.03
        elif signal_category == "unknown":
            p_signal_violation = 0.05
        else:
            p_signal_violation = 0.01
    elif signal_category == "stop":
        p_signal_violation = 0.62 + 0.28 * branch_factor
    elif signal_category == "caution":
        p_signal_violation = 0.28 + 0.24 * branch_factor
    elif signal_category == "go":
        p_signal_violation = 0.03 + 0.07 * branch_factor
    else:
        p_signal_violation = 0.10 + 0.08 * branch_factor
    p_signal_violation = _clip01(p_signal_violation)

    speed_severity = _clip01(conflict_agent_speed_mps / 14.0)
    p_collision_or_near_miss = (
        0.46 * p_conflict_overlap
        + 0.34 * p_signal_violation
        + 0.20 * speed_severity * branch_factor
    )
    if min_conflict_eta_gap_s is not None and float(min_conflict_eta_gap_s) <= 1.5:
        p_collision_or_near_miss += 0.10
    if semantic_label == "stop":
        p_collision_or_near_miss *= 0.45
    p_collision_or_near_miss = _clip01(p_collision_or_near_miss)

    severity_score = (
        0.22
        + 0.40 * speed_severity
        + 0.25 * branch_factor
        + (0.15 if signal_category == "stop" and semantic_label != "stop" else 0.0)
        + (0.10 if min_conflict_eta_gap_s is not None and float(min_conflict_eta_gap_s) <= 1.5 else 0.0)
    )
    severity_score = _clip01(severity_score)

    critical_event_prior = _clip01(
        1.0
        - (1.0 - p_conflict_overlap)
        * (1.0 - p_signal_violation)
        * (1.0 - 0.85 * p_collision_or_near_miss)
    )
    risk_score_total = _clip01(critical_event_prior * (0.5 + 0.5 * severity_score))
    return {
        "p_conflict_overlap": float(p_conflict_overlap),
        "p_signal_violation": float(p_signal_violation),
        "p_collision_or_near_miss": float(p_collision_or_near_miss),
        "severity_score": float(severity_score),
        "critical_event_prior": float(critical_event_prior),
        "risk_score_total": float(risk_score_total),
    }


def _risk_tier(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.55:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def _build_explanation(
    *,
    semantic_label: str,
    signal_category: str,
    min_conflict_eta_gap_s: Optional[float],
    conflict_agent_speed_mps: float,
    num_conflict_agents: int,
) -> list[str]:
    notes: list[str] = []
    if signal_category in {"stop", "caution"} and semantic_label != "stop":
        notes.append("non-stop maneuver under restrictive signal")
    if min_conflict_eta_gap_s is not None and float(min_conflict_eta_gap_s) <= 1.5:
        notes.append(f"tight conflict ETA gap ({float(min_conflict_eta_gap_s):.2f}s)")
    elif num_conflict_agents > 0:
        notes.append(f"{int(num_conflict_agents)} nearby conflict agent(s)")
    if float(conflict_agent_speed_mps) >= 8.0:
        notes.append(f"high conflict-agent speed ({float(conflict_agent_speed_mps):.1f} m/s)")
    if semantic_label == "left":
        notes.append("crossing maneuver carries elevated conflict exposure")
    elif semantic_label == "left_lane_change":
        notes.append("left lane change carries moderate lateral conflict exposure")
    elif semantic_label == "right_lane_change":
        notes.append("right lane change carries moderate lateral conflict exposure")
    elif semantic_label == "u_turn":
        notes.append("u-turn maneuver carries elevated exposure")
    if not notes:
        notes.append("low-conflict compliant maneuver prior")
    return notes


def _proxy_payload(
    *,
    canonical: CanonicalScenario,
    sdc_id: str,
    current_time_index: int,
    selected_slot_id: str,
    selected_path_id: Optional[str],
    semantic_label: str,
    reference_light: Mapping[str, Any],
    conflict_result: Optional[Any],
    selected_path_xy: np.ndarray,
) -> Dict[str, Any]:
    track = canonical.tracks.get(str(sdc_id))
    heading = 0.0
    if track is not None and 0 <= int(current_time_index) < track.heading.shape[0]:
        heading = _safe_float(track.heading[int(current_time_index)], default=0.0)
    if selected_path_xy.shape[0] >= 2:
        delta = np.asarray(selected_path_xy[1], dtype=np.float32) - np.asarray(selected_path_xy[0], dtype=np.float32)
        if np.linalg.norm(delta) > 1e-3:
            heading = float(math.atan2(float(delta[1]), float(delta[0])))

    signal_state = reference_light.get("signal_state_at_decision")
    signal_category = _signal_state_category(signal_state)
    min_conflict_eta_gap_s = None
    conflict_agents = []
    if conflict_result is not None:
        conflict_agents = [record.to_dict() for record in list(getattr(conflict_result, "conflict_agents", []) or [])]
        finite_gaps = [
            float(record.get("eta_gap_s"))
            for record in conflict_agents
            if record.get("eta_gap_s") is not None and np.isfinite(float(record.get("eta_gap_s")))
        ]
        min_conflict_eta_gap_s = min(finite_gaps) if finite_gaps else None
    compliance_label = _inferred_compliance_label(
        semantic_label=semantic_label,
        signal_category=signal_category,
    )
    entry_timing = _inferred_entry_timing(
        semantic_label=semantic_label,
        min_conflict_eta_gap_s=min_conflict_eta_gap_s,
        has_conflicts=bool(conflict_agents),
    )
    return {
        "scenario_id": str(canonical.scenario_id),
        "agent_id": str(sdc_id),
        "decision_time_idx": int(current_time_index),
        "context": {
            "sdc_id": str(sdc_id),
            "traffic_light_id": str(reference_light.get("light_id") or ""),
            "stop_point_xy": list(reference_light.get("stop_point_xy") or [0.0, 0.0]),
            "approach_heading": float(heading),
            "signal_state_at_decision": signal_state,
            "objects_of_interest": list(canonical.objects_of_interest),
            "conflict_agents": conflict_agents,
        },
        "supervised_decision": {
            "branch_id": str(selected_path_id or selected_slot_id or semantic_label),
            "branch_label": str(semantic_label),
            "compliance_label": str(compliance_label),
            "entry_timing": None if entry_timing == "no_conflict" else str(entry_timing),
        },
        "raw_recovered_decision": {
            "crossed_stop_region": bool(semantic_label != "stop"),
        },
    }


def score_semantic_slot_dag_risks(
    *,
    canonical: CanonicalScenario,
    contract: Mapping[str, Any],
    world_paths: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
) -> Dict[str, Dict[str, Any]]:
    sdc_track = canonical.tracks.get(str(sdc_id))
    if sdc_track is None:
        return {}
    t_idx = int(np.clip(int(current_time_index), 0, max(0, canonical.length - 1)))
    current_xy = np.asarray(sdc_track.position_xy[t_idx], dtype=np.float32)

    slots = []
    for slot in list(contract.get("highlighted_paths", []) or []):
        row = dict(slot or {})
        row["slot_id"] = str(row.get("slot_id") or "")
        row["path_id"] = None if row.get("path_id") is None else str(row.get("path_id"))
        row["semantic_label"] = normalize_semantic_label(row.get("semantic_label"), default="straight")
        row["source_kind"] = str(row.get("source_kind") or "")
        slots.append(row)

    risk_by_slot: Dict[str, Dict[str, Any]] = {}
    factual_slot_id: Optional[str] = None
    for slot in slots:
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        if str(slot.get("source_kind") or "") == "ground_truth":
            factual_slot_id = slot_id
        selected_path = world_paths.get(slot_id)
        selected_path_xy = (
            np.asarray(getattr(selected_path, "waypoints_xy_world", []), dtype=np.float32).reshape(-1, 2)
            if selected_path is not None
            else np.zeros((0, 2), dtype=np.float32)
        )
        semantic_label = normalize_semantic_label(slot.get("semantic_label"), default="straight")
        reference_light = _pick_reference_light(
            canonical,
            current_time_index=t_idx,
            current_xy=current_xy,
            selected_path_xy=selected_path_xy,
        )
        conflict_result = None
        if reference_light.get("stop_point_xy") is not None:
            conflict_result = analyze_conflicts(
                canonical,
                agent_id=str(sdc_id),
                stop_point_xy=tuple(reference_light["stop_point_xy"]),
                decision_time_idx=t_idx,
            )
        conflict_agents = list(getattr(conflict_result, "conflict_agents", []) or [])
        finite_gaps = [
            float(record.eta_gap_s)
            for record in conflict_agents
            if record.eta_gap_s is not None and np.isfinite(float(record.eta_gap_s))
        ]
        min_gap = min(finite_gaps) if finite_gaps else None
        conflict_speed_mps = max(
            [_safe_float(getattr(record, "current_speed_mps", 0.0), default=0.0) for record in conflict_agents] or [0.0]
        )
        signal_state = reference_light.get("signal_state_at_decision")
        signal_category = _signal_state_category(signal_state)
        components = _risk_components(
            semantic_label=semantic_label,
            signal_category=signal_category,
            min_conflict_eta_gap_s=min_gap,
            conflict_agent_speed_mps=conflict_speed_mps,
            num_conflict_agents=len(conflict_agents),
        )
        compliance_label = _inferred_compliance_label(
            semantic_label=semantic_label,
            signal_category=signal_category,
        )
        entry_timing = _inferred_entry_timing(
            semantic_label=semantic_label,
            min_conflict_eta_gap_s=min_gap,
            has_conflicts=bool(conflict_agents),
        )
        proxy_payload = _proxy_payload(
            canonical=canonical,
            sdc_id=str(sdc_id),
            current_time_index=t_idx,
            selected_slot_id=slot_id,
            selected_path_id=(None if slot.get("path_id") is None else str(slot.get("path_id"))),
            semantic_label=semantic_label,
            reference_light=reference_light,
            conflict_result=conflict_result,
            selected_path_xy=selected_path_xy,
        )
        dag_view = local_intervention_to_bayesian_dag(proxy_payload).to_dict()
        risk_by_slot[slot_id] = {
            "version": "dag_risk_v1",
            "calibrated_probability": False,
            "selected_slot_id": slot_id,
            "selected_path_id": None if slot.get("path_id") is None else str(slot.get("path_id")),
            "semantic_label": semantic_label,
            "source_kind": ("factual_gt" if str(slot.get("source_kind") or "") == "ground_truth" else "alternative_sdc_path"),
            "reference_light_id": str(reference_light.get("light_id") or ""),
            "reference_stop_point_xy": _jsonify(reference_light.get("stop_point_xy")),
            "reference_stop_point_distance_to_path_m": _safe_float(reference_light.get("path_distance_m"), default=float("inf")),
            "reference_stop_point_distance_to_current_m": _safe_float(reference_light.get("current_distance_m"), default=float("inf")),
            "signal_state_at_decision": signal_state,
            "signal_state_category": signal_category,
            "proxy_compliance_label": compliance_label,
            "proxy_entry_timing": entry_timing,
            "num_conflict_agents": int(len(conflict_agents)),
            "min_conflict_eta_gap_s": (None if min_gap is None else float(min_gap)),
            "max_conflict_agent_speed_mps": float(conflict_speed_mps),
            "risk_components": components,
            "critical_event_prior": float(components["critical_event_prior"]),
            "risk_score_total": float(components["risk_score_total"]),
            "risk_tier": _risk_tier(float(components["risk_score_total"])),
            "explanation": _build_explanation(
                semantic_label=semantic_label,
                signal_category=signal_category,
                min_conflict_eta_gap_s=min_gap,
                conflict_agent_speed_mps=conflict_speed_mps,
                num_conflict_agents=len(conflict_agents),
            ),
            "dag_view": dag_view,
        }

    factual_risk = risk_by_slot.get(str(factual_slot_id or ""), {})
    factual_total = factual_risk.get("risk_score_total")
    factual_prior = factual_risk.get("critical_event_prior")
    ranked = sorted(
        risk_by_slot.items(),
        key=lambda item: (-float(item[1].get("risk_score_total", 0.0)), stable_string_sort_key(item[0])),
    )
    rank_by_slot = {slot_id: idx + 1 for idx, (slot_id, _) in enumerate(ranked)}
    for slot_id, bundle in risk_by_slot.items():
        bundle["risk_rank_within_example"] = int(rank_by_slot.get(slot_id, 0))
        bundle["factual_reference_slot_id"] = None if factual_slot_id is None else str(factual_slot_id)
        bundle["risk_uplift_vs_gt"] = (
            None
            if factual_total is None
            else float(bundle.get("risk_score_total", 0.0) - float(factual_total))
        )
        bundle["critical_event_prior_uplift_vs_gt"] = (
            None
            if factual_prior is None
            else float(bundle.get("critical_event_prior", 0.0) - float(factual_prior))
        )
    return risk_by_slot
