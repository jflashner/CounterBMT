from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import CanonicalScenario, stable_string_sort_key


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


@dataclass
class ETARecord:
    track_id: str
    object_type: str
    eta_s: Optional[float]
    eta_gap_s: Optional[float]
    current_distance_to_core_m: float
    current_speed_mps: float
    will_enter_core: bool
    current_position_xy: Tuple[float, float]
    is_object_of_interest: bool

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class ConflictAnalysisResult:
    core_center_xy: Tuple[float, float]
    core_radius_m: float
    target_agent_id: str
    target_eta_s: Optional[float]
    sdc_eta_s: Optional[float]
    eta_table: List[ETARecord] = field(default_factory=list)
    conflict_agents: List[ETARecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


def analyze_conflicts(
    canonical: CanonicalScenario,
    *,
    agent_id: Optional[str] = None,
    stop_point_xy: Tuple[float, float],
    decision_time_idx: int,
    core_radius_m: float = 8.0,
    neighborhood_radius_m: float = 45.0,
    eta_gap_threshold_s: float = 3.0,
) -> ConflictAnalysisResult:
    stop_xy = np.asarray(stop_point_xy, dtype=np.float32)
    target_agent_id = str(agent_id or canonical.sdc_id)
    if target_agent_id not in canonical.tracks:
        raise KeyError(f"agent_id={target_agent_id!r} not found in canonical.tracks")
    target_track = canonical.tracks[target_agent_id]
    dt_s = _estimate_dt_s(canonical)
    target_eta = _future_eta_to_core(
        track_position_xy=target_track.position_xy,
        track_valid=target_track.valid,
        start_idx=decision_time_idx,
        center_xy=stop_xy,
        core_radius_m=core_radius_m,
        dt_s=dt_s,
    )

    ooi = set(canonical.objects_of_interest)
    eta_records: List[ETARecord] = []
    for track_id, track in sorted(canonical.tracks.items(), key=lambda item: stable_string_sort_key(item[0])):
        if track_id == target_agent_id or track.object_type != "VEHICLE":
            continue
        if decision_time_idx >= track.valid.shape[0] or not bool(track.valid[decision_time_idx]) or not np.isfinite(track.position_xy[decision_time_idx]).all():
            continue
        current_xy = np.asarray(track.position_xy[decision_time_idx], dtype=np.float32)
        current_dist = float(np.linalg.norm(current_xy - stop_xy))
        if current_dist > float(neighborhood_radius_m):
            continue
        eta = _future_eta_to_core(
            track_position_xy=track.position_xy,
            track_valid=track.valid,
            start_idx=decision_time_idx,
            center_xy=stop_xy,
            core_radius_m=core_radius_m,
            dt_s=dt_s,
        )
        current_speed = float(np.linalg.norm(np.asarray(track.velocity_xy[decision_time_idx], dtype=np.float32)))
        eta_gap = None if eta is None or target_eta is None else float(abs(eta - target_eta))
        eta_records.append(
            ETARecord(
                track_id=track_id,
                object_type=track.object_type,
                eta_s=eta,
                eta_gap_s=eta_gap,
                current_distance_to_core_m=current_dist,
                current_speed_mps=current_speed,
                will_enter_core=eta is not None,
                current_position_xy=(float(current_xy[0]), float(current_xy[1])),
                is_object_of_interest=track_id in ooi,
            )
        )

    conflicts = [
        record
        for record in eta_records
        if record.eta_s is not None
        and target_eta is not None
        and record.eta_gap_s is not None
        and float(record.eta_gap_s) <= float(eta_gap_threshold_s)
    ]
    conflicts = sorted(conflicts, key=lambda record: (record.eta_gap_s if record.eta_gap_s is not None else float("inf"), record.track_id))
    return ConflictAnalysisResult(
        core_center_xy=(float(stop_xy[0]), float(stop_xy[1])),
        core_radius_m=float(core_radius_m),
        target_agent_id=target_agent_id,
        target_eta_s=target_eta,
        sdc_eta_s=target_eta,
        eta_table=eta_records,
        conflict_agents=conflicts,
    )


def _future_eta_to_core(
    *,
    track_position_xy: np.ndarray,
    track_valid: np.ndarray,
    start_idx: int,
    center_xy: np.ndarray,
    core_radius_m: float,
    dt_s: float,
) -> Optional[float]:
    start_idx = int(np.clip(int(start_idx), 0, max(0, track_position_xy.shape[0] - 1)))
    for idx in range(start_idx, track_position_xy.shape[0]):
        if not bool(track_valid[idx]) or not np.isfinite(track_position_xy[idx]).all():
            continue
        dist = float(np.linalg.norm(np.asarray(track_position_xy[idx], dtype=np.float32) - center_xy))
        if dist <= float(core_radius_m):
            return float((idx - start_idx) * dt_s)
    return None


def _estimate_dt_s(canonical: CanonicalScenario) -> float:
    ts = np.asarray(canonical.ts, dtype=np.float32)
    if ts.shape[0] < 2:
        return 0.1
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size > 0 else 0.1
