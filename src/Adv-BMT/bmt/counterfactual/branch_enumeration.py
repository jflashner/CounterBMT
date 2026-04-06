from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contract_local_intervention import ArtifactProvenance, CommitmentMetrics
from .geometry import (
    angle_delta,
    any_point_within_radius,
    classify_heading_delta,
    cluster_heading_values,
    cumulative_polyline_lengths,
    heading_from_points,
    heading_from_track_window,
    min_distance_to_points,
    nearest_point_index,
    polyline_length,
)
from .types import CanonicalScenario, stable_string_sort_key

PATH_CHOICE_MIN_BRANCH_MARGIN = 0.75
PATH_CHOICE_MAX_FINAL_HEADING_ERROR_RAD = math.radians(35.0)
INTERSECTION_CORE_RADIUS_M = 8.0


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
class LaneFeatureSummary:
    feature_id: str
    feature_type: str
    polyline_xy: np.ndarray
    start_point_xy: Tuple[float, float]
    end_point_xy: Tuple[float, float]
    length_m: float
    heading_start: float
    heading_end: float
    nearest_stop_dist_m: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class LocalPatchTrackSnapshot:
    track_id: str
    object_type: str
    position_xy: Tuple[float, float]
    distance_to_center_m: float
    is_sdc: bool
    is_object_of_interest: bool

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class LocalPatch:
    stop_point_xy: Tuple[float, float]
    radius_m: float
    lane_features: List[LaneFeatureSummary] = field(default_factory=list)
    nearby_tracks: List[LocalPatchTrackSnapshot] = field(default_factory=list)
    nearby_traffic_light_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class BranchTerminalPose:
    x: float
    y: float
    heading: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class BranchCandidate:
    branch_id: str
    branch_label: str
    source_feature_id: str
    heading: float
    heading_delta: float
    start_point_xy: Tuple[float, float]
    terminal_pose: BranchTerminalPose
    rank_score: float
    source_kind: str = "geometry"
    polyline_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class DecisionWindow:
    decision_time_idx: int
    window_start_idx: int
    window_end_idx: int
    first_time_under_35m: int
    approach_heading: float
    crossed_stop_region: bool
    cross_time_idx: Optional[int]
    stop_region_radius_m: float
    stop_point_xy: Tuple[float, float]
    agent_id: str
    agent_role: str
    current_time_index_global: int
    decision_time_index_global: int
    cross_time_index_global: Optional[int]
    branch_commit_index_global: Optional[int]
    decision_time_index_rel_to_current: int
    cross_time_index_rel_to_current: Optional[int]
    branch_commit_index_rel_to_current: Optional[int]
    control_available_at_current: bool

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class GTBranchRecovery:
    branch_id: str
    branch_label: str
    terminal_pose: BranchTerminalPose
    crossed_stop_region: bool
    cross_time_idx: Optional[int]
    branch_recall_hit: bool
    recovered_from_existing_candidate: bool
    realized_heading: float
    provenance: Optional[ArtifactProvenance] = None
    commitment_metrics: Optional[CommitmentMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class _BranchFitScore:
    candidate: BranchCandidate
    score: float
    downstream_progress_m: float
    final_heading_error_rad: float
    mean_lateral_error_m: float


def extract_local_patch(
    canonical: CanonicalScenario,
    *,
    stop_point_xy: Tuple[float, float],
    radius_m: float = 30.0,
    time_index: Optional[int] = None,
) -> LocalPatch:
    lane_features = select_lane_like_features(canonical, stop_point_xy=stop_point_xy, radius_m=radius_m)
    nearby_tracks: List[LocalPatchTrackSnapshot] = []
    ooi = set(canonical.objects_of_interest)
    if time_index is None:
        time_index = canonical.current_time_index
    time_index = int(np.clip(int(time_index), 0, max(0, canonical.length - 1)))

    for track_id, track in sorted(canonical.tracks.items(), key=lambda item: stable_string_sort_key(item[0])):
        if time_index >= track.valid.shape[0] or not bool(track.valid[time_index]) or not np.isfinite(track.position_xy[time_index]).all():
            continue
        pos = np.asarray(track.position_xy[time_index], dtype=np.float32)
        dist = float(np.linalg.norm(pos - np.asarray(stop_point_xy, dtype=np.float32)))
        if dist > float(radius_m):
            continue
        nearby_tracks.append(
            LocalPatchTrackSnapshot(
                track_id=track_id,
                object_type=track.object_type,
                position_xy=(float(pos[0]), float(pos[1])),
                distance_to_center_m=dist,
                is_sdc=track_id == canonical.sdc_id,
                is_object_of_interest=track_id in ooi,
            )
        )

    nearby_lights = [
        light_id
        for light_id, light in sorted(canonical.traffic_lights.items(), key=lambda item: stable_string_sort_key(item[0]))
        if light.stop_point_xy is not None and float(np.linalg.norm(np.asarray(light.stop_point_xy, dtype=np.float32) - np.asarray(stop_point_xy, dtype=np.float32))) <= float(radius_m)
    ]
    return LocalPatch(
        stop_point_xy=stop_point_xy,
        radius_m=float(radius_m),
        lane_features=lane_features,
        nearby_tracks=nearby_tracks,
        nearby_traffic_light_ids=nearby_lights,
    )


def select_lane_like_features(
    canonical: CanonicalScenario,
    *,
    stop_point_xy: Tuple[float, float],
    radius_m: float = 30.0,
) -> List[LaneFeatureSummary]:
    lane_features: List[LaneFeatureSummary] = []
    for feature_id, feature in sorted(canonical.map_features.items(), key=lambda item: stable_string_sort_key(item[0])):
        if not str(feature.feature_type).startswith("LANE_"):
            continue
        if feature.polyline_xy.shape[0] < 2 or not any_point_within_radius(feature.polyline_xy, stop_point_xy, radius_m):
            continue
        lane_features.append(
            LaneFeatureSummary(
                feature_id=feature_id,
                feature_type=feature.feature_type,
                polyline_xy=np.asarray(feature.polyline_xy, dtype=np.float32),
                start_point_xy=(float(feature.polyline_xy[0, 0]), float(feature.polyline_xy[0, 1])),
                end_point_xy=(float(feature.polyline_xy[-1, 0]), float(feature.polyline_xy[-1, 1])),
                length_m=polyline_length(feature.polyline_xy),
                heading_start=heading_from_points(feature.polyline_xy[0], feature.polyline_xy[min(1, feature.polyline_xy.shape[0] - 1)]),
                heading_end=heading_from_points(feature.polyline_xy[max(0, feature.polyline_xy.shape[0] - 2)], feature.polyline_xy[-1]),
                nearest_stop_dist_m=min_distance_to_points(feature.polyline_xy, stop_point_xy),
            )
        )
    return lane_features


def choose_decision_window(
    canonical: CanonicalScenario,
    *,
    agent_id: Optional[str] = None,
    agent_role: str = "target_agent",
    stop_point_xy: Tuple[float, float],
    distance_threshold_m: float = 35.0,
    stop_region_radius_m: float = 4.0,
    lookback_steps: int = 8,
) -> DecisionWindow:
    resolved_agent_id = str(agent_id or canonical.sdc_id)
    if resolved_agent_id not in canonical.tracks:
        raise KeyError(f"agent_id={resolved_agent_id!r} not found in canonical.tracks")
    track = canonical.tracks[resolved_agent_id]
    stop_xy = np.asarray(stop_point_xy, dtype=np.float32)
    dist_curve = np.linalg.norm(np.asarray(track.position_xy, dtype=np.float32) - stop_xy[None, :], axis=-1)
    valid_mask = np.asarray(track.valid, dtype=bool) & np.isfinite(track.position_xy).all(axis=-1)
    within = valid_mask & (dist_curve <= float(distance_threshold_m))
    candidate_times = np.flatnonzero(within)
    if candidate_times.size == 0:
        raise ValueError("No valid target-agent timestep within 35m of stop point")

    decision_time_idx: Optional[int] = None
    approach_heading: Optional[float] = None
    for t in candidate_times.tolist():
        heading = heading_from_track_window(track.position_xy, track.valid, end_idx=int(t), lookback_steps=lookback_steps)
        if heading is None:
            continue
        branches: List[BranchCandidate] = []
        if str(resolved_agent_id) == str(canonical.sdc_id) and getattr(canonical, "sdc_paths", {}):
            try:
                from .sdc_path_branches import enumerate_branch_candidates_from_sdc_paths

                branches = enumerate_branch_candidates_from_sdc_paths(
                    canonical,
                    agent_id=str(resolved_agent_id),
                    decision_time_idx=int(t),
                    approach_heading=float(heading),
                )
            except Exception:
                branches = []
        try:
            if not branches:
                from .branch_routes_v2 import enumerate_branch_candidates_from_routes_v2

                _, branches = enumerate_branch_candidates_from_routes_v2(
                    canonical,
                    agent_id=resolved_agent_id,
                    current_time_idx=int(canonical.current_time_index),
                    decision_time_idx=int(t),
                    stop_point_xy=stop_point_xy,
                    approach_heading=heading,
                )
        except Exception:
            if not branches:
                branches = []
        if not branches:
            branches = enumerate_branch_candidates(
                select_lane_like_features(canonical, stop_point_xy=stop_point_xy, radius_m=30.0),
                stop_point_xy=stop_point_xy,
                approach_heading=heading,
            )
        if len(branches) >= 2:
            decision_time_idx = int(t)
            approach_heading = float(heading)
            break

    if decision_time_idx is None or approach_heading is None:
        decision_time_idx = int(candidate_times[0])
        approach_heading = heading_from_track_window(track.position_xy, track.valid, end_idx=decision_time_idx, lookback_steps=lookback_steps)
        if approach_heading is None:
            approach_heading = float(track.heading[decision_time_idx]) if np.isfinite(track.heading[decision_time_idx]) else 0.0

    cross_time_idx = _first_stopline_crossing_index(
        track_position_xy=track.position_xy,
        track_valid=track.valid,
        start_idx=decision_time_idx,
        stop_point_xy=stop_point_xy,
        approach_heading=float(approach_heading),
    )
    current_idx = int(canonical.current_time_index)
    return DecisionWindow(
        decision_time_idx=decision_time_idx,
        window_start_idx=max(0, decision_time_idx - 10),
        window_end_idx=min(int(canonical.length) - 1, decision_time_idx + 20),
        first_time_under_35m=int(candidate_times[0]),
        approach_heading=float(approach_heading),
        crossed_stop_region=bool(cross_time_idx is not None),
        cross_time_idx=cross_time_idx,
        stop_region_radius_m=float(stop_region_radius_m),
        stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
        agent_id=resolved_agent_id,
        agent_role=str(agent_role),
        current_time_index_global=current_idx,
        decision_time_index_global=decision_time_idx,
        cross_time_index_global=cross_time_idx,
        branch_commit_index_global=None,
        decision_time_index_rel_to_current=int(decision_time_idx - current_idx),
        cross_time_index_rel_to_current=(None if cross_time_idx is None else int(cross_time_idx - current_idx)),
        branch_commit_index_rel_to_current=None,
        control_available_at_current=bool(decision_time_idx >= current_idx),
    )


def enumerate_branch_candidates(
    lane_features: Sequence[LaneFeatureSummary],
    *,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
    min_branch_length_m: float = 8.0,
    heading_cluster_threshold_deg: float = 25.0,
) -> List[BranchCandidate]:
    raw_candidates: List[BranchCandidate] = []
    counter = 0
    stop_xy = np.asarray(stop_point_xy, dtype=np.float32)
    for lane in lane_features:
        polyline = np.asarray(lane.polyline_xy, dtype=np.float32)
        nearest_idx = nearest_point_index(polyline, stop_point_xy)
        for direction in (1, -1):
            candidate_polyline = _polyline_direction_from_index(polyline, nearest_idx, direction)
            if candidate_polyline.shape[0] < 2:
                continue
            cum = cumulative_polyline_lengths(candidate_polyline)
            if float(cum[-1]) < float(min_branch_length_m):
                continue
            first_far_idx = int(np.searchsorted(cum, 1.5, side="left"))
            first_far_idx = min(max(first_far_idx, 0), candidate_polyline.shape[0] - 2)
            start_point = candidate_polyline[first_far_idx]
            end_idx = int(np.searchsorted(cum, min(float(cum[-1]), 15.0), side="left"))
            end_idx = min(max(end_idx, first_far_idx + 1), candidate_polyline.shape[0] - 1)
            terminal_point = candidate_polyline[end_idx]
            outward_dist = float(np.linalg.norm(terminal_point - stop_xy))
            if outward_dist < 4.0:
                continue
            heading = heading_from_points(start_point, terminal_point)
            delta = angle_delta(heading, approach_heading)
            label = classify_heading_delta(delta)
            rank_score = float(np.linalg.norm(start_point - stop_xy) + 0.5 * abs(delta))
            terminal_heading = heading_from_points(candidate_polyline[max(0, end_idx - 1)], candidate_polyline[end_idx])
            raw_candidates.append(
                BranchCandidate(
                    branch_id=f"branch_{counter:02d}",
                    branch_label=label,
                    source_feature_id=lane.feature_id,
                    heading=float(heading),
                    heading_delta=float(delta),
                    start_point_xy=(float(start_point[0]), float(start_point[1])),
                    terminal_pose=BranchTerminalPose(
                        x=float(terminal_point[0]),
                        y=float(terminal_point[1]),
                        heading=float(terminal_heading),
                    ),
                    rank_score=rank_score,
                    source_kind="geometry",
                    polyline_xy=candidate_polyline,
                )
            )
            counter += 1

    if not raw_candidates:
        return []
    clusters = cluster_heading_values([candidate.heading for candidate in raw_candidates], threshold_deg=heading_cluster_threshold_deg)
    clustered: List[BranchCandidate] = []
    for cluster in clusters:
        best = sorted((raw_candidates[idx] for idx in cluster), key=lambda candidate: (candidate.branch_label, candidate.rank_score, candidate.source_feature_id))[0]
        clustered.append(best)

    best_by_label: Dict[str, BranchCandidate] = {}
    for candidate in clustered:
        current = best_by_label.get(candidate.branch_label)
        if current is None or (candidate.rank_score, candidate.branch_id) < (current.rank_score, current.branch_id):
            best_by_label[candidate.branch_label] = candidate
    ordered = []
    for label in ("left", "straight", "right", "u_turn"):
        if label in best_by_label:
            ordered.append(best_by_label[label])
    return ordered


def recover_ground_truth_branch(
    canonical: CanonicalScenario,
    *,
    decision_window: DecisionWindow,
    branch_candidates: List[BranchCandidate],
    agent_id: Optional[str] = None,
) -> Tuple[GTBranchRecovery, List[BranchCandidate]]:
    resolved_agent_id = str(agent_id or decision_window.agent_id or canonical.sdc_id)
    if resolved_agent_id not in canonical.tracks:
        raise KeyError(f"agent_id={resolved_agent_id!r} not found in canonical.tracks")
    track = canonical.tracks[resolved_agent_id]
    current_idx = int(canonical.current_time_index)
    analysis_start_idx = min(max(decision_window.decision_time_idx, current_idx), max(0, track.valid.shape[0] - 1))
    valid_future = [
        idx
        for idx in range(int(analysis_start_idx), track.valid.shape[0])
        if bool(track.valid[idx]) and np.isfinite(track.position_xy[idx]).all()
    ]

    if len(valid_future) >= 2:
        realized_heading = _track_heading_at_index(track.position_xy, track.heading, track.valid, valid_future[-1], fallback=decision_window.approach_heading)
        terminal_xy = np.asarray(track.position_xy[valid_future[-1]], dtype=np.float32)
        trajectory_xy = np.asarray(track.position_xy[valid_future], dtype=np.float32)
    elif valid_future:
        realized_heading = _track_heading_at_index(track.position_xy, track.heading, track.valid, valid_future[0], fallback=decision_window.approach_heading)
        terminal_xy = np.asarray(track.position_xy[valid_future[-1]], dtype=np.float32)
        trajectory_xy = np.asarray(track.position_xy[valid_future], dtype=np.float32)
    else:
        realized_heading = float(decision_window.approach_heading)
        terminal_xy = np.asarray(track.position_xy[decision_window.decision_time_idx], dtype=np.float32)
        trajectory_xy = np.asarray(track.position_xy[decision_window.decision_time_idx : decision_window.decision_time_idx + 1], dtype=np.float32)

    ordered_candidates = list(branch_candidates)
    if not ordered_candidates:
        synthetic = BranchCandidate(
            branch_id="branch_gt_recovered",
            branch_label=classify_heading_delta(angle_delta(realized_heading, decision_window.approach_heading)),
            source_feature_id="gt_recovered",
            heading=float(realized_heading),
            heading_delta=float(angle_delta(realized_heading, decision_window.approach_heading)),
            start_point_xy=(float(trajectory_xy[0, 0]), float(trajectory_xy[0, 1])),
            terminal_pose=BranchTerminalPose(x=float(terminal_xy[0]), y=float(terminal_xy[1]), heading=float(realized_heading)),
            rank_score=0.0,
            source_kind="gt_recovered",
            polyline_xy=np.asarray(trajectory_xy, dtype=np.float32),
        )
        ordered_candidates = [synthetic]

    fit_scores = [
        _score_branch_candidate(
            candidate,
            trajectory_xy=trajectory_xy,
            trajectory_heading=realized_heading,
            stop_point_xy=decision_window.stop_point_xy,
        )
        for candidate in ordered_candidates
    ]
    fit_scores = sorted(fit_scores, key=lambda item: (item.score, item.candidate.rank_score, item.candidate.branch_id))
    best_fit = fit_scores[0]
    second_best_score = float(fit_scores[1].score) if len(fit_scores) > 1 else float("inf")
    branch_margin = float(second_best_score - best_fit.score) if np.isfinite(second_best_score) else float("inf")

    signed_stopline_progress_m = _signed_stopline_progress(
        terminal_xy,
        stop_point_xy=decision_window.stop_point_xy,
        approach_heading=decision_window.approach_heading,
    )
    dt_s = _estimate_dt_s(canonical)
    intersection_core_dwell_s = _intersection_core_dwell(
        track_position_xy=track.position_xy,
        track_valid=track.valid,
        start_idx=analysis_start_idx,
        stop_point_xy=decision_window.stop_point_xy,
        core_radius_m=INTERSECTION_CORE_RADIUS_M,
        dt_s=dt_s,
    )
    branch_commit_idx = _find_branch_commit_index(
        track_position_xy=track.position_xy,
        track_heading=track.heading,
        track_valid=track.valid,
        start_idx=analysis_start_idx,
        stop_point_xy=decision_window.stop_point_xy,
        approach_heading=decision_window.approach_heading,
        branch_candidates=ordered_candidates,
    )

    provenance = ArtifactProvenance(
        agent_id=resolved_agent_id,
        agent_role=str(decision_window.agent_role),
        current_time_index_global=current_idx,
        decision_time_index_global=int(decision_window.decision_time_idx),
        cross_time_index_global=decision_window.cross_time_idx,
        branch_commit_index_global=branch_commit_idx,
        decision_time_index_rel_to_current=int(decision_window.decision_time_idx - current_idx),
        cross_time_index_rel_to_current=(None if decision_window.cross_time_idx is None else int(decision_window.cross_time_idx - current_idx)),
        branch_commit_index_rel_to_current=(None if branch_commit_idx is None else int(branch_commit_idx - current_idx)),
        control_available_at_current=bool(decision_window.decision_time_idx >= current_idx),
    )
    commitment_metrics = CommitmentMetrics(
        signed_stopline_progress_m=float(signed_stopline_progress_m),
        downstream_progress_along_branch_m=float(best_fit.downstream_progress_m),
        intersection_core_dwell_s=float(intersection_core_dwell_s),
        best_branch_score=float(best_fit.score),
        second_best_branch_score=float(second_best_score) if np.isfinite(second_best_score) else float("inf"),
        branch_margin=float(branch_margin),
        final_heading_error_rad=float(best_fit.final_heading_error_rad),
        mean_lateral_error_to_best_branch_m=float(best_fit.mean_lateral_error_m),
    )

    decision_window.branch_commit_index_global = branch_commit_idx
    decision_window.branch_commit_index_rel_to_current = None if branch_commit_idx is None else int(branch_commit_idx - current_idx)

    gt = GTBranchRecovery(
        branch_id=best_fit.candidate.branch_id,
        branch_label=best_fit.candidate.branch_label,
        terminal_pose=best_fit.candidate.terminal_pose,
        crossed_stop_region=bool(decision_window.crossed_stop_region),
        cross_time_idx=decision_window.cross_time_idx,
        branch_recall_hit=best_fit.candidate.source_kind != "gt_recovered",
        recovered_from_existing_candidate=best_fit.candidate.source_kind != "gt_recovered",
        realized_heading=float(realized_heading),
        provenance=provenance,
        commitment_metrics=commitment_metrics,
    )
    ordered = sorted(
        ordered_candidates,
        key=lambda candidate: (
            ["left", "straight", "right", "u_turn"].index(candidate.branch_label) if candidate.branch_label in {"left", "straight", "right", "u_turn"} else 99,
            candidate.rank_score,
            candidate.branch_id,
        ),
    )
    return gt, ordered


def _polyline_direction_from_index(polyline_xy: np.ndarray, nearest_idx: int, direction: int) -> np.ndarray:
    if int(direction) > 0:
        return np.asarray(polyline_xy[nearest_idx:], dtype=np.float32)
    return np.asarray(polyline_xy[: nearest_idx + 1][::-1], dtype=np.float32)


def _track_heading_at_index(
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray,
    idx: int,
    *,
    fallback: float,
) -> float:
    idx = int(np.clip(int(idx), 0, max(0, position_xy.shape[0] - 1)))
    if idx < heading.shape[0] and np.isfinite(heading[idx]):
        return float(heading[idx])
    prev_valid = [
        t
        for t in range(max(0, idx - 5), idx + 1)
        if bool(valid_mask[t]) and np.isfinite(position_xy[t]).all()
    ]
    if len(prev_valid) >= 2:
        return float(heading_from_points(position_xy[prev_valid[-2]], position_xy[prev_valid[-1]]))
    return float(fallback)


def _first_stopline_crossing_index(
    *,
    track_position_xy: np.ndarray,
    track_valid: np.ndarray,
    start_idx: int,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
) -> Optional[int]:
    for idx in range(int(start_idx), track_position_xy.shape[0]):
        if not bool(track_valid[idx]) or not np.isfinite(track_position_xy[idx]).all():
            continue
        progress = _signed_stopline_progress(track_position_xy[idx], stop_point_xy=stop_point_xy, approach_heading=approach_heading)
        if progress >= 0.0:
            return int(idx)
    return None


def _signed_stopline_progress(
    point_xy: Sequence[float],
    *,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
) -> float:
    delta = np.asarray(point_xy, dtype=np.float32)[:2] - np.asarray(stop_point_xy, dtype=np.float32)
    forward = np.asarray([math.cos(float(approach_heading)), math.sin(float(approach_heading))], dtype=np.float32)
    return float(np.dot(delta, forward))


def _project_point_to_polyline(polyline_xy: np.ndarray, point_xy: Sequence[float]) -> Tuple[float, float, float]:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)[:2]
    if polyline.shape[0] == 0:
        return 0.0, float("inf"), 0.0
    if polyline.shape[0] == 1:
        return 0.0, float(np.linalg.norm(point - polyline[0])), 0.0

    cumulative = cumulative_polyline_lengths(polyline)
    best_s = 0.0
    best_dist = float("inf")
    best_heading = heading_from_points(polyline[0], polyline[1])
    for idx in range(polyline.shape[0] - 1):
        p0 = polyline[idx]
        p1 = polyline[idx + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-8:
            continue
        alpha = float(np.clip(np.dot(point - p0, seg) / seg_len_sq, 0.0, 1.0))
        proj = p0 + alpha * seg
        dist = float(np.linalg.norm(point - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(cumulative[idx] + alpha * math.sqrt(seg_len_sq))
            best_heading = heading_from_points(p0, p1)
    return best_s, best_dist, best_heading


def _score_branch_candidate(
    candidate: BranchCandidate,
    *,
    trajectory_xy: np.ndarray,
    trajectory_heading: float,
    stop_point_xy: Tuple[float, float],
) -> _BranchFitScore:
    polyline = np.asarray(candidate.polyline_xy, dtype=np.float32)
    if polyline.shape[0] < 2 or trajectory_xy.shape[0] == 0:
        return _BranchFitScore(
            candidate=candidate,
            score=float("inf"),
            downstream_progress_m=0.0,
            final_heading_error_rad=float("inf"),
            mean_lateral_error_m=float("inf"),
        )

    stop_s, _, _ = _project_point_to_polyline(polyline, stop_point_xy)
    projected_s: List[float] = []
    lateral_errors: List[float] = []
    final_branch_heading = candidate.terminal_pose.heading
    for point in trajectory_xy:
        s, dist, heading = _project_point_to_polyline(polyline, point)
        projected_s.append(float(s))
        lateral_errors.append(float(dist))
        final_branch_heading = float(heading)

    downstream_progress = max(0.0, max(projected_s) - stop_s)
    mean_lateral_error = float(np.mean(lateral_errors)) if lateral_errors else float("inf")
    final_heading_error = abs(angle_delta(float(trajectory_heading), float(final_branch_heading)))
    terminal_dist = float(np.linalg.norm(np.asarray(trajectory_xy[-1], dtype=np.float32) - np.asarray([candidate.terminal_pose.x, candidate.terminal_pose.y], dtype=np.float32)))
    score = float(mean_lateral_error + 0.75 * final_heading_error + 0.05 * terminal_dist)
    return _BranchFitScore(
        candidate=candidate,
        score=score,
        downstream_progress_m=float(downstream_progress),
        final_heading_error_rad=float(final_heading_error),
        mean_lateral_error_m=float(mean_lateral_error),
    )


def _estimate_dt_s(canonical: CanonicalScenario) -> float:
    ts = np.asarray(canonical.ts, dtype=np.float32)
    if ts.shape[0] < 2:
        return 0.1
    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size > 0 else 0.1


def _intersection_core_dwell(
    *,
    track_position_xy: np.ndarray,
    track_valid: np.ndarray,
    start_idx: int,
    stop_point_xy: Tuple[float, float],
    core_radius_m: float,
    dt_s: float,
) -> float:
    center = np.asarray(stop_point_xy, dtype=np.float32)
    count = 0
    for idx in range(int(start_idx), track_position_xy.shape[0]):
        if not bool(track_valid[idx]) or not np.isfinite(track_position_xy[idx]).all():
            continue
        if float(np.linalg.norm(np.asarray(track_position_xy[idx], dtype=np.float32) - center)) <= float(core_radius_m):
            count += 1
    return float(count * dt_s)


def _find_branch_commit_index(
    *,
    track_position_xy: np.ndarray,
    track_heading: np.ndarray,
    track_valid: np.ndarray,
    start_idx: int,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
    branch_candidates: Sequence[BranchCandidate],
    branch_margin_threshold: float = PATH_CHOICE_MIN_BRANCH_MARGIN,
    heading_error_threshold_rad: float = PATH_CHOICE_MAX_FINAL_HEADING_ERROR_RAD,
) -> Optional[int]:
    valid_idx = [
        idx
        for idx in range(int(start_idx), track_position_xy.shape[0])
        if bool(track_valid[idx]) and np.isfinite(track_position_xy[idx]).all()
    ]
    if len(valid_idx) < 2 or len(branch_candidates) < 2:
        return None

    for idx in valid_idx[1:]:
        prefix_idx = [
            t
            for t in valid_idx
            if t <= idx
        ]
        if len(prefix_idx) < 2:
            continue
        trajectory_xy = np.asarray(track_position_xy[prefix_idx], dtype=np.float32)
        trajectory_heading = _track_heading_at_index(track_position_xy, track_heading, track_valid, idx, fallback=approach_heading)
        fit_scores = sorted(
            (
                _score_branch_candidate(
                    candidate,
                    trajectory_xy=trajectory_xy,
                    trajectory_heading=trajectory_heading,
                    stop_point_xy=stop_point_xy,
                )
                for candidate in branch_candidates
            ),
            key=lambda item: (item.score, item.candidate.rank_score, item.candidate.branch_id),
        )
        if len(fit_scores) < 2:
            continue
        best = fit_scores[0]
        second = fit_scores[1]
        signed_progress = _signed_stopline_progress(
            track_position_xy[idx],
            stop_point_xy=stop_point_xy,
            approach_heading=approach_heading,
        )
        branch_margin = float(second.score - best.score)
        if (
            signed_progress >= 3.0
            and best.downstream_progress_m >= 8.0
            and branch_margin >= float(branch_margin_threshold)
            and best.final_heading_error_rad <= float(heading_error_threshold_rad)
        ):
            return int(idx)
    return None
