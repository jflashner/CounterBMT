from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .branch_enumeration import BranchCandidate, BranchTerminalPose
from .geometry import (
    angle_delta,
    any_point_within_radius,
    classify_heading_delta,
    cumulative_polyline_lengths,
    heading_from_points,
    polyline_length,
)
from .types import CanonicalScenario, stable_string_sort_key


LANE_LIKE_TYPES = ("LANE_",)
ROUTE_GRAPH_RADIUS_M = 90.0
HOST_MATCH_MAX_DIST_M = 12.0
HOST_MATCH_MAX_HEADING_ERROR_RAD = math.radians(75.0)
EDGE_ENDPOINT_DIST_M = 4.0
EDGE_HEADING_CONTINUITY_RAD = math.radians(55.0)
MAX_ROUTE_DEPTH = 6
MAX_ROUTE_LENGTH_M = 95.0
MIN_ROUTE_LENGTH_AFTER_DECISION_M = 12.0
LOOKAHEAD_HEADING_M = 18.0
EXIT_GATE_LENGTH_M = 5.0
EXIT_GATE_WIDTH_M = 4.0
GT_AMBIGUITY_MARGIN = 1.5
GT_MAX_BEST_SCORE = 12.0


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _finite_heading(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(value)
    except Exception:
        return float(default)
    return scalar if math.isfinite(scalar) else float(default)


@dataclass
class DirectedLaneNode:
    node_id: str
    feature_id: str
    direction: int
    polyline_xy: np.ndarray
    length_m: float
    heading_start: float
    heading_end: float
    start_point_xy: Tuple[float, float]
    end_point_xy: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class HostLaneMatch:
    node_id: str
    feature_id: str
    direction: int
    distance_m: float
    heading_error_rad: float
    projected_s_m: float
    projected_xy: Tuple[float, float]
    tangent_heading: float
    score: float
    used_heading_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class ExitGateV2:
    centerline_xy: np.ndarray
    polygon_xy: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class BranchRouteFamilyV2:
    branch_id: str
    branch_label: str
    route_node_ids: List[str]
    route_lane_feature_ids: List[str]
    polyline_xy: np.ndarray
    branch_suffix_xy: np.ndarray
    split_point_xy: Tuple[float, float]
    exit_gate: ExitGateV2
    heading_delta: float
    rank_score: float
    route_length_m: float
    decision_to_exit_length_m: float
    source_feature_id: str

    def to_branch_candidate(self) -> BranchCandidate:
        polyline = np.asarray(self.polyline_xy, dtype=np.float32)
        terminal_heading = float(
            heading_from_points(polyline[-2], polyline[-1]) if polyline.shape[0] >= 2 else 0.0
        )
        return BranchCandidate(
            branch_id=str(self.branch_id),
            branch_label=str(self.branch_label),
            source_feature_id=str(self.source_feature_id),
            heading=float(terminal_heading),
            heading_delta=float(self.heading_delta),
            start_point_xy=(float(polyline[0, 0]), float(polyline[0, 1])) if polyline.shape[0] else (0.0, 0.0),
            terminal_pose=BranchTerminalPose(
                x=float(polyline[-1, 0]) if polyline.shape[0] else 0.0,
                y=float(polyline[-1, 1]) if polyline.shape[0] else 0.0,
                heading=terminal_heading,
            ),
            rank_score=float(self.rank_score),
            source_kind="route_v2",
            polyline_xy=np.asarray(polyline, dtype=np.float32),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["exit_gate"] = self.exit_gate.to_dict()
        return _jsonify(payload)


@dataclass
class BranchRoutesV2Result:
    agent_id: str
    current_time_idx: int
    decision_time_idx: int
    stop_point_xy: Tuple[float, float]
    approach_heading: float
    host_lane_current: Optional[HostLaneMatch]
    host_lane_decision: Optional[HostLaneMatch]
    current_to_decision_connected: bool
    connector_node_ids: List[str]
    local_graph_radius_m: float
    local_lane_node_count: int
    local_lane_edge_count: int
    shared_stem_xy: np.ndarray
    split_point_xy: Optional[Tuple[float, float]]
    route_families: List[BranchRouteFamilyV2] = field(default_factory=list)

    def to_branch_candidates(self) -> List[BranchCandidate]:
        return [family.to_branch_candidate() for family in self.route_families]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(
            {
                "agent_id": self.agent_id,
                "current_time_idx": int(self.current_time_idx),
                "decision_time_idx": int(self.decision_time_idx),
                "stop_point_xy": tuple(float(v) for v in self.stop_point_xy),
                "approach_heading": float(self.approach_heading),
                "host_lane_current": None if self.host_lane_current is None else self.host_lane_current.to_dict(),
                "host_lane_decision": None if self.host_lane_decision is None else self.host_lane_decision.to_dict(),
                "current_to_decision_connected": bool(self.current_to_decision_connected),
                "connector_node_ids": list(self.connector_node_ids),
                "local_graph_radius_m": float(self.local_graph_radius_m),
                "local_lane_node_count": int(self.local_lane_node_count),
                "local_lane_edge_count": int(self.local_lane_edge_count),
                "shared_stem_xy": np.asarray(self.shared_stem_xy, dtype=np.float32),
                "split_point_xy": None if self.split_point_xy is None else tuple(float(v) for v in self.split_point_xy),
                "route_families": [family.to_dict() for family in self.route_families],
            }
        )


def _valid_track_pose(track: Any, time_index: int) -> Optional[Tuple[np.ndarray, float]]:
    valid = np.asarray(track.valid, dtype=bool)
    position_xy = np.asarray(track.position_xy, dtype=np.float32)
    heading = np.asarray(track.heading, dtype=np.float32)
    if valid.size == 0:
        return None
    idx = int(np.clip(int(time_index), 0, valid.shape[0] - 1))
    if not bool(valid[idx]) or not np.isfinite(position_xy[idx]).all():
        before = np.flatnonzero(valid[: idx + 1])
        if before.size > 0:
            idx = int(before[-1])
        else:
            after = np.flatnonzero(valid[idx:])
            if after.size == 0:
                return None
            idx = int(idx + after[0])
    return np.asarray(position_xy[idx, :2], dtype=np.float32), _finite_heading(heading[idx])


def _lane_like_features_near_centers(
    canonical: CanonicalScenario,
    *,
    center_points_xy: Sequence[Sequence[float]],
    radius_m: float,
) -> List[Tuple[str, Any]]:
    centers = [np.asarray(center[:2], dtype=np.float32) for center in center_points_xy]
    features: List[Tuple[str, Any]] = []
    for feature_id, feature in sorted(canonical.map_features.items(), key=lambda item: stable_string_sort_key(item[0])):
        feature_type = str(feature.feature_type)
        if not feature_type.startswith(LANE_LIKE_TYPES):
            continue
        polyline = np.asarray(feature.polyline_xy, dtype=np.float32)
        if polyline.shape[0] < 2:
            continue
        if not any(any_point_within_radius(polyline, center, radius_m) for center in centers):
            continue
        features.append((str(feature_id), feature))
    return features


def _make_directed_lane_nodes(
    canonical: CanonicalScenario,
    *,
    center_points_xy: Sequence[Sequence[float]],
    radius_m: float,
) -> Dict[str, DirectedLaneNode]:
    nodes: Dict[str, DirectedLaneNode] = {}
    for feature_id, feature in _lane_like_features_near_centers(canonical, center_points_xy=center_points_xy, radius_m=radius_m):
        polyline = np.asarray(feature.polyline_xy, dtype=np.float32)
        if polyline.shape[0] < 2 or polyline_length(polyline) < 3.0:
            continue
        for direction, suffix in ((1, "fwd"), (-1, "rev")):
            directed = np.asarray(polyline if direction > 0 else polyline[::-1], dtype=np.float32)
            if directed.shape[0] < 2:
                continue
            node_id = f"{feature_id}:{suffix}"
            nodes[node_id] = DirectedLaneNode(
                node_id=node_id,
                feature_id=str(feature_id),
                direction=int(direction),
                polyline_xy=directed,
                length_m=float(polyline_length(directed)),
                heading_start=float(heading_from_points(directed[0], directed[min(1, directed.shape[0] - 1)])),
                heading_end=float(heading_from_points(directed[max(0, directed.shape[0] - 2)], directed[-1])),
                start_point_xy=(float(directed[0, 0]), float(directed[0, 1])),
                end_point_xy=(float(directed[-1, 0]), float(directed[-1, 1])),
            )
    return nodes


def _project_point_to_polyline(polyline_xy: np.ndarray, point_xy: Sequence[float]) -> Tuple[float, float, float, np.ndarray]:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)[:2]
    if polyline.shape[0] == 0:
        return 0.0, float("inf"), 0.0, np.zeros((2,), dtype=np.float32)
    if polyline.shape[0] == 1:
        return 0.0, float(np.linalg.norm(point - polyline[0])), 0.0, np.asarray(polyline[0], dtype=np.float32)

    cumulative = cumulative_polyline_lengths(polyline)
    best_s = 0.0
    best_dist = float("inf")
    best_heading = heading_from_points(polyline[0], polyline[1])
    best_proj = np.asarray(polyline[0], dtype=np.float32)
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
            best_heading = float(heading_from_points(p0, p1))
            best_proj = np.asarray(proj, dtype=np.float32)
    return best_s, best_dist, best_heading, best_proj


def _sample_point_along_polyline(polyline_xy: np.ndarray, s_m: float) -> np.ndarray:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    if polyline.shape[0] == 0:
        return np.zeros((2,), dtype=np.float32)
    if polyline.shape[0] == 1:
        return np.asarray(polyline[0], dtype=np.float32)
    cumulative = cumulative_polyline_lengths(polyline)
    target_s = float(np.clip(float(s_m), 0.0, float(cumulative[-1])))
    idx = int(np.searchsorted(cumulative, target_s, side="right") - 1)
    idx = min(max(idx, 0), polyline.shape[0] - 2)
    p0 = polyline[idx]
    p1 = polyline[idx + 1]
    seg_len = float(max(cumulative[idx + 1] - cumulative[idx], 1e-6))
    alpha = float(np.clip((target_s - cumulative[idx]) / seg_len, 0.0, 1.0))
    return np.asarray(p0 + alpha * (p1 - p0), dtype=np.float32)


def _slice_polyline(polyline_xy: np.ndarray, *, start_s: float = 0.0, end_s: Optional[float] = None) -> np.ndarray:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    if polyline.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cumulative = cumulative_polyline_lengths(polyline)
    total = float(cumulative[-1]) if cumulative.size else 0.0
    start_val = float(np.clip(float(start_s), 0.0, total))
    end_val = float(total if end_s is None else np.clip(float(end_s), start_val, total))
    if end_val <= start_val + 1e-4:
        point = _sample_point_along_polyline(polyline, start_val)
        return point.reshape(1, 2)

    points: List[np.ndarray] = [_sample_point_along_polyline(polyline, start_val)]
    for idx, s_val in enumerate(cumulative.tolist()):
        if start_val < float(s_val) < end_val:
            points.append(np.asarray(polyline[idx], dtype=np.float32))
    points.append(_sample_point_along_polyline(polyline, end_val))
    out = np.asarray(points, dtype=np.float32)
    if out.shape[0] <= 1:
        return out
    deduped = [out[0]]
    for point in out[1:]:
        if float(np.linalg.norm(point - deduped[-1])) > 1e-3:
            deduped.append(point)
    return np.asarray(deduped, dtype=np.float32)


def _match_pose_to_host_lane(
    nodes: Mapping[str, DirectedLaneNode],
    *,
    pose_xy: Sequence[float],
    pose_heading: float,
) -> Optional[HostLaneMatch]:
    best: Optional[HostLaneMatch] = None
    best_fallback: Optional[HostLaneMatch] = None
    for node in nodes.values():
        projected_s, distance_m, tangent_heading, projected_xy = _project_point_to_polyline(node.polyline_xy, pose_xy)
        heading_error = abs(angle_delta(tangent_heading, pose_heading))
        score = float(distance_m + 5.0 * heading_error)
        match = HostLaneMatch(
            node_id=str(node.node_id),
            feature_id=str(node.feature_id),
            direction=int(node.direction),
            distance_m=float(distance_m),
            heading_error_rad=float(heading_error),
            projected_s_m=float(projected_s),
            projected_xy=(float(projected_xy[0]), float(projected_xy[1])),
            tangent_heading=float(tangent_heading),
            score=score,
            used_heading_fallback=False,
        )
        if distance_m <= HOST_MATCH_MAX_DIST_M and heading_error <= HOST_MATCH_MAX_HEADING_ERROR_RAD:
            if best is None or match.score < best.score:
                best = match
        if best_fallback is None or (distance_m, heading_error, score, node.node_id) < (
            best_fallback.distance_m,
            best_fallback.heading_error_rad,
            best_fallback.score,
            best_fallback.node_id,
        ):
            best_fallback = match
    if best is not None:
        return best
    if best_fallback is None:
        return None
    best_fallback.used_heading_fallback = True
    return best_fallback


def _build_lane_graph(nodes: Mapping[str, DirectedLaneNode]) -> Dict[str, List[Tuple[str, float]]]:
    adjacency: Dict[str, List[Tuple[str, float]]] = {str(node_id): [] for node_id in nodes}
    for src_id, src in nodes.items():
        src_end = np.asarray(src.end_point_xy, dtype=np.float32)
        for dst_id, dst in nodes.items():
            if src_id == dst_id:
                continue
            if str(src.feature_id) == str(dst.feature_id) and int(src.direction) != int(dst.direction):
                continue
            dst_start = np.asarray(dst.start_point_xy, dtype=np.float32)
            endpoint_gap = float(np.linalg.norm(src_end - dst_start))
            if endpoint_gap > EDGE_ENDPOINT_DIST_M:
                continue
            heading_gap = abs(angle_delta(dst.heading_start, src.heading_end))
            if heading_gap > EDGE_HEADING_CONTINUITY_RAD:
                continue
            cost = float(endpoint_gap + 2.0 * heading_gap)
            adjacency[src_id].append((dst_id, cost))
    for node_id in adjacency:
        adjacency[node_id].sort(key=lambda item: (item[1], item[0]))
    return adjacency


def _shortest_connector_path(
    adjacency: Mapping[str, Sequence[Tuple[str, float]]],
    *,
    start_node_id: str,
    end_node_id: str,
    max_depth: int = 6,
) -> List[str]:
    if str(start_node_id) == str(end_node_id):
        return [str(start_node_id)]
    frontier: List[Tuple[float, str, List[str]]] = [(0.0, str(start_node_id), [str(start_node_id)])]
    best_cost: Dict[str, float] = {str(start_node_id): 0.0}
    while frontier:
        frontier.sort(key=lambda item: (item[0], len(item[2]), item[1]))
        cost, node_id, path = frontier.pop(0)
        if len(path) > max_depth:
            continue
        for next_id, edge_cost in adjacency.get(node_id, []):
            if next_id in path:
                continue
            next_cost = float(cost + edge_cost)
            if next_cost >= best_cost.get(next_id, float("inf")):
                continue
            next_path = path + [str(next_id)]
            if next_id == str(end_node_id):
                return next_path
            best_cost[next_id] = next_cost
            frontier.append((next_cost, str(next_id), next_path))
    return []


def _enumerate_suffix_sequences(
    nodes: Mapping[str, DirectedLaneNode],
    adjacency: Mapping[str, Sequence[Tuple[str, float]]],
    *,
    start_node_id: str,
    start_s_m: float = 0.0,
    max_depth: int = MAX_ROUTE_DEPTH,
    max_total_length_m: float = MAX_ROUTE_LENGTH_M,
) -> List[List[str]]:
    results: List[List[str]] = []
    start_remaining_length = max(0.0, float(nodes[str(start_node_id)].length_m) - float(start_s_m))
    stack: List[Tuple[str, List[str], float]] = [(str(start_node_id), [str(start_node_id)], float(start_remaining_length))]
    while stack:
        node_id, path, length_m = stack.pop()
        outgoing = [item for item in adjacency.get(node_id, []) if item[0] not in path]
        if length_m >= MIN_ROUTE_LENGTH_AFTER_DECISION_M and (not outgoing or len(path) >= max_depth or length_m >= max_total_length_m):
            results.append(list(path))
            continue
        if len(path) >= max_depth:
            results.append(list(path))
            continue
        for next_id, edge_cost in reversed(outgoing[:4]):
            next_length = float(length_m + float(nodes[next_id].length_m))
            if next_length > max_total_length_m:
                results.append(list(path))
                continue
            stack.append((str(next_id), path + [str(next_id)], next_length))
    if not results:
        results.append([str(start_node_id)])
    deduped: List[List[str]] = []
    seen: set[Tuple[str, ...]] = set()
    for path in results:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _stitch_node_sequence(
    node_ids: Sequence[str],
    *,
    nodes: Mapping[str, DirectedLaneNode],
    first_node_start_s: float,
) -> np.ndarray:
    stitched: List[np.ndarray] = []
    for idx, node_id in enumerate(node_ids):
        node = nodes[str(node_id)]
        polyline = np.asarray(node.polyline_xy, dtype=np.float32)
        part = _slice_polyline(polyline, start_s=(float(first_node_start_s) if idx == 0 else 0.0))
        if part.shape[0] == 0:
            continue
        if stitched:
            if float(np.linalg.norm(part[0] - stitched[-1][-1])) <= 1.5:
                part = part[1:]
        if part.shape[0] == 0:
            continue
        stitched.append(part)
    if not stitched:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(stitched, axis=0)


def _common_prefix_node_ids(sequences: Sequence[Sequence[str]]) -> List[str]:
    if not sequences:
        return []
    prefix = list(sequences[0])
    for sequence in sequences[1:]:
        limit = min(len(prefix), len(sequence))
        keep = 0
        for idx in range(limit):
            if str(prefix[idx]) != str(sequence[idx]):
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            break
    return prefix


def _heading_from_route(polyline_xy: np.ndarray, *, decision_point_xy: Sequence[float], approach_heading: float) -> Tuple[float, float]:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    if polyline.shape[0] < 2:
        return float(approach_heading), 0.0
    decision_s, _, _, _ = _project_point_to_polyline(polyline, decision_point_xy)
    start_pt = _sample_point_along_polyline(polyline, min(float(decision_s + 2.0), float(cumulative_polyline_lengths(polyline)[-1])))
    end_pt = _sample_point_along_polyline(polyline, min(float(decision_s + LOOKAHEAD_HEADING_M), float(cumulative_polyline_lengths(polyline)[-1])))
    heading = float(heading_from_points(start_pt, end_pt))
    return heading, float(angle_delta(heading, approach_heading))


def _slice_from_point(polyline_xy: np.ndarray, point_xy: Sequence[float]) -> np.ndarray:
    s_val, _, _, _ = _project_point_to_polyline(np.asarray(polyline_xy, dtype=np.float32), point_xy)
    return _slice_polyline(np.asarray(polyline_xy, dtype=np.float32), start_s=float(s_val))


def _build_exit_gate(polyline_xy: np.ndarray) -> ExitGateV2:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    if polyline.shape[0] < 2:
        return ExitGateV2(centerline_xy=np.asarray(polyline, dtype=np.float32), polygon_xy=np.zeros((0, 2), dtype=np.float32))
    cumulative = cumulative_polyline_lengths(polyline)
    total = float(cumulative[-1])
    gate_line = _slice_polyline(polyline, start_s=max(0.0, total - EXIT_GATE_LENGTH_M), end_s=total)
    if gate_line.shape[0] < 2:
        gate_line = np.asarray(polyline[-2:], dtype=np.float32)
    p0 = np.asarray(gate_line[0], dtype=np.float32)
    p1 = np.asarray(gate_line[-1], dtype=np.float32)
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        normal = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        unit = direction / norm
        normal = np.asarray([-unit[1], unit[0]], dtype=np.float32)
    offset = normal * (EXIT_GATE_WIDTH_M / 2.0)
    polygon = np.asarray([p0 + offset, p1 + offset, p1 - offset, p0 - offset], dtype=np.float32)
    return ExitGateV2(centerline_xy=np.asarray(gate_line, dtype=np.float32), polygon_xy=polygon)


def _route_rank_score(
    *,
    current_match: Optional[HostLaneMatch],
    decision_match: Optional[HostLaneMatch],
    connector_node_ids: Sequence[str],
    route_length_m: float,
    heading_delta: float,
) -> float:
    match_score = 0.0
    if current_match is not None:
        match_score += float(current_match.score)
    if decision_match is not None:
        match_score += float(decision_match.score)
    connector_penalty = 0.5 * max(0, len(connector_node_ids) - 1)
    length_penalty = 0.02 * max(0.0, 30.0 - float(route_length_m))
    heading_penalty = 0.25 * abs(float(heading_delta))
    return float(match_score + connector_penalty + length_penalty + heading_penalty)


def build_branch_routes_v2(
    canonical: CanonicalScenario,
    *,
    agent_id: str,
    current_time_idx: int,
    decision_time_idx: int,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
    radius_m: float = ROUTE_GRAPH_RADIUS_M,
) -> BranchRoutesV2Result:
    if str(agent_id) not in canonical.tracks:
        return BranchRoutesV2Result(
            agent_id=str(agent_id),
            current_time_idx=int(current_time_idx),
            decision_time_idx=int(decision_time_idx),
            stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
            approach_heading=float(approach_heading),
            host_lane_current=None,
            host_lane_decision=None,
            current_to_decision_connected=False,
            connector_node_ids=[],
            local_graph_radius_m=float(radius_m),
            local_lane_node_count=0,
            local_lane_edge_count=0,
            shared_stem_xy=np.zeros((0, 2), dtype=np.float32),
            split_point_xy=None,
            route_families=[],
        )

    track = canonical.tracks[str(agent_id)]
    current_pose = _valid_track_pose(track, int(current_time_idx))
    decision_pose = _valid_track_pose(track, int(decision_time_idx))
    if current_pose is None or decision_pose is None:
        return BranchRoutesV2Result(
            agent_id=str(agent_id),
            current_time_idx=int(current_time_idx),
            decision_time_idx=int(decision_time_idx),
            stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
            approach_heading=float(approach_heading),
            host_lane_current=None,
            host_lane_decision=None,
            current_to_decision_connected=False,
            connector_node_ids=[],
            local_graph_radius_m=float(radius_m),
            local_lane_node_count=0,
            local_lane_edge_count=0,
            shared_stem_xy=np.zeros((0, 2), dtype=np.float32),
            split_point_xy=None,
            route_families=[],
        )

    current_pose_xy, current_heading = current_pose
    decision_pose_xy, decision_heading = decision_pose
    nodes = _make_directed_lane_nodes(
        canonical,
        center_points_xy=[current_pose_xy, decision_pose_xy, stop_point_xy],
        radius_m=float(radius_m),
    )
    adjacency = _build_lane_graph(nodes)
    edge_count = int(sum(len(v) for v in adjacency.values()))

    current_match = _match_pose_to_host_lane(nodes, pose_xy=current_pose_xy, pose_heading=current_heading) if nodes else None
    decision_match = _match_pose_to_host_lane(nodes, pose_xy=decision_pose_xy, pose_heading=decision_heading) if nodes else None

    if decision_match is None:
        return BranchRoutesV2Result(
            agent_id=str(agent_id),
            current_time_idx=int(current_time_idx),
            decision_time_idx=int(decision_time_idx),
            stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
            approach_heading=float(approach_heading),
            host_lane_current=current_match,
            host_lane_decision=None,
            current_to_decision_connected=False,
            connector_node_ids=[],
            local_graph_radius_m=float(radius_m),
            local_lane_node_count=int(len(nodes)),
            local_lane_edge_count=edge_count,
            shared_stem_xy=np.zeros((0, 2), dtype=np.float32),
            split_point_xy=None,
            route_families=[],
        )

    connector_node_ids: List[str] = [str(decision_match.node_id)]
    route_start_s = float(decision_match.projected_s_m)
    connected = False
    if current_match is not None:
        if str(current_match.node_id) == str(decision_match.node_id):
            connector_node_ids = [str(decision_match.node_id)]
            route_start_s = float(min(current_match.projected_s_m, decision_match.projected_s_m))
            connected = True
        else:
            connector = _shortest_connector_path(
                adjacency,
                start_node_id=str(current_match.node_id),
                end_node_id=str(decision_match.node_id),
            )
            if connector:
                connector_node_ids = list(connector)
                route_start_s = float(current_match.projected_s_m)
                connected = True

    suffix_sequences = _enumerate_suffix_sequences(
        nodes,
        adjacency,
        start_node_id=str(decision_match.node_id),
        start_s_m=float(decision_match.projected_s_m),
    )

    sequence_records: List[Dict[str, Any]] = []
    connector_prefix = list(connector_node_ids[:-1]) if connector_node_ids else []
    for suffix_ids in suffix_sequences:
        full_node_ids = connector_prefix + list(suffix_ids)
        polyline = _stitch_node_sequence(full_node_ids, nodes=nodes, first_node_start_s=route_start_s)
        if polyline.shape[0] < 2:
            continue
        route_length = float(polyline_length(polyline))
        if route_length < MIN_ROUTE_LENGTH_AFTER_DECISION_M:
            continue
        sequence_records.append(
            {
                "route_node_ids": list(full_node_ids),
                "route_lane_feature_ids": [str(nodes[node_id].feature_id) for node_id in full_node_ids],
                "polyline_xy": np.asarray(polyline, dtype=np.float32),
                "route_length_m": float(route_length),
                "rank_score": _route_rank_score(
                    current_match=current_match,
                    decision_match=decision_match,
                    connector_node_ids=connector_node_ids,
                    route_length_m=route_length,
                    heading_delta=0.0,
                ),
                "source_feature_id": str(nodes[full_node_ids[-1]].feature_id),
            }
        )

    if not sequence_records:
        return BranchRoutesV2Result(
            agent_id=str(agent_id),
            current_time_idx=int(current_time_idx),
            decision_time_idx=int(decision_time_idx),
            stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
            approach_heading=float(approach_heading),
            host_lane_current=current_match,
            host_lane_decision=decision_match,
            current_to_decision_connected=connected,
            connector_node_ids=list(connector_node_ids),
            local_graph_radius_m=float(radius_m),
            local_lane_node_count=int(len(nodes)),
            local_lane_edge_count=edge_count,
            shared_stem_xy=np.zeros((0, 2), dtype=np.float32),
            split_point_xy=None,
            route_families=[],
        )

    common_prefix_ids = _common_prefix_node_ids([record["route_node_ids"] for record in sequence_records])
    if common_prefix_ids:
        shared_stem_xy = _stitch_node_sequence(common_prefix_ids, nodes=nodes, first_node_start_s=route_start_s)
    else:
        shared_stem_xy = np.zeros((0, 2), dtype=np.float32)
    split_point_xy: Tuple[float, float]
    if shared_stem_xy.shape[0] > 0:
        split_point_xy = (float(shared_stem_xy[-1, 0]), float(shared_stem_xy[-1, 1]))
    else:
        split_point_xy = (float(decision_pose_xy[0]), float(decision_pose_xy[1]))

    family_candidates: Dict[str, BranchRouteFamilyV2] = {}
    for record in sequence_records:
        polyline = np.asarray(record["polyline_xy"], dtype=np.float32)
        suffix_xy = _slice_from_point(polyline, split_point_xy)
        heading_polyline = suffix_xy if suffix_xy.shape[0] >= 2 else polyline
        terminal_heading, heading_delta = _heading_from_route(
            heading_polyline,
            decision_point_xy=np.asarray(split_point_xy, dtype=np.float32),
            approach_heading=float(approach_heading),
        )
        branch_label = classify_heading_delta(float(heading_delta))
        if str(branch_label) == "unknown":
            continue
        rank_score = _route_rank_score(
            current_match=current_match,
            decision_match=decision_match,
            connector_node_ids=connector_node_ids,
            route_length_m=float(record["route_length_m"]),
            heading_delta=heading_delta,
        )
        candidate = BranchRouteFamilyV2(
            branch_id=f"route_v2_{branch_label}",
            branch_label=str(branch_label),
            route_node_ids=list(record["route_node_ids"]),
            route_lane_feature_ids=list(record["route_lane_feature_ids"]),
            polyline_xy=np.asarray(polyline, dtype=np.float32),
            branch_suffix_xy=np.asarray(suffix_xy, dtype=np.float32),
            split_point_xy=tuple(float(v) for v in split_point_xy),
            exit_gate=_build_exit_gate(suffix_xy if suffix_xy.shape[0] >= 2 else polyline),
            heading_delta=float(heading_delta),
            rank_score=float(rank_score),
            route_length_m=float(record["route_length_m"]),
            decision_to_exit_length_m=max(0.0, float(record["route_length_m"] - decision_match.projected_s_m)),
            source_feature_id=str(record["source_feature_id"]),
        )
        current_best = family_candidates.get(str(branch_label))
        if current_best is None or (
            candidate.rank_score,
            candidate.route_length_m,
            candidate.branch_id,
        ) < (
            current_best.rank_score,
            current_best.route_length_m,
            current_best.branch_id,
        ):
            family_candidates[str(branch_label)] = candidate

    ordered_families = [family_candidates[label] for label in ("left", "straight", "right", "u_turn") if label in family_candidates]
    if not ordered_families:
        return BranchRoutesV2Result(
            agent_id=str(agent_id),
            current_time_idx=int(current_time_idx),
            decision_time_idx=int(decision_time_idx),
            stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
            approach_heading=float(approach_heading),
            host_lane_current=current_match,
            host_lane_decision=decision_match,
            current_to_decision_connected=connected,
            connector_node_ids=list(connector_node_ids),
            local_graph_radius_m=float(radius_m),
            local_lane_node_count=int(len(nodes)),
            local_lane_edge_count=edge_count,
            shared_stem_xy=np.zeros((0, 2), dtype=np.float32),
            split_point_xy=None,
            route_families=[],
        )

    for family in ordered_families:
        family.split_point_xy = tuple(float(v) for v in split_point_xy)
        family.branch_suffix_xy = _slice_from_point(family.polyline_xy, split_point_xy)

    return BranchRoutesV2Result(
        agent_id=str(agent_id),
        current_time_idx=int(current_time_idx),
        decision_time_idx=int(decision_time_idx),
        stop_point_xy=(float(stop_point_xy[0]), float(stop_point_xy[1])),
        approach_heading=float(approach_heading),
        host_lane_current=current_match,
        host_lane_decision=decision_match,
        current_to_decision_connected=bool(connected),
        connector_node_ids=list(connector_node_ids),
        local_graph_radius_m=float(radius_m),
        local_lane_node_count=int(len(nodes)),
        local_lane_edge_count=edge_count,
        shared_stem_xy=np.asarray(shared_stem_xy, dtype=np.float32),
        split_point_xy=None if split_point_xy is None else tuple(float(v) for v in split_point_xy),
        route_families=ordered_families,
    )


def enumerate_branch_candidates_from_routes_v2(
    canonical: CanonicalScenario,
    *,
    agent_id: str,
    current_time_idx: int,
    decision_time_idx: int,
    stop_point_xy: Tuple[float, float],
    approach_heading: float,
    radius_m: float = ROUTE_GRAPH_RADIUS_M,
) -> Tuple[BranchRoutesV2Result, List[BranchCandidate]]:
    result = build_branch_routes_v2(
        canonical,
        agent_id=str(agent_id),
        current_time_idx=int(current_time_idx),
        decision_time_idx=int(decision_time_idx),
        stop_point_xy=stop_point_xy,
        approach_heading=float(approach_heading),
        radius_m=float(radius_m),
    )
    return result, result.to_branch_candidates()


def _score_polyline_to_trajectory(polyline_xy: np.ndarray, trajectory_xy: np.ndarray, final_heading: float) -> Tuple[float, float, float]:
    polyline = np.asarray(polyline_xy, dtype=np.float32)
    trajectory = np.asarray(trajectory_xy, dtype=np.float32)
    if polyline.shape[0] < 2 or trajectory.shape[0] == 0:
        return float("inf"), float("inf"), float("inf")
    lateral_errors: List[float] = []
    projected_s: List[float] = []
    for point in trajectory:
        s_val, dist, _, _ = _project_point_to_polyline(polyline, point)
        lateral_errors.append(float(dist))
        projected_s.append(float(s_val))
    route_heading, _ = _heading_from_route(polyline, decision_point_xy=trajectory[0], approach_heading=final_heading)
    heading_error = abs(angle_delta(final_heading, route_heading))
    terminal_dist = float(np.linalg.norm(np.asarray(trajectory[-1], dtype=np.float32) - np.asarray(polyline[-1], dtype=np.float32)))
    score = float(np.mean(lateral_errors) + 0.75 * heading_error + 0.08 * terminal_dist)
    return score, float(np.mean(lateral_errors)), float(heading_error)


def audit_gt_future_against_branch_routes_v2(
    canonical: CanonicalScenario,
    *,
    agent_id: str,
    decision_time_idx: int,
    route_result: BranchRoutesV2Result,
) -> Dict[str, Any]:
    if str(agent_id) not in canonical.tracks:
        return {"gt_branch_label": None, "ambiguous": True, "drop_for_training": True, "scores": []}
    track = canonical.tracks[str(agent_id)]
    valid_idx = [
        idx
        for idx in range(int(decision_time_idx), track.valid.shape[0])
        if bool(track.valid[idx]) and np.isfinite(track.position_xy[idx]).all()
    ]
    if not valid_idx or not route_result.route_families:
        return {"gt_branch_label": None, "ambiguous": True, "drop_for_training": True, "scores": []}
    trajectory_xy = np.asarray(track.position_xy[valid_idx], dtype=np.float32)
    final_heading = _finite_heading(track.heading[valid_idx[-1]], default=route_result.approach_heading)
    scores: List[Dict[str, Any]] = []
    for family in route_result.route_families:
        score, mean_lat, heading_err = _score_polyline_to_trajectory(family.polyline_xy, trajectory_xy, final_heading)
        scores.append(
            {
                "branch_id": family.branch_id,
                "branch_label": family.branch_label,
                "score": float(score),
                "mean_lateral_error_m": float(mean_lat),
                "final_heading_error_rad": float(heading_err),
            }
        )
    scores.sort(key=lambda item: (item["score"], item["branch_label"]))
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    margin = None if second is None else float(second["score"] - best["score"])
    ambiguous = bool(
        (margin is not None and margin < GT_AMBIGUITY_MARGIN)
        or float(best["score"]) > GT_MAX_BEST_SCORE
    )
    return {
        "gt_branch_label": best["branch_label"],
        "gt_branch_id": best["branch_id"],
        "best_score": float(best["score"]),
        "score_margin": margin,
        "ambiguous": bool(ambiguous),
        "drop_for_training": bool(ambiguous),
        "scores": scores,
    }


def render_branch_routes_overlay_v2(
    *,
    output_path: str | Path,
    canonical: CanonicalScenario,
    route_result: BranchRoutesV2Result,
    current_pose_xy: Sequence[float],
    decision_pose_xy: Sequence[float],
    gt_future_xy: np.ndarray,
    factual_rollout_xy: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 8.8))
    ax.set_facecolor("#f8fafc")

    center = np.asarray(decision_pose_xy, dtype=np.float32)
    radius_m = 60.0

    for feature_id, feature in sorted(canonical.map_features.items(), key=lambda item: stable_string_sort_key(item[0])):
        feature_type = str(feature.feature_type)
        polyline = np.asarray(feature.polyline_xy, dtype=np.float32)
        if polyline.shape[0] < 2:
            continue
        if not any_point_within_radius(polyline, center, radius_m):
            continue
        if feature_type.startswith("LANE_"):
            ax.plot(polyline[:, 0], polyline[:, 1], color="#94a3b8", linewidth=1.0, alpha=0.45, zorder=1)
        elif feature_type.startswith("ROAD_EDGE") or feature_type.startswith("ROAD_LINE"):
            ax.plot(polyline[:, 0], polyline[:, 1], color="#cbd5e1", linewidth=0.9, alpha=0.35, zorder=0)

    if route_result.shared_stem_xy.shape[0] > 0:
        stem = np.asarray(route_result.shared_stem_xy, dtype=np.float32)
        ax.plot(stem[:, 0], stem[:, 1], color="#0f172a", linewidth=3.0, alpha=0.95, zorder=10, label="shared_stem")

    palette = {"left": "#0ea5e9", "straight": "#22c55e", "right": "#f97316", "u_turn": "#7c3aed"}
    for family in route_result.route_families:
        suffix = np.asarray(family.branch_suffix_xy, dtype=np.float32)
        polyline = np.asarray(family.polyline_xy, dtype=np.float32)
        color = palette.get(str(family.branch_label), "#334155")
        ax.plot(polyline[:, 0], polyline[:, 1], color=color, linewidth=2.2, alpha=0.35, zorder=9)
        if suffix.shape[0] > 0:
            ax.plot(suffix[:, 0], suffix[:, 1], color=color, linewidth=3.0, alpha=0.95, zorder=12, label=str(family.branch_label))
        gate = np.asarray(family.exit_gate.centerline_xy, dtype=np.float32)
        if gate.shape[0] >= 2:
            ax.plot(gate[:, 0], gate[:, 1], color=color, linewidth=4.0, alpha=0.95, zorder=13)

    if route_result.split_point_xy is not None:
        ax.scatter([route_result.split_point_xy[0]], [route_result.split_point_xy[1]], s=80, color="#111827", marker="P", zorder=15)

    ax.scatter([current_pose_xy[0]], [current_pose_xy[1]], s=70, color="#a855f7", marker="s", zorder=16, label="current")
    ax.scatter([decision_pose_xy[0]], [decision_pose_xy[1]], s=70, color="#6d28d9", marker="o", zorder=16, label="decision")

    gt_xy = np.asarray(gt_future_xy, dtype=np.float32)
    if gt_xy.shape[0] > 0:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#dc2626", linewidth=3.0, alpha=0.95, zorder=17, label="gt_future")
    if factual_rollout_xy is not None:
        factual_xy = np.asarray(factual_rollout_xy, dtype=np.float32)
        if factual_xy.shape[0] > 0:
            ax.plot(factual_xy[:, 0], factual_xy[:, 1], color="#111827", linewidth=2.6, linestyle="--", alpha=0.95, zorder=17, label="factual_rollout")

    ax.set_xlim(float(center[0] - radius_m), float(center[0] + radius_m))
    ax.set_ylim(float(center[1] - radius_m), float(center[1] + radius_m))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    filtered_h = []
    filtered_l = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        filtered_h.append(handle)
        filtered_l.append(label)
    if filtered_h:
        ax.legend(filtered_h, filtered_l, loc="lower right", fontsize=8, framealpha=0.9)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.965)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
