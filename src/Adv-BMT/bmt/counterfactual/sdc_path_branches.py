from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .branch_enumeration import BranchCandidate, BranchTerminalPose
from .geometry import angle_delta, classify_heading_delta, cumulative_polyline_lengths, heading_from_points, nearest_point_index, polyline_length
from .types import CanonicalScenario, stable_string_sort_key

MIN_SDC_PATH_BRANCH_LENGTH_M = 8.0


def _valid_sdc_path_polyline(path: Any) -> np.ndarray:
    polyline = np.asarray(getattr(path, "polyline_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    valid = np.asarray(getattr(path, "valid", np.ones((polyline.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    if polyline.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if valid.shape[0] != polyline.shape[0]:
        valid = np.ones((polyline.shape[0],), dtype=bool)
    clipped = polyline[valid]
    return np.asarray(clipped, dtype=np.float32)


def _semantic_label_from_metadata(metadata: Dict[str, Any]) -> str | None:
    for key in ("semantic_label", "branch_label", "label"):
        value = metadata.get(key)
        text = "" if value is None else str(value).strip().lower()
        if text in {"left", "straight", "right", "u_turn"}:
            return text
    return None


def _candidate_from_sdc_path(
    *,
    path_id: str,
    path: Any,
    decision_xy: Sequence[float],
    approach_heading: float,
) -> tuple[BranchCandidate, bool, float, float] | None:
    polyline = _valid_sdc_path_polyline(path)
    if polyline.shape[0] < 2:
        return None
    start_idx = nearest_point_index(polyline, decision_xy)
    candidate_polyline = np.asarray(polyline[start_idx:], dtype=np.float32)
    if candidate_polyline.shape[0] < 2:
        return None
    route_length = polyline_length(candidate_polyline)
    if route_length < float(MIN_SDC_PATH_BRANCH_LENGTH_M):
        return None

    cum = cumulative_polyline_lengths(candidate_polyline)
    first_far_idx = int(np.searchsorted(cum, 1.5, side="left"))
    first_far_idx = min(max(first_far_idx, 0), candidate_polyline.shape[0] - 2)
    start_point = candidate_polyline[first_far_idx]

    metadata = dict(getattr(path, "metadata", {}) or {})
    terminal_heading = heading_from_points(candidate_polyline[-2], candidate_polyline[-1])
    heading_delta = angle_delta(terminal_heading, float(approach_heading))
    branch_label = _semantic_label_from_metadata(metadata) or classify_heading_delta(heading_delta)
    decision_offset = float(np.linalg.norm(start_point - np.asarray(decision_xy, dtype=np.float32)[:2]))
    on_route = bool(metadata.get("on_route", False))

    candidate = BranchCandidate(
        branch_id=str(path_id),
        branch_label=str(branch_label),
        source_feature_id=str(path_id),
        heading=float(terminal_heading),
        heading_delta=float(heading_delta),
        start_point_xy=(float(start_point[0]), float(start_point[1])),
        terminal_pose=BranchTerminalPose(
            x=float(candidate_polyline[-1, 0]),
            y=float(candidate_polyline[-1, 1]),
            heading=float(terminal_heading),
        ),
        rank_score=float(decision_offset + 0.05 * abs(float(heading_delta))),
        source_kind="sdc_path",
        polyline_xy=np.asarray(candidate_polyline, dtype=np.float32),
    )
    return candidate, on_route, route_length, decision_offset


def enumerate_branch_candidates_from_sdc_paths(
    canonical: CanonicalScenario,
    *,
    agent_id: str,
    decision_time_idx: int,
    approach_heading: float,
) -> List[BranchCandidate]:
    if str(agent_id) != str(canonical.sdc_id):
        return []
    if not canonical.sdc_paths:
        return []
    track = canonical.tracks.get(str(agent_id))
    if track is None:
        return []
    idx = int(np.clip(int(decision_time_idx), 0, max(0, track.position_xy.shape[0] - 1)))
    if not bool(track.valid[idx]) or not np.isfinite(track.position_xy[idx]).all():
        return []
    decision_xy = np.asarray(track.position_xy[idx], dtype=np.float32)

    grouped: Dict[str, tuple[BranchCandidate, bool, float, float]] = {}
    for path_id, path in sorted(canonical.sdc_paths.items(), key=lambda item: stable_string_sort_key(item[0])):
        built = _candidate_from_sdc_path(
            path_id=str(path_id),
            path=path,
            decision_xy=decision_xy,
            approach_heading=float(approach_heading),
        )
        if built is None:
            continue
        candidate, on_route, route_length, decision_offset = built
        current = grouped.get(str(candidate.branch_label))
        if current is None:
            grouped[str(candidate.branch_label)] = built
            continue
        _, current_on_route, current_route_length, current_offset = current
        if (on_route, route_length, -decision_offset, str(candidate.branch_id)) > (
            current_on_route,
            current_route_length,
            -current_offset,
            str(current[0].branch_id),
        ):
            grouped[str(candidate.branch_label)] = built

    ordered_labels = {"left": 0, "straight": 1, "right": 2, "u_turn": 3}
    selected = [item[0] for item in grouped.values()]
    return sorted(
        selected,
        key=lambda candidate: (
            ordered_labels.get(str(candidate.branch_label), 99),
            float(candidate.rank_score),
            str(candidate.branch_id),
        ),
    )
