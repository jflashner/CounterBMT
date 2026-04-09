from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .geometry import heading_from_points
from .normalize import load_raw_scenario

SDC_PATH_CONTROL_SCHEMA_VERSION = "sdc_path_control_v1"
SDC_PATH_SEMANTIC_LABEL_ORDER = (
    "left",
    "right",
    "left_lane_change",
    "right_lane_change",
    "straight",
    "stop",
)
SDC_PATH_SEMANTIC_LABEL_TO_ID = {
    label: idx for idx, label in enumerate(SDC_PATH_SEMANTIC_LABEL_ORDER)
}

DEFAULT_RESAMPLE_SPACING_M = 2.0
DEFAULT_SEPARABILITY_SCALE_M = 6.0
DEFAULT_SEPARABILITY_HEADING_WEIGHT_M = 2.0
DEFAULT_PATH_DEADBAND_M = 1.0
DEFAULT_DISCONTINUITY_STITCH_RADIUS_M = 2.0
DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M = 6.0


@dataclass
class ResampledLocalPath:
    waypoints_xy: np.ndarray
    headings: np.ndarray
    arc_lengths_m: np.ndarray


def sanitize_resampled_local_path(path: ResampledLocalPath) -> ResampledLocalPath:
    xy = _sanitize_polyline(path.waypoints_xy)
    headings = np.asarray(path.headings, dtype=np.float32).reshape(-1)
    arc = np.asarray(path.arc_lengths_m, dtype=np.float32).reshape(-1)
    n = int(min(xy.shape[0], headings.shape[0], arc.shape[0]))
    if n <= 0:
        return ResampledLocalPath(
            waypoints_xy=np.zeros((0, 2), dtype=np.float32),
            headings=np.zeros((0,), dtype=np.float32),
            arc_lengths_m=np.zeros((0,), dtype=np.float32),
        )
    return ResampledLocalPath(
        waypoints_xy=np.asarray(xy[:n], dtype=np.float32),
        headings=np.asarray(headings[:n], dtype=np.float32),
        arc_lengths_m=np.asarray(arc[:n], dtype=np.float32),
    )


def normalize_semantic_label(value: Any, *, default: str = "straight") -> str:
    text = str(value or "").strip().lower()
    if text in SDC_PATH_SEMANTIC_LABEL_TO_ID:
        return text
    return str(default)


def semantic_label_to_id(value: Any, *, default: str = "straight") -> int:
    label = normalize_semantic_label(value, default=default)
    return int(SDC_PATH_SEMANTIC_LABEL_TO_ID[label])


def _finite_xy_rows(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    array = array[:, :2]
    mask = np.isfinite(array).all(axis=-1)
    return np.asarray(array[mask], dtype=np.float32)


def _sanitize_polyline(points_xy: Any, *, dedup_eps: float = 1e-3) -> np.ndarray:
    xy = _finite_xy_rows(points_xy)
    if xy.shape[0] == 0:
        return xy
    kept = [xy[0]]
    for idx in range(1, int(xy.shape[0])):
        if float(np.linalg.norm(xy[idx] - kept[-1])) > float(dedup_eps):
            kept.append(xy[idx])
    if len(kept) == 1:
        return np.asarray(kept, dtype=np.float32)
    return np.asarray(kept, dtype=np.float32)


def split_polyline_on_discontinuities(
    points_xy: Any,
    *,
    jump_threshold_m: Optional[float] = None,
    jump_scale: float = 4.0,
    min_points: int = 2,
) -> List[np.ndarray]:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] == 0:
        return []
    if xy.shape[0] == 1:
        return [xy] if int(min_points) <= 1 else []

    steps = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    positive_steps = steps[steps > 1e-3]
    if jump_threshold_m is None:
        typical_step = float(np.median(positive_steps)) if positive_steps.size > 0 else DEFAULT_RESAMPLE_SPACING_M
        jump_threshold = max(4.0, typical_step * float(jump_scale))
    else:
        jump_threshold = float(jump_threshold_m)

    split_after = np.flatnonzero(steps > jump_threshold).tolist()
    start_idx = 0
    segments: List[np.ndarray] = []
    for break_idx in split_after:
        end_idx = int(break_idx + 1)
        segment = np.asarray(xy[start_idx:end_idx], dtype=np.float32)
        if segment.shape[0] >= int(min_points):
            segments.append(segment)
        start_idx = end_idx
    final_segment = np.asarray(xy[start_idx:], dtype=np.float32)
    if final_segment.shape[0] >= int(min_points):
        segments.append(final_segment)
    return segments


def stitch_polyline_discontinuities(
    points_xy: Any,
    *,
    handoff_radius_m: float = DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    jump_threshold_m: float = DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
    min_points: int = 2,
) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] <= 1:
        return xy
    segments = split_polyline_on_discontinuities(
        xy,
        jump_threshold_m=float(jump_threshold_m),
        min_points=min_points,
    )
    if len(segments) <= 1:
        return xy

    radius = max(float(handoff_radius_m), 1e-3)
    stitched = np.asarray(segments[0], dtype=np.float32)
    for next_segment in segments[1:]:
        current = _sanitize_polyline(stitched)
        upcoming = _sanitize_polyline(next_segment)
        if current.shape[0] == 0:
            stitched = upcoming
            continue
        if upcoming.shape[0] == 0:
            stitched = current
            continue

        next_start = np.asarray(upcoming[0], dtype=np.float32)
        distances = np.linalg.norm(current - next_start.reshape(1, 2), axis=-1)
        within = np.flatnonzero(distances <= radius)
        if within.size > 0:
            # Use the latest plausible handoff so we remove the dead-end tail
            # without jumping prematurely if the segments run near each other.
            cut_idx = int(within[-1])
            stitched_prefix = np.asarray(current[: cut_idx + 1], dtype=np.float32)
            stitched = np.concatenate([stitched_prefix, upcoming], axis=0).astype(np.float32)
        else:
            stitched = np.concatenate([current, upcoming], axis=0).astype(np.float32)
        stitched = _sanitize_polyline(stitched)
    return stitched


def polyline_arc_lengths(points_xy: Any) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(step, dtype=np.float32)], axis=0)


def polyline_length_m(points_xy: Any) -> float:
    arc = polyline_arc_lengths(points_xy)
    if arc.size == 0:
        return 0.0
    return float(arc[-1])


def nearest_point_index(points_xy: Any, point_xy: Sequence[float]) -> int:
    xy = _sanitize_polyline(points_xy)
    point = _finite_xy_rows(np.asarray(point_xy, dtype=np.float32))
    if xy.shape[0] == 0 or point.shape[0] == 0:
        return 0
    d = np.linalg.norm(xy - point[0][None, :], axis=-1)
    return int(np.argmin(d))


def trim_polyline_from_point(points_xy: Any, point_xy: Sequence[float], *, prepend_point: bool = True) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    point = _finite_xy_rows(np.asarray(point_xy, dtype=np.float32))
    if xy.shape[0] == 0:
        return xy
    if point.shape[0] == 0:
        return xy
    start_idx = nearest_point_index(xy, point[0])
    trimmed = np.asarray(xy[start_idx:], dtype=np.float32)
    if prepend_point and trimmed.shape[0] > 0 and float(np.linalg.norm(trimmed[0] - point[0])) > 1e-3:
        trimmed = np.vstack([point[0], trimmed]).astype(np.float32)
    return _sanitize_polyline(trimmed)


def resample_polyline_xy(points_xy: Any, *, spacing_m: float = DEFAULT_RESAMPLE_SPACING_M) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] < 2:
        return xy
    spacing = max(float(spacing_m), 0.25)
    arc = polyline_arc_lengths(xy)
    total = float(arc[-1])
    if total <= spacing:
        return xy
    sample_arc = np.arange(0.0, total + 0.5 * spacing, spacing, dtype=np.float32)
    if sample_arc[-1] < total:
        sample_arc = np.concatenate([sample_arc, np.asarray([total], dtype=np.float32)], axis=0)
    x = np.interp(sample_arc, arc, xy[:, 0]).astype(np.float32)
    y = np.interp(sample_arc, arc, xy[:, 1]).astype(np.float32)
    return _sanitize_polyline(np.stack([x, y], axis=-1).astype(np.float32))


def polyline_headings(points_xy: Any) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if xy.shape[0] == 1:
        return np.zeros((1,), dtype=np.float32)
    headings = np.zeros((xy.shape[0],), dtype=np.float32)
    for idx in range(int(xy.shape[0])):
        if idx == 0:
            headings[idx] = float(heading_from_points(xy[0], xy[1]))
        elif idx == int(xy.shape[0] - 1):
            headings[idx] = float(heading_from_points(xy[-2], xy[-1]))
        else:
            headings[idx] = float(heading_from_points(xy[idx - 1], xy[idx + 1]))
    return headings


def wrap_angle(angle: Any) -> Any:
    if torch.is_tensor(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def world_to_sdc_up_frame(points_xy_world: Any, *, origin_xy_world: Sequence[float], origin_heading_world: float) -> np.ndarray:
    xy = _finite_xy_rows(points_xy_world)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    origin = np.asarray(origin_xy_world, dtype=np.float32).reshape(1, 2)
    centered = xy - origin
    rot = (math.pi / 2.0) - float(origin_heading_world)
    c = math.cos(rot)
    s = math.sin(rot)
    x_new = c * centered[:, 0] - s * centered[:, 1]
    y_new = s * centered[:, 0] + c * centered[:, 1]
    return np.stack([x_new, y_new], axis=-1).astype(np.float32)


def headings_world_to_sdc_up(headings_world: Any, *, origin_heading_world: float) -> np.ndarray:
    headings = np.asarray(headings_world, dtype=np.float32).reshape(-1)
    if headings.size == 0:
        return np.zeros((0,), dtype=np.float32)
    local = wrap_angle(headings - float(origin_heading_world) + np.float32(math.pi / 2.0))
    return np.asarray(local, dtype=np.float32)


def extract_sdc_current_pose(raw_scenario: Mapping[str, Any], *, sdc_id: str, current_time_index: int) -> Tuple[np.ndarray, float]:
    track = dict(dict(raw_scenario.get("tracks", {})).get(str(sdc_id), {}))
    state = dict(track.get("state", {}))
    position = np.asarray(state.get("position", []), dtype=np.float32)
    heading = np.asarray(state.get("heading", []), dtype=np.float32).reshape(-1)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[1] < 2 or heading.ndim != 1 or heading.shape[0] == 0:
        raise ValueError(f"Missing SDC pose for track {sdc_id}")
    idx = int(np.clip(int(current_time_index), 0, min(position.shape[0], heading.shape[0]) - 1))
    if valid.shape[0] == position.shape[0] and not bool(valid[idx]):
        valid_idx = np.flatnonzero(valid[idx:])
        if valid_idx.size > 0:
            idx = idx + int(valid_idx[0])
    return np.asarray(position[idx, :2], dtype=np.float32), float(heading[idx])


def extract_ground_truth_sdc_route_xy(raw_scenario: Mapping[str, Any], *, sdc_id: str, current_time_index: int) -> np.ndarray:
    track = dict(dict(raw_scenario.get("tracks", {})).get(str(sdc_id), {}))
    state = dict(track.get("state", {}))
    position = np.asarray(state.get("position", []), dtype=np.float32)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    idx = int(np.clip(int(current_time_index), 0, position.shape[0] - 1))
    if valid.shape[0] == position.shape[0]:
        future = position[idx:, :2][valid[idx:]]
    else:
        future = position[idx:, :2]
    return _sanitize_polyline(future)


def _extract_valid_sdc_path_xy(raw_scenario: Mapping[str, Any], path_id: str) -> np.ndarray:
    payload = dict(dict(raw_scenario.get("sdc_paths", {})).get(str(path_id), {}))
    polyline = np.asarray(payload.get("polyline_xyz", []), dtype=np.float32)
    valid = np.asarray(payload.get("valid", []), dtype=bool).reshape(-1)
    if polyline.ndim != 2 or polyline.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    if valid.shape[0] == polyline.shape[0]:
        xy = polyline[valid][:, :2]
    else:
        xy = polyline[:, :2]
    return _sanitize_polyline(xy)


def list_on_route_candidate_path_ids(raw_scenario: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for path_id, payload in sorted(dict(raw_scenario.get("sdc_paths", {})).items(), key=lambda item: str(item[0])):
        metadata = dict(dict(payload).get("metadata", {}) or {})
        if bool(metadata.get("on_route", False)):
            out.append(str(path_id))
    return out


def build_local_selected_path(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    source_kind: str,
    selected_path_id: Optional[str],
    spacing_m: float = DEFAULT_RESAMPLE_SPACING_M,
) -> ResampledLocalPath:
    current_xy_world, current_heading_world = extract_sdc_current_pose(
        raw_scenario,
        sdc_id=str(sdc_id),
        current_time_index=int(current_time_index),
    )
    if str(source_kind) == "factual_gt":
        path_world = extract_ground_truth_sdc_route_xy(
            raw_scenario,
            sdc_id=str(sdc_id),
            current_time_index=int(current_time_index),
        )
    else:
        if not selected_path_id:
            raise ValueError("selected_path_id is required for alternative_sdc_path rows")
        candidate_xy = _extract_valid_sdc_path_xy(raw_scenario, str(selected_path_id))
        path_world = trim_polyline_from_point(candidate_xy, current_xy_world, prepend_point=True)
    resampled_world = resample_polyline_xy(path_world, spacing_m=float(spacing_m))
    local_xy = world_to_sdc_up_frame(
        resampled_world,
        origin_xy_world=current_xy_world,
        origin_heading_world=current_heading_world,
    )
    local_headings = polyline_headings(local_xy)
    local_arc = polyline_arc_lengths(local_xy)
    return ResampledLocalPath(
        waypoints_xy=np.asarray(local_xy, dtype=np.float32),
        headings=np.asarray(local_headings, dtype=np.float32),
        arc_lengths_m=np.asarray(local_arc, dtype=np.float32),
    )


def build_selected_path_world(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    source_kind: str,
    selected_path_id: Optional[str],
) -> np.ndarray:
    current_xy_world, _ = extract_sdc_current_pose(
        raw_scenario,
        sdc_id=str(sdc_id),
        current_time_index=int(current_time_index),
    )
    if str(source_kind) == "factual_gt":
        return extract_ground_truth_sdc_route_xy(
            raw_scenario,
            sdc_id=str(sdc_id),
            current_time_index=int(current_time_index),
        )
    if not selected_path_id:
        raise ValueError("selected_path_id is required for alternative_sdc_path rows")
    candidate_xy = _extract_valid_sdc_path_xy(raw_scenario, str(selected_path_id))
    return trim_polyline_from_point(candidate_xy, current_xy_world, prepend_point=True)


def polyline_segment_valid_mask(
    points_xy: Any,
    *,
    jump_threshold_m: float = DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
) -> np.ndarray:
    xy = _sanitize_polyline(points_xy)
    if xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    mask = np.zeros((xy.shape[0],), dtype=np.float32)
    if xy.shape[0] < 2:
        return mask
    step_m = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    contiguous = step_m <= float(jump_threshold_m)
    mask[: contiguous.shape[0]] = contiguous.astype(np.float32)
    return mask


def build_local_competing_paths(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    selected_path_id: Optional[str],
    spacing_m: float = DEFAULT_RESAMPLE_SPACING_M,
) -> Dict[str, ResampledLocalPath]:
    current_xy_world, current_heading_world = extract_sdc_current_pose(
        raw_scenario,
        sdc_id=str(sdc_id),
        current_time_index=int(current_time_index),
    )
    out: Dict[str, ResampledLocalPath] = {}
    for path_id in list_on_route_candidate_path_ids(raw_scenario):
        if selected_path_id is not None and str(path_id) == str(selected_path_id):
            continue
        candidate_xy = _extract_valid_sdc_path_xy(raw_scenario, str(path_id))
        trimmed = trim_polyline_from_point(candidate_xy, current_xy_world, prepend_point=True)
        resampled_world = resample_polyline_xy(trimmed, spacing_m=float(spacing_m))
        local_xy = world_to_sdc_up_frame(
            resampled_world,
            origin_xy_world=current_xy_world,
            origin_heading_world=current_heading_world,
        )
        if local_xy.shape[0] < 2:
            continue
        out[str(path_id)] = ResampledLocalPath(
            waypoints_xy=np.asarray(local_xy, dtype=np.float32),
            headings=np.asarray(polyline_headings(local_xy), dtype=np.float32),
            arc_lengths_m=np.asarray(polyline_arc_lengths(local_xy), dtype=np.float32),
        )
    return out


def compute_path_separability_profile(
    selected_path: ResampledLocalPath,
    competing_paths: Mapping[str, ResampledLocalPath],
    *,
    scale_m: float = DEFAULT_SEPARABILITY_SCALE_M,
    heading_weight_m: float = DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
) -> Dict[str, Any]:
    sanitized_selected = sanitize_resampled_local_path(selected_path)
    selected_xy = np.asarray(sanitized_selected.waypoints_xy, dtype=np.float32)
    selected_heading = np.asarray(sanitized_selected.headings, dtype=np.float32).reshape(-1)
    if selected_xy.shape[0] == 0:
        return {
            "separability": np.zeros((0,), dtype=np.float32),
            "min_distance_m": np.zeros((0,), dtype=np.float32),
            "heading_delta_rad": np.zeros((0,), dtype=np.float32),
            "nearest_competing_path_id": [],
        }
    if not competing_paths:
        return {
            "separability": np.ones((selected_xy.shape[0],), dtype=np.float32),
            "min_distance_m": np.full((selected_xy.shape[0],), np.inf, dtype=np.float32),
            "heading_delta_rad": np.zeros((selected_xy.shape[0],), dtype=np.float32),
            "nearest_competing_path_id": [None] * int(selected_xy.shape[0]),
        }

    min_distance = np.full((selected_xy.shape[0],), np.inf, dtype=np.float32)
    min_heading_delta = np.zeros((selected_xy.shape[0],), dtype=np.float32)
    nearest_path_id: List[Optional[str]] = [None] * int(selected_xy.shape[0])
    for path_id, path in competing_paths.items():
        sanitized_competitor = sanitize_resampled_local_path(path)
        competitor_xy = np.asarray(sanitized_competitor.waypoints_xy, dtype=np.float32)
        competitor_heading = np.asarray(sanitized_competitor.headings, dtype=np.float32).reshape(-1)
        if competitor_xy.shape[0] == 0:
            continue
        diff = selected_xy[:, None, :] - competitor_xy[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        nearest_idx = np.argmin(d, axis=-1)
        nearest_d = d[np.arange(selected_xy.shape[0]), nearest_idx]
        nearest_heading = competitor_heading[np.asarray(nearest_idx, dtype=np.int64)]
        heading_delta = np.abs(np.arctan2(np.sin(selected_heading - nearest_heading), np.cos(selected_heading - nearest_heading))).astype(np.float32)
        improved = nearest_d < min_distance
        if bool(np.any(improved)):
            min_distance[improved] = nearest_d[improved].astype(np.float32)
            min_heading_delta[improved] = heading_delta[improved]
            for idx in np.flatnonzero(improved):
                nearest_path_id[int(idx)] = str(path_id)
    combined = min_distance + float(heading_weight_m) * min_heading_delta
    normalized = np.clip(combined / max(float(scale_m), 1e-3), 0.0, 1.0).astype(np.float32)
    return {
        "separability": normalized,
        "min_distance_m": min_distance.astype(np.float32),
        "heading_delta_rad": min_heading_delta.astype(np.float32),
        "nearest_competing_path_id": nearest_path_id,
    }


def load_sdc_path_control_row(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("rt", encoding="utf-8") as f:
        return dict(__import__("json").load(f))


def is_sdc_path_control_row(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    schema_version = str(row.get("schema_version") or "").strip()
    if schema_version == SDC_PATH_CONTROL_SCHEMA_VERSION:
        return True
    required = (
        "selected_path_waypoints_local_xy",
        "selected_path_waypoints_local_heading",
        "selected_path_arc_lengths_m",
        "selected_path_separability",
        "semantic_label",
    )
    return all(key in row for key in required)


def build_sdc_path_dataset_fields(
    *,
    scenario_id: str,
    decoder_track_names: Sequence[Any],
    horizon: int,
    row: Mapping[str, Any],
    require_trainable: bool,
    include_stop: bool = True,
) -> Dict[str, Any]:
    debug_meta = {
        "schema_version": str(row.get("schema_version") or SDC_PATH_CONTROL_SCHEMA_VERSION),
        "scenario_id": str(scenario_id),
        "sdc_id": str(row.get("sdc_id") or ""),
        "selected_path_id": None if row.get("selected_path_id") is None else str(row.get("selected_path_id")),
        "source_kind": str(row.get("source_kind") or ""),
        "semantic_label": normalize_semantic_label(row.get("semantic_label")),
        "candidate_count": int(row.get("candidate_count") or 0),
    }
    sdc_id = str(row.get("sdc_id") or "")
    decoder_track_names = [str(value) for value in np.asarray(decoder_track_names, dtype=object).reshape(-1).tolist()]
    decision_agent_mask = np.zeros((len(decoder_track_names),), dtype=np.float32)
    if sdc_id and sdc_id in decoder_track_names:
        decision_agent_mask[decoder_track_names.index(sdc_id)] = 1.0
    time_window_mask = np.ones((int(horizon),), dtype=np.float32)

    semantic_label = normalize_semantic_label(row.get("semantic_label"))
    semantic_confidence = float(row.get("semantic_confidence") or 0.0)
    use_for_training = bool(row.get("use_for_training", True))
    if semantic_label == "stop" and not include_stop:
        use_for_training = False
        debug_meta["drop_reason"] = "stop_row_disabled"

    control_available = bool(decision_agent_mask.sum() > 0) and use_for_training
    if require_trainable and not control_available:
        debug_meta.setdefault("drop_reason", "sdc_not_modeled_or_row_disabled")

    waypoints_xy = np.asarray(row.get("selected_path_waypoints_local_xy", []), dtype=np.float32)
    waypoints_heading = np.asarray(row.get("selected_path_waypoints_local_heading", []), dtype=np.float32).reshape(-1)
    path_arc = np.asarray(row.get("selected_path_arc_lengths_m", []), dtype=np.float32).reshape(-1)
    path_sep = np.asarray(row.get("selected_path_separability", []), dtype=np.float32).reshape(-1)
    max_len = min(int(waypoints_xy.shape[0]), int(waypoints_heading.shape[0]), int(path_arc.shape[0]), int(path_sep.shape[0]))
    if max_len <= 0:
        waypoints_xy = np.zeros((0, 2), dtype=np.float32)
        waypoints_heading = np.zeros((0,), dtype=np.float32)
        path_arc = np.zeros((0,), dtype=np.float32)
        path_sep = np.zeros((0,), dtype=np.float32)
    else:
        waypoints_xy = np.asarray(waypoints_xy[:max_len], dtype=np.float32)
        waypoints_heading = np.asarray(waypoints_heading[:max_len], dtype=np.float32)
        path_arc = np.asarray(path_arc[:max_len], dtype=np.float32)
        path_sep = np.asarray(path_sep[:max_len], dtype=np.float32)
    path_waypoints = np.concatenate(
        [
            waypoints_xy,
            np.sin(waypoints_heading).reshape(-1, 1).astype(np.float32),
            np.cos(waypoints_heading).reshape(-1, 1).astype(np.float32),
            path_arc.reshape(-1, 1).astype(np.float32),
        ],
        axis=-1,
    ) if waypoints_xy.shape[0] > 0 else np.zeros((0, 5), dtype=np.float32)
    path_mask = np.ones((path_waypoints.shape[0],), dtype=np.float32)

    is_factual = str(row.get("source_kind") or "") == "factual_gt"
    fields = {
        "cf/sdc_semantic_label_id": int(semantic_label_to_id(semantic_label)),
        "cf/sdc_semantic_confidence": np.float32(semantic_confidence),
        "cf/sdc_path_waypoints": path_waypoints.astype(np.float32),
        "cf/sdc_path_waypoint_mask": path_mask.astype(np.float32),
        "cf/sdc_path_separability": path_sep.astype(np.float32),
        "cf/sdc_path_arc_lengths": path_arc.astype(np.float32),
        "cf/sdc_is_factual": int(is_factual),
        "cf/sdc_control_available": int(control_available),
        "cf/sdc_debug_meta": dict(debug_meta),
        "cf/time_window_mask": time_window_mask.astype(np.float32),
        "cf/decision_agent_mask": decision_agent_mask.astype(np.float32),
        "cf/conditioning_eligible": int(control_available),
        "cf/control_available": int(control_available),
        "cf/path_supervision_mask": int(control_available),
        "cf/compliance_supervision_mask": 0,
        "cf/timing_supervision_mask": 0,
        "cf/debug_meta": dict(debug_meta),
    }
    return fields


def torch_world_to_sdc_up(points_xy_world: torch.Tensor, *, origin_xy_world: torch.Tensor, origin_heading_world: torch.Tensor) -> torch.Tensor:
    origin = origin_xy_world.to(dtype=points_xy_world.dtype)
    heading = origin_heading_world.to(dtype=points_xy_world.dtype)
    rot = (math.pi / 2.0) - heading
    c = torch.cos(rot)
    s = torch.sin(rot)
    centered = points_xy_world - origin.unsqueeze(-2)
    x_new = c.unsqueeze(-1) * centered[..., 0] - s.unsqueeze(-1) * centered[..., 1]
    y_new = s.unsqueeze(-1) * centered[..., 0] + c.unsqueeze(-1) * centered[..., 1]
    return torch.stack([x_new, y_new], dim=-1)


def torch_heading_to_sdc_up(headings_world: torch.Tensor, *, origin_heading_world: torch.Tensor) -> torch.Tensor:
    return wrap_angle(headings_world - origin_heading_world.unsqueeze(-1) + (math.pi / 2.0))


def project_points_to_path_torch(
    points_local_xy: torch.Tensor,
    *,
    path_waypoints_local_xy: torch.Tensor,
    path_waypoint_mask: torch.Tensor,
    path_waypoint_heading: torch.Tensor,
    path_waypoint_arc: torch.Tensor,
    path_waypoint_separability: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if points_local_xy.ndim != 3:
        raise ValueError(f"points_local_xy must be [B,T,2], got {tuple(points_local_xy.shape)}")
    if path_waypoints_local_xy.ndim != 3:
        raise ValueError(f"path_waypoints_local_xy must be [B,M,2], got {tuple(path_waypoints_local_xy.shape)}")
    d = torch.linalg.norm(
        points_local_xy[:, :, None, :] - path_waypoints_local_xy[:, None, :, :],
        dim=-1,
    )
    large = torch.full_like(d, 1e6)
    mask = path_waypoint_mask[:, None, :] > 0
    d = torch.where(mask, d, large)
    nearest_idx = torch.argmin(d, dim=-1)
    nearest_distance = torch.gather(d, dim=-1, index=nearest_idx.unsqueeze(-1)).squeeze(-1)
    nearest_heading = torch.gather(path_waypoint_heading, dim=-1, index=nearest_idx)
    nearest_arc = torch.gather(path_waypoint_arc, dim=-1, index=nearest_idx)
    nearest_sep = torch.gather(path_waypoint_separability, dim=-1, index=nearest_idx)
    return {
        "nearest_idx": nearest_idx,
        "nearest_distance": nearest_distance,
        "nearest_heading": nearest_heading,
        "nearest_arc": nearest_arc,
        "nearest_separability": nearest_sep,
    }


def load_raw_scenario_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_pkl = row.get("scenario_pkl")
    if not scenario_pkl:
        raise ValueError("sdc_path_control row is missing scenario_pkl")
    return load_raw_scenario(Path(str(scenario_pkl)).expanduser())
