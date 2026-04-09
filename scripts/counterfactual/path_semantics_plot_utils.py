from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from bmt.counterfactual.geometry import heading_from_points, polyline_length

PLOT_RADIUS_M = 48.0
SDC_VERTICAL_FRACTION = 0.10
PAST_STEPS = 12
FIG_SIZE_INCH = 8.8
FIG_DPI = 180
CONTEXT_SELECTION_RADIUS_M = (2.0 * PLOT_RADIUS_M * (1.0 - SDC_VERTICAL_FRACTION)) + 12.0
STAY_LANE_GUIDE_LENGTH_M = 36.0
ROAD_COLOR = "#334155"
LANE_COLOR = "#cbd5e1"
CROSSWALK_FACE = "#e2e8f0"
AGENT_COLOR = "#cbd5e1"
SDC_ARROW_COLOR = "#111827"
SDC_DOT_COLOR = "#f43f5e"
START_LANE_SHADE = "#60a5fa"
FINAL_LANE_SHADE = "#f59e0b"
STAY_GUIDE_COLOR = "#0f172a"
ROUTE_DISCONTINUITY_JUMP_M = 6.0


def _finite_xy_rows(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    array = array[:, :2]
    mask = np.isfinite(array).all(axis=-1)
    return np.asarray(array[mask], dtype=np.float64)


def _feature_xy(feature: Mapping[str, Any]) -> np.ndarray:
    if "polyline" in feature:
        return _finite_xy_rows(np.asarray(feature["polyline"], dtype=np.float64))
    if "polygon" in feature:
        return _finite_xy_rows(np.asarray(feature["polygon"], dtype=np.float64))
    return np.zeros((0, 2), dtype=np.float64)


def _trim_to_radius(xy: np.ndarray, *, center_xy: np.ndarray, radius_m: float) -> np.ndarray:
    array = _finite_xy_rows(xy)
    if array.shape[0] == 0:
        return array
    d = np.linalg.norm(array - center_xy[None, :], axis=-1)
    keep = d <= float(radius_m)
    clipped = array[keep]
    return np.asarray(clipped, dtype=np.float64)


def _min_pointset_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    a = _finite_xy_rows(a_xy)
    b = _finite_xy_rows(b_xy)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("inf")
    diff = a[:, None, :] - b[None, :, :]
    return float(np.min(np.linalg.norm(diff, axis=-1)))


def _select_map_context(raw_scenario: Mapping[str, Any], *, center_xy: np.ndarray, radius_m: float) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"lane_centerlines": [], "road_boundaries": [], "crosswalks": []}
    for feature_id, feature in sorted(raw_scenario.get("map_features", {}).items(), key=lambda item: str(item[0])):
        feature_type = str(feature.get("type", ""))
        xy = _trim_to_radius(_feature_xy(feature), center_xy=center_xy, radius_m=radius_m)
        if xy.shape[0] < 2:
            continue
        payload = {"feature_id": str(feature_id), "feature_type": feature_type, "xy_world": xy}
        if feature_type.startswith("LANE_") or feature_type == "DRIVEWAY":
            grouped["lane_centerlines"].append(payload)
        elif feature_type.startswith("ROAD_EDGE") or feature_type.startswith("ROAD_LINE"):
            grouped["road_boundaries"].append(payload)
        elif feature_type == "CROSSWALK":
            grouped["crosswalks"].append(payload)
    return grouped


def _select_traffic_lights(raw_scenario: Mapping[str, Any], *, center_xy: np.ndarray, radius_m: float, time_index: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for light_id, light in sorted(raw_scenario.get("dynamic_map_states", {}).items(), key=lambda item: str(item[0])):
        if str(light.get("type", "")) != "TRAFFIC_LIGHT":
            continue
        stop_point = _finite_xy_rows(np.asarray([light.get("stop_point", [])], dtype=np.float64))
        if stop_point.shape[0] == 0:
            continue
        dist = float(np.linalg.norm(stop_point[0] - center_xy))
        if dist > float(radius_m):
            continue
        object_state = list(dict(light.get("state", {})).get("object_state", []))
        state = None
        if object_state:
            idx = int(np.clip(int(time_index), 0, len(object_state) - 1))
            state = object_state[idx]
        rows.append(
            {
                "light_id": str(light_id),
                "stop_point_xy_world": stop_point[0],
                "state": None if state is None else str(state),
            }
        )
    return rows


def _select_nearby_agents(raw_scenario: Mapping[str, Any], *, sdc_id: str, center_xy: np.ndarray, current_idx: int, radius_m: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for track_id, track in sorted(raw_scenario.get("tracks", {}).items(), key=lambda item: str(item[0])):
        if str(track_id) == str(sdc_id):
            continue
        state = dict(track.get("state", {}))
        valid = np.asarray(state.get("valid", []), dtype=bool)
        position = np.asarray(state.get("position", []), dtype=np.float64)
        if valid.ndim != 1 or position.ndim != 2 or position.shape[0] == 0:
            continue
        idx = int(np.clip(int(current_idx), 0, valid.shape[0] - 1))
        if not bool(valid[idx]):
            continue
        current_xy = _finite_xy_rows(position[idx])
        if current_xy.shape[0] == 0:
            continue
        dist = float(np.linalg.norm(current_xy[0] - center_xy))
        if dist > float(radius_m):
            continue
        start_idx = max(0, idx - int(PAST_STEPS))
        past_xy = _finite_xy_rows(position[start_idx : idx + 1][valid[start_idx : idx + 1]])
        rows.append({"track_id": str(track_id), "current_xy": current_xy[0], "past_xy": past_xy, "distance_m": dist})
    rows.sort(key=lambda row: (row["distance_m"], row["track_id"]))
    return rows[:18]


def _world_to_sdc_up_frame(xy_world: np.ndarray, *, center_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    xy = _finite_xy_rows(xy_world)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = xy - np.asarray(center_xy, dtype=np.float64).reshape(1, 2)
    rot = (math.pi / 2.0) - float(heading_rad)
    c = math.cos(rot)
    s = math.sin(rot)
    x_new = c * centered[:, 0] - s * centered[:, 1]
    y_new = s * centered[:, 0] + c * centered[:, 1]
    return np.stack([x_new, y_new], axis=-1).astype(np.float64)


def _wrapped_angle_delta(a: float, b: float) -> float:
    return float(math.atan2(math.sin(float(b) - float(a)), math.cos(float(b) - float(a))))


def _nearest_lane_feature_id(
    point_xy: np.ndarray,
    *,
    lane_features: Sequence[Mapping[str, Any]],
    max_distance_m: float = 3.5,
) -> Optional[str]:
    point = _finite_xy_rows(point_xy)
    if point.shape[0] == 0:
        return None
    best_feature_id = None
    best_distance = float("inf")
    for feature in lane_features:
        feature_id = str(feature.get("feature_id") or "")
        if not feature_id:
            continue
        lane_xy = _finite_xy_rows(np.asarray(feature.get("xy_world", []), dtype=np.float64))
        if lane_xy.shape[0] < 2:
            continue
        distance = _min_pointset_distance(point, lane_xy)
        if distance < best_distance:
            best_distance = distance
            best_feature_id = feature_id
    if best_feature_id is None or best_distance > float(max_distance_m):
        return None
    return best_feature_id


def _lane_ids_along_route(
    route_xy: np.ndarray,
    *,
    highlighted_metadata: Optional[Mapping[str, Any]],
    map_context: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Optional[str]]:
    route = _finite_xy_rows(route_xy)
    if route.shape[0] == 0:
        return []
    metadata = dict(highlighted_metadata or {})
    point_lane_ids = metadata.get("valid_point_road_part_ids")
    if isinstance(point_lane_ids, list) and len(point_lane_ids) == route.shape[0]:
        lane_ids: List[Optional[str]] = []
        for value in point_lane_ids:
            try:
                lane_id = int(value)
            except Exception:
                lane_id = -1
            lane_ids.append(None if lane_id < 0 else str(lane_id))
        return lane_ids
    lane_features = list(map_context.get("lane_centerlines", []))
    lane_ids = [_nearest_lane_feature_id(route[idx], lane_features=lane_features) for idx in range(route.shape[0])]
    return lane_ids


def _lane_feature_lookup(map_context: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    for feature in map_context.get("lane_centerlines", []):
        feature_id = str(feature.get("feature_id") or "")
        if not feature_id:
            continue
        xy = _finite_xy_rows(np.asarray(feature.get("xy_world", []), dtype=np.float64))
        if xy.shape[0] >= 2:
            lookup[feature_id] = xy
    return lookup


def _lane_features(map_context: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    return list(map_context.get("lane_centerlines", []))


def _nearest_polyline_index(polyline_xy: np.ndarray, point_xy: np.ndarray) -> int:
    xy = _finite_xy_rows(polyline_xy)
    point = _finite_xy_rows(np.asarray(point_xy, dtype=np.float64))
    if xy.shape[0] == 0 or point.shape[0] == 0:
        return 0
    d = np.linalg.norm(xy - point[0][None, :], axis=-1)
    return int(np.argmin(d))


def _oriented_lane_polyline(
    lane_xy: np.ndarray,
    *,
    current_xy: np.ndarray,
    current_heading: float,
) -> tuple[np.ndarray, int]:
    xy = _finite_xy_rows(lane_xy)
    if xy.shape[0] < 2:
        return xy, 0
    nearest_idx = _nearest_polyline_index(xy, current_xy)
    lo = max(0, nearest_idx - 1)
    hi = min(int(xy.shape[0] - 1), nearest_idx + 1)
    if hi == lo:
        segment_heading = float(current_heading)
    else:
        segment_heading = float(heading_from_points(xy[lo], xy[hi]))
    reverse_heading = float(math.atan2(math.sin(segment_heading + math.pi), math.cos(segment_heading + math.pi)))
    aligned_forward = abs(_wrapped_angle_delta(current_heading, segment_heading))
    aligned_reverse = abs(_wrapped_angle_delta(current_heading, reverse_heading))
    if aligned_reverse < aligned_forward:
        xy = np.asarray(xy[::-1], dtype=np.float64)
        nearest_idx = int(xy.shape[0] - 1 - nearest_idx)
    return xy, int(nearest_idx)


def _prefix_by_length(polyline_xy: np.ndarray, *, start_idx: int, max_length_m: float) -> np.ndarray:
    xy = _finite_xy_rows(polyline_xy)
    if xy.shape[0] == 0:
        return xy
    start_idx = int(np.clip(int(start_idx), 0, xy.shape[0] - 1))
    out = [xy[start_idx]]
    traveled = 0.0
    for idx in range(start_idx + 1, int(xy.shape[0])):
        step = float(np.linalg.norm(xy[idx] - xy[idx - 1]))
        traveled += step
        out.append(xy[idx])
        if traveled >= float(max_length_m):
            break
    return np.asarray(out, dtype=np.float64)


def _current_lane_guide(
    *,
    lane_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    max_length_m: float = STAY_LANE_GUIDE_LENGTH_M,
) -> np.ndarray:
    oriented_xy, nearest_idx = _oriented_lane_polyline(lane_xy, current_xy=current_xy, current_heading=current_heading)
    if oriented_xy.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    guide = _prefix_by_length(oriented_xy, start_idx=nearest_idx, max_length_m=max_length_m)
    if guide.shape[0] == 0:
        return guide
    if np.linalg.norm(guide[0] - current_xy) > 1e-3:
        guide = np.vstack([np.asarray(current_xy, dtype=np.float64).reshape(1, 2), guide])
    return np.asarray(guide, dtype=np.float64)


def _lane_segment_around_anchor(
    lane_xy: np.ndarray,
    *,
    anchor_xy: np.ndarray,
    before_length_m: float,
    after_length_m: float,
) -> np.ndarray:
    xy = _finite_xy_rows(lane_xy)
    anchor = _finite_xy_rows(np.asarray(anchor_xy, dtype=np.float64))
    if xy.shape[0] < 2 or anchor.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    nearest_idx = _nearest_polyline_index(xy, anchor[0])
    start_idx = nearest_idx
    traveled = 0.0
    while start_idx > 0 and traveled < float(before_length_m):
        traveled += float(np.linalg.norm(xy[start_idx] - xy[start_idx - 1]))
        start_idx -= 1
    end_idx = nearest_idx
    traveled = 0.0
    while end_idx < int(xy.shape[0] - 1) and traveled < float(after_length_m):
        traveled += float(np.linalg.norm(xy[end_idx + 1] - xy[end_idx]))
        end_idx += 1
    segment = np.asarray(xy[start_idx : end_idx + 1], dtype=np.float64)
    if segment.shape[0] >= 1 and np.min(np.linalg.norm(segment - anchor[0][None, :], axis=-1)) > 1e-3:
        segment = np.vstack([segment, anchor[0]])
    return np.asarray(segment, dtype=np.float64)


def _lane_transition_info(
    route_xy: np.ndarray,
    *,
    highlighted_metadata: Optional[Mapping[str, Any]],
    map_context: Mapping[str, Sequence[Mapping[str, Any]]],
    current_heading: float,
    terminal_heading: Optional[float] = None,
) -> Dict[str, Any]:
    lane_ids = _lane_ids_along_route(route_xy, highlighted_metadata=highlighted_metadata, map_context=map_context)
    route = _finite_xy_rows(route_xy)
    start_lane_id = next((lane_id for lane_id in lane_ids if lane_id), None)
    final_lane_id = next((lane_id for lane_id in reversed(lane_ids) if lane_id), None)
    switch_index = None
    for idx, lane_id in enumerate(lane_ids):
        if idx == 0 or lane_id is None or start_lane_id is None:
            continue
        if lane_id != start_lane_id:
            switch_index = idx
            break
    if terminal_heading is None and route.shape[0] >= 2:
        terminal_heading = float(heading_from_points(route[-2], route[-1]))
    heading_delta = None
    if terminal_heading is not None:
        heading_delta = abs(_wrapped_angle_delta(current_heading, terminal_heading))
    lane_change_like = bool(
        start_lane_id
        and final_lane_id
        and start_lane_id != final_lane_id
        and switch_index is not None
        and heading_delta is not None
        and heading_delta <= 0.65
    )
    crossing_point = None
    if switch_index is not None and route.shape[0] > switch_index:
        crossing_point = np.asarray(route[switch_index], dtype=np.float64)
    return {
        "lane_ids": lane_ids,
        "start_lane_id": start_lane_id,
        "final_lane_id": final_lane_id,
        "switch_index": switch_index,
        "crossing_point": crossing_point,
        "lane_change_like": lane_change_like,
        "heading_delta_rad": heading_delta,
    }


def _split_route_segments(
    xy_world: np.ndarray,
    *,
    point_road_part_ids: Optional[np.ndarray] = None,
    jump_threshold_m: float = ROUTE_DISCONTINUITY_JUMP_M,
) -> List[np.ndarray]:
    xy = _finite_xy_rows(xy_world)
    if xy.shape[0] < 2:
        return []
    road_ids = None
    if point_road_part_ids is not None:
        arr = np.asarray(point_road_part_ids).reshape(-1)
        if arr.shape[0] == xy.shape[0]:
            road_ids = arr.astype(np.int64)
    split_points = [0]
    for idx in range(1, xy.shape[0]):
        jump_m = float(np.linalg.norm(xy[idx] - xy[idx - 1]))
        road_change = bool(
            road_ids is not None
            and int(road_ids[idx]) >= 0
            and int(road_ids[idx - 1]) >= 0
            and int(road_ids[idx]) != int(road_ids[idx - 1])
        )
        if jump_m > float(jump_threshold_m) or (road_change and jump_m > 2.5):
            split_points.append(idx)
    split_points.append(int(xy.shape[0]))
    segments: List[np.ndarray] = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        segment = np.asarray(xy[start:end], dtype=np.float64)
        if segment.shape[0] >= 2:
            segments.append(segment)
    return segments

