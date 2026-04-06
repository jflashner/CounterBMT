from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.geometry import heading_from_points, polyline_length
from bmt.counterfactual.vlm_semantics.client import OpenAIVLMSemanticClient
from bmt.counterfactual.vlm_semantics.sdc_path_contract import (
    SLOT_IDS,
    make_empty_sdc_path_contract,
    normalize_sdc_path_contract,
    sdc_path_semantic_json_schema,
)
from bmt.counterfactual.vlm_semantics.sdc_path_prompt import build_single_sdc_path_semantic_prompt
from bmt.counterfactual.waymax_adapter import (
    raw_scenario_from_waymax_state,
    resolve_waymax_config,
    save_raw_waymax_scenario_pickle,
    waymax_available,
)
from waymax.dataloader import womd_dataloader


DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"
PLOT_RADIUS_M = 48.0
SDC_VERTICAL_FRACTION = 0.10
PAST_STEPS = 12
FIG_SIZE_INCH = 8.8
FIG_DPI = 180
CONTEXT_SELECTION_RADIUS_M = (2.0 * PLOT_RADIUS_M * (1.0 - SDC_VERTICAL_FRACTION)) + 12.0
STAY_LANE_GUIDE_LENGTH_M = 36.0
GT_COLOR = "#16a34a"
ALT_COLORS = ["#2563eb", "#f97316", "#7c3aed"]
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
ZOOM_PADDING_RATIO = 1.10
ZOOM_MIN_HALF_EXTENT_M = 20.0


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and label Waymax SDC path semantics with GPT-5.4-mini.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--max-scenes-scanned", type=int, default=80)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--min-route-length-m", type=float, default=15.0)
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--image-detail", type=str, default="high", choices=("low", "high", "original", "auto"))
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dotenv", type=str, default=".env")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--save-pkls", action="store_true")
    return parser.parse_args()


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


def _heading_arrow(pose_xy: np.ndarray, heading: float, *, length_m: float = 10.0) -> np.ndarray:
    x, y = float(pose_xy[0]), float(pose_xy[1])
    return np.asarray([[x, y], [x + length_m * math.cos(float(heading)), y + length_m * math.sin(float(heading))]], dtype=np.float64)


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


def _auto_half_extent(series_list: Sequence[np.ndarray], *, default_half_extent: float) -> float:
    valid_arrays = [arr for arr in (_finite_xy_rows(series) for series in series_list) if arr.shape[0] > 0]
    if not valid_arrays:
        return float(default_half_extent)
    stacked = np.concatenate(valid_arrays, axis=0)
    max_abs = float(np.max(np.abs(stacked)))
    half_extent = max(20.0, max_abs + 3.0)
    return float(min(default_half_extent, half_extent))


def _wrapped_angle_delta(a: float, b: float) -> float:
    return float(math.atan2(math.sin(float(b) - float(a)), math.cos(float(b) - float(a))))


def _path_focus_points(route_segments_local: Sequence[np.ndarray]) -> np.ndarray:
    segments = [_finite_xy_rows(segment) for segment in route_segments_local]
    segments = [segment for segment in segments if segment.shape[0] >= 2]
    if not segments:
        return np.zeros((0, 2), dtype=np.float64)
    best_seg_idx = 0
    best_point_idx = int(segments[0].shape[0] - 1)
    best_score = -1.0
    for seg_idx, segment in enumerate(segments):
        if segment.shape[0] < 2:
            continue
        initial_heading = float(heading_from_points(segment[0], segment[1]))
        local_best_score = -1.0
        local_best_point_idx = int(segment.shape[0] - 1)
        for idx in range(1, int(segment.shape[0])):
            heading = float(heading_from_points(segment[idx - 1], segment[idx]))
            score = abs(_wrapped_angle_delta(initial_heading, heading))
            if score > local_best_score:
                local_best_score = score
                local_best_point_idx = idx
        if local_best_score > best_score:
            best_score = local_best_score
            best_seg_idx = seg_idx
            best_point_idx = local_best_point_idx
    focus_parts: List[np.ndarray] = []
    for seg_idx, segment in enumerate(segments):
        if seg_idx < best_seg_idx:
            focus_parts.append(segment)
            continue
        if seg_idx == best_seg_idx:
            focus_parts.append(np.asarray(segment[: best_point_idx + 1], dtype=np.float64))
        break
    if not focus_parts:
        return np.asarray(segments[0], dtype=np.float64)
    stacked = np.concatenate([part for part in focus_parts if part.shape[0] > 0], axis=0)
    return np.asarray(stacked, dtype=np.float64)


def _path_zoom_half_extent(
    *,
    route_segments_local: Sequence[np.ndarray],
    context_series_local: Sequence[np.ndarray],
    min_half_extent_m: float = ZOOM_MIN_HALF_EXTENT_M,
    padding_ratio: float = ZOOM_PADDING_RATIO,
    default_half_extent_m: float = PLOT_RADIUS_M,
) -> float:
    focus_points = _path_focus_points(route_segments_local)
    valid_arrays = [focus_points]
    valid_arrays.extend(_finite_xy_rows(series) for series in context_series_local)
    valid_arrays = [arr for arr in valid_arrays if arr.shape[0] > 0]
    if not valid_arrays:
        return float(default_half_extent_m)
    stacked = np.concatenate(valid_arrays, axis=0)
    max_abs = float(np.max(np.abs(stacked)))
    half_extent = max(float(min_half_extent_m), float(padding_ratio) * max_abs)
    return float(min(float(default_half_extent_m), half_extent))


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


def _path_to_rows(raw_scenario: Mapping[str, Any], *, sdc_id: str, current_idx: int, min_route_length_m: float) -> tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    sdc_track = dict(raw_scenario["tracks"][str(sdc_id)]["state"])
    position = np.asarray(sdc_track.get("position", []), dtype=np.float64)
    valid = np.asarray(sdc_track.get("valid", []), dtype=bool)
    gt_future = _finite_xy_rows(position[current_idx:][valid[current_idx:]])
    gt_past = _finite_xy_rows(position[max(0, current_idx - PAST_STEPS) : current_idx + 1][valid[max(0, current_idx - PAST_STEPS) : current_idx + 1]])
    rows: List[Dict[str, Any]] = []
    for path_id, raw_path in sorted(raw_scenario.get("sdc_paths", {}).items(), key=lambda item: str(item[0])):
        coords = np.asarray(raw_path.get("polyline_xyz", []), dtype=np.float64)
        valid_mask = np.asarray(raw_path.get("valid", []), dtype=bool).reshape(-1)
        metadata = dict(raw_path.get("metadata", {}) or {})
        path_road_part_ids = np.asarray(metadata.get("point_road_part_ids", []), dtype=np.int64).reshape(-1)
        valid_xy = _finite_xy_rows(coords[valid_mask][:, :2]) if coords.ndim == 2 and coords.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)
        valid_point_ids = path_road_part_ids[valid_mask][: valid_xy.shape[0]] if path_road_part_ids.shape[0] == valid_mask.shape[0] else None
        segments_xy = _split_route_segments(valid_xy, point_road_part_ids=valid_point_ids)
        if not segments_xy:
            continue
        route_length_m = float(sum(polyline_length(segment.astype(np.float32)) for segment in segments_xy))
        if route_length_m < float(min_route_length_m):
            continue
        last_segment = np.asarray(segments_xy[-1], dtype=np.float64)
        terminal_heading = float(heading_from_points(last_segment[-2], last_segment[-1]))
        rows.append(
            {
                "path_id": str(path_id),
                "polyline_xy": valid_xy,
                "segments_xy": [segment.tolist() for segment in segments_xy],
                "num_points": int(valid_xy.shape[0]),
                "route_length_m": route_length_m,
                "on_route": bool(metadata.get("on_route", False)),
                "terminal_xy": np.asarray(last_segment[-1], dtype=np.float64),
                "terminal_heading": terminal_heading,
                "valid_point_road_part_ids": None if valid_point_ids is None else [int(v) for v in valid_point_ids.tolist()],
                "metadata": metadata,
            }
        )
    return gt_future, gt_past, rows


def _select_alternate_paths(
    path_rows: Sequence[Mapping[str, Any]],
    *,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in path_rows if bool(row.get("on_route", False))]
    if len(rows) < 3:
        return []
    sampled = rng.sample(rows, k=3)
    return [dict(row) for row in sampled]


def _render_single_image(
    *,
    output_path: Path,
    title: str,
    sidebar_lines: Sequence[str],
    center_xy: np.ndarray,
    current_xy: np.ndarray,
    current_heading: float,
    gt_past_xy: np.ndarray,
    highlighted_xy: np.ndarray,
    highlighted_segments_xy: Optional[Sequence[np.ndarray]],
    highlighted_color: str,
    highlighted_label: str,
    highlighted_gradient_values: Optional[Sequence[float]] = None,
    gradient_label_low: str = "shared",
    gradient_label_high: str = "distinct",
    map_context: Mapping[str, Sequence[Mapping[str, Any]]],
    traffic_lights: Sequence[Mapping[str, Any]],
    nearby_agents: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    fig = plt.figure(figsize=(FIG_SIZE_INCH, FIG_SIZE_INCH))
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_facecolor("#f8fafc")

    lane_lookup = _lane_feature_lookup(map_context)
    lane_features = _lane_features(map_context)
    route_xy_world = np.asarray(highlighted_xy, dtype=np.float64)
    transition_info = _lane_transition_info(
        route_xy_world,
        highlighted_metadata={"valid_point_road_part_ids": list()},
        map_context=map_context,
        current_heading=current_heading,
    )
    if highlighted_segments_xy:
        last_segment = _finite_xy_rows(np.asarray(list(highlighted_segments_xy)[-1], dtype=np.float64))
        if last_segment.shape[0] >= 2:
            transition_info = _lane_transition_info(
                route_xy_world,
                highlighted_metadata={"valid_point_road_part_ids": list()},
                map_context=map_context,
                current_heading=current_heading,
                terminal_heading=float(heading_from_points(last_segment[-2], last_segment[-1])),
            )

    start_lane_id = transition_info.get("start_lane_id") or _nearest_lane_feature_id(
        current_xy,
        lane_features=lane_features,
        max_distance_m=6.0,
    )
    final_anchor_xy = route_xy_world[-1] if route_xy_world.shape[0] > 0 else current_xy
    final_lane_id = transition_info.get("final_lane_id") or _nearest_lane_feature_id(
        final_anchor_xy,
        lane_features=lane_features,
        max_distance_m=6.0,
    )
    start_lane_xy = None if not start_lane_id else lane_lookup.get(str(start_lane_id))
    final_lane_xy = None if not final_lane_id else lane_lookup.get(str(final_lane_id))
    stay_lane_guide = (
        np.zeros((0, 2), dtype=np.float64)
        if start_lane_xy is None
        else _current_lane_guide(lane_xy=start_lane_xy, current_xy=current_xy, current_heading=current_heading)
    )

    def _plot_lane_corridor(
        lane_xy_world: Optional[np.ndarray],
        *,
        anchor_xy_world: np.ndarray,
        color: str,
        alpha_outer: float,
        alpha_inner: float,
        linewidth_outer: float,
        linewidth_inner: float,
        before_length_m: float,
        after_length_m: float,
        zorder: int,
    ) -> None:
        if lane_xy_world is None:
            return
        lane_window_world = _lane_segment_around_anchor(
            lane_xy_world,
            anchor_xy=anchor_xy_world,
            before_length_m=before_length_m,
            after_length_m=after_length_m,
        )
        lane_local = _world_to_sdc_up_frame(lane_window_world, center_xy=center_xy, heading_rad=current_heading)
        if lane_local.shape[0] < 2:
            return
        ax.plot(
            lane_local[:, 0],
            lane_local[:, 1],
            color=color,
            linewidth=linewidth_outer,
            alpha=alpha_outer,
            zorder=zorder,
            solid_capstyle="round",
        )
        ax.plot(
            lane_local[:, 0],
            lane_local[:, 1],
            color=color,
            linewidth=linewidth_inner,
            alpha=alpha_inner,
            zorder=zorder + 0.1,
            solid_capstyle="round",
        )

    _plot_lane_corridor(
        start_lane_xy,
        anchor_xy_world=current_xy,
        color=START_LANE_SHADE,
        alpha_outer=0.24,
        alpha_inner=0.34,
        linewidth_outer=30.0,
        linewidth_inner=22.0,
        before_length_m=18.0,
        after_length_m=58.0,
        zorder=1,
    )
    if final_lane_xy is not None:
        final_is_distinct = str(final_lane_id or "") != str(start_lane_id or "")
        _plot_lane_corridor(
            final_lane_xy,
            anchor_xy_world=final_anchor_xy,
            color=(FINAL_LANE_SHADE if final_is_distinct else START_LANE_SHADE),
            alpha_outer=(0.20 if final_is_distinct else 0.14),
            alpha_inner=(0.28 if final_is_distinct else 0.20),
            linewidth_outer=(26.0 if final_is_distinct else 20.0),
            linewidth_inner=(18.0 if final_is_distinct else 14.0),
            before_length_m=34.0,
            after_length_m=28.0,
            zorder=2,
        )

    for feature in map_context.get("crosswalks", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color=CROSSWALK_FACE, alpha=0.35, zorder=1)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=ROAD_COLOR, linewidth=2.8, alpha=0.98, zorder=4)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(
            np.asarray(feature["xy_world"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=LANE_COLOR, linewidth=1.1, alpha=0.30, zorder=5)

    for agent in nearby_agents:
        past_xy = _world_to_sdc_up_frame(
            np.asarray(agent["past_xy"], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if past_xy.shape[0] >= 2:
            ax.plot(past_xy[:, 0], past_xy[:, 1], color=AGENT_COLOR, linewidth=1.0, alpha=0.55, zorder=3)
        current_agent_xy = _world_to_sdc_up_frame(
            np.asarray([agent["current_xy"]], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if current_agent_xy.shape[0] > 0:
            ax.scatter([current_agent_xy[0, 0]], [current_agent_xy[0, 1]], c=AGENT_COLOR, s=14, alpha=0.85, zorder=4)

    for light in traffic_lights:
        stop_xy = _world_to_sdc_up_frame(
            np.asarray([light["stop_point_xy_world"]], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if stop_xy.shape[0] == 0:
            continue
        state = str(light.get("state") or "unknown")
        color = "#ef4444" if "STOP" in state or "RED" in state else ("#22c55e" if "GO" in state or "GREEN" in state else "#eab308")
        ax.scatter([stop_xy[0, 0]], [stop_xy[0, 1]], c=color, marker="s", s=80, edgecolors="black", linewidths=0.8, zorder=5)

    gt_past_local = _world_to_sdc_up_frame(gt_past_xy, center_xy=center_xy, heading_rad=current_heading)
    if gt_past_local.shape[0] >= 2:
        ax.plot(gt_past_local[:, 0], gt_past_local[:, 1], color="#111827", linewidth=2.8, alpha=0.95, zorder=6)

    stay_lane_local = _world_to_sdc_up_frame(stay_lane_guide, center_xy=center_xy, heading_rad=current_heading)
    if stay_lane_local.shape[0] >= 2:
        ax.plot(
            stay_lane_local[:, 0],
            stay_lane_local[:, 1],
            color=STAY_GUIDE_COLOR,
            linewidth=2.0,
            alpha=0.45,
            linestyle=(0, (5, 4)),
            zorder=7,
        )

    route_segments = list(highlighted_segments_xy or [])
    if not route_segments:
        route_segments = [np.asarray(highlighted_xy, dtype=np.float64)]
    transformed_route_segments = [
        _world_to_sdc_up_frame(np.asarray(segment, dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        for segment in route_segments
    ]
    transformed_route_segments = [segment for segment in transformed_route_segments if segment.shape[0] >= 2]
    gradient_values = None if highlighted_gradient_values is None else np.asarray(list(highlighted_gradient_values), dtype=np.float64).reshape(-1)
    if transformed_route_segments:
        if gradient_values is not None and gradient_values.size > 0:
            norm = Normalize(vmin=0.0, vmax=1.0)
            cmap = plt.cm.viridis
            grad_cursor = 0
            for seg_idx, segment in enumerate(transformed_route_segments):
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color="#0f172a",
                    linewidth=6.2,
                    alpha=0.18,
                    zorder=8.8,
                    solid_capstyle="round",
                )
                for point_idx in range(1, int(segment.shape[0])):
                    value_idx = min(grad_cursor + point_idx - 1, int(gradient_values.size - 1))
                    color = cmap(norm(float(np.clip(gradient_values[value_idx], 0.0, 1.0))))
                    ax.plot(
                        segment[point_idx - 1 : point_idx + 1, 0],
                        segment[point_idx - 1 : point_idx + 1, 1],
                        color=color,
                        linewidth=5.2,
                        alpha=0.99,
                        zorder=9,
                        solid_capstyle="round",
                    )
                grad_cursor += int(segment.shape[0])
                if seg_idx > 0:
                    ax.scatter([segment[0, 0]], [segment[0, 1]], c="#111827", s=30, marker="x", linewidths=1.2, zorder=10)
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cax = fig.add_axes([0.88, 0.10, 0.03, 0.18])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_ticks([0.0, 1.0])
            cbar.set_ticklabels([str(gradient_label_low), str(gradient_label_high)])
            cbar.ax.tick_params(labelsize=7, length=0)
            cbar.outline.set_linewidth(0.6)
        else:
            for seg_idx, segment in enumerate(transformed_route_segments):
                ax.plot(
                    segment[:, 0],
                    segment[:, 1],
                    color=highlighted_color,
                    linewidth=5.2,
                    alpha=0.98,
                    zorder=9,
                    solid_capstyle="round",
                )
                if seg_idx > 0:
                    ax.scatter([segment[0, 0]], [segment[0, 1]], c=highlighted_color, s=30, marker="x", linewidths=1.2, zorder=10)
        final_segment = transformed_route_segments[-1]
        ax.scatter([final_segment[-1, 0]], [final_segment[-1, 1]], c=highlighted_color, s=80, edgecolors="white", linewidths=1.1, zorder=10)

    current_local = np.zeros((1, 2), dtype=np.float64)
    arrow_xy = np.asarray([[0.0, 0.0], [0.0, 11.0]], dtype=np.float64)
    ax.scatter([current_local[0, 0]], [current_local[0, 1]], c=SDC_DOT_COLOR, s=105, edgecolors="white", linewidths=1.4, zorder=12)
    ax.annotate(
        "",
        xy=(arrow_xy[1, 0], arrow_xy[1, 1]),
        xytext=(arrow_xy[0, 0], arrow_xy[0, 1]),
        arrowprops={"arrowstyle": "-|>", "linewidth": 3.2, "color": SDC_ARROW_COLOR},
        zorder=13,
    )
    ax.text(0.0, 0.0, " SDC", color=SDC_ARROW_COLOR, fontsize=11, weight="bold", zorder=14)
    ax.text(-3.1, 6.0, "L", color=SDC_ARROW_COLOR, fontsize=10, weight="bold", ha="center", va="center", zorder=14)
    ax.text(3.1, 6.0, "R", color=SDC_ARROW_COLOR, fontsize=10, weight="bold", ha="center", va="center", zorder=14)

    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI)
    plt.close(fig)
    pixel_size = {
        "width_px": int(round(float(FIG_SIZE_INCH) * float(FIG_DPI))),
        "height_px": int(round(float(FIG_SIZE_INCH) * float(FIG_DPI))),
    }
    return {
        "half_extent_m": float(half_extent),
        "pixel_size": pixel_size,
        "start_lane_id": None if start_lane_id is None else str(start_lane_id),
        "final_lane_id": None if final_lane_id is None else str(final_lane_id),
        "lane_change_like": bool(transition_info.get("lane_change_like", False)),
    }


def _build_example_payload(
    *,
    raw_scenario: Mapping[str, Any],
    example_id: str,
    scenario_id: str,
    sdc_id: str,
    current_time_index: int,
    gt_future_xy: np.ndarray,
    gt_past_xy: np.ndarray,
    alt_paths: Sequence[Mapping[str, Any]],
    image_dir: Path,
) -> Dict[str, Any]:
    current_state = dict(raw_scenario["tracks"][str(sdc_id)]["state"])
    current_position = np.asarray(current_state["position"], dtype=np.float64)
    current_heading_seq = np.asarray(current_state["heading"], dtype=np.float64)
    current_valid = np.asarray(current_state["valid"], dtype=bool)
    idx = int(np.clip(int(current_time_index), 0, current_position.shape[0] - 1))
    while idx > 0 and not bool(current_valid[idx]):
        idx -= 1
    current_xy = _finite_xy_rows(current_position[idx])[0]
    current_heading = float(current_heading_seq[idx]) if idx < current_heading_seq.shape[0] and np.isfinite(current_heading_seq[idx]) else 0.0
    map_context = _select_map_context(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M)
    traffic_lights = _select_traffic_lights(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M, time_index=idx)
    nearby_agents = _select_nearby_agents(raw_scenario, sdc_id=sdc_id, center_xy=current_xy, current_idx=idx, radius_m=CONTEXT_SELECTION_RADIUS_M)

    slot_metadata = [
        {
            "slot_id": "gt",
            "source_kind": "ground_truth",
            "path_id": None,
            "on_route": True,
            "route_length_m": float(polyline_length(gt_future_xy.astype(np.float32))) if gt_future_xy.shape[0] >= 2 else 0.0,
        }
    ]
    images = {}
    image_zoom_half_extent_m: Dict[str, float] = {}
    image_pixel_size: Dict[str, Dict[str, int]] = {}
    lane_overlay_metadata: Dict[str, Dict[str, Any]] = {}
    gt_sidebar = [
        f"example={example_id}",
        f"scenario={scenario_id}",
        f"sdc={sdc_id}",
        f"time={idx}",
        "slot=gt",
        "highlight=ground truth future",
        "heading arrow = current SDC heading",
    ]
    images["gt"] = str((image_dir / "gt.png").resolve())
    gt_render_info = _render_single_image(
        output_path=Path(images["gt"]),
        title=f"{example_id} | GT future",
        sidebar_lines=gt_sidebar,
        center_xy=current_xy,
        current_xy=current_xy,
        current_heading=current_heading,
        gt_past_xy=gt_past_xy,
        highlighted_xy=gt_future_xy,
        highlighted_segments_xy=[np.asarray(gt_future_xy, dtype=np.float64)] if gt_future_xy.shape[0] >= 2 else None,
        highlighted_color=GT_COLOR,
        highlighted_label="GT",
        map_context=map_context,
        traffic_lights=traffic_lights,
        nearby_agents=nearby_agents,
    )
    image_zoom_half_extent_m["gt"] = float(gt_render_info["half_extent_m"])
    image_pixel_size["gt"] = dict(gt_render_info["pixel_size"])
    lane_overlay_metadata["gt"] = {
        "start_lane_id": gt_render_info.get("start_lane_id"),
        "final_lane_id": gt_render_info.get("final_lane_id"),
        "lane_change_like": bool(gt_render_info.get("lane_change_like", False)),
    }

    for alt_index, path_row in enumerate(alt_paths, start=1):
        slot_id = f"alt_{alt_index}"
        slot_metadata.append(
            {
                "slot_id": slot_id,
                "source_kind": "sdc_path",
                "path_id": str(path_row["path_id"]),
                "on_route": bool(path_row["on_route"]),
                "route_length_m": float(path_row["route_length_m"]),
            }
        )
        images[slot_id] = str((image_dir / f"{slot_id}.png").resolve())
        sidebar = [
            f"example={example_id}",
            f"scenario={scenario_id}",
            f"sdc={sdc_id}",
            f"time={idx}",
            f"slot={slot_id}",
            f"path_id={path_row['path_id']}",
            f"on_route={bool(path_row['on_route'])}",
            f"route_length_m={float(path_row['route_length_m']):.1f}",
            "heading arrow = current SDC heading",
        ]
        render_info = _render_single_image(
            output_path=Path(images[slot_id]),
            title=f"{example_id} | {slot_id} candidate",
            sidebar_lines=sidebar,
            center_xy=current_xy,
            current_xy=current_xy,
            current_heading=current_heading,
            gt_past_xy=gt_past_xy,
            highlighted_xy=np.asarray(path_row["polyline_xy"], dtype=np.float64),
            highlighted_segments_xy=[np.asarray(segment, dtype=np.float64) for segment in list(path_row.get("segments_xy") or [])],
            highlighted_color=ALT_COLORS[(alt_index - 1) % len(ALT_COLORS)],
            highlighted_label=slot_id.upper(),
            map_context=map_context,
            traffic_lights=traffic_lights,
            nearby_agents=nearby_agents,
        )
        image_zoom_half_extent_m[slot_id] = float(render_info["half_extent_m"])
        image_pixel_size[slot_id] = dict(render_info["pixel_size"])
        lane_overlay_metadata[slot_id] = {
            "start_lane_id": render_info.get("start_lane_id"),
            "final_lane_id": render_info.get("final_lane_id"),
            "lane_change_like": bool(render_info.get("lane_change_like", False)),
        }

    return {
        "example_id": str(example_id),
        "scenario_id": str(scenario_id),
        "sdc_id": str(sdc_id),
        "current_time_index": int(idx),
        "slot_metadata": slot_metadata,
        "images": images,
        "image_zoom_half_extent_m": image_zoom_half_extent_m,
        "image_pixel_size": image_pixel_size,
        "lane_overlay_metadata": lane_overlay_metadata,
        "map_counts": {key: int(len(value)) for key, value in map_context.items()},
        "traffic_lights_count": int(len(traffic_lights)),
        "nearby_agents_count": int(len(nearby_agents)),
    }


def _slot_path_row(contract: Mapping[str, Any], *, slot_row: Mapping[str, Any]) -> Dict[str, Any]:
    slot_id = str(slot_row.get("slot_id") or "gt")
    source_kind = str(slot_row.get("source_kind") or "sdc_path")
    path_id = slot_row.get("path_id")
    for row in list(contract.get("highlighted_paths") or []):
        if str(row.get("slot_id") or "") == slot_id:
            return {
                "slot_id": slot_id,
                "source_kind": source_kind,
                "path_id": None if path_id is None else str(path_id),
                "semantic_label": str(row.get("semantic_label") or "straight"),
                "confidence": float(row.get("confidence") or 0.0),
                "is_valid_target": bool(row.get("is_valid_target") or False),
                "rationale_short": str(row.get("rationale_short") or ""),
            }
    return {
        "slot_id": slot_id,
        "source_kind": source_kind,
        "path_id": None if path_id is None else str(path_id),
        "semantic_label": "straight",
        "confidence": 0.0,
        "is_valid_target": False,
        "rationale_short": "",
    }


def _aggregate_scene_ambiguity(slot_contracts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    level_rank = {"low": 0, "medium": 1, "high": 2}
    best_level = "low"
    best_confidence = 1.0
    rationale = "Aggregated from per-slot labeling."
    for contract in slot_contracts:
        ambiguity = dict(contract.get("scene_ambiguity") or {})
        level = str(ambiguity.get("level") or "medium")
        confidence = float(ambiguity.get("confidence") or 0.0)
        if level_rank.get(level, 1) > level_rank.get(best_level, 0):
            best_level = level
        best_confidence = min(best_confidence, confidence if confidence > 0.0 else 1.0)
    if not slot_contracts:
        best_level = "medium"
        best_confidence = 0.0
    return {
        "level": best_level,
        "confidence": float(best_confidence),
        "rationale_short": rationale,
    }


def main() -> int:
    args = parse_args()
    if not waymax_available():
        raise SystemExit("waymax is not installed in this environment")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    config = resolve_waymax_config(
        config_name=str(args.config_name),
        path=str(args.path),
        include_sdc_paths=True,
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )
    if dataclasses.is_dataclass(config):
        if hasattr(config, "num_shards"):
            config = dataclasses.replace(config, num_shards=1, deterministic=True)

    client = OpenAIVLMSemanticClient(dotenv_path=args.dotenv)
    render_rows: List[Dict[str, Any]] = []
    raw_contract_rows: List[Dict[str, Any]] = []
    aggregate_index_rows: List[Dict[str, Any]] = []
    prompt_manifest_rows: List[Dict[str, Any]] = []

    scene_iter = itertools.islice(womd_dataloader.simulator_state_generator(config=config), int(args.scene_offset), None)
    num_scanned = 0
    num_selected = 0

    for scene_index, state in enumerate(scene_iter, start=int(args.scene_offset)):
        if num_selected >= int(args.num_examples):
            break
        if num_scanned >= int(args.max_scenes_scanned):
            break
        num_scanned += 1

        fallback_scenario_id = f"waymax_scene_{scene_index:05d}"
        raw = raw_scenario_from_waymax_state(
            state,
            scenario_id=fallback_scenario_id,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or fallback_scenario_id)
        sdc_id = str(dict(raw.get("metadata", {})).get("sdc_id") or "")
        if not sdc_id or str(sdc_id) not in dict(raw.get("tracks", {})):
            continue
        current_time_index = int(dict(raw.get("metadata", {})).get("current_time_index") or 0)
        gt_future_xy, gt_past_xy, path_rows = _path_to_rows(
            raw,
            sdc_id=sdc_id,
            current_idx=current_time_index,
            min_route_length_m=float(args.min_route_length_m),
        )
        if gt_future_xy.shape[0] < 5:
            continue
        example_rng = random.Random(f"{int(args.random_seed)}::{scenario_id}::{sdc_id}::{current_time_index}")
        eligible_on_route_paths = [str(row["path_id"]) for row in path_rows if bool(row.get("on_route", False))]
        alt_paths = _select_alternate_paths(path_rows, rng=example_rng)
        if len(alt_paths) != 3:
            continue

        example_id = f"{scenario_id}__sdc_{sdc_id}__t_{current_time_index:03d}"
        example_dir = outdir / "examples" / example_id
        image_dir = example_dir / "images"
        payload = _build_example_payload(
            raw_scenario=raw,
            example_id=example_id,
            scenario_id=scenario_id,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
            gt_future_xy=gt_future_xy,
            gt_past_xy=gt_past_xy,
            alt_paths=alt_paths,
            image_dir=image_dir,
        )
        payload["scene_index"] = int(scene_index)
        payload["sampling_key"] = f"{int(args.random_seed)}::{scenario_id}::{sdc_id}::{current_time_index}"
        payload["eligible_on_route_path_ids"] = eligible_on_route_paths
        payload["num_eligible_on_route_paths"] = int(len(eligible_on_route_paths))
        payload["selected_alt_paths"] = [
            {
                "path_id": str(row["path_id"]),
                "on_route": bool(row["on_route"]),
                "route_length_m": float(row["route_length_m"]),
            }
            for row in alt_paths
        ]
        write_json(example_dir / "render_metadata.json", payload)
        if bool(args.save_pkls):
            save_raw_waymax_scenario_pickle(state, out_path=example_dir / f"sd_waymo_v1.3.1_{scenario_id}.pkl", current_time_index=current_time_index)

        slot_rows = list(payload.get("slot_metadata") or [])
        slot_raw_contracts: Dict[str, Dict[str, Any]] = {}
        slot_normalized_contracts: Dict[str, Dict[str, Any]] = {}

        for slot_row in slot_rows:
            slot_id = str(slot_row.get("slot_id") or "")
            if slot_id not in SLOT_IDS:
                continue
            prompt_text = build_single_sdc_path_semantic_prompt(payload, slot_row=slot_row)
            prompt_path = example_dir / f"prompt_{slot_id}.txt"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            request_payload = {
                "example_id": example_id,
                "slot_id": slot_id,
                "model": str(args.model),
                "image_detail": str(args.image_detail),
                "image_paths": [payload["images"][slot_id]],
                "prompt": prompt_text,
                "json_schema": sdc_path_semantic_json_schema(),
            }
            request_json_path = example_dir / f"request_{slot_id}.json"
            write_json(request_json_path, request_payload)
            prompt_manifest_rows.append(
                {
                    "example_id": example_id,
                    "scenario_id": scenario_id,
                    "sdc_id": sdc_id,
                    "slot_id": slot_id,
                    "prompt_path": str(prompt_path.resolve()),
                    "request_json": str(request_json_path.resolve()),
                    "image_paths": request_payload["image_paths"],
                }
            )
            if bool(args.skip_api) or not client.available:
                continue
            raw_contract = client.label_contract(
                prompt=prompt_text,
                image_paths=request_payload["image_paths"],
                model_name=str(args.model),
                image_detail=str(args.image_detail),
                max_completion_tokens=1000,
                json_schema=sdc_path_semantic_json_schema(),
            )
            raw_path = example_dir / f"contract_raw_{slot_id}.json"
            write_json(raw_path, raw_contract)
            normalized_slot_contract = normalize_sdc_path_contract(
                raw_contract,
                example_id=example_id,
                scenario_id=scenario_id,
                sdc_id=sdc_id,
                current_time_index=current_time_index,
                model_name=str(args.model),
            )
            normalized_path = example_dir / f"contract_normalized_{slot_id}.json"
            write_json(normalized_path, normalized_slot_contract)
            slot_raw_contracts[slot_id] = raw_contract
            slot_normalized_contracts[slot_id] = normalized_slot_contract

        normalized_contract = None
        if slot_normalized_contracts:
            aggregated_payload = make_empty_sdc_path_contract(
                example_id=example_id,
                scenario_id=scenario_id,
                sdc_id=sdc_id,
                current_time_index=current_time_index,
                model_name=str(args.model),
            )
            aggregated_payload["scene_ambiguity"] = _aggregate_scene_ambiguity(list(slot_normalized_contracts.values()))
            aggregated_payload["highlighted_paths"] = [
                _slot_path_row(slot_normalized_contracts[slot_id], slot_row=slot_row)
                for slot_row in slot_rows
                for slot_id in [str(slot_row.get("slot_id") or "")]
                if slot_id in slot_normalized_contracts
            ]
            aggregated_payload["use_for_training"] = bool(
                aggregated_payload["highlighted_paths"]
                and all(bool(row.get("is_valid_target")) for row in aggregated_payload["highlighted_paths"])
                and all(bool(contract.get("use_for_training")) for contract in slot_normalized_contracts.values())
            )
            aggregated_payload["notes"] = [
                f"per_slot_requests={len(slot_normalized_contracts)}",
            ]
            normalized_contract = normalize_sdc_path_contract(
                aggregated_payload,
                example_id=example_id,
                scenario_id=scenario_id,
                sdc_id=sdc_id,
                current_time_index=current_time_index,
                model_name=str(args.model),
            )
            aggregated_raw = {
                "example_id": example_id,
                "scenario_id": scenario_id,
                "sdc_id": sdc_id,
                "current_time_index": int(current_time_index),
                "mode": "per_slot_requests",
                "slot_raw_contracts": slot_raw_contracts,
            }
            write_json(example_dir / "contract_raw.json", aggregated_raw)
            write_json(example_dir / "contract_normalized.json", normalized_contract)
            raw_contract_rows.append(
                {
                    "example_id": example_id,
                    "scenario_id": scenario_id,
                    "sdc_id": sdc_id,
                    "raw_contract_path": str((example_dir / "contract_raw.json").resolve()),
                    "normalized_contract_path": str((example_dir / "contract_normalized.json").resolve()),
                    "contract": normalized_contract,
                }
            )

        aggregate_index_rows.append(
            {
                "example_id": example_id,
                "scenario_id": scenario_id,
                "sdc_id": sdc_id,
                "scene_index": int(scene_index),
                "current_time_index": int(current_time_index),
                "slot_metadata": payload["slot_metadata"],
                "images": payload["images"],
                "prompt_paths": {
                    str(slot_row.get("slot_id") or ""): str((example_dir / f"prompt_{slot_row.get('slot_id')}.txt").resolve())
                    for slot_row in slot_rows
                    if str(slot_row.get("slot_id") or "") in SLOT_IDS
                },
                "request_jsons": {
                    str(slot_row.get("slot_id") or ""): str((example_dir / f"request_{slot_row.get('slot_id')}.json").resolve())
                    for slot_row in slot_rows
                    if str(slot_row.get("slot_id") or "") in SLOT_IDS
                },
                "contract": normalized_contract,
            }
        )
        render_rows.append(payload)
        num_selected += 1

    render_manifest_path = outdir / "sdc_path_vlm_render_manifest.json"
    write_json(
        render_manifest_path,
        {
            "path": str(args.path),
            "num_examples_requested": int(args.num_examples),
            "num_examples_rendered": int(len(render_rows)),
            "num_scenes_scanned": int(num_scanned),
            "rows": render_rows,
        },
    )
    request_manifest_path = outdir / "sdc_path_vlm_request_manifest.jsonl"
    write_jsonl(request_manifest_path, prompt_manifest_rows)
    raw_contracts_path = outdir / "sdc_path_vlm_contracts_raw.jsonl"
    write_jsonl(raw_contracts_path, raw_contract_rows)
    index_path = outdir / "sdc_path_semantics_index.jsonl"
    write_jsonl(index_path, aggregate_index_rows)
    summary = {
        "path": str(args.path),
        "num_examples_requested": int(args.num_examples),
        "num_examples_rendered": int(len(render_rows)),
        "num_examples_labeled": int(len(raw_contract_rows)),
        "num_requests_written": int(len(prompt_manifest_rows)),
        "num_scenes_scanned": int(num_scanned),
        "model": str(args.model),
        "image_detail": str(args.image_detail),
        "skip_api": bool(args.skip_api),
        "api_available": bool(client.available),
        "render_manifest_json": str(render_manifest_path.resolve()),
        "request_manifest_jsonl": str(request_manifest_path.resolve()),
        "raw_contracts_jsonl": str(raw_contracts_path.resolve()),
        "semantics_index_jsonl": str(index_path.resolve()),
    }
    write_json(outdir / "sdc_path_vlm_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
