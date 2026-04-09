from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from bmt.counterfactual.sdc_path_control import (
    _extract_valid_sdc_path_xy,
    _sanitize_polyline,
    extract_ground_truth_sdc_route_xy,
    extract_sdc_current_pose,
    trim_polyline_from_point,
)
from scripts.counterfactual.label_waymax_sdc_path_semantics import PLOT_RADIUS_M, SDC_VERTICAL_FRACTION, _world_to_sdc_up_frame


def selected_raw_route_world(raw_scenario: Mapping[str, Any], row: Mapping[str, Any]) -> np.ndarray:
    sdc_id = str(row.get("sdc_id") or "")
    current_time_index = int(row.get("current_time_index") or 0)
    selected_path_id = row.get("selected_path_id")
    if selected_path_id is None:
        return extract_ground_truth_sdc_route_xy(
            raw_scenario,
            sdc_id=sdc_id,
            current_time_index=current_time_index,
        )
    current_xy_world, _ = extract_sdc_current_pose(
        raw_scenario,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
    )
    raw_xy = _extract_valid_sdc_path_xy(raw_scenario, str(selected_path_id))
    return trim_polyline_from_point(raw_xy, current_xy_world, prepend_point=True)


def polyline_segment_distance_to_points(points_xy: Any, polyline_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    polyline = _sanitize_polyline(polyline_xy)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if polyline.shape[0] == 0:
        return np.full((points.shape[0],), np.inf, dtype=np.float32)
    if polyline.shape[0] == 1:
        return np.linalg.norm(points - polyline[0][None, :], axis=-1).astype(np.float32)

    seg_start = np.asarray(polyline[:-1], dtype=np.float32)
    seg_end = np.asarray(polyline[1:], dtype=np.float32)
    seg_vec = seg_end - seg_start
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=-1).clip(min=1e-6)

    rel = points[:, None, :] - seg_start[None, :, :]
    t = np.sum(rel * seg_vec[None, :, :], axis=-1) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=-1), axis=-1).astype(np.float32)


def segment_distance_field_in_sdc_frame(
    *,
    polyline_world_xy: Any,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
    grid_step_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    world_xy = sdc_up_to_world_frame(local_xy, center_xy_world=center_xy_world, heading_world_rad=heading_world_rad)
    dist = polyline_segment_distance_to_points(world_xy, polyline_world_xy).reshape(xx.shape)
    return xx, yy, dist


def sdc_up_to_world_frame(
    points_xy_local: Any,
    *,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
) -> np.ndarray:
    local_xy = np.asarray(points_xy_local, dtype=np.float32).reshape(-1, 2)
    if local_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rot = float(heading_world_rad) - (math.pi / 2.0)
    c = math.cos(rot)
    s = math.sin(rot)
    x_world = c * local_xy[:, 0] - s * local_xy[:, 1] + float(center_xy_world[0])
    y_world = s * local_xy[:, 0] + c * local_xy[:, 1] + float(center_xy_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def world_to_sdc_up_frame(
    points_world_xy: Any,
    *,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
) -> np.ndarray:
    return np.asarray(
        _world_to_sdc_up_frame(
            np.asarray(points_world_xy, dtype=np.float32),
            center_xy=np.asarray(center_xy_world, dtype=np.float32),
            heading_rad=float(heading_world_rad),
        ),
        dtype=np.float32,
    )


def tube_reward_from_distance(
    distance_m: Any,
    *,
    tube_radius_m: float,
    inside_reward: float = 1.0,
    outside_scale: float = 1.0,
) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=np.float32)
    radius = max(float(tube_radius_m), 1e-3)
    reward = np.full(distance.shape, float(inside_reward), dtype=np.float32)
    outside = distance > radius
    if bool(np.any(outside)):
        reward[outside] = -(distance[outside] - radius) * float(outside_scale)
    return reward.astype(np.float32)


def return_to_go(reward_t: Any, *, gamma: float = 1.0) -> np.ndarray:
    reward = np.asarray(reward_t, dtype=np.float32).reshape(-1)
    out = np.zeros_like(reward, dtype=np.float32)
    running = 0.0
    for idx in range(int(reward.shape[0]) - 1, -1, -1):
        running = float(reward[idx]) + float(gamma) * float(running)
        out[idx] = running
    return out


def group_normalized_advantages(values: Any, *, axis: int = 0, eps: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mean = np.mean(array, axis=axis, keepdims=True)
    std = np.std(array, axis=axis, keepdims=True)
    return ((array - mean) / np.maximum(std, float(eps))).astype(np.float32)
