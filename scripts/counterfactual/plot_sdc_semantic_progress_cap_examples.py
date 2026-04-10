from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_path_control import (
    _extract_valid_sdc_path_xy,
    extract_ground_truth_sdc_route_xy,
    extract_sdc_current_pose,
    polyline_arc_lengths,
    polyline_length_m,
    split_polyline_on_discontinuities,
    trim_polyline_from_point,
)
from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from scripts.counterfactual.path_semantics_plot_utils import (
    AGENT_COLOR,
    CROSSWALK_FACE,
    LANE_COLOR,
    PAST_STEPS,
    ROAD_COLOR,
    ROUTE_DISCONTINUITY_JUMP_M,
    SDC_DOT_COLOR,
    _finite_xy_rows,
    _select_map_context,
    _select_nearby_agents,
    _select_traffic_lights,
    _world_to_sdc_up_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot scene context with raw route, divergence point, and divergence-radius progress endpoint."
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Example formatted as scenario_id:slot_id . Repeatable.",
    )
    parser.add_argument("--radius-from-divergence-m", type=float, default=80.0)
    parser.add_argument("--tube-radius-m", type=float, default=3.0)
    parser.add_argument("--grid-step-m", type=float, default=0.35)
    parser.add_argument("--jump-threshold-m", type=float, default=6.0)
    return parser.parse_args()


def _parse_example_spec(spec: str) -> Tuple[str, str]:
    text = str(spec).strip()
    if ":" not in text:
        raise ValueError(f"Example spec must be scenario_id:slot_id, got {spec!r}")
    scenario_id, slot_id = text.split(":", 1)
    return str(scenario_id).strip(), str(slot_id).strip()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _select_row(rows: Sequence[Mapping[str, Any]], *, row_index: int, scenario_id: str, slot_id: str) -> int:
    if int(row_index) >= 0:
        idx = int(row_index)
        if idx >= len(rows):
            raise IndexError(f"row_index {idx} out of range for {len(rows)} rows")
        return idx
    target_scenario = str(scenario_id).strip()
    target_slot = str(slot_id).strip()
    for idx, row in enumerate(rows):
        if str(row.get("scenario_id") or "").strip() != target_scenario:
            continue
        if target_slot and str(row.get("selected_slot_id") or "").strip() != target_slot:
            continue
        return int(idx)
    raise KeyError(f"Could not find row for scenario_id={target_scenario!r} slot_id={target_slot!r}")


def _finite_nonnegative_min(values: Sequence[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float32).reshape(-1)
    valid = np.isfinite(array) & (array >= 0.0)
    if not np.any(valid):
        return 0.0
    return float(np.min(array[valid]))


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
    polyline = _finite_xy_rows(np.asarray(polyline_xy, dtype=np.float32))
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


def segment_distance_field_in_sdc_frame(
    *,
    polyline_world_xy: Any,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
    grid_step_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_extent = 48.0
    vertical_span = 2.0 * half_extent
    y_min = -0.10 * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    world_xy = sdc_up_to_world_frame(local_xy, center_xy_world=center_xy_world, heading_world_rad=heading_world_rad)
    dist = polyline_segment_distance_to_points(world_xy, polyline_world_xy).reshape(xx.shape)
    return xx, yy, dist


def _sample_point_along_polyline(points_xy: np.ndarray, arc_m: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((2,), dtype=np.float32)
    if points.shape[0] == 1:
        return points[0].astype(np.float32)
    arc = polyline_arc_lengths(points)
    target = float(np.clip(float(arc_m), 0.0, float(arc[-1])))
    idx = int(np.searchsorted(arc, target, side="right"))
    if idx <= 0:
        return points[0].astype(np.float32)
    if idx >= points.shape[0]:
        return points[-1].astype(np.float32)
    left_idx = idx - 1
    right_idx = idx
    left_arc = float(arc[left_idx])
    right_arc = float(arc[right_idx])
    denom = max(right_arc - left_arc, 1e-6)
    alpha = float((target - left_arc) / denom)
    point = (1.0 - alpha) * points[left_idx] + alpha * points[right_idx]
    return np.asarray(point, dtype=np.float32)


def _resample_polyline_interval(points_xy: np.ndarray, start_arc_m: float, end_arc_m: float, spacing_m: float = 0.5) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    total_arc = float(polyline_length_m(points))
    start_arc = float(np.clip(float(start_arc_m), 0.0, total_arc))
    end_arc = float(np.clip(float(end_arc_m), start_arc, total_arc))
    if end_arc <= start_arc + 1e-6:
        return _sample_point_along_polyline(points, start_arc)[None, :]
    num_steps = max(2, int(math.ceil((end_arc - start_arc) / max(float(spacing_m), 1e-3))) + 1)
    sample_arcs = np.linspace(start_arc, end_arc, num=num_steps, dtype=np.float32)
    sampled = np.stack([_sample_point_along_polyline(points, float(val)) for val in sample_arcs], axis=0)
    return np.asarray(sampled, dtype=np.float32)


def _first_exit_arc_from_circle_center(points_xy: np.ndarray, center_xy: np.ndarray, radius_m: float, *, start_arc_m: float = 0.0) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return 0.0
    total_arc = float(polyline_length_m(points))
    onset = float(np.clip(float(start_arc_m), 0.0, total_arc))
    center = np.asarray(center_xy, dtype=np.float32).reshape(-1)
    if center.shape[0] < 2:
        center = np.zeros((2,), dtype=np.float32)
    arc = polyline_arc_lengths(points)
    right_idx = int(np.searchsorted(arc, onset, side="right"))
    right_idx = min(max(right_idx, 1), int(points.shape[0] - 1))
    start_point = _sample_point_along_polyline(points, onset)
    tail_points = np.concatenate([start_point[None, :], points[right_idx:]], axis=0).astype(np.float32)
    if tail_points.shape[0] < 2:
        return onset
    rel = tail_points - center[None, :]
    dist_sq = np.sum(rel * rel, axis=-1)
    radius_sq = float(radius_m) * float(radius_m)
    tail_seg_len = np.linalg.norm(tail_points[1:] - tail_points[:-1], axis=-1).astype(np.float32)
    tail_arc = np.concatenate([[0.0], np.cumsum(tail_seg_len, dtype=np.float32)], axis=0)
    for idx in range(1, int(tail_points.shape[0])):
        if float(dist_sq[idx]) >= radius_sq and float(dist_sq[idx - 1]) < radius_sq:
            seg_start = tail_points[idx - 1] - center
            seg_end = tail_points[idx] - center
            seg = seg_end - seg_start
            a = float(np.dot(seg, seg))
            b = float(2.0 * np.dot(seg_start, seg))
            c = float(np.dot(seg_start, seg_start) - radius_sq)
            t = 1.0
            if a > 1e-8:
                disc = max(b * b - 4.0 * a * c, 0.0)
                root = math.sqrt(disc)
                candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
                valid_t = [val for val in candidates if -1e-6 <= val <= 1.0 + 1e-6]
                if valid_t:
                    t = float(np.clip(valid_t[0], 0.0, 1.0))
            return float(np.clip(onset + float(tail_arc[idx - 1] + t * tail_seg_len[idx - 1]), 0.0, total_arc))
    return total_arc


def _extract_tube_contour_segments(xx: np.ndarray, yy: np.ndarray, distance_field: np.ndarray, *, level: float) -> List[np.ndarray]:
    tmp_fig, tmp_ax = plt.subplots(figsize=(2, 2), dpi=72)
    try:
        contour = tmp_ax.contour(xx, yy, distance_field, levels=[float(level)])
        segments = [
            np.asarray(seg, dtype=np.float32).reshape(-1, 2)
            for seg in contour.allsegs[0]
            if np.asarray(seg).reshape(-1, 2).shape[0] >= 2
        ]
    finally:
        plt.close(tmp_fig)
    return segments


def _find_rightward_ray_hit(polyline_local: np.ndarray, *, y0: float = 0.0) -> Optional[Tuple[int, float, np.ndarray]]:
    points = np.asarray(polyline_local, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return None
    best: Optional[Tuple[float, int, float, np.ndarray]] = None
    eps = 1e-6
    for idx in range(int(points.shape[0] - 1)):
        p0 = points[idx]
        p1 = points[idx + 1]
        y_start = float(p0[1] - y0)
        y_end = float(p1[1] - y0)
        if abs(y_start) <= eps and abs(y_end) <= eps:
            x_min = min(float(p0[0]), float(p1[0]))
            x_max = max(float(p0[0]), float(p1[0]))
            if x_max < -eps:
                continue
            x_hit = max(0.0, x_min)
            if x_hit > x_max + eps:
                continue
            denom = float(p1[0] - p0[0])
            if abs(denom) <= eps:
                t = 0.0
            else:
                t = float(np.clip((x_hit - float(p0[0])) / denom, 0.0, 1.0))
            point = np.asarray([x_hit, float(y0)], dtype=np.float32)
        else:
            if (y_start > 0.0 and y_end > 0.0) or (y_start < 0.0 and y_end < 0.0):
                continue
            denom = float(y_end - y_start)
            if abs(denom) <= eps:
                continue
            t = float(np.clip((-y_start) / denom, 0.0, 1.0))
            x_hit = float((1.0 - t) * p0[0] + t * p1[0])
            if x_hit < -eps:
                continue
            point = np.asarray([x_hit, float(y0)], dtype=np.float32)
        candidate = (float(point[0]), idx, t, point)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return None
    _, idx, t, point = best
    return idx, float(t), np.asarray(point, dtype=np.float32)


def _orient_contour_forward(polyline_local: np.ndarray, seg_idx: int, t: float) -> np.ndarray:
    points = np.asarray(polyline_local, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 2:
        return points
    closed = points.shape[0] >= 3 and float(np.linalg.norm(points[0] - points[-1])) <= 1.5
    if closed:
        points = np.asarray(points[:-1], dtype=np.float32)
    start = ((1.0 - float(t)) * points[seg_idx] + float(t) * points[(seg_idx + 1) % points.shape[0]]).astype(np.float32)

    def _build(direction: int) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if closed:
            if direction > 0:
                order = [((seg_idx + 1 + step) % points.shape[0]) for step in range(points.shape[0])]
            else:
                order = [((seg_idx - step) % points.shape[0]) for step in range(points.shape[0])]
            tail = points[np.asarray(order, dtype=np.int64)]
        else:
            if direction > 0:
                tail = points[seg_idx + 1 :]
            else:
                tail = points[seg_idx::-1]
        if tail.shape[0] == 0:
            return start[None, :]
        trace = np.vstack([start[None, :], tail]).astype(np.float32)
        keep = [0]
        for idx in range(1, int(trace.shape[0])):
            if float(np.linalg.norm(trace[idx] - trace[keep[-1]])) > 1e-3:
                keep.append(idx)
        return np.asarray(trace[np.asarray(keep, dtype=np.int64)], dtype=np.float32)

    def _score(trace: np.ndarray) -> float:
        if trace.shape[0] < 2:
            return -1e9
        step = min(int(trace.shape[0]), 25)
        y_gain = float(np.max(trace[:step, 1] - trace[0, 1]))
        x_penalty = 0.1 * float(np.max(np.abs(trace[:step, 0] - trace[0, 0])))
        return y_gain - x_penalty

    forward_trace = _build(+1)
    backward_trace = _build(-1)
    return forward_trace if _score(forward_trace) >= _score(backward_trace) else backward_trace


def _build_actual_right_wall_trace_local(
    *,
    xx: np.ndarray,
    yy: np.ndarray,
    distance_field: np.ndarray,
    tube_radius_m: float,
) -> np.ndarray:
    contour_segments = _extract_tube_contour_segments(xx, yy, distance_field, level=float(tube_radius_m))
    if not contour_segments:
        return np.zeros((0, 2), dtype=np.float32)
    chosen: Optional[np.ndarray] = None
    best_x = None
    for seg in contour_segments:
        hit = _find_rightward_ray_hit(seg, y0=0.0)
        if hit is None:
            continue
        seg_idx, t, point = hit
        if best_x is None or float(point[0]) < float(best_x):
            best_x = float(point[0])
            chosen = _orient_contour_forward(seg, seg_idx, t)
    if chosen is not None and chosen.shape[0] >= 2:
        return np.asarray(chosen, dtype=np.float32)

    seed = np.asarray([float(tube_radius_m), 0.0], dtype=np.float32)
    best_seg = None
    best_seg_idx = 0
    best_dist = None
    for seg in contour_segments:
        points = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 2:
            continue
        d = np.linalg.norm(points - seed[None, :], axis=-1)
        idx = int(np.argmin(d))
        dist = float(d[idx])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_seg = points
            best_seg_idx = min(idx, int(points.shape[0] - 2))
    if best_seg is None:
        return np.zeros((0, 2), dtype=np.float32)
    return _orient_contour_forward(best_seg, best_seg_idx, 0.0)


def _extract_scene_render_context(raw_scenario: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    sdc_id = str(row["sdc_id"])
    current_idx = int(row["current_time_index"])
    track_state = dict(raw_scenario["tracks"][str(sdc_id)]["state"])
    position = np.asarray(track_state.get("position", []), dtype=np.float64)
    heading = np.asarray(track_state.get("heading", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(track_state.get("valid", []), dtype=bool).reshape(-1)
    idx = int(np.clip(current_idx, 0, max(0, position.shape[0] - 1)))
    while idx > 0 and valid.shape[0] > idx and not bool(valid[idx]):
        idx -= 1
    current_xy = _finite_xy_rows(position[idx])[0]
    current_heading = float(heading[idx]) if heading.shape[0] > idx and np.isfinite(heading[idx]) else 0.0
    gt_past_xy = _finite_xy_rows(
        position[max(0, idx - int(PAST_STEPS)) : idx + 1][valid[max(0, idx - int(PAST_STEPS)) : idx + 1]]
    )
    return {
        "current_time_index": int(idx),
        "current_xy": np.asarray(current_xy, dtype=np.float64),
        "current_heading": float(current_heading),
        "gt_past_xy": np.asarray(gt_past_xy, dtype=np.float64),
        "map_context": _select_map_context(raw_scenario, center_xy=current_xy, radius_m=108.0),
        "traffic_lights": _select_traffic_lights(raw_scenario, center_xy=current_xy, radius_m=108.0, time_index=idx),
        "nearby_agents": _select_nearby_agents(raw_scenario, sdc_id=sdc_id, center_xy=current_xy, current_idx=idx, radius_m=108.0),
    }


def _draw_scene_context(
    *,
    ax,
    render_context: Mapping[str, Any],
    info_box_text: str = "",
) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    gt_past_xy = np.asarray(render_context["gt_past_xy"], dtype=np.float64)
    map_context = render_context["map_context"]
    traffic_lights = render_context["traffic_lights"]
    nearby_agents = render_context["nearby_agents"]

    ax.set_facecolor("#f8fafc")
    for feature in map_context.get("crosswalks", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color=CROSSWALK_FACE, alpha=0.35, zorder=1)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=ROAD_COLOR, linewidth=2.6, alpha=0.92, zorder=2)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=LANE_COLOR, linewidth=1.0, alpha=0.28, zorder=3)

    for agent in nearby_agents:
        past_xy = _world_to_sdc_up_frame(np.asarray(agent["past_xy"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if past_xy.shape[0] >= 2:
            ax.plot(past_xy[:, 0], past_xy[:, 1], color=AGENT_COLOR, linewidth=1.0, alpha=0.55, zorder=4)
        current_agent_xy = _world_to_sdc_up_frame(np.asarray([agent["current_xy"]], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if current_agent_xy.shape[0] > 0:
            ax.scatter([current_agent_xy[0, 0]], [current_agent_xy[0, 1]], c=AGENT_COLOR, s=14, alpha=0.85, zorder=4.2)

    for light in traffic_lights:
        stop_xy = _world_to_sdc_up_frame(np.asarray([light["stop_point_xy_world"]], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if stop_xy.shape[0] == 0:
            continue
        state = str(light.get("state") or "unknown")
        color = "#ef4444" if "STOP" in state or "RED" in state else ("#22c55e" if "GO" in state or "GREEN" in state else "#eab308")
        ax.scatter([stop_xy[0, 0]], [stop_xy[0, 1]], c=color, marker="s", s=70, edgecolors="black", linewidths=0.7, zorder=4.5)

    gt_past_local = _world_to_sdc_up_frame(gt_past_xy, center_xy=center_xy, heading_rad=current_heading)
    if gt_past_local.shape[0] >= 2:
        ax.plot(gt_past_local[:, 0], gt_past_local[:, 1], color="#111827", linewidth=2.6, alpha=0.95, zorder=5)
    ax.scatter([0.0], [0.0], c=SDC_DOT_COLOR, s=56, edgecolors="white", linewidths=0.8, zorder=6)

    if info_box_text:
        ax.text(
            0.02,
            0.975,
            str(info_box_text),
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
            zorder=15,
        )

    half_extent = 48.0
    vertical_span = 2.0 * half_extent
    y_min = -0.10 * vertical_span
    y_max = y_min + vertical_span
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_single_example(
    *,
    ax,
    fig,
    row: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
    radius_from_divergence_m: float,
    tube_radius_m: float,
    grid_step_m: float,
    jump_threshold_m: float,
) -> Dict[str, Any]:
    render_context = _extract_scene_render_context(raw_scenario, row)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float32)
    current_heading = float(render_context["current_heading"])
    path_world = np.asarray(selected_raw_route_world(raw_scenario, row), dtype=np.float32)
    path_length_m = float(polyline_length_m(path_world))
    divergence_onset_m = _finite_nonnegative_min(row.get("candidate_family_divergence_onsets_m", []))
    segments_world = [
        np.asarray(seg, dtype=np.float32)
        for seg in split_polyline_on_discontinuities(path_world, jump_threshold_m=float(jump_threshold_m))
        if np.asarray(seg).shape[0] >= 2
    ]

    xx, yy, distance_field = segment_distance_field_in_sdc_frame(
        polyline_world_xy=path_world,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
        grid_step_m=float(grid_step_m),
    )
    divergence_world = _sample_point_along_polyline(path_world, divergence_onset_m)
    divergence_local = world_to_sdc_up_frame(
        divergence_world[None, :],
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
    )[0]
    progress_path_local = _build_actual_right_wall_trace_local(
        xx=xx,
        yy=yy,
        distance_field=distance_field,
        tube_radius_m=float(tube_radius_m),
    )
    if progress_path_local.shape[0] >= 2:
        progress_path_world = sdc_up_to_world_frame(
            progress_path_local,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
    else:
        progress_path_world = np.zeros((0, 2), dtype=np.float32)
    progress_path_length_m = float(polyline_length_m(progress_path_world))
    progress_cap_arc_m = _first_exit_arc_from_circle_center(
        progress_path_local,
        divergence_local,
        float(radius_from_divergence_m),
        start_arc_m=0.0,
    )
    cap_local = _sample_point_along_polyline(progress_path_local, progress_cap_arc_m)
    cap_world = sdc_up_to_world_frame(
        cap_local[None, :],
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
    )[0]
    reward_segment_local = _resample_polyline_interval(
        progress_path_local,
        0.0,
        progress_cap_arc_m,
        spacing_m=0.5,
    )
    reward_segment_world = sdc_up_to_world_frame(
        reward_segment_local,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
    )
    post_cap_local = _resample_polyline_interval(
        progress_path_local,
        progress_cap_arc_m,
        float(polyline_length_m(progress_path_local)),
        spacing_m=0.5,
    )
    post_cap_world = sdc_up_to_world_frame(
        post_cap_local,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
    )

    _draw_scene_context(
        ax=ax,
        render_context=render_context,
        info_box_text=(
            f"scene={row['scenario_id']}\n"
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}\n"
            f"divergence={divergence_onset_m:.1f}m\n"
            f"progress_cap={progress_cap_arc_m:.1f}m\n"
            f"radius_from_divergence={float(radius_from_divergence_m):.1f}m"
        ),
    )

    ax.contour(
        xx,
        yy,
        distance_field,
        levels=[float(tube_radius_m)],
        colors=["#f59e0b"],
        linewidths=1.5,
        linestyles=["--"],
        zorder=8.0,
        alpha=0.85,
    )

    for seg_idx, seg_world in enumerate(segments_world):
        seg_local = world_to_sdc_up_frame(
            seg_world,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        if seg_local.shape[0] < 2:
            continue
        ax.plot(
            seg_local[:, 0],
            seg_local[:, 1],
            color="#2563eb",
            linewidth=3.8,
            alpha=0.75,
            zorder=9.0,
            solid_capstyle="round",
        )
        if seg_idx > 0:
            ax.scatter(
                [seg_local[0, 0]],
                [seg_local[0, 1]],
                c="#111827",
                s=26,
                marker="x",
                linewidths=1.0,
                zorder=11.0,
            )

    if progress_path_local.shape[0] >= 2:
        ax.plot(
            progress_path_local[:, 0],
            progress_path_local[:, 1],
            color="#be185d",
            linewidth=2.8,
            alpha=0.95,
            zorder=9.8,
            linestyle="-.",
            solid_capstyle="round",
        )

    if reward_segment_local.shape[0] >= 2:
        ax.plot(
            reward_segment_local[:, 0],
            reward_segment_local[:, 1],
            color="#dc2626",
            linewidth=5.0,
            alpha=0.96,
            zorder=10.5,
            solid_capstyle="round",
        )

    if post_cap_local.shape[0] >= 2:
        ax.plot(
            post_cap_local[:, 0],
            post_cap_local[:, 1],
            color="#9ca3af",
            linewidth=3.2,
            alpha=0.75,
            zorder=9.4,
            solid_capstyle="round",
        )

    circle = plt.Circle(
        tuple(divergence_local.tolist()),
        radius=float(radius_from_divergence_m),
        edgecolor="#ef4444",
        facecolor="none",
        linestyle=":",
        linewidth=1.6,
        alpha=0.8,
        zorder=10.2,
    )
    ax.add_patch(circle)
    ax.scatter(
        [divergence_local[0]],
        [divergence_local[1]],
        s=90,
        c="#10b981",
        edgecolors="white",
        linewidths=0.8,
        marker="o",
        zorder=12.0,
    )
    ax.scatter(
        [cap_local[0]],
        [cap_local[1]],
        s=120,
        c="#ef4444",
        edgecolors="white",
        linewidths=0.8,
        marker="*",
        zorder=12.2,
    )
    ax.text(
        divergence_local[0] + 1.2,
        divergence_local[1] + 1.2,
        f"diverge\n{divergence_onset_m:.1f}m",
        color="#065f46",
        fontsize=8.5,
        zorder=12.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#10b981", alpha=0.82),
    )
    ax.text(
        cap_local[0] + 1.2,
        cap_local[1] - 2.3,
        f"cap\n{progress_cap_arc_m:.1f}m",
        color="#7f1d1d",
        fontsize=8.5,
        zorder=12.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ef4444", alpha=0.84),
    )

    return {
        "scenario_id": str(row.get("scenario_id") or ""),
        "selected_slot_id": str(row.get("selected_slot_id") or ""),
        "requested_semantic_label": str(row.get("requested_semantic_label") or ""),
        "selected_path_id": row.get("selected_path_id"),
        "divergence_onset_m": float(divergence_onset_m),
        "progress_cap_arc_m": float(progress_cap_arc_m),
        "radius_from_divergence_m": float(radius_from_divergence_m),
        "path_length_m": float(path_length_m),
        "progress_path_type": "actual_right_wall_contour_trace",
        "progress_path_length_m": float(progress_path_length_m),
        "cap_fraction_of_progress_path": float(progress_cap_arc_m / max(progress_path_length_m, 1e-6)),
        "cap_fraction_of_path": float(progress_cap_arc_m / max(path_length_m, 1e-6)),
        "tube_radius_m": float(tube_radius_m),
        "jump_threshold_m": float(jump_threshold_m),
        "num_segments": int(len(segments_world)),
    }


def main() -> int:
    args = parse_args()
    if not list(args.example):
        raise ValueError("Provide at least one --example scenario_id:slot_id")

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    example_specs = [_parse_example_spec(spec) for spec in list(args.example)]
    cols = min(2, max(1, len(example_specs)))
    grid_rows = int(math.ceil(float(len(example_specs)) / float(cols)))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(8.3 * cols, 8.3 * grid_rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)

    manifest: List[Dict[str, Any]] = []
    for ax, (scenario_id, slot_id) in zip(axes, example_specs):
        row_index = _select_row(rows, row_index=-1, scenario_id=scenario_id, slot_id=slot_id)
        row = dict(rows[row_index])
        raw_scenario = load_raw_scenario_from_row(row)
        meta = _plot_single_example(
            ax=ax,
            fig=fig,
            row=row,
            raw_scenario=raw_scenario,
            radius_from_divergence_m=float(args.radius_from_divergence_m),
            tube_radius_m=float(args.tube_radius_m),
            grid_step_m=float(args.grid_step_m),
            jump_threshold_m=float(args.jump_threshold_m),
        )
        slug = f"{scenario_id}__{slot_id}".replace("/", "_")
        single_path = outdir / f"{slug}_progress_cap.png"
        single_fig = plt.figure(figsize=(8.2, 8.2), dpi=180)
        single_ax = single_fig.add_axes([0.02, 0.02, 0.96, 0.96])
        _plot_single_example(
            ax=single_ax,
            fig=single_fig,
            row=row,
            raw_scenario=raw_scenario,
            radius_from_divergence_m=float(args.radius_from_divergence_m),
            tube_radius_m=float(args.tube_radius_m),
            grid_step_m=float(args.grid_step_m),
            jump_threshold_m=float(args.jump_threshold_m),
        )
        single_fig.savefig(single_path, bbox_inches="tight")
        plt.close(single_fig)
        meta["png"] = str(single_path)
        meta["row_index"] = int(row_index)
        manifest.append(meta)

    for ax in axes[len(example_specs) :]:
        ax.axis("off")

    fig.suptitle(
        f"Scene-context overlays for divergence-radius progress reward ({float(args.radius_from_divergence_m):.0f}m circle)",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])
    grid_path = outdir / "progress_cap_examples_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)

    manifest_path = outdir / "progress_cap_examples_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "radius_from_divergence_m": float(args.radius_from_divergence_m),
                "tube_radius_m": float(args.tube_radius_m),
                "grid_step_m": float(args.grid_step_m),
                "jump_threshold_m": float(args.jump_threshold_m),
                "grid_png": str(grid_path),
                "examples": manifest,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "progress_cap_examples_manifest_json": str(manifest_path),
                "progress_cap_examples_grid_png": str(grid_path),
                "examples": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
