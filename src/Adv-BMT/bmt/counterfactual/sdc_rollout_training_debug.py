from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, List, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PLOT_RADIUS_M = 60.0
DEFAULT_VERTICAL_FRACTION = 0.63


def _iter_map_feature_polylines(raw_scenario: Mapping[str, Any]) -> Iterable[np.ndarray]:
    for feature in dict(raw_scenario.get("map_features", {}) or {}).values():
        payload = dict(feature or {})
        for key in ("polyline", "polygon"):
            value = payload.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] >= 2 and arr.shape[0] > 1:
                yield np.asarray(arr[:, :2], dtype=np.float32)
                break


def normalize_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [item.strip().strip("'\"") for item in text.split(",") if item.strip().strip("'\"")]
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def world_to_sdc_up_frame(
    points_world_xy: Any,
    *,
    center_xy_world: Sequence[float],
    heading_world_rad: float,
) -> np.ndarray:
    points = np.asarray(points_world_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    center = np.asarray(center_xy_world, dtype=np.float32).reshape(2)
    shift = points - center[None, :]
    rot = float(heading_world_rad) - (math.pi / 2.0)
    c = math.cos(rot)
    s = math.sin(rot)
    local_x = c * shift[:, 0] + s * shift[:, 1]
    local_y = -s * shift[:, 0] + c * shift[:, 1]
    return np.stack([local_x, local_y], axis=-1).astype(np.float32)


def split_polyline_by_segment_mask(
    polyline_world_xy: Any,
    *,
    point_mask: Any | None = None,
    segment_mask: Any | None = None,
) -> List[np.ndarray]:
    points = np.asarray(polyline_world_xy, dtype=np.float32).reshape(-1, 2)
    if point_mask is not None:
        point_mask_arr = np.asarray(point_mask, dtype=np.float32).reshape(-1) > 0.5
        if point_mask_arr.shape[0] > 0:
            points = points[: point_mask_arr.shape[0]][point_mask_arr[: points.shape[0]]]
    if points.shape[0] < 2:
        return []
    if segment_mask is None:
        return [points.astype(np.float32)]

    seg_mask = np.asarray(segment_mask, dtype=np.float32).reshape(-1) > 0.5
    num_seg = min(int(seg_mask.shape[0]), int(points.shape[0] - 1))
    if num_seg <= 0:
        return [points.astype(np.float32)]
    seg_mask = seg_mask[:num_seg]

    segments: List[np.ndarray] = []
    run_start = None
    for seg_idx, is_valid in enumerate(seg_mask.tolist()):
        if is_valid and run_start is None:
            run_start = int(seg_idx)
        if (not is_valid) and run_start is not None:
            segment = points[run_start : seg_idx + 1]
            if segment.shape[0] >= 2:
                segments.append(segment.astype(np.float32))
            run_start = None
    if run_start is not None:
        segment = points[run_start : num_seg + 1]
        if segment.shape[0] >= 2:
            segments.append(segment.astype(np.float32))
    if not segments:
        return [points.astype(np.float32)]
    return segments


def polyline_segment_distance_to_points(
    points_xy: Any,
    *,
    path_segments_world: Sequence[np.ndarray],
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if not path_segments_world:
        return np.full((points.shape[0],), np.inf, dtype=np.float32)

    best = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for segment in path_segments_world:
        seg = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
        if seg.shape[0] < 2:
            continue
        seg_start = seg[:-1]
        seg_end = seg[1:]
        seg_vec = seg_end - seg_start
        seg_len_sq = np.sum(seg_vec * seg_vec, axis=-1).clip(min=1e-6)
        rel = points[:, None, :] - seg_start[None, :, :]
        t = np.sum(rel * seg_vec[None, :, :], axis=-1) / seg_len_sq[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        dist = np.min(np.linalg.norm(points[:, None, :] - closest, axis=-1), axis=-1)
        best = np.minimum(best, dist.astype(np.float32))
    return best.astype(np.float32)


def _stack_plot_points(
    *,
    current_xy: np.ndarray,
    path_segments_xy: Sequence[np.ndarray],
    trajectories_xy: Sequence[np.ndarray],
) -> np.ndarray:
    pieces: List[np.ndarray] = [np.asarray(current_xy, dtype=np.float32).reshape(1, 2)]
    pieces.extend(
        np.asarray(segment, dtype=np.float32).reshape(-1, 2)
        for segment in path_segments_xy
        if np.asarray(segment, dtype=np.float32).reshape(-1, 2).shape[0] > 0
    )
    pieces.extend(
        np.asarray(traj, dtype=np.float32).reshape(-1, 2)
        for traj in trajectories_xy
        if np.asarray(traj, dtype=np.float32).reshape(-1, 2).shape[0] > 0
    )
    return np.concatenate(pieces, axis=0).astype(np.float32)


def compute_auto_plot_limits(
    *,
    current_xy: np.ndarray,
    path_segments_xy: Sequence[np.ndarray],
    trajectories_xy: Sequence[np.ndarray],
    tube_radius_m: float,
    grid_step_m: float,
    min_half_extent_m: float = 12.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = _stack_plot_points(
        current_xy=np.asarray(current_xy, dtype=np.float32).reshape(2),
        path_segments_xy=path_segments_xy,
        trajectories_xy=trajectories_xy,
    )
    margin = max(float(tube_radius_m) + 4.0 * float(grid_step_m), 4.0)
    x_min = float(points[:, 0].min()) - margin
    x_max = float(points[:, 0].max()) + margin
    y_min = float(points[:, 1].min()) - margin
    y_max = float(points[:, 1].max()) + margin

    half_extent = max(
        float(min_half_extent_m),
        0.5 * max(x_max - x_min, y_max - y_min),
    )
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    return (
        (x_center - half_extent, x_center + half_extent),
        (y_center - half_extent, y_center + half_extent),
    )


def segment_distance_field_in_frame(
    *,
    path_segments_xy: Sequence[np.ndarray],
    grid_step_m: float,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    plot_radius_m: float = DEFAULT_PLOT_RADIUS_M,
    vertical_fraction: float = DEFAULT_VERTICAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_limits is None or y_limits is None:
        half_extent = float(plot_radius_m)
        vertical_span = 2.0 * half_extent
        x_min = -half_extent
        x_max = half_extent
        y_min = -float(vertical_fraction) * vertical_span
        y_max = y_min + vertical_span
    else:
        x_min, x_max = float(x_limits[0]), float(x_limits[1])
        y_min, y_max = float(y_limits[0]), float(y_limits[1])
    xs = np.arange(x_min, x_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    frame_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    dist = polyline_segment_distance_to_points(frame_xy, path_segments_world=path_segments_xy).reshape(xx.shape)
    return xx, yy, dist.astype(np.float32)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _load_scene_context_in_reward_frame(
    *,
    scenario_pkl: str | None,
    current_time_index: int | None,
    sdc_id: str | None,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    margin_m: float = 20.0,
) -> dict[str, Any]:
    if not scenario_pkl:
        return {"map_segments": [], "agent_points": np.zeros((0, 2), dtype=np.float32)}
    try:
        from bmt.counterfactual.normalize import load_raw_scenario
        from bmt.counterfactual.sdc_semantic_control import extract_model_frame, world_xy_to_model_frame
    except Exception:
        return {"map_segments": [], "agent_points": np.zeros((0, 2), dtype=np.float32)}

    raw_scenario = load_raw_scenario(Path(str(scenario_pkl)).expanduser())
    map_center, map_heading = extract_model_frame(raw_scenario)
    x_min, x_max = float(x_limits[0]) - float(margin_m), float(x_limits[1]) + float(margin_m)
    y_min, y_max = float(y_limits[0]) - float(margin_m), float(y_limits[1]) + float(margin_m)

    map_segments: List[np.ndarray] = []
    for polyline_world in _iter_map_feature_polylines(raw_scenario):
        polyline_frame = world_xy_to_model_frame(polyline_world, map_center=map_center, map_heading=map_heading)
        if polyline_frame.shape[0] < 2:
            continue
        seg_x_min = float(np.min(polyline_frame[:, 0]))
        seg_x_max = float(np.max(polyline_frame[:, 0]))
        seg_y_min = float(np.min(polyline_frame[:, 1]))
        seg_y_max = float(np.max(polyline_frame[:, 1]))
        if seg_x_max < x_min or seg_x_min > x_max or seg_y_max < y_min or seg_y_min > y_max:
            continue
        map_segments.append(np.asarray(polyline_frame, dtype=np.float32))

    agent_points: List[np.ndarray] = []
    if current_time_index is not None:
        time_idx = int(max(int(current_time_index), 0))
        for track_id, track_payload in dict(raw_scenario.get("tracks", {}) or {}).items():
            if sdc_id and str(track_id) == str(sdc_id):
                continue
            state = dict(dict(track_payload).get("state") or {})
            position_world = np.asarray(state.get("position", []), dtype=np.float32)
            valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
            if position_world.ndim != 2 or position_world.shape[1] < 2 or position_world.shape[0] == 0:
                continue
            idx = min(time_idx, position_world.shape[0] - 1)
            if valid.shape[0] > idx and not bool(valid[idx]):
                continue
            point_frame = world_xy_to_model_frame(
                position_world[idx : idx + 1, :2],
                map_center=map_center,
                map_heading=map_heading,
            )
            if point_frame.shape[0] == 0:
                continue
            x_val, y_val = float(point_frame[0, 0]), float(point_frame[0, 1])
            if x_val < x_min or x_val > x_max or y_val < y_min or y_val > y_max:
                continue
            agent_points.append(np.asarray(point_frame[0], dtype=np.float32))

    return {
        "map_segments": map_segments,
        "agent_points": (
            np.stack(agent_points, axis=0).astype(np.float32)
            if agent_points
            else np.zeros((0, 2), dtype=np.float32)
        ),
    }


def write_rollout_tube_training_debug(
    *,
    outdir: str | Path,
    scenario_id: str,
    slot_id: str,
    requested_semantic_label: str,
    global_step: int,
    current_xy_world: Any,
    current_heading_world: float,
    path_world: Any,
    point_mask: Any,
    segment_mask: Any,
    trajectories_world: Any,
    reward_t: Any,
    return_to_go_t: Any,
    advantage_t: Any,
    action_token_t: Any,
    action_logprob_t: Any,
    tube_distance_t: Any,
    valid_mask_t: Any,
    tube_radius_m: float,
    inside_reward: float,
    outside_scale: float,
    discount: float,
    grid_step_m: float = 0.35,
    scenario_pkl: str | None = None,
    current_time_index: int | None = None,
    sdc_id: str | None = None,
    traffic_speed_penalty_t: Any | None = None,
    traffic_speed_reference_t: Any | None = None,
    traffic_speed_neighbor_mean_t: Any | None = None,
    traffic_speed_neighbor_median_t: Any | None = None,
    traffic_speed_floor_threshold_t: Any | None = None,
    traffic_speed_cap_threshold_t: Any | None = None,
    traffic_speed_cap_penalty_t: Any | None = None,
    traffic_speed_sdc_t: Any | None = None,
    traffic_speed_neighbor_count_t: Any | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    path_world_arr = np.asarray(path_world, dtype=np.float32).reshape(-1, 2)
    point_mask_arr = np.asarray(point_mask, dtype=np.float32).reshape(-1)
    segment_mask_arr = np.asarray(segment_mask, dtype=np.float32).reshape(-1)
    trajectories_world_arr = np.asarray(trajectories_world, dtype=np.float32)
    reward_arr = np.asarray(reward_t, dtype=np.float32)
    rtg_arr = np.asarray(return_to_go_t, dtype=np.float32)
    advantage_arr = np.asarray(advantage_t, dtype=np.float32)
    action_token_arr = np.asarray(action_token_t, dtype=np.int64)
    action_logprob_arr = np.asarray(action_logprob_t, dtype=np.float32)
    tube_distance_arr = np.asarray(tube_distance_t, dtype=np.float32)
    valid_mask_arr = np.asarray(valid_mask_t, dtype=bool)
    traffic_speed_penalty_arr = None if traffic_speed_penalty_t is None else np.asarray(traffic_speed_penalty_t, dtype=np.float32)
    traffic_speed_reference_arr = None if traffic_speed_reference_t is None else np.asarray(traffic_speed_reference_t, dtype=np.float32)
    traffic_speed_neighbor_mean_arr = (
        None if traffic_speed_neighbor_mean_t is None else np.asarray(traffic_speed_neighbor_mean_t, dtype=np.float32)
    )
    traffic_speed_neighbor_median_arr = (
        None if traffic_speed_neighbor_median_t is None else np.asarray(traffic_speed_neighbor_median_t, dtype=np.float32)
    )
    traffic_speed_floor_threshold_arr = (
        None if traffic_speed_floor_threshold_t is None else np.asarray(traffic_speed_floor_threshold_t, dtype=np.float32)
    )
    traffic_speed_cap_threshold_arr = (
        None if traffic_speed_cap_threshold_t is None else np.asarray(traffic_speed_cap_threshold_t, dtype=np.float32)
    )
    traffic_speed_cap_penalty_arr = (
        None if traffic_speed_cap_penalty_t is None else np.asarray(traffic_speed_cap_penalty_t, dtype=np.float32)
    )
    traffic_speed_sdc_arr = None if traffic_speed_sdc_t is None else np.asarray(traffic_speed_sdc_t, dtype=np.float32)
    traffic_speed_neighbor_count_arr = (
        None if traffic_speed_neighbor_count_t is None else np.asarray(traffic_speed_neighbor_count_t, dtype=np.float32)
    )
    current_xy = np.asarray(current_xy_world, dtype=np.float32).reshape(2)
    current_heading = float(current_heading_world)

    # Despite the legacy argument names, the training tube reward is computed in the
    # same frame as decoder positions and selected raw paths. Plot directly in that
    # frame so the contour matches the reward geometry exactly.
    path_segments_frame = split_polyline_by_segment_mask(
        path_world_arr,
        point_mask=point_mask_arr,
        segment_mask=segment_mask_arr,
    )
    trajectory_frame_list = [
        np.asarray(traj, dtype=np.float32).reshape(-1, 2)
        for traj in trajectories_world_arr
    ]

    total_return = reward_arr.sum(axis=-1).astype(np.float32)
    scalar_advantage = np.zeros_like(total_return, dtype=np.float32)
    if total_return.size > 1:
        denom = max(float(total_return.std()), 1e-6)
        scalar_advantage = ((total_return - float(total_return.mean())) / denom).astype(np.float32)
    valid_any = bool(np.any(valid_mask_arr))

    x_limits, y_limits = compute_auto_plot_limits(
        current_xy=current_xy,
        path_segments_xy=path_segments_frame,
        trajectories_xy=trajectory_frame_list,
        tube_radius_m=float(tube_radius_m),
        grid_step_m=float(grid_step_m),
    )
    xx, yy, dist_field = segment_distance_field_in_frame(
        path_segments_xy=path_segments_frame,
        grid_step_m=float(grid_step_m),
        x_limits=x_limits,
        y_limits=y_limits,
    )
    scene_context = _load_scene_context_in_reward_frame(
        scenario_pkl=scenario_pkl,
        current_time_index=current_time_index,
        sdc_id=sdc_id,
        x_limits=x_limits,
        y_limits=y_limits,
    )

    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(vmin=float(np.min(scalar_advantage)), vmax=float(np.max(scalar_advantage)))
    if float(np.max(scalar_advantage) - np.min(scalar_advantage)) < 1e-6:
        norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    rollout_colors = [cmap(norm(float(value))) for value in scalar_advantage.tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.0), dpi=180)
    inside = np.ma.masked_where(dist_field > float(tube_radius_m), dist_field)
    axes[0, 0].contourf(
        xx,
        yy,
        inside,
        levels=np.linspace(0.0, float(tube_radius_m), num=8),
        cmap="Blues_r",
        alpha=0.25,
        zorder=1.0,
        antialiased=True,
    )
    axes[0, 0].contour(
        xx,
        yy,
        dist_field,
        levels=[float(tube_radius_m)],
        colors=["#f59e0b"],
        linewidths=1.8,
        linestyles=["--"],
        zorder=2.0,
    )
    for map_segment in scene_context["map_segments"]:
        axes[0, 0].plot(
            map_segment[:, 0],
            map_segment[:, 1],
            color="#cbd5e1",
            linewidth=0.8,
            alpha=0.75,
            zorder=2.2,
        )
    agent_points = np.asarray(scene_context["agent_points"], dtype=np.float32).reshape(-1, 2)
    if agent_points.shape[0] > 0:
        axes[0, 0].scatter(
            agent_points[:, 0],
            agent_points[:, 1],
            c="#9ca3af",
            s=10,
            alpha=0.55,
            zorder=2.4,
        )
    for seg_idx, seg_frame in enumerate(path_segments_frame):
        if seg_frame.shape[0] < 2:
            continue
        axes[0, 0].plot(seg_frame[:, 0], seg_frame[:, 1], color="#2563eb", linewidth=4.0, alpha=0.96, zorder=3.0)
        if seg_idx > 0:
            axes[0, 0].scatter(
                [seg_frame[0, 0]],
                [seg_frame[0, 1]],
                c="#111827",
                s=28,
                marker="x",
                linewidths=1.2,
                zorder=3.5,
            )
    for rollout_idx, (traj_frame, color) in enumerate(zip(trajectory_frame_list, rollout_colors)):
        if traj_frame.shape[0] < 2:
            continue
        axes[0, 0].plot(traj_frame[:, 0], traj_frame[:, 1], color=color, linewidth=2.2, alpha=0.92, zorder=4.0)
        axes[0, 0].scatter(
            [traj_frame[-1, 0]],
            [traj_frame[-1, 1]],
            c=[color],
            s=32,
            edgecolors="white",
            linewidths=0.6,
            zorder=4.2,
        )
        axes[0, 0].text(
            float(traj_frame[-1, 0]) + 0.5,
            float(traj_frame[-1, 1]) + 0.5,
            f"{rollout_idx}",
            fontsize=8,
            color="#111827",
            zorder=4.5,
        )
    axes[0, 0].scatter([0.0], [0.0], c="#111827", s=30, marker="o", zorder=5.0)
    axes[0, 0].set_title("Actual Training Rollouts + Tube", fontsize=12)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].grid(alpha=0.18)
    axes[0, 0].set_xlim(*x_limits)
    axes[0, 0].set_ylim(*y_limits)
    info_box = (
        f"scene={scenario_id}\n"
        f"slot={slot_id}\n"
        f"label={requested_semantic_label}\n"
        f"step={int(global_step)}\n"
        f"group={int(trajectories_world_arr.shape[0])}\n"
        f"tube={float(tube_radius_m):.1f}m"
    )
    axes[0, 0].text(
        0.02,
        0.98,
        info_box,
        transform=axes[0, 0].transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.82, edgecolor="#d1d5db"),
        zorder=6.0,
    )

    for rollout_idx, color in enumerate(rollout_colors):
        step_idx = np.arange(reward_arr.shape[-1], dtype=np.int64)
        axes[0, 1].plot(step_idx, reward_arr[rollout_idx], color=color, linewidth=1.8, alpha=0.94)
    axes[0, 1].set_title("Per-Step Reward", fontsize=12)
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("reward")
    axes[0, 1].grid(alpha=0.25)

    for rollout_idx, color in enumerate(rollout_colors):
        step_idx = np.arange(rtg_arr.shape[-1], dtype=np.int64)
        axes[1, 0].plot(step_idx, rtg_arr[rollout_idx], color=color, linewidth=1.8, alpha=0.94)
    axes[1, 0].set_title("Return-to-Go", fontsize=12)
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("RTG")
    axes[1, 0].grid(alpha=0.25)

    for rollout_idx, color in enumerate(rollout_colors):
        step_idx = np.arange(advantage_arr.shape[-1], dtype=np.int64)
        axes[1, 1].plot(step_idx, advantage_arr[rollout_idx], color=color, linewidth=1.8, alpha=0.94)
    axes[1, 1].axhline(0.0, color="#111827", linewidth=1.0, alpha=0.5)
    axes[1, 1].set_title("Stepwise Group-Normalized Advantage", fontsize=12)
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("advantage")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        "Exact training rollout-group debug\n"
        f"inside_reward={float(inside_reward):.2f} outside_scale={float(outside_scale):.2f} gamma={float(discount):.2f}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.965])
    analysis_png = outdir / "group_rollout_advantage_analysis.png"
    fig.savefig(analysis_png, bbox_inches="tight")
    plt.close(fig)

    rollouts = []
    inside_mask = tube_distance_arr <= float(tube_radius_m)
    for rollout_idx in range(int(trajectories_world_arr.shape[0])):
        first_exit = np.flatnonzero(~inside_mask[rollout_idx] & valid_mask_arr[rollout_idx])
        rollouts.append(
            {
                "rollout_id": int(rollout_idx),
                "trajectory_world_xy": trajectories_world_arr[rollout_idx].tolist(),
                "reward_t": reward_arr[rollout_idx].tolist(),
                "return_to_go_t": rtg_arr[rollout_idx].tolist(),
                "step_group_advantage_t": advantage_arr[rollout_idx].tolist(),
                "action_token_t": action_token_arr[rollout_idx].tolist(),
                "action_logprob_t": action_logprob_arr[rollout_idx].tolist(),
                "tube_distance_t": tube_distance_arr[rollout_idx].tolist(),
                "valid_mask_t": valid_mask_arr[rollout_idx].astype(bool).tolist(),
                "inside_valid_tube_t": (inside_mask[rollout_idx] & valid_mask_arr[rollout_idx]).astype(bool).tolist(),
                "inside_fraction": float(np.mean((inside_mask[rollout_idx] & valid_mask_arr[rollout_idx]).astype(np.float32))),
                "total_return": float(total_return[rollout_idx]),
                "scalar_group_advantage": float(scalar_advantage[rollout_idx]),
                "first_exit_step": (None if first_exit.size == 0 else int(first_exit[0])),
                "traffic_speed_penalty_t": (
                    None if traffic_speed_penalty_arr is None else traffic_speed_penalty_arr[rollout_idx].tolist()
                ),
                "traffic_speed_reference_t": (
                    None if traffic_speed_reference_arr is None else traffic_speed_reference_arr[rollout_idx].tolist()
                ),
                "traffic_speed_neighbor_mean_t": (
                    None
                    if traffic_speed_neighbor_mean_arr is None
                    else traffic_speed_neighbor_mean_arr[rollout_idx].tolist()
                ),
                "traffic_speed_neighbor_median_t": (
                    None
                    if traffic_speed_neighbor_median_arr is None
                    else traffic_speed_neighbor_median_arr[rollout_idx].tolist()
                ),
                "traffic_speed_floor_threshold_t": (
                    None
                    if traffic_speed_floor_threshold_arr is None
                    else traffic_speed_floor_threshold_arr[rollout_idx].tolist()
                ),
                "traffic_speed_cap_threshold_t": (
                    None
                    if traffic_speed_cap_threshold_arr is None
                    else traffic_speed_cap_threshold_arr[rollout_idx].tolist()
                ),
                "traffic_speed_cap_penalty_t": (
                    None
                    if traffic_speed_cap_penalty_arr is None
                    else traffic_speed_cap_penalty_arr[rollout_idx].tolist()
                ),
                "traffic_speed_sdc_t": (
                    None if traffic_speed_sdc_arr is None else traffic_speed_sdc_arr[rollout_idx].tolist()
                ),
                "traffic_speed_neighbor_count_t": (
                    None if traffic_speed_neighbor_count_arr is None else traffic_speed_neighbor_count_arr[rollout_idx].tolist()
                ),
            }
        )

    summary = {
        "scenario_id": str(scenario_id),
        "selected_slot_id": str(slot_id),
        "requested_semantic_label": str(requested_semantic_label),
        "global_step": int(global_step),
        "tube_radius_m": float(tube_radius_m),
        "inside_reward": float(inside_reward),
        "outside_scale": float(outside_scale),
        "discount": float(discount),
        "grid_step_m": float(grid_step_m),
        "analysis_png": str(analysis_png),
        "current_xy_world": current_xy.tolist(),
        "current_heading_world": float(current_heading),
        "num_path_segments": int(len(path_segments_frame)),
        "path_world_xy": path_world_arr.tolist(),
        "path_point_mask": point_mask_arr.tolist(),
        "path_segment_mask": segment_mask_arr.tolist(),
        "scenario_pkl": str(scenario_pkl or ""),
        "current_time_index": (None if current_time_index is None else int(current_time_index)),
        "sdc_id": (None if sdc_id is None else str(sdc_id)),
        "num_context_map_segments": int(len(scene_context["map_segments"])),
        "num_context_agents": int(agent_points.shape[0]),
        "plot_frame": "training_reward_frame",
        "plot_xlim": [float(x_limits[0]), float(x_limits[1])],
        "plot_ylim": [float(y_limits[0]), float(y_limits[1])],
        "path_extent_xy": {
            "x_min": float(path_world_arr[:, 0].min()) if path_world_arr.size else 0.0,
            "x_max": float(path_world_arr[:, 0].max()) if path_world_arr.size else 0.0,
            "y_min": float(path_world_arr[:, 1].min()) if path_world_arr.size else 0.0,
            "y_max": float(path_world_arr[:, 1].max()) if path_world_arr.size else 0.0,
        },
        "traffic_speed_penalty_mean": (
            None
            if traffic_speed_penalty_arr is None
            else (float(np.mean(traffic_speed_penalty_arr[valid_mask_arr])) if valid_any else None)
        ),
        "traffic_speed_reference_mean": (
            None
            if traffic_speed_reference_arr is None
            else (float(np.mean(traffic_speed_reference_arr[valid_mask_arr])) if valid_any else None)
        ),
        "traffic_speed_neighbor_median_mean": (
            None
            if traffic_speed_neighbor_median_arr is None
            else (float(np.mean(traffic_speed_neighbor_median_arr[valid_mask_arr])) if valid_any else None)
        ),
        "traffic_speed_floor_threshold_mean": (
            None
            if traffic_speed_floor_threshold_arr is None
            else (float(np.mean(traffic_speed_floor_threshold_arr[valid_mask_arr])) if valid_any else None)
        ),
        "traffic_speed_sdc_mean": (
            None
            if traffic_speed_sdc_arr is None
            else (float(np.mean(traffic_speed_sdc_arr[valid_mask_arr])) if valid_any else None)
        ),
        "traffic_speed_neighbor_count_mean": (
            None
            if traffic_speed_neighbor_count_arr is None
            else (float(np.mean(traffic_speed_neighbor_count_arr[valid_mask_arr])) if valid_any else None)
        ),
        "rollouts": rollouts,
    }
    if extra_summary:
        summary.update(_json_ready(dict(extra_summary)))
    summary_path = outdir / "group_rollout_advantage_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    return {
        "analysis_png": str(analysis_png),
        "summary_json": str(summary_path),
    }
