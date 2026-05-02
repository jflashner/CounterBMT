from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _read_jsonl,
    _select_row,
    _selected_path_world_from_row,
)
from scripts.counterfactual.path_semantics_plot_utils import _finite_xy_rows, _world_to_sdc_up_frame
from scripts.counterfactual.path_semantics_plot_utils import (
    PAST_STEPS,
    _select_map_context,
    _select_nearby_agents,
    _select_traffic_lights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a standalone CounterDrive rollout scene panel and optionally export reusable rollout data."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--summary", type=str, help="group_rollout_advantage_summary.json from the rollout analysis script.")
    source.add_argument("--rollout-data", type=str, help="Previously exported rollout_scene_data.json.")
    parser.add_argument("--control-index", type=str, default="", help="Control index used to recover scene context when --summary is set.")
    parser.add_argument("--scenario-id", type=str, default="", help="Override scenario id when selecting the row from --control-index.")
    parser.add_argument("--slot-id", type=str, default="", help="Override slot id when selecting the row from --control-index.")
    parser.add_argument("--data-output", type=str, default="", help="Optional JSON path for exported scene and trajectory data.")
    parser.add_argument("--npz-output", type=str, default="", help="Optional NPZ path for rollout trajectories and metrics.")
    parser.add_argument("--output", type=str, required=True, help="Output figure path, usually .png or .pdf.")
    parser.add_argument("--context-radius-m", type=float, default=-1.0)
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--figsize", type=float, nargs=2, default=(5.4, 5.0), metavar=("W", "H"))
    parser.add_argument("--crop-radius-m", type=float, default=48.0)
    parser.add_argument("--vertical-fraction", type=float, default=0.10)
    parser.add_argument("--x-half-extent-m", type=float, default=-1.0)
    parser.add_argument("--y-min-m", type=float, default=float("nan"))
    parser.add_argument("--y-max-m", type=float, default=float("nan"))
    parser.add_argument("--show-route", action="store_true", default=True)
    parser.add_argument("--hide-route", action="store_false", dest="show_route")
    parser.add_argument("--show-tube", action="store_true")
    parser.add_argument("--tube-radius-m", type=float, default=-1.0)
    parser.add_argument("--tube-grid-step-m", type=float, default=0.35)
    parser.add_argument("--tube-color", type=str, default="#60a5fa")
    parser.add_argument("--tube-alpha", type=float, default=0.22)
    parser.add_argument("--tube-boundary-color", type=str, default="#2563eb")
    parser.add_argument("--tube-boundary-alpha", type=float, default=0.80)
    parser.add_argument("--plain-scene", action="store_true")
    parser.add_argument("--show-agent-cars", action="store_true")
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--show-reward-colorbar", action="store_true")
    parser.add_argument("--line-width", type=float, default=2.7)
    parser.add_argument("--route-line-width", type=float, default=5.2)
    parser.add_argument("--alpha", type=float, default=0.88)
    parser.add_argument("--route-jump-threshold-m", type=float, default=3.0)
    parser.add_argument("--rollout-jump-threshold-m", type=float, default=10.0)
    parser.add_argument("--color-by", choices=("return", "advantage", "id"), default="return")
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--route-color", type=str, default="#2563eb")
    parser.add_argument("--ego-color", type=str, default="#111827")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def _rollout_metric(record: Mapping[str, Any], *, color_by: str, fallback: int) -> float:
    if color_by == "return":
        return float(record.get("total_return", fallback) or 0.0)
    if color_by == "advantage":
        return float(record.get("scalar_group_advantage", fallback) or 0.0)
    return float(fallback)


def _split_xy_by_jump(xy: np.ndarray, *, threshold_m: float) -> list[np.ndarray]:
    points = _finite_xy_rows(np.asarray(xy, dtype=np.float64))
    if points.shape[0] < 2:
        return []
    threshold = float(threshold_m)
    if threshold <= 0.0:
        return [points]
    step = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    break_after = np.flatnonzero(step > threshold)
    segments: list[np.ndarray] = []
    start = 0
    for idx in break_after.tolist():
        segment = points[start : idx + 1]
        if segment.shape[0] >= 2:
            segments.append(segment)
        start = idx + 1
    segment = points[start:]
    if segment.shape[0] >= 2:
        segments.append(segment)
    return segments


def _distance_to_polyline_segments(points_xy: np.ndarray, segments_xy: Sequence[np.ndarray]) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    out = np.full((points.shape[0],), np.inf, dtype=np.float64)
    for segment in segments_xy:
        poly = _finite_xy_rows(np.asarray(segment, dtype=np.float64))
        if poly.shape[0] < 2:
            continue
        start = poly[:-1]
        end = poly[1:]
        vec = end - start
        denom = np.sum(vec * vec, axis=1).clip(min=1e-9)
        rel = points[:, None, :] - start[None, :, :]
        t = np.sum(rel * vec[None, :, :], axis=-1) / denom[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = start[None, :, :] + t[:, :, None] * vec[None, :, :]
        dist = np.min(np.linalg.norm(points[:, None, :] - closest, axis=-1), axis=1)
        out = np.minimum(out, dist)
    return out


def _draw_tube(
    *,
    ax,
    route_segments_world: Sequence[np.ndarray],
    center_xy: np.ndarray,
    heading: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    tube_radius_m: float,
    grid_step_m: float,
    route_jump_threshold_m: float,
    color: str,
    alpha: float,
    boundary_color: str,
    boundary_alpha: float,
) -> None:
    local_segments: list[np.ndarray] = []
    for segment_world in route_segments_world:
        for split_world in _split_xy_by_jump(np.asarray(segment_world), threshold_m=route_jump_threshold_m):
            split_local = _world_to_sdc_up_frame(split_world, center_xy=center_xy, heading_rad=heading)
            if split_local.shape[0] >= 2:
                local_segments.append(split_local)
    if not local_segments:
        return
    step = max(float(grid_step_m), 0.05)
    xs = np.arange(float(xlim[0]), float(xlim[1]) + 0.5 * step, step, dtype=np.float64)
    ys = np.arange(float(ylim[0]), float(ylim[1]) + 0.5 * step, step, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    query = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    dist = _distance_to_polyline_segments(query, local_segments).reshape(xx.shape)
    inside = np.ma.masked_where(dist > float(tube_radius_m), dist)
    ax.contourf(
        xx,
        yy,
        inside,
        levels=np.linspace(0.0, float(tube_radius_m), num=8),
        colors=[str(color)],
        alpha=float(alpha),
        zorder=8.1,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        dist,
        levels=[float(tube_radius_m)],
        colors=[str(boundary_color)],
        linewidths=1.25,
        alpha=float(boundary_alpha),
        zorder=8.3,
    )


def _agent_heading_from_past(agent: Mapping[str, Any], *, center_xy: np.ndarray, heading: float) -> float:
    past = _finite_xy_rows(np.asarray(agent.get("past_xy", []), dtype=np.float64))
    if past.shape[0] >= 2:
        local = _world_to_sdc_up_frame(past, center_xy=center_xy, heading_rad=heading)
        for idx in range(local.shape[0] - 1, 0, -1):
            delta = local[idx] - local[idx - 1]
            if float(np.linalg.norm(delta)) > 0.15:
                return float(np.arctan2(delta[1], delta[0]))
    return float(np.pi / 2.0)


def _draw_agent_cars(ax, render_context: Mapping[str, Any], *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    heading = float(render_context["current_heading"])
    for agent in render_context.get("nearby_agents", []):
        current = _finite_xy_rows(np.asarray([agent.get("current_xy", [])], dtype=np.float64))
        if current.shape[0] == 0:
            continue
        local = _world_to_sdc_up_frame(current, center_xy=center_xy, heading_rad=heading)[0]
        if local[0] < xlim[0] - 3.0 or local[0] > xlim[1] + 3.0 or local[1] < ylim[0] - 3.0 or local[1] > ylim[1] + 3.0:
            continue
        theta = _agent_heading_from_past(agent, center_xy=center_xy, heading=heading)
        direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float64)
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        length = 4.4
        width = 1.9
        corners = np.asarray(
            [
                local + 0.5 * length * direction + 0.5 * width * normal,
                local + 0.5 * length * direction - 0.5 * width * normal,
                local - 0.5 * length * direction - 0.5 * width * normal,
                local - 0.5 * length * direction + 0.5 * width * normal,
            ],
            dtype=np.float64,
        )
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="#cbd5e1",
                edgecolor="#475569",
                linewidth=0.8,
                alpha=0.95,
                zorder=11.1,
            )
        )
        arrow_start = local - 0.12 * length * direction
        arrow_end = local + 0.34 * length * direction
        ax.annotate(
            "",
            xy=(float(arrow_end[0]), float(arrow_end[1])),
            xytext=(float(arrow_start[0]), float(arrow_start[1])),
            arrowprops=dict(arrowstyle="-|>", color="#334155", lw=0.8, mutation_scale=6),
            zorder=11.3,
        )


def _draw_plain_scene_ax(
    *,
    ax,
    render_context: Mapping[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    heading = float(render_context["current_heading"])
    map_context = dict(render_context.get("map_context", {}) or {})
    ax.set_facecolor("#f8fafc")

    for feature in map_context.get("crosswalks", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature.get("xy_world", []), dtype=np.float64), center_xy=center_xy, heading_rad=heading)
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color="#e2e8f0", alpha=0.32, zorder=1)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature.get("xy_world", []), dtype=np.float64), center_xy=center_xy, heading_rad=heading)
        if xy.shape[0] >= 2:
            for segment in _split_xy_by_jump(xy, threshold_m=3.5):
                ax.plot(segment[:, 0], segment[:, 1], color="#cbd5e1", linewidth=1.0, alpha=0.24, zorder=3)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature.get("xy_world", []), dtype=np.float64), center_xy=center_xy, heading_rad=heading)
        if xy.shape[0] >= 2:
            for segment in _split_xy_by_jump(xy, threshold_m=3.5):
                ax.plot(segment[:, 0], segment[:, 1], color="#334155", linewidth=2.7, alpha=0.96, zorder=4)

    for light in render_context.get("traffic_lights", []):
        stop_xy = _world_to_sdc_up_frame(np.asarray([light.get("stop_point_xy_world", [])], dtype=np.float64), center_xy=center_xy, heading_rad=heading)
        if stop_xy.shape[0] == 0:
            continue
        state = str(light.get("state") or "unknown")
        color = "#ef4444" if "STOP" in state or "RED" in state else ("#22c55e" if "GO" in state or "GREEN" in state else "#eab308")
        ax.scatter([stop_xy[0, 0]], [stop_xy[0, 1]], c=color, marker="s", s=76, edgecolors="black", linewidths=0.8, zorder=9)

    gt_past = _finite_xy_rows(np.asarray(render_context.get("gt_past_xy", []), dtype=np.float64))
    gt_local = _world_to_sdc_up_frame(gt_past, center_xy=center_xy, heading_rad=heading)
    if gt_local.shape[0] >= 2:
        ax.plot(gt_local[:, 0], gt_local[:, 1], color="#111827", linewidth=2.2, alpha=0.85, zorder=10)
    ax.scatter([0.0], [0.0], c="#f43f5e", s=42, edgecolors="white", linewidths=0.75, zorder=10.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _extract_scene_render_context_with_radius(raw_scenario: Mapping[str, Any], row: Mapping[str, Any], *, radius_m: float) -> dict[str, Any]:
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
    past_slice = slice(max(0, idx - int(PAST_STEPS)), idx + 1)
    gt_past_xy = _finite_xy_rows(position[past_slice][valid[past_slice]])
    return {
        "current_time_index": int(idx),
        "current_xy": np.asarray(current_xy, dtype=np.float64),
        "current_heading": float(current_heading),
        "gt_past_xy": np.asarray(gt_past_xy, dtype=np.float64),
        "map_context": _select_map_context(raw_scenario, center_xy=current_xy, radius_m=float(radius_m)),
        "traffic_lights": _select_traffic_lights(
            raw_scenario,
            center_xy=current_xy,
            radius_m=float(radius_m),
            time_index=idx,
        ),
        "nearby_agents": _select_nearby_agents(
            raw_scenario,
            sdc_id=sdc_id,
            center_xy=current_xy,
            current_idx=idx,
            radius_m=float(radius_m),
        ),
    }


def _export_from_summary(
    summary_path: Path,
    control_index_path: Path,
    scenario_id: str,
    slot_id: str,
    *,
    context_radius_m: float = -1.0,
) -> dict[str, Any]:
    summary = _read_json(summary_path)
    scenario = scenario_id or str(summary.get("scenario_id") or "")
    slot = slot_id or str(summary.get("selected_slot_id") or "")
    rows = _read_jsonl(control_index_path)
    row_index = _select_row(rows, row_index=-1, scenario_id=scenario, slot_id=slot)
    row = dict(rows[row_index])
    raw_scenario = load_raw_scenario_from_row(row)
    if float(context_radius_m) > 0.0:
        render_context = _extract_scene_render_context_with_radius(raw_scenario, row, radius_m=float(context_radius_m))
    else:
        render_context = _extract_scene_render_context(raw_scenario, row)
    selected_path_id, path_segments_world = _selected_path_world_from_row(row)
    route_parts = [_finite_xy_rows(np.asarray(seg, dtype=np.float64)) for seg in path_segments_world]
    route_parts = [part for part in route_parts if part.shape[0] >= 2]
    route_world = np.concatenate(route_parts, axis=0) if route_parts else np.zeros((0, 2), dtype=np.float64)
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    heading = float(render_context["current_heading"])

    rollouts = []
    for record in summary.get("rollouts", []):
        traj_world = _finite_xy_rows(np.asarray(record.get("trajectory_world_xy", []), dtype=np.float64))
        traj_local = _world_to_sdc_up_frame(traj_world, center_xy=center_xy, heading_rad=heading)
        rollouts.append(
            {
                "rollout_id": int(record.get("rollout_id", len(rollouts))),
                "seed": int(record.get("seed", 0)),
                "trajectory_world_xy": traj_world,
                "trajectory_local_xy": traj_local,
                "total_return": record.get("total_return"),
                "scalar_group_advantage": record.get("scalar_group_advantage"),
                "inside_fraction": record.get("inside_fraction"),
                "first_exit_step": record.get("first_exit_step"),
            }
        )

    return {
        "source_summary": str(summary_path),
        "source_control_index": str(control_index_path),
        "scenario_id": scenario,
        "selected_slot_id": slot,
        "requested_semantic_label": summary.get("requested_semantic_label") or row.get("requested_semantic_label"),
        "selected_path_id": summary.get("selected_path_id") or selected_path_id,
        "tube_radius_m": summary.get("tube_radius_m"),
        "render_context": render_context,
        "route_world_xy": route_world,
        "route_segments_world_xy": route_parts,
        "rollouts": rollouts,
    }


def _save_npz(path: str | Path, data: Mapping[str, Any]) -> Path:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rollouts = list(data.get("rollouts", []))
    max_len = max((len(r.get("trajectory_world_xy", [])) for r in rollouts), default=0)
    world = np.full((len(rollouts), max_len, 2), np.nan, dtype=np.float32)
    local = np.full((len(rollouts), max_len, 2), np.nan, dtype=np.float32)
    ids = np.zeros((len(rollouts),), dtype=np.int32)
    returns = np.full((len(rollouts),), np.nan, dtype=np.float32)
    advantages = np.full((len(rollouts),), np.nan, dtype=np.float32)
    inside = np.full((len(rollouts),), np.nan, dtype=np.float32)
    for idx, record in enumerate(rollouts):
        ids[idx] = int(record.get("rollout_id", idx))
        returns[idx] = float(record.get("total_return", np.nan))
        advantages[idx] = float(record.get("scalar_group_advantage", np.nan))
        inside[idx] = float(record.get("inside_fraction", np.nan))
        traj_world = np.asarray(record.get("trajectory_world_xy", []), dtype=np.float32)
        traj_local = np.asarray(record.get("trajectory_local_xy", []), dtype=np.float32)
        world[idx, : traj_world.shape[0], :] = traj_world[:, :2]
        local[idx, : traj_local.shape[0], :] = traj_local[:, :2]
    np.savez_compressed(
        out,
        trajectory_world_xy=world,
        trajectory_local_xy=local,
        rollout_id=ids,
        total_return=returns,
        scalar_group_advantage=advantages,
        inside_fraction=inside,
    )
    return out


def _render(data: Mapping[str, Any], args: argparse.Namespace) -> Path:
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=int(args.dpi))
    route_world = np.asarray(data.get("route_world_xy", []), dtype=np.float64)
    route_segments = [np.asarray(seg, dtype=np.float64) for seg in data.get("route_segments_world_xy", [])]
    half_extent = float(args.crop_radius_m)
    x_half_extent = half_extent if float(args.x_half_extent_m) <= 0.0 else float(args.x_half_extent_m)
    vertical_span = 2.0 * half_extent
    y_min = -float(args.vertical_fraction) * vertical_span
    y_max = y_min + vertical_span
    if np.isfinite(float(args.y_min_m)):
        y_min = float(args.y_min_m)
    if np.isfinite(float(args.y_max_m)):
        y_max = float(args.y_max_m)
    xlim = (-x_half_extent, x_half_extent)
    ylim = (y_min, y_max)
    info_text = ""
    if args.show_info:
        info_text = (
            f"scene={data.get('scenario_id')}\n"
            f"slot={data.get('selected_slot_id')}  label={data.get('requested_semantic_label')}\n"
            f"group={len(data.get('rollouts', []))}"
        )
    render_context = dict(data["render_context"])
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    heading = float(render_context["current_heading"])
    if args.plain_scene:
        _draw_plain_scene_ax(ax=ax, render_context=render_context, xlim=xlim, ylim=ylim)
        if info_text:
            ax.text(
                0.02,
                0.975,
                info_text,
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
                zorder=15,
            )
    else:
        _draw_vlm_style_scene_ax(
            fig=fig,
            ax=ax,
            render_context=render_context,
            highlighted_segments_world=[],
            highlighted_gradient_values=None,
            representative_route_world=route_world if args.show_route and route_world.size else None,
            info_box_text=info_text,
            show_colorbar=False,
        )
    if args.show_tube:
        radius = float(args.tube_radius_m)
        if radius <= 0.0:
            radius = float(data.get("tube_radius_m") or 3.0)
        _draw_tube(
            ax=ax,
            route_segments_world=route_segments,
            center_xy=center_xy,
            heading=heading,
            xlim=xlim,
            ylim=ylim,
            tube_radius_m=radius,
            grid_step_m=float(args.tube_grid_step_m),
            route_jump_threshold_m=float(args.route_jump_threshold_m),
            color=str(args.tube_color),
            alpha=float(args.tube_alpha),
            boundary_color=str(args.tube_boundary_color),
            boundary_alpha=float(args.tube_boundary_alpha),
        )
    if args.show_route:
        for seg_idx, segment_world in enumerate(route_segments):
            segment = _finite_xy_rows(np.asarray(segment_world, dtype=np.float64))
            split_segments = _split_xy_by_jump(segment, threshold_m=float(args.route_jump_threshold_m))
            for local_idx, split_world in enumerate(split_segments):
                segment_local = _world_to_sdc_up_frame(split_world, center_xy=center_xy, heading_rad=heading)
                if segment_local.shape[0] < 2:
                    continue
                ax.plot(
                    segment_local[:, 0],
                    segment_local[:, 1],
                    color=str(args.route_color),
                    linewidth=float(args.route_line_width),
                    alpha=0.34,
                    zorder=8.4,
                    solid_capstyle="round",
                )
                if seg_idx > 0 or local_idx > 0:
                    ax.scatter(
                        [segment_local[0, 0]],
                        [segment_local[0, 1]],
                        c=str(args.route_color),
                        s=18,
                        marker="x",
                        linewidths=0.8,
                        zorder=8.6,
                        alpha=0.65,
                    )

    rollouts = list(data.get("rollouts", []))
    values = np.asarray(
        [_rollout_metric(record, color_by=str(args.color_by), fallback=idx) for idx, record in enumerate(rollouts)],
        dtype=np.float64,
    )
    if values.size == 0 or not np.isfinite(values).any() or np.nanmax(values) <= np.nanmin(values):
        normed = np.linspace(0.20, 0.86, num=max(len(rollouts), 1), dtype=np.float64)
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        normed = (values - vmin) / max(vmax - vmin, 1e-6)
    cmap = plt.get_cmap(str(args.cmap))
    for idx, record in enumerate(rollouts):
        traj_world = _finite_xy_rows(np.asarray(record.get("trajectory_world_xy", []), dtype=np.float64))
        color = cmap(float(normed[idx]))
        traj_segments_world = _split_xy_by_jump(traj_world, threshold_m=float(args.rollout_jump_threshold_m))
        last_local = None
        for traj_segment_world in traj_segments_world:
            traj_local = _world_to_sdc_up_frame(traj_segment_world, center_xy=center_xy, heading_rad=heading)
            if traj_local.shape[0] < 2:
                continue
            ax.plot(
                traj_local[:, 0],
                traj_local[:, 1],
                color=color,
                linewidth=float(args.line_width),
                alpha=float(args.alpha),
                zorder=12.5,
                solid_capstyle="round",
            )
            last_local = traj_local
        if last_local is None:
            continue
        ax.scatter(
            [last_local[-1, 0]],
            [last_local[-1, 1]],
            c=[color],
            s=30,
            edgecolors="white",
            linewidths=0.75,
            zorder=13.0,
        )
        if args.show_labels:
            ax.text(
                float(last_local[-1, 0]) + 0.65,
                float(last_local[-1, 1]) + 0.65,
                str(record.get("rollout_id", idx)),
                fontsize=7.5,
                color="#111827",
                zorder=13.3,
            )

    if args.show_agent_cars:
        _draw_agent_cars(ax, render_context, xlim=xlim, ylim=ylim)
    if args.show_reward_colorbar and values.size > 0 and np.isfinite(values).any():
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_clim(float(np.nanmin(values)), float(np.nanmax(values)))
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
        cbar.ax.tick_params(labelsize=7, length=2)
        cbar.set_label("rollout return", fontsize=7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main() -> int:
    args = parse_args()
    if args.rollout_data:
        data = _read_json(args.rollout_data)
    else:
        if not args.control_index:
            raise ValueError("--control-index is required when exporting from --summary")
        data = _export_from_summary(
            Path(args.summary).expanduser(),
            Path(args.control_index).expanduser(),
            scenario_id=str(args.scenario_id or ""),
            slot_id=str(args.slot_id or ""),
            context_radius_m=float(args.context_radius_m),
        )
    if args.data_output:
        _write_json(args.data_output, data)
    if args.npz_output:
        _save_npz(args.npz_output, data)
    out = _render(data, args)
    print(json.dumps({"output": str(out), "data_output": args.data_output, "npz_output": args.npz_output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
