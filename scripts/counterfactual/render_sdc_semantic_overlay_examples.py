from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from PIL import Image, ImageDraw

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_semantic_control import extract_model_frame, load_raw_scenario_from_row
from bmt.counterfactual.sdc_path_control import (
    _extract_valid_sdc_path_xy,
    extract_ground_truth_sdc_route_xy,
    extract_sdc_current_pose,
    split_polyline_on_discontinuities,
    trim_polyline_from_point,
)
from bmt.dataset.dataset import InfgenDataset
from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule, _load_eval_model
from bmt.eval.scenario_evaluator import Evaluator
from bmt.tokenization import get_tokenizer
from bmt.utils.config import REPO_ROOT
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _model_to_world,
)
from scripts.counterfactual.path_semantics_plot_utils import _world_to_sdc_up_frame
from scripts.counterfactual.render_sdc_semantic_eval_examples import (
    _annotate_and_resize,
    _load_config,
    _read_jsonl,
    _resolve_device,
    _save_grid,
    _scenario_sort_key,
    _to_builtin,
    _to_numpy_output,
)


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


def _polyline_segment_distance_to_points(points_xy: Any, polyline_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    polyline = np.asarray(polyline_xy, dtype=np.float32).reshape(-1, 2)
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
    half_extent = 48.0
    vertical_span = 2.0 * half_extent
    y_min = -0.10 * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    world_xy = sdc_up_to_world_frame(local_xy, center_xy_world=center_xy_world, heading_world_rad=heading_world_rad)
    dist = _polyline_segment_distance_to_points(world_xy, polyline_world_xy).reshape(xx.shape)
    return xx, yy, dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render semantic-control overlay panels that show scene context, requested path/tube, "
            "and generated SDC rollout for validation examples."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml",
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sampling-method", type=str, default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--grid-columns", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=780)
    parser.add_argument("--tube-radius-m", type=float, default=3.0)
    parser.add_argument("--grid-step-m", type=float, default=0.35)
    parser.add_argument("--jump-threshold-m", type=float, default=6.0)
    parser.add_argument("--full-scene-view", action="store_true")
    return parser.parse_args()


def _row_slot_id(row: Mapping[str, Any], row_idx: int) -> str:
    selected_slot = str(row.get("selected_slot_id") or "").strip()
    slot_id = str(row.get("slot_id") or "").strip()
    if selected_slot:
        return selected_slot
    if slot_id:
        return slot_id
    if str(row.get("source_kind") or "") == "factual_gt":
        return "factual_gt"
    return f"row_{int(row_idx):04d}"


def _extract_sdc_rollout_world(output_np: Mapping[str, Any], raw_scenario: Mapping[str, Any]) -> np.ndarray:
    reconstructed_pos = np.asarray(output_np.get("decoder/reconstructed_position", []), dtype=np.float32)
    reconstructed_valid = np.asarray(output_np.get("decoder/reconstructed_valid_mask", []), dtype=bool)
    if reconstructed_pos.ndim != 3 or reconstructed_pos.shape[-1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    sdc_index = int(np.asarray(output_np.get("decoder/sdc_index", 0)).reshape(-1)[0])
    if sdc_index < 0 or sdc_index >= reconstructed_pos.shape[1]:
        sdc_index = 0

    traj_model = np.asarray(reconstructed_pos[:, sdc_index, :2], dtype=np.float32)
    if reconstructed_valid.ndim == 2:
        valid_mask = np.asarray(reconstructed_valid[:, sdc_index], dtype=bool)
    else:
        valid_mask = np.ones((traj_model.shape[0],), dtype=bool)

    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)

    map_center_world, map_heading_world = extract_model_frame(raw_scenario)
    traj_world = _model_to_world(
        traj_model[valid_mask],
        map_center_world=np.asarray(map_center_world, dtype=np.float32),
        map_heading_world=float(map_heading_world),
    )
    return np.asarray(traj_world, dtype=np.float32)


def _extract_gt_sdc_world(gt_data_dict: Mapping[str, Any], raw_scenario: Mapping[str, Any]) -> np.ndarray:
    agent_position = np.asarray(gt_data_dict.get("decoder/agent_position", []), dtype=np.float32)
    agent_valid = np.asarray(gt_data_dict.get("decoder/agent_valid_mask", []), dtype=bool)
    if agent_position.ndim != 3 or agent_position.shape[-1] < 2:
        return np.zeros((0, 2), dtype=np.float32)

    time_steps, num_agents = agent_position.shape[:2]
    if agent_valid.shape == (num_agents, time_steps):
        agent_valid = agent_valid.T
    if agent_valid.shape != (time_steps, num_agents):
        agent_valid = np.ones((time_steps, num_agents), dtype=bool)

    sdc_index = int(np.asarray(gt_data_dict.get("decoder/sdc_index", 0)).reshape(-1)[0])
    if sdc_index < 0 or sdc_index >= num_agents:
        sdc_index = 0

    traj_model = np.asarray(agent_position[:, sdc_index, :2], dtype=np.float32)
    valid_mask = np.asarray(agent_valid[:, sdc_index], dtype=bool)
    if not np.any(valid_mask):
        return np.zeros((0, 2), dtype=np.float32)

    map_center_world, map_heading_world = extract_model_frame(raw_scenario)
    traj_world = _model_to_world(
        traj_model[valid_mask],
        map_center_world=np.asarray(map_center_world, dtype=np.float32),
        map_heading_world=float(map_heading_world),
    )
    return np.asarray(traj_world, dtype=np.float32)


def _compute_single_mode_sade_sfde(
    gt_data_dict: Mapping[str, Any],
    pred_data_dict: Mapping[str, Any],
) -> Dict[str, float]:
    gt_pos = np.asarray(gt_data_dict.get("decoder/agent_position", []), dtype=np.float32)[..., :2]
    pred_pos = np.asarray(pred_data_dict.get("decoder/reconstructed_position", []), dtype=np.float32)[..., :2]
    gt_valid = np.asarray(gt_data_dict.get("decoder/agent_valid_mask", []), dtype=bool)

    if gt_pos.ndim != 3 or pred_pos.ndim != 3:
        return {"sade": float("nan"), "sfde": float("nan")}

    t_gt, n_gt = gt_pos.shape[:2]
    t_pred, n_pred = pred_pos.shape[:2]
    t_eval = int(min(t_gt, t_pred))
    n_eval = int(min(n_gt, n_pred))
    if t_eval <= 0 or n_eval <= 0:
        return {"sade": float("nan"), "sfde": float("nan")}

    gt_pos = np.asarray(gt_pos[:t_eval, :n_eval], dtype=np.float32)
    pred_pos = np.asarray(pred_pos[:t_eval, :n_eval], dtype=np.float32)

    if gt_valid.shape == (n_gt, t_gt):
        gt_valid = gt_valid.T
    if gt_valid.shape != (t_gt, n_gt):
        gt_valid = np.ones((t_gt, n_gt), dtype=bool)
    gt_valid = np.asarray(gt_valid[:t_eval, :n_eval], dtype=bool)

    error = np.linalg.norm(gt_pos - pred_pos, axis=-1).astype(np.float32)
    agent_valid = np.any(gt_valid, axis=0)
    if not np.any(agent_valid):
        return {"sade": float("nan"), "sfde": float("nan")}

    valid_counts = gt_valid.sum(axis=0).clip(min=1)
    last_valid_ind = gt_valid.cumsum(axis=0).argmax(axis=0)
    fde = error[last_valid_ind, np.arange(n_eval)]
    sfde = float(fde[agent_valid].sum() / max(int(agent_valid.sum()), 1))

    sade_per_agent = (error * gt_valid).sum(axis=0) / valid_counts
    sade = float(sade_per_agent[agent_valid].sum() / max(int(agent_valid.sum()), 1))
    return {"sade": sade, "sfde": sfde}


def _compute_single_mode_sdc_ade_fde(
    gt_data_dict: Mapping[str, Any],
    pred_data_dict: Mapping[str, Any],
) -> Dict[str, float]:
    gt_pos = np.asarray(gt_data_dict.get("decoder/agent_position", []), dtype=np.float32)[..., :2]
    pred_pos = np.asarray(pred_data_dict.get("decoder/reconstructed_position", []), dtype=np.float32)[..., :2]
    gt_valid = np.asarray(gt_data_dict.get("decoder/agent_valid_mask", []), dtype=bool)
    if gt_pos.ndim != 3 or pred_pos.ndim != 3:
        return {"sdc_ade": float("nan"), "sdc_fde": float("nan")}

    t_gt, n_gt = gt_pos.shape[:2]
    t_pred, n_pred = pred_pos.shape[:2]
    t_eval = int(min(t_gt, t_pred))
    n_eval = int(min(n_gt, n_pred))
    if t_eval <= 0 or n_eval <= 0:
        return {"sdc_ade": float("nan"), "sdc_fde": float("nan")}

    gt_pos = np.asarray(gt_pos[:t_eval, :n_eval], dtype=np.float32)
    pred_pos = np.asarray(pred_pos[:t_eval, :n_eval], dtype=np.float32)

    if gt_valid.shape == (n_gt, t_gt):
        gt_valid = gt_valid.T
    if gt_valid.shape != (t_gt, n_gt):
        gt_valid = np.ones((t_gt, n_gt), dtype=bool)
    gt_valid = np.asarray(gt_valid[:t_eval, :n_eval], dtype=bool)

    sdc_index = int(np.asarray(gt_data_dict.get("decoder/sdc_index", 0)).reshape(-1)[0])
    if sdc_index < 0 or sdc_index >= n_eval:
        sdc_index = 0

    gt_traj = gt_pos[:, sdc_index]
    pred_traj = pred_pos[:, sdc_index]
    valid_mask = gt_valid[:, sdc_index]
    if not np.any(valid_mask):
        return {"sdc_ade": float("nan"), "sdc_fde": float("nan")}

    error = np.linalg.norm(gt_traj - pred_traj, axis=-1).astype(np.float32)
    ade = float(error[valid_mask].mean())
    fde = float(error[np.where(valid_mask)[0][-1]])
    return {"sdc_ade": ade, "sdc_fde": fde}


def _set_full_scene_limits(
    ax,
    *,
    render_context: Mapping[str, Any],
    center_xy_world: Sequence[float],
    heading_world_rad: float,
    route_segments_world: Sequence[np.ndarray],
    rollout_world_xy: np.ndarray,
    gt_sdc_world_xy: np.ndarray,
) -> None:
    local_points: List[np.ndarray] = []

    def _append_world_xy(points_world_xy: Any) -> None:
        points = np.asarray(points_world_xy, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] == 0:
            return
        local = world_to_sdc_up_frame(
            points,
            center_xy_world=center_xy_world,
            heading_world_rad=heading_world_rad,
        )
        if local.shape[0] > 0:
            local_points.append(local)

    _append_world_xy(np.asarray(render_context.get("gt_past_xy", []), dtype=np.float32))
    _append_world_xy(np.asarray(center_xy_world, dtype=np.float32).reshape(1, 2))
    _append_world_xy(rollout_world_xy)
    _append_world_xy(gt_sdc_world_xy)
    for seg_world in route_segments_world:
        _append_world_xy(seg_world)

    map_context = render_context.get("map_context", {}) or {}
    for key in ("road_boundaries", "lane_centerlines", "crosswalks"):
        for feature in map_context.get(key, []) or []:
            _append_world_xy(feature.get("xy_world", []))

    for light in render_context.get("traffic_lights", []) or []:
        stop_xy = np.asarray(light.get("stop_point_xy_world", []), dtype=np.float32).reshape(-1, 2)
        _append_world_xy(stop_xy)

    for agent in render_context.get("nearby_agents", []) or []:
        _append_world_xy(agent.get("past_xy", []))
        _append_world_xy(np.asarray(agent.get("current_xy", []), dtype=np.float32).reshape(-1, 2))

    if not local_points:
        return

    merged = np.concatenate(local_points, axis=0)
    finite = merged[np.all(np.isfinite(merged), axis=-1)]
    if finite.shape[0] == 0:
        return

    x_min = float(np.min(finite[:, 0]))
    x_max = float(np.max(finite[:, 0]))
    y_min = float(np.min(finite[:, 1]))
    y_max = float(np.max(finite[:, 1]))
    span_x = max(x_max - x_min, 20.0)
    span_y = max(y_max - y_min, 20.0)
    pad = max(8.0, 0.06 * max(span_x, span_y))

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_aspect("equal", adjustable="box")


def _render_overlay_panel(
    *,
    row: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
    rollout_world_xy: np.ndarray,
    gt_sdc_world_xy: np.ndarray,
    out_path: Path,
    tube_radius_m: float,
    grid_step_m: float,
    jump_threshold_m: float,
    title_text: str,
    full_scene_view: bool,
) -> Image.Image:
    render_context = _extract_scene_render_context(raw_scenario, row)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float32)
    current_heading = float(render_context["current_heading"])
    route_world = np.asarray(selected_raw_route_world(raw_scenario, row), dtype=np.float32)
    route_segments_world = [
        np.asarray(seg, dtype=np.float32)
        for seg in split_polyline_on_discontinuities(route_world, jump_threshold_m=float(jump_threshold_m))
        if np.asarray(seg).shape[0] >= 2
    ]

    fig = plt.figure(figsize=(7.8, 7.8), dpi=180)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=render_context,
        highlighted_segments_world=route_segments_world,
        highlighted_gradient_values=None,
        representative_route_world=route_world,
        rollout_trajectory_world=np.asarray(rollout_world_xy, dtype=np.float32),
        current_marker_world=current_xy,
        info_box_text=title_text,
        show_colorbar=False,
    )

    xx, yy, distance_field = segment_distance_field_in_sdc_frame(
        polyline_world_xy=route_world,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
        grid_step_m=float(grid_step_m),
    )
    inside = np.ma.masked_where(distance_field > float(tube_radius_m), distance_field)
    ax.contourf(
        xx,
        yy,
        inside,
        levels=np.linspace(0.0, float(tube_radius_m), num=8),
        cmap="Blues_r",
        alpha=0.22,
        zorder=6.1,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        distance_field,
        levels=[float(tube_radius_m)],
        colors=["#f59e0b"],
        linewidths=1.8,
        linestyles=["--"],
        zorder=11.5,
    )

    for seg_idx, seg_world in enumerate(route_segments_world):
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
            linewidth=4.8,
            alpha=0.98,
            zorder=12.0,
            solid_capstyle="round",
        )
        if seg_idx > 0:
            ax.scatter(
                [seg_local[0, 0]],
                [seg_local[0, 1]],
                c="#111827",
                s=28,
                marker="x",
                linewidths=1.1,
                zorder=12.3,
            )

    if rollout_world_xy.shape[0] >= 2:
        rollout_local = world_to_sdc_up_frame(
            rollout_world_xy,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        ax.plot(
            rollout_local[:, 0],
            rollout_local[:, 1],
            color="#111827",
            linewidth=2.8,
            alpha=0.95,
            linestyle=(0, (5, 3)),
            zorder=13.0,
        )
        ax.scatter(
            [rollout_local[-1, 0]],
            [rollout_local[-1, 1]],
            c="#111827",
            s=42,
            edgecolors="white",
            linewidths=0.8,
            zorder=13.2,
        )

    if gt_sdc_world_xy.shape[0] >= 2:
        gt_local = world_to_sdc_up_frame(
            gt_sdc_world_xy,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
        )
        ax.plot(
            gt_local[:, 0],
            gt_local[:, 1],
            color="#16a34a",
            linewidth=3.0,
            alpha=0.95,
            zorder=12.8,
            solid_capstyle="round",
        )
        ax.scatter(
            [gt_local[-1, 0]],
            [gt_local[-1, 1]],
            c="#16a34a",
            s=42,
            edgecolors="white",
            linewidths=0.8,
            zorder=13.1,
        )

    if full_scene_view:
        _set_full_scene_limits(
            ax,
            render_context=render_context,
            center_xy_world=current_xy,
            heading_world_rad=current_heading,
            route_segments_world=route_segments_world,
            rollout_world_xy=np.asarray(rollout_world_xy, dtype=np.float32),
            gt_sdc_world_xy=np.asarray(gt_sdc_world_xy, dtype=np.float32),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    with Image.open(out_path) as img:
        return img.convert("RGB")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    control_index = Path(args.control_index).expanduser().resolve()
    rows = _read_jsonl(control_index)
    selected_sids: List[str] = []
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if sid and sid not in selected_sids:
            selected_sids.append(sid)
        if len(selected_sids) >= int(args.num_scenes):
            break
    selected_sid_set = set(selected_sids)

    runtime_config = _load_config(args)
    model = _load_eval_model(runtime_config, str(Path(args.ckpt).expanduser().resolve()))
    config = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.create(_to_builtin(model.config)), runtime_config)
    model.config = config
    tokenizer = get_tokenizer(config)
    dataset = InfgenDataset(config, "test", backward_prediction=False)
    module = EvaluationLightningModule(
        model=model,
        evaluator=Evaluator(key_metrics_only=True),
        tokenizer=tokenizer,
        config=config,
        dataset=dataset,
        eval_mode="GPTmodel",
        multi_mode=False,
        num_modes=1,
        save_path=str(outdir / "unused_metrics"),
    )
    module.eval()
    module.model.eval()
    device = _resolve_device(args.device)
    module.model.to(device)

    scenario_to_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {sid: [] for sid in selected_sids}
    for idx, row in enumerate(rows):
        sid = str(row.get("scenario_id") or "")
        if sid in selected_sid_set:
            scenario_to_rows[sid].append((idx, row))

    manifest: List[Dict[str, Any]] = []
    for scenario_id in selected_sids:
        row_entries = sorted(scenario_to_rows[scenario_id], key=lambda item: _scenario_sort_key(item[1]))
        if not row_entries:
            continue

        scenario_dir = outdir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        panels: List[Tuple[str, Image.Image]] = []
        row_manifest: List[Dict[str, Any]] = []

        for row_idx, row in row_entries:
            raw_scenario = load_raw_scenario_from_row(row)
            raw_data = dataset[row_idx]
            input_data = module.preprocess_GPTmodel(copy.deepcopy(raw_data), backward_prediction=False)
            with torch.no_grad():
                output_data = module.GPT_AR(input_data, backward_prediction=False, teacher_forcing=False)
            output_data = tokenizer.detokenize(
                output_data,
                detokenizing_gt=False,
                backward_prediction=False,
                teacher_forcing=False,
            )
            output_np = _to_numpy_output(output_data)
            rollout_world_xy = _extract_sdc_rollout_world(output_np, raw_scenario)
            gt_sdc_world_xy = _extract_gt_sdc_world(raw_data, raw_scenario)
            metrics = _compute_single_mode_sade_sfde(raw_data, output_np)
            sdc_metrics = _compute_single_mode_sdc_ade_fde(raw_data, output_np)

            slot_id = _row_slot_id(row, int(row_idx))
            source_kind = str(row.get("source_kind") or "")
            label = str(row.get("requested_semantic_label") or "")
            path_id = row.get("selected_path_id")
            stem = f"{scenario_id}__{slot_id}".replace("/", "_")
            panel_png = scenario_dir / f"{stem}__overlay.png"
            title = f"{slot_id} | {source_kind} | {label}"
            info_box_text = (
                f"scene={scenario_id}\n"
                f"slot={slot_id}\n"
                f"requested={label or 'n/a'}\n"
                f"SDC_ADE={sdc_metrics['sdc_ade']:.2f}  SDC_FDE={sdc_metrics['sdc_fde']:.2f}\n"
                f"all_agent_SADE={metrics['sade']:.2f}  all_agent_SFDE={metrics['sfde']:.2f}\n"
                f"path={path_id}\n"
                "blue=target route\n"
                "green=gt sdc\n"
                "orange=tube boundary\n"
                "black dashed=generated sdc"
            )
            panel_img = _render_overlay_panel(
                row=row,
                raw_scenario=raw_scenario,
                rollout_world_xy=rollout_world_xy,
                gt_sdc_world_xy=gt_sdc_world_xy,
                out_path=panel_png,
                tube_radius_m=float(args.tube_radius_m),
                grid_step_m=float(args.grid_step_m),
                jump_threshold_m=float(args.jump_threshold_m),
                title_text=info_box_text,
                full_scene_view=bool(args.full_scene_view),
            )
            panels.append((title, panel_img))
            row_manifest.append(
                {
                    "row_index": int(row_idx),
                    "slot_id": slot_id,
                    "source_kind": source_kind,
                    "requested_semantic_label": label,
                    "selected_path_id": path_id,
                    "sade": float(metrics["sade"]),
                    "sfde": float(metrics["sfde"]),
                    "sdc_ade": float(sdc_metrics["sdc_ade"]),
                    "sdc_fde": float(sdc_metrics["sdc_fde"]),
                    "overlay_png": str(panel_png),
                }
            )

        grid_png = scenario_dir / f"{scenario_id}__overlay_grid.png"
        _save_grid(
            panels,
            grid_png,
            columns=int(args.grid_columns),
            tile_size=int(args.tile_size),
        )

        manifest.append(
            {
                "scenario_id": scenario_id,
                "grid_png": str(grid_png),
                "rows": row_manifest,
            }
        )

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"num_scenes": len(manifest), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
