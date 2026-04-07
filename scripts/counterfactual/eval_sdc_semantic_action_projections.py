from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_path_control import compute_path_separability_profile, split_polyline_on_discontinuities
from bmt.counterfactual.sdc_semantic_control import (
    DEFAULT_FAMILY_BACKWARD_SLACK_M,
    DEFAULT_FAMILY_GUIDE_BANDWIDTH_M,
    DEFAULT_FAMILY_HEADING_BETA_RAD,
    DEFAULT_FAMILY_HEADING_DEADBAND_RAD,
    DEFAULT_FAMILY_PATH_DEADBAND_M,
    DEFAULT_FAMILY_TEACHER_TEMPERATURE,
    load_raw_scenario_from_row,
    project_points_to_family_paths_torch,
)
from bmt.dataset.dataset import InfgenDataset
from bmt.models.motionlm_lightning import sanitize_logits_for_loss
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config
from scripts.counterfactual.label_waymax_sdc_path_semantics import (
    AGENT_COLOR,
    CONTEXT_SELECTION_RADIUS_M,
    CROSSWALK_FACE,
    FIG_DPI,
    FIG_SIZE_INCH,
    FINAL_LANE_SHADE,
    LANE_COLOR,
    PAST_STEPS,
    PLOT_RADIUS_M,
    ROAD_COLOR,
    ROUTE_DISCONTINUITY_JUMP_M,
    SDC_ARROW_COLOR,
    SDC_DOT_COLOR,
    SDC_VERTICAL_FRACTION,
    STAY_GUIDE_COLOR,
    START_LANE_SHADE,
    _current_lane_guide,
    _finite_xy_rows,
    _lane_feature_lookup,
    _lane_features,
    _lane_segment_around_anchor,
    _lane_transition_info,
    _nearest_lane_feature_id,
    _select_map_context,
    _select_nearby_agents,
    _select_traffic_lights,
    _split_route_segments,
    _world_to_sdc_up_frame,
)
from scripts.counterfactual.render_waymax_sdc_postsplit_semantics import (
    DEFAULT_RESAMPLE_SPACING_M,
    DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    DEFAULT_SEPARABILITY_SCALE_M,
    _display_gradient_values,
    _resampled_local_path_from_world_segments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect semantic-family path projections and best action tokens on a single SDC control row."
    )
    parser.add_argument("--config", type=str, default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--scenario-id", type=str, default="")
    parser.add_argument("--slot-id", type=str, default="")
    parser.add_argument("--row-index", type=int, default=-1)
    parser.add_argument("--all-scene-slots", action="store_true")
    parser.add_argument("--include-gt", action="store_true")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--crop-radius-m", type=float, default=24.0)
    parser.add_argument("--gradient-display-reference", type=float, default=0.75)
    parser.add_argument("--gradient-display-gamma", type=float, default=1.10)
    parser.add_argument("--resample-spacing-m", type=float, default=DEFAULT_RESAMPLE_SPACING_M)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument("--separability-heading-weight-m", type=float, default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M)
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    config = cfg_from_yaml_file(cfg_path, config)
    data_dir = str(Path(args.data_dir).expanduser().resolve())
    control_index = str(Path(args.control_index).expanduser().resolve())
    config.DATA.TRAINING_DATA_DIR = data_dir
    config.DATA.TEST_DATA_DIR = data_dir
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = control_index
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL = control_index
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX = ""
    config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = ""
    config.DATA.COUNTERFACTUAL_MODE = "sdc_semantic_only"
    config.DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER = False
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    teacher_ckpt = str(args.teacher_ckpt or args.ckpt).strip()
    config.MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT = teacher_ckpt
    return config


def _resolve_device(requested: str) -> torch.device:
    text = str(requested).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _to_torch_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"b", "i", "u", "f", "c"}:
                output[key] = torch.from_numpy(value).to(device)
            else:
                output[key] = value
        else:
            output[key] = value
    return output


def _plot_segmented_polyline(ax, points_xy: np.ndarray, *, label: str | None = None, **kwargs) -> None:
    for idx, segment in enumerate(split_polyline_on_discontinuities(points_xy)):
        if segment.shape[0] < 2:
            continue
        ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)


def _plot_world_map(ax, raw_scenario: Mapping[str, Any], *, center_xy: np.ndarray, radius_m: float):
    for feature in dict(raw_scenario.get("map_features", {})).values():
        polyline = np.asarray(dict(feature).get("polyline", []), dtype=np.float32)
        if polyline.ndim != 2 or polyline.shape[1] < 2:
            continue
        xy = polyline[:, :2]
        if not np.isfinite(xy).all():
            continue
        if np.max(np.linalg.norm(xy - center_xy.reshape(1, 2), axis=-1)) > radius_m * 1.7:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=0.7, alpha=0.8)


def _model_to_world(points_model_xy: np.ndarray, *, map_center_world: np.ndarray, map_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_model_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if float(map_heading_world) == 0.0:
        return (xy + np.asarray(map_center_world, dtype=np.float32).reshape(1, 3)[:, :2]).astype(np.float32)
    c = math.cos(float(map_heading_world))
    s = math.sin(float(map_heading_world))
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(map_center_world[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(map_center_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


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
    map_context = _select_map_context(raw_scenario, center_xy=current_xy, radius_m=CONTEXT_SELECTION_RADIUS_M)
    traffic_lights = _select_traffic_lights(
        raw_scenario,
        center_xy=current_xy,
        radius_m=CONTEXT_SELECTION_RADIUS_M,
        time_index=idx,
    )
    nearby_agents = _select_nearby_agents(
        raw_scenario,
        sdc_id=sdc_id,
        center_xy=current_xy,
        current_idx=idx,
        radius_m=CONTEXT_SELECTION_RADIUS_M,
    )
    return {
        "current_time_index": int(idx),
        "current_xy": np.asarray(current_xy, dtype=np.float64),
        "current_heading": float(current_heading),
        "gt_past_xy": np.asarray(gt_past_xy, dtype=np.float64),
        "map_context": map_context,
        "traffic_lights": traffic_lights,
        "nearby_agents": nearby_agents,
    }


def _scene_rows_for_scenario(rows: Sequence[Mapping[str, Any]], scenario_id: str) -> List[Mapping[str, Any]]:
    target = str(scenario_id)
    return [row for row in rows if str(row.get("scenario_id") or "") == target]


def _path_world_segments(path_world_xy: Any) -> List[np.ndarray]:
    path_world = _finite_xy_rows(np.asarray(path_world_xy, dtype=np.float64))
    return [
        np.asarray(seg, dtype=np.float32)
        for seg in split_polyline_on_discontinuities(path_world, jump_threshold_m=float(ROUTE_DISCONTINUITY_JUMP_M))
        if np.asarray(seg).shape[0] >= 2
    ]


def _selected_path_id_for_row(row: Mapping[str, Any]) -> str:
    selected = row.get("selected_path_id")
    if selected is not None:
        return str(selected)
    metadata = dict(row.get("metadata") or {})
    selected = metadata.get("selected_path_id")
    return "" if selected is None else str(selected)


def _selected_path_world_from_row(row: Mapping[str, Any]) -> tuple[str, List[np.ndarray]]:
    selected_path_id = _selected_path_id_for_row(row)
    family_path_ids = list(row.get("candidate_family_path_ids") or [])
    family_paths_world = list(row.get("candidate_family_resampled_paths_world") or [])
    if selected_path_id:
        for idx, path_id in enumerate(family_path_ids):
            if str(path_id) == selected_path_id and idx < len(family_paths_world):
                return selected_path_id, _path_world_segments(family_paths_world[idx])
    if family_paths_world:
        fallback_id = ""
        if family_path_ids:
            first_id = family_path_ids[0]
            fallback_id = "" if first_id is None else str(first_id)
        return fallback_id, _path_world_segments(family_paths_world[0])
    return selected_path_id, []


def _raw_gt_future_segments(raw_scenario: Mapping[str, Any], row: Mapping[str, Any]) -> List[np.ndarray]:
    sdc_id = str(row["sdc_id"])
    current_idx = int(row["current_time_index"])
    state = dict(raw_scenario["tracks"][sdc_id]["state"])
    position = np.asarray(state.get("position", []), dtype=np.float64)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[0] == 0 or valid.shape[0] == 0:
        return []
    idx = int(np.clip(current_idx, 0, max(0, position.shape[0] - 1)))
    gt_future = _finite_xy_rows(position[idx:][valid[idx:]])
    if gt_future.shape[0] < 2:
        return []
    return [np.asarray(gt_future, dtype=np.float32)]


def _raw_sdc_path_segments(
    raw_scenario: Mapping[str, Any],
    *,
    path_id: str,
    current_xy_world: np.ndarray,
) -> List[np.ndarray]:
    raw_path = dict(raw_scenario.get("sdc_paths", {}).get(str(path_id), {}) or {})
    coords = np.asarray(raw_path.get("polyline_xyz", []), dtype=np.float64)
    valid_mask = np.asarray(raw_path.get("valid", []), dtype=bool).reshape(-1)
    metadata = dict(raw_path.get("metadata", {}) or {})
    road_part_ids = np.asarray(metadata.get("point_road_part_ids", []), dtype=np.int64).reshape(-1)
    if coords.ndim != 2 or coords.shape[1] < 2 or valid_mask.shape[0] == 0:
        return []
    valid_xy = _finite_xy_rows(coords[valid_mask][:, :2])
    if valid_xy.shape[0] < 2:
        return []
    valid_ids = road_part_ids[valid_mask][: valid_xy.shape[0]] if road_part_ids.shape[0] == valid_mask.shape[0] else None
    nearest_idx = int(np.argmin(np.linalg.norm(valid_xy - np.asarray(current_xy_world, dtype=np.float64).reshape(1, 2), axis=-1)))
    trimmed_xy = np.asarray(valid_xy[nearest_idx:], dtype=np.float64)
    if trimmed_xy.shape[0] < 2:
        return []
    if float(np.linalg.norm(trimmed_xy[0] - np.asarray(current_xy_world, dtype=np.float64))) > 1e-3:
        trimmed_xy = np.concatenate([np.asarray(current_xy_world, dtype=np.float64).reshape(1, 2), trimmed_xy], axis=0)
    trimmed_ids = None
    if valid_ids is not None and valid_ids.shape[0] == valid_xy.shape[0]:
        tail_ids = np.asarray(valid_ids[nearest_idx:], dtype=np.int64)
        if tail_ids.size > 0 and trimmed_xy.shape[0] == tail_ids.shape[0] + 1:
            tail_ids = np.concatenate([tail_ids[:1], tail_ids], axis=0)
        trimmed_ids = tail_ids[: trimmed_xy.shape[0]]
    segments = _split_route_segments(trimmed_xy, point_road_part_ids=trimmed_ids)
    return [np.asarray(seg, dtype=np.float32) for seg in segments if np.asarray(seg).shape[0] >= 2]


def _build_family_gradient_render_items(
    *,
    raw_scenario: Mapping[str, Any],
    row: Mapping[str, Any],
    scene_rows: Sequence[Mapping[str, Any]],
    current_xy_world: np.ndarray,
    current_heading_world: float,
    spacing_m: float,
    separability_scale_m: float,
    separability_heading_weight_m: float,
    gradient_display_reference: float,
    gradient_display_gamma: float,
) -> List[Dict[str, Any]]:
    family_path_ids = list(row.get("candidate_family_path_ids") or [])
    family_items: List[Dict[str, Any]] = []
    gt_row = next((scene_row for scene_row in scene_rows if str(scene_row.get("selected_slot_id") or "") == "gt"), None)
    gt_local_path = None
    if gt_row is not None:
        gt_segments_world = _raw_gt_future_segments(raw_scenario, gt_row)
        gt_local_path, _, _ = _resampled_local_path_from_world_segments(
            gt_segments_world,
            center_xy_world=np.asarray(current_xy_world, dtype=np.float32),
            origin_heading_world=float(current_heading_world),
            spacing_m=float(spacing_m),
        )
    for idx, path_id_raw in enumerate(family_path_ids):
        path_id = "" if path_id_raw is None else str(path_id_raw)
        if path_id:
            segments_world = _raw_sdc_path_segments(
                raw_scenario,
                path_id=path_id,
                current_xy_world=np.asarray(current_xy_world, dtype=np.float64),
            )
        else:
            segments_world = _raw_gt_future_segments(raw_scenario, row)
        local_path, _, world_resampled_segments = _resampled_local_path_from_world_segments(
            segments_world,
            center_xy_world=np.asarray(current_xy_world, dtype=np.float32),
            origin_heading_world=float(current_heading_world),
            spacing_m=float(spacing_m),
        )
        family_items.append(
            {
                "path_id": path_id,
                "segments_world": [np.asarray(seg, dtype=np.float32) for seg in world_resampled_segments],
                "local_path": local_path,
                "gradient_values": np.zeros((int(local_path.arc_lengths_m.shape[0]),), dtype=np.float32),
                "separability_values": np.zeros((int(local_path.arc_lengths_m.shape[0]),), dtype=np.float32),
            }
        )

    if str(row.get("selected_slot_id") or "") == "gt":
        competitor_paths: Dict[str, Any] = {}
        for other_row in scene_rows:
            other_slot = str(other_row.get("selected_slot_id") or "")
            if other_slot == "gt":
                continue
            other_path_id = _selected_path_id_for_row(other_row)
            other_segments_world = _raw_sdc_path_segments(
                raw_scenario,
                path_id=other_path_id,
                current_xy_world=np.asarray(current_xy_world, dtype=np.float64),
            )
            other_local_path, _, _ = _resampled_local_path_from_world_segments(
                other_segments_world,
                center_xy_world=np.asarray(current_xy_world, dtype=np.float32),
                origin_heading_world=float(current_heading_world),
                spacing_m=float(spacing_m),
            )
            if other_local_path.waypoints_xy.shape[0] >= 2:
                competitor_paths[other_path_id or other_slot] = other_local_path
        for item in family_items:
            local_path = item["local_path"]
            if local_path.waypoints_xy.shape[0] < 2 or not competitor_paths:
                separability = np.zeros((int(local_path.arc_lengths_m.shape[0]),), dtype=np.float32)
            else:
                sep = compute_path_separability_profile(
                    local_path,
                    competitor_paths,
                    scale_m=float(separability_scale_m),
                    heading_weight_m=float(separability_heading_weight_m),
                )
                separability = np.asarray(sep["separability"], dtype=np.float32).reshape(-1)
            item["separability_values"] = separability
            item["gradient_values"] = _display_gradient_values(
                separability,
                reference=float(gradient_display_reference),
                gamma=float(gradient_display_gamma),
            )
        return family_items

    for item in family_items:
        local_path = item["local_path"]
        if local_path.waypoints_xy.shape[0] < 2 or gt_local_path is None or gt_local_path.waypoints_xy.shape[0] < 2:
            separability = np.zeros((int(local_path.arc_lengths_m.shape[0]),), dtype=np.float32)
        else:
            sep = compute_path_separability_profile(
                local_path,
                {"gt": gt_local_path},
                scale_m=float(separability_scale_m),
                heading_weight_m=float(separability_heading_weight_m),
            )
            separability = np.asarray(sep["separability"], dtype=np.float32).reshape(-1)
        item["separability_values"] = separability
        item["gradient_values"] = _display_gradient_values(
            separability,
            reference=float(gradient_display_reference),
            gamma=float(gradient_display_gamma),
        )
    return family_items


def _add_vlm_colorbar(fig, ax, *, low_label: str = "shared", high_label: str = "distinct") -> None:
    bbox = ax.get_position()
    cax_width = 0.012
    cax_height = min(0.18, bbox.height * 0.42)
    cax_x = max(0.01, bbox.x1 - cax_width - 0.008)
    cax_y = max(0.02, bbox.y0 + 0.02)
    cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])
    norm = Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels([str(low_label), str(high_label)])
    cbar.ax.tick_params(labelsize=6.5, length=0)
    cbar.outline.set_linewidth(0.6)


def _draw_vlm_style_scene_ax(
    *,
    fig,
    ax,
    render_context: Mapping[str, Any],
    highlighted_segments_world: Sequence[np.ndarray],
    highlighted_gradient_values: Sequence[float] | None,
    representative_route_world: np.ndarray | None,
    projection_line_world: np.ndarray | None = None,
    best_arrow_world: np.ndarray | None = None,
    current_marker_world: np.ndarray | None = None,
    sampled_points_world: np.ndarray | None = None,
    sampled_step_labels: Sequence[int] | None = None,
    info_box_text: str = "",
    show_colorbar: bool = True,
) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    gt_past_xy = np.asarray(render_context["gt_past_xy"], dtype=np.float64)
    map_context = render_context["map_context"]
    traffic_lights = render_context["traffic_lights"]
    nearby_agents = render_context["nearby_agents"]

    ax.set_facecolor("#f8fafc")
    lane_lookup = _lane_feature_lookup(map_context)
    lane_features = _lane_features(map_context)
    route_xy_world = _finite_xy_rows(
        np.asarray(representative_route_world if representative_route_world is not None else current_xy.reshape(1, 2), dtype=np.float64)
    )
    transition_info = _lane_transition_info(
        route_xy_world,
        highlighted_metadata={"valid_point_road_part_ids": list()},
        map_context=map_context,
        current_heading=current_heading,
    )
    if highlighted_segments_world:
        last_segment = _finite_xy_rows(np.asarray(list(highlighted_segments_world)[-1], dtype=np.float64))
        if last_segment.shape[0] >= 2:
            transition_info = _lane_transition_info(
                route_xy_world,
                highlighted_metadata={"valid_point_road_part_ids": list()},
                map_context=map_context,
                current_heading=current_heading,
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
        lane_xy_world: np.ndarray | None,
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
        ax.plot(lane_local[:, 0], lane_local[:, 1], color=color, linewidth=linewidth_outer, alpha=alpha_outer, zorder=zorder, solid_capstyle="round")
        ax.plot(lane_local[:, 0], lane_local[:, 1], color=color, linewidth=linewidth_inner, alpha=alpha_inner, zorder=zorder + 0.1, solid_capstyle="round")

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
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color=CROSSWALK_FACE, alpha=0.35, zorder=1)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=ROAD_COLOR, linewidth=2.8, alpha=0.98, zorder=4)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=LANE_COLOR, linewidth=1.1, alpha=0.30, zorder=5)

    for agent in nearby_agents:
        past_xy = _world_to_sdc_up_frame(np.asarray(agent["past_xy"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if past_xy.shape[0] >= 2:
            ax.plot(past_xy[:, 0], past_xy[:, 1], color=AGENT_COLOR, linewidth=1.0, alpha=0.55, zorder=3)
        current_agent_xy = _world_to_sdc_up_frame(np.asarray([agent["current_xy"]], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if current_agent_xy.shape[0] > 0:
            ax.scatter([current_agent_xy[0, 0]], [current_agent_xy[0, 1]], c=AGENT_COLOR, s=14, alpha=0.85, zorder=4)

    for light in traffic_lights:
        stop_xy = _world_to_sdc_up_frame(np.asarray([light["stop_point_xy_world"]], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
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
        ax.plot(stay_lane_local[:, 0], stay_lane_local[:, 1], color=STAY_GUIDE_COLOR, linewidth=2.0, alpha=0.45, linestyle=(0, (5, 4)), zorder=7)

    route_segments = [_finite_xy_rows(np.asarray(seg, dtype=np.float64)) for seg in highlighted_segments_world]
    route_segments = [seg for seg in route_segments if seg.shape[0] >= 2]
    transformed_route_segments = [
        _world_to_sdc_up_frame(seg, center_xy=center_xy, heading_rad=current_heading) for seg in route_segments
    ]
    transformed_route_segments = [seg for seg in transformed_route_segments if seg.shape[0] >= 2]
    gradient_values = None if highlighted_gradient_values is None else np.asarray(list(highlighted_gradient_values), dtype=np.float64).reshape(-1)
    if transformed_route_segments:
        if gradient_values is not None and gradient_values.size > 0:
            norm = Normalize(vmin=0.0, vmax=1.0)
            cmap = plt.cm.viridis
            grad_cursor = 0
            for seg_idx, segment in enumerate(transformed_route_segments):
                ax.plot(segment[:, 0], segment[:, 1], color="#0f172a", linewidth=6.2, alpha=0.18, zorder=8.8, solid_capstyle="round")
                for point_idx in range(1, int(segment.shape[0])):
                    value_idx = min(grad_cursor + point_idx - 1, int(gradient_values.size - 1))
                    color = cmap(norm(float(np.clip(gradient_values[value_idx], 0.0, 1.0))))
                    ax.plot(segment[point_idx - 1 : point_idx + 1, 0], segment[point_idx - 1 : point_idx + 1, 1], color=color, linewidth=5.2, alpha=0.99, zorder=9, solid_capstyle="round")
                grad_cursor += int(segment.shape[0])
                if seg_idx > 0:
                    ax.scatter([segment[0, 0]], [segment[0, 1]], c="#111827", s=30, marker="x", linewidths=1.2, zorder=10)
            if show_colorbar:
                _add_vlm_colorbar(fig, ax)
        else:
            for seg_idx, segment in enumerate(transformed_route_segments):
                ax.plot(segment[:, 0], segment[:, 1], color="#2563eb", linewidth=5.2, alpha=0.98, zorder=9, solid_capstyle="round")
                if seg_idx > 0:
                    ax.scatter([segment[0, 0]], [segment[0, 1]], c="#2563eb", s=30, marker="x", linewidths=1.2, zorder=10)
        final_segment = transformed_route_segments[-1]
        ax.scatter([final_segment[-1, 0]], [final_segment[-1, 1]], c="#2563eb", s=80, edgecolors="white", linewidths=1.1, zorder=10)

    if sampled_points_world is not None and len(np.asarray(sampled_points_world).reshape(-1, 2)) > 0:
        sampled_local = _world_to_sdc_up_frame(np.asarray(sampled_points_world, dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        ax.scatter(sampled_local[:, 0], sampled_local[:, 1], c="#111827", s=26, zorder=11)
        if sampled_step_labels is not None:
            for step, point in zip(sampled_step_labels, sampled_local):
                ax.text(float(point[0]) + 0.6, float(point[1]) + 0.6, str(int(step)), fontsize=8, color="#111827", zorder=11)

    if projection_line_world is not None:
        proj_local = _world_to_sdc_up_frame(np.asarray(projection_line_world, dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if proj_local.shape[0] >= 2:
            ax.plot(proj_local[:, 0], proj_local[:, 1], color="#06b6d4", linewidth=2.6, alpha=0.95, zorder=11)
            ax.scatter([proj_local[-1, 0]], [proj_local[-1, 1]], c="#06b6d4", s=34, edgecolors="white", linewidths=0.8, zorder=12)

    if current_marker_world is not None:
        current_marker_local = _world_to_sdc_up_frame(
            np.asarray([current_marker_world], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if current_marker_local.shape[0] > 0:
            ax.scatter(
                [current_marker_local[0, 0]],
                [current_marker_local[0, 1]],
                c="#111827",
                s=52,
                edgecolors="white",
                linewidths=0.9,
                zorder=12.5,
            )

    if best_arrow_world is not None:
        best_arrow_local = _world_to_sdc_up_frame(np.asarray(best_arrow_world, dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if best_arrow_local.shape[0] >= 2:
            delta = best_arrow_local[1] - best_arrow_local[0]
            ax.arrow(
                float(best_arrow_local[0, 0]),
                float(best_arrow_local[0, 1]),
                float(delta[0]),
                float(delta[1]),
                width=0.18,
                head_width=1.0,
                head_length=1.3,
                color="#d97706",
                length_includes_head=True,
                zorder=13.5,
            )
            ax.scatter([best_arrow_local[1, 0]], [best_arrow_local[1, 1]], c="#d97706", s=28, edgecolors="white", linewidths=0.7, zorder=14)

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

    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _select_row(rows: Sequence[Mapping[str, Any]], *, row_index: int, scenario_id: str, slot_id: str) -> int:
    if int(row_index) >= 0:
        if int(row_index) >= len(rows):
            raise IndexError(f"--row-index {row_index} out of range for {len(rows)} rows")
        return int(row_index)
    if not str(scenario_id).strip():
        raise ValueError("Provide either --row-index or --scenario-id")
    matches = []
    for idx, row in enumerate(rows):
        if str(row.get("scenario_id") or "") != str(scenario_id):
            continue
        if str(slot_id).strip() and str(row.get("selected_slot_id") or "") != str(slot_id):
            continue
        matches.append(idx)
    if not matches:
        raise ValueError(f"No row found for scenario_id={scenario_id!r} slot_id={slot_id!r}")
    if len(matches) > 1 and not str(slot_id).strip():
        slots = [str(rows[idx].get("selected_slot_id") or "") for idx in matches]
        raise ValueError(f"Multiple rows found for scenario_id={scenario_id!r}. Specify --slot-id from {slots}")
    return int(matches[0])


def _sample_step_indices(valid_mask: np.ndarray, *, num_samples: int) -> np.ndarray:
    valid_steps = np.flatnonzero(valid_mask.astype(bool))
    if valid_steps.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if valid_steps.size <= int(num_samples):
        return valid_steps.astype(np.int64)
    picks = np.linspace(0, valid_steps.size - 1, num=max(int(num_samples), 1), dtype=np.int64)
    return valid_steps[np.unique(picks)].astype(np.int64)


def _sample_step_indices_by_arc(valid_mask: np.ndarray, arc_m: np.ndarray, *, num_samples: int) -> np.ndarray:
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    arc = np.asarray(arc_m, dtype=np.float32).reshape(-1)
    if mask.shape[0] == 0 or arc.shape[0] != mask.shape[0]:
        return _sample_step_indices(mask, num_samples=num_samples)
    valid_steps = np.flatnonzero(mask & np.isfinite(arc))
    if valid_steps.size == 0:
        return _sample_step_indices(mask, num_samples=num_samples)
    if valid_steps.size <= int(num_samples):
        return valid_steps.astype(np.int64)
    valid_arc = arc[valid_steps]
    lo = float(np.min(valid_arc))
    hi = float(np.max(valid_arc))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-3:
        return _sample_step_indices(mask, num_samples=num_samples)
    targets = np.linspace(lo, hi, num=max(int(num_samples), 1), dtype=np.float32)
    chosen: List[int] = []
    used: set[int] = set()
    for target in targets.tolist():
        nearest_order = np.argsort(np.abs(valid_arc - float(target)))
        for order_idx in nearest_order.tolist():
            step_idx = int(valid_steps[order_idx])
            if step_idx not in used:
                used.add(step_idx)
                chosen.append(step_idx)
                break
    if len(chosen) < int(num_samples):
        for step_idx in valid_steps.tolist():
            if int(step_idx) not in used:
                used.add(int(step_idx))
                chosen.append(int(step_idx))
            if len(chosen) >= int(num_samples):
                break
    return np.asarray(sorted(chosen), dtype=np.int64)


def _select_scene_row_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_id: str,
    include_gt: bool,
) -> List[int]:
    target = str(scenario_id).strip()
    if not target:
        raise ValueError("--all-scene-slots requires --scenario-id")
    matches: List[tuple[int, str]] = []
    for idx, row in enumerate(rows):
        if str(row.get("scenario_id") or "") != target:
            continue
        slot_id = str(row.get("selected_slot_id") or "")
        if not include_gt and slot_id == "gt":
            continue
        matches.append((idx, slot_id))
    if not matches:
        raise ValueError(
            f"No rows found for scenario_id={target!r} include_gt={include_gt}"
        )

    def _slot_sort_key(item: tuple[int, str]) -> tuple[int, str]:
        _, slot_id = item
        if slot_id == "gt":
            return (0, slot_id)
        if slot_id.startswith("alt_"):
            suffix = slot_id.split("_", 1)[-1]
            if suffix.isdigit():
                return (1, f"{int(suffix):04d}")
        return (2, slot_id)

    matches.sort(key=_slot_sort_key)
    return [idx for idx, _ in matches]


def _build_action_debug_bundle(model, output: Dict[str, Any], semantic_context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    output_logit = output["decoder/output_logit"]
    device = output_logit.device
    dtype = output_logit.dtype
    teacher_logits = model._run_policy_teacher(output)
    if teacher_logits is None:
        raise RuntimeError("Policy teacher is required for semantic-family inspection.")
    teacher_logits = sanitize_logits_for_loss(teacher_logits.to(device=device, dtype=dtype))
    decision_agent_mask = semantic_context["decision_agent_mask"]
    student_logits_sdc = sanitize_logits_for_loss((output_logit * decision_agent_mask[:, None, :, None]).sum(dim=2))
    teacher_logits_sdc = sanitize_logits_for_loss((teacher_logits * decision_agent_mask[:, None, :, None]).sum(dim=2))

    candidate_projection = project_points_to_family_paths_torch(
        semantic_context["sdc_next_pos_candidates_world"],
        family_path_polylines_world=semantic_context["family_paths_world"],
        family_path_mask=semantic_context["family_path_mask"],
        family_path_tangents_world=semantic_context["family_tangents_world"],
        family_path_arc_lengths=semantic_context["family_arc_lengths"],
    )

    position_deadband = float(
        model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_PATH_DEADBAND_M", DEFAULT_FAMILY_PATH_DEADBAND_M)
    )
    heading_deadband = float(
        model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_DEADBAND_RAD", DEFAULT_FAMILY_HEADING_DEADBAND_RAD)
    )
    heading_beta = float(
        model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_BETA_RAD", DEFAULT_FAMILY_HEADING_BETA_RAD)
    )
    backward_slack = float(
        model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_BACKWARD_SLACK_M", DEFAULT_FAMILY_BACKWARD_SLACK_M)
    )
    energy_temperature = float(
        model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_GUIDE_TEMPERATURE", DEFAULT_FAMILY_TEACHER_TEMPERATURE)
    )
    position_weight = float(model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_PATH_PROX_WEIGHT", 1.0))
    heading_weight = float(model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_HEADING_WEIGHT", 0.75))
    backward_weight = float(model.config.MODEL.get("LOCAL_CONTROL_SDC_FAMILY_BACKWARD_WEIGHT", 0.5))

    distance = candidate_projection["nearest_distance"]
    nearest_arc = candidate_projection["nearest_arc"]
    current_arc = semantic_context["current_projection"]["nearest_arc"][:, :, None, :]
    candidate_heading = semantic_context["sdc_candidate_heading_world"][:, :, :, None]
    nearest_heading = candidate_projection["nearest_heading"]
    heading_delta = torch.atan2(
        torch.sin(candidate_heading - nearest_heading),
        torch.cos(candidate_heading - nearest_heading),
    ).abs()
    position_over = torch.relu(distance - position_deadband)
    position_penalty = F.smooth_l1_loss(position_over, torch.zeros_like(position_over), reduction="none")
    heading_over = torch.relu(heading_delta - heading_deadband)
    heading_penalty = F.smooth_l1_loss(
        heading_over,
        torch.zeros_like(heading_over),
        beta=max(heading_beta, 1e-3),
        reduction="none",
    )
    backward_penalty = torch.relu(current_arc - nearest_arc - backward_slack)
    energy_by_action_family = (
        position_weight * position_penalty
        + heading_weight * heading_penalty
        + backward_weight * backward_penalty
    )
    energy_by_family_action = energy_by_action_family.permute(0, 1, 3, 2)
    family_weights = semantic_context["family_weights"]
    weighted_action_energy = (energy_by_family_action * family_weights[:, None, :, None]).sum(dim=2)
    best_token = torch.argmin(weighted_action_energy, dim=-1)
    best_token_energy = weighted_action_energy.gather(dim=-1, index=best_token.unsqueeze(-1)).squeeze(-1)

    best_family_energy = torch.gather(
        energy_by_family_action,
        dim=-1,
        index=best_token[:, :, None, None].expand(-1, -1, energy_by_family_action.shape[2], 1),
    ).squeeze(-1)
    best_family_idx = torch.argmin(best_family_energy, dim=-1)
    best_family_energy_value = best_family_energy.gather(dim=-1, index=best_family_idx.unsqueeze(-1)).squeeze(-1)

    teacher_log_probs = F.log_softmax(teacher_logits_sdc, dim=-1)
    student_log_probs = F.log_softmax(student_logits_sdc, dim=-1)
    family_teacher = torch.softmax(
        teacher_log_probs[:, :, None, :] - (energy_by_family_action / max(energy_temperature, 1e-3)),
        dim=-1,
    )
    family_teacher = torch.nan_to_num(family_teacher, nan=0.0, posinf=0.0, neginf=0.0)
    family_teacher = (family_teacher * family_weights[:, None, :, None]).sum(dim=2)
    family_teacher = family_teacher / family_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    student_probs = torch.softmax(student_logits_sdc, dim=-1)
    base_teacher_probs = torch.softmax(teacher_logits_sdc, dim=-1)
    guide_weight = (semantic_context["current_gate"] * family_weights[:, None, :]).sum(dim=-1)

    return {
        "candidate_projection": candidate_projection,
        "weighted_action_energy": weighted_action_energy,
        "best_token": best_token,
        "best_token_energy": best_token_energy,
        "best_family_idx": best_family_idx,
        "best_family_energy": best_family_energy_value,
        "position_penalty": position_penalty,
        "heading_penalty": heading_penalty,
        "backward_penalty": backward_penalty,
        "family_teacher": family_teacher,
        "student_probs": student_probs,
        "base_teacher_probs": base_teacher_probs,
        "guide_weight": guide_weight,
    }


def _save_overview_plot(
    *,
    out_path: Path,
    render_context: Mapping[str, Any],
    row: Mapping[str, Any],
    family_render_items: Sequence[Mapping[str, Any]],
    sampled_world_xy: np.ndarray,
    sampled_steps: np.ndarray,
):
    fig = plt.figure(figsize=(FIG_SIZE_INCH, FIG_SIZE_INCH), dpi=FIG_DPI)
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    all_segments_world: List[np.ndarray] = []
    all_gradient_values: List[np.ndarray] = []
    representative_route_world = None
    for idx, item in enumerate(family_render_items):
        segments_world = [np.asarray(seg, dtype=np.float64) for seg in list(item.get("segments_world") or []) if np.asarray(seg).shape[0] >= 2]
        if not segments_world:
            continue
        if representative_route_world is None:
            representative_route_world = np.concatenate(segments_world, axis=0)
        all_segments_world.extend(segments_world)
        gradient_src = item.get("gradient_values")
        gradients = np.asarray([] if gradient_src is None else gradient_src, dtype=np.float32).reshape(-1)
        if gradients.size > 0:
            all_gradient_values.append(gradients)
    gradient_values = np.concatenate(all_gradient_values, axis=0) if all_gradient_values else None
    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=render_context,
        highlighted_segments_world=all_segments_world,
        highlighted_gradient_values=gradient_values,
        representative_route_world=representative_route_world,
        sampled_points_world=sampled_world_xy,
        sampled_step_labels=sampled_steps.tolist(),
        info_box_text=(
            f"scene={row['scenario_id']}\n"
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}"
        ),
        show_colorbar=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _save_step_grid(
    *,
    out_path: Path,
    render_context: Mapping[str, Any],
    row: Mapping[str, Any],
    family_render_items: Sequence[Mapping[str, Any]],
    sampled_records: List[Dict[str, Any]],
):
    num = len(sampled_records)
    cols = min(3, max(1, num))
    rows = int(math.ceil(float(num) / float(cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(4.9 * cols, 4.9 * rows), dpi=FIG_DPI)
    axes = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes, sampled_records):
        chosen_family_idx = int(record["best_family_idx"])
        chosen_item = family_render_items[chosen_family_idx]
        highlighted_segments_world = [np.asarray(seg, dtype=np.float64) for seg in list(chosen_item.get("segments_world") or []) if np.asarray(seg).shape[0] >= 2]
        representative_route_world = (
            np.concatenate(highlighted_segments_world, axis=0) if highlighted_segments_world else np.zeros((0, 2), dtype=np.float64)
        )
        info_box_text = (
            f"step={record['step_index']}  token={record['best_token']}\n"
            f"path={record.get('best_family_path_id', chosen_item.get('path_id', ''))}\n"
            f"weightedE={record['weighted_energy']:.3f}  familyE={record['chosen_family_energy']:.3f}\n"
            f"pos={record['position_penalty']:.3f}  head={record['heading_penalty']:.3f}  back={record['backward_penalty']:.3f}\n"
            f"gate={record['guide_weight']:.3f}  student_p={record['student_prob']:.3f}  family_p={record['family_teacher_prob']:.3f}"
        )
        _draw_vlm_style_scene_ax(
            fig=fig,
            ax=ax,
            render_context=render_context,
            highlighted_segments_world=highlighted_segments_world,
            highlighted_gradient_values=np.asarray(
                [] if chosen_item.get("gradient_values") is None else chosen_item.get("gradient_values"),
                dtype=np.float32,
            ),
            representative_route_world=representative_route_world,
            projection_line_world=np.asarray([record["current_world_xy"], record["projection_world_xy"]], dtype=np.float64),
            best_arrow_world=np.asarray([record["current_world_xy"], record["best_next_world_xy"]], dtype=np.float64),
            current_marker_world=np.asarray(record["current_world_xy"], dtype=np.float64),
            info_box_text=info_box_text,
            show_colorbar=True,
        )

    for ax in axes[num:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)


def _scalarize_loss_stat(loss_stat: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in loss_stat.items():
        if hasattr(value, "numel"):
            if value.numel() != 1:
                continue
            output[key] = float(value.detach().cpu().item())
        else:
            output[key] = value
    return output


def _sanitize_slot_name(slot_id: str) -> str:
    text = str(slot_id).strip()
    if not text:
        return "slot"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def _run_single_row(
    *,
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    row_index: int,
    dataset: InfgenDataset,
    model,
    device: torch.device,
    outdir: Path,
) -> Dict[str, Any]:
    row = dict(rows[row_index])
    sample = dataset[row_index]
    batch = dataset.collate_batch([sample])
    batch_torch = _to_torch_device(batch, device=device)

    with torch.no_grad():
        output = model(copy.deepcopy(batch_torch))
        loss, loss_stat = model.get_loss(output)
        semantic_context = model._extract_sdc_semantic_context(output)
    if semantic_context is None:
        raise RuntimeError("Selected row did not produce an sdc_semantic_only context.")

    action_bundle = _build_action_debug_bundle(model, output, semantic_context)
    raw_scenario = load_raw_scenario_from_row(row)
    render_context = _extract_scene_render_context(raw_scenario, row)
    scene_rows = _scene_rows_for_scenario(rows, str(row.get("scenario_id") or ""))
    family_render_items = _build_family_gradient_render_items(
        raw_scenario=raw_scenario,
        row=row,
        scene_rows=scene_rows,
        current_xy_world=np.asarray(render_context["current_xy"], dtype=np.float64),
        current_heading_world=float(render_context["current_heading"]),
        spacing_m=float(args.resample_spacing_m),
        separability_scale_m=float(args.separability_scale_m),
        separability_heading_weight_m=float(args.separability_heading_weight_m),
        gradient_display_reference=float(args.gradient_display_reference),
        gradient_display_gamma=float(args.gradient_display_gamma),
    )
    map_center_world = np.asarray(row.get("candidate_family_map_center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    map_heading_world = float(row.get("candidate_family_map_heading", 0.0) or 0.0)

    family_paths_world = [
        np.asarray(path_xy, dtype=np.float32).reshape(-1, 2)
        for path_xy in list(row.get("candidate_family_resampled_paths_world", []) or [])
    ]
    sdc_valid_by_t = np.asarray(semantic_context["sdc_valid_by_t"][0].detach().cpu(), dtype=bool)
    family_weights = np.asarray(semantic_context["family_weights"][0].detach().cpu(), dtype=np.float32)
    current_projection_idx = np.asarray(semantic_context["current_projection"]["nearest_idx"][0].detach().cpu(), dtype=np.int64)
    current_projection_arc = np.asarray(semantic_context["current_projection"]["nearest_arc"][0].detach().cpu(), dtype=np.float32)
    selected_path_id = _selected_path_id_for_row(row)
    family_path_ids = list(row.get("candidate_family_path_ids") or [])
    selected_family_idx = 0
    if selected_path_id:
        for idx, path_id in enumerate(family_path_ids):
            if str(path_id) == selected_path_id:
                selected_family_idx = int(idx)
                break
    selected_family_idx = int(np.clip(selected_family_idx, 0, max(0, current_projection_arc.shape[1] - 1)))
    sampled_steps = _sample_step_indices_by_arc(
        sdc_valid_by_t,
        current_projection_arc[:, selected_family_idx],
        num_samples=int(args.num_samples),
    )
    if sampled_steps.size == 0:
        raise RuntimeError("No valid SDC steps found for inspection.")

    current_model_xy = np.asarray(semantic_context["sdc_current_pos_world"][0].detach().cpu(), dtype=np.float32)
    current_world_xy = _model_to_world(current_model_xy, map_center_world=map_center_world, map_heading_world=map_heading_world)

    position_penalty = np.asarray(action_bundle["position_penalty"][0].detach().cpu(), dtype=np.float32)
    heading_penalty = np.asarray(action_bundle["heading_penalty"][0].detach().cpu(), dtype=np.float32)
    backward_penalty = np.asarray(action_bundle["backward_penalty"][0].detach().cpu(), dtype=np.float32)
    candidate_projection_idx = np.asarray(action_bundle["candidate_projection"]["nearest_idx"][0].detach().cpu(), dtype=np.int64)
    best_token = np.asarray(action_bundle["best_token"][0].detach().cpu(), dtype=np.int64)
    best_token_energy = np.asarray(action_bundle["best_token_energy"][0].detach().cpu(), dtype=np.float32)
    best_family_idx = np.asarray(action_bundle["best_family_idx"][0].detach().cpu(), dtype=np.int64)
    best_family_energy = np.asarray(action_bundle["best_family_energy"][0].detach().cpu(), dtype=np.float32)
    guide_weight = np.asarray(action_bundle["guide_weight"][0].detach().cpu(), dtype=np.float32)
    student_probs = np.asarray(action_bundle["student_probs"][0].detach().cpu(), dtype=np.float32)
    family_teacher_probs = np.asarray(action_bundle["family_teacher"][0].detach().cpu(), dtype=np.float32)
    best_next_model = np.asarray(semantic_context["sdc_next_pos_candidates_world"][0].detach().cpu(), dtype=np.float32)
    best_next_world_all = np.stack(
        [
            _model_to_world(best_next_model[t], map_center_world=map_center_world, map_heading_world=map_heading_world)
            for t in range(best_next_model.shape[0])
        ],
        axis=0,
    )

    sampled_records: List[Dict[str, Any]] = []
    sampled_points_world = []
    for step_idx in sampled_steps.tolist():
        token_idx = int(best_token[step_idx])
        family_idx = int(best_family_idx[step_idx])
        path_world = family_paths_world[family_idx]
        if path_world.shape[0] == 0:
            continue
        proj_idx = int(np.clip(current_projection_idx[step_idx, family_idx], 0, max(0, path_world.shape[0] - 1)))
        proj_world = np.asarray(path_world[proj_idx], dtype=np.float32)
        next_world = np.asarray(best_next_world_all[step_idx, token_idx], dtype=np.float32)
        curr_world = np.asarray(current_world_xy[step_idx], dtype=np.float32)
        sampled_points_world.append(curr_world)
        token_family_proj_idx = int(
            np.clip(candidate_projection_idx[step_idx, token_idx, family_idx], 0, max(0, path_world.shape[0] - 1))
        )
        sampled_records.append(
            {
                "step_index": int(step_idx),
                "projected_arc_m": float(semantic_context["current_projection"]["nearest_arc"][0, step_idx, family_idx].detach().cpu()),
                "divergence_onset_m": float(semantic_context["family_divergence_onsets"][0, family_idx].detach().cpu()),
                "best_token": token_idx,
                "best_family_idx": family_idx,
                "best_family_path_id": str((row.get("candidate_family_path_ids") or [""])[family_idx]),
                "current_world_xy": curr_world.tolist(),
                "projection_world_xy": proj_world.tolist(),
                "candidate_projection_world_xy": np.asarray(path_world[token_family_proj_idx], dtype=np.float32).tolist(),
                "best_next_world_xy": next_world.tolist(),
                "weighted_energy": float(best_token_energy[step_idx]),
                "chosen_family_energy": float(best_family_energy[step_idx]),
                "position_penalty": float(position_penalty[step_idx, token_idx, family_idx]),
                "heading_penalty": float(heading_penalty[step_idx, token_idx, family_idx]),
                "backward_penalty": float(backward_penalty[step_idx, token_idx, family_idx]),
                "guide_weight": float(guide_weight[step_idx]),
                "student_prob": float(student_probs[step_idx, token_idx]),
                "family_teacher_prob": float(family_teacher_probs[step_idx, token_idx]),
                "family_weight": float(family_weights[family_idx]),
            }
        )

    overview_path = outdir / "scene_overview.png"
    grid_path = outdir / "sampled_action_projection_grid.png"
    summary_path = outdir / "action_projection_summary.json"

    sampled_points_world_arr = np.asarray(sampled_points_world, dtype=np.float32).reshape(-1, 2)
    _save_overview_plot(
        out_path=overview_path,
        render_context=render_context,
        row=row,
        family_render_items=family_render_items,
        sampled_world_xy=sampled_points_world_arr,
        sampled_steps=sampled_steps,
    )
    _save_step_grid(
        out_path=grid_path,
        render_context=render_context,
        row=row,
        family_render_items=family_render_items,
        sampled_records=sampled_records,
    )

    summary = {
        "row_index": int(row_index),
        "scenario_id": str(row.get("scenario_id") or ""),
        "selected_slot_id": str(row.get("selected_slot_id") or ""),
        "requested_semantic_label": str(row.get("requested_semantic_label") or ""),
        "requested_semantic_confidence": float(row.get("requested_semantic_confidence") or 0.0),
        "checkpoint_load_report": None,
        "total_loss": float(loss.detach().cpu().item()),
        "loss_stat": _scalarize_loss_stat(loss_stat),
        "num_sampled_steps": int(len(sampled_records)),
        "sampled_steps": sampled_records,
        "scene_overview_png": str(overview_path),
        "sampled_action_projection_grid_png": str(grid_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode="training")
    device = _resolve_device(args.device)

    model, load_report = load_model_from_checkpoint_forgiving(
        config=config,
        ckpt_path=args.ckpt,
        load_mode=str(args.load_mode),
        strict_state_dict=(str(args.load_mode) == "strict_state_dict"),
        map_location=str(device),
    )
    model = model.to(device)
    model.eval()
    model._trainer = type("TrainerStub", (), {"world_size": 1, "lr_scheduler_configs": None, "optimizers": None})()
    if bool(args.all_scene_slots):
        row_indices = _select_scene_row_indices(
            rows,
            scenario_id=str(args.scenario_id).strip(),
            include_gt=bool(args.include_gt),
        )
        scene_results: List[Dict[str, Any]] = []
        for row_index in row_indices:
            slot_id = _sanitize_slot_name(str(rows[row_index].get("selected_slot_id") or "slot"))
            slot_outdir = outdir / slot_id
            slot_outdir.mkdir(parents=True, exist_ok=True)
            summary = _run_single_row(
                args=args,
                rows=rows,
                row_index=row_index,
                dataset=dataset,
                model=model,
                device=device,
                outdir=slot_outdir,
            )
            summary["checkpoint_load_report"] = load_report if not scene_results else None
            summary_path = slot_outdir / "action_projection_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            scene_results.append(
                {
                    "row_index": int(row_index),
                    "scenario_id": str(summary["scenario_id"]),
                    "selected_slot_id": str(summary["selected_slot_id"]),
                    "requested_semantic_label": str(summary["requested_semantic_label"]),
                    "action_projection_summary_json": str(summary_path),
                    "scene_overview_png": str(summary["scene_overview_png"]),
                    "sampled_action_projection_grid_png": str(summary["sampled_action_projection_grid_png"]),
                }
            )
        manifest = {
            "scenario_id": str(args.scenario_id).strip(),
            "include_gt": bool(args.include_gt),
            "num_processed_slots": int(len(scene_results)),
            "slots": scene_results,
        }
        manifest_path = outdir / "scene_action_projection_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"scene_action_projection_manifest_json": str(manifest_path), "slots": scene_results}, indent=2, sort_keys=True))
    else:
        row_index = _select_row(
            rows,
            row_index=int(args.row_index),
            scenario_id=str(args.scenario_id).strip(),
            slot_id=str(args.slot_id).strip(),
        )
        summary = _run_single_row(
            args=args,
            rows=rows,
            row_index=row_index,
            dataset=dataset,
            model=model,
            device=device,
            outdir=outdir,
        )
        summary["checkpoint_load_report"] = load_report
        summary_path = outdir / "action_projection_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(
            json.dumps(
                {
                    "scene_overview_png": str(summary["scene_overview_png"]),
                    "sampled_action_projection_grid_png": str(summary["sampled_action_projection_grid_png"]),
                    "action_projection_summary_json": str(summary_path),
                    "row_index": int(row_index),
                    "scenario_id": str(summary["scenario_id"]),
                    "selected_slot_id": str(summary["selected_slot_id"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
