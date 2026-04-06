from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from bmt.counterfactual.path_eval_bundle import (
    FRAME_AGENT_RELATIVE_AT_DECISION,
    FRAME_MODEL_OUTPUT,
    FRAME_WORLD,
    ade_fde,
    agent_relative_error_to_anchor,
    agent_relative_pose_to_world,
    anchor_pose_from_control_code,
    agent_relative_xy_to_world_xy,
    build_bundle_inventory,
    build_confusion_and_breakdown,
    branch_candidates_world,
    classify_branch_from_world_pose,
    discover_local_scenario_pkls,
    find_bundle_checkpoint,
    find_materialized_eval_dir,
    gt_final_world_pose_from_raw,
    load_json,
    load_jsonl,
    load_materialized_controls,
    load_model_and_tokenizer_for_bundle,
    load_raw_scenario,
    mean_or_none,
    mode_bucket,
    normalize_predicted_branch,
    nearest_point_on_polyline,
    parse_example_id,
    percentile_dict,
    pose_to_dict,
    preprocess_raw_scenario_for_audit,
    raw_track_world_state,
    restore_world_trajectory,
    rewrite_path_index_rows_for_bundle,
    run_control_variant,
    trajectory_mean_displacement,
    non_target_displacement,
    world_pose_to_agent_relative,
    world_xy_to_agent_relative_xy,
    write_json,
    write_jsonl,
)
from bmt.counterfactual.frame_safe_plotting import (
    PlotValidationError,
    TaggedXYSeries,
    render_tagged_series_collection,
    set_axes_from_points,
)
from scripts.counterfactual.validate_eval_anchor_sanity import run_anchor_sanity


REQUIRED_AUDIT_SPECS = (
    {"scenario_id": "114f6fdcfd17cd14", "agent_id": "2530", "decision_time_idx": 57},
    {"scenario_id": "120334eaa6906bf1", "agent_id": "1186", "decision_time_idx": 25},
    {"scenario_id": "13d80a412371e2", "agent_id": "1679", "decision_time_idx": 40},
    {"scenario_id": "114f6fdcfd17cd14", "agent_id": "2524", "decision_time_idx": 51},
)
REQUESTED_BRANCH_LABELS = ("left", "right", "straight")
MODE_BUCKETS = ("factual", "alternative")
LOCAL_BEV_HALF_EXTENT_M = 90.0
LOCAL_SANITY_LIMIT_M = 200.0
DETERMINISTIC_DECODE_MODE = "argmax"
DETERMINISTIC_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a local path-control eval bundle without rescanning the dataset.")
    parser.add_argument("--bundle-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--load-mode",
        type=str,
        default="forgiving_state_dict",
        choices=("forgiving_state_dict", "strict_state_dict"),
    )
    return parser.parse_args()


def _row_priority(row: Mapping[str, Any]) -> Tuple[int, float, float, str]:
    predicted_label = normalize_predicted_branch(row.get("predicted_branch_label"))
    anchor_distance = float(row.get("final_pose_to_requested_anchor_m") or 0.0)
    branch_margin = float(row.get("branch_score_margin") or 0.0)
    return (
        1 if predicted_label == "other" else 0,
        anchor_distance,
        -branch_margin,
        str(row.get("example_id", "")),
    )


def _required_lookup_key(row: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("scenario_id", "")),
        str(row.get("agent_id", "")),
        int(row.get("decision_time_idx", 0)),
    )


def _find_matching_required_row(rows: Sequence[Dict[str, Any]], spec: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = [
        row for row in rows
        if _required_lookup_key(row)
        == (str(spec["scenario_id"]), str(spec["agent_id"]), int(spec["decision_time_idx"]))
    ]
    if not candidates:
        return None
    factual = [row for row in candidates if str(row.get("mode_bucket")) == "factual"]
    if factual:
        factual.sort(key=_row_priority, reverse=True)
        return factual[0]
    candidates.sort(key=_row_priority, reverse=True)
    return candidates[0]


def _build_selected_manifest(
    *,
    available_rows: Sequence[Dict[str, Any]],
    outdir: Path,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    selected_example_ids: set[str] = set()
    bucket_counts: Counter[Tuple[str, str]] = Counter()

    def add_row(row: Dict[str, Any]) -> None:
        example_id = str(row["example_id"])
        if example_id in selected_example_ids:
            return
        selected.append(row)
        selected_example_ids.add(example_id)
        bucket = (str(row.get("mode_bucket")), str(row.get("requested_branch_label")))
        if bool(row.get("requested_branch_match")) is False and bucket[0] in MODE_BUCKETS and bucket[1] in REQUESTED_BRANCH_LABELS:
            bucket_counts[bucket] += 1

    for spec in REQUIRED_AUDIT_SPECS:
        row = _find_matching_required_row(available_rows, spec)
        if row is not None:
            add_row(row)

    for current_mode in MODE_BUCKETS:
        for requested_label in REQUESTED_BRANCH_LABELS:
            needed = 2 - bucket_counts[(current_mode, requested_label)]
            if needed <= 0:
                continue
            candidates = [
                row for row in available_rows
                if str(row.get("mode_bucket")) == current_mode
                and str(row.get("requested_branch_label")) == requested_label
                and bool(row.get("requested_branch_match")) is False
                and str(row.get("example_id")) not in selected_example_ids
            ]
            candidates.sort(key=_row_priority, reverse=True)
            for candidate in candidates[:needed]:
                add_row(candidate)

    manifest: List[Dict[str, Any]] = []
    for row in selected:
        manifest.append(
            {
                "example_id": str(row["example_id"]),
                "scenario_id": str(row["scenario_id"]),
                "agent_id": str(row["agent_id"]),
                "decision_time_idx": int(row["decision_time_idx"]),
                "requested_branch_label": row.get("requested_branch_label"),
                "factual_or_alternative": str(row["mode_bucket"]),
                "selected_mode": str(row["mode"]),
                "legacy_predicted_branch_label": row.get("predicted_branch_label"),
                "legacy_requested_branch_match": row.get("requested_branch_match"),
                "legacy_final_pose_to_requested_anchor_m": row.get("final_pose_to_requested_anchor_m"),
                "local_scenario_pkl": row.get("local_scenario_pkl"),
                "local_materialized_eval_input": row.get("local_materialized_eval_input"),
                "has_sweep_png": bool(row.get("has_sweep_png")),
                "control_sweep_png": row.get("control_sweep_png"),
            }
        )
    write_json(outdir / "selected_examples_manifest.json", manifest)
    return manifest


def _agent_relative_xy(xy_world: np.ndarray, *, agent_pose_world: Mapping[str, Any]) -> np.ndarray:
    xy = np.asarray(xy_world, dtype=np.float64)
    dx = xy[..., 0] - float(agent_pose_world.get("x", 0.0))
    dy = xy[..., 1] - float(agent_pose_world.get("y", 0.0))
    heading = float(agent_pose_world.get("heading", 0.0))
    c = math.cos(heading)
    s = math.sin(heading)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    return np.stack([x_rel, y_rel], axis=-1)


def _set_axis_bounds(ax, points: np.ndarray, *, padding: float = 10.0) -> None:
    if points.size == 0:
        ax.set_xlim(-padding, padding)
        ax.set_ylim(-padding, padding)
        return
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    center = (min_xy + max_xy) / 2.0
    half_extent = max(float(np.max(max_xy - min_xy)) / 2.0 + padding, padding)
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)


def _variant_style(mode: str) -> Tuple[str, str]:
    if mode == "no_control":
        return "#111827", "no_control"
    if mode == "factual":
        return "#e11d48", "factual"
    alt_palette = ["#0ea5e9", "#22c55e", "#f97316", "#7c3aed", "#14b8a6", "#f59e0b"]
    if mode.startswith("alternative_"):
        try:
            rank = int(mode.split("_", 1)[1])
        except Exception:
            rank = 0
        return alt_palette[rank % len(alt_palette)], mode
    return "#56b4e9", mode


def _variant_linestyle(mode: str) -> str:
    if mode == "no_control":
        return "--"
    if mode == "factual":
        return "-"
    alt_styles = ["-.", ":", "--", "-.", ":", "--"]
    if mode.startswith("alternative_"):
        try:
            rank = int(mode.split("_", 1)[1])
        except Exception:
            rank = 0
        return alt_styles[rank % len(alt_styles)]
    return "-"


def _track_pose_at_index(raw_scenario: Mapping[str, Any], *, track_id: str, time_index: int) -> Optional[Dict[str, float]]:
    state = raw_track_world_state(raw_scenario, track_id=str(track_id))
    valid = np.asarray(state["valid"], dtype=bool)
    if valid.size == 0:
        return None
    idx = int(np.clip(int(time_index), 0, valid.shape[0] - 1))
    if not bool(valid[idx]):
        valid_before = np.flatnonzero(valid[: idx + 1])
        if valid_before.size == 0:
            return None
        idx = int(valid_before[-1])
    position = np.asarray(state["position"], dtype=np.float64)
    heading = np.asarray(state["heading"], dtype=np.float64)
    return {
        "x": float(position[idx, 0]),
        "y": float(position[idx, 1]),
        "heading": float(heading[idx]),
        "index": int(idx),
    }


def _feature_xy_world(feature: Mapping[str, Any]) -> np.ndarray:
    if "polyline" in feature:
        xy = np.asarray(feature["polyline"], dtype=np.float64)
    elif "polygon" in feature:
        xy = np.asarray(feature["polygon"], dtype=np.float64)
    else:
        return np.zeros((0, 2), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(xy[:, :2], dtype=np.float64)


def _trim_xy_to_radius(xy_world: np.ndarray, *, center_xy: Sequence[float], radius_m: float) -> np.ndarray:
    xy = np.asarray(xy_world, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    mask = np.linalg.norm(xy[:, :2] - center[None, :], axis=-1) <= float(radius_m)
    trimmed = xy[mask]
    if trimmed.shape[0] >= 2:
        return np.asarray(trimmed[:, :2], dtype=np.float64)
    if bool(mask.any()):
        return np.asarray(trimmed[:, :2], dtype=np.float64)
    return np.zeros((0, 2), dtype=np.float64)


def _select_map_context(
    raw_scenario: Mapping[str, Any],
    *,
    center_xy: Sequence[float],
    radius_m: float,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {
        "lane_centerlines": [],
        "road_boundaries": [],
        "crosswalks": [],
    }
    for feature_id, feature in sorted(raw_scenario.get("map_features", {}).items(), key=lambda item: str(item[0])):
        feature_type = str(feature.get("type", ""))
        xy = _feature_xy_world(feature)
        trimmed = _trim_xy_to_radius(xy, center_xy=center_xy, radius_m=float(radius_m))
        if trimmed.shape[0] == 0:
            continue
        payload = {
            "feature_id": str(feature_id),
            "feature_type": feature_type,
            "xy_world": trimmed,
        }
        if feature_type.startswith("LANE_") or feature_type == "DRIVEWAY":
            grouped["lane_centerlines"].append(payload)
        elif feature_type.startswith("ROAD_EDGE") or feature_type.startswith("ROAD_LINE"):
            grouped["road_boundaries"].append(payload)
        elif feature_type == "CROSSWALK":
            grouped["crosswalks"].append(payload)
    return grouped


def _dynamic_light_state_at_index(light_obj: Mapping[str, Any], time_index: int) -> Optional[str]:
    state_seq = list(dict(light_obj.get("state", {})).get("object_state", []))
    if not state_seq:
        return None
    idx = int(np.clip(int(time_index), 0, len(state_seq) - 1))
    if state_seq[idx] is not None:
        return str(state_seq[idx])
    for fallback in range(idx, -1, -1):
        if state_seq[fallback] is not None:
            return str(state_seq[fallback])
    return None


def _select_traffic_light_context(
    raw_scenario: Mapping[str, Any],
    *,
    center_xy: Sequence[float],
    time_index: int,
    focus_light_id: Optional[str],
    radius_m: float,
) -> List[Dict[str, Any]]:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    lights: List[Dict[str, Any]] = []
    for light_id, light_obj in sorted(raw_scenario.get("dynamic_map_states", {}).items(), key=lambda item: str(item[0])):
        if str(light_obj.get("type", "")) != "TRAFFIC_LIGHT":
            continue
        stop_point = np.asarray(light_obj.get("stop_point", []), dtype=np.float64)
        if stop_point.size < 2:
            continue
        dist = float(np.linalg.norm(stop_point[:2] - center))
        if str(light_id) != str(focus_light_id) and dist > float(radius_m):
            continue
        lights.append(
            {
                "light_id": str(light_id),
                "stop_point_xy_world": np.asarray(stop_point[:2], dtype=np.float64),
                "state_at_decision": _dynamic_light_state_at_index(light_obj, time_index=int(time_index)),
                "is_focus": bool(str(light_id) == str(focus_light_id)),
                "distance_to_center_m": dist,
            }
        )
    return lights


def _select_nearby_agents(
    raw_scenario: Mapping[str, Any],
    *,
    center_xy: Sequence[float],
    current_time_idx: int,
    radius_m: float,
    past_steps: int,
    exclude_track_id: str,
    max_agents: int = 18,
) -> List[Dict[str, Any]]:
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    rows: List[Dict[str, Any]] = []
    for track_id, track in sorted(raw_scenario.get("tracks", {}).items(), key=lambda item: str(item[0])):
        if str(track_id) == str(exclude_track_id):
            continue
        state = dict(track.get("state", {}))
        valid = np.asarray(state.get("valid", []), dtype=bool)
        position = np.asarray(state.get("position", []), dtype=np.float64)
        if valid.size == 0 or position.ndim != 2 or position.shape[0] == 0:
            continue
        idx = int(np.clip(int(current_time_idx), 0, valid.shape[0] - 1))
        if not bool(valid[idx]):
            continue
        current_xy = np.asarray(position[idx, :2], dtype=np.float64)
        dist = float(np.linalg.norm(current_xy - center))
        if dist > float(radius_m):
            continue
        start_idx = max(0, idx - int(past_steps))
        past_mask = valid[start_idx : idx + 1]
        past_xy = np.asarray(position[start_idx : idx + 1, :2], dtype=np.float64)[past_mask]
        rows.append(
            {
                "track_id": str(track_id),
                "object_type": str(track.get("type", "")),
                "current_xy_world": current_xy,
                "past_xy_world": past_xy,
                "distance_to_center_m": dist,
            }
        )
    rows.sort(key=lambda item: (float(item["distance_to_center_m"]), str(item["track_id"])))
    return rows[: int(max_agents)]


def _branch_color(branch_label: Optional[str]) -> str:
    if branch_label == "left":
        return "#d55e00"
    if branch_label == "straight":
        return "#0072b2"
    if branch_label == "right":
        return "#009e73"
    return "#7b3294"


def _heading_arrow_xy(pose: Mapping[str, Any], *, length_m: float = 6.0) -> np.ndarray:
    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    heading = float(pose.get("heading", 0.0))
    return np.asarray(
        [
            [x, y],
            [x + float(length_m) * math.cos(heading), y + float(length_m) * math.sin(heading)],
        ],
        dtype=np.float64,
    )


def _local_clip_xy(xy_local: np.ndarray, *, limit_m: float = LOCAL_BEV_HALF_EXTENT_M + 18.0) -> np.ndarray:
    xy = np.asarray(xy_local, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    mask = (np.abs(xy[:, 0]) <= float(limit_m)) & (np.abs(xy[:, 1]) <= float(limit_m))
    clipped = xy[mask]
    if clipped.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(clipped[:, :2], dtype=np.float64)


def _placeholder_plot(output_path: Path, *, message: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_axis_off()
    ax.text(0.5, 0.5, str(message), ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_frame_safe_plot(
    *,
    output_path: Path,
    example_id: str,
    plot_name: str,
    expected_frame: str,
    title: str,
    xlabel: str,
    ylabel: str,
    series_list: Sequence[TaggedXYSeries],
    plot_failures: List[Dict[str, Any]],
    fixed_half_extent: Optional[float] = None,
    padding: float = 10.0,
    extra_text_lines: Sequence[str] = (),
) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {
            "rendered": False,
            "extent": None,
            "local_frame_sanity_passed": False,
            "frame": expected_frame,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    try:
        plotted = render_tagged_series_collection(
            ax,
            series_list=series_list,
            expected_frame=expected_frame,
            example_id=example_id,
            plot_name=plot_name,
            failures=plot_failures,
            local_limit_abs_m=LOCAL_SANITY_LIMIT_M,
        )
        extent = set_axes_from_points(ax, plotted, padding=padding, fixed_half_extent=fixed_half_extent)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if extra_text_lines:
            ax.text(
                0.01,
                0.99,
                "\n".join(str(line) for line in extra_text_lines if line),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = {}
            for handle, label in zip(handles, labels):
                if label:
                    dedup.setdefault(label, handle)
            if dedup:
                ax.legend(
                    dedup.values(),
                    dedup.keys(),
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    fontsize=7,
                )
        fig.tight_layout()
        fig.savefig(output_path, dpi=170)
        plt.close(fig)
        return {
            "rendered": True,
            "extent": extent,
            "local_frame_sanity_passed": True if expected_frame == FRAME_AGENT_RELATIVE_AT_DECISION else None,
            "frame": expected_frame,
        }
    except PlotValidationError as exc:
        plt.close(fig)
        _placeholder_plot(output_path, message=f"{plot_name} failed\n{exc}")
        return {
            "rendered": False,
            "extent": None,
            "local_frame_sanity_passed": False if expected_frame == FRAME_AGENT_RELATIVE_AT_DECISION else None,
            "frame": expected_frame,
        }


def _plot_local_target_frame(
    ax,
    *,
    example_id: str,
    requested_branch_label: Optional[str],
    gt_xy_world: np.ndarray,
    gt_valid: np.ndarray,
    branch_candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    focus_mode: str,
    focus_anchor_rel: Optional[Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
) -> None:
    valid_gt_world = np.asarray(gt_xy_world)[np.asarray(gt_valid, dtype=bool)]
    valid_gt_rel = _agent_relative_xy(valid_gt_world, agent_pose_world=agent_pose_world) if valid_gt_world.size else np.zeros((0, 2))
    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            rel = _agent_relative_xy(polyline[:, :2], agent_pose_world=agent_pose_world)
            ax.plot(rel[:, 0], rel[:, 1], color="#c0c0c0", linewidth=1.0, alpha=0.7)
    for mode, result in variants.items():
        color, label = _variant_style(mode)
        linestyle = _variant_linestyle(mode)
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size == 0:
            continue
        rel = _agent_relative_xy(xy, agent_pose_world=agent_pose_world)
        ax.plot(rel[:, 0], rel[:, 1], color=color, linewidth=2.0 if mode == focus_mode else 1.4, label=label)
    if valid_gt_rel.size:
        ax.plot(valid_gt_rel[:, 0], valid_gt_rel[:, 1], color="#000000", linewidth=2.0, linestyle="--", label="GT future")
        ax.scatter(valid_gt_rel[-1, 0], valid_gt_rel[-1, 1], color="#009e73", s=45, label="GT final")
    ax.scatter([0.0], [0.0], color="#7b3294", s=45, label="decision pose")
    if focus_anchor_rel is not None:
        ax.scatter(
            [float(focus_anchor_rel["x"])],
            [float(focus_anchor_rel["y"])],
            color="#d55e00",
            s=50,
            label=f"requested anchor ({requested_branch_label})",
        )
    ax.set_title(f"{example_id}\nlocal_target_frame [{focus_mode}]")
    ax.set_xlabel("x_rel (m)")
    ax.set_ylabel("y_rel (m)")
    points = []
    if valid_gt_rel.size:
        points.append(valid_gt_rel)
    if focus_anchor_rel is not None:
        points.append(np.asarray([[float(focus_anchor_rel["x"]), float(focus_anchor_rel["y"])]], dtype=np.float64))
    for result in variants.values():
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size:
            points.append(_agent_relative_xy(xy, agent_pose_world=agent_pose_world))
    if points:
        _set_axis_bounds(ax, np.concatenate(points, axis=0), padding=12.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)


def _plot_world_frame(
    ax,
    *,
    example_id: str,
    requested_branch_label: Optional[str],
    gt_xy_world: np.ndarray,
    gt_valid: np.ndarray,
    branch_candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    focus_mode: str,
    focus_anchor_world: Optional[Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
) -> None:
    valid_gt_world = np.asarray(gt_xy_world)[np.asarray(gt_valid, dtype=bool)]
    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#c0c0c0", linewidth=1.0, alpha=0.7)
    for mode, result in variants.items():
        color, label = _variant_style(mode)
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size == 0:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0 if mode == focus_mode else 1.4, label=label)
    if valid_gt_world.size:
        ax.plot(valid_gt_world[:, 0], valid_gt_world[:, 1], color="#000000", linewidth=2.0, linestyle="--", label="GT future")
        ax.scatter(valid_gt_world[-1, 0], valid_gt_world[-1, 1], color="#009e73", s=45, label="GT final")
    ax.scatter([float(agent_pose_world["x"])], [float(agent_pose_world["y"])], color="#7b3294", s=45, label="decision pose")
    if focus_anchor_world is not None:
        ax.scatter(
            [float(focus_anchor_world["x"])],
            [float(focus_anchor_world["y"])],
            color="#d55e00",
            s=50,
            label=f"requested anchor ({requested_branch_label})",
        )
    ax.set_title(f"{example_id}\nworld_frame [{focus_mode}]")
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    points = []
    if valid_gt_world.size:
        points.append(valid_gt_world)
    if focus_anchor_world is not None:
        points.append(np.asarray([[float(focus_anchor_world["x"]), float(focus_anchor_world["y"])]], dtype=np.float64))
    points.append(np.asarray([[float(agent_pose_world["x"]), float(agent_pose_world["y"])]], dtype=np.float64))
    for result in variants.values():
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size:
            points.append(xy)
    _set_axis_bounds(ax, np.concatenate(points, axis=0), padding=15.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)


def _plot_target_only_delta(
    ax,
    *,
    example_id: str,
    variants: Mapping[str, Mapping[str, Any]],
    focus_mode: str,
    focus_effect_rows: Mapping[str, Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
) -> None:
    for mode, result in variants.items():
        color, label = _variant_style(mode)
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size == 0:
            continue
        rel = _agent_relative_xy(xy, agent_pose_world=agent_pose_world)
        legend = label
        if mode in focus_effect_rows:
            disp = focus_effect_rows[mode].get("target_mean_displacement_vs_no_control")
            if disp is not None:
                legend = f"{label} ({disp:.2f}m)"
        ax.plot(
            rel[:, 0],
            rel[:, 1],
            color=color,
            linewidth=2.8 if mode == focus_mode else 2.0,
            linestyle=linestyle,
            alpha=0.98 if mode == focus_mode else 0.94,
            label=legend,
        )
    ax.set_title(f"{example_id}\ntarget_only_delta")
    ax.set_xlabel("x_rel (m)")
    ax.set_ylabel("y_rel (m)")
    points = []
    for result in variants.values():
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy.size:
            points.append(_agent_relative_xy(xy, agent_pose_world=agent_pose_world))
    if points:
        _set_axis_bounds(ax, np.concatenate(points, axis=0), padding=10.0)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)


def _plot_anchor_vs_gt(
    ax,
    *,
    example_id: str,
    focus_mode: str,
    gt_final_world: Mapping[str, Any],
    focus_anchor_world: Optional[Mapping[str, Any]],
    distance_m: Optional[float],
    heading_error_rad: Optional[float],
) -> None:
    ax.scatter([float(gt_final_world["x"])], [float(gt_final_world["y"])], color="#009e73", s=55, label="GT final")
    points = [np.asarray([[float(gt_final_world["x"]), float(gt_final_world["y"])]], dtype=np.float64)]
    if focus_anchor_world is not None:
        ax.scatter([float(focus_anchor_world["x"])], [float(focus_anchor_world["y"])], color="#d55e00", s=55, label="requested anchor")
        ax.plot(
            [float(gt_final_world["x"]), float(focus_anchor_world["x"])],
            [float(gt_final_world["y"]), float(focus_anchor_world["y"])],
            color="#999999",
            linewidth=1.2,
            linestyle="--",
        )
        points.append(np.asarray([[float(focus_anchor_world["x"]), float(focus_anchor_world["y"])]], dtype=np.float64))
    _set_axis_bounds(ax, np.concatenate(points, axis=0), padding=12.0)
    ax.set_aspect("equal", adjustable="box")
    title = f"{example_id}\nanchor_vs_gt [{focus_mode}]"
    if distance_m is not None and heading_error_rad is not None:
        title += f" | dist={distance_m:.1f}m dh={heading_error_rad:.2f}rad"
    ax.set_title(title)
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.legend(loc="best", fontsize=7)


def _compose_contact_sheet(output_path: Path, image_paths: Sequence[Path]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, image_path in zip(axes.flat, image_paths[:4]):
        ax.imshow(mpimg.imread(image_path))
        ax.set_axis_off()
        ax.set_title(image_path.stem, fontsize=9)
    for ax in axes.flat[len(image_paths[:4]):]:
        ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _render_visual_bundle(
    *,
    output_dir: Path,
    example_id: str,
    requested_branch_label: Optional[str],
    gt_xy_world: np.ndarray,
    gt_valid: np.ndarray,
    gt_final_world: Mapping[str, Any],
    branch_candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    focus_mode: str,
    focus_anchor_rel: Optional[Mapping[str, Any]],
    focus_anchor_world: Optional[Mapping[str, Any]],
    focus_distance_m: Optional[float],
    focus_heading_error_rad: Optional[float],
    effect_rows: Mapping[str, Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    local_path = output_dir / "local_target_frame.png"
    world_path = output_dir / "world_frame.png"
    target_delta_path = output_dir / "target_only_delta.png"
    anchor_gt_path = output_dir / "anchor_vs_gt.png"

    fig_local, ax_local = plt.subplots(figsize=(6.0, 6.0))
    _plot_local_target_frame(
        ax_local,
        example_id=example_id,
        requested_branch_label=requested_branch_label,
        gt_xy_world=gt_xy_world,
        gt_valid=gt_valid,
        branch_candidates=branch_candidates,
        variants=variants,
        focus_mode=focus_mode,
        focus_anchor_rel=focus_anchor_rel,
        agent_pose_world=agent_pose_world,
    )
    fig_local.tight_layout()
    fig_local.savefig(local_path, dpi=160)
    plt.close(fig_local)

    fig_world, ax_world = plt.subplots(figsize=(6.0, 6.0))
    _plot_world_frame(
        ax_world,
        example_id=example_id,
        requested_branch_label=requested_branch_label,
        gt_xy_world=gt_xy_world,
        gt_valid=gt_valid,
        branch_candidates=branch_candidates,
        variants=variants,
        focus_mode=focus_mode,
        focus_anchor_world=focus_anchor_world,
        agent_pose_world=agent_pose_world,
    )
    fig_world.tight_layout()
    fig_world.savefig(world_path, dpi=160)
    plt.close(fig_world)

    fig_delta, ax_delta = plt.subplots(figsize=(6.0, 6.0))
    _plot_target_only_delta(
        ax_delta,
        example_id=example_id,
        variants=variants,
        focus_mode=focus_mode,
        focus_effect_rows=effect_rows,
        agent_pose_world=agent_pose_world,
    )
    fig_delta.tight_layout()
    fig_delta.savefig(target_delta_path, dpi=160)
    plt.close(fig_delta)

    fig_anchor, ax_anchor = plt.subplots(figsize=(6.0, 6.0))
    _plot_anchor_vs_gt(
        ax_anchor,
        example_id=example_id,
        focus_mode=focus_mode,
        gt_final_world=gt_final_world,
        focus_anchor_world=focus_anchor_world,
        distance_m=focus_distance_m,
        heading_error_rad=focus_heading_error_rad,
    )
    fig_anchor.tight_layout()
    fig_anchor.savefig(anchor_gt_path, dpi=160)
    plt.close(fig_anchor)

    _compose_contact_sheet(output_dir / "contact_sheet.png", [local_path, world_path, target_delta_path, anchor_gt_path])


def _build_example_variant_bundle(
    *,
    manifest_item: Mapping[str, Any],
    bundle_root: Path,
    config: Any,
    model: Any,
    tokenizer: Any,
    scenario_cache: Dict[str, Dict[str, Any]],
    base_sample_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    local_scenario_pkl = str(manifest_item["local_scenario_pkl"])
    materialized_dir = str(manifest_item["local_materialized_eval_input"])
    raw_scenario = scenario_cache.get(local_scenario_pkl)
    if raw_scenario is None:
        raw_scenario = load_raw_scenario(local_scenario_pkl)
        scenario_cache[local_scenario_pkl] = raw_scenario
    base_sample = base_sample_cache.get(local_scenario_pkl)
    if base_sample is None:
        base_sample = preprocess_raw_scenario_for_audit(raw_scenario, config=config, tokenizer=tokenizer)
        base_sample_cache[local_scenario_pkl] = base_sample

    materialized = load_materialized_controls(materialized_dir)
    factual_control = materialized["factual_control_code"]
    alternatives = list(materialized["alternative_control_codes"])
    branch_candidates = list(materialized["branch_candidates"])
    agent_pose_world = dict(factual_control.get("debug", {}).get("agent_pose_at_decision", {})) if factual_control else {}
    provenance = dict(factual_control.get("debug", {}).get("source_provenance", {})) if factual_control else {}
    current_time_idx = int(provenance.get("current_time_index_global", manifest_item["decision_time_idx"]))
    decision_time_idx = int(provenance.get("decision_time_index_global", manifest_item["decision_time_idx"]))
    current_pose_world = _track_pose_at_index(raw_scenario, track_id=str(manifest_item["agent_id"]), time_index=current_time_idx)
    decision_pose_world = _track_pose_at_index(raw_scenario, track_id=str(manifest_item["agent_id"]), time_index=decision_time_idx)
    local_intervention = materialized.get("local_intervention_train_view") or {}
    stop_point_xy = (
        np.asarray(local_intervention.get("context", {}).get("stop_point_xy", []), dtype=np.float64)[:2]
        if local_intervention
        else np.zeros((0,), dtype=np.float64)
    )
    focus_light_id = None
    if factual_control:
        focus_light_id = str(factual_control.get("debug", {}).get("light_id", "")).strip() or None
    if focus_light_id is None and local_intervention:
        focus_light_id = str(local_intervention.get("context", {}).get("traffic_light_id", "")).strip() or None
    if focus_light_id is None:
        focus_light_id = str(parse_example_id(str(manifest_item["example_id"])).get("light_id") or "").strip() or None
    center_xy = (
        np.asarray([float(agent_pose_world.get("x", 0.0)), float(agent_pose_world.get("y", 0.0))], dtype=np.float64)
        if agent_pose_world
        else (np.asarray(stop_point_xy, dtype=np.float64) if stop_point_xy.size >= 2 else np.zeros((2,), dtype=np.float64))
    )
    map_context = _select_map_context(raw_scenario, center_xy=center_xy, radius_m=110.0)
    traffic_lights = _select_traffic_light_context(
        raw_scenario,
        center_xy=center_xy,
        time_index=decision_time_idx,
        focus_light_id=focus_light_id,
        radius_m=95.0,
    )
    nearby_agents = _select_nearby_agents(
        raw_scenario,
        center_xy=center_xy,
        current_time_idx=current_time_idx,
        radius_m=85.0,
        past_steps=8,
        exclude_track_id=str(manifest_item["agent_id"]),
    )

    variants: Dict[str, Dict[str, Any]] = {}
    variants["no_control"] = run_control_variant(
        base_sample=base_sample,
        scenario_id=str(manifest_item["scenario_id"]),
        mode="no_control",
        control_code=None,
        model=model,
        tokenizer=tokenizer,
        sampling_method=DETERMINISTIC_DECODE_MODE,
        temperature=1.0,
        topp=1.0,
        seed=DETERMINISTIC_SEED,
        deterministic_agent_ids=True,
    )
    if factual_control is not None:
        variants["factual"] = run_control_variant(
            base_sample=base_sample,
            scenario_id=str(manifest_item["scenario_id"]),
            mode="factual",
            control_code=factual_control,
            model=model,
            tokenizer=tokenizer,
            sampling_method=DETERMINISTIC_DECODE_MODE,
            temperature=1.0,
            topp=1.0,
            seed=DETERMINISTIC_SEED,
            deterministic_agent_ids=True,
        )
    for alternative in alternatives:
        rank = int(alternative.get("alternative_rank", 0))
        control_code = dict(alternative.get("control_code", {}))
        if not control_code:
            continue
        variants[f"alternative_{rank}"] = run_control_variant(
            base_sample=base_sample,
            scenario_id=str(manifest_item["scenario_id"]),
            mode=f"alternative_{rank}",
            control_code=control_code,
            model=model,
            tokenizer=tokenizer,
            sampling_method=DETERMINISTIC_DECODE_MODE,
            temperature=1.0,
            topp=1.0,
            seed=DETERMINISTIC_SEED,
            deterministic_agent_ids=True,
        )

    no_control = variants["no_control"]
    no_control_branch = classify_branch_from_world_pose(no_control["target_final_pose_world"], branch_candidates)

    for mode, result in variants.items():
        corrected_branch = classify_branch_from_world_pose(result["target_final_pose_world"], branch_candidates)
        result["corrected_branch"] = corrected_branch
        result["requested_branch_match"] = (
            bool(corrected_branch.get("branch_label") == result.get("requested_branch_label"))
            if result.get("requested_branch_label") is not None
            else None
        )
        requested_anchor = result.get("requested_anchor")
        if requested_anchor is not None and agent_pose_world:
            corrected_anchor_distance, corrected_heading_error, final_rel_pose = agent_relative_error_to_anchor(
                pose_world=result["target_final_pose_world"],
                anchor_rel=requested_anchor,
                agent_pose_world=agent_pose_world,
            )
            requested_anchor_world = agent_relative_pose_to_world(requested_anchor, agent_pose_world=agent_pose_world)
        else:
            corrected_anchor_distance = None
            corrected_heading_error = None
            final_rel_pose = None
            requested_anchor_world = None
        result["corrected_anchor_distance_m"] = corrected_anchor_distance
        result["corrected_heading_error_rad"] = corrected_heading_error
        result["target_final_pose_rel_to_decision"] = pose_to_dict(final_rel_pose) if final_rel_pose is not None else None
        result["requested_anchor_world"] = pose_to_dict(requested_anchor_world) if requested_anchor_world is not None else None
        result["no_control_branch_changed"] = (
            None if mode == "no_control" else bool(corrected_branch.get("branch_label") != no_control_branch.get("branch_label"))
        )
        result["legacy_frame_branch_mismatch"] = True
        result["legacy_frame_anchor_mismatch"] = bool(result.get("requested_anchor") is not None)

    gt_state = raw_track_world_state(raw_scenario, track_id=str(manifest_item["agent_id"]))
    return {
        "raw_scenario": raw_scenario,
        "gt_state": gt_state,
        "branch_candidates": branch_candidates,
        "variants": variants,
        "agent_pose_world": agent_pose_world,
        "current_pose_world": current_pose_world,
        "decision_pose_world": decision_pose_world,
        "current_time_idx": current_time_idx,
        "decision_time_idx": decision_time_idx,
        "map_context": map_context,
        "traffic_lights": traffic_lights,
        "nearby_agents": nearby_agents,
        "local_intervention": local_intervention,
        "materialized": materialized,
    }


def _branch_candidate_for_label(
    branch_candidates: Sequence[Mapping[str, Any]],
    branch_label: Optional[str],
) -> Optional[Dict[str, Any]]:
    for candidate in branch_candidates:
        if str(candidate.get("branch_label")) == str(branch_label):
            return dict(candidate)
    return None


def _build_variant_series(
    *,
    variants: Mapping[str, Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
    expected_frame: str,
    focus_mode: str,
) -> List[TaggedXYSeries]:
    series: List[TaggedXYSeries] = []
    for mode, result in variants.items():
        color, label = _variant_style(mode)
        linestyle = _variant_linestyle(mode)
        mask = np.asarray(result["target_valid_mask"], dtype=bool)
        xy_world = np.asarray(result["target_positions_world"], dtype=np.float64)[mask]
        if xy_world.size == 0:
            continue
        if expected_frame == FRAME_AGENT_RELATIVE_AT_DECISION:
            xy = _local_clip_xy(world_xy_to_agent_relative_xy(xy_world, agent_pose_world=agent_pose_world))
        else:
            xy = np.asarray(xy_world, dtype=np.float64)
        if xy.shape[0] == 0:
            continue
        series.append(
            TaggedXYSeries(
                name=f"variant_{mode}",
                xy=xy,
                frame=expected_frame,
                draw_style="line",
                color=color,
                label=label,
                linewidth=3.0 if mode == focus_mode else 2.2,
                linestyle=linestyle,
                alpha=0.98 if mode == focus_mode else 0.94,
                zorder=20 if mode == focus_mode else 16,
            )
        )
    return series


def _build_context_rich_visuals(
    *,
    output_dir: Path,
    example_id: str,
    selected_mode: str,
    requested_branch_label: Optional[str],
    factual_or_alternative: str,
    path_head_predicted_label: Optional[str],
    raw_scenario: Mapping[str, Any],
    gt_state: Mapping[str, Any],
    branch_candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, Mapping[str, Any]],
    agent_pose_world: Mapping[str, Any],
    current_pose_world: Optional[Mapping[str, Any]],
    decision_pose_world: Optional[Mapping[str, Any]],
    current_time_idx: int,
    decision_time_idx: int,
    map_context: Mapping[str, Sequence[Mapping[str, Any]]],
    traffic_lights: Sequence[Mapping[str, Any]],
    nearby_agents: Sequence[Mapping[str, Any]],
    plot_failures: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return {
            "local_plot_extent_x": None,
            "local_plot_extent_y": None,
            "local_frame_sanity_passed": False,
            "frame_tags": {},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_result = dict(variants[selected_mode])
    requested_anchor = selected_result.get("requested_anchor")
    requested_anchor_world = (
        agent_relative_pose_to_world(requested_anchor, agent_pose_world=agent_pose_world)
        if requested_anchor is not None
        else None
    )
    requested_branch_candidate = _branch_candidate_for_label(branch_candidates, requested_branch_label)
    requested_branch_polyline_world = (
        np.asarray(requested_branch_candidate.get("polyline_xy", []), dtype=np.float64)
        if requested_branch_candidate is not None
        else np.zeros((0, 2), dtype=np.float64)
    )

    gt_xy_world = np.asarray(gt_state["position"], dtype=np.float64)[:, :2]
    gt_valid = np.asarray(gt_state["valid"], dtype=bool)
    gt_future_world = np.asarray(selected_result["gt_target_positions_world"], dtype=np.float64)[
        np.asarray(selected_result["gt_target_valid_mask"], dtype=bool)
    ]
    gt_final_world = np.asarray(
        [[selected_result["gt_final_pose_world"].x, selected_result["gt_final_pose_world"].y]],
        dtype=np.float64,
    )

    past_start_idx = max(0, int(decision_time_idx) - 12)
    target_past_world = np.asarray(gt_xy_world[past_start_idx : int(decision_time_idx) + 1], dtype=np.float64)[
        np.asarray(gt_valid[past_start_idx : int(decision_time_idx) + 1], dtype=bool)
    ]
    if current_pose_world is None:
        current_pose_world = decision_pose_world or agent_pose_world
    if decision_pose_world is None:
        decision_pose_world = agent_pose_world

    requested_branch_nearest_dist, requested_branch_nearest_point = nearest_point_on_polyline(
        [float(selected_result["gt_final_pose_world"].x), float(selected_result["gt_final_pose_world"].y)],
        requested_branch_polyline_world,
    )
    gt_branch_scorer = classify_branch_from_world_pose(selected_result["gt_final_pose_world"], branch_candidates)
    gt_to_anchor_m = None
    gt_to_anchor_heading = None
    if requested_anchor is not None:
        gt_to_anchor_m, gt_to_anchor_heading, _ = agent_relative_error_to_anchor(
            pose_world=selected_result["gt_final_pose_world"],
            anchor_rel=requested_anchor,
            agent_pose_world=agent_pose_world,
        )

    local_frame_tags: Dict[str, str] = {}
    world_frame_tags: Dict[str, str] = {}
    local_label_once: set[str] = set()
    world_label_once: set[str] = set()

    def local_label(label: str) -> Optional[str]:
        if label in local_label_once:
            return None
        local_label_once.add(label)
        return label

    def world_label(label: str) -> Optional[str]:
        if label in world_label_once:
            return None
        world_label_once.add(label)
        return label

    def add_local_series(series_list: List[TaggedXYSeries], *, name: str, xy_world: np.ndarray, **kwargs: Any) -> None:
        xy_local = _local_clip_xy(world_xy_to_agent_relative_xy(np.asarray(xy_world, dtype=np.float64), agent_pose_world=agent_pose_world))
        if xy_local.shape[0] == 0:
            return
        local_frame_tags[name] = FRAME_AGENT_RELATIVE_AT_DECISION
        series_list.append(TaggedXYSeries(name=name, xy=xy_local, frame=FRAME_AGENT_RELATIVE_AT_DECISION, **kwargs))

    def add_world_series(series_list: List[TaggedXYSeries], *, name: str, xy_world: np.ndarray, **kwargs: Any) -> None:
        xy = np.asarray(xy_world, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] == 0:
            return
        world_frame_tags[name] = FRAME_WORLD
        series_list.append(TaggedXYSeries(name=name, xy=xy[:, :2], frame=FRAME_WORLD, **kwargs))

    local_series: List[TaggedXYSeries] = []
    world_series: List[TaggedXYSeries] = []
    overlay_series: List[TaggedXYSeries] = []

    for feature in map_context.get("crosswalks", []):
        add_local_series(
            local_series,
            name=f"crosswalk_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="polygon",
            color="#a7f3d0",
            alpha=0.18,
            label=None,
            zorder=1,
            fill_alpha=0.10,
        )
        add_world_series(
            world_series,
            name=f"crosswalk_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="polygon",
            color="#a7f3d0",
            alpha=0.18,
            label=None,
            zorder=1,
            fill_alpha=0.10,
        )
    for feature in map_context.get("road_boundaries", []):
        add_local_series(
            local_series,
            name=f"road_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="line",
            color="#b8c0cc",
            linewidth=0.9,
            alpha=0.34,
            label=None,
            zorder=2,
        )
        add_world_series(
            world_series,
            name=f"road_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="line",
            color="#b8c0cc",
            linewidth=0.9,
            alpha=0.34,
            label=None,
            zorder=2,
        )
    for feature in map_context.get("lane_centerlines", []):
        add_local_series(
            local_series,
            name=f"lane_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="line",
            color="#d1d5db",
            linewidth=0.78,
            alpha=0.32,
            label=None,
            zorder=3,
        )
        add_world_series(
            world_series,
            name=f"lane_{feature['feature_id']}",
            xy_world=np.asarray(feature["xy_world"], dtype=np.float64),
            draw_style="line",
            color="#d1d5db",
            linewidth=0.78,
            alpha=0.32,
            label=None,
            zorder=3,
        )

    for candidate in branch_candidates:
        polyline_world = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)
        branch_label = str(candidate.get("branch_label"))
        color = _branch_color(branch_label)
        add_local_series(
            overlay_series,
            name=f"branch_{candidate.get('branch_id')}",
            xy_world=polyline_world,
            draw_style="line",
            color=color,
            linewidth=2.1 if branch_label == requested_branch_label else 1.4,
            alpha=1.0 if branch_label == requested_branch_label else 0.78,
            label=local_label(f"branch {branch_label}"),
            annotate=branch_label,
            annotate_index=-1,
            zorder=9 if branch_label == requested_branch_label else 7,
        )
        add_local_series(
            local_series,
            name=f"branch_{candidate.get('branch_id')}",
            xy_world=polyline_world,
            draw_style="line",
            color=color,
            linewidth=1.35 if branch_label == requested_branch_label else 0.9,
            linestyle="--",
            alpha=0.62 if branch_label == requested_branch_label else 0.28,
            label=local_label(f"branch {branch_label}"),
            zorder=5,
        )
        add_world_series(
            world_series,
            name=f"branch_{candidate.get('branch_id')}",
            xy_world=polyline_world,
            draw_style="line",
            color=color,
            linewidth=1.35 if branch_label == requested_branch_label else 0.9,
            linestyle="--",
            alpha=0.62 if branch_label == requested_branch_label else 0.28,
            label=world_label(f"branch {branch_label}"),
            zorder=5,
        )

    for agent in nearby_agents:
        add_local_series(
            local_series,
            name=f"nearby_past_{agent['track_id']}",
            xy_world=np.asarray(agent["past_xy_world"], dtype=np.float64),
            draw_style="line",
            color="#94a3b8",
            linewidth=0.95,
            alpha=0.42,
            linestyle="--",
            label=None,
            zorder=6,
        )
        add_local_series(
            local_series,
            name=f"nearby_now_{agent['track_id']}",
            xy_world=np.asarray([agent["current_xy_world"]], dtype=np.float64),
            draw_style="scatter",
            color="#64748b",
            markersize=18.0,
            alpha=0.7,
            label=None,
            zorder=8,
            annotate=str(agent["track_id"]),
        )
        add_world_series(
            world_series,
            name=f"nearby_past_{agent['track_id']}",
            xy_world=np.asarray(agent["past_xy_world"], dtype=np.float64),
            draw_style="line",
            color="#94a3b8",
            linewidth=0.95,
            alpha=0.42,
            linestyle="--",
            label=None,
            zorder=6,
        )
        add_world_series(
            world_series,
            name=f"nearby_now_{agent['track_id']}",
            xy_world=np.asarray([agent["current_xy_world"]], dtype=np.float64),
            draw_style="scatter",
            color="#64748b",
            markersize=18.0,
            alpha=0.7,
            label=None,
            zorder=8,
            annotate=str(agent["track_id"]),
        )

    add_local_series(
        local_series,
        name="target_past",
        xy_world=target_past_world,
        draw_style="line",
        color="#6d28d9",
        linewidth=2.2,
        alpha=0.82,
        label=local_label("target past"),
        zorder=13,
    )
    add_world_series(
        world_series,
        name="target_past",
        xy_world=target_past_world,
        draw_style="line",
        color="#6d28d9",
        linewidth=2.2,
        alpha=0.82,
        label=world_label("target past"),
        zorder=13,
    )
    add_local_series(
        local_series,
        name="gt_future",
        xy_world=gt_future_world,
        draw_style="line",
        color="#000000",
        linewidth=2.0,
        linestyle="--",
        label=local_label("GT future"),
        zorder=18,
    )
    add_world_series(
        world_series,
        name="gt_future",
        xy_world=gt_future_world,
        draw_style="line",
        color="#000000",
        linewidth=2.0,
        linestyle="--",
        label=world_label("GT future"),
        zorder=18,
    )

    local_series.extend(_build_variant_series(variants=variants, agent_pose_world=agent_pose_world, expected_frame=FRAME_AGENT_RELATIVE_AT_DECISION, focus_mode=selected_mode))
    world_series.extend(_build_variant_series(variants=variants, agent_pose_world=agent_pose_world, expected_frame=FRAME_WORLD, focus_mode=selected_mode))

    stop_points_world = np.asarray(
        [light["stop_point_xy_world"] for light in traffic_lights if np.asarray(light["stop_point_xy_world"]).shape[0] == 2],
        dtype=np.float64,
    ) if traffic_lights else np.zeros((0, 2), dtype=np.float64)
    if stop_points_world.size:
        add_local_series(
            local_series,
            name="stop_points",
            xy_world=stop_points_world,
            draw_style="scatter",
            color="#ff7f0e",
            marker="*",
            label=local_label("stop point"),
            markersize=80.0,
            zorder=21,
        )
        add_world_series(
            world_series,
            name="stop_points",
            xy_world=stop_points_world,
            draw_style="scatter",
            color="#ff7f0e",
            marker="*",
            label=world_label("stop point"),
            markersize=80.0,
            zorder=21,
        )

    for light in traffic_lights:
        light_state = str(light.get("state_at_decision"))
        if "GO" in light_state:
            light_color = "#1b9e77"
        elif "CAUTION" in light_state:
            light_color = "#e6ab02"
        else:
            light_color = "#d95f02"
        add_local_series(
            local_series,
            name=f"light_{light['light_id']}",
            xy_world=np.asarray([light["stop_point_xy_world"]], dtype=np.float64),
            draw_style="scatter",
            color=light_color,
            marker="D",
            label=local_label("traffic light"),
            markersize=52.0 if light.get("is_focus") else 30.0,
            zorder=22,
            annotate=(f"{light['light_id']}:{light_state}" if light.get("is_focus") else None),
        )
        add_world_series(
            world_series,
            name=f"light_{light['light_id']}",
            xy_world=np.asarray([light["stop_point_xy_world"]], dtype=np.float64),
            draw_style="scatter",
            color=light_color,
            marker="D",
            label=world_label("traffic light"),
            markersize=52.0 if light.get("is_focus") else 30.0,
            zorder=22,
            annotate=(f"{light['light_id']}:{light_state}" if light.get("is_focus") else None),
        )

    add_local_series(
        local_series,
        name="decision_pose",
        xy_world=np.asarray([[float(decision_pose_world["x"]), float(decision_pose_world["y"])]], dtype=np.float64),
        draw_style="scatter",
        color="#7b3294",
        label=local_label("decision"),
        markersize=48.0,
        zorder=24,
    )
    add_world_series(
        world_series,
        name="decision_pose",
        xy_world=np.asarray([[float(decision_pose_world["x"]), float(decision_pose_world["y"])]], dtype=np.float64),
        draw_style="scatter",
        color="#7b3294",
        label=world_label("decision"),
        markersize=48.0,
        zorder=24,
    )
    add_local_series(
        local_series,
        name="current_pose",
        xy_world=np.asarray([[float(current_pose_world["x"]), float(current_pose_world["y"])]], dtype=np.float64),
        draw_style="scatter",
        color="#984ea3",
        marker="s",
        label=local_label("current"),
        markersize=36.0,
        zorder=23,
    )
    add_world_series(
        world_series,
        name="current_pose",
        xy_world=np.asarray([[float(current_pose_world["x"]), float(current_pose_world["y"])]], dtype=np.float64),
        draw_style="scatter",
        color="#984ea3",
        marker="s",
        label=world_label("current"),
        markersize=36.0,
        zorder=23,
    )
    add_local_series(
        local_series,
        name="decision_heading",
        xy_world=_heading_arrow_xy(decision_pose_world, length_m=7.0),
        draw_style="line",
        color="#4d004b",
        linewidth=2.0,
        label=None,
        zorder=24,
    )
    add_world_series(
        world_series,
        name="decision_heading",
        xy_world=_heading_arrow_xy(decision_pose_world, length_m=7.0),
        draw_style="line",
        color="#4d004b",
        linewidth=2.0,
        label=None,
        zorder=24,
    )
    add_local_series(
        local_series,
        name="gt_final",
        xy_world=gt_final_world,
        draw_style="scatter",
        color="#009e73",
        label=local_label("GT final"),
        markersize=44.0,
        zorder=25,
    )
    add_world_series(
        world_series,
        name="gt_final",
        xy_world=gt_final_world,
        draw_style="scatter",
        color="#009e73",
        label=world_label("GT final"),
        markersize=44.0,
        zorder=25,
    )
    if requested_anchor_world is not None:
        add_local_series(
            local_series,
            name="requested_anchor",
            xy_world=np.asarray([[float(requested_anchor_world.x), float(requested_anchor_world.y)]], dtype=np.float64),
            draw_style="scatter",
            color="#e41a1c",
            label=local_label("requested anchor"),
            markersize=48.0,
            zorder=26,
        )
        add_world_series(
            world_series,
            name="requested_anchor",
            xy_world=np.asarray([[float(requested_anchor_world.x), float(requested_anchor_world.y)]], dtype=np.float64),
            draw_style="scatter",
            color="#e41a1c",
            label=world_label("requested anchor"),
            markersize=48.0,
            zorder=26,
        )
        add_local_series(
            overlay_series,
            name="requested_anchor",
            xy_world=np.asarray([[float(requested_anchor_world.x), float(requested_anchor_world.y)]], dtype=np.float64),
            draw_style="scatter",
            color="#e41a1c",
            label=local_label("requested anchor"),
            markersize=46.0,
            zorder=26,
        )

    local_result = _render_frame_safe_plot(
        output_path=output_dir / "local_bev.png",
        example_id=example_id,
        plot_name="local_bev",
        expected_frame=FRAME_AGENT_RELATIVE_AT_DECISION,
        title=f"{example_id}\nlocal BEV [{selected_mode}]",
        xlabel="x_rel_at_decision (m)",
        ylabel="y_rel_at_decision (m)",
        series_list=local_series,
        plot_failures=plot_failures,
        fixed_half_extent=LOCAL_BEV_HALF_EXTENT_M,
        padding=12.0,
        extra_text_lines=[
            f"requested={requested_branch_label} | selected={selected_mode} | bucket={factual_or_alternative}",
            f"path_head={path_head_predicted_label} | current={current_time_idx} | decision={decision_time_idx}",
            f"map lanes={len(map_context.get('lane_centerlines', []))} roads={len(map_context.get('road_boundaries', []))} crosswalks={len(map_context.get('crosswalks', []))}",
            f"lights={len(traffic_lights)} nearby={len(nearby_agents)}",
        ],
    )
    world_result = _render_frame_safe_plot(
        output_path=output_dir / "world_bev.png",
        example_id=example_id,
        plot_name="world_bev",
        expected_frame=FRAME_WORLD,
        title=f"{example_id}\nworld BEV [{selected_mode}]",
        xlabel="x_world (m)",
        ylabel="y_world (m)",
        series_list=world_series,
        plot_failures=plot_failures,
        padding=18.0,
        extra_text_lines=[
            f"requested={requested_branch_label} | selected={selected_mode}",
            f"path_head={path_head_predicted_label} | realized={selected_result['corrected_branch'].get('branch_label')}",
        ],
    )

    target_only_local_series = [
        TaggedXYSeries(
            name="gt_future",
            xy=_local_clip_xy(world_xy_to_agent_relative_xy(gt_future_world, agent_pose_world=agent_pose_world)),
            frame=FRAME_AGENT_RELATIVE_AT_DECISION,
            draw_style="line",
            color="#000000",
            label="GT future",
            linewidth=2.0,
            linestyle="--",
            zorder=12,
        )
    ]
    target_only_local_series.extend(
        _build_variant_series(
            variants=variants,
            agent_pose_world=agent_pose_world,
            expected_frame=FRAME_AGENT_RELATIVE_AT_DECISION,
            focus_mode=selected_mode,
        )
    )
    target_delta_result = _render_frame_safe_plot(
        output_path=output_dir / "target_only_delta.png",
        example_id=example_id,
        plot_name="target_only_delta",
        expected_frame=FRAME_AGENT_RELATIVE_AT_DECISION,
        title=f"{example_id}\ntarget-only delta [{selected_mode}]",
        xlabel="x_rel_at_decision (m)",
        ylabel="y_rel_at_decision (m)",
        series_list=target_only_local_series,
        plot_failures=plot_failures,
        fixed_half_extent=LOCAL_BEV_HALF_EXTENT_M,
        padding=10.0,
    )

    overlay_result = _render_frame_safe_plot(
        output_path=output_dir / "branch_candidates_overlay.png",
        example_id=example_id,
        plot_name="branch_candidates_overlay",
        expected_frame=FRAME_AGENT_RELATIVE_AT_DECISION,
        title=f"{example_id}\nbranch candidates overlay",
        xlabel="x_rel_at_decision (m)",
        ylabel="y_rel_at_decision (m)",
        series_list=overlay_series,
        plot_failures=plot_failures,
        fixed_half_extent=LOCAL_BEV_HALF_EXTENT_M,
        padding=10.0,
        extra_text_lines=[f"requested={requested_branch_label}", f"path_head={path_head_predicted_label}"],
    )

    anchor_semantics_series: List[TaggedXYSeries] = []
    if requested_branch_polyline_world.shape[0] >= 2:
        add_world_series(
            anchor_semantics_series,
            name="requested_branch_polyline",
            xy_world=requested_branch_polyline_world,
            draw_style="line",
            color=_branch_color(requested_branch_label),
            linewidth=2.0,
            label=f"requested branch ({requested_branch_label})",
            zorder=8,
        )
    add_world_series(
        anchor_semantics_series,
        name="gt_future",
        xy_world=gt_future_world,
        draw_style="line",
        color="#000000",
        linewidth=1.8,
        linestyle="--",
        label="GT future",
        zorder=10,
    )
    add_world_series(
        anchor_semantics_series,
        name="gt_final",
        xy_world=gt_final_world,
        draw_style="scatter",
        color="#009e73",
        label="GT final",
        markersize=46.0,
        zorder=20,
    )
    if requested_anchor_world is not None:
        add_world_series(
            anchor_semantics_series,
            name="requested_anchor",
            xy_world=np.asarray([[float(requested_anchor_world.x), float(requested_anchor_world.y)]], dtype=np.float64),
            draw_style="scatter",
            color="#e41a1c",
            label="requested anchor",
            markersize=46.0,
            zorder=21,
        )
    if requested_branch_nearest_point is not None:
        add_world_series(
            anchor_semantics_series,
            name="requested_branch_nearest",
            xy_world=np.asarray([requested_branch_nearest_point], dtype=np.float64),
            draw_style="scatter",
            color="#377eb8",
            label="nearest point on requested branch",
            markersize=42.0,
            zorder=22,
        )
    anchor_semantics_result = _render_frame_safe_plot(
        output_path=output_dir / "anchor_semantics.png",
        example_id=example_id,
        plot_name="anchor_semantics",
        expected_frame=FRAME_WORLD,
        title=f"{example_id}\nanchor semantics",
        xlabel="x_world (m)",
        ylabel="y_world (m)",
        series_list=anchor_semantics_series,
        plot_failures=plot_failures,
        padding=18.0,
        extra_text_lines=[
            f"gt->anchor={None if gt_to_anchor_m is None else round(float(gt_to_anchor_m), 2)}m",
            f"gt->branch_polyline={None if requested_branch_nearest_dist is None else round(float(requested_branch_nearest_dist), 2)}m",
            f"gt_branch={gt_branch_scorer.get('branch_label')} | requested={requested_branch_label}",
        ],
    )

    _compose_contact_sheet(
        output_dir / "contact_sheet.png",
        [
            output_dir / "local_bev.png",
            output_dir / "world_bev.png",
            output_dir / "target_only_delta.png",
            output_dir / "anchor_semantics.png",
        ],
    )

    metadata = {
        "example_id": str(example_id),
        "requested_branch": requested_branch_label,
        "factual_or_alternative": factual_or_alternative,
        "path_head_predicted_label": path_head_predicted_label,
        "realized_branch_no_control": variants["no_control"]["corrected_branch"].get("branch_label"),
        "realized_branch_factual": (variants["factual"]["corrected_branch"].get("branch_label") if "factual" in variants else None),
        "realized_branch_alternative": (selected_result["corrected_branch"].get("branch_label") if selected_mode.startswith("alternative_") else None),
        "alternative_realized_branch_labels": {
            mode: result["corrected_branch"].get("branch_label")
            for mode, result in variants.items()
            if mode.startswith("alternative_")
        },
        "target_mean_displacement_vs_no_control": None,
        "non_target_mean_displacement_vs_no_control": None,
        "gt_to_anchor_m": gt_to_anchor_m,
        "gt_to_branch_polyline_m": requested_branch_nearest_dist,
        "gt_branch_label": gt_branch_scorer.get("branch_label"),
        "frame_tags": {
            "local_bev": local_frame_tags,
            "world_bev": world_frame_tags,
            "target_only_delta": {
                series.name: FRAME_AGENT_RELATIVE_AT_DECISION
                for series in target_only_local_series
            },
            "anchor_semantics": {
                series.name: FRAME_WORLD
                for series in anchor_semantics_series
            },
            "branch_candidates_overlay": {
                series.name: FRAME_AGENT_RELATIVE_AT_DECISION
                for series in overlay_series
            },
        },
        "context_counts": {
            "lane_centerlines": int(len(map_context.get("lane_centerlines", []))),
            "road_boundaries": int(len(map_context.get("road_boundaries", []))),
            "crosswalks": int(len(map_context.get("crosswalks", []))),
            "traffic_lights": int(len(traffic_lights)),
            "nearby_agents": int(len(nearby_agents)),
        },
        "local_plot_extent_x": (None if local_result["extent"] is None else [local_result["extent"]["x_min"], local_result["extent"]["x_max"]]),
        "local_plot_extent_y": (None if local_result["extent"] is None else [local_result["extent"]["y_min"], local_result["extent"]["y_max"]]),
        "local_frame_sanity_passed": bool(local_result.get("local_frame_sanity_passed", False)),
        "plots": {
            "local_bev": str((output_dir / "local_bev.png").resolve()),
            "world_bev": str((output_dir / "world_bev.png").resolve()),
            "target_only_delta": str((output_dir / "target_only_delta.png").resolve()),
            "anchor_semantics": str((output_dir / "anchor_semantics.png").resolve()),
            "branch_candidates_overlay": str((output_dir / "branch_candidates_overlay.png").resolve()),
            "contact_sheet": str((output_dir / "contact_sheet.png").resolve()),
        },
        "requested_anchor_world": (None if requested_anchor_world is None else pose_to_dict(requested_anchor_world)),
        "requested_branch_nearest_point_world": (
            None if requested_branch_nearest_point is None else [float(requested_branch_nearest_point[0]), float(requested_branch_nearest_point[1])]
        ),
        "plot_results": {
            "local_bev": local_result,
            "world_bev": world_result,
            "target_only_delta": target_delta_result,
            "anchor_semantics": anchor_semantics_result,
            "branch_candidates_overlay": overlay_result,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


def _build_selected_bundle_row(
    *,
    manifest_item: Mapping[str, Any],
    selected_mode_result: Mapping[str, Any],
    no_control_result: Mapping[str, Any],
    original_eval_row: Mapping[str, Any],
) -> Dict[str, Any]:
    target_mean_disp, target_final_delta = trajectory_mean_displacement(
        np.asarray(selected_mode_result["target_positions_world"], dtype=np.float64),
        np.asarray(selected_mode_result["target_valid_mask"], dtype=bool),
        np.asarray(no_control_result["target_positions_world"], dtype=np.float64),
        np.asarray(no_control_result["target_valid_mask"], dtype=bool),
    )
    non_target_mean, non_target_max = non_target_displacement(
        np.asarray(selected_mode_result["non_target_positions_world"], dtype=np.float64),
        np.asarray(no_control_result["non_target_positions_world"], dtype=np.float64),
    )
    ade_factual, fde_factual = (None, None)
    if str(manifest_item["selected_mode"]) == "factual":
        ade_factual, fde_factual = ade_fde(
            np.asarray(selected_mode_result["target_positions_world"], dtype=np.float64),
            np.asarray(selected_mode_result["target_valid_mask"], dtype=bool),
            np.asarray(selected_mode_result["gt_target_positions_world"], dtype=np.float64),
            np.asarray(selected_mode_result["gt_target_valid_mask"], dtype=bool),
        )

    target_heading_delta_vs_no_control = None
    target_valid = np.asarray(selected_mode_result["target_valid_mask"], dtype=bool)
    base_valid = np.asarray(no_control_result["target_valid_mask"], dtype=bool)
    overlap = target_valid & base_valid
    if bool(overlap.any()):
        final_idx = int(np.flatnonzero(overlap)[-1])
        target_heading_delta_vs_no_control = float(
            abs(
                math.atan2(
                    math.sin(
                        float(selected_mode_result["target_headings_world"][final_idx])
                        - float(no_control_result["target_headings_world"][final_idx])
                    ),
                    math.cos(
                        float(selected_mode_result["target_headings_world"][final_idx])
                        - float(no_control_result["target_headings_world"][final_idx])
                    ),
                )
            )
        )

    return {
        "example_id": str(manifest_item["example_id"]),
        "scenario_id": str(manifest_item["scenario_id"]),
        "agent_id": str(manifest_item["agent_id"]),
        "decision_time_idx": int(manifest_item["decision_time_idx"]),
        "mode": str(manifest_item["selected_mode"]),
        "mode_bucket": str(manifest_item["factual_or_alternative"]),
        "requested_branch_label": selected_mode_result.get("requested_branch_label"),
        "predicted_branch_label": selected_mode_result["corrected_branch"].get("branch_label"),
        "requested_branch_match": selected_mode_result.get("requested_branch_match"),
        "branch_score_margin": selected_mode_result["corrected_branch"].get("score_margin"),
        "final_pose_to_requested_anchor_m": selected_mode_result.get("corrected_anchor_distance_m"),
        "final_heading_error_to_requested_anchor_rad": selected_mode_result.get("corrected_heading_error_rad"),
        "ade_factual": ade_factual,
        "fde_factual": fde_factual,
        "changed_from_no_control": bool((target_mean_disp or 0.0) > 0.1),
        "target_mean_displacement_vs_no_control": target_mean_disp,
        "target_final_pose_delta_vs_no_control": target_final_delta,
        "target_heading_delta_vs_no_control": target_heading_delta_vs_no_control,
        "target_branch_changed_from_no_control": selected_mode_result.get("no_control_branch_changed"),
        "non_target_mean_displacement_vs_no_control": non_target_mean,
        "non_target_max_displacement_vs_no_control": non_target_max,
        "path_head_predicted_label": selected_mode_result.get("path_head", {}).get("label"),
        "path_head_margin": selected_mode_result.get("path_head", {}).get("margin"),
        "legacy_predicted_branch_label": original_eval_row.get("predicted_branch_label"),
        "legacy_requested_branch_match": original_eval_row.get("requested_branch_match"),
        "legacy_final_pose_to_requested_anchor_m": original_eval_row.get("final_pose_to_requested_anchor_m"),
    }


def _write_frame_audit(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    for row in selected_rows:
        failures.append(
            {
                "example_id": str(row["example_id"]),
                "mode": str(row["mode"]),
                "issue": "legacy_branch_scoring_compared_model_output_frame_to_world_branch_candidates",
                "left_frame": FRAME_MODEL_OUTPUT,
                "right_frame": FRAME_WORLD,
            }
        )
        if row.get("requested_branch_label") is not None:
            failures.append(
                {
                    "example_id": str(row["example_id"]),
                    "mode": str(row["mode"]),
                    "issue": "legacy_anchor_comparison_used_model_output_frame_as_if_world_before_agent_relative_projection",
                    "left_frame": FRAME_MODEL_OUTPUT,
                    "right_frame": FRAME_AGENT_RELATIVE_AT_DECISION,
                }
            )
    summary = {
        "num_rows": int(len(selected_rows)),
        "num_branch_frame_mismatch_failures": int(len(selected_rows)),
        "num_anchor_frame_mismatch_failures": int(sum(1 for row in selected_rows if row.get("requested_branch_label") is not None)),
        "direct_frame_bug_detected": True,
        "corrected_branch_scorer_frame": FRAME_WORLD,
        "corrected_anchor_comparison_frame": FRAME_AGENT_RELATIVE_AT_DECISION,
    }
    write_json(outdir / "frame_audit_summary.json", summary)
    write_jsonl(outdir / "frame_audit_failures.jsonl", failures)
    return summary


def _write_control_effect_summary(rows: Sequence[Mapping[str, Any]], outdir: Path) -> Dict[str, Any]:
    summary = {
        "num_rows": int(len(rows)),
        "mean_target_mean_displacement_vs_no_control": mean_or_none(row.get("target_mean_displacement_vs_no_control") for row in rows),
        "mean_target_final_pose_delta_vs_no_control": mean_or_none(row.get("target_final_pose_delta_vs_no_control") for row in rows),
        "mean_target_heading_delta_vs_no_control": mean_or_none(row.get("target_heading_delta_vs_no_control") for row in rows),
        "branch_changed_from_no_control_rate": float(
            sum(bool(row.get("target_branch_changed_from_no_control")) for row in rows) / len(rows)
        ) if rows else 0.0,
        "mean_non_target_mean_displacement_vs_no_control": mean_or_none(row.get("non_target_mean_displacement_vs_no_control") for row in rows),
        "mean_non_target_max_displacement_vs_no_control": mean_or_none(row.get("non_target_max_displacement_vs_no_control") for row in rows),
    }
    write_json(outdir / "control_effect_summary.json", summary)
    write_jsonl(outdir / "control_effect_per_example.jsonl", rows)
    return summary


def _write_path_head_vs_trajectory(rows: Sequence[Mapping[str, Any]], outdir: Path) -> Dict[str, Any]:
    summary = {
        "num_rows": int(len(rows)),
        "path_head_matches_requested_rate": float(
            sum(
                bool(row.get("path_head_predicted_label") == row.get("requested_branch_label"))
                for row in rows
                if row.get("requested_branch_label") is not None
            )
            / max(1, sum(1 for row in rows if row.get("requested_branch_label") is not None))
        ),
        "realized_branch_matches_requested_rate": float(
            sum(bool(row.get("requested_branch_match")) for row in rows if row.get("requested_branch_label") is not None)
            / max(1, sum(1 for row in rows if row.get("requested_branch_label") is not None))
        ),
        "path_head_matches_requested_but_realized_mismatch_count": int(
            sum(
                bool(row.get("path_head_predicted_label") == row.get("requested_branch_label"))
                and not bool(row.get("requested_branch_match"))
                for row in rows
                if row.get("requested_branch_label") is not None
            )
        ),
        "both_path_head_and_realized_mismatch_count": int(
            sum(
                (row.get("path_head_predicted_label") != row.get("requested_branch_label"))
                and not bool(row.get("requested_branch_match"))
                for row in rows
                if row.get("requested_branch_label") is not None
            )
        ),
    }
    write_json(outdir / "path_head_vs_trajectory_summary.json", summary)
    write_jsonl(outdir / "path_head_vs_trajectory_per_example.jsonl", rows)
    return summary


def _write_deterministic_pair_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    outdir: Path,
    previous_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    near_zero_threshold_m = 1e-3
    summary = {
        "num_rows": int(len(rows)),
        "seed_or_decode_mode": DETERMINISTIC_DECODE_MODE,
        "decode_seed": int(DETERMINISTIC_SEED),
        "target_mean_displacement_vs_no_control": mean_or_none(row.get("target_mean_displacement_vs_no_control") for row in rows),
        "non_target_mean_displacement_vs_no_control": mean_or_none(row.get("non_target_mean_displacement_vs_no_control") for row in rows),
        "non_target_max_displacement_vs_no_control": mean_or_none(row.get("non_target_max_displacement_vs_no_control") for row in rows),
        "branch_changed_from_no_control": float(
            sum(bool(row.get("target_branch_changed_from_no_control")) for row in rows) / len(rows)
        ) if rows else 0.0,
        "num_rows_non_target_mean_below_1e-3m": int(
            sum(
                row.get("non_target_mean_displacement_vs_no_control") is not None
                and float(row.get("non_target_mean_displacement_vs_no_control")) <= near_zero_threshold_m
                for row in rows
            )
        ),
        "num_rows_non_target_max_below_1e-3m": int(
            sum(
                row.get("non_target_max_displacement_vs_no_control") is not None
                and float(row.get("non_target_max_displacement_vs_no_control")) <= near_zero_threshold_m
                for row in rows
            )
        ),
        "all_non_target_deltas_near_zero": bool(
            rows and all(
                row.get("non_target_max_displacement_vs_no_control") is not None
                and float(row.get("non_target_max_displacement_vs_no_control")) <= near_zero_threshold_m
                for row in rows
            )
        ),
        "previous_summary": (dict(previous_summary) if previous_summary is not None else None),
    }
    write_json(outdir / "deterministic_pair_summary.json", summary)
    return summary


def _write_anchor_semantics_summary(rows: Sequence[Mapping[str, Any]], outdir: Path) -> Dict[str, Any]:
    gt_to_anchor = [float(row["gt_to_anchor_m"]) for row in rows if row.get("gt_to_anchor_m") is not None]
    gt_to_branch = [float(row["gt_to_branch_polyline_m"]) for row in rows if row.get("gt_to_branch_polyline_m") is not None]
    summary = {
        "num_rows": int(len(rows)),
        "mean_gt_to_anchor_m": mean_or_none(row.get("gt_to_anchor_m") for row in rows),
        "mean_gt_to_branch_polyline_m": mean_or_none(row.get("gt_to_branch_polyline_m") for row in rows),
        "gt_to_anchor_percentiles_m": percentile_dict(gt_to_anchor),
        "gt_to_branch_polyline_percentiles_m": percentile_dict(gt_to_branch),
        "gt_branch_matches_requested_rate": float(
            sum(bool(row.get("gt_branch_matches_requested")) for row in rows) / len(rows)
        ) if rows else 0.0,
        "num_anchor_far_over_20m": int(sum(float(row.get("gt_to_anchor_m") or 0.0) > 20.0 for row in rows)),
        "num_anchor_far_but_branch_close_under_5m": int(
            sum(
                float(row.get("gt_to_anchor_m") or 0.0) > 20.0
                and row.get("gt_to_branch_polyline_m") is not None
                and float(row.get("gt_to_branch_polyline_m")) < 5.0
                for row in rows
            )
        ),
    }
    write_json(outdir / "anchor_semantics_summary.json", summary)
    return summary


def _write_branch_scorer_sanity_summary(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    factual_anchor_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
) -> Dict[str, Any]:
    by_class: Dict[str, Dict[str, Any]] = {}
    for label in REQUESTED_BRANCH_LABELS:
        class_rows = [row for row in selected_rows if str(row.get("requested_branch_label")) == label]
        by_class[label] = {
            "num_rows": int(len(class_rows)),
            "requested_match_rate": float(sum(bool(row.get("requested_branch_match")) for row in class_rows) / len(class_rows)) if class_rows else 0.0,
            "mean_branch_score_margin": mean_or_none(row.get("branch_score_margin") for row in class_rows),
        }
    summary = {
        "num_rows": int(len(selected_rows)),
        "requested_branch_match_rate": float(
            sum(bool(row.get("requested_branch_match")) for row in selected_rows) / len(selected_rows)
        ) if selected_rows else 0.0,
        "by_requested_class": by_class,
        "factual_gt_branch_matches_requested_rate": float(
            sum(bool(row.get("gt_branch_matches_requested")) for row in factual_anchor_rows) / len(factual_anchor_rows)
        ) if factual_anchor_rows else 0.0,
        "decode_mode": DETERMINISTIC_DECODE_MODE,
    }
    write_json(outdir / "branch_scorer_sanity_summary.json", summary)
    return summary


def _select_original_eval_row(
    original_eval_rows_by_example: Mapping[str, List[Dict[str, Any]]],
    *,
    example_id: str,
    mode: str,
) -> Dict[str, Any]:
    for row in original_eval_rows_by_example.get(example_id, []):
        if str(row.get("mode")) == mode:
            return row
    return {}


def main() -> int:
    args = parse_args()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = build_bundle_inventory(bundle_root)
    write_json(outdir / "bundle_inventory.json", inventory)

    path_index_rows = load_jsonl(bundle_root / "outputs" / "pr6_path_index_5000" / "path_index_curated_val.jsonl")
    rewritten_index_rows, rewrite_report = rewrite_path_index_rows_for_bundle(path_index_rows, bundle_root=bundle_root)

    eval_rows = load_jsonl(bundle_root / "outputs" / "pr6_path_eval_20260401_run1" / "path_control_eval_per_example.jsonl")
    original_eval_rows_by_example: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        original_eval_rows_by_example[str(row["example_id"])].append(row)
    index_row_by_example = {str(row["example_id"]): row for row in rewritten_index_rows}

    available_rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        if str(row.get("mode")) == "no_control":
            continue
        entry = index_row_by_example.get(str(row["example_id"]))
        if entry is None:
            continue
        light_id = str(entry.get("light_id", "")) or str(parse_example_id(row["example_id"]).get("light_id") or "")
        local_materialized = find_materialized_eval_dir(
            bundle_root=bundle_root,
            example_id=str(row["example_id"]),
            scenario_id=str(row["scenario_id"]),
            agent_id=str(row["agent_id"]),
            decision_time_idx=int(row["decision_time_idx"]),
            light_id=light_id,
        )
        local_scenario_pkl = entry.get("scenario_pkl_local")
        if not local_scenario_pkl or local_materialized is None:
            continue
        enriched = dict(row)
        enriched.update(
            {
                "mode_bucket": mode_bucket(str(row["mode"])),
                "light_id": light_id,
                "local_scenario_pkl": str(local_scenario_pkl),
                "local_materialized_eval_input": str(local_materialized),
                "has_sweep_png": bool(row.get("control_sweep_png")),
                "control_sweep_png": row.get("control_sweep_png"),
            }
        )
        available_rows.append(enriched)

    rewrite_report.update(
        {
            "num_eval_rows": int(len(eval_rows)),
            "num_available_eval_rows_with_local_bundle_assets": int(len(available_rows)),
            "available_scenario_ids": sorted({str(row["scenario_id"]) for row in available_rows}),
        }
    )
    write_json(outdir / "path_rewrite_report.json", rewrite_report)

    manifest = _build_selected_manifest(available_rows=available_rows, outdir=outdir)

    checkpoint_path = find_bundle_checkpoint(bundle_root)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No last.ckpt found under bundle root {bundle_root}")

    config, model, tokenizer, load_report = load_model_and_tokenizer_for_bundle(
        ckpt_path=checkpoint_path,
        config_path=(args.config or None),
        load_mode=args.load_mode,
    )
    write_json(outdir / "bundle_checkpoint_load_report.json", load_report)

    # Part C: anchor sanity on factual selections first.
    run_anchor_sanity(outdir / "selected_examples_manifest.json", outdir)

    scenario_cache: Dict[str, Dict[str, Any]] = {}
    base_sample_cache: Dict[str, Dict[str, Any]] = {}
    bundle_rows: List[Dict[str, Any]] = []
    control_effect_rows: List[Dict[str, Any]] = []
    path_head_rows: List[Dict[str, Any]] = []
    factual_anchor_rows: List[Dict[str, Any]] = []
    plot_failures: List[Dict[str, Any]] = []

    corrected_visual_root = outdir / "corrected_visuals"
    previous_control_effect_summary = None
    previous_control_effect_path = outdir / "control_effect_summary.json"
    if previous_control_effect_path.is_file():
        try:
            previous_control_effect_summary = load_json(previous_control_effect_path)
        except Exception:
            previous_control_effect_summary = None

    for manifest_item in manifest:
        example_bundle = _build_example_variant_bundle(
            manifest_item=manifest_item,
            bundle_root=bundle_root,
            config=config,
            model=model,
            tokenizer=tokenizer,
            scenario_cache=scenario_cache,
            base_sample_cache=base_sample_cache,
        )
        variants = example_bundle["variants"]
        selected_mode = str(manifest_item["selected_mode"])
        selected_result = variants[selected_mode]
        no_control_result = variants["no_control"]
        original_eval_row = _select_original_eval_row(
            original_eval_rows_by_example,
            example_id=str(manifest_item["example_id"]),
            mode=selected_mode,
        )

        bundle_row = _build_selected_bundle_row(
            manifest_item=manifest_item,
            selected_mode_result=selected_result,
            no_control_result=no_control_result,
            original_eval_row=original_eval_row,
        )
        bundle_rows.append(bundle_row)
        control_effect_rows.append(
            {
                "example_id": bundle_row["example_id"],
                "scenario_id": bundle_row["scenario_id"],
                "agent_id": bundle_row["agent_id"],
                "mode": bundle_row["mode"],
                "requested_branch_label": bundle_row["requested_branch_label"],
                "predicted_branch_label": bundle_row["predicted_branch_label"],
                "target_mean_displacement_vs_no_control": bundle_row["target_mean_displacement_vs_no_control"],
                "target_final_pose_delta_vs_no_control": bundle_row["target_final_pose_delta_vs_no_control"],
                "target_heading_delta_vs_no_control": bundle_row["target_heading_delta_vs_no_control"],
                "target_branch_changed_from_no_control": bundle_row["target_branch_changed_from_no_control"],
                "non_target_mean_displacement_vs_no_control": bundle_row["non_target_mean_displacement_vs_no_control"],
                "non_target_max_displacement_vs_no_control": bundle_row["non_target_max_displacement_vs_no_control"],
            }
        )

        path_head_rows.append(
            {
                "example_id": bundle_row["example_id"],
                "scenario_id": bundle_row["scenario_id"],
                "agent_id": bundle_row["agent_id"],
                "selected_mode": bundle_row["mode"],
                "requested_branch_label": bundle_row["requested_branch_label"],
                "path_head_predicted_label": bundle_row["path_head_predicted_label"],
                "realized_generated_branch_label": bundle_row["predicted_branch_label"],
                "no_control_realized_branch_label": variants["no_control"]["corrected_branch"].get("branch_label"),
                "factual_realized_branch_label": variants["factual"]["corrected_branch"].get("branch_label") if "factual" in variants else None,
                "alternative_realized_branch_labels": {
                    mode: result["corrected_branch"].get("branch_label")
                    for mode, result in variants.items()
                    if mode.startswith("alternative_")
                },
                "requested_branch_match": bundle_row["requested_branch_match"],
            }
        )
        requested_branch_label = selected_result.get("requested_branch_label")
        requested_branch_candidate = _branch_candidate_for_label(example_bundle["branch_candidates"], requested_branch_label)
        requested_branch_polyline_world = (
            np.asarray(requested_branch_candidate.get("polyline_xy", []), dtype=np.float64)
            if requested_branch_candidate is not None
            else np.zeros((0, 2), dtype=np.float64)
        )
        gt_final_pose_world = selected_result["gt_final_pose_world"]
        gt_to_branch_polyline_m, _ = nearest_point_on_polyline(
            [float(gt_final_pose_world.x), float(gt_final_pose_world.y)],
            requested_branch_polyline_world,
        )
        gt_branch = classify_branch_from_world_pose(gt_final_pose_world, example_bundle["branch_candidates"])
        gt_to_anchor_m = None
        if selected_result.get("requested_anchor") is not None:
            gt_to_anchor_m, _, _ = agent_relative_error_to_anchor(
                pose_world=gt_final_pose_world,
                anchor_rel=selected_result["requested_anchor"],
                agent_pose_world=example_bundle["agent_pose_world"],
            )
        if str(manifest_item["factual_or_alternative"]) == "factual":
            factual_anchor_rows.append(
                {
                    "example_id": str(manifest_item["example_id"]),
                    "scenario_id": str(manifest_item["scenario_id"]),
                    "agent_id": str(manifest_item["agent_id"]),
                    "requested_branch_label": requested_branch_label,
                    "gt_branch_label": gt_branch.get("branch_label"),
                    "gt_branch_matches_requested": bool(gt_branch.get("branch_label") == requested_branch_label),
                    "gt_to_anchor_m": gt_to_anchor_m,
                    "gt_to_branch_polyline_m": gt_to_branch_polyline_m,
                    "branch_score_margin": gt_branch.get("score_margin"),
                }
            )

        metadata = _build_context_rich_visuals(
            output_dir=corrected_visual_root / str(manifest_item["example_id"]),
            example_id=str(manifest_item["example_id"]),
            selected_mode=selected_mode,
            requested_branch_label=requested_branch_label,
            factual_or_alternative=str(manifest_item["factual_or_alternative"]),
            path_head_predicted_label=selected_result.get("path_head", {}).get("label"),
            raw_scenario=example_bundle["raw_scenario"],
            gt_state=example_bundle["gt_state"],
            branch_candidates=example_bundle["branch_candidates"],
            variants=variants,
            agent_pose_world=example_bundle["agent_pose_world"],
            current_pose_world=example_bundle["current_pose_world"],
            decision_pose_world=example_bundle["decision_pose_world"],
            current_time_idx=example_bundle["current_time_idx"],
            decision_time_idx=example_bundle["decision_time_idx"],
            map_context=example_bundle["map_context"],
            traffic_lights=example_bundle["traffic_lights"],
            nearby_agents=example_bundle["nearby_agents"],
            plot_failures=plot_failures,
        )
        metadata["target_mean_displacement_vs_no_control"] = bundle_row["target_mean_displacement_vs_no_control"]
        metadata["non_target_mean_displacement_vs_no_control"] = bundle_row["non_target_mean_displacement_vs_no_control"]
        metadata["gt_to_anchor_m"] = gt_to_anchor_m
        metadata["gt_to_branch_polyline_m"] = gt_to_branch_polyline_m
        metadata["gt_branch_label"] = gt_branch.get("branch_label")
        write_json(corrected_visual_root / str(manifest_item["example_id"]) / "metadata.json", metadata)

    write_jsonl(outdir / "path_control_eval_per_example_bundle.jsonl", bundle_rows)

    confusion, breakdown = build_confusion_and_breakdown(bundle_rows)
    legacy_selected_rows = [row for row in available_rows if any(row["example_id"] == item["example_id"] and row["mode"] == item["selected_mode"] for item in manifest)]
    summary_bundle = {
        "ran": True,
        "bundle_root": str(bundle_root),
        "num_examples_selected": int(len(manifest)),
        "num_rows": int(len(bundle_rows)),
        "decode_mode": DETERMINISTIC_DECODE_MODE,
        "decode_seed": int(DETERMINISTIC_SEED),
        "selected_examples": [str(item["example_id"]) for item in manifest],
        "legacy_requested_branch_match_rate": float(
            sum(bool(row.get("requested_branch_match")) for row in legacy_selected_rows) / len(legacy_selected_rows)
        ) if legacy_selected_rows else 0.0,
        "corrected_requested_branch_match_rate": float(
            sum(bool(row.get("requested_branch_match")) for row in bundle_rows) / len(bundle_rows)
        ) if bundle_rows else 0.0,
        "legacy_mean_final_pose_to_requested_anchor_m": mean_or_none(row.get("final_pose_to_requested_anchor_m") for row in legacy_selected_rows),
        "corrected_mean_final_pose_to_requested_anchor_m": mean_or_none(row.get("final_pose_to_requested_anchor_m") for row in bundle_rows),
        "factual_ade_mean": mean_or_none(row.get("ade_factual") for row in bundle_rows if row.get("mode") == "factual"),
        "factual_fde_mean": mean_or_none(row.get("fde_factual") for row in bundle_rows if row.get("mode") == "factual"),
        "artifacts": {
            "path_control_eval_summary_bundle_json": str((outdir / "path_control_eval_summary_bundle.json").resolve()),
            "path_control_eval_confusion_matrix_bundle_json": str((outdir / "path_control_eval_confusion_matrix_bundle.json").resolve()),
            "corrected_visuals_dir": str(corrected_visual_root.resolve()),
            "plot_failures_json": str((outdir / "plot_failures.json").resolve()),
            "deterministic_pair_summary_json": str((outdir / "deterministic_pair_summary.json").resolve()),
        },
    }
    write_json(outdir / "path_control_eval_summary_bundle.json", summary_bundle)
    write_json(outdir / "path_control_eval_confusion_matrix_bundle.json", confusion)
    write_json(outdir / "path_control_eval_branch_breakdown_bundle.json", breakdown)

    _write_frame_audit(selected_rows=bundle_rows, outdir=outdir)
    _write_control_effect_summary(control_effect_rows, outdir)
    _write_path_head_vs_trajectory(path_head_rows, outdir)
    _write_deterministic_pair_summary(control_effect_rows, outdir=outdir, previous_summary=previous_control_effect_summary)
    _write_anchor_semantics_summary(factual_anchor_rows, outdir)
    _write_branch_scorer_sanity_summary(selected_rows=bundle_rows, factual_anchor_rows=factual_anchor_rows, outdir=outdir)
    write_json(outdir / "plot_failures.json", plot_failures)

    print(json.dumps(summary_bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
