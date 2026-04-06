from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bmt.counterfactual.frame_safe_plotting import PlotValidationError, TaggedXYSeries, render_tagged_series_collection
from bmt.counterfactual.path_eval_bundle import (
    FRAME_AGENT_RELATIVE_AT_DECISION,
    FRAME_WORLD,
    anchor_pose_from_control_code,
    agent_relative_pose_to_world,
    load_json,
    load_materialized_controls,
    load_raw_scenario,
    raw_track_world_state,
    world_xy_to_agent_relative_xy,
    write_json,
)


LOCAL_SANITY_LIMIT_M = 200.0
WORLD_PATCH_HALF_EXTENT_M = 40.0
CANDIDATE_PALETTE = ["#0f766e", "#2563eb", "#ea580c", "#7c3aed", "#dc2626", "#0891b2"]
PLOT_BG_COLOR = "#f8fafc"
MAP_CROSSWALK_COLOR = "#cbd5e1"
MAP_LANE_COLOR = "#64748b"
MAP_ROAD_COLOR = "#94a3b8"
NEARBY_AGENT_COLOR = "#cbd5e1"


def _track_pose_at_index(raw_scenario: Mapping[str, Any], *, track_id: str, time_index: int) -> Optional[Dict[str, float]]:
    state = raw_track_world_state(raw_scenario, track_id=str(track_id))
    valid = np.asarray(state["valid"], dtype=bool)
    if valid.size == 0:
        return None
    idx = int(np.clip(int(time_index), 0, valid.shape[0] - 1))
    if not bool(valid[idx]):
        valid_before = np.flatnonzero(valid[: idx + 1])
        if valid_before.size > 0:
            idx = int(valid_before[-1])
        else:
            valid_after = np.flatnonzero(valid[idx:])
            if valid_after.size == 0:
                return None
            idx = int(idx + valid_after[0])
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
    clipped = xy[mask]
    if clipped.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(clipped[:, :2], dtype=np.float64)


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
        trimmed = _trim_xy_to_radius(xy, center_xy=center_xy, radius_m=radius_m)
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
    max_agents: int = 14,
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
                "current_xy_world": current_xy,
                "past_xy_world": past_xy,
                "distance_to_center_m": dist,
            }
        )
    rows.sort(key=lambda item: (float(item["distance_to_center_m"]), str(item["track_id"])))
    return rows[: int(max_agents)]


def _heading_arrow_xy(pose: Mapping[str, Any], *, length_m: float = 5.0) -> np.ndarray:
    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    heading = float(pose.get("heading", 0.0))
    return np.asarray([[x, y], [x + length_m * math.cos(heading), y + length_m * math.sin(heading)]], dtype=np.float64)


def _candidate_color(index: int) -> str:
    return CANDIDATE_PALETTE[int(index) % len(CANDIDATE_PALETTE)]


def _local_sanity_for_world_series(
    *,
    agent_pose_world: Mapping[str, Any],
    series_items: Sequence[Tuple[str, np.ndarray]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    failures: List[Dict[str, Any]] = []
    for name, xy_world in series_items:
        xy = np.asarray(xy_world, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[0] == 0:
            continue
        local_xy = world_xy_to_agent_relative_xy(xy, agent_pose_world=agent_pose_world)
        over = np.argwhere(np.abs(local_xy) > float(LOCAL_SANITY_LIMIT_M))
        if over.size > 0:
            point_idx, axis_idx = over[0].tolist()
            failures.append(
                {
                    "series_name": str(name),
                    "axis": "x" if int(axis_idx) == 0 else "y",
                    "point_index": int(point_idx),
                    "value_m": float(local_xy[int(point_idx), int(axis_idx)]),
                    "limit_abs_m": float(LOCAL_SANITY_LIMIT_M),
                }
            )
    return bool(len(failures) == 0), failures


def _estimate_split_point(branch_candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    polylines = [np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)[:, :2] for candidate in branch_candidates]
    polylines = [polyline for polyline in polylines if polyline.ndim == 2 and polyline.shape[0] >= 5]
    if len(polylines) < 2:
        return {"x": None, "y": None, "frame": FRAME_WORLD, "confidence": 0.0}
    max_shared_idx = min(polyline.shape[0] for polyline in polylines)
    for idx in range(max_shared_idx):
        points = np.stack([polyline[idx] for polyline in polylines], axis=0)
        if float(np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1))) > 4.0:
            anchor = np.mean(points, axis=0)
            confidence = min(1.0, 0.45 + 0.05 * float(len(polylines)))
            return {
                "x": float(anchor[0]),
                "y": float(anchor[1]),
                "frame": FRAME_WORLD,
                "confidence": float(confidence),
            }
    anchor = np.mean(np.stack([polyline[0] for polyline in polylines], axis=0), axis=0)
    return {
        "x": float(anchor[0]),
        "y": float(anchor[1]),
        "frame": FRAME_WORLD,
        "confidence": 0.25,
    }


def _build_world_plot(
    *,
    output_path: Path,
    title: str,
    frame_label: str,
    decision_pose_world: Mapping[str, Any],
    current_pose_world: Mapping[str, Any],
    target_past_world: np.ndarray,
    map_context: Mapping[str, Sequence[Mapping[str, Any]]],
    traffic_lights: Sequence[Mapping[str, Any]],
    nearby_agents: Sequence[Mapping[str, Any]],
    branch_candidates: Sequence[Mapping[str, Any]],
    candidate_id_map: Sequence[Mapping[str, Any]],
    split_point_guess: Mapping[str, Any],
    gt_future_world: Optional[np.ndarray],
    gt_final_world: Optional[np.ndarray],
    requested_anchor_world: Optional[np.ndarray],
    sidebar_lines: Sequence[str],
    plot_failures: List[Dict[str, Any]],
    local_sanity_passed: bool,
    local_sanity_failures: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    ax.set_facecolor(PLOT_BG_COLOR)

    series_list: List[TaggedXYSeries] = []
    world_series_for_sanity: List[Tuple[str, np.ndarray]] = []
    center_xy = np.asarray([float(decision_pose_world["x"]), float(decision_pose_world["y"])], dtype=np.float64)

    for feature in map_context.get("crosswalks", []):
        xy = np.asarray(feature["xy_world"], dtype=np.float64)
        series_list.append(
            TaggedXYSeries(
                name=f"crosswalk_{feature['feature_id']}",
                xy=xy,
                frame=FRAME_WORLD,
                draw_style="polygon",
                color=MAP_CROSSWALK_COLOR,
                alpha=0.42,
                fill_alpha=0.30,
                linewidth=1.0,
                zorder=1,
            )
        )
        world_series_for_sanity.append((f"crosswalk_{feature['feature_id']}", xy))
    for feature in map_context.get("lane_centerlines", []):
        xy = np.asarray(feature["xy_world"], dtype=np.float64)
        series_list.append(
            TaggedXYSeries(
                name=f"lane_{feature['feature_id']}",
                xy=xy,
                frame=FRAME_WORLD,
                draw_style="line",
                color=MAP_LANE_COLOR,
                alpha=0.62,
                linewidth=1.4,
                zorder=2,
            )
        )
        world_series_for_sanity.append((f"lane_{feature['feature_id']}", xy))
    for feature in map_context.get("road_boundaries", []):
        xy = np.asarray(feature["xy_world"], dtype=np.float64)
        series_list.append(
            TaggedXYSeries(
                name=f"road_{feature['feature_id']}",
                xy=xy,
                frame=FRAME_WORLD,
                draw_style="line",
                color=MAP_ROAD_COLOR,
                alpha=0.52,
                linewidth=1.35,
                linestyle="--",
                zorder=2,
            )
        )
        world_series_for_sanity.append((f"road_{feature['feature_id']}", xy))

    for agent in nearby_agents:
        past_xy = np.asarray(agent["past_xy_world"], dtype=np.float64)
        current_xy = np.asarray(agent["current_xy_world"], dtype=np.float64).reshape(1, 2)
        series_list.append(
            TaggedXYSeries(
                name=f"nearby_past_{agent['track_id']}",
                xy=past_xy,
                frame=FRAME_WORLD,
                draw_style="line",
                color=NEARBY_AGENT_COLOR,
                alpha=0.22,
                linewidth=0.9,
                zorder=3,
            )
        )
        series_list.append(
            TaggedXYSeries(
                name=f"nearby_current_{agent['track_id']}",
                xy=current_xy,
                frame=FRAME_WORLD,
                draw_style="scatter",
                color=NEARBY_AGENT_COLOR,
                alpha=0.28,
                markersize=14.0,
                marker="o",
                zorder=4,
            )
        )
        world_series_for_sanity.append((f"nearby_past_{agent['track_id']}", past_xy))
        world_series_for_sanity.append((f"nearby_current_{agent['track_id']}", current_xy))

    for idx, candidate in enumerate(branch_candidates):
        xy = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)[:, :2]
        xy = _trim_xy_to_radius(xy, center_xy=center_xy, radius_m=WORLD_PATCH_HALF_EXTENT_M + 8.0)
        candidate_id = candidate_id_map[idx]["candidate_id"] if idx < len(candidate_id_map) else f"B{idx}"
        color = _candidate_color(idx)
        series_list.append(
            TaggedXYSeries(
                name=f"candidate_{candidate_id}",
                xy=xy,
                frame=FRAME_WORLD,
                draw_style="line",
                color=color,
                alpha=0.95,
                linewidth=2.8,
                linestyle=":",
                zorder=8,
                annotate=str(candidate_id),
                annotate_index=min(max(xy.shape[0] - 12, 0), xy.shape[0] - 1) if xy.shape[0] else -1,
            )
        )
        world_series_for_sanity.append((f"candidate_{candidate_id}", xy))

    for light in traffic_lights:
        stop_point = np.asarray(light["stop_point_xy_world"], dtype=np.float64).reshape(1, 2)
        color = "#ef4444" if bool(light.get("is_focus")) else "#f59e0b"
        state = str(light.get("state_at_decision") or "unknown")
        label = f"TL {light['light_id']} {state}"
        series_list.append(TaggedXYSeries(name=f"light_{light['light_id']}", xy=stop_point, frame=FRAME_WORLD, draw_style="scatter", color=color, alpha=0.92, markersize=52.0, marker="s", zorder=10, annotate=label))
        world_series_for_sanity.append((f"light_{light['light_id']}", stop_point))

    series_list.append(TaggedXYSeries(name="target_past", xy=np.asarray(target_past_world, dtype=np.float64), frame=FRAME_WORLD, draw_style="line", color="#111827", alpha=0.96, linewidth=3.0, zorder=11))
    series_list.append(TaggedXYSeries(name="target_current", xy=np.asarray([[float(current_pose_world['x']), float(current_pose_world['y'])]], dtype=np.float64), frame=FRAME_WORLD, draw_style="scatter", color="#a855f7", alpha=1.0, markersize=58.0, marker="s", zorder=13, annotate="current"))
    series_list.append(TaggedXYSeries(name="target_decision", xy=np.asarray([[float(decision_pose_world['x']), float(decision_pose_world['y'])]], dtype=np.float64), frame=FRAME_WORLD, draw_style="scatter", color="#6d28d9", alpha=1.0, markersize=58.0, marker="o", zorder=13, annotate="decision"))
    series_list.append(TaggedXYSeries(name="target_decision_heading", xy=_heading_arrow_xy(decision_pose_world), frame=FRAME_WORLD, draw_style="line", color="#6d28d9", alpha=0.9, linewidth=1.8, zorder=12))
    world_series_for_sanity.extend(
        [
            ("target_past", np.asarray(target_past_world, dtype=np.float64)),
            ("target_current", np.asarray([[float(current_pose_world["x"]), float(current_pose_world["y"])]], dtype=np.float64)),
            ("target_decision", np.asarray([[float(decision_pose_world["x"]), float(decision_pose_world["y"])]], dtype=np.float64)),
        ]
    )

    if gt_future_world is not None and np.asarray(gt_future_world).size > 0:
        gt_xy = np.asarray(gt_future_world, dtype=np.float64)
        series_list.append(TaggedXYSeries(name="gt_future", xy=gt_xy, frame=FRAME_WORLD, draw_style="line", color="#dc2626", alpha=0.95, linewidth=3.0, zorder=14))
        world_series_for_sanity.append(("gt_future", gt_xy))
    if gt_final_world is not None and np.asarray(gt_final_world).size > 0:
        gt_final = np.asarray(gt_final_world, dtype=np.float64).reshape(-1, 2)
        series_list.append(TaggedXYSeries(name="gt_final", xy=gt_final, frame=FRAME_WORLD, draw_style="scatter", color="#dc2626", alpha=1.0, markersize=56.0, marker="X", zorder=15, annotate="GT final"))
        world_series_for_sanity.append(("gt_final", gt_final))
    if requested_anchor_world is not None and np.asarray(requested_anchor_world).size > 0:
        anchor_xy = np.asarray(requested_anchor_world, dtype=np.float64).reshape(-1, 2)
        series_list.append(TaggedXYSeries(name="requested_anchor", xy=anchor_xy, frame=FRAME_WORLD, draw_style="scatter", color="#d97706", alpha=1.0, markersize=62.0, marker="*", zorder=16, annotate="anchor"))
        world_series_for_sanity.append(("requested_anchor", anchor_xy))
    split_x = split_point_guess.get("x")
    split_y = split_point_guess.get("y")
    if split_x is not None and split_y is not None:
        split_xy = np.asarray([[float(split_x), float(split_y)]], dtype=np.float64)
        series_list.append(TaggedXYSeries(name="split_point_guess", xy=split_xy, frame=FRAME_WORLD, draw_style="scatter", color="#111827", alpha=1.0, markersize=34.0, marker="P", zorder=17, annotate="S"))
        world_series_for_sanity.append(("split_point_guess", split_xy))

    render_tagged_series_collection(
        ax,
        series_list=series_list,
        expected_frame=FRAME_WORLD,
        example_id=title,
        plot_name=output_path.name,
        failures=plot_failures,
        local_limit_abs_m=LOCAL_SANITY_LIMIT_M,
    )

    ax.set_xlim(float(center_xy[0] - WORLD_PATCH_HALF_EXTENT_M), float(center_xy[0] + WORLD_PATCH_HALF_EXTENT_M))
    ax.set_ylim(float(center_xy[1] - WORLD_PATCH_HALF_EXTENT_M), float(center_xy[1] + WORLD_PATCH_HALF_EXTENT_M))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    scale_x0 = float(center_xy[0] - WORLD_PATCH_HALF_EXTENT_M + 6.0)
    scale_y0 = float(center_xy[1] - WORLD_PATCH_HALF_EXTENT_M + 6.0)
    ax.plot([scale_x0, scale_x0 + 10.0], [scale_y0, scale_y0], color="#111827", linewidth=2.0, zorder=20)
    ax.text(scale_x0 + 5.0, scale_y0 + 1.5, "10m", ha="center", va="bottom", fontsize=8)
    north_x = float(center_xy[0] + WORLD_PATCH_HALF_EXTENT_M - 8.0)
    north_y = float(center_xy[1] + WORLD_PATCH_HALF_EXTENT_M - 14.0)
    ax.annotate("N", xy=(north_x, north_y + 7.0), xytext=(north_x, north_y), arrowprops={"arrowstyle": "-|>", "color": "#111827", "lw": 1.5}, ha="center", va="bottom", fontsize=9)
    ax.text(
        0.02,
        0.98,
        "\n".join(str(line) for line in sidebar_lines if line),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "#e2e8f0", "boxstyle": "round,pad=0.28"},
        zorder=25,
    )
    if not local_sanity_passed:
        ax.text(
            0.5,
            0.5,
            "PLOT INVALID",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#b91c1c",
            fontsize=28,
            fontweight="bold",
            alpha=0.32,
            rotation=22,
        )
        if local_sanity_failures:
            ax.text(
                0.5,
                0.43,
                f"local sanity failed: {local_sanity_failures[0].get('series_name')}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#7f1d1d",
                fontsize=10,
                alpha=0.82,
            )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.01, top=0.965)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_vlm_semantic_views(
    *,
    records: Sequence[Mapping[str, Any]],
    outdir: str | Path,
) -> Dict[str, Any]:
    outdir_path = Path(outdir).expanduser()
    outdir_path.mkdir(parents=True, exist_ok=True)
    corrected_root = outdir_path / "corrected_visuals_vlm"
    corrected_root.mkdir(parents=True, exist_ok=True)

    render_rows: List[Dict[str, Any]] = []
    plot_failures: List[Dict[str, Any]] = []

    for record in records:
        example_id = str(record["example_id"])
        example_root = corrected_root / example_id
        example_root.mkdir(parents=True, exist_ok=True)

        raw_scenario = load_raw_scenario(record["local_scenario_pkl"])
        materialized_controls = load_materialized_controls(record["local_materialized_eval_input"])
        selected_mode = str(record.get("selected_mode") or "factual")
        selected_control_code = None
        if selected_mode == "factual":
            selected_control_code = materialized_controls.get("factual_control_code")
        elif selected_mode.startswith("alternative_"):
            try:
                rank = int(selected_mode.split("_", 1)[1])
            except Exception:
                rank = -1
            alternatives = list(materialized_controls.get("alternative_control_codes") or [])
            if 0 <= rank < len(alternatives):
                selected_control_code = alternatives[rank]
        selected_control_code = dict(selected_control_code or {})

        source_provenance = dict(dict(selected_control_code.get("debug") or {}).get("source_provenance") or {})
        current_time_idx = int(source_provenance.get("current_time_index_global") or 0)
        decision_time_idx = int(source_provenance.get("decision_time_index_global") or record.get("decision_time_idx") or 0)
        target_track_id = str(record["agent_id"])

        current_pose_world = _track_pose_at_index(raw_scenario, track_id=target_track_id, time_index=current_time_idx)
        decision_pose_world = _track_pose_at_index(raw_scenario, track_id=target_track_id, time_index=decision_time_idx)
        if current_pose_world is None or decision_pose_world is None:
            plot_failures.append(
                {
                    "example_id": example_id,
                    "reason": "missing_target_pose",
                    "current_time_idx": current_time_idx,
                    "decision_time_idx": decision_time_idx,
                }
            )
            continue

        state = raw_track_world_state(raw_scenario, track_id=target_track_id)
        valid = np.asarray(state["valid"], dtype=bool)
        position = np.asarray(state["position"], dtype=np.float64)[:, :2]
        target_past_world = position[max(0, current_time_idx - 15) : current_time_idx + 1][valid[max(0, current_time_idx - 15) : current_time_idx + 1]]
        gt_future_world = position[decision_time_idx:][valid[decision_time_idx:]]
        gt_final_world = gt_future_world[-1:] if gt_future_world.size else np.zeros((0, 2), dtype=np.float64)

        branch_candidates = list(materialized_controls.get("branch_candidates") or [])
        candidate_id_map = list(record.get("candidate_id_map") or [])
        selected_anchor_rel = anchor_pose_from_control_code(selected_control_code)
        requested_anchor_world = None
        if selected_anchor_rel is not None:
            requested_anchor_world_pose = agent_relative_pose_to_world(selected_anchor_rel, agent_pose_world=decision_pose_world)
            requested_anchor_world = np.asarray([[requested_anchor_world_pose.x, requested_anchor_world_pose.y]], dtype=np.float64)

        focus_light_id = str(record.get("path_index_row", {}).get("light_id") or load_json(Path(record["local_materialized_eval_input"]) / "factual_control_code.json").get("light_id") or "")
        map_context = _select_map_context(raw_scenario, center_xy=[decision_pose_world["x"], decision_pose_world["y"]], radius_m=55.0)
        traffic_lights = _select_traffic_light_context(
            raw_scenario,
            center_xy=[decision_pose_world["x"], decision_pose_world["y"]],
            time_index=decision_time_idx,
            focus_light_id=focus_light_id or None,
            radius_m=60.0,
        )
        nearby_agents = _select_nearby_agents(
            raw_scenario,
            center_xy=[decision_pose_world["x"], decision_pose_world["y"]],
            current_time_idx=current_time_idx,
            radius_m=45.0,
            past_steps=8,
            exclude_track_id=target_track_id,
        )
        split_point_guess = _estimate_split_point(branch_candidates)

        sanity_passed, sanity_failures = _local_sanity_for_world_series(
            agent_pose_world=decision_pose_world,
            series_items=[
                ("target_past", target_past_world),
                ("gt_future", gt_future_world),
                ("requested_anchor", np.asarray(np.zeros((0, 2), dtype=np.float64) if requested_anchor_world is None else requested_anchor_world, dtype=np.float64)),
                *[
                    (
                        f"candidate_{idx}",
                        _trim_xy_to_radius(
                            np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)[:, :2],
                            center_xy=[decision_pose_world["x"], decision_pose_world["y"]],
                            radius_m=WORLD_PATCH_HALF_EXTENT_M + 8.0,
                        ),
                    )
                    for idx, candidate in enumerate(branch_candidates)
                ],
                *[(f"light_{light['light_id']}", np.asarray(light["stop_point_xy_world"], dtype=np.float64).reshape(1, 2)) for light in traffic_lights],
            ],
        )
        if not sanity_passed:
            plot_failures.append(
                {
                    "example_id": example_id,
                    "reason": "local_frame_sanity_limit_exceeded",
                    "failures": list(sanity_failures),
                }
            )

        sidebar_lines = [
            f"id: {example_id}",
            f"agent: {target_track_id}  t={decision_time_idx}",
            f"mode: {selected_mode}",
            f"lights: {', '.join(str(v) for v in record.get('light_group_ids') or []) or 'none'}",
            f"cands: {', '.join(item['candidate_id'] for item in candidate_id_map) or 'none'}",
            f"frame: world",
        ]

        context_only_path = example_root / "context_only.png"
        context_plus_gt_path = example_root / "context_plus_gt.png"
        context_plus_anchor_path = example_root / "context_plus_anchor.png"
        _build_world_plot(
            output_path=context_only_path,
            title=f"{example_id}\ncontext_only",
            frame_label="world",
            decision_pose_world=decision_pose_world,
            current_pose_world=current_pose_world,
            target_past_world=target_past_world,
            map_context=map_context,
            traffic_lights=traffic_lights,
            nearby_agents=nearby_agents,
            branch_candidates=branch_candidates,
            candidate_id_map=candidate_id_map,
            split_point_guess=split_point_guess,
            gt_future_world=None,
            gt_final_world=None,
            requested_anchor_world=None,
            sidebar_lines=sidebar_lines,
            plot_failures=plot_failures,
            local_sanity_passed=sanity_passed,
            local_sanity_failures=sanity_failures,
        )
        _build_world_plot(
            output_path=context_plus_gt_path,
            title=f"{example_id}\ncontext_plus_gt",
            frame_label="world",
            decision_pose_world=decision_pose_world,
            current_pose_world=current_pose_world,
            target_past_world=target_past_world,
            map_context=map_context,
            traffic_lights=traffic_lights,
            nearby_agents=nearby_agents,
            branch_candidates=branch_candidates,
            candidate_id_map=candidate_id_map,
            split_point_guess=split_point_guess,
            gt_future_world=gt_future_world,
            gt_final_world=gt_final_world,
            requested_anchor_world=None,
            sidebar_lines=sidebar_lines,
            plot_failures=plot_failures,
            local_sanity_passed=sanity_passed,
            local_sanity_failures=sanity_failures,
        )
        _build_world_plot(
            output_path=context_plus_anchor_path,
            title=f"{example_id}\ncontext_plus_anchor",
            frame_label="world",
            decision_pose_world=decision_pose_world,
            current_pose_world=current_pose_world,
            target_past_world=target_past_world,
            map_context=map_context,
            traffic_lights=traffic_lights,
            nearby_agents=nearby_agents,
            branch_candidates=branch_candidates,
            candidate_id_map=candidate_id_map,
            split_point_guess=split_point_guess,
            gt_future_world=None,
            gt_final_world=None,
            requested_anchor_world=requested_anchor_world,
            sidebar_lines=sidebar_lines,
            plot_failures=plot_failures,
            local_sanity_passed=sanity_passed,
            local_sanity_failures=sanity_failures,
        )

        metadata = {
            "example_id": example_id,
            "scenario_id": str(record["scenario_id"]),
            "agent_id": target_track_id,
            "decision_time_idx": decision_time_idx,
            "selected_mode": selected_mode,
            "requested_branch_label": str(record.get("requested_branch_label") or ""),
            "selected_candidate_id": record.get("selected_candidate_id"),
            "selected_candidate_geometry_branch_id": record.get("selected_candidate_geometry_branch_id"),
            "selected_candidate_geometry_label": record.get("selected_candidate_geometry_label"),
            "candidate_id_map": candidate_id_map,
            "light_group_ids": list(record.get("light_group_ids") or []),
            "split_point_guess": split_point_guess,
            "frame_label": "world",
            "local_sanity_passed": bool(sanity_passed),
            "local_sanity_failures": list(sanity_failures),
            "local_plot_extent_x": [-WORLD_PATCH_HALF_EXTENT_M, WORLD_PATCH_HALF_EXTENT_M],
            "local_plot_extent_y": [-WORLD_PATCH_HALF_EXTENT_M, WORLD_PATCH_HALF_EXTENT_M],
        }
        metadata_path = example_root / "metadata.json"
        write_json(metadata_path, metadata)

        render_rows.append(
            {
                "example_id": example_id,
                "scenario_id": str(record["scenario_id"]),
                "agent_id": target_track_id,
                "decision_time_idx": decision_time_idx,
                "selected_mode": selected_mode,
                "requested_branch_label": str(record.get("requested_branch_label") or ""),
                "geometry_branch_label": str(record.get("geometry_branch_label") or ""),
                "geometry_branch_id": str(record.get("geometry_branch_id") or ""),
                "geometry_light_group_id": record.get("path_index_row", {}).get("light_group_id"),
                "geometry_primary_light_id": record.get("path_index_row", {}).get("primary_light_id"),
                "local_scenario_pkl": str(record["local_scenario_pkl"]),
                "local_materialized_eval_input": str(record["local_materialized_eval_input"]),
                "candidate_id_map": candidate_id_map,
                "selected_candidate_id": record.get("selected_candidate_id"),
                "selected_candidate_geometry_branch_id": record.get("selected_candidate_geometry_branch_id"),
                "selected_candidate_geometry_label": record.get("selected_candidate_geometry_label"),
                "light_group_ids": list(record.get("light_group_ids") or []),
                "split_point_guess": split_point_guess,
                "frame_label": "world",
                "images": {
                    "context_only": str(context_only_path),
                    "context_plus_gt": str(context_plus_gt_path),
                    "context_plus_anchor": str(context_plus_anchor_path),
                },
                "metadata_json": str(metadata_path),
                "local_sanity_passed": bool(sanity_passed),
            }
        )

    render_manifest = {
        "num_examples": int(len(render_rows)),
        "num_plot_failures": int(len(plot_failures)),
        "rows": render_rows,
    }
    write_json(outdir_path / "vlm_render_manifest.json", render_manifest)
    write_json(outdir_path / "plot_failures.json", plot_failures)
    return render_manifest
