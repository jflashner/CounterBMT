from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .types import CanonicalScenario

try:  # pragma: no cover - import error path depends on env
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - handled at runtime
    plt = None


def render_bev_overview(canonical: CanonicalScenario, path: str | Path) -> None:
    if plt is None:  # pragma: no cover - exercised only in minimal envs
        raise RuntimeError("matplotlib is required to render bev_overview.png")

    fig, ax = plt.subplots(figsize=(10, 10))

    for feature in canonical.map_features.values():
        if feature.polyline_xy.shape[0] > 1:
            ax.plot(
                feature.polyline_xy[:, 0],
                feature.polyline_xy[:, 1],
                color=_map_feature_color(feature.feature_type),
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )
        if feature.polygon_xy is not None and feature.polygon_xy.shape[0] > 2:
            ax.fill(
                feature.polygon_xy[:, 0],
                feature.polygon_xy[:, 1],
                color=_map_feature_fill_color(feature.feature_type),
                alpha=0.12,
                zorder=0,
            )

    ooi = set(canonical.objects_of_interest)
    for track_id, track in canonical.tracks.items():
        valid = np.asarray(track.valid, dtype=bool) & np.isfinite(track.position_xy).all(axis=-1)
        if not np.any(valid):
            continue
        xy = track.position_xy[valid]
        is_sdc = track_id == canonical.sdc_id
        is_ooi = track_id in ooi
        color = "#1f77b4" if is_sdc else ("#ff7f0e" if is_ooi else "#6b7280")
        linewidth = 2.8 if is_sdc else (1.8 if is_ooi else 0.9)
        alpha = 1.0 if is_sdc else (0.9 if is_ooi else 0.45)
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=3)

        current_idx = min(max(canonical.current_time_index, 0), track.position_xy.shape[0] - 1)
        if valid[current_idx]:
            current_xy = track.position_xy[current_idx]
            ax.scatter(
                [current_xy[0]],
                [current_xy[1]],
                s=38 if is_sdc else 24,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

    for light in canonical.traffic_lights.values():
        if light.stop_point_xy is None:
            continue
        current_idx = min(max(canonical.current_time_index, 0), max(0, len(light.object_state) - 1))
        state = light.object_state[current_idx] if light.object_state else None
        ax.scatter(
            [light.stop_point_xy[0]],
            [light.stop_point_xy[1]],
            s=42,
            c=_traffic_light_color(state),
            marker="s",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.set_title(f"BEV Overview: {canonical.scenario_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(Path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def _map_feature_color(feature_type: str) -> str:
    key = str(feature_type).upper()
    if "LANE" in key:
        return "#b0b7c3"
    if "ROAD_EDGE" in key:
        return "#4b5563"
    if "ROAD_LINE" in key:
        return "#d1a72c"
    if "CROSSWALK" in key:
        return "#10b981"
    if "STOP_SIGN" in key:
        return "#ef4444"
    return "#9ca3af"


def _map_feature_fill_color(feature_type: str) -> str:
    key = str(feature_type).upper()
    if "CROSSWALK" in key:
        return "#86efac"
    return "#d1d5db"


def _traffic_light_color(state: str | None) -> str:
    text = "" if state is None else str(state).upper()
    if "GO" in text or "GREEN" in text:
        return "#22c55e"
    if "CAUTION" in text or "YELLOW" in text:
        return "#f59e0b"
    if "STOP" in text or "RED" in text:
        return "#ef4444"
    return "#9ca3af"


def plot_stop_point_distance_curve(
    *,
    ts: np.ndarray,
    distance_curve_m: np.ndarray,
    threshold_m: float,
    first_time_under_threshold_idx: int | None,
    t_min_dist_idx: int | None,
    out_path: str | Path,
) -> None:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render stop point distance plots")
    time_axis = np.asarray(ts, dtype=np.float32)
    if time_axis.shape[0] != distance_curve_m.shape[0]:
        time_axis = np.arange(distance_curve_m.shape[0], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, distance_curve_m, color="#1f2937", linewidth=2.0)
    ax.axhline(float(threshold_m), color="#ef4444", linestyle="--", linewidth=1.2, label=f"{threshold_m:.0f} m threshold")
    if first_time_under_threshold_idx is not None and 0 <= int(first_time_under_threshold_idx) < time_axis.shape[0]:
        idx = int(first_time_under_threshold_idx)
        ax.scatter([time_axis[idx]], [distance_curve_m[idx]], c="#2563eb", s=36, label="first under threshold")
    if t_min_dist_idx is not None and 0 <= int(t_min_dist_idx) < time_axis.shape[0]:
        idx = int(t_min_dist_idx)
        ax.scatter([time_axis[idx]], [distance_curve_m[idx]], c="#16a34a", s=36, label="min distance")
    ax.set_xlabel("time")
    ax.set_ylabel("distance to stop point (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_local_patch(
    *,
    stop_point_xy: Tuple[float, float],
    radius_m: float,
    lane_features: Sequence[Dict[str, Any]],
    nearby_tracks: Sequence[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render local patch plots")
    fig, ax = plt.subplots(figsize=(8, 8))
    center = np.asarray(stop_point_xy, dtype=np.float32)
    circle = plt.Circle((center[0], center[1]), float(radius_m), color="#d1d5db", fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(circle)
    for feature in lane_features:
        polyline = np.asarray(feature.get("polyline_xy", []), dtype=np.float32)
        if polyline.ndim == 2 and polyline.shape[0] > 1:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#6b7280", linewidth=1.2, alpha=0.8)
    for track in nearby_tracks:
        pos = np.asarray(track.get("position_xy", []), dtype=np.float32)
        if pos.shape[0] >= 2:
            color = "#2563eb" if track.get("is_sdc") else ("#f97316" if track.get("is_object_of_interest") else "#111827")
            ax.scatter([pos[0]], [pos[1]], c=color, s=35)
            ax.text(float(pos[0]), float(pos[1]), str(track.get("track_id")), fontsize=7)
    ax.scatter([center[0]], [center[1]], c="#dc2626", marker="s", s=50)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_branch_candidates(
    *,
    stop_point_xy: Tuple[float, float],
    lane_features: Sequence[Dict[str, Any]],
    branch_candidates: Sequence[Dict[str, Any]],
    sdc_past_xy: np.ndarray,
    sdc_future_xy: np.ndarray,
    current_xy: Tuple[float, float],
    decision_xy: Tuple[float, float],
    approach_heading: float,
    current_heading: float,
    current_time_idx: int,
    decision_time_idx: int,
    agent_id: str,
    gt_branch_id: Optional[str],
    out_path: str | Path,
) -> None:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render branch candidate plots")
    fig, ax = plt.subplots(figsize=(8, 8))
    for feature in lane_features:
        polyline = np.asarray(feature.get("polyline_xy", []), dtype=np.float32)
        if polyline.ndim == 2 and polyline.shape[0] > 1:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#9ca3af", linewidth=1.0, alpha=0.75)
    colors = {"left": "#2563eb", "straight": "#16a34a", "right": "#ea580c", "u_turn": "#7c3aed"}
    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float32)
        label = str(candidate.get("branch_label", "unknown"))
        color = colors.get(label, "#111827")
        is_gt = gt_branch_id is not None and str(candidate.get("branch_id")) == str(gt_branch_id)
        if polyline.ndim == 2 and polyline.shape[0] > 1:
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=color,
                linewidth=3.0 if is_gt else 1.8,
                alpha=0.95 if is_gt else 0.7,
                linestyle="-" if is_gt else "--",
            )
        terminal = candidate.get("terminal_pose", {})
        if terminal:
            ax.scatter([terminal.get("x")], [terminal.get("y")], c=color, s=50 if is_gt else 35)
            ax.text(
                float(terminal.get("x")),
                float(terminal.get("y")),
                str(candidate.get("branch_id")),
                fontsize=7,
                color=color,
            )
    if sdc_past_xy.ndim == 2 and sdc_past_xy.shape[0] > 1:
        ax.plot(sdc_past_xy[:, 0], sdc_past_xy[:, 1], color="#111827", linewidth=2.2, alpha=0.85, label="actual past")
    if sdc_future_xy.ndim == 2 and sdc_future_xy.shape[0] > 1:
        ax.plot(sdc_future_xy[:, 0], sdc_future_xy[:, 1], color="#111827", linewidth=2.5, linestyle="--", label="actual future")
    stop_xy = np.asarray(stop_point_xy, dtype=np.float32)
    ax.scatter([stop_xy[0]], [stop_xy[1]], c="#dc2626", marker="s", s=55, label="stop point")
    ax.text(float(stop_xy[0]), float(stop_xy[1]), "stop", fontsize=8, color="#991b1b")
    current_xy_arr = np.asarray(current_xy, dtype=np.float32)
    ax.scatter([current_xy_arr[0]], [current_xy_arr[1]], c="#2563eb", s=60, edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(float(current_xy_arr[0]), float(current_xy_arr[1]), f"current t={current_time_idx}", fontsize=8, color="#1d4ed8")
    dec_xy = np.asarray(decision_xy, dtype=np.float32)
    ax.scatter([dec_xy[0]], [dec_xy[1]], c="#7c3aed", s=55, marker="D", edgecolors="white", linewidths=0.8, zorder=6)
    ax.text(float(dec_xy[0]), float(dec_xy[1]), f"decision t={decision_time_idx}", fontsize=8, color="#6d28d9")
    ax.arrow(
        float(current_xy_arr[0]),
        float(current_xy_arr[1]),
        4.0 * np.cos(float(current_heading)),
        4.0 * np.sin(float(current_heading)),
        color="#0f172a",
        width=0.12,
        length_includes_head=True,
        zorder=6,
    )
    ax.arrow(
        float(dec_xy[0]),
        float(dec_xy[1]),
        3.0 * np.cos(float(approach_heading)),
        3.0 * np.sin(float(approach_heading)),
        color="#7c3aed",
        width=0.08,
        length_includes_head=True,
        zorder=5,
    )
    ax.set_title(f"Branch Candidates: agent {agent_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_conflict_plot(
    *,
    stop_point_xy: Tuple[float, float],
    core_radius_m: float,
    sdc_position_xy: Tuple[float, float],
    eta_table: Sequence[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    if plt is None:  # pragma: no cover
        raise RuntimeError("matplotlib is required to render conflict plots")
    fig, ax = plt.subplots(figsize=(8, 8))
    stop_xy = np.asarray(stop_point_xy, dtype=np.float32)
    core = plt.Circle((stop_xy[0], stop_xy[1]), float(core_radius_m), color="#d1d5db", fill=False, linestyle="--", linewidth=1.2)
    ax.add_patch(core)
    ax.scatter([stop_xy[0]], [stop_xy[1]], c="#dc2626", marker="s", s=50)
    ax.scatter([sdc_position_xy[0]], [sdc_position_xy[1]], c="#2563eb", s=45, label="SDC")
    for record in eta_table:
        pos = record.get("current_position_xy", [None, None])
        if pos[0] is None or pos[1] is None:
            continue
        is_conflict = record.get("eta_gap_s") is not None and float(record.get("eta_gap_s")) <= 3.0 and record.get("eta_s") is not None
        color = "#ef4444" if is_conflict else "#6b7280"
        ax.scatter([pos[0]], [pos[1]], c=color, s=30)
        ax.text(float(pos[0]), float(pos[1]), str(record.get("track_id")), fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
