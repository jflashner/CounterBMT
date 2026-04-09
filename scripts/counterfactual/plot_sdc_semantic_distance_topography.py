from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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

from bmt.counterfactual.sdc_path_control import split_polyline_on_discontinuities
from bmt.counterfactual.sdc_semantic_control import DEFAULT_FAMILY_PATH_DEADBAND_M, load_raw_scenario_from_row
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _finite_xy_rows,
)
from scripts.counterfactual.label_waymax_sdc_path_semantics import PLOT_RADIUS_M, SDC_VERTICAL_FRACTION, _world_to_sdc_up_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a topographic nearest-path-distance overlay for one semantic control row."
    )
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--scenario-id", type=str, required=True)
    parser.add_argument("--slot-id", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--proposed-deadband-m", type=float, default=3.0)
    parser.add_argument("--current-deadband-m", type=float, default=DEFAULT_FAMILY_PATH_DEADBAND_M)
    parser.add_argument("--grid-step-m", type=float, default=0.35)
    parser.add_argument("--max-contour-distance-m", type=float, default=10.0)
    parser.add_argument("--progress-margin-m", type=float, default=0.25)
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _select_row(rows: Sequence[Mapping[str, Any]], *, scenario_id: str, slot_id: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("scenario_id") or "") != str(scenario_id):
            continue
        if str(row.get("selected_slot_id") or "") != str(slot_id):
            continue
        return dict(row)
    raise ValueError(f"No row found for scenario_id={scenario_id!r} slot_id={slot_id!r}")


def _selected_path_world(row: Mapping[str, Any]) -> np.ndarray:
    selected_path_id = row.get("selected_path_id")
    family_ids = list(row.get("candidate_family_path_ids") or [])
    family_paths = list(row.get("candidate_family_resampled_paths_world") or [])
    if selected_path_id is not None:
        for idx, path_id in enumerate(family_ids):
            if path_id == selected_path_id and idx < len(family_paths):
                return _finite_xy_rows(np.asarray(family_paths[idx], dtype=np.float64))
    if family_paths:
        return _finite_xy_rows(np.asarray(family_paths[0], dtype=np.float64))
    raise ValueError("Row does not contain any candidate family path world points.")


def _path_segments_and_markers(path_world: np.ndarray, *, jump_threshold_m: float = 4.0):
    segments = [
        np.asarray(seg, dtype=np.float64)
        for seg in split_polyline_on_discontinuities(path_world, jump_threshold_m=float(jump_threshold_m))
        if np.asarray(seg).shape[0] >= 2
    ]
    discontinuity_markers: List[np.ndarray] = []
    for segment in segments[1:]:
        discontinuity_markers.append(np.asarray(segment[0], dtype=np.float64))
    return segments, discontinuity_markers


def _sdc_up_to_world_frame(xy_local: np.ndarray, *, center_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    xy = np.asarray(xy_local, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    rot = float(heading_rad) - (math.pi / 2.0)
    c = math.cos(rot)
    s = math.sin(rot)
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(center_xy[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(center_xy[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float64)


def _nearest_waypoint_distance_field(
    *,
    path_world_xy: np.ndarray,
    center_xy: np.ndarray,
    heading_rad: float,
    grid_step_m: float,
):
    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float64)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    local_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    world_points = _sdc_up_to_world_frame(local_points, center_xy=center_xy, heading_rad=heading_rad)
    diff = world_points[:, None, :] - np.asarray(path_world_xy, dtype=np.float64)[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    nearest_distance = np.min(d, axis=-1).reshape(xx.shape)
    return xx, yy, nearest_distance


def _polyline_arc_lengths(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(seg, dtype=np.float64)], axis=0)


def _nearest_waypoint_projection_field(
    *,
    path_world_xy: np.ndarray,
    center_xy: np.ndarray,
    heading_rad: float,
    grid_step_m: float,
):
    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float64)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    local_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    world_points = _sdc_up_to_world_frame(local_points, center_xy=center_xy, heading_rad=heading_rad)
    path_world = np.asarray(path_world_xy, dtype=np.float64).reshape(-1, 2)
    diff = world_points[:, None, :] - path_world[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    nearest_idx = np.argmin(d, axis=-1)
    nearest_distance = np.take_along_axis(d, nearest_idx[:, None], axis=-1).reshape(-1)
    arc_lengths = _polyline_arc_lengths(path_world)
    nearest_arc = arc_lengths[nearest_idx]
    return xx, yy, nearest_distance.reshape(xx.shape), nearest_idx.reshape(xx.shape), nearest_arc.reshape(xx.shape)


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    row = _select_row(rows, scenario_id=str(args.scenario_id), slot_id=str(args.slot_id))
    raw_scenario = load_raw_scenario_from_row(row)
    render_context = _extract_scene_render_context(raw_scenario, row)
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])

    path_world = _selected_path_world(row)
    path_segments_world, discontinuity_markers_world = _path_segments_and_markers(path_world)
    scenario_slug = str(row["scenario_id"]).replace("/", "_")
    slot_slug = str(row["selected_slot_id"]).replace("/", "_")

    xx, yy, raw_distance, nearest_idx_field, nearest_arc_field = _nearest_waypoint_projection_field(
        path_world_xy=path_world,
        center_xy=current_xy,
        heading_rad=current_heading,
        grid_step_m=float(args.grid_step_m),
    )
    raw_distance = np.asarray(raw_distance, dtype=np.float64)
    path_arc_lengths = _polyline_arc_lengths(path_world)
    current_idx = int(
        np.argmin(np.linalg.norm(np.asarray(path_world, dtype=np.float64) - current_xy.reshape(1, 2), axis=-1))
    )
    current_arc = float(path_arc_lengths[current_idx]) if path_arc_lengths.size > 0 else 0.0
    proposed_deadband = max(float(args.proposed_deadband_m), 1e-3)
    current_deadband = max(float(args.current_deadband_m), 1e-3)
    progress_margin = max(float(args.progress_margin_m), 1e-4)
    proposed_penalty = np.maximum(raw_distance - proposed_deadband, 0.0)
    progress_delta_arc = np.asarray(nearest_arc_field, dtype=np.float64) - current_arc
    progress_penalty = np.maximum(progress_margin - progress_delta_arc, 0.0)

    fig = plt.figure(figsize=(8.3, 8.3), dpi=180)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=render_context,
        highlighted_segments_world=path_segments_world,
        highlighted_gradient_values=None,
        representative_route_world=path_world,
        info_box_text=(
            f"scene={row['scenario_id']}\n"
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}\n"
            f"metric=nearest resampled waypoint distance\n"
            f"current_deadband={current_deadband:.1f}m  proposed_deadband={proposed_deadband:.1f}m"
        ),
        show_colorbar=False,
    )

    max_contour = max(float(args.max_contour_distance_m), proposed_deadband + 1.0)
    fill_levels = np.linspace(0.0, proposed_deadband, num=10)
    fill_values = np.ma.masked_where(raw_distance > proposed_deadband, raw_distance)
    ax.contourf(
        xx,
        yy,
        fill_values,
        levels=fill_levels,
        cmap="Blues_r",
        alpha=0.26,
        zorder=6.3,
        antialiased=True,
    )
    topo_levels = np.arange(1.0, max_contour + 1e-6, 1.0)
    contours = ax.contour(
        xx,
        yy,
        np.clip(raw_distance, 0.0, max_contour),
        levels=topo_levels,
        colors="#0f172a",
        linewidths=0.65,
        alpha=0.38,
        zorder=6.7,
    )
    ax.clabel(contours, fmt="%dm", fontsize=6, inline=True)
    ax.contour(
        xx,
        yy,
        raw_distance,
        levels=[current_deadband],
        colors=["#22c55e"],
        linewidths=1.6,
        linestyles=["--"],
        zorder=11.2,
    )
    ax.contour(
        xx,
        yy,
        raw_distance,
        levels=[proposed_deadband],
        colors=["#ef4444"],
        linewidths=2.0,
        zorder=11.3,
    )
    penalty_contours = ax.contour(
        xx,
        yy,
        np.clip(proposed_penalty, 0.0, max_contour),
        levels=np.arange(0.5, max_contour + 1e-6, 1.0),
        colors="#7c3aed",
        linewidths=0.4,
        alpha=0.18,
        zorder=6.6,
    )
    for collection in penalty_contours.collections:
        collection.set_linestyle((0, (2, 3)))

    if discontinuity_markers_world:
        discontinuity_local = _world_to_sdc_up_frame(
            np.asarray(discontinuity_markers_world, dtype=np.float64),
            center_xy=current_xy,
            heading_rad=current_heading,
        )
        ax.scatter(
            discontinuity_local[:, 0],
            discontinuity_local[:, 1],
            marker="x",
            s=70,
            c="#ef4444",
            linewidths=1.6,
            zorder=12.0,
        )

    ax.text(
        0.02,
        0.02,
        "Blue fill: proposed zero-cost basin (d <= proposed deadband)\n"
        "Green dashed: current 1m deadband boundary\n"
        "Red solid: proposed deadband boundary\n"
        "Red x: discontinuity handoff",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
        zorder=15,
    )

    png_path = outdir / f"{scenario_slug}_{slot_slug}_distance_topography.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    gated_progress_penalty = np.where(raw_distance <= proposed_deadband, progress_penalty, np.nan)
    raw_zero_fraction = float(np.mean(progress_penalty <= 1e-6))
    gated_valid_mask = np.isfinite(gated_progress_penalty)
    gated_zero_fraction = float(
        np.mean(gated_progress_penalty[gated_valid_mask] <= 1e-6) if np.any(gated_valid_mask) else 0.0
    )

    progress_fig, progress_axes = plt.subplots(1, 2, figsize=(15.8, 8.3), dpi=180)
    progress_fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.05, wspace=0.04)
    progress_panels = [
        (
            progress_axes[0],
            progress_penalty,
            (
                f"scene={row['scenario_id']}\n"
                f"slot={row['selected_slot_id']}\n"
                f"requested={row['requested_semantic_label']}\n"
                f"metric=raw progress penalty from projected arc\n"
                f"current_arc={current_arc:.2f}m  margin={progress_margin:.2f}m\n"
                f"zero_penalty_fraction={raw_zero_fraction:.3f}"
            ),
            "Raw progress term",
            False,
        ),
        (
            progress_axes[1],
            gated_progress_penalty,
            (
                f"scene={row['scenario_id']}\n"
                f"slot={row['selected_slot_id']}\n"
                f"requested={row['requested_semantic_label']}\n"
                f"metric=progress penalty gated by distance <= {proposed_deadband:.1f}m\n"
                f"current_arc={current_arc:.2f}m  margin={progress_margin:.2f}m\n"
                f"zero_penalty_fraction_within_gate={gated_zero_fraction:.3f}"
            ),
            "Distance-gated progress term",
            True,
        ),
    ]
    progress_levels = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    arc_levels = np.arange(
        float(math.floor(np.min(progress_delta_arc))),
        float(math.ceil(np.max(progress_delta_arc))) + 1e-6,
        2.0,
        dtype=np.float64,
    )
    discontinuity_local = None
    if discontinuity_markers_world:
        discontinuity_local = _world_to_sdc_up_frame(
            np.asarray(discontinuity_markers_world, dtype=np.float64),
            center_xy=current_xy,
            heading_rad=current_heading,
        )
    for panel_ax, panel_penalty, info_text, panel_title, show_gate in progress_panels:
        _draw_vlm_style_scene_ax(
            fig=progress_fig,
            ax=panel_ax,
            render_context=render_context,
            highlighted_segments_world=path_segments_world,
            highlighted_gradient_values=None,
            representative_route_world=path_world,
            info_box_text=info_text,
            show_colorbar=False,
        )
        panel_ax.set_title(panel_title, fontsize=11, pad=8)
        zero_penalty_mask = np.ma.masked_where(panel_penalty > 1e-6, panel_penalty)
        panel_ax.contourf(
            xx,
            yy,
            zero_penalty_mask,
            levels=np.linspace(0.0, 1e-3, num=2),
            colors=["#86efac"],
            alpha=0.24,
            zorder=6.2,
            antialiased=True,
        )
        panel_ax.contour(
            xx,
            yy,
            progress_delta_arc,
            levels=[0.0],
            colors=["#f59e0b"],
            linewidths=1.2,
            linestyles=["--"],
            zorder=10.9,
        )
        panel_ax.contour(
            xx,
            yy,
            progress_delta_arc,
            levels=[progress_margin],
            colors=["#16a34a"],
            linewidths=1.8,
            zorder=11.0,
        )
        finite_penalty = panel_penalty[np.isfinite(panel_penalty)]
        valid_levels = progress_levels[progress_levels < float(np.max(finite_penalty) + 1e-6)] if finite_penalty.size > 0 else np.zeros((0,), dtype=np.float64)
        if valid_levels.size > 0:
            progress_contours = panel_ax.contour(
                xx,
                yy,
                panel_penalty,
                levels=valid_levels,
                colors="#7c2d12",
                linewidths=0.65,
                alpha=0.42,
                zorder=6.6,
            )
            panel_ax.clabel(progress_contours, fmt="%.2fm", fontsize=6, inline=True)
        if arc_levels.size > 1:
            arc_contours = panel_ax.contour(
                xx,
                yy,
                progress_delta_arc,
                levels=arc_levels,
                colors="#1d4ed8",
                linewidths=0.35,
                alpha=0.14,
                zorder=6.1,
            )
            panel_ax.clabel(arc_contours, fmt="%.0fm", fontsize=5, inline=True)
        if show_gate:
            panel_ax.contour(
                xx,
                yy,
                raw_distance,
                levels=[proposed_deadband],
                colors=["#ef4444"],
                linewidths=1.8,
                zorder=11.1,
            )
        if discontinuity_local is not None:
            panel_ax.scatter(
                discontinuity_local[:, 0],
                discontinuity_local[:, 1],
                marker="x",
                s=70,
                c="#ef4444",
                linewidths=1.6,
                zorder=12.0,
            )

    progress_fig.text(
        0.5,
        0.02,
        "Left: raw rollout progress term from projected arc only. "
        "Right: same term, but only shown inside the proposed distance corridor. "
        "If the left panel is mostly green, that means projected progress alone is too permissive.",
        fontsize=8.0,
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
    )
    progress_png_path = outdir / f"{scenario_slug}_{slot_slug}_progress_topography.png"
    progress_fig.savefig(progress_png_path, dpi=180)
    plt.close(progress_fig)

    metadata = {
        "scenario_id": str(row["scenario_id"]),
        "slot_id": str(row["selected_slot_id"]),
        "requested_semantic_label": str(row["requested_semantic_label"]),
        "selected_path_id": row.get("selected_path_id"),
        "current_deadband_m": current_deadband,
        "proposed_deadband_m": proposed_deadband,
        "progress_margin_m": progress_margin,
        "current_arc_m": current_arc,
        "raw_progress_zero_fraction": raw_zero_fraction,
        "gated_progress_zero_fraction": gated_zero_fraction,
        "grid_step_m": float(args.grid_step_m),
        "num_path_points": int(path_world.shape[0]),
        "num_discontinuities": int(max(0, len(path_segments_world) - 1)),
        "distance_metric": "nearest_resampled_waypoint_distance",
        "png": str(png_path),
        "progress_png": str(progress_png_path),
    }
    metadata_path = outdir / f"{scenario_slug}_{slot_slug}_distance_topography.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
