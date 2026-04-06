from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import (
    enumerate_branch_candidates_from_sdc_paths,
    iter_waymax_simulator_states,
    normalize_scenario,
    raw_scenario_from_waymax_state,
    resolve_waymax_config,
    save_raw_waymax_scenario_pickle,
    waymax_available,
)

DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a few live Waymax/WOMD scenes and plot SDC paths with road context.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--num-scenes", type=int, default=3)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--save-pkls", action="store_true")
    parser.add_argument("--padding-m", type=float, default=20.0)
    parser.add_argument("--max-alternate-paths", type=int, default=5)
    return parser.parse_args()


def _finite_xy_rows(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[-1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    array = array[:, :2]
    mask = np.isfinite(array).all(axis=-1)
    return np.asarray(array[mask], dtype=np.float32)


def _compute_plot_limits(canonical: object, *, padding_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    points: list[np.ndarray] = []
    for feature in canonical.map_features.values():
        polyline = _finite_xy_rows(np.asarray(feature.polyline_xy, dtype=np.float32))
        if polyline.shape[0] > 0:
            points.append(polyline)
    for track in canonical.tracks.values():
        polyline = np.asarray(track.position_xy, dtype=np.float32)
        valid = np.asarray(track.valid, dtype=bool)
        polyline = _finite_xy_rows(polyline[valid])
        if polyline.shape[0] > 0:
            points.append(polyline)
    for path in canonical.sdc_paths.values():
        polyline = np.asarray(path.polyline_xy, dtype=np.float32)
        valid = np.asarray(path.valid, dtype=bool)
        polyline = _finite_xy_rows(polyline[valid])
        if polyline.shape[0] > 0:
            points.append(polyline)
    for light in canonical.traffic_lights.values():
        if light.stop_point_xy is not None:
            light_xy = _finite_xy_rows(np.asarray([light.stop_point_xy], dtype=np.float32))
            if light_xy.shape[0] > 0:
                points.append(light_xy)
    if not points:
        return (-30.0, 30.0), (-30.0, 30.0)
    stacked = np.concatenate(points, axis=0)
    stacked = _finite_xy_rows(stacked)
    if stacked.shape[0] == 0:
        return (-30.0, 30.0), (-30.0, 30.0)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    center = 0.5 * (min_xy + max_xy)
    half_extent = 0.5 * np.max(max_xy - min_xy) + float(padding_m)
    half_extent = max(float(half_extent), 25.0)
    return (float(center[0] - half_extent), float(center[0] + half_extent)), (
        float(center[1] - half_extent),
        float(center[1] + half_extent),
    )


def _plot_scene(raw_scenario: dict, *, out_path: Path, padding_m: float, max_alternate_paths: int) -> dict:
    canonical = normalize_scenario(raw_scenario)
    sdc_track = canonical.tracks[str(canonical.sdc_id)]
    decision_idx = int(canonical.current_time_index)
    if decision_idx >= sdc_track.position_xy.shape[0]:
        decision_idx = max(0, sdc_track.position_xy.shape[0] - 1)
    approach_heading = float(sdc_track.heading[decision_idx]) if np.isfinite(sdc_track.heading[decision_idx]) else 0.0
    branches = enumerate_branch_candidates_from_sdc_paths(
        canonical,
        agent_id=str(canonical.sdc_id),
        decision_time_idx=decision_idx,
        approach_heading=approach_heading,
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    for feature in canonical.map_features.values():
        polyline = _finite_xy_rows(np.asarray(feature.polyline_xy, dtype=np.float32))
        if polyline.shape[0] >= 2:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#d3d3d3", linewidth=1.0, alpha=0.75, zorder=1)

    for light in canonical.traffic_lights.values():
        if light.stop_point_xy is not None:
            light_xy = _finite_xy_rows(np.asarray([light.stop_point_xy], dtype=np.float32))
            if light_xy.shape[0] == 0:
                continue
            ax.scatter(
                [light_xy[0, 0]],
                [light_xy[0, 1]],
                c="#ffd23f",
                marker="s",
                s=70,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label=None,
            )

    for track_id, track in canonical.tracks.items():
        polyline = np.asarray(track.position_xy, dtype=np.float32)
        valid = np.asarray(track.valid, dtype=bool)
        polyline = _finite_xy_rows(polyline[valid])
        if polyline.shape[0] < 2:
            continue
        is_sdc = str(track_id) == str(canonical.sdc_id)
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=("#111111" if is_sdc else "#8f99a3"),
            linewidth=(2.8 if is_sdc else 1.0),
            alpha=(1.0 if is_sdc else 0.45),
            zorder=(4 if is_sdc else 2),
            label=("sdc track" if is_sdc else None),
        )

    all_path_rows = []
    for path_id, path in sorted(canonical.sdc_paths.items()):
        polyline = np.asarray(path.polyline_xy, dtype=np.float32)
        valid = np.asarray(path.valid, dtype=bool)
        polyline = _finite_xy_rows(polyline[valid])
        metadata = dict(path.metadata or {})
        on_route = bool(metadata.get("on_route", False))
        all_path_rows.append(
            {
                "path_id": str(path_id),
                "on_route": on_route,
                "num_points": int(polyline.shape[0]),
                "polyline_xy": polyline,
            }
        )

    rows_with_any_valid = [row for row in all_path_rows if int(row["num_points"]) >= 1]
    displayable_paths = [row for row in all_path_rows if int(row["num_points"]) >= 2]
    on_route_paths = [row for row in displayable_paths if bool(row["on_route"])]
    alternate_paths = [row for row in displayable_paths if not bool(row["on_route"])]
    plotted_paths = on_route_paths[:1] + alternate_paths[: max(0, int(max_alternate_paths))]

    path_palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    path_rows = []
    for idx, row in enumerate(plotted_paths):
        polyline = np.asarray(row["polyline_xy"], dtype=np.float32)
        color = path_palette[idx % len(path_palette)]
        on_route = bool(row["on_route"])
        label_suffix = " [on_route]" if on_route else " [alt]"
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=color,
            linewidth=(2.8 if on_route else 1.8),
            alpha=(0.98 if on_route else 0.7),
            linestyle=("-" if on_route else "--"),
            zorder=(6 if on_route else 3),
            label=f"{row['path_id']}{label_suffix}",
        )
        ax.scatter([polyline[-1, 0]], [polyline[-1, 1]], c=color, s=48, zorder=7)
        path_rows.append(
            {
                "path_id": str(row["path_id"]),
                "on_route": on_route,
                "num_points": int(row["num_points"]),
            }
        )

    for branch in branches:
        polyline = _finite_xy_rows(np.asarray(branch.polyline_xy, dtype=np.float32))
        if polyline.shape[0] < 2:
            continue
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color="#000000",
            linewidth=4.0,
            alpha=0.18,
            zorder=8,
            label=None,
        )

    decision_xy = _finite_xy_rows(np.asarray(sdc_track.position_xy[decision_idx], dtype=np.float32))
    if decision_xy.shape[0] > 0:
        ax.scatter(
            [decision_xy[0, 0]],
            [decision_xy[0, 1]],
            c="#7b3294",
            s=90,
            marker="o",
            edgecolors="white",
            linewidths=1.2,
            zorder=9,
            label="decision/current",
        )

    xlim, ylim = _compute_plot_limits(canonical, padding_m=padding_m)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{canonical.scenario_id} | sdc={canonical.sdc_id} | "
        f"{len(canonical.sdc_paths)} sdc_paths | "
        f"alts {len(alternate_paths)} total, {min(len(alternate_paths), max(0, int(max_alternate_paths)))} shown"
    )
    ax.grid(alpha=0.12)
    ax.legend(loc="best", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    branch_rows = [
        {
            "branch_id": str(branch.branch_id),
            "branch_label": str(branch.branch_label),
            "source_kind": str(branch.source_kind),
            "rank_score": float(branch.rank_score),
        }
        for branch in branches
    ]
    return {
        "scenario_id": str(canonical.scenario_id),
        "sdc_id": str(canonical.sdc_id),
        "decision_time_idx": int(decision_idx),
        "num_sdc_paths_raw": int(len(canonical.sdc_paths)),
        "num_rows_with_any_valid": int(len(rows_with_any_valid)),
        "num_sdc_paths_displayable": int(len(displayable_paths)),
        "num_on_route_paths_total": int(len(on_route_paths)),
        "num_on_route_paths_plotted": int(sum(1 for row in path_rows if bool(row["on_route"]))),
        "num_alternate_paths_total": int(len(alternate_paths)),
        "num_alternate_paths_plotted": int(sum(1 for row in path_rows if not bool(row["on_route"]))),
        "num_branch_candidates": int(len(branches)),
        "sdc_paths": path_rows,
        "branch_candidates": branch_rows,
        "overlay_png": str(out_path),
    }


def main() -> int:
    args = parse_args()
    if not waymax_available():
        raise SystemExit("waymax is not installed in this environment")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    config = resolve_waymax_config(
        config_name=str(args.config_name),
        path=str(args.path),
        include_sdc_paths=True,
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )

    rows = []
    data_iter = iter_waymax_simulator_states(config)
    selected_iter = itertools.islice(data_iter, int(args.scene_offset), int(args.scene_offset) + int(args.num_scenes))
    for local_idx, state in enumerate(selected_iter):
        raw = raw_scenario_from_waymax_state(
            state,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or f"waymax_scene_{local_idx:04d}")
        overlay_path = outdir / f"{local_idx:03d}_{scenario_id}_sdc_paths.png"
        row = _plot_scene(
            raw,
            out_path=overlay_path,
            padding_m=float(args.padding_m),
            max_alternate_paths=int(args.max_alternate_paths),
        )
        if bool(args.save_pkls):
            pkl_path = outdir / f"sd_waymo_v1.3.1_{scenario_id}.pkl"
            save_raw_waymax_scenario_pickle(
                state,
                out_path=pkl_path,
                current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
            )
            row["output_pkl"] = str(pkl_path)
        print(
            json.dumps(
                {
                    "scene_index": int(args.scene_offset) + local_idx,
                    "scenario_id": scenario_id,
                    "num_sdc_paths_raw": row["num_sdc_paths_raw"],
                    "num_rows_with_any_valid": row["num_rows_with_any_valid"],
                    "num_sdc_paths_displayable": row["num_sdc_paths_displayable"],
                    "num_on_route_paths_total": row["num_on_route_paths_total"],
                    "num_on_route_paths_plotted": row["num_on_route_paths_plotted"],
                    "num_alternate_paths_total": row["num_alternate_paths_total"],
                    "num_alternate_paths_plotted": row["num_alternate_paths_plotted"],
                    "num_branch_candidates": row["num_branch_candidates"],
                    "overlay_png": row["overlay_png"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        rows.append(row)

    summary = {
        "config_name": str(args.config_name),
        "path": str(args.path),
        "num_scenes_requested": int(args.num_scenes),
        "scene_offset": int(args.scene_offset),
        "num_scenes_written": int(len(rows)),
        "num_paths": int(args.num_paths),
        "num_points_per_path": int(args.num_points_per_path),
        "max_alternate_paths": int(args.max_alternate_paths),
        "rows": rows,
    }
    summary_path = outdir / "waymax_sdc_paths_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "num_scenes_written": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
