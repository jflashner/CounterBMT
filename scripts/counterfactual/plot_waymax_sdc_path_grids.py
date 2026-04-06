from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
    iter_waymax_simulator_states,
    normalize_scenario,
    raw_scenario_from_waymax_state,
    resolve_waymax_config,
    save_raw_waymax_scenario_pickle,
    waymax_available,
)
from bmt.counterfactual.sdc_path_control import split_polyline_on_discontinuities
from bmt.counterfactual.types import stable_string_sort_key

DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all displayable SDC paths for each Waymax/WOMD scene in a grid.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--padding-m", type=float, default=18.0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--save-pkls", action="store_true")
    return parser.parse_args()


def _finite_xy_rows(points: Any) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    array = array[:, :2]
    mask = np.isfinite(array).all(axis=-1)
    return np.asarray(array[mask], dtype=np.float32)


def _valid_segments(points_xy: Any, valid: Any | None = None) -> List[np.ndarray]:
    xy = np.asarray(points_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return []
    xy = xy[:, :2]
    if valid is None:
        valid_mask = np.isfinite(xy).all(axis=-1)
    else:
        valid_mask = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != xy.shape[0]:
            valid_mask = np.ones((xy.shape[0],), dtype=bool)
        valid_mask = valid_mask & np.isfinite(xy).all(axis=-1)

    segments: List[np.ndarray] = []
    start: int | None = None
    for idx, is_valid in enumerate(valid_mask.tolist()):
        if is_valid and start is None:
            start = idx
        elif (not is_valid) and start is not None:
            chunk = np.asarray(xy[start:idx], dtype=np.float32)
            segments.extend(split_polyline_on_discontinuities(chunk))
            start = None
    if start is not None:
        chunk = np.asarray(xy[start:], dtype=np.float32)
        segments.extend(split_polyline_on_discontinuities(chunk))
    return [segment for segment in segments if np.asarray(segment).shape[0] >= 2]


def _plot_segmented(ax, points_xy: Any, *, valid: Any | None = None, label: str | None = None, **kwargs) -> None:
    segments = _valid_segments(points_xy, valid)
    for idx, segment in enumerate(segments):
        ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)


def _collect_displayable_paths(canonical: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for order_idx, (path_id, path) in enumerate(sorted(canonical.sdc_paths.items(), key=lambda item: stable_string_sort_key(item[0]))):
        polyline = np.asarray(path.polyline_xy, dtype=np.float32)
        valid = np.asarray(path.valid, dtype=bool).reshape(-1)
        segments = _valid_segments(polyline, valid)
        num_points = int(sum(segment.shape[0] for segment in segments))
        if num_points < 2:
            continue
        metadata = dict(path.metadata or {})
        path_length_m = 0.0
        for segment in segments:
            if segment.shape[0] >= 2:
                path_length_m += float(np.linalg.norm(np.diff(segment, axis=0), axis=-1).sum())
        rows.append(
            {
                "order_idx": int(order_idx),
                "path_id": str(path_id),
                "on_route": bool(metadata.get("on_route", False)),
                "segments": segments,
                "num_points": num_points,
                "path_length_m": float(path_length_m),
            }
        )
    return rows


def _compute_plot_limits(canonical: Any, *, padding_m: float) -> tuple[tuple[float, float], tuple[float, float]]:
    points: List[np.ndarray] = []
    for feature in canonical.map_features.values():
        polyline = _finite_xy_rows(feature.polyline_xy)
        if polyline.shape[0] >= 2:
            points.append(polyline)
    for track in canonical.tracks.values():
        segments = _valid_segments(track.position_xy, track.valid)
        points.extend(segments)
    for path in canonical.sdc_paths.values():
        segments = _valid_segments(path.polyline_xy, path.valid)
        points.extend(segments)

    if not points:
        return (-30.0, 30.0), (-30.0, 30.0)
    stacked = np.concatenate(points, axis=0)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    center = 0.5 * (min_xy + max_xy)
    half_extent = 0.5 * np.max(max_xy - min_xy) + float(padding_m)
    half_extent = max(25.0, float(half_extent))
    return (
        float(center[0] - half_extent),
        float(center[0] + half_extent),
    ), (
        float(center[1] - half_extent),
        float(center[1] + half_extent),
    )


def _draw_scene_context(ax, canonical: Any, *, xlim: Sequence[float], ylim: Sequence[float]) -> None:
    for feature in canonical.map_features.values():
        polyline = _finite_xy_rows(feature.polyline_xy)
        if polyline.shape[0] >= 2:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#d5d9e0", linewidth=0.8, alpha=0.8, zorder=1)
    for track_id, track in canonical.tracks.items():
        is_sdc = str(track_id) == str(canonical.sdc_id)
        _plot_segmented(
            ax,
            track.position_xy,
            valid=track.valid,
            color=("#111827" if is_sdc else "#9ca3af"),
            linewidth=(1.8 if is_sdc else 0.8),
            alpha=(0.9 if is_sdc else 0.35),
            zorder=(3 if is_sdc else 2),
        )
    sdc_track = canonical.tracks[str(canonical.sdc_id)]
    decision_idx = int(np.clip(int(canonical.current_time_index), 0, max(0, len(sdc_track.heading) - 1)))
    current_xy = _finite_xy_rows(np.asarray(sdc_track.position_xy[decision_idx], dtype=np.float32))
    if current_xy.shape[0] > 0:
        ax.scatter([current_xy[0, 0]], [current_xy[0, 1]], c="#7b3294", s=45, edgecolors="white", linewidths=0.8, zorder=5)
        heading = float(sdc_track.heading[decision_idx]) if np.isfinite(sdc_track.heading[decision_idx]) else 0.0
        arrow_dx = 6.5 * math.cos(heading)
        arrow_dy = 6.5 * math.sin(heading)
        ax.arrow(
            float(current_xy[0, 0]),
            float(current_xy[0, 1]),
            arrow_dx,
            arrow_dy,
            width=0.18,
            head_width=1.0,
            head_length=1.4,
            color="#111827",
            length_includes_head=True,
            zorder=6,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_scene_grid(raw_scenario: Mapping[str, Any], *, out_path: Path, padding_m: float, columns: int) -> Dict[str, Any]:
    canonical = normalize_scenario(raw_scenario)
    displayable_paths = _collect_displayable_paths(canonical)
    xlim, ylim = _compute_plot_limits(canonical, padding_m=padding_m)

    if not displayable_paths:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
        _draw_scene_context(ax, canonical, xlim=xlim, ylim=ylim)
        ax.set_title(f"{canonical.scenario_id}\nNo displayable SDC paths")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return {
            "scenario_id": str(canonical.scenario_id),
            "sdc_id": str(canonical.sdc_id),
            "num_sdc_paths_raw": int(len(canonical.sdc_paths)),
            "num_displayable_paths": 0,
            "grid_png": str(out_path),
            "paths": [],
        }

    ncols = max(1, int(columns))
    nrows = int(math.ceil(float(len(displayable_paths)) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 4.0 * nrows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(-1)

    path_rows: List[Dict[str, Any]] = []
    for ax, row in zip(axes_arr, displayable_paths):
        _draw_scene_context(ax, canonical, xlim=xlim, ylim=ylim)
        color = "#16a34a" if bool(row["on_route"]) else "#ea580c"
        linestyle = "-" if bool(row["on_route"]) else "--"
        for seg_idx, segment in enumerate(row["segments"]):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                color=color,
                linewidth=2.4,
                alpha=0.98,
                linestyle=linestyle,
                zorder=7,
                label="selected path" if seg_idx == 0 else None,
            )
        end_segment = row["segments"][-1]
        ax.scatter([end_segment[-1, 0]], [end_segment[-1, 1]], c=color, s=26, zorder=8)
        ax.set_title(
            f"[{row['order_idx']:02d}] {row['path_id']}\n"
            f"{'on_route' if row['on_route'] else 'off_route'} | "
            f"{row['path_length_m']:.1f} m",
            fontsize=9,
        )
        path_rows.append(
            {
                "order_idx": int(row["order_idx"]),
                "path_id": str(row["path_id"]),
                "on_route": bool(row["on_route"]),
                "num_points": int(row["num_points"]),
                "path_length_m": float(row["path_length_m"]),
            }
        )

    for ax in axes_arr[len(displayable_paths):]:
        ax.axis("off")

    fig.suptitle(
        f"{canonical.scenario_id} | sdc={canonical.sdc_id} | "
        f"displayable paths={len(displayable_paths)} / raw={len(canonical.sdc_paths)}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "scenario_id": str(canonical.scenario_id),
        "sdc_id": str(canonical.sdc_id),
        "num_sdc_paths_raw": int(len(canonical.sdc_paths)),
        "num_displayable_paths": int(len(displayable_paths)),
        "grid_png": str(out_path),
        "paths": path_rows,
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

    rows: List[Dict[str, Any]] = []
    data_iter = iter_waymax_simulator_states(config)
    selected_iter = itertools.islice(data_iter, int(args.scene_offset), int(args.scene_offset) + int(args.num_scenes))
    for local_idx, state in enumerate(selected_iter):
        global_idx = int(args.scene_offset) + local_idx
        raw = raw_scenario_from_waymax_state(
            state,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or f"waymax_scene_{global_idx:04d}")
        grid_path = outdir / f"{global_idx:03d}_{scenario_id}_sdc_path_grid.png"
        row = _plot_scene_grid(
            raw,
            out_path=grid_path,
            padding_m=float(args.padding_m),
            columns=int(args.columns),
        )
        if bool(args.save_pkls):
            pkl_path = outdir / f"sd_waymo_v1.3.1_{scenario_id}.pkl"
            save_raw_waymax_scenario_pickle(
                state,
                out_path=pkl_path,
                current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
            )
            row["output_pkl"] = str(pkl_path)
        rows.append(row)
        print(
            json.dumps(
                {
                    "scene_index": global_idx,
                    "scenario_id": row["scenario_id"],
                    "num_sdc_paths_raw": row["num_sdc_paths_raw"],
                    "num_displayable_paths": row["num_displayable_paths"],
                    "grid_png": row["grid_png"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "config_name": str(args.config_name),
        "path": str(args.path),
        "num_scenes_requested": int(args.num_scenes),
        "scene_offset": int(args.scene_offset),
        "num_scenes_written": int(len(rows)),
        "num_paths": int(args.num_paths),
        "num_points_per_path": int(args.num_points_per_path),
        "columns": int(args.columns),
        "rows": rows,
    }
    summary_path = outdir / "waymax_sdc_path_grids_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), "num_scenes_written": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
