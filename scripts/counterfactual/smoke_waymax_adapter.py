from __future__ import annotations

import argparse
import json
import math
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

from bmt.counterfactual import enumerate_branch_candidates_from_sdc_paths, normalize_scenario, raw_scenario_from_waymax_state, save_raw_waymax_scenario_pickle, waymax_available


class _DummyTrajectory:
    def __init__(self, positions_xy: np.ndarray, other_xy: np.ndarray) -> None:
        self.x = np.asarray([positions_xy[:, 0], other_xy[:, 0]], dtype=np.float32)
        self.y = np.asarray([positions_xy[:, 1], other_xy[:, 1]], dtype=np.float32)
        self.z = np.zeros_like(self.x)
        self.yaw = np.asarray(
            [
                _heading_series(positions_xy),
                _heading_series(other_xy),
            ],
            dtype=np.float32,
        )
        self.vel_x = np.gradient(self.x, axis=1).astype(np.float32)
        self.vel_y = np.gradient(self.y, axis=1).astype(np.float32)
        self.valid = np.ones_like(self.x, dtype=bool)


class _DummyMetadata:
    def __init__(self) -> None:
        self.ids = np.asarray([101, 202], dtype=np.int64)
        self.object_types = np.asarray([1, 1], dtype=np.int64)
        self.is_sdc = np.asarray([True, False], dtype=bool)


class _DummyLights:
    def __init__(self, stop_xy: tuple[float, float], timesteps: int) -> None:
        self.ids = np.asarray([900], dtype=np.int64)
        self.lane_ids = np.asarray([55], dtype=np.int64)
        self.state = np.asarray([[3] * timesteps], dtype=np.int64)
        self.x = np.asarray([[float(stop_xy[0])] * timesteps], dtype=np.float32)
        self.y = np.asarray([[float(stop_xy[1])] * timesteps], dtype=np.float32)
        self.z = np.zeros_like(self.x)
        self.valid = np.asarray([[True] * timesteps], dtype=bool)


class _DummyRoadgraph:
    def __init__(self, feature_ids: np.ndarray, coords_xy: np.ndarray) -> None:
        self.ids = np.asarray(feature_ids, dtype=np.int64)
        self.types = np.ones_like(self.ids, dtype=np.int64)
        self.x = np.asarray(coords_xy[:, 0], dtype=np.float32)
        self.y = np.asarray(coords_xy[:, 1], dtype=np.float32)
        self.z = np.zeros((coords_xy.shape[0],), dtype=np.float32)
        self.valid = np.ones((coords_xy.shape[0],), dtype=bool)


class _DummyPaths:
    def __init__(self, paths_xy: list[np.ndarray], on_route_index: int = 0) -> None:
        max_len = max(path.shape[0] for path in paths_xy)
        x = np.zeros((len(paths_xy), max_len), dtype=np.float32)
        y = np.zeros((len(paths_xy), max_len), dtype=np.float32)
        z = np.zeros((len(paths_xy), max_len), dtype=np.float32)
        valid = np.zeros((len(paths_xy), max_len), dtype=bool)
        for idx, path in enumerate(paths_xy):
            n = path.shape[0]
            x[idx, :n] = path[:, 0]
            y[idx, :n] = path[:, 1]
            valid[idx, :n] = True
            if n < max_len:
                x[idx, n:] = path[-1, 0]
                y[idx, n:] = path[-1, 1]
        self.ids = np.arange(len(paths_xy), dtype=np.int64)
        self.x = x
        self.y = y
        self.z = z
        self.valid = valid
        on_route = np.zeros((len(paths_xy),), dtype=bool)
        if 0 <= on_route_index < len(paths_xy):
            on_route[on_route_index] = True
        self.on_route = on_route


class _DummyState:
    def __init__(
        self,
        *,
        scene_id: str,
        positions_xy: np.ndarray,
        other_xy: np.ndarray,
        stop_xy: tuple[float, float],
        road_feature_ids: np.ndarray,
        road_coords_xy: np.ndarray,
        paths_xy: list[np.ndarray],
        current_time_index: int,
    ) -> None:
        self.id = str(scene_id)
        self.current_time_index = int(current_time_index)
        self.log_trajectory = _DummyTrajectory(positions_xy=positions_xy, other_xy=other_xy)
        self.object_metadata = _DummyMetadata()
        self.log_traffic_light = _DummyLights(stop_xy=stop_xy, timesteps=positions_xy.shape[0])
        self.roadgraph_points = _DummyRoadgraph(feature_ids=road_feature_ids, coords_xy=road_coords_xy)
        self.sdc_paths = _DummyPaths(paths_xy=paths_xy, on_route_index=0)


def _heading_series(points_xy: np.ndarray) -> np.ndarray:
    headings = np.zeros((points_xy.shape[0],), dtype=np.float32)
    if points_xy.shape[0] < 2:
        return headings
    deltas = np.diff(points_xy, axis=0)
    segment_headings = np.arctan2(deltas[:, 1], deltas[:, 0]).astype(np.float32)
    headings[:-1] = segment_headings
    headings[-1] = segment_headings[-1]
    return headings


def _scene_one() -> _DummyState:
    sdc = np.asarray(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [6.0, 0.0],
            [9.0, 0.0],
            [12.0, 0.5],
            [15.0, 1.0],
        ],
        dtype=np.float32,
    )
    other = np.asarray(
        [
            [6.0, -3.0],
            [6.5, -2.0],
            [7.0, -1.0],
            [7.5, 0.0],
            [8.0, 1.0],
            [8.5, 2.0],
        ],
        dtype=np.float32,
    )
    road = np.asarray(
        [
            [0.0, 0.0],
            [5.0, 0.0],
            [10.0, 0.0],
            [15.0, 0.0],
            [10.0, 5.0],
            [10.0, 10.0],
            [10.0, -5.0],
            [10.0, -10.0],
        ],
        dtype=np.float32,
    )
    feature_ids = np.asarray([10, 10, 10, 10, 20, 20, 30, 30], dtype=np.int64)
    paths = [
        np.asarray([[6.0, 0.0], [10.0, 0.0], [14.0, 0.0], [18.0, 0.0]], dtype=np.float32),
        np.asarray([[6.0, 0.0], [10.0, 0.0], [12.0, 2.0], [12.0, 6.0]], dtype=np.float32),
        np.asarray([[6.0, 0.0], [10.0, 0.0], [12.0, -2.0], [12.0, -6.0]], dtype=np.float32),
    ]
    return _DummyState(
        scene_id="synthetic_waymax_scene_a",
        positions_xy=sdc,
        other_xy=other,
        stop_xy=(9.0, 0.0),
        road_feature_ids=feature_ids,
        road_coords_xy=road,
        paths_xy=paths,
        current_time_index=2,
    )


def _scene_two() -> _DummyState:
    sdc = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.5],
            [4.0, 1.0],
            [6.0, 1.5],
            [8.0, 2.0],
            [10.0, 2.5],
        ],
        dtype=np.float32,
    )
    other = np.asarray(
        [
            [4.0, -4.0],
            [4.2, -3.0],
            [4.4, -2.0],
            [4.6, -1.0],
            [4.8, 0.0],
            [5.0, 1.0],
        ],
        dtype=np.float32,
    )
    road = np.asarray(
        [
            [0.0, 0.0],
            [3.0, 0.8],
            [6.0, 1.6],
            [9.0, 2.4],
            [12.0, 3.2],
            [12.0, 7.0],
            [12.0, -1.0],
        ],
        dtype=np.float32,
    )
    feature_ids = np.asarray([11, 11, 11, 11, 11, 21, 31], dtype=np.int64)
    paths = [
        np.asarray([[4.0, 1.0], [8.0, 2.0], [12.0, 3.0], [16.0, 4.0]], dtype=np.float32),
        np.asarray([[4.0, 1.0], [8.0, 2.0], [11.0, 4.0], [12.0, 8.0]], dtype=np.float32),
    ]
    return _DummyState(
        scene_id="synthetic_waymax_scene_b",
        positions_xy=sdc,
        other_xy=other,
        stop_xy=(7.0, 1.8),
        road_feature_ids=feature_ids,
        road_coords_xy=road,
        paths_xy=paths,
        current_time_index=2,
    )


def _plot_sdc_paths(raw_scenario: dict, *, out_path: Path) -> dict:
    canonical = normalize_scenario(raw_scenario)
    sdc_track = canonical.tracks[str(canonical.sdc_id)]
    decision_idx = int(canonical.current_time_index)
    approach_heading = float(sdc_track.heading[decision_idx]) if np.isfinite(sdc_track.heading[decision_idx]) else 0.0
    branches = enumerate_branch_candidates_from_sdc_paths(
        canonical,
        agent_id=str(canonical.sdc_id),
        decision_time_idx=decision_idx,
        approach_heading=approach_heading,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    for feature in canonical.map_features.values():
        polyline = np.asarray(feature.polyline_xy, dtype=np.float32)
        if polyline.shape[0] >= 2:
            ax.plot(polyline[:, 0], polyline[:, 1], color="#d0d0d0", linewidth=1.2, alpha=0.8, zorder=1)
    for light in canonical.traffic_lights.values():
        if light.stop_point_xy is not None:
            ax.scatter([light.stop_point_xy[0]], [light.stop_point_xy[1]], c="#ffcc00", marker="s", s=64, edgecolors="black", linewidths=0.8, zorder=4)

    for track_id, track in canonical.tracks.items():
        polyline = np.asarray(track.position_xy, dtype=np.float32)
        valid = np.asarray(track.valid, dtype=bool)
        polyline = polyline[valid]
        if polyline.shape[0] < 2:
            continue
        color = "#111111" if str(track_id) == str(canonical.sdc_id) else "#7f8c8d"
        alpha = 1.0 if str(track_id) == str(canonical.sdc_id) else 0.5
        linewidth = 2.4 if str(track_id) == str(canonical.sdc_id) else 1.2
        ax.plot(polyline[:, 0], polyline[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=3)

    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    branch_rows = []
    for idx, branch in enumerate(branches):
        polyline = np.asarray(branch.polyline_xy, dtype=np.float32)
        color = palette[idx % len(palette)]
        ax.plot(polyline[:, 0], polyline[:, 1], color=color, linewidth=3.0, alpha=0.95, zorder=5, label=f"{branch.branch_id} ({branch.branch_label})")
        ax.scatter([polyline[-1, 0]], [polyline[-1, 1]], c=color, s=55, zorder=6)
        branch_rows.append(
            {
                "branch_id": str(branch.branch_id),
                "branch_label": str(branch.branch_label),
                "source_kind": str(branch.source_kind),
                "rank_score": float(branch.rank_score),
            }
        )

    current_xy = np.asarray(sdc_track.position_xy[decision_idx], dtype=np.float32)
    ax.scatter([current_xy[0]], [current_xy[1]], c="#6a0dad", s=80, marker="o", edgecolors="white", linewidths=1.0, zorder=7, label="sdc decision")
    ax.set_title(f"{canonical.scenario_id} | SDC paths")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "scenario_id": str(canonical.scenario_id),
        "sdc_id": str(canonical.sdc_id),
        "decision_time_idx": decision_idx,
        "num_sdc_paths": int(len(canonical.sdc_paths)),
        "num_branches": int(len(branches)),
        "branches": branch_rows,
        "overlay_png": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local smoke test for the Waymax adapter and render SDC path overlays.")
    parser.add_argument("--outdir", type=str, default="outputs/waymax_adapter_smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    if not waymax_available():
        raise SystemExit("waymax is not installed in the current environment")

    scenes = [_scene_one(), _scene_two()]
    scene_summaries = []
    for scene in scenes:
        raw = raw_scenario_from_waymax_state(scene)
        pkl_path = outdir / f"sd_waymo_v1.3.1_{raw['id']}.pkl"
        save_raw_waymax_scenario_pickle(scene, out_path=pkl_path)
        overlay_png = outdir / f"{raw['id']}_sdc_paths_overlay.png"
        overlay_summary = _plot_sdc_paths(raw, out_path=overlay_png)
        overlay_summary["output_pkl"] = str(pkl_path)
        scene_summaries.append(overlay_summary)

    summary = {
        "waymax_available": True,
        "num_scenes": int(len(scene_summaries)),
        "scenes": scene_summaries,
    }
    (outdir / "waymax_adapter_smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
