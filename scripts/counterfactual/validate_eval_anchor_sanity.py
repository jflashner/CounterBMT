from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

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
    FRAME_WORLD,
    agent_relative_error_to_anchor,
    agent_relative_pose_to_world,
    anchor_pose_from_control_code,
    branch_candidates_world,
    classify_branch_from_world_pose,
    gt_final_world_pose_from_raw,
    load_json,
    load_raw_scenario,
    mean_or_none,
    percentile_dict,
    pose_to_dict,
    raw_track_world_state,
    world_pose_to_agent_relative,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GT/anchor consistency on a selected local path-control eval bundle.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def _world_xy_to_agent_relative(xy_world: np.ndarray, *, agent_pose_world: Mapping[str, Any]) -> np.ndarray:
    xy = np.asarray(xy_world, dtype=np.float64)
    dx = xy[..., 0] - float(agent_pose_world.get("x", 0.0))
    dy = xy[..., 1] - float(agent_pose_world.get("y", 0.0))
    heading = float(agent_pose_world.get("heading", 0.0))
    c = math.cos(heading)
    s = math.sin(heading)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    return np.stack([x_rel, y_rel], axis=-1)


def _plot_gt_anchor_views(
    *,
    output_local: Path,
    output_world: Path,
    example_id: str,
    requested_branch_label: str,
    gt_xy_world: np.ndarray,
    gt_valid: np.ndarray,
    branch_candidates: Sequence[Mapping[str, Any]],
    anchor_world: Mapping[str, Any],
    anchor_rel_pose: Mapping[str, Any],
    agent_pose_world: Mapping[str, Any],
    distance_m: float,
    heading_error_rad: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    valid_xy_world = np.asarray(gt_xy_world, dtype=np.float64)[np.asarray(gt_valid, dtype=bool)]
    if valid_xy_world.size == 0:
        return
    valid_xy_rel = _world_xy_to_agent_relative(valid_xy_world, agent_pose_world=agent_pose_world)

    def _set_bounds(ax, points: np.ndarray, *, padding: float = 10.0):
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

    fig_local, ax_local = plt.subplots(figsize=(6.0, 6.0))
    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            polyline_rel = _world_xy_to_agent_relative(polyline[:, :2], agent_pose_world=agent_pose_world)
            ax_local.plot(polyline_rel[:, 0], polyline_rel[:, 1], color="#c0c0c0", linewidth=1.0, alpha=0.8)
    ax_local.plot(valid_xy_rel[:, 0], valid_xy_rel[:, 1], color="#222222", linewidth=2.0, label="GT future")
    ax_local.scatter([0.0], [0.0], color="#7b3294", s=40, label="decision pose")
    ax_local.scatter(
        [float(anchor_rel_pose["x"])],
        [float(anchor_rel_pose["y"])],
        color="#d55e00",
        s=45,
        label=f"requested anchor ({requested_branch_label})",
    )
    ax_local.scatter(valid_xy_rel[-1, 0], valid_xy_rel[-1, 1], color="#009e73", s=45, label="GT final")
    ax_local.set_title(
        f"{example_id}\nlocal_target_frame | dist={distance_m:.1f}m | dh={heading_error_rad:.2f}rad"
    )
    ax_local.set_xlabel("x_rel (m)")
    ax_local.set_ylabel("y_rel (m)")
    ax_local.legend(loc="best", fontsize=8)
    ax_local.set_aspect("equal", adjustable="box")
    _set_bounds(
        ax_local,
        np.concatenate(
            [
                valid_xy_rel,
                np.asarray([[float(anchor_rel_pose['x']), float(anchor_rel_pose['y'])]], dtype=np.float64),
            ],
            axis=0,
        ),
    )
    output_local.parent.mkdir(parents=True, exist_ok=True)
    fig_local.tight_layout()
    fig_local.savefig(output_local, dpi=160)
    plt.close(fig_local)

    fig_world, ax_world = plt.subplots(figsize=(6.0, 6.0))
    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float64)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            ax_world.plot(polyline[:, 0], polyline[:, 1], color="#c0c0c0", linewidth=1.0, alpha=0.8)
    ax_world.plot(valid_xy_world[:, 0], valid_xy_world[:, 1], color="#222222", linewidth=2.0, label="GT future")
    ax_world.scatter(
        [float(agent_pose_world["x"])],
        [float(agent_pose_world["y"])],
        color="#7b3294",
        s=40,
        label="decision pose",
    )
    ax_world.scatter(
        [float(anchor_world["x"])],
        [float(anchor_world["y"])],
        color="#d55e00",
        s=45,
        label=f"requested anchor ({requested_branch_label})",
    )
    ax_world.scatter(valid_xy_world[-1, 0], valid_xy_world[-1, 1], color="#009e73", s=45, label="GT final")
    ax_world.set_title(
        f"{example_id}\nworld_frame | dist={distance_m:.1f}m | dh={heading_error_rad:.2f}rad"
    )
    ax_world.set_xlabel("x_world (m)")
    ax_world.set_ylabel("y_world (m)")
    ax_world.legend(loc="best", fontsize=8)
    ax_world.set_aspect("equal", adjustable="box")
    _set_bounds(
        ax_world,
        np.concatenate(
            [
                valid_xy_world,
                np.asarray([[float(anchor_world['x']), float(anchor_world['y'])]], dtype=np.float64),
                np.asarray([[float(agent_pose_world['x']), float(agent_pose_world['y'])]], dtype=np.float64),
            ],
            axis=0,
        ),
        padding=15.0,
    )
    output_world.parent.mkdir(parents=True, exist_ok=True)
    fig_world.tight_layout()
    fig_world.savefig(output_world, dpi=160)
    plt.close(fig_world)


def run_anchor_sanity(manifest_path: str | Path, outdir: str | Path) -> Dict[str, Any]:
    manifest = load_json(manifest_path)
    selected = [item for item in manifest if str(item.get("factual_or_alternative")) == "factual"]
    output_root = Path(outdir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root = output_root / "visuals" / "gt_anchor_sanity"

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for item in selected:
        local_scenario_pkl = item.get("local_scenario_pkl")
        materialized_dir = item.get("local_materialized_eval_input")
        if not local_scenario_pkl or not materialized_dir:
            continue
        raw_scenario = load_raw_scenario(local_scenario_pkl)
        factual_control = load_json(Path(materialized_dir) / "factual_control_code.json")
        branch_candidates = branch_candidates_world(materialized_dir)

        requested_anchor_rel = anchor_pose_from_control_code(factual_control)
        if requested_anchor_rel is None:
            continue
        agent_pose_world = dict(factual_control.get("debug", {}).get("agent_pose_at_decision", {}))
        if not agent_pose_world:
            continue
        gt_final_world = gt_final_world_pose_from_raw(raw_scenario, track_id=str(item["agent_id"]))
        gt_distance_m, gt_heading_error_rad, gt_final_rel = agent_relative_error_to_anchor(
            pose_world=gt_final_world,
            anchor_rel=requested_anchor_rel,
            agent_pose_world=agent_pose_world,
        )
        requested_anchor_world = agent_relative_pose_to_world(requested_anchor_rel, agent_pose_world=agent_pose_world)
        gt_branch = classify_branch_from_world_pose(gt_final_world, branch_candidates)
        requested_branch_label = str(factual_control.get("path_token", {}).get("branch_label"))
        gt_state = raw_track_world_state(raw_scenario, track_id=str(item["agent_id"]))
        gt_valid = np.asarray(gt_state["valid"], dtype=bool)
        gt_xy_world = np.asarray(gt_state["position"], dtype=np.float64)[:, :2]

        row = {
            "example_id": str(item["example_id"]),
            "scenario_id": str(item["scenario_id"]),
            "agent_id": str(item["agent_id"]),
            "decision_time_idx": int(item["decision_time_idx"]),
            "requested_branch_label": requested_branch_label,
            "gt_branch_label": gt_branch.get("branch_label"),
            "gt_branch_matches_requested": bool(gt_branch.get("branch_label") == requested_branch_label),
            "gt_branch_score_margin": gt_branch.get("score_margin"),
            "gt_final_pose_world": pose_to_dict(gt_final_world),
            "gt_final_pose_rel_to_decision": pose_to_dict(gt_final_rel),
            "requested_anchor_rel": pose_to_dict(requested_anchor_rel),
            "requested_anchor_world": pose_to_dict(requested_anchor_world),
            "gt_final_pose_to_requested_anchor_m": float(gt_distance_m),
            "gt_final_heading_error_to_requested_anchor_rad": float(gt_heading_error_rad),
            "gt_final_pose_frame": FRAME_WORLD,
            "requested_anchor_frame": FRAME_AGENT_RELATIVE_AT_DECISION,
            "anchor_comparison_frame": FRAME_AGENT_RELATIVE_AT_DECISION,
            "branch_scorer_frame": FRAME_WORLD,
            "local_target_frame_png": str((visual_root / f"{item['example_id']}_local_target_frame.png").resolve()),
            "world_frame_png": str((visual_root / f"{item['example_id']}_world_frame.png").resolve()),
        }
        rows.append(row)

        _plot_gt_anchor_views(
            output_local=visual_root / f"{item['example_id']}_local_target_frame.png",
            output_world=visual_root / f"{item['example_id']}_world_frame.png",
            example_id=str(item["example_id"]),
            requested_branch_label=requested_branch_label,
            gt_xy_world=gt_xy_world,
            gt_valid=gt_valid,
            branch_candidates=branch_candidates,
            anchor_world=pose_to_dict(requested_anchor_world),
            anchor_rel_pose=pose_to_dict(requested_anchor_rel),
            agent_pose_world=agent_pose_world,
            distance_m=float(gt_distance_m),
            heading_error_rad=float(gt_heading_error_rad),
        )

        if (gt_distance_m > 20.0) or (gt_branch.get("branch_label") != requested_branch_label):
            failures.append(row)

    distances = [float(row["gt_final_pose_to_requested_anchor_m"]) for row in rows]
    heading_errors = [float(row["gt_final_heading_error_to_requested_anchor_rad"]) for row in rows]
    summary = {
        "num_rows": int(len(rows)),
        "gt_branch_matches_requested_rate": float(
            sum(bool(row["gt_branch_matches_requested"]) for row in rows) / len(rows)
        ) if rows else 0.0,
        "mean_gt_final_pose_to_requested_anchor_m": mean_or_none(distances),
        "mean_gt_final_heading_error_to_requested_anchor_rad": mean_or_none(heading_errors),
        **percentile_dict(distances),
        "num_rows_over_20m": int(sum(distance > 20.0 for distance in distances)),
        "num_rows_over_50m": int(sum(distance > 50.0 for distance in distances)),
        "num_rows_over_100m": int(sum(distance > 100.0 for distance in distances)),
    }

    write_json(output_root / "gt_anchor_sanity_summary.json", summary)
    write_jsonl(output_root / "gt_anchor_sanity_per_example.jsonl", rows)
    write_jsonl(output_root / "gt_anchor_sanity_failures.jsonl", failures)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_anchor_sanity(args.manifest, args.outdir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
