from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
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

from bmt.dataset.dataset import InfgenDataset
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.models.motionlm_lightning import sanitize_logits_for_loss
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _adapt_rollout_output_for_semantic_eval,
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _gather_sdc_action_tokens,
    _load_config,
    _model_to_world,
    _optional_positive_float,
    _prepare_batch_for_autoregressive_rollout,
    _read_jsonl,
    _resolve_device,
    _select_row,
    _to_torch_device,
)
from scripts.counterfactual.sdc_semantic_tube_utils import (
    group_normalized_advantages,
    return_to_go,
    segment_distance_field_in_sdc_frame,
    selected_raw_route_world,
    tube_reward_from_distance,
    polyline_segment_distance_to_points,
    world_to_sdc_up_frame,
)
from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from bmt.counterfactual.sdc_path_control import polyline_length_m, split_polyline_on_discontinuities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample grouped autoregressive rollouts and inspect tube-based returns, RTG, and advantages."
    )
    parser.add_argument("--config", type=str, default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--scenario-id", type=str, required=True)
    parser.add_argument("--slot-id", type=str, required=True)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--tube-radius-m", type=float, default=3.0)
    parser.add_argument("--inside-reward", type=float, default=1.0)
    parser.add_argument("--outside-scale", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--sampling-method", type=str, default="softmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--grid-step-m", type=float, default=0.35)
    parser.add_argument("--jump-threshold-m", type=float, default=6.0)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def _tube_overlay(
    *,
    ax,
    fig,
    render_context: Mapping[str, Any],
    path_world: np.ndarray,
    path_segments_world: List[np.ndarray],
    tube_radius_m: float,
    grid_step_m: float,
    trajectory_local_list: List[np.ndarray],
    rollout_labels: List[str],
    rollout_colors: List[Any],
    info_box_text: str,
) -> None:
    current_xy = np.asarray(render_context["current_xy"], dtype=np.float32)
    current_heading = float(render_context["current_heading"])
    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=render_context,
        highlighted_segments_world=[],
        highlighted_gradient_values=None,
        representative_route_world=path_world,
        info_box_text=info_box_text,
        show_colorbar=False,
    )
    xx, yy, dist_field = segment_distance_field_in_sdc_frame(
        polyline_world_xy=path_world,
        center_xy_world=current_xy,
        heading_world_rad=current_heading,
        grid_step_m=float(grid_step_m),
    )
    inside = np.ma.masked_where(dist_field > float(tube_radius_m), dist_field)
    ax.contourf(
        xx,
        yy,
        inside,
        levels=np.linspace(0.0, float(tube_radius_m), num=8),
        cmap="Blues_r",
        alpha=0.24,
        zorder=6.1,
        antialiased=True,
    )
    ax.contour(
        xx,
        yy,
        dist_field,
        levels=[float(tube_radius_m)],
        colors=["#f59e0b"],
        linewidths=1.8,
        linestyles=["--"],
        zorder=11.5,
    )
    for seg_idx, seg_world in enumerate(path_segments_world):
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
            linewidth=4.2,
            alpha=0.98,
            zorder=10.0,
            solid_capstyle="round",
        )
        if seg_idx > 0:
            ax.scatter(
                [seg_local[0, 0]],
                [seg_local[0, 1]],
                c="#111827",
                s=26,
                marker="x",
                linewidths=1.0,
                zorder=10.8,
            )
    for traj_local, label, color in zip(trajectory_local_list, rollout_labels, rollout_colors):
        if traj_local.shape[0] < 2:
            continue
        ax.plot(
            traj_local[:, 0],
            traj_local[:, 1],
            color=color,
            linewidth=2.2,
            alpha=0.92,
            zorder=12.6,
        )
        ax.scatter(
            [traj_local[-1, 0]],
            [traj_local[-1, 1]],
            c=[color],
            s=28,
            edgecolors="white",
            linewidths=0.6,
            zorder=13.0,
        )
        ax.text(
            float(traj_local[-1, 0]) + 0.6,
            float(traj_local[-1, 1]) + 0.6,
            str(label),
            fontsize=8,
            color="#111827",
            zorder=13.2,
        )


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.control_index).expanduser().resolve())
    row_index = _select_row(
        rows,
        row_index=-1,
        scenario_id=str(args.scenario_id).strip(),
        slot_id=str(args.slot_id).strip(),
    )
    row = dict(rows[row_index])
    raw_scenario = load_raw_scenario_from_row(row)
    render_context = _extract_scene_render_context(raw_scenario, row)
    path_world = np.asarray(selected_raw_route_world(raw_scenario, row), dtype=np.float32)
    path_segments_world = [
        np.asarray(seg, dtype=np.float32)
        for seg in split_polyline_on_discontinuities(path_world, jump_threshold_m=float(args.jump_threshold_m))
        if np.asarray(seg).shape[0] >= 2
    ]

    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode="training")
    device = _resolve_device(args.device)
    model, load_report = load_model_from_checkpoint_forgiving(
        config=config,
        ckpt_path=str(args.ckpt),
        load_mode=str(args.load_mode),
        strict_state_dict=(str(args.load_mode) == "strict_state_dict"),
        map_location=str(device),
    )
    model = model.to(device)
    model.eval()
    model._trainer = type("TrainerStub", (), {"world_size": 1, "lr_scheduler_configs": None, "optimizers": None})()

    sample = dataset[row_index]
    batch = dataset.collate_batch([sample])
    batch_torch = _to_torch_device(batch, device=device)
    rollout_base = _prepare_batch_for_autoregressive_rollout(batch_torch, raw_scenario=raw_scenario)

    path_model = np.asarray(
        batch["cf/sdc_selected_raw_path_model"][0]
        if "cf/sdc_selected_raw_path_model" in batch
        else batch["cf/sdc_selected_raw_path_world"][0],
        dtype=np.float32,
    )
    path_model_mask = np.asarray(batch["cf/sdc_selected_raw_path_mask"][0], dtype=np.float32).reshape(-1) > 0.5
    if path_model_mask.shape[0] > 0:
        path_model = path_model[: path_model_mask.shape[0]][path_model_mask[: path_model.shape[0]]]

    map_center_world = np.asarray(row.get("candidate_family_map_center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    map_heading_world = float(row.get("candidate_family_map_heading", 0.0) or 0.0)
    rollout_records: List[Dict[str, Any]] = []

    for rollout_idx in range(int(args.num_rollouts)):
        current_seed = int(args.seed) + int(rollout_idx)
        _set_seed(current_seed)
        with torch.no_grad():
            rollout_output = model.model.autoregressive_rollout(
                copy.deepcopy(rollout_base),
                num_decode_steps=None,
                sampling_method=str(args.sampling_method),
                temperature=_optional_positive_float(float(args.temperature)),
                topp=_optional_positive_float(float(args.topp)),
                autoregressive_start_step=0,
            )
            output = _adapt_rollout_output_for_semantic_eval(base_batch=batch_torch, rollout_output=rollout_output)
            semantic_context = model._extract_sdc_semantic_context(output)
            if semantic_context is None:
                raise RuntimeError("Failed to build semantic context for rollout reward analysis.")

            decision_agent_mask = semantic_context["decision_agent_mask"]
            sdc_agent_idx = int(np.argmax(np.asarray(decision_agent_mask[0].detach().cpu(), dtype=np.float32)))
            rollout_next_model_xy = np.asarray(
                output["decoder/rollout_next_position"][0, :, sdc_agent_idx, :].detach().cpu(),
                dtype=np.float32,
            )
            rollout_next_world_xy = _model_to_world(
                rollout_next_model_xy,
                map_center_world=map_center_world,
                map_heading_world=map_heading_world,
            )
            trajectory_world_xy = np.concatenate(
                [
                    np.asarray(render_context["current_xy"], dtype=np.float32).reshape(1, 2),
                    rollout_next_world_xy,
                ],
                axis=0,
            ).astype(np.float32)
            student_logits_sdc = sanitize_logits_for_loss(
                (output["decoder/output_logit"] * decision_agent_mask[:, None, :, None]).sum(dim=2)
            )
            selected_tokens = _gather_sdc_action_tokens(output["decoder/output_action"], decision_agent_mask)
            selected_log_probs = F.log_softmax(student_logits_sdc, dim=-1).gather(
                dim=-1,
                index=selected_tokens.unsqueeze(-1),
            ).squeeze(-1)

        distance_m = polyline_segment_distance_to_points(rollout_next_world_xy, path_world)
        distance_model_m = polyline_segment_distance_to_points(rollout_next_model_xy, path_model)
        reward_t = tube_reward_from_distance(
            distance_model_m,
            tube_radius_m=float(args.tube_radius_m),
            inside_reward=float(args.inside_reward),
            outside_scale=float(args.outside_scale),
        )
        rtg_t = return_to_go(reward_t, gamma=float(args.discount))
        inside_mask = distance_model_m <= float(args.tube_radius_m)
        first_exit = np.flatnonzero(~inside_mask)
        rollout_records.append(
            {
                "rollout_id": int(rollout_idx),
                "seed": int(current_seed),
                "trajectory_world_xy": trajectory_world_xy.tolist(),
                "trajectory_model_xy": rollout_next_model_xy.astype(np.float32).tolist(),
                "distance_to_tube_m": distance_model_m.astype(np.float32).tolist(),
                "inside_valid_tube": inside_mask.astype(bool).tolist(),
                "reward_t": reward_t.astype(np.float32).tolist(),
                "return_to_go_t": rtg_t.astype(np.float32).tolist(),
                "action_token_t": np.asarray(selected_tokens[0].detach().cpu(), dtype=np.int64).tolist(),
                "action_logprob_t": np.asarray(selected_log_probs[0].detach().cpu(), dtype=np.float32).tolist(),
                "inside_fraction": float(np.mean(inside_mask.astype(np.float32))),
                "first_exit_step": (None if first_exit.size == 0 else int(first_exit[0])),
                "total_return": float(rtg_t[0] if rtg_t.size > 0 else 0.0),
            }
        )

    reward_matrix = np.asarray([record["reward_t"] for record in rollout_records], dtype=np.float32)
    rtg_matrix = np.asarray([record["return_to_go_t"] for record in rollout_records], dtype=np.float32)
    total_return = np.asarray([record["total_return"] for record in rollout_records], dtype=np.float32)
    scalar_advantage = group_normalized_advantages(total_return, axis=0)
    step_advantage = group_normalized_advantages(rtg_matrix, axis=0)

    for rollout_idx, record in enumerate(rollout_records):
        record["scalar_group_advantage"] = float(scalar_advantage[rollout_idx])
        record["step_group_advantage_t"] = np.asarray(step_advantage[rollout_idx], dtype=np.float32).tolist()

    color_norm = plt.Normalize(vmin=float(np.min(scalar_advantage)), vmax=float(np.max(scalar_advantage)))
    cmap = plt.cm.RdYlGn
    rollout_colors = [cmap(color_norm(float(value))) for value in scalar_advantage.tolist()]
    trajectory_local_list = [
        world_to_sdc_up_frame(
            np.asarray(record["trajectory_world_xy"], dtype=np.float32),
            center_xy_world=np.asarray(render_context["current_xy"], dtype=np.float32),
            heading_world_rad=float(render_context["current_heading"]),
        )
        for record in rollout_records
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5), dpi=180)
    _tube_overlay(
        ax=axes[0, 0],
        fig=fig,
        render_context=render_context,
        path_world=path_world,
        path_segments_world=path_segments_world,
        tube_radius_m=float(args.tube_radius_m),
        grid_step_m=float(args.grid_step_m),
        trajectory_local_list=trajectory_local_list,
        rollout_labels=[str(record["rollout_id"]) for record in rollout_records],
        rollout_colors=rollout_colors,
        info_box_text=(
            f"scene={row['scenario_id']}\n"
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}\n"
            f"tube_radius={float(args.tube_radius_m):.1f}m\n"
            f"num_rollouts={int(args.num_rollouts)}"
        ),
    )
    axes[0, 0].set_title("Tube Region + Sampled Rollouts", fontsize=12)

    for record, color in zip(rollout_records, rollout_colors):
        step_idx = np.arange(len(record["reward_t"]), dtype=np.int64)
        axes[0, 1].plot(step_idx, record["reward_t"], color=color, linewidth=1.9, alpha=0.92)
    axes[0, 1].set_title("Per-Step Tube Reward", fontsize=12)
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("reward")
    axes[0, 1].grid(alpha=0.25)

    for record, color in zip(rollout_records, rollout_colors):
        step_idx = np.arange(len(record["return_to_go_t"]), dtype=np.int64)
        axes[1, 0].plot(step_idx, record["return_to_go_t"], color=color, linewidth=1.9, alpha=0.92)
    axes[1, 0].set_title("Return-to-Go", fontsize=12)
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("RTG")
    axes[1, 0].grid(alpha=0.25)

    for record, color in zip(rollout_records, rollout_colors):
        step_idx = np.arange(len(record["step_group_advantage_t"]), dtype=np.int64)
        axes[1, 1].plot(step_idx, record["step_group_advantage_t"], color=color, linewidth=1.9, alpha=0.92)
    axes[1, 1].axhline(0.0, color="#111827", linewidth=1.0, alpha=0.5)
    axes[1, 1].set_title("Stepwise Group-Normalized Advantage", fontsize=12)
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("advantage")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(
        "Tube-based grouped rollout analysis\n"
        f"inside_reward={float(args.inside_reward):.2f}  outside_scale={float(args.outside_scale):.2f}  gamma={float(args.discount):.2f}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.965])
    analysis_png = outdir / "group_rollout_advantage_analysis.png"
    fig.savefig(analysis_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scenario_id": str(row.get("scenario_id") or ""),
        "selected_slot_id": str(row.get("selected_slot_id") or ""),
        "requested_semantic_label": str(row.get("requested_semantic_label") or ""),
        "selected_path_id": row.get("selected_path_id"),
        "tube_radius_m": float(args.tube_radius_m),
        "inside_reward": float(args.inside_reward),
        "outside_scale": float(args.outside_scale),
        "discount": float(args.discount),
        "sampling_method": str(args.sampling_method),
        "temperature": _optional_positive_float(float(args.temperature)),
        "topp": _optional_positive_float(float(args.topp)),
        "num_rollouts": int(args.num_rollouts),
        "path_length_m": float(polyline_length_m(path_world)),
        "num_path_segments": int(len(path_segments_world)),
        "row_index": int(row_index),
        "checkpoint_load_report": load_report,
        "analysis_png": str(analysis_png),
        "rollouts": rollout_records,
    }
    summary_path = outdir / "group_rollout_advantage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis_png": str(analysis_png),
                "group_rollout_advantage_summary_json": str(summary_path),
                "top_rollouts": sorted(
                    [
                        {
                            "rollout_id": int(record["rollout_id"]),
                            "total_return": float(record["total_return"]),
                            "scalar_group_advantage": float(record["scalar_group_advantage"]),
                            "inside_fraction": float(record["inside_fraction"]),
                            "first_exit_step": record["first_exit_step"],
                        }
                        for record in rollout_records
                    ],
                    key=lambda item: float(item["total_return"]),
                    reverse=True,
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
