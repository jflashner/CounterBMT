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

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row, project_points_to_family_paths_torch, world_xy_to_model_frame
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from scripts.counterfactual.eval_sdc_semantic_action_projections import (
    _compute_family_action_energy_terms,
    _draw_vlm_style_scene_ax,
    _extract_scene_render_context,
    _load_config,
    _model_to_world,
    _read_jsonl,
    _resolve_device,
    _selected_path_world_from_row,
    _select_scene_row_indices,
    _to_torch_device,
)
from scripts.counterfactual.label_waymax_sdc_path_semantics import PLOT_RADIUS_M, SDC_VERTICAL_FRACTION, _world_to_sdc_up_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a best-token vector field over one scene for each semantic alternate."
    )
    parser.add_argument("--config", type=str, default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--scenario-id", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--grid-step-m", type=float, default=2.0)
    parser.add_argument("--crop-radius-m", type=float, default=PLOT_RADIUS_M)
    parser.add_argument("--step-index", type=int, default=-1, help="If negative, use the first valid SDC step.")
    return parser.parse_args()


def _sdc_up_to_world_frame(xy_local: np.ndarray, *, center_xy: np.ndarray, heading_rad: float) -> np.ndarray:
    xy = np.asarray(xy_local, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rot = float(heading_rad) - (math.pi / 2.0)
    c = math.cos(rot)
    s = math.sin(rot)
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(center_xy[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(center_xy[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def _weighted_angle(headings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        (weights * torch.sin(headings)).sum(dim=-1),
        (weights * torch.cos(headings)).sum(dim=-1),
    )


def _resolve_step_index(valid_by_t: np.ndarray, requested_step: int) -> int:
    valid = np.asarray(valid_by_t, dtype=bool).reshape(-1)
    if valid.size == 0 or not valid.any():
        raise ValueError("No valid SDC steps available for vector-field plotting.")
    if requested_step >= 0:
        if requested_step >= valid.size:
            raise IndexError(f"Requested step {requested_step} exceeds valid horizon {valid.size}.")
        if not bool(valid[requested_step]):
            raise ValueError(f"Requested step {requested_step} is not valid for the SDC.")
        return int(requested_step)
    return int(np.flatnonzero(valid)[0])


def _scene_local_grid(crop_radius_m: float, grid_step_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_extent = float(crop_radius_m)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    xs = np.arange(-half_extent, half_extent + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    ys = np.arange(y_min, y_max + 0.5 * float(grid_step_m), float(grid_step_m), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    local_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float32)
    return xx, yy, local_points


def _build_slot_vector_field(
    *,
    row: Mapping[str, Any],
    row_index: int,
    dataset: InfgenDataset,
    model,
    device: torch.device,
    crop_radius_m: float,
    grid_step_m: float,
    requested_step_index: int,
) -> Dict[str, Any]:
    raw_scenario = load_raw_scenario_from_row(row)
    sample = dataset[row_index]
    batch = dataset.collate_batch([sample])
    batch_torch = _to_torch_device(batch, device=device)

    with torch.no_grad():
        output = model(copy.deepcopy(batch_torch))
        semantic_context = model._extract_sdc_semantic_context(output)
    if semantic_context is None:
        raise RuntimeError("Selected row did not produce an sdc_semantic_only context.")

    valid_by_t = np.asarray(semantic_context["sdc_valid_by_t"][0].detach().cpu(), dtype=bool)
    step_index = _resolve_step_index(valid_by_t, requested_step_index)
    decision_agent_mask = semantic_context["decision_agent_mask"][0].detach().cpu().numpy()
    sdc_agent_idx = int(np.argmax(decision_agent_mask))

    current_vel = (
        batch_torch["decoder/modeled_agent_velocity"][0, step_index, sdc_agent_idx, :2]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    current_speed = float(np.linalg.norm(current_vel))

    render_context = _extract_scene_render_context(raw_scenario, row)
    center_xy_world = np.asarray(render_context["current_xy"], dtype=np.float32)
    center_heading_world = float(render_context["current_heading"])
    map_center_world = np.asarray(row.get("candidate_family_map_center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    map_heading_world = float(row.get("candidate_family_map_heading", 0.0) or 0.0)

    xx, yy, local_points = _scene_local_grid(crop_radius_m=float(crop_radius_m), grid_step_m=float(grid_step_m))
    world_points = _sdc_up_to_world_frame(local_points, center_xy=center_xy_world, heading_rad=center_heading_world)
    model_points = world_xy_to_model_frame(world_points, map_center=map_center_world, map_heading=map_heading_world)

    dtype = semantic_context["family_paths_world"].dtype
    model_points_t = torch.as_tensor(model_points, device=device, dtype=dtype).reshape(1, -1, 2)
    family_paths_world = semantic_context["family_paths_world"][:, :1 * 0 + semantic_context["family_paths_world"].shape[1]]
    family_path_mask = semantic_context["family_path_mask"]
    family_tangents_world = semantic_context["family_tangents_world"]
    family_arc_lengths = semantic_context["family_arc_lengths"]
    family_weights = semantic_context["family_weights"]

    current_projection = project_points_to_family_paths_torch(
        model_points_t,
        family_path_polylines_world=family_paths_world,
        family_path_mask=family_path_mask,
        family_path_tangents_world=family_tangents_world,
        family_path_arc_lengths=family_arc_lengths,
    )
    current_heading_model = _weighted_angle(current_projection["nearest_heading"], family_weights[:, None, :])
    current_vel_model = torch.stack(
        [
            torch.cos(current_heading_model) * current_speed,
            torch.sin(current_heading_model) * current_speed,
        ],
        dim=-1,
    )

    num_points = int(model_points.shape[0])
    num_actions = int(getattr(model._tokenizer, "num_actions"))
    dummy_logits = torch.zeros((1, num_points, 1, num_actions), device=device, dtype=dtype)
    dummy_data = {
        "decoder/modeled_agent_position": model_points_t[:, :, None, :],
        "decoder/modeled_agent_heading": current_heading_model[:, :, None],
        "decoder/modeled_agent_velocity": current_vel_model[:, :, None, :],
        "decoder/input_action_valid_mask": torch.ones((1, num_points, 1), device=device, dtype=torch.bool),
    }
    candidate_bundle = model._next_state_candidates_from_action_space(dummy_logits, dummy_data)
    sdc_next_pos_candidates_world = candidate_bundle["next_pos_candidates_world"][:, :, 0]
    sdc_candidate_heading_world = candidate_bundle["candidate_heading_world"][:, :, 0]

    custom_semantic_context = {
        "sdc_next_pos_candidates_world": sdc_next_pos_candidates_world,
        "sdc_candidate_heading_world": sdc_candidate_heading_world,
        "family_paths_world": family_paths_world,
        "family_path_mask": family_path_mask,
        "family_tangents_world": family_tangents_world,
        "family_arc_lengths": family_arc_lengths,
        "family_weights": family_weights,
        "current_projection": current_projection,
    }
    dummy_student_logits = torch.zeros((1, num_points, num_actions), device=device, dtype=dtype)
    energy_terms = _compute_family_action_energy_terms(model, custom_semantic_context, dummy_student_logits)
    weighted_energy = energy_terms["weighted_action_energy"][0]
    best_token = torch.argmin(weighted_energy, dim=-1)
    best_energy = weighted_energy.gather(dim=-1, index=best_token.unsqueeze(-1)).squeeze(-1)
    best_next_model = torch.gather(
        sdc_next_pos_candidates_world[0],
        dim=1,
        index=best_token[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)

    best_next_world = _model_to_world(
        best_next_model.detach().cpu().numpy().astype(np.float32),
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    local_best_next = _world_to_sdc_up_frame(best_next_world, center_xy=center_xy_world, heading_rad=center_heading_world)
    local_current = local_points.astype(np.float32)
    local_delta = local_best_next - local_current

    selected_path_id, selected_segments_world = _selected_path_world_from_row(row)
    actual_current_model = np.asarray(semantic_context["sdc_current_pos_world"][0, step_index].detach().cpu(), dtype=np.float32).reshape(1, 2)
    actual_current_world = _model_to_world(actual_current_model, map_center_world=map_center_world, map_heading_world=map_heading_world)
    actual_current_local = _world_to_sdc_up_frame(actual_current_world, center_xy=center_xy_world, heading_rad=center_heading_world)

    return {
        "row": dict(row),
        "render_context": render_context,
        "selected_path_id": selected_path_id,
        "selected_segments_world": selected_segments_world,
        "step_index": int(step_index),
        "current_speed_mps": current_speed,
        "grid_x": xx,
        "grid_y": yy,
        "local_delta": local_delta.reshape(xx.shape + (2,)),
        "best_token": best_token.detach().cpu().numpy().astype(np.int64).reshape(xx.shape),
        "best_energy": best_energy.detach().cpu().numpy().astype(np.float32).reshape(xx.shape),
        "actual_current_local_xy": actual_current_local.reshape(-1, 2),
    }


def _plot_slot_panel(
    *,
    fig,
    ax,
    slot_field: Mapping[str, Any],
    norm: Normalize,
):
    row = slot_field["row"]
    _draw_vlm_style_scene_ax(
        fig=fig,
        ax=ax,
        render_context=slot_field["render_context"],
        highlighted_segments_world=slot_field["selected_segments_world"],
        highlighted_gradient_values=None,
        representative_route_world=None,
        info_box_text=(
            f"slot={row['selected_slot_id']}\n"
            f"requested={row['requested_semantic_label']}\n"
            f"step={slot_field['step_index']}  speed={slot_field['current_speed_mps']:.2f} m/s\n"
            f"heading source=family-weighted nearest tangent\n"
            f"score=semantic family energy"
        ),
        show_colorbar=False,
    )
    grid_x = np.asarray(slot_field["grid_x"], dtype=np.float32)
    grid_y = np.asarray(slot_field["grid_y"], dtype=np.float32)
    local_delta = np.asarray(slot_field["local_delta"], dtype=np.float32)
    best_energy = np.asarray(slot_field["best_energy"], dtype=np.float32)
    quiver = ax.quiver(
        grid_x,
        grid_y,
        local_delta[..., 0],
        local_delta[..., 1],
        np.clip(best_energy, norm.vmin, norm.vmax),
        cmap="magma_r",
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        alpha=0.9,
        zorder=11.0,
    )
    actual_current = np.asarray(slot_field["actual_current_local_xy"], dtype=np.float32)
    if actual_current.size > 0:
        ax.scatter(
            actual_current[:, 0],
            actual_current[:, 1],
            marker="*",
            s=110,
            c="#facc15",
            edgecolors="#111827",
            linewidths=0.7,
            zorder=12.0,
        )
    ax.set_title(f"{row['selected_slot_id']}  ({row['requested_semantic_label']})", fontsize=12, pad=8)
    return quiver


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
        load_mode="forgiving_state_dict",
        strict_state_dict=False,
        map_location=str(device),
    )
    model = model.to(device)
    model.eval()
    model._trainer = type("TrainerStub", (), {"world_size": 1, "lr_scheduler_configs": None, "optimizers": None})()

    row_indices = _select_scene_row_indices(rows, scenario_id=str(args.scenario_id).strip(), include_gt=False)
    if not row_indices:
        raise ValueError(f"No alternate rows found for scenario_id={args.scenario_id!r}")

    slot_fields = [
        _build_slot_vector_field(
            row=rows[row_index],
            row_index=row_index,
            dataset=dataset,
            model=model,
            device=device,
            crop_radius_m=float(args.crop_radius_m),
            grid_step_m=float(args.grid_step_m),
            requested_step_index=int(args.step_index),
        )
        for row_index in row_indices
    ]

    all_energy = np.concatenate([np.asarray(item["best_energy"], dtype=np.float32).reshape(-1) for item in slot_fields], axis=0)
    finite_energy = all_energy[np.isfinite(all_energy)]
    vmax = float(np.quantile(finite_energy, 0.95)) if finite_energy.size > 0 else 1.0
    vmin = float(np.min(finite_energy)) if finite_energy.size > 0 else 0.0
    if not math.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    combined_fig, axes = plt.subplots(1, len(slot_fields), figsize=(6.1 * len(slot_fields), 6.9), dpi=180)
    if len(slot_fields) == 1:
        axes = [axes]
    quiver_artist = None
    slot_summaries: List[Dict[str, Any]] = []
    for ax, slot_field in zip(axes, slot_fields):
        quiver_artist = _plot_slot_panel(fig=combined_fig, ax=ax, slot_field=slot_field, norm=norm)
        row = slot_field["row"]
        slot_outdir = outdir / str(row["selected_slot_id"])
        slot_outdir.mkdir(parents=True, exist_ok=True)
        slot_png = slot_outdir / "best_token_vector_field.png"
        single_fig = plt.figure(figsize=(7.2, 7.8), dpi=180)
        single_ax = single_fig.add_axes([0.02, 0.02, 0.96, 0.96])
        single_quiver = _plot_slot_panel(fig=single_fig, ax=single_ax, slot_field=slot_field, norm=norm)
        cbar = single_fig.colorbar(single_quiver, ax=single_ax, fraction=0.036, pad=0.015)
        cbar.set_label("Best-token semantic energy", fontsize=9)
        single_fig.savefig(slot_png, dpi=180)
        plt.close(single_fig)
        slot_summaries.append(
            {
                "selected_slot_id": str(row["selected_slot_id"]),
                "requested_semantic_label": str(row["requested_semantic_label"]),
                "step_index": int(slot_field["step_index"]),
                "current_speed_mps": float(slot_field["current_speed_mps"]),
                "selected_path_id": str(slot_field["selected_path_id"]),
                "best_token_vector_field_png": str(slot_png),
            }
        )

    if quiver_artist is not None:
        cbar = combined_fig.colorbar(quiver_artist, ax=list(axes), fraction=0.025, pad=0.02)
        cbar.set_label("Best-token semantic energy", fontsize=10)
    combined_fig.suptitle(
        f"Scene {args.scenario_id}: best-scoring token vector field per alternate",
        fontsize=14,
        y=0.995,
    )
    combined_png = outdir / "scene_best_token_vector_fields.png"
    combined_fig.savefig(combined_png, dpi=180)
    plt.close(combined_fig)

    manifest = {
        "scenario_id": str(args.scenario_id),
        "checkpoint": str(Path(args.ckpt).expanduser().resolve()),
        "checkpoint_load_report": load_report,
        "grid_step_m": float(args.grid_step_m),
        "crop_radius_m": float(args.crop_radius_m),
        "step_index_request": int(args.step_index),
        "heading_source": "family-weighted nearest tangent at each grid point",
        "speed_source": "row SDC speed at inspected step",
        "combined_png": str(combined_png),
        "slots": slot_summaries,
    }
    manifest_path = outdir / "vector_field_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
