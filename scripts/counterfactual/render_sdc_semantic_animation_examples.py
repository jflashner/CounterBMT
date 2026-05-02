from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from easydict import EasyDict
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import torch
from PIL import Image

try:
    import seaborn  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from matplotlib import cm

    seaborn_stub = types.ModuleType("seaborn")

    def _fallback_color_palette(name: str = "colorblind", n_colors: int = 10):
        cmap = cm.get_cmap("tab20", int(max(1, n_colors)))
        return [tuple(float(v) for v in cmap(i)[:3]) for i in range(int(max(1, n_colors)))]

    seaborn_stub.color_palette = _fallback_color_palette  # type: ignore[attr-defined]
    sys.modules["seaborn"] = seaborn_stub

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.forward_supervision import (
    preprocess_raw_scenario_for_forward_supervision,
    summarize_forward_supervision_for_sample,
)
from bmt.counterfactual.normalize import load_raw_scenario
from bmt.counterfactual.sdc_semantic_control import extract_model_frame, load_raw_scenario_from_row
from bmt.dataset.dataset import InfgenDataset
from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule, _load_eval_model
from bmt.eval.scenario_evaluator import Evaluator
from bmt.tokenization import get_tokenizer
from scripts.counterfactual.eval_sdc_semantic_action_projections import _extract_scene_render_context
from scripts.counterfactual.probe_agent_semantic_rollout import (
    _build_eval_module,
    _build_control_sample,
    _load_config as _load_probe_config,
    _load_model as _load_probe_model,
    _normalize_track_id,
    _optional_positive_float,
    _run_rollout,
)
from scripts.counterfactual.path_semantics_plot_utils import (
    AGENT_COLOR,
    CROSSWALK_FACE,
    LANE_COLOR,
    PLOT_RADIUS_M,
    ROAD_COLOR,
    SDC_VERTICAL_FRACTION,
    _world_to_sdc_up_frame,
)
from scripts.counterfactual.render_sdc_semantic_eval_examples import (
    _ensure_plot_fields,
    _load_config,
    _read_jsonl,
    _resolve_device,
    _scenario_sort_key,
    _to_builtin,
    _to_numpy_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render GT and predicted multi-agent animations for SDC semantic-control validation examples "
            "using a trained checkpoint."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml",
    )
    parser.add_argument("--control-index", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sampling-method", type=str, default="argmax")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topp", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--draw-traffic", action="store_true")
    parser.add_argument("--full-scene-view", action="store_true")
    parser.add_argument(
        "--non-sdc-cases-json",
        type=str,
        default="",
        help="Optional JSON list of arbitrary-agent semantic intervention cases to render with GIFs.",
    )
    return parser.parse_args()


def _row_slot_id(row: Dict[str, Any], row_idx: int) -> str:
    selected_slot = str(row.get("selected_slot_id") or "").strip()
    slot_id = str(row.get("slot_id") or "").strip()
    if selected_slot:
        return selected_slot
    if slot_id:
        return slot_id
    if str(row.get("source_kind") or "") == "factual_gt":
        return "factual_gt"
    return f"row_{int(row_idx):04d}"


def _sanitize_token(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip()) or "item"


def _merge_pred_with_raw(raw_data: Dict[str, Any], output_np: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(raw_data)
    merged.update(output_np)
    return _ensure_plot_fields(merged)


def _figure_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(np.ascontiguousarray(rgb))


def _to_world_xy(points_model_xy: np.ndarray, *, map_center_world: np.ndarray, map_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_model_xy, dtype=np.float32).reshape(-1, 2)
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if float(map_heading_world) == 0.0:
        return (xy + np.asarray(map_center_world, dtype=np.float32).reshape(1, 3)[:, :2]).astype(np.float32)
    c = math.cos(float(map_heading_world))
    s = math.sin(float(map_heading_world))
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(map_center_world[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(map_center_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def _convert_agent_positions_to_world(
    pos_model: np.ndarray,
    *,
    raw_scenario: Dict[str, Any],
) -> np.ndarray:
    pos_array = np.asarray(pos_model, dtype=np.float32)
    if pos_array.ndim != 3 or pos_array.shape[-1] < 2:
        return np.zeros((0, 0, 2), dtype=np.float32)
    time_steps, num_agents = pos_array.shape[:2]
    map_center_world, map_heading_world = extract_model_frame(raw_scenario)
    world_flat = _to_world_xy(
        pos_array[..., :2].reshape(-1, 2),
        map_center_world=np.asarray(map_center_world, dtype=np.float32),
        map_heading_world=float(map_heading_world),
    )
    return world_flat.reshape(time_steps, num_agents, 2).astype(np.float32)


def _infer_sdc_current_time_index(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_track_id: str,
    sdc_reference_world_xy: np.ndarray,
) -> int:
    tracks = dict(raw_scenario.get("tracks", {}) or {})
    track = dict(tracks.get(str(sdc_track_id), {}) or {})
    state = dict(track.get("state", {}) or {})
    position = np.asarray(state.get("position", []), dtype=np.float32)
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[0] == 0 or valid.shape[0] == 0 or sdc_reference_world_xy.shape[0] == 0:
        return 0
    valid_indices = np.flatnonzero(valid[: position.shape[0]])
    if valid_indices.size == 0:
        return 0
    anchor_xy = np.asarray(sdc_reference_world_xy[0], dtype=np.float32).reshape(1, 2)
    candidate_xy = np.asarray(position[valid_indices, :2], dtype=np.float32)
    nearest = int(np.argmin(np.linalg.norm(candidate_xy - anchor_xy, axis=-1)))
    return int(valid_indices[nearest])


def _reference_world_and_valid_from_batch(
    *,
    batch_torch: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pos_model = np.asarray(batch_torch["decoder/agent_position"][0, :, :, :2].detach().cpu(), dtype=np.float32)
    valid_mask = np.asarray(batch_torch["decoder/agent_valid_mask"][0].detach().cpu(), dtype=bool)
    world_pos = _convert_agent_positions_to_world(pos_model, raw_scenario=dict(raw_scenario))
    return world_pos, valid_mask


def _reference_world_and_valid_from_sample(
    *,
    sample: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pos_model = np.asarray(sample["decoder/agent_position"], dtype=np.float32)[..., :2]
    valid_mask = np.asarray(sample["decoder/agent_valid_mask"], dtype=bool)
    world_pos = _convert_agent_positions_to_world(pos_model, raw_scenario=dict(raw_scenario))
    return world_pos, valid_mask


def _rollout_world_and_valid_from_eval_output(
    *,
    eval_output: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    current_model = np.asarray(eval_output["decoder/modeled_agent_position"][0, :, :, :2].detach().cpu(), dtype=np.float32)
    rollout_next_model = np.asarray(eval_output["decoder/rollout_next_position"][0, :, :, :2].detach().cpu(), dtype=np.float32)
    rollout_model = np.concatenate([current_model[:1], rollout_next_model], axis=0).astype(np.float32)
    current_valid = np.asarray(eval_output["decoder/modeled_agent_valid_mask"][0].detach().cpu(), dtype=bool)
    next_valid = np.asarray(eval_output["decoder/input_action_valid_mask"][0].detach().cpu(), dtype=bool)
    rollout_valid = np.concatenate([current_valid[:1], next_valid], axis=0).astype(bool)
    world_pos = _convert_agent_positions_to_world(rollout_model, raw_scenario=dict(raw_scenario))
    return world_pos, rollout_valid


def _rollout_world_and_valid_from_output_np(
    *,
    output_np: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    pos_model = np.asarray(output_np["decoder/reconstructed_position"], dtype=np.float32)[..., :2]
    valid_mask = np.asarray(output_np["decoder/reconstructed_valid_mask"], dtype=bool)
    world_pos = _convert_agent_positions_to_world(pos_model, raw_scenario=dict(raw_scenario))
    return world_pos, valid_mask


def _draw_local_scene_background(ax, render_context: Dict[str, Any]) -> None:
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    map_context = render_context["map_context"]
    traffic_lights = render_context["traffic_lights"]

    ax.set_facecolor("#f8fafc")
    for feature in map_context.get("crosswalks", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 3:
            ax.fill(xy[:, 0], xy[:, 1], color=CROSSWALK_FACE, alpha=0.35, zorder=1)
    for feature in map_context.get("road_boundaries", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=ROAD_COLOR, linewidth=2.8, alpha=0.98, zorder=2)
    for feature in map_context.get("lane_centerlines", []):
        xy = _world_to_sdc_up_frame(np.asarray(feature["xy_world"], dtype=np.float64), center_xy=center_xy, heading_rad=current_heading)
        if xy.shape[0] >= 2:
            ax.plot(xy[:, 0], xy[:, 1], color=LANE_COLOR, linewidth=1.1, alpha=0.30, zorder=3)
    for light in traffic_lights:
        stop_xy = _world_to_sdc_up_frame(
            np.asarray([light["stop_point_xy_world"]], dtype=np.float64),
            center_xy=center_xy,
            heading_rad=current_heading,
        )
        if stop_xy.shape[0] == 0:
            continue
        state = str(light.get("state") or "unknown")
        color = "#ef4444" if ("STOP" in state or "RED" in state) else ("#22c55e" if ("GO" in state or "GREEN" in state) else "#eab308")
        ax.scatter([stop_xy[0, 0]], [stop_xy[0, 1]], c=color, marker="s", s=70, edgecolors="black", linewidths=0.8, zorder=4)

    half_extent = float(PLOT_RADIUS_M)
    vertical_span = 2.0 * half_extent
    y_min = -float(SDC_VERTICAL_FRACTION) * vertical_span
    y_max = y_min + vertical_span
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def _choose_visible_agents(
    world_pos: np.ndarray,
    valid_mask: np.ndarray,
    *,
    center_xy: np.ndarray,
    radius_m: float,
    sdc_index: int,
    force_include_agents: Sequence[int] | None = None,
) -> np.ndarray:
    if world_pos.ndim != 3 or valid_mask.ndim != 2:
        return np.asarray([sdc_index], dtype=np.int64)
    num_agents = int(world_pos.shape[1])
    keep: List[int] = [int(sdc_index)]
    if force_include_agents is not None:
        keep.extend(int(idx) for idx in force_include_agents)
    for agent_idx in range(num_agents):
        if agent_idx == int(sdc_index):
            continue
        mask = np.asarray(valid_mask[:, agent_idx], dtype=bool)
        if not np.any(mask):
            continue
        agent_xy = np.asarray(world_pos[:, agent_idx, :], dtype=np.float32)[mask]
        if agent_xy.shape[0] == 0:
            continue
        if float(np.min(np.linalg.norm(agent_xy - center_xy.reshape(1, 2), axis=-1))) <= float(radius_m):
            keep.append(int(agent_idx))
    return np.asarray(sorted(set(keep)), dtype=np.int64)


def _render_local_multiactor_gif(
    *,
    raw_scenario: Dict[str, Any],
    row: Dict[str, Any],
    world_pos: np.ndarray,
    valid_mask: np.ndarray,
    sdc_index: int,
    out_path: Path,
    fps: float,
    dpi: int,
    title_lines: Sequence[str],
    force_include_agents: Sequence[int] | None = None,
    special_agent_styles: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    render_context = _extract_scene_render_context(raw_scenario, row)
    center_xy = np.asarray(render_context["current_xy"], dtype=np.float64)
    current_heading = float(render_context["current_heading"])
    visible_agents = _choose_visible_agents(
        world_pos,
        np.asarray(valid_mask, dtype=bool),
        center_xy=np.asarray(center_xy, dtype=np.float32),
        radius_m=float(PLOT_RADIUS_M) * 1.15,
        sdc_index=int(sdc_index),
        force_include_agents=force_include_agents,
    )
    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(20, int(len(visible_agents)) + 1)))[:, :3]

    frames: List[Image.Image] = []
    total_steps = int(world_pos.shape[0])
    for step_idx in range(total_steps):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        _draw_local_scene_background(ax, render_context)

        for color_idx, agent_idx in enumerate(visible_agents.tolist()):
            mask = np.asarray(valid_mask[: step_idx + 1, agent_idx], dtype=bool)
            if not np.any(mask):
                continue
            history_world = np.asarray(world_pos[: step_idx + 1, agent_idx, :], dtype=np.float64)[mask]
            local_xy = _world_to_sdc_up_frame(history_world, center_xy=center_xy, heading_rad=current_heading)
            if local_xy.shape[0] == 0:
                continue
            is_sdc = int(agent_idx) == int(sdc_index)
            style = dict((special_agent_styles or {}).get(int(agent_idx), {}))
            color = style.get("color", ("#111827" if is_sdc else tuple(float(v) for v in palette[color_idx % len(palette)])))
            linewidth = float(style.get("linewidth", (2.8 if is_sdc else 1.4)))
            alpha = float(style.get("alpha", (0.95 if is_sdc else 0.78)))
            zorder = int(style.get("zorder", (8 if is_sdc else 6)))
            marker_size = float(style.get("marker_size", (48 if is_sdc else 24)))
            if local_xy.shape[0] >= 2:
                ax.plot(
                    local_xy[:, 0],
                    local_xy[:, 1],
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
            ax.scatter(
                [local_xy[-1, 0]],
                [local_xy[-1, 1]],
                c=[color],
                s=marker_size,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.98,
                zorder=zorder + 1,
            )

        info_box_text = "\n".join([str(line) for line in title_lines] + [f"step {step_idx + 1}/{total_steps}"])
        ax.text(
            0.02,
            0.975,
            info_box_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#cbd5e1"},
            zorder=20,
        )
        fig.tight_layout(pad=0.05)
        frames.append(_figure_to_image(fig))
        plt.close(fig)

    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(round(1000.0 / max(float(fps), 1.0)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )


def _render_sdc_control_index_cases(args: argparse.Namespace, *, outdir: Path) -> None:
    control_index = Path(args.control_index).expanduser().resolve()
    rows = _read_jsonl(control_index)
    all_sids: List[str] = []
    for row in rows:
        sid = str(row.get("scenario_id") or "")
        if sid and sid not in all_sids:
            all_sids.append(sid)

    scene_offset = max(0, int(args.scene_offset))
    scene_limit = max(0, int(args.num_scenes))
    selected_sids = all_sids[scene_offset : scene_offset + scene_limit]
    selected_sid_set = set(selected_sids)

    runtime_config = _load_config(args)
    model = _load_eval_model(runtime_config, str(Path(args.ckpt).expanduser().resolve()))
    config = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.create(_to_builtin(model.config)), runtime_config)
    model.config = config
    tokenizer = get_tokenizer(config)
    dataset = InfgenDataset(config, "test", backward_prediction=False)
    module = EvaluationLightningModule(
        model=model,
        evaluator=Evaluator(key_metrics_only=True),
        tokenizer=tokenizer,
        config=config,
        dataset=dataset,
        eval_mode="GPTmodel",
        multi_mode=False,
        num_modes=1,
        save_path=str(outdir / "unused_metrics"),
    )
    module.eval()
    module.model.eval()
    device = _resolve_device(args.device)
    module.model.to(device)

    scenario_to_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {sid: [] for sid in selected_sids}
    for idx, row in enumerate(rows):
        sid = str(row.get("scenario_id") or "")
        if sid in selected_sid_set:
            scenario_to_rows[sid].append((idx, row))

    manifest: List[Dict[str, Any]] = []
    for scenario_id in selected_sids:
        row_entries = sorted(scenario_to_rows[scenario_id], key=lambda item: _scenario_sort_key(item[1]))
        if not row_entries:
            continue

        scenario_dir = outdir / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        gt_entry = None
        for row_idx, row in row_entries:
            if str(row.get("source_kind") or "") == "factual_gt" or str(row.get("selected_slot_id") or "") == "gt":
                gt_entry = (row_idx, row)
                break
        if gt_entry is None:
            gt_entry = row_entries[0]

        gt_row_idx, gt_row = gt_entry
        gt_raw = _ensure_plot_fields(copy.deepcopy(dataset[gt_row_idx]))
        raw_scenario = load_raw_scenario_from_row(gt_row)
        gt_video = scenario_dir / f"{scenario_id}__ground_truth.gif"
        gt_world_pos = _convert_agent_positions_to_world(
            np.asarray(gt_raw["decoder/agent_position"], dtype=np.float32)[..., :2],
            raw_scenario=raw_scenario,
        )
        gt_valid = np.asarray(gt_raw["decoder/agent_valid_mask"], dtype=bool)
        gt_sdc_index = int(np.asarray(gt_raw.get("decoder/sdc_index", 0)).reshape(-1)[0])
        _render_local_multiactor_gif(
            raw_scenario=raw_scenario,
            row=dict(gt_row),
            world_pos=gt_world_pos,
            valid_mask=gt_valid,
            sdc_index=gt_sdc_index,
            out_path=gt_video,
            fps=float(args.fps),
            dpi=int(args.dpi),
            title_lines=[
                scenario_id,
                "ground truth",
            ],
        )

        row_manifest: List[Dict[str, Any]] = []
        for row_idx, row in row_entries:
            raw_data = copy.deepcopy(dataset[row_idx])
            input_data = module.preprocess_GPTmodel(copy.deepcopy(raw_data), backward_prediction=False)
            with torch.no_grad():
                output_data = module.GPT_AR(input_data, backward_prediction=False, teacher_forcing=False)
            output_data = tokenizer.detokenize(
                output_data,
                detokenizing_gt=False,
                backward_prediction=False,
                teacher_forcing=False,
            )
            output_np = _to_numpy_output(output_data)
            pred_anim_data = _merge_pred_with_raw(raw_data, output_np)

            slot_id = _row_slot_id(row, int(row_idx))
            requested_label = str(row.get("requested_semantic_label") or "")
            source_kind = str(row.get("source_kind") or "")
            stem = f"{scenario_id}__{_sanitize_token(slot_id)}__pred"
            pred_video = scenario_dir / f"{stem}.gif"
            pred_world_pos = _convert_agent_positions_to_world(
                np.asarray(pred_anim_data["decoder/reconstructed_position"], dtype=np.float32)[..., :2],
                raw_scenario=raw_scenario,
            )
            pred_valid = np.asarray(pred_anim_data["decoder/reconstructed_valid_mask"], dtype=bool)
            pred_sdc_index = int(np.asarray(pred_anim_data.get("decoder/sdc_index", 0)).reshape(-1)[0])
            _render_local_multiactor_gif(
                raw_scenario=raw_scenario,
                row=dict(row),
                world_pos=pred_world_pos,
                valid_mask=pred_valid,
                sdc_index=pred_sdc_index,
                out_path=pred_video,
                fps=float(args.fps),
                dpi=int(args.dpi),
                title_lines=[
                    scenario_id,
                    f"slot={slot_id}",
                    (f"label={requested_label}" if requested_label else f"source={source_kind or 'unknown'}"),
                ],
            )

            row_manifest.append(
                {
                    "row_index": int(row_idx),
                    "slot_id": slot_id,
                    "source_kind": source_kind,
                    "requested_semantic_label": requested_label,
                    "selected_path_id": row.get("selected_path_id"),
                    "pred_gif": str(pred_video),
                }
            )

        manifest.append(
            {
                "scenario_id": scenario_id,
                "ground_truth_gif": str(gt_video),
                "rows": row_manifest,
            }
        )

    manifest_path = outdir / f"manifest_offset{scene_offset:03d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"num_scenes": len(manifest), "manifest": str(manifest_path)}, indent=2))


def _load_non_sdc_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = list(payload.get("cases") or [])
    elif isinstance(payload, list):
        cases = list(payload)
    else:
        raise ValueError(f"Expected JSON list or object with 'cases', got {type(payload).__name__}")
    normalized: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"Case {idx} must be a JSON object.")
        normalized.append(dict(case))
    return normalized


def _render_non_sdc_cases(args: argparse.Namespace, *, outdir: Path) -> None:
    cases_path = Path(args.non_sdc_cases_json).expanduser().resolve()
    cases = _load_non_sdc_cases(cases_path)
    scene_offset = max(0, int(args.scene_offset))
    scene_limit = max(0, int(args.num_scenes))
    selected_cases = cases[scene_offset : scene_offset + scene_limit] if scene_limit > 0 else cases[scene_offset:]

    config = _load_probe_config(args)
    device = _resolve_device(args.device)
    model, load_report = _load_probe_model(config=config, ckpt_path=args.ckpt, load_mode="forgiving_state_dict")
    model = model.to(device)
    module, tokenizer = _build_eval_module(
        config=config,
        ckpt_path=args.ckpt,
        device=device,
        save_path=outdir / "unused_eval_metrics",
        model=model,
    )
    if str(args.sampling_method).strip():
        module.config.SAMPLING.SAMPLING_METHOD = str(args.sampling_method)
    rollout_temperature = _optional_positive_float(float(args.temperature))
    rollout_topp = _optional_positive_float(float(args.topp))
    if rollout_temperature is not None:
        module.config.SAMPLING.TEMPERATURE = float(rollout_temperature)
    if rollout_topp is not None:
        module.config.SAMPLING.TOPP = float(rollout_topp)

    manifest: List[Dict[str, Any]] = []
    for case_idx, case in enumerate(selected_cases):
        scenario_pkl = Path(str(case.get("scenario_pkl") or "")).expanduser()
        if not scenario_pkl.is_absolute():
            scenario_pkl = (Path.cwd() / scenario_pkl).resolve()
        raw_scenario = load_raw_scenario(scenario_pkl)
        base_sample = preprocess_raw_scenario_for_forward_supervision(raw_scenario, config=config, in_evaluation=True)
        base_sample["metadata/scenario_id"] = str(raw_scenario.get("id") or base_sample.get("metadata/scenario_id", ""))
        forward_summary = summarize_forward_supervision_for_sample(base_sample, raw_scenario=raw_scenario)

        agent_id = _normalize_track_id(case.get("agent_id"))
        if not agent_id:
            raise ValueError(f"Case {case_idx} is missing agent_id")
        target_summary = next((row for row in forward_summary.agents if row.raw_track_id == agent_id), None)
        if target_summary is None:
            raise ValueError(f"Agent '{agent_id}' is not modeled in scenario '{forward_summary.scenario_id}'")
        target_slot = int(target_summary.model_agent_slot)
        modeled_agent_ids = list(forward_summary.modeled_agent_ids)
        try:
            sdc_slot = modeled_agent_ids.index(str(forward_summary.sdc_id))
        except ValueError:
            sdc_slot = 0

        horizon = int(np.asarray(base_sample["decoder/target_action_valid_mask"]).shape[0])
        time_window_mask = np.zeros((horizon,), dtype=np.float32)
        start_step = int(case.get("start_step", 0))
        requested_end_step = int(case.get("end_step", -1))
        start_step = max(0, start_step)
        end_step = horizon - 1 if requested_end_step < 0 else min(max(start_step, requested_end_step), horizon - 1)
        time_window_mask[start_step : end_step + 1] = 1.0
        decision_agent_mask = np.zeros((len(modeled_agent_ids),), dtype=np.float32)
        decision_agent_mask[target_slot] = 1.0
        semantic_label = str(case.get("semantic_label") or "")
        semantic_confidence = float(case.get("semantic_confidence", 1.0))

        controlled_sample = _build_control_sample(
            base_sample=base_sample,
            semantic_label=semantic_label,
            semantic_confidence=semantic_confidence,
            time_window_mask=time_window_mask,
            decision_agent_mask=decision_agent_mask,
        )
        baseline = _run_rollout(module, tokenizer, raw_sample=base_sample)
        controlled = _run_rollout(module, tokenizer, raw_sample=controlled_sample)

        reference_world_pos, reference_valid = _reference_world_and_valid_from_sample(sample=base_sample, raw_scenario=raw_scenario)
        baseline_world_pos, baseline_valid = _rollout_world_and_valid_from_output_np(
            output_np=baseline["output_np"],
            raw_scenario=raw_scenario,
        )
        controlled_world_pos, controlled_valid = _rollout_world_and_valid_from_output_np(
            output_np=controlled["output_np"],
            raw_scenario=raw_scenario,
        )
        sdc_reference_world_xy = np.asarray(reference_world_pos[:, sdc_slot, :], dtype=np.float32)
        current_time_index = _infer_sdc_current_time_index(
            raw_scenario=raw_scenario,
            sdc_track_id=str(forward_summary.sdc_id),
            sdc_reference_world_xy=sdc_reference_world_xy,
        )
        row_context = {
            "sdc_id": str(forward_summary.sdc_id),
            "current_time_index": int(current_time_index),
        }

        scenario_id = str(forward_summary.scenario_id)
        case_name = str(case.get("case_name") or f"{scenario_id}__adv_{_sanitize_token(agent_id)}__{_sanitize_token(semantic_label)}")
        case_dir = outdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        force_include = [int(sdc_slot), int(target_slot)]

        reference_gif = case_dir / f"{case_name}__reference.gif"
        baseline_gif = case_dir / f"{case_name}__baseline.gif"
        controlled_gif = case_dir / f"{case_name}__controlled.gif"
        _render_local_multiactor_gif(
            raw_scenario=dict(raw_scenario),
            row=dict(row_context),
            world_pos=reference_world_pos,
            valid_mask=reference_valid,
            sdc_index=int(sdc_slot),
            out_path=reference_gif,
            fps=float(args.fps),
            dpi=int(args.dpi),
            title_lines=[scenario_id, "reference", f"adv={agent_id}", f"victim={forward_summary.sdc_id}"],
            force_include_agents=force_include,
            special_agent_styles={
                int(sdc_slot): {"color": "#16a34a", "linewidth": 2.8, "marker_size": 48, "zorder": 8},
                int(target_slot): {"color": "#64748b", "linewidth": 2.6, "marker_size": 48, "zorder": 9},
            },
        )
        _render_local_multiactor_gif(
            raw_scenario=dict(raw_scenario),
            row=dict(row_context),
            world_pos=baseline_world_pos,
            valid_mask=baseline_valid,
            sdc_index=int(sdc_slot),
            out_path=baseline_gif,
            fps=float(args.fps),
            dpi=int(args.dpi),
            title_lines=[scenario_id, "baseline rollout", f"adv={agent_id}", f"victim={forward_summary.sdc_id}"],
            force_include_agents=force_include,
            special_agent_styles={
                int(sdc_slot): {"color": "#16a34a", "linewidth": 2.8, "marker_size": 48, "zorder": 8},
                int(target_slot): {"color": "#2563eb", "linewidth": 2.8, "marker_size": 48, "zorder": 9},
            },
        )
        _render_local_multiactor_gif(
            raw_scenario=dict(raw_scenario),
            row=dict(row_context),
            world_pos=controlled_world_pos,
            valid_mask=controlled_valid,
            sdc_index=int(sdc_slot),
            out_path=controlled_gif,
            fps=float(args.fps),
            dpi=int(args.dpi),
            title_lines=[scenario_id, f"controlled ({semantic_label})", f"adv={agent_id}", f"victim={forward_summary.sdc_id}"],
            force_include_agents=force_include,
            special_agent_styles={
                int(sdc_slot): {"color": "#16a34a", "linewidth": 2.8, "marker_size": 48, "zorder": 8},
                int(target_slot): {"color": "#dc2626", "linewidth": 2.8, "marker_size": 48, "zorder": 9},
            },
        )
        manifest.append(
            {
                "scenario_id": scenario_id,
                "case_name": case_name,
                "agent_id": agent_id,
                "semantic_label": semantic_label,
                "victim_agent_id": str(forward_summary.sdc_id),
                "current_time_index": int(current_time_index),
                "checkpoint_load_report": load_report,
                "artifacts": {
                    "reference_gif": str(reference_gif),
                    "baseline_gif": str(baseline_gif),
                    "controlled_gif": str(controlled_gif),
                },
            }
        )

    manifest_path = outdir / f"manifest_offset{scene_offset:03d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"num_cases": len(manifest), "manifest": str(manifest_path)}, indent=2))


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if str(args.non_sdc_cases_json).strip():
        _render_non_sdc_cases(args, outdir=outdir)
    else:
        if not str(args.control_index).strip() or not str(args.data_dir).strip():
            raise ValueError("--control-index and --data-dir are required unless --non-sdc-cases-json is provided.")
        _render_sdc_control_index_cases(args, outdir=outdir)


if __name__ == "__main__":
    main()
