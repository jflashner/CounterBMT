from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.getLogger("metadrive.type").setLevel(logging.ERROR)

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
from bmt.counterfactual.sdc_semantic_control import load_raw_scenario_from_row
from bmt.dataset.dataset import InfgenDataset
from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PR10.1 semantic-only SDC dataset/model/loss smokes.")
    parser.add_argument("--config", type=str, default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--debug-examples", type=int, default=2, help="Number of unique scenes/examples to emit debug artifacts for.")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    config = cfg_from_yaml_file(cfg_path, config)
    control_index = str(Path(args.control_index).expanduser())
    data_dir = str(Path(args.data_dir).expanduser())
    config.DATA.TRAINING_DATA_DIR = data_dir
    config.DATA.TEST_DATA_DIR = data_dir
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = control_index
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL = control_index
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX = ""
    config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = ""
    config.DATA.COUNTERFACTUAL_MODE = "sdc_semantic_only"
    config.DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER = True
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    config.MODEL.LOCAL_CONTROL_USE_ANCHOR = False
    config.MODEL.LOCAL_CONTROL_USE_COMPLIANCE = False
    config.MODEL.LOCAL_CONTROL_USE_TIMING = False
    teacher_ckpt = str(args.teacher_ckpt or args.ckpt).strip()
    config.MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT = teacher_ckpt
    return config


def _to_torch_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"b", "i", "u", "f", "c"}:
                output[key] = torch.from_numpy(value).to(device)
            else:
                output[key] = value
        else:
            output[key] = value
    return output


def _label_from_id(value: int) -> str:
    labels = ("left", "right", "left_lane_change", "right_lane_change", "straight", "stop")
    if 0 <= int(value) < len(labels):
        return labels[int(value)]
    return f"label_{value}"


def _summarize_batch(batch: Dict[str, Any], rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    semantic_ids = np.asarray(batch["cf/sdc_semantic_label_id"]).reshape(-1)
    confidence = np.asarray(batch["cf/sdc_semantic_confidence"]).reshape(-1)
    factual = np.asarray(batch["cf/sdc_is_factual"]).reshape(-1)
    available = np.asarray(batch["cf/sdc_control_available"]).reshape(-1)
    family_mask = np.asarray(batch["cf/sdc_family_path_mask"], dtype=np.float32)
    family_arc = np.asarray(batch["cf/sdc_family_arc_lengths"], dtype=np.float32)
    family_onsets = np.asarray(batch["cf/sdc_family_divergence_onsets"], dtype=np.float32)
    family_conf = np.asarray(batch["cf/sdc_family_confidences"], dtype=np.float32)
    return {
        "batch_size": int(len(rows)),
        "example_ids": [str(dict(row.get("metadata", {}) or {}).get("example_id") or "") for row in rows],
        "scenario_ids": [str(row.get("scenario_id") or "") for row in rows],
        "slot_ids": [str(row.get("selected_slot_id") or "") for row in rows],
        "source_kinds": [str(row.get("source_kind") or "") for row in rows],
        "semantic_labels": [_label_from_id(v) for v in semantic_ids.tolist()],
        "semantic_confidence": [float(v) for v in confidence.tolist()],
        "sdc_is_factual": [bool(v) for v in factual.tolist()],
        "sdc_control_available": [bool(v) for v in available.tolist()],
        "family_size": [int(mask.sum(axis=-1).gt(0).sum() if hasattr(mask.sum(axis=-1), "gt") else np.sum(mask.sum(axis=-1) > 0)) for mask in family_mask],
        "family_waypoint_count": [int(mask.sum()) for mask in family_mask],
        "family_max_arc_m": [float(arc[mask > 0].max()) if np.any(mask > 0) else 0.0 for arc, mask in zip(family_arc, family_mask)],
        "family_divergence_onset_min_m": [float(np.min(onset[np.isfinite(onset)])) if np.any(np.isfinite(onset)) else float("inf") for onset in family_onsets],
        "family_confidence_mean": [float(conf[conf >= 0].mean()) if conf.size > 0 else 0.0 for conf in family_conf],
    }


def _aggregate_batch_summaries(batch_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch_summaries:
        return {"num_rows_processed": 0, "num_examples_processed": 0, "num_batches": 0, "batch_size_used": 0}
    flat: Dict[str, List[Any]] = {
        "example_ids": [],
        "scenario_ids": [],
        "slot_ids": [],
        "source_kinds": [],
        "semantic_labels": [],
        "semantic_confidence": [],
        "sdc_is_factual": [],
        "sdc_control_available": [],
        "family_size": [],
        "family_waypoint_count": [],
        "family_max_arc_m": [],
        "family_divergence_onset_min_m": [],
        "family_confidence_mean": [],
    }
    batch_size_total = 0
    for summary in batch_summaries:
        batch_size_total += int(summary.get("batch_size") or 0)
        for key in flat.keys():
            flat[key].extend(list(summary.get(key, [])))
    example_counts: Dict[str, int] = {}
    for example_id in flat["example_ids"]:
        example_counts[str(example_id)] = int(example_counts.get(str(example_id), 0) + 1)

    def _stats(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return {"min": 0.0, "mean": 0.0, "max": 0.0}
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return {"min": float("inf"), "mean": float("inf"), "max": float("inf")}
        return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}

    def _hist(values: List[Any]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for value in values:
            hist[str(value)] = int(hist.get(str(value), 0) + 1)
        return hist

    return {
        "num_rows_processed": int(batch_size_total),
        "num_examples_processed": int(len(example_counts)),
        "num_batches": int(len(batch_summaries)),
        "batch_size_used": int(batch_summaries[0].get("batch_size") or 0),
        "rows_per_example_stats": _stats(list(example_counts.values())),
        "source_kind_histogram": _hist(flat["source_kinds"]),
        "semantic_label_histogram": _hist(flat["semantic_labels"]),
        "sdc_control_available_fraction": float(np.asarray(flat["sdc_control_available"], dtype=np.float32).mean()) if flat["sdc_control_available"] else 0.0,
        "semantic_confidence_stats": _stats(flat["semantic_confidence"]),
        "family_size_stats": _stats(flat["family_size"]),
        "family_waypoint_count_stats": _stats(flat["family_waypoint_count"]),
        "family_max_arc_m_stats": _stats(flat["family_max_arc_m"]),
        "family_divergence_onset_min_m_stats": _stats(flat["family_divergence_onset_min_m"]),
    }


def _serialize_metrics(loss: torch.Tensor, loss_stat: Mapping[str, Any]) -> Dict[str, Any]:
    scalar_metrics: Dict[str, float] = {}
    for key, value in loss_stat.items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                scalar_metrics[str(key)] = float(tensor.item())
        elif isinstance(value, (int, float)):
            scalar_metrics[str(key)] = float(value)
    return {"total_loss": float(loss.detach().cpu().item()), "scalar_metrics": scalar_metrics}


def _plot_segmented_polyline(ax, points_xy: np.ndarray, *, label: str | None = None, **kwargs) -> None:
    segments = split_polyline_on_discontinuities(points_xy)
    for idx, segment in enumerate(segments):
        if segment.shape[0] < 2:
            continue
        ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)


def _world_to_sdc_up(points_world_xy: np.ndarray, *, origin_xy_world: np.ndarray, origin_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_world_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = xy - origin_xy_world.reshape(1, 2)
    rot = (math.pi / 2.0) - float(origin_heading_world)
    c = math.cos(rot)
    s = math.sin(rot)
    return np.stack([c * centered[:, 0] - s * centered[:, 1], s * centered[:, 0] + c * centered[:, 1]], axis=-1).astype(np.float32)


def _local_to_world(points_local_xy: np.ndarray, *, origin_xy_world: np.ndarray, origin_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_local_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rot = float(origin_heading_world) - (math.pi / 2.0)
    c = math.cos(rot)
    s = math.sin(rot)
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(origin_xy_world[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(origin_xy_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def _model_to_world(points_model_xy: np.ndarray, *, map_center_world: np.ndarray, map_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_model_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if float(map_heading_world) == 0.0:
        return (xy + np.asarray(map_center_world, dtype=np.float32).reshape(1, 3)[:, :2]).astype(np.float32)
    c = math.cos(float(map_heading_world))
    s = math.sin(float(map_heading_world))
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(map_center_world[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(map_center_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def _plot_world_map(ax, raw_scenario: Mapping[str, Any], *, center_xy: np.ndarray, radius_m: float):
    for feature in dict(raw_scenario.get("map_features", {})).values():
        polyline = np.asarray(dict(feature).get("polyline", []), dtype=np.float32)
        if polyline.ndim != 2 or polyline.shape[1] < 2:
            continue
        xy = polyline[:, :2]
        if not np.isfinite(xy).all():
            continue
        if np.max(np.linalg.norm(xy - center_xy.reshape(1, 2), axis=-1)) > radius_m * 1.7:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=0.7, alpha=0.7)


def _save_world_bev(
    *,
    out_path: Path,
    row: Mapping[str, Any],
    raw_scenario: Mapping[str, Any],
    predicted_world_xy: np.ndarray,
    gt_local_xy: np.ndarray,
):
    current_xy = np.asarray(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["position"][int(row["current_time_index"])][:2], dtype=np.float32)
    current_heading = float(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["heading"][int(row["current_time_index"])])
    family_paths = [np.asarray(path_xy, dtype=np.float32).reshape(-1, 2) for path_xy in list(row.get("candidate_family_resampled_paths_world", []) or [])]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.set_facecolor("#ffffff")
    _plot_world_map(ax, raw_scenario, center_xy=current_xy, radius_m=55.0)
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea"]
    for idx, path_xy in enumerate(family_paths):
        _plot_segmented_polyline(ax, path_xy, color=colors[idx % len(colors)], linewidth=2.2, alpha=0.9, label=f"family path {idx + 1}" if idx < 4 else None)
    if gt_local_xy.shape[0] >= 2:
        gt_world = _local_to_world(gt_local_xy, origin_xy_world=current_xy, origin_heading_world=current_heading)
        _plot_segmented_polyline(ax, gt_world, color="#16a34a", linewidth=1.6, linestyle="--", label="GT rollout")
    if predicted_world_xy.shape[0] >= 2:
        _plot_segmented_polyline(ax, predicted_world_xy, color="#dc2626", linewidth=2.2, label="predicted SDC rollout")
    ax.scatter([current_xy[0]], [current_xy[1]], color="#111827", s=42, label="SDC current pose")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(current_xy[0] - 35.0, current_xy[0] + 35.0)
    ax.set_ylim(current_xy[1] - 15.0, current_xy[1] + 55.0)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(
        f"World Semantic Control BEV\n{row['scenario_id']} | slot={row['selected_slot_id']} | requested={row['requested_semantic_label']}"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_projection_debug(
    *,
    out_path: Path,
    row: Mapping[str, Any],
    predicted_model_xy: np.ndarray,
    projected_model_xy: np.ndarray,
    family_paths_model_xy: List[np.ndarray],
    projected_distance: np.ndarray,
    family_gate: np.ndarray,
):
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), dpi=180)
    axes[0].set_facecolor("#f8fafc")
    for idx, family_path_xy in enumerate(family_paths_model_xy):
        if family_path_xy.shape[0] < 2:
            continue
        _plot_segmented_polyline(
            axes[0],
            family_path_xy,
            color="#cbd5e1",
            linewidth=1.0,
            alpha=0.8,
            label="family paths" if idx == 0 else None,
        )
    if predicted_model_xy.shape[0] > 0:
        axes[0].scatter(predicted_model_xy[:, 0], predicted_model_xy[:, 1], c="#dc2626", s=18, label="predicted points")
    if projected_model_xy.shape[0] > 0:
        axes[0].scatter(projected_model_xy[:, 0], projected_model_xy[:, 1], c="#2563eb", s=18, label="nearest family projections")
        for p, q in zip(predicted_model_xy, projected_model_xy):
            axes[0].plot([p[0], q[0]], [p[1], q[1]], color="#94a3b8", linewidth=0.8, alpha=0.8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title(
        f"Projection Debug\nslot={row['selected_slot_id']} | requested={row['requested_semantic_label']}"
    )
    axes[0].set_xlabel("Model-frame x (m)")
    axes[0].set_ylabel("Model-frame y (m)")
    axes[0].grid(alpha=0.2, linewidth=0.4)
    axes[0].legend(loc="best", fontsize=7)

    steps = np.arange(projected_distance.shape[0], dtype=np.int64)
    axes[1].plot(steps, projected_distance, color="#2563eb", linewidth=2.0, label="weighted family distance")
    axes[1].plot(steps, family_gate, color="#16a34a", linewidth=1.6, label="family guide gate")
    axes[1].set_title("Family Distance And Guide Gate")
    axes[1].set_xlabel("Prediction step")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.2, linewidth=0.4)
    axes[1].legend(loc="best", fontsize=7)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_contact_sheet(*, out_path: Path, image_paths: List[Path]):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=160)
    for ax, path in zip(axes.flat, image_paths):
        image = plt.imread(path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(path.stem, fontsize=8)
    for ax in axes.flat[len(image_paths):]:
        ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _aggregate_loss_records(records: List[Dict[str, Any]], *, weights: List[float]) -> Dict[str, Any]:
    if not records:
        return {"total_loss": 0.0, "scalar_metrics": {}}
    scalar_keys = sorted({key for record in records for key in dict(record.get("scalar_metrics", {})).keys()})
    scalar_metrics: Dict[str, float] = {}
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = np.where(weight_arr > 0.0, weight_arr, 1.0)
    for key in scalar_keys:
        values = np.asarray([float(dict(record.get("scalar_metrics", {})).get(key, 0.0)) for record in records], dtype=np.float64)
        scalar_metrics[key] = float(np.average(values, weights=weight_arr))
    total_loss = np.asarray([float(record.get("total_loss", 0.0)) for record in records], dtype=np.float64)
    return {"total_loss": float(np.average(total_loss, weights=weight_arr)), "scalar_metrics": scalar_metrics}


def _aggregate_eval(records: List[Dict[str, float]], *, weights: List[float]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = np.where(weight_arr > 0.0, weight_arr, 1.0)
    return {
        key: float(np.average(np.asarray([float(record.get(key, 0.0)) for record in records], dtype=np.float64), weights=weight_arr))
        for key in keys
    }


def _row_example_key(row: Mapping[str, Any]) -> str:
    metadata = dict(row.get("metadata", {}) or {})
    example_id = str(metadata.get("example_id") or "").strip()
    if example_id:
        return example_id
    scenario_id = str(row.get("scenario_id") or "").strip()
    sdc_id = str(row.get("sdc_id") or "").strip()
    current_time_index = int(row.get("current_time_index") or 0)
    return f"{scenario_id}__sdc_{sdc_id}__t_{current_time_index:03d}"


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(Path(args.control_index).expanduser())
    if int(args.max_rows) > 0:
        rows = rows[: int(args.max_rows)]
    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode="training")
    num_samples = min(len(dataset), len(rows))
    if num_samples <= 0:
        raise ValueError("No rows available for semantic-only SDC smoke.")

    model, load_report = load_model_from_checkpoint_forgiving(
        config=config,
        ckpt_path=args.ckpt,
        load_mode=str(args.load_mode),
        strict_state_dict=(str(args.load_mode) == "strict_state_dict"),
        map_location="cpu",
    )
    model.eval()
    model._trainer = type("TrainerStub", (), {"world_size": 1, "lr_scheduler_configs": None, "optimizers": None})()

    batch_summaries: List[Dict[str, Any]] = []
    loss_records: List[Dict[str, Any]] = []
    loss_weights: List[float] = []
    eval_records: List[Dict[str, float]] = []
    eval_weights: List[float] = []
    debug_scene_limit = max(0, int(args.debug_examples))
    debug_scene_keys: List[str] = []
    for row in rows[:num_samples]:
        key = _row_example_key(row)
        if key not in debug_scene_keys:
            debug_scene_keys.append(key)
        if len(debug_scene_keys) >= debug_scene_limit:
            break
    debug_scene_key_set = set(debug_scene_keys)
    debug_payloads: List[Dict[str, Any]] = []

    for batch_start in range(0, num_samples, int(args.batch_size)):
        batch_end = min(batch_start + int(args.batch_size), num_samples)
        batch_idx = (batch_start // int(args.batch_size)) + 1
        batch_count = int(math.ceil(float(num_samples) / float(args.batch_size)))
        print(
            f"Processing semantic-only SDC smoke batch {batch_idx}/{batch_count} (rows {batch_start}:{batch_end})",
            flush=True,
        )
        samples = [dataset[idx] for idx in range(batch_start, batch_end)]
        batch = dataset.collate_batch(samples)
        batch_rows = rows[batch_start:batch_end]
        batch_summaries.append(_summarize_batch(batch, batch_rows))

        batch_torch = _to_torch_device(batch, device=torch.device("cpu"))
        with torch.no_grad():
            output = model(copy.deepcopy(batch_torch))
            loss, loss_stat = model.get_loss(output)
            semantic_context = model._extract_sdc_semantic_context(output)
            no_control_input = {k: copy.deepcopy(v) for k, v in batch_torch.items() if not str(k).startswith("cf/")}
            no_control_output = model(copy.deepcopy(no_control_input))
            no_control_pred = model._expected_next_state_from_logits(no_control_output["decoder/output_logit"], no_control_output)
            control_pred = model._expected_next_state_from_logits(output["decoder/output_logit"], output)

        serialized = _serialize_metrics(loss, loss_stat)
        loss_records.append(serialized)
        loss_weights.append(float(serialized["scalar_metrics"].get("num_trained_tokens", len(batch_rows))))

        if semantic_context is not None:
            family_distance = (semantic_context["expected_projection"]["nearest_distance"] * semantic_context["family_weights"][:, None, :]).sum(dim=-1)
            family_gate = semantic_context["family_gate_mean"]
            batch_eval: Dict[str, float] = {
                "semantic_label_match": float(serialized["scalar_metrics"].get("cf/sdc_semantic_acc", 0.0)),
                "family_gate_mean": float(family_gate[semantic_context["sdc_valid_by_t"]].mean().item()) if bool(semantic_context["sdc_valid_by_t"].any()) else 0.0,
                "projected_family_distance_mean_m": float(family_distance[semantic_context["sdc_valid_by_t"]].mean().item()) if bool(semantic_context["sdc_valid_by_t"].any()) else 0.0,
                "family_guide_loss": float(serialized["scalar_metrics"].get("cf/sdc_family_guide_loss", 0.0)),
            }
            decision_agent_mask = semantic_context["decision_agent_mask"][:, None, :, None]
            changed_from_no_control = torch.linalg.norm(
                (control_pred["expected_pos_world"] - no_control_pred["expected_pos_world"]) * decision_agent_mask,
                dim=-1,
            )
            non_sdc_mask = (1.0 - semantic_context["decision_agent_mask"])[:, None, :]
            non_sdc_drift = torch.linalg.norm(
                (control_pred["expected_pos_world"] - no_control_pred["expected_pos_world"]) * non_sdc_mask[:, :, :, None],
                dim=-1,
            )
            batch_eval["changed_from_no_control_sdc_mean_m"] = float(
                changed_from_no_control[semantic_context["sdc_token_mask"]].mean().item()
            ) if bool(semantic_context["sdc_token_mask"].any()) else 0.0
            batch_eval["non_sdc_drift_mean_m"] = float(
                non_sdc_drift[batch_torch["decoder/target_action_valid_mask"]].mean().item()
            ) if bool(batch_torch["decoder/target_action_valid_mask"].any()) else 0.0
            eval_records.append(batch_eval)
            eval_weights.append(float(semantic_context["sdc_valid_by_t"].sum().item() or len(batch_rows)))

        if semantic_context is not None and debug_scene_key_set:
            for local_idx, row in enumerate(batch_rows):
                if _row_example_key(row) not in debug_scene_key_set:
                    continue
                debug_payloads.append(
                    {
                        "row": row,
                        "pred_model": np.asarray(semantic_context["sdc_expected_pos_world"][local_idx].detach().cpu(), dtype=np.float32),
                        "family_paths_model": np.asarray(semantic_context["family_paths_world"][local_idx].detach().cpu(), dtype=np.float32),
                        "family_path_mask": np.asarray(semantic_context["family_path_mask"][local_idx].detach().cpu(), dtype=np.float32),
                        "family_distance": np.asarray(
                            (semantic_context["expected_projection"]["nearest_distance"][local_idx] * semantic_context["family_weights"][local_idx][None, :]).sum(dim=-1).detach().cpu(),
                            dtype=np.float32,
                        ),
                        "family_gate": np.asarray(semantic_context["family_gate_mean"][local_idx].detach().cpu(), dtype=np.float32),
                        "nearest_idx": np.asarray(
                            semantic_context["expected_projection"]["nearest_idx"][local_idx].detach().cpu(),
                            dtype=np.int64,
                        ),
                        "nearest_distance_per_family": np.asarray(
                            semantic_context["expected_projection"]["nearest_distance"][local_idx].detach().cpu(),
                            dtype=np.float32,
                        ),
                    }
                )

    batch_summary = _aggregate_batch_summaries(batch_summaries)
    batch_summary_path = outdir / "sdc_semantic_batch_smoke.json"
    batch_summary_path.write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")

    metrics = _aggregate_loss_records(loss_records, weights=loss_weights)
    metrics["checkpoint_load_report"] = load_report
    metrics["teacher_loaded"] = model.policy_teacher is not None
    if eval_records:
        metrics["sdc_eval"] = _aggregate_eval(eval_records, weights=eval_weights)
    loss_summary_path = outdir / "sdc_semantic_loss_smoke.json"
    loss_summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    debug_records = []
    for batch_idx, payload in enumerate(debug_payloads):
        row = payload["row"]
        raw_scenario = load_raw_scenario_from_row(row)
        current_xy = np.asarray(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["position"][int(row["current_time_index"])][:2], dtype=np.float32)
        current_heading = float(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["heading"][int(row["current_time_index"])])
        pred_model = np.asarray(payload["pred_model"], dtype=np.float32)
        map_center_world = np.asarray(row.get("candidate_family_map_center", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
        map_heading_world = float(row.get("candidate_family_map_heading", 0.0) or 0.0)
        pred_world = _model_to_world(pred_model, map_center_world=map_center_world, map_heading_world=map_heading_world)
        family_paths_tensor = np.asarray(payload["family_paths_model"], dtype=np.float32)
        family_path_mask = np.asarray(payload["family_path_mask"], dtype=np.float32)
        family_paths_model = []
        for path_xy, path_mask in zip(family_paths_tensor, family_path_mask):
            valid = np.asarray(path_mask, dtype=np.float32) > 0.0
            family_paths_model.append(np.asarray(path_xy, dtype=np.float32)[valid])
        projected_model = np.zeros_like(pred_model)
        nearest_idx = np.asarray(payload["nearest_idx"], dtype=np.int64)
        nearest_distance_per_family = np.asarray(payload["nearest_distance_per_family"], dtype=np.float32)
        if family_paths_model and nearest_idx.ndim == 2 and nearest_distance_per_family.ndim == 2:
            nearest_family_idx = np.argmin(nearest_distance_per_family, axis=-1)
            gathered = []
            for step_idx, family_idx in enumerate(nearest_family_idx.tolist()):
                if step_idx >= nearest_idx.shape[0] or family_idx >= nearest_idx.shape[1]:
                    continue
                path_xy = family_paths_model[int(family_idx)]
                if path_xy.shape[0] == 0:
                    continue
                point_idx = int(np.clip(nearest_idx[step_idx, int(family_idx)], 0, max(0, path_xy.shape[0] - 1)))
                gathered.append(path_xy[point_idx])
            if gathered:
                projected_model = np.asarray(gathered, dtype=np.float32)
        gt_world = np.asarray(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["position"][int(row["current_time_index"]):, :2], dtype=np.float32)
        gt_local = _world_to_sdc_up(gt_world, origin_xy_world=current_xy, origin_heading_world=current_heading)

        example_dir = outdir / "debug_examples" / f"{batch_idx:02d}_{row['scenario_id']}__{row['selected_slot_id']}"
        world_path = example_dir / "world_path_control_bev.png"
        projection_path = example_dir / "projection_debug.png"
        contact_sheet_path = example_dir / "semantic_control_contact_sheet.png"
        family_overlay_path = example_dir / "semantic_family_overlay.png"
        family_profile_path = example_dir / "family_separability_profile.png"

        upstream_overlay = Path(str(dict(row.get("debug_artifacts", {}) or {}).get("semantic_family_overlay_png") or ""))
        upstream_profile = Path(str(dict(row.get("debug_artifacts", {}) or {}).get("family_separability_profile_png") or ""))
        if upstream_overlay.is_file():
            family_overlay_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(upstream_overlay, family_overlay_path)
        if upstream_profile.is_file():
            family_profile_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(upstream_profile, family_profile_path)

        _save_world_bev(
            out_path=world_path,
            row=row,
            raw_scenario=raw_scenario,
            predicted_world_xy=pred_world,
            gt_local_xy=gt_local,
        )
        _save_projection_debug(
            out_path=projection_path,
            row=row,
            predicted_model_xy=pred_model,
            projected_model_xy=projected_model,
            family_paths_model_xy=family_paths_model,
            projected_distance=np.asarray(payload["family_distance"], dtype=np.float32),
            family_gate=np.asarray(payload["family_gate"], dtype=np.float32),
        )
        image_paths = [path for path in [family_overlay_path, family_profile_path, world_path, projection_path] if path.exists()]
        _save_contact_sheet(out_path=contact_sheet_path, image_paths=image_paths)
        debug_records.append(
            {
                "example_id": _row_example_key(row),
                "scenario_id": str(row["scenario_id"]),
                "selected_slot_id": str(row["selected_slot_id"]),
                "semantic_family_overlay_png": str(family_overlay_path),
                "family_separability_profile_png": str(family_profile_path),
                "world_path_control_bev_png": str(world_path),
                "projection_debug_png": str(projection_path),
                "semantic_control_contact_sheet_png": str(contact_sheet_path),
            }
        )

    (outdir / "debug_manifest.json").write_text(json.dumps(debug_records, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "sdc_semantic_batch_smoke": str(batch_summary_path),
                "sdc_semantic_loss_smoke": str(loss_summary_path),
                "debug_manifest": str(outdir / "debug_manifest.json"),
                "num_debug_examples": int(len(debug_scene_keys)),
                "num_debug_rows": int(len(debug_records)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
