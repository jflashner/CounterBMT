from __future__ import annotations

import argparse
import copy
import json
import logging
import math
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

from bmt.counterfactual.sdc_path_control import (
    build_local_competing_paths,
    build_local_selected_path,
    load_raw_scenario_from_row,
    split_polyline_on_discontinuities,
)
from bmt.dataset.dataset import InfgenDataset
from bmt.models.motionlm_lightning import MotionLMLightning
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local PR10 SDC-path dataset/model/loss smokes.")
    parser.add_argument("--config", type=str, default="src/Adv-BMT/cfgs/motion_forward_sdc_path_control_strict_local.yaml")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--debug-examples", type=int, default=2)
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
    config.DATA.COUNTERFACTUAL_MODE = "sdc_path"
    config.DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER = True
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    config.MODEL.LOCAL_CONTROL_USE_PATH = True
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
    waypoints = np.asarray(batch["cf/sdc_path_waypoints"], dtype=np.float32)
    waypoint_mask = np.asarray(batch["cf/sdc_path_waypoint_mask"], dtype=np.float32)
    separability = np.asarray(batch["cf/sdc_path_separability"], dtype=np.float32)
    scenario_ids = [str(v) for v in np.asarray(batch["metadata/scenario_id"]).reshape(-1).tolist()]
    return {
        "batch_size": int(len(rows)),
        "example_ids": [str(dict(row.get("metadata", {}) or {}).get("example_id") or "") for row in rows],
        "scenario_ids": scenario_ids,
        "selected_path_ids": [str(row.get("selected_path_id")) for row in rows],
        "source_kinds": [str(row.get("source_kind")) for row in rows],
        "semantic_labels": [_label_from_id(v) for v in semantic_ids.tolist()],
        "semantic_confidence": [float(v) for v in confidence.tolist()],
        "sdc_is_factual": [bool(v) for v in factual.tolist()],
        "sdc_control_available": [bool(v) for v in available.tolist()],
        "candidate_count": [int(row.get("candidate_count") or 0) for row in rows],
        "path_waypoint_count": [int(mask.sum()) for mask in waypoint_mask],
        "path_length_m": [float(np.max(wp[:, 4][mask > 0])) if np.any(mask > 0) else 0.0 for wp, mask in zip(waypoints, waypoint_mask)],
        "path_separability_min": [float(separability[idx][waypoint_mask[idx] > 0].min()) if np.any(waypoint_mask[idx] > 0) else 0.0 for idx in range(len(rows))],
        "path_separability_mean": [float(separability[idx][waypoint_mask[idx] > 0].mean()) if np.any(waypoint_mask[idx] > 0) else 0.0 for idx in range(len(rows))],
        "path_separability_max": [float(separability[idx][waypoint_mask[idx] > 0].max()) if np.any(waypoint_mask[idx] > 0) else 0.0 for idx in range(len(rows))],
    }


def _aggregate_batch_summaries(batch_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch_summaries:
        return {
            "num_rows_processed": 0,
            "num_examples_processed": 0,
            "num_batches": 0,
            "batch_size_used": 0,
        }

    flat: Dict[str, List[Any]] = {
        "example_ids": [],
        "scenario_ids": [],
        "selected_path_ids": [],
        "source_kinds": [],
        "semantic_labels": [],
        "semantic_confidence": [],
        "sdc_is_factual": [],
        "sdc_control_available": [],
        "candidate_count": [],
        "path_waypoint_count": [],
        "path_length_m": [],
        "path_separability_min": [],
        "path_separability_mean": [],
        "path_separability_max": [],
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
        return {
            "min": float(arr.min()),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }

    def _hist(values: List[Any]) -> Dict[str, int]:
        hist: Dict[str, int] = {}
        for value in values:
            key = str(value)
            hist[key] = int(hist.get(key, 0) + 1)
        return hist

    preview = []
    for idx in range(min(8, len(flat["example_ids"]))):
        preview.append(
            {
                "example_id": str(flat["example_ids"][idx]),
                "selected_path_id": str(flat["selected_path_ids"][idx]),
                "source_kind": str(flat["source_kinds"][idx]),
                "semantic_label": str(flat["semantic_labels"][idx]),
            }
        )

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
        "candidate_count_stats": _stats(flat["candidate_count"]),
        "path_waypoint_count_stats": _stats(flat["path_waypoint_count"]),
        "path_length_m_stats": _stats(flat["path_length_m"]),
        "path_separability_mean_stats": _stats(flat["path_separability_mean"]),
        "preview_rows": preview,
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
    return {
        "total_loss": float(loss.detach().cpu().item()),
        "scalar_metrics": scalar_metrics,
    }


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


def _plot_segmented_polyline(ax, points_xy: np.ndarray, *, label: str | None = None, **kwargs) -> None:
    segments = split_polyline_on_discontinuities(points_xy)
    for idx, segment in enumerate(segments):
        ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)


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


def _save_local_bev(
    *,
    out_path: Path,
    row: Mapping[str, Any],
    competing_paths: Mapping[str, Any],
    predicted_local_xy: np.ndarray,
    gt_local_xy: np.ndarray,
):
    selected_xy = np.asarray(row["selected_path_waypoints_local_xy"], dtype=np.float32)
    separability = np.asarray(row["selected_path_separability"], dtype=np.float32).reshape(-1)
    metadata = dict(row.get("metadata", {}) or {})
    example_id = str(metadata.get("example_id") or row.get("scenario_id") or "")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.set_facecolor("#f8fafc")
    for idx, path in enumerate(competing_paths.values()):
        xy = np.asarray(path.waypoints_xy, dtype=np.float32)
        if xy.shape[0] >= 2:
            _plot_segmented_polyline(
                ax,
                xy,
                color="#cbd5e1",
                linewidth=1.0,
                alpha=0.9,
                label="competing on-route paths" if idx == 0 else None,
            )
    if selected_xy.shape[0] >= 2:
        for idx in range(1, int(selected_xy.shape[0])):
            color = plt.cm.viridis(float(np.clip(separability[min(idx, separability.shape[0] - 1)], 0.0, 1.0)))
            ax.plot(selected_xy[idx - 1:idx + 1, 0], selected_xy[idx - 1:idx + 1, 1], color=color, linewidth=2.8)
        ax.plot([], [], color=plt.cm.viridis(0.85), linewidth=2.8, label="selected path")
    if gt_local_xy.shape[0] >= 2:
        _plot_segmented_polyline(ax, gt_local_xy, color="#16a34a", linewidth=1.8, alpha=0.9, linestyle="--", label="GT rollout")
    if predicted_local_xy.shape[0] >= 2:
        _plot_segmented_polyline(ax, predicted_local_xy, color="#dc2626", linewidth=2.2, alpha=0.95, label="predicted SDC rollout")
        ax.scatter(predicted_local_xy[:, 0], predicted_local_xy[:, 1], c="#dc2626", s=10)
    ax.scatter([0.0], [0.0], color="#111827", s=42, label="SDC current pose")
    ax.arrow(0.0, 0.0, 0.0, 6.5, width=0.14, head_width=0.9, head_length=1.3, color="#111827", length_includes_head=True)
    ax.set_title(
        f"Local SDC Path Control BEV\n{example_id} | slot={metadata.get('slot_id', row.get('selected_path_id'))} | "
        f"{row['source_kind']} | semantic={row['semantic_label']}"
    )
    ax.set_xlabel("Local x (m)")
    ax.set_ylabel("Local y (m, forward)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_world_bev(
    *,
    out_path: Path,
    raw_scenario: Mapping[str, Any],
    row: Mapping[str, Any],
    predicted_local_xy: np.ndarray,
    gt_local_xy: np.ndarray,
):
    current_xy = np.asarray(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["position"][int(row["current_time_index"])][:2], dtype=np.float32)
    current_heading = float(raw_scenario["tracks"][str(row["sdc_id"])]["state"]["heading"][int(row["current_time_index"])])
    metadata = dict(row.get("metadata", {}) or {})
    example_id = str(metadata.get("example_id") or row.get("scenario_id") or "")
    selected_world = _local_to_world(np.asarray(row["selected_path_waypoints_local_xy"], dtype=np.float32), origin_xy_world=current_xy, origin_heading_world=current_heading)
    predicted_world = _local_to_world(predicted_local_xy, origin_xy_world=current_xy, origin_heading_world=current_heading)
    gt_world = _local_to_world(gt_local_xy, origin_xy_world=current_xy, origin_heading_world=current_heading)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.set_facecolor("#ffffff")
    _plot_world_map(ax, raw_scenario, center_xy=current_xy, radius_m=55.0)
    if selected_world.shape[0] >= 2:
        _plot_segmented_polyline(ax, selected_world, color="#2563eb", linewidth=2.4, label="selected path")
    if gt_world.shape[0] >= 2:
        _plot_segmented_polyline(ax, gt_world, color="#16a34a", linewidth=1.8, linestyle="--", label="GT rollout")
    if predicted_world.shape[0] >= 2:
        _plot_segmented_polyline(ax, predicted_world, color="#dc2626", linewidth=2.2, label="predicted SDC rollout")
        ax.scatter(predicted_world[:, 0], predicted_world[:, 1], c="#dc2626", s=10)
    ax.scatter([current_xy[0]], [current_xy[1]], color="#111827", s=42, label="SDC current pose")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(current_xy[0] - 35.0, current_xy[0] + 35.0)
    ax.set_ylim(current_xy[1] - 15.0, current_xy[1] + 55.0)
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title(
        f"World SDC Path Control BEV\n{example_id} | slot={metadata.get('slot_id', row.get('selected_path_id'))} | "
        f"{row['source_kind']} | semantic={row['semantic_label']}"
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_projection_debug(
    *,
    out_path: Path,
    predicted_local_xy: np.ndarray,
    projected_local_xy: np.ndarray,
    projected_arc: np.ndarray,
    separability: np.ndarray,
):
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), dpi=180)
    axes[0].set_facecolor("#f8fafc")
    if predicted_local_xy.shape[0] > 0:
        axes[0].scatter(predicted_local_xy[:, 0], predicted_local_xy[:, 1], c="#dc2626", s=18, label="predicted points")
    if projected_local_xy.shape[0] > 0:
        axes[0].scatter(projected_local_xy[:, 0], projected_local_xy[:, 1], c="#2563eb", s=18, label="nearest path projections")
        for p, q in zip(predicted_local_xy, projected_local_xy):
            axes[0].plot([p[0], q[0]], [p[1], q[1]], color="#94a3b8", linewidth=0.8, alpha=0.8)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_title("Projection Debug")
    axes[0].set_xlabel("Local x (m)")
    axes[0].set_ylabel("Local y (m, forward)")
    axes[0].grid(alpha=0.2, linewidth=0.4)
    axes[0].legend(loc="best", fontsize=7)
    steps = np.arange(projected_arc.shape[0], dtype=np.int64)
    axes[1].plot(steps, projected_arc, color="#2563eb", linewidth=2.0, label="projected arc")
    axes[1].plot(steps, separability, color="#16a34a", linewidth=1.6, label="separability")
    axes[1].set_title("Projected Progress And Separability")
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
    return {
        "total_loss": float(np.average(total_loss, weights=weight_arr)),
        "scalar_metrics": scalar_metrics,
    }


def _aggregate_sdc_eval(records: List[Dict[str, float]], *, weights: List[float]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    weight_arr = np.asarray(weights, dtype=np.float64)
    weight_arr = np.where(weight_arr > 0.0, weight_arr, 1.0)
    out: Dict[str, float] = {}
    for key in keys:
        values = np.asarray([float(record.get(key, 0.0)) for record in records], dtype=np.float64)
        out[key] = float(np.average(values, weights=weight_arr))
    return out


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
        raise ValueError("No rows available for SDC-path smoke.")

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
    sdc_eval_records: List[Dict[str, float]] = []
    sdc_eval_weights: List[float] = []

    debug_rows = min(int(args.debug_examples), num_samples)
    debug_records = []
    debug_payloads: List[Dict[str, Any]] = []

    for batch_start in range(0, num_samples, int(args.batch_size)):
        batch_end = min(batch_start + int(args.batch_size), num_samples)
        batch_idx = (batch_start // int(args.batch_size)) + 1
        batch_count = int(math.ceil(float(num_samples) / float(args.batch_size)))
        print(
            f"Processing SDC-path smoke batch {batch_idx}/{batch_count} "
            f"(rows {batch_start}:{batch_end})",
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
            sdc_path_context = model._extract_sdc_path_context(output)
            no_control_input = {k: copy.deepcopy(v) for k, v in batch_torch.items() if not str(k).startswith("cf/")}
            no_control_output = model(copy.deepcopy(no_control_input))
            no_control_pred = model._expected_next_state_from_logits(no_control_output["decoder/output_logit"], no_control_output)
            control_pred = model._expected_next_state_from_logits(output["decoder/output_logit"], output)

        serialized = _serialize_metrics(loss, loss_stat)
        loss_records.append(serialized)
        loss_weights.append(float(serialized["scalar_metrics"].get("num_trained_tokens", len(batch_rows))))

        if sdc_path_context is not None:
            valid = sdc_path_context["sdc_valid_by_t"]
            proj = sdc_path_context["projection"]
            distance = proj["nearest_distance"][valid]
            arc = proj["nearest_arc"]
            sep = proj["nearest_separability"][valid]
            batch_eval: Dict[str, float] = {
                "semantic_label_match": float(serialized["scalar_metrics"].get("cf/sdc_semantic_acc", 0.0)),
            }
            if distance.numel() > 0:
                weighted_distance = (distance * sep).sum() / sep.sum().clamp_min(1e-4)
                batch_eval.update(
                    {
                        "nearest_selected_path_distance_mean_m": float(distance.mean().item()),
                        "nearest_selected_path_distance_max_m": float(distance.max().item()),
                        "separability_weighted_path_distance_mean_m": float(weighted_distance.item()),
                        "separability_weighted_path_adherence_score": float(torch.exp(-weighted_distance).item()),
                        "projected_progress_final_m": float(arc[valid].max().item()),
                        "projected_progress_backward_fraction": float((arc[:, 1:] < arc[:, :-1]).float()[valid[:, 1:] & valid[:, :-1]].mean().item()) if bool((valid[:, 1:] & valid[:, :-1]).any()) else 0.0,
                    }
                )

            decision_agent_mask = sdc_path_context["decision_agent_mask"][:, None, :, None]
            changed_from_no_control = torch.linalg.norm(
                (control_pred["expected_pos_world"] - no_control_pred["expected_pos_world"]) * decision_agent_mask,
                dim=-1,
            )
            non_sdc_mask = (1.0 - sdc_path_context["decision_agent_mask"])[:, None, :]
            non_sdc_drift = torch.linalg.norm(
                (control_pred["expected_pos_world"] - no_control_pred["expected_pos_world"]) * non_sdc_mask[:, :, :, None],
                dim=-1,
            )
            batch_eval["changed_from_no_control_sdc_mean_m"] = float(
                changed_from_no_control[sdc_path_context["sdc_token_mask"]].mean().item()
            ) if bool(sdc_path_context["sdc_token_mask"].any()) else 0.0
            batch_eval["non_sdc_drift_mean_m"] = float(
                non_sdc_drift[batch_torch["decoder/target_action_valid_mask"]].mean().item()
            ) if bool(batch_torch["decoder/target_action_valid_mask"].any()) else 0.0
            sdc_eval_records.append(batch_eval)
            sdc_eval_weights.append(float(sdc_path_context["sdc_valid_by_t"].sum().item() or len(batch_rows)))

        while len(debug_payloads) < debug_rows and len(debug_payloads) < batch_end:
            local_idx = len(debug_payloads) - batch_start
            if local_idx < 0 or local_idx >= len(batch_rows):
                break
            debug_payloads.append(
                {
                    "row": batch_rows[local_idx],
                    "pred_local": np.asarray(sdc_path_context["sdc_expected_pos_local"][local_idx].detach().cpu(), dtype=np.float32) if sdc_path_context is not None else np.zeros((0, 2), dtype=np.float32),
                    "proj_idx": np.asarray(sdc_path_context["projection"]["nearest_idx"][local_idx].detach().cpu(), dtype=np.int64) if sdc_path_context is not None else np.zeros((0,), dtype=np.int64),
                    "projected_arc": np.asarray(sdc_path_context["projection"]["nearest_arc"][local_idx].detach().cpu(), dtype=np.float32) if sdc_path_context is not None else np.zeros((0,), dtype=np.float32),
                    "projected_sep": np.asarray(sdc_path_context["projection"]["nearest_separability"][local_idx].detach().cpu(), dtype=np.float32) if sdc_path_context is not None else np.zeros((0,), dtype=np.float32),
                }
            )

    batch_summary = _aggregate_batch_summaries(batch_summaries)
    batch_summary_path = outdir / "sdc_path_batch_smoke.json"
    batch_summary_path.write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")

    metrics = _aggregate_loss_records(loss_records, weights=loss_weights)
    metrics["checkpoint_load_report"] = load_report
    metrics["teacher_loaded"] = model.policy_teacher is not None
    if sdc_eval_records:
        metrics["sdc_eval"] = _aggregate_sdc_eval(sdc_eval_records, weights=sdc_eval_weights)

    loss_summary_path = outdir / "sdc_path_loss_smoke.json"
    loss_summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    for batch_idx, payload in enumerate(debug_payloads):
        row = payload["row"]
        raw_scenario = load_raw_scenario_from_row(row)
        selected_local = np.asarray(row["selected_path_waypoints_local_xy"], dtype=np.float32)
        gt_local_path = build_local_selected_path(
            raw_scenario=raw_scenario,
            sdc_id=str(row["sdc_id"]),
            current_time_index=int(row["current_time_index"]),
            source_kind="factual_gt",
            selected_path_id=None,
        )
        competing = build_local_competing_paths(
            raw_scenario=raw_scenario,
            sdc_id=str(row["sdc_id"]),
            current_time_index=int(row["current_time_index"]),
            selected_path_id=None if str(row["selected_path_id"]) == "gt" else str(row["selected_path_id"]),
        )
        pred_local = np.asarray(payload["pred_local"], dtype=np.float32)
        proj_idx = np.asarray(payload["proj_idx"], dtype=np.int64)
        projected_local = selected_local[np.clip(proj_idx, 0, max(0, selected_local.shape[0] - 1))] if selected_local.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)
        projected_arc = np.asarray(payload["projected_arc"], dtype=np.float32)
        projected_sep = np.asarray(payload["projected_sep"], dtype=np.float32)

        example_dir = outdir / "debug_examples" / f"{batch_idx:02d}_{row['scenario_id']}__{row['selected_path_id']}"
        local_path = example_dir / "local_path_control_bev.png"
        world_path = example_dir / "world_path_control_bev.png"
        separability_path = example_dir / "separability_profile_plot.png"
        projection_path = example_dir / "projection_debug_plot.png"
        contact_sheet_path = example_dir / "contact_sheet.png"

        _save_local_bev(
            out_path=local_path,
            row=row,
            competing_paths=competing,
            predicted_local_xy=pred_local,
            gt_local_xy=np.asarray(gt_local_path.waypoints_xy, dtype=np.float32),
        )
        _save_world_bev(
            out_path=world_path,
            raw_scenario=raw_scenario,
            row=row,
            predicted_local_xy=pred_local,
            gt_local_xy=np.asarray(gt_local_path.waypoints_xy, dtype=np.float32),
        )
        fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
        arc = np.asarray(row["selected_path_arc_lengths_m"], dtype=np.float32)
        sep = np.asarray(row["selected_path_separability"], dtype=np.float32)
        ax.plot(arc, sep, color="#2563eb", linewidth=2.0)
        ax.fill_between(arc, 0.0, sep, color="#93c5fd", alpha=0.35)
        ax.set_ylim(-0.02, 1.02)
        metadata = dict(row.get("metadata", {}) or {})
        ax.set_title(
            f"Separability Profile\n{metadata.get('example_id', row['scenario_id'])} | "
            f"slot={metadata.get('slot_id', row['selected_path_id'])}"
        )
        ax.set_xlabel("Arc length (m)")
        ax.set_ylabel("Separability")
        ax.grid(alpha=0.2, linewidth=0.4)
        fig.savefig(separability_path, dpi=180)
        plt.close(fig)
        _save_projection_debug(
            out_path=projection_path,
            predicted_local_xy=pred_local,
            projected_local_xy=projected_local,
            projected_arc=projected_arc,
            separability=projected_sep,
        )
        _save_contact_sheet(
            out_path=contact_sheet_path,
            image_paths=[local_path, world_path, separability_path, projection_path],
        )
        debug_records.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "example_id": str(dict(row.get("metadata", {}) or {}).get("example_id") or ""),
                "selected_path_id": str(row["selected_path_id"]),
                "local_path_control_bev_png": str(local_path),
                "world_path_control_bev_png": str(world_path),
                "separability_profile_plot_png": str(separability_path),
                "projection_debug_plot_png": str(projection_path),
                "contact_sheet_png": str(contact_sheet_path),
            }
        )

    (outdir / "debug_manifest.json").write_text(json.dumps(debug_records, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(
        {
            "sdc_path_batch_smoke": str(batch_summary_path),
            "sdc_path_loss_smoke": str(loss_summary_path),
            "debug_manifest": str(outdir / "debug_manifest.json"),
            "num_debug_examples": int(len(debug_records)),
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
