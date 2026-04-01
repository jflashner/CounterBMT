from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

from bmt.counterfactual import build_counterfactual_dataset_fields, default_counterfactual_dataset_fields, load_control_code
from bmt.dataset.preprocessor import preprocess_scenario_description
from bmt.tokenization import get_tokenizer
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils import utils
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config
from scripts.counterfactual.mine_local_interventions import materialize_candidate_debug_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate path-control adherence over a held-out control index.")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-control-sweep-images", type=int, default=6)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument(
        "--load-mode",
        type=str,
        default="forgiving_state_dict",
        choices=("forgiving_state_dict", "strict_state_dict", "legacy_merge"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    entries = _load_jsonl(Path(args.control_index).expanduser())
    selected_entries = _select_entries(entries, num_examples=int(args.num_examples), seed=int(args.seed))
    summary_path = outdir / "path_control_eval_summary.json"
    examples_path = outdir / "path_control_eval_per_example.jsonl"
    legacy_summary_path = outdir / "eval_smoke_summary.json"
    legacy_examples_path = outdir / "eval_smoke_examples.jsonl"
    confusion_path = outdir / "path_control_eval_confusion_matrix.json"
    breakdown_path = outdir / "path_control_eval_branch_breakdown.json"

    if not args.ckpt or not Path(args.ckpt).expanduser().exists():
        summary = {
            "ran": False,
            "reason": "missing_checkpoint",
            "num_examples_requested": int(args.num_examples),
            "num_examples_selected": len(selected_entries),
            "control_index": str(Path(args.control_index).expanduser()),
        }
        _write_json(summary_path, summary)
        _write_json(legacy_summary_path, summary)
        examples_path.write_text("", encoding="utf-8")
        legacy_examples_path.write_text("", encoding="utf-8")
        _write_json(confusion_path, {"ran": False, "labels": ["left", "straight", "right"], "matrix": {}})
        _write_json(breakdown_path, {"ran": False, "breakdown_by_requested_class": {}})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    config = _load_config(args)
    model, load_report = _load_model(config=config, ckpt_path=args.ckpt, load_mode=args.load_mode)
    tokenizer = get_tokenizer(config)

    rows: List[Dict[str, Any]] = []
    control_sweep_pngs: List[str] = []
    for entry_idx, entry in enumerate(selected_entries):
        evaluation = _evaluate_entry(
            entry,
            config=config,
            model=model,
            tokenizer=tokenizer,
            outdir=outdir,
            control_sweep_index=(entry_idx if entry_idx < int(args.num_control_sweep_images) else None),
        )
        rows.extend(evaluation["rows"])
        if evaluation.get("control_sweep_png"):
            control_sweep_pngs.append(str(evaluation["control_sweep_png"]))

    confusion_matrix, branch_breakdown = _build_branch_breakdown(rows)

    summary = {
        "ran": True,
        "control_index": str(Path(args.control_index).expanduser()),
        "ckpt": str(Path(args.ckpt).expanduser()),
        "load_mode": str(args.load_mode),
        "checkpoint_load_report": load_report,
        "num_examples_requested": int(args.num_examples),
        "num_examples_selected": len(selected_entries),
        "num_rows": len(rows),
        "num_controlled_rows": sum(1 for row in rows if row.get("mode_bucket") != "no_control"),
        "num_control_sweep_pngs": len(control_sweep_pngs),
        "control_sweep_pngs": control_sweep_pngs,
        "requested_branch_match_rate": _fraction(sum(bool(row.get("requested_branch_match")) for row in rows if row.get("mode") != "no_control"), sum(1 for row in rows if row.get("mode") != "no_control")),
        "factual_ade_mean": _mean(row.get("ade_factual") for row in rows if row.get("mode") == "factual"),
        "factual_fde_mean": _mean(row.get("fde_factual") for row in rows if row.get("mode") == "factual"),
        "artifacts": {
            "path_control_eval_summary_json": str(summary_path),
            "path_control_eval_per_example_jsonl": str(examples_path),
            "path_control_eval_confusion_matrix_json": str(confusion_path),
            "path_control_eval_branch_breakdown_json": str(breakdown_path),
        },
    }
    _write_json(summary_path, summary)
    _write_json(legacy_summary_path, summary)
    _write_jsonl(examples_path, rows)
    _write_jsonl(legacy_examples_path, rows)
    _write_json(confusion_path, confusion_matrix)
    _write_json(breakdown_path, branch_breakdown)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / cfg_path).resolve()
        config = cfg_from_yaml_file(cfg_path, config)
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    return config


def _load_model(*, config: Any, ckpt_path: str, load_mode: str):
    from bmt.models.motionlm_lightning import MotionLMLightning

    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs/motion_default.yaml", global_config)
    resolved_ckpt = Path(ckpt_path).expanduser()
    if not resolved_ckpt.is_absolute():
        resolved_ckpt = (REPO_ROOT / resolved_ckpt).resolve()
    if load_mode == "legacy_merge":
        model = utils.load_from_checkpoint(
            checkpoint_path=str(resolved_ckpt),
            cls=MotionLMLightning,
            config=config,
            default_config=default_config,
            strict=False,
            checkpoint_surgery_func=utils.checkpoint_surgery_func,
            map_location="cpu",
        )
        load_report = {
            "ckpt_path": str(resolved_ckpt),
            "load_mode": "legacy_merge",
            "num_ckpt_state_dict_keys": None,
            "num_loaded_keys": None,
            "num_missing_keys": None,
            "num_unexpected_keys": None,
            "num_shape_mismatch_keys": None,
            "strict_state_dict_used": False,
        }
    else:
        model, load_report = load_model_from_checkpoint_forgiving(
            config=config,
            ckpt_path=str(resolved_ckpt),
            load_mode=load_mode,
            strict_state_dict=(load_mode == "strict_state_dict"),
            map_location="cpu",
            checkpoint_surgery_func=utils.checkpoint_surgery_func,
        )
    model.eval()
    return model, load_report


def _evaluate_entry(
    entry: Dict[str, Any],
    *,
    config: Any,
    model: Any,
    tokenizer: Any,
    outdir: Path,
    control_sweep_index: Optional[int] = None,
) -> Dict[str, Any]:
    entry = _ensure_entry_debug_bundle(entry, outdir=outdir, config=config)
    raw_scenario = _load_raw_scenario(entry["scenario_pkl"])
    base_sample = preprocess_scenario_description(
        scenario=raw_scenario,
        config=copy.deepcopy(config),
        in_evaluation=True,
        keep_all_data=False,
        backward_prediction=False,
        tokenizer=tokenizer,
    )
    base_sample["metadata/scenario_id"] = raw_scenario["id"]

    no_control = _run_variant(
        mode_name="no_control",
        sample=base_sample,
        control_code=None,
        model=model,
        tokenizer=tokenizer,
    )
    factual_control = dict(entry) if not str(entry.get("factual_control_code_path", "")).strip() else load_control_code(entry["factual_control_code_path"])
    factual = _run_variant(
        mode_name="factual",
        sample=base_sample,
        control_code=factual_control,
        model=model,
        tokenizer=tokenizer,
    )

    alternative_rows = []
    for alt in _load_json(Path(entry["alternative_control_codes_path"])) if str(entry.get("alternative_control_codes_path", "")).strip() else []:
        alt_control = dict(alt.get("control_code", {}))
        if not alt_control:
            continue
        alternative_rows.append(
            _run_variant(
                mode_name=f"alternative_{alt.get('alternative_rank', 0)}",
                sample=base_sample,
                control_code=alt_control,
                model=model,
                tokenizer=tokenizer,
                alternative_meta=alt,
            )
        )

    branch_candidates = _load_json(Path(entry["train_view_path"]).parent / "branch_candidates.json") if str(entry.get("train_view_path", "")).strip() else {"branch_candidates": []}
    all_rows = [no_control, factual] + alternative_rows
    baseline_positions = no_control["target_positions"]
    for row in all_rows:
        requested_branch_label = row.get("requested_branch_label")
        predicted_branch = _classify_predicted_branch(
            final_pose=row["target_final_pose_world"],
            branch_candidates=branch_candidates.get("branch_candidates", []),
        )
        row.update(
            {
                "scenario_id": str(entry["scenario_id"]),
                "agent_id": str(entry["agent_id"]),
                "decision_time_idx": int(entry["decision_time_idx"]),
                "predicted_branch_label": predicted_branch["branch_label"],
                "requested_branch_label": requested_branch_label,
                "requested_branch_match": bool(predicted_branch["branch_label"] == requested_branch_label) if requested_branch_label is not None else None,
                "branch_score_margin": predicted_branch["score_margin"],
                "final_pose_to_requested_anchor_m": _anchor_distance(row),
                "final_heading_error_to_requested_anchor_rad": _anchor_heading_error(row),
                "changed_from_no_control": _changed_from_baseline(row["target_positions"], baseline_positions),
                "non_target_mean_displacement_vs_no_control": _non_target_displacement(row["non_target_positions"], no_control["non_target_positions"])[0],
                "non_target_max_displacement_vs_no_control": _non_target_displacement(row["non_target_positions"], no_control["non_target_positions"])[1],
            }
        )
        if row["mode"] != "factual":
            row["ade_factual"] = None
            row["fde_factual"] = None

    factual_ade, factual_fde = _ade_fde(
        factual["target_positions"],
        factual["target_valid_mask"],
        factual["gt_target_positions"],
        factual["gt_target_valid_mask"],
    )
    factual["ade_factual"] = factual_ade
    factual["fde_factual"] = factual_fde

    control_sweep_png = None
    if control_sweep_index is not None:
        control_sweep_png = outdir / f"control_sweep_{int(control_sweep_index):03d}.png"
        _write_control_sweep_png(
            entry=entry,
            rows=all_rows,
            branch_candidates=branch_candidates.get("branch_candidates", []),
            output_path=control_sweep_png,
        )

    return {
        "rows": [_prune_row(row, entry=entry, control_sweep_png=control_sweep_png) for row in all_rows],
        "control_sweep_png": (str(control_sweep_png) if control_sweep_png is not None else None),
    }


def _run_variant(
    *,
    mode_name: str,
    sample: Dict[str, Any],
    control_code: Optional[Dict[str, Any]],
    model: Any,
    tokenizer: Any,
    alternative_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample_with_cf = dict(sample)
    decoder_track_names = sample_with_cf.get("decoder/track_name", [])
    horizon = int(sample_with_cf["decoder/agent_position"].shape[0])
    scenario_id = str(sample_with_cf["metadata/scenario_id"])
    if control_code is None:
        cf_fields = default_counterfactual_dataset_fields(
            scenario_id=scenario_id,
            decoder_track_names=decoder_track_names,
            horizon=horizon,
        )
    else:
        cf_fields = build_counterfactual_dataset_fields(
            scenario_id=scenario_id,
            decoder_track_names=decoder_track_names,
            horizon=horizon,
            control_code=control_code,
            control_code_path="",
            require_trainable=False,
        )
    sample_with_cf.update(cf_fields)
    batch = _single_sample_to_batch(sample_with_cf)
    device = torch.device("cpu")
    batch = _to_torch_device(batch, device=device)
    batch["in_evaluation"] = torch.ones((1,), dtype=torch.bool, device=device)

    with torch.no_grad():
        rollout = model.model.autoregressive_rollout(
            batch,
            num_decode_steps=None,
            sampling_method=str(model.config.SAMPLING.get("SAMPLING_METHOD", "topp")),
            temperature=float(model.config.SAMPLING.TEMPERATURE),
            topp=float(model.config.SAMPLING.TOPP),
        )
        rollout = tokenizer.detokenize(rollout, detokenizing_gt=False, backward_prediction=False)

    decision_mask = np.asarray(sample_with_cf["cf/decision_agent_mask"], dtype=np.float32) > 0
    if bool(decision_mask.any()):
        target_slot = int(np.flatnonzero(decision_mask)[0])
    else:
        target_slot = 0
    predicted_position = _to_numpy(rollout["decoder/reconstructed_position"])[0]
    predicted_heading = _to_numpy(rollout["decoder/reconstructed_heading"])[0]
    predicted_valid_mask = _to_numpy(rollout["decoder/reconstructed_valid_mask"])[0].astype(bool)
    gt_position = np.asarray(sample_with_cf["decoder/agent_position"], dtype=np.float32)
    gt_valid_mask = np.asarray(sample_with_cf["decoder/agent_valid_mask"], dtype=bool)

    target_positions = predicted_position[:, target_slot, :2]
    target_valid_mask = predicted_valid_mask[:, target_slot]
    target_heading = predicted_heading[:, target_slot]
    gt_target_positions = gt_position[:, target_slot, :2]
    gt_target_valid_mask = gt_valid_mask[:, target_slot]
    non_target_positions = predicted_position[:, ~decision_mask, :2] if decision_mask.shape[0] == predicted_position.shape[1] else predicted_position[:, 1:, :2]

    requested_branch_label = None
    requested_anchor = None
    if control_code is not None:
        requested_branch_label = str(control_code.get("path_token", {}).get("branch_label"))
        requested_anchor = dict(control_code.get("terminal_anchor", {}))
    return {
        "mode": mode_name,
        "requested_branch_label": requested_branch_label,
        "requested_anchor": requested_anchor,
        "target_positions": target_positions,
        "target_valid_mask": target_valid_mask,
        "target_final_pose_world": _extract_final_pose(target_positions, target_heading, target_valid_mask),
        "non_target_positions": non_target_positions,
        "gt_target_positions": gt_target_positions,
        "gt_target_valid_mask": gt_target_valid_mask,
        "alternative_meta": alternative_meta,
        "control_code": control_code,
    }


def _classify_predicted_branch(*, final_pose: Dict[str, float], branch_candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scored: List[Tuple[float, str]] = []
    for candidate in branch_candidates:
        terminal_pose = candidate.get("terminal_pose") or candidate.get("target_terminal_pose") or {}
        dx = float(final_pose["x"] - float(terminal_pose.get("x", 0.0)))
        dy = float(final_pose["y"] - float(terminal_pose.get("y", 0.0)))
        d_heading = _wrap_angle(float(final_pose["heading"] - float(terminal_pose.get("heading", 0.0))))
        score = math.hypot(dx, dy) + 2.0 * abs(d_heading)
        scored.append((score, str(candidate.get("branch_label"))))
    if not scored:
        return {"branch_label": None, "score_margin": None}
    scored.sort(key=lambda item: item[0])
    best_score, best_label = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    return {
        "branch_label": best_label,
        "score_margin": float((second_score - best_score) if second_score is not None else 0.0),
    }


def _anchor_distance(row: Dict[str, Any]) -> Optional[float]:
    requested_anchor = row.get("requested_anchor")
    control_code = row.get("control_code")
    if not requested_anchor or not control_code:
        return None
    debug = dict(control_code.get("debug", {}))
    pose = dict(debug.get("agent_pose_at_decision", {}))
    if not pose:
        return None
    rel_x, rel_y, _ = _world_pose_to_agent_frame(
        x=float(row["target_final_pose_world"]["x"]),
        y=float(row["target_final_pose_world"]["y"]),
        heading=float(row["target_final_pose_world"]["heading"]),
        agent_x=float(pose.get("x", 0.0)),
        agent_y=float(pose.get("y", 0.0)),
        agent_heading=float(pose.get("heading", 0.0)),
    )
    dx = rel_x - float(requested_anchor.get("target_x_rel", 0.0))
    dy = rel_y - float(requested_anchor.get("target_y_rel", 0.0))
    return float(math.hypot(dx, dy))


def _anchor_heading_error(row: Dict[str, Any]) -> Optional[float]:
    requested_anchor = row.get("requested_anchor")
    control_code = row.get("control_code")
    if not requested_anchor or not control_code:
        return None
    debug = dict(control_code.get("debug", {}))
    pose = dict(debug.get("agent_pose_at_decision", {}))
    if not pose:
        return None
    _, _, rel_heading = _world_pose_to_agent_frame(
        x=float(row["target_final_pose_world"]["x"]),
        y=float(row["target_final_pose_world"]["y"]),
        heading=float(row["target_final_pose_world"]["heading"]),
        agent_x=float(pose.get("x", 0.0)),
        agent_y=float(pose.get("y", 0.0)),
        agent_heading=float(pose.get("heading", 0.0)),
    )
    requested_heading = math.atan2(
        float(requested_anchor.get("target_sin_heading_rel", 0.0)),
        float(requested_anchor.get("target_cos_heading_rel", 1.0)),
    )
    return float(abs(_wrap_angle(rel_heading - requested_heading)))


def _world_pose_to_agent_frame(*, x: float, y: float, heading: float, agent_x: float, agent_y: float, agent_heading: float) -> Tuple[float, float, float]:
    dx = float(x - agent_x)
    dy = float(y - agent_y)
    c = math.cos(float(agent_heading))
    s = math.sin(float(agent_heading))
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    heading_rel = _wrap_angle(float(heading - agent_heading))
    return x_rel, y_rel, heading_rel


def _changed_from_baseline(target_positions: np.ndarray, baseline_positions: np.ndarray, threshold_m: float = 0.1) -> bool:
    if target_positions.shape != baseline_positions.shape:
        return True
    displacement = np.linalg.norm(np.asarray(target_positions) - np.asarray(baseline_positions), axis=-1)
    return bool(np.any(displacement > float(threshold_m)))


def _non_target_displacement(pred: np.ndarray, base: np.ndarray) -> Tuple[float, float]:
    if pred.size == 0 or base.size == 0 or pred.shape != base.shape:
        return 0.0, 0.0
    displacement = np.linalg.norm(np.asarray(pred) - np.asarray(base), axis=-1)
    return float(np.mean(displacement)), float(np.max(displacement))


def _ade_fde(pred: np.ndarray, pred_valid: np.ndarray, gt: np.ndarray, gt_valid: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    pred = np.asarray(pred)
    pred_valid = np.asarray(pred_valid, dtype=bool)
    gt = np.asarray(gt)
    gt_valid = np.asarray(gt_valid, dtype=bool)
    length = min(int(pred.shape[0]), int(gt.shape[0]), int(pred_valid.shape[0]), int(gt_valid.shape[0]))
    if length <= 0:
        return None, None
    pred = pred[:length]
    pred_valid = pred_valid[:length]
    gt = gt[:length]
    gt_valid = gt_valid[:length]
    mask = pred_valid & gt_valid
    if not bool(mask.any()):
        return None, None
    errors = np.linalg.norm(pred[mask] - gt[mask], axis=-1)
    ade = float(np.mean(errors))
    final_idx = int(np.flatnonzero(mask)[-1])
    fde = float(np.linalg.norm(pred[final_idx] - gt[final_idx]))
    return ade, fde


def _extract_final_pose(positions: np.ndarray, headings: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    if bool(np.asarray(valid_mask, dtype=bool).any()):
        idx = int(np.flatnonzero(np.asarray(valid_mask, dtype=bool))[-1])
    else:
        idx = int(positions.shape[0] - 1)
    return {
        "x": float(positions[idx, 0]),
        "y": float(positions[idx, 1]),
        "heading": float(headings[idx]),
    }


def _single_sample_to_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
    object_keys = {
        "raw_scenario_description",
        "original_SD",
        "encoder/track_name",
        "decoder/track_name",
        "eval/track_name",
        "cf/debug_meta",
    }
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in object_keys:
            batch[key] = [value]
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                batch[key] = np.asarray([value])
            else:
                batch[key] = utils.numpy_to_torch(value[None])
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            batch[key] = utils.numpy_to_torch(np.asarray([value]))
        elif isinstance(value, str):
            batch[key] = np.asarray([value])
        else:
            batch[key] = [value]
    return batch


def _to_torch_device(value: Any, *, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_torch_device(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch_device(item, device=device) for item in value)
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_raw_scenario(path: str | Path):
    import pickle

    with Path(path).expanduser().open("rb") as f:
        return pickle.load(f)


def _ensure_entry_debug_bundle(entry: Dict[str, Any], *, outdir: Path, config: Any) -> Dict[str, Any]:
    if str(entry.get("train_view_path", "")).strip() and str(entry.get("alternative_control_codes_path", "")).strip():
        return dict(entry)
    if not str(entry.get("scenario_pkl", "")).strip():
        return dict(entry)
    light_id = str(entry.get("light_id", "")).strip()
    agent_id = str(entry.get("agent_id", "")).strip()
    if not light_id or not agent_id:
        return dict(entry)
    materialized_root = outdir / "materialized_eval_inputs"
    result = materialize_candidate_debug_bundle(
        scenario_pkl=str(entry["scenario_pkl"]),
        light_id=light_id,
        agent_id=agent_id,
        outdir=materialized_root,
        config=config,
        include_pngs=False,
    )
    if result is None:
        return dict(entry)
    merged = dict(entry)
    merged.update(
        {
            "train_view_path": result.get("train_view_path", ""),
            "factual_control_code_path": result.get("factual_control_code_path", ""),
            "alternative_control_codes_path": result.get("alternative_control_codes_path", ""),
        }
    )
    return merged


def _load_json(path: Path):
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _select_entries(entries: List[Dict[str, Any]], *, num_examples: int, seed: int) -> List[Dict[str, Any]]:
    if num_examples <= 0 or len(entries) <= num_examples:
        return list(entries)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(entries), size=int(num_examples), replace=False))
    return [entries[int(idx)] for idx in indices.tolist()]


def _wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def _fraction(num: int, denom: int) -> float:
    return float(num / denom) if denom > 0 else 0.0


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(np.asarray(filtered, dtype=np.float32)))


def _prune_row(row: Dict[str, Any], *, entry: Dict[str, Any], control_sweep_png: Optional[Path]) -> Dict[str, Any]:
    return {
        "example_id": str(entry.get("example_id", "")),
        "scenario_id": row["scenario_id"],
        "agent_id": row["agent_id"],
        "light_id": str(row.get("light_id", entry.get("light_id", ""))),
        "decision_time_idx": row["decision_time_idx"],
        "mode": row["mode"],
        "mode_bucket": _mode_bucket(str(row["mode"])),
        "predicted_branch_label": row["predicted_branch_label"],
        "requested_branch_label": row["requested_branch_label"],
        "requested_branch_match": row["requested_branch_match"],
        "branch_score_margin": row["branch_score_margin"],
        "final_pose_to_requested_anchor_m": row["final_pose_to_requested_anchor_m"],
        "final_heading_error_to_requested_anchor_rad": row["final_heading_error_to_requested_anchor_rad"],
        "ade_factual": row.get("ade_factual"),
        "fde_factual": row.get("fde_factual"),
        "changed_from_no_control": row["changed_from_no_control"],
        "non_target_mean_displacement_vs_no_control": row["non_target_mean_displacement_vs_no_control"],
        "non_target_max_displacement_vs_no_control": row["non_target_max_displacement_vs_no_control"],
        "control_sweep_png": (str(control_sweep_png) if control_sweep_png is not None else None),
    }


def _mode_bucket(mode: str) -> str:
    if mode == "no_control":
        return "no_control"
    if mode == "factual":
        return "factual"
    if mode.startswith("alternative_"):
        return "alternative"
    return str(mode)


def _build_branch_breakdown(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requested_labels = ["left", "straight", "right"]
    predicted_labels = ["left", "straight", "right", "none", "other"]
    confusion_counts = {
        requested_label: {predicted_label: 0 for predicted_label in predicted_labels}
        for requested_label in requested_labels
    }
    breakdown: Dict[str, Dict[str, Any]] = {}

    for label in requested_labels:
        label_rows = [
            row for row in rows
            if row.get("mode_bucket") != "no_control" and str(row.get("requested_branch_label")) == label
        ]
        factual_rows = [row for row in label_rows if row.get("mode_bucket") == "factual"]
        alternative_rows = [row for row in label_rows if row.get("mode_bucket") == "alternative"]
        for row in label_rows:
            predicted_label = _normalize_predicted_label(row.get("predicted_branch_label"))
            confusion_counts[label][predicted_label] += 1
        breakdown[label] = {
            "num_rows": len(label_rows),
            "num_matches": sum(bool(row.get("requested_branch_match")) for row in label_rows),
            "match_rate": _fraction(
                sum(bool(row.get("requested_branch_match")) for row in label_rows),
                len(label_rows),
            ),
            "factual_rows": len(factual_rows),
            "factual_match_rate": _fraction(
                sum(bool(row.get("requested_branch_match")) for row in factual_rows),
                len(factual_rows),
            ),
            "alternative_rows": len(alternative_rows),
            "alternative_match_rate": _fraction(
                sum(bool(row.get("requested_branch_match")) for row in alternative_rows),
                len(alternative_rows),
            ),
            "mean_branch_score_margin": _mean(row.get("branch_score_margin") for row in label_rows),
            "mean_final_pose_to_requested_anchor_m": _mean(
                row.get("final_pose_to_requested_anchor_m") for row in label_rows
            ),
            "mean_final_heading_error_to_requested_anchor_rad": _mean(
                row.get("final_heading_error_to_requested_anchor_rad") for row in label_rows
            ),
            "mean_non_target_mean_displacement_vs_no_control": _mean(
                row.get("non_target_mean_displacement_vs_no_control") for row in label_rows
            ),
        }

    confusion_matrix = {
        "labels": requested_labels,
        "predicted_labels": predicted_labels,
        "matrix": confusion_counts,
    }
    branch_breakdown = {
        "requested_labels": requested_labels,
        "breakdown_by_requested_class": breakdown,
    }
    return confusion_matrix, branch_breakdown


def _normalize_predicted_label(value: Any) -> str:
    if value is None:
        return "none"
    label = str(value)
    if label in {"left", "straight", "right"}:
        return label
    if not label or label.lower() == "none":
        return "none"
    return "other"


def _write_control_sweep_png(
    *,
    entry: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    branch_candidates: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    branch_colors = {"left": "#d55e00", "straight": "#0072b2", "right": "#009e73", "u_turn": "#cc79a7"}
    mode_colors = {
        "no_control": "#444444",
        "factual": "#e69f00",
        "alternative": "#56b4e9",
    }

    for candidate in branch_candidates:
        polyline = np.asarray(candidate.get("polyline_xy", []), dtype=np.float32)
        if polyline.ndim == 2 and polyline.shape[0] >= 2 and polyline.shape[1] >= 2:
            color = branch_colors.get(str(candidate.get("branch_label")), "#bbbbbb")
            ax.plot(polyline[:, 0], polyline[:, 1], linestyle="--", linewidth=1.25, color=color, alpha=0.35)

    gt_drawn = False
    for row in rows:
        mode_bucket = _mode_bucket(str(row.get("mode")))
        positions = np.asarray(row.get("target_positions", []), dtype=np.float32)
        valid_mask = np.asarray(row.get("target_valid_mask", []), dtype=bool)
        if positions.ndim != 2 or positions.shape[0] == 0:
            continue
        if valid_mask.shape[0] == positions.shape[0] and bool(valid_mask.any()):
            positions = positions[valid_mask]
        color = mode_colors.get(mode_bucket, "#666666")
        label = str(row.get("mode"))
        requested_branch = row.get("requested_branch_label")
        if requested_branch is not None:
            label = f"{label}:{requested_branch}"
        ax.plot(positions[:, 0], positions[:, 1], linewidth=2.2, color=color, label=label)
        ax.scatter([positions[0, 0]], [positions[0, 1]], color=color, s=16, alpha=0.8)
        ax.scatter([positions[-1, 0]], [positions[-1, 1]], color=color, s=28, alpha=0.95, marker="o")

        if not gt_drawn:
            gt_positions = np.asarray(row.get("gt_target_positions", []), dtype=np.float32)
            gt_valid_mask = np.asarray(row.get("gt_target_valid_mask", []), dtype=bool)
            if gt_positions.ndim == 2 and gt_positions.shape[0] > 0:
                if gt_valid_mask.shape[0] == gt_positions.shape[0] and bool(gt_valid_mask.any()):
                    gt_positions = gt_positions[gt_valid_mask]
                ax.plot(gt_positions[:, 0], gt_positions[:, 1], linewidth=2.0, color="#111111", linestyle=":", label="ground_truth")
                gt_drawn = True

    title = (
        f"{entry.get('scenario_id', '')} | agent {entry.get('agent_id', '')} | "
        f"t={entry.get('decision_time_idx', '')}"
    )
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
        ax.legend(dedup.values(), dedup.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
