from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Dict

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

from bmt.counterfactual import decode_decision_agent_mask, decode_path_token_tensor
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 1-step curated path-control training dry-run.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--train-control-index", type=str, required=True)
    parser.add_argument("--val-control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--max-steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "train_dryrun_summary.json"
    metrics_path = outdir / "train_dryrun_metrics.json"
    batch_summary_path = outdir / "train_dryrun_batch_summary.json"
    stdout_path = outdir / "train_dryrun_stdout.log"
    stderr_path = outdir / "train_dryrun_stderr.log"

    summary: Dict[str, Any] = {
        "checkpoint_warm_start_succeeded": False,
        "manual_forward_completed": False,
        "completed_forward_and_backward": False,
        "completed_forward_only": False,
        "failure_reason": None,
        "load_mode": str(args.load_mode),
    }
    metrics: Dict[str, Any] = {}

    try:
        config = _load_config(args)
        dataset = InfgenDataset(config=config, mode="training")
        samples = [dataset[idx] for idx in range(int(args.batch_size))]
        batch = dataset.collate_batch(samples)
        batch_summary = _summarize_batch(batch)
        batch_summary_path.write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")

        model, load_report = load_model_from_checkpoint_forgiving(
            config=config,
            ckpt_path=args.ckpt,
            load_mode=str(args.load_mode),
            strict_state_dict=(str(args.load_mode) == "strict_state_dict"),
            map_location="cpu",
        )
        summary["checkpoint_warm_start_succeeded"] = bool(load_report.get("num_loaded_keys", 0) > 0)
        summary["checkpoint_load_report"] = load_report

        model.eval()
        model._trainer = types.SimpleNamespace(world_size=1, lr_scheduler_configs=None, optimizers=None)
        batch_torch = _to_torch_device(batch, device=torch.device("cpu"))
        with torch.no_grad():
            output = model(copy.deepcopy(batch_torch))
            loss, loss_stat = model.get_loss(output)
        summary["manual_forward_completed"] = True
        metrics = _serialize_loss_metrics(loss, loss_stat)
        metrics["path_loss_active"] = bool(batch_summary["path_active_mask_sum"] > 0)
        metrics["anchor_loss_active"] = bool(batch_summary["anchor_active_mask_sum"] > 0)
        metrics["all_reported_losses_finite"] = bool(
            metrics.get("finite_total_loss", False) and all(metrics.get("finite_scalar_metrics", {}).values())
        )
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

        command = [
            sys.executable,
            str(REPO_ROOT / "bmt" / "train_motion.py"),
            "--config-name",
            "motion_forward_path_control_strict_local.yaml",
            f"DATA.TRAINING_DATA_DIR={str(Path(args.data_dir).expanduser())}",
            f"DATA.TEST_DATA_DIR={str(Path(args.data_dir).expanduser())}",
            f"DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN={str(Path(args.train_control_index).expanduser())}",
            f"DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL={str(Path(args.val_control_index).expanduser())}",
            f"batch_size={int(args.batch_size)}",
            f"val_batch_size={int(args.batch_size)}",
            f"num_workers={int(args.num_workers)}",
            f"val_num_workers={int(args.num_workers)}",
            f"pretrain={str(Path(args.ckpt).expanduser())}",
            f"log_dir={str((outdir / 'trainer_logs').resolve())}",
            f"seed=0",
            f"+max_steps={int(args.max_steps)}",
            "limit_train_batches=1",
            "limit_val_batches=1",
            "+val_interval=1",
            "num_sanity_val_steps=0",
            "wandb=false",
            f"CKPT_LOAD_MODE={str(args.load_mode)}",
        ]
        env = os.environ.copy()
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT.parents[1]),
            env=env,
            capture_output=True,
            text=True,
        )
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        summary["canonical_train_command"] = command
        summary["canonical_entrypoint_returncode"] = int(process.returncode)
        summary["stdout_log_path"] = str(stdout_path)
        summary["stderr_log_path"] = str(stderr_path)
        train_step_completed = _train_step_completed_from_logs(process.stdout)
        summary["train_step_completed_before_failure"] = bool(train_step_completed)
        if process.returncode == 0:
            summary["completed_forward_and_backward"] = True
        else:
            summary["completed_forward_and_backward"] = bool(train_step_completed)
            summary["completed_forward_only"] = bool(summary["manual_forward_completed"] and not train_step_completed)
            summary["failure_reason"] = _tail_lines(process.stderr or process.stdout, max_lines=40)
    except Exception as exc:
        summary["completed_forward_only"] = bool(summary["manual_forward_completed"])
        summary["failure_reason"] = {
            "message": str(exc),
            "traceback_summary": traceback.format_exc().splitlines()[-20:],
        }
        if metrics:
            metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
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
    config.DATA.TRAINING_DATA_DIR = str(Path(args.data_dir).expanduser())
    config.DATA.TEST_DATA_DIR = str(Path(args.data_dir).expanduser())
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = str(Path(args.train_control_index).expanduser())
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL = str(Path(args.val_control_index).expanduser())
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX = ""
    config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = ""
    config.DATA.COUNTERFACTUAL_MODE = "path_only"
    config.DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER = True
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    config.MODEL.LOCAL_CONTROL_USE_PATH = True
    config.MODEL.LOCAL_CONTROL_USE_ANCHOR = True
    config.MODEL.LOCAL_CONTROL_USE_COMPLIANCE = False
    config.MODEL.LOCAL_CONTROL_USE_TIMING = False
    return config


def _summarize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    decoder_track_names = batch.get("decoder/track_name", [])
    branch_labels = []
    raw_track_ids = []
    model_agent_slots = []
    path_mask = _to_numpy(batch["cf/path_supervision_mask"]).astype(np.int64)
    for sample_idx in range(len(path_mask)):
        if int(path_mask[sample_idx]) > 0:
            branch_payload = decode_path_token_tensor(_to_numpy(batch["cf/path_token"])[sample_idx])
            branch_labels.append(str(branch_payload["branch_label"]))
        decoded_mask = decode_decision_agent_mask(
            _to_numpy(batch["cf/decision_agent_mask"])[sample_idx],
            decoder_track_names=decoder_track_names[sample_idx] if decoder_track_names else None,
        )
        active_track_names = [str(value) for value in decoded_mask["active_track_names"]]
        if not active_track_names:
            sample_debug_meta = dict(batch["cf/debug_meta"][sample_idx]) if isinstance(batch.get("cf/debug_meta"), list) else {}
            if sample_debug_meta.get("agent_id"):
                active_track_names = [str(sample_debug_meta["agent_id"])]
        raw_track_ids.extend(active_track_names)
        model_agent_slots.extend(int(value) for value in decoded_mask["active_agent_indices"])
    return {
        "batch_size": int(len(path_mask)),
        "scenario_ids": _to_numpy(batch["metadata/scenario_id"]).astype(str).tolist(),
        "path_active_mask_sum": int(path_mask.sum()),
        "anchor_active_mask_sum": int(path_mask.sum()),
        "decision_agent_mask_sum": float(_to_numpy(batch["cf/decision_agent_mask"]).astype(np.float32).sum()),
        "time_window_mask_sum": float(_to_numpy(batch["cf/time_window_mask"]).astype(np.float32).sum()),
        "branch_labels_present": sorted(set(branch_labels)),
        "raw_track_ids": raw_track_ids,
        "model_agent_slots": model_agent_slots,
    }


def _serialize_loss_metrics(loss: torch.Tensor, loss_stat: Dict[str, Any]) -> Dict[str, Any]:
    scalar_metrics: Dict[str, float] = {}
    finite_scalar_metrics: Dict[str, bool] = {}
    for key, value in loss_stat.items():
        if hasattr(value, "detach"):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                number = float(tensor.item())
                scalar_metrics[str(key)] = number
                finite_scalar_metrics[str(key)] = bool(torch.isfinite(tensor).item())
        elif isinstance(value, (int, float)):
            scalar_metrics[str(key)] = float(value)
            finite_scalar_metrics[str(key)] = bool(torch.isfinite(torch.tensor(float(value))).item())
    return {
        "total_loss": float(loss.detach().cpu().item()),
        "finite_total_loss": bool(torch.isfinite(loss.detach().cpu()).item()),
        "scalar_metrics": scalar_metrics,
        "finite_scalar_metrics": finite_scalar_metrics,
    }


def _tail_lines(text: str, *, max_lines: int) -> Dict[str, Any]:
    lines = [line for line in str(text).splitlines() if line.strip()]
    return {
        "tail_lines": lines[-max_lines:],
    }


def _train_step_completed_from_logs(stdout_text: str) -> bool:
    text = str(stdout_text)
    return ("Epoch 0: 100%" in text) and ("total_loss=" in text)


def _to_numpy(value: Any):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _to_torch_device(value: Any, *, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_torch_device(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch_device(item, device=device) for item in value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
