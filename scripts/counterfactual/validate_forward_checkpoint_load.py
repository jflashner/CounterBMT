from __future__ import annotations

import argparse
import copy
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from scripts.counterfactual.inspect_batch_control import (
    _run_forward_smoke,
    _to_numpy,
)

from bmt.counterfactual import (
    decode_compliance_token_tensor,
    decode_decision_agent_mask,
    decode_path_token_tensor,
    decode_terminal_anchor_tensor,
    decode_time_window_mask,
    decode_timing_token_tensor,
)
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate forgiving checkpoint load on a curated path-only batch.")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", type=str, default="training", choices=("training", "test"))
    parser.add_argument("--forward-control-mode", type=str, default="strict_local", choices=("interactive", "strict_local"))
    parser.add_argument("--counterfactual-mode", type=str, default="path_only")
    parser.add_argument("--run-forward", action="store_true")
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

    checkpoint_report_path = outdir / "checkpoint_load_report.json"
    loaded_module_summary_path = outdir / "loaded_module_summary.json"
    runtime_path = outdir / "curated_forward_runtime_smoke.json"
    selected_control_path = outdir / "curated_selected_control.json"
    batch_summary_path = outdir / "curated_forward_batch_summary.json"

    checkpoint_report: Dict[str, Any] = {
        "ckpt_path": str(Path(args.ckpt).expanduser()),
        "load_mode": str(args.load_mode),
        "status": "not_started",
    }
    runtime_summary: Dict[str, Any] = {
        "ran": False,
        "reason": "not_started",
    }

    try:
        config = _load_config(args)
        dataset = InfgenDataset(config=config, mode=args.mode)
        samples = [dataset[idx] for idx in range(int(args.batch_size))]
        batch = dataset.collate_batch(samples)

        debug_meta = batch["cf/debug_meta"]
        available_flags = [bool(item.get("available")) for item in debug_meta]
        selected_example_idx = available_flags.index(True) if any(available_flags) else 0
        decoder_track_names = None
        if "decoder/track_name" in batch:
            decoder_track_names = batch["decoder/track_name"][selected_example_idx]

        decoded = {
            "path_token": decode_path_token_tensor(_to_numpy(batch["cf/path_token"])[selected_example_idx]),
            "compliance_token": decode_compliance_token_tensor(_to_numpy(batch["cf/compliance_token"])[selected_example_idx]),
            "timing_token": decode_timing_token_tensor(_to_numpy(batch["cf/timing_token"])[selected_example_idx]),
            "terminal_anchor": decode_terminal_anchor_tensor(_to_numpy(batch["cf/terminal_anchor"])[selected_example_idx]),
            "time_window": decode_time_window_mask(_to_numpy(batch["cf/time_window_mask"])[selected_example_idx]),
            "target_agent_slot": decode_decision_agent_mask(
                _to_numpy(batch["cf/decision_agent_mask"])[selected_example_idx],
                decoder_track_names=decoder_track_names,
            ),
            "debug_meta": debug_meta[selected_example_idx],
        }
        selected_control = {
            "scenario_id": _to_numpy(batch["metadata/scenario_id"]).astype(str).tolist()[selected_example_idx],
            "selected_example_idx": int(selected_example_idx),
            **decoded,
        }
        selected_control_path.write_text(json.dumps(selected_control, indent=2, sort_keys=True), encoding="utf-8")

        batch_summary = {
            "mode": args.mode,
            "batch_size": len(samples),
            "selected_example_idx": int(selected_example_idx),
            "scenario_ids": _to_numpy(batch["metadata/scenario_id"]).astype(str).tolist(),
            "available_control_codes": int(sum(available_flags)),
            "path_active_mask_sum": int(_to_numpy(batch["cf/path_supervision_mask"]).astype("int64").sum()),
            "anchor_active_mask_sum": int(_to_numpy(batch["cf/path_supervision_mask"]).astype("int64").sum()),
            "decision_agent_mask_sum": float(_to_numpy(batch["cf/decision_agent_mask"]).astype("float32").sum()),
            "time_window_mask_sum": float(_to_numpy(batch["cf/time_window_mask"]).astype("float32").sum()),
            "branch_labels_present": sorted(
                {
                    str(decode_path_token_tensor(_to_numpy(batch["cf/path_token"])[idx])["branch_label"])
                    for idx in range(len(samples))
                    if int(_to_numpy(batch["cf/path_supervision_mask"])[idx]) > 0
                }
            ),
            "raw_track_ids": [str(value) for value in decoded["target_agent_slot"]["active_track_names"]],
            "model_agent_slots": [int(value) for value in decoded["target_agent_slot"]["active_agent_indices"]],
            "counterfactual_mode": str(args.counterfactual_mode),
            "forward_control_mode": str(args.forward_control_mode),
        }
        batch_summary_path.write_text(json.dumps(batch_summary, indent=2, sort_keys=True), encoding="utf-8")

        if args.run_forward:
            forward_summary = _run_forward_smoke(
                config=config,
                batch=batch,
                selected_example_idx=selected_example_idx,
                ckpt_path=args.ckpt,
                load_mode=args.load_mode,
            )
            checkpoint_report = dict(forward_summary.get("checkpoint_load_report", checkpoint_report))
            checkpoint_report["status"] = "loaded"
            runtime_summary = dict(forward_summary)
            runtime_summary["ran"] = True
            loaded_module_summary = dict(forward_summary.get("loaded_module_summary", {}))
            loaded_module_summary_path.write_text(
                json.dumps(loaded_module_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            runtime_summary = {
                "ran": False,
                "reason": "run_forward_disabled",
                "selected_example_idx": int(selected_example_idx),
            }
    except Exception as exc:
        checkpoint_report.setdefault("status", "failed")
        checkpoint_report["failure_reason"] = str(exc)
        runtime_summary = {
            "ran": False,
            "reason": "checkpoint_or_forward_failed",
            "failure_reason": str(exc),
            "traceback_summary": traceback.format_exc().splitlines()[-20:],
        }

    checkpoint_report_path.write_text(json.dumps(checkpoint_report, indent=2, sort_keys=True), encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(runtime_summary, indent=2, sort_keys=True))
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

    control_index = str(Path(args.control_index).expanduser())
    config.DATA.COUNTERFACTUAL_CONTROL_INDEX = control_index
    if args.mode == "training":
        config.DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN = control_index
        config.DATA.TRAINING_DATA_DIR = str(Path(args.data_dir).expanduser())
    else:
        config.DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL = control_index
        config.DATA.TEST_DATA_DIR = str(Path(args.data_dir).expanduser())
    config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = ""
    config.DATA.COUNTERFACTUAL_MODE = str(args.counterfactual_mode)
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = str(args.forward_control_mode)
    if str(args.counterfactual_mode) == "path_only":
        config.MODEL.LOCAL_CONTROL_USE_PATH = True
        config.MODEL.LOCAL_CONTROL_USE_ANCHOR = True
        config.MODEL.LOCAL_CONTROL_USE_COMPLIANCE = False
        config.MODEL.LOCAL_CONTROL_USE_TIMING = False
    return config


if __name__ == "__main__":
    raise SystemExit(main())
