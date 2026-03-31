from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import decode_decision_agent_mask, decode_path_token_tensor
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a path-only batch.")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--control-source", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--mode", type=str, default="training", choices=("training", "test"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode=args.mode)
    samples = [dataset[idx] for idx in range(int(args.offset), int(args.offset) + int(args.batch_size))]
    batch = dataset.collate_batch(samples)

    branch_labels_present: List[str] = []
    raw_track_ids: List[str] = []
    model_agent_slots: List[int] = []
    decoder_track_names = batch.get("decoder/track_name", [])

    for sample_idx in range(len(samples)):
        branch_payload = decode_path_token_tensor(_to_numpy(batch["cf/path_token"])[sample_idx])
        if int(_to_numpy(batch["cf/path_supervision_mask"])[sample_idx]) > 0:
            branch_labels_present.append(str(branch_payload["branch_label"]))
        decoded_mask = decode_decision_agent_mask(
            _to_numpy(batch["cf/decision_agent_mask"])[sample_idx],
            decoder_track_names=decoder_track_names[sample_idx] if decoder_track_names else None,
        )
        raw_track_ids.extend(str(value) for value in decoded_mask["active_track_names"])
        model_agent_slots.extend(int(value) for value in decoded_mask["active_agent_indices"])

    summary = {
        "mode": args.mode,
        "batch_size": len(samples),
        "counterfactual_mode": "path_only",
        "path_active_mask_sum": int(np.asarray(_to_numpy(batch["cf/path_supervision_mask"]), dtype=np.int64).sum()),
        "anchor_active_mask_sum": int(np.asarray(_to_numpy(batch["cf/path_supervision_mask"]), dtype=np.int64).sum()),
        "decision_agent_mask_sum": float(np.asarray(_to_numpy(batch["cf/decision_agent_mask"]), dtype=np.float32).sum()),
        "time_window_mask_sum": float(np.asarray(_to_numpy(batch["cf/time_window_mask"]), dtype=np.float32).sum()),
        "branch_labels_present": sorted(set(branch_labels_present)),
        "raw_track_ids": raw_track_ids,
        "model_agent_slots": model_agent_slots,
        "scenario_ids": _to_numpy(batch["metadata/scenario_id"]).astype(str).tolist(),
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / cfg_path).resolve()
        config = cfg_from_yaml_file(cfg_path, config)
    elif default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)

    control_source = Path(args.control_source).expanduser()
    if control_source.suffix == ".jsonl":
        config.DATA.COUNTERFACTUAL_CONTROL_INDEX = str(control_source)
        config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = ""
    else:
        config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = str(control_source)
        config.DATA.COUNTERFACTUAL_CONTROL_INDEX = ""
    config.DATA.COUNTERFACTUAL_MODE = "path_only"
    if args.mode == "training":
        config.DATA.TRAINING_DATA_DIR = str(Path(args.data_dir).expanduser())
    else:
        config.DATA.TEST_DATA_DIR = str(Path(args.data_dir).expanduser())
    return config


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
