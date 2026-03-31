from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

if "hydra" not in sys.modules:
    hydra_stub = types.ModuleType("hydra")

    def _hydra_main(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    hydra_stub.main = _hydra_main
    sys.modules["hydra"] = hydra_stub

try:
    from scenarionet import read_dataset_summary as _scenarionet_read_dataset_summary  # type: ignore
    from scenarionet import read_scenario as _scenarionet_read_scenario  # type: ignore
except Exception:
    scenarionet_stub = types.ModuleType("scenarionet")

    def _read_dataset_summary(dataset_path):
        dataset_root = Path(dataset_path).expanduser()
        files = sorted(path.name for path in dataset_root.glob("*.pkl"))
        summary = {name: {} for name in files}
        mapping = {name: str(dataset_root) for name in files}
        return summary, files, mapping

    def _read_scenario(dataset_path, mapping, scenario_file_name):
        dataset_root = Path(dataset_path).expanduser()
        base_dir = Path(mapping.get(scenario_file_name, dataset_root))
        scenario_path = base_dir / scenario_file_name
        with scenario_path.open("rb") as f:
            return pickle.load(f)

    scenarionet_stub.read_dataset_summary = _read_dataset_summary
    scenarionet_stub.read_scenario = _read_scenario
    sys.modules["scenarionet"] = scenarionet_stub

from bmt.counterfactual.forward_supervision import (
    _ensure_runtime_imports,
    build_forward_supervision_summary_payload,
    summarize_forward_supervision_for_batch,
)
_ensure_runtime_imports()
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.config import cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the actual forward-supervised agent set used by Adv-BMT.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="training", choices=("training", "test"))
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    return parser.parse_args()


def _load_config(args: argparse.Namespace):
    config = global_config
    config_path = Path(args.config).expanduser() if args.config else (Path(global_config.ROOT_DIR) / "cfgs" / "motion_default.yaml")
    config = cfg_from_yaml_file(config_path, config)
    if args.data_dir:
        data_dir = str(Path(args.data_dir).expanduser())
        config.DATA.TRAINING_DATA_DIR = data_dir
        config.DATA.TEST_DATA_DIR = data_dir
    return config


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode=args.mode)
    samples = [dataset[idx] for idx in range(int(args.offset), int(args.offset) + int(args.batch_size))]
    batch = dataset.collate_batch(samples)
    examples = summarize_forward_supervision_for_batch(batch)
    summary = build_forward_supervision_summary_payload(examples)
    summary.update(
        {
            "mode": args.mode,
            "batch_size": len(samples),
            "offset": int(args.offset),
        }
    )

    (outdir / "forward_supervision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (outdir / "forward_supervision_examples.jsonl").write_text(
        "".join(json.dumps(example.to_dict(), sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
