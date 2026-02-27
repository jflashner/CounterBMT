"""Batch export canonical forward artifacts to ScenarioNet replay packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.eval.compare import build_artifact_index
from counter_bmt_v2.eval.replay_export import export_replays_from_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export replay scenarios for multiple model artifact folders.")
    p.add_argument("--artifacts-root", type=str, required=True, help="Root containing per-model artifact dirs")
    p.add_argument("--dataset-dir", type=str, required=True, help="ScenarioNet dataset root")
    p.add_argument("--scenario-subset-file", type=str, required=True, help="JSON with subset entries including scenario_id and relative_path")
    p.add_argument("--output-dir", type=str, required=True, help="Replay export output root")
    p.add_argument("--max-scenarios", type=int, default=8)
    p.add_argument("--mode-index", type=int, default=0)
    p.add_argument("--include-ground-truth", action="store_true")
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifacts_root = Path(args.artifacts_root)
    dataset_dir = Path(args.dataset_dir)
    subset_file = Path(args.scenario_subset_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_raw = json.loads(subset_file.read_text(encoding="utf-8"))
    entries = subset_raw.get("entries", []) if isinstance(subset_raw, dict) else []
    scenario_relpath_by_id = {
        str(x.get("scenario_id")): str(x.get("relative_path"))
        for x in entries
        if isinstance(x, dict) and x.get("scenario_id") and x.get("relative_path")
    }
    selected = [str(x.get("scenario_id")) for x in entries[: max(0, int(args.max_scenarios))] if isinstance(x, dict)]

    model_dirs: Dict[str, Path] = {}
    for p in sorted(artifacts_root.iterdir()):
        if p.is_dir():
            step_eval = p / "step_eval"
            if step_eval.is_dir():
                model_dirs[p.name] = step_eval
    artifact_index = build_artifact_index(model_dirs)

    exports = export_replays_from_artifacts(
        artifact_index=artifact_index,
        dataset_dir=dataset_dir,
        scenario_relpath_by_id=scenario_relpath_by_id,
        out_dir=out_dir,
        selected_scenarios=selected,
        mode_index=int(args.mode_index),
        include_ground_truth=bool(args.include_ground_truth),
    )
    payload = {
        "artifacts_root": str(artifacts_root),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "num_models": int(len(artifact_index)),
        "num_selected_scenarios": int(len(selected)),
        "exports": exports,
    }
    if str(args.output_json).strip():
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
