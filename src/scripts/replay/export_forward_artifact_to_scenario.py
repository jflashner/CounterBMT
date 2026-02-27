"""Export one forward-eval rollout artifact to ScenarioNet replay files.

This bridges CounterBMT v2 forward-eval artifacts (`*.npz`) to ScenarioNet /
MetaDrive replay format using the legacy-but-stable export helpers in
`counter_bmt.scenario_export`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from counter_bmt.scenario_export import (
    create_dataset_summary,
    create_replay_script,
    export_ground_truth_scenario,
    export_trajectory_only,
)


def _safe_scenario_id(raw: object) -> str:
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        raw = raw.reshape(-1)[0]
    return str(raw)


def _find_scenario_file(scenario_root: Path, scenario_id: str) -> Optional[Path]:
    # ScenarioNet format: sd_*_<scenario_id>.pkl, often nested under shard dirs.
    candidates = sorted(scenario_root.rglob(f"sd_*{scenario_id}.pkl"))
    if not candidates:
        return None
    return candidates[0]


def _load_artifact(npz_path: Path) -> tuple[np.ndarray, str]:
    data = np.load(npz_path, allow_pickle=True)
    if "pred_pos_ktn2" not in data:
        raise KeyError(f"{npz_path} missing key `pred_pos_ktn2`")
    if "scenario_id" not in data:
        raise KeyError(f"{npz_path} missing key `scenario_id`")
    pred_pos = np.asarray(data["pred_pos_ktn2"], dtype=np.float32)  # [K,T,N,2]
    if pred_pos.ndim != 4:
        raise ValueError(f"Expected pred_pos_ktn2 rank=4, got shape={pred_pos.shape}")
    scenario_id = _safe_scenario_id(data["scenario_id"])
    if not scenario_id:
        raise ValueError(f"Empty scenario_id in artifact: {npz_path}")
    return pred_pos, scenario_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export forward-eval artifact rollout to ScenarioNet replay files")
    p.add_argument("--artifact-npz", type=str, required=True, help="Path to one forward_eval_artifacts *.npz file")
    p.add_argument("--scenario-root", type=str, required=True, help="ScenarioNet dataset root used to find original scenario pkl")
    p.add_argument("--output-dir", type=str, required=True, help="Output replay directory")
    p.add_argument("--mode-index", type=int, default=0, help="Trajectory mode index in pred_pos_ktn2")
    p.add_argument("--intervention-name", type=str, default="counterfactual_rollout")
    p.add_argument("--include-ground-truth", action="store_true", help="Also export original GT scenario into replay folder")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifact_npz = Path(args.artifact_npz)
    scenario_root = Path(args.scenario_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not artifact_npz.is_file():
        raise FileNotFoundError(f"artifact not found: {artifact_npz}")
    if not scenario_root.is_dir():
        raise FileNotFoundError(f"scenario root not found: {scenario_root}")

    pred_pos_ktn2, scenario_id = _load_artifact(artifact_npz)
    mode = int(np.clip(int(args.mode_index), 0, max(0, pred_pos_ktn2.shape[0] - 1)))
    ego_xy_t2 = np.asarray(pred_pos_ktn2[mode, :, 0, :], dtype=np.float32)

    scenario_file = _find_scenario_file(scenario_root, scenario_id)
    if scenario_file is None:
        raise FileNotFoundError(
            f"Could not locate scenario file for scenario_id={scenario_id} under {scenario_root}"
        )

    with open(scenario_file, "rb") as f:
        original_scenario = pickle.load(f)
    map_center = original_scenario.get("metadata", {}).get("map_center", None)

    safe_sid = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in scenario_id)
    safe_int = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(args.intervention_name))

    cf_path = output_dir / f"sd_counterfactual_1.0_{safe_sid}_{safe_int}_m{mode}.pkl"
    cf_saved = export_trajectory_only(
        trajectory=ego_xy_t2,
        original_scenario=original_scenario,
        output_path=cf_path,
        intervention_name=str(args.intervention_name),
        original_file_path=scenario_file,
        map_center=map_center,
    )
    if cf_saved is None:
        raise RuntimeError("Failed to export counterfactual scenario")

    saved_paths = [Path(cf_saved)]
    if bool(args.include_ground_truth):
        gt_path = output_dir / f"sd_counterfactual_1.0_{safe_sid}_ground_truth.pkl"
        gt_saved = export_ground_truth_scenario(
            original_file_path=scenario_file,
            output_path=gt_path,
        )
        if gt_saved is not None:
            saved_paths.append(Path(gt_saved))

    create_dataset_summary(saved_paths, output_dir)
    replay_script = create_replay_script(saved_paths, output_dir / "replay_scenarios.py")

    print("Export complete")
    print(f"  artifact: {artifact_npz}")
    print(f"  scenario_id: {scenario_id}")
    print(f"  source_scenario: {scenario_file}")
    print(f"  counterfactual: {cf_saved}")
    if bool(args.include_ground_truth) and len(saved_paths) > 1:
        print(f"  ground_truth: {saved_paths[1]}")
    print(f"  dataset_summary: {output_dir / 'dataset_summary.pkl'}")
    print(f"  replay_script: {replay_script}")
    print("")
    print("Replay (2D):")
    print(f"  python -m scenarionet.sim -d {output_dir} --render 2D")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

