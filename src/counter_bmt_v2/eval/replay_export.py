"""Replay export helpers for head-to-head artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from counter_bmt.scenario_export import (
    create_dataset_summary,
    create_replay_script,
    export_ground_truth_scenario,
    export_trajectory_only,
)


def _safe_name(x: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(x))


def export_replays_from_artifacts(
    *,
    artifact_index: Mapping[str, Mapping[str, Path]],
    dataset_dir: Path,
    scenario_relpath_by_id: Mapping[str, str],
    out_dir: Path,
    selected_scenarios: Sequence[str],
    mode_index: int = 0,
    include_ground_truth: bool = False,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for model_id, by_sid in artifact_index.items():
        model_out = out_dir / _safe_name(model_id)
        model_out.mkdir(parents=True, exist_ok=True)
        exported: List[Path] = []
        for sid in selected_scenarios:
            sid = str(sid)
            ap = by_sid.get(sid)
            rel = scenario_relpath_by_id.get(sid, "")
            if ap is None or not rel:
                continue
            scenario_file = (dataset_dir / rel).resolve()
            if not scenario_file.is_file():
                continue
            with np.load(ap, allow_pickle=True) as d:
                pred = np.asarray(d["pred_pos_ktn2"], dtype=np.float32)
                mode = int(np.clip(int(mode_index), 0, max(0, pred.shape[0] - 1)))
                ego_xy = np.asarray(pred[mode, :, 0, :], dtype=np.float32)

            with scenario_file.open("rb") as f:
                raw = pickle.load(f)
            map_center = raw.get("metadata", {}).get("map_center", None)

            scenario_dir = model_out / _safe_name(sid)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scenario_saved: List[Path] = []
            cf_path = scenario_dir / f"sd_counterfactual_1.0_{_safe_name(sid)}_{_safe_name(model_id)}.pkl"
            saved = export_trajectory_only(
                trajectory=ego_xy,
                original_scenario=raw,
                output_path=cf_path,
                intervention_name=str(model_id),
                original_file_path=scenario_file,
                map_center=map_center,
            )
            if saved is not None:
                p_saved = Path(saved)
                exported.append(p_saved)
                scenario_saved.append(p_saved)

            if bool(include_ground_truth):
                gt_path = scenario_dir / f"sd_counterfactual_1.0_{_safe_name(sid)}_ground_truth.pkl"
                gt_saved = export_ground_truth_scenario(
                    original_file_path=scenario_file,
                    output_path=gt_path,
                )
                if gt_saved is not None:
                    p_saved = Path(gt_saved)
                    exported.append(p_saved)
                    scenario_saved.append(p_saved)

            if scenario_saved:
                create_dataset_summary(scenario_saved, scenario_dir)
                create_replay_script(scenario_saved, scenario_dir / "replay_scenarios.py")

        if exported:
            create_dataset_summary(exported, model_out)
            create_replay_script(exported, model_out / "replay_scenarios.py")
        out[str(model_id)] = [str(p) for p in exported]
    return out
