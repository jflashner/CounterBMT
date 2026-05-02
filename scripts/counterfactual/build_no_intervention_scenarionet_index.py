from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.normalize import load_raw_scenario
from bmt.counterfactual.sdc_path_control import (
    DEFAULT_RESAMPLE_SPACING_M,
    extract_ground_truth_sdc_route_xy,
    polyline_arc_lengths,
    polyline_headings,
    resample_polyline_xy,
)
from bmt.counterfactual.sdc_semantic_control import (
    SDC_SEMANTIC_CONTROL_SCHEMA_VERSION,
    extract_model_frame,
    tangents_from_headings,
    world_direction_to_model_frame,
    world_xy_to_model_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build factual no_intervention semantic-control rows from a ScenarioNet root."
    )
    parser.add_argument("--scenario-root", type=str, required=True)
    parser.add_argument("--output-index", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resample-spacing-m", type=float, default=DEFAULT_RESAMPLE_SPACING_M)
    parser.add_argument("--source-tag", type=str, default="scenarionet")
    return parser.parse_args()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_scenario_path(root: Path, mapping: Mapping[str, Any], file_name: str) -> Path:
    folder = Path(str(mapping.get(file_name, "") or ""))
    if folder.is_absolute():
        return folder / file_name
    return root / folder / file_name


def _metadata(raw_scenario: Mapping[str, Any], summary_entry: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = dict(raw_scenario.get("metadata", {}) or {})
    if not metadata:
        metadata = dict(summary_entry or {})
    return metadata


def _build_row(
    *,
    raw_scenario: Mapping[str, Any],
    summary_entry: Mapping[str, Any],
    scenario_pkl: Path,
    scenario_file_name: str,
    spacing_m: float,
    source_tag: str,
) -> Dict[str, Any] | None:
    metadata = _metadata(raw_scenario, summary_entry)
    scenario_id = str(
        metadata.get("scenario_id")
        or summary_entry.get("scenario_id")
        or metadata.get("id")
        or raw_scenario.get("id")
        or Path(scenario_file_name).stem
    )
    sdc_id = str(metadata.get("sdc_id") or summary_entry.get("sdc_id") or "")
    if not sdc_id:
        return None
    current_time_index = int(metadata.get("current_time_index") or summary_entry.get("current_time_index") or 0)
    gt_xy = extract_ground_truth_sdc_route_xy(
        raw_scenario,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
    )
    if gt_xy.shape[0] < 2:
        return None
    resampled_world = resample_polyline_xy(gt_xy, spacing_m=float(spacing_m))
    if resampled_world.shape[0] < 2:
        return None
    headings_world = polyline_headings(resampled_world)
    tangents_world = tangents_from_headings(headings_world)
    arc_lengths_m = polyline_arc_lengths(resampled_world)
    map_center, map_heading = extract_model_frame(raw_scenario)
    model_path = world_xy_to_model_frame(resampled_world, map_center=map_center, map_heading=map_heading)
    model_tangents = world_direction_to_model_frame(tangents_world, map_heading=map_heading)

    return {
        "schema_version": SDC_SEMANTIC_CONTROL_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_file_name": str(scenario_file_name),
        "scenario_pkl": str(scenario_pkl.resolve()),
        "sdc_id": sdc_id,
        "current_time_index": int(current_time_index),
        "requested_semantic_label": "no_intervention",
        "requested_semantic_confidence": 1.0,
        "semantic_label": "no_intervention",
        "use_for_training": True,
        "source_kind": "factual_gt",
        "selected_slot_id": "gt",
        "selected_path_id": None,
        "candidate_family_path_ids": [None],
        "candidate_family_slot_ids": ["gt"],
        "candidate_family_confidences": [1.0],
        "candidate_family_resampled_paths_world": [np.asarray(resampled_world, dtype=np.float32).tolist()],
        "candidate_family_resampled_path_tangents_world": [np.asarray(tangents_world, dtype=np.float32).tolist()],
        "candidate_family_arc_lengths_m": [np.asarray(arc_lengths_m, dtype=np.float32).tolist()],
        "candidate_family_divergence_onsets_m": [0.0],
        "candidate_family_frame": "model_map_centered",
        "candidate_family_map_center": np.asarray(map_center, dtype=np.float32).tolist(),
        "candidate_family_map_heading": float(map_heading),
        "candidate_family_resampled_paths_model": [np.asarray(model_path, dtype=np.float32).tolist()],
        "candidate_family_resampled_path_tangents_model": [np.asarray(model_tangents, dtype=np.float32).tolist()],
        "no_intervention_anchor": True,
        "no_intervention_anchor_repeat_index": 0,
        "counterfactual_objective": "ce_only",
        "metadata": {
            "source_tag": str(source_tag),
            "scenario_file_name": str(scenario_file_name),
            "family_slot_ids": ["gt"],
            "family_size": 1,
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(args.scenario_root).expanduser().resolve()
    summary = dict(_load_pickle(root / "dataset_summary.pkl"))
    mapping = dict(_load_pickle(root / "dataset_mapping.pkl"))
    file_names = list(summary.keys())
    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(file_names)
    if int(args.max_rows) > 0:
        file_names = file_names[: int(args.max_rows)]

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for file_name in file_names:
        file_name = str(file_name)
        scenario_pkl = _resolve_scenario_path(root, mapping, file_name)
        try:
            raw_scenario = load_raw_scenario(scenario_pkl)
            row = _build_row(
                raw_scenario=raw_scenario,
                summary_entry=dict(summary.get(file_name, {}) or {}),
                scenario_pkl=scenario_pkl,
                scenario_file_name=file_name,
                spacing_m=float(args.resample_spacing_m),
                source_tag=str(args.source_tag),
            )
        except Exception as exc:
            row = None
            skipped.append({"scenario_file_name": file_name, "reason": repr(exc)})
        if row is None:
            if not skipped or skipped[-1].get("scenario_file_name") != file_name:
                skipped.append({"scenario_file_name": file_name, "reason": "missing_sdc_or_gt_route"})
            continue
        rows.append(row)

    output_index = Path(args.output_index).expanduser()
    _write_jsonl(output_index, rows)
    summary_payload = {
        "scenario_root": str(root),
        "output_index": str(output_index),
        "input_scenarios_considered": int(len(file_names)),
        "output_rows": int(len(rows)),
        "skipped_rows": int(len(skipped)),
        "skipped_examples": skipped[:25],
        "source_tag": str(args.source_tag),
    }
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else output_index.with_suffix(".summary.json")
    _write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
