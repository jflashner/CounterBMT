from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from bmt.counterfactual.path_eval_bundle import (
    build_bundle_inventory,
    load_json,
    load_jsonl,
    load_materialized_controls,
)


def _load_selected_control_code(materialized_controls: Mapping[str, Any], *, selected_mode: str) -> Optional[Dict[str, Any]]:
    factual = materialized_controls.get("factual_control_code")
    alternatives = list(materialized_controls.get("alternative_control_codes") or [])
    if selected_mode == "factual":
        return None if factual is None else dict(factual)
    if selected_mode.startswith("alternative_"):
        try:
            rank = int(str(selected_mode).split("_", 1)[1])
        except Exception:
            return None
        if 0 <= rank < len(alternatives):
            return dict(alternatives[rank])
    return None


def _build_candidate_id_map(
    branch_candidates: Sequence[Mapping[str, Any]],
    *,
    selected_control_code: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    selected_branch_id = str(dict(selected_control_code or {}).get("path_token", {}).get("branch_id") or "")
    rows: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(list(branch_candidates)):
        candidate_id = f"B{idx}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "branch_id": str(candidate.get("branch_id") or ""),
                "geometry_branch_label": str(candidate.get("branch_label") or "unknown"),
                "source_feature_id": str(candidate.get("source_feature_id") or ""),
                "is_selected_candidate": bool(selected_branch_id and str(candidate.get("branch_id") or "") == selected_branch_id),
            }
        )
    return rows


def load_bundle_selected_examples(
    *,
    bundle_root: str | Path,
    selected_manifest_path: Optional[str | Path] = None,
    max_examples: int = 0,
) -> Dict[str, Any]:
    root = Path(bundle_root).expanduser()
    inventory = build_bundle_inventory(root)
    manifest_path = Path(selected_manifest_path).expanduser() if selected_manifest_path else root / "audit_local" / "selected_examples_manifest.json"
    selected_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if max_examples > 0:
        selected_rows = list(selected_rows)[: int(max_examples)]
    path_index_rows = load_jsonl(inventory["path_index_curated_val_jsonl"])
    index_by_example_id = {str(row.get("example_id") or ""): row for row in path_index_rows}
    eval_rows = load_jsonl(inventory["path_control_eval_per_example_jsonl"])
    eval_by_key = {
        (str(row.get("example_id") or ""), str(row.get("mode") or "")): row
        for row in eval_rows
        if str(row.get("example_id") or "").strip()
    }

    examples: List[Dict[str, Any]] = []
    for row in selected_rows:
        materialized_dir = Path(str(row["local_materialized_eval_input"])).expanduser()
        materialized_controls = load_materialized_controls(materialized_dir)
        selected_mode = str(row.get("selected_mode") or row.get("factual_or_alternative") or "factual")
        selected_control_code = _load_selected_control_code(materialized_controls, selected_mode=selected_mode)
        branch_candidates = list(materialized_controls.get("branch_candidates") or [])
        candidate_id_map = _build_candidate_id_map(branch_candidates, selected_control_code=selected_control_code)
        selected_candidate = next((item for item in candidate_id_map if bool(item.get("is_selected_candidate"))), None)
        path_index_row = dict(index_by_example_id.get(str(row.get("example_id") or ""), {}) or {})
        eval_row = dict(eval_by_key.get((str(row.get("example_id") or ""), selected_mode), {}) or {})
        examples.append(
            {
                "example_id": str(row["example_id"]),
                "scenario_id": str(row["scenario_id"]),
                "agent_id": str(row["agent_id"]),
                "decision_time_idx": int(row["decision_time_idx"]),
                "requested_branch_label": str(row.get("requested_branch_label") or ""),
                "selected_mode": selected_mode,
                "factual_or_alternative": str(row.get("factual_or_alternative") or ""),
                "control_sweep_png": row.get("control_sweep_png"),
                "has_sweep_png": bool(row.get("has_sweep_png")),
                "local_scenario_pkl": str(row["local_scenario_pkl"]),
                "local_materialized_eval_input": str(materialized_dir),
                "path_index_row": path_index_row,
                "eval_row": eval_row,
                "candidate_id_map": candidate_id_map,
                "selected_candidate_id": None if selected_candidate is None else str(selected_candidate["candidate_id"]),
                "selected_candidate_geometry_branch_id": None if selected_candidate is None else str(selected_candidate["branch_id"]),
                "selected_candidate_geometry_label": None if selected_candidate is None else str(selected_candidate["geometry_branch_label"]),
                "geometry_branch_label": str(path_index_row.get("branch_label") or row.get("requested_branch_label") or ""),
                "geometry_branch_id": str(path_index_row.get("branch_id") or ""),
                "light_group_ids": [
                    value
                    for value in [
                        path_index_row.get("light_group_id"),
                        path_index_row.get("primary_light_id"),
                    ]
                    if value not in (None, "")
                ],
            }
        )
    return {
        "bundle_inventory": inventory,
        "selected_examples": examples,
        "selected_manifest_path": str(manifest_path),
    }


def load_materialized_manifest_examples(
    *,
    materialized_manifest_path: str | Path,
    path_index_path: str | Path,
    max_examples: int = 0,
) -> Dict[str, Any]:
    manifest_rows = load_jsonl(materialized_manifest_path)
    if max_examples > 0:
        manifest_rows = list(manifest_rows)[: int(max_examples)]
    path_index_rows = load_jsonl(path_index_path)
    index_by_example_id = {str(row.get("example_id") or ""): dict(row) for row in path_index_rows}
    examples: List[Dict[str, Any]] = []
    for row in manifest_rows:
        artifact_dir = str(row.get("artifact_dir") or "")
        if not artifact_dir:
            continue
        materialized_dir = Path(artifact_dir).expanduser()
        materialized_controls = load_materialized_controls(materialized_dir)
        example_id = str(row.get("example_id") or "")
        path_index_row = dict(index_by_example_id.get(example_id) or {})
        selected_control_code = dict(materialized_controls.get("factual_control_code") or {})
        branch_candidates = list(materialized_controls.get("branch_candidates") or [])
        candidate_id_map = _build_candidate_id_map(branch_candidates, selected_control_code=selected_control_code)
        selected_candidate = next((item for item in candidate_id_map if bool(item.get("is_selected_candidate"))), None)
        examples.append(
            {
                "example_id": example_id,
                "scenario_id": str(row.get("scenario_id") or path_index_row.get("scenario_id") or ""),
                "agent_id": str(row.get("agent_id") or path_index_row.get("agent_id") or ""),
                "decision_time_idx": int(row.get("decision_time_idx") or path_index_row.get("decision_time_idx") or 0),
                "requested_branch_label": str(path_index_row.get("branch_label") or ""),
                "selected_mode": "factual",
                "factual_or_alternative": "factual",
                "control_sweep_png": None,
                "has_sweep_png": False,
                "local_scenario_pkl": str(row.get("scenario_pkl") or path_index_row.get("scenario_pkl") or ""),
                "local_materialized_eval_input": str(materialized_dir),
                "path_index_row": path_index_row,
                "eval_row": {},
                "candidate_id_map": candidate_id_map,
                "selected_candidate_id": None if selected_candidate is None else str(selected_candidate["candidate_id"]),
                "selected_candidate_geometry_branch_id": None if selected_candidate is None else str(selected_candidate["branch_id"]),
                "selected_candidate_geometry_label": None if selected_candidate is None else str(selected_candidate["geometry_branch_label"]),
                "geometry_branch_label": str(path_index_row.get("branch_label") or ""),
                "geometry_branch_id": str(path_index_row.get("branch_id") or ""),
                "light_group_ids": [
                    value
                    for value in [
                        path_index_row.get("light_group_id"),
                        path_index_row.get("primary_light_id"),
                    ]
                    if value not in (None, "")
                ],
            }
        )
    return {
        "bundle_inventory": {},
        "selected_examples": examples,
        "selected_manifest_path": str(Path(materialized_manifest_path).expanduser()),
    }
