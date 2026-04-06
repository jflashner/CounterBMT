from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract import normalize_contract


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _candidate_semantics_by_id(contract: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for entry in list(contract.get("candidate_semantics") or []):
        candidate_id = str(entry.get("candidate_id") or "")
        if candidate_id:
            result[candidate_id] = dict(entry)
    return result


def merge_pass_contracts(
    *,
    example_id: str,
    scenario_id: str,
    agent_id: str,
    image_set_ids: Mapping[str, Sequence[str]],
    model_name: str,
    context_only_contract: Optional[Mapping[str, Any]],
    context_plus_gt_contract: Optional[Mapping[str, Any]],
    context_plus_anchor_contract: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = normalize_contract(
        context_only_contract,
        example_id=example_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        image_set_ids=image_set_ids,
        model_name=model_name,
    )
    gt_contract = normalize_contract(
        context_plus_gt_contract,
        example_id=example_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        image_set_ids=image_set_ids,
        model_name=model_name,
    )
    anchor_contract = normalize_contract(
        context_plus_anchor_contract,
        example_id=example_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        image_set_ids=image_set_ids,
        model_name=model_name,
    )

    merged["gt_semantics"] = copy.deepcopy(gt_contract.get("gt_semantics") or merged["gt_semantics"])
    merged["anchor_audit"] = copy.deepcopy(anchor_contract.get("anchor_audit") or merged["anchor_audit"])
    merged["notes"] = [
        *[str(value) for value in list(merged.get("notes") or []) if str(value).strip()],
        *[str(value) for value in list(gt_contract.get("notes") or []) if str(value).strip()],
        *[str(value) for value in list(anchor_contract.get("notes") or []) if str(value).strip()],
    ]

    confidences = [
        _safe_float(merged.get("contract_confidence"), default=0.0),
        _safe_float(gt_contract.get("contract_confidence"), default=0.0),
        _safe_float(anchor_contract.get("contract_confidence"), default=0.0),
    ]
    merged["contract_confidence"] = float(sum(confidences) / max(len(confidences), 1))
    ambiguity = dict(merged.get("scene_ambiguity") or {})
    merged["use_for_training"] = bool(
        bool(merged.get("use_for_training"))
        and str(ambiguity.get("level") or "") != "high"
        and str(dict(anchor_contract.get("anchor_audit") or {}).get("anchor_validity") or "") != "invalid"
    )
    return merged


def fuse_geometry_and_vlm_contracts(
    *,
    raw_contract_rows: Sequence[Mapping[str, Any]],
    path_index_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    confidence_threshold: float = 0.75,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    index_by_example_id = {
        str(row.get("example_id") or ""): dict(row)
        for row in list(path_index_rows or [])
        if str(row.get("example_id") or "").strip()
    }
    fused_rows: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    low_confidence_examples: List[str] = []
    training_true = 0
    training_false = 0

    for raw_row in raw_contract_rows:
        contract = dict(raw_row.get("contract") or {})
        example_id = str(raw_row.get("example_id") or contract.get("example_id") or "")
        selected_candidate_id = str(raw_row.get("selected_candidate_id") or "")
        candidate_map = {
            str(item.get("candidate_id") or ""): dict(item)
            for item in list(raw_row.get("candidate_id_map") or [])
            if str(item.get("candidate_id") or "").strip()
        }
        candidate_semantics = _candidate_semantics_by_id(contract)
        selected_candidate_semantics = dict(candidate_semantics.get(selected_candidate_id) or {})
        geometry_semantic_label = str(raw_row.get("geometry_branch_label") or "")
        vlm_semantic_label = str(selected_candidate_semantics.get("semantic_label") or "unknown")
        contract_confidence = _safe_float(contract.get("contract_confidence"), default=0.0)
        use_vlm = bool(
            contract_confidence >= float(confidence_threshold)
            and str(vlm_semantic_label) not in {"", "unknown"}
        )
        if contract_confidence < float(confidence_threshold):
            low_confidence_examples.append(example_id)

        path_row = copy.deepcopy(index_by_example_id.get(example_id) or {})
        fused = copy.deepcopy(path_row) if path_row else {}
        fused.update(
            {
                "example_id": example_id,
                "scenario_id": str(raw_row.get("scenario_id") or fused.get("scenario_id") or ""),
                "agent_id": str(raw_row.get("agent_id") or fused.get("agent_id") or ""),
                "decision_time_idx": int(raw_row.get("decision_time_idx") or fused.get("decision_time_idx") or 0),
                "selected_mode": str(raw_row.get("selected_mode") or "factual"),
                "geometry_branch_label": geometry_semantic_label,
                "geometry_branch_id": str(raw_row.get("geometry_branch_id") or ""),
                "geometry_light_group_id": raw_row.get("geometry_light_group_id"),
                "geometry_primary_light_id": raw_row.get("geometry_primary_light_id"),
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_geometry_branch_id": str(raw_row.get("selected_candidate_geometry_branch_id") or ""),
                "selected_candidate_geometry_label": str(raw_row.get("selected_candidate_geometry_label") or ""),
                "vlm_semantic_label": vlm_semantic_label if use_vlm else geometry_semantic_label,
                "vlm_candidate_id_map": list(raw_row.get("candidate_id_map") or []),
                "vlm_controlling_light_group": dict(contract.get("controlling_light_group") or {}),
                "vlm_split_point": dict(contract.get("split_point") or {}),
                "vlm_exit_zone": list(contract.get("exit_zones") or []),
                "vlm_anchor_validity": str(dict(contract.get("anchor_audit") or {}).get("anchor_validity") or "unknown"),
                "vlm_contract_confidence": contract_confidence,
                "vlm_use_for_training": bool(contract.get("use_for_training")),
                "vlm_scene_ambiguity": dict(contract.get("scene_ambiguity") or {}),
                "vlm_contract": contract,
                "vlm_high_confidence": bool(use_vlm),
                "vlm_geometry_agreement": bool(
                    not geometry_semantic_label
                    or vlm_semantic_label in {"", "unknown"}
                    or str(geometry_semantic_label) == str(vlm_semantic_label)
                ),
            }
        )
        if bool(fused.get("vlm_use_for_training")):
            training_true += 1
        else:
            training_false += 1

        if use_vlm and geometry_semantic_label and vlm_semantic_label not in {"", "unknown"} and geometry_semantic_label != vlm_semantic_label:
            disagreements.append(
                {
                    "example_id": example_id,
                    "selected_mode": fused.get("selected_mode"),
                    "selected_candidate_id": selected_candidate_id,
                    "geometry_branch_label": geometry_semantic_label,
                    "vlm_semantic_label": vlm_semantic_label,
                    "contract_confidence": contract_confidence,
                    "selected_candidate_geometry_branch_id": fused.get("selected_candidate_geometry_branch_id"),
                    "selected_candidate_metadata": candidate_map.get(selected_candidate_id),
                }
            )
        fused_rows.append(fused)

    disagreement_report = {
        "num_rows": int(len(fused_rows)),
        "num_disagreements_high_confidence": int(len(disagreements)),
        "num_low_confidence_examples": int(len(low_confidence_examples)),
        "confidence_threshold": float(confidence_threshold),
        "first_50_disagreements": disagreements[:50],
        "first_50_low_confidence_example_ids": low_confidence_examples[:50],
    }
    training_eligibility_report = {
        "num_rows": int(len(fused_rows)),
        "num_use_for_training_true": int(training_true),
        "num_use_for_training_false": int(training_false),
        "num_scene_ambiguity_high": int(
            sum(1 for row in fused_rows if str(dict(row.get("vlm_scene_ambiguity") or {}).get("level") or "") == "high")
        ),
        "num_anchor_invalid": int(sum(1 for row in fused_rows if str(row.get("vlm_anchor_validity") or "") == "invalid")),
        "num_vlm_high_confidence": int(sum(1 for row in fused_rows if bool(row.get("vlm_high_confidence")))),
    }
    return fused_rows, disagreement_report, training_eligibility_report
