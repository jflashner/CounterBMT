from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from .contract import VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION


def _json_block(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True)


def _candidate_ids_only(rows: Sequence[Mapping[str, Any]]) -> Sequence[Dict[str, Any]]:
    return [
        {
            "candidate_id": str(row.get("candidate_id") or ""),
            "source_feature_id": str(row.get("source_feature_id") or ""),
            "is_selected_candidate": bool(row.get("is_selected_candidate")),
        }
        for row in rows
        if str(row.get("candidate_id") or "").strip()
    ]


def build_context_only_prompt(record: Mapping[str, Any]) -> str:
    return f"""
You are auditing a road-scene control example and must emit structured JSON only.

Task:
- Identify which traffic-light group most likely controls the target vehicle.
- Estimate where the shared stem ends and the branch split begins.
- Map neutral candidate ids B0/B1/B2... to human branch semantics: left / straight / right / u_turn / unknown.
- Mark ambiguous scenes as ambiguous instead of forcing a guess.
- Keep rationales very short.

Important rules:
- Do NOT use the requested branch label. Treat candidate ids as neutral geometry handles.
- Abstain with semantic_label=unknown or selected_light_group_id=null if uncertain.
- Set use_for_training=false if the scene is highly ambiguous.
- Output JSON matching schema_version={VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "agent_id": record.get("agent_id"),
    "decision_time_idx": record.get("decision_time_idx"),
    "selected_mode": record.get("selected_mode"),
    "candidate_ids": _candidate_ids_only(record.get("candidate_id_map") or []),
    "light_group_ids": record.get("light_group_ids"),
    "split_point_guess": record.get("split_point_guess"),
    "frame_label": record.get("frame_label"),
})}

Fill these fields from this pass:
- scene_ambiguity
- controlling_light_group
- split_point
- candidate_semantics
- exit_zones
- contract_confidence
- use_for_training
- notes

Leave gt_semantics conservative if not visible.
Leave anchor_audit conservative if the anchor is not shown.
""".strip()


def build_context_plus_gt_prompt(record: Mapping[str, Any]) -> str:
    return f"""
You are auditing the GT semantics for one counterfactual scene example.

Task:
- Identify which candidate branch B0/B1/B2... best matches the target vehicle's GT future.
- Assign a human semantic label to that GT branch.
- If GT is ambiguous, abstain with unknown.

Important rules:
- Do NOT assume the requested branch is correct.
- Use only what is visible in the GT overlay.
- Output JSON matching schema_version={VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "agent_id": record.get("agent_id"),
    "decision_time_idx": record.get("decision_time_idx"),
    "candidate_ids": _candidate_ids_only(record.get("candidate_id_map") or []),
    "frame_label": record.get("frame_label"),
})}

Focus on:
- gt_semantics
- contract_confidence
- notes

Leave anchor_audit conservative if uncertain.
""".strip()


def build_context_plus_anchor_prompt(record: Mapping[str, Any]) -> str:
    return f"""
You are auditing whether a requested anchor is semantically valid for one scene.

Task:
- Decide whether the requested anchor lies on or near the intended candidate branch family.
- Identify which candidate semantic label the anchor best corresponds to.
- Rate anchor validity as one of: valid, weak, invalid, unknown.

Important rules:
- The anchor is a branch-terminal cue, not necessarily the exact GT endpoint.
- Do NOT infer GT semantics from this pass.
- Output JSON matching schema_version={VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION}.

Scene metadata:
{_json_block({
    "example_id": record.get("example_id"),
    "scenario_id": record.get("scenario_id"),
    "agent_id": record.get("agent_id"),
    "decision_time_idx": record.get("decision_time_idx"),
    "requested_branch_label": record.get("requested_branch_label"),
    "selected_candidate_id": record.get("selected_candidate_id"),
    "selected_candidate_geometry_branch_id": record.get("selected_candidate_geometry_branch_id"),
    "candidate_ids": _candidate_ids_only(record.get("candidate_id_map") or []),
    "frame_label": record.get("frame_label"),
})}

Focus on:
- anchor_audit
- contract_confidence
- notes

Leave gt_semantics conservative if uncertain.
""".strip()


def build_all_prompts(record: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "context_only": build_context_only_prompt(record),
        "context_plus_gt": build_context_plus_gt_prompt(record),
        "context_plus_anchor": build_context_plus_anchor_prompt(record),
    }
