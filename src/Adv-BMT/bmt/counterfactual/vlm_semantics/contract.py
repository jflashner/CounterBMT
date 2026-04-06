from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence


VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION = "vlm_semantic_contract_v1"
SEMANTIC_LABELS = ("left", "straight", "right", "u_turn", "unknown")
ANCHOR_VALIDITY_LABELS = ("valid", "weak", "invalid", "unknown")
AMBIGUITY_LEVELS = ("low", "medium", "high")
FRAME_LABELS = ("world", "agent_relative_at_decision")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return bool(default)


def _safe_label(value: Any, *, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else str(default)


def semantic_contract_json_schema() -> Dict[str, Any]:
    return {
        "name": VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string"},
                "example_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "agent_id": {"type": "string"},
                "image_set_ids": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "context_only": {"type": "array", "items": {"type": "string"}},
                        "context_plus_gt": {"type": "array", "items": {"type": "string"}},
                        "context_plus_anchor": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["context_only", "context_plus_gt", "context_plus_anchor"],
                },
                "model_name": {"type": "string"},
                "contract_confidence": {"type": "number"},
                "use_for_training": {"type": "boolean"},
                "scene_ambiguity": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "level": {"type": "string", "enum": list(AMBIGUITY_LEVELS)},
                        "confidence": {"type": "number"},
                        "rationale_short": {"type": "string"},
                    },
                    "required": ["level", "confidence", "rationale_short"],
                },
                "controlling_light_group": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "selected_light_group_id": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "alternatives_considered": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "light_group_id": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["light_group_id", "confidence"],
                            },
                        },
                    },
                    "required": ["selected_light_group_id", "confidence", "alternatives_considered"],
                },
                "split_point": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "x": {"type": ["number", "null"]},
                        "y": {"type": ["number", "null"]},
                        "frame": {"type": "string", "enum": list(FRAME_LABELS)},
                        "confidence": {"type": "number"},
                    },
                    "required": ["x", "y", "frame", "confidence"],
                },
                "candidate_semantics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "semantic_label": {"type": "string", "enum": list(SEMANTIC_LABELS)},
                            "confidence": {"type": "number"},
                            "is_valid_branch": {"type": "boolean"},
                            "rationale_short": {"type": "string"},
                        },
                        "required": [
                            "candidate_id",
                            "semantic_label",
                            "confidence",
                            "is_valid_branch",
                            "rationale_short",
                        ],
                    },
                },
                "exit_zones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "semantic_label": {"type": "string", "enum": list(SEMANTIC_LABELS)},
                            "polygon_xy": {
                                "type": ["array", "null"],
                                "items": {"type": "array", "items": {"type": "number"}},
                            },
                            "gate_centerline": {
                                "type": ["array", "null"],
                                "items": {"type": "array", "items": {"type": "number"}},
                            },
                            "frame": {"type": "string", "enum": list(FRAME_LABELS)},
                            "confidence": {"type": "number"},
                        },
                        "required": ["semantic_label", "polygon_xy", "gate_centerline", "frame", "confidence"],
                    },
                },
                "gt_semantics": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "gt_branch_candidate_id": {"type": ["string", "null"]},
                        "gt_semantic_label": {"type": "string", "enum": list(SEMANTIC_LABELS)},
                        "confidence": {"type": "number"},
                    },
                    "required": ["gt_branch_candidate_id", "gt_semantic_label", "confidence"],
                },
                "anchor_audit": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "requested_anchor_on_candidate_branch": {"type": ["boolean", "null"]},
                        "requested_anchor_semantic_label": {"type": "string", "enum": list(SEMANTIC_LABELS)},
                        "anchor_validity": {"type": "string", "enum": list(ANCHOR_VALIDITY_LABELS)},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "requested_anchor_on_candidate_branch",
                        "requested_anchor_semantic_label",
                        "anchor_validity",
                        "confidence",
                    ],
                },
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "schema_version",
                "example_id",
                "scenario_id",
                "agent_id",
                "image_set_ids",
                "model_name",
                "contract_confidence",
                "use_for_training",
                "scene_ambiguity",
                "controlling_light_group",
                "split_point",
                "candidate_semantics",
                "exit_zones",
                "gt_semantics",
                "anchor_audit",
                "notes",
            ],
        },
    }


def make_empty_contract(
    *,
    example_id: str,
    scenario_id: str,
    agent_id: str,
    image_set_ids: Optional[Mapping[str, Sequence[str]]] = None,
    model_name: str = "",
) -> Dict[str, Any]:
    return {
        "schema_version": VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION,
        "example_id": str(example_id),
        "scenario_id": str(scenario_id),
        "agent_id": str(agent_id),
        "image_set_ids": {
            "context_only": list((image_set_ids or {}).get("context_only", [])),
            "context_plus_gt": list((image_set_ids or {}).get("context_plus_gt", [])),
            "context_plus_anchor": list((image_set_ids or {}).get("context_plus_anchor", [])),
        },
        "model_name": str(model_name),
        "contract_confidence": 0.0,
        "use_for_training": False,
        "scene_ambiguity": {
            "level": "medium",
            "confidence": 0.0,
            "rationale_short": "",
        },
        "controlling_light_group": {
            "selected_light_group_id": None,
            "confidence": 0.0,
            "alternatives_considered": [],
        },
        "split_point": {
            "x": None,
            "y": None,
            "frame": "world",
            "confidence": 0.0,
        },
        "candidate_semantics": [],
        "exit_zones": [],
        "gt_semantics": {
            "gt_branch_candidate_id": None,
            "gt_semantic_label": "unknown",
            "confidence": 0.0,
        },
        "anchor_audit": {
            "requested_anchor_on_candidate_branch": None,
            "requested_anchor_semantic_label": "unknown",
            "anchor_validity": "unknown",
            "confidence": 0.0,
        },
        "notes": [],
    }


def normalize_contract(
    payload: Optional[Mapping[str, Any]],
    *,
    example_id: str,
    scenario_id: str,
    agent_id: str,
    image_set_ids: Optional[Mapping[str, Sequence[str]]] = None,
    model_name: str = "",
) -> Dict[str, Any]:
    contract = make_empty_contract(
        example_id=example_id,
        scenario_id=scenario_id,
        agent_id=agent_id,
        image_set_ids=image_set_ids,
        model_name=model_name,
    )
    if not payload:
        return contract

    normalized = copy.deepcopy(contract)
    normalized["schema_version"] = VLM_SEMANTIC_CONTRACT_SCHEMA_VERSION
    normalized["example_id"] = str(example_id)
    normalized["scenario_id"] = str(scenario_id)
    normalized["agent_id"] = str(agent_id)
    normalized["model_name"] = str(model_name or payload.get("model_name") or "")
    normalized["contract_confidence"] = _safe_float(payload.get("contract_confidence"), default=0.0)
    normalized["use_for_training"] = _safe_bool(payload.get("use_for_training"), default=False)

    normalized["image_set_ids"] = {
        "context_only": [str(value) for value in list((image_set_ids or {}).get("context_only", contract["image_set_ids"]["context_only"]))],
        "context_plus_gt": [str(value) for value in list((image_set_ids or {}).get("context_plus_gt", contract["image_set_ids"]["context_plus_gt"]))],
        "context_plus_anchor": [str(value) for value in list((image_set_ids or {}).get("context_plus_anchor", contract["image_set_ids"]["context_plus_anchor"]))],
    }

    scene_ambiguity = dict(payload.get("scene_ambiguity") or {})
    normalized["scene_ambiguity"] = {
        "level": _safe_label(scene_ambiguity.get("level"), allowed=AMBIGUITY_LEVELS, default="medium"),
        "confidence": _safe_float(scene_ambiguity.get("confidence"), default=0.0),
        "rationale_short": str(scene_ambiguity.get("rationale_short") or ""),
    }

    controlling_light_group = dict(payload.get("controlling_light_group") or {})
    normalized["controlling_light_group"] = {
        "selected_light_group_id": (
            None
            if controlling_light_group.get("selected_light_group_id") in (None, "")
            else str(controlling_light_group.get("selected_light_group_id"))
        ),
        "confidence": _safe_float(controlling_light_group.get("confidence"), default=0.0),
        "alternatives_considered": [
            {
                "light_group_id": str(item.get("light_group_id") or ""),
                "confidence": _safe_float(item.get("confidence"), default=0.0),
            }
            for item in list(controlling_light_group.get("alternatives_considered") or [])
            if str(item.get("light_group_id") or "").strip()
        ],
    }

    split_point = dict(payload.get("split_point") or {})
    normalized["split_point"] = {
        "x": None if split_point.get("x") is None else _safe_float(split_point.get("x"), default=0.0),
        "y": None if split_point.get("y") is None else _safe_float(split_point.get("y"), default=0.0),
        "frame": _safe_label(split_point.get("frame"), allowed=FRAME_LABELS, default="world"),
        "confidence": _safe_float(split_point.get("confidence"), default=0.0),
    }

    normalized["candidate_semantics"] = [
        {
            "candidate_id": str(entry.get("candidate_id") or ""),
            "semantic_label": _safe_label(entry.get("semantic_label"), allowed=SEMANTIC_LABELS, default="unknown"),
            "confidence": _safe_float(entry.get("confidence"), default=0.0),
            "is_valid_branch": _safe_bool(entry.get("is_valid_branch"), default=False),
            "rationale_short": str(entry.get("rationale_short") or ""),
        }
        for entry in list(payload.get("candidate_semantics") or [])
        if str(entry.get("candidate_id") or "").strip()
    ]

    normalized["exit_zones"] = []
    for entry in list(payload.get("exit_zones") or []):
        semantic_label = _safe_label(entry.get("semantic_label"), allowed=SEMANTIC_LABELS, default="unknown")
        polygon_xy = entry.get("polygon_xy")
        gate_centerline = entry.get("gate_centerline")
        normalized["exit_zones"].append(
            {
                "semantic_label": semantic_label,
                "polygon_xy": polygon_xy if isinstance(polygon_xy, list) else None,
                "gate_centerline": gate_centerline if isinstance(gate_centerline, list) else None,
                "frame": _safe_label(entry.get("frame"), allowed=FRAME_LABELS, default="world"),
                "confidence": _safe_float(entry.get("confidence"), default=0.0),
            }
        )

    gt_semantics = dict(payload.get("gt_semantics") or {})
    normalized["gt_semantics"] = {
        "gt_branch_candidate_id": None if gt_semantics.get("gt_branch_candidate_id") in (None, "") else str(gt_semantics.get("gt_branch_candidate_id")),
        "gt_semantic_label": _safe_label(gt_semantics.get("gt_semantic_label"), allowed=SEMANTIC_LABELS, default="unknown"),
        "confidence": _safe_float(gt_semantics.get("confidence"), default=0.0),
    }

    anchor_audit = dict(payload.get("anchor_audit") or {})
    normalized["anchor_audit"] = {
        "requested_anchor_on_candidate_branch": (
            None if anchor_audit.get("requested_anchor_on_candidate_branch") is None
            else _safe_bool(anchor_audit.get("requested_anchor_on_candidate_branch"), default=False)
        ),
        "requested_anchor_semantic_label": _safe_label(anchor_audit.get("requested_anchor_semantic_label"), allowed=SEMANTIC_LABELS, default="unknown"),
        "anchor_validity": _safe_label(anchor_audit.get("anchor_validity"), allowed=ANCHOR_VALIDITY_LABELS, default="unknown"),
        "confidence": _safe_float(anchor_audit.get("confidence"), default=0.0),
    }

    normalized["notes"] = [str(value) for value in list(payload.get("notes") or []) if str(value).strip()]
    return normalized


def should_escalate_contract(contract: Mapping[str, Any], *, confidence_threshold: float = 0.65) -> bool:
    confidence = _safe_float(contract.get("contract_confidence"), default=0.0)
    ambiguity = dict(contract.get("scene_ambiguity") or {})
    ambiguity_level = str(ambiguity.get("level") or "")
    anchor_audit = dict(contract.get("anchor_audit") or {})
    anchor_validity = str(anchor_audit.get("anchor_validity") or "")
    if confidence < float(confidence_threshold):
        return True
    if ambiguity_level == "high":
        return True
    if anchor_validity in {"unknown", "weak"} and _safe_float(anchor_audit.get("confidence"), default=0.0) < float(confidence_threshold):
        return True
    return False
