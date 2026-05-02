from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION = "sdc_path_semantic_contract_v2"
SDC_PATH_LABELS = (
    "left",
    "right",
    "left_lane_change",
    "right_lane_change",
    "straight",
    "stop",
)
RISK_LEVELS = ("low", "medium", "high")
AMBIGUITY_LEVELS = ("low", "medium", "high")
SOURCE_KINDS = ("ground_truth", "sdc_path")
SLOT_IDS = ("gt", "alt_1", "alt_2", "alt_3")


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


def _safe_enum(value: Any, *, allowed: Sequence[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else str(default)


def sdc_path_semantic_json_schema() -> Dict[str, Any]:
    return {
        "name": SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string"},
                "example_id": {"type": "string"},
                "scenario_id": {"type": "string"},
                "sdc_id": {"type": "string"},
                "current_time_index": {"type": "integer"},
                "model_name": {"type": "string"},
                "prompt_version": {"type": "string"},
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
                "use_for_training": {"type": "boolean"},
                "highlighted_paths": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "slot_id": {"type": "string", "enum": list(SLOT_IDS)},
                            "source_kind": {"type": "string", "enum": list(SOURCE_KINDS)},
                            "path_id": {"type": ["string", "null"]},
                            "semantic_label": {"type": "string", "enum": list(SDC_PATH_LABELS)},
                            "risk_level": {"type": "string", "enum": list(RISK_LEVELS)},
                            "risk_rationale_short": {"type": "string"},
                            "confidence": {"type": "number"},
                            "is_valid_target": {"type": "boolean"},
                            "rationale_short": {"type": "string"},
                        },
                        "required": [
                            "slot_id",
                            "source_kind",
                            "path_id",
                            "semantic_label",
                            "risk_level",
                            "risk_rationale_short",
                            "confidence",
                            "is_valid_target",
                            "rationale_short",
                        ],
                    },
                },
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "schema_version",
                "example_id",
                "scenario_id",
                "sdc_id",
                "current_time_index",
                "model_name",
                "prompt_version",
                "scene_ambiguity",
                "use_for_training",
                "highlighted_paths",
                "notes",
            ],
        },
    }


def make_empty_sdc_path_contract(
    *,
    example_id: str,
    scenario_id: str,
    sdc_id: str,
    current_time_index: int,
    model_name: str = "",
    prompt_version: str = "sdc_path_semantics_v1",
) -> Dict[str, Any]:
    return {
        "schema_version": SDC_PATH_SEMANTIC_CONTRACT_SCHEMA_VERSION,
        "example_id": str(example_id),
        "scenario_id": str(scenario_id),
        "sdc_id": str(sdc_id),
        "current_time_index": int(current_time_index),
        "model_name": str(model_name),
        "prompt_version": str(prompt_version),
        "scene_ambiguity": {
            "level": "medium",
            "confidence": 0.0,
            "rationale_short": "",
        },
        "use_for_training": False,
        "highlighted_paths": [],
        "notes": [],
    }


def normalize_sdc_path_contract(
    payload: Optional[Mapping[str, Any]],
    *,
    example_id: str,
    scenario_id: str,
    sdc_id: str,
    current_time_index: int,
    model_name: str,
    prompt_version: str = "sdc_path_semantics_v1",
) -> Dict[str, Any]:
    contract = make_empty_sdc_path_contract(
        example_id=example_id,
        scenario_id=scenario_id,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
        model_name=model_name,
        prompt_version=prompt_version,
    )
    payload_dict = dict(payload or {})
    contract["schema_version"] = str(payload_dict.get("schema_version") or contract["schema_version"])
    contract["model_name"] = str(model_name)
    contract["prompt_version"] = str(prompt_version)
    ambiguity = dict(payload_dict.get("scene_ambiguity") or {})
    contract["scene_ambiguity"] = {
        "level": _safe_enum(ambiguity.get("level"), allowed=AMBIGUITY_LEVELS, default="medium"),
        "confidence": _safe_float(ambiguity.get("confidence"), 0.0),
        "rationale_short": str(ambiguity.get("rationale_short") or ""),
    }
    contract["use_for_training"] = _safe_bool(payload_dict.get("use_for_training"), default=False)

    normalized_paths: List[Dict[str, Any]] = []
    for item in list(payload_dict.get("highlighted_paths") or []):
        row = dict(item or {})
        normalized_paths.append(
            {
                "slot_id": _safe_enum(row.get("slot_id"), allowed=SLOT_IDS, default="gt"),
                "source_kind": _safe_enum(row.get("source_kind"), allowed=SOURCE_KINDS, default="sdc_path"),
                "path_id": None if row.get("path_id") is None else str(row.get("path_id")),
                "semantic_label": _safe_enum(row.get("semantic_label"), allowed=SDC_PATH_LABELS, default="straight"),
                "risk_level": _safe_enum(row.get("risk_level"), allowed=RISK_LEVELS, default="medium"),
                "risk_rationale_short": str(row.get("risk_rationale_short") or ""),
                "confidence": _safe_float(row.get("confidence"), 0.0),
                "is_valid_target": _safe_bool(row.get("is_valid_target"), default=False),
                "rationale_short": str(row.get("rationale_short") or ""),
            }
        )
    normalized_paths.sort(key=lambda row: SLOT_IDS.index(row["slot_id"]) if row["slot_id"] in SLOT_IDS else 99)
    contract["highlighted_paths"] = normalized_paths
    contract["notes"] = [str(note) for note in list(payload_dict.get("notes") or []) if str(note).strip()]
    return contract
