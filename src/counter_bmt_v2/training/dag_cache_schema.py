"""Shared DAG cache schema helpers for CounterBMT v2."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG

SCHEMA_VERSION_V2_COMPACT10 = "counter_bmt_v2_dag_cache_v2_compact10"
SCHEMA_VERSION_V3_MANEUVER_OUTCOME = "counter_bmt_v2_dag_cache_v3_maneuver_outcome"
SCHEMA_VERSION = SCHEMA_VERSION_V3_MANEUVER_OUTCOME
SUPPORTED_SCHEMA_VERSIONS = (
    SCHEMA_VERSION_V2_COMPACT10,
    SCHEMA_VERSION_V3_MANEUVER_OUTCOME,
)


def schema_version_for_contract(contract_name: str) -> str:
    name = str(contract_name).strip().lower()
    if name == "maneuver_outcome_v1":
        return SCHEMA_VERSION_V3_MANEUVER_OUTCOME
    return SCHEMA_VERSION_V2_COMPACT10


def detect_schema_version(payload: Mapping[str, Any]) -> str:
    return str(payload.get("schema_version", "")).strip()


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if is_dataclass(obj):
        return _jsonify(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def dag_to_cache_payload(dag: BayesianDAG) -> Dict[str, Any]:
    """Serialize a BayesianDAG into canonical cache payload."""
    nodes = []
    for node in dag.nodes.values():
        nodes.append(
            {
                "node_id": str(node.node_id),
                "node_type": str(node.node_type),
                "value": _jsonify(node.value),
                "timestamp_s": None if node.timestamp_s is None else float(node.timestamp_s),
                "metadata": _jsonify(dict(node.metadata)),
            }
        )

    edges = []
    for edge in dag.edges:
        edges.append(
            {
                "parent_id": str(edge.parent_id),
                "child_id": str(edge.child_id),
                "confidence": float(edge.confidence),
                "mechanism": str(edge.mechanism),
            }
        )

    metadata: Dict[str, Any] = {"source": "counter_bmt_v2"}
    extra_meta = getattr(dag, "_contract_metadata", None)
    if isinstance(extra_meta, Mapping):
        metadata.update(_jsonify(dict(extra_meta)))
    contract_name = str(metadata.get("contract_name", "")).strip()
    if not contract_name:
        node_types = {
            str(getattr(node, "node_type", "")).strip().lower()
            for node in dag.nodes.values()
        }
        if node_types and node_types.issubset({"maneuver", "outcome"}):
            contract_name = "maneuver_outcome_v1"
            metadata["contract_name"] = contract_name
            metadata.setdefault("contract_version", "1")
        else:
            contract_name = "compact10"
            metadata["contract_name"] = contract_name
            metadata.setdefault("contract_version", "1")
    schema_version = schema_version_for_contract(contract_name)

    return {
        "schema_version": schema_version,
        "scenario_id": str(dag.scenario_id),
        "nodes": nodes,
        "edges": edges,
        "cpts": _jsonify(dict(dag.cpts)),
        "metadata": metadata,
    }


def _normalize_allowed_schema_versions(
    allowed_schema_versions: Sequence[str] | Iterable[str] | None,
) -> set[str]:
    if allowed_schema_versions is None:
        return set(SUPPORTED_SCHEMA_VERSIONS)
    out = {str(v).strip() for v in allowed_schema_versions if str(v).strip()}
    return out if out else set(SUPPORTED_SCHEMA_VERSIONS)


def validate_cache_payload(
    payload: Mapping[str, Any],
    *,
    allowed_schema_versions: Sequence[str] | Iterable[str] | None = None,
) -> bool:
    """Return True when payload satisfies cache schema + contract requirements."""
    schema_version = detect_schema_version(payload)
    if schema_version not in _normalize_allowed_schema_versions(allowed_schema_versions):
        return False
    if not str(payload.get("scenario_id", "")).strip():
        return False
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False

    node_ids = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            return False
        nid = str(node.get("node_id", "")).strip()
        if not nid:
            return False
        node_ids.add(nid)

    for edge in edges:
        if not isinstance(edge, Mapping):
            return False
        pid = str(edge.get("parent_id", "")).strip()
        cid = str(edge.get("child_id", "")).strip()
        if not pid or not cid:
            return False
        if pid not in node_ids or cid not in node_ids:
            return False

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    contract_name = str(metadata.get("contract_name", "")).strip()
    expected_contract = ""
    if schema_version == SCHEMA_VERSION_V2_COMPACT10:
        expected_contract = "compact10"
    elif schema_version == SCHEMA_VERSION_V3_MANEUVER_OUTCOME:
        expected_contract = "maneuver_outcome_v1"
    if expected_contract and contract_name != expected_contract:
        return False
    if not str(metadata.get("contract_version", "")).strip():
        return False
    report = metadata.get("contract_report")
    if not isinstance(report, Mapping):
        return False
    if "passed" not in report:
        return False
    if not bool(report.get("passed")):
        return False

    return True
