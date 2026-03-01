"""Shared DAG cache schema helpers for CounterBMT v2."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Mapping

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG


SCHEMA_VERSION = "counter_bmt_v2_dag_cache_v2_compact10"


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
    """Serialize a BayesianDAG into canonical v2 compact cache payload."""
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

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": str(dag.scenario_id),
        "nodes": nodes,
        "edges": edges,
        "cpts": _jsonify(dict(dag.cpts)),
        "metadata": metadata,
    }


def validate_cache_payload(payload: Mapping[str, Any]) -> bool:
    """Return True when payload satisfies compact v2 cache requirements."""
    if str(payload.get("schema_version", "")) != SCHEMA_VERSION:
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
    if str(metadata.get("contract_name", "")).strip() != "compact10":
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
