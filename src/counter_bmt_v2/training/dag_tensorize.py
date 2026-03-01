"""Tensorization utilities for DAG latent conditioning."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from counter_bmt_v2.training.dag_cache_schema import (
    SCHEMA_VERSION_V3_MANEUVER_OUTCOME,
    detect_schema_version,
)


_NODE_TYPE_TO_ID_V2 = {
    "context": 0,
    "ego_state": 1,
    "interaction": 2,
    "maneuver": 3,
    "decision": 4,
    "risk": 5,
    "outcome": 6,
}

_BEHAVIOR_CLASS_TO_ID_V2 = {
    "straight": 0,
    "left_turn": 1,
    "right_turn": 2,
    "lane_change_left": 3,
    "lane_change_right": 4,
    "stop": 5,
    "maintain_speed": 6,
    "accelerate": 7,
    "decelerate": 8,
    "yield_or_proceed": 9,
}

_MANEUVER_CLASS_TO_ID_V3 = {
    "straight": 0,
    "left_turn": 1,
    "right_turn": 2,
    "lane_change_left": 3,
    "lane_change_right": 4,
    "stop": 5,
    "accelerate": 6,
    "decelerate": 7,
    "yield": 8,
    "merge": 9,
    "u_turn": 10,
    "park": 11,
}

_OUTCOME_NODE_TO_ID_V3 = {
    "collision_outcome": 0,
    "progress_outcome": 1,
    "compliance_outcome": 2,
}


def _to_float_or_zero(x: Any) -> float:
    try:
        y = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(y):
        return 0.0
    return float(y)


def _behavior_onehot_v2(node_type: str, value: Any, metadata: Dict[str, Any]) -> np.ndarray:
    out = np.zeros((10,), dtype=np.float32)
    if node_type not in {"maneuver", "decision"}:
        return out
    cls = str(metadata.get("behavior_class", "")).strip().lower()
    if not cls:
        cls = str(value).strip().lower()
    idx = _BEHAVIOR_CLASS_TO_ID_V2.get(cls)
    if idx is None:
        idx = int(_BEHAVIOR_CLASS_TO_ID_V2["straight"] if node_type == "maneuver" else _BEHAVIOR_CLASS_TO_ID_V2["maintain_speed"])
    out[int(idx)] = 1.0
    return out


def _maneuver_onehot_v3(node_type: str, value: Any, metadata: Dict[str, Any]) -> np.ndarray:
    out = np.zeros((12,), dtype=np.float32)
    if node_type != "maneuver":
        return out
    cls = str(metadata.get("behavior_class", "")).strip().lower() or str(value).strip().lower()
    idx = _MANEUVER_CLASS_TO_ID_V3.get(cls, 0)
    out[int(idx)] = 1.0
    return out


def _outcome_onehot_v3(node_type: str, node_id: str) -> np.ndarray:
    out = np.zeros((3,), dtype=np.float32)
    if node_type != "outcome":
        return out
    idx = _OUTCOME_NODE_TO_ID_V3.get(str(node_id), None)
    if idx is not None:
        out[int(idx)] = 1.0
    return out


def _cpt_summary(cpt_spec: Dict[str, Any]) -> Tuple[float, float, float, float]:
    values = list(cpt_spec.get("values", []))
    parents = list(cpt_spec.get("parents", []))
    cpt = cpt_spec.get("cpt", {})
    rows = [row for row in cpt.values() if isinstance(row, dict)]
    if not rows and values:
        p = np.ones((len(values),), dtype=np.float32) / max(1.0, float(len(values)))
        return float(-np.sum(p * np.log(np.clip(p, 1e-8, 1.0)))), float(len(values)), float(len(parents)), 1.0

    ent: List[float] = []
    for row in rows:
        p = np.asarray([max(0.0, float(row.get(v, 0.0))) for v in values], dtype=np.float32)
        s = float(np.sum(p))
        if s <= 0.0:
            continue
        p = p / s
        ent.append(float(-np.sum(p * np.log(np.clip(p, 1e-8, 1.0)))))
    avg_ent = float(np.mean(ent)) if ent else 0.0
    return avg_ent, float(len(values)), float(len(parents)), float(len(rows))


def _edge_mechanism_bucket(mech: str) -> int:
    m = str(mech).lower()
    if "risk" in m:
        return 1
    if "speed" in m:
        return 2
    if "maneuver" in m:
        return 3
    if "outcome" in m:
        return 4
    if "decision" in m:
        return 5
    return 0


def _node_features_v2(
    *,
    node_id: str,
    node_type: str,
    value: Any,
    metadata: Dict[str, Any],
    timestamp: float,
    indeg: float,
    outdeg: float,
    cpt_spec: Dict[str, Any],
    d_node_in: int,
) -> np.ndarray:
    t_id = int(_NODE_TYPE_TO_ID_V2.get(node_type, 0))
    onehot = np.zeros((7,), dtype=np.float32)
    onehot[t_id] = 1.0

    behavior_oh = _behavior_onehot_v2(node_type, value, metadata)
    value_scalar = _to_float_or_zero(value)
    cpt_ent, _cpt_vals, cpt_par, cpt_rows = _cpt_summary(cpt_spec)
    core = np.asarray(
        [
            *onehot.tolist(),
            *behavior_oh.tolist(),
            float(timestamp),
            float(indeg),
            float(outdeg),
            float(value_scalar),
            float(cpt_ent),
            float(cpt_par),
            float(cpt_rows),
        ],
        dtype=np.float32,
    )
    if core.size >= d_node_in:
        return core[:d_node_in]
    out = np.zeros((int(d_node_in),), dtype=np.float32)
    out[: core.size] = core
    return out


def _node_features_v3(
    *,
    node_id: str,
    node_type: str,
    value: Any,
    metadata: Dict[str, Any],
    indeg: float,
    outdeg: float,
    horizon_s: float,
    d_node_in: int,
) -> np.ndarray:
    # Fixed 24-D packing for maneuver_outcome_v1:
    # type(2) + maneuver_class(12) + outcome_class(3) + observed(1) + interval(4) + degree(2)
    type_oh = np.zeros((2,), dtype=np.float32)
    if node_type == "maneuver":
        type_oh[0] = 1.0
    elif node_type == "outcome":
        type_oh[1] = 1.0

    maneuver_oh = _maneuver_onehot_v3(node_type, value, metadata)
    outcome_oh = _outcome_onehot_v3(node_type, node_id)

    observed = float(bool(metadata.get("observed", False)))

    start_s = _to_float_or_zero(metadata.get("start_s", 0.0))
    end_s = _to_float_or_zero(metadata.get("end_s", 0.0))
    duration_s = _to_float_or_zero(metadata.get("duration_s", max(0.0, end_s - start_s)))
    mid_s = _to_float_or_zero(metadata.get("mid_s", 0.5 * (start_s + end_s)))
    denom = float(max(1e-3, horizon_s))
    interval = np.asarray(
        [
            np.clip(start_s / denom, 0.0, 1.0),
            np.clip(end_s / denom, 0.0, 1.0),
            np.clip(duration_s / denom, 0.0, 1.0),
            np.clip(mid_s / denom, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    if node_type != "maneuver":
        interval[:] = 0.0

    degree = np.asarray([float(indeg), float(outdeg)], dtype=np.float32)
    core = np.concatenate([type_oh, maneuver_oh, outcome_oh, np.asarray([observed], dtype=np.float32), interval, degree], axis=0)
    if core.size != 24:
        raise ValueError(f"v3 node feature packing must be 24 dims, got {core.size}")
    if d_node_in != 24:
        out = np.zeros((int(d_node_in),), dtype=np.float32)
        out[: min(int(d_node_in), 24)] = core[: min(int(d_node_in), 24)]
        return out
    return core


def _tensorize_one(
    dag: Dict[str, Any],
    *,
    max_nodes: int,
    max_edges: int,
    d_node_in: int,
    d_edge_in: int,
) -> Dict[str, np.ndarray]:
    nodes = list(dag.get("nodes", []))
    edges = list(dag.get("edges", []))
    cpts = dict(dag.get("cpts", {}))
    schema_version = detect_schema_version(dag)
    is_v3 = schema_version == SCHEMA_VERSION_V3_MANEUVER_OUTCOME

    if not nodes:
        if is_v3:
            nodes = [
                {
                    "node_id": "maneuver_0",
                    "node_type": "maneuver",
                    "value": "straight",
                    "timestamp_s": 0.0,
                    "metadata": {
                        "start_s": 0.0,
                        "end_s": 0.0,
                        "duration_s": 0.0,
                        "mid_s": 0.0,
                        "observed": False,
                    },
                }
            ]
        else:
            nodes = [
                {"node_id": "empty", "node_type": "ego_state", "value": 0.0, "timestamp_s": 0.0, "metadata": {}}
            ]

    nodes = nodes[: int(max_nodes)]
    node_ids = [str(n.get("node_id", f"node_{i}")) for i, n in enumerate(nodes)]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(nodes)

    indeg = np.zeros((n,), dtype=np.float32)
    outdeg = np.zeros((n,), dtype=np.float32)
    for e in edges:
        u = str(e.get("parent_id", ""))
        v = str(e.get("child_id", ""))
        if u in idx and v in idx and u != v:
            outdeg[idx[u]] += 1.0
            indeg[idx[v]] += 1.0

    # Normalize interval against max maneuver end time for this graph.
    horizon_s = 1.0
    if is_v3:
        end_vals = []
        for nrec in nodes:
            if str(nrec.get("node_type", "")).lower() != "maneuver":
                continue
            md = nrec.get("metadata", {})
            if isinstance(md, dict):
                end_vals.append(_to_float_or_zero(md.get("end_s", 0.0)))
        if end_vals:
            horizon_s = float(max(1e-3, max(end_vals)))

    node_feat = np.zeros((int(max_nodes), int(d_node_in)), dtype=np.float32)
    node_mask = np.zeros((int(max_nodes),), dtype=bool)
    for i, nrec in enumerate(nodes):
        node_mask[i] = True
        node_id = str(nrec.get("node_id", f"node_{i}"))
        ntype = str(nrec.get("node_type", "")).lower()
        timestamp = float(nrec.get("timestamp_s") or 0.0)
        metadata = nrec.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        cpt_spec = cpts.get(node_id, {})
        if is_v3:
            node_feat[i, :] = _node_features_v3(
                node_id=node_id,
                node_type=ntype,
                value=nrec.get("value"),
                metadata=metadata,
                indeg=float(indeg[i]),
                outdeg=float(outdeg[i]),
                horizon_s=horizon_s,
                d_node_in=int(d_node_in),
            )
        else:
            node_feat[i, :] = _node_features_v2(
                node_id=node_id,
                node_type=ntype,
                value=nrec.get("value"),
                metadata=metadata,
                timestamp=timestamp,
                indeg=float(indeg[i]),
                outdeg=float(outdeg[i]),
                cpt_spec=cpt_spec if isinstance(cpt_spec, dict) else {},
                d_node_in=int(d_node_in),
            )

    edge_src = np.zeros((int(max_edges),), dtype=np.int32)
    edge_dst = np.zeros((int(max_edges),), dtype=np.int32)
    edge_feat = np.zeros((int(max_edges), int(d_edge_in)), dtype=np.float32)
    edge_mask = np.zeros((int(max_edges),), dtype=bool)

    e_i = 0
    for erec in edges:
        if e_i >= int(max_edges):
            break
        u = str(erec.get("parent_id", ""))
        v = str(erec.get("child_id", ""))
        if u not in idx or v not in idx or u == v:
            continue
        edge_mask[e_i] = True
        edge_src[e_i] = int(idx[u])
        edge_dst[e_i] = int(idx[v])
        conf = float(erec.get("confidence", 0.5))
        bucket = int(np.clip(_edge_mechanism_bucket(str(erec.get("mechanism", ""))), 0, 6))
        mech_oh = np.zeros((max(0, int(d_edge_in) - 2),), dtype=np.float32)
        if mech_oh.size > 0:
            mech_oh[min(bucket, mech_oh.size - 1)] = 1.0
        ef = np.concatenate([[conf, float(bucket)], mech_oh], axis=0).astype(np.float32)
        edge_feat[e_i, : min(edge_feat.shape[1], ef.size)] = ef[: edge_feat.shape[1]]
        e_i += 1

    global_feat = np.asarray(
        [
            float(n),
            float(np.sum(edge_mask)),
            float(np.mean(indeg[:n])) if n > 0 else 0.0,
            float(np.mean(outdeg[:n])) if n > 0 else 0.0,
        ],
        dtype=np.float32,
    )
    return {
        "dag_node_feat": node_feat,
        "dag_node_mask": node_mask,
        "dag_edge_src": edge_src,
        "dag_edge_dst": edge_dst,
        "dag_edge_feat": edge_feat,
        "dag_edge_mask": edge_mask,
        "dag_global_feat": global_feat,
    }


def tensorize_dag_batch(
    dags: Sequence[Dict[str, Any]],
    *,
    max_nodes: int,
    max_edges: int,
    d_node_in: int = 24,
    d_edge_in: int = 8,
) -> Dict[str, np.ndarray]:
    if not dags:
        raise ValueError("tensorize_dag_batch expects non-empty DAG list")

    one = [
        _tensorize_one(
            d,
            max_nodes=max_nodes,
            max_edges=max_edges,
            d_node_in=d_node_in,
            d_edge_in=d_edge_in,
        )
        for d in dags
    ]
    out: Dict[str, np.ndarray] = {}
    for k in one[0].keys():
        out[k] = np.stack([o[k] for o in one], axis=0)
    return out
