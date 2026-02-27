"""Tensorization utilities for DAG latent conditioning."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


_NODE_TYPE_TO_ID = {
    "ego_state": 0,
    "maneuver": 1,
    "decision": 2,
    "outcome": 3,
}


def _text_hash_feature(text: str, n: int = 8) -> np.ndarray:
    out = np.zeros((n,), dtype=np.float32)
    b = text.encode("utf-8")
    if not b:
        return out
    for i, v in enumerate(b):
        out[i % n] += (float(v % 29) / 28.0) - 0.5
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

    if not nodes:
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

    node_feat = np.zeros((int(max_nodes), int(d_node_in)), dtype=np.float32)
    node_mask = np.zeros((int(max_nodes),), dtype=bool)
    for i, nrec in enumerate(nodes):
        node_mask[i] = True
        t_id = int(_NODE_TYPE_TO_ID.get(str(nrec.get("node_type", "")).lower(), 0))
        onehot = np.zeros((4,), dtype=np.float32)
        onehot[t_id] = 1.0

        timestamp = float(nrec.get("timestamp_s") or 0.0)
        value_hash = _text_hash_feature(str(nrec.get("value", "")), n=8)
        cpt_spec = cpts.get(str(nrec.get("node_id", "")), {})
        cpt_ent, cpt_vals, cpt_par, cpt_rows = _cpt_summary(cpt_spec)
        core = np.asarray(
            [
                *onehot.tolist(),
                timestamp,
                float(indeg[i]),
                float(outdeg[i]),
                *value_hash.tolist(),
                cpt_ent,
                cpt_vals,
                cpt_par,
                cpt_rows,
            ],
            dtype=np.float32,
        )
        if core.size > d_node_in:
            node_feat[i, :] = core[:d_node_in]
        else:
            node_feat[i, : core.size] = core

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

