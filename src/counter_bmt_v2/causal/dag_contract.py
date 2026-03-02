"""Compact DAG contract enforcement for stable latent learning."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from counter_bmt_v2.contracts import BayesianDAG, DAGEdge, DAGNode


_MANEUVER_CLASSES: Tuple[str, ...] = (
    "straight",
    "left_turn",
    "right_turn",
    "lane_change_left",
    "lane_change_right",
    "stop",
)

_MANEUVER_CLASSES_COMPACT12: Tuple[str, ...] = (
    "straight",
    "left_turn",
    "right_turn",
    "lane_change_left",
    "lane_change_right",
    "stop",
    "accelerate",
    "decelerate",
    "yield",
    "merge",
    "u_turn",
    "park",
)

_OUTCOME_NODE_IDS_MO: Tuple[str, ...] = (
    "collision_outcome",
    "progress_outcome",
    "compliance_outcome",
)

_OUTCOME_VALUES_MO: Dict[str, Tuple[str, ...]] = {
    "collision_outcome": ("collision_avoided", "collision_possible"),
    "progress_outcome": ("progress_good", "progress_limited"),
    "compliance_outcome": ("compliant", "violation_possible"),
}

_DECISION_CLASSES: Tuple[str, ...] = (
    "maintain_speed",
    "accelerate",
    "decelerate",
    "yield_or_proceed",
)

_OUTCOME_CLASSES: Tuple[str, ...] = ("collision_avoided", "collision_possible")

_TYPE_ORDER: Tuple[str, ...] = (
    "context",
    "ego_state",
    "interaction",
    "maneuver",
    "decision",
    "risk",
    "outcome",
)

_TYPE_TO_TIER: Dict[str, int] = {
    "context": 0,
    "ego_state": 0,
    "interaction": 1,
    "maneuver": 1,
    "decision": 1,
    "risk": 1,
    "outcome": 2,
}

_MECHANISM_BUCKETS: Tuple[str, ...] = (
    "speed_to_maneuver",
    "maneuver_to_decision",
    "decision_to_outcome",
    "maneuver_to_outcome",
    "risk_to_outcome",
    "context_to_decision",
    "interaction_to_maneuver",
    "generic",
)


@dataclass(frozen=True)
class DAGContractConfig:
    name: str = "compact10"
    version: str = "1"
    mode: str = "hard"
    max_nodes: int = 14
    max_edges: int = 20
    max_parents_per_node: int = 3
    max_outgoing_per_node: int = 5
    max_depth: int = 4


@dataclass
class DAGContractViolation:
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGContractReport:
    contract_name: str
    contract_version: str
    mode: str
    passed: bool
    before_nodes: int
    after_nodes: int
    before_edges: int
    after_edges: int
    max_depth: int
    density: float
    normalization_counts: Dict[str, int] = field(default_factory=dict)
    violation_counts: Dict[str, int] = field(default_factory=dict)
    violations: List[DAGContractViolation] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "mode": self.mode,
            "passed": bool(self.passed),
            "before_nodes": int(self.before_nodes),
            "after_nodes": int(self.after_nodes),
            "before_edges": int(self.before_edges),
            "after_edges": int(self.after_edges),
            "max_depth": int(self.max_depth),
            "density": float(self.density),
            "normalization_counts": dict(self.normalization_counts),
            "violation_counts": dict(self.violation_counts),
            "num_violations": int(len(self.violations)),
        }


def _to_str(x: Any) -> str:
    return str(x).strip()


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(x)
    except Exception:
        return None
    if not math.isfinite(y):
        return None
    return y


def _normalize_node_type(node_type: str, node_id: str) -> Optional[str]:
    t = _to_str(node_type).lower().replace("-", "_").replace(" ", "_")
    nid = _to_str(node_id).lower()
    if t in _TYPE_TO_TIER:
        return t
    if "ego" in t or nid == "ego_initial_speed":
        return "ego_state"
    if "maneuver" in t or "lane_change" in t or "turn" in t:
        return "maneuver"
    if "decision" in t or "yield" in t or "speed_choice" in t:
        return "decision"
    if "outcome" in t or "collision" in t:
        return "outcome"
    if "risk" in t:
        return "risk"
    if "interaction" in t:
        return "interaction"
    if "context" in t:
        return "context"
    return None


def _normalize_maneuver(v: str) -> Tuple[str, Optional[str]]:
    x = _to_str(v).lower().replace("-", "_").replace(" ", "_")
    if x in _MANEUVER_CLASSES:
        return x, None
    if "lane" in x and "left" in x:
        return "lane_change_left", "mapped_lane_left"
    if "lane" in x and "right" in x:
        return "lane_change_right", "mapped_lane_right"
    if "left" in x and "turn" in x:
        return "left_turn", "mapped_left_turn"
    if "right" in x and "turn" in x:
        return "right_turn", "mapped_right_turn"
    if "stop" in x or "halt" in x:
        return "stop", "mapped_stop"
    if "straight" in x or "follow" in x or "cruise" in x:
        return "straight", "mapped_straight"
    return "straight", "fallback_straight"


def _normalize_maneuver_compact12(v: str) -> Tuple[str, Optional[str]]:
    x = _to_str(v).lower().replace("-", "_").replace(" ", "_")
    if x in _MANEUVER_CLASSES_COMPACT12:
        return x, None
    if "lane" in x and "left" in x:
        return "lane_change_left", "mapped_lane_left"
    if "lane" in x and "right" in x:
        return "lane_change_right", "mapped_lane_right"
    if "left" in x and "turn" in x:
        return "left_turn", "mapped_left_turn"
    if "right" in x and "turn" in x:
        return "right_turn", "mapped_right_turn"
    if "u_turn" in x or "uturn" in x:
        return "u_turn", "mapped_u_turn"
    if "merge" in x:
        return "merge", "mapped_merge"
    if "yield" in x or "wait" in x:
        return "yield", "mapped_yield"
    if "park" in x or "idle" in x:
        return "park", "mapped_park"
    if "acceler" in x or "speed_up" in x:
        return "accelerate", "mapped_accelerate"
    if "deceler" in x or "slow" in x or "brake" in x:
        return "decelerate", "mapped_decelerate"
    if "stop" in x or "halt" in x:
        return "stop", "mapped_stop"
    if "straight" in x or "follow" in x or "cruise" in x:
        return "straight", "mapped_straight"
    return "straight", "fallback_straight"


def _normalize_outcome_node_id(node_id: str, value: Any) -> Tuple[Optional[str], Optional[str]]:
    nid = _to_str(node_id).lower().replace("-", "_").replace(" ", "_")
    val = _to_str(value).lower().replace("-", "_").replace(" ", "_")
    if nid in _OUTCOME_NODE_IDS_MO:
        return nid, None
    if "collision" in nid or "collision" in val or "crash" in val:
        return "collision_outcome", "mapped_collision_outcome"
    if "progress" in nid or "arrival" in nid or "goal" in nid:
        return "progress_outcome", "mapped_progress_outcome"
    if "compliance" in nid or "rule" in nid or "legal" in nid or "violation" in val:
        return "compliance_outcome", "mapped_compliance_outcome"
    return None, None


def _normalize_outcome_value_mo(outcome_id: str, value: Any) -> Tuple[str, Optional[str]]:
    valid = _OUTCOME_VALUES_MO.get(str(outcome_id), ())
    x = _to_str(value).lower().replace("-", "_").replace(" ", "_")
    if x in valid:
        return x, None
    if outcome_id == "collision_outcome":
        if "possible" in x or "collision" in x or "crash" in x or "unsafe" in x:
            return "collision_possible", "mapped_collision_possible"
        return "collision_avoided", "mapped_collision_avoided"
    if outcome_id == "progress_outcome":
        if "limited" in x or "blocked" in x or "stuck" in x or "slow" in x:
            return "progress_limited", "mapped_progress_limited"
        return "progress_good", "mapped_progress_good"
    if outcome_id == "compliance_outcome":
        if "violation" in x or "non" in x or "illegal" in x:
            return "violation_possible", "mapped_violation_possible"
        return "compliant", "mapped_compliant"
    return "unknown", "mapped_unknown_outcome"


def _normalize_decision(v: str) -> Tuple[str, Optional[str]]:
    x = _to_str(v).lower().replace("-", "_").replace(" ", "_")
    if x in _DECISION_CLASSES:
        return x, None
    if "maintain" in x or "keep" in x:
        return "maintain_speed", "mapped_maintain"
    if "acceler" in x or "speed_up" in x:
        return "accelerate", "mapped_accelerate"
    if "deceler" in x or "slow" in x or "brake" in x:
        return "decelerate", "mapped_decelerate"
    if "yield" in x or "proceed" in x or "gap" in x:
        return "yield_or_proceed", "mapped_yield_or_proceed"
    return "maintain_speed", "fallback_maintain_speed"


def _normalize_outcome(v: str) -> Tuple[str, Optional[str]]:
    x = _to_str(v).lower().replace("-", "_").replace(" ", "_")
    if x in _OUTCOME_CLASSES:
        return x, None
    if "avoid" in x or "safe" in x or "no_collision" in x:
        return "collision_avoided", "mapped_collision_avoided"
    if "collision" in x or "crash" in x or "conflict" in x:
        return "collision_possible", "mapped_collision_possible"
    return "collision_avoided", "fallback_collision_avoided"


def _normalize_mechanism(m: str) -> str:
    x = _to_str(m).lower().replace(" ", "_")
    if "speed" in x and ("maneuver" in x or "turn" in x or "lane" in x):
        return "speed_to_maneuver"
    if "maneuver" in x and "decision" in x:
        return "maneuver_to_decision"
    if "decision" in x and ("outcome" in x or "collision" in x):
        return "decision_to_outcome"
    if "maneuver" in x and ("outcome" in x or "collision" in x):
        return "maneuver_to_outcome"
    if "risk" in x and ("outcome" in x or "collision" in x):
        return "risk_to_outcome"
    if "context" in x and "decision" in x:
        return "context_to_decision"
    if "interaction" in x and "maneuver" in x:
        return "interaction_to_maneuver"
    if x in _MECHANISM_BUCKETS:
        return x
    return "generic"


def _value_to_canonical(node_type: str, value: Any) -> Tuple[Any, Optional[str]]:
    if node_type == "maneuver":
        return _normalize_maneuver(_to_str(value))
    if node_type == "decision":
        return _normalize_decision(_to_str(value))
    if node_type == "outcome":
        return _normalize_outcome(_to_str(value))
    if node_type == "risk":
        f = _to_float(value)
        if f is None:
            return 0.0, "mapped_risk_zero"
        return float(max(0.0, min(1.0, f))), None
    if node_type == "ego_state":
        f = _to_float(value)
        if f is None:
            return 0.0, "mapped_ego_state_zero"
        return float(f), None
    return _to_str(value), None


def _topological_depth(node_ids: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Tuple[bool, int]:
    indeg = {n: 0 for n in node_ids}
    adj: Dict[str, List[str]] = {n: [] for n in node_ids}
    for u, v in edges:
        if u not in indeg or v not in indeg or u == v:
            return False, 0
        indeg[v] += 1
        adj[u].append(v)
    q = [n for n in node_ids if indeg[n] == 0]
    depth = {n: 0 for n in node_ids}
    seen = 0
    while q:
        cur = q.pop(0)
        seen += 1
        for nxt in adj[cur]:
            depth[nxt] = max(depth[nxt], depth[cur] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if seen != len(node_ids):
        return False, 0
    return True, int(max(depth.values()) if depth else 0)


def _normalize_distribution(values: Sequence[str], row: Mapping[str, Any]) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    total = 0.0
    for v in values:
        p = _to_float(row.get(v))
        if p is None or p < 0.0:
            p = 0.0
        probs[str(v)] = float(p)
        total += float(p)
    if total <= 0.0:
        u = 1.0 / float(max(1, len(values)))
        return {str(v): float(u) for v in values}
    return {str(v): float(probs[str(v)] / total) for v in values}


def _default_values_for_type(node_type: str) -> List[str]:
    if node_type == "maneuver":
        return list(_MANEUVER_CLASSES)
    if node_type == "decision":
        return list(_DECISION_CLASSES)
    if node_type == "outcome":
        return list(_OUTCOME_CLASSES)
    return ["unknown"]


def _default_outcome_values_mo(node_id: str) -> List[str]:
    return list(_OUTCOME_VALUES_MO.get(str(node_id), ("unknown",)))


def _canonicalize_maneuver_outcome_payload(
    payload: Mapping[str, Any],
    cfg: DAGContractConfig,
) -> Tuple[Dict[str, Any], DAGContractReport]:
    p = copy.deepcopy(dict(payload))
    violations: List[DAGContractViolation] = []
    normalization_counts: Dict[str, int] = {}

    raw_nodes = p.get("nodes", [])
    raw_edges = p.get("edges", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []
    before_nodes = len(raw_nodes)
    before_edges = len(raw_edges)

    canonical_nodes: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for i, rec in enumerate(raw_nodes):
        if not isinstance(rec, Mapping):
            violations.append(
                DAGContractViolation("invalid_node", "Node entry is not a mapping", {"index": i})
            )
            continue
        raw_id = _to_str(rec.get("node_id", f"node_{i}")) or f"node_{i}"
        ntype = _to_str(rec.get("node_type", "")).lower().replace("-", "_").replace(" ", "_")
        value = rec.get("value")
        metadata = dict(rec.get("metadata", {})) if isinstance(rec.get("metadata", {}), Mapping) else {}
        ts = _to_float(rec.get("timestamp_s"))

        if ntype == "maneuver":
            cls, reason = _normalize_maneuver_compact12(_to_str(value))
            if reason:
                normalization_counts[reason] = int(normalization_counts.get(reason, 0) + 1)
            start_s = _to_float(metadata.get("start_s"))
            end_s = _to_float(metadata.get("end_s"))
            has_start = "start_s" in metadata
            has_end = "end_s" in metadata
            if (not has_start) or (not has_end):
                violations.append(
                    DAGContractViolation(
                        "invalid_interval",
                        "Maneuver node is missing required interval metadata (start_s/end_s)",
                        {"node_id": raw_id, "has_start_s": has_start, "has_end_s": has_end},
                    )
                )
                continue
            if start_s is None or end_s is None or end_s < start_s:
                violations.append(
                    DAGContractViolation(
                        "invalid_interval",
                        "Maneuver node requires valid start/end timestamps",
                        {"node_id": raw_id, "start_s": start_s, "end_s": end_s},
                    )
                )
                continue
            duration_s = float(max(0.0, end_s - start_s))
            mid_s = float(0.5 * (start_s + end_s))
            metadata["start_s"] = float(start_s)
            metadata["end_s"] = float(end_s)
            metadata["duration_s"] = duration_s
            metadata["mid_s"] = mid_s
            metadata["behavior_class"] = cls
            metadata.setdefault("alternatives", list(_MANEUVER_CLASSES_COMPACT12))
            if raw_id in seen_ids:
                raw_id = f"{raw_id}_{i}"
                normalization_counts["dedup_node_id"] = int(normalization_counts.get("dedup_node_id", 0) + 1)
            seen_ids.add(raw_id)
            canonical_nodes.append(
                {
                    "node_id": raw_id,
                    "node_type": "maneuver",
                    "value": cls,
                    "timestamp_s": mid_s,
                    "metadata": metadata,
                    "_old_id": raw_id,
                    "_sort": (0, float(start_s), float(end_s), raw_id),
                }
            )
            continue

        if ntype == "outcome":
            out_id, reason = _normalize_outcome_node_id(raw_id, value)
            if reason:
                normalization_counts[reason] = int(normalization_counts.get(reason, 0) + 1)
            if out_id is None:
                violations.append(
                    DAGContractViolation(
                        "unknown_outcome_type",
                        "Outcome node id/value cannot be mapped to required anchors",
                        {"node_id": raw_id, "value": _to_str(value)},
                    )
                )
                continue
            out_val, reason = _normalize_outcome_value_mo(out_id, value)
            if reason:
                normalization_counts[reason] = int(normalization_counts.get(reason, 0) + 1)
            metadata.setdefault("alternatives", _default_outcome_values_mo(out_id))
            if out_id in seen_ids:
                normalization_counts["dedup_outcome_node"] = int(
                    normalization_counts.get("dedup_outcome_node", 0) + 1
                )
                continue
            seen_ids.add(out_id)
            canonical_nodes.append(
                {
                    "node_id": out_id,
                    "node_type": "outcome",
                    "value": out_val,
                    "timestamp_s": None,
                    "metadata": metadata,
                    "_old_id": raw_id,
                    "_sort": (1, 0.0, 0.0, out_id),
                }
            )
            continue

        violations.append(
            DAGContractViolation(
                "invalid_node_type",
                "Only maneuver/outcome nodes are allowed for maneuver_outcome_v1",
                {"node_id": raw_id, "node_type": ntype},
            )
        )

    maneuvers = [n for n in canonical_nodes if n["node_type"] == "maneuver"]
    outcomes = [n for n in canonical_nodes if n["node_type"] == "outcome"]
    maneuvers.sort(key=lambda r: r["_sort"])
    outcomes.sort(key=lambda r: r["_sort"])

    if len(maneuvers) <= 0:
        violations.append(
            DAGContractViolation("missing_anchor", "At least one maneuver node required", {})
        )
    if len(maneuvers) > 8:
        normalization_counts["capped_maneuver_nodes"] = int(len(maneuvers) - 8)
        maneuvers = maneuvers[:8]

    outcome_map = {n["node_id"]: n for n in outcomes}
    for req in _OUTCOME_NODE_IDS_MO:
        if req not in outcome_map:
            violations.append(
                DAGContractViolation(
                    "missing_required_outcome",
                    "Required outcome node missing",
                    {"node_id": req},
                )
            )

    # Canonical IDs and deterministic order.
    canonical_nodes_out: List[Dict[str, Any]] = []
    maneuver_old_to_new: Dict[str, str] = {}
    id_map: Dict[str, str] = {}
    for i, rec in enumerate(maneuvers):
        old = str(rec.get("_old_id", rec["node_id"]))
        new = f"maneuver_{i}"
        maneuver_old_to_new[old] = new
        id_map[old] = new
        if old != new:
            normalization_counts["renamed_node_id"] = int(normalization_counts.get("renamed_node_id", 0) + 1)
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        out["node_id"] = new
        canonical_nodes_out.append(out)

    for req in _OUTCOME_NODE_IDS_MO:
        if req not in outcome_map:
            continue
        rec = outcome_map[req]
        old = str(rec.get("_old_id", req))
        id_map[old] = req
        out = {k: v for k, v in rec.items() if not k.startswith("_")}
        out["node_id"] = req
        out["timestamp_s"] = None
        canonical_nodes_out.append(out)

    by_new_id = {n["node_id"]: n for n in canonical_nodes_out}
    id_map.update(maneuver_old_to_new)
    for req in _OUTCOME_NODE_IDS_MO:
        if req in outcome_map:
            id_map[req] = req

    # Canonical edges: only maneuver -> outcome.
    edge_best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for i, rec in enumerate(raw_edges):
        if not isinstance(rec, Mapping):
            violations.append(
                DAGContractViolation("invalid_edge", "Edge entry is not a mapping", {"index": i})
            )
            continue
        u_old = _to_str(rec.get("parent_id", ""))
        v_old = _to_str(rec.get("child_id", ""))
        if u_old not in id_map or v_old not in id_map:
            continue
        u = id_map[u_old]
        v = id_map[v_old]
        if u == v:
            normalization_counts["drop_self_edge"] = int(normalization_counts.get("drop_self_edge", 0) + 1)
            continue
        u_type = by_new_id.get(u, {}).get("node_type", "")
        v_type = by_new_id.get(v, {}).get("node_type", "")
        if not (u_type == "maneuver" and v_type == "outcome"):
            violations.append(
                DAGContractViolation(
                    "invalid_edge_type",
                    "Only maneuver->outcome edges are allowed",
                    {"parent_id": u, "child_id": v, "parent_type": u_type, "child_type": v_type},
                )
            )
            continue
        conf = _to_float(rec.get("confidence"))
        if conf is None:
            conf = 0.7
        conf = float(max(0.0, min(1.0, conf)))
        mech = _normalize_mechanism(_to_str(rec.get("mechanism", "")))
        key = (u, v)
        prev = edge_best.get(key)
        if prev is None or conf > float(prev["confidence"]):
            edge_best[key] = {"parent_id": u, "child_id": v, "confidence": conf, "mechanism": mech}
        elif conf == float(prev["confidence"]) and mech < str(prev["mechanism"]):
            edge_best[key] = {"parent_id": u, "child_id": v, "confidence": conf, "mechanism": mech}
    canonical_edges = sorted(edge_best.values(), key=lambda r: (r["parent_id"], r["child_id"]))
    parents_count: Dict[str, int] = {}
    outgoing_count: Dict[str, int] = {}
    for e in canonical_edges:
        u = str(e["parent_id"])
        v = str(e["child_id"])
        parents_count[v] = int(parents_count.get(v, 0) + 1)
        outgoing_count[u] = int(outgoing_count.get(u, 0) + 1)
    for node_id, count in parents_count.items():
        if count > int(cfg.max_parents_per_node):
            violations.append(
                DAGContractViolation(
                    "max_parents_exceeded",
                    "Node exceeds max_parents_per_node",
                    {"node_id": node_id, "count": count, "limit": int(cfg.max_parents_per_node)},
                )
            )
    for node_id, count in outgoing_count.items():
        if count > int(cfg.max_outgoing_per_node):
            violations.append(
                DAGContractViolation(
                    "max_outgoing_exceeded",
                    "Node exceeds max_outgoing_per_node",
                    {"node_id": node_id, "count": count, "limit": int(cfg.max_outgoing_per_node)},
                )
            )

    node_ids = [n["node_id"] for n in canonical_nodes_out]
    edge_pairs = [(e["parent_id"], e["child_id"]) for e in canonical_edges]
    is_dag, depth = _topological_depth(node_ids, edge_pairs)
    if not is_dag:
        violations.append(DAGContractViolation("cycle_detected", "Graph is cyclic", {}))
        depth = 0
    if depth > int(cfg.max_depth):
        violations.append(
            DAGContractViolation(
                "max_depth_exceeded",
                "Graph depth exceeds contract bound",
                {"depth": int(depth), "limit": int(cfg.max_depth)},
            )
        )
    if len(canonical_nodes_out) > int(cfg.max_nodes):
        violations.append(
            DAGContractViolation(
                "max_nodes_exceeded",
                "Node count exceeds contract bound",
                {"count": len(canonical_nodes_out), "limit": int(cfg.max_nodes)},
            )
        )
    if len(canonical_edges) > int(cfg.max_edges):
        violations.append(
            DAGContractViolation(
                "max_edges_exceeded",
                "Edge count exceeds contract bound",
                {"count": len(canonical_edges), "limit": int(cfg.max_edges)},
            )
        )

    # CPT canonicalization after ID mapping.
    raw_cpts = p.get("cpts", {})
    if not isinstance(raw_cpts, Mapping):
        raw_cpts = {}
    rev_map: Dict[str, List[str]] = {}
    for old_id, new_id in id_map.items():
        rev_map.setdefault(new_id, []).append(old_id)

    cpts_out: Dict[str, Dict[str, Any]] = {}
    for node in canonical_nodes_out:
        nid = node["node_id"]
        ntype = node["node_type"]
        spec_raw = None
        for old in rev_map.get(nid, []):
            if old in raw_cpts and isinstance(raw_cpts[old], Mapping):
                spec_raw = raw_cpts[old]
                break
        if spec_raw is None:
            spec_raw = {}

        if ntype == "maneuver":
            values = list(_MANEUVER_CLASSES_COMPACT12)
            node_val = _to_str(node.get("value", ""))
            if node_val and node_val not in values:
                values.append(node_val)
        else:
            values = _default_outcome_values_mo(nid)
            node_val = _to_str(node.get("value", ""))
            if node_val and node_val not in values:
                values.append(node_val)
        values = list(dict.fromkeys(values))

        rows_raw = spec_raw.get("cpt", {}) if isinstance(spec_raw, Mapping) else {}
        rows: Dict[str, Dict[str, float]] = {}
        if isinstance(rows_raw, Mapping):
            for key, row in rows_raw.items():
                if not isinstance(row, Mapping):
                    continue
                rows[_to_str(key) or "*"] = _normalize_distribution(values, row)
        if not rows:
            rows["*"] = _normalize_distribution(values, {})
            normalization_counts["filled_missing_cpt"] = int(
                normalization_counts.get("filled_missing_cpt", 0) + 1
            )
        parents = [e["parent_id"] for e in canonical_edges if e["child_id"] == nid]
        cpts_out[nid] = {"values": values, "parents": parents, "cpt": rows}

    density = 0.0
    n_nodes = float(len(canonical_nodes_out))
    if n_nodes > 1:
        density = float(len(canonical_edges) / (n_nodes * (n_nodes - 1.0)))

    violation_counts: Dict[str, int] = {}
    for v in violations:
        violation_counts[v.code] = int(violation_counts.get(v.code, 0) + 1)

    passed = len(violations) == 0
    report = DAGContractReport(
        contract_name=str(cfg.name),
        contract_version=str(cfg.version),
        mode=str(cfg.mode),
        passed=bool(passed),
        before_nodes=int(before_nodes),
        after_nodes=int(len(canonical_nodes_out)),
        before_edges=int(before_edges),
        after_edges=int(len(canonical_edges)),
        max_depth=int(depth),
        density=float(density),
        normalization_counts=normalization_counts,
        violation_counts=violation_counts,
        violations=violations,
    )

    metadata = dict(p.get("metadata", {})) if isinstance(p.get("metadata", {}), Mapping) else {}
    metadata["contract_name"] = str(cfg.name)
    metadata["contract_version"] = str(cfg.version)
    metadata["contract_mode"] = str(cfg.mode)
    metadata["contract_report"] = report.summary()

    out_payload = {
        "schema_version": _to_str(p.get("schema_version", "")),
        "scenario_id": _to_str(p.get("scenario_id", "")),
        "nodes": canonical_nodes_out,
        "edges": canonical_edges,
        "cpts": cpts_out,
        "metadata": metadata,
    }
    return out_payload, report


def canonicalize_dag_payload(
    payload: Mapping[str, Any],
    config: DAGContractConfig | None = None,
) -> Tuple[Dict[str, Any], DAGContractReport]:
    cfg = config or DAGContractConfig()
    if str(cfg.name) == "maneuver_outcome_v1":
        cfg = DAGContractConfig(
            name=str(cfg.name),
            version=str(cfg.version),
            mode=str(cfg.mode),
            max_nodes=11,
            max_edges=24,
            max_parents_per_node=8,
            max_outgoing_per_node=3,
            max_depth=2,
        )
        return _canonicalize_maneuver_outcome_payload(payload, cfg)
    p = copy.deepcopy(dict(payload))
    violations: List[DAGContractViolation] = []
    normalization_counts: Dict[str, int] = {}

    raw_nodes = p.get("nodes", [])
    raw_edges = p.get("edges", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []
    if not isinstance(raw_edges, list):
        raw_edges = []
    before_nodes = len(raw_nodes)
    before_edges = len(raw_edges)

    canonical_nodes: List[Dict[str, Any]] = []
    seen_node_ids = set()
    for i, n in enumerate(raw_nodes):
        if not isinstance(n, Mapping):
            violations.append(DAGContractViolation("invalid_node", "Node entry is not a mapping", {"index": i}))
            continue
        node_id = _to_str(n.get("node_id", f"node_{i}")) or f"node_{i}"
        node_type = _normalize_node_type(_to_str(n.get("node_type", "")), node_id)
        if node_type is None:
            violations.append(
                DAGContractViolation(
                    "unknown_node_type",
                    "Node type is not in compact contract ontology",
                    {"node_id": node_id, "node_type": _to_str(n.get("node_type", ""))},
                )
            )
            continue
        if node_id in seen_node_ids:
            normalization_counts["dedup_node_id"] = int(normalization_counts.get("dedup_node_id", 0) + 1)
            continue
        seen_node_ids.add(node_id)
        ts = _to_float(n.get("timestamp_s"))
        value, reason = _value_to_canonical(node_type, n.get("value"))
        if reason:
            normalization_counts[reason] = int(normalization_counts.get(reason, 0) + 1)
        md = dict(n.get("metadata", {})) if isinstance(n.get("metadata", {}), Mapping) else {}
        if node_type in {"maneuver", "decision"}:
            md["behavior_class"] = str(value)
        canonical_nodes.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "value": value,
                "timestamp_s": ts,
                "metadata": md,
                "_orig_index": i,
            }
        )

    node_by_id = {n["node_id"]: n for n in canonical_nodes}
    if "ego_initial_speed" not in node_by_id:
        violations.append(
            DAGContractViolation("missing_anchor", "Required anchor node missing", {"node_id": "ego_initial_speed"})
        )
    if "collision_outcome" not in node_by_id:
        violations.append(
            DAGContractViolation("missing_anchor", "Required anchor node missing", {"node_id": "collision_outcome"})
        )
    if sum(1 for n in canonical_nodes if n["node_type"] == "maneuver") <= 0:
        violations.append(DAGContractViolation("missing_anchor", "At least one maneuver node required", {}))
    if sum(1 for n in canonical_nodes if n["node_type"] == "decision") <= 0:
        violations.append(DAGContractViolation("missing_anchor", "At least one decision node required", {}))

    # Canonical deterministic IDs.
    id_map: Dict[str, str] = {}
    taken = set()
    for anchor in ("ego_initial_speed", "collision_outcome"):
        if anchor in node_by_id:
            id_map[anchor] = anchor
            taken.add(anchor)
    for node_type in _TYPE_ORDER:
        pool = [n for n in canonical_nodes if n["node_type"] == node_type and n["node_id"] not in id_map]
        pool.sort(key=lambda r: (float("inf") if r["timestamp_s"] is None else float(r["timestamp_s"]), r["node_id"]))
        for i, n in enumerate(pool):
            base = f"{node_type}_{i}"
            name = base
            j = 1
            while name in taken:
                name = f"{base}_{j}"
                j += 1
            id_map[n["node_id"]] = name
            taken.add(name)
            if name != n["node_id"]:
                normalization_counts["renamed_node_id"] = int(normalization_counts.get("renamed_node_id", 0) + 1)

    canonical_nodes_out: List[Dict[str, Any]] = []
    for n in canonical_nodes:
        new_id = id_map.get(n["node_id"])
        if new_id is None:
            continue
        out = {k: v for k, v in n.items() if not k.startswith("_")}
        out["node_id"] = new_id
        canonical_nodes_out.append(out)
    canonical_nodes_out.sort(
        key=lambda r: (_TYPE_ORDER.index(r["node_type"]), float("inf") if r["timestamp_s"] is None else r["timestamp_s"], r["node_id"])
    )

    by_new_id = {n["node_id"]: n for n in canonical_nodes_out}

    # Canonical edges.
    edge_best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for i, e in enumerate(raw_edges):
        if not isinstance(e, Mapping):
            violations.append(DAGContractViolation("invalid_edge", "Edge entry is not a mapping", {"index": i}))
            continue
        u_old = _to_str(e.get("parent_id", ""))
        v_old = _to_str(e.get("child_id", ""))
        if u_old not in id_map or v_old not in id_map:
            continue
        u = id_map[u_old]
        v = id_map[v_old]
        if u == v:
            normalization_counts["drop_self_edge"] = int(normalization_counts.get("drop_self_edge", 0) + 1)
            continue
        conf = _to_float(e.get("confidence"))
        if conf is None:
            conf = 0.7
        conf = float(max(0.0, min(1.0, conf)))
        mech = _normalize_mechanism(_to_str(e.get("mechanism", "")))
        key = (u, v)
        prev = edge_best.get(key)
        if prev is None or conf > float(prev["confidence"]):
            edge_best[key] = {"parent_id": u, "child_id": v, "confidence": conf, "mechanism": mech}
        elif conf == float(prev["confidence"]) and mech < str(prev["mechanism"]):
            edge_best[key] = {"parent_id": u, "child_id": v, "confidence": conf, "mechanism": mech}

    canonical_edges = sorted(edge_best.values(), key=lambda x: (x["parent_id"], x["child_id"]))

    # Validate edge tier policy and degree caps.
    parents_count: Dict[str, int] = {}
    outgoing_count: Dict[str, int] = {}
    for e in canonical_edges:
        u = e["parent_id"]
        v = e["child_id"]
        u_type = by_new_id[u]["node_type"]
        v_type = by_new_id[v]["node_type"]
        t_u = _TYPE_TO_TIER.get(u_type, -1)
        t_v = _TYPE_TO_TIER.get(v_type, -1)
        allowed_same = (u_type, v_type) == ("maneuver", "decision")
        if not (t_u < t_v or allowed_same):
            violations.append(
                DAGContractViolation(
                    "invalid_tier_edge",
                    "Edge violates compact tier policy",
                    {"parent_id": u, "child_id": v, "parent_type": u_type, "child_type": v_type},
                )
            )
        parents_count[v] = int(parents_count.get(v, 0) + 1)
        outgoing_count[u] = int(outgoing_count.get(u, 0) + 1)

    for n, c in parents_count.items():
        if c > int(cfg.max_parents_per_node):
            violations.append(
                DAGContractViolation(
                    "max_parents_exceeded",
                    "Node exceeds max_parents_per_node",
                    {"node_id": n, "count": c, "limit": int(cfg.max_parents_per_node)},
                )
            )
    for n, c in outgoing_count.items():
        if c > int(cfg.max_outgoing_per_node):
            violations.append(
                DAGContractViolation(
                    "max_outgoing_exceeded",
                    "Node exceeds max_outgoing_per_node",
                    {"node_id": n, "count": c, "limit": int(cfg.max_outgoing_per_node)},
                )
            )

    node_ids = [n["node_id"] for n in canonical_nodes_out]
    edge_pairs = [(e["parent_id"], e["child_id"]) for e in canonical_edges]
    is_dag, depth = _topological_depth(node_ids, edge_pairs)
    if not is_dag:
        violations.append(DAGContractViolation("cycle_detected", "Graph is cyclic", {}))
        depth = 0
    if depth > int(cfg.max_depth):
        violations.append(
            DAGContractViolation(
                "max_depth_exceeded",
                "Graph depth exceeds contract bound",
                {"depth": int(depth), "limit": int(cfg.max_depth)},
            )
        )

    if len(canonical_nodes_out) > int(cfg.max_nodes):
        violations.append(
            DAGContractViolation(
                "max_nodes_exceeded",
                "Node count exceeds contract bound",
                {"count": len(canonical_nodes_out), "limit": int(cfg.max_nodes)},
            )
        )
    if len(canonical_edges) > int(cfg.max_edges):
        violations.append(
            DAGContractViolation(
                "max_edges_exceeded",
                "Edge count exceeds contract bound",
                {"count": len(canonical_edges), "limit": int(cfg.max_edges)},
            )
        )

    # CPT canonicalization after ID mapping.
    raw_cpts = p.get("cpts", {})
    if not isinstance(raw_cpts, Mapping):
        raw_cpts = {}
    cpts_out: Dict[str, Dict[str, Any]] = {}
    rev_map: Dict[str, List[str]] = {}
    for old_id, new_id in id_map.items():
        rev_map.setdefault(new_id, []).append(old_id)

    for n in canonical_nodes_out:
        nid = n["node_id"]
        ntype = n["node_type"]
        if ntype not in {"maneuver", "decision", "outcome"}:
            continue
        spec_raw = None
        for old in rev_map.get(nid, []):
            if old in raw_cpts and isinstance(raw_cpts[old], Mapping):
                spec_raw = raw_cpts[old]
                break
        if spec_raw is None:
            spec_raw = {}
        values_raw = list(spec_raw.get("values", [])) if isinstance(spec_raw.get("values", []), list) else []
        values: List[str] = []
        if values_raw:
            for v in values_raw:
                if ntype == "maneuver":
                    nv, reason = _normalize_maneuver(_to_str(v))
                elif ntype == "decision":
                    nv, reason = _normalize_decision(_to_str(v))
                else:
                    nv, reason = _normalize_outcome(_to_str(v))
                if reason:
                    normalization_counts[reason] = int(normalization_counts.get(reason, 0) + 1)
                values.append(nv)
        if not values:
            values = _default_values_for_type(ntype)
        val_node = _to_str(n.get("value", ""))
        if val_node and val_node not in values:
            values.append(val_node)
        values = list(dict.fromkeys(values))
        rows_raw = spec_raw.get("cpt", {}) if isinstance(spec_raw, Mapping) else {}
        rows: Dict[str, Dict[str, float]] = {}
        if isinstance(rows_raw, Mapping):
            for key, row in rows_raw.items():
                if not isinstance(row, Mapping):
                    continue
                rows[_to_str(key) or "*"] = _normalize_distribution(values, row)
        if not rows:
            rows["*"] = _normalize_distribution(values, {})
            normalization_counts["filled_missing_cpt"] = int(normalization_counts.get("filled_missing_cpt", 0) + 1)

        parents = [e["parent_id"] for e in canonical_edges if e["child_id"] == nid]
        cpts_out[nid] = {"values": values, "parents": parents, "cpt": rows}

    density = 0.0
    n_nodes = float(len(canonical_nodes_out))
    if n_nodes > 1:
        density = float(len(canonical_edges) / (n_nodes * (n_nodes - 1.0)))

    violation_counts: Dict[str, int] = {}
    for v in violations:
        violation_counts[v.code] = int(violation_counts.get(v.code, 0) + 1)

    passed = len(violations) == 0
    report = DAGContractReport(
        contract_name=str(cfg.name),
        contract_version=str(cfg.version),
        mode=str(cfg.mode),
        passed=bool(passed),
        before_nodes=int(before_nodes),
        after_nodes=int(len(canonical_nodes_out)),
        before_edges=int(before_edges),
        after_edges=int(len(canonical_edges)),
        max_depth=int(depth),
        density=float(density),
        normalization_counts=normalization_counts,
        violation_counts=violation_counts,
        violations=violations,
    )

    metadata = dict(p.get("metadata", {})) if isinstance(p.get("metadata", {}), Mapping) else {}
    metadata["contract_name"] = str(cfg.name)
    metadata["contract_version"] = str(cfg.version)
    metadata["contract_mode"] = str(cfg.mode)
    metadata["contract_report"] = report.summary()

    out_payload = {
        "schema_version": _to_str(p.get("schema_version", "")),
        "scenario_id": _to_str(p.get("scenario_id", "")),
        "nodes": canonical_nodes_out,
        "edges": canonical_edges,
        "cpts": cpts_out,
        "metadata": metadata,
    }
    return out_payload, report


def validate_dag_payload(
    payload: Mapping[str, Any],
    config: DAGContractConfig | None = None,
) -> Tuple[bool, List[DAGContractViolation], DAGContractReport]:
    canonical, report = canonicalize_dag_payload(payload, config=config)
    return bool(report.passed), list(report.violations), report


def enforce_dag_contract(
    payload: Mapping[str, Any],
    config: DAGContractConfig | None = None,
) -> Tuple[bool, Dict[str, Any], DAGContractReport]:
    cfg = config or DAGContractConfig()
    canonical, report = canonicalize_dag_payload(payload, config=cfg)
    if str(cfg.mode) == "hard":
        report.passed = bool(len(report.violations) == 0)
    return bool(report.passed), canonical, report


def payload_to_bayesian_dag(payload: Mapping[str, Any]) -> BayesianDAG:
    sid = _to_str(payload.get("scenario_id", "unknown"))
    dag = BayesianDAG(scenario_id=sid)
    for n in payload.get("nodes", []) if isinstance(payload.get("nodes", []), list) else []:
        if not isinstance(n, Mapping):
            continue
        nid = _to_str(n.get("node_id", ""))
        if not nid:
            continue
        dag.nodes[nid] = DAGNode(
            node_id=nid,
            node_type=_to_str(n.get("node_type", "unknown")),
            value=n.get("value"),
            timestamp_s=_to_float(n.get("timestamp_s")),
            metadata=dict(n.get("metadata", {})) if isinstance(n.get("metadata", {}), Mapping) else {},
        )
    edges_in = payload.get("edges", []) if isinstance(payload.get("edges", []), list) else []
    for e in edges_in:
        if not isinstance(e, Mapping):
            continue
        u = _to_str(e.get("parent_id", ""))
        v = _to_str(e.get("child_id", ""))
        if not u or not v or u not in dag.nodes or v not in dag.nodes:
            continue
        conf = _to_float(e.get("confidence"))
        dag.edges.append(
            DAGEdge(parent_id=u, child_id=v, confidence=float(conf if conf is not None else 0.7), mechanism=_to_str(e.get("mechanism", "")))
        )
    cpts = payload.get("cpts", {})
    dag.cpts = dict(cpts) if isinstance(cpts, Mapping) else {}
    return dag
