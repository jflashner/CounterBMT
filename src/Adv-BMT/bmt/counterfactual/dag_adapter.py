from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .contract_local_intervention import LocalInterventionV1


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class SparseDAGNode:
    node_id: str
    node_type: str
    value: Any
    timestamp_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SparseDAGEdge:
    parent_id: str
    child_id: str
    confidence: float = 1.0
    mechanism: str = ""


@dataclass
class SparseDAGView:
    scenario_id: str
    nodes: List[SparseDAGNode] = field(default_factory=list)
    edges: List[SparseDAGEdge] = field(default_factory=list)
    cpts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


def _payload_from_intervention(intervention: LocalInterventionV1 | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(intervention, LocalInterventionV1):
        return intervention.to_dict()
    return dict(intervention)


def local_intervention_to_bayesian_dag(intervention: LocalInterventionV1 | Dict[str, Any]) -> SparseDAGView:
    payload = _payload_from_intervention(intervention)
    context = dict(payload.get("context", {}))
    decision = dict(payload.get("supervised_decision", payload.get("gt_decision", payload.get("raw_recovered_decision", {}))))
    conflict_agents = context.get("conflict_agents", [])
    min_conflict_eta_gap = None
    if conflict_agents:
        finite = [
            float(agent.get("eta_gap_s"))
            for agent in conflict_agents
            if agent.get("eta_gap_s") is not None and np.isfinite(float(agent.get("eta_gap_s")))
        ]
        min_conflict_eta_gap = min(finite) if finite else None

    entry_timing_value = decision.get("entry_timing") or ("no_conflict" if not conflict_agents else "undetermined")
    collision_value = "collision_avoided"
    if conflict_agents and decision.get("entry_timing") == "before_conflict" and decision.get("compliance_label") == "red_light_violation":
        collision_value = "collision_possible"

    nodes = [
        SparseDAGNode(node_id="context/signal_state", node_type="context", value=context.get("signal_state_at_decision"), metadata={}),
        SparseDAGNode(node_id="context/conflict_eta", node_type="context", value=min_conflict_eta_gap, metadata={"num_conflict_agents": len(conflict_agents)}),
        SparseDAGNode(node_id="decision/path_choice", node_type="decision", value=decision.get("branch_label"), metadata={"branch_id": decision.get("branch_id")}),
        SparseDAGNode(node_id="decision/compliance", node_type="decision", value=decision.get("compliance_label"), metadata={}),
        SparseDAGNode(node_id="decision/entry_timing", node_type="decision", value=entry_timing_value, metadata={}),
        SparseDAGNode(
            node_id="outcome/stopline_crossing",
            node_type="outcome",
            value=("crossed" if bool(payload.get("raw_recovered_decision", payload.get("gt_decision", {})).get("crossed_stop_region")) else "not_crossed"),
            metadata={},
        ),
        SparseDAGNode(node_id="outcome/collision", node_type="outcome", value=collision_value, metadata={}),
        SparseDAGNode(node_id="outcome/interaction_order", node_type="outcome", value=entry_timing_value, metadata={}),
    ]
    edges = [
        SparseDAGEdge(parent_id="context/signal_state", child_id="decision/compliance", confidence=1.0, mechanism="signal_informs_compliance"),
        SparseDAGEdge(parent_id="context/conflict_eta", child_id="decision/entry_timing", confidence=1.0, mechanism="conflict_eta_informs_entry_timing"),
        SparseDAGEdge(parent_id="decision/path_choice", child_id="outcome/stopline_crossing", confidence=1.0, mechanism="path_choice_to_crossing"),
        SparseDAGEdge(parent_id="decision/path_choice", child_id="outcome/collision", confidence=0.9, mechanism="path_choice_to_collision"),
        SparseDAGEdge(parent_id="decision/compliance", child_id="outcome/stopline_crossing", confidence=1.0, mechanism="compliance_to_crossing"),
        SparseDAGEdge(parent_id="decision/compliance", child_id="outcome/collision", confidence=1.0, mechanism="compliance_to_collision"),
        SparseDAGEdge(parent_id="decision/entry_timing", child_id="outcome/interaction_order", confidence=1.0, mechanism="entry_timing_to_order"),
        SparseDAGEdge(parent_id="decision/entry_timing", child_id="outcome/collision", confidence=0.9, mechanism="entry_timing_to_collision"),
    ]
    return SparseDAGView(
        scenario_id=str(payload.get("scenario_id", "")),
        nodes=nodes,
        edges=edges,
        cpts={},
        metadata={
            "projection_name": "local_intervention_to_bayesian_dag_v1",
            "source_contract": payload.get("contract_name"),
            "source_schema_version": payload.get("schema_version"),
            "agent_id": payload.get("agent_id"),
            "decision_time_idx": int(payload.get("decision_time_idx", 0)),
        },
    )


def project_local_intervention_to_sparse_dag(intervention: LocalInterventionV1 | Dict[str, Any]) -> SparseDAGView:
    return local_intervention_to_bayesian_dag(intervention)
