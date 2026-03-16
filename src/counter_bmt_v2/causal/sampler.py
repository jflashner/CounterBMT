"""Intervention sampling from Bayesian DAG state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Sequence

import numpy as np

from counter_bmt_v2.contracts import BayesianDAG, DAGNode, DAGEdge, Intervention
from counter_bmt_v2.training.dag_cache_schema import dag_to_cache_payload


class InterventionSampler(Protocol):
    def sample(self, dag: BayesianDAG, *, rare: bool = False, seed: int = 0) -> Intervention:
        """Sample one intervention from DAG."""


@dataclass
class SimpleInterventionSampler(InterventionSampler):
    fallback_value: str = "straight"

    def sample(self, dag: BayesianDAG, *, rare: bool = False, seed: int = 0) -> Intervention:
        rng = np.random.default_rng(seed)
        candidates: List[str] = [
            node_id
            for node_id, node in dag.nodes.items()
            if node.node_type in {"maneuver", "decision"}
        ]
        if not candidates:
            return Intervention(variable="maneuver_0", value=self.fallback_value, description="fallback intervention")

        node_id = candidates[int(rng.integers(0, len(candidates)))]
        node = dag.nodes[node_id]

        alternatives = node.metadata.get("alternatives", [])
        if not alternatives:
            alternatives = [node.value]

        options = [x for x in alternatives if x != node.value] or alternatives
        value = options[int(rng.integers(0, len(options)))]

        if rare and node.node_type == "maneuver":
            # Bias toward stronger perturbations for tail exploration.
            for pref in ["stop", "lane_change_left", "lane_change_right"]:
                if pref in options:
                    value = pref
                    break

        return Intervention(
            variable=node_id,
            value=value,
            original_value=node.value,
            timestamp_s=node.timestamp_s,
            aggressiveness=node.metadata.get("aggressiveness", "normal"),
            description=f"sampled intervention on {node_id}: {node.value} -> {value}",
        )


def _topological_order(dag: BayesianDAG, *, allowed_types: Sequence[str] | None = None) -> List[str]:
    allowed = None if allowed_types is None else {str(t) for t in allowed_types}
    node_ids = [
        nid for nid, node in dag.nodes.items()
        if allowed is None or str(node.node_type) in allowed
    ]
    keep = set(node_ids)
    indeg: Dict[str, int] = {nid: 0 for nid in node_ids}
    children: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for edge in dag.edges:
        if edge.parent_id in keep and edge.child_id in keep:
            indeg[edge.child_id] += 1
            children[edge.parent_id].append(edge.child_id)
    ready = sorted([nid for nid, deg in indeg.items() if deg == 0])
    out: List[str] = []
    while ready:
        nid = ready.pop(0)
        out.append(nid)
        for child in sorted(children.get(nid, [])):
            indeg[child] -= 1
            if indeg[child] == 0:
                ready.append(child)
                ready.sort()
    if len(out) != len(node_ids):
        remaining = [nid for nid in node_ids if nid not in set(out)]
        out.extend(sorted(remaining))
    return out


def _normalize_parent_key(parent_ids: Sequence[str], assignments: Mapping[str, Any]) -> str:
    return ", ".join(f"{pid}={assignments.get(pid)}" for pid in parent_ids)


def _normalize_probs(raw_row: Mapping[str, Any], values: Sequence[Any]) -> np.ndarray:
    probs = np.asarray([float(raw_row.get(v, 0.0)) for v in values], dtype=np.float32)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.ones((len(values),), dtype=np.float32) / max(1.0, float(len(values)))
    return (probs / total).astype(np.float32)


def _rare_bias_probs(probs: np.ndarray, *, rare: bool, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    if p.size == 0:
        return p
    if not rare:
        return p
    inv = np.power(np.clip(p, eps, 1.0), -0.5).astype(np.float32)
    inv_sum = float(np.sum(inv))
    if inv_sum <= 0.0:
        return np.ones_like(p) / max(1.0, float(p.size))
    return (inv / inv_sum).astype(np.float32)


def _node_alternatives(node: DAGNode, cpt_spec: Mapping[str, Any]) -> List[Any]:
    alts = node.metadata.get("alternatives", [])
    if isinstance(alts, Sequence) and not isinstance(alts, (str, bytes)):
        vals = [x for x in alts]
    else:
        vals = []
    cpt_vals = cpt_spec.get("values", [])
    if isinstance(cpt_vals, Sequence) and not isinstance(cpt_vals, (str, bytes)):
        for v in cpt_vals:
            if v not in vals:
                vals.append(v)
    if node.value not in vals:
        vals.append(node.value)
    return vals


def apply_intervention_assignments(dag: BayesianDAG, intervention: Intervention) -> BayesianDAG:
    sampled = BayesianDAG(
        scenario_id=dag.scenario_id,
        nodes={},
        edges=[
            DAGEdge(
                parent_id=str(edge.parent_id),
                child_id=str(edge.child_id),
                confidence=float(edge.confidence),
                mechanism=str(edge.mechanism),
            )
            for edge in dag.edges
        ],
        cpts=dict(dag.cpts),
    )
    assignments = dict(intervention.assignments)
    changed: List[str] = []
    for node_id, node in dag.nodes.items():
        value = assignments.get(node_id, node.value)
        meta = dict(node.metadata)
        meta["counterfactual_value"] = value
        meta["observed_value"] = node.value
        meta["is_counterfactual"] = bool(value != node.value)
        if value != node.value:
            changed.append(str(node_id))
        sampled.nodes[node_id] = DAGNode(
            node_id=str(node.node_id),
            node_type=str(node.node_type),
            value=value,
            timestamp_s=node.timestamp_s,
            metadata=meta,
        )
    setattr(
        sampled,
        "_contract_metadata",
        {
            "intervention_variable": str(intervention.variable),
            "intervention_value": intervention.value,
            "counterfactual_changed_nodes": list(changed),
            "source_dag_schema": str(intervention.source_dag_schema),
        },
    )
    return sampled


@dataclass
class TopologicalDAGAssignmentSampler(InterventionSampler):
    allowed_node_types: Sequence[str] = ("maneuver", "outcome")
    rare_probability_power: float = -0.5

    def _sample_value(
        self,
        *,
        rng: np.random.Generator,
        node: DAGNode,
        cpt_spec: Mapping[str, Any],
        assignments: Mapping[str, Any],
        rare: bool,
    ) -> Any:
        values = _node_alternatives(node, cpt_spec)
        if not values:
            return node.value

        parents_raw = cpt_spec.get("parents", [])
        parents = [str(p) for p in parents_raw] if isinstance(parents_raw, Sequence) and not isinstance(parents_raw, (str, bytes)) else []
        cpt = cpt_spec.get("cpt", {})
        row: Mapping[str, Any] | None = None
        if isinstance(cpt, Mapping) and parents:
            exact_key = _normalize_parent_key(parents, assignments)
            exact_row = cpt.get(exact_key)
            if isinstance(exact_row, Mapping):
                row = exact_row
            elif isinstance(cpt.get("*"), Mapping):
                row = cpt.get("*")  # type: ignore[assignment]
        elif isinstance(cpt, Mapping) and isinstance(cpt.get("*"), Mapping):
            row = cpt.get("*")  # type: ignore[assignment]

        if isinstance(row, Mapping):
            probs = _normalize_probs(row, values)
        else:
            probs = np.ones((len(values),), dtype=np.float32) / max(1.0, float(len(values)))

        if rare:
            p = np.asarray(probs, dtype=np.float32)
            strength = abs(float(self.rare_probability_power))
            weights = np.power(np.clip(p, 1e-6, 1.0), -strength).astype(np.float32)
            probs = weights / max(1e-6, float(np.sum(weights)))

        idx = int(rng.choice(len(values), p=probs))
        return values[idx]

    def sample(self, dag: BayesianDAG, *, rare: bool = False, seed: int = 0) -> Intervention:
        rng = np.random.default_rng(seed)
        order = _topological_order(dag, allowed_types=self.allowed_node_types)
        if not order:
            return Intervention(
                variable="maneuver_0",
                value="straight",
                description="fallback sampled DAG assignment",
                assignments={"maneuver_0": "straight"},
                assignment_order=["maneuver_0"],
                is_counterfactual=True,
                source_dag_schema=str(dag_to_cache_payload(dag).get("schema_version", "")),
            )

        assignments: Dict[str, Any] = {}
        first_changed: tuple[str, Any, Any, float | None] | None = None
        for node_id in order:
            node = dag.nodes[node_id]
            cpt_spec = dag.cpts.get(node_id, {})
            sampled_value = self._sample_value(
                rng=rng,
                node=node,
                cpt_spec=cpt_spec if isinstance(cpt_spec, Mapping) else {},
                assignments=assignments,
                rare=bool(rare),
            )
            assignments[node_id] = sampled_value
            if first_changed is None and sampled_value != node.value:
                first_changed = (node_id, sampled_value, node.value, node.timestamp_s)

        if first_changed is None:
            primary_id = order[0]
            primary_node = dag.nodes[primary_id]
            first_changed = (primary_id, assignments.get(primary_id, primary_node.value), primary_node.value, primary_node.timestamp_s)

        variable, value, original_value, timestamp_s = first_changed
        return Intervention(
            variable=str(variable),
            value=value,
            original_value=original_value,
            timestamp_s=timestamp_s,
            aggressiveness="rare" if bool(rare) else "normal",
            description=f"sampled DAG assignment with primary node {variable}={value}",
            assignments=assignments,
            assignment_order=list(order),
            source_dag_schema=str(dag_to_cache_payload(dag).get("schema_version", "")),
            is_counterfactual=True,
            metadata={"changed_nodes": [nid for nid in order if assignments.get(nid) != dag.nodes[nid].value]},
        )
