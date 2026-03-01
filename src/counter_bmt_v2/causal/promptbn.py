"""PromptBN-style one-shot DAG builder.

Grounded in PromptBN (Zhang et al., 2025):
- single-step LLM structure induction
- dual representation (node-centric + edge-centric)
- structural consistency validation
- DAG acyclicity validation
- retry-until-valid (bounded)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from counter_bmt_v2.causal.dag_contract import (
    DAGContractConfig,
    enforce_dag_contract,
    payload_to_bayesian_dag,
)
from counter_bmt_v2.causal.dag import DAGBuilder, SimpleDAGBuilder
from counter_bmt_v2.contracts import BayesianDAG, DAGEdge, DAGNode, ScenarioInput, VLMFeatures
from counter_bmt_v2.llm import OpenAIChatClient
from counter_bmt_v2.training.dag_cache_schema import schema_version_for_contract

logger = logging.getLogger(__name__)

_MANEUVER_ALTERNATIVES_COMPACT12: List[str] = [
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
]

_OUTCOME_ALTERNATIVES_MO: Dict[str, List[str]] = {
    "collision_outcome": ["collision_avoided", "collision_possible"],
    "progress_outcome": ["progress_good", "progress_limited"],
    "compliance_outcome": ["compliant", "violation_possible"],
}


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return {}
    return {}


def _check_dag(node_ids: Set[str], edges: List[Tuple[str, str]]) -> bool:
    adj: Dict[str, Set[str]] = {n: set() for n in node_ids}
    indeg: Dict[str, int] = {n: 0 for n in node_ids}

    for u, v in edges:
        if u not in node_ids or v not in node_ids:
            return False
        if u == v:
            return False
        if v not in adj[u]:
            adj[u].add(v)
            indeg[v] += 1

    q = [n for n in node_ids if indeg[n] == 0]
    seen = 0
    while q:
        cur = q.pop()
        seen += 1
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return seen == len(node_ids)


def _normalize_cpt_rows(cpt: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, row in cpt.items():
        vals: Dict[str, float] = {}
        total = 0.0
        for k, v in row.items():
            try:
                x = float(v)
            except Exception:
                x = 0.0
            x = max(0.0, x)
            vals[str(k)] = x
            total += x

        if total <= 0.0 and vals:
            u = 1.0 / float(len(vals))
            vals = {k: u for k in vals}
        elif total > 0.0:
            vals = {k: x / total for k, x in vals.items()}

        out[str(key)] = vals
    return out


@dataclass
class PromptBNDAGBuilder(DAGBuilder):
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    max_retries: int = 4
    use_simple_fallback: bool = True
    dag_contract: str = "maneuver_outcome_v1"
    dag_contract_mode: str = "hard"

    def __post_init__(self) -> None:
        self._fallback = SimpleDAGBuilder()
        self._client: Optional[OpenAIChatClient] = None
        try:
            self._client = OpenAIChatClient(model=self.model, api_key=self.api_key)
        except Exception as exc:
            if not self.use_simple_fallback:
                raise
            logger.warning("PromptBNDAGBuilder fallback to simple DAG: %s", exc)

    def build(self, scene: ScenarioInput, features: VLMFeatures) -> BayesianDAG:
        if self._client is None:
            return self._fallback.build(scene, features)

        variables = self._build_variable_schema(scene, features)
        prompt = self._build_prompt(scene, variables)

        for attempt in range(1, self.max_retries + 1):
            raw = self._client.complete(prompt=prompt, images_base64=None, temperature=0.1, max_tokens=2500)
            parsed = _extract_json_object(raw)

            ok, dag = self._parse_and_validate(scene, variables, parsed)
            if ok and dag is not None:
                self._project_edges_for_contract(dag)
                dag.cpts = self._ensure_minimal_cpts(dag)
                dag.cpts.update(self._extract_cpts(parsed, dag))
                dag.cpts = {k: v for k, v in dag.cpts.items() if k in dag.nodes}
                dag.cpts = self._postprocess_cpts(dag)
                dag.nodes.setdefault(
                    "collision_outcome",
                    DAGNode(
                        node_id="collision_outcome",
                        node_type="outcome",
                        value="collision_avoided",
                        metadata={"alternatives": ["collision_avoided", "collision_possible"]},
                    ),
                )
                default_outcomes = ["collision_outcome"]
                if str(self.dag_contract) == "maneuver_outcome_v1":
                    default_outcomes = ["collision_outcome", "progress_outcome", "compliance_outcome"]
                for out_node in default_outcomes:
                    if out_node not in dag.nodes:
                        continue
                    if any(e.child_id == out_node for e in dag.edges):
                        continue
                    for node in dag.nodes.values():
                        if node.node_type == "maneuver":
                            dag.edges.append(
                                DAGEdge(
                                    parent_id=node.node_id,
                                    child_id=out_node,
                                    confidence=0.7,
                                    mechanism="maneuver_to_outcome",
                                )
                            )
                payload = {
                    "schema_version": schema_version_for_contract(str(self.dag_contract)),
                    "scenario_id": str(scene.scenario_id),
                    "nodes": [
                        {
                            "node_id": str(n.node_id),
                            "node_type": str(n.node_type),
                            "value": n.value,
                            "timestamp_s": n.timestamp_s,
                            "metadata": dict(n.metadata),
                        }
                        for n in dag.nodes.values()
                    ],
                    "edges": [
                        {
                            "parent_id": str(e.parent_id),
                            "child_id": str(e.child_id),
                            "confidence": float(e.confidence),
                            "mechanism": str(e.mechanism),
                        }
                        for e in dag.edges
                    ],
                    "cpts": dict(dag.cpts),
                    "metadata": {"source": "counter_bmt_v2_promptbn"},
                }
                cfg = DAGContractConfig(
                    name=str(self.dag_contract),
                    mode=str(self.dag_contract_mode),
                )
                contract_ok, canonical_payload, contract_report = enforce_dag_contract(payload, config=cfg)
                if not contract_ok:
                    logger.warning(
                        "PromptBN DAG failed contract on attempt %d/%d: violations=%s",
                        attempt,
                        self.max_retries,
                        contract_report.violation_counts,
                    )
                    continue
                canonical_payload["schema_version"] = schema_version_for_contract(str(self.dag_contract))
                canonical_payload["scenario_id"] = str(scene.scenario_id)
                canonical_meta = canonical_payload.get("metadata", {})
                if not isinstance(canonical_meta, dict):
                    canonical_meta = {"metadata_raw": str(canonical_meta)}
                canonical_meta.setdefault("source", "counter_bmt_v2_promptbn")
                canonical_payload["metadata"] = canonical_meta
                canonical_dag = payload_to_bayesian_dag(canonical_payload)
                setattr(canonical_dag, "_contract_metadata", canonical_meta)
                return canonical_dag

            logger.warning("PromptBN DAG output invalid on attempt %d/%d", attempt, self.max_retries)

        if self.use_simple_fallback:
            return self._fallback.build(scene, features)
        raise RuntimeError("PromptBNDAGBuilder failed to produce a valid DAG within retry budget")

    def _build_variable_schema(self, scene: ScenarioInput, features: VLMFeatures) -> List[Dict[str, Any]]:
        vars_out: List[Dict[str, Any]] = []

        if str(self.dag_contract) == "maneuver_outcome_v1":
            maneuvers = sorted(list(features.maneuvers), key=lambda m: (float(m.start_s), float(m.end_s)))
            for i, m in enumerate(maneuvers[:8]):
                start_s = float(m.start_s)
                end_s = float(max(float(m.end_s), start_s))
                duration_s = float(max(0.0, end_s - start_s))
                mid_s = float(0.5 * (start_s + end_s))
                vars_out.append(
                    {
                        "node_id": f"maneuver_{i}",
                        "node_type": "maneuver",
                        "value": str(m.maneuver_type.value),
                        "timestamp_s": mid_s,
                        "start_s": start_s,
                        "end_s": end_s,
                        "duration_s": duration_s,
                        "mid_s": mid_s,
                        "description": m.reasoning or "critical maneuver segment",
                        "alternatives": list(_MANEUVER_ALTERNATIVES_COMPACT12),
                        "observed": True,
                    }
                )
            if not any(str(v.get("node_type", "")) == "maneuver" for v in vars_out):
                vars_out.append(
                    {
                        "node_id": "maneuver_0",
                        "node_type": "maneuver",
                        "value": "straight",
                        "timestamp_s": 0.0,
                        "start_s": 0.0,
                        "end_s": 0.0,
                        "duration_s": 0.0,
                        "mid_s": 0.0,
                        "description": "default maneuver anchor",
                        "alternatives": list(_MANEUVER_ALTERNATIVES_COMPACT12),
                        "observed": True,
                    }
                )

            vars_out.extend(
                [
                    {
                        "node_id": "collision_outcome",
                        "node_type": "outcome",
                        "value": "collision_avoided",
                        "description": "collision consequence at scenario horizon",
                        "alternatives": list(_OUTCOME_ALTERNATIVES_MO["collision_outcome"]),
                        "observed": True,
                    },
                    {
                        "node_id": "progress_outcome",
                        "node_type": "outcome",
                        "value": "progress_good",
                        "description": "goal/progress consequence at scenario horizon",
                        "alternatives": list(_OUTCOME_ALTERNATIVES_MO["progress_outcome"]),
                        "observed": True,
                    },
                    {
                        "node_id": "compliance_outcome",
                        "node_type": "outcome",
                        "value": "compliant",
                        "description": "rule-compliance consequence at scenario horizon",
                        "alternatives": list(_OUTCOME_ALTERNATIVES_MO["compliance_outcome"]),
                        "observed": True,
                    },
                ]
            )
            return vars_out

        init_speed = 10.0
        if scene.ego_trajectory_xy is not None and len(scene.ego_trajectory_xy) > 1:
            dx = scene.ego_trajectory_xy[1] - scene.ego_trajectory_xy[0]
            init_speed = float((dx[0] ** 2 + dx[1] ** 2) ** 0.5 / 0.1)

        vars_out.append(
            {
                "node_id": "ego_initial_speed",
                "node_type": "ego_state",
                "value": init_speed,
                "description": "ego speed at scenario start",
                "alternatives": [max(0.0, init_speed * 0.5), init_speed, init_speed * 1.5],
            }
        )

        for i, m in enumerate(features.maneuvers):
            vars_out.append(
                {
                    "node_id": f"maneuver_{i}",
                    "node_type": "maneuver",
                    "value": m.maneuver_type.value,
                    "timestamp_s": m.start_s,
                    "description": m.reasoning or "maneuver event",
                    "alternatives": [
                        "straight",
                        "lane_change_left",
                        "lane_change_right",
                        "left_turn",
                        "right_turn",
                        "stop",
                    ],
                }
            )
        if not any(str(v.get("node_type", "")) == "maneuver" for v in vars_out):
            vars_out.append(
                {
                    "node_id": "maneuver_0",
                    "node_type": "maneuver",
                    "value": "straight",
                    "timestamp_s": 0.0,
                    "description": "default maneuver anchor",
                    "alternatives": [
                        "straight",
                        "lane_change_left",
                        "lane_change_right",
                        "left_turn",
                        "right_turn",
                        "stop",
                    ],
                }
            )

        for i, d in enumerate(features.decisions):
            vars_out.append(
                {
                    "node_id": f"decision_{i}",
                    "node_type": "decision",
                    "value": d.choice,
                    "timestamp_s": d.timestamp_s,
                    "description": d.reasoning or "decision event",
                    "alternatives": d.alternatives or [d.choice],
                }
            )
        if not any(str(v.get("node_type", "")) == "decision" for v in vars_out):
            vars_out.append(
                {
                    "node_id": "decision_0",
                    "node_type": "decision",
                    "value": "maintain_speed",
                    "timestamp_s": 0.0,
                    "description": "default decision anchor",
                    "alternatives": ["maintain_speed", "accelerate", "decelerate", "yield_or_proceed"],
                }
            )

        vars_out.append(
            {
                "node_id": "collision_outcome",
                "node_type": "outcome",
                "value": "collision_avoided",
                "description": "collision outcome at scenario horizon",
                "alternatives": ["collision_avoided", "collision_possible"],
            }
        )
        return vars_out

    def _build_prompt(self, scene: ScenarioInput, variables: List[Dict[str, Any]]) -> str:
        cpt_scope = "maneuver and outcome" if str(self.dag_contract) == "maneuver_outcome_v1" else "maneuver, decision, and outcome"
        edge_rule = (
            "Edges MUST be only maneuver -> outcome. Do NOT emit outcome->* or maneuver->maneuver edges."
            if str(self.dag_contract) == "maneuver_outcome_v1"
            else "Edges may follow general causal semantics from parent causes to child effects."
        )
        return f"""
You are discovering a Bayesian Network structure from variable metadata.

Follow PromptBN-style constraints:
1) Output both node-centric and edge-centric structures.
2) Use ONLY node IDs from the provided metadata.
3) Ensure edge set forms a valid DAG (no cycles).
4) Include discrete CPTs for {cpt_scope} nodes.
5) Keep probabilities in each CPT row normalized to sum to 1.
6) {edge_rule}

Scenario: {scene.scenario_id}
Variables metadata:
{json.dumps(variables, indent=2)}

Output JSON only with this schema:
{{
  "bn": {{
    "nodes": [
      {{
        "node_id": "...",
        "parents": ["..."],
        "reasoning": "..."
      }}
    ],
    "edges": [
      {{"from": "...", "to": "...", "mechanism": "...", "confidence": 0.0}}
    ],
    "cpts": [
      {{
        "node_id": "...",
        "values": ["..."],
        "parents": ["..."],
        "cpt": {{
          "*": {{"v1": 0.5, "v2": 0.5}},
          "parent=a": {{"v1": 0.7, "v2": 0.3}}
        }}
      }}
    ],
    "network_summary": "..."
  }}
}}
""".strip()

    def _project_edges_for_contract(self, dag: BayesianDAG) -> None:
        """Hard-shape edges for contract mode to reduce retry churn."""
        if str(self.dag_contract) != "maneuver_outcome_v1":
            return
        maneuvers = {n.node_id for n in dag.nodes.values() if str(n.node_type) == "maneuver"}
        outcomes = {n.node_id for n in dag.nodes.values() if str(n.node_type) == "outcome"}
        if not maneuvers or not outcomes:
            dag.edges = []
            return

        best: Dict[Tuple[str, str], DAGEdge] = {}
        for e in dag.edges:
            u = str(e.parent_id)
            v = str(e.child_id)
            if u not in maneuvers or v not in outcomes:
                continue
            key = (u, v)
            prev = best.get(key)
            if prev is None or float(e.confidence) > float(prev.confidence):
                best[key] = DAGEdge(
                    parent_id=u,
                    child_id=v,
                    confidence=float(max(0.0, min(1.0, float(e.confidence)))),
                    mechanism=str(e.mechanism or "maneuver_to_outcome"),
                )

        # Guarantee at least one incoming edge per required outcome if maneuvers exist.
        required_outcomes = ("collision_outcome", "progress_outcome", "compliance_outcome")
        first_m = sorted(maneuvers)[0]
        for o in required_outcomes:
            if o in outcomes and not any(k[1] == o for k in best.keys()):
                best[(first_m, o)] = DAGEdge(
                    parent_id=first_m,
                    child_id=o,
                    confidence=0.7,
                    mechanism="maneuver_to_outcome",
                )

        dag.edges = [best[k] for k in sorted(best.keys())]

    def _parse_and_validate(
        self,
        scene: ScenarioInput,
        variables: List[Dict[str, Any]],
        parsed: Dict[str, Any],
    ) -> Tuple[bool, Optional[BayesianDAG]]:
        bn = parsed.get("bn", parsed)
        raw_nodes = bn.get("nodes", [])
        raw_edges = bn.get("edges", [])

        allowed = {v["node_id"] for v in variables}
        node_parent_map: Dict[str, Set[str]] = {}

        for n in raw_nodes:
            nid = str(n.get("node_id", ""))
            if nid not in allowed:
                continue
            parents = {str(p) for p in n.get("parents", []) if str(p) in allowed and str(p) != nid}
            node_parent_map[nid] = parents

        edge_pairs: Set[Tuple[str, str]] = set()
        for e in raw_edges:
            u = str(e.get("from", ""))
            v = str(e.get("to", ""))
            if u in allowed and v in allowed and u != v:
                edge_pairs.add((u, v))

        if str(self.dag_contract) == "maneuver_outcome_v1":
            # In maneuver_outcome mode we allow partial node-centric outputs and
            # project edge shape later. If edge list is empty but parents were
            # provided in node-centric form, use those links.
            if not edge_pairs and node_parent_map:
                for nid, parents in node_parent_map.items():
                    for p in parents:
                        if p in allowed and p != nid:
                            edge_pairs.add((p, nid))
        else:
            # PromptBN expects full node-centric coverage.
            if set(node_parent_map.keys()) != allowed:
                return False, None
            # Structural consistency validation (PromptBN): node parents == incoming edge set.
            for nid, parents in node_parent_map.items():
                incoming = {u for (u, v) in edge_pairs if v == nid}
                if parents != incoming:
                    return False, None

        # Ensure every edge references known nodes, and graph is acyclic.
        if not _check_dag(allowed, list(edge_pairs)):
            return False, None

        var_map = {v["node_id"]: v for v in variables}
        dag = BayesianDAG(scenario_id=scene.scenario_id)

        for nid in allowed:
            meta = var_map[nid]
            node_meta = {
                "alternatives": meta.get("alternatives", []),
                "description": meta.get("description", ""),
            }
            for key in ("start_s", "end_s", "duration_s", "mid_s", "observed"):
                if key in meta:
                    node_meta[key] = meta.get(key)
            dag.nodes[nid] = DAGNode(
                node_id=nid,
                node_type=str(meta.get("node_type", "unknown")),
                value=meta.get("value"),
                timestamp_s=meta.get("timestamp_s"),
                metadata=node_meta,
            )

        edge_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for e in raw_edges:
            u = str(e.get("from", ""))
            v = str(e.get("to", ""))
            if (u, v) in edge_pairs:
                edge_info[(u, v)] = e

        for u, v in sorted(edge_pairs):
            info = edge_info.get((u, v), {})
            dag.edges.append(
                DAGEdge(
                    parent_id=u,
                    child_id=v,
                    confidence=float(info.get("confidence", 0.7)),
                    mechanism=str(info.get("mechanism", "")),
                )
            )

        return True, dag

    def _ensure_minimal_cpts(self, dag: BayesianDAG) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for node in dag.nodes.values():
            if node.node_type not in {"maneuver", "decision", "outcome"}:
                continue
            if str(self.dag_contract) == "maneuver_outcome_v1" and node.node_type == "decision":
                continue
            values = [str(x) for x in node.metadata.get("alternatives", [])]
            if str(node.value) not in values and node.value is not None:
                values = [str(node.value)] + values
            values = list(dict.fromkeys(values))
            if not values:
                values = [str(node.value) if node.value is not None else "unknown"]
            p = 1.0 / float(len(values))
            out[node.node_id] = {
                "values": values,
                "parents": [e.parent_id for e in dag.edges if e.child_id == node.node_id],
                "cpt": {"*": {v: p for v in values}},
            }
        return out

    def _extract_cpts(self, parsed: Dict[str, Any], dag: BayesianDAG) -> Dict[str, Dict[str, Any]]:
        bn = parsed.get("bn", parsed)
        raw = bn.get("cpts", [])
        out: Dict[str, Dict[str, Any]] = {}
        for item in raw:
            nid = str(item.get("node_id", ""))
            if nid not in dag.nodes:
                continue
            cpt = item.get("cpt", {})
            if not isinstance(cpt, dict):
                continue
            out[nid] = {
                "values": [str(v) for v in item.get("values", [])],
                "parents": [str(p) for p in item.get("parents", [])],
                "cpt": _normalize_cpt_rows({str(k): dict(v) for k, v in cpt.items() if isinstance(v, dict)}),
            }
        return out

    def _postprocess_cpts(self, dag: BayesianDAG) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for nid, spec in dag.cpts.items():
            values = [str(v) for v in spec.get("values", [])]
            if not values:
                values = [str(dag.nodes[nid].value)]
            rowed = {}
            for key, row in spec.get("cpt", {}).items():
                normalized = _normalize_cpt_rows({"tmp": row}).get("tmp", {})
                # Ensure all values present.
                full = {v: float(normalized.get(v, 0.0)) for v in values}
                s = sum(full.values())
                if s <= 0.0:
                    u = 1.0 / float(len(values))
                    full = {v: u for v in values}
                else:
                    full = {v: x / s for v, x in full.items()}
                rowed[str(key)] = full

            if not rowed:
                u = 1.0 / float(len(values))
                rowed = {"*": {v: u for v in values}}

            if str(self.dag_contract) == "maneuver_outcome_v1":
                parents = [e.parent_id for e in dag.edges if e.child_id == nid]
            else:
                parents = [str(p) for p in spec.get("parents", []) if p in dag.nodes]

            out[nid] = {
                "values": values,
                "parents": [str(p) for p in parents if str(p) in dag.nodes],
                "cpt": rowed,
            }
        return out
