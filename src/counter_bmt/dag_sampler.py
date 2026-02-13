"""
dag_sampler.py

Bayesian DAG sampling module for CounterBMT.

Three self-contained classes:
  - ObservedCPTBuilder: converts LLM CPTs into pgmpy TabularCPDs
  - DAGSampler: ancestral + posterior sampling via pgmpy
  - SampledInterventionBuilder: converts samples into intervention dicts

Usage:
    from counter_bmt.dag_sampler import DAGSampler, SampledInterventionBuilder

    sampler = DAGSampler(dag)
    samples = sampler.sample_rare(n=5)
    builder = SampledInterventionBuilder()
    for sample in samples:
        interventions = builder.build_interventions(sample, dag)
"""

import logging
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Node types that are always continuous / observed (never sampled)
_CONTINUOUS_NODE_TYPES = {"ego_state", "agent_state", "environmental"}

# Node types that are discrete and can be sampled
_DISCRETE_NODE_TYPES = {"maneuver", "decision", "outcome", "severity"}


# =============================================================================
# ObservedCPTBuilder
# =============================================================================


class ObservedCPTBuilder:
    """
    Converts LLM-produced CPTs stored in node.metadata["cpt"] into
    pgmpy-compatible TabularCPD objects.

    Continuous parents (ego_state, agent_state, environmental) are treated
    as *observed evidence* and collapsed out of the CPT.  Only discrete
    parents remain as conditioning variables.
    """

    def __init__(self, dag):
        """
        Args:
            dag: ScenarioDAG with CPTs already stored in node.metadata["cpt"].
        """
        self.dag = dag

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def build_cpds(self) -> Tuple[List, List[str]]:
        """
        Returns:
            (cpds, discrete_node_ids)
            cpds: list of pgmpy TabularCPD objects
            discrete_node_ids: list of node ids included in the BN
        """
        from pgmpy.factors.discrete import TabularCPD

        discrete_ids = self._discrete_node_ids()
        cpds = []

        for nid in discrete_ids:
            node = self.dag.nodes[nid]
            cpt_data = node.metadata.get("cpt", {})
            values = self._values_for(nid)

            if not values:
                logger.warning(f"Node {nid} has no discrete values, skipping")
                continue

            # Identify discrete parents only
            all_parents = self.dag.get_parents(nid)
            discrete_parents = [p for p in all_parents if p in discrete_ids]

            # Build the CPD array
            cpd = self._build_single_cpd(nid, values, discrete_parents, cpt_data)
            if cpd is not None:
                cpds.append(cpd)

        return cpds, discrete_ids

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _discrete_node_ids(self) -> List[str]:
        """Return node ids that are discrete (maneuver/decision/outcome)."""
        ids = []
        for nid, node in self.dag.nodes.items():
            ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
            if ntype in _DISCRETE_NODE_TYPES:
                ids.append(nid)
        return ids

    def _values_for(self, nid: str) -> List[str]:
        """Get the discrete value set for a node."""
        node = self.dag.nodes[nid]
        alts = node.metadata.get("alternatives", [])
        val = node.value
        values = []
        if val is not None and not isinstance(val, dict):
            values.append(str(val))
        for a in alts:
            s = str(a)
            if s not in values:
                values.append(s)
        return values

    def _build_single_cpd(
        self,
        nid: str,
        values: List[str],
        discrete_parents: List[str],
        cpt_data: Dict,
    ):
        """Build a TabularCPD for one node."""
        from pgmpy.factors.discrete import TabularCPD

        cpt_table = cpt_data.get("cpt", {})
        n_vals = len(values)

        if not discrete_parents:
            # Root node or all parents are continuous → single column
            probs = self._lookup_row(cpt_table, {}, values)
            cpd = TabularCPD(
                variable=nid,
                variable_card=n_vals,
                values=[[p] for p in probs],
                state_names={nid: values},
            )
            return cpd

        # Enumerate all parent-value combos
        parent_values = {pid: self._values_for(pid) for pid in discrete_parents}
        parent_cards = [len(parent_values[pid]) for pid in discrete_parents]
        combos = list(itertools.product(*(parent_values[pid] for pid in discrete_parents)))

        # Build column-major matrix: shape (n_vals, prod(parent_cards))
        columns = []
        for combo in combos:
            assignment = dict(zip(discrete_parents, combo))
            probs = self._lookup_row(cpt_table, assignment, values)
            columns.append(probs)

        # Transpose to pgmpy shape: rows = values, cols = parent combos
        matrix = np.array(columns).T.tolist()  # shape (n_vals, n_combos)

        state_names = {nid: values}
        for pid in discrete_parents:
            state_names[pid] = parent_values[pid]

        cpd = TabularCPD(
            variable=nid,
            variable_card=n_vals,
            values=matrix,
            evidence=discrete_parents,
            evidence_card=parent_cards,
            state_names=state_names,
        )
        return cpd

    def _lookup_row(
        self,
        cpt_table: Dict,
        assignment: Dict[str, str],
        values: List[str],
    ) -> List[float]:
        """
        Look up probabilities from the CPT table for a given parent assignment.
        Falls back to wildcard '*' or uniform.
        """
        n = len(values)
        uniform = [1.0 / n] * n

        if not cpt_table:
            return uniform

        # Build lookup key: "parent1=val1,parent2=val2" (sorted by parent id)
        if assignment:
            key = ",".join(f"{k}={assignment[k]}" for k in sorted(assignment))
        else:
            key = "*"

        # Try exact match
        row = cpt_table.get(key)

        # Try with wildcards for continuous parents embedded in key
        if row is None:
            for cpt_key, cpt_row in cpt_table.items():
                if self._key_matches(cpt_key, assignment):
                    row = cpt_row
                    break

        # Fallback to wildcard
        if row is None:
            row = cpt_table.get("*")

        if row is None:
            return uniform

        # Extract probabilities in value order
        probs = []
        for v in values:
            p = row.get(str(v), 0.0)
            if isinstance(p, (int, float)):
                probs.append(float(p))
            else:
                probs.append(0.0)

        # Normalize if needed
        total = sum(probs)
        if total <= 0:
            return uniform
        if abs(total - 1.0) > 1e-6:
            probs = [p / total for p in probs]

        return probs

    def _key_matches(self, cpt_key: str, assignment: Dict[str, str]) -> bool:
        """
        Check if a CPT key matches a discrete assignment, ignoring
        continuous-parent entries (marked with =*).
        """
        if cpt_key == "*":
            return not assignment
        parts = cpt_key.split(",")
        for part in parts:
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v == "*":
                continue  # Continuous parent wildcard, skip
            if k in assignment and assignment[k] != v:
                return False
        return True


# =============================================================================
# DAGSampler
# =============================================================================


class DAGSampler:
    """
    Wraps a ScenarioDAG + pgmpy BayesianNetwork for sampling.

    Supports:
      - Ancestral (forward) sampling
      - Posterior sampling conditioned on evidence (Option C)
      - Convenience: sample_rare() for tail-of-distribution scenarios
    """

    def __init__(self, dag):
        """
        Args:
            dag: ScenarioDAG with CPTs in node.metadata["cpt"].
        """
        from pgmpy.models import BayesianNetwork

        self.dag = dag
        self._builder = ObservedCPTBuilder(dag)
        cpds, discrete_ids = self._builder.build_cpds()
        self._discrete_ids = discrete_ids

        if not cpds:
            raise ValueError("No discrete CPDs could be built from the DAG")

        # Build pgmpy BN with only discrete nodes and discrete-to-discrete edges
        discrete_set = set(discrete_ids)
        edges = []
        for edge in dag.edges:
            if edge.parent_id in discrete_set and edge.child_id in discrete_set:
                edges.append((edge.parent_id, edge.child_id))

        self._bn = BayesianNetwork(edges)
        # Add isolated discrete nodes (no edges to/from other discrete nodes)
        for nid in discrete_ids:
            if nid not in self._bn.nodes():
                self._bn.add_node(nid)

        for cpd in cpds:
            self._bn.add_cpds(cpd)

        if not self._bn.check_model():
            logger.warning("pgmpy model check failed; CPTs may be inconsistent")

        logger.info(
            f"DAGSampler: BN with {len(discrete_ids)} discrete nodes, "
            f"{len(edges)} edges, {len(cpds)} CPDs"
        )

    # ------------------------------------------------------------------
    # Public sampling methods
    # ------------------------------------------------------------------

    def sample_ancestral(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Forward / ancestral sampling from the joint distribution.

        Returns:
            List of dicts, each mapping node_id -> sampled value string.
        """
        from pgmpy.sampling import BayesianModelSampling

        sampler = BayesianModelSampling(self._bn)
        df = sampler.forward_sample(size=n)
        samples = []
        for _, row in df.iterrows():
            samples.append({col: row[col] for col in df.columns})
        return samples

    def sample_posterior(
        self,
        evidence: Dict[str, str],
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Sample from the posterior P(layer0-1 | evidence) using exact
        inference (Variable Elimination), then draw from the computed
        posterior.

        Args:
            evidence: dict mapping node_id -> observed value string
                      e.g. {"collision_outcome": "collision_possible"}
            n: number of samples

        Returns:
            List of dicts, each mapping node_id -> sampled value string.
        """
        from pgmpy.inference import VariableElimination

        ve = VariableElimination(self._bn)

        # Query all non-evidence discrete nodes
        query_vars = [nid for nid in self._discrete_ids if nid not in evidence]

        if not query_vars:
            # Nothing to query; just return evidence repeated
            return [dict(evidence) for _ in range(n)]

        # Compute posterior for each query variable individually
        # (joint query can be expensive; marginals are sufficient for
        # independent sampling when we later do ancestral re-sampling)
        posteriors = {}
        for var in query_vars:
            try:
                result = ve.query([var], evidence=evidence)
                vals = result.state_names[var]
                probs = result.values
                posteriors[var] = (vals, probs)
            except Exception as e:
                logger.warning(f"Posterior query failed for {var}: {e}")
                # Fallback to prior
                vals = self._builder._values_for(var)
                probs = np.ones(len(vals)) / len(vals)
                posteriors[var] = (vals, probs)

        # Draw samples from the marginal posteriors
        samples = []
        for _ in range(n):
            sample = dict(evidence)
            for var in query_vars:
                vals, probs = posteriors[var]
                # Ensure probs sum to 1
                p = np.array(probs, dtype=float)
                p = np.clip(p, 0, None)
                total = p.sum()
                if total > 0:
                    p = p / total
                else:
                    p = np.ones(len(vals)) / len(vals)
                idx = np.random.choice(len(vals), p=p)
                sample[var] = vals[idx]
            samples.append(sample)

        return samples

    def sample_rare(
        self,
        n: int = 1,
        outcome_value: str = "collision_possible",
        outcome_node: str = "collision_outcome",
    ) -> List[Dict[str, Any]]:
        """
        Convenience: sample from P(maneuvers, decisions | outcome = outcome_value).

        This is Option C — backward inference to find the rare combinations
        of maneuvers and decisions that lead to a specific (safety-critical)
        outcome.
        """
        evidence = {outcome_node: outcome_value}
        return self.sample_posterior(evidence=evidence, n=n)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def discrete_node_ids(self) -> List[str]:
        return list(self._discrete_ids)

    @property
    def bn(self):
        """Access the underlying pgmpy BayesianNetwork."""
        return self._bn


# =============================================================================
# SampledInterventionBuilder
# =============================================================================


class SampledInterventionBuilder:
    """
    Converts a sampled DAG assignment (dict of node_id -> sampled_value)
    into intervention dicts compatible with compile_from_dag_intervention().
    """

    def build_interventions(
        self,
        sample: Dict[str, Any],
        dag,
    ) -> List[Dict[str, Any]]:
        """
        For each sampled maneuver/decision whose value differs from the
        ground-truth node value, produce an intervention dict.

        Args:
            sample: dict mapping node_id -> sampled value string
            dag: ScenarioDAG (to look up original values and metadata)

        Returns:
            List of intervention dicts, each with keys:
              variable, value, original_value, aggressiveness, timestamp, description
        """
        interventions = []

        for node_id, sampled_value in sample.items():
            if node_id not in dag.nodes:
                continue

            node = dag.nodes[node_id]
            ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)

            # Only create interventions for maneuvers and decisions
            if ntype not in ("maneuver", "decision"):
                continue

            original = str(node.value) if node.value is not None else None
            sampled = str(sampled_value)

            # Skip if sampled value matches the ground truth
            if sampled == original:
                continue

            interventions.append({
                "variable": node_id,
                "value": sampled,
                "original_value": original,
                "aggressiveness": node.metadata.get("aggressiveness", "normal"),
                "timestamp": node.timestamp,
                "description": f"Sampled: set {node.name} to {sampled} (was {original})",
            })

        return interventions
