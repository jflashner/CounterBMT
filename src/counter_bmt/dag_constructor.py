"""
Grounded DAG Constructor for Safety-Critical Driving Scenarios

Enforced structure:
- Layer 0 (INITIAL): Agent states from trajectory (positions, speeds, headings)
- Layer 1 (EVENTS): VLM-extracted maneuvers and decisions  
- Layer 2 (OUTCOME): Collision outcome node

The LLM only infers edges between existing nodes - no hallucinated nodes.

Author: CounterBMT Project
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from copy import deepcopy
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    EGO_STATE = "ego_state"
    AGENT_STATE = "agent_state"
    ENVIRONMENTAL = "environmental"
    MANEUVER = "maneuver"
    DECISION = "decision"
    INTERACTION = "interaction"
    OUTCOME = "outcome"
    SEVERITY = "severity"


class EdgeType(Enum):
    DIRECT = "direct"
    TEMPORAL = "temporal"
    MEDIATED = "mediated"


def _to_python_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python_types(v) for v in obj]
    return obj


@dataclass
class DAGNode:
    id: str
    name: str
    node_type: NodeType
    layer: int
    timestamp: Optional[float] = None
    value: Optional[Any] = None
    description: str = ""
    is_intervened: bool = False
    is_observed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "type": self.node_type.value,
                "layer": self.layer, "timestamp": _to_python_types(self.timestamp), 
                "value": _to_python_types(self.value),
                "description": self.description, "is_intervened": self.is_intervened,
                "is_observed": self.is_observed, "metadata": _to_python_types(self.metadata)}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGNode":
        return cls(id=d["id"], name=d["name"], node_type=NodeType(d["type"]),
                   layer=d.get("layer", 1), timestamp=d.get("timestamp"),
                   value=d.get("value"), description=d.get("description", ""),
                   is_intervened=d.get("is_intervened", False),
                   is_observed=d.get("is_observed", True), metadata=d.get("metadata", {}))


@dataclass
class DAGEdge:
    parent_id: str
    child_id: str
    edge_type: EdgeType = EdgeType.DIRECT
    mechanism: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"parent": self.parent_id, "child": self.child_id,
                "edge_type": self.edge_type.value, "mechanism": self.mechanism,
                "confidence": self.confidence, "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DAGEdge":
        return cls(parent_id=d["parent"], child_id=d["child"],
                   edge_type=EdgeType(d.get("edge_type", "direct")),
                   mechanism=d.get("mechanism", ""), confidence=d.get("confidence", 1.0),
                   metadata=d.get("metadata", {}))


@dataclass
class Intervention:
    """
    Represents a counterfactual intervention on a DAG node.
    
    Includes aggressiveness and timestamp for proper mapping to
    granular token sets in the InterventionCompiler.
    """
    variable_id: str
    value: Any
    original_value: Optional[Any] = None
    description: str = ""
    aggressiveness: str = "normal"  # passive, normal, aggressive
    timestamp: Optional[float] = None  # Time of intervention for time-based biasing
    
    def to_dict(self): 
        return {
            "variable": self.variable_id, 
            "value": self.value,
            "original_value": self.original_value, 
            "description": self.description,
            "aggressiveness": self.aggressiveness,
            "timestamp": self.timestamp
        }


@dataclass
class CounterfactualResult:
    intervention: Intervention
    outcome_variable: str
    original_outcome: Optional[Any]
    counterfactual_outcome: Optional[Any]
    effect_direction: str
    confidence: float
    reasoning: str
    affected_paths: List[List[str]] = field(default_factory=list)
    def to_dict(self): return {"intervention": self.intervention.to_dict(),
        "outcome_variable": self.outcome_variable, "original_outcome": self.original_outcome,
        "counterfactual_outcome": self.counterfactual_outcome, "effect_direction": self.effect_direction,
        "confidence": self.confidence, "reasoning": self.reasoning, "affected_paths": self.affected_paths}


# =============================================================================
# Intervention Sequences (Sequential Multi-Maneuver Interventions)
# =============================================================================

class InterventionDiversity(Enum):
    """Controls how extreme/diverse the generated interventions are."""
    LOW = "low"          # Conservative: only safe alternatives
    MEDIUM = "medium"    # Balanced (current default)
    HIGH = "high"        # Extreme: includes aggressive/risky alternatives
    LATERAL = "lateral"  # Only lateral maneuvers (turns, lane changes) - no straight/accel/brake


@dataclass
class InterventionStep:
    """
    A single step in an intervention sequence.
    
    Represents one maneuver at a specific time in a sequential chain.
    """
    maneuver: str           # "accelerate", "lane_change_right", "turn_right", etc.
    start_time_s: float     # When to start this maneuver
    duration_s: float       # How long the maneuver lasts
    intensity: str = "normal"  # "gentle", "normal", "aggressive"
    
    def to_dict(self) -> Dict:
        return {
            "maneuver": self.maneuver,
            "start_time_s": self.start_time_s,
            "duration_s": self.duration_s,
            "intensity": self.intensity
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "InterventionStep":
        return cls(
            maneuver=d["maneuver"],
            start_time_s=d["start_time_s"],
            duration_s=d["duration_s"],
            intensity=d.get("intensity", "normal")
        )


@dataclass  
class InterventionSequence:
    """
    A sequence of maneuvers to apply at different times.
    
    Example: accelerate → lane_change_right → turn_right
    
    This enables complex driving "scripts" in a single counterfactual trajectory.
    """
    steps: List[InterventionStep]
    description: str = ""
    sequence_id: str = ""
    
    def __post_init__(self):
        # Sort steps by start time
        self.steps = sorted(self.steps, key=lambda s: s.start_time_s)
        
        # Generate description if not provided
        if not self.description and self.steps:
            maneuvers = [s.maneuver for s in self.steps]
            self.description = " → ".join(maneuvers)
    
    @property
    def total_duration(self) -> float:
        """Total duration covered by all steps."""
        if not self.steps:
            return 0.0
        last_step = self.steps[-1]
        return last_step.start_time_s + last_step.duration_s
    
    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "total_duration": self.total_duration
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "InterventionSequence":
        return cls(
            steps=[InterventionStep.from_dict(s) for s in d.get("steps", [])],
            description=d.get("description", ""),
            sequence_id=d.get("sequence_id", "")
        )
    
    @classmethod
    def from_string(cls, s: str) -> "InterventionSequence":
        """
        Parse a sequence from a string specification.
        
        Format: "maneuver1:start-end:intensity,maneuver2:start-end:intensity,..."
        Example: "accelerate:0-2:aggressive,lane_change_right:2-4:normal,turn_right:4-7:gentle"
        """
        steps = []
        parts = s.split(",")
        for part in parts:
            tokens = part.strip().split(":")
            if len(tokens) >= 2:
                maneuver = tokens[0]
                time_range = tokens[1].split("-")
                start = float(time_range[0])
                end = float(time_range[1]) if len(time_range) > 1 else start + 2.0
                intensity = tokens[2] if len(tokens) > 2 else "normal"
                steps.append(InterventionStep(
                    maneuver=maneuver,
                    start_time_s=start,
                    duration_s=end - start,
                    intensity=intensity
                ))
        return cls(steps=steps)


class ScenarioDAG:
    """Grounded causal DAG with layer structure."""
    
    def __init__(self, scenario_id: str = ""):
        self.scenario_id = scenario_id
        self.nodes: Dict[str, DAGNode] = {}
        self.edges: List[DAGEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adj: Dict[str, Set[str]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, node: DAGNode) -> None:
        self.nodes[node.id] = node
        self._adjacency.setdefault(node.id, set())
        self._reverse_adj.setdefault(node.id, set())
    
    def add_edge(self, edge: DAGEdge) -> bool:
        if edge.parent_id not in self.nodes or edge.child_id not in self.nodes:
            return False
        if edge.parent_id == edge.child_id:
            return False
        # Layer ordering
        if self.nodes[edge.parent_id].layer > self.nodes[edge.child_id].layer:
            logger.warning(f"Edge {edge.parent_id}->{edge.child_id} violates layer order")
            return False
        # Duplicate check
        for e in self.edges:
            if e.parent_id == edge.parent_id and e.child_id == edge.child_id:
                return False
        # Cycle check
        self._adjacency[edge.parent_id].add(edge.child_id)
        self._reverse_adj[edge.child_id].add(edge.parent_id)
        if self._has_cycle():
            self._adjacency[edge.parent_id].remove(edge.child_id)
            self._reverse_adj[edge.child_id].remove(edge.parent_id)
            return False
        self.edges.append(edge)
        return True
    
    def remove_incoming_edges(self, node_id: str) -> List[DAGEdge]:
        removed = []
        for p in list(self._reverse_adj.get(node_id, [])):
            for e in self.edges:
                if e.parent_id == p and e.child_id == node_id:
                    removed.append(e); break
            self.edges = [e for e in self.edges if not (e.parent_id == p and e.child_id == node_id)]
            self._adjacency[p].discard(node_id)
            self._reverse_adj[node_id].discard(p)
        return removed
    
    def get_parents(self, nid: str) -> List[str]: return list(self._reverse_adj.get(nid, []))
    def get_children(self, nid: str) -> List[str]: return list(self._adjacency.get(nid, []))
    def get_roots(self) -> List[str]: return [n for n, p in self._reverse_adj.items() if not p]
    def get_leaves(self) -> List[str]: return [n for n, c in self._adjacency.items() if not c]
    def get_nodes_by_layer(self, layer: int) -> List[DAGNode]:
        return [n for n in self.nodes.values() if n.layer == layer]
    
    def find_all_paths(self, src: str, tgt: str) -> List[List[str]]:
        paths = []
        def dfs(cur, path):
            if cur == tgt: paths.append(path.copy()); return
            for c in self._adjacency.get(cur, []):
                if c not in path: path.append(c); dfs(c, path); path.pop()
        dfs(src, [src])
        return paths
    
    def topological_sort(self) -> List[str]:
        in_deg = {n: len(p) for n, p in self._reverse_adj.items()}
        q = [n for n, d in in_deg.items() if d == 0]
        result = []
        while q:
            n = q.pop(0); result.append(n)
            for c in self._adjacency.get(n, []):
                in_deg[c] -= 1
                if in_deg[c] == 0: q.append(c)
        return result
    
    def _has_cycle(self) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.nodes}
        def dfs(node):
            color[node] = GRAY
            for c in self._adjacency.get(node, []):
                if color[c] == GRAY: return True
                if color[c] == WHITE and dfs(c): return True
            color[node] = BLACK
            return False
        return any(color[n] == WHITE and dfs(n) for n in self.nodes)
    
    def do(self, interventions: Dict[str, Any]) -> "ScenarioDAG":
        new_dag = self.copy()
        for var_id, val in interventions.items():
            if var_id in new_dag.nodes:
                new_dag.remove_incoming_edges(var_id)
                new_dag.nodes[var_id].value = val
                new_dag.nodes[var_id].is_intervened = True
        new_dag.metadata["interventions"] = interventions
        return new_dag
    
    def copy(self) -> "ScenarioDAG":
        new = ScenarioDAG(self.scenario_id)
        new.nodes = {k: DAGNode.from_dict(v.to_dict()) for k, v in self.nodes.items()}
        new.edges = [DAGEdge.from_dict(e.to_dict()) for e in self.edges]
        new._adjacency = {k: set(v) for k, v in self._adjacency.items()}
        new._reverse_adj = {k: set(v) for k, v in self._reverse_adj.items()}
        new.metadata = deepcopy(self.metadata)
        return new
    
    def get_intervenable_nodes(self) -> List[DAGNode]:
        types = {NodeType.DECISION, NodeType.MANEUVER, NodeType.EGO_STATE}
        return [n for n in self.nodes.values() if n.node_type in types and n.layer < 2]
    
    def enumerate_interventions(
        self, 
        diversity: InterventionDiversity = InterventionDiversity.MEDIUM,
        speed_range: Optional[Tuple[float, float]] = None
    ) -> List[Intervention]:
        """
        Enumerate all possible interventions on intervenable nodes.
        
        Args:
            diversity: Controls how extreme the interventions are
                - LOW: Only safe/conservative alternatives
                - MEDIUM: Balanced alternatives (default)
                - HIGH: Includes aggressive/extreme alternatives
            speed_range: Optional (min_multiplier, max_multiplier) for speed changes
                        Default depends on diversity level
        
        Includes aggressiveness and timestamp from node metadata for
        proper mapping to granular token sets.
        """
        interventions = []

        def _node_priority(n: DAGNode) -> int:
            if n.node_type == NodeType.DECISION:
                return 0
            if n.node_type == NodeType.MANEUVER:
                return 1
            if n.node_type == NodeType.EGO_STATE:
                return 2
            return 3

        nodes = sorted(self.get_intervenable_nodes(), key=_node_priority)
        for node in nodes:
            alts = self._get_alternatives(node, diversity, speed_range)
            for alt in alts:
                if alt != node.value:
                    interventions.append(Intervention(
                        variable_id=node.id, 
                        value=alt, 
                        original_value=node.value,
                        description=f"Set {node.name} to {alt} (was {node.value})",
                        aggressiveness=node.metadata.get('aggressiveness', 'normal'),
                        timestamp=node.timestamp
                    ))
        return interventions
    
    def enumerate_sequences(
        self,
        sequence_length: int = 2,
        diversity: InterventionDiversity = InterventionDiversity.MEDIUM,
        prediction_horizon_s: float = 9.5
    ) -> List[InterventionSequence]:
        """
        Enumerate sequential intervention chains (multi-maneuver trajectories).
        
        Generates sequences of maneuvers that happen at different times,
        e.g., "accelerate → lane_change_right → turn_right"
        
        Args:
            sequence_length: Number of maneuvers per sequence (2, 3, etc.)
            diversity: Controls which maneuvers are included
            prediction_horizon_s: Total time horizon (default 9.5s for 19 timesteps)
            
        Returns:
            List of InterventionSequence objects
        """
        if sequence_length < 2:
            return []
        
        # Get available maneuvers based on diversity
        maneuvers = self._get_sequence_maneuvers(diversity)
        
        # Calculate time per maneuver
        maneuver_duration = prediction_horizon_s / sequence_length
        
        # Generate combinations
        import itertools
        sequences = []
        seq_id = 0
        
        for combo in itertools.product(maneuvers, repeat=sequence_length):
            # Skip sequences with all same maneuvers
            if len(set(combo)) == 1:
                continue
            
            # Skip incompatible sequences (e.g., left_turn after lane_change_right on highway)
            if not self._is_valid_sequence(combo):
                continue
            
            steps = []
            for i, maneuver in enumerate(combo):
                steps.append(InterventionStep(
                    maneuver=maneuver,
                    start_time_s=i * maneuver_duration,
                    duration_s=maneuver_duration,
                    intensity="normal"
                ))
            
            sequences.append(InterventionSequence(
                steps=steps,
                sequence_id=f"seq_{seq_id:03d}"
            ))
            seq_id += 1
        
        return sequences
    
    def _get_sequence_maneuvers(self, diversity: InterventionDiversity) -> List[str]:
        """Get available maneuvers for sequence generation based on diversity."""
        if diversity == InterventionDiversity.LOW:
            return ["straight", "decelerate", "lane_change_left", "lane_change_right"]
        elif diversity == InterventionDiversity.HIGH:
            return [
                "straight", "accelerate", "decelerate", "hard_brake",
                "lane_change_left", "lane_change_right", 
                "turn_left", "turn_right", "swerve"
            ]
        elif diversity == InterventionDiversity.LATERAL:
            # Only lateral maneuvers - no straight, accelerate, or brake
            return [
                "lane_change_left", "lane_change_right",
                "turn_left", "turn_right", "swerve"
            ]
        else:  # MEDIUM
            return [
                "straight", "accelerate", "decelerate",
                "lane_change_left", "lane_change_right",
                "turn_left", "turn_right"
            ]
    
    def _is_valid_sequence(self, maneuvers: Tuple[str, ...]) -> bool:
        """Check if a maneuver sequence is physically plausible."""
        # Can't turn immediately after opposite lane change
        for i in range(len(maneuvers) - 1):
            if maneuvers[i] == "lane_change_left" and maneuvers[i+1] == "turn_right":
                return False
            if maneuvers[i] == "lane_change_right" and maneuvers[i+1] == "turn_left":
                return False
            # Can't do hard brake then accelerate immediately
            if maneuvers[i] == "hard_brake" and maneuvers[i+1] == "accelerate":
                return False
        return True
    
    def _get_alternatives(
        self, 
        node: DAGNode, 
        diversity: InterventionDiversity = InterventionDiversity.MEDIUM,
        speed_range: Optional[Tuple[float, float]] = None
    ) -> List[Any]:
        """Get alternatives for a node based on diversity level."""
        if "alternatives" in node.metadata: 
            return node.metadata["alternatives"]
        
        if node.node_type == NodeType.DECISION:
            return self._get_decision_alternatives(node.value, diversity)
        
        if node.node_type == NodeType.MANEUVER:
            return self._get_maneuver_alternatives(node.value, diversity)
        
        if node.node_type == NodeType.EGO_STATE and "speed" in node.name.lower():
            return self._get_speed_alternatives(node.value, diversity, speed_range)
        
        return [node.value]
    
    def _get_decision_alternatives(self, value: str, diversity: InterventionDiversity) -> List[str]:
        """Get decision alternatives based on diversity."""
        if diversity == InterventionDiversity.LOW:
            alts = {
                "proceed": ["proceed", "yield"],
                "yield": ["yield", "stop"],
                "maintain_course": ["maintain_course", "brake"],
                "accept_gap": ["accept_gap", "reject_gap"]
            }
        elif diversity == InterventionDiversity.HIGH:
            alts = {
                "proceed": ["proceed", "yield", "stop", "accelerate_through"],
                "yield": ["proceed", "yield", "stop", "ignore_yield"],
                "maintain_course": ["maintain_course", "brake", "swerve", "hard_brake"],
                "accept_gap": ["accept_gap", "reject_gap", "force_gap"]
            }
        else:  # MEDIUM
            alts = {
                "proceed": ["proceed", "yield", "stop"],
                "yield": ["proceed", "yield", "stop"],
                "maintain_course": ["maintain_course", "brake", "swerve"],
                "accept_gap": ["accept_gap", "reject_gap"]
            }
        return alts.get(value, [value])
    
    def _get_maneuver_alternatives(self, value: str, diversity: InterventionDiversity) -> List[str]:
        """Get maneuver alternatives based on diversity."""
        if diversity == InterventionDiversity.LOW:
            alts = {
                "left_turn": ["left_turn", "straight"],
                "right_turn": ["right_turn", "straight"],
                "lane_change_left": ["lane_change_left", "straight"],
                "lane_change_right": ["lane_change_right", "straight"],
                "straight": ["straight", "decelerate"],
                "stop": ["stop", "decelerate"],
            }
        elif diversity == InterventionDiversity.HIGH:
            alts = {
                "left_turn": ["left_turn", "straight", "stop", "sharp_left_turn", "u_turn"],
                "right_turn": ["right_turn", "straight", "stop", "sharp_right_turn"],
                "lane_change_left": ["lane_change_left", "lane_change_right", "straight", "aggressive_lane_left"],
                "lane_change_right": ["lane_change_right", "lane_change_left", "straight", "aggressive_lane_right"],
                "straight": ["straight", "lane_change_left", "lane_change_right", "accelerate", "hard_brake", "swerve"],
                "stop": ["stop", "decelerate", "straight", "reverse"],
                "decelerate": ["decelerate", "stop", "straight", "hard_brake"],
                "accelerate": ["accelerate", "straight", "decelerate", "hard_accelerate"],
            }
        elif diversity == InterventionDiversity.LATERAL:
            # Only lateral maneuvers - no straight, accelerate, brake, or stop
            alts = {
                "left_turn": ["left_turn", "right_turn", "lane_change_left", "swerve"],
                "right_turn": ["right_turn", "left_turn", "lane_change_right", "swerve"],
                "lane_change_left": ["lane_change_left", "lane_change_right", "turn_left", "swerve"],
                "lane_change_right": ["lane_change_right", "lane_change_left", "turn_right", "swerve"],
                "straight": ["lane_change_left", "lane_change_right", "turn_left", "turn_right", "swerve"],
                "stop": ["lane_change_left", "lane_change_right", "swerve"],
                "decelerate": ["lane_change_left", "lane_change_right", "swerve"],
                "accelerate": ["lane_change_left", "lane_change_right", "swerve"],
            }
        else:  # MEDIUM
            alts = {
                "left_turn": ["left_turn", "straight", "stop"],
                "right_turn": ["right_turn", "straight", "stop"],
                "lane_change_left": ["lane_change_left", "lane_change_right", "straight"],
                "lane_change_right": ["lane_change_right", "lane_change_left", "straight"],
                "straight": ["straight", "lane_change_left", "lane_change_right", "stop"],
                "stop": ["stop", "decelerate", "straight"],
                "decelerate": ["decelerate", "stop", "straight", "accelerate"],
                "accelerate": ["accelerate", "straight", "decelerate"],
            }
        return alts.get(value, [value])
    
    def _get_speed_alternatives(
        self, 
        value: Any, 
        diversity: InterventionDiversity,
        speed_range: Optional[Tuple[float, float]] = None
    ) -> List[float]:
        """Get speed alternatives based on diversity and optional range."""
        if not isinstance(value, (int, float)) or value <= 0:
            return [value]
        
        v = float(value)
        
        # Use provided range or default based on diversity
        if speed_range:
            min_mult, max_mult = speed_range
        else:
            if diversity == InterventionDiversity.LOW:
                min_mult, max_mult = 0.75, 1.25
            elif diversity == InterventionDiversity.HIGH:
                min_mult, max_mult = 0.0, 2.0  # Include full stop and double speed
            else:  # MEDIUM
                min_mult, max_mult = 0.5, 1.5
        
        # Generate speed alternatives
        multipliers = []
        if min_mult == 0.0:
            multipliers.append(0.0)  # Full stop
            multipliers.extend([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
            if max_mult >= 1.75:
                multipliers.append(1.75)
            if max_mult >= 2.0:
                multipliers.append(2.0)
        else:
            # Generate evenly spaced multipliers
            step = (max_mult - min_mult) / 4
            multipliers = [min_mult + i * step for i in range(5)]
        
        return [v * m for m in multipliers if min_mult <= m <= max_mult]
    
    def to_dict(self) -> Dict: return {"scenario_id": self.scenario_id,
        "nodes": [n.to_dict() for n in self.nodes.values()],
        "edges": [e.to_dict() for e in self.edges], "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ScenarioDAG":
        dag = cls(d.get("scenario_id", ""))
        for nd in d.get("nodes", []): dag.add_node(DAGNode.from_dict(nd))
        for ed in d.get("edges", []): dag.add_edge(DAGEdge.from_dict(ed))
        dag.metadata = d.get("metadata", {})
        return dag
    
    def to_json(self, indent=2) -> str: return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, s: str) -> "ScenarioDAG": return cls.from_dict(json.loads(s))
    
    def summary(self) -> str:
        lines = [f"ScenarioDAG: {self.scenario_id}",
                 f"  Nodes: {len(self.nodes)}, Edges: {len(self.edges)}",
                 f"  Layer 0 (Initial): {[n.name for n in self.get_nodes_by_layer(0)]}",
                 f"  Layer 1 (Events): {[n.name for n in self.get_nodes_by_layer(1)]}",
                 f"  Layer 2 (Outcome): {[n.name for n in self.get_nodes_by_layer(2)]}"]
        return "\n".join(lines)


# =============================================================================
# LLM Clients
# =============================================================================

class DAGClient(ABC):
    @abstractmethod
    def infer_edges(self, nodes: List[Dict], scenario_context: str) -> List[Dict]:
        """Infer causal edges between existing nodes. NO new nodes."""
        pass
    
    @abstractmethod
    def evaluate_counterfactual(self, dag: ScenarioDAG, intervention: Intervention, 
                                outcome_var: str) -> CounterfactualResult:
        pass


class MockDAGClient(DAGClient):
    """Mock client using domain heuristics."""
    
    def infer_edges(self, nodes: List[Dict], scenario_context: str) -> List[Dict]:
        edges = []
        layer0 = [n for n in nodes if n.get("layer", 1) == 0]
        layer1 = [n for n in nodes if n.get("layer", 1) == 1]
        layer2 = [n for n in nodes if n.get("layer", 1) == 2]
        
        # Layer 0 -> Layer 1 (initial states influence events)
        for n0 in layer0:
            for n1 in layer1:
                # Speed influences decisions
                if "speed" in n0.get("name", "").lower() and n1.get("type") == "decision":
                    edges.append({"parent": n0["id"], "child": n1["id"],
                                  "mechanism": "Speed influences decision", "confidence": 0.8})
                # Position influences maneuvers
                elif "position" in n0.get("name", "").lower() and n1.get("type") == "maneuver":
                    edges.append({"parent": n0["id"], "child": n1["id"],
                                  "mechanism": "Position influences maneuver", "confidence": 0.7})
                # Other agent state influences decisions
                elif n0.get("type") == "agent_state" and n1.get("type") == "decision":
                    edges.append({"parent": n0["id"], "child": n1["id"],
                                  "mechanism": "Other agent influences decision", "confidence": 0.85})
        
        # Layer 1 -> Layer 1 (temporal/causal ordering by timestamp)
        sorted_l1 = sorted(layer1, key=lambda x: x.get("timestamp", 0) or 0)
        for i, n1 in enumerate(sorted_l1):
            for n2 in sorted_l1[i+1:]:
                # Decisions influence maneuvers
                if n1.get("type") == "decision" and n2.get("type") == "maneuver":
                    edges.append({"parent": n1["id"], "child": n2["id"],
                                  "mechanism": "Decision leads to maneuver", "confidence": 0.9})
                # Earlier decisions influence later decisions
                elif n1.get("type") == "decision" and n2.get("type") == "decision":
                    if (n2.get("timestamp", 0) or 0) - (n1.get("timestamp", 0) or 0) < 0.5:
                        edges.append({"parent": n1["id"], "child": n2["id"],
                                      "mechanism": "Sequential decisions", "confidence": 0.75})
        
        # Layer 1 -> Layer 2 (events influence outcome)
        for n1 in layer1:
            for n2 in layer2:
                edges.append({"parent": n1["id"], "child": n2["id"],
                              "mechanism": f"{n1.get('type', 'Event')} affects outcome",
                              "confidence": 0.85})
        
        return edges
    
    def evaluate_counterfactual(self, dag: ScenarioDAG, intervention: Intervention,
                                outcome_var: str) -> CounterfactualResult:
        paths = dag.find_all_paths(intervention.variable_id, outcome_var)
        effect = "decrease" if intervention.value in ["yield", "stop", "brake", "reject_gap"] else "increase"
        return CounterfactualResult(
            intervention=intervention, outcome_variable=outcome_var,
            original_outcome="collision_possible", 
            counterfactual_outcome="collision_avoided" if effect == "decrease" else "collision_likely",
            effect_direction=effect, confidence=0.75,
            reasoning=f"Changing {intervention.variable_id} to {intervention.value} would {effect} collision risk via {len(paths)} path(s)",
            affected_paths=paths)


class GPT4oDAGClient(DAGClient):
    """GPT-4o client for edge inference."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key: raise ValueError("OPENAI_API_KEY not found")
        import openai
        self.client = openai.OpenAI(api_key=self.api_key)
        self.call_log = []
    
    def _call(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0.2, max_tokens=2000,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return resp.choices[0].message.content
    
    def infer_edges(self, nodes: List[Dict], scenario_context: str) -> List[Dict]:
        system = """You are an expert in causal inference for driving scenarios.
Given a list of EXISTING nodes, infer causal edges between them.

CRITICAL RULES:
1. You can ONLY create edges between nodes in the provided list
2. You CANNOT create new nodes
3. Edges must respect layer ordering: Layer 0 (initial states) -> Layer 1 (events) -> Layer 2 (outcome)
4. Return ONLY valid JSON array of edges

IMPORTANT - Consider these causal relationships:
- OTHER AGENT STATES (agent_*_state) SHOULD influence ego decisions and maneuvers when logical!
  The presence, position, and speed of other vehicles affects whether the ego yields, stops, changes lanes, etc.
- Ego initial state (position, speed, heading) influences early maneuvers
- Maneuvers can causally chain (decelerate -> stop -> accelerate)
- Decisions influence maneuvers and vice versa
- All layer 1 events can affect the collision outcome

Each edge: {"parent": "node_id", "child": "node_id", "mechanism": "description", "confidence": 0.0-1.0}"""

        user = f"""Scenario: {scenario_context}

Nodes (you can ONLY use these node IDs):
{json.dumps(nodes, indent=2)}

IMPORTANT: Make sure to include edges from agent_*_state nodes to relevant decisions/maneuvers.
Other vehicles' positions and speeds causally influence the ego's driving decisions.

Infer causal edges. Return JSON array only, no other text:"""
        
        # Log what we're sending to the LLM
        logger.info("=" * 60)
        logger.info("GPT-4o DAG Edge Inference Request")
        logger.info("=" * 60)
        logger.info(f"Context: {scenario_context[:200]}...")
        logger.info(f"Number of nodes being sent: {len(nodes)}")
        logger.info("Nodes being sent to LLM:")
        for n in nodes:
            val_str = str(n.get('value', 'N/A'))[:60]
            logger.info(f"  [{n['layer']}] {n['id']}: {n['name']} = {val_str}")
        logger.info("-" * 60)
        
        resp = self._call(system, user)
        
        # Log the response
        logger.info("GPT-4o Response:")
        logger.info(resp[:800] if len(resp) > 800 else resp)
        logger.info("=" * 60)
        
        # Store in call log
        self.call_log.append({
            "type": "infer_edges",
            "n_nodes": len(nodes),
            "context": scenario_context[:100],
            "response_length": len(resp)
        })
        
        try:
            if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
            elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
            edges = json.loads(resp.strip())
            logger.info(f"Successfully parsed {len(edges)} edges")
            return edges
        except:
            logger.error(f"Failed to parse edge response: {resp}")
            return []
    
    def evaluate_counterfactual(self, dag: ScenarioDAG, intervention: Intervention,
                                outcome_var: str) -> CounterfactualResult:
        paths = dag.find_all_paths(intervention.variable_id, outcome_var)
        
        system = """Evaluate the counterfactual effect of an intervention in a driving scenario.
Return JSON: {"original_outcome": "...", "counterfactual_outcome": "...", 
"effect_direction": "increase|decrease|unchanged", "confidence": 0.0-1.0, "reasoning": "..."}"""
        
        nodes_summary = [{"id": n.id, "name": n.name, "value": _to_python_types(n.value), "layer": n.layer} 
                         for n in dag.nodes.values()]
        user = f"""Intervention: do({intervention.variable_id} = {intervention.value})
Original value: {intervention.original_value}
Outcome variable: {outcome_var}
Causal paths affected: {len(paths)}

Nodes: {json.dumps(nodes_summary)}

Return JSON only:"""
        
        # Log counterfactual evaluation
        logger.debug(f"Counterfactual: do({intervention.variable_id}={intervention.value})")
        logger.debug(f"  Original: {intervention.original_value}, Paths: {len(paths)}")
        
        resp = self._call(system, user)
        try:
            if "```json" in resp: resp = resp.split("```json")[1].split("```")[0]
            elif "```" in resp: resp = resp.split("```")[1].split("```")[0]
            r = json.loads(resp.strip())
            
            logger.info(f"CF: do({intervention.variable_id}={intervention.value}) -> {r.get('effect_direction', 'unknown')}")
            
            return CounterfactualResult(
                intervention=intervention, outcome_variable=outcome_var,
                original_outcome=r.get("original_outcome"),
                counterfactual_outcome=r.get("counterfactual_outcome"),
                effect_direction=r.get("effect_direction", "unknown"),
                confidence=r.get("confidence", 0.5),
                reasoning=r.get("reasoning", ""),
                affected_paths=paths)
        except:
            return CounterfactualResult(intervention=intervention, outcome_variable=outcome_var,
                original_outcome=None, counterfactual_outcome=None, effect_direction="unknown",
                confidence=0.0, reasoning="Parse error", affected_paths=paths)


# =============================================================================
# Grounded DAG Constructor
# =============================================================================

class GroundedDAGConstructor:
    """
    Constructs DAGs grounded in extracted features with enforced structure.
    
    The LLM can only infer edges - nodes come from:
    - Trajectory data (ego/agent initial states)
    - VLM extraction (maneuvers, decisions)
    - Fixed outcome node
    """
    
    def __init__(self, client: DAGClient):
        self.client = client
    
    def construct(self, scenario_features: Any, trajectory: Optional[np.ndarray] = None,
                  other_agents: Optional[List[Dict]] = None, scenario_id: str = "") -> ScenarioDAG:
        """
        Construct a grounded DAG.
        
        Args:
            scenario_features: VLM-extracted features (maneuvers, decisions)
            trajectory: Ego trajectory array (T, 4) with [x, y, heading, speed]
            other_agents: List of other agent state dicts
            scenario_id: Scenario identifier
        """
        # Get scenario ID
        if hasattr(scenario_features, 'scenario_id'):
            scenario_id = scenario_features.scenario_id
        elif isinstance(scenario_features, dict):
            scenario_id = scenario_features.get('scenario_id', scenario_id)
        
        dag = ScenarioDAG(scenario_id)
        
        # Layer 0: Initial state nodes
        self._add_initial_state_nodes(dag, trajectory, other_agents)
        
        # Layer 1: Event nodes from VLM extraction
        self._add_event_nodes(dag, scenario_features)
        
        # Layer 2: Outcome node
        dag.add_node(DAGNode(
            id="collision_outcome", name="Collision Outcome", 
            node_type=NodeType.OUTCOME, layer=2,
            description="Whether collision occurs or is avoided"))
        
        # Build node list for LLM
        nodes_for_llm = [n.to_dict() for n in dag.nodes.values()]
        
        # Build context string
        context = self._build_context(scenario_features, trajectory)
        
        # Infer edges using LLM (only edges, no new nodes)
        edge_data = self.client.infer_edges(nodes_for_llm, context)
        
        # Add edges with validation
        valid_node_ids = set(dag.nodes.keys())
        for ed in edge_data:
            if ed.get("parent") in valid_node_ids and ed.get("child") in valid_node_ids:
                dag.add_edge(DAGEdge(
                    parent_id=ed["parent"], child_id=ed["child"],
                    mechanism=ed.get("mechanism", ""),
                    confidence=ed.get("confidence", 0.8)))
            else:
                logger.warning(f"Skipping invalid edge: {ed}")
        
        # Ensure connectivity: all layer 1 nodes connect to outcome
        self._ensure_connectivity(dag)
        
        dag.metadata["llm_reasoning"] = f"Edges inferred between {len(dag.nodes)} grounded nodes"
        logger.info(f"Constructed DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        return dag
    
    def _add_initial_state_nodes(self, dag: ScenarioDAG, trajectory: Optional[np.ndarray],
                                  other_agents: Optional[List[Dict]]) -> None:
        """Add layer 0 nodes from trajectory data."""
        if trajectory is not None and len(trajectory) > 0:
            # Initial ego state
            t0 = trajectory[0]
            if len(t0) >= 4:
                x, y, heading, speed = t0[0], t0[1], t0[2], t0[3]
            elif len(t0) >= 2:
                x, y = t0[0], t0[1]
                heading = t0[2] if len(t0) > 2 else 0
                speed = t0[3] if len(t0) > 3 else 0
            else:
                x, y, heading, speed = 0, 0, 0, 0
            
            dag.add_node(DAGNode(
                id="ego_initial_position", name="Ego Initial Position",
                node_type=NodeType.EGO_STATE, layer=0, timestamp=0.0,
                value={"x": float(x), "y": float(y)},
                description=f"Ego position at t=0: ({x:.1f}, {y:.1f})"))
            
            dag.add_node(DAGNode(
                id="ego_initial_speed", name="Ego Initial Speed",
                node_type=NodeType.EGO_STATE, layer=0, timestamp=0.0,
                value=float(speed),
                description=f"Ego speed at t=0: {speed:.1f} m/s",
                metadata={"alternatives": [speed * 0.5, speed * 0.75, speed, speed * 1.25] if speed > 0 else [0, 5, 10]}))
            
            dag.add_node(DAGNode(
                id="ego_initial_heading", name="Ego Initial Heading",
                node_type=NodeType.EGO_STATE, layer=0, timestamp=0.0,
                value=float(heading),
                description=f"Ego heading at t=0: {heading:.2f} rad"))
        
        # Other agents
        if other_agents:
            for i, agent in enumerate(other_agents):
                agent_id = agent.get("agent_id", f"agent_{i}")
                agent_type = agent.get("type", "vehicle")
                pos = agent.get("position", (0, 0))
                speed = agent.get("speed", 0)
                
                dag.add_node(DAGNode(
                    id=f"{agent_id}_state", name=f"{agent_type.title()} {agent_id}",
                    node_type=NodeType.AGENT_STATE, layer=0, timestamp=0.0,
                    value={"position": pos, "speed": speed, "type": agent_type},
                    description=f"{agent_type} at ({pos[0]:.1f}, {pos[1]:.1f}), speed {speed:.1f}"))
    
    def _add_event_nodes(self, dag: ScenarioDAG, features: Any) -> None:
        """Add layer 1 nodes from VLM-extracted features."""
        # Get maneuvers and decisions with correct attribute names from VLM extractor
        # VLM extractor uses: maneuver_sequence, critical_decisions
        # Each ManeuverSegment has: maneuver_type, start_timestamp, end_timestamp, aggressiveness
        # Each CriticalDecisionPoint has: decision_type, ground_truth_choice, timestamp, alternatives
        
        maneuvers = []
        decisions = []
        
        if isinstance(features, dict):
            maneuvers = features.get("maneuver_sequence", []) or features.get("maneuvers", [])
            decisions = features.get("critical_decisions", []) or features.get("decisions", [])
        else:
            # Check maneuver_sequence first (VLM extractor uses this name)
            if hasattr(features, "maneuver_sequence") and features.maneuver_sequence:
                maneuvers = features.maneuver_sequence
            elif hasattr(features, "maneuvers") and features.maneuvers:
                maneuvers = features.maneuvers
            
            # Check critical_decisions first (VLM extractor uses this name)
            if hasattr(features, "critical_decisions") and features.critical_decisions:
                decisions = features.critical_decisions
            elif hasattr(features, "decisions") and features.decisions:
                decisions = features.decisions
        
        logger.info(f"Adding {len(maneuvers)} maneuver nodes and {len(decisions)} decision nodes")
        
        # Add maneuver nodes
        for i, m in enumerate(maneuvers):
            if isinstance(m, dict):
                m_type = m.get("maneuver_type", m.get("type", "unknown"))
                m_ts = m.get("start_timestamp", 0)
                m_desc = m.get("description", "")
                m_aggr = m.get("aggressiveness", "normal")
            else:
                # ManeuverSegment dataclass uses maneuver_type (which is an enum)
                m_type_raw = getattr(m, "maneuver_type", None) or getattr(m, "type", "unknown")
                m_type = m_type_raw.value if hasattr(m_type_raw, 'value') else str(m_type_raw)
                m_ts = getattr(m, "start_timestamp", 0)
                m_desc = getattr(m, "description", "")
                m_aggr_raw = getattr(m, "aggressiveness", "normal")
                m_aggr = m_aggr_raw.value if hasattr(m_aggr_raw, 'value') else str(m_aggr_raw)
            
            dag.add_node(DAGNode(
                id=f"maneuver_{i}", name=f"Maneuver: {m_type}",
                node_type=NodeType.MANEUVER, layer=1, timestamp=float(m_ts),
                value=m_type, description=m_desc,
                metadata={"aggressiveness": m_aggr, 
                          "alternatives": self._maneuver_alternatives(m_type)}))
        
        # Add decision nodes
        for i, d in enumerate(decisions):
            if isinstance(d, dict):
                d_type = d.get("decision_type", d.get("type", "unknown"))
                d_choice = d.get("ground_truth_choice", d.get("choice", "unknown"))
                d_ts = d.get("timestamp", 0)
                d_desc = d.get("description", "")
                d_alts = d.get("alternatives", [])
                d_conf = d.get("confidence", 1.0)
            else:
                # CriticalDecisionPoint dataclass uses decision_type (enum) and ground_truth_choice
                d_type_raw = getattr(d, "decision_type", None) or getattr(d, "type", "unknown")
                d_type = d_type_raw.value if hasattr(d_type_raw, 'value') else str(d_type_raw)
                d_choice = getattr(d, "ground_truth_choice", None) or getattr(d, "choice", "unknown")
                d_ts = getattr(d, "timestamp", 0)
                d_desc = getattr(d, "description", "")
                d_alts = getattr(d, "alternatives", [])
                d_conf = getattr(d, "confidence", 1.0)
            
            dag.add_node(DAGNode(
                id=f"decision_{i}", name=f"Decision: {d_type}",
                node_type=NodeType.DECISION, layer=1, timestamp=d_ts,
                value=d_choice, description=d_desc,
                metadata={"alternatives": d_alts if d_alts else self._decision_alternatives(d_choice),
                          "confidence": d_conf}))
    
    def _maneuver_alternatives(self, m_type: str) -> List[str]:
        """
        Get alternative maneuvers for counterfactual generation.
        
        Returns alternatives that map to the new granular token sets:
        - Lane changes: lane_change_{left,right}
        - Turns: left_turn, right_turn (mapped to turn_{left,right}_moderate)
        """
        alts = {
            # Turns - alternatives include going straight or stopping
            "left_turn": ["left_turn", "straight", "stop"],
            "right_turn": ["right_turn", "straight", "stop"],
            # Lane changes - alternatives include staying straight
            "lane_change_left": ["lane_change_left", "lane_change_right", "straight"],
            "lane_change_right": ["lane_change_right", "lane_change_left", "straight"],
            # Basic maneuvers
            "straight": ["straight", "lane_change_left", "lane_change_right", "stop"],
            "stop": ["stop", "decelerate", "straight"],
            "decelerate": ["decelerate", "stop", "straight", "accelerate"],
            "accelerate": ["accelerate", "straight", "decelerate"],
        }
        return alts.get(m_type, [m_type])
    
    def _decision_alternatives(self, choice: str) -> List[str]:
        alts = {"proceed": ["proceed", "yield", "stop"], "yield": ["proceed", "yield", "stop"],
                "maintain_course": ["maintain_course", "brake", "swerve"],
                "accept_gap": ["accept_gap", "reject_gap"], "reject_gap": ["accept_gap", "reject_gap"]}
        return alts.get(choice, [choice])
    
    def _build_context(self, features: Any, trajectory: Optional[np.ndarray]) -> str:
        """Build context string for LLM."""
        if isinstance(features, dict):
            sid = features.get("scenario_id", "unknown")
        else:
            sid = getattr(features, "scenario_id", "unknown")
        
        lines = [f"Scenario: {sid}"]
        if trajectory is not None:
            lines.append(f"Trajectory length: {len(trajectory)} steps")
            if len(trajectory) > 0:
                t0, tf = trajectory[0], trajectory[-1]
                lines.append(f"Start: ({t0[0]:.1f}, {t0[1]:.1f}), End: ({tf[0]:.1f}, {tf[1]:.1f})")
        return "\n".join(lines)
    
    def _ensure_connectivity(self, dag: ScenarioDAG) -> None:
        """Ensure all layer 1 nodes connect to outcome."""
        outcome_id = "collision_outcome"
        for node in dag.get_nodes_by_layer(1):
            # Check if this node has a path to outcome
            if outcome_id not in dag._adjacency.get(node.id, set()) and outcome_id != node.id:
                dag.add_edge(DAGEdge(
                    parent_id=node.id, child_id=outcome_id,
                    mechanism="Event affects collision outcome",
                    confidence=0.7))
    
    def evaluate_counterfactuals(self, dag: ScenarioDAG, 
                                  outcome_var: str = "collision_outcome") -> List[CounterfactualResult]:
        """Evaluate all possible interventions."""
        results = []
        for intv in dag.enumerate_interventions():
            result = self.client.evaluate_counterfactual(dag, intv, outcome_var)
            results.append(result)
            logger.info(f"CF: do({intv.variable_id}={intv.value}) -> {result.effect_direction}")
        return results


# Backwards compatibility aliases
DAGConstructor = GroundedDAGConstructor


# =============================================================================
# Main
# =============================================================================

def main():
    """Example usage."""
    print("=" * 60)
    print("Grounded DAG Constructor Example")
    print("=" * 60)
    
    # Mock features
    features = {
        "scenario_id": "grounded_test_001",
        "maneuvers": [
            {"type": "left_turn", "start_timestamp": 0.2, "description": "Left turn at intersection",
             "aggressiveness": "normal"},
            {"type": "straight", "start_timestamp": 0.6, "description": "Continue straight"}
        ],
        "decisions": [
            {"type": "proceed_or_yield", "choice": "proceed", "timestamp": 0.15,
             "alternatives": ["proceed", "yield"], "description": "Chose to proceed"},
            {"type": "gap_acceptance", "choice": "accept_gap", "timestamp": 0.5,
             "alternatives": ["accept_gap", "reject_gap"], "description": "Accepted gap"}
        ]
    }
    
    # Mock trajectory
    trajectory = np.array([
        [0, 0, 0, 10],      # t=0: position (0,0), heading 0, speed 10
        [10, 0, 0.1, 10],   # t=0.1
        [20, 2, 0.3, 9],    # ...
        [28, 8, 0.5, 8],
        [35, 15, 0.7, 7],
    ])
    
    # Other agents
    other_agents = [
        {"agent_id": "vehicle_1", "type": "vehicle", "position": (50, -5), "speed": 12},
        {"agent_id": "ped_1", "type": "pedestrian", "position": (30, 10), "speed": 1.5}
    ]
    
    client = MockDAGClient()
    constructor = GroundedDAGConstructor(client)
    
    dag = constructor.construct(features, trajectory, other_agents)
    
    print("\n" + dag.summary())
    print("\nNodes:")
    for n in dag.nodes.values():
        print(f"  [{n.layer}] {n.id}: {n.name} = {n.value}")
    
    print("\nEdges:")
    for e in dag.edges:
        print(f"  {e.parent_id} -> {e.child_id}: {e.mechanism}")
    
    print("\nInterventions:")
    for intv in dag.enumerate_interventions():
        print(f"  {intv.description}")
    
    print("\nCounterfactuals:")
    for cf in constructor.evaluate_counterfactuals(dag):
        print(f"  do({cf.intervention.variable_id}={cf.intervention.value}): {cf.effect_direction}")
    
    print("\n" + dag.to_json())


if __name__ == "__main__":
    main()