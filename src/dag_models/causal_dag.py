"""
Causal DAG for crash analysis.

Defines the causal graph structure based on:
- Haddon Matrix (pre-crash → crash → post-crash phases)
- VEACON ontology for traffic safety features
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from collections import deque


@dataclass
class DAGNode:
    """A node in the causal DAG."""
    name: str
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    phase: str = "unknown"  # "pre_crash", "crash", or "post_crash"
    description: str = ""
    value_type: str = "categorical"  # "categorical", "continuous", "ordinal"
    valid_values: Optional[List[Any]] = None


class CausalDAG:
    """
    Causal DAG for crash scenario modeling.
    
    Structure follows the Haddon Matrix temporal phases:
    1. Pre-crash: Environmental conditions, vehicle state, conflict type
    2. Crash: Braking effectiveness, impact dynamics
    3. Post-crash: Injury outcome
    
    Edges encode causal relationships (e.g., weather → surface condition → braking)
    """
    
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self._build_default_dag()
    
    def _build_default_dag(self):
        """Build the default crash causal DAG."""
        
        # =====================================================================
        # PRE-CRASH PHASE NODES
        # =====================================================================
        
        self.add_node(DAGNode(
            name="environment.weather",
            parents=[],
            phase="pre_crash",
            description="Weather conditions at time of event",
            value_type="categorical",
            valid_values=["clear", "rain", "snow", "fog", "sleet"]
        ))
        
        self.add_node(DAGNode(
            name="environment.light",
            parents=[],
            phase="pre_crash",
            description="Lighting conditions",
            value_type="categorical",
            valid_values=["daylight", "dark", "dark_lighted", "dawn", "dusk"]
        ))
        
        self.add_node(DAGNode(
            name="environment.road_type",
            parents=[],
            phase="pre_crash",
            description="Type of roadway",
            value_type="categorical",
            valid_values=["highway", "urban", "rural", "intersection"]
        ))
        
        self.add_node(DAGNode(
            name="environment.speed_limit_kph",
            parents=["environment.road_type"],
            phase="pre_crash",
            description="Posted speed limit in km/h",
            value_type="continuous",
            valid_values=[30, 50, 70, 90, 100, 120]
        ))
        
        self.add_node(DAGNode(
            name="environment.surface_condition",
            parents=["environment.weather"],
            phase="pre_crash",
            description="Road surface condition",
            value_type="categorical",
            valid_values=["dry", "wet", "snow_ice"]
        ))
        
        self.add_node(DAGNode(
            name="environment.visibility",
            parents=["environment.weather", "environment.light"],
            phase="pre_crash",
            description="Visibility level",
            value_type="categorical",
            valid_values=["good", "moderate", "poor"]
        ))
        
        self.add_node(DAGNode(
            name="environment.traffic_density",
            parents=["environment.road_type"],
            phase="pre_crash",
            description="Traffic density level",
            value_type="categorical",
            valid_values=["low", "medium", "high"]
        ))
        
        self.add_node(DAGNode(
            name="accident.conflict_type",
            parents=["environment.road_type", "environment.traffic_density"],
            phase="pre_crash",
            description="Type of conflict/crash scenario",
            value_type="categorical",
            valid_values=["lead_vehicle", "following_vehicle", "adjacent_vehicle", 
                         "opposing_vehicle", "pedestrian", "cyclist", "fixed_object", 
                         "single_vehicle"]
        ))
        
        self.add_node(DAGNode(
            name="vehicle_state.pre_crash_speed_kph",
            parents=["environment.speed_limit_kph", "environment.traffic_density",
                    "environment.surface_condition"],
            phase="pre_crash",
            description="Vehicle speed before crash in km/h",
            value_type="continuous",
            valid_values=list(range(0, 200, 10))
        ))
        
        # =====================================================================
        # CRASH PHASE NODES
        # =====================================================================
        
        self.add_node(DAGNode(
            name="crash_dynamics.braking_effectiveness",
            parents=["environment.surface_condition", "vehicle_state.pre_crash_speed_kph"],
            phase="crash",
            description="Effectiveness of braking action",
            value_type="categorical",
            valid_values=["none", "low", "medium", "high"]
        ))
        
        self.add_node(DAGNode(
            name="accident.event_type",
            parents=["crash_dynamics.braking_effectiveness", "accident.conflict_type",
                    "environment.visibility"],
            phase="crash",
            description="Whether event resulted in crash",
            value_type="categorical",
            valid_values=["normal", "near_crash", "crash"]
        ))
        
        self.add_node(DAGNode(
            name="crash_state.speed_at_impact_kph",
            parents=["vehicle_state.pre_crash_speed_kph", "crash_dynamics.braking_effectiveness"],
            phase="crash",
            description="Vehicle speed at moment of impact in km/h",
            value_type="continuous",
            valid_values=list(range(0, 200, 10))
        ))
        
        self.add_node(DAGNode(
            name="crash_state.point_of_impact",
            parents=["accident.conflict_type"],
            phase="crash",
            description="Primary point of vehicle impact",
            value_type="categorical",
            valid_values=["front", "rear", "left", "right", "rollover"]
        ))
        
        self.add_node(DAGNode(
            name="crash_state.delta_v_kph",
            parents=["crash_state.speed_at_impact_kph", "accident.event_type",
                    "accident.conflict_type"],
            phase="crash",
            description="Change in velocity during collision in km/h",
            value_type="continuous",
            valid_values=list(range(0, 100, 5))
        ))
        
        self.add_node(DAGNode(
            name="crash_state.max_accel_mps2",
            parents=["crash_state.delta_v_kph"],
            phase="crash",
            description="Maximum acceleration during crash in m/s²",
            value_type="continuous",
            valid_values=list(range(0, 100, 5))
        ))
        
        # =====================================================================
        # POST-CRASH PHASE NODES
        # =====================================================================
        
        self.add_node(DAGNode(
            name="injury_outcome.severity_score",
            parents=["crash_state.delta_v_kph", "crash_state.max_accel_mps2",
                    "crash_state.point_of_impact", "accident.event_type"],
            phase="post_crash",
            description="Injury severity score from 0 (none) to 1 (fatal)",
            value_type="continuous",
            valid_values=[round(x * 0.1, 1) for x in range(11)]
        ))
        
        # Build children lists from parent relationships
        self._build_children()
    
    def add_node(self, node: DAGNode):
        """Add a node to the DAG."""
        self.nodes[node.name] = node
    
    def _build_children(self):
        """Build children lists from parent relationships."""
        for name, node in self.nodes.items():
            for parent_name in node.parents:
                if parent_name in self.nodes:
                    if name not in self.nodes[parent_name].children:
                        self.nodes[parent_name].children.append(name)
    
    def get_node(self, name: str) -> Optional[DAGNode]:
        """Get a node by name."""
        return self.nodes.get(name)
    
    def get_parents(self, name: str) -> List[str]:
        """Get parent node names."""
        node = self.nodes.get(name)
        return node.parents if node else []
    
    def get_children(self, name: str) -> List[str]:
        """Get child node names."""
        node = self.nodes.get(name)
        return node.children if node else []
    
    def get_ancestors(self, name: str) -> Set[str]:
        """Get all ancestor node names (recursive parents)."""
        ancestors = set()
        to_visit = deque(self.get_parents(name))
        
        while to_visit:
            current = to_visit.popleft()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))
        
        return ancestors
    
    def get_descendants(self, name: str) -> Set[str]:
        """Get all descendant node names (recursive children)."""
        descendants = set()
        to_visit = deque(self.get_children(name))
        
        while to_visit:
            current = to_visit.popleft()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))
        
        return descendants
    
    def topological_sort(self, nodes: Optional[Set[str]] = None) -> List[str]:
        """
        Return nodes in topological order (parents before children).
        
        Args:
            nodes: Optional subset of nodes to sort. If None, sorts all nodes.
        
        Returns:
            List of node names in topological order.
        """
        if nodes is None:
            nodes = set(self.nodes.keys())
        
        # Kahn's algorithm
        in_degree = {n: 0 for n in nodes}
        for n in nodes:
            for parent in self.get_parents(n):
                if parent in nodes:
                    in_degree[n] += 1
        
        # Start with nodes that have no parents in the subset
        queue = deque([n for n in nodes if in_degree[n] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            for child in self.get_children(current):
                if child in nodes:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        return result
    
    def get_nodes_by_phase(self, phase: str) -> List[str]:
        """Get all nodes in a specific phase."""
        return [name for name, node in self.nodes.items() if node.phase == phase]
    
    def get_intervention_targets(self) -> List[str]:
        """
        Get nodes that are good targets for intervention.
        
        These are typically:
        - Root nodes (no parents) - environmental factors
        - Pre-crash nodes that are modifiable
        """
        targets = []
        for name, node in self.nodes.items():
            if node.phase == "pre_crash":
                targets.append(name)
        return targets
    
    def visualize(self) -> str:
        """Generate a text visualization of the DAG."""
        lines = ["CAUSAL DAG STRUCTURE", "=" * 50, ""]
        
        for phase in ["pre_crash", "crash", "post_crash"]:
            lines.append(f"[{phase.upper().replace('_', '-')} PHASE]")
            lines.append("-" * 30)
            
            for name in self.get_nodes_by_phase(phase):
                node = self.nodes[name]
                parents_str = ", ".join(node.parents) if node.parents else "(root)"
                lines.append(f"  {name}")
                lines.append(f"    ← {parents_str}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export DAG structure to dictionary."""
        return {
            name: {
                "parents": node.parents,
                "children": node.children,
                "phase": node.phase,
                "description": node.description,
                "value_type": node.value_type,
                "valid_values": node.valid_values
            }
            for name, node in self.nodes.items()
        }


def create_default_dag() -> CausalDAG:
    """Create and return the default causal DAG."""
    return CausalDAG()


if __name__ == "__main__":
    # Demo
    dag = create_default_dag()
    print(dag.visualize())
    
    print("\nDescendants of 'environment.weather':")
    print(dag.get_descendants("environment.weather"))
    
    print("\nAncestors of 'injury_outcome.severity_score':")
    print(dag.get_ancestors("injury_outcome.severity_score"))
    
    print("\nTopological order of all nodes:")
    print(dag.topological_sort())
