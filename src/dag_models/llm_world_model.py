"""
LLM World Model for Counterfactual Propagation.

Uses an LLM as a universal conditional probability oracle to propagate
causal effects through the DAG. When intervening on a node, the LLM
reasons about physics and causality to determine downstream effects.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
import json
import re

# Import from sibling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.causal_dag import CausalDAG, create_default_dag
from data.veacon_schema import VeaconEvent, create_example_event


class LLMWorldModel:
    """
    LLM-based world model for counterfactual crash analysis.
    
    Core idea: Instead of learning explicit P(child|parents) distributions,
    use an LLM's implicit world knowledge to simulate causal propagation.
    
    When we intervene on a node (e.g., weather: rain → clear):
    1. Identify all descendants in the causal DAG
    2. Process them in topological order
    3. For each descendant, query the LLM: "Given these parent states, what should this child be?"
    4. LLM reasons about physics/causality and returns the new value
    """
    
    def __init__(
        self,
        dag: Optional[CausalDAG] = None,
        llm_client: Any = None,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3
    ):
        """
        Initialize the LLM world model.
        
        Args:
            dag: CausalDAG instance. Uses default if not provided.
            llm_client: Anthropic client or compatible API client.
            model_name: LLM model to use for reasoning.
            temperature: Sampling temperature (lower = more deterministic).
        """
        self.dag = dag or create_default_dag()
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature
    
    def propagate_intervention(
        self,
        event: VeaconEvent,
        intervention: Dict[str, Any],
        verbose: bool = False
    ) -> VeaconEvent:
        """
        Apply an intervention and propagate effects through the DAG.
        
        Args:
            event: Original VeaconEvent
            intervention: Dict of {node_name: new_value} to intervene on
            verbose: Whether to print reasoning steps
        
        Returns:
            New VeaconEvent with propagated counterfactual values
        
        Example:
            >>> event = create_example_event()  # rainy crash
            >>> cf_event = model.propagate_intervention(
            ...     event, 
            ...     {"environment.weather": "clear"}
            ... )
            >>> # cf_event now has dry surface, better braking, lower severity
        """
        # Deep copy to avoid modifying original
        cf_event = deepcopy(event)
        cf_dict = cf_event.to_flat_dict()
        
        # Apply the intervention directly
        for node, value in intervention.items():
            cf_dict[node] = value
            if verbose:
                print(f"[INTERVENTION] {node} := {value}")
        
        # Find all descendants of intervened nodes
        all_descendants = set()
        for node in intervention.keys():
            all_descendants.update(self.dag.get_descendants(node))
        
        if verbose:
            print(f"[PROPAGATION] Descendants to update: {all_descendants}")
        
        # Process descendants in topological order
        sorted_descendants = self.dag.topological_sort(all_descendants)
        
        for node in sorted_descendants:
            if node in intervention:
                continue  # Skip nodes we directly intervened on
            
            # Get current parent values
            parent_values = {}
            for parent in self.dag.get_parents(node):
                parent_values[parent] = cf_dict.get(parent)
            
            # Query LLM for new value
            new_value = self._query_llm_for_node(
                node, parent_values, cf_dict, verbose
            )
            
            if new_value is not None:
                cf_dict[node] = new_value
                if verbose:
                    print(f"[UPDATE] {node} = {new_value}")
        
        # Convert back to VeaconEvent
        return VeaconEvent.from_flat_dict(cf_dict)
    
    def _query_llm_for_node(
        self,
        node: str,
        parent_values: Dict[str, Any],
        current_state: Dict[str, Any],
        verbose: bool = False
    ) -> Any:
        """
        Query the LLM to determine a node's value given its parents.
        
        If no LLM client is available, uses rule-based fallback.
        """
        dag_node = self.dag.get_node(node)
        if dag_node is None:
            return current_state.get(node)
        
        # If we have an LLM client, use it
        if self.llm_client is not None:
            return self._query_llm_api(node, parent_values, dag_node, verbose)
        
        # Otherwise, use rule-based fallback
        return self._rule_based_propagation(node, parent_values, current_state)
    
    def _query_llm_api(
        self,
        node: str,
        parent_values: Dict[str, Any],
        dag_node: Any,
        verbose: bool
    ) -> Any:
        """Query the actual LLM API."""
        prompt = self._build_propagation_prompt(node, parent_values, dag_node)
        
        try:
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            return self._parse_llm_response(response_text, dag_node)
            
        except Exception as e:
            if verbose:
                print(f"[LLM ERROR] {e}")
            return None
    
    def _build_propagation_prompt(
        self,
        node: str,
        parent_values: Dict[str, Any],
        dag_node: Any
    ) -> str:
        """Build the prompt for LLM causal reasoning."""
        
        valid_values_str = ""
        if dag_node.valid_values:
            valid_values_str = f"\nValid values: {dag_node.valid_values}"
        
        prompt = f"""You are a traffic safety expert reasoning about crash causality.

Given the following parent node values in a causal graph:
{json.dumps(parent_values, indent=2)}

Determine what the value of the child node should be:
- Node: {node}
- Description: {dag_node.description}
- Type: {dag_node.value_type}{valid_values_str}

Think step-by-step about the causal physics:
1. How do the parent values affect this child node?
2. What is the most likely value given these conditions?

Respond with JSON in this exact format:
{{"reasoning": "your step-by-step reasoning", "value": <the value>}}

The value must be one of the valid values if specified, or a reasonable number for continuous variables."""

        return prompt
    
    def _parse_llm_response(self, response_text: str, dag_node: Any) -> Any:
        """Parse the LLM response to extract the value."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"value"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("value")
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return None
    
    def _rule_based_propagation(
        self,
        node: str,
        parent_values: Dict[str, Any],
        current_state: Dict[str, Any]
    ) -> Any:
        """
        Rule-based fallback for propagation without LLM.
        
        These rules encode basic physics and causal knowledge.
        """
        
        # Weather → Surface condition
        if node == "environment.surface_condition":
            weather = parent_values.get("environment.weather", "clear")
            if weather in ["rain", "sleet"]:
                return "wet"
            elif weather == "snow":
                return "snow_ice"
            else:
                return "dry"
        
        # Weather + Light → Visibility
        if node == "environment.visibility":
            weather = parent_values.get("environment.weather", "clear")
            light = parent_values.get("environment.light", "daylight")
            
            if weather in ["fog", "snow"] or light == "dark":
                return "poor"
            elif weather == "rain" or light in ["dawn", "dusk"]:
                return "moderate"
            else:
                return "good"
        
        # Surface + Speed → Braking effectiveness
        if node == "crash_dynamics.braking_effectiveness":
            surface = parent_values.get("environment.surface_condition", "dry")
            
            if surface == "snow_ice":
                return "low"
            elif surface == "wet":
                return "medium"
            else:
                return "high"
        
        # Pre-crash speed + Braking → Impact speed
        if node == "crash_state.speed_at_impact_kph":
            pre_speed = parent_values.get("vehicle_state.pre_crash_speed_kph", 50)
            braking = parent_values.get("crash_dynamics.braking_effectiveness", "medium")
            
            if pre_speed is None:
                pre_speed = 50
            
            reduction_factor = {
                "none": 1.0,
                "low": 0.85,
                "medium": 0.65,
                "high": 0.4
            }.get(braking, 0.65)
            
            return round(pre_speed * reduction_factor, 1)
        
        # Braking + Visibility + Conflict → Event type
        if node == "accident.event_type":
            braking = parent_values.get("crash_dynamics.braking_effectiveness", "medium")
            visibility = current_state.get("environment.visibility", "good")
            
            if braking == "high" and visibility == "good":
                return "near_crash"
            else:
                return "crash"
        
        # Impact speed + Event type → Delta-V
        if node == "crash_state.delta_v_kph":
            impact_speed = parent_values.get("crash_state.speed_at_impact_kph", 50)
            event_type = parent_values.get("accident.event_type", "crash")
            
            if impact_speed is None:
                impact_speed = 50
            
            if event_type == "near_crash":
                return round(impact_speed * 0.1, 1)  # Minor contact
            elif event_type == "crash":
                return round(impact_speed * 0.6, 1)  # Significant collision
            else:
                return 0
        
        # Delta-V → Max acceleration
        if node == "crash_state.max_accel_mps2":
            delta_v = parent_values.get("crash_state.delta_v_kph", 30)
            if delta_v is None:
                delta_v = 30
            # Rough estimate: assuming 100ms collision duration
            return round(delta_v / 3.6 / 0.1, 1)
        
        # Delta-V + Event type → Severity
        if node == "injury_outcome.severity_score":
            delta_v = parent_values.get("crash_state.delta_v_kph", 30)
            event_type = parent_values.get("accident.event_type", "crash")
            
            if delta_v is None:
                delta_v = 30
            
            if event_type == "near_crash":
                return round(min(0.1, delta_v / 100), 2)
            elif event_type == "crash":
                # Sigmoid-like mapping
                severity = 1 / (1 + 2.718 ** (-(delta_v - 40) / 15))
                return round(severity, 2)
            else:
                return 0.0
        
        # Default: return current value
        return current_state.get(node)
    
    def batch_propagate(
        self,
        event: VeaconEvent,
        interventions: List[Dict[str, Any]]
    ) -> List[VeaconEvent]:
        """
        Apply multiple interventions and get counterfactual events.
        
        Useful for exploring "what-if" scenarios.
        """
        return [
            self.propagate_intervention(event, intervention)
            for intervention in interventions
        ]
    
    def compute_severity_change(
        self,
        original: VeaconEvent,
        counterfactual: VeaconEvent
    ) -> float:
        """Compute the change in severity between original and counterfactual."""
        return counterfactual.injury_outcome.severity_score - original.injury_outcome.severity_score


def demo_propagation():
    """Demonstrate counterfactual propagation."""
    
    print("=" * 60)
    print("LLM WORLD MODEL - COUNTERFACTUAL PROPAGATION DEMO")
    print("=" * 60)
    
    # Create model (without actual LLM client - uses rules)
    model = LLMWorldModel()
    
    # Create example crash event
    event = create_example_event()
    print("\n📍 ORIGINAL EVENT:")
    original_dict = event.to_flat_dict()
    for key, value in original_dict.items():
        if value is not None:
            print(f"   {key}: {value}")
    
    # Intervention: Change weather from rain to clear
    print("\n🔧 INTERVENTION: weather := clear")
    print("-" * 40)
    
    cf_event = model.propagate_intervention(
        event,
        {"environment.weather": "clear"},
        verbose=True
    )
    
    print("\n📍 COUNTERFACTUAL EVENT:")
    cf_dict = cf_event.to_flat_dict()
    for key in original_dict:
        orig_val = original_dict.get(key)
        cf_val = cf_dict.get(key)
        if orig_val != cf_val:
            print(f"   {key}: {orig_val} → {cf_val} ⚡")
        elif cf_val is not None:
            print(f"   {key}: {cf_val}")
    
    # Compute severity change
    severity_change = model.compute_severity_change(event, cf_event)
    print(f"\n📊 SEVERITY CHANGE: {severity_change:+.2f}")
    print(f"   Original: {event.injury_outcome.severity_score:.2f}")
    print(f"   Counterfactual: {cf_event.injury_outcome.severity_score:.2f}")
    
    if severity_change < 0:
        print(f"   → Intervention REDUCED severity by {abs(severity_change)*100:.1f}%")
    else:
        print(f"   → Intervention INCREASED severity by {severity_change*100:.1f}%")


if __name__ == "__main__":
    demo_propagation()
