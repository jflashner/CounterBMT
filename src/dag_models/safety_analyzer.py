"""
Safety-Critical Feature Analyzer.

Identifies which features are most safety-critical in a given crash scenario
using counterfactual reasoning: "Which interventions would most reduce severity?"
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.causal_dag import CausalDAG, create_default_dag
from models.llm_world_model import LLMWorldModel
from data.veacon_schema import VeaconEvent, create_example_event


@dataclass
class InterventionResult:
    """Result of a single intervention."""
    node: str
    original_value: Any
    intervention_value: Any
    original_severity: float
    counterfactual_severity: float
    severity_reduction: float
    reasoning: Optional[str] = None


@dataclass
class SafetyRanking:
    """Complete safety-critical feature ranking for an event."""
    event_id: Optional[str]
    original_severity: float
    rankings: List[InterventionResult] = field(default_factory=list)
    
    def get_top_k(self, k: int = 5) -> List[InterventionResult]:
        """Get top-k most impactful interventions."""
        return self.rankings[:k]
    
    def get_total_potential_reduction(self) -> float:
        """Get maximum severity reduction from best intervention."""
        if not self.rankings:
            return 0.0
        return self.rankings[0].severity_reduction


class SafetyCriticalAnalyzer:
    """
    Analyzes crash events to identify safety-critical features.
    
    For each modifiable feature, tries safer alternatives and measures
    the counterfactual severity reduction. Features are ranked by their
    potential to reduce injury severity.
    """
    
    # Define safer alternatives for each feature
    SAFER_ALTERNATIVES = {
        "environment.weather": {
            "rain": ["clear"],
            "snow": ["clear", "rain"],
            "fog": ["clear"],
            "sleet": ["clear", "rain"]
        },
        "environment.light": {
            "dark": ["daylight", "dark_lighted"],
            "dawn": ["daylight"],
            "dusk": ["daylight"]
        },
        "environment.surface_condition": {
            "wet": ["dry"],
            "snow_ice": ["dry", "wet"]
        },
        "environment.visibility": {
            "poor": ["good", "moderate"],
            "moderate": ["good"]
        },
        "crash_dynamics.braking_effectiveness": {
            "none": ["high", "medium", "low"],
            "low": ["high", "medium"],
            "medium": ["high"]
        },
        # For continuous variables, we define reduction factors
        "vehicle_state.pre_crash_speed_kph": "reduce_30_50",
        "environment.speed_limit_kph": "reduce_20"
    }
    
    def __init__(
        self,
        world_model: Optional[LLMWorldModel] = None,
        dag: Optional[CausalDAG] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            world_model: LLMWorldModel for counterfactual propagation.
            dag: CausalDAG defining causal structure.
        """
        self.dag = dag or create_default_dag()
        self.world_model = world_model or LLMWorldModel(dag=self.dag)
    
    def compute_safety_impact(
        self,
        event: VeaconEvent,
        node_name: str
    ) -> Optional[InterventionResult]:
        """
        Compute safety impact of intervening on a specific feature.
        
        Tries all safer alternatives and returns the best result.
        """
        event_dict = event.to_flat_dict()
        current_value = event_dict.get(node_name)
        original_severity = event.injury_outcome.severity_score
        
        if current_value is None:
            return None
        
        # Get alternatives for this feature
        alternatives = self._get_safer_alternatives(node_name, current_value)
        
        if not alternatives:
            return None
        
        # Try each alternative and find the best
        best_result = None
        best_reduction = 0
        
        for alt_value in alternatives:
            # Apply intervention
            cf_event = self.world_model.propagate_intervention(
                event,
                {node_name: alt_value}
            )
            
            cf_severity = cf_event.injury_outcome.severity_score
            reduction = original_severity - cf_severity
            
            if reduction > best_reduction:
                best_reduction = reduction
                best_result = InterventionResult(
                    node=node_name,
                    original_value=current_value,
                    intervention_value=alt_value,
                    original_severity=original_severity,
                    counterfactual_severity=cf_severity,
                    severity_reduction=reduction
                )
        
        return best_result
    
    def _get_safer_alternatives(
        self,
        node_name: str,
        current_value: Any
    ) -> List[Any]:
        """Get safer alternative values for a node."""
        
        if node_name not in self.SAFER_ALTERNATIVES:
            return []
        
        alternatives_spec = self.SAFER_ALTERNATIVES[node_name]
        
        # Categorical: look up alternatives
        if isinstance(alternatives_spec, dict):
            return alternatives_spec.get(current_value, [])
        
        # Continuous: apply reduction
        if alternatives_spec == "reduce_30_50":
            if isinstance(current_value, (int, float)) and current_value > 0:
                return [
                    round(current_value * 0.7, 1),  # 30% reduction
                    round(current_value * 0.5, 1)   # 50% reduction
                ]
        
        if alternatives_spec == "reduce_20":
            if isinstance(current_value, (int, float)) and current_value > 0:
                return [round(current_value * 0.8, 1)]
        
        return []
    
    def rank_safety_critical_features(
        self,
        event: VeaconEvent,
        include_all: bool = False
    ) -> SafetyRanking:
        """
        Rank all features by their safety-critical importance.
        
        Args:
            event: The crash event to analyze.
            include_all: Whether to include features with no improvement.
        
        Returns:
            SafetyRanking with features sorted by severity reduction potential.
        """
        # Get all potential intervention targets
        intervention_targets = self._get_intervention_targets(event)
        
        # Compute impact for each
        results = []
        for node_name in intervention_targets:
            result = self.compute_safety_impact(event, node_name)
            if result is not None:
                if include_all or result.severity_reduction > 0:
                    results.append(result)
        
        # Sort by severity reduction (descending)
        results.sort(key=lambda r: r.severity_reduction, reverse=True)
        
        return SafetyRanking(
            event_id=event.event_id,
            original_severity=event.injury_outcome.severity_score,
            rankings=results
        )
    
    def _get_intervention_targets(self, event: VeaconEvent) -> List[str]:
        """Get nodes that are valid intervention targets."""
        event_dict = event.to_flat_dict()
        
        # Only consider nodes that:
        # 1. Have a value in the event
        # 2. Are in our safer alternatives list
        # 3. Are ancestors of severity (can affect outcome)
        
        severity_ancestors = self.dag.get_ancestors("injury_outcome.severity_score")
        
        targets = []
        for node_name in self.SAFER_ALTERNATIVES.keys():
            if node_name in event_dict and event_dict[node_name] is not None:
                if node_name in severity_ancestors or node_name in self.dag.nodes:
                    targets.append(node_name)
        
        return targets
    
    def generate_explanation(
        self,
        ranking: SafetyRanking,
        top_k: int = 3
    ) -> str:
        """Generate a human-readable explanation of safety-critical factors."""
        
        lines = [
            "=" * 60,
            "SAFETY-CRITICAL FEATURE ANALYSIS",
            "=" * 60,
            "",
            f"Original Severity: {ranking.original_severity:.2f}",
            ""
        ]
        
        if not ranking.rankings:
            lines.append("No interventions found that could reduce severity.")
            return "\n".join(lines)
        
        lines.append(f"Top {min(top_k, len(ranking.rankings))} safety-critical factors:")
        lines.append("-" * 40)
        
        for i, result in enumerate(ranking.get_top_k(top_k), 1):
            pct_reduction = (result.severity_reduction / ranking.original_severity) * 100
            
            lines.append(f"\n{i}. {result.node}")
            lines.append(f"   Current: {result.original_value}")
            lines.append(f"   If changed to: {result.intervention_value}")
            lines.append(f"   Severity: {result.original_severity:.2f} → {result.counterfactual_severity:.2f}")
            lines.append(f"   Reduction: {result.severity_reduction:.2f} ({pct_reduction:.1f}%)")
        
        # Summary
        best = ranking.rankings[0]
        lines.extend([
            "",
            "-" * 40,
            "SUMMARY:",
            f"The most impactful intervention would be changing",
            f"'{best.node}' from '{best.original_value}' to '{best.intervention_value}',",
            f"which could reduce severity by {best.severity_reduction:.2f}",
            f"({(best.severity_reduction/ranking.original_severity)*100:.1f}% improvement)."
        ])
        
        return "\n".join(lines)
    
    def compare_scenarios(
        self,
        events: List[VeaconEvent],
        labels: Optional[List[str]] = None
    ) -> str:
        """Compare safety-critical factors across multiple events."""
        
        if labels is None:
            labels = [f"Event {i+1}" for i in range(len(events))]
        
        lines = ["CROSS-SCENARIO COMPARISON", "=" * 60, ""]
        
        for event, label in zip(events, labels):
            ranking = self.rank_safety_critical_features(event)
            lines.append(f"\n{label}:")
            lines.append(f"  Severity: {ranking.original_severity:.2f}")
            
            if ranking.rankings:
                top = ranking.rankings[0]
                lines.append(f"  Top factor: {top.node}")
                lines.append(f"  Potential reduction: {top.severity_reduction:.2f}")
            else:
                lines.append("  No improvement opportunities found")
        
        return "\n".join(lines)


def demo_safety_analysis():
    """Demonstrate safety-critical feature analysis."""
    
    print("=" * 60)
    print("SAFETY-CRITICAL FEATURE ANALYSIS DEMO")
    print("=" * 60)
    
    # Create analyzer
    analyzer = SafetyCriticalAnalyzer()
    
    # Create example crash event
    event = create_example_event()
    
    print("\n📍 ANALYZING CRASH EVENT:")
    event_dict = event.to_flat_dict()
    print(f"   Weather: {event_dict['environment.weather']}")
    print(f"   Surface: {event_dict['environment.surface_condition']}")
    print(f"   Light: {event_dict['environment.light']}")
    print(f"   Pre-crash speed: {event_dict['vehicle_state.pre_crash_speed_kph']} km/h")
    print(f"   Original severity: {event.injury_outcome.severity_score:.2f}")
    
    # Rank safety-critical features
    ranking = analyzer.rank_safety_critical_features(event)
    
    # Generate and print explanation
    explanation = analyzer.generate_explanation(ranking)
    print("\n" + explanation)
    
    # Show all rankings
    print("\n\n📊 FULL RANKING:")
    print("-" * 40)
    for i, r in enumerate(ranking.rankings, 1):
        print(f"{i}. {r.node}: {r.original_value} → {r.intervention_value}")
        print(f"   Severity reduction: {r.severity_reduction:.3f}")


if __name__ == "__main__":
    demo_safety_analysis()
