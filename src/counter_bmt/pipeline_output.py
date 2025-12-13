"""
Pipeline Output Module for CounterBMT

Provides comprehensive output logging and export functionality for the CounterBMT pipeline.
Captures and exports:
- Causal DAG structure and metadata
- Counterfactual interventions and results
- LLM interaction logs
- Trajectory metrics
- Scenario metadata

Author: CounterBMT Project
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import hashlib

import numpy as np

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return super().default(obj)


@dataclass
class LLMLogEntry:
    """Single LLM interaction log entry."""
    
    timestamp: str
    log_type: str  # "vlm_extraction", "dag_construction", "counterfactual_eval"
    model: str
    prompt: str
    response: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InterventionResult:
    """Result of a single intervention."""
    
    intervention_id: str
    intervention_name: str
    description: str
    target_node: str
    original_value: Any
    new_value: Any
    
    # Predicted effect from LLM
    predicted_effect: str  # "increase", "decrease", "none"
    prediction_confidence: float
    prediction_reasoning: str
    
    # Actual generation results
    trajectories: List[List[List[float]]] = field(default_factory=list)  # List of (T, 2) trajectories
    travel_distances: List[float] = field(default_factory=list)
    mean_travel_distance: float = 0.0
    std_travel_distance: float = 0.0
    
    # Comparison with baseline
    baseline_distance: float = 0.0
    distance_change_percent: float = 0.0
    effect_matches_prediction: bool = False
    
    # Bias configuration used
    bias_groups: int = 0
    total_biased_tokens: int = 0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        # Ensure trajectories are lists, not numpy arrays
        if isinstance(d.get('trajectories'), np.ndarray):
            d['trajectories'] = d['trajectories'].tolist()
        return d


@dataclass  
class DAGNodeExport:
    """Exportable DAG node."""
    
    id: str
    name: str
    node_type: str
    layer: int
    value: Any
    is_interventionable: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DAGEdgeExport:
    """Exportable DAG edge."""
    
    parent_id: str
    child_id: str
    confidence: float
    mechanism: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DAGExport:
    """Complete DAG export structure."""
    
    scenario_id: str
    nodes: List[DAGNodeExport] = field(default_factory=list)
    edges: List[DAGEdgeExport] = field(default_factory=list)
    
    # Summary statistics
    n_nodes: int = 0
    n_edges: int = 0
    n_initial_state_nodes: int = 0
    n_event_nodes: int = 0
    n_outcome_nodes: int = 0
    
    # VLM extraction results
    maneuvers: List[Dict] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = {
            'scenario_id': self.scenario_id,
            'nodes': [n.to_dict() if hasattr(n, 'to_dict') else n for n in self.nodes],
            'edges': [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.edges],
            'summary': {
                'n_nodes': self.n_nodes,
                'n_edges': self.n_edges,
                'n_initial_state_nodes': self.n_initial_state_nodes,
                'n_event_nodes': self.n_event_nodes,
                'n_outcome_nodes': self.n_outcome_nodes
            },
            'vlm_extraction': {
                'maneuvers': self.maneuvers,
                'decisions': self.decisions
            }
        }
        return d


@dataclass
class PipelineOutput:
    """Complete pipeline output for a scenario."""
    
    # Identifiers
    scenario_id: str
    run_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_version: str = "1.0.0"
    
    # Input configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Stage results
    stage_results: Dict[str, Dict] = field(default_factory=dict)
    
    # DAG
    dag: Optional[DAGExport] = None
    
    # Baseline trajectory
    baseline_trajectory: List[List[float]] = field(default_factory=list)
    baseline_travel_distance: float = 0.0
    
    # All interventions enumerated
    possible_interventions: List[Dict] = field(default_factory=list)
    
    # Interventions that were evaluated
    evaluated_interventions: List[InterventionResult] = field(default_factory=list)
    
    # LLM logs
    llm_logs: List[LLMLogEntry] = field(default_factory=list)
    
    # Metrics summary
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Errors/warnings
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'scenario_id': self.scenario_id,
            'run_timestamp': self.run_timestamp,
            'pipeline_version': self.pipeline_version,
            'config': self.config,
            'stage_results': self.stage_results,
            'dag': self.dag.to_dict() if self.dag else None,
            'baseline': {
                'trajectory': self.baseline_trajectory,
                'travel_distance': self.baseline_travel_distance
            },
            'interventions': {
                'possible': self.possible_interventions,
                'evaluated': [i.to_dict() for i in self.evaluated_interventions]
            },
            'llm_logs': [log.to_dict() for log in self.llm_logs],
            'metrics_summary': self.metrics_summary,
            'warnings': self.warnings,
            'errors': self.errors
        }


class PipelineOutputManager:
    """
    Manager for collecting and exporting pipeline outputs.
    
    Usage:
        manager = PipelineOutputManager(scenario_id, output_dir)
        manager.set_config({...})
        manager.add_llm_log(...)
        manager.set_dag(dag)
        manager.add_intervention_result(...)
        manager.export_all()
    """
    
    def __init__(self, scenario_id: str, output_dir: Union[str, Path]):
        """
        Initialize output manager.
        
        Args:
            scenario_id: Scenario identifier
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output = PipelineOutput(scenario_id=scenario_id)
        
        # Track timing
        self._stage_start_times: Dict[str, datetime] = {}
    
    def set_config(self, config: Dict[str, Any]):
        """Set pipeline configuration."""
        self.output.config = config
    
    def start_stage(self, stage_name: str):
        """Mark stage start for timing."""
        self._stage_start_times[stage_name] = datetime.now()
        self.output.stage_results[stage_name] = {'status': 'running'}
    
    def complete_stage(self, stage_name: str, result: Dict[str, Any]):
        """Complete a stage with results."""
        end_time = datetime.now()
        start_time = self._stage_start_times.get(stage_name, end_time)
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        self.output.stage_results[stage_name] = {
            'status': 'success',
            'duration_ms': duration_ms,
            **result
        }
    
    def fail_stage(self, stage_name: str, error: str):
        """Mark a stage as failed."""
        self.output.stage_results[stage_name] = {
            'status': 'failed',
            'error': error
        }
        self.output.errors.append(f"Stage {stage_name}: {error}")
    
    def add_llm_log(
        self,
        log_type: str,
        model: str,
        prompt: str,
        response: str,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        **metadata
    ):
        """Add an LLM interaction log entry."""
        entry = LLMLogEntry(
            timestamp=datetime.now().isoformat(),
            log_type=log_type,
            model=model,
            prompt=prompt,
            response=response,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            metadata=metadata
        )
        self.output.llm_logs.append(entry)
    
    def set_dag(self, dag, maneuvers: List = None, decisions: List = None):
        """
        Set DAG from ScenarioDAG object.
        
        Args:
            dag: ScenarioDAG object
            maneuvers: Optional list of extracted maneuvers
            decisions: Optional list of extracted decisions
        """
        export = DAGExport(scenario_id=dag.scenario_id)
        
        # Export nodes
        for node_id, node in dag.nodes.items():
            # Check for interventionable attribute (may be named differently or not exist)
            is_interventionable = getattr(node, 'interventionable', 
                                         getattr(node, 'is_interventionable', 
                                                getattr(node, 'is_intervened', False)))
            export.nodes.append(DAGNodeExport(
                id=node_id,
                name=node.name,
                node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                layer=node.layer,
                value=node.value,
                is_interventionable=is_interventionable
            ))
        
        # Export edges
        for edge in dag.edges:
            export.edges.append(DAGEdgeExport(
                parent_id=edge.parent_id,
                child_id=edge.child_id,
                confidence=edge.confidence,
                mechanism=edge.mechanism
            ))
        
        # Summary
        export.n_nodes = len(export.nodes)
        export.n_edges = len(export.edges)
        export.n_initial_state_nodes = len([n for n in export.nodes if n.layer == 0])
        export.n_event_nodes = len([n for n in export.nodes if n.layer == 1])
        export.n_outcome_nodes = len([n for n in export.nodes if n.layer == 2])
        
        # VLM extraction
        if maneuvers:
            export.maneuvers = [
                m.to_dict() if hasattr(m, 'to_dict') else m.__dict__ if hasattr(m, '__dict__') else str(m)
                for m in maneuvers
            ]
        if decisions:
            export.decisions = [
                d.to_dict() if hasattr(d, 'to_dict') else d.__dict__ if hasattr(d, '__dict__') else str(d)
                for d in decisions
            ]
        
        self.output.dag = export
    
    def set_baseline(self, trajectory: np.ndarray, travel_distance: float):
        """Set baseline trajectory."""
        self.output.baseline_trajectory = trajectory.tolist() if isinstance(trajectory, np.ndarray) else trajectory
        self.output.baseline_travel_distance = float(travel_distance)
    
    def add_possible_intervention(self, intervention: Dict):
        """Add a possible intervention to the list."""
        self.output.possible_interventions.append(intervention)
    
    def add_intervention_result(
        self,
        intervention_id: str,
        intervention_name: str,
        description: str,
        target_node: str,
        original_value: Any,
        new_value: Any,
        predicted_effect: str,
        prediction_confidence: float,
        prediction_reasoning: str,
        trajectories: List[np.ndarray],
        bias_groups: int = 0,
        total_biased_tokens: int = 0
    ):
        """Add result for a single intervention."""
        # Convert trajectories
        traj_list = [t.tolist() if isinstance(t, np.ndarray) else t for t in trajectories]
        
        # Compute distances
        distances = [self._compute_distance(t) for t in trajectories]
        mean_dist = np.mean(distances) if distances else 0
        std_dist = np.std(distances) if distances else 0
        
        # Compare with baseline
        baseline_dist = self.output.baseline_travel_distance
        change_percent = ((mean_dist - baseline_dist) / baseline_dist * 100) if baseline_dist > 0 else 0
        
        # Check if effect matches prediction
        actual_effect = "decrease" if change_percent < -5 else ("increase" if change_percent > 5 else "none")
        matches = (actual_effect == predicted_effect) or (
            predicted_effect == "decrease" and change_percent < 0
        ) or (
            predicted_effect == "increase" and change_percent > 0
        )
        
        result = InterventionResult(
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            description=description,
            target_node=target_node,
            original_value=original_value,
            new_value=new_value,
            predicted_effect=predicted_effect,
            prediction_confidence=prediction_confidence,
            prediction_reasoning=prediction_reasoning,
            trajectories=traj_list,
            travel_distances=distances,
            mean_travel_distance=float(mean_dist),
            std_travel_distance=float(std_dist),
            baseline_distance=baseline_dist,
            distance_change_percent=float(change_percent),
            effect_matches_prediction=matches,
            bias_groups=bias_groups,
            total_biased_tokens=total_biased_tokens
        )
        
        self.output.evaluated_interventions.append(result)
    
    def set_metrics_summary(self, summary: Dict[str, Any]):
        """Set metrics summary."""
        self.output.metrics_summary = summary
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.output.warnings.append(warning)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.output.errors.append(error)
    
    def export_all(self) -> Dict[str, Path]:
        """
        Export all outputs to files.
        
        Returns:
            Dictionary mapping output types to file paths
        """
        paths = {}
        
        # Main JSON output
        main_output_path = self.output_dir / "pipeline_output.json"
        with open(main_output_path, 'w') as f:
            json.dump(self.output.to_dict(), f, cls=NumpyJSONEncoder, indent=2)
        paths['main'] = main_output_path
        logger.info(f"Exported main output to {main_output_path}")
        
        # Separate DAG export
        if self.output.dag:
            dag_path = self.output_dir / "dag_structure.json"
            with open(dag_path, 'w') as f:
                json.dump(self.output.dag.to_dict(), f, cls=NumpyJSONEncoder, indent=2)
            paths['dag'] = dag_path
        
        # LLM logs export
        if self.output.llm_logs:
            logs_path = self.output_dir / "llm_logs.json"
            with open(logs_path, 'w') as f:
                json.dump([log.to_dict() for log in self.output.llm_logs], f, cls=NumpyJSONEncoder, indent=2)
            paths['llm_logs'] = logs_path
        
        # Intervention results
        if self.output.evaluated_interventions:
            interventions_path = self.output_dir / "intervention_results.json"
            with open(interventions_path, 'w') as f:
                json.dump({
                    'baseline_distance': self.output.baseline_travel_distance,
                    'interventions': [i.to_dict() for i in self.output.evaluated_interventions]
                }, f, cls=NumpyJSONEncoder, indent=2)
            paths['interventions'] = interventions_path
        
        # Human-readable summary
        summary_path = self.output_dir / "summary.txt"
        summary_text = self._generate_text_summary()
        summary_path.write_text(summary_text)
        paths['summary'] = summary_path
        
        return paths
    
    def _compute_distance(self, trajectory) -> float:
        """Compute travel distance for a trajectory."""
        trajectory = np.asarray(trajectory)
        if len(trajectory) < 2:
            return 0.0
        diffs = np.diff(trajectory, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
    
    def _generate_text_summary(self) -> str:
        """Generate human-readable text summary."""
        lines = [
            "=" * 70,
            f"COUNTERBMT PIPELINE SUMMARY",
            "=" * 70,
            f"Scenario ID: {self.output.scenario_id}",
            f"Run Time: {self.output.run_timestamp}",
            "",
            "STAGE RESULTS:",
            "-" * 40,
        ]
        
        for stage, result in self.output.stage_results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration_ms', 'N/A')
            lines.append(f"  {stage}: {status} ({duration}ms)")
        
        if self.output.dag:
            lines.extend([
                "",
                "DAG SUMMARY:",
                "-" * 40,
                f"  Total Nodes: {self.output.dag.n_nodes}",
                f"  Total Edges: {self.output.dag.n_edges}",
                f"  Initial States: {self.output.dag.n_initial_state_nodes}",
                f"  Events: {self.output.dag.n_event_nodes}",
                f"  Outcomes: {self.output.dag.n_outcome_nodes}",
            ])
        
        lines.extend([
            "",
            "BASELINE TRAJECTORY:",
            "-" * 40,
            f"  Travel Distance: {self.output.baseline_travel_distance:.2f}m",
        ])
        
        if self.output.evaluated_interventions:
            lines.extend([
                "",
                "INTERVENTION RESULTS:",
                "-" * 40,
            ])
            for result in self.output.evaluated_interventions:
                effect_symbol = "↓" if result.distance_change_percent < -5 else (
                    "↑" if result.distance_change_percent > 5 else "→"
                )
                match_symbol = "✓" if result.effect_matches_prediction else "✗"
                lines.append(
                    f"  {result.intervention_name[:40]}: "
                    f"{result.mean_travel_distance:.1f}m ({result.distance_change_percent:+.1f}%) "
                    f"{effect_symbol} [predicted: {result.predicted_effect}] {match_symbol}"
                )
        
        if self.output.llm_logs:
            lines.extend([
                "",
                "LLM INTERACTIONS:",
                "-" * 40,
                f"  Total calls: {len(self.output.llm_logs)}",
            ])
            type_counts = {}
            for log in self.output.llm_logs:
                type_counts[log.log_type] = type_counts.get(log.log_type, 0) + 1
            for log_type, count in type_counts.items():
                lines.append(f"    {log_type}: {count}")
        
        if self.output.warnings:
            lines.extend([
                "",
                "WARNINGS:",
                "-" * 40,
            ])
            for w in self.output.warnings:
                lines.append(f"  ⚠ {w}")
        
        if self.output.errors:
            lines.extend([
                "",
                "ERRORS:",
                "-" * 40,
            ])
            for e in self.output.errors:
                lines.append(f"  ✗ {e}")
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)


def export_scenario_package(
    scenario_id: str,
    output_dir: Path,
    dag,
    baseline_trajectory: np.ndarray,
    counterfactual_results: Dict[str, Any],
    llm_logs: List[Dict] = None,
    config: Dict = None
) -> Path:
    """
    Convenience function to export a complete scenario package.
    
    Args:
        scenario_id: Scenario identifier
        output_dir: Output directory
        dag: ScenarioDAG object
        baseline_trajectory: Baseline trajectory
        counterfactual_results: Results from counterfactual generation
        llm_logs: Optional LLM interaction logs
        config: Optional pipeline configuration
        
    Returns:
        Path to the output directory
    """
    manager = PipelineOutputManager(scenario_id, output_dir)
    
    if config:
        manager.set_config(config)
    
    # Set DAG
    manager.set_dag(dag)
    
    # Set baseline
    baseline_dist = counterfactual_results.get('baseline_travel_distance', 0)
    manager.set_baseline(baseline_trajectory, baseline_dist)
    
    # Add intervention results
    for name, data in counterfactual_results.get('counterfactuals', {}).items():
        if isinstance(data, dict):
            manager.add_intervention_result(
                intervention_id=f"intervention_{name}",
                intervention_name=name,
                description=data.get('description', ''),
                target_node=data.get('target_node', ''),
                original_value=data.get('original_value'),
                new_value=data.get('new_value'),
                predicted_effect=data.get('predicted_effect', 'unknown'),
                prediction_confidence=data.get('prediction_confidence', 0),
                prediction_reasoning=data.get('prediction_reasoning', ''),
                trajectories=data.get('trajectories', []),
                bias_groups=data.get('bias_groups', 0),
                total_biased_tokens=data.get('total_biased_tokens', 0)
            )
    
    # Add LLM logs
    if llm_logs:
        for log in llm_logs:
            manager.add_llm_log(**log)
    
    manager.export_all()
    
    return output_dir

