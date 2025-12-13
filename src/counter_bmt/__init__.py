"""
CounterBMT: Counterfactual Trajectory Generation for Safety-Critical Scenarios

This package provides tools for:
1. Loading and visualizing Waymo scenarios via ScenarioNet
2. Extracting safety-critical features using VLMs
3. Constructing grounded causal DAGs from extracted features
4. Generating counterfactual scenarios via do-calculus
5. Guiding BMT trajectory generation with counterfactual constraints

Usage:
    from counter_bmt import (
        # Visualization
        ScenarioNetDatabase,
        ScenarioNetVisualizer,
        prepare_for_vlm,
        
        # VLM Feature Extraction
        VLMSafetyCriticalExtractor,
        GPT4oClient,
        MockGPT4oClient,
        TimestampedImage,
        ScenarioFeatures,
        
        # DAG Construction
        GroundedDAGConstructor,
        GPT4oDAGClient,
        MockDAGClient,
        ScenarioDAG,
        
        # DAG Visualization
        visualize_dag,
        export_dag_to_dot,
    )
"""

# Visualization module
from .scenarionet_visualizer import (
    ScenarioNetDatabase,
    ScenarioNetVisualizer,
    prepare_for_vlm,
    extract_trajectory_from_scenario,
)

# VLM Feature Extraction module
from .vlm_extractor import (
    # Enums
    ManeuverType,
    DecisionType,
    Aggressiveness,
    
    # Data classes
    TimestampedImage,
    ManeuverSegment,
    CriticalDecisionPoint,
    ScenarioFeatures,
    
    # Clients
    GPT4oClient,
    MockGPT4oClient,
    
    # Extractor
    VLMSafetyCriticalExtractor,
)

# DAG Construction module
from .dag_constructor import (
    # Enums
    NodeType,
    EdgeType,
    
    # Data classes
    DAGNode,
    DAGEdge,
    Intervention,
    CounterfactualResult,
    
    # DAG class
    ScenarioDAG,
    
    # Clients
    DAGClient,
    GPT4oDAGClient,
    MockDAGClient,
    
    # Constructor (both names for compatibility)
    GroundedDAGConstructor,
    DAGConstructor,
)

# DAG Visualization module
from .dag_visualization import (
    visualize_dag,
    export_dag_to_dot,
    print_dag_summary,
)

# BMT Integration module
from .bmt_generator import (
    MotionTokenSpace,
    InterventionCompiler,
    BiasedTokenSampler,
    CounterBMTGenerator,
    TokenBias,
)

# Trajectory Metrics module
from .trajectory_metrics import (
    TrajectoryMetrics,
    TrajectoryMetricsCalculator,
    CounterfactualComparison,
    compute_intervention_effectiveness,
    generate_metrics_summary,
)

# Trajectory Visualization module
from .trajectory_visualization import (
    visualize_trajectory_comparison,
    visualize_single_comparison,
    visualize_intervention_summary,
    create_scenario_report,
    VisualizationConfig,
)

# Pipeline Output module
from .pipeline_output import (
    PipelineOutput,
    PipelineOutputManager,
    LLMLogEntry,
    InterventionResult,
    DAGExport,
    export_scenario_package,
)

# Scenario Export module (for MetaDrive/ScenarioNet replay)
from .scenario_export import (
    export_counterfactual_scenario,
    export_all_counterfactuals,
    export_trajectory_only,
    create_replay_script,
)

__all__ = [
    # Visualization
    "ScenarioNetDatabase",
    "ScenarioNetVisualizer", 
    "prepare_for_vlm",
    "extract_trajectory_from_scenario",
    
    # VLM Extraction - Enums
    "ManeuverType",
    "DecisionType",
    "Aggressiveness",
    
    # VLM Extraction - Data classes
    "TimestampedImage",
    "ManeuverSegment",
    "CriticalDecisionPoint",
    "ScenarioFeatures",
    
    # VLM Extraction - Clients
    "GPT4oClient",
    "MockGPT4oClient",
    
    # VLM Extraction - Extractor
    "VLMSafetyCriticalExtractor",
    
    # DAG - Enums
    "NodeType",
    "EdgeType",
    
    # DAG - Data classes
    "DAGNode",
    "DAGEdge",
    "Intervention",
    "CounterfactualResult",
    
    # DAG - Core class
    "ScenarioDAG",
    
    # DAG - Clients
    "DAGClient",
    "GPT4oDAGClient",
    "MockDAGClient",
    
    # DAG - Constructor
    "GroundedDAGConstructor",
    "DAGConstructor",
    
    # DAG Visualization
    "visualize_dag",
    "export_dag_to_dot",
    "print_dag_summary",
    
    # BMT Integration
    "MotionTokenSpace",
    "InterventionCompiler",
    "BiasedTokenSampler",
    "CounterBMTGenerator",
    "TokenBias",
    
    # Trajectory Metrics
    "TrajectoryMetrics",
    "TrajectoryMetricsCalculator",
    "CounterfactualComparison",
    "compute_intervention_effectiveness",
    "generate_metrics_summary",
    
    # Trajectory Visualization
    "visualize_trajectory_comparison",
    "visualize_single_comparison",
    "visualize_intervention_summary",
    "create_scenario_report",
    "VisualizationConfig",
    
    # Pipeline Output
    "PipelineOutput",
    "PipelineOutputManager",
    "LLMLogEntry",
    "InterventionResult",
    "DAGExport",
    "export_scenario_package",
    
    # Scenario Export (MetaDrive/ScenarioNet replay)
    "export_counterfactual_scenario",
    "export_all_counterfactuals",
    "export_trajectory_only",
    "create_replay_script",
]

__version__ = "0.1.0"