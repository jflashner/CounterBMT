"""
Test Grounded DAG Constructor with Waymo Data and GPT-4o API

This script tests the full pipeline:
1. Load Waymo scenario via ScenarioNet
2. Extract features with GPT-4o VLM
3. Construct grounded causal DAG
4. Enumerate interventions and evaluate counterfactuals
5. Visualize and save results

Usage:
    python test_grounded_dag.py --data-dir ./exp_converted
    python test_grounded_dag.py --data-dir ./exp_converted --scenario-index 5
    python test_grounded_dag.py --data-dir ./exp_converted --num-scenarios 10

Author: CounterBMT Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from package
from counter_bmt.dag_constructor import (
    GroundedDAGConstructor,
    GPT4oDAGClient,
    ScenarioDAG,
)
from counter_bmt.scenarionet_visualizer import (
    ScenarioNetDatabase,
    prepare_for_vlm,
)
from counter_bmt.vlm_extractor import (
    VLMSafetyCriticalExtractor,
    GPT4oClient,
    TimestampedImage,
)
from counter_bmt.dag_visualization import (
    visualize_dag,
    export_dag_to_dot,
    print_dag_summary,
)


def extract_other_agents_from_scenario(data_dir: str, scenario_index: int, max_agents: int = 5) -> Tuple[List[Dict], str]:
    """
    Extract other agent states from scenario using ScenarioNet database.
    
    Returns:
        Tuple of (list of agent dicts, scenario_id)
    """
    from counter_bmt.scenarionet_visualizer import extract_trajectory_from_scenario
    
    _, scenario_id, other_agents = extract_trajectory_from_scenario(
        data_dir=str(data_dir),
        scenario_index=scenario_index
    )
    
    # Limit to max_agents
    if len(other_agents) > max_agents:
        other_agents = other_agents[:max_agents]
    
    logger.info(f"Extracted {len(other_agents)} valid other agents from scenario {scenario_id}")
    for agent in other_agents:
        pos = agent['position']
        logger.info(f"  {agent['agent_id']}: {agent['type']} at ({pos[0]:.1f}, {pos[1]:.1f}), speed={agent['speed']:.1f}")
    
    return other_agents, scenario_id


def process_scenario(
    scenario_index: int,
    data_dir: Path,
    output_dir: Path,
    vlm_client,
    dag_client,
    num_frames: int = 8
) -> dict:
    """
    Process a single Waymo scenario through the full pipeline.
    
    Returns:
        Dictionary with results or None on failure
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing scenario index {scenario_index}")
    logger.info(f"{'='*60}")
    
    # Generate screenshots and get trajectory
    try:
        saved_images, trajectory, scenario_id = prepare_for_vlm(
            data_dir=str(data_dir),
            scenario_index=scenario_index,
            output_dir=str(output_dir / "screenshots"),
            num_frames=num_frames
        )
    except Exception as e:
        logger.error(f"Failed to load scenario: {e}")
        return None
    
    logger.info(f"Scenario ID: {scenario_id}")
    logger.info(f"Generated {len(saved_images)} frames")
    logger.info(f"Trajectory: {len(trajectory)} timesteps")
    
    # Log trajectory details
    logger.info("\nTrajectory data (from scenario file):")
    logger.info(f"  Shape: {trajectory.shape}")
    if len(trajectory) > 0:
        logger.info(f"  t=0: pos=({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}), "
                   f"heading={trajectory[0][2]:.2f} rad, speed={trajectory[0][3]:.1f} m/s")
        if len(trajectory) > 1:
            logger.info(f"  t=end: pos=({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}), "
                       f"heading={trajectory[-1][2]:.2f} rad, speed={trajectory[-1][3]:.1f} m/s")
    
    # Extract features with VLM
    logger.info("\n" + "=" * 60)
    logger.info("VLM Feature Extraction (GPT-4o Vision)")
    logger.info("=" * 60)
    logger.info("Images being sent to VLM:")
    for path, timestamp in saved_images:
        logger.info(f"  {Path(path).name} @ t={timestamp:.2f}s")
    logger.info("-" * 60)
    logger.info("NOTE: VLM only sees the IMAGES, not the trajectory data.")
    logger.info("      It infers maneuvers/decisions from visual analysis only.")
    logger.info("-" * 60)
    
    images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
    extractor = VLMSafetyCriticalExtractor(vlm_client)
    features = extractor.extract(images, scenario_id, trajectory)
    
    # Get maneuvers/decisions - VLM extractor uses these attribute names:
    # - maneuver_sequence (list of ManeuverSegment)
    # - critical_decisions (list of CriticalDecisionPoint)
    # ManeuverSegment has: maneuver_type (enum), start_timestamp, end_timestamp, aggressiveness (enum)
    # CriticalDecisionPoint has: decision_type (enum), ground_truth_choice, timestamp, alternatives
    
    maneuvers = []
    decisions = []
    
    if hasattr(features, 'maneuver_sequence') and features.maneuver_sequence:
        maneuvers = features.maneuver_sequence
    
    if hasattr(features, 'critical_decisions') and features.critical_decisions:
        decisions = features.critical_decisions
    
    logger.info(f"Extracted: {len(maneuvers)} maneuvers, {len(decisions)} decisions")
    
    # Get other agents from scenario file
    other_agents, _ = extract_other_agents_from_scenario(str(data_dir), scenario_index)
    
    # Construct grounded DAG
    logger.info("\n" + "=" * 60)
    logger.info("DAG Edge Inference (GPT-4o Text)")
    logger.info("=" * 60)
    logger.info("The DAG edge inference LLM receives:")
    logger.info("  - Ego initial state: position, speed, heading (from trajectory)")
    logger.info(f"  - Other agents: {len(other_agents)} (from scenario file)")
    logger.info(f"  - Maneuvers: {len(maneuvers)} (from VLM)")
    logger.info(f"  - Decisions: {len(decisions)} (from VLM)")
    logger.info("NOTE: This LLM sees the ACTUAL trajectory data, not images.")
    logger.info("-" * 60)
    
    constructor = GroundedDAGConstructor(dag_client)
    dag = constructor.construct(features, trajectory, other_agents, scenario_id)
    
    # Print summary
    print_dag_summary(dag)
    
    # Enumerate interventions
    interventions = dag.enumerate_interventions()
    logger.info(f"\nPossible interventions: {len(interventions)}")
    for intv in interventions:
        logger.info(f"  - {intv.description}")
    
    # Evaluate counterfactuals
    logger.info("\nEvaluating counterfactuals...")
    counterfactuals = constructor.evaluate_counterfactuals(dag)
    
    logger.info(f"\nCounterfactual Results ({len(counterfactuals)}):")
    for cf in counterfactuals:
        logger.info(f"  do({cf.intervention.variable_id}={cf.intervention.value})")
        logger.info(f"    Effect: {cf.effect_direction}, Confidence: {cf.confidence:.2f}")
        logger.info(f"    Reasoning: {cf.reasoning[:80]}...")
    
    # Visualize
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = scenario_output_dir / "dag.png"
    visualize_dag(dag, viz_path, title=f"Grounded DAG: {scenario_id}")
    
    dot_path = scenario_output_dir / "dag.dot"
    export_dag_to_dot(dag, dot_path)
    
    # Save results
    def safe_get_attr(obj, attr, default='unknown'):
        """Safely get attribute from object or dict, handling enums."""
        if isinstance(obj, dict):
            val = obj.get(attr, default)
        else:
            val = getattr(obj, attr, default)
        # Handle enum values
        if hasattr(val, 'value'):
            return val.value
        return val
    
    result = {
        "scenario_id": scenario_id,
        "scenario_index": scenario_index,
        "timestamp": datetime.now().isoformat(),
        "extraction": {
            "n_frames": len(saved_images),
            "n_trajectory_steps": len(trajectory),
            "n_maneuvers": len(maneuvers),
            "n_decisions": len(decisions),
            "n_other_agents": len(other_agents),
            "maneuvers": [
                {"type": safe_get_attr(m, 'maneuver_type'),
                 "start_timestamp": safe_get_attr(m, 'start_timestamp', 0),
                 "end_timestamp": safe_get_attr(m, 'end_timestamp', 0),
                 "aggressiveness": safe_get_attr(m, 'aggressiveness', 'normal')}
                for m in maneuvers
            ],
            "decisions": [
                {"type": safe_get_attr(d, 'decision_type'),
                 "choice": safe_get_attr(d, 'ground_truth_choice'),
                 "timestamp": safe_get_attr(d, 'timestamp', 0),
                 "alternatives": safe_get_attr(d, 'alternatives', [])}
                for d in decisions
            ]
        },
        "dag": dag.to_dict(),
        "interventions": [intv.to_dict() for intv in interventions],
        "counterfactuals": [cf.to_dict() for cf in counterfactuals],
        "outputs": {
            "visualization": str(viz_path),
            "dot_file": str(dot_path)
        }
    }
    
    result_path = scenario_output_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"\nSaved results to {result_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Grounded DAG Constructor with Waymo data and GPT-4o"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to ScenarioNet converted Waymo data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/grounded_dag",
        help="Output directory for results"
    )
    parser.add_argument(
        "--scenario-index", type=int, default=0,
        help="Index of scenario to process (default: 0)"
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=1,
        help="Number of scenarios to process (default: 1)"
    )
    parser.add_argument(
        "--num-frames", type=int, default=8,
        help="Number of frames to extract per scenario (default: 8)"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Initialize clients
    logger.info("Initializing GPT-4o clients...")
    try:
        vlm_client = GPT4oClient()
        dag_client = GPT4oDAGClient()
    except ValueError as e:
        logger.error(f"Failed to initialize API clients: {e}")
        logger.error("Make sure OPENAI_API_KEY environment variable is set")
        return 1
    
    # Process scenarios
    results = []
    for i in range(args.num_scenarios):
        scenario_idx = args.scenario_index + i
        try:
            result = process_scenario(
                scenario_index=scenario_idx,
                data_dir=data_dir,
                output_dir=output_dir,
                vlm_client=vlm_client,
                dag_client=dag_client,
                num_frames=args.num_frames
            )
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing scenario {scenario_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(data_dir),
            "n_scenarios_processed": len(results),
            "scenarios": [r["scenario_id"] for r in results]
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: Processed {len(results)} scenarios")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())