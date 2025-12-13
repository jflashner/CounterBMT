"""
CounterBMT Full Pipeline

End-to-end pipeline that:
1. Loads Waymo scenario via ScenarioNet
2. Extracts features with GPT-4o VLM
3. Constructs grounded causal DAG
4. Enumerates interventions and compiles to token biases
5. Generates counterfactual trajectories with BMT
6. Compares and visualizes results

Usage:
    # Full pipeline with BMT generation
    python run_full_pipeline.py --data-dir ./exp_converted --bmt-checkpoint ./models/bmt.ckpt
    
    # Specific scenario
    python run_full_pipeline.py --data-dir ./exp_converted --scenario-index 5 --bmt-checkpoint ./models/bmt.ckpt
    
    # Skip BMT (DAG only mode for testing)
    python run_full_pipeline.py --data-dir ./exp_converted --skip-bmt
    
    # Use mock clients (no API calls)
    python run_full_pipeline.py --data-dir ./exp_converted --mock

Author: CounterBMT Project
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Import Pipeline Components
# =============================================================================

def import_dag_components():
    """Import DAG construction components."""
    from counter_bmt.dag_constructor import (
        GroundedDAGConstructor,
        GPT4oDAGClient,
        MockDAGClient,
        ScenarioDAG,
    )
    from counter_bmt.scenarionet_visualizer import (
        ScenarioNetDatabase,
        prepare_for_vlm,
        extract_trajectory_from_scenario,
        extract_all_trajectories,
    )
    from counter_bmt.vlm_extractor import (
        VLMSafetyCriticalExtractor,
        GPT4oClient,
        MockGPT4oClient,
        TimestampedImage,
    )
    from counter_bmt.dag_visualization import (
        visualize_dag,
        export_dag_to_dot,
        print_dag_summary,
    )
    return {
        'GroundedDAGConstructor': GroundedDAGConstructor,
        'GPT4oDAGClient': GPT4oDAGClient,
        'MockDAGClient': MockDAGClient,
        'ScenarioDAG': ScenarioDAG,
        'prepare_for_vlm': prepare_for_vlm,
        'extract_trajectory_from_scenario': extract_trajectory_from_scenario,
        'extract_all_trajectories': extract_all_trajectories,
        'VLMSafetyCriticalExtractor': VLMSafetyCriticalExtractor,
        'GPT4oClient': GPT4oClient,
        'MockGPT4oClient': MockGPT4oClient,
        'TimestampedImage': TimestampedImage,
        'visualize_dag': visualize_dag,
        'export_dag_to_dot': export_dag_to_dot,
        'print_dag_summary': print_dag_summary,
    }


def import_bmt_components():
    """Import BMT generation components."""
    from counter_bmt.bmt_generator import (
        CounterBMTGenerator,
        MotionTokenSpace,
        InterventionCompiler,
        BiasedTokenSampler,
        TokenBias,
        compare_trajectories,
        plot_trajectory_comparison,
    )
    return {
        'CounterBMTGenerator': CounterBMTGenerator,
        'MotionTokenSpace': MotionTokenSpace,
        'InterventionCompiler': InterventionCompiler,
        'BiasedTokenSampler': BiasedTokenSampler,
        'TokenBias': TokenBias,
        'compare_trajectories': compare_trajectories,
        'plot_trajectory_comparison': plot_trajectory_comparison,
    }


def import_analysis_components():
    """Import trajectory metrics and visualization components."""
    from counter_bmt.trajectory_metrics import (
        TrajectoryMetricsCalculator,
        compute_intervention_effectiveness,
        generate_metrics_summary,
    )
    from counter_bmt.trajectory_visualization import (
        visualize_trajectory_comparison,
        visualize_intervention_summary,
        create_scenario_report,
        VisualizationConfig,
    )
    from counter_bmt.pipeline_output import (
        PipelineOutputManager,
        export_scenario_package,
    )
    return {
        'TrajectoryMetricsCalculator': TrajectoryMetricsCalculator,
        'compute_intervention_effectiveness': compute_intervention_effectiveness,
        'generate_metrics_summary': generate_metrics_summary,
        'visualize_trajectory_comparison': visualize_trajectory_comparison,
        'visualize_intervention_summary': visualize_intervention_summary,
        'create_scenario_report': create_scenario_report,
        'VisualizationConfig': VisualizationConfig,
        'PipelineOutputManager': PipelineOutputManager,
        'export_scenario_package': export_scenario_package,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def safe_get_attr(obj, attr, default='unknown'):
    """Safely get attribute from object or dict, handling enums."""
    if isinstance(obj, dict):
        val = obj.get(attr, default)
    else:
        val = getattr(obj, attr, default)
    if hasattr(val, 'value'):
        return val.value
    return val


def extract_other_agents(data_dir: str, scenario_index: int, max_agents: int = 5) -> Tuple[List[Dict], str]:
    """Extract other agent states from scenario."""
    dag_comps = import_dag_components()
    extract_trajectory_from_scenario = dag_comps['extract_trajectory_from_scenario']
    
    _, scenario_id, other_agents = extract_trajectory_from_scenario(
        data_dir=str(data_dir),
        scenario_index=scenario_index
    )
    
    if len(other_agents) > max_agents:
        other_agents = other_agents[:max_agents]
    
    logger.info(f"Extracted {len(other_agents)} other agents")
    for agent in other_agents[:3]:  # Log first 3
        pos = agent['position']
        logger.info(f"  {agent['agent_id']}: {agent['type']} at ({pos[0]:.1f}, {pos[1]:.1f})")
    
    return other_agents, scenario_id


def load_scenario_for_bmt_input(data_dir: str, scenario_index: int, config=None, tokenizer=None) -> Dict:
    """
    Load and preprocess scenario for BMT input format.
    """
    from metadrive.scenario.utils import read_dataset_summary
    import pickle
    
    data_path = Path(data_dir)
    summary_dict, summary_list, mapping = read_dataset_summary(str(data_path))
    
    if scenario_index >= len(summary_list):
        raise ValueError(f"Scenario index {scenario_index} out of range (max: {len(summary_list)-1})")
    
    scenario_file = summary_list[scenario_index]
    folder = mapping.get(scenario_file, "")
    file_path = data_path / folder / scenario_file  # Use data_path as base
    
    with open(file_path, 'rb') as f:
        scenario_desc = pickle.load(f)
    
    scenario_id = scenario_desc.get('id', f'scenario_{scenario_index}')
    
    # If config provided, preprocess for BMT
    if config is not None:
        from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
        preprocessed = preprocess_scenario_description_for_motionlm(
            scenario=scenario_desc,
            config=config,
            in_evaluation=True,
            keep_all_data=True,
            tokenizer=tokenizer
        )
        preprocessed['metadata/scenario_id'] = scenario_id
        return {
            'raw_data': scenario_desc,
            'preprocessed': preprocessed,
            'scenario_id': scenario_id,
            'file_path': str(file_path)
        }
    
    return {
        'raw_data': scenario_desc,
        'scenario_id': scenario_id,
        'file_path': str(file_path)
    }


# =============================================================================
# Pipeline Stages
# =============================================================================

def stage_1_load_and_visualize(
    data_dir: Path,
    scenario_index: int,
    output_dir: Path,
    num_frames: int = 8
) -> Dict:
    """
    Stage 1: Load scenario and generate screenshots for VLM.
    
    Returns:
        Dict with scenario_id, trajectory, saved_images, other_agents
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: Load Scenario & Generate Frames")
    logger.info("=" * 60)
    
    dag_comps = import_dag_components()
    prepare_for_vlm = dag_comps['prepare_for_vlm']
    
    # Generate screenshots
    saved_images, trajectory, scenario_id = prepare_for_vlm(
        data_dir=str(data_dir),
        scenario_index=scenario_index,
        output_dir=str(output_dir / "screenshots"),
        num_frames=num_frames
    )
    
    logger.info(f"Scenario ID: {scenario_id}")
    logger.info(f"Generated {len(saved_images)} frames")
    logger.info(f"Trajectory shape: {trajectory.shape}")
    
    if len(trajectory) > 0:
        logger.info(f"  Start: pos=({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}), "
                   f"speed={trajectory[0][3]:.1f} m/s")
        logger.info(f"  End:   pos=({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}), "
                   f"speed={trajectory[-1][3]:.1f} m/s")
    
    # Extract other agents
    other_agents, _ = extract_other_agents(str(data_dir), scenario_index)
    
    return {
        'scenario_id': scenario_id,
        'scenario_index': scenario_index,
        'trajectory': trajectory,
        'saved_images': saved_images,
        'other_agents': other_agents,
    }


def stage_2_vlm_extraction(
    stage1_result: Dict,
    vlm_client,
) -> Dict:
    """
    Stage 2: Extract maneuvers and decisions using VLM.
    
    Returns:
        Dict with features, maneuvers, decisions
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: VLM Feature Extraction")
    logger.info("=" * 60)
    
    dag_comps = import_dag_components()
    VLMSafetyCriticalExtractor = dag_comps['VLMSafetyCriticalExtractor']
    TimestampedImage = dag_comps['TimestampedImage']
    
    saved_images = stage1_result['saved_images']
    trajectory = stage1_result['trajectory']
    scenario_id = stage1_result['scenario_id']
    
    logger.info(f"Sending {len(saved_images)} images to VLM...")
    for path, timestamp in saved_images:
        logger.info(f"  {Path(path).name} @ t={timestamp:.2f}s")
    
    # Create timestamped images
    images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
    
    # Extract features
    extractor = VLMSafetyCriticalExtractor(vlm_client)
    features = extractor.extract(images, scenario_id, trajectory)
    
    # Get maneuvers and decisions
    maneuvers = []
    decisions = []
    
    if hasattr(features, 'maneuver_sequence') and features.maneuver_sequence:
        maneuvers = features.maneuver_sequence
    if hasattr(features, 'critical_decisions') and features.critical_decisions:
        decisions = features.critical_decisions
    
    logger.info(f"Extracted: {len(maneuvers)} maneuvers, {len(decisions)} decisions")
    
    for m in maneuvers:
        m_type = safe_get_attr(m, 'maneuver_type')
        m_start = safe_get_attr(m, 'start_timestamp', 0)
        logger.info(f"  Maneuver: {m_type} @ t={m_start:.2f}s")
    
    for d in decisions:
        d_type = safe_get_attr(d, 'decision_type')
        d_choice = safe_get_attr(d, 'ground_truth_choice')
        d_time = safe_get_attr(d, 'timestamp', 0)
        logger.info(f"  Decision: {d_type} -> {d_choice} @ t={d_time:.2f}s")
    
    return {
        'features': features,
        'maneuvers': maneuvers,
        'decisions': decisions,
    }


def stage_3_dag_construction(
    stage1_result: Dict,
    stage2_result: Dict,
    dag_client,
    output_dir: Path,
) -> Dict:
    """
    Stage 3: Construct grounded causal DAG and enumerate interventions.
    
    Returns:
        Dict with dag, interventions, counterfactuals
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3: DAG Construction & Intervention Enumeration")
    logger.info("=" * 60)
    
    dag_comps = import_dag_components()
    GroundedDAGConstructor = dag_comps['GroundedDAGConstructor']
    visualize_dag = dag_comps['visualize_dag']
    export_dag_to_dot = dag_comps['export_dag_to_dot']
    print_dag_summary = dag_comps['print_dag_summary']
    
    trajectory = stage1_result['trajectory']
    other_agents = stage1_result['other_agents']
    scenario_id = stage1_result['scenario_id']
    features = stage2_result['features']
    
    # Construct DAG
    constructor = GroundedDAGConstructor(dag_client)
    dag = constructor.construct(features, trajectory, other_agents, scenario_id)
    
    # Print summary
    print_dag_summary(dag)
    
    # Enumerate interventions
    interventions = dag.enumerate_interventions()
    logger.info(f"\nPossible interventions: {len(interventions)}")
    for intv in interventions:
        logger.info(f"  - {intv.description}")
    
    # Evaluate counterfactuals (LLM predictions)
    logger.info("\nEvaluating counterfactuals with LLM...")
    counterfactuals = constructor.evaluate_counterfactuals(dag)
    
    for cf in counterfactuals:
        logger.info(f"  do({cf.intervention.variable_id}={cf.intervention.value})")
        logger.info(f"    -> {cf.effect_direction} (confidence: {cf.confidence:.2f})")
    
    # Visualize DAG
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = scenario_output_dir / "dag.png"
    visualize_dag(dag, viz_path, title=f"Grounded DAG: {scenario_id}")
    
    dot_path = scenario_output_dir / "dag.dot"
    export_dag_to_dot(dag, dot_path)
    
    logger.info(f"Saved DAG visualization to {viz_path}")
    
    return {
        'dag': dag,
        'interventions': interventions,
        'counterfactuals': counterfactuals,
        'viz_path': str(viz_path),
        'dot_path': str(dot_path),
    }


def stage_4_compile_interventions(
    stage3_result: Dict,
    max_interventions: int = 5,
) -> Dict:
    """
    Stage 4: Compile interventions to token biases for BMT.
    
    Returns:
        Dict with compiled_interventions
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: Compile Interventions to Token Biases")
    logger.info("=" * 60)
    
    bmt_comps = import_bmt_components()
    InterventionCompiler = bmt_comps['InterventionCompiler']
    MotionTokenSpace = bmt_comps['MotionTokenSpace']
    
    interventions = stage3_result['interventions']
    counterfactuals = stage3_result['counterfactuals']
    
    # Limit interventions
    if len(interventions) > max_interventions:
        logger.info(f"Limiting to {max_interventions} interventions (from {len(interventions)})")
        interventions = interventions[:max_interventions]
    
    # Create compiler
    token_space = MotionTokenSpace()
    compiler = InterventionCompiler(token_space)
    
    compiled = []
    for i, intv in enumerate(interventions):
        # Convert Intervention object to dict
        int_dict = {
            'variable': intv.variable_id,
            'value': intv.value,
            'original_value': intv.original_value,
            'description': intv.description
        }
        
        # Compile to token biases
        token_biases = compiler.compile_from_dag_intervention(int_dict)
        
        # Find matching counterfactual prediction
        effect_prediction = {}
        for cf in counterfactuals:
            if cf.intervention.variable_id == intv.variable_id and cf.intervention.value == intv.value:
                effect_prediction = {
                    'effect_direction': cf.effect_direction,
                    'confidence': cf.confidence,
                    'reasoning': cf.reasoning,
                }
                break
        
        compiled.append({
            'intervention': int_dict,
            'token_biases': [b.to_dict() for b in token_biases],
            'effect_prediction': effect_prediction,
        })
        
        n_tokens = sum(len(b.token_ids) for b in token_biases)
        logger.info(f"  [{i+1}] {intv.description}")
        logger.info(f"      -> {len(token_biases)} bias groups, {n_tokens} total tokens")
    
    return {
        'compiled_interventions': compiled,
        'token_space_info': {
            'n_tokens': token_space.n_tokens,
            'n_acc_bins': token_space.n_acc_bins,
            'n_yaw_bins': token_space.n_yaw_bins,
        }
    }


def stage_5_bmt_generation(
    stage1_result: Dict,
    stage4_result: Dict,
    bmt_checkpoint: str,
    output_dir: Path,
    n_samples: int = 3,
    temperature: Optional[float] = None,
) -> Dict:
    """
    Stage 5: Generate counterfactual trajectories with BMT.
    
    Returns:
        Dict with baseline_trajectory, counterfactual_results
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 5: BMT Counterfactual Generation")
    logger.info("=" * 60)
    
    import torch
    from bmt.utils import utils as bmt_utils
    from bmt.models.motionlm import set_biased_sampler, reset_timestep
    from bmt.utils.utils import numpy_to_torch
    
    bmt_comps = import_bmt_components()
    BiasedTokenSampler = bmt_comps['BiasedTokenSampler']
    TokenBias = bmt_comps['TokenBias']
    compare_trajectories = bmt_comps['compare_trajectories']
    
    scenario_id = stage1_result['scenario_id']
    compiled_interventions = stage4_result['compiled_interventions']
    
    # Load BMT model
    logger.info(f"Loading BMT model from: {bmt_checkpoint}")
    
    try:
        pl_model = bmt_utils.get_model(checkpoint_path=bmt_checkpoint)
        pl_model = pl_model.eval()
        config = pl_model.config
        tokenizer = pl_model.model.tokenizer
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pl_model = pl_model.to(device)
        
        logger.info(f"  Model loaded on {device}")
        logger.info(f"  Sampling: {config.SAMPLING.SAMPLING_METHOD}, temp={config.SAMPLING.TEMPERATURE}")
    except Exception as e:
        logger.error(f"Failed to load BMT model: {e}")
        return {'status': 'model_load_failed', 'error': str(e)}
    
    # Load and preprocess scenario for BMT
    logger.info("Loading scenario for BMT input...")
    data_dir_path = stage1_result.get('data_dir', str(output_dir.parent / "exp_converted"))
    scenario_data = load_scenario_for_bmt_input(
        data_dir_path,
        stage1_result['scenario_index'],
        config,
        tokenizer
    )
    
    # Ground truth will be extracted from preprocessed BMT input (same coordinate frame)
    
    # Use preprocessed if available, else raw
    input_data = scenario_data.get('preprocessed', scenario_data['raw_data'])
    
    # Prepare input
    input_dict = _prepare_bmt_input(input_data, device, config)
    logger.info(f"  Prepared {len(input_dict)} input tensors")
    
    # Extract ground truth from preprocessed BMT input (same coordinate frame as predictions!)
    gt_ego_traj = None
    other_agents_traj = None
    if 'decoder/agent_position' in input_dict:
        import torch
        gt_positions = input_dict['decoder/agent_position']
        if isinstance(gt_positions, torch.Tensor):
            gt_positions = gt_positions.cpu().numpy()
        # Shape: [B, T, N, 2] or [T, N, 2]
        if gt_positions.ndim == 4:
            gt_ego_traj = gt_positions[0, :, 0, :2]  # Ego is agent 0
            other_agents_traj = gt_positions[0, :, 1:, :2].transpose(1, 0, 2)  # [N-1, T, 2]
        elif gt_positions.ndim == 3:
            gt_ego_traj = gt_positions[:, 0, :2]
            other_agents_traj = gt_positions[:, 1:, :2].transpose(1, 0, 2)
        logger.info(f"  GT from BMT input: ego={gt_ego_traj.shape if gt_ego_traj is not None else None}, "
                    f"other_agents={other_agents_traj.shape if other_agents_traj is not None else None}")
    
    # Generate baseline
    logger.info("\nGenerating baseline trajectory...")
    baseline_output = _run_bmt_generation(
        pl_model, config, tokenizer, input_dict, device,
        use_bias=False, sampler=None, temperature=temperature
    )
    baseline_traj = _extract_ego_trajectory(baseline_output)
    
    if baseline_traj is not None:
        logger.info(f"  Baseline trajectory: {baseline_traj.shape}")
        travel = np.sum(np.linalg.norm(np.diff(baseline_traj, axis=0), axis=1))
        logger.info(f"  Travel distance: {travel:.1f}m")
    
    # Generate counterfactuals
    cf_results = []
    
    for i, comp_int in enumerate(compiled_interventions):
        intervention = comp_int['intervention']
        var_id = intervention['variable']
        new_val = intervention['value']
        
        logger.info(f"\n[Intervention {i+1}/{len(compiled_interventions)}]")
        logger.info(f"  do({var_id} = {new_val})")
        
        # Reconstruct TokenBias objects
        token_biases = []
        for b in comp_int['token_biases']:
            token_biases.append(TokenBias(
                token_ids=b['token_ids'],
                bias_value=b['bias_value'],
                timestep_range=tuple(b['timestep_range']),
                description=b.get('description', '')
            ))
        
        if not token_biases:
            logger.warning("  No token biases, skipping")
            continue
        
        # Create sampler
        sampler = BiasedTokenSampler(token_biases)
        logger.info(f"  {len(token_biases)} bias groups active")
        
        # Generate samples
        cf_trajectories = []
        cf_bmt_outputs = []  # Store BMT outputs for replay export
        for sample_idx in range(n_samples):
            logger.info(f"  Generating sample {sample_idx + 1}/{n_samples}...")
            
            cf_output = _run_bmt_generation(
                pl_model, config, tokenizer, input_dict, device,
                use_bias=True, sampler=sampler, temperature=temperature
            )
            
            cf_traj = _extract_ego_trajectory(cf_output)
            if cf_traj is not None:
                cf_trajectories.append(cf_traj)
                # Store first sample's BMT output for replay export
                if sample_idx == 0:
                    cf_bmt_outputs.append(cf_output)
        
        # Compare with baseline
        comparison = None
        if baseline_traj is not None and len(cf_trajectories) > 0:
            # Use first sample for comparison
            comparison = _compare_trajectories_simple(baseline_traj, cf_trajectories[0])
            logger.info(f"  Travel: {comparison['baseline_travel']:.1f}m -> {comparison['counterfactual_travel']:.1f}m")
            logger.info(f"  Reduction: {(1 - comparison['travel_reduction_ratio']) * 100:.1f}%")
        
        cf_results.append({
            'intervention': intervention,
            'effect_prediction': comp_int.get('effect_prediction', {}),
            'n_samples': len(cf_trajectories),
            'trajectories': [t.tolist() for t in cf_trajectories],
            'comparison': comparison,
            '_bmt_output': cf_bmt_outputs[0] if cf_bmt_outputs else None,  # For replay export
        })
    
    # Cleanup
    set_biased_sampler(None)
    
    # Save results
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute baseline travel distance
    baseline_travel_distance = 0.0
    if baseline_traj is not None and len(baseline_traj) > 1:
        baseline_travel_distance = float(np.sum(np.sqrt(np.sum(np.diff(baseline_traj, axis=0)**2, axis=1))))
    
    generation_results = {
        'scenario_id': scenario_id,
        'baseline_trajectory': baseline_traj.tolist() if baseline_traj is not None else None,
        'baseline_travel_distance': baseline_travel_distance,
        'ground_truth_trajectory': gt_ego_traj.tolist() if gt_ego_traj is not None else None,
        'other_agents_trajectories': other_agents_traj.tolist() if other_agents_traj is not None else None,
        'counterfactual_results': cf_results,
        'model_info': {
            'checkpoint': bmt_checkpoint,
            'sampling_method': config.SAMPLING.SAMPLING_METHOD,
            'temperature': temperature or config.SAMPLING.TEMPERATURE,
        }
    }
    
    results_path = scenario_output_dir / "generation_results.json"
    
    # Remove _bmt_output from results before JSON serialization (not JSON-serializable)
    cf_results_for_json = []
    for cf in cf_results:
        cf_json = {k: v for k, v in cf.items() if k != '_bmt_output'}
        cf_results_for_json.append(cf_json)
    generation_results['counterfactual_results'] = cf_results_for_json
    
    with open(results_path, 'w') as f:
        json.dump(generation_results, f, indent=2)
    logger.info(f"\nSaved generation results to {results_path}")
    
    # Export counterfactual scenarios for replay in ScenarioNet/MetaDrive
    try:
        from counter_bmt.scenario_export import (
            export_trajectory_only, 
            create_replay_script,
            export_ground_truth_scenario,
        )
        
        replay_dir = scenario_output_dir / "replay_scenarios"
        replay_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract map_center from preprocessed data (used by Adv-BMT for coordinate transform)
        map_center = None
        if 'preprocessed' in scenario_data:
            preprocessed = scenario_data['preprocessed']
            if 'metadata/map_center' in preprocessed:
                map_center = preprocessed['metadata/map_center']
                logger.info(f"Found map_center for coordinate transform: {map_center}")
        
        exported_paths = []
        
        # First, export the GROUND TRUTH scenario for comparison
        # This comes first so it appears as scenario 0 in the simulator
        gt_output_path = replay_dir / f"sd_counterfactual_1.0_{scenario_id}_00_GROUND_TRUTH.pkl"
        gt_path = export_ground_truth_scenario(
            original_file_path=scenario_data.get('file_path'),
            output_path=gt_output_path,
        )
        if gt_path:
            exported_paths.append(gt_path)
            logger.info(f"Exported ground truth scenario as scenario 0")
        
        # Then export counterfactual scenarios
        for i, cf in enumerate(cf_results):
            if cf.get('trajectories') and len(cf['trajectories']) > 0:
                # Get intervention name
                int_desc = cf.get('intervention', {}).get('description', f'intervention_{i}')
                
                # Get first trajectory sample
                traj = np.array(cf['trajectories'][0])
                
                # Export using trajectory-only method (simpler, more reliable)
                # Filename must start with 'sd_' or be numeric to pass MetaDrive validation
                # Use i+1 since ground truth is 0
                safe_name = _sanitize_intervention_name(int_desc)[:30]
                output_path = replay_dir / f"sd_counterfactual_1.0_{scenario_id}_cf_{i+1:02d}_{safe_name}.pkl"
                
                path = export_trajectory_only(
                    trajectory=traj,
                    original_scenario=scenario_data['raw_data'],
                    output_path=output_path,
                    intervention_name=int_desc,
                    original_file_path=scenario_data.get('file_path'),
                    map_center=map_center,
                )
                if path:
                    exported_paths.append(path)
        
        if exported_paths:
            # Create replay script (this also regenerates dataset_summary.pkl)
            replay_script_path = scenario_output_dir / "replay_scenarios.py"
            create_replay_script(exported_paths, replay_script_path)
            logger.info(f"Exported {len(exported_paths)} scenarios for replay to {replay_dir}")
            logger.info(f"  Scenario 0: GROUND TRUTH (original)")
            logger.info(f"  Scenarios 1-{len(exported_paths)-1}: Counterfactuals")
            logger.info(f"Run: python -m scenarionet.sim -d {replay_dir} --render 2D")
            generation_results['replay_scenarios'] = [str(p) for p in exported_paths]
            
    except Exception as e:
        logger.warning(f"Could not export replay scenarios: {e}")
        import traceback
        traceback.print_exc()
    
    # Note: generation_results['counterfactual_results'] stays as cf_results_for_json
    # (without _bmt_output) to ensure JSON serializability. Use cf_results locally
    # for any processing that needs the full data.
    
    # Generate visualizations and comprehensive metrics
    try:
        analysis_comps = import_analysis_components()
        
        # Compute detailed metrics using full metrics pipeline
        if baseline_traj is not None and cf_results:
            calculator = analysis_comps['TrajectoryMetricsCalculator']()
            
            # Build counterfactual trajectories dict for metrics and visualization
            cf_trajs_dict = {}
            expected_effects = {}
            for cf_result in cf_results:
                intervention = cf_result.get('intervention', {})
                int_name = intervention.get('description')
                if not int_name:
                    var_id = intervention.get('variable', 'unknown')
                    new_val = intervention.get('value', '')
                    int_name = f"do({var_id}={new_val})"
                trajs = [np.array(t) for t in cf_result.get('trajectories', [])]
                if trajs:
                    cf_trajs_dict[int_name] = trajs
                    # Get predicted effect for accuracy tracking
                    expected_effects[int_name] = cf_result.get('predicted_effect', 'decrease')
            
            # Generate comprehensive metrics summary (ADV-BMT paper metrics)
            if cf_trajs_dict:
                # Get ground truth and other agents from generation results (if available)
                gt_traj = generation_results.get('ground_truth_trajectory')
                if gt_traj is not None:
                    gt_traj = np.array(gt_traj)[:, :2]  # Only use x,y
                
                other_agents = generation_results.get('other_agents_trajectories')
                if other_agents is not None:
                    other_agents = np.array(other_agents)
                
                # GT and predictions are now in the SAME coordinate frame (from BMT preprocessing)
                # No rotation needed - just truncate to same length
                
                baseline_aligned = baseline_traj.copy()
                cf_trajs_aligned = cf_trajs_dict  # No modification needed
                
                # Align ground truth (truncate to prediction length)
                gt_aligned = None
                if gt_traj is not None and len(gt_traj) > 0:
                    pred_len = len(baseline_aligned)
                    gt_aligned = gt_traj[:pred_len] if len(gt_traj) >= pred_len else gt_traj
                    
                    # Debug: Log trajectory comparison (now in same coordinate frame!)
                    logger.info(f"=== Trajectory Alignment Debug ===")
                    logger.info(f"  Baseline: start={baseline_aligned[0]}, end={baseline_aligned[-1]}, len={len(baseline_aligned)}")
                    logger.info(f"  GT:       start={gt_aligned[0]}, end={gt_aligned[-1]}, len={len(gt_aligned)}")
                    logger.info(f"  Start diff: {np.linalg.norm(baseline_aligned[0] - gt_aligned[0]):.2f}m")
                    logger.info(f"  End diff:   {np.linalg.norm(baseline_aligned[-1] - gt_aligned[-1]):.2f}m")
                    
                    # Travel distance comparison
                    baseline_travel = np.sum(np.linalg.norm(np.diff(baseline_aligned, axis=0), axis=1))
                    gt_travel = np.sum(np.linalg.norm(np.diff(gt_aligned, axis=0), axis=1))
                    logger.info(f"  Baseline travel: {baseline_travel:.2f}m, GT travel: {gt_travel:.2f}m")
                
                # Other agents (truncate to prediction length)
                other_agents_aligned = None
                if other_agents is not None and len(other_agents) > 0:
                    pred_len = len(baseline_aligned)
                    other_agents_aligned = other_agents[:, :pred_len, :] if other_agents.shape[1] >= pred_len else other_agents
                
                metrics_summary = analysis_comps['generate_metrics_summary'](
                    baseline=baseline_aligned,
                    interventions=cf_trajs_aligned,
                    expected_effects=expected_effects,
                    ground_truth=gt_aligned,
                    other_agents=other_agents_aligned,
                )
                
                # === SANITY CHECK: Baseline vs Ground Truth ===
                # This shows how well BMT predicts the actual logged behavior (without interventions)
                if gt_aligned is not None:
                    calc = analysis_comps['TrajectoryMetricsCalculator']()
                    min_len = min(len(baseline_aligned), len(gt_aligned))
                    baseline_vs_gt_ade = calc._compute_ade(baseline_aligned[:min_len], gt_aligned[:min_len])
                    baseline_vs_gt_fde = calc._compute_fde(baseline_aligned[:min_len], gt_aligned[:min_len])
                    
                    sanity_check = {
                        'baseline_vs_gt_ade': float(baseline_vs_gt_ade),
                        'baseline_vs_gt_fde': float(baseline_vs_gt_fde),
                        'baseline_travel_distance': float(calc._compute_travel_distance(baseline_aligned)),
                        'gt_travel_distance': float(calc._compute_travel_distance(gt_aligned[:min_len])),
                        'trajectory_length': min_len,
                    }
                    metrics_summary['sanity_check'] = sanity_check
                    
                    logger.info(f"\n=== SANITY CHECK: Baseline vs Ground Truth ===")
                    logger.info(f"  Baseline ADE vs GT: {baseline_vs_gt_ade:.2f} m")
                    logger.info(f"  Baseline FDE vs GT: {baseline_vs_gt_fde:.2f} m")
                    logger.info(f"  Baseline travel:    {sanity_check['baseline_travel_distance']:.2f} m")
                    logger.info(f"  GT travel:          {sanity_check['gt_travel_distance']:.2f} m")
                    logger.info(f"=" * 50)
                
                # Save detailed metrics to separate JSON file
                metrics_path = scenario_output_dir / "detailed_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_summary, f, indent=2)
                logger.info(f"Saved detailed metrics to {metrics_path}")
                
                # Add metrics summary to generation results
                generation_results['detailed_metrics'] = metrics_summary
                generation_results['overall_stats'] = metrics_summary.get('overall', {})
                
                # Update the main results file with metrics
                with open(results_path, 'w') as f:
                    json.dump(generation_results, f, indent=2)
            
            # Generate trajectory comparison visualization
            if cf_trajs_dict:
                viz_path = scenario_output_dir / "trajectory_comparison.png"
                analysis_comps['visualize_trajectory_comparison'](
                    baseline=baseline_traj,
                    counterfactuals=cf_trajs_dict,
                    output_path=viz_path,
                    scenario_id=scenario_id,
                )
                logger.info(f"Saved trajectory visualization to {viz_path}")
            
            # Generate intervention summary chart
            summary_viz_path = scenario_output_dir / "intervention_summary.png"
            summary_results = {
                'generation_results': {
                    'baseline_travel_distance': float(np.sum(np.sqrt(np.sum(np.diff(baseline_traj, axis=0)**2, axis=1)))),
                    'counterfactuals': {}
                }
            }
            for cf_result in cf_results:
                intervention = cf_result.get('intervention', {})
                int_name = intervention.get('description')
                if not int_name:
                    var_id = intervention.get('variable', 'unknown')
                    new_val = intervention.get('value', '')
                    int_name = f"do({var_id}={new_val})"
                if cf_result.get('comparison'):
                    summary_results['generation_results']['counterfactuals'][int_name] = {
                        'mean_travel_distance': cf_result['comparison'].get('counterfactual_travel', 0)
                    }
            analysis_comps['visualize_intervention_summary'](summary_results, summary_viz_path)
            logger.info(f"Saved intervention summary to {summary_viz_path}")
            
            # Generate text summary with metrics
            _save_metrics_summary_text(scenario_output_dir, scenario_id, generation_results, metrics_summary if cf_trajs_dict else None)
            
    except Exception as e:
        logger.warning(f"Could not generate visualizations/metrics: {e}")
        import traceback
        traceback.print_exc()
    
    return generation_results


# =============================================================================
# Metrics Summary Helper
# =============================================================================

def _save_metrics_summary_text(output_dir: Path, scenario_id: str, 
                                generation_results: Dict, metrics_summary: Optional[Dict] = None):
    """Save a comprehensive text summary with ADV-BMT paper metrics."""
    summary_path = output_dir / "metrics_summary.txt"
    
    lines = [
        "=" * 70,
        "COUNTERBMT DETAILED METRICS SUMMARY",
        "=" * 70,
        f"Scenario ID: {scenario_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Baseline metrics
    if metrics_summary and 'baseline' in metrics_summary:
        baseline = metrics_summary['baseline']
        lines.extend([
            "BASELINE TRAJECTORY METRICS:",
            "-" * 40,
            f"  Travel Distance:    {baseline.get('travel_distance', 0):.2f} m",
            f"  Trajectory Length:  {baseline.get('trajectory_length', 0)} steps",
            f"  Mean Speed:         {baseline.get('mean_speed', 0):.2f} m/s",
            f"  Max Speed:          {baseline.get('max_speed', 0):.2f} m/s",
            f"  Mean Acceleration:  {baseline.get('mean_acceleration', 0):.2f} m/s²",
            f"  Max Acceleration:   {baseline.get('max_acceleration', 0):.2f} m/s²",
            f"  Max Jerk:           {baseline.get('max_jerk', 0):.2f} m/s³",
            "",
        ])
    
    # SANITY CHECK: Baseline vs Ground Truth
    if metrics_summary and 'sanity_check' in metrics_summary:
        sc = metrics_summary['sanity_check']
        lines.extend([
            "*** SANITY CHECK: BMT Baseline vs Ground Truth ***",
            "-" * 40,
            "  (This shows BMT prediction accuracy WITHOUT any intervention)",
            f"  Baseline ADE vs GT: {sc.get('baseline_vs_gt_ade', 0):.2f} m",
            f"  Baseline FDE vs GT: {sc.get('baseline_vs_gt_fde', 0):.2f} m",
            f"  Baseline Travel:    {sc.get('baseline_travel_distance', 0):.2f} m",
            f"  GT Travel:          {sc.get('gt_travel_distance', 0):.2f} m",
            "",
        ])
    
    # ADV-BMT Aggregate Metrics (Table a & b from paper)
    if metrics_summary and 'aggregate_advbmt' in metrics_summary:
        agg = metrics_summary['aggregate_advbmt']
        lines.extend([
            "ADV-BMT PAPER METRICS (Aggregate):",
            "-" * 40,
            "",
            "  Realism Metrics (Table a):",
            f"    SFDE_avg:       {agg.get('sfde_avg', 0):.2f} m",
            f"    SFDE_min:       {agg.get('sfde_min', 0):.2f} m",
            f"    SADE_avg:       {agg.get('sade_avg', 0):.2f} m",
            f"    SADE_min:       {agg.get('sade_min', 0):.2f} m",
            f"    VehColl_avg:    {agg.get('veh_coll_avg', 0):.4f}",
            f"    VehColl_min:    {agg.get('veh_coll_min', 0):.4f}",
            f"    JSD_velocity:   {agg.get('jsd_velocity', 0):.4f}",
            f"    JSD_TTC:        {agg.get('jsd_ttc', 0):.4f}",
            "",
            "  Diversity Metrics (Table b):",
            f"    Mean SDD:       {agg.get('mean_sdd', 0):.2f} m",
            f"    Mean FDD:       {agg.get('mean_fdd', 0):.2f} m",
            f"    Mean ADD:       {agg.get('mean_add', 0):.2f} m",
            "",
        ])
    
    # Overall statistics
    if metrics_summary and 'overall' in metrics_summary:
        overall = metrics_summary['overall']
        lines.extend([
            "OVERALL INTERVENTION STATISTICS:",
            "-" * 40,
            f"  Total Interventions:    {overall.get('n_interventions', 0)}",
            f"  Effective (>5% change): {overall.get('effective_interventions', 0)}",
            f"  Effectiveness Rate:     {overall.get('effectiveness_rate', 0)*100:.1f}%",
            f"  Match Expected Effect:  {overall.get('matching_expected', 0)}",
            f"  Prediction Accuracy:    {overall.get('prediction_accuracy', 0)*100:.1f}%",
            "",
        ])
    
    # Per-intervention metrics
    if metrics_summary and 'interventions' in metrics_summary:
        lines.extend([
            "PER-INTERVENTION METRICS:",
            "-" * 40,
        ])
        for name, data in metrics_summary['interventions'].items():
            lines.append(f"\n  {name}:")
            lines.append(f"    Distance Change: {data.get('distance_change_percent', 0):+.1f}%")
            lines.append(f"    Mean CF Distance: {data.get('mean_counterfactual_distance', 0):.2f} m")
            
            # ADV-BMT metrics for this intervention
            adv = data.get('advbmt_metrics', {})
            if adv:
                lines.append(f"    --- ADV-BMT Realism ---")
                lines.append(f"    SFDE_avg: {adv.get('sfde_avg', 0):.2f} m  |  SFDE_min: {adv.get('sfde_min', 0):.2f} m")
                lines.append(f"    SADE_avg: {adv.get('sade_avg', 0):.2f} m  |  SADE_min: {adv.get('sade_min', 0):.2f} m")
                lines.append(f"    VehColl_avg: {adv.get('veh_coll_avg', 0):.4f}  |  VehColl_min: {adv.get('veh_coll_min', 0):.4f}")
                lines.append(f"    JSD_velocity: {adv.get('jsd_velocity', 0):.4f}  |  JSD_TTC: {adv.get('jsd_ttc', 0):.4f}")
                lines.append(f"    --- ADV-BMT Diversity ---")
                lines.append(f"    SDD: {adv.get('sdd', 0):.2f} m  |  FDD: {adv.get('fdd', 0):.2f} m  |  ADD: {adv.get('add', 0):.2f} m")
            
            # Effect assessment
            lines.append(f"    Effect Direction: {data.get('effect_direction', 'unknown')}")
            lines.append(f"    Matches Expected: {'✓' if data.get('effect_matches_expected') else '✗'}")
        lines.append("")
    
    lines.extend([
        "=" * 70,
        "END OF METRICS SUMMARY",
        "=" * 70,
    ])
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved metrics summary to {summary_path}")


# =============================================================================
# BMT Helper Functions
# =============================================================================

def _prepare_bmt_input(raw_data: Dict, device: str, config) -> Dict:
    """Prepare scenario data for BMT input."""
    import torch
    from bmt.utils.utils import numpy_to_torch
    import copy
    
    input_dict = {}
    for k, v in raw_data.items():
        if isinstance(v, np.ndarray) and 'track_name' not in k:
            input_dict[k] = v.copy()
        elif hasattr(v, 'clone'):  # torch tensor
            input_dict[k] = v.clone()
        elif isinstance(v, str):
            input_dict[k] = v
        else:
            input_dict[k] = copy.deepcopy(v)
    
    input_dict = numpy_to_torch(input_dict, device=device)
    
    # Ensure float tensors are float32 (model uses float32 weights)
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.float64:
                input_dict[k] = v.float()  # Convert double to float32
    
    # Add batch dimension if needed
    if 'decoder/agent_position' in input_dict:
        pos = input_dict['decoder/agent_position']
        if isinstance(pos, torch.Tensor) and pos.dim() == 3:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1:
                    input_dict[k] = v.unsqueeze(0)
    
    input_dict["in_evaluation"] = torch.tensor([True], dtype=torch.bool).to(device)
    
    return input_dict


def _run_bmt_generation(
    pl_model, config, tokenizer, input_dict, device,
    use_bias: bool, sampler, temperature: Optional[float]
) -> Dict:
    """Run single BMT generation."""
    import torch
    import copy
    from bmt.models.motionlm import set_biased_sampler, reset_timestep
    
    # Deep copy
    input_copy = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_copy[k] = v.clone()
        elif isinstance(v, str):
            input_copy[k] = v
        else:
            input_copy[k] = copy.deepcopy(v)
    
    # Setup bias
    if use_bias and sampler is not None:
        reset_timestep()
        set_biased_sampler(sampler)
    else:
        set_biased_sampler(None)
    
    # Note: Data is already tokenized by preprocess_scenario_description_for_motionlm
    
    # Generate
    sampling_temp = temperature if temperature is not None else config.SAMPLING.TEMPERATURE
    
    with torch.no_grad():
        output_dict = pl_model.model.autoregressive_rollout(
            input_copy,
            num_decode_steps=None,
            sampling_method=config.SAMPLING.SAMPLING_METHOD,
            temperature=sampling_temp,
        )
    
    # Detokenize
    flip_heading = getattr(config.TOKENIZATION, 'FLIP_WRONG_HEADING', True)
    output_dict = tokenizer.detokenize(
        output_dict,
        detokenizing_gt=False,
        backward_prediction=False,
        flip_wrong_heading=flip_heading,
    )
    
    set_biased_sampler(None)
    return output_dict


def _extract_ego_trajectory(output_dict: Dict) -> Optional[np.ndarray]:
    """Extract ego trajectory from BMT output."""
    import torch
    
    if output_dict is None:
        return None
    
    for key in ['decoder/reconstructed_position', 'decoder/agent_position']:
        if key in output_dict:
            positions = output_dict[key]
            break
    else:
        return None
    
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    
    if positions.ndim == 4:  # [B, T, N, D]
        return positions[0, :, 0, :2]
    elif positions.ndim == 3:  # [T, N, D]
        return positions[:, 0, :2]
    
    return None


def _sanitize_intervention_name(name: str) -> str:
    """Sanitize intervention name for use in filenames."""
    sanitized = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized = sanitized.replace(':', '_').replace('(', '').replace(')', '')
    sanitized = sanitized.replace(',', '_').replace('.', '_')
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized[:40]


def _compare_trajectories_simple(baseline: np.ndarray, counterfactual: np.ndarray) -> Dict:
    """Compare two trajectories."""
    T = min(len(baseline), len(counterfactual))
    b, c = baseline[:T], counterfactual[:T]
    
    diff = np.linalg.norm(c - b, axis=1)
    b_travel = np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1))
    c_travel = np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))
    
    return {
        'max_displacement_diff': float(diff.max()),
        'mean_displacement_diff': float(diff.mean()),
        'final_displacement_diff': float(np.linalg.norm(c[-1] - b[-1])),
        'baseline_travel': float(b_travel),
        'counterfactual_travel': float(c_travel),
        'travel_reduction_ratio': float(c_travel / max(b_travel, 0.01)),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_full_pipeline(
    data_dir: Path,
    output_dir: Path,
    scenario_index: int,
    bmt_checkpoint: Optional[str] = None,
    use_mock: bool = False,
    num_frames: int = 8,
    max_interventions: int = 5,
    n_samples: int = 3,
    temperature: Optional[float] = None,
) -> Dict:
    """
    Run the complete CounterBMT pipeline.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CounterBMT Full Pipeline")
    logger.info("=" * 70)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Scenario index: {scenario_index}")
    logger.info(f"Use mock: {use_mock}")
    logger.info(f"BMT checkpoint: {bmt_checkpoint}")
    logger.info("=" * 70)
    
    # Initialize clients
    dag_comps = import_dag_components()
    
    if use_mock:
        logger.info("\nUsing MOCK clients (no API calls)")
        vlm_client = dag_comps['MockGPT4oClient']()
        dag_client = dag_comps['MockDAGClient']()
    else:
        logger.info("\nInitializing GPT-4o clients...")
        try:
            vlm_client = dag_comps['GPT4oClient']()
            dag_client = dag_comps['GPT4oDAGClient']()
        except ValueError as e:
            logger.error(f"Failed to initialize API clients: {e}")
            logger.error("Set OPENAI_API_KEY or use --mock")
            return {'status': 'api_init_failed', 'error': str(e)}
    
    results = {
        'scenario_index': scenario_index,
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    try:
        # Stage 1: Load and visualize
        stage1 = stage_1_load_and_visualize(data_dir, scenario_index, output_dir, num_frames)
        stage1['data_dir'] = str(data_dir)  # Save for later
        results['stages']['stage1'] = {'status': 'success', 'scenario_id': stage1['scenario_id']}
        results['scenario_id'] = stage1['scenario_id']
        
        # Stage 2: VLM extraction
        stage2 = stage_2_vlm_extraction(stage1, vlm_client)
        results['stages']['stage2'] = {
            'status': 'success',
            'n_maneuvers': len(stage2['maneuvers']),
            'n_decisions': len(stage2['decisions']),
        }
        
        # Stage 3: DAG construction
        stage3 = stage_3_dag_construction(stage1, stage2, dag_client, output_dir)
        results['stages']['stage3'] = {
            'status': 'success',
            'n_nodes': len(stage3['dag'].nodes),
            'n_edges': len(stage3['dag'].edges),
            'n_interventions': len(stage3['interventions']),
        }
        
        # Stage 4: Compile interventions
        stage4 = stage_4_compile_interventions(stage3, max_interventions)
        results['stages']['stage4'] = {
            'status': 'success',
            'n_compiled': len(stage4['compiled_interventions']),
        }
        
        # Stage 5: BMT generation (if checkpoint provided)
        if bmt_checkpoint:
            stage5 = stage_5_bmt_generation(
                stage1, stage4, bmt_checkpoint, output_dir,
                n_samples=n_samples, temperature=temperature
            )
            results['stages']['stage5'] = {
                'status': stage5.get('status', 'success'),
                'n_counterfactuals': len(stage5.get('counterfactual_results', [])),
            }
            results['generation_results'] = stage5
        else:
            logger.info("\n[Skipping Stage 5: No BMT checkpoint provided]")
            results['stages']['stage5'] = {'status': 'skipped', 'reason': 'no_checkpoint'}
        
        results['status'] = 'success'
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['error'] = str(e)
    
    # Save overall results
    scenario_id = results.get('scenario_id', f'scenario_{scenario_index}')
    scenario_output_dir = output_dir / scenario_id
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = scenario_output_dir / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate comprehensive output package
    analysis_comps = None
    try:
        analysis_comps = import_analysis_components()
    except Exception as e:
        logger.warning(f"Could not import analysis components: {e}")
    
    # Generate visualizations (separate try-except for resilience)
    if analysis_comps:
        gen_results = results.get('generation_results', {})
        
        # Trajectory visualization
        try:
            if gen_results and gen_results.get('baseline_trajectory'):
                baseline_arr = np.array(gen_results['baseline_trajectory'])
                cf_trajs_dict = {}
                for cf_result in gen_results.get('counterfactual_results', []):
                    int_name = cf_result.get('intervention', {}).get('description', 'unknown')
                    trajs = [np.array(t) for t in cf_result.get('trajectories', [])]
                    if trajs:
                        cf_trajs_dict[int_name] = trajs
                
                if cf_trajs_dict:
                    viz_path = scenario_output_dir / "trajectory_comparison.png"
                    analysis_comps['visualize_trajectory_comparison'](
                        baseline=baseline_arr,
                        counterfactuals=cf_trajs_dict,
                        output_path=viz_path,
                        scenario_id=scenario_id,
                    )
                    logger.info(f"Saved trajectory comparison to {viz_path}")
        except Exception as e:
            logger.warning(f"Could not generate trajectory visualization: {e}")
        
        # Intervention summary chart
        try:
            if gen_results and gen_results.get('baseline_trajectory'):
                baseline_arr = np.array(gen_results['baseline_trajectory'])
                baseline_dist = float(np.sum(np.sqrt(np.sum(np.diff(baseline_arr, axis=0)**2, axis=1))))
                
                summary_results = {
                    'generation_results': {
                        'baseline_travel_distance': baseline_dist,
                        'counterfactuals': {}
                    }
                }
                for cf_result in gen_results.get('counterfactual_results', []):
                    int_name = cf_result.get('intervention', {}).get('description', 'unknown')
                    if cf_result.get('comparison'):
                        summary_results['generation_results']['counterfactuals'][int_name] = {
                            'mean_travel_distance': cf_result['comparison'].get('counterfactual_travel', 0)
                        }
                
                summary_viz_path = scenario_output_dir / "intervention_summary.png"
                analysis_comps['visualize_intervention_summary'](summary_results, summary_viz_path)
                logger.info(f"Saved intervention summary to {summary_viz_path}")
        except Exception as e:
            logger.warning(f"Could not generate intervention summary: {e}")
        
        # Comprehensive output manager
        try:
            output_manager = analysis_comps['PipelineOutputManager'](scenario_id, scenario_output_dir)
            
            # Set config
            output_manager.set_config({
                'data_dir': str(data_dir),
                'scenario_index': scenario_index,
                'bmt_checkpoint': bmt_checkpoint,
                'use_mock': use_mock,
                'num_frames': num_frames,
                'max_interventions': max_interventions,
                'n_samples': n_samples,
            })
            
            # Add stage results
            for stage_name, stage_result in results.get('stages', {}).items():
                if stage_result.get('status') == 'success':
                    output_manager.complete_stage(stage_name, stage_result)
                elif stage_result.get('status') == 'failed':
                    output_manager.fail_stage(stage_name, stage_result.get('error', 'Unknown'))
            
            # Set DAG if available
            try:
                if 'stage3' in locals() and stage3.get('dag'):
                    output_manager.set_dag(
                        stage3['dag'],
                        maneuvers=stage2.get('maneuvers', []) if 'stage2' in locals() else None,
                        decisions=stage2.get('decisions', []) if 'stage2' in locals() else None,
                    )
            except Exception as e:
                logger.warning(f"Could not set DAG in output manager: {e}")
            
            # Set baseline and intervention results if available
            if gen_results:
                baseline_traj = gen_results.get('baseline_trajectory')
                if baseline_traj:
                    baseline_arr = np.array(baseline_traj)
                    baseline_dist = float(np.sum(np.sqrt(np.sum(np.diff(baseline_arr, axis=0)**2, axis=1))))
                    output_manager.set_baseline(baseline_arr, baseline_dist)
                
                # Add intervention results
                for cf_result in gen_results.get('counterfactual_results', []):
                    try:
                        intervention = cf_result.get('intervention', {})
                        effect_pred = cf_result.get('effect_prediction', {})
                        
                        # Extract variable ID and value from intervention dict
                        var_id = intervention.get('variable', intervention.get('node_id', 'unknown'))
                        new_val = intervention.get('value', intervention.get('new_value'))
                        original_val = intervention.get('original_value')
                        description = intervention.get('description', f"Set {var_id} to {new_val}")
                        
                        output_manager.add_intervention_result(
                            intervention_id=var_id,
                            intervention_name=description,
                            description=description,
                            target_node=var_id,
                            original_value=original_val,
                            new_value=new_val,
                            predicted_effect=effect_pred.get('effect', 'unknown'),
                            prediction_confidence=float(effect_pred.get('confidence', 0) or 0),
                            prediction_reasoning=effect_pred.get('reasoning', ''),
                            trajectories=[np.array(t) for t in cf_result.get('trajectories', [])],
                            bias_groups=len(cf_result.get('bias_groups', [])),
                            total_biased_tokens=cf_result.get('total_biased_tokens', 0),
                        )
                    except Exception as e:
                        logger.warning(f"Could not add intervention result: {e}")
            
            # Export all outputs
            export_paths = output_manager.export_all()
            logger.info(f"Generated comprehensive output package: {list(export_paths.keys())}")
        except Exception as e:
            logger.warning(f"Could not generate comprehensive output package: {e}")
        
        # Generate scenario report (separate try-except)
        try:
            if gen_results and gen_results.get('baseline_trajectory'):
                dag_data = None
                if 'stage3' in locals() and stage3.get('dag') and hasattr(stage3.get('dag'), 'to_dict'):
                    try:
                        dag_data = stage3['dag'].to_dict()
                    except:
                        pass
                
                report_path = analysis_comps['create_scenario_report'](
                    scenario_id=scenario_id,
                    output_dir=scenario_output_dir,
                    baseline_trajectory=np.array(gen_results['baseline_trajectory']),
                    counterfactual_results=gen_results,
                    dag_data=dag_data,
                    llm_logs=None,
                )
                logger.info(f"Generated scenario report: {report_path}")
        except Exception as e:
            logger.warning(f"Could not generate scenario report: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline Complete")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 70)
    
    return results


# =============================================================================
# Batch Processing
# =============================================================================

def run_batch_pipeline(
    data_dir: Path,
    output_dir: Path,
    start_index: int = 0,
    end_index: Optional[int] = None,
    num_scenarios: int = 10,
    bmt_checkpoint: Optional[str] = None,
    use_mock: bool = False,
    num_frames: int = 8,
    max_interventions: int = 5,
    n_samples: int = 3,
    temperature: Optional[float] = None,
    continue_on_error: bool = True,
) -> int:
    """
    Run the CounterBMT pipeline on multiple scenarios in batch mode.
    
    Generates aggregate metrics across all scenarios for comparison with ADV-BMT paper.
    
    Returns:
        0 if all succeeded, 1 if any failed
    """
    from counter_bmt.scenarionet_visualizer import ScenarioNetDatabase
    
    # Initialize database to get scenario count
    logger.info("=" * 70)
    logger.info("COUNTERBMT BATCH PROCESSING")
    logger.info("=" * 70)
    
    db = ScenarioNetDatabase(data_dir)
    total_available = len(db)
    
    # Determine range
    if end_index is None:
        end_index = min(start_index + num_scenarios, total_available)
    end_index = min(end_index, total_available)
    
    scenario_indices = list(range(start_index, end_index))
    n_scenarios = len(scenario_indices)
    
    logger.info(f"Processing scenarios {start_index} to {end_index-1} ({n_scenarios} total)")
    logger.info(f"Total available scenarios: {total_available}")
    logger.info(f"BMT Checkpoint: {bmt_checkpoint or 'SKIPPED'}")
    logger.info(f"Mock mode: {use_mock}")
    logger.info("=" * 70)
    
    # Results tracking
    batch_results = {
        'config': {
            'data_dir': str(data_dir),
            'start_index': start_index,
            'end_index': end_index,
            'n_scenarios': n_scenarios,
            'use_mock': use_mock,
            'bmt_checkpoint': bmt_checkpoint,
        },
        'scenarios': [],
        'aggregate_metrics': {},
        'failures': [],
    }
    
    # Aggregate metric accumulators - comprehensive ADV-BMT metrics
    metrics_accumulators = {
        # Basic metrics
        'baseline_distance': [],
        'gt_distance': [],
        'cf_change_percent': [],
        'effective_count': [],
        
        # Sanity check (baseline vs GT)
        'baseline_vs_gt_ade': [],
        'baseline_vs_gt_fde': [],
        
        # ADV-BMT Realism metrics (per intervention)
        'sfde_avg': [],
        'sfde_min': [],
        'sade_avg': [],
        'sade_min': [],
        'veh_coll_avg': [],
        'veh_coll_min': [],
        'jsd_velocity': [],
        'jsd_ttc': [],
        
        # ADV-BMT Diversity metrics (per intervention)
        'fdd': [],
        'sdd': [],
        'add': [],
        
        # Aggregate ADV-BMT (per scenario)
        'scenario_sfde_avg': [],
        'scenario_sade_avg': [],
        'scenario_jsd_velocity': [],
        'scenario_fdd': [],
        'scenario_sdd': [],
        'scenario_add': [],
    }
    
    for i, scenario_idx in enumerate(scenario_indices):
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH PROGRESS: {i+1}/{n_scenarios} (Scenario index: {scenario_idx})")
        logger.info("=" * 70)
        
        try:
            result = run_full_pipeline(
                data_dir=data_dir,
                output_dir=output_dir,
                scenario_index=scenario_idx,
                bmt_checkpoint=bmt_checkpoint,
                use_mock=use_mock,
                num_frames=num_frames,
                max_interventions=max_interventions,
                n_samples=n_samples,
                temperature=temperature,
            )
            
            # Get generation results from stage 5
            gen_results = result.get('generation_results', {})
            
            scenario_summary = {
                'index': scenario_idx,
                'scenario_id': result.get('scenario_id', 'unknown'),
                'status': result.get('status', 'unknown'),
                'baseline_distance': gen_results.get('baseline_travel_distance', 0) or result.get('baseline_travel_distance', 0),
                'n_interventions': len(gen_results.get('counterfactual_results', [])),
            }
            
            # Extract metrics if available - detailed_metrics is inside generation_results
            dm = gen_results.get('detailed_metrics', {})
            if dm:
                
                # Basic metrics
                baseline_dist = dm.get('baseline', {}).get('travel_distance', 0)
                metrics_accumulators['baseline_distance'].append(baseline_dist)
                
                overall = dm.get('overall', {})
                scenario_summary['effectiveness_rate'] = overall.get('effectiveness_rate', 0)
                scenario_summary['prediction_accuracy'] = overall.get('prediction_accuracy', 0)
                metrics_accumulators['effective_count'].append(overall.get('effective_interventions', 0))
                
                # Sanity check metrics (baseline vs ground truth)
                sanity = dm.get('sanity_check', {})
                if sanity.get('baseline_vs_gt_ade') is not None:
                    metrics_accumulators['baseline_vs_gt_ade'].append(sanity['baseline_vs_gt_ade'])
                if sanity.get('baseline_vs_gt_fde') is not None:
                    metrics_accumulators['baseline_vs_gt_fde'].append(sanity['baseline_vs_gt_fde'])
                if sanity.get('gt_travel_distance') is not None:
                    metrics_accumulators['gt_distance'].append(sanity['gt_travel_distance'])
                
                # Aggregate ADV-BMT metrics (scenario level)
                agg_adv = dm.get('aggregate_advbmt', {})
                if agg_adv:
                    for key in ['sfde_avg', 'sade_avg', 'jsd_velocity']:
                        if agg_adv.get(key) is not None:
                            metrics_accumulators[f'scenario_{key}'].append(agg_adv[key])
                    for key in ['mean_fdd', 'mean_sdd', 'mean_add']:
                        short_key = key.replace('mean_', '')
                        if agg_adv.get(key) is not None:
                            metrics_accumulators[f'scenario_{short_key}'].append(agg_adv[key])
                
                # Collect per-intervention ADV-BMT metrics
                for int_name, int_data in dm.get('interventions', {}).items():
                    metrics_accumulators['cf_change_percent'].append(int_data.get('distance_change_percent', 0))
                    
                    # ADV-BMT metrics from each intervention
                    adv = int_data.get('advbmt_metrics', {})
                    if adv:
                        for key in ['sfde_avg', 'sfde_min', 'sade_avg', 'sade_min', 
                                   'veh_coll_avg', 'veh_coll_min', 'jsd_velocity', 'jsd_ttc']:
                            if adv.get(key) is not None:
                                metrics_accumulators[key].append(adv[key])
                        for key in ['fdd', 'sdd', 'add']:
                            if adv.get(key) is not None:
                                metrics_accumulators[key].append(adv[key])
            
            batch_results['scenarios'].append(scenario_summary)
            logger.info(f"✓ Scenario {scenario_idx} completed successfully")
            
        except Exception as e:
            logger.error(f"✗ Scenario {scenario_idx} failed: {e}")
            batch_results['failures'].append({
                'index': scenario_idx,
                'error': str(e)
            })
            if not continue_on_error:
                logger.error("Stopping batch due to error (use --continue-on-error to skip)")
                break
    
    # Compute aggregate metrics with quartiles
    n_successful = len(batch_results['scenarios'])
    n_failed = len(batch_results['failures'])
    
    def compute_stats(values: List[float]) -> Dict[str, float]:
        """Compute mean, std, and quartiles for a list of values."""
        if not values:
            return {'mean': None, 'std': None, 'q1': None, 'median': None, 'q3': None, 'min': None, 'max': None, 'n': 0}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'q1': float(np.percentile(arr, 25)),
            'median': float(np.percentile(arr, 50)),
            'q3': float(np.percentile(arr, 75)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'n': len(values),
        }
    
    # Basic summary
    batch_results['aggregate_metrics'] = {
        'n_scenarios_processed': n_successful,
        'n_scenarios_failed': n_failed,
        'success_rate': n_successful / n_scenarios if n_scenarios > 0 else 0,
        'total_interventions': len(metrics_accumulators['cf_change_percent']),
        'total_effective_interventions': sum(metrics_accumulators['effective_count']),
    }
    
    # Compute detailed statistics for each metric
    batch_results['detailed_statistics'] = {}
    
    # Basic metrics
    batch_results['detailed_statistics']['baseline_distance'] = compute_stats(metrics_accumulators['baseline_distance'])
    batch_results['detailed_statistics']['gt_distance'] = compute_stats(metrics_accumulators['gt_distance'])
    batch_results['detailed_statistics']['cf_change_percent'] = compute_stats(metrics_accumulators['cf_change_percent'])
    
    # Sanity check metrics
    batch_results['detailed_statistics']['baseline_vs_gt_ade'] = compute_stats(metrics_accumulators['baseline_vs_gt_ade'])
    batch_results['detailed_statistics']['baseline_vs_gt_fde'] = compute_stats(metrics_accumulators['baseline_vs_gt_fde'])
    
    # ADV-BMT Realism metrics (from individual interventions)
    batch_results['detailed_statistics']['advbmt_realism'] = {
        'sfde_avg': compute_stats(metrics_accumulators['sfde_avg']),
        'sfde_min': compute_stats(metrics_accumulators['sfde_min']),
        'sade_avg': compute_stats(metrics_accumulators['sade_avg']),
        'sade_min': compute_stats(metrics_accumulators['sade_min']),
        'veh_coll_avg': compute_stats(metrics_accumulators['veh_coll_avg']),
        'veh_coll_min': compute_stats(metrics_accumulators['veh_coll_min']),
        'jsd_velocity': compute_stats(metrics_accumulators['jsd_velocity']),
        'jsd_ttc': compute_stats(metrics_accumulators['jsd_ttc']),
    }
    
    # ADV-BMT Diversity metrics (from individual interventions)
    batch_results['detailed_statistics']['advbmt_diversity'] = {
        'fdd': compute_stats(metrics_accumulators['fdd']),
        'sdd': compute_stats(metrics_accumulators['sdd']),
        'add': compute_stats(metrics_accumulators['add']),
    }
    
    # Scenario-level aggregate ADV-BMT metrics
    batch_results['detailed_statistics']['scenario_advbmt'] = {
        'sfde_avg': compute_stats(metrics_accumulators['scenario_sfde_avg']),
        'sade_avg': compute_stats(metrics_accumulators['scenario_sade_avg']),
        'jsd_velocity': compute_stats(metrics_accumulators['scenario_jsd_velocity']),
        'fdd': compute_stats(metrics_accumulators['scenario_fdd']),
        'sdd': compute_stats(metrics_accumulators['scenario_sdd']),
        'add': compute_stats(metrics_accumulators['scenario_add']),
    }
    
    # Save batch results
    batch_output_path = output_dir / "batch_results.json"
    with open(batch_output_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    # Save aggregate summary text
    _save_aggregate_summary(output_dir, batch_results)
    
    logger.info("\n" + "=" * 70)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Scenarios Processed: {n_successful}/{n_scenarios}")
    logger.info(f"Failures: {n_failed}")
    logger.info(f"Results saved to: {batch_output_path}")
    
    # Log key aggregate metrics
    stats = batch_results['detailed_statistics']
    
    if stats['baseline_vs_gt_ade']['n'] > 0:
        logger.info(f"\n=== SANITY CHECK (Baseline vs Ground Truth) ===")
        logger.info(f"  ADE: mean={stats['baseline_vs_gt_ade']['mean']:.2f}m, "
                   f"median={stats['baseline_vs_gt_ade']['median']:.2f}m, "
                   f"Q1-Q3=[{stats['baseline_vs_gt_ade']['q1']:.2f}, {stats['baseline_vs_gt_ade']['q3']:.2f}]")
        logger.info(f"  FDE: mean={stats['baseline_vs_gt_fde']['mean']:.2f}m, "
                   f"median={stats['baseline_vs_gt_fde']['median']:.2f}m, "
                   f"Q1-Q3=[{stats['baseline_vs_gt_fde']['q1']:.2f}, {stats['baseline_vs_gt_fde']['q3']:.2f}]")
    
    realism = stats.get('advbmt_realism', {})
    if realism.get('sfde_avg', {}).get('n', 0) > 0:
        logger.info(f"\n=== ADV-BMT REALISM METRICS ===")
        logger.info(f"  SFDE_avg: mean={realism['sfde_avg']['mean']:.2f}, median={realism['sfde_avg']['median']:.2f}")
        logger.info(f"  SFDE_min: mean={realism['sfde_min']['mean']:.2f}, median={realism['sfde_min']['median']:.2f}")
        logger.info(f"  SADE_avg: mean={realism['sade_avg']['mean']:.2f}, median={realism['sade_avg']['median']:.2f}")
        logger.info(f"  SADE_min: mean={realism['sade_min']['mean']:.2f}, median={realism['sade_min']['median']:.2f}")
        logger.info(f"  VehColl_avg: mean={realism['veh_coll_avg']['mean']:.3f}")
        logger.info(f"  JSD_velocity: mean={realism['jsd_velocity']['mean']:.3f}")
    
    diversity = stats.get('advbmt_diversity', {})
    if diversity.get('fdd', {}).get('n', 0) > 0:
        logger.info(f"\n=== ADV-BMT DIVERSITY METRICS ===")
        logger.info(f"  FDD: mean={diversity['fdd']['mean']:.2f}, median={diversity['fdd']['median']:.2f}")
        logger.info(f"  SDD: mean={diversity['sdd']['mean']:.2f}, median={diversity['sdd']['median']:.2f}")
        logger.info(f"  ADD: mean={diversity['add']['mean']:.2f}, median={diversity['add']['median']:.2f}")
    
    logger.info("=" * 70)
    
    return 0 if n_failed == 0 else 1


def _save_aggregate_summary(output_dir: Path, batch_results: Dict):
    """Save comprehensive human-readable aggregate summary with quartiles."""
    summary_path = output_dir / "batch_summary.txt"
    
    agg = batch_results['aggregate_metrics']
    stats = batch_results.get('detailed_statistics', {})
    
    def format_stats(s: Dict, unit: str = "") -> str:
        """Format statistics with mean, median, and quartiles."""
        if not s or s.get('n', 0) == 0:
            return "N/A"
        return (f"mean={s['mean']:.2f}{unit}, median={s['median']:.2f}{unit}, "
                f"Q1-Q3=[{s['q1']:.2f}, {s['q3']:.2f}], range=[{s['min']:.2f}, {s['max']:.2f}] (n={s['n']})")
    
    def format_stats_short(s: Dict) -> str:
        """Format statistics in short form for tables."""
        if not s or s.get('n', 0) == 0:
            return "N/A"
        return f"{s['mean']:.2f} ({s['median']:.2f})"
    
    lines = [
        "=" * 80,
        "COUNTERBMT BATCH PROCESSING SUMMARY",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PROCESSING RESULTS:",
        "-" * 50,
        f"  Scenarios Processed: {agg.get('n_scenarios_processed', 0)}",
        f"  Scenarios Failed:    {agg.get('n_scenarios_failed', 0)}",
        f"  Success Rate:        {agg.get('success_rate', 0)*100:.1f}%",
        f"  Total Interventions: {agg.get('total_interventions', 0)}",
        f"  Effective Interventions: {agg.get('total_effective_interventions', 0)}",
        "",
    ]
    
    # Sanity Check Section
    lines.extend([
        "=" * 80,
        "SANITY CHECK: BMT BASELINE vs GROUND TRUTH",
        "=" * 80,
        "(Lower is better - measures how well BMT predicts without intervention)",
        "",
    ])
    
    ade_stats = stats.get('baseline_vs_gt_ade', {})
    fde_stats = stats.get('baseline_vs_gt_fde', {})
    
    if ade_stats.get('n', 0) > 0:
        lines.extend([
            f"  ADE (m): {format_stats(ade_stats, 'm')}",
            f"  FDE (m): {format_stats(fde_stats, 'm')}",
            "",
            f"  Baseline Distance: {format_stats(stats.get('baseline_distance', {}), 'm')}",
            f"  GT Distance:       {format_stats(stats.get('gt_distance', {}), 'm')}",
        ])
    else:
        lines.append("  No sanity check data available.")
    
    # ADV-BMT Realism Metrics Section
    lines.extend([
        "",
        "=" * 80,
        "ADV-BMT REALISM METRICS (per intervention)",
        "=" * 80,
        "(Compare with ADV-BMT paper Table (a))",
        "",
        "  Format: mean (median)",
        "",
    ])
    
    realism = stats.get('advbmt_realism', {})
    if realism.get('sfde_avg', {}).get('n', 0) > 0:
        # Create a table-like format
        lines.extend([
            "  Metric          Mean    Median    Q1      Q3      Min     Max     N",
            "  " + "-" * 70,
        ])
        
        metric_names = {
            'sfde_avg': 'SFDE_avg',
            'sfde_min': 'SFDE_min', 
            'sade_avg': 'SADE_avg',
            'sade_min': 'SADE_min',
            'veh_coll_avg': 'VehColl_avg',
            'veh_coll_min': 'VehColl_min',
            'jsd_velocity': 'JSD_velocity',
            'jsd_ttc': 'JSD_TTC',
        }
        
        for key, name in metric_names.items():
            s = realism.get(key, {})
            if s.get('n', 0) > 0:
                lines.append(f"  {name:14s}  {s['mean']:6.2f}  {s['median']:6.2f}  {s['q1']:6.2f}  {s['q3']:6.2f}  {s['min']:6.2f}  {s['max']:6.2f}  {s['n']:4d}")
    else:
        lines.append("  No realism metrics available.")
    
    # ADV-BMT Diversity Metrics Section
    lines.extend([
        "",
        "=" * 80,
        "ADV-BMT DIVERSITY METRICS (per intervention)",
        "=" * 80,
        "(Compare with ADV-BMT paper Table (b))",
        "",
    ])
    
    diversity = stats.get('advbmt_diversity', {})
    if diversity.get('fdd', {}).get('n', 0) > 0:
        lines.extend([
            "  Metric          Mean    Median    Q1      Q3      Min     Max     N",
            "  " + "-" * 70,
        ])
        
        for key, name in [('fdd', 'FDD'), ('sdd', 'SDD'), ('add', 'ADD')]:
            s = diversity.get(key, {})
            if s.get('n', 0) > 0:
                lines.append(f"  {name:14s}  {s['mean']:6.2f}  {s['median']:6.2f}  {s['q1']:6.2f}  {s['q3']:6.2f}  {s['min']:6.2f}  {s['max']:6.2f}  {s['n']:4d}")
    else:
        lines.append("  No diversity metrics available.")
    
    # Intervention Effectiveness
    lines.extend([
        "",
        "=" * 80,
        "INTERVENTION EFFECTIVENESS",
        "=" * 80,
    ])
    
    cf_change = stats.get('cf_change_percent', {})
    if cf_change.get('n', 0) > 0:
        lines.extend([
            f"  Distance Change (%): {format_stats(cf_change, '%')}",
        ])
    
    # Comparison with ADV-BMT paper
    lines.extend([
        "",
        "=" * 80,
        "COMPARISON WITH ADV-BMT PAPER (Forward method)",
        "=" * 80,
        "",
        "  ADV-BMT Paper Values (Forward):",
        "    SFDE_avg: 3.52    SFDE_min: 2.35",
        "    SADE_avg: 2.39    SADE_min: 1.98",
        "    VehColl_min: 0.03  VehColl_avg: 0.05",
        "    JSD_velocity: 0.23  JSD_TTC: 0.30",
        "    FDD: 10.78  ADD: 4.40",
        "",
        "  CounterBMT Results (this batch):",
    ])
    
    if realism.get('sfde_avg', {}).get('n', 0) > 0:
        lines.extend([
            f"    SFDE_avg: {realism['sfde_avg']['mean']:.2f}    SFDE_min: {realism['sfde_min']['mean']:.2f}",
            f"    SADE_avg: {realism['sade_avg']['mean']:.2f}    SADE_min: {realism['sade_min']['mean']:.2f}",
            f"    VehColl_min: {realism['veh_coll_min']['mean']:.2f}  VehColl_avg: {realism['veh_coll_avg']['mean']:.2f}",
            f"    JSD_velocity: {realism['jsd_velocity']['mean']:.2f}  JSD_TTC: {realism['jsd_ttc']['mean']:.2f}",
        ])
    if diversity.get('fdd', {}).get('n', 0) > 0:
        lines.extend([
            f"    FDD: {diversity['fdd']['mean']:.2f}  ADD: {diversity['add']['mean']:.2f}",
        ])
    
    # Per-scenario breakdown (condensed)
    lines.extend([
        "",
        "=" * 80,
        "PER-SCENARIO BREAKDOWN",
        "=" * 80,
    ])
    
    for scenario in batch_results['scenarios'][:20]:  # Limit to first 20
        status_icon = "✓" if scenario['status'] == 'success' else "✗"
        lines.append(f"  {status_icon} [{scenario['index']:4d}] {scenario['scenario_id']}: "
                    f"{scenario.get('baseline_distance', 0):.1f}m, "
                    f"{scenario.get('n_interventions', 0)} interventions")
    
    if len(batch_results['scenarios']) > 20:
        lines.append(f"  ... and {len(batch_results['scenarios']) - 20} more scenarios")
    
    # Failures section
    if batch_results['failures']:
        lines.extend([
            "",
            "FAILURES:",
            "-" * 50,
        ])
        for fail in batch_results['failures'][:10]:
            lines.append(f"  ✗ [{fail['index']:4d}] {fail['error'][:60]}...")
        if len(batch_results['failures']) > 10:
            lines.append(f"  ... and {len(batch_results['failures']) - 10} more failures")
    
    lines.extend([
        "",
        "=" * 80,
    ])
    
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved batch summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="CounterBMT Full Pipeline")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to ScenarioNet converted Waymo data")
    parser.add_argument("--output-dir", type=str, default="./outputs/counterbmt_pipeline",
                        help="Output directory")
    parser.add_argument("--scenario-index", type=int, default=0,
                        help="Scenario index to process")
    parser.add_argument("--bmt-checkpoint", type=str, default=None,
                        help="Path to BMT checkpoint (skip BMT if not provided)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock clients (no API calls)")
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Number of frames for VLM")
    parser.add_argument("--max-interventions", type=int, default=5,
                        help="Maximum interventions to process")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Samples per intervention")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature")
    
    # Batch processing arguments
    parser.add_argument("--batch", action="store_true",
                        help="Run batch processing on multiple scenarios")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Starting scenario index for batch mode")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Ending scenario index for batch mode (exclusive)")
    parser.add_argument("--num-scenarios", type=int, default=10,
                        help="Number of scenarios to process in batch mode (if end-index not specified)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue processing even if a scenario fails")
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        return run_batch_pipeline(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir),
            start_index=args.start_index,
            end_index=args.end_index,
            num_scenarios=args.num_scenarios,
            bmt_checkpoint=args.bmt_checkpoint,
            use_mock=args.mock,
            num_frames=args.num_frames,
            max_interventions=args.max_interventions,
            n_samples=args.n_samples,
            temperature=args.temperature,
            continue_on_error=args.continue_on_error,
        )
    
    # Single scenario mode
    results = run_full_pipeline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        scenario_index=args.scenario_index,
        bmt_checkpoint=args.bmt_checkpoint,
        use_mock=args.mock,
        num_frames=args.num_frames,
        max_interventions=args.max_interventions,
        n_samples=args.n_samples,
        temperature=args.temperature,
    )
    
    return 0 if results.get('status') == 'success' else 1


if __name__ == "__main__":
    sys.exit(main())