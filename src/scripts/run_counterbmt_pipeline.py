"""
CounterBMT Pipeline - BMT Generation Functions

This module provides the complete BMT generation integration for run_counterbmt_pipeline.py.

All token bias classes are now imported from counter_bmt.bmt_generator
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


def run_bmt_generation(
    scenario_data: Dict,
    compiled_interventions: List[Dict],
    bmt_checkpoint: str,
    config_path: Optional[str] = None,
    output_dir: str = "./outputs",
    n_samples: int = 3,
    temperature: Optional[float] = None,
    save_trajectories: bool = True
) -> Dict:
    """
    Run BMT trajectory generation with counterfactual biases.
    
    Args:
        scenario_data: Dict containing raw scenario data and metadata
        compiled_interventions: List of compiled intervention dicts, each with:
            - intervention: original intervention dict
            - token_biases: list of bias dicts with token_ids, bias_value, timestep_range
            - effect_prediction: predicted effect from DAG
        bmt_checkpoint: Path to BMT model checkpoint
        config_path: Optional path to config override
        output_dir: Directory to save results
        n_samples: Number of samples per intervention
        temperature: Override sampling temperature
        save_trajectories: Whether to save trajectory files
        
    Returns:
        Dict with generation results
    """
    import torch
    from bmt.utils import utils as bmt_utils
    from bmt.models.motionlm import set_biased_sampler, reset_timestep
    from bmt.utils.utils import numpy_to_torch
    
    # Import from counter_bmt.bmt_generator (updated import path)
    from counter_bmt.bmt_generator import BiasedTokenSampler, TokenBias, MotionTokenSpace, InterventionCompiler
    
    logger.info("=" * 60)
    logger.info("BMT Counterfactual Generation")
    logger.info("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load BMT Model
    # =========================================================================
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
    
    # =========================================================================
    # Step 2: Prepare Input Data
    # =========================================================================
    logger.info("Preparing scenario data...")
    
    # Get raw data from scenario_data dict
    raw_data = scenario_data.get('raw_data', scenario_data)
    scenario_id = scenario_data.get('scenario_id', 'unknown')
    
    # Prepare input dict for BMT
    input_dict = _prepare_bmt_input(raw_data, device, config)
    
    logger.info(f"  Scenario: {scenario_id}")
    logger.info(f"  Input keys: {len(input_dict)} tensors")
    
    # =========================================================================
    # Step 3: Generate Baseline Trajectory
    # =========================================================================
    logger.info("Generating baseline trajectory...")
    
    baseline_output = _run_single_generation(
        pl_model=pl_model,
        config=config,
        tokenizer=tokenizer,
        input_dict=input_dict,
        device=device,
        use_bias=False,
        sampler=None,
        temperature=temperature
    )
    
    baseline_traj = _extract_ego_trajectory(baseline_output)
    logger.info(f"  Baseline trajectory shape: {baseline_traj.shape if baseline_traj is not None else 'None'}")
    
    # =========================================================================
    # Step 4: Generate Counterfactual Trajectories
    # =========================================================================
    results = {
        'status': 'success',
        'scenario_id': scenario_id,
        'baseline': {
            'trajectory': baseline_traj.tolist() if baseline_traj is not None else None,
            'output_keys': list(baseline_output.keys()) if baseline_output else []
        },
        'counterfactuals': []
    }
    
    for i, comp_int in enumerate(compiled_interventions):
        intervention = comp_int.get('intervention', {})
        var_id = intervention.get('variable', f'intervention_{i}')
        new_val = intervention.get('value', 'unknown')
        
        logger.info(f"\n[Intervention {i+1}/{len(compiled_interventions)}]")
        logger.info(f"  do({var_id} = {new_val})")
        
        # Reconstruct TokenBias objects from serialized dict
        token_bias_data = comp_int.get('token_biases', [])
        token_biases = []
        
        for b in token_bias_data:
            # Handle both full token_ids and truncated versions
            if 'token_ids' in b and len(b['token_ids']) > 0:
                token_ids = b['token_ids']
            else:
                # If token_ids were truncated, we need to recompile
                logger.warning(f"  Token IDs missing, recompiling intervention...")
                compiler = InterventionCompiler(MotionTokenSpace())
                recompiled = compiler.compile_from_dag_intervention(intervention)
                token_biases = recompiled
                break
            
            token_biases.append(TokenBias(
                token_ids=token_ids,
                bias_value=b['bias_value'],
                timestep_range=tuple(b['timestep_range'])
            ))
        
        if not token_biases:
            logger.warning(f"  No token biases for intervention, skipping")
            continue
        
        logger.info(f"  {len(token_biases)} bias groups")
        
        # Create sampler
        sampler = BiasedTokenSampler(token_biases)
        
        # Generate samples
        cf_trajectories = []
        for sample_idx in range(n_samples):
            logger.info(f"  Generating sample {sample_idx + 1}/{n_samples}...")
            
            cf_output = _run_single_generation(
                pl_model=pl_model,
                config=config,
                tokenizer=tokenizer,
                input_dict=input_dict,
                device=device,
                use_bias=True,
                sampler=sampler,
                temperature=temperature
            )
            
            cf_traj = _extract_ego_trajectory(cf_output)
            if cf_traj is not None:
                cf_trajectories.append(cf_traj.tolist())
        
        # Compare with baseline
        comparison = None
        if baseline_traj is not None and len(cf_trajectories) > 0:
            cf_traj_arr = np.array(cf_trajectories[0])
            comparison = _compare_trajectories(baseline_traj, cf_traj_arr)
            logger.info(f"  Travel reduction: {comparison.get('travel_reduction_ratio', 1.0):.2%}")
        
        # Store result
        cf_result = {
            'intervention': intervention,
            'effect_prediction': comp_int.get('effect_prediction', {}),
            'n_samples': len(cf_trajectories),
            'trajectories': cf_trajectories,
            'comparison': comparison
        }
        results['counterfactuals'].append(cf_result)
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    if save_trajectories:
        results_path = output_path / "generation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nSaved results to: {results_path}")
    
    # Cleanup
    set_biased_sampler(None)
    
    logger.info("\n" + "=" * 60)
    logger.info("Generation Complete")
    logger.info(f"  Baseline: 1 trajectory")
    logger.info(f"  Counterfactuals: {len(results['counterfactuals'])} interventions")
    logger.info("=" * 60)
    
    return results


def _prepare_bmt_input(raw_data: Dict, device: str, config) -> Dict:
    """
    Prepare raw scenario data for BMT input format.
    
    Handles numpy -> torch conversion, batch dimension, double precision.
    """
    import torch
    from bmt.utils.utils import numpy_to_torch
    
    # Deep copy
    input_dict = {}
    for k, v in raw_data.items():
        if isinstance(v, np.ndarray) and 'track_name' not in k:
            input_dict[k] = v.copy()
        elif isinstance(v, torch.Tensor):
            input_dict[k] = v.clone()
        elif isinstance(v, str):
            input_dict[k] = v
        else:
            input_dict[k] = copy.deepcopy(v)
    
    # Convert to torch
    input_dict = numpy_to_torch(input_dict, device=device)
    
    # Convert to double precision for specific keys
    double_keys = [
        "decoder/agent_position", 
        "decoder/agent_heading", 
        "decoder/agent_velocity",
        "decoder/reconstructed_position",
        "decoder/reconstructed_heading", 
        "decoder/reconstructed_velocity",
        "decoder/agent_shape",
        "decoder/current_agent_shape",
        "decoder/current_agent_position",
        "encoder/current_agent_position",
        "encoder/current_agent_velocity",
    ]
    
    for k in double_keys:
        if k in input_dict and isinstance(input_dict[k], torch.Tensor):
            if input_dict[k].dtype in [torch.float32, torch.float16]:
                input_dict[k] = input_dict[k].double()
    
    # Add batch dimension if needed (check decoder/agent_position shape)
    needs_batch = False
    if 'decoder/agent_position' in input_dict:
        pos = input_dict['decoder/agent_position']
        if isinstance(pos, torch.Tensor) and pos.dim() == 3:  # [T, N, D]
            needs_batch = True
    
    if needs_batch:
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                input_dict[k] = v.unsqueeze(0)
    
    # Set evaluation flag
    input_dict["in_evaluation"] = torch.tensor([True], dtype=torch.bool).to(device)
    
    return input_dict


def _run_single_generation(
    pl_model,
    config,
    tokenizer,
    input_dict: Dict,
    device: str,
    use_bias: bool,
    sampler,
    temperature: Optional[float] = None
) -> Dict:
    """
    Run a single BMT generation pass.
    """
    import torch
    from bmt.models.motionlm import set_biased_sampler, reset_timestep
    
    # Deep copy input (torch tensors need clone)
    input_copy = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_copy[k] = v.clone()
        elif isinstance(v, str):
            input_copy[k] = v
        else:
            input_copy[k] = copy.deepcopy(v)
    
    # Setup biased sampling
    if use_bias and sampler is not None:
        reset_timestep()
        set_biased_sampler(sampler)
    else:
        set_biased_sampler(None)
    
    # Tokenize
    tok_data, _ = tokenizer.tokenize(input_copy, backward_prediction=False)
    input_copy.update(tok_data)
    
    # Set temperature
    sampling_temp = temperature if temperature is not None else config.SAMPLING.TEMPERATURE
    
    # Generate
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
    
    # Cleanup
    set_biased_sampler(None)
    
    return output_dict


def _extract_ego_trajectory(output_dict: Dict) -> Optional[np.ndarray]:
    """
    Extract ego vehicle trajectory from BMT output.
    
    Returns [T, 2] array of (x, y) positions.
    """
    import torch
    
    if output_dict is None:
        return None
    
    # Priority: reconstructed > raw positions
    position_keys = [
        'decoder/reconstructed_position',
        'decoder/agent_position',
    ]
    
    positions = None
    for key in position_keys:
        if key in output_dict:
            positions = output_dict[key]
            break
    
    if positions is None:
        logger.warning("No position data in output")
        return None
    
    # Convert to numpy
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    
    # Handle different shapes
    if positions.ndim == 4:  # [B, T, N, D]
        ego_pos = positions[0, :, 0, :2]  # First batch, ego agent, xy
    elif positions.ndim == 3:  # [T, N, D]
        ego_pos = positions[:, 0, :2]
    else:
        logger.warning(f"Unexpected position shape: {positions.shape}")
        return None
    
    return ego_pos


def _compare_trajectories(baseline: np.ndarray, counterfactual: np.ndarray) -> Dict:
    """
    Compare baseline and counterfactual trajectories.
    
    Both inputs should be [T, 2] arrays.
    """
    T = min(len(baseline), len(counterfactual))
    b = baseline[:T]
    c = counterfactual[:T]
    
    # Displacement difference per timestep
    diff = np.linalg.norm(c - b, axis=1)
    
    # Total travel distance
    b_travel = np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1))
    c_travel = np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))
    
    # Final position displacement from start
    b_final = np.linalg.norm(b[-1] - b[0])
    c_final = np.linalg.norm(c[-1] - c[0])
    
    return {
        'max_displacement_diff': float(diff.max()),
        'mean_displacement_diff': float(diff.mean()),
        'final_displacement_diff': float(np.linalg.norm(c[-1] - b[-1])),
        'baseline_travel': float(b_travel),
        'counterfactual_travel': float(c_travel),
        'travel_reduction_ratio': float(c_travel / max(b_travel, 0.01)),
        'baseline_final_displacement': float(b_final),
        'counterfactual_final_displacement': float(c_final),
    }


# =============================================================================
# Scenario Loading Helpers
# =============================================================================

def load_scenario_from_scenarionet(
    data_dir: str, 
    scenario_index: int
) -> Dict:
    """
    Load scenario data from ScenarioNet format for BMT input.
    
    Args:
        data_dir: Path to converted ScenarioNet data
        scenario_index: Index of scenario to load
        
    Returns:
        Dict with raw_data and metadata suitable for BMT
    """
    from metadrive.scenario.utils import read_dataset_summary
    import pickle
    
    logger.info(f"Loading scenario {scenario_index} from {data_dir}")
    
    # Read dataset summary
    summary_dict, summary_list, mapping = read_dataset_summary(data_dir)
    
    if scenario_index >= len(summary_list):
        raise ValueError(f"Scenario index {scenario_index} out of range (max: {len(summary_list)-1})")
    
    # Get scenario file path
    scenario_file = summary_list[scenario_index]
    folder = mapping.get(scenario_file, data_dir)
    file_path = Path(folder) / scenario_file
    
    # Load scenario description
    with open(file_path, 'rb') as f:
        scenario_desc = pickle.load(f)
    
    scenario_id = scenario_desc.get('id', f'scenario_{scenario_index}')
    
    logger.info(f"  Loaded: {scenario_id}")
    
    return {
        'raw_data': scenario_desc,
        'scenario_id': scenario_id,
        'data_dir': data_dir,
        'scenario_index': scenario_index,
        'file_path': str(file_path)
    }


def load_scenario_for_bmt(
    data_dir: str,
    scenario_index: int,
    config
) -> Dict:
    """
    Load and preprocess scenario for direct BMT input.
    
    This handles the full preprocessing pipeline.
    """
    from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
    import pickle
    from metadrive.scenario.utils import read_dataset_summary
    
    # Load raw scenario
    scenario_data = load_scenario_from_scenarionet(data_dir, scenario_index)
    raw_scenario = scenario_data['raw_data']
    
    # Preprocess for BMT
    preprocessed = preprocess_scenario_description_for_motionlm(
        scenario=raw_scenario,
        config=config,
        in_evaluation=True,
        keep_all_data=True,
        cache=None
    )
    
    # Add scenario metadata
    preprocessed['metadata/scenario_id'] = scenario_data['scenario_id']
    
    scenario_data['preprocessed'] = preprocessed
    
    return scenario_data


# =============================================================================
# Main Entry Point (for standalone usage)
# =============================================================================

def main():
    """
    Main entry point for running the CounterBMT pipeline.
    
    Usage:
        python run_counterbmt_pipeline.py \\
            --data-dir src/exp_converted \\
            --scenario-index 0 \\
            --bmt-checkpoint models/checkpoint.ckpt \\
            --output-dir outputs/counterbmt_results
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="CounterBMT Pipeline")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to ScenarioNet data directory")
    parser.add_argument("--scenario-index", type=int, default=0,
                        help="Scenario index to process")
    parser.add_argument("--bmt-checkpoint", type=str, required=True,
                        help="Path to BMT checkpoint")
    parser.add_argument("--output-dir", type=str, default="./outputs/counterbmt",
                        help="Output directory")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples per intervention")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (None = use config)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import DAG components
    from counter_bmt.dag_constructor import GroundedDAGConstructor, GPT4oDAGClient, MockDAGClient
    from counter_bmt.vlm_extractor import VLMSafetyCriticalExtractor, MockGPT4oClient
    from counter_bmt.bmt_generator import InterventionCompiler, MotionTokenSpace
    
    logger.info("=" * 60)
    logger.info("CounterBMT Pipeline")
    logger.info("=" * 60)
    
    # Load scenario
    scenario_data = load_scenario_from_scenarionet(args.data_dir, args.scenario_index)
    
    # For demo, use mock DAG client
    # In production, use: client = GPT4oDAGClient()
    dag_client = MockDAGClient()
    constructor = GroundedDAGConstructor(dag_client)
    
    # Create mock features for demo
    mock_features = {
        "scenario_id": scenario_data['scenario_id'],
        "maneuvers": [
            {"type": "straight", "start_timestamp": 0.0, "description": "Initial straight"},
            {"type": "decelerate", "start_timestamp": 2.0, "description": "Slowing down"},
        ],
        "decisions": [
            {"type": "proceed_or_yield", "choice": "proceed", "timestamp": 1.0,
             "alternatives": ["proceed", "yield"], "description": "Chose to proceed"}
        ]
    }
    
    # Construct DAG
    dag = constructor.construct(mock_features, scenario_id=scenario_data['scenario_id'])
    logger.info(f"\n{dag.summary()}")
    
    # Get interventions
    interventions = dag.enumerate_interventions()
    logger.info(f"\nFound {len(interventions)} possible interventions")
    
    # Compile interventions
    compiler = InterventionCompiler(MotionTokenSpace())
    compiled = []
    
    for intv in interventions[:3]:  # Limit to first 3 for demo
        int_dict = {
            'variable': intv.variable_id,
            'value': intv.value,
            'original_value': intv.original_value,
            'description': intv.description
        }
        token_biases = compiler.compile_from_dag_intervention(int_dict)
        compiled.append({
            'intervention': int_dict,
            'token_biases': [b.to_dict() for b in token_biases],
            'effect_prediction': {}
        })
        logger.info(f"  Compiled: {intv.description}")
    
    # Run BMT generation
    results = run_bmt_generation(
        scenario_data=scenario_data,
        compiled_interventions=compiled,
        bmt_checkpoint=args.bmt_checkpoint,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        temperature=args.temperature
    )
    
    logger.info("\nPipeline complete!")
    return results


if __name__ == "__main__":
    main()