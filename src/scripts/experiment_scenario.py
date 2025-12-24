#!/usr/bin/env python3
"""
CounterBMT Scenario Experimentation Script

Interactive script for experimenting with individual scenarios and custom interventions.
Allows manual specification of interventions like lane changes, speed adjustments, etc.

Usage:
    # List available scenarios
    python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 --list
    
    # Run with a specific scenario and intervention
    python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \
        --scenario-index 10 \
        --intervention "speed:reduce:0.5" \
        --output-dir outputs/experiments
    
    # Run with multiple interventions
    python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \
        --scenario-id abc123 \
        --intervention "speed:reduce:0.5" \
        --intervention "lane:left" \
        --n-samples 5
    
    # Interactive mode
    python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 --interactive

Intervention Format:
    speed:reduce:FACTOR     - Reduce speed by factor (0.5 = half speed)
    speed:increase:FACTOR   - Increase speed by factor (1.5 = 50% faster)
    speed:set:VALUE         - Set speed to specific value (m/s)
    lane:left               - Bias toward left lane change
    lane:right              - Bias toward right lane change
    lane:stay               - Bias toward staying in lane
    maneuver:stop           - Bias toward stopping
    maneuver:accelerate     - Bias toward acceleration
    maneuver:decelerate     - Bias toward deceleration
    maneuver:turn_left      - Bias toward left turn
    maneuver:turn_right     - Bias toward right turn
    yaw:left:STRENGTH       - Bias yaw rate left (0.0-1.0)
    yaw:right:STRENGTH      - Bias yaw rate right (0.0-1.0)

Time-Based Interventions:
    Append @START or @START-END to apply intervention during specific PREDICTION timesteps.
    
    IMPORTANT: BMT predicts ~19 timesteps (0-18) at 0.5s intervals = ~9.5 seconds of future.
    The timesteps here refer to PREDICTION steps, not total trajectory frames!
    
    Use 's' suffix for seconds (at 0.5s per step, so 1s = 2 timesteps, 5s = 10 timesteps).
    
    Examples:
        "lane:left@5"            - Start left lane change at prediction step 5 (~2.5s)
        "lane:left@5-10"         - Lane change between steps 5-10 (~2.5-5s)
        "speed:reduce:0.5@2s"    - Slow down starting at 2 seconds (step 4)
        "maneuver:stop@3s-5s"    - Apply stopping bias from 3-5 seconds (steps 6-10)
        "lane:left@0-8,lane:right@10-18"  - Chain: left early, then right later
    
    Without @time, the intervention applies to ALL prediction timesteps (0-18).

Author: CounterBMT Project
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Adv-BMT"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Intervention Definitions
# =============================================================================

INTERVENTION_PRESETS = {
    # Speed interventions
    'slow_down': {
        'type': 'speed',
        'action': 'reduce',
        'factor': 0.5,
        'description': 'Reduce speed to 50%'
    },
    'speed_up': {
        'type': 'speed',
        'action': 'increase',
        'factor': 1.5,
        'description': 'Increase speed by 50%'
    },
    'stop': {
        'type': 'speed',
        'action': 'set',
        'value': 0.0,
        'description': 'Come to a stop'
    },
    
    # Lane change interventions
    'change_left': {
        'type': 'lane',
        'direction': 'left',
        'strength': 0.8,
        'description': 'Change to left lane'
    },
    'change_right': {
        'type': 'lane',
        'direction': 'right',
        'strength': 0.8,
        'description': 'Change to right lane'
    },
    'stay_in_lane': {
        'type': 'lane',
        'direction': 'stay',
        'strength': 0.9,
        'description': 'Stay in current lane'
    },
    
    # Maneuver interventions
    'turn_left': {
        'type': 'maneuver',
        'maneuver': 'turn_left',
        'strength': 0.8,
        'description': 'Make a left turn'
    },
    'turn_right': {
        'type': 'maneuver',
        'maneuver': 'turn_right',
        'strength': 0.8,
        'description': 'Make a right turn'
    },
    'go_straight': {
        'type': 'maneuver',
        'maneuver': 'straight',
        'strength': 0.8,
        'description': 'Continue straight'
    },
    'aggressive_accel': {
        'type': 'acceleration',
        'bias': 'positive',
        'strength': 0.7,
        'description': 'Accelerate aggressively'
    },
    'hard_brake': {
        'type': 'acceleration',
        'bias': 'negative',
        'strength': 0.9,
        'description': 'Brake hard'
    },
}


def parse_time_spec(time_str: str, steps_per_second: float = 2.0) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a time specification string into start/end PREDICTION timesteps.
    
    Args:
        time_str: Time spec like "5", "5-10", "2s", "2s-4s"
        steps_per_second: Prediction steps per second (BMT uses 0.5s per step = 2 steps/sec)
    
    Returns:
        (start_timestep, end_timestep) - end is None if only start specified
        
    Note: BMT predicts ~19 timesteps (0-18), so max valid timestep is 18.
    """
    if not time_str:
        return None, None
    
    # Check for range
    if '-' in time_str:
        start_str, end_str = time_str.split('-', 1)
    else:
        start_str = time_str
        end_str = None
    
    def parse_single(s: str) -> int:
        s = s.strip()
        if s.endswith('s'):
            # Seconds - convert to prediction timesteps
            # BMT uses 0.5s per step, so 2 steps per second
            seconds = float(s[:-1])
            return int(seconds * steps_per_second)
        else:
            return int(s)
    
    start = parse_single(start_str)
    end = parse_single(end_str) if end_str else None
    
    # Clamp to valid BMT prediction range (0-18)
    MAX_BMT_STEP = 18
    if start is not None:
        start = min(start, MAX_BMT_STEP)
    if end is not None:
        end = min(end, MAX_BMT_STEP + 1)  # +1 because end is exclusive in range()
    
    return start, end


def parse_intervention_string(intervention_str: str) -> Dict[str, Any]:
    """
    Parse an intervention string into a structured dict.
    
    Format: TYPE:ACTION:VALUE[@TIME] or TYPE:ACTION[@TIME]
    
    Examples:
        "speed:reduce:0.5" -> reduce speed by 50% (all timesteps)
        "lane:left" -> bias toward left lane change (all timesteps)
        "lane:left@20" -> lane change starting at timestep 20
        "lane:left@20-40" -> lane change between timesteps 20-40
        "speed:reduce:0.5@2s" -> slow down starting at 2 seconds
        "maneuver:stop@3s-5s" -> stop between 3-5 seconds
    """
    # Split off time specification if present
    if '@' in intervention_str:
        base_str, time_str = intervention_str.rsplit('@', 1)
        start_time, end_time = parse_time_spec(time_str)
    else:
        base_str = intervention_str
        start_time, end_time = None, None
    
    parts = base_str.strip().lower().split(':')
    
    if len(parts) < 2:
        raise ValueError(f"Invalid intervention format: {intervention_str}")
    
    int_type = parts[0]
    action = parts[1]
    value = float(parts[2]) if len(parts) > 2 else None
    
    intervention = {
        'type': int_type,
        'action': action,
        'raw_string': intervention_str,
        'start_timestep': start_time,
        'end_timestep': end_time,
    }
    
    # Build time description suffix
    time_desc = ""
    if start_time is not None:
        if end_time is not None:
            time_desc = f" (t={start_time}-{end_time})"
        else:
            time_desc = f" (from t={start_time})"
    
    if int_type == 'speed':
        if action == 'reduce':
            intervention['factor'] = value or 0.5
            intervention['description'] = f"Reduce speed to {(value or 0.5)*100:.0f}%{time_desc}"
        elif action == 'increase':
            intervention['factor'] = value or 1.5
            intervention['description'] = f"Increase speed by {((value or 1.5)-1)*100:.0f}%{time_desc}"
        elif action == 'set':
            intervention['value'] = value or 0.0
            intervention['description'] = f"Set speed to {value or 0.0:.1f} m/s{time_desc}"
    
    elif int_type == 'lane':
        intervention['direction'] = action
        intervention['strength'] = value or 0.8
        intervention['description'] = f"Lane change: {action}{time_desc}"
    
    elif int_type == 'maneuver':
        intervention['maneuver'] = action
        intervention['strength'] = value or 0.8
        intervention['description'] = f"Maneuver: {action}{time_desc}"
    
    elif int_type == 'yaw':
        intervention['direction'] = action
        intervention['strength'] = value or 0.5
        intervention['description'] = f"Yaw bias: {action} (strength={value or 0.5:.1f}){time_desc}"
    
    elif int_type == 'acceleration' or int_type == 'accel':
        intervention['type'] = 'acceleration'
        intervention['bias'] = action
        intervention['strength'] = value or 0.5
        intervention['description'] = f"Acceleration: {action}{time_desc}"
    
    else:
        intervention['description'] = f"{intervention_str}{time_desc}"
    
    return intervention


# =============================================================================
# Time-Aware Token Biasing
# =============================================================================

class TimedTokenBias:
    """A token bias that only applies during specific timesteps."""
    
    def __init__(
        self,
        token_ids: List[int],
        bias_value: float,
        start_timestep: Optional[int] = None,
        end_timestep: Optional[int] = None,
        description: str = "",
    ):
        self.token_ids = token_ids
        self.bias_value = bias_value
        self.start_timestep = start_timestep  # None = from beginning
        self.end_timestep = end_timestep      # None = until end
        self.description = description
    
    def is_active(self, timestep: int) -> bool:
        """Check if this bias should be active at the given timestep."""
        if self.start_timestep is not None and timestep < self.start_timestep:
            return False
        if self.end_timestep is not None and timestep > self.end_timestep:
            return False
        return True


class TimedTokenSampler:
    """
    A token sampler that applies different biases at different timesteps.
    
    Compatible with BMT's set_biased_sampler() interface.
    """
    
    def __init__(self, timed_biases: List[TimedTokenBias]):
        self.timed_biases = timed_biases
        # Build timestep lookup for efficient bias application
        # BMT predicts ~19 timesteps (0-18), not 91
        self._timestep_biases: Dict[int, List[TimedTokenBias]] = {}
        for bias in timed_biases:
            # If no timing specified, apply to all BMT prediction timesteps (0-18)
            start = bias.start_timestep if bias.start_timestep is not None else 0
            end = bias.end_timestep if bias.end_timestep is not None else 19
            for t in range(start, end):
                if t not in self._timestep_biases:
                    self._timestep_biases[t] = []
                self._timestep_biases[t].append(bias)
    
    def reset(self):
        """Reset for a new rollout (no-op for this implementation)."""
        pass
    
    def apply_bias(self, logits, timestep: int, agent_id: Optional[int] = None):
        """
        Apply time-appropriate biases to logits.
        
        This signature matches BMT's expected BiasedTokenSampler interface.
        
        Args:
            logits: Token logits tensor
            timestep: Current prediction timestep
            agent_id: Optional agent ID for agent-specific biasing
            
        Returns:
            Modified logits
        """
        import torch
        
        # Get active biases for this timestep
        active_biases = self._timestep_biases.get(timestep, [])
        
        
        if not active_biases:
            return logits
        
        # Clone logits to avoid in-place modification issues
        logits = logits.clone()
        
        n_modified = 0
        for bias in active_biases:
            if bias.token_ids:
                for token_id in bias.token_ids:
                    if token_id < logits.shape[-1]:
                        logits[..., token_id] += bias.bias_value
                        n_modified += 1
        
        return logits
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the timed biases."""
        lines = []
        for bias in self.timed_biases:
            time_range = ""
            if bias.start_timestep is not None or bias.end_timestep is not None:
                start = bias.start_timestep if bias.start_timestep is not None else 0
                end = bias.end_timestep if bias.end_timestep is not None else "end"
                time_range = f" [t={start}-{end}]"
            lines.append(f"  - {bias.description}{time_range}")
        return "\n".join(lines)


def intervention_to_timed_bias(
    intervention: Dict[str, Any], 
    token_space,
    base_bias: float = 5.0
) -> List[TimedTokenBias]:
    """
    Convert an intervention specification to timed token biases.
    
    Args:
        intervention: Parsed intervention dict (with optional start_timestep/end_timestep)
        token_space: BMT MotionTokenSpace instance
        base_bias: Base bias strength (default: 5.0, try 8-10 for stronger effects)
        
    Returns:
        List of TimedTokenBias objects
    """
    biases = []
    int_type = intervention['type']
    
    # Get timing info
    start_t = intervention.get('start_timestep')
    end_t = intervention.get('end_timestep')
    
    def add_bias(token_ids, bias_value, description):
        """Helper to add a timed bias."""
        if token_ids:
            biases.append(TimedTokenBias(
                token_ids=token_ids,
                bias_value=bias_value,
                start_timestep=start_t,
                end_timestep=end_t,
                description=description
            ))
    
    # Helper to get tokens using the correct MotionTokenSpace API
    def get_tokens(behavior: str) -> List[int]:
        """Get tokens for a behavior using token_space.get_tokens_by_behavior()."""
        return token_space.get_tokens_by_behavior(behavior)
    
    # Use provided base_bias strength
    BASE_BIAS = base_bias
    
    if int_type == 'speed':
        action = intervention['action']
        if action == 'reduce' or action == 'set':
            factor = intervention.get('factor', 0.5)
            decel_tokens = get_tokens('decelerate')
            add_bias(decel_tokens, BASE_BIAS * (1 - factor) * 2, intervention['description'])
        elif action == 'increase':
            factor = intervention.get('factor', 1.5)
            accel_tokens = get_tokens('accelerate')
            add_bias(accel_tokens, BASE_BIAS * (factor - 1) * 2, intervention['description'])
    
    elif int_type == 'lane':
        direction = intervention['direction']
        strength = intervention.get('strength', 0.8)
        
        if direction == 'left':
            add_bias(get_tokens('turn_left'), BASE_BIAS * strength, intervention['description'])
        elif direction == 'right':
            add_bias(get_tokens('turn_right'), BASE_BIAS * strength, intervention['description'])
        elif direction == 'stay':
            add_bias(get_tokens('straight'), BASE_BIAS * strength, intervention['description'])
    
    elif int_type == 'maneuver':
        maneuver = intervention['maneuver']
        strength = intervention.get('strength', 0.8)
        
        if maneuver in ['stop', 'decelerate', 'brake']:
            add_bias(get_tokens('decelerate'), BASE_BIAS * strength * 1.5, intervention['description'])
        elif maneuver in ['accelerate', 'speed_up']:
            add_bias(get_tokens('accelerate'), BASE_BIAS * strength, intervention['description'])
        elif maneuver in ['turn_left', 'left_turn', 'left']:
            add_bias(get_tokens('turn_left'), BASE_BIAS * strength * 1.5, intervention['description'])
        elif maneuver in ['turn_right', 'right_turn', 'right']:
            add_bias(get_tokens('turn_right'), BASE_BIAS * strength * 1.5, intervention['description'])
        elif maneuver in ['straight', 'continue']:
            add_bias(get_tokens('straight'), BASE_BIAS * strength, intervention['description'])
    
    elif int_type == 'yaw':
        direction = intervention['direction']
        strength = intervention.get('strength', 0.5)
        
        if direction == 'left':
            tokens = get_tokens('turn_left')
        else:
            tokens = get_tokens('turn_right')
        add_bias(tokens, BASE_BIAS * strength * 1.5, intervention['description'])
    
    elif int_type == 'acceleration':
        bias_dir = intervention.get('bias', 'positive')
        strength = intervention.get('strength', 0.5)
        
        if bias_dir == 'positive':
            tokens = get_tokens('accelerate')
        else:
            tokens = get_tokens('decelerate')
        add_bias(tokens, BASE_BIAS * strength, intervention['description'])
    
    return biases


# Keep old function for backwards compatibility
def intervention_to_token_bias(intervention: Dict[str, Any], token_space) -> List:
    """Legacy wrapper - converts to non-timed biases."""
    from counter_bmt.bmt_generator import TokenBias
    
    timed_biases = intervention_to_timed_bias(intervention, token_space)
    return [TokenBias(
        token_ids=tb.token_ids,
        bias_value=tb.bias_value,
        description=tb.description
    ) for tb in timed_biases]


# =============================================================================
# Scenario Loading
# =============================================================================

def list_scenarios(data_dir: Path, limit: int = 20):
    """List available scenarios in the dataset."""
    summary_path = data_dir / "dataset_summary.pkl"
    
    if not summary_path.exists():
        logger.error(f"Dataset summary not found: {summary_path}")
        return
    
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
    
    print(f"\nAvailable Scenarios in {data_dir}:")
    print("=" * 60)
    print(f"Total: {len(summary)} scenarios\n")
    
    for i, (filename, metadata) in enumerate(list(summary.items())[:limit]):
        scenario_id = filename.replace('sd_waymo_v1.2_', '').replace('.pkl', '')
        print(f"  [{i:3d}] {scenario_id}")
    
    if len(summary) > limit:
        print(f"\n  ... and {len(summary) - limit} more")
    
    print(f"\nUse --scenario-index N or --scenario-id ID to select a scenario")


def load_scenario(data_dir: Path, scenario_index: int = None, scenario_id: str = None):
    """Load a scenario by index or ID."""
    summary_path = data_dir / "dataset_summary.pkl"
    
    with open(summary_path, 'rb') as f:
        summary = pickle.load(f)
    
    scenario_files = list(summary.keys())
    
    if scenario_id:
        # Find by ID
        matching = [f for f in scenario_files if scenario_id in f]
        if not matching:
            raise ValueError(f"Scenario ID '{scenario_id}' not found")
        scenario_file = matching[0]
        scenario_index = scenario_files.index(scenario_file)
    elif scenario_index is not None:
        if scenario_index >= len(scenario_files):
            raise ValueError(f"Scenario index {scenario_index} out of range (max: {len(scenario_files)-1})")
        scenario_file = scenario_files[scenario_index]
    else:
        raise ValueError("Must specify scenario_index or scenario_id")
    
    file_path = data_dir / scenario_file
    with open(file_path, 'rb') as f:
        scenario_data = pickle.load(f)
    
    extracted_id = scenario_file.replace('sd_waymo_v1.2_', '').replace('.pkl', '')
    
    return {
        'raw_data': scenario_data,
        'scenario_id': extracted_id,
        'scenario_index': scenario_index,
        'file_path': str(file_path),
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def run_experiment(
    data_dir: Path,
    scenario_index: int = None,
    scenario_id: str = None,
    interventions: List[str] = None,
    preset: str = None,
    output_dir: Path = None,
    bmt_checkpoint: str = None,
    n_samples: int = 3,
    temperature: float = 1.0,
    combine_interventions: bool = False,
    bias_strength: float = 5.0,
) -> Dict:
    """
    Run an experiment with specified interventions.
    
    Returns dict with results and paths to exported scenarios.
    """
    import torch
    
    # Default checkpoint path - resolve to absolute path
    if bmt_checkpoint is None:
        bmt_checkpoint = "src/Adv-BMT/bmt/ckpt/last.ckpt"
    
    # Convert to absolute path to avoid path resolution issues
    bmt_checkpoint = str(Path(bmt_checkpoint).resolve())
    
    # Load scenario
    logger.info(f"Loading scenario...")
    scenario_data = load_scenario(data_dir, scenario_index, scenario_id)
    scenario_id = scenario_data['scenario_id']
    logger.info(f"Loaded scenario: {scenario_id}")
    
    # Parse interventions
    parsed_interventions = []
    
    if preset and preset in INTERVENTION_PRESETS:
        parsed_interventions.append(INTERVENTION_PRESETS[preset])
        logger.info(f"Using preset intervention: {preset}")
    
    if interventions:
        for int_str in interventions:
            # Support comma-separated chained interventions
            # e.g., "lane:left@0-30,lane:right@60-91"
            if ',' in int_str and '@' in int_str:
                # This is a chained intervention - parse each part
                for part in int_str.split(','):
                    part = part.strip()
                    if part:
                        parsed = parse_intervention_string(part)
                        parsed_interventions.append(parsed)
                        logger.info(f"Parsed chained intervention: {parsed['description']}")
            else:
                parsed = parse_intervention_string(int_str)
                parsed_interventions.append(parsed)
                logger.info(f"Parsed intervention: {parsed['description']}")
    
    if not parsed_interventions:
        logger.warning("No interventions specified - generating baseline only")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path("outputs/experiments")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / f"{scenario_id}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    replay_dir = experiment_dir / "replay_scenarios"
    replay_dir.mkdir(parents=True, exist_ok=True)
    
    # Load BMT model (same approach as run_full_pipeline.py)
    logger.info(f"Loading BMT model from {bmt_checkpoint}...")
    
    from bmt.utils import utils as bmt_utils
    
    pl_model = bmt_utils.get_model(checkpoint_path=bmt_checkpoint)
    pl_model = pl_model.eval()
    config = pl_model.config
    tokenizer = pl_model.model.tokenizer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pl_model = pl_model.to(device)
    
    logger.info(f"Model loaded on {device}")
    
    # Preprocess scenario
    from bmt.dataset.preprocessor import preprocess_scenario_description_for_motionlm
    
    preprocessed = preprocess_scenario_description_for_motionlm(
        scenario=scenario_data['raw_data'],
        config=config,
        in_evaluation=True,
        keep_all_data=True,
        tokenizer=tokenizer
    )
    
    scenario_data['preprocessed'] = preprocessed
    
    # Get map_center for coordinate transform
    map_center = preprocessed.get('metadata/map_center')
    
    # Prepare input
    from scripts.run_full_pipeline import _prepare_bmt_input
    input_dict = _prepare_bmt_input(preprocessed, device, config)
    
    # Generate baseline
    logger.info("\nGenerating baseline trajectory...")
    
    baseline_output = pl_model.model.autoregressive_rollout(
        input_dict,
        num_decode_steps=None,
        sampling_method=config.SAMPLING.SAMPLING_METHOD,
        temperature=temperature,
    )
    
    # CRITICAL: Detokenize to convert sampled tokens to actual positions!
    flip_heading = getattr(config.TOKENIZATION, 'FLIP_WRONG_HEADING', True)
    baseline_output = tokenizer.detokenize(
        baseline_output,
        detokenizing_gt=False,
        backward_prediction=False,
        flip_wrong_heading=flip_heading,
    )
    
    # Extract baseline trajectory
    if 'decoder/reconstructed_position' in baseline_output:
        baseline_traj = baseline_output['decoder/reconstructed_position']
    elif 'decoder/agent_position' in baseline_output:
        baseline_traj = baseline_output['decoder/agent_position']
    else:
        logger.error("No position data in baseline output")
        return None
    
    if hasattr(baseline_traj, 'cpu'):
        baseline_traj = baseline_traj.cpu().numpy()
    
    if baseline_traj.ndim == 4:
        baseline_traj = baseline_traj[0, :, 0, :2]
    elif baseline_traj.ndim == 3:
        baseline_traj = baseline_traj[:, 0, :2]
    
    logger.info(f"Baseline trajectory: {baseline_traj.shape}")
    
    # Generate counterfactuals with interventions
    counterfactual_results = []
    
    if parsed_interventions:
        from counter_bmt.bmt_generator import MotionTokenSpace
        from bmt.models.motionlm import set_biased_sampler, reset_timestep
        
        # BMT uses 33x33 = 1089 tokens (not the N_ACC_BINS from config)
        # The config values are for internal tokenization, not the final vocabulary
        token_space = MotionTokenSpace(
            n_acc_bins=33,  # Fixed: BMT vocabulary is 33x33=1089
            n_yaw_bins=33,
        )
        
        # Decide whether to combine all interventions or run separately
        if combine_interventions and len(parsed_interventions) > 1:
            # Combine all interventions into a single sampler
            logger.info(f"\nCombining {len(parsed_interventions)} interventions into single trajectory...")
            
            all_timed_biases = []
            combined_description = []
            
            for intervention in parsed_interventions:
                timed_biases = intervention_to_timed_bias(intervention, token_space, bias_strength)
                all_timed_biases.extend(timed_biases)
                combined_description.append(intervention['description'])
            
            if all_timed_biases:
                sampler = TimedTokenSampler(all_timed_biases)
                logger.info(f"Combined interventions:\n{sampler.get_summary()}")
                logger.info(f"Sampler has {len(sampler._timestep_biases)} timesteps with biases")
                
                # Generate samples with combined interventions
                samples = []
                for sample_idx in range(n_samples):
                    reset_timestep()
                    set_biased_sampler(sampler)
                    
                    cf_output = pl_model.model.autoregressive_rollout(
                        input_dict,
                        num_decode_steps=None,
                        sampling_method='topp',
                        temperature=temperature,
                    )
                    
                    # Detokenize to convert sampled tokens to actual positions!
                    cf_output = tokenizer.detokenize(
                        cf_output,
                        detokenizing_gt=False,
                        backward_prediction=False,
                        flip_wrong_heading=flip_heading,
                    )
                    
                    if 'decoder/reconstructed_position' in cf_output:
                        cf_traj = cf_output['decoder/reconstructed_position']
                    else:
                        cf_traj = cf_output['decoder/agent_position']
                    
                    if hasattr(cf_traj, 'cpu'):
                        cf_traj = cf_traj.cpu().numpy()
                    
                    if cf_traj.ndim == 4:
                        cf_traj = cf_traj[0, :, 0, :2]
                    elif cf_traj.ndim == 3:
                        cf_traj = cf_traj[:, 0, :2]
                    
                    samples.append(cf_traj)
                
                counterfactual_results.append({
                    'intervention': {
                        'description': ' + '.join(combined_description),
                        'combined': True,
                        'components': parsed_interventions,
                    },
                    'trajectories': samples,
                })
                
                logger.info(f"Generated {len(samples)} samples with combined interventions")
        else:
            # Run each intervention separately
            for intervention in parsed_interventions:
                logger.info(f"\nGenerating counterfactual: {intervention['description']}...")
                
                # Check for time-based intervention
                has_timing = intervention.get('start_timestep') is not None or intervention.get('end_timestep') is not None
                
                # Convert intervention to timed token biases
                timed_biases = intervention_to_timed_bias(intervention, token_space, bias_strength)
                
                if not timed_biases:
                    logger.warning(f"Could not create token biases for: {intervention['description']}")
                    continue
                
                # Create time-aware sampler
                sampler = TimedTokenSampler(timed_biases)
                
                if has_timing:
                    logger.info(f"  Time-based: {sampler.get_summary()}")
                
                # Generate multiple samples
                samples = []
                for sample_idx in range(n_samples):
                    # Reset timestep and set biased sampler
                    reset_timestep()
                    set_biased_sampler(sampler)
                    
                    cf_output = pl_model.model.autoregressive_rollout(
                        input_dict,
                        num_decode_steps=None,
                        sampling_method='topp',
                        temperature=temperature,
                    )
                    
                    # Detokenize to convert sampled tokens to actual positions!
                    cf_output = tokenizer.detokenize(
                        cf_output,
                        detokenizing_gt=False,
                        backward_prediction=False,
                        flip_wrong_heading=flip_heading,
                    )
                    
                    # Extract trajectory
                    if 'decoder/reconstructed_position' in cf_output:
                        cf_traj = cf_output['decoder/reconstructed_position']
                    else:
                        cf_traj = cf_output['decoder/agent_position']
                    
                    if hasattr(cf_traj, 'cpu'):
                        cf_traj = cf_traj.cpu().numpy()
                    
                    if cf_traj.ndim == 4:
                        cf_traj = cf_traj[0, :, 0, :2]
                    elif cf_traj.ndim == 3:
                        cf_traj = cf_traj[:, 0, :2]
                    
                    samples.append(cf_traj)
                
                counterfactual_results.append({
                    'intervention': intervention,
                    'trajectories': samples,
                })
                
                logger.info(f"Generated {len(samples)} samples")
                
                # Compare baseline vs counterfactual
                if samples and baseline_traj is not None:
                    cf_sample = np.array(samples[0])
                    baseline_np = np.array(baseline_traj)
                    max_diff = np.abs(cf_sample - baseline_np).max()
                    logger.info(f"  Max trajectory deviation: {max_diff:.2f}m")
        
        # Cleanup - reset biased sampler
        set_biased_sampler(None)
    
    # Export scenarios for replay
    logger.info("\nExporting scenarios for replay...")
    
    from counter_bmt.scenario_export import (
        export_trajectory_only,
        export_ground_truth_scenario,
        create_replay_script,
    )
    
    exported_paths = []
    
    # Export ground truth
    gt_path = replay_dir / f"sd_experiment_1.0_{scenario_id}_00_GROUND_TRUTH.pkl"
    if export_ground_truth_scenario(scenario_data['file_path'], gt_path):
        exported_paths.append(gt_path)
        logger.info(f"Exported ground truth")
    
    # Export baseline
    baseline_path = replay_dir / f"sd_experiment_1.0_{scenario_id}_01_BASELINE.pkl"
    if export_trajectory_only(
        trajectory=baseline_traj,
        original_scenario=scenario_data['raw_data'],
        output_path=baseline_path,
        intervention_name="BASELINE (no intervention)",
        original_file_path=scenario_data['file_path'],
        map_center=map_center,
    ):
        exported_paths.append(baseline_path)
        logger.info(f"Exported baseline")
    
    # Export counterfactuals
    cf_idx = 2
    for cf_result in counterfactual_results:
        intervention = cf_result['intervention']
        
        for sample_idx, traj in enumerate(cf_result['trajectories']):
            int_name = intervention['description'].replace(' ', '_')[:30]
            cf_path = replay_dir / f"sd_experiment_1.0_{scenario_id}_{cf_idx:02d}_{int_name}_s{sample_idx}.pkl"
            
            if export_trajectory_only(
                trajectory=traj,
                original_scenario=scenario_data['raw_data'],
                output_path=cf_path,
                intervention_name=f"{intervention['description']} (sample {sample_idx})",
                original_file_path=scenario_data['file_path'],
                map_center=map_center,
            ):
                exported_paths.append(cf_path)
            
            cf_idx += 1
    
    # Create replay script
    if exported_paths:
        replay_script = experiment_dir / "replay_scenarios.py"
        create_replay_script(exported_paths, replay_script)
        logger.info(f"Created replay script: {replay_script}")
    
    # Save experiment config
    experiment_config = {
        'scenario_id': scenario_id,
        'scenario_index': scenario_data['scenario_index'],
        'interventions': [i.get('description', str(i)) for i in parsed_interventions],
        'n_samples': n_samples,
        'temperature': temperature,
        'timestamp': timestamp,
        'exported_scenarios': [str(p) for p in exported_paths],
    }
    
    config_path = experiment_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nScenario: {scenario_id}")
    print(f"Interventions: {len(parsed_interventions)}")
    print(f"Total trajectories: {len(exported_paths)}")
    print(f"\nOutput directory: {experiment_dir}")
    print(f"\nTo replay scenarios:")
    print(f"  python -m scenarionet.sim -d {replay_dir} --render 2D")
    print("=" * 60)
    
    return {
        'experiment_dir': str(experiment_dir),
        'scenario_id': scenario_id,
        'baseline_trajectory': baseline_traj.tolist(),
        'counterfactual_results': counterfactual_results,
        'exported_paths': [str(p) for p in exported_paths],
    }


def interactive_mode(data_dir: Path, bmt_checkpoint: str = None):
    """Run in interactive mode for quick experimentation."""
    print("\n" + "=" * 60)
    print("CounterBMT Interactive Experiment Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  list              - List available scenarios")
    print("  load <index>      - Load scenario by index")
    print("  load id:<id>      - Load scenario by ID")
    print("  presets           - Show intervention presets")
    print("  run <intervention> - Run with intervention")
    print("  preset <name>     - Run with preset (e.g., 'slow_down')")
    print("  help              - Show this help")
    print("  quit              - Exit")
    print("\nIntervention Examples:")
    print("  run speed:reduce:0.5       - Slow down to 50% (all timesteps)")
    print("  run lane:left@20           - Lane change left starting at t=20")
    print("  run lane:left@20-40        - Lane change left during t=20-40")
    print("  run maneuver:stop@2s       - Start stopping at 2 seconds")
    print("  run maneuver:stop@2s-4s    - Stop between 2-4 seconds")
    print("=" * 60)
    
    current_scenario = None
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if action == 'quit' or action == 'exit':
                print("Goodbye!")
                break
            
            elif action == 'help':
                print("\nCommands:")
                print("  list              - List available scenarios")
                print("  load <index>      - Load scenario by index")
                print("  presets           - Show intervention presets")
                print("  run <intervention> - Run experiment")
                print("  preset <name>     - Run with preset")
            
            elif action == 'list':
                list_scenarios(data_dir)
            
            elif action == 'presets':
                print("\nAvailable Presets:")
                for name, preset in INTERVENTION_PRESETS.items():
                    print(f"  {name:20s} - {preset['description']}")
            
            elif action == 'load':
                if args.startswith('id:'):
                    scenario_id = args[3:]
                    current_scenario = load_scenario(data_dir, scenario_id=scenario_id)
                else:
                    idx = int(args)
                    current_scenario = load_scenario(data_dir, scenario_index=idx)
                print(f"Loaded scenario: {current_scenario['scenario_id']}")
            
            elif action == 'run':
                if not current_scenario:
                    print("No scenario loaded. Use 'load <index>' first.")
                    continue
                
                interventions = [args] if args else []
                run_experiment(
                    data_dir=data_dir,
                    scenario_index=current_scenario['scenario_index'],
                    interventions=interventions,
                    bmt_checkpoint=bmt_checkpoint,
                )
            
            elif action == 'preset':
                if not current_scenario:
                    print("No scenario loaded. Use 'load <index>' first.")
                    continue
                
                if args not in INTERVENTION_PRESETS:
                    print(f"Unknown preset: {args}")
                    print("Use 'presets' to see available presets")
                    continue
                
                run_experiment(
                    data_dir=data_dir,
                    scenario_index=current_scenario['scenario_index'],
                    preset=args,
                    bmt_checkpoint=bmt_checkpoint,
                )
            
            else:
                print(f"Unknown command: {action}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CounterBMT Scenario Experimentation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List scenarios
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 --list

  # Run with speed reduction (all timesteps)
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \\
      --scenario-index 10 --intervention "speed:reduce:0.5"

  # Run with preset
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \\
      --scenario-index 10 --preset slow_down

  # Time-based intervention: lane change starting at timestep 20
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \\
      --scenario-index 10 --intervention "lane:left@20"

  # Time-based: slow down between 2-4 seconds
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \\
      --scenario-index 10 --intervention "speed:reduce:0.5@2s-4s"

  # Chained interventions: go left then go right (combined into single trajectory)
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 \\
      --scenario-index 10 --intervention "lane:left@0-30,lane:right@60-91" --combine

  # Interactive mode
  python experiment_scenario.py --data-dir data/scenarionet_waymo_training_500 --interactive
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to ScenarioNet dataset')
    parser.add_argument('--scenario-index', type=int, default=None,
                        help='Scenario index to use')
    parser.add_argument('--scenario-id', type=str, default=None,
                        help='Scenario ID to use')
    parser.add_argument('--intervention', '-i', action='append', default=[],
                        help='Intervention to apply (can specify multiple)')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        help='Use a preset intervention')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--bmt-checkpoint', type=str, default=None,
                        help='Path to BMT checkpoint')
    parser.add_argument('--n-samples', type=int, default=3,
                        help='Number of samples per intervention')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--combine', action='store_true',
                        help='Combine multiple interventions into a single trajectory '
                             '(useful for time-based sequences like lane:left@0-30,lane:right@60-91)')
    parser.add_argument('--bias-strength', type=float, default=5.0,
                        help='Base bias strength for interventions (default: 5.0, try 8-10 for stronger effects)')
    parser.add_argument('--list', action='store_true',
                        help='List available scenarios')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--presets', action='store_true',
                        help='Show available presets')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    if args.list:
        list_scenarios(data_dir)
        return
    
    if args.presets:
        print("\nAvailable Intervention Presets:")
        print("=" * 60)
        for name, preset in INTERVENTION_PRESETS.items():
            print(f"  {name:20s} - {preset['description']}")
        return
    
    if args.interactive:
        interactive_mode(data_dir, args.bmt_checkpoint)
        return
    
    if args.scenario_index is None and args.scenario_id is None:
        parser.print_help()
        print("\nError: Must specify --scenario-index, --scenario-id, --list, or --interactive")
        sys.exit(1)
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_experiment(
        data_dir=data_dir,
        scenario_index=args.scenario_index,
        scenario_id=args.scenario_id,
        interventions=args.intervention,
        preset=args.preset,
        output_dir=output_dir,
        bmt_checkpoint=args.bmt_checkpoint,
        n_samples=args.n_samples,
        temperature=args.temperature,
        combine_interventions=args.combine,
        bias_strength=args.bias_strength,
    )


if __name__ == "__main__":
    main()

