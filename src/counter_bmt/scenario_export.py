"""
Scenario Export Module for CounterBMT

Exports counterfactual trajectories in ScenarioNet/MetaDrive-compatible format
for replay and visualization.

The exported pickle files can be loaded directly by MetaDrive's ScenarioEnv
to replay the counterfactual scenarios.

Usage:
    from counter_bmt.scenario_export import export_counterfactual_scenario
    
    # Export a single counterfactual
    export_counterfactual_scenario(
        original_scenario=scenario_desc,
        bmt_output=output_dict,
        output_path="counterfactual_scenario.pkl",
        intervention_name="speed_reduction"
    )

Author: CounterBMT Project
"""

import copy
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


def export_counterfactual_scenario(
    original_scenario: Dict[str, Any],
    bmt_output: Dict[str, Any],
    output_path: Union[str, Path],
    intervention_name: str = "counterfactual",
    ego_only: bool = True,
) -> Path:
    """
    Export a counterfactual scenario in ScenarioNet-compatible format.
    
    Takes the original scenario description and overwrites the ego trajectory
    with the BMT-generated counterfactual trajectory.
    
    Args:
        original_scenario: Original ScenarioNet scenario description dict
        bmt_output: BMT model output containing reconstructed trajectories
        output_path: Path to save the pickle file
        intervention_name: Name of the intervention (for metadata)
        ego_only: If True, only overwrite ego trajectory; if False, overwrite all agents
        
    Returns:
        Path to the saved scenario file
    """
    output_path = Path(output_path)
    
    # Deep copy to avoid modifying original
    new_scenario = copy.deepcopy(original_scenario)
    
    # Get SDC (ego) track name
    sdc_id = new_scenario.get('metadata', {}).get('sdc_id')
    if sdc_id is None:
        logger.warning("No sdc_id found in scenario metadata")
        return None
    
    sdc_track_name = str(sdc_id)
    
    # Extract trajectories from BMT output
    if 'decoder/reconstructed_position' in bmt_output:
        positions = _to_numpy(bmt_output['decoder/reconstructed_position'])
        velocities = _to_numpy(bmt_output.get('decoder/reconstructed_velocity'))
        headings = _to_numpy(bmt_output.get('decoder/reconstructed_heading'))
        valid_mask = _to_numpy(bmt_output.get('decoder/reconstructed_valid_mask'))
    elif 'decoder/agent_position' in bmt_output:
        # Fallback to agent_position if reconstructed not available
        positions = _to_numpy(bmt_output['decoder/agent_position'])
        velocities = _to_numpy(bmt_output.get('decoder/agent_velocity'))
        headings = _to_numpy(bmt_output.get('decoder/agent_heading'))
        valid_mask = _to_numpy(bmt_output.get('decoder/agent_valid_mask'))
    else:
        logger.error("No position data found in BMT output")
        return None
    
    # Handle batch dimension
    if positions.ndim == 4:  # [B, T, N, D]
        positions = positions[0]
        if velocities is not None:
            velocities = velocities[0]
        if headings is not None:
            headings = headings[0]
        if valid_mask is not None:
            valid_mask = valid_mask[0]
    
    T, N = positions.shape[:2]
    
    # Get ego index (usually 0)
    ego_idx = 0
    if 'decoder/sdc_index' in bmt_output:
        ego_idx = int(_to_numpy(bmt_output['decoder/sdc_index']))
    
    # Overwrite ego trajectory
    if sdc_track_name in new_scenario.get('tracks', {}):
        track = new_scenario['tracks'][sdc_track_name]
        
        # Get original trajectory length - must match for MetaDrive compatibility
        orig_pos = np.array(track['state'].get('position', []))
        orig_len = len(orig_pos)
        
        # Extract ego trajectory
        ego_pos = positions[:, ego_idx, :2]  # (T, 2)
        
        # Pad or truncate to match original length
        if T < orig_len:
            # Pad by repeating last position
            pad_len = orig_len - T
            last_pos = ego_pos[-1:] if len(ego_pos) > 0 else np.zeros((1, 2))
            ego_pos = np.vstack([ego_pos, np.repeat(last_pos, pad_len, axis=0)])
            logger.info(f"Padded trajectory from {T} to {orig_len} timesteps")
        elif T > orig_len and orig_len > 0:
            # Truncate
            ego_pos = ego_pos[:orig_len]
            logger.info(f"Truncated trajectory from {T} to {orig_len} timesteps")
        
        final_len = len(ego_pos)
        
        # Preserve z-coordinate from original if exists
        if orig_len > 0 and orig_pos.shape[-1] == 3:
            z_coord = orig_pos[:final_len, 2:3]
            ego_pos_3d = np.concatenate([ego_pos, z_coord], axis=1)
            track['state']['position'] = ego_pos_3d.astype(np.float32)
        else:
            track['state']['position'] = ego_pos.astype(np.float32)
        
        # Update velocity - pad/truncate to match
        if velocities is not None:
            ego_vel = velocities[:, ego_idx]
            if ego_vel.ndim == 1:
                ego_vel = np.stack([ego_vel, np.zeros_like(ego_vel)], axis=-1)
            # Pad or truncate
            if len(ego_vel) < final_len:
                pad_len = final_len - len(ego_vel)
                ego_vel = np.vstack([ego_vel, np.repeat(ego_vel[-1:], pad_len, axis=0)])
            elif len(ego_vel) > final_len:
                ego_vel = ego_vel[:final_len]
            track['state']['velocity'] = ego_vel.astype(np.float32)
        
        # Update heading - pad/truncate to match
        if headings is not None:
            ego_heading = headings[:, ego_idx]
            if len(ego_heading) < final_len:
                pad_len = final_len - len(ego_heading)
                ego_heading = np.concatenate([ego_heading, np.repeat(ego_heading[-1:], pad_len)])
            elif len(ego_heading) > final_len:
                ego_heading = ego_heading[:final_len]
            track['state']['heading'] = ego_heading.astype(np.float32)
        
        # Update valid mask - preserve original or pad/truncate
        if valid_mask is not None:
            ego_valid = valid_mask[:, ego_idx]
            if len(ego_valid) < final_len:
                pad_len = final_len - len(ego_valid)
                ego_valid = np.concatenate([ego_valid, np.zeros(pad_len, dtype=bool)])
            elif len(ego_valid) > final_len:
                ego_valid = ego_valid[:final_len]
            track['state']['valid'] = ego_valid.astype(bool)
        else:
            # Preserve original valid mask
            pass
        
        logger.info(f"Updated ego trajectory: {final_len} timesteps (original: {orig_len})")
    else:
        logger.warning(f"SDC track '{sdc_track_name}' not found in scenario tracks")
    
    # Update metadata
    if 'metadata' not in new_scenario:
        new_scenario['metadata'] = {}
    new_scenario['metadata']['counterfactual'] = True
    new_scenario['metadata']['intervention'] = intervention_name
    new_scenario['metadata']['source_scenario'] = original_scenario.get('id', 'unknown')
    new_scenario['metadata']['dataset'] = 'waymo_counterfactual'
    
    # Optionally update scenario ID
    orig_id = new_scenario.get('id', 'unknown')
    new_scenario['id'] = f"{orig_id}_cf_{_sanitize_name(intervention_name)}"
    
    # Save to pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(new_scenario, f)
    
    logger.info(f"Exported counterfactual scenario to: {output_path}")
    return output_path


def export_all_counterfactuals(
    original_scenario: Dict[str, Any],
    counterfactual_outputs: List[Dict[str, Any]],
    intervention_names: List[str],
    output_dir: Union[str, Path],
    scenario_id: str,
) -> List[Path]:
    """
    Export all counterfactual scenarios from a pipeline run.
    
    Args:
        original_scenario: Original ScenarioNet scenario description
        counterfactual_outputs: List of BMT outputs for each intervention
        intervention_names: List of intervention names
        output_dir: Directory to save the pickle files
        scenario_id: Base scenario ID
        
    Returns:
        List of paths to saved scenario files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, (cf_output, int_name) in enumerate(zip(counterfactual_outputs, intervention_names)):
        safe_name = _sanitize_name(int_name)
        # Filename must start with 'sd_' to pass MetaDrive's is_scenario_file() validation
        output_path = output_dir / f"sd_counterfactual_1.0_{scenario_id}_cf_{i}_{safe_name[:30]}.pkl"
        
        path = export_counterfactual_scenario(
            original_scenario=original_scenario,
            bmt_output=cf_output,
            output_path=output_path,
            intervention_name=int_name,
        )
        
        if path:
            saved_paths.append(path)
    
    # Create dataset_summary.pkl for MetaDrive compatibility
    if saved_paths:
        create_dataset_summary(saved_paths, output_dir)
    
    logger.info(f"Exported {len(saved_paths)} counterfactual scenarios to {output_dir}")
    return saved_paths


def export_trajectory_only(
    trajectory: np.ndarray,
    original_scenario: Dict[str, Any],
    output_path: Union[str, Path],
    intervention_name: str = "counterfactual",
    original_file_path: Optional[Union[str, Path]] = None,
    map_center: Optional[np.ndarray] = None,
) -> Path:
    """
    Export a scenario with only the trajectory (position) data updated.
    
    This is useful when you have the trajectory but not the full BMT output.
    Velocity and heading will be computed from the trajectory.
    
    The trajectory will be:
    1. Transformed from BMT's local coordinate frame to global coordinates
       (using map_center if provided, following Adv-BMT's approach)
    2. Padded or truncated to match the original scenario length
    
    Args:
        trajectory: (T, 2) or (T, 3) numpy array of positions in BMT local frame
        original_scenario: Original ScenarioNet scenario description
        output_path: Path to save the pickle file
        intervention_name: Name of the intervention
        original_file_path: Optional path to original .pkl file for fresh loading
        map_center: Optional (3,) array - if provided, use Adv-BMT style transform
                    (just add map_center, no rotation)
        
    Returns:
        Path to the saved scenario file
    """
    output_path = Path(output_path)
    trajectory = np.asarray(trajectory).copy()
    
    # Load fresh original scenario from file if path provided (ensures unmodified data)
    if original_file_path and Path(original_file_path).exists():
        with open(original_file_path, 'rb') as f:
            fresh_original = pickle.load(f)
        new_scenario = copy.deepcopy(fresh_original)
        logger.info(f"Loaded fresh original scenario from {original_file_path}")
    else:
        # Fall back to provided scenario
        new_scenario = copy.deepcopy(original_scenario)
    
    # Get SDC track
    sdc_id = new_scenario.get('metadata', {}).get('sdc_id')
    if sdc_id is None:
        logger.warning("No sdc_id found in scenario metadata")
        return None
    
    sdc_track_name = str(sdc_id)
    
    if sdc_track_name not in new_scenario.get('tracks', {}):
        logger.warning(f"SDC track '{sdc_track_name}' not found")
        return None
    
    track = new_scenario['tracks'][sdc_track_name]
    
    # Get original trajectory - needed for coordinate transformation
    orig_pos = np.array(track['state'].get('position', []))
    orig_heading = np.array(track['state'].get('heading', []))
    orig_len = len(orig_pos)
    T = len(trajectory)
    
    if orig_len == 0:
        logger.warning("Original trajectory is empty")
        return None
    
    # =========================================================================
    # COORDINATE TRANSFORMATION: BMT local frame -> Global frame
    # =========================================================================
    # Following Adv-BMT's approach (see scenarionet_utils.py):
    # - BMT preprocessing subtracts map_center from positions
    # - To transform back: just add map_center (NO rotation needed)
    # 
    # If map_center is not provided, fall back to using original start position
    
    if map_center is not None:
        # Adv-BMT style: just add map_center (no rotation)
        map_center = np.asarray(map_center).flatten()
        trajectory_global = trajectory[:, :2] + map_center[:2]
        logger.info(f"Transformed using map_center: added ({map_center[0]:.1f}, {map_center[1]:.1f})")
    else:
        # Fallback: translate so trajectory starts at original start position
        orig_start_pos = orig_pos[0, :2]
        bmt_start_pos = trajectory[0, :2]
        offset = orig_start_pos - bmt_start_pos
        trajectory_global = trajectory[:, :2] + offset
        logger.info(f"Transformed using offset: ({offset[0]:.1f}, {offset[1]:.1f})")
    
    # Pad or truncate trajectory to match original length
    if T < orig_len:
        # Pad by repeating last position
        pad_len = orig_len - T
        last_pos = trajectory_global[-1:] if len(trajectory_global) > 0 else np.zeros((1, 2))
        trajectory_global = np.vstack([trajectory_global, np.repeat(last_pos, pad_len, axis=0)])
        logger.info(f"Padded trajectory from {T} to {orig_len} timesteps")
    elif T > orig_len:
        # Truncate
        trajectory_global = trajectory_global[:orig_len]
        logger.info(f"Truncated trajectory from {T} to {orig_len} timesteps")
    
    T = len(trajectory_global)  # Update T after padding/truncation
    
    # Add z-coordinate from original
    if orig_pos.shape[-1] == 3:
        z = orig_pos[:T, 2:3]
        trajectory_final = np.concatenate([trajectory_global, z], axis=1)
    else:
        trajectory_final = trajectory_global
    
    track['state']['position'] = trajectory_final.astype(np.float32)
    
    # Compute velocity from TRANSFORMED positions (finite difference)
    dt = 0.1  # 10Hz
    if T > 1:
        velocity = np.diff(trajectory_global, axis=0) / dt
        velocity = np.vstack([velocity, velocity[-1:]])  # Repeat last velocity
        track['state']['velocity'] = velocity.astype(np.float32)
    
    # Compute heading from velocity (in global frame)
    # This is tricky because arctan2 returns values in [-π, π], so a vehicle
    # moving in the -X direction can have heading near -π or +π depending
    # on small Y variations. Lane changes can cause the heading to "jump"
    # across the ±π boundary, which MetaDrive interprets as a 360° spin.
    if T > 1:
        heading_raw = np.arctan2(velocity[:, 1], velocity[:, 0])
        
        # Key insight: We want the heading to be CONTINUOUS throughout the trajectory.
        # np.unwrap() handles this by adding/subtracting 2π to prevent jumps > π.
        heading_unwrapped = np.unwrap(heading_raw)
        
        # Now anchor to the original heading at the START of the trajectory
        # This ensures the counterfactual starts with the same orientation
        if len(orig_heading) > 0:
            # Get the original starting heading
            orig_start_heading = float(orig_heading[0])
            
            # Calculate offset between our computed heading and original
            heading_offset = orig_start_heading - heading_unwrapped[0]
            
            # Apply offset to shift entire trajectory
            heading_shifted = heading_unwrapped + heading_offset
            
            # CRITICAL: Check if we're on the same "branch" as the original trajectory
            # If original goes from -179° to -175° (staying negative), we should too.
            # But if our computed heading went from +179° to +175° (positive branch),
            # we need to shift by 2π to match.
            
            # Use the original trajectory's average heading to determine which branch to use
            orig_mean_heading = float(np.mean(orig_heading[:min(10, len(orig_heading))]))
            our_start_heading = heading_shifted[0]
            
            # If our start heading is more than 90° different from original mean, we're on wrong branch
            branch_diff = our_start_heading - orig_mean_heading
            while branch_diff > np.pi:
                heading_shifted -= 2 * np.pi
                branch_diff = heading_shifted[0] - orig_mean_heading
                logger.debug(f"Shifted heading -2π (branch correction)")
            while branch_diff < -np.pi:
                heading_shifted += 2 * np.pi
                branch_diff = heading_shifted[0] - orig_mean_heading
                logger.debug(f"Shifted heading +2π (branch correction)")
            
            logger.debug(f"Heading: orig_start={np.degrees(orig_start_heading):.1f}°, "
                        f"computed_start={np.degrees(heading_raw[0]):.1f}°, "
                        f"final_start={np.degrees(heading_shifted[0]):.1f}°")
            
            heading_continuous = heading_shifted
        else:
            heading_continuous = heading_unwrapped
        
        track['state']['heading'] = heading_continuous.astype(np.float32)
        
        # Log heading summary for debugging
        heading_start = np.degrees(heading_continuous[0])
        heading_end = np.degrees(heading_continuous[-1])
        heading_min = np.degrees(np.min(heading_continuous))
        heading_max = np.degrees(np.max(heading_continuous))
        logger.info(f"Heading for '{intervention_name}': start={heading_start:.1f}°, end={heading_end:.1f}°, "
                   f"range=[{heading_min:.1f}°, {heading_max:.1f}°]")
    
    # Update valid mask - match original length
    # Mark padded timesteps as invalid if we padded
    orig_valid = track['state'].get('valid')
    if orig_valid is not None:
        valid = np.array(orig_valid, dtype=bool)
    else:
        valid = np.ones(T, dtype=bool)
    track['state']['valid'] = valid
    
    # Update metadata
    new_scenario['metadata']['counterfactual'] = True
    new_scenario['metadata']['intervention'] = intervention_name
    new_scenario['metadata']['source_scenario'] = original_scenario.get('id', 'unknown')
    
    orig_id = new_scenario.get('id', 'unknown')
    new_scenario['id'] = f"{orig_id}_cf_{_sanitize_name(intervention_name)}"
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(new_scenario, f)
    
    logger.info(f"Exported trajectory-only scenario to: {output_path}")
    return output_path


def create_dataset_summary(scenario_paths: List[Path], output_dir: Union[str, Path]) -> Path:
    """
    Create a dataset_summary.pkl file for MetaDrive compatibility.
    
    MetaDrive requires this file to index and load scenarios from a directory.
    Without it, ScenarioEnv cannot find valid scenarios to load.
    
    This function also scans the output_dir for any sd_*.pkl files that might
    not be in the scenario_paths list.
    
    Args:
        scenario_paths: List of paths to exported scenario files
        output_dir: Directory to save the summary file
        
    Returns:
        Path to the created dataset_summary.pkl
    """
    output_dir = Path(output_dir)
    summary_dict = {}
    
    # Collect all scenario files - both from provided paths AND by scanning directory
    all_scenario_files = set()
    
    # Add provided paths
    for path in scenario_paths:
        path = Path(path)
        if path.exists() and path.name.startswith('sd_') and path.name.endswith('.pkl'):
            all_scenario_files.add(path)
    
    # Also scan directory for any sd_*.pkl files we might have missed
    if output_dir.exists():
        for f in output_dir.glob('sd_*.pkl'):
            if f.name not in ['dataset_summary.pkl', 'dataset_mapping.pkl']:
                all_scenario_files.add(f)
    
    # Process all scenario files
    for scenario_path in sorted(all_scenario_files):
        if not scenario_path.exists():
            logger.warning(f"Scenario file not found: {scenario_path}")
            continue
            
        # Load scenario to get metadata
        try:
            with open(scenario_path, 'rb') as f:
                scenario = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load scenario {scenario_path}: {e}")
            continue
        
        filename = scenario_path.name
        metadata = scenario.get('metadata', {})
        
        # MetaDrive expects these fields in the summary
        # Copy all metadata and ensure required fields exist
        summary_entry = dict(metadata)
        summary_entry.update({
            'scenario_id': scenario.get('id', filename.replace('.pkl', '')),
            'sdc_id': metadata.get('sdc_id', ''),
            'dataset': metadata.get('dataset', 'counterfactual'),
            'counterfactual': metadata.get('counterfactual', True),
            'intervention': metadata.get('intervention', ''),
        })
        
        summary_dict[filename] = summary_entry
    
    if not summary_dict:
        logger.warning("No valid scenarios found to create summary")
        return None
    
    # Save dataset_summary.pkl
    summary_path = output_dir / "dataset_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary_dict, f)
    
    # Also create dataset_mapping.pkl (maps filenames to subdirectories, all empty for us)
    mapping_dict = {filename: "" for filename in summary_dict}
    mapping_path = output_dir / "dataset_mapping.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping_dict, f)
    
    logger.info(f"Created dataset summary with {len(summary_dict)} scenarios at {summary_path}")
    return summary_path


def export_ground_truth_scenario(
    original_file_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Export the original ground truth scenario for comparison with counterfactuals.
    
    This creates a copy of the original scenario with a clear "ground_truth" label
    so it can be played alongside counterfactuals in the simulator.
    
    Args:
        original_file_path: Path to the original scenario .pkl file
        output_path: Path to save the ground truth scenario
        
    Returns:
        Path to the saved ground truth scenario file
    """
    original_file_path = Path(original_file_path)
    output_path = Path(output_path)
    
    if not original_file_path.exists():
        logger.warning(f"Original scenario not found: {original_file_path}")
        return None
    
    # Load original scenario
    with open(original_file_path, 'rb') as f:
        scenario = pickle.load(f)
    
    # Make a copy and update metadata
    gt_scenario = copy.deepcopy(scenario)
    
    if 'metadata' not in gt_scenario:
        gt_scenario['metadata'] = {}
    
    gt_scenario['metadata']['counterfactual'] = False
    gt_scenario['metadata']['intervention'] = 'GROUND TRUTH (Original)'
    gt_scenario['metadata']['is_ground_truth'] = True
    
    # Update ID to indicate ground truth
    orig_id = gt_scenario.get('id', 'unknown')
    gt_scenario['id'] = f"{orig_id}_GROUND_TRUTH"
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(gt_scenario, f)
    
    logger.info(f"Exported ground truth scenario to: {output_path}")
    return output_path


def create_replay_script(
    scenario_paths: List[Path],
    output_path: Union[str, Path],
    data_dir: str = "./",
) -> Path:
    """
    Create a Python script to replay the exported scenarios.
    Also creates the dataset_summary.pkl needed by MetaDrive/ScenarioNet.
    
    Args:
        scenario_paths: List of paths to exported scenario files
        output_path: Path to save the replay script
        data_dir: Base data directory for MetaDrive
        
    Returns:
        Path to the replay script
    """
    output_path = Path(output_path)
    
    # Create dataset_summary.pkl for MetaDrive compatibility
    if scenario_paths:
        scenario_dir = Path(scenario_paths[0]).parent
        create_dataset_summary(scenario_paths, scenario_dir)
    
    # Get the replay scenarios directory path
    replay_dir = str(scenario_dir) if scenario_paths else "replay_scenarios"
    
    script = f'''#!/usr/bin/env python3
"""
CounterBMT Scenario Replay Script

Auto-generated script to replay counterfactual scenarios.

RECOMMENDED: Use ScenarioNet's 2D simulator (works reliably):
    python -m scenarionet.sim -d {replay_dir} --render 2D

Alternative: Use this script with MetaDrive's 3D renderer:
    python replay_scenarios.py --list           # List available scenarios
    python replay_scenarios.py --render         # Replay all with 3D rendering
    python replay_scenarios.py --scenario 0     # Replay specific scenario

Requirements:
    - ScenarioNet installed (for 2D): pip install scenarionet
    - MetaDrive installed (for 3D): pip install metadrive-simulator
"""

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

# Directory containing scenario files
SCENARIO_DIR = "{replay_dir}"

# Scenario files generated by CounterBMT
SCENARIO_FILES = [
'''
    
    for path in scenario_paths:
        script += f'    "{path}",\n'
    
    script += ''']

def load_scenario(path):
    """Load a scenario from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def replay_with_scenarionet_2d():
    """Replay using ScenarioNet's 2D simulator (recommended)."""
    print(f"\\nLaunching ScenarioNet 2D simulator...")
    print(f"  Directory: {SCENARIO_DIR}")
    subprocess.run([sys.executable, "-m", "scenarionet.sim", "-d", SCENARIO_DIR, "--render", "2D"])

def replay_scenario_metadrive(scenario_path, render=True):
    """Replay a single scenario in MetaDrive 3D."""
    try:
        from metadrive.envs.scenario_env import ScenarioEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    except ImportError:
        print("MetaDrive not installed. Install with: pip install metadrive-simulator")
        print("Or use ScenarioNet 2D: python -m scenarionet.sim -d", SCENARIO_DIR, "--render 2D")
        return
    
    scenario = load_scenario(scenario_path)
    scenario_id = scenario.get('id', 'unknown')
    
    print(f"\\nReplaying scenario: {scenario_id}")
    print(f"  Intervention: {scenario.get('metadata', {}).get('intervention', 'N/A')}")
    
    # Create environment
    env = ScenarioEnv({
        "use_render": render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": str(Path(scenario_path).parent),
        "num_scenarios": 1,
        "start_scenario_index": 0,
    })
    
    try:
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done:
            action = [0, 0]  # ReplayPolicy handles actual action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            if render:
                env.render()
        
        print(f"  Completed: {step} steps")
        
    finally:
        env.close()

def main():
    parser = argparse.ArgumentParser(description="Replay CounterBMT counterfactual scenarios")
    parser.add_argument("--scenario", type=int, default=None, help="Scenario index to replay (MetaDrive 3D)")
    parser.add_argument("--render", action="store_true", help="Enable 3D rendering (MetaDrive)")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--2d", dest="use_2d", action="store_true", help="Use ScenarioNet 2D renderer (recommended)")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available scenarios:")
        for i, path in enumerate(SCENARIO_FILES):
            scenario = load_scenario(path)
            print(f"  [{i}] {scenario.get('id', 'unknown')}")
            print(f"      Intervention: {scenario.get('metadata', {}).get('intervention', 'N/A')}")
        print(f"\\nTo visualize, run:")
        print(f"  python -m scenarionet.sim -d {SCENARIO_DIR} --render 2D")
        return
    
    if args.use_2d:
        replay_with_scenarionet_2d()
        return
    
    if args.scenario is not None:
        if 0 <= args.scenario < len(SCENARIO_FILES):
            replay_scenario_metadrive(SCENARIO_FILES[args.scenario], render=args.render)
        else:
            print(f"Invalid scenario index. Available: 0-{len(SCENARIO_FILES)-1}")
    elif args.render:
        # Replay all scenarios with MetaDrive
        for path in SCENARIO_FILES:
            replay_scenario_metadrive(path, render=True)
            input("Press Enter to continue to next scenario...")
    else:
        # Default: show help
        print("CounterBMT Scenario Replay")
        print("=" * 40)
        print(f"\\nScenario directory: {SCENARIO_DIR}")
        print(f"Number of scenarios: {len(SCENARIO_FILES)}")
        print("\\nRECOMMENDED - Use ScenarioNet 2D renderer:")
        print(f"  python -m scenarionet.sim -d {SCENARIO_DIR} --render 2D")
        print("\\nAlternative - Use MetaDrive 3D renderer:")
        print("  python replay_scenarios.py --render")
        print("  python replay_scenarios.py --scenario 0 --render")
        print("\\nList scenarios:")
        print("  python replay_scenarios.py --list")

if __name__ == "__main__":
    main()
'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(script)
    
    logger.info(f"Created replay script: {output_path}")
    return output_path


# =============================================================================
# Helper Functions
# =============================================================================

def _to_numpy(tensor) -> Optional[np.ndarray]:
    """Convert tensor to numpy array."""
    if tensor is None:
        return None
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().numpy()
    return np.asarray(tensor)


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in filenames."""
    # Replace problematic characters
    sanitized = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    sanitized = sanitized.replace(':', '_').replace('(', '').replace(')', '')
    sanitized = sanitized.replace(',', '_').replace('.', '_')
    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized[:50]  # Limit length

