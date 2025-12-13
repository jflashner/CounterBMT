"""
scenarionet_visualizer.py

ScenarioNet/MetaDrive-based visualization for Waymo scenarios.
Uses the official ScenarioEnv simulation environment for rendering.

Usage:
    from scenarionet_visualizer import ScenarioNetVisualizer, prepare_for_vlm
    
    # Generate frames for VLM analysis
    saved_images, trajectory, scenario_id = prepare_for_vlm(
        data_dir="./exp_converted",
        scenario_index=0,
        num_frames=8
    )

Author: CounterBMT Project
"""

import os
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import MetaDrive/ScenarioNet
try:
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    HAS_METADRIVE = True
except ImportError:
    HAS_METADRIVE = False
    logger.warning("MetaDrive not installed. Install with: pip install metadrive-simulator")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed. Install with: pip install pillow")


class ScenarioNetDatabase:
    """Database interface for ScenarioNet converted data."""
    
    def __init__(self, data_dir: str):
        """
        Initialize database from ScenarioNet data directory.
        
        Args:
            data_dir: Path to directory with converted scenarios
                     Structure can be:
                     - data_dir/*.pkl (flat)
                     - data_dir/_0/*.pkl, data_dir/_1/*.pkl (sharded)
        """
        self.data_dir = Path(data_dir)
        self.scenario_files = []
        self.scenario_ids = []
        self.summary = {}
        
        self._load_database()
    
    def _load_database(self):
        """Load scenario metadata from directory."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Load dataset summary if exists
        summary_path = self.data_dir / "dataset_summary.pkl"
        if summary_path.exists():
            with open(summary_path, 'rb') as f:
                self.summary = pickle.load(f)
        
        # Find all scenario files - check multiple patterns and locations
        # Pattern 1: Flat structure - sd_*.pkl in root
        # Pattern 2: Sharded structure - _0/sd_*.pkl, _1/sd_*.pkl, etc.
        
        scenario_files = []
        
        # Check root directory
        for f in self.data_dir.glob("sd_*.pkl"):
            scenario_files.append(f)
        
        # Check numbered subdirectories (_0, _1, etc.)
        for subdir in sorted(self.data_dir.glob("_*")):
            if subdir.is_dir():
                for f in subdir.glob("sd_*.pkl"):
                    scenario_files.append(f)
        
        # Sort by filename for consistent ordering
        self.scenario_files = sorted(scenario_files, key=lambda x: x.name)
        
        # Extract scenario IDs from filenames
        # Format: sd_waymo_v1.2_<scenario_id>.pkl or sd_<source>_<scenario_id>.pkl
        for f in self.scenario_files:
            # Get the last part after splitting by underscore (before .pkl)
            parts = f.stem.split('_')
            if len(parts) >= 2:
                # Last part is the scenario ID
                self.scenario_ids.append(parts[-1])
            else:
                self.scenario_ids.append(f.stem)
        
        logger.info(f"Loaded {len(self.scenario_files)} scenarios from {self.data_dir}")
        if len(self.scenario_files) > 0:
            logger.info(f"  First scenario: {self.scenario_files[0].name}")
            logger.info(f"  Last scenario: {self.scenario_files[-1].name}")
    
    def __len__(self) -> int:
        return len(self.scenario_files)
    
    def get_scenario_path(self, index: int) -> Path:
        """Get path to scenario file by index."""
        if index < 0 or index >= len(self.scenario_files):
            raise IndexError(f"Scenario index {index} out of range [0, {len(self.scenario_files)})")
        return self.scenario_files[index]
    
    def get_scenario_id(self, index: int) -> str:
        """Get scenario ID by index."""
        if index < 0 or index >= len(self.scenario_ids):
            raise IndexError(f"Scenario index {index} out of range")
        return self.scenario_ids[index]
    
    def load_scenario(self, index: int) -> Dict[str, Any]:
        """Load full scenario data by index."""
        path = self.get_scenario_path(index)
        with open(path, 'rb') as f:
            return pickle.load(f)


class ScenarioNetVisualizer:
    """
    Visualizer using MetaDrive's ScenarioEnv for rendering.
    
    This provides high-quality top-down renders of scenarios
    using the official simulation environment.
    """
    
    def __init__(
        self,
        data_dir: str,
        film_size: Tuple[int, int] = (1200, 1200),
        screen_size: Tuple[int, int] = (800, 800)
    ):
        """
        Initialize visualizer.
        
        Args:
            data_dir: Path to ScenarioNet data directory
            film_size: Size of the rendered film (higher = more detail)
            screen_size: Output image size
        """
        if not HAS_METADRIVE:
            raise ImportError("MetaDrive required: pip install metadrive-simulator")
        
        self.data_dir = str(Path(data_dir).absolute())  # Use absolute path
        self.film_size = film_size
        self.screen_size = screen_size
        self.env = None
        self.db = ScenarioNetDatabase(data_dir)
        
        # Count scenarios
        self.num_scenarios = len(self.db)
        logger.info(f"Found {self.num_scenarios} scenarios in {data_dir}")
        
        # Debug: show directory structure
        data_path = Path(data_dir)
        logger.info(f"Data directory contents:")
        for item in sorted(data_path.iterdir()):
            if item.is_dir():
                pkl_count = len(list(item.glob("sd_*.pkl")))
                logger.info(f"  {item.name}/ ({pkl_count} scenario files)")
            else:
                logger.info(f"  {item.name}")
    
    def _create_env(self, num_scenarios: Optional[int] = None):
        """Create or recreate the simulation environment."""
        # Close existing env
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
        
        if num_scenarios is None:
            num_scenarios = self.num_scenarios
        
        # Hide pygame window
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
        # Use config similar to ScenarioNet's own replay script
        self.env = ScenarioEnv({
            "use_render": False,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_logo": False,
            "show_fps": False,
            "num_scenarios": num_scenarios,
            "horizon": 1000,  # Max steps per episode
            "data_directory": self.data_dir,
            "vehicle_config": dict(
                show_navi_mark=False,
                show_line_to_dest=False,
                show_dest_mark=False,
                no_wheel_friction=True,
            ),
        })
        
        logger.info(f"Created ScenarioEnv with {num_scenarios} scenarios")
    
    def render_scenario(
        self,
        scenario_index: int,
        num_frames: int = 8,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, float]], np.ndarray, str]:
        """
        Render frames from a scenario using MetaDrive simulation.
        
        Frames are evenly spaced across the entire scenario duration.
        
        Args:
            scenario_index: Index of scenario to render
            num_frames: Number of frames to capture (evenly spaced)
            output_dir: Directory to save frames (None = temp)
        
        Returns:
            Tuple of:
                - List of (image_path, timestamp) tuples
                - Ego trajectory array [T, 4] (x, y, heading, speed)
                - Scenario ID
        """
        if self.env is None:
            self._create_env()
        
        # Setup output directory
        if output_dir is None:
            output_dir = f"./scenario_frames_{scenario_index}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Reset to target scenario
        try:
            obs, info = self.env.reset(seed=scenario_index)
        except Exception as e:
            logger.error(f"Failed to reset to scenario {scenario_index}: {e}")
            self._create_env()
            obs, info = self.env.reset(seed=scenario_index)
        
        # Get scenario ID and length
        try:
            scenario_id = self.env.engine.data_manager.current_scenario_id
            scenario_length = self.env.engine.data_manager.current_scenario_length
        except:
            scenario_id = self.db.get_scenario_id(scenario_index)
            scenario_length = 91  # Default fallback (typical Waymo length)
        
        logger.info(f"Rendering scenario {scenario_index}: {scenario_id} (length: {scenario_length} steps)")
        
        # Calculate which steps to capture frames at (evenly spaced)
        # e.g., for 8 frames over 91 steps: [0, 13, 26, 39, 52, 65, 78, 90]
        if num_frames >= scenario_length:
            frame_steps = list(range(scenario_length))
        else:
            frame_steps = [int(i * (scenario_length - 1) / (num_frames - 1)) for i in range(num_frames)]
        
        logger.info(f"Will capture {len(frame_steps)} frames at steps: {frame_steps}")
        
        # Collect trajectory and frames
        all_frames = {}  # step -> frame
        trajectory = []
        dt = 0.1  # 10 Hz
        
        # Run simulation until scenario ends
        for step in range(10000):
            # Get ego state
            try:
                ego = self.env.vehicle
                pos = ego.position
                heading = ego.heading_theta
                speed = ego.speed
                trajectory.append([pos[0], pos[1], heading, speed])
            except Exception as e:
                logger.warning(f"Could not get ego state at step {step}: {e}")
                trajectory.append([0, 0, 0, 0])
            
            # Capture frame if this is one of our target steps
            if step in frame_steps:
                try:
                    frame = self.env.render(
                        mode="top_down",
                        film_size=self.film_size,
                        screen_size=self.screen_size
                    )
                    all_frames[step] = (frame, step * dt)
                    logger.debug(f"Captured frame at step {step}")
                except Exception as e:
                    logger.warning(f"Render failed at step {step}: {e}")
            
            # Step simulation
            try:
                obs, reward, terminated, truncated, info = self.env.step([0, 0])
            except Exception as e:
                logger.warning(f"Step failed at {step}: {e}")
                break
            
            # Check if scenario ended (ScenarioNet's approach)
            if self.env.episode_step >= scenario_length:
                logger.info(f"Scenario completed at step {step}")
                break
        
        # Save frames in order
        saved_images = []
        for step in sorted(all_frames.keys()):
            frame, timestamp = all_frames[step]
            if frame is not None:
                filename = f"frame_{timestamp:.2f}.png"
                filepath = output_path / filename
                
                if HAS_PIL:
                    img = Image.fromarray(frame)
                    img.save(filepath)
                else:
                    import matplotlib.pyplot as plt
                    plt.imsave(str(filepath), frame)
                
                saved_images.append((str(filepath), timestamp))
                logger.info(f"Saved {filename}")
        
        # Convert trajectory to numpy
        trajectory = np.array(trajectory, dtype=np.float32)
        
        logger.info(f"Captured {len(saved_images)} frames, {len(trajectory)} trajectory steps")
        
        return saved_images, trajectory, scenario_id
    
    def get_scenario_info(self, scenario_index: int) -> Dict[str, Any]:
        """Get metadata about a scenario."""
        scenario_data = self.db.load_scenario(scenario_index)
        
        info = {
            "scenario_id": scenario_data.get("id", "unknown"),
            "num_tracks": len(scenario_data.get("tracks", {})),
            "sdc_id": scenario_data.get("metadata", {}).get("sdc_id", "unknown"),
        }
        
        # Get track types
        track_types = {}
        for track_id, track in scenario_data.get("tracks", {}).items():
            t = track.get("type", "unknown")
            track_types[t] = track_types.get(t, 0) + 1
        info["track_types"] = track_types
        
        return info
    
    def close(self):
        """Close the simulation environment."""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass
            self.env = None


def prepare_for_vlm(
    data_dir: str,
    scenario_index: int = 0,
    output_dir: Optional[str] = None,
    num_frames: int = 8,
    film_size: Tuple[int, int] = (1200, 1200),
    screen_size: Tuple[int, int] = (800, 800)
) -> Tuple[List[Tuple[str, float]], np.ndarray, str]:
    """
    Convenience function to prepare scenario frames for VLM analysis.
    
    Args:
        data_dir: Path to ScenarioNet data directory
        scenario_index: Which scenario to visualize
        output_dir: Where to save frames
        num_frames: Number of frames to capture
        film_size: Internal render size (higher = more detail)
        screen_size: Output image size
    
    Returns:
        Tuple of:
            - List of (image_path, timestamp) tuples
            - Ego trajectory array [T, 4] with columns [x, y, heading, speed]
            - Scenario ID string
    
    Example:
        saved_images, trajectory, scenario_id = prepare_for_vlm(
            data_dir="./exp_converted",
            scenario_index=0,
            num_frames=8
        )
        
        # Use with VLM extractor
        from vlm_extractor import VLMSafetyCriticalExtractor, TimestampedImage
        images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
        features = extractor.extract(images, scenario_id, trajectory)
    """
    if output_dir is None:
        output_dir = f"./vlm_frames/{scenario_index}"
    
    visualizer = ScenarioNetVisualizer(
        data_dir=data_dir,
        film_size=film_size,
        screen_size=screen_size
    )
    
    try:
        saved_images, trajectory, scenario_id = visualizer.render_scenario(
            scenario_index=scenario_index,
            num_frames=num_frames,
            output_dir=output_dir,
        )
        
        return saved_images, trajectory, scenario_id
        
    finally:
        visualizer.close()


def extract_trajectory_from_scenario(
    data_dir: str,
    scenario_index: int
) -> Tuple[np.ndarray, str, List[Dict]]:
    """
    Extract trajectory data directly from scenario file without simulation.
    
    This is faster than running simulation but doesn't produce frames.
    
    Args:
        data_dir: Path to ScenarioNet data
        scenario_index: Scenario index
    
    Returns:
        Tuple of:
            - Ego trajectory [T, 4]
            - Scenario ID
            - List of other agent info dicts
    """
    db = ScenarioNetDatabase(data_dir)
    scenario_data = db.load_scenario(scenario_index)
    
    scenario_id = scenario_data.get("id", "unknown")
    sdc_id = scenario_data.get("metadata", {}).get("sdc_id")
    
    # Extract ego trajectory
    ego_trajectory = []
    other_agents = []
    
    for track_id, track in scenario_data.get("tracks", {}).items():
        state = track.get("state", {})
        position = state.get("position", [])
        heading = state.get("heading", [])
        velocity = state.get("velocity", [])
        valid = state.get("valid", [])
        
        if str(track_id) == str(sdc_id):
            # This is the ego vehicle
            for t in range(len(position)):
                if t < len(valid) and valid[t]:
                    pos = position[t]
                    h = heading[t] if t < len(heading) else 0
                    vel = velocity[t] if t < len(velocity) else [0, 0]
                    speed = np.sqrt(vel[0]**2 + vel[1]**2) if len(vel) >= 2 else 0
                    ego_trajectory.append([pos[0], pos[1], h, speed])
        else:
            # Other agent - get initial state
            if len(position) > 0 and len(position[0]) >= 2:
                # Check if valid at t=0
                if len(valid) == 0 or valid[0]:
                    pos = position[0]
                    # Filter out origin
                    if abs(pos[0]) > 1.0 or abs(pos[1]) > 1.0:
                        vel = velocity[0] if len(velocity) > 0 else [0, 0]
                        speed = np.sqrt(vel[0]**2 + vel[1]**2) if len(vel) >= 2 else 0
                        
                        agent_type = track.get("type", "vehicle")
                        if isinstance(agent_type, int):
                            agent_type = {1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}.get(agent_type, 'vehicle')
                        
                        other_agents.append({
                            "agent_id": f"agent_{track_id}",
                            "type": str(agent_type),
                            "position": (float(pos[0]), float(pos[1])),
                            "speed": float(speed)
                        })
    
    ego_trajectory = np.array(ego_trajectory, dtype=np.float32)
    
    logger.info(f"Extracted trajectory: {len(ego_trajectory)} steps, {len(other_agents)} other agents")
    
    return ego_trajectory, scenario_id, other_agents


def extract_all_trajectories(
    data_dir: str,
    scenario_index: int,
    normalize_to_origin: bool = True
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Extract ground truth trajectories for ego and all other agents.
    
    This provides the data needed for ADV-BMT paper metrics:
    - Ground truth ego trajectory for SFDE/SADE calculations
    - Other agent trajectories for collision rate and TTC JSD
    
    Args:
        data_dir: Path to ScenarioNet data
        scenario_index: Scenario index
        normalize_to_origin: If True, transform all trajectories so ego starts at origin.
                           This is needed because ScenarioNet uses global coordinates
                           while BMT predictions use local coordinates.
    
    Returns:
        Tuple of:
            - ego_trajectory: (T, 2) ground truth ego positions [x, y]
            - other_agents_trajectories: (N, T, 2) other agent positions [x, y]
            - scenario_id: Scenario identifier
    """
    db = ScenarioNetDatabase(data_dir)
    scenario_data = db.load_scenario(scenario_index)
    
    scenario_id = scenario_data.get("id", "unknown")
    sdc_id = scenario_data.get("metadata", {}).get("sdc_id")
    
    ego_trajectory = None
    ego_heading_start = 0.0
    other_trajectories = []
    max_timesteps = 0
    
    # First pass: find ego and determine max timesteps
    for track_id, track in scenario_data.get("tracks", {}).items():
        state = track.get("state", {})
        position = state.get("position", [])
        heading = state.get("heading", [])
        valid = state.get("valid", [])
        
        if str(track_id) == str(sdc_id):
            # Ego vehicle
            traj = []
            for t in range(len(position)):
                if t < len(valid) and valid[t]:
                    pos = position[t]
                    traj.append([pos[0], pos[1]])
            if traj:
                ego_trajectory = np.array(traj, dtype=np.float32)
                max_timesteps = max(max_timesteps, len(traj))
                # Get initial heading for rotation
                if len(heading) > 0:
                    ego_heading_start = heading[0]
    
    if ego_trajectory is None:
        logger.warning(f"No ego trajectory found for scenario {scenario_id}")
        return np.array([]), np.array([]), scenario_id
    
    # Store ego origin for normalization
    ego_origin = ego_trajectory[0].copy()
    
    # Second pass: extract other agent trajectories
    for track_id, track in scenario_data.get("tracks", {}).items():
        if str(track_id) == str(sdc_id):
            continue  # Skip ego
        
        state = track.get("state", {})
        position = state.get("position", [])
        valid = state.get("valid", [])
        
        # Only include tracks that are active during the scenario
        # Build trajectory array matching ego length
        agent_traj = np.zeros((max_timesteps, 2), dtype=np.float32)
        has_valid = False
        last_valid_pos = None
        
        for t in range(min(len(position), max_timesteps)):
            if t < len(valid) and valid[t]:
                pos = position[t]
                # Filter out origin/invalid positions
                if abs(pos[0]) > 0.1 or abs(pos[1]) > 0.1:
                    agent_traj[t] = [pos[0], pos[1]]
                    last_valid_pos = [pos[0], pos[1]]
                    has_valid = True
            elif last_valid_pos is not None:
                # Forward-fill with last valid position
                agent_traj[t] = last_valid_pos
        
        # Only include if agent has valid positions
        if has_valid:
            other_trajectories.append(agent_traj)
    
    # Normalize to ego-centric coordinates (origin at ego start, aligned with ego heading)
    if normalize_to_origin and len(ego_trajectory) > 0:
        # Translate so ego starts at origin
        ego_trajectory = ego_trajectory - ego_origin
        
        # Rotate so ego initially faces +x direction (heading = 0)
        # This matches BMT's local coordinate frame
        cos_h = np.cos(-ego_heading_start)
        sin_h = np.sin(-ego_heading_start)
        rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        ego_trajectory = ego_trajectory @ rotation_matrix.T
        
        # Transform other agents to same frame
        for i, agent_traj in enumerate(other_trajectories):
            agent_traj = agent_traj - ego_origin
            other_trajectories[i] = agent_traj @ rotation_matrix.T
        
        logger.info(f"  Normalized to ego-centric frame (origin={ego_origin}, heading={np.degrees(ego_heading_start):.1f}°)")
    
    # Stack other trajectories: (N, T, 2)
    if other_trajectories:
        other_agents_array = np.stack(other_trajectories, axis=0)
    else:
        other_agents_array = np.zeros((0, max_timesteps, 2), dtype=np.float32)
    
    logger.info(f"Extracted ground truth: ego={ego_trajectory.shape}, "
                f"other_agents={other_agents_array.shape}")
    
    return ego_trajectory, other_agents_array, scenario_id


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ScenarioNet Visualizer")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to ScenarioNet data")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario index")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    
    args = parser.parse_args()
    
    if args.list:
        db = ScenarioNetDatabase(args.data_dir)
        print(f"\nFound {len(db)} scenarios in {args.data_dir}:")
        for i, sid in enumerate(db.scenario_ids[:20]):
            print(f"  {i}: {sid}")
        if len(db) > 20:
            print(f"  ... and {len(db) - 20} more")
    else:
        print(f"\nRendering scenario {args.scenario}...")
        saved_images, trajectory, scenario_id = prepare_for_vlm(
            data_dir=args.data_dir,
            scenario_index=args.scenario,
            num_frames=args.num_frames,
            output_dir=args.output_dir
        )
        
        print(f"\nScenario: {scenario_id}")
        print(f"Frames saved: {len(saved_images)}")
        print(f"Trajectory steps: {len(trajectory)}")
        
        if len(trajectory) > 0:
            print(f"\nTrajectory preview:")
            print(f"  Start: ({trajectory[0][0]:.1f}, {trajectory[0][1]:.1f}), "
                  f"heading={trajectory[0][2]:.2f}, speed={trajectory[0][3]:.1f}")
            print(f"  End: ({trajectory[-1][0]:.1f}, {trajectory[-1][1]:.1f}), "
                  f"heading={trajectory[-1][2]:.2f}, speed={trajectory[-1][3]:.1f}")