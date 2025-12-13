"""
Trajectory Metrics Module for CounterBMT

Calculates trajectory evaluation metrics based on the ADV-BMT paper:
"Generating Adversarial Driving Scenarios in High-Fidelity Simulators"

Metrics include:

Table (a) - Realism Metrics:
- SFDE (Scene Final Displacement Error) - avg/min
- SADE (Scene Average Displacement Error) - avg/min
- VehColl (Vehicle Collision rate) - avg/min
- JSDvelocity (Jensen-Shannon Divergence of velocity)
- JSDTTC (Jensen-Shannon Divergence of Time-To-Collision)

Table (b) - Diversity Metrics:
- SDD (Scene Displacement Diversity)
- FDD (Final Displacement Diversity)
- ADD (Average Displacement Diversity)

Author: CounterBMT Project
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy

logger = logging.getLogger(__name__)


def _to_python(val):
    """Convert numpy types to Python types for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


@dataclass
class TrajectoryMetrics:
    """Container for computed trajectory metrics."""
    
    # Basic trajectory properties
    trajectory_length: int = 0
    travel_distance: float = 0.0
    
    # Displacement errors (relative to ground truth)
    ade: Optional[float] = None  # Average Displacement Error (SADE)
    fde: Optional[float] = None  # Final Displacement Error (SFDE)
    
    # Kinematic metrics
    mean_speed: float = 0.0
    max_speed: float = 0.0
    mean_acceleration: float = 0.0
    max_acceleration: float = 0.0
    max_jerk: float = 0.0
    
    # Safety metrics
    min_distance_to_agents: Optional[float] = None
    collision_detected: bool = False
    time_to_collision: Optional[float] = None  # TTC in seconds
    
    # Deviation metrics
    max_lateral_deviation: float = 0.0
    path_divergence_point: Optional[int] = None
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'trajectory_length': int(self.trajectory_length),
            'travel_distance': float(self.travel_distance),
            'ade': _to_python(self.ade),
            'fde': _to_python(self.fde),
            'mean_speed': float(self.mean_speed),
            'max_speed': float(self.max_speed),
            'mean_acceleration': float(self.mean_acceleration),
            'max_acceleration': float(self.max_acceleration),
            'max_jerk': float(self.max_jerk),
            'min_distance_to_agents': _to_python(self.min_distance_to_agents),
            'collision_detected': bool(self.collision_detected),
            'time_to_collision': _to_python(self.time_to_collision),
            'max_lateral_deviation': float(self.max_lateral_deviation),
            'path_divergence_point': _to_python(self.path_divergence_point),
            'metadata': self.metadata
        }


@dataclass
class ADVBMTMetrics:
    """
    Container for ADV-BMT paper metrics.
    
    These metrics evaluate the realism and diversity of generated trajectories
    compared to ground truth.
    """
    
    # === Realism Metrics (Table a) ===
    # Scene Final Displacement Error
    sfde_avg: float = 0.0  # Average FDE across samples
    sfde_min: float = 0.0  # Minimum FDE (best sample)
    
    # Scene Average Displacement Error
    sade_avg: float = 0.0  # Average ADE across samples
    sade_min: float = 0.0  # Minimum ADE (best sample)
    
    # Vehicle Collision Rate
    veh_coll_avg: float = 0.0  # Average collision rate
    veh_coll_min: float = 0.0  # Minimum collision rate (best sample)
    
    # Jensen-Shannon Divergence metrics
    jsd_velocity: float = 0.0  # JSD of velocity distributions
    jsd_ttc: float = 0.0  # JSD of Time-To-Collision distributions
    
    # === Diversity Metrics (Table b) ===
    sdd: float = 0.0  # Scene Displacement Diversity (max pairwise at any timestep)
    fdd: float = 0.0  # Final Displacement Diversity (max pairwise at final position)
    add: float = 0.0  # Average Displacement Diversity (mean of max pairwise per timestep)
    
    # === Additional useful metrics ===
    n_samples: int = 0
    n_collisions: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            # Realism metrics
            'sfde_avg': float(self.sfde_avg),
            'sfde_min': float(self.sfde_min),
            'sade_avg': float(self.sade_avg),
            'sade_min': float(self.sade_min),
            'veh_coll_avg': float(self.veh_coll_avg),
            'veh_coll_min': float(self.veh_coll_min),
            'jsd_velocity': float(self.jsd_velocity),
            'jsd_ttc': float(self.jsd_ttc),
            # Diversity metrics
            'sdd': float(self.sdd),
            'fdd': float(self.fdd),
            'add': float(self.add),
            # Additional
            'n_samples': int(self.n_samples),
            'n_collisions': int(self.n_collisions),
        }


@dataclass
class CounterfactualComparison:
    """Comparison results between baseline and counterfactual trajectories."""
    
    intervention_name: str
    baseline_metrics: TrajectoryMetrics
    counterfactual_metrics: TrajectoryMetrics
    
    # Comparison results
    distance_change_percent: float = 0.0
    speed_change_percent: float = 0.0
    trajectory_similarity: float = 0.0
    
    # Effectiveness assessment
    intervention_effective: bool = False
    effect_direction: str = "none"
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'intervention_name': self.intervention_name,
            'baseline_metrics': self.baseline_metrics.to_dict(),
            'counterfactual_metrics': self.counterfactual_metrics.to_dict(),
            'distance_change_percent': float(self.distance_change_percent),
            'speed_change_percent': float(self.speed_change_percent),
            'trajectory_similarity': float(self.trajectory_similarity),
            'intervention_effective': bool(self.intervention_effective),
            'effect_direction': str(self.effect_direction),
            'confidence': float(self.confidence)
        }


class TrajectoryMetricsCalculator:
    """
    Calculator for trajectory metrics including ADV-BMT paper metrics.
    
    Usage:
        calculator = TrajectoryMetricsCalculator(dt=0.1, collision_threshold=2.0)
        
        # Single trajectory metrics
        metrics = calculator.compute_metrics(trajectory, ground_truth, other_agents)
        
        # ADV-BMT paper metrics for multiple samples
        adv_metrics = calculator.compute_advbmt_metrics(
            samples, ground_truth, other_agents
        )
    """
    
    def __init__(self, dt: float = 0.1, collision_threshold: float = 2.0):
        """
        Initialize calculator.
        
        Args:
            dt: Time step between trajectory points (seconds)
            collision_threshold: Distance threshold for collision detection (meters)
        """
        self.dt = dt
        self.collision_threshold = collision_threshold
    
    # =========================================================================
    # ADV-BMT Paper Metrics
    # =========================================================================
    
    def compute_advbmt_metrics(
        self,
        predicted_trajectories: List[np.ndarray],
        ground_truth: Optional[np.ndarray] = None,
        other_agents: Optional[np.ndarray] = None,
        gt_velocities: Optional[np.ndarray] = None,
    ) -> ADVBMTMetrics:
        """
        Compute all ADV-BMT paper metrics for a set of trajectory predictions.
        
        Args:
            predicted_trajectories: List of (T, 2) predicted trajectories
            ground_truth: (T, 2) ground truth trajectory for ego agent
            other_agents: (N, T, 2) trajectories of other agents
            gt_velocities: (T,) ground truth velocity profile (optional, computed if not provided)
            
        Returns:
            ADVBMTMetrics object with all paper metrics
        """
        if not predicted_trajectories:
            return ADVBMTMetrics()
        
        metrics = ADVBMTMetrics(n_samples=len(predicted_trajectories))
        trajectories = [np.asarray(t) for t in predicted_trajectories]
        
        # === Compute Realism Metrics ===
        if ground_truth is not None:
            gt = np.asarray(ground_truth)
            
            # SFDE and SADE
            fde_values = []
            ade_values = []
            for traj in trajectories:
                min_len = min(len(traj), len(gt))
                fde = self._compute_fde(traj[:min_len], gt[:min_len])
                ade = self._compute_ade(traj[:min_len], gt[:min_len])
                fde_values.append(fde)
                ade_values.append(ade)
            
            metrics.sfde_avg = float(np.mean(fde_values))
            metrics.sfde_min = float(np.min(fde_values))
            metrics.sade_avg = float(np.mean(ade_values))
            metrics.sade_min = float(np.min(ade_values))
            
            # JSD Velocity
            if gt_velocities is None:
                gt_velocities = self._compute_velocities(gt)
            metrics.jsd_velocity = self._compute_velocity_jsd(trajectories, gt_velocities)
        
        # Vehicle Collision Rate
        if other_agents is not None:
            other_agents = np.asarray(other_agents)
            collision_counts = []
            for traj in trajectories:
                n_collisions = self._count_collisions(traj, other_agents)
                collision_counts.append(n_collisions)
            
            total_timesteps = len(trajectories[0]) if trajectories else 1
            metrics.veh_coll_avg = float(np.mean(collision_counts) / total_timesteps)
            metrics.veh_coll_min = float(np.min(collision_counts) / total_timesteps)
            metrics.n_collisions = int(np.sum(collision_counts))
            
            # JSD TTC
            metrics.jsd_ttc = self._compute_ttc_jsd(trajectories, other_agents, ground_truth)
        
        # === Compute Diversity Metrics ===
        metrics.fdd = self._compute_fdd(trajectories)
        metrics.sdd = self._compute_sdd(trajectories)
        metrics.add = self._compute_add(trajectories)
        
        return metrics
    
    def _compute_fdd(self, trajectories: List[np.ndarray]) -> float:
        """
        Compute Final Displacement Diversity (FDD).
        
        Maximum pairwise distance between final positions of trajectory samples.
        """
        if len(trajectories) < 2:
            return 0.0
        
        final_positions = np.array([t[-1] for t in trajectories])
        distances = cdist(final_positions, final_positions)
        return float(np.max(distances))
    
    def _compute_sdd(self, trajectories: List[np.ndarray]) -> float:
        """
        Compute Scene Displacement Diversity (SDD).
        
        Maximum pairwise distance at any timestep across all samples.
        """
        if len(trajectories) < 2:
            return 0.0
        
        # Find minimum common length
        min_len = min(len(t) for t in trajectories)
        
        max_diversity = 0.0
        for t in range(min_len):
            positions_at_t = np.array([traj[t] for traj in trajectories])
            distances = cdist(positions_at_t, positions_at_t)
            max_diversity = max(max_diversity, np.max(distances))
        
        return float(max_diversity)
    
    def _compute_add(self, trajectories: List[np.ndarray]) -> float:
        """
        Compute Average Displacement Diversity (ADD).
        
        Average of maximum pairwise distances across all timesteps.
        """
        if len(trajectories) < 2:
            return 0.0
        
        min_len = min(len(t) for t in trajectories)
        
        max_distances = []
        for t in range(min_len):
            positions_at_t = np.array([traj[t] for traj in trajectories])
            distances = cdist(positions_at_t, positions_at_t)
            max_distances.append(np.max(distances))
        
        return float(np.mean(max_distances))
    
    def _compute_velocities(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute velocity profile from positions."""
        if len(trajectory) < 2:
            return np.array([0.0])
        velocity = np.diff(trajectory, axis=0) / self.dt
        speeds = np.sqrt(np.sum(velocity**2, axis=1))
        return speeds
    
    def _compute_velocity_jsd(
        self, trajectories: List[np.ndarray], gt_velocities: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon Divergence between predicted and ground truth velocity distributions.
        """
        if len(trajectories) == 0:
            return 0.0
        
        # Compute velocities for all predictions
        pred_velocities = []
        for traj in trajectories:
            vels = self._compute_velocities(traj)
            pred_velocities.extend(vels)
        
        pred_velocities = np.array(pred_velocities)
        gt_velocities = np.asarray(gt_velocities)
        
        # Create histograms (probability distributions)
        # Use same bins for both
        all_vels = np.concatenate([pred_velocities, gt_velocities])
        bins = np.linspace(0, np.max(all_vels) + 1, 50)
        
        pred_hist, _ = np.histogram(pred_velocities, bins=bins, density=True)
        gt_hist, _ = np.histogram(gt_velocities, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred_hist = pred_hist + eps
        gt_hist = gt_hist + eps
        
        # Normalize
        pred_hist = pred_hist / pred_hist.sum()
        gt_hist = gt_hist / gt_hist.sum()
        
        # JSD = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
        m = 0.5 * (pred_hist + gt_hist)
        jsd = 0.5 * entropy(pred_hist, m) + 0.5 * entropy(gt_hist, m)
        
        return float(jsd)
    
    def _compute_ttc_jsd(
        self,
        trajectories: List[np.ndarray],
        other_agents: np.ndarray,
        ground_truth: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Jensen-Shannon Divergence of Time-To-Collision distributions.
        """
        # Compute TTC for predictions
        pred_ttcs = []
        for traj in trajectories:
            ttc = self._compute_min_ttc(traj, other_agents)
            if ttc is not None:
                pred_ttcs.append(ttc)
        
        if not pred_ttcs:
            return 0.0
        
        # Compute TTC for ground truth if available
        if ground_truth is not None:
            gt_ttc = self._compute_min_ttc(ground_truth, other_agents)
            if gt_ttc is None:
                gt_ttcs = [10.0]  # Default large TTC if no collision risk
            else:
                gt_ttcs = [gt_ttc]
        else:
            # Use mean of predictions as reference
            gt_ttcs = [np.mean(pred_ttcs)]
        
        pred_ttcs = np.array(pred_ttcs)
        gt_ttcs = np.array(gt_ttcs)
        
        # Create histograms
        bins = np.linspace(0, max(np.max(pred_ttcs), np.max(gt_ttcs)) + 1, 30)
        
        pred_hist, _ = np.histogram(pred_ttcs, bins=bins, density=True)
        gt_hist, _ = np.histogram(gt_ttcs, bins=bins, density=True)
        
        eps = 1e-10
        pred_hist = pred_hist + eps
        gt_hist = gt_hist + eps
        
        pred_hist = pred_hist / pred_hist.sum()
        gt_hist = gt_hist / gt_hist.sum()
        
        m = 0.5 * (pred_hist + gt_hist)
        jsd = 0.5 * entropy(pred_hist, m) + 0.5 * entropy(gt_hist, m)
        
        return float(jsd)
    
    def _compute_min_ttc(
        self, trajectory: np.ndarray, other_agents: np.ndarray
    ) -> Optional[float]:
        """
        Compute minimum Time-To-Collision for a trajectory.
        
        Uses a simplified constant velocity assumption.
        """
        if other_agents is None or len(other_agents) == 0:
            return None
        
        trajectory = np.asarray(trajectory)
        min_ttc = float('inf')
        
        T = min(len(trajectory), other_agents.shape[1])
        
        for t in range(T - 1):
            ego_pos = trajectory[t]
            ego_vel = (trajectory[min(t+1, T-1)] - trajectory[t]) / self.dt
            
            for agent_idx in range(other_agents.shape[0]):
                agent_pos = other_agents[agent_idx, t]
                agent_vel = (other_agents[agent_idx, min(t+1, T-1)] - other_agents[agent_idx, t]) / self.dt
                
                # Relative position and velocity
                rel_pos = agent_pos - ego_pos
                rel_vel = agent_vel - ego_vel
                
                # Distance
                dist = np.linalg.norm(rel_pos)
                if dist < self.collision_threshold:
                    min_ttc = 0  # Already in collision
                    break
                
                # TTC approximation: time until distance becomes < threshold
                # Using linear approximation
                closing_speed = -np.dot(rel_pos, rel_vel) / (dist + 1e-6)
                if closing_speed > 0:
                    ttc = (dist - self.collision_threshold) / closing_speed
                    if ttc > 0:
                        min_ttc = min(min_ttc, ttc)
            
            if min_ttc == 0:
                break
        
        return float(min_ttc) if min_ttc != float('inf') else None
    
    def _count_collisions(
        self, trajectory: np.ndarray, other_agents: np.ndarray
    ) -> int:
        """Count number of timesteps with collision."""
        T = min(len(trajectory), other_agents.shape[1])
        collision_count = 0
        
        for t in range(T):
            for agent_idx in range(other_agents.shape[0]):
                dist = np.linalg.norm(trajectory[t] - other_agents[agent_idx, t])
                if dist < self.collision_threshold:
                    collision_count += 1
                    break  # Count each timestep only once
        
        return collision_count
    
    # =========================================================================
    # Standard Metrics
    # =========================================================================
    
    def compute_metrics(
        self,
        trajectory: np.ndarray,
        reference_trajectory: Optional[np.ndarray] = None,
        other_agents: Optional[np.ndarray] = None
    ) -> TrajectoryMetrics:
        """
        Compute comprehensive metrics for a single trajectory.
        
        Args:
            trajectory: (T, 2) array of (x, y) positions
            reference_trajectory: Optional (T, 2) reference for computing ADE/FDE
            other_agents: Optional (N, T, 2) positions of other agents
            
        Returns:
            TrajectoryMetrics object
        """
        trajectory = np.asarray(trajectory)
        metrics = TrajectoryMetrics()
        
        # Basic properties
        metrics.trajectory_length = len(trajectory)
        metrics.travel_distance = self._compute_travel_distance(trajectory)
        
        # Kinematic metrics
        speed, accel, jerk = self._compute_kinematics(trajectory)
        metrics.mean_speed = float(np.nanmean(speed)) if len(speed) > 0 else 0.0
        metrics.max_speed = float(np.nanmax(speed)) if len(speed) > 0 else 0.0
        metrics.mean_acceleration = float(np.nanmean(np.abs(accel))) if len(accel) > 0 else 0.0
        metrics.max_acceleration = float(np.nanmax(np.abs(accel))) if len(accel) > 0 else 0.0
        metrics.max_jerk = float(np.nanmax(np.abs(jerk))) if len(jerk) > 0 else 0.0
        
        # Displacement errors relative to reference
        if reference_trajectory is not None:
            reference_trajectory = np.asarray(reference_trajectory)
            min_len = min(len(trajectory), len(reference_trajectory))
            metrics.ade = self._compute_ade(trajectory[:min_len], reference_trajectory[:min_len])
            metrics.fde = self._compute_fde(trajectory[:min_len], reference_trajectory[:min_len])
            metrics.max_lateral_deviation = self._compute_max_lateral_deviation(
                trajectory[:min_len], reference_trajectory[:min_len]
            )
            metrics.path_divergence_point = self._find_divergence_point(
                trajectory[:min_len], reference_trajectory[:min_len]
            )
        
        # Safety metrics relative to other agents
        if other_agents is not None:
            other_agents = np.asarray(other_agents)
            metrics.min_distance_to_agents = self._compute_min_distance_to_agents(
                trajectory, other_agents
            )
            metrics.collision_detected = metrics.min_distance_to_agents < self.collision_threshold
            metrics.time_to_collision = self._compute_min_ttc(trajectory, other_agents)
        
        return metrics
    
    def compare_trajectories(
        self,
        baseline: np.ndarray,
        counterfactual: np.ndarray,
        intervention_name: str,
        other_agents: Optional[np.ndarray] = None
    ) -> CounterfactualComparison:
        """Compare baseline and counterfactual trajectories."""
        baseline = np.asarray(baseline)
        counterfactual = np.asarray(counterfactual)
        
        baseline_metrics = self.compute_metrics(baseline, other_agents=other_agents)
        counterfactual_metrics = self.compute_metrics(
            counterfactual, 
            reference_trajectory=baseline,
            other_agents=other_agents
        )
        
        comparison = CounterfactualComparison(
            intervention_name=intervention_name,
            baseline_metrics=baseline_metrics,
            counterfactual_metrics=counterfactual_metrics
        )
        
        if baseline_metrics.travel_distance > 0:
            comparison.distance_change_percent = (
                (counterfactual_metrics.travel_distance - baseline_metrics.travel_distance) 
                / baseline_metrics.travel_distance * 100
            )
        
        if baseline_metrics.mean_speed > 0:
            comparison.speed_change_percent = (
                (counterfactual_metrics.mean_speed - baseline_metrics.mean_speed)
                / baseline_metrics.mean_speed * 100
            )
        
        if counterfactual_metrics.ade is not None and baseline_metrics.travel_distance > 0:
            normalized_ade = counterfactual_metrics.ade / baseline_metrics.travel_distance
            comparison.trajectory_similarity = max(0, 1 - normalized_ade)
        
        comparison.intervention_effective = abs(comparison.distance_change_percent) > 5.0
        if comparison.distance_change_percent > 5:
            comparison.effect_direction = "increase"
        elif comparison.distance_change_percent < -5:
            comparison.effect_direction = "decrease"
        else:
            comparison.effect_direction = "none"
        
        if counterfactual_metrics.fde is not None:
            comparison.confidence = min(1.0, counterfactual_metrics.fde / 10.0)
        
        return comparison
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _compute_travel_distance(self, trajectory: np.ndarray) -> float:
        """Compute total travel distance."""
        if len(trajectory) < 2:
            return 0.0
        diffs = np.diff(trajectory, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return float(np.sum(distances))
    
    def _compute_kinematics(
        self, trajectory: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute speed, acceleration, and jerk from positions."""
        if len(trajectory) < 2:
            return np.array([]), np.array([]), np.array([])
        
        velocity = np.diff(trajectory, axis=0) / self.dt
        speed = np.sqrt(np.sum(velocity**2, axis=1))
        
        if len(speed) < 2:
            return speed, np.array([]), np.array([])
        acceleration = np.diff(speed) / self.dt
        
        if len(acceleration) < 2:
            return speed, acceleration, np.array([])
        jerk = np.diff(acceleration) / self.dt
        
        return speed, acceleration, jerk
    
    def _compute_ade(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Average Displacement Error."""
        distances = np.sqrt(np.sum((traj1 - traj2)**2, axis=1))
        return float(np.mean(distances))
    
    def _compute_fde(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Compute Final Displacement Error."""
        return float(np.sqrt(np.sum((traj1[-1] - traj2[-1])**2)))
    
    def _compute_max_lateral_deviation(
        self, trajectory: np.ndarray, reference: np.ndarray
    ) -> float:
        """Compute maximum lateral deviation from reference path."""
        distances = np.sqrt(np.sum((trajectory - reference)**2, axis=1))
        return float(np.max(distances))
    
    def _find_divergence_point(
        self, trajectory: np.ndarray, reference: np.ndarray, threshold: float = 2.0
    ) -> Optional[int]:
        """Find the timestep where trajectories significantly diverge."""
        distances = np.sqrt(np.sum((trajectory - reference)**2, axis=1))
        divergent_indices = np.where(distances > threshold)[0]
        if len(divergent_indices) > 0:
            return int(divergent_indices[0])
        return None
    
    def _compute_min_distance_to_agents(
        self, trajectory: np.ndarray, other_agents: np.ndarray
    ) -> float:
        """Compute minimum distance to any other agent across all timesteps."""
        min_dist = float('inf')
        T = min(len(trajectory), other_agents.shape[1])
        
        for t in range(T):
            for agent_idx in range(other_agents.shape[0]):
                dist = np.sqrt(np.sum((trajectory[t] - other_agents[agent_idx, t])**2))
                min_dist = min(min_dist, dist)
        
        return float(min_dist) if min_dist != float('inf') else None
    
    def _compute_sample_diversity(self, trajectories: List[np.ndarray]) -> float:
        """Compute FDD (Final Displacement Diversity) for backward compatibility."""
        return self._compute_fdd(trajectories)


# =============================================================================
# High-Level Functions
# =============================================================================

def compute_intervention_effectiveness(
    baseline_trajectory: np.ndarray,
    counterfactual_trajectories: List[np.ndarray],
    intervention_name: str,
    expected_effect: str = "decrease",
    ground_truth: Optional[np.ndarray] = None,
    other_agents: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Evaluate the effectiveness of an intervention with ADV-BMT metrics.
    
    Args:
        baseline_trajectory: Original/baseline predicted trajectory
        counterfactual_trajectories: List of counterfactual trajectory samples
        intervention_name: Name of the intervention
        expected_effect: Expected effect on travel distance ("increase" or "decrease")
        ground_truth: Optional ground truth trajectory for realism metrics
        other_agents: Optional (N, T, 2) trajectories of other agents
        
    Returns:
        Dictionary with effectiveness and ADV-BMT metrics
    """
    calculator = TrajectoryMetricsCalculator()
    
    baseline_metrics = calculator.compute_metrics(baseline_trajectory, other_agents=other_agents)
    baseline_distance = baseline_metrics.travel_distance
    
    # Compute per-sample comparisons
    cf_distances = []
    cf_comparisons = []
    
    for cf_traj in counterfactual_trajectories:
        comparison = calculator.compare_trajectories(
            baseline_trajectory, cf_traj, intervention_name, other_agents
        )
        cf_comparisons.append(comparison)
        cf_distances.append(comparison.counterfactual_metrics.travel_distance)
    
    mean_cf_distance = np.mean(cf_distances) if cf_distances else 0.0
    distance_change = (mean_cf_distance - baseline_distance) / baseline_distance * 100 if baseline_distance > 0 else 0.0
    
    effect_matches = (
        (expected_effect == "decrease" and distance_change < -5) or
        (expected_effect == "increase" and distance_change > 5)
    )
    
    # Compute ADV-BMT metrics
    advbmt_metrics = calculator.compute_advbmt_metrics(
        counterfactual_trajectories,
        ground_truth=ground_truth,
        other_agents=other_agents,
    )
    
    return {
        'intervention_name': intervention_name,
        'baseline_distance': float(baseline_distance),
        'mean_counterfactual_distance': float(mean_cf_distance),
        'distance_change_percent': float(distance_change),
        'std_counterfactual_distance': float(np.std(cf_distances)) if cf_distances else 0.0,
        'expected_effect': expected_effect,
        'effect_matches_expected': bool(effect_matches),
        # ADV-BMT metrics
        'advbmt_metrics': advbmt_metrics.to_dict(),
        # Diversity metrics (for backward compatibility)
        'fdd': float(advbmt_metrics.fdd),
        'sdd': float(advbmt_metrics.sdd),
        'add': float(advbmt_metrics.add),
        # Detailed comparisons
        'comparisons': [c.to_dict() for c in cf_comparisons]
    }


def generate_metrics_summary(
    baseline: np.ndarray,
    interventions: Dict[str, List[np.ndarray]],
    expected_effects: Optional[Dict[str, str]] = None,
    ground_truth: Optional[np.ndarray] = None,
    other_agents: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Generate a comprehensive metrics summary for all interventions.
    
    Args:
        baseline: Baseline predicted trajectory
        interventions: Dict mapping intervention names to lists of trajectory samples
        expected_effects: Optional dict mapping intervention names to expected effects
        ground_truth: Optional ground truth trajectory
        other_agents: Optional (N, T, 2) trajectories of other agents
        
    Returns:
        Comprehensive summary dictionary with ADV-BMT metrics
    """
    expected_effects = expected_effects or {}
    calculator = TrajectoryMetricsCalculator()
    
    summary = {
        'baseline': calculator.compute_metrics(baseline, other_agents=other_agents).to_dict(),
        'interventions': {},
        'overall': {
            'n_interventions': len(interventions),
            'effective_interventions': 0,
            'matching_expected': 0,
        },
        # Aggregate ADV-BMT metrics across all interventions
        'aggregate_advbmt': {
            'sfde_avg': 0.0,
            'sfde_min': float('inf'),
            'sade_avg': 0.0,
            'sade_min': float('inf'),
            'veh_coll_avg': 0.0,
            'veh_coll_min': float('inf'),
            'jsd_velocity': 0.0,
            'jsd_ttc': 0.0,
            'mean_fdd': 0.0,
            'mean_sdd': 0.0,
            'mean_add': 0.0,
        }
    }
    
    all_sfde = []
    all_sade = []
    all_veh_coll = []
    all_jsd_vel = []
    all_jsd_ttc = []
    all_fdd = []
    all_sdd = []
    all_add = []
    
    for name, trajectories in interventions.items():
        expected = expected_effects.get(name, "decrease")
        effectiveness = compute_intervention_effectiveness(
            baseline, trajectories, name, expected,
            ground_truth=ground_truth,
            other_agents=other_agents,
        )
        summary['interventions'][name] = effectiveness
        
        if abs(effectiveness['distance_change_percent']) > 5:
            summary['overall']['effective_interventions'] += 1
        if effectiveness['effect_matches_expected']:
            summary['overall']['matching_expected'] += 1
        
        # Collect ADV-BMT metrics
        adv = effectiveness.get('advbmt_metrics', {})
        if adv.get('sfde_avg', 0) > 0:
            all_sfde.append(adv['sfde_avg'])
        if adv.get('sade_avg', 0) > 0:
            all_sade.append(adv['sade_avg'])
        all_veh_coll.append(adv.get('veh_coll_avg', 0))
        if adv.get('jsd_velocity', 0) > 0:
            all_jsd_vel.append(adv['jsd_velocity'])
        if adv.get('jsd_ttc', 0) > 0:
            all_jsd_ttc.append(adv['jsd_ttc'])
        all_fdd.append(adv.get('fdd', 0))
        all_sdd.append(adv.get('sdd', 0))
        all_add.append(adv.get('add', 0))
    
    # Compute overall statistics
    n_int = summary['overall']['n_interventions']
    if n_int > 0:
        summary['overall']['effectiveness_rate'] = (
            summary['overall']['effective_interventions'] / n_int
        )
        summary['overall']['prediction_accuracy'] = (
            summary['overall']['matching_expected'] / n_int
        )
    
    # Aggregate ADV-BMT metrics
    if all_sfde:
        summary['aggregate_advbmt']['sfde_avg'] = float(np.mean(all_sfde))
        summary['aggregate_advbmt']['sfde_min'] = float(np.min(all_sfde))
    if all_sade:
        summary['aggregate_advbmt']['sade_avg'] = float(np.mean(all_sade))
        summary['aggregate_advbmt']['sade_min'] = float(np.min(all_sade))
    if all_veh_coll:
        summary['aggregate_advbmt']['veh_coll_avg'] = float(np.mean(all_veh_coll))
        summary['aggregate_advbmt']['veh_coll_min'] = float(np.min(all_veh_coll))
    if all_jsd_vel:
        summary['aggregate_advbmt']['jsd_velocity'] = float(np.mean(all_jsd_vel))
    if all_jsd_ttc:
        summary['aggregate_advbmt']['jsd_ttc'] = float(np.mean(all_jsd_ttc))
    if all_fdd:
        summary['aggregate_advbmt']['mean_fdd'] = float(np.mean(all_fdd))
    if all_sdd:
        summary['aggregate_advbmt']['mean_sdd'] = float(np.mean(all_sdd))
    if all_add:
        summary['aggregate_advbmt']['mean_add'] = float(np.mean(all_add))
    
    return summary
