"""
LLM-Guided Intervention Planning

This module uses an LLM to generate context-aware intervention parameters
instead of hardcoded physics calculations. The LLM reasons about:
- Required acceleration/deceleration ranges
- Appropriate time durations
- Multi-phase intervention planning

This approach is more robust than hardcoding because:
1. The LLM can adapt to different scenario contexts
2. It can handle complex multi-phase maneuvers
3. It provides explainable reasoning for each phase

Enhanced with trajectory context:
- Full velocity/acceleration time series
- State at intervention timestamp
- Current maneuver and decision context
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import math
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Trajectory Context Builder
# =============================================================================

@dataclass
class TrajectoryState:
    """State of the ego vehicle at a specific time."""
    time_s: float
    position: Tuple[float, float]
    speed: float  # m/s
    heading: float  # rad
    acceleration: float  # m/s²
    yaw_rate: float  # rad/s
    
    def to_dict(self) -> Dict:
        return {
            'time_s': self.time_s,
            'position': list(self.position),
            'speed': self.speed,
            'heading': self.heading,
            'acceleration': self.acceleration,
            'yaw_rate': self.yaw_rate
        }


@dataclass 
class TrajectoryContext:
    """
    Rich trajectory context for LLM intervention planning.
    
    Provides temporal information about the ego vehicle's motion,
    including the state at the intervention timestamp.
    """
    # Full trajectory time series
    trajectory_states: List[TrajectoryState]
    
    # Intervention timing
    intervention_time_s: Optional[float] = None
    state_at_intervention: Optional[TrajectoryState] = None
    
    # Current behavior context (from VLM)
    current_maneuver: Optional[str] = None
    current_aggressiveness: str = "normal"
    upcoming_decisions: List[str] = field(default_factory=list)
    
    # Summary statistics
    avg_speed: float = 0.0
    max_speed: float = 0.0
    avg_acceleration: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'trajectory_states': [s.to_dict() for s in self.trajectory_states],
            'intervention_time_s': self.intervention_time_s,
            'state_at_intervention': self.state_at_intervention.to_dict() if self.state_at_intervention else None,
            'current_maneuver': self.current_maneuver,
            'current_aggressiveness': self.current_aggressiveness,
            'upcoming_decisions': self.upcoming_decisions,
            'avg_speed': self.avg_speed,
            'max_speed': self.max_speed,
            'avg_acceleration': self.avg_acceleration
        }


def build_trajectory_context(
    trajectory: np.ndarray,
    dt: float = 0.1,
    intervention_time_s: Optional[float] = None,
    current_maneuver: Optional[str] = None,
    aggressiveness: str = "normal",
    upcoming_decisions: Optional[List[str]] = None
) -> TrajectoryContext:
    """
    Build rich trajectory context from position data.
    
    Args:
        trajectory: Nx2 array of (x, y) positions
        dt: Time step between positions (default 0.1s for ScenarioNet)
        intervention_time_s: Time when intervention should occur
        current_maneuver: Current maneuver from VLM extraction
        aggressiveness: Driving aggressiveness level
        upcoming_decisions: List of upcoming decision points
        
    Returns:
        TrajectoryContext with computed velocities, accelerations, etc.
    """
    if trajectory is None or len(trajectory) < 2:
        return TrajectoryContext(
            trajectory_states=[],
            intervention_time_s=intervention_time_s,
            current_maneuver=current_maneuver,
            current_aggressiveness=aggressiveness,
            upcoming_decisions=upcoming_decisions or []
        )
    
    trajectory = np.array(trajectory)
    n_points = len(trajectory)
    
    states = []
    speeds = []
    accelerations = []
    
    for i in range(n_points):
        time_s = i * dt
        pos = (float(trajectory[i, 0]), float(trajectory[i, 1]))
        
        # Compute velocity (finite difference)
        if i < n_points - 1:
            vel = (trajectory[i+1] - trajectory[i]) / dt
            speed = float(np.linalg.norm(vel))
            heading = float(np.arctan2(vel[1], vel[0]))
        elif i > 0:
            vel = (trajectory[i] - trajectory[i-1]) / dt
            speed = float(np.linalg.norm(vel))
            heading = float(np.arctan2(vel[1], vel[0]))
        else:
            speed = 0.0
            heading = 0.0
        
        # Compute acceleration (second derivative)
        if i > 0 and i < n_points - 1:
            vel_prev = (trajectory[i] - trajectory[i-1]) / dt
            vel_next = (trajectory[i+1] - trajectory[i]) / dt
            speed_prev = float(np.linalg.norm(vel_prev))
            speed_next = float(np.linalg.norm(vel_next))
            acc = (speed_next - speed_prev) / (2 * dt)
        elif i == 0 and n_points > 2:
            vel_curr = (trajectory[1] - trajectory[0]) / dt
            vel_next = (trajectory[2] - trajectory[1]) / dt
            speed_curr = float(np.linalg.norm(vel_curr))
            speed_next = float(np.linalg.norm(vel_next))
            acc = (speed_next - speed_curr) / dt
        else:
            acc = 0.0
        
        # Compute yaw rate
        if i > 0 and i < n_points - 1:
            # Use previous and next headings
            vel_prev = (trajectory[i] - trajectory[i-1]) / dt
            vel_next = (trajectory[i+1] - trajectory[i]) / dt
            heading_prev = float(np.arctan2(vel_prev[1], vel_prev[0]))
            heading_next = float(np.arctan2(vel_next[1], vel_next[0]))
            # Handle angle wrapping
            yaw_diff = heading_next - heading_prev
            if yaw_diff > np.pi:
                yaw_diff -= 2 * np.pi
            elif yaw_diff < -np.pi:
                yaw_diff += 2 * np.pi
            yaw_rate = yaw_diff / (2 * dt)
        else:
            yaw_rate = 0.0
        
        state = TrajectoryState(
            time_s=time_s,
            position=pos,
            speed=speed,
            heading=heading,
            acceleration=acc,
            yaw_rate=yaw_rate
        )
        states.append(state)
        speeds.append(speed)
        accelerations.append(acc)
    
    # Find state at intervention time
    state_at_intervention = None
    if intervention_time_s is not None and states:
        # Find nearest state
        intervention_idx = min(int(intervention_time_s / dt), len(states) - 1)
        state_at_intervention = states[intervention_idx]
    
    return TrajectoryContext(
        trajectory_states=states,
        intervention_time_s=intervention_time_s,
        state_at_intervention=state_at_intervention,
        current_maneuver=current_maneuver,
        current_aggressiveness=aggressiveness,
        upcoming_decisions=upcoming_decisions or [],
        avg_speed=float(np.mean(speeds)) if speeds else 0.0,
        max_speed=float(np.max(speeds)) if speeds else 0.0,
        avg_acceleration=float(np.mean(accelerations)) if accelerations else 0.0
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class InterventionPhase:
    """
    A single phase of an intervention plan.
    
    Interventions can have multiple phases, e.g.:
    - Phase 1: Decelerate from 16 to 12 m/s (2 seconds)
    - Phase 2: Maintain 12 m/s (remainder of prediction)
    """
    phase_name: str
    start_time_s: float
    end_time_s: float
    acc_range: Tuple[float, float]  # (min, max) in m/s²
    yaw_range: Tuple[float, float]  # (min, max) in rad/s
    bias_strength_multiplier: float = 1.0  # Relative to base bias strength
    reasoning: str = ""
    
    @property
    def start_timestep(self) -> int:
        """Convert start time to BMT timestep (2 steps/second)."""
        return int(self.start_time_s * 2)
    
    @property
    def end_timestep(self) -> int:
        """Convert end time to BMT timestep (2 steps/second)."""
        return min(int(self.end_time_s * 2), 19)  # Cap at prediction horizon
    
    def to_dict(self) -> Dict:
        return {
            'phase_name': self.phase_name,
            'start_time_s': self.start_time_s,
            'end_time_s': self.end_time_s,
            'start_timestep': self.start_timestep,
            'end_timestep': self.end_timestep,
            'acc_range': list(self.acc_range),
            'yaw_range': list(self.yaw_range),
            'bias_strength_multiplier': self.bias_strength_multiplier,
            'reasoning': self.reasoning
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "InterventionPhase":
        return cls(
            phase_name=d.get('phase_name', d.get('phase', 'unknown')),
            start_time_s=d.get('start_time_s', d.get('start_time', 0)),
            end_time_s=d.get('end_time_s', d.get('end_time', 9.5)),
            acc_range=tuple(d.get('acc_range', [-10, 10])),
            yaw_range=tuple(d.get('yaw_range', [-0.15, 0.15])),
            bias_strength_multiplier=d.get('bias_strength_multiplier', 1.0),
            reasoning=d.get('reasoning', '')
        )


@dataclass
class InterventionPlan:
    """
    Complete intervention plan with multiple phases.
    """
    intervention_id: str
    intervention_type: str  # 'speed', 'maneuver', 'decision'
    phases: List[InterventionPhase]
    summary: str = ""
    llm_reasoning: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'intervention_id': self.intervention_id,
            'intervention_type': self.intervention_type,
            'phases': [p.to_dict() for p in self.phases],
            'summary': self.summary,
            'llm_reasoning': self.llm_reasoning
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "InterventionPlan":
        return cls(
            intervention_id=d.get('intervention_id', 'unknown'),
            intervention_type=d.get('intervention_type', 'unknown'),
            phases=[InterventionPhase.from_dict(p) for p in d.get('phases', [])],
            summary=d.get('summary', ''),
            llm_reasoning=d.get('llm_reasoning', '')
        )


# =============================================================================
# LLM Intervention Planner
# =============================================================================

class LLMInterventionPlanner:
    """
    Uses an LLM to generate context-aware intervention plans.
    
    Instead of hardcoding physics calculations, the LLM reasons about:
    - How long an intervention should last
    - What acceleration/yaw ranges are appropriate
    - Whether multiple phases are needed
    """
    
    # Vehicle dynamics constraints (shared with LLM for context)
    VEHICLE_CONSTRAINTS = {
        'max_comfortable_accel': 3.0,      # m/s²
        'max_hard_accel': 6.0,             # m/s²
        'max_comfortable_decel': -3.0,     # m/s²
        'max_hard_decel': -8.0,            # m/s² (emergency braking)
        'max_lane_change_yaw': 0.4,        # rad/s
        'max_turn_yaw': 0.9,               # rad/s
        'prediction_horizon_s': 9.5,       # seconds (19 timesteps * 0.5s)
        'timestep_duration_s': 0.5,        # seconds per timestep
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize the planner.
        
        Args:
            llm_client: GPT-4o client (or compatible). If None, uses fallback logic.
        """
        self.client = llm_client
        self.call_count = 0
    
    def plan_intervention(
        self,
        intervention: Dict,
        ego_state: Optional[Dict] = None,
        scenario_context: Optional[Dict] = None,
        trajectory_context: Optional[TrajectoryContext] = None,
        debug_output_dir: Optional[str] = None,
        intervention_idx: int = 0
    ) -> InterventionPlan:
        """
        Generate an intervention plan using the LLM.
        
        Args:
            intervention: Dict with 'variable', 'value', 'original_value', 'description', 'timestamp'
            ego_state: Dict with 'speed', 'heading', 'position' (optional, legacy)
            scenario_context: Dict with 'road_type', 'traffic_density', etc. (optional)
            trajectory_context: TrajectoryContext with full trajectory data and timing (preferred)
            debug_output_dir: Optional path to save LLM debug logs
            intervention_idx: Index of this intervention for logging
            
        Returns:
            InterventionPlan with one or more phases
        """
        from pathlib import Path
        
        # Determine intervention type
        int_type = self._infer_intervention_type(intervention)
        
        # Extract intervention timestamp
        intervention_time = intervention.get('timestamp')
        
        # If trajectory_context provided, use state at intervention time
        effective_ego_state = ego_state
        if trajectory_context and trajectory_context.state_at_intervention:
            state = trajectory_context.state_at_intervention
            effective_ego_state = {
                'speed': state.speed,
                'heading': state.heading,
                'acceleration': state.acceleration,
                'yaw_rate': state.yaw_rate,
                'position': list(state.position),
                'time_s': state.time_s
            }
        
        # If no LLM client, use physics-based fallback
        if self.client is None:
            logger.debug("No LLM client, using physics-based fallback")
            plan = self._fallback_plan(intervention, int_type, effective_ego_state, trajectory_context)
            
            # Log fallback plan if debug enabled
            if debug_output_dir:
                self._save_debug_log(
                    debug_output_dir, intervention_idx, intervention,
                    "(fallback - no LLM client)", 
                    "(fallback plan generated)", 
                    plan
                )
            
            return plan
        
        # Build prompt with trajectory context
        prompt = self._build_prompt(
            intervention, int_type, effective_ego_state, 
            scenario_context, trajectory_context
        )
        
        # Query LLM
        response = None
        try:
            response = self.client.complete(prompt, temperature=0.2, max_tokens=1500)
            self.call_count += 1
            
            # Parse response
            plan = self._parse_response(response, intervention, int_type)
            
            # Save debug log
            if debug_output_dir:
                self._save_debug_log(
                    debug_output_dir, intervention_idx, intervention,
                    prompt, response, plan
                )
            
            return plan
            
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback")
            plan = self._fallback_plan(intervention, int_type, effective_ego_state, trajectory_context)
            
            # Save debug log with error
            if debug_output_dir:
                self._save_debug_log(
                    debug_output_dir, intervention_idx, intervention,
                    prompt, f"ERROR: {e}\nResponse: {response}",
                    plan
                )
            
            return plan
    
    def _save_debug_log(
        self,
        output_dir: str,
        intervention_idx: int,
        intervention: Dict,
        prompt: str,
        response: str,
        plan: 'InterventionPlan'
    ):
        """Save LLM debug log to file."""
        from pathlib import Path
        
        try:
            debug_dir = Path(output_dir) / "llm_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            log_path = debug_dir / f"intervention_{intervention_idx + 1}_llm.json"
            
            debug_data = {
                'intervention_idx': intervention_idx + 1,
                'intervention': intervention,
                'prompt': prompt,
                'response': response,
                'parsed_plan': plan.to_dict() if plan else None,
                'phases_summary': []
            }
            
            if plan and plan.phases:
                for phase in plan.phases:
                    debug_data['phases_summary'].append({
                        'name': phase.phase_name,
                        'time_range': f"{phase.start_time_s:.1f}s - {phase.end_time_s:.1f}s",
                        'acc_range': f"[{phase.acc_range[0]:.2f}, {phase.acc_range[1]:.2f}] m/s²",
                        'yaw_range': f"[{phase.yaw_range[0]:.3f}, {phase.yaw_range[1]:.3f}] rad/s",
                        'yaw_direction': 'LEFT' if phase.yaw_range[0] > 0 or phase.yaw_range[1] > 0 else ('RIGHT' if phase.yaw_range[1] < 0 else 'STRAIGHT'),
                        'reasoning': phase.reasoning
                    })
            
            with open(log_path, 'w') as f:
                json.dump(debug_data, f, indent=2, default=str)
            
            logger.debug(f"Saved LLM debug log to {log_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save LLM debug log: {e}")
    
    def _infer_intervention_type(self, intervention: Dict) -> str:
        """Infer intervention type from variable name."""
        variable = intervention.get('variable', '').lower()
        
        if 'speed' in variable:
            return 'speed'
        if 'maneuver' in variable:
            return 'maneuver'
        if 'decision' in variable:
            return 'decision'
        
        # Check value for clues
        value = str(intervention.get('value', '')).lower()
        if any(x in value for x in ['lane_change', 'turn', 'straight', 'stop']):
            return 'maneuver'
        
        return 'generic'
    
    def _build_prompt(
        self,
        intervention: Dict,
        int_type: str,
        ego_state: Optional[Dict],
        scenario_context: Optional[Dict],
        trajectory_context: Optional[TrajectoryContext] = None
    ) -> str:
        """Build the LLM prompt for intervention planning with full trajectory context."""
        
        # Extract intervention details
        variable = intervention.get('variable', 'unknown')
        new_value = intervention.get('value', 'unknown')
        original_value = intervention.get('original_value', 'unknown')
        description = intervention.get('description', '')
        intervention_time = intervention.get('timestamp')
        
        # Build intervention timing section
        timing_section = ""
        if intervention_time is not None:
            timing_section = f"""
## Intervention Timing
- Intervention occurs at: {intervention_time:.1f} seconds into scenario
- Plan phases should start from t=0 (start of BMT prediction horizon)
- Note: The intervention time indicates WHEN the change should happen
"""
        
        # Build ego state section (using state at intervention time if available)
        if ego_state:
            ego_section = f"""
## Ego Vehicle State at Intervention Time
- Speed: {ego_state.get('speed', 'unknown'):.2f} m/s
- Heading: {ego_state.get('heading', 'unknown'):.3f} rad
- Acceleration: {ego_state.get('acceleration', 'unknown')} m/s²
- Yaw Rate: {ego_state.get('yaw_rate', 'unknown')} rad/s
- Position: {ego_state.get('position', 'unknown')}
"""
            if ego_state.get('time_s') is not None:
                ego_section += f"- Time: {ego_state.get('time_s'):.1f}s\n"
        else:
            ego_section = """
## Ego Vehicle State
- Not provided (use reasonable assumptions)
"""
        
        # Build trajectory history section
        trajectory_section = ""
        if trajectory_context and trajectory_context.trajectory_states:
            # Sample trajectory at key points (every 0.5s for conciseness)
            states = trajectory_context.trajectory_states
            sample_interval = max(1, len(states) // 20)  # Max ~20 rows
            
            trajectory_section = """
## Trajectory History (sampled)
| Time(s) | Speed(m/s) | Accel(m/s²) | Heading(rad) | Yaw Rate(rad/s) |
|---------|------------|-------------|--------------|-----------------|
"""
            for i, state in enumerate(states):
                if i % sample_interval == 0 or i == len(states) - 1:
                    marker = " ← INTERVENTION" if (intervention_time and 
                             abs(state.time_s - intervention_time) < 0.1) else ""
                    trajectory_section += (f"| {state.time_s:.1f} | {state.speed:.1f} | "
                                          f"{state.acceleration:+.2f} | {state.heading:.2f} | "
                                          f"{state.yaw_rate:+.3f} |{marker}\n")
            
            trajectory_section += f"""
### Trajectory Summary
- Average speed: {trajectory_context.avg_speed:.1f} m/s
- Maximum speed: {trajectory_context.max_speed:.1f} m/s
- Average acceleration: {trajectory_context.avg_acceleration:+.2f} m/s²
"""
            
            if trajectory_context.current_maneuver:
                trajectory_section += f"- Current maneuver: {trajectory_context.current_maneuver}\n"
            if trajectory_context.current_aggressiveness:
                trajectory_section += f"- Driving style: {trajectory_context.current_aggressiveness}\n"
            if trajectory_context.upcoming_decisions:
                trajectory_section += f"- Upcoming decisions: {', '.join(trajectory_context.upcoming_decisions)}\n"
        
        # Build context section
        if scenario_context:
            context_section = f"""
## Scenario Context
- Road type: {scenario_context.get('road_type', 'unknown')}
- Traffic density: {scenario_context.get('traffic_density', 'unknown')}
- Weather: {scenario_context.get('weather', 'clear')}
"""
        else:
            context_section = ""
        
        # Type-specific guidance
        if int_type == 'speed':
            type_guidance = """
## Speed Change Planning Guidelines
- Calculate the time needed: time = |speed_difference| / avg_deceleration
- Use comfortable deceleration (-2 to -3 m/s²) for normal changes
- Use harder deceleration (-4 to -6 m/s²) for urgent changes
- After reaching target speed, add a "maintain" phase with small acc range (-0.5 to 0.5)
- Keep yaw range small for straight driving (±0.1 rad/s)
- NEVER apply deceleration bias longer than needed to reach target speed
- Consider the CURRENT speed from trajectory, not original speed
"""
        elif int_type == 'maneuver':
            type_guidance = """
## Maneuver Planning Guidelines

CRITICAL - Yaw Sign Convention:
- POSITIVE yaw rate = turning LEFT (counter-clockwise)
- NEGATIVE yaw rate = turning RIGHT (clockwise)
- For lane_change_LEFT: use POSITIVE yaw values (e.g., [0.08, 0.25])
- For lane_change_RIGHT: use NEGATIVE yaw values (e.g., [-0.25, -0.08])
- For turn_LEFT: use POSITIVE yaw values (e.g., [0.3, 0.6])
- For turn_RIGHT: use NEGATIVE yaw values (e.g., [-0.6, -0.3])

For lane changes:
- Preparation phase (0.5s): slight yaw in the correct direction
- Execution phase (2-3s): yaw magnitude 0.08-0.25 rad/s (use correct sign!)
- Stabilization phase (1s): return yaw to ~0, maintain speed

For turns:
- Approach phase: decelerate if needed
- Turn phase: yaw magnitude 0.3-0.6 rad/s (use correct sign for direction!)
- Exit phase: accelerate back to speed, reduce yaw

Consider current trajectory behavior when planning transitions.
"""
        elif int_type == 'decision':
            type_guidance = """
## Decision Point Planning Guidelines
For yield/stop decisions:
- Deceleration phase: apply braking until safe speed or stop
- Wait/proceed phase: maintain low speed or accelerate away

For proceed decisions:
- Maintain or slight acceleration

Consider current speed and acceleration when planning.
"""
        else:
            type_guidance = """
## General Planning Guidelines
- Break intervention into logical phases
- Each phase should have a clear purpose
- Use realistic acceleration and yaw ranges
- Consider current vehicle state for smooth transitions
"""
        
        prompt = f"""You are a vehicle dynamics expert planning trajectory interventions for a driving simulator.

{timing_section}
{ego_section}
{trajectory_section}
{context_section}

## Intervention to Plan
- Variable: {variable}
- New Value: {new_value}
- Original Value: {original_value}
- Description: {description}

## Vehicle Dynamics Constraints
- Max comfortable acceleration: {self.VEHICLE_CONSTRAINTS['max_comfortable_accel']} m/s²
- Max hard acceleration: {self.VEHICLE_CONSTRAINTS['max_hard_accel']} m/s²
- Max comfortable deceleration: {self.VEHICLE_CONSTRAINTS['max_comfortable_decel']} m/s²
- Max emergency deceleration: {self.VEHICLE_CONSTRAINTS['max_hard_decel']} m/s²
- Max lane change yaw rate: {self.VEHICLE_CONSTRAINTS['max_lane_change_yaw']} rad/s
- Max turn yaw rate: {self.VEHICLE_CONSTRAINTS['max_turn_yaw']} rad/s
- Prediction horizon: {self.VEHICLE_CONSTRAINTS['prediction_horizon_s']} seconds

{type_guidance}

## Task
Create an intervention plan with one or more phases. For each phase, specify:
1. phase_name: Short descriptive name
2. start_time_s: Start time in seconds (0.0 = start of prediction)
3. end_time_s: End time in seconds (max 9.5)
4. acc_range: [min, max] acceleration in m/s²
5. yaw_range: [min, max] yaw rate in rad/s
6. bias_strength_multiplier: 0.5-1.5 (how strongly to apply this phase)
7. reasoning: Brief explanation (include physics calculations)

IMPORTANT:
- Phases must not overlap in time
- Total time should not exceed 9.5 seconds
- Use physics to calculate appropriate durations based on CURRENT state
- For speed changes, calculate: time_needed = |current_speed - target_speed| / avg_decel
- Consider current acceleration - if already decelerating, less intervention needed

Output your response as JSON:
```json
{{
  "phases": [
    {{
      "phase_name": "...",
      "start_time_s": 0.0,
      "end_time_s": ...,
      "acc_range": [..., ...],
      "yaw_range": [..., ...],
      "bias_strength_multiplier": 1.0,
      "reasoning": "..."
    }}
  ],
  "summary": "Brief overall summary",
  "reasoning": "Overall reasoning for the plan"
}}
```
"""
        return prompt
    
    def _parse_response(
        self,
        response: str,
        intervention: Dict,
        int_type: str
    ) -> InterventionPlan:
        """Parse LLM response into an InterventionPlan."""
        
        # Extract JSON from response
        try:
            # Find JSON block
            json_match = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_match = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_match = response[start:end].strip()
            
            data = json.loads(json_match)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return self._fallback_plan(intervention, int_type, None)
        
        # Build phases
        phases = []
        for p in data.get('phases', []):
            phase = InterventionPhase(
                phase_name=p.get('phase_name', 'unknown'),
                start_time_s=float(p.get('start_time_s', 0)),
                end_time_s=float(p.get('end_time_s', 9.5)),
                acc_range=tuple(p.get('acc_range', [-10, 10])),
                yaw_range=tuple(p.get('yaw_range', [-0.15, 0.15])),
                bias_strength_multiplier=float(p.get('bias_strength_multiplier', 1.0)),
                reasoning=p.get('reasoning', '')
            )
            phases.append(phase)
        
        # Validate phases
        phases = self._validate_phases(phases)
        
        return InterventionPlan(
            intervention_id=intervention.get('variable', 'unknown'),
            intervention_type=int_type,
            phases=phases,
            summary=data.get('summary', ''),
            llm_reasoning=data.get('reasoning', '')
        )
    
    def _validate_phases(self, phases: List[InterventionPhase]) -> List[InterventionPhase]:
        """Validate and fix phase timing issues."""
        if not phases:
            return phases
        
        # Sort by start time
        phases = sorted(phases, key=lambda p: p.start_time_s)
        
        # Fix overlaps and clamp to horizon
        for i, phase in enumerate(phases):
            # Clamp to prediction horizon
            phase.end_time_s = min(phase.end_time_s, 9.5)
            phase.start_time_s = min(phase.start_time_s, 9.5)
            
            # Ensure start < end
            if phase.start_time_s >= phase.end_time_s:
                phase.end_time_s = min(phase.start_time_s + 1.0, 9.5)
            
            # Fix overlap with previous phase
            if i > 0:
                prev_end = phases[i-1].end_time_s
                if phase.start_time_s < prev_end:
                    phase.start_time_s = prev_end
        
        # Remove invalid phases (zero duration)
        phases = [p for p in phases if p.start_time_s < p.end_time_s]
        
        return phases
    
    def _fallback_plan(
        self,
        intervention: Dict,
        int_type: str,
        ego_state: Optional[Dict],
        trajectory_context: Optional[TrajectoryContext] = None
    ) -> InterventionPlan:
        """
        Generate a physics-based fallback plan when LLM is unavailable.
        
        Uses simple calculations based on intervention type and trajectory context.
        """
        variable = intervention.get('variable', 'unknown')
        new_value = intervention.get('value')
        original_value = intervention.get('original_value')
        
        phases = []
        
        if int_type == 'speed':
            phases = self._fallback_speed_plan(new_value, original_value, ego_state, trajectory_context)
        elif int_type == 'maneuver':
            phases = self._fallback_maneuver_plan(new_value, original_value, trajectory_context)
        elif int_type == 'decision':
            phases = self._fallback_decision_plan(new_value, original_value, trajectory_context)
        else:
            # Generic fallback
            phases = [InterventionPhase(
                phase_name='intervention',
                start_time_s=0.0,
                end_time_s=6.0,
                acc_range=(-2.0, 2.0),
                yaw_range=(-0.2, 0.2),
                reasoning='Generic intervention phase'
            )]
        
        return InterventionPlan(
            intervention_id=variable,
            intervention_type=int_type,
            phases=phases,
            summary=f"Fallback plan for {int_type}",
            llm_reasoning="Generated using physics-based fallback (no LLM)"
        )
    
    def _fallback_speed_plan(
        self,
        new_value: Any,
        original_value: Any,
        ego_state: Optional[Dict],
        trajectory_context: Optional[TrajectoryContext] = None
    ) -> List[InterventionPhase]:
        """Physics-based fallback for speed interventions using trajectory context."""
        phases = []
        
        try:
            new_speed = float(new_value)
            orig_speed = float(original_value) if original_value else None
        except (TypeError, ValueError):
            # Can't parse speeds, use generic
            return [InterventionPhase(
                phase_name='speed_change',
                start_time_s=0.0,
                end_time_s=4.0,
                acc_range=(-3.0, 3.0),
                yaw_range=(-0.1, 0.1),
                reasoning='Generic speed change'
            )]
        
        # Get CURRENT speed from trajectory context (preferred) or ego_state
        current_speed = None
        current_accel = 0.0
        
        if trajectory_context and trajectory_context.state_at_intervention:
            current_speed = trajectory_context.state_at_intervention.speed
            current_accel = trajectory_context.state_at_intervention.acceleration
            logger.debug(f"Using trajectory context: speed={current_speed:.1f}, accel={current_accel:.1f}")
        elif ego_state:
            current_speed = ego_state.get('speed')
            current_accel = ego_state.get('acceleration', 0.0)
        
        # Use current speed if available, otherwise fall back to original value
        if current_speed is None:
            current_speed = orig_speed if orig_speed is not None else 15.0
        
        speed_diff = new_speed - current_speed
        
        if abs(speed_diff) < 1.0:
            # Minimal change, just maintain
            phases.append(InterventionPhase(
                phase_name='maintain',
                start_time_s=0.0,
                end_time_s=9.5,
                acc_range=(-0.5, 0.5),
                yaw_range=(-0.1, 0.1),
                reasoning=f'Current speed {current_speed:.1f} near target {new_speed:.1f}, maintain'
            ))
        else:
            # Calculate transition duration based on CURRENT speed
            if speed_diff < 0:
                # Decelerating
                avg_decel = -2.5  # Comfortable deceleration
                
                # If already decelerating, account for it
                if current_accel < -0.5:
                    # Already braking, need less additional deceleration
                    effective_decel = avg_decel - current_accel * 0.3
                    effective_decel = max(effective_decel, -4.0)
                else:
                    effective_decel = avg_decel
                
                time_needed = abs(speed_diff) / abs(effective_decel)
                time_needed = min(time_needed, 6.0)  # Cap at 6 seconds
                
                acc_range = (-4.0, -1.5)
            else:
                # Accelerating
                avg_accel = 2.0  # Comfortable acceleration
                
                # If already accelerating, account for it
                if current_accel > 0.5:
                    effective_accel = avg_accel + current_accel * 0.3
                    effective_accel = min(effective_accel, 4.0)
                else:
                    effective_accel = avg_accel
                
                time_needed = speed_diff / effective_accel
                time_needed = min(time_needed, 6.0)
                
                acc_range = (1.0, 4.0)
            
            # Phase 1: Transition
            phases.append(InterventionPhase(
                phase_name='transition',
                start_time_s=0.0,
                end_time_s=time_needed,
                acc_range=acc_range,
                yaw_range=(-0.1, 0.1),
                reasoning=f'{"Decelerate" if speed_diff < 0 else "Accelerate"} from {current_speed:.1f} to {new_speed:.1f} m/s ({time_needed:.1f}s)'
            ))
            
            # Phase 2: Maintain
            if time_needed < 9.0:
                phases.append(InterventionPhase(
                    phase_name='maintain',
                    start_time_s=time_needed,
                    end_time_s=9.5,
                    acc_range=(-0.5, 0.5),
                    yaw_range=(-0.1, 0.1),
                    bias_strength_multiplier=0.7,
                    reasoning=f'Maintain target speed {new_speed:.1f} m/s'
                ))
        
        return phases
    
    def _fallback_maneuver_plan(
        self,
        new_value: Any,
        original_value: Any,
        trajectory_context: Optional[TrajectoryContext] = None
    ) -> List[InterventionPhase]:
        """Physics-based fallback for maneuver interventions."""
        maneuver = str(new_value).lower().replace(' ', '_').replace('-', '_')
        
        if 'lane_change' in maneuver:
            # Lane change phases
            direction = 1.0 if 'left' in maneuver else -1.0
            yaw_base = 0.15 * direction
            
            return [
                InterventionPhase(
                    phase_name='initiate',
                    start_time_s=0.0,
                    end_time_s=1.0,
                    acc_range=(-0.5, 0.5),
                    yaw_range=(0.05 * direction, 0.12 * direction) if direction > 0 else (0.12 * direction, 0.05 * direction),
                    reasoning='Begin lane change'
                ),
                InterventionPhase(
                    phase_name='execute',
                    start_time_s=1.0,
                    end_time_s=3.5,
                    acc_range=(-1.0, 1.0),
                    yaw_range=(0.08 * direction, 0.25 * direction) if direction > 0 else (0.25 * direction, 0.08 * direction),
                    reasoning='Execute lane change maneuver'
                ),
                InterventionPhase(
                    phase_name='stabilize',
                    start_time_s=3.5,
                    end_time_s=5.0,
                    acc_range=(-0.5, 0.5),
                    yaw_range=(-0.1, 0.1),
                    bias_strength_multiplier=0.8,
                    reasoning='Stabilize in new lane'
                )
            ]
        
        elif 'turn' in maneuver:
            direction = 1.0 if 'left' in maneuver else -1.0
            
            return [
                InterventionPhase(
                    phase_name='approach',
                    start_time_s=0.0,
                    end_time_s=1.5,
                    acc_range=(-3.0, -1.0),
                    yaw_range=(-0.1, 0.1),
                    reasoning='Decelerate for turn'
                ),
                InterventionPhase(
                    phase_name='turn',
                    start_time_s=1.5,
                    end_time_s=4.5,
                    acc_range=(-1.5, 1.0),
                    yaw_range=(0.25 * direction, 0.6 * direction) if direction > 0 else (0.6 * direction, 0.25 * direction),
                    reasoning='Execute turn'
                ),
                InterventionPhase(
                    phase_name='exit',
                    start_time_s=4.5,
                    end_time_s=6.5,
                    acc_range=(0.5, 2.5),
                    yaw_range=(-0.15, 0.15),
                    reasoning='Accelerate out of turn'
                )
            ]
        
        elif 'stop' in maneuver:
            return [
                InterventionPhase(
                    phase_name='brake',
                    start_time_s=0.0,
                    end_time_s=4.0,
                    acc_range=(-6.0, -2.0),
                    yaw_range=(-0.1, 0.1),
                    reasoning='Braking to stop'
                ),
                InterventionPhase(
                    phase_name='stopped',
                    start_time_s=4.0,
                    end_time_s=9.5,
                    acc_range=(-0.5, 0.0),
                    yaw_range=(-0.05, 0.05),
                    reasoning='Maintain stopped position'
                )
            ]
        
        elif 'straight' in maneuver:
            return [InterventionPhase(
                phase_name='straight',
                start_time_s=0.0,
                end_time_s=9.5,
                acc_range=(-1.0, 1.0),
                yaw_range=(-0.1, 0.1),
                reasoning='Maintain straight heading'
            )]
        
        else:
            # Generic maneuver
            return [InterventionPhase(
                phase_name='maneuver',
                start_time_s=0.0,
                end_time_s=6.0,
                acc_range=(-2.0, 2.0),
                yaw_range=(-0.3, 0.3),
                reasoning='Generic maneuver'
            )]
    
    def _fallback_decision_plan(
        self,
        new_value: Any,
        original_value: Any,
        trajectory_context: Optional[TrajectoryContext] = None
    ) -> List[InterventionPhase]:
        """Physics-based fallback for decision interventions."""
        decision = str(new_value).lower()
        
        if 'yield' in decision or 'stop' in decision:
            return [
                InterventionPhase(
                    phase_name='decelerate',
                    start_time_s=0.0,
                    end_time_s=3.0,
                    acc_range=(-5.0, -2.0),
                    yaw_range=(-0.1, 0.1),
                    reasoning='Yield/stop decision - decelerate'
                ),
                InterventionPhase(
                    phase_name='wait',
                    start_time_s=3.0,
                    end_time_s=9.5,
                    acc_range=(-1.0, 0.5),
                    yaw_range=(-0.1, 0.1),
                    bias_strength_multiplier=0.6,
                    reasoning='Wait for safe gap'
                )
            ]
        
        elif 'proceed' in decision or 'accept' in decision:
            return [InterventionPhase(
                phase_name='proceed',
                start_time_s=0.0,
                end_time_s=9.5,
                acc_range=(0.0, 3.0),
                yaw_range=(-0.1, 0.1),
                reasoning='Proceed/accept gap - maintain or accelerate'
            )]
        
        else:
            return [InterventionPhase(
                phase_name='decision',
                start_time_s=0.0,
                end_time_s=6.0,
                acc_range=(-2.0, 2.0),
                yaw_range=(-0.15, 0.15),
                reasoning='Generic decision phase'
            )]


# =============================================================================
# Mock Client for Testing
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing without API calls."""
    
    def complete(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        """Return a mock response based on prompt content."""
        
        # Detect intervention type from prompt
        if "speed" in prompt.lower() and "m/s" in prompt.lower():
            # Extract speeds if possible
            import re
            speeds = re.findall(r'(\d+\.?\d*)\s*m/s', prompt)
            if len(speeds) >= 2:
                orig = float(speeds[0])
                new = float(speeds[1])
                diff = new - orig
                
                if diff < 0:
                    time_needed = min(abs(diff) / 2.5, 5.0)
                    return json.dumps({
                        "phases": [
                            {
                                "phase_name": "decelerate",
                                "start_time_s": 0.0,
                                "end_time_s": time_needed,
                                "acc_range": [-3.5, -1.5],
                                "yaw_range": [-0.1, 0.1],
                                "bias_strength_multiplier": 1.0,
                                "reasoning": f"Decelerate for {time_needed:.1f}s to reduce speed"
                            },
                            {
                                "phase_name": "maintain",
                                "start_time_s": time_needed,
                                "end_time_s": 9.5,
                                "acc_range": [-0.5, 0.5],
                                "yaw_range": [-0.1, 0.1],
                                "bias_strength_multiplier": 0.7,
                                "reasoning": "Maintain target speed"
                            }
                        ],
                        "summary": f"Speed reduction from {orig:.1f} to {new:.1f} m/s",
                        "reasoning": "Calculated deceleration time based on speed difference"
                    })
        
        # Default mock response
        return json.dumps({
            "phases": [
                {
                    "phase_name": "intervention",
                    "start_time_s": 0.0,
                    "end_time_s": 5.0,
                    "acc_range": [-2.0, 2.0],
                    "yaw_range": [-0.15, 0.15],
                    "bias_strength_multiplier": 1.0,
                    "reasoning": "Mock intervention phase"
                }
            ],
            "summary": "Mock intervention plan",
            "reasoning": "Generated by mock client"
        })


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLM Intervention Planner")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM client")
    parser.add_argument("--test-speed", action="store_true", help="Test speed intervention")
    parser.add_argument("--test-maneuver", action="store_true", help="Test maneuver intervention")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create planner
    if args.mock:
        client = MockLLMClient()
    else:
        client = None  # Use fallback
    
    planner = LLMInterventionPlanner(client)
    
    if args.test_speed:
        intervention = {
            'variable': 'ego_initial_speed',
            'value': 12.0,
            'original_value': 16.0,
            'description': 'Reduce ego speed from 16 to 12 m/s'
        }
        ego_state = {'speed': 16.0, 'heading': 0.1}
        
        plan = planner.plan_intervention(intervention, ego_state)
        print("\n=== Speed Intervention Plan ===")
        print(json.dumps(plan.to_dict(), indent=2))
    
    if args.test_maneuver:
        intervention = {
            'variable': 'maneuver_0',
            'value': 'lane_change_left',
            'original_value': 'straight',
            'description': 'Change lane to the left'
        }
        
        plan = planner.plan_intervention(intervention)
        print("\n=== Maneuver Intervention Plan ===")
        print(json.dumps(plan.to_dict(), indent=2))
    
    if not args.test_speed and not args.test_maneuver:
        print("Use --test-speed or --test-maneuver to run tests")
        print("Add --mock to use mock LLM client")
