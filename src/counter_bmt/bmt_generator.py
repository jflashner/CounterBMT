"""
CounterBMT Generator - Complete BMT Integration

This module provides the complete implementation for counterfactual trajectory generation
with BMT, including all token bias classes.

Classes:
    - TokenBias: Dataclass for token bias specification
    - MotionTokenSpace: BMT token vocabulary (1089 tokens)
    - InterventionCompiler: Translates DAG interventions → TokenBias objects
    - BiasedTokenSampler: Applies biases to logits during sampling
    - CounterBMTGenerator: High-level interface for counterfactual generation
"""

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np

logger = logging.getLogger(__name__)

# PyTorch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Token Bias Data Structures
# =============================================================================

@dataclass
class TokenBias:
    """
    Specification for biasing specific tokens during sampling.
    
    Attributes:
        token_ids: List of token IDs to bias
        bias_value: Logit bias to apply (positive = encourage, negative = discourage)
        timestep_range: (start, end) timesteps to apply bias (inclusive start, exclusive end)
        agent_id: Optional agent ID (0 = ego, None = all agents)
        description: Human-readable description
    """
    token_ids: List[int]
    bias_value: float
    timestep_range: Tuple[int, int] = (0, 19)  # Full BMT prediction horizon
    agent_id: Optional[int] = None
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'token_ids': self.token_ids,
            'bias_value': self.bias_value,
            'timestep_range': self.timestep_range,
            'agent_id': self.agent_id,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TokenBias":
        return cls(
            token_ids=d['token_ids'],
            bias_value=d['bias_value'],
            timestep_range=tuple(d.get('timestep_range', (0, 19))),
            agent_id=d.get('agent_id'),
            description=d.get('description', '')
        )


# =============================================================================
# Motion Token Space
# =============================================================================

class MotionTokenSpace:
    """
    BMT motion token vocabulary.
    
    Token space: 33 acceleration bins × 33 yaw rate bins = 1089 tokens
    - Acceleration: [-10, 10] m/s² 
    - Yaw rate: [-π/2, π/2] rad/s
    
    Token index formula: token_id = acc_bin * 33 + yaw_bin
    """
    
    def __init__(
        self,
        n_acc_bins: int = 33,
        n_yaw_bins: int = 33,
        acc_range: Tuple[float, float] = (-10.0, 10.0),
        yaw_range: Tuple[float, float] = (-math.pi/2, math.pi/2)
    ):
        self.n_acc_bins = n_acc_bins
        self.n_yaw_bins = n_yaw_bins
        self.n_tokens = n_acc_bins * n_yaw_bins
        
        self.acc_min, self.acc_max = acc_range
        self.yaw_min, self.yaw_max = yaw_range
        
        # Bin edges
        self.acc_edges = np.linspace(self.acc_min, self.acc_max, n_acc_bins + 1)
        self.yaw_edges = np.linspace(self.yaw_min, self.yaw_max, n_yaw_bins + 1)
        
        # Bin centers
        self.acc_centers = (self.acc_edges[:-1] + self.acc_edges[1:]) / 2
        self.yaw_centers = (self.yaw_edges[:-1] + self.yaw_edges[1:]) / 2
        
        # Precompute behavior token sets
        self._behavior_tokens = self._compute_behavior_tokens()
    
    def token_to_action(self, token_id: int) -> Tuple[float, float]:
        """Convert token ID to (acceleration, yaw_rate)."""
        acc_bin = token_id // self.n_yaw_bins
        yaw_bin = token_id % self.n_yaw_bins
        return float(self.acc_centers[acc_bin]), float(self.yaw_centers[yaw_bin])
    
    def action_to_token(self, acc: float, yaw: float) -> int:
        """Convert (acceleration, yaw_rate) to token ID."""
        acc_bin = np.clip(np.searchsorted(self.acc_edges[1:], acc), 0, self.n_acc_bins - 1)
        yaw_bin = np.clip(np.searchsorted(self.yaw_edges[1:], yaw), 0, self.n_yaw_bins - 1)
        return int(acc_bin * self.n_yaw_bins + yaw_bin)
    
    def _compute_behavior_tokens(self) -> Dict[str, List[int]]:
        """Precompute token sets for common behaviors."""
        behaviors = {
            'stop': [],           # Strong deceleration
            'hard_brake': [],     # Emergency braking
            'decelerate': [],     # Moderate slowing
            'maintain': [],       # Constant speed
            'accelerate': [],     # Speed up
            'straight': [],       # No steering
            'turn_left': [],      # Turning left (any)
            'turn_right': [],     # Turning right (any)
            'swerve_left': [],    # Sharp left (emergency)
            'swerve_right': [],   # Sharp right (emergency)
            # Lane change specific tokens with proper yaw ranges
            'lane_change_left_gentle': [],    # Gentle left lane change (0.05-0.15 rad/s)
            'lane_change_left': [],           # Normal left lane change (0.08-0.25 rad/s)
            'lane_change_left_aggressive': [],# Aggressive left lane change (0.20-0.40 rad/s)
            'lane_change_right_gentle': [],   # Gentle right lane change
            'lane_change_right': [],          # Normal right lane change
            'lane_change_right_aggressive': [],# Aggressive right lane change
            # Turn tokens (intersection-style, sharper than lane changes)
            'turn_left_moderate': [],         # Normal intersection turn (yaw 0.30-0.60 rad/s)
            'turn_right_moderate': [],        
            'turn_left_sharp': [],            # Sharp turn, U-turn (yaw 0.50-0.90 rad/s)
            'turn_right_sharp': [],           
            'turn_left_with_brake': [],       # Turn with deceleration (realistic approach)
            'turn_right_with_brake': [],
        }
        
        for token_id in range(self.n_tokens):
            acc, yaw = self.token_to_action(token_id)
            
            # Acceleration-based
            if acc < -5.0:
                behaviors['stop'].append(token_id)
            if acc < -7.0:
                behaviors['hard_brake'].append(token_id)
            if acc < -2.0:
                behaviors['decelerate'].append(token_id)
            if -1.5 < acc < 1.5:
                behaviors['maintain'].append(token_id)
            if acc > 2.0:
                behaviors['accelerate'].append(token_id)
            
            # Steering-based (original broad categories)
            if abs(yaw) < 0.15:
                behaviors['straight'].append(token_id)
            if yaw > 0.2:
                behaviors['turn_left'].append(token_id)
            if yaw < -0.2:
                behaviors['turn_right'].append(token_id)
            if yaw > 0.5:
                behaviors['swerve_left'].append(token_id)
            if yaw < -0.5:
                behaviors['swerve_right'].append(token_id)
            
            # Lane change specific (constrained yaw + moderate acceleration)
            # These are based on vehicle dynamics for comfortable lane changes
            is_moderate_acc = -2.0 < acc < 2.0
            
            # Gentle lane change: yaw 0.05-0.15 rad/s (very smooth, 5-7 second lane change)
            if is_moderate_acc and 0.05 < yaw < 0.15:
                behaviors['lane_change_left_gentle'].append(token_id)
            if is_moderate_acc and -0.15 < yaw < -0.05:
                behaviors['lane_change_right_gentle'].append(token_id)
            
            # Normal lane change: yaw 0.08-0.25 rad/s (3-5 second lane change)
            if is_moderate_acc and 0.08 < yaw < 0.25:
                behaviors['lane_change_left'].append(token_id)
            if is_moderate_acc and -0.25 < yaw < -0.08:
                behaviors['lane_change_right'].append(token_id)
            
            # Aggressive lane change: yaw 0.20-0.40 rad/s (2-3 second lane change)
            if is_moderate_acc and 0.20 < yaw < 0.40:
                behaviors['lane_change_left_aggressive'].append(token_id)
            if is_moderate_acc and -0.40 < yaw < -0.20:
                behaviors['lane_change_right_aggressive'].append(token_id)
            
            # ===== TURN TOKENS (intersection-style) =====
            # Turns require higher yaw rates than lane changes
            # and often involve deceleration for safety
            
            # Moderate turn: yaw 0.30-0.60 rad/s (typical intersection turn)
            # Allow slight decel to maintain speed through turn (-3 to +1.5 m/s²)
            is_turn_acc = -3.0 < acc < 1.5
            
            if is_turn_acc and 0.30 < yaw < 0.60:
                behaviors['turn_left_moderate'].append(token_id)
            if is_turn_acc and -0.60 < yaw < -0.30:
                behaviors['turn_right_moderate'].append(token_id)
            
            # Sharp turn: yaw 0.50-0.90 rad/s (tight turns, U-turns)
            # Usually with braking for control
            is_sharp_turn_acc = acc < 0  # Require some braking for sharp turns
            
            if is_sharp_turn_acc and 0.50 < yaw < 0.90:
                behaviors['turn_left_sharp'].append(token_id)
            if is_sharp_turn_acc and -0.90 < yaw < -0.50:
                behaviors['turn_right_sharp'].append(token_id)
            
            # Turn with braking: moderate yaw + deceleration (realistic turn approach)
            # This captures the "slow down before turn" behavior
            is_braking = acc < -1.5
            
            if is_braking and 0.25 < yaw < 0.60:
                behaviors['turn_left_with_brake'].append(token_id)
            if is_braking and -0.60 < yaw < -0.25:
                behaviors['turn_right_with_brake'].append(token_id)
        
        return behaviors
    
    def get_tokens_by_behavior(self, behavior: str) -> List[int]:
        """Get token IDs for a named behavior."""
        behavior = behavior.lower().replace(' ', '_').replace('-', '_')
        
        # Direct match
        if behavior in self._behavior_tokens:
            return self._behavior_tokens[behavior]
        
        # Fuzzy matching - prioritize lane_change over generic turn
        if 'lane_change' in behavior or 'lanechange' in behavior:
            # Lane change specific matching
            if 'left' in behavior:
                if 'gentle' in behavior:
                    return self._behavior_tokens['lane_change_left_gentle']
                elif 'aggressive' in behavior:
                    return self._behavior_tokens['lane_change_left_aggressive']
                else:
                    return self._behavior_tokens['lane_change_left']
            elif 'right' in behavior:
                if 'gentle' in behavior:
                    return self._behavior_tokens['lane_change_right_gentle']
                elif 'aggressive' in behavior:
                    return self._behavior_tokens['lane_change_right_aggressive']
                else:
                    return self._behavior_tokens['lane_change_right']
        
        # Turn-specific matching (intersection turns)
        if 'turn' in behavior and ('moderate' in behavior or 'sharp' in behavior or 'brake' in behavior):
            if 'left' in behavior:
                if 'moderate' in behavior:
                    return self._behavior_tokens['turn_left_moderate']
                elif 'sharp' in behavior:
                    return self._behavior_tokens['turn_left_sharp']
                elif 'brake' in behavior or 'with_brake' in behavior:
                    return self._behavior_tokens['turn_left_with_brake']
            elif 'right' in behavior:
                if 'moderate' in behavior:
                    return self._behavior_tokens['turn_right_moderate']
                elif 'sharp' in behavior:
                    return self._behavior_tokens['turn_right_sharp']
                elif 'brake' in behavior or 'with_brake' in behavior:
                    return self._behavior_tokens['turn_right_with_brake']
        
        if 'stop' in behavior or 'brake' in behavior:
            return self._behavior_tokens['stop']
        if 'decel' in behavior or 'slow' in behavior:
            return self._behavior_tokens['decelerate']
        if 'accel' in behavior or 'speed' in behavior:
            return self._behavior_tokens['accelerate']
        if 'maintain' in behavior or 'constant' in behavior:
            return self._behavior_tokens['maintain']
        if 'straight' in behavior or 'forward' in behavior:
            return self._behavior_tokens['straight']
        if 'left' in behavior:
            if 'swerve' in behavior:
                return self._behavior_tokens['swerve_left']
            return self._behavior_tokens['turn_left']
        if 'right' in behavior:
            if 'swerve' in behavior:
                return self._behavior_tokens['swerve_right']
            return self._behavior_tokens['turn_right']
        
        logger.warning(f"Unknown behavior: {behavior}")
        return []
    
    def get_lane_change_tokens(
        self, 
        direction: str = 'right',
        style: str = 'normal'
    ) -> List[int]:
        """
        Get tokens appropriate for lane change maneuvers.
        
        Uses vehicle dynamics-based yaw rate ranges:
        - gentle: 0.05-0.15 rad/s (smooth 5-7 second lane change)
        - normal: 0.08-0.25 rad/s (comfortable 3-5 second lane change)
        - aggressive: 0.20-0.40 rad/s (quick 2-3 second lane change)
        
        Args:
            direction: 'left' or 'right'
            style: 'gentle', 'normal', or 'aggressive'
            
        Returns:
            List of token IDs for the specified lane change maneuver
        """
        key = f"lane_change_{direction}"
        if style == 'gentle':
            key += '_gentle'
        elif style == 'aggressive':
            key += '_aggressive'
        # 'normal' style uses the base key without suffix
        
        if key in self._behavior_tokens:
            return self._behavior_tokens[key]
        
        logger.warning(f"Unknown lane change: {key}")
        return []
    
    def get_turn_tokens(
        self,
        direction: str = 'right',
        style: str = 'moderate'
    ) -> List[int]:
        """
        Get tokens appropriate for intersection turn maneuvers.
        
        Uses vehicle dynamics-based yaw rate ranges (higher than lane changes):
        - moderate: yaw 0.30-0.60 rad/s (typical intersection turn)
        - sharp: yaw 0.50-0.90 rad/s (tight turn, U-turn)
        - with_brake: yaw 0.25-0.60 rad/s + deceleration (realistic approach)
        
        Args:
            direction: 'left' or 'right'
            style: 'moderate', 'sharp', or 'with_brake'
            
        Returns:
            List of token IDs for the specified turn maneuver
        """
        key = f"turn_{direction}_{style}"
        
        if key in self._behavior_tokens:
            return self._behavior_tokens[key]
        
        logger.warning(f"Unknown turn: {key}")
        return []
    
    def describe_token_set(self, behavior: str) -> str:
        """Get a human-readable description of a token set's yaw/acc ranges."""
        tokens = self.get_tokens_by_behavior(behavior)
        if not tokens:
            return f"No tokens for behavior: {behavior}"
        
        accs = [self.token_to_action(t)[0] for t in tokens]
        yaws = [self.token_to_action(t)[1] for t in tokens]
        
        return (f"{behavior}: {len(tokens)} tokens, "
                f"acc=[{min(accs):.2f}, {max(accs):.2f}] m/s², "
                f"yaw=[{min(yaws):.2f}, {max(yaws):.2f}] rad/s")
    
    def get_tokens_by_constraint(
        self,
        acc_min: Optional[float] = None,
        acc_max: Optional[float] = None,
        yaw_min: Optional[float] = None,
        yaw_max: Optional[float] = None
    ) -> List[int]:
        """Get tokens matching acceleration/yaw constraints."""
        tokens = []
        for token_id in range(self.n_tokens):
            acc, yaw = self.token_to_action(token_id)
            
            if acc_min is not None and acc < acc_min:
                continue
            if acc_max is not None and acc > acc_max:
                continue
            if yaw_min is not None and yaw < yaw_min:
                continue
            if yaw_max is not None and yaw > yaw_max:
                continue
            
            tokens.append(token_id)
        return tokens


# =============================================================================
# Intervention Compiler
# =============================================================================

class InterventionCompiler:
    """
    Compiles high-level interventions into token biases.
    
    Translates DAG interventions (e.g., "change maneuver to stop") 
    into specific token biases for BMT sampling.
    
    Uses granular token sets for reliable maneuver generation:
    - Lane changes: lane_change_{left,right}_{gentle,normal,aggressive}
    - Turns: turn_{left,right}_{moderate,sharp,with_brake}
    """
    
    # Default bias strengths
    DEFAULT_ENCOURAGE_BIAS = 8.0  # Increased for more reliable intervention
    DEFAULT_DISCOURAGE_BIAS = -3.0
    
    # BMT timestep = 0.5s, prediction horizon = 19 steps (9.5s)
    DT = 0.5
    MAX_TIMESTEPS = 19
    
    # Mapping from VLM maneuver types to token set keys
    # Format: maneuver_type -> (token_key_prefix, uses_style)
    MANEUVER_TOKEN_MAPPING = {
        # Lane changes use style (gentle/normal/aggressive)
        'lane_change_left': ('lane_change_left', True),
        'lane_change_right': ('lane_change_right', True),
        # Turns use style (moderate/sharp/with_brake)
        'left_turn': ('turn_left', True),
        'right_turn': ('turn_right', True),
        # Simple maneuvers without style variants
        'straight': ('straight', False),
        'stop': ('stop', False),
        'decelerate': ('decelerate', False),
        'accelerate': ('accelerate', False),
    }
    
    # Mapping from VLM aggressiveness to token style
    AGGRESSIVENESS_TO_LANE_STYLE = {
        'passive': 'gentle',
        'normal': 'normal', 
        'aggressive': 'aggressive',
    }
    
    AGGRESSIVENESS_TO_TURN_STYLE = {
        'passive': 'with_brake',  # Slow, careful turns
        'normal': 'moderate',
        'aggressive': 'sharp',
    }
    
    def __init__(
        self,
        token_space: Optional[MotionTokenSpace] = None,
        llm_planner: Optional[Any] = None
    ):
        """
        Initialize the intervention compiler.
        
        Args:
            token_space: MotionTokenSpace instance for token lookup
            llm_planner: Optional LLMInterventionPlanner for smart phase planning
        """
        self.token_space = token_space or MotionTokenSpace()
        self.llm_planner = llm_planner
        
        # Intervention type handlers
        self._handlers = {
            'maneuver': self._compile_maneuver_intervention,
            'decision': self._compile_decision_intervention,
            'speed': self._compile_speed_intervention,
            'ego_state': self._compile_ego_state_intervention,
        }
    
    def compile_from_dag_intervention(
        self,
        intervention: Dict,
        encourage_bias: float = None,
        discourage_bias: float = None,
        ego_state: Optional[Dict] = None,
        scenario_context: Optional[Dict] = None,
        use_llm_planning: bool = True,
        trajectory_context: Optional[Any] = None,
        debug_output_dir: Optional[str] = None,
        intervention_idx: int = 0
    ) -> List[TokenBias]:
        """
        Compile a DAG intervention dict to token biases.
        
        Args:
            intervention: Dict with keys:
                - variable: node ID (e.g., "maneuver_0", "decision_1")
                - value: new value to encourage
                - original_value: value to discourage (optional)
                - description: human description (optional)
                - aggressiveness: optional style hint ("passive", "normal", "aggressive")
                - timestamp: optional timestamp for time-based bias
            encourage_bias: Bias for encouraging tokens (default: 8.0)
            discourage_bias: Bias for discouraging tokens (default: -3.0)
            ego_state: Optional dict with ego vehicle state (speed, heading, position)
            scenario_context: Optional dict with scenario context (road_type, traffic, etc.)
            use_llm_planning: Whether to use LLM for smart phase planning (default: True)
            trajectory_context: Optional TrajectoryContext with full trajectory data and timing
            debug_output_dir: Optional path to save LLM debug logs
            intervention_idx: Index of this intervention for logging
            
        Returns:
            List of TokenBias objects
        """
        encourage = encourage_bias or self.DEFAULT_ENCOURAGE_BIAS
        discourage = discourage_bias or self.DEFAULT_DISCOURAGE_BIAS
        
        variable = intervention.get('variable', '')
        new_value = intervention.get('value', '')
        original_value = intervention.get('original_value')
        aggressiveness = intervention.get('aggressiveness', 'normal')
        timestamp = intervention.get('timestamp')
        
        # Determine intervention type from variable name
        int_type = self._infer_intervention_type(variable)
        
        # Try LLM-based planning if available and enabled
        if use_llm_planning and self.llm_planner is not None:
            try:
                plan = self.llm_planner.plan_intervention(
                    intervention=intervention,
                    ego_state=ego_state,
                    scenario_context=scenario_context,
                    trajectory_context=trajectory_context,
                    debug_output_dir=debug_output_dir,
                    intervention_idx=intervention_idx
                )
                
                if plan and plan.phases:
                    biases = self._compile_from_plan(plan, encourage, discourage)
                    if biases:
                        logger.info(f"LLM planning generated {len(biases)} bias phases for '{variable}'")
                        return biases
                        
            except Exception as e:
                logger.warning(f"LLM planning failed for '{variable}': {e}, using fallback")
        
        # Fallback to hardcoded handlers
        # Special handling for maneuver interventions to pass extra params
        if int_type == 'maneuver':
            return self._compile_maneuver_intervention(
                variable=variable,
                new_value=new_value,
                original_value=original_value,
                encourage_bias=encourage,
                discourage_bias=discourage,
                aggressiveness=aggressiveness,
                timestamp=timestamp
            )
        
        handler = self._handlers.get(int_type, self._compile_generic_intervention)
        
        return handler(
            variable=variable,
            new_value=new_value,
            original_value=original_value,
            encourage_bias=encourage,
            discourage_bias=discourage
        )
    
    def _compile_from_plan(
        self,
        plan: Any,  # InterventionPlan from llm_intervention_planner
        encourage_bias: float,
        discourage_bias: float
    ) -> List[TokenBias]:
        """
        Convert an LLM-generated intervention plan to TokenBias objects.
        
        Each phase in the plan becomes a TokenBias with tokens matching
        the phase's acceleration and yaw constraints.
        
        For lateral movement phases (lane changes, turns), we expand the 
        acceleration constraint to ensure enough tokens are biased.
        """
        biases = []
        
        # Minimum number of tokens per phase for effective biasing
        MIN_TOKENS_FOR_EFFECTIVE_BIAS = 8
        
        # Default acceleration ranges for different phase types
        LANE_CHANGE_ACC_RANGE = (-2.0, 2.0)  # For lane changes, acc is flexible
        TURN_ACC_RANGE = (-4.0, 2.0)  # Turns often involve braking
        
        for phase in plan.phases:
            acc_min, acc_max = phase.acc_range
            yaw_min, yaw_max = phase.yaw_range
            
            # Fix inverted ranges (LLM sometimes returns min > max for negative values)
            if acc_min > acc_max:
                acc_min, acc_max = acc_max, acc_min
                logger.debug(f"Phase '{phase.phase_name}': Swapped inverted acc range")
            if yaw_min > yaw_max:
                yaw_min, yaw_max = yaw_max, yaw_min
                logger.debug(f"Phase '{phase.phase_name}': Swapped inverted yaw range")
            
            # Detect if this is a lateral movement phase (significant yaw)
            # Lane change yaw: 0.08-0.40 rad/s, Turn yaw: 0.25-0.90 rad/s
            is_lateral_phase = abs(yaw_min) > 0.05 or abs(yaw_max) > 0.05
            is_turn_phase = abs(yaw_min) > 0.25 or abs(yaw_max) > 0.25
            
            # Get tokens with original constraints
            tokens = self.token_space.get_tokens_by_constraint(
                acc_min=acc_min,
                acc_max=acc_max,
                yaw_min=yaw_min,
                yaw_max=yaw_max
            )
            
            # If too few tokens for lateral phases, expand acceleration range
            # For lane changes and turns, the yaw is what matters, not exact acceleration
            if is_lateral_phase and len(tokens) < MIN_TOKENS_FOR_EFFECTIVE_BIAS:
                if is_turn_phase:
                    expanded_acc_range = TURN_ACC_RANGE
                else:
                    expanded_acc_range = LANE_CHANGE_ACC_RANGE
                
                logger.info(f"Phase '{phase.phase_name}': Expanding acc range from "
                           f"[{acc_min:.2f}, {acc_max:.2f}] to {expanded_acc_range} "
                           f"for effective lateral biasing")
                
                tokens = self.token_space.get_tokens_by_constraint(
                    acc_min=expanded_acc_range[0],
                    acc_max=expanded_acc_range[1],
                    yaw_min=yaw_min,
                    yaw_max=yaw_max
                )
                
                # Update for logging
                acc_min, acc_max = expanded_acc_range
            
            if not tokens:
                logger.warning(f"No tokens for phase '{phase.phase_name}' with "
                             f"acc=[{acc_min:.2f}, {acc_max:.2f}], "
                             f"yaw=[{yaw_min:.3f}, {yaw_max:.3f}]")
                continue
            
            # Apply bias strength multiplier
            phase_bias = encourage_bias * phase.bias_strength_multiplier
            
            biases.append(TokenBias(
                token_ids=tokens,
                bias_value=phase_bias,
                timestep_range=(phase.start_timestep, phase.end_timestep),
                description=f"{phase.phase_name}: {phase.reasoning}"
            ))
            
            logger.debug(f"Phase '{phase.phase_name}': {len(tokens)} tokens, "
                        f"acc=[{acc_min:.2f}, {acc_max:.2f}], "
                        f"yaw=[{yaw_min:.3f}, {yaw_max:.3f}], "
                        f"steps {phase.start_timestep}-{phase.end_timestep}, "
                        f"bias={phase_bias:.1f}")
        
        return biases
    
    def _infer_intervention_type(self, variable: str) -> str:
        """Infer intervention type from variable name."""
        variable_lower = variable.lower()
        
        if 'maneuver' in variable_lower:
            return 'maneuver'
        if 'decision' in variable_lower:
            return 'decision'
        if 'speed' in variable_lower:
            return 'speed'
        if 'ego' in variable_lower:
            return 'ego_state'
        
        return 'generic'
    
    def map_maneuver_to_tokens(
        self,
        maneuver_type: str,
        aggressiveness: str = 'normal'
    ) -> List[int]:
        """
        Map a VLM-extracted maneuver to the appropriate token set.
        
        Uses granular token sets based on maneuver type and aggressiveness:
        - Lane changes: gentle/normal/aggressive based on aggressiveness
        - Turns: with_brake/moderate/sharp based on aggressiveness
        
        Args:
            maneuver_type: Maneuver type from VLM (e.g., "lane_change_left", "left_turn")
            aggressiveness: Aggressiveness from VLM ("passive", "normal", "aggressive")
            
        Returns:
            List of token IDs for the specified maneuver
        """
        # Normalize maneuver type
        maneuver_type = maneuver_type.lower().replace(' ', '_').replace('-', '_')
        aggressiveness = aggressiveness.lower() if aggressiveness else 'normal'
        
        # Look up mapping
        if maneuver_type in self.MANEUVER_TOKEN_MAPPING:
            prefix, uses_style = self.MANEUVER_TOKEN_MAPPING[maneuver_type]
            
            if uses_style:
                # Determine style based on maneuver type and aggressiveness
                if 'lane_change' in maneuver_type:
                    style = self.AGGRESSIVENESS_TO_LANE_STYLE.get(aggressiveness, 'normal')
                    # Use get_lane_change_tokens for lane changes
                    direction = 'left' if 'left' in maneuver_type else 'right'
                    return self.token_space.get_lane_change_tokens(direction, style)
                elif 'turn' in prefix:
                    style = self.AGGRESSIVENESS_TO_TURN_STYLE.get(aggressiveness, 'moderate')
                    # Use get_turn_tokens for turns
                    direction = 'left' if 'left' in maneuver_type else 'right'
                    return self.token_space.get_turn_tokens(direction, style)
            else:
                # Simple behavior without style
                return self.token_space.get_tokens_by_behavior(prefix)
        
        # Fallback to generic behavior lookup
        logger.debug(f"No mapping for maneuver '{maneuver_type}', using generic lookup")
        return self.token_space.get_tokens_by_behavior(maneuver_type)
    
    def _compile_maneuver_intervention(
        self,
        variable: str,
        new_value: str,
        original_value: Optional[str],
        encourage_bias: float,
        discourage_bias: float,
        aggressiveness: str = 'normal',
        timestamp: Optional[float] = None
    ) -> List[TokenBias]:
        """
        Compile maneuver change intervention using granular token sets.
        
        Args:
            variable: Node ID (e.g., "maneuver_0")
            new_value: New maneuver value (e.g., "lane_change_left")
            original_value: Original maneuver to discourage
            encourage_bias: Bias strength for encouraging tokens
            discourage_bias: Bias strength for discouraging tokens
            aggressiveness: Maneuver aggressiveness ("passive", "normal", "aggressive")
            timestamp: Optional timestamp for time-based bias range
        """
        biases = []
        
        # Determine timestep range
        if timestamp is not None:
            # Convert timestamp to BMT prediction steps (2 steps/second)
            start_step = max(0, int(timestamp * 2) - 2)
            end_step = min(start_step + 8, self.MAX_TIMESTEPS)
        else:
            # Default: use maneuver index heuristic
            try:
                maneuver_idx = int(variable.split('_')[-1])
                # Each maneuver ~2 seconds = 4 steps
                start_step = min(maneuver_idx * 4, self.MAX_TIMESTEPS - 4)
                end_step = min(start_step + 8, self.MAX_TIMESTEPS)
            except (ValueError, IndexError):
                start_step, end_step = 0, 8
        
        # Get tokens using granular mapping
        encourage_tokens = self.map_maneuver_to_tokens(new_value, aggressiveness)
        
        if encourage_tokens:
            token_info = f"{len(encourage_tokens)} tokens"
            biases.append(TokenBias(
                token_ids=encourage_tokens,
                bias_value=encourage_bias,
                timestep_range=(start_step, end_step),
                description=f"Encourage {new_value} ({aggressiveness}) [{token_info}]"
            ))
            logger.info(f"Mapped '{new_value}' ({aggressiveness}) -> {len(encourage_tokens)} tokens, "
                       f"steps {start_step}-{end_step}")
        else:
            logger.warning(f"No tokens found for maneuver: {new_value}")
        
        # Discourage original behavior if different
        if original_value and original_value != new_value:
            discourage_tokens = self.map_maneuver_to_tokens(original_value, aggressiveness)
            # Remove overlap with encourage tokens
            discourage_tokens = [t for t in discourage_tokens if t not in encourage_tokens]
            if discourage_tokens:
                biases.append(TokenBias(
                    token_ids=discourage_tokens,
                    bias_value=discourage_bias,
                    timestep_range=(start_step, end_step),
                    description=f"Discourage {original_value}"
                ))
        
        return biases
    
    def _compile_decision_intervention(
        self,
        variable: str,
        new_value: str,
        original_value: Optional[str],
        encourage_bias: float,
        discourage_bias: float
    ) -> List[TokenBias]:
        """Compile decision change intervention."""
        biases = []
        
        # Map decisions to behaviors
        decision_to_behavior = {
            'yield': 'decelerate',
            'stop': 'stop',
            'proceed': 'maintain',
            'brake': 'hard_brake',
            'swerve': 'swerve_left',
            'swerve_left': 'swerve_left',
            'swerve_right': 'swerve_right',
            'accept_gap': 'accelerate',
            'reject_gap': 'decelerate',
        }
        
        # Determine timestep from decision index
        try:
            decision_idx = int(variable.split('_')[-1])
            start_step = min(decision_idx * 3, self.MAX_TIMESTEPS - 4)
            end_step = min(start_step + 6, self.MAX_TIMESTEPS)
        except (ValueError, IndexError):
            start_step, end_step = 0, 6
        
        # Get behavior for new decision
        new_behavior = decision_to_behavior.get(new_value.lower(), new_value)
        encourage_tokens = self.token_space.get_tokens_by_behavior(new_behavior)
        
        if encourage_tokens:
            biases.append(TokenBias(
                token_ids=encourage_tokens,
                bias_value=encourage_bias,
                timestep_range=(start_step, end_step),
                description=f"Decision: {new_value}"
            ))
        
        # Discourage original
        if original_value:
            orig_behavior = decision_to_behavior.get(original_value.lower(), original_value)
            discourage_tokens = self.token_space.get_tokens_by_behavior(orig_behavior)
            discourage_tokens = [t for t in discourage_tokens if t not in encourage_tokens]
            if discourage_tokens:
                biases.append(TokenBias(
                    token_ids=discourage_tokens,
                    bias_value=discourage_bias,
                    timestep_range=(start_step, end_step),
                    description=f"Discourage: {original_value}"
                ))
        
        return biases
    
    def _compile_speed_intervention(
        self,
        variable: str,
        new_value: Any,
        original_value: Optional[Any],
        encourage_bias: float,
        discourage_bias: float
    ) -> List[TokenBias]:
        """
        Compile speed change intervention.
        
        Uses realistic acceleration ranges for vehicles:
        - Gentle acceleration: 0.5-2.0 m/s² (comfortable, normal driving)
        - Moderate acceleration: 2.0-4.0 m/s² (assertive)
        - Hard acceleration: 4.0-6.0 m/s² (aggressive, near performance limit)
        - Gentle deceleration: -0.5 to -2.0 m/s² (comfortable coasting/braking)
        - Moderate deceleration: -2.0 to -4.0 m/s² (normal braking)
        - Hard braking: -4.0 to -8.0 m/s² (emergency braking, ~0.5-0.8g)
        
        Also adds bias for straight driving (low yaw) to prevent erratic steering.
        """
        biases = []
        
        try:
            new_speed = float(new_value)
            orig_speed = float(original_value) if original_value else None
        except (TypeError, ValueError):
            # Handle string values like "slower", "faster"
            if isinstance(new_value, str):
                if 'slow' in new_value.lower() or 'reduce' in new_value.lower():
                    return self._compile_maneuver_intervention(
                        variable, 'decelerate', 'maintain', encourage_bias, discourage_bias
                    )
                elif 'fast' in new_value.lower() or 'increase' in new_value.lower():
                    return self._compile_maneuver_intervention(
                        variable, 'accelerate', 'maintain', encourage_bias, discourage_bias
                    )
            return biases
        
        # Calculate required speed change
        speed_diff = new_speed - (orig_speed or new_speed)
        
        # Determine realistic acceleration range based on speed ratio
        if orig_speed is not None and orig_speed > 0:
            speed_ratio = new_speed / orig_speed
            
            if speed_ratio < 0.3:
                # Emergency stop: hard braking -6 to -10 m/s²
                acc_min, acc_max = -10.0, -6.0
                behavior_desc = "emergency_brake"
            elif speed_ratio < 0.6:
                # Strong deceleration: -4 to -6 m/s²
                acc_min, acc_max = -6.0, -4.0
                behavior_desc = "hard_brake"
            elif speed_ratio < 0.85:
                # Moderate deceleration: -2 to -4 m/s²
                acc_min, acc_max = -4.0, -2.0
                behavior_desc = "decelerate"
            elif speed_ratio < 1.15:
                # Maintain speed: -1.5 to +1.5 m/s²
                acc_min, acc_max = -1.5, 1.5
                behavior_desc = "maintain"
            elif speed_ratio < 1.3:
                # Gentle acceleration: 1 to 3 m/s²
                acc_min, acc_max = 1.0, 3.0
                behavior_desc = "gentle_accelerate"
            elif speed_ratio < 1.5:
                # Moderate acceleration: 2 to 4 m/s²
                acc_min, acc_max = 2.0, 4.0
                behavior_desc = "accelerate"
            else:
                # Strong acceleration: 3 to 6 m/s² (realistic max for most vehicles)
                acc_min, acc_max = 3.0, 6.0
                behavior_desc = "hard_accelerate"
        else:
            # No original speed reference
            if new_speed < 3:
                acc_min, acc_max = -6.0, -2.0
                behavior_desc = "stop"
            elif new_speed < 10:
                acc_min, acc_max = -1.5, 1.5
                behavior_desc = "maintain_slow"
            else:
                acc_min, acc_max = -1.5, 1.5
                behavior_desc = "maintain"
        
        # Get tokens with realistic acceleration AND low yaw (straight driving)
        # This prevents erratic steering during speed changes
        yaw_limit = 0.15  # Low yaw for straight driving
        
        speed_tokens = self.token_space.get_tokens_by_constraint(
            acc_min=acc_min,
            acc_max=acc_max,
            yaw_min=-yaw_limit,
            yaw_max=yaw_limit
        )
        
        if speed_tokens:
            biases.append(TokenBias(
                token_ids=speed_tokens,
                bias_value=encourage_bias,
                timestep_range=(0, 12),  # Apply for first 6 seconds
                description=f"Speed: {orig_speed:.1f} -> {new_speed:.1f} m/s ({behavior_desc}, acc=[{acc_min:.1f},{acc_max:.1f}])"
            ))
            logger.info(f"Speed intervention: {len(speed_tokens)} tokens, "
                       f"acc=[{acc_min:.1f}, {acc_max:.1f}], yaw=[{-yaw_limit:.2f}, {yaw_limit:.2f}]")
        else:
            # Fallback to behavior-based tokens if constraint-based fails
            logger.warning(f"No tokens for acc=[{acc_min}, {acc_max}], falling back to behavior")
            if speed_ratio < 0.8:
                behavior = 'decelerate'
            elif speed_ratio > 1.2:
                behavior = 'accelerate'
            else:
                behavior = 'maintain'
            tokens = self.token_space.get_tokens_by_behavior(behavior)
            if tokens:
                biases.append(TokenBias(
                    token_ids=tokens,
                    bias_value=encourage_bias,
                    timestep_range=(0, 10),
                    description=f"Speed: {orig_speed} -> {new_speed} m/s (fallback: {behavior})"
                ))
        
        # Also add a weaker bias for straight driving throughout the prediction
        straight_tokens = self.token_space.get_tokens_by_behavior('straight')
        if straight_tokens:
            biases.append(TokenBias(
                token_ids=straight_tokens,
                bias_value=encourage_bias * 0.3,  # Weaker bias
                timestep_range=(0, 19),  # Full horizon
                description="Maintain straight heading during speed change"
            ))
        
        return biases
    
    def _compile_ego_state_intervention(
        self,
        variable: str,
        new_value: Any,
        original_value: Optional[Any],
        encourage_bias: float,
        discourage_bias: float
    ) -> List[TokenBias]:
        """Compile ego state intervention (speed, position)."""
        if 'speed' in variable.lower():
            return self._compile_speed_intervention(
                variable, new_value, original_value, encourage_bias, discourage_bias
            )
        
        # For position/heading changes, use generic
        return self._compile_generic_intervention(
            variable, new_value, original_value, encourage_bias, discourage_bias
        )
    
    def _compile_generic_intervention(
        self,
        variable: str,
        new_value: Any,
        original_value: Optional[Any],
        encourage_bias: float,
        discourage_bias: float
    ) -> List[TokenBias]:
        """Generic intervention handler for unknown types."""
        biases = []
        
        # Try to interpret new_value as a behavior
        if isinstance(new_value, str):
            tokens = self.token_space.get_tokens_by_behavior(new_value)
            if tokens:
                biases.append(TokenBias(
                    token_ids=tokens,
                    bias_value=encourage_bias,
                    timestep_range=(0, 8),
                    description=f"Generic: {new_value}"
                ))
        
        return biases
    
    def compile_sequence(
        self,
        sequence: Any,  # InterventionSequence from dag_constructor
        encourage_bias: float = None
    ) -> List[TokenBias]:
        """
        Compile a sequential intervention chain into token biases.
        
        This enables complex driving "scripts" like:
        "accelerate → lane_change_right → turn_right"
        
        Each step in the sequence gets biases applied during its time window.
        
        Args:
            sequence: InterventionSequence with steps containing:
                - maneuver: str (e.g., "accelerate", "lane_change_right")
                - start_time_s: float
                - duration_s: float
                - intensity: str ("gentle", "normal", "aggressive")
            encourage_bias: Bias strength (default: DEFAULT_ENCOURAGE_BIAS)
            
        Returns:
            List of TokenBias objects, one or more per step
        """
        encourage = encourage_bias or self.DEFAULT_ENCOURAGE_BIAS
        biases = []
        
        # Import to access step attributes
        steps = sequence.steps if hasattr(sequence, 'steps') else []
        
        for i, step in enumerate(steps):
            maneuver = step.maneuver if hasattr(step, 'maneuver') else step.get('maneuver', 'straight')
            start_time = step.start_time_s if hasattr(step, 'start_time_s') else step.get('start_time_s', 0)
            duration = step.duration_s if hasattr(step, 'duration_s') else step.get('duration_s', 2.0)
            intensity = step.intensity if hasattr(step, 'intensity') else step.get('intensity', 'normal')
            
            # Convert times to BMT timesteps (2 steps per second)
            start_step = max(0, int(start_time * 2))
            end_step = min(int((start_time + duration) * 2), self.MAX_TIMESTEPS)
            
            # Skip if timestep range is invalid
            if start_step >= end_step:
                logger.warning(f"Skipping sequence step {i}: invalid time range "
                              f"{start_time}s-{start_time + duration}s (steps {start_step}-{end_step})")
                continue
            
            # Map maneuver to tokens
            tokens = self._get_sequence_step_tokens(maneuver, intensity)
            
            if tokens:
                biases.append(TokenBias(
                    token_ids=tokens,
                    bias_value=encourage,
                    timestep_range=(start_step, end_step),
                    description=f"Seq[{i}] {maneuver} ({intensity}) @ {start_time:.1f}-{start_time + duration:.1f}s"
                ))
                logger.info(f"Sequence step {i}: {maneuver} ({intensity}) -> "
                           f"{len(tokens)} tokens, steps {start_step}-{end_step}")
            else:
                logger.warning(f"No tokens found for sequence step {i}: {maneuver}")
        
        return biases
    
    def _get_sequence_step_tokens(self, maneuver: str, intensity: str) -> List[int]:
        """
        Get tokens for a sequence step maneuver.
        
        Args:
            maneuver: Maneuver type (e.g., "accelerate", "lane_change_right")
            intensity: Intensity level ("gentle", "normal", "aggressive")
            
        Returns:
            List of token IDs
        """
        # Map intensity to style names
        intensity_to_lane_style = {
            'gentle': 'gentle',
            'normal': 'normal',
            'aggressive': 'aggressive'
        }
        intensity_to_turn_style = {
            'gentle': 'with_brake',
            'normal': 'moderate',
            'aggressive': 'sharp'
        }
        
        # Handle different maneuver types
        if maneuver.startswith('lane_change'):
            direction = 'right' if 'right' in maneuver else 'left'
            style = intensity_to_lane_style.get(intensity, 'normal')
            return self.token_space.get_lane_change_tokens(direction, style)
        
        elif maneuver in ('turn_left', 'left_turn'):
            style = intensity_to_turn_style.get(intensity, 'moderate')
            return self.token_space.get_turn_tokens('left', style)
        
        elif maneuver in ('turn_right', 'right_turn'):
            style = intensity_to_turn_style.get(intensity, 'moderate')
            return self.token_space.get_turn_tokens('right', style)
        
        elif maneuver == 'hard_brake':
            return self.token_space.get_tokens_by_constraint(
                acc_min=-8.0, acc_max=-4.0, yaw_min=-0.1, yaw_max=0.1
            )
        
        elif maneuver == 'swerve':
            # Quick yaw change
            return self.token_space.get_tokens_by_constraint(
                acc_min=-2.0, acc_max=2.0, yaw_min=0.2, yaw_max=0.5
            ) + self.token_space.get_tokens_by_constraint(
                acc_min=-2.0, acc_max=2.0, yaw_min=-0.5, yaw_max=-0.2
            )
        
        elif maneuver in ('accelerate', 'hard_accelerate'):
            acc_min = 2.0 if maneuver == 'hard_accelerate' else 1.0
            acc_max = 4.0 if maneuver == 'hard_accelerate' else 3.0
            return self.token_space.get_tokens_by_constraint(
                acc_min=acc_min, acc_max=acc_max, yaw_min=-0.1, yaw_max=0.1
            )
        
        elif maneuver in ('decelerate', 'slow_down'):
            return self.token_space.get_tokens_by_constraint(
                acc_min=-3.0, acc_max=-0.5, yaw_min=-0.1, yaw_max=0.1
            )
        
        else:
            # Try generic behavior lookup
            return self.token_space.get_tokens_by_behavior(maneuver)


# =============================================================================
# Biased Token Sampler
# =============================================================================

class BiasedTokenSampler:
    """
    Applies token biases during BMT sampling.
    
    Used as a hook in autoregressive_rollout.sample_action().
    """
    
    def __init__(
        self, 
        token_biases: List[TokenBias],
        token_space: Optional[MotionTokenSpace] = None
    ):
        """
        Args:
            token_biases: List of TokenBias objects
            token_space: Optional token space (for validation)
        """
        self.token_biases = token_biases
        self.token_space = token_space or MotionTokenSpace()
        
        # Build timestep lookup for efficient bias application
        self._timestep_biases: Dict[int, List[TokenBias]] = {}
        for bias in token_biases:
            start, end = bias.timestep_range
            for t in range(start, end):
                if t not in self._timestep_biases:
                    self._timestep_biases[t] = []
                self._timestep_biases[t].append(bias)
        
        logger.debug(f"BiasedTokenSampler: {len(token_biases)} biases, "
                     f"timesteps {min(self._timestep_biases.keys(), default=0)}-"
                     f"{max(self._timestep_biases.keys(), default=0)}")
    
    def apply_bias(
        self, 
        logits: Any,  # torch.Tensor or np.ndarray
        timestep: int,
        agent_id: Optional[int] = None
    ) -> Any:
        """
        Apply biases to logits for the given timestep.
        
        Args:
            logits: Logits tensor/array, shape [..., n_tokens]
            timestep: Current prediction timestep
            agent_id: Optional agent ID for agent-specific biasing
            
        Returns:
            Modified logits (same type as input)
        """
        # Get biases for this timestep
        active_biases = self._timestep_biases.get(timestep, [])
        
        if not active_biases:
            return logits
        
        # Handle both torch and numpy
        is_torch = HAS_TORCH and isinstance(logits, torch.Tensor)
        
        if is_torch:
            logits = logits.clone()
        else:
            logits = np.array(logits, copy=True)
        
        for bias in active_biases:
            # Check agent filter
            if bias.agent_id is not None and agent_id is not None:
                if bias.agent_id != agent_id:
                    continue
            
            # Apply bias to specified tokens
            for token_id in bias.token_ids:
                if token_id < logits.shape[-1]:
                    logits[..., token_id] += bias.bias_value
        
        return logits
    
    def get_active_biases(self, timestep: int) -> List[TokenBias]:
        """Get biases active at a specific timestep."""
        return self._timestep_biases.get(timestep, [])
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"BiasedTokenSampler: {len(self.token_biases)} bias groups"]
        for i, bias in enumerate(self.token_biases):
            lines.append(f"  [{i}] {len(bias.token_ids)} tokens, "
                        f"bias={bias.bias_value:+.1f}, "
                        f"steps {bias.timestep_range[0]}-{bias.timestep_range[1]}: "
                        f"{bias.description}")
        return "\n".join(lines)


# =============================================================================
# CounterBMT Generator
# =============================================================================

class CounterBMTGenerator:
    """
    Complete implementation for counterfactual trajectory generation with BMT.
    
    Usage:
        # Initialize with checkpoint path
        generator = CounterBMTGenerator.from_checkpoint("path/to/checkpoint.ckpt")
        
        # Or initialize with pre-loaded model
        generator = CounterBMTGenerator(model=pl_model, config=config, tokenizer=tokenizer)
        
        # Generate counterfactual
        result = generator.generate_counterfactual(scenario_data, intervention)
    """
    
    def __init__(
        self,
        model=None,
        config=None,
        tokenizer=None,
        device: str = 'cuda'
    ):
        """
        Initialize generator with model components.
        
        Args:
            model: BMT lightning model (MotionLMLightning)
            config: BMT config object
            tokenizer: BMT tokenizer
            device: Device for computation
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        
        # Token space and compiler
        self.token_space = MotionTokenSpace()
        self.compiler = InterventionCompiler(self.token_space)
        self.sampler = None
        
        self._is_loaded = model is not None
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cuda',
        config_override: Optional[Dict] = None
    ) -> "CounterBMTGenerator":
        """
        Load generator from BMT checkpoint.
        
        Args:
            checkpoint_path: Path to BMT checkpoint (.ckpt)
            device: Device for model
            config_override: Optional config overrides
            
        Returns:
            Initialized CounterBMTGenerator
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for model loading")
        
        from bmt.utils import utils as bmt_utils
        
        logger.info(f"Loading BMT model from: {checkpoint_path}")
        
        # Load model
        pl_model = bmt_utils.get_model(checkpoint_path=checkpoint_path)
        pl_model = pl_model.eval()
        
        # Get config and tokenizer
        config = pl_model.config
        tokenizer = pl_model.model.tokenizer
        
        # Apply config overrides
        if config_override:
            for key, val in config_override.items():
                setattr(config, key, val)
        
        # Move to device
        actual_device = device if torch.cuda.is_available() else 'cpu'
        pl_model = pl_model.to(actual_device)
        
        logger.info(f"Model loaded on {actual_device}")
        logger.info(f"Sampling: {config.SAMPLING.SAMPLING_METHOD}, temp={config.SAMPLING.TEMPERATURE}")
        
        generator = cls(
            model=pl_model,
            config=config,
            tokenizer=tokenizer,
            device=actual_device
        )
        generator._is_loaded = True
        
        return generator
    
    def generate_counterfactual(
        self,
        scenario_data: Dict,
        intervention: Dict,
        n_samples: int = 1,
        temperature: Optional[float] = None,
        return_baseline: bool = True
    ) -> Dict:
        """
        Generate counterfactual trajectory for an intervention.
        
        Args:
            scenario_data: Scenario dict (from InfgenDataset or preprocessed)
            intervention: Intervention dict from DAG with keys:
                - variable: node ID
                - value: counterfactual value
                - original_value: original value  
            n_samples: Number of counterfactual trajectory samples
            temperature: Override sampling temperature (None = use config)
            return_baseline: Also generate unbiased baseline trajectory
            
        Returns:
            Dict with:
                - counterfactual_trajectories: List of trajectory arrays
                - baseline_trajectory: Baseline trajectory (if return_baseline)
                - intervention: Input intervention
                - token_biases: Compiled biases info
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Use from_checkpoint() or provide model in __init__")
        
        # Compile intervention to token biases
        token_biases = self.compiler.compile_from_dag_intervention(intervention)
        
        logger.info(f"Compiled {len(token_biases)} token biases for intervention: "
                    f"{intervention.get('variable')} = {intervention.get('value')}")
        
        # Create biased sampler
        self.sampler = BiasedTokenSampler(token_biases, self.token_space)
        
        # Prepare results
        results = {
            'intervention': intervention,
            'token_biases': [
                {
                    'n_tokens': len(b.token_ids),
                    'bias_value': b.bias_value,
                    'timestep_range': b.timestep_range
                }
                for b in token_biases
            ],
            'counterfactual_trajectories': [],
        }
        
        # Generate baseline if requested
        if return_baseline:
            logger.info("Generating baseline trajectory...")
            baseline = self._generate_trajectory(
                scenario_data,
                temperature=temperature,
                use_bias=False
            )
            results['baseline_trajectory'] = baseline
        
        # Generate counterfactual samples
        for i in range(n_samples):
            logger.info(f"Generating counterfactual sample {i+1}/{n_samples}...")
            cf_traj = self._generate_trajectory(
                scenario_data,
                temperature=temperature,
                use_bias=True
            )
            results['counterfactual_trajectories'].append(cf_traj)
        
        return results
    
    def _generate_trajectory(
        self,
        scenario_data: Dict,
        temperature: Optional[float] = None,
        use_bias: bool = False
    ) -> Dict:
        """
        Generate a single trajectory with BMT.
        
        Args:
            scenario_data: Input scenario dict
            temperature: Sampling temperature (None = use config)
            use_bias: Whether to apply counterfactual bias
            
        Returns:
            Dict with trajectory data including:
                - positions: [T, N_agents, 2] array
                - headings: [T, N_agents] array
                - velocities: [T, N_agents, 2] array
                - valid_mask: [T, N_agents] array
        """
        from bmt.models.motionlm import set_biased_sampler, reset_timestep
        
        # Prepare input data
        input_dict = self._prepare_input(scenario_data)
        
        # Setup biased sampling
        if use_bias and self.sampler is not None:
            reset_timestep()
            set_biased_sampler(self.sampler)
            logger.debug("Biased sampling enabled")
        else:
            set_biased_sampler(None)
            logger.debug("Standard sampling")
        
        # Tokenize input
        tok_data, _ = self.tokenizer.tokenize(input_dict, backward_prediction=False)
        input_dict.update(tok_data)
        
        # Set sampling temperature
        sampling_temp = temperature if temperature is not None else self.config.SAMPLING.TEMPERATURE
        
        # Run autoregressive generation
        with torch.no_grad():
            output_dict = self.model.model.autoregressive_rollout(
                input_dict,
                num_decode_steps=None,
                sampling_method=self.config.SAMPLING.SAMPLING_METHOD,
                temperature=sampling_temp,
            )
        
        # Detokenize to get trajectories
        output_dict = self.tokenizer.detokenize(
            output_dict,
            detokenizing_gt=False,
            backward_prediction=False,
            flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING,
        )
        
        # Cleanup
        set_biased_sampler(None)
        
        # Extract trajectory data
        return self._extract_trajectories(output_dict)
    
    def _prepare_input(self, scenario_data: Dict) -> Dict:
        """
        Prepare scenario data for BMT input.
        
        Handles:
        - Converting numpy arrays to torch tensors
        - Adding batch dimension if needed
        - Moving to correct device
        - Setting evaluation flags
        """
        from bmt.utils.utils import numpy_to_torch
        
        # Deep copy to avoid modifying original
        input_dict = {}
        for k, v in scenario_data.items():
            if isinstance(v, np.ndarray) and 'track_name' not in k:
                input_dict[k] = v.copy()
            elif isinstance(v, torch.Tensor):
                input_dict[k] = v.clone()
            else:
                input_dict[k] = copy.deepcopy(v) if not isinstance(v, str) else v
        
        # Convert to torch tensors
        input_dict = numpy_to_torch(input_dict, device=self.device)
        
        # Convert specific keys to double precision (BMT requirement)
        double_keys = [
            "decoder/agent_position", 
            "decoder/agent_heading", 
            "decoder/agent_velocity",
            "decoder/reconstructed_position",
            "decoder/reconstructed_heading", 
            "decoder/reconstructed_velocity",
            "decoder/agent_shape",
            "decoder/current_agent_shape",
            "decoder/current_agent_position"
        ]
        
        for k in double_keys:
            if k in input_dict and isinstance(input_dict[k], torch.Tensor):
                if input_dict[k].dtype in [torch.float32, torch.float16]:
                    input_dict[k] = input_dict[k].double()
        
        # Add batch dimension if needed
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                # Check if batch dim is missing (heuristic: first dim should be 1 for batched)
                if 'decoder/agent' in k and v.dim() == 3:  # [T, N, D] -> [B, T, N, D]
                    input_dict[k] = v.unsqueeze(0)
                elif 'encoder/agent' in k and v.dim() == 3:
                    input_dict[k] = v.unsqueeze(0)
        
        # Set evaluation flags
        input_dict["in_evaluation"] = torch.tensor([True], dtype=torch.bool).to(self.device)
        
        return input_dict
    
    def _extract_trajectories(self, output_dict: Dict) -> Dict:
        """
        Extract trajectory arrays from BMT output.
        
        Args:
            output_dict: Raw BMT output after detokenization
            
        Returns:
            Dict with numpy arrays for positions, headings, velocities, valid_mask
        """
        result = {}
        
        # Key priority - after detokenization, predictions are in 'reconstructed' fields
        position_key = None
        for k in ['decoder/reconstructed_position', 'decoder/agent_position']:
            if k in output_dict:
                position_key = k
                break
        
        if position_key is None:
            logger.warning("No position data found in output")
            return result
        
        # Extract and convert to numpy
        for key_suffix, out_name in [
            ('position', 'positions'),
            ('heading', 'headings'),
            ('velocity', 'velocities'),
            ('valid_mask', 'valid_mask')
        ]:
            # Try reconstructed first, then raw
            for prefix in ['decoder/reconstructed_', 'decoder/agent_']:
                key = prefix + key_suffix
                if key in output_dict:
                    val = output_dict[key]
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().numpy()
                    # Remove batch dim if present
                    if val.ndim == 4:  # [B, T, N, D]
                        val = val[0]  # Take first batch
                    elif val.ndim == 3 and key_suffix == 'valid_mask':  # [B, T, N]
                        val = val[0]
                    result[out_name] = val
                    break
        
        return result
    
    def generate_batch_counterfactuals(
        self,
        scenario_data: Dict,
        interventions: List[Dict],
        n_samples_per: int = 1,
        temperature: Optional[float] = None
    ) -> List[Dict]:
        """
        Generate counterfactuals for multiple interventions.
        
        Args:
            scenario_data: BMT-format scenario
            interventions: List of intervention dicts
            n_samples_per: Samples per intervention
            temperature: Sampling temperature
            
        Returns:
            List of result dicts, one per intervention
        """
        results = []
        
        for i, intervention in enumerate(interventions):
            logger.info(f"Processing intervention {i+1}/{len(interventions)}")
            result = self.generate_counterfactual(
                scenario_data,
                intervention,
                n_samples=n_samples_per,
                temperature=temperature,
                return_baseline=(i == 0)  # Only first needs baseline
            )
            results.append(result)
        
        return results


# =============================================================================
# Utility functions for trajectory comparison
# =============================================================================

def compare_trajectories(baseline: Dict, counterfactual: Dict, agent_idx: int = 0) -> Dict:
    """
    Compare baseline and counterfactual trajectories.
    
    Args:
        baseline: Baseline trajectory dict from generator
        counterfactual: Counterfactual trajectory dict
        agent_idx: Which agent to compare (0 = ego)
        
    Returns:
        Dict with comparison metrics
    """
    if 'positions' not in baseline or 'positions' not in counterfactual:
        return {'error': 'Missing position data'}
    
    b_pos = baseline['positions'][:, agent_idx, :2]  # [T, 2]
    c_pos = counterfactual['positions'][:, agent_idx, :2]
    
    # Ensure same length
    T = min(len(b_pos), len(c_pos))
    b_pos = b_pos[:T]
    c_pos = c_pos[:T]
    
    # Displacement difference
    diff = np.linalg.norm(c_pos - b_pos, axis=1)
    
    # Travel distance
    b_travel = np.sum(np.linalg.norm(np.diff(b_pos, axis=0), axis=1))
    c_travel = np.sum(np.linalg.norm(np.diff(c_pos, axis=0), axis=1))
    
    # Final displacement from start
    b_final_disp = np.linalg.norm(b_pos[-1] - b_pos[0])
    c_final_disp = np.linalg.norm(c_pos[-1] - c_pos[0])
    
    return {
        'max_displacement_diff': float(diff.max()),
        'mean_displacement_diff': float(diff.mean()),
        'baseline_travel_distance': float(b_travel),
        'counterfactual_travel_distance': float(c_travel),
        'baseline_final_displacement': float(b_final_disp),
        'counterfactual_final_displacement': float(c_final_disp),
        'travel_reduction_ratio': float(c_travel / b_travel) if b_travel > 0 else 1.0,
    }


def plot_trajectory_comparison(
    baseline: Dict, 
    counterfactual: Dict, 
    agent_idx: int = 0,
    save_path: Optional[str] = None,
    title: str = "Baseline vs Counterfactual Trajectory"
):
    """
    Visualize baseline vs counterfactual trajectories.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return
    
    if 'positions' not in baseline or 'positions' not in counterfactual:
        logger.warning("Missing position data for plotting")
        return
    
    b_pos = baseline['positions'][:, agent_idx, :2]
    c_pos = counterfactual['positions'][:, agent_idx, :2]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectories
    ax.plot(b_pos[:, 0], b_pos[:, 1], 'b-o', label='Baseline', 
            markersize=4, alpha=0.7, linewidth=2)
    ax.plot(c_pos[:, 0], c_pos[:, 1], 'r-s', label='Counterfactual',
            markersize=4, alpha=0.7, linewidth=2)
    
    # Mark start/end
    ax.scatter([b_pos[0, 0]], [b_pos[0, 1]], c='green', s=150, 
               marker='*', zorder=10, label='Start')
    ax.scatter([b_pos[-1, 0]], [b_pos[-1, 1]], c='blue', s=100,
               marker='x', zorder=10, linewidths=3)
    ax.scatter([c_pos[-1, 0]], [c_pos[-1, 1]], c='red', s=100,
               marker='x', zorder=10, linewidths=3)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()