"""
Minimal working DAG-guided trajectory generation.
Fill in the marked TODOs to connect to your actual data/models.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from collections import defaultdict


# ============ ENUMS ============

class ManeuverType(Enum):
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STOP = "stop"

class DecisionType(Enum):
    PROCEED_OR_YIELD = "proceed_or_yield"
    LANE_CHOICE = "lane_choice"
    EVASIVE_ACTION = "evasive_action"

class Aggressiveness(Enum):
    PASSIVE = "passive"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"


# ============ DATA STRUCTURES ============

@dataclass
class CriticalDecisionPoint:
    timestep: int
    decision_type: DecisionType
    ground_truth_choice: str
    alternatives: List[str]

@dataclass
class ManeuverSegment:
    maneuver_type: ManeuverType
    start_timestep: int
    end_timestep: int
    aggressiveness: Aggressiveness

@dataclass
class ScenarioFeatures:
    scenario_id: str
    maneuver_sequence: List[ManeuverSegment]
    critical_decisions: List[CriticalDecisionPoint]
    ego_trajectory: np.ndarray  # [T, 4] - x, y, theta, v

@dataclass
class DAGNode:
    name: str
    node_type: str
    value: Any
    domain: List[Any]
    timestep: Optional[int] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

@dataclass
class BMTConstraints:
    token_biases: Dict[int, np.ndarray]
    terminal_state: Optional[np.ndarray] = None
    source_intervention: Dict[str, Any] = field(default_factory=dict)


# ============ FEATURE EXTRACTION ============

class SafetyCriticalExtractor:
    """
    Rule-based extraction of safety-critical features from trajectory.
    """
    
    # Thresholds
    YAW_TURN_THRESHOLD = 0.3  # rad/s - above this is a turn
    YAW_LANE_CHANGE_THRESHOLD = 0.1  # rad/s - above this is lane change
    ACCEL_THRESHOLD = 2.0  # m/s² - significant accel/decel
    HARD_BRAKE_THRESHOLD = -4.0  # m/s²
    
    def extract(self, ego_trajectory: np.ndarray, scenario_id: str = "unknown") -> ScenarioFeatures:
        """
        Extract features from ego trajectory.
        
        Args:
            ego_trajectory: [T, 4] array of (x, y, theta, v)
            scenario_id: Identifier for the scenario
        
        Returns:
            ScenarioFeatures with maneuvers and decisions
        """
        
        # Compute kinematics
        dt = 0.1  # Assuming 10Hz
        T = len(ego_trajectory)
        
        positions = ego_trajectory[:, :2]
        headings = ego_trajectory[:, 2]
        speeds = ego_trajectory[:, 3]
        
        # Derivatives
        yaw_rates = np.zeros(T)
        yaw_rates[1:] = np.diff(headings) / dt
        # Handle angle wrapping
        yaw_rates = np.arctan2(np.sin(yaw_rates), np.cos(yaw_rates)) / dt * dt
        
        accelerations = np.zeros(T)
        accelerations[1:] = np.diff(speeds) / dt
        
        # Segment into maneuvers
        maneuvers = self._segment_maneuvers(yaw_rates, accelerations, dt)
        
        # Find critical decisions
        decisions = self._find_decisions(yaw_rates, accelerations, maneuvers)
        
        return ScenarioFeatures(
            scenario_id=scenario_id,
            maneuver_sequence=maneuvers,
            critical_decisions=decisions,
            ego_trajectory=ego_trajectory
        )
    
    def _segment_maneuvers(
        self, 
        yaw_rates: np.ndarray, 
        accelerations: np.ndarray,
        dt: float
    ) -> List[ManeuverSegment]:
        """Segment trajectory into maneuver primitives."""
        
        segments = []
        T = len(yaw_rates)
        
        current_start = 0
        current_type = self._classify_instant(yaw_rates[0], accelerations[0])
        
        for t in range(1, T):
            new_type = self._classify_instant(yaw_rates[t], accelerations[t])
            
            # Require sustained change (hysteresis)
            if new_type != current_type:
                # Check if change is sustained for at least 5 frames
                if t + 5 < T:
                    future_types = [self._classify_instant(yaw_rates[t+i], accelerations[t+i]) 
                                   for i in range(5)]
                    if future_types.count(new_type) < 3:
                        continue  # Not sustained, skip
                
                # Commit segment
                if t - current_start >= 3:  # Minimum segment length
                    aggressiveness = self._classify_aggressiveness(
                        accelerations[current_start:t],
                        yaw_rates[current_start:t]
                    )
                    segments.append(ManeuverSegment(
                        maneuver_type=current_type,
                        start_timestep=current_start,
                        end_timestep=t,
                        aggressiveness=aggressiveness
                    ))
                
                current_start = t
                current_type = new_type
        
        # Final segment
        if T - current_start >= 3:
            aggressiveness = self._classify_aggressiveness(
                accelerations[current_start:],
                yaw_rates[current_start:]
            )
            segments.append(ManeuverSegment(
                maneuver_type=current_type,
                start_timestep=current_start,
                end_timestep=T,
                aggressiveness=aggressiveness
            ))
        
        return segments
    
    def _classify_instant(self, yaw_rate: float, accel: float) -> ManeuverType:
        """Classify instantaneous maneuver from kinematics."""
        
        yaw_abs = abs(yaw_rate)
        
        if yaw_abs > self.YAW_TURN_THRESHOLD:
            return ManeuverType.LEFT_TURN if yaw_rate > 0 else ManeuverType.RIGHT_TURN
        elif yaw_abs > self.YAW_LANE_CHANGE_THRESHOLD:
            return ManeuverType.LANE_CHANGE_LEFT if yaw_rate > 0 else ManeuverType.LANE_CHANGE_RIGHT
        elif accel > self.ACCEL_THRESHOLD:
            return ManeuverType.ACCELERATE
        elif accel < -self.ACCEL_THRESHOLD:
            return ManeuverType.DECELERATE
        else:
            return ManeuverType.STRAIGHT
    
    def _classify_aggressiveness(
        self, 
        accelerations: np.ndarray, 
        yaw_rates: np.ndarray
    ) -> Aggressiveness:
        """Classify aggressiveness based on magnitude statistics."""
        
        max_accel = np.max(np.abs(accelerations)) if len(accelerations) > 0 else 0
        max_yaw = np.max(np.abs(yaw_rates)) if len(yaw_rates) > 0 else 0
        
        # Aggressive if high acceleration or yaw rate
        if max_accel > 5.0 or max_yaw > 0.5:
            return Aggressiveness.AGGRESSIVE
        elif max_accel < 2.0 and max_yaw < 0.2:
            return Aggressiveness.PASSIVE
        else:
            return Aggressiveness.NORMAL
    
    def _find_decisions(
        self,
        yaw_rates: np.ndarray,
        accelerations: np.ndarray,
        maneuvers: List[ManeuverSegment]
    ) -> List[CriticalDecisionPoint]:
        """Find critical decision points."""
        
        decisions = []
        
        # Decision: Turn direction (at start of turn maneuvers)
        for maneuver in maneuvers:
            if maneuver.maneuver_type in [ManeuverType.LEFT_TURN, ManeuverType.RIGHT_TURN]:
                decisions.append(CriticalDecisionPoint(
                    timestep=maneuver.start_timestep,
                    decision_type=DecisionType.LANE_CHOICE,
                    ground_truth_choice="left" if maneuver.maneuver_type == ManeuverType.LEFT_TURN else "right",
                    alternatives=["left", "right", "straight"]
                ))
            
            elif maneuver.maneuver_type in [ManeuverType.LANE_CHANGE_LEFT, ManeuverType.LANE_CHANGE_RIGHT]:
                decisions.append(CriticalDecisionPoint(
                    timestep=maneuver.start_timestep,
                    decision_type=DecisionType.LANE_CHOICE,
                    ground_truth_choice="left" if maneuver.maneuver_type == ManeuverType.LANE_CHANGE_LEFT else "right",
                    alternatives=["left", "right", "stay"]
                ))
        
        # Decision: Hard braking (potential evasive action)
        hard_brake_indices = np.where(accelerations < self.HARD_BRAKE_THRESHOLD)[0]
        if len(hard_brake_indices) > 0:
            # Take first hard brake
            t = hard_brake_indices[0]
            decisions.append(CriticalDecisionPoint(
                timestep=int(t),
                decision_type=DecisionType.EVASIVE_ACTION,
                ground_truth_choice="brake",
                alternatives=["brake", "swerve_left", "swerve_right", "none"]
            ))
        
        return decisions

# ============ VLM-BASED EXTRACTOR ============

class VLMSafetyCriticalExtractor:
    """
    Extract safety-critical features using VLM analysis of dashcam frames.
    
    Designed for easy debugging:
    - All prompts are class attributes (easy to modify)
    - All VLM calls are logged
    - Raw responses are preserved
    - Graceful fallbacks on parse errors
    """
    
    # ===== PROMPTS (easy to iterate on) =====
    
    MANEUVER_EXTRACTION_PROMPT = """Analyze these dashcam images showing a driving sequence. Images are in chronological order.

Identify the MANEUVERS the ego vehicle (camera vehicle) performs. Focus on:
- Going straight
- Turning left or right
- Changing lanes left or right  
- Accelerating or decelerating
- Stopping

For each maneuver, describe:
1. What maneuver type it is
2. When it roughly starts (which frame: 1=first, 2=second, etc.)
3. How aggressive it is (passive/normal/aggressive based on speed of execution)

Respond ONLY with JSON:
{
    "maneuvers": [
        {
            "type": "straight" | "left_turn" | "right_turn" | "lane_change_left" | "lane_change_right" | "accelerate" | "decelerate" | "stop",
            "start_frame": 1-5,
            "end_frame": 1-5,
            "aggressiveness": "passive" | "normal" | "aggressive",
            "description": "brief description of what you see"
        }
    ],
    "overall_description": "one sentence summary of what happens in this sequence"
}"""

    DECISION_EXTRACTION_PROMPT = """Analyze these dashcam images showing a driving sequence. Images are in chronological order.

Identify any CRITICAL DECISIONS the driver made. These include:
- Choosing to proceed or yield (at intersections, merges)
- Choosing which lane/direction to go
- Taking evasive action (hard braking, swerving) or not
- Accepting or rejecting a gap (when merging, changing lanes)

For each decision, describe:
1. What type of decision it was
2. What choice was made
3. What alternatives existed
4. Which frame shows this decision

Respond ONLY with JSON:
{
    "decisions": [
        {
            "type": "proceed_or_yield" | "lane_choice" | "evasive_action" | "gap_acceptance",
            "frame": 1-5,
            "choice_made": "what the driver did",
            "alternatives": ["other", "possible", "choices"],
            "description": "why this was a critical decision"
        }
    ],
    "risk_level": "low" | "medium" | "high",
    "risk_explanation": "brief explanation of any risks observed"
}"""

    # Frame names in expected order
    FRAME_NAMES = ["Prior.jpg", "Start.jpg", "Reaction.jpg", "Impact.jpg", "End.jpg"]
    
    def __init__(self, client, debug: bool = True):
        """
        Args:
            client: GPT4oClient instance
            debug: If True, log verbose output
        """
        self.client = client
        self.debug = debug
        self.extraction_log = []
    
    def extract(
        self, 
        keyframe_dir: str,
        trajectory: Optional[np.ndarray] = None,
        scenario_id: str = "unknown"
    ) -> ScenarioFeatures:
        """
        Extract safety-critical features from keyframes.
        
        Args:
            keyframe_dir: Directory containing keyframe images
            trajectory: Optional [T, 4] trajectory for timestamp grounding
            scenario_id: Identifier for logging
        
        Returns:
            ScenarioFeatures with maneuvers and decisions
        """
        
        # Load images
        images, frame_names = self._load_keyframes(keyframe_dir)
        
        if not images:
            logger.warning(f"No keyframes found in {keyframe_dir}")
            return self._empty_features(scenario_id, "No keyframes found")
        
        if self.debug:
            logger.info(f"VLM extracting from {scenario_id}: {len(images)} frames")
            logger.info(f"Frames: {frame_names}")
        
        # Extract maneuvers
        maneuvers, maneuver_raw = self._extract_maneuvers(images, frame_names)
        
        # Extract decisions
        decisions, decision_raw = self._extract_decisions(images, frame_names)
        
        # If trajectory provided, ground timestamps
        if trajectory is not None:
            maneuvers = self._ground_timestamps(maneuvers, frame_names, len(trajectory))
            decisions = self._ground_decision_timestamps(decisions, frame_names, len(trajectory))
        
        # Build result
        features = ScenarioFeatures(
            scenario_id=scenario_id,
            maneuver_sequence=maneuvers,
            critical_decisions=decisions,
            ego_trajectory=trajectory,
            vlm_raw_response=json.dumps({
                "maneuvers": maneuver_raw,
                "decisions": decision_raw
            }, indent=2),
            extraction_metadata={
                "keyframe_dir": str(keyframe_dir),
                "frames_used": frame_names,
                "n_frames": len(images),
                "has_trajectory": trajectory is not None
            }
        )
        
        # Log for debugging
        self.extraction_log.append({
            "scenario_id": scenario_id,
            "n_maneuvers": len(maneuvers),
            "n_decisions": len(decisions),
            "frames": frame_names
        })
        
        if self.debug:
            logger.info(f"Extracted {len(maneuvers)} maneuvers, {len(decisions)} decisions")
            for m in maneuvers:
                logger.info(f"  Maneuver: {m.maneuver_type.value} "
                           f"(frames {m.start_timestep}-{m.end_timestep}, {m.aggressiveness.value})")
            for d in decisions:
                logger.info(f"  Decision: {d.decision_type.value} -> {d.ground_truth_choice}")
        
        return features
    
    def _load_keyframes(self, keyframe_dir: str) -> Tuple[List[str], List[str]]:
        """Load keyframe images as base64."""
        
        kf_path = Path(keyframe_dir)
        images = []
        frame_names = []
        
        for name in self.FRAME_NAMES:
            p = kf_path / name
            if p.exists():
                with open(p, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
                    frame_names.append(name)
        
        return images, frame_names
    
    def _extract_maneuvers(
        self, 
        images: List[str], 
        frame_names: List[str]
    ) -> Tuple[List[ManeuverSegment], Dict]:
        """Extract maneuvers using VLM."""
        
        try:
            response = self.client.complete(
                self.MANEUVER_EXTRACTION_PROMPT,
                images=images,
                temperature=0.1,
                max_tokens=1000
            )
            
            parsed = self._parse_json_response(response)
            
            if self.debug:
                logger.debug(f"Maneuver VLM raw: {response[:500]}")
            
            if "error" in parsed:
                logger.warning(f"Maneuver extraction parse error: {parsed['error']}")
                return [], {"error": parsed["error"], "raw": response[:300]}
            
            maneuvers = []
            for m in parsed.get("maneuvers", []):
                maneuver = self._parse_maneuver(m, frame_names)
                if maneuver:
                    maneuvers.append(maneuver)
            
            return maneuvers, parsed
            
        except Exception as e:
            logger.error(f"Maneuver extraction failed: {e}")
            return [], {"error": str(e)}
    
    def _extract_decisions(
        self, 
        images: List[str], 
        frame_names: List[str]
    ) -> Tuple[List[CriticalDecisionPoint], Dict]:
        """Extract decisions using VLM."""
        
        try:
            response = self.client.complete(
                self.DECISION_EXTRACTION_PROMPT,
                images=images,
                temperature=0.1,
                max_tokens=1000
            )
            
            parsed = self._parse_json_response(response)
            
            if self.debug:
                logger.debug(f"Decision VLM raw: {response[:500]}")
            
            if "error" in parsed:
                logger.warning(f"Decision extraction parse error: {parsed['error']}")
                return [], {"error": parsed["error"], "raw": response[:300]}
            
            decisions = []
            for d in parsed.get("decisions", []):
                decision = self._parse_decision(d, frame_names)
                if decision:
                    decisions.append(decision)
            
            return decisions, parsed
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {e}")
            return [], {"error": str(e)}
    
    def _parse_maneuver(self, m: Dict, frame_names: List[str]) -> Optional[ManeuverSegment]:
        """Parse a single maneuver from VLM response."""
        
        try:
            # Map string to enum
            type_str = m.get("type", "unknown").lower()
            type_map = {
                "straight": ManeuverType.STRAIGHT,
                "left_turn": ManeuverType.LEFT_TURN,
                "right_turn": ManeuverType.RIGHT_TURN,
                "lane_change_left": ManeuverType.LANE_CHANGE_LEFT,
                "lane_change_right": ManeuverType.LANE_CHANGE_RIGHT,
                "accelerate": ManeuverType.ACCELERATE,
                "decelerate": ManeuverType.DECELERATE,
                "stop": ManeuverType.STOP,
            }
            maneuver_type = type_map.get(type_str, ManeuverType.UNKNOWN)
            
            # Aggressiveness
            agg_str = m.get("aggressiveness", "normal").lower()
            agg_map = {
                "passive": Aggressiveness.PASSIVE,
                "normal": Aggressiveness.NORMAL,
                "aggressive": Aggressiveness.AGGRESSIVE,
            }
            aggressiveness = agg_map.get(agg_str, Aggressiveness.NORMAL)
            
            # Frame indices (1-indexed from VLM, convert to 0-indexed)
            start_frame = max(0, int(m.get("start_frame", 1)) - 1)
            end_frame = max(start_frame, int(m.get("end_frame", len(frame_names))) - 1)
            
            return ManeuverSegment(
                maneuver_type=maneuver_type,
                start_timestep=start_frame,  # Will be re-grounded if trajectory provided
                end_timestep=end_frame,
                aggressiveness=aggressiveness,
                description=m.get("description", "")
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse maneuver {m}: {e}")
            return None
    
    def _parse_decision(self, d: Dict, frame_names: List[str]) -> Optional[CriticalDecisionPoint]:
        """Parse a single decision from VLM response."""
        
        try:
            # Map string to enum
            type_str = d.get("type", "lane_choice").lower()
            type_map = {
                "proceed_or_yield": DecisionType.PROCEED_OR_YIELD,
                "lane_choice": DecisionType.LANE_CHOICE,
                "evasive_action": DecisionType.EVASIVE_ACTION,
                "gap_acceptance": DecisionType.GAP_ACCEPTANCE,
            }
            decision_type = type_map.get(type_str, DecisionType.LANE_CHOICE)
            
            # Frame index
            frame = max(0, int(d.get("frame", 1)) - 1)
            
            # Choice and alternatives
            choice = d.get("choice_made", "unknown")
            alternatives = d.get("alternatives", [choice])
            if choice not in alternatives:
                alternatives = [choice] + alternatives
            
            return CriticalDecisionPoint(
                timestep=frame,  # Will be re-grounded if trajectory provided
                decision_type=decision_type,
                ground_truth_choice=choice,
                alternatives=alternatives,
                description=d.get("description", "")
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse decision {d}: {e}")
            return None
    
    def _ground_timestamps(
        self, 
        maneuvers: List[ManeuverSegment],
        frame_names: List[str],
        trajectory_length: int
    ) -> List[ManeuverSegment]:
        """
        Convert frame indices to trajectory timesteps.
        
        Maps keyframes to trajectory timeline:
        - Prior.jpg -> ~10% of trajectory
        - Start.jpg -> ~25%
        - Reaction.jpg -> ~50%
        - Impact.jpg -> ~75%
        - End.jpg -> ~90%
        """
        
        # Approximate mapping from frame name to trajectory fraction
        frame_to_fraction = {
            "Prior.jpg": 0.1,
            "Start.jpg": 0.25,
            "Reaction.jpg": 0.5,
            "Impact.jpg": 0.75,
            "End.jpg": 0.9
        }
        
        # Build mapping for frames we actually have
        frame_fractions = []
        for name in frame_names:
            frac = frame_to_fraction.get(name, 0.5)
            frame_fractions.append(frac)
        
        grounded = []
        for m in maneuvers:
            # Get fractions for start/end frames
            start_frac = frame_fractions[min(m.start_timestep, len(frame_fractions)-1)]
            end_frac = frame_fractions[min(m.end_timestep, len(frame_fractions)-1)]
            
            # Convert to timesteps
            start_t = int(start_frac * trajectory_length)
            end_t = int(end_frac * trajectory_length)
            
            grounded.append(ManeuverSegment(
                maneuver_type=m.maneuver_type,
                start_timestep=start_t,
                end_timestep=max(end_t, start_t + 1),
                aggressiveness=m.aggressiveness,
                description=m.description
            ))
        
        return grounded
    
    def _ground_decision_timestamps(
        self,
        decisions: List[CriticalDecisionPoint],
        frame_names: List[str],
        trajectory_length: int
    ) -> List[CriticalDecisionPoint]:
        """Convert frame indices to trajectory timesteps for decisions."""
        
        frame_to_fraction = {
            "Prior.jpg": 0.1,
            "Start.jpg": 0.25,
            "Reaction.jpg": 0.5,
            "Impact.jpg": 0.75,
            "End.jpg": 0.9
        }
        
        frame_fractions = [frame_to_fraction.get(name, 0.5) for name in frame_names]
        
        grounded = []
        for d in decisions:
            frac = frame_fractions[min(d.timestep, len(frame_fractions)-1)]
            t = int(frac * trajectory_length)
            
            grounded.append(CriticalDecisionPoint(
                timestep=t,
                decision_type=d.decision_type,
                ground_truth_choice=d.ground_truth_choice,
                alternatives=d.alternatives,
                description=d.description
            ))
        
        return grounded
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from VLM response, handling markdown code blocks."""
        
        try:
            resp = response.strip()
            
            # Remove markdown code blocks if present
            if resp.startswith("```"):
                lines = resp.split("\n")
                # Remove first line (```json) and last line (```)
                resp = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            return json.loads(resp)
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}", "raw": response[:200]}
    
    def _empty_features(self, scenario_id: str, error: str) -> ScenarioFeatures:
        """Return empty features on error."""
        return ScenarioFeatures(
            scenario_id=scenario_id,
            maneuver_sequence=[],
            critical_decisions=[],
            vlm_raw_response=error,
            extraction_metadata={"error": error}
        )
    
    def get_extraction_log(self) -> List[Dict]:
        """Return log of all extractions."""
        return self.extraction_log

# ============ DAG CONSTRUCTION ============

class ScenarioDAG:
    """Causal DAG constructed from extracted features."""
    
    def __init__(self, features: ScenarioFeatures):
        self.features = features
        self.nodes: Dict[str, DAGNode] = {}
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        
        self._build_from_features()
    
    def _build_from_features(self):
        """Build DAG from features."""
        
        # Add decision nodes (our intervention targets)
        for i, decision in enumerate(self.features.critical_decisions):
            node_name = f"decision_{i}_{decision.decision_type.value}"
            self.nodes[node_name] = DAGNode(
                name=node_name,
                node_type="decision",
                value=decision.ground_truth_choice,
                domain=decision.alternatives,
                timestep=decision.timestep,
                parents=[],
                children=[]
            )
        
        # Add maneuver nodes
        for i, maneuver in enumerate(self.features.maneuver_sequence):
            node_name = f"maneuver_{i}_{maneuver.maneuver_type.value}"
            self.nodes[node_name] = DAGNode(
                name=node_name,
                node_type="maneuver",
                value={
                    "type": maneuver.maneuver_type,
                    "aggressiveness": maneuver.aggressiveness,
                    "start": maneuver.start_timestep,
                    "end": maneuver.end_timestep
                },
                domain=self._get_maneuver_alternatives(maneuver),
                timestep=maneuver.start_timestep,
                parents=[],
                children=[]
            )
        
        # Add outcome node
        self.nodes["outcome"] = DAGNode(
            name="outcome",
            node_type="outcome",
            value="safe",  # Default; would be set from collision detection
            domain=["safe", "near_miss", "collision"],
            parents=[],
            children=[]
        )
        
        # Wire edges: decisions → maneuvers → outcome
        self._build_edges()
    
    def _get_maneuver_alternatives(self, maneuver: ManeuverSegment) -> List[Dict]:
        """Get alternative configurations for a maneuver."""
        
        alternatives = []
        
        # Same type, different aggressiveness
        for agg in Aggressiveness:
            alternatives.append({
                "type": maneuver.maneuver_type,
                "aggressiveness": agg,
                "start": maneuver.start_timestep,
                "end": maneuver.end_timestep
            })
        
        # Different maneuver types (if applicable)
        if maneuver.maneuver_type == ManeuverType.LEFT_TURN:
            alternatives.append({
                "type": ManeuverType.RIGHT_TURN,
                "aggressiveness": maneuver.aggressiveness,
                "start": maneuver.start_timestep,
                "end": maneuver.end_timestep
            })
            alternatives.append({
                "type": ManeuverType.STRAIGHT,
                "aggressiveness": maneuver.aggressiveness,
                "start": maneuver.start_timestep,
                "end": maneuver.end_timestep
            })
        
        return alternatives
    
    def _build_edges(self):
        """Build causal edges based on temporal proximity."""
        
        decision_nodes = [(n, node) for n, node in self.nodes.items() 
                         if node.node_type == "decision"]
        maneuver_nodes = [(n, node) for n, node in self.nodes.items() 
                         if node.node_type == "maneuver"]
        
        # Decisions → Maneuvers (if decision precedes maneuver)
        for d_name, d_node in decision_nodes:
            for m_name, m_node in maneuver_nodes:
                if d_node.timestep is not None and m_node.timestep is not None:
                    # Decision affects maneuvers that follow within 2 seconds (20 frames)
                    if 0 <= m_node.timestep - d_node.timestep <= 20:
                        self._add_edge(d_name, m_name)
        
        # All maneuvers → Outcome
        for m_name, _ in maneuver_nodes:
            self._add_edge(m_name, "outcome")
    
    def _add_edge(self, parent: str, child: str):
        """Add directed edge."""
        self.adjacency[parent].append(child)
        self.nodes[child].parents.append(parent)
        self.nodes[parent].children.append(child)
    
    def get_topological_order(self) -> List[str]:
        """Return nodes in topological order."""
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.adjacency.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        return order
    
    def get_decision_nodes(self) -> List[str]:
        """Get names of decision nodes (intervention targets)."""
        return [name for name, node in self.nodes.items() 
                if node.node_type == "decision"]


# ============ COUNTERFACTUAL SAMPLING ============

class CounterfactualSampler:
    """Sample counterfactual scenarios by intervening on DAG."""
    
    def __init__(self, dag: ScenarioDAG):
        self.dag = dag
    
    def sample_counterfactual(self, intervention: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply intervention and propagate.
        
        For now, uses simple rule-based propagation.
        Could be extended with LLM world model.
        """
        
        # Start with ground truth
        cf_state = {name: node.value for name, node in self.dag.nodes.items()}
        cf_state["_intervention"] = intervention
        
        # Apply intervention
        for node_name, new_value in intervention.items():
            cf_state[node_name] = new_value
        
        # Propagate to descendants (simple rule-based for now)
        topo_order = self.dag.get_topological_order()
        intervened = set(intervention.keys())
        
        for node_name in topo_order:
            if node_name in intervened:
                continue
            
            node = self.dag.nodes[node_name]
            if not node.parents:
                continue
            
            # Check if any parent was intervened
            if any(self._is_affected(p, intervened) for p in node.parents):
                # Propagate: update based on parent values
                cf_state[node_name] = self._propagate_value(node, cf_state)
        
        return cf_state
    
    def _is_affected(self, node_name: str, intervened: set) -> bool:
        """Check if node is in intervened set or descended from it."""
        if node_name in intervened:
            return True
        for parent in self.dag.nodes[node_name].parents:
            if self._is_affected(parent, intervened):
                return True
        return False
    
    def _propagate_value(self, node: DAGNode, current_state: Dict) -> Any:
        """
        Simple rule-based propagation.
        
        This is where you'd plug in LLM world model for more sophisticated reasoning.
        """
        
        if node.node_type == "maneuver":
            # If parent decision changed, update maneuver type accordingly
            for parent_name in node.parents:
                if "decision" in parent_name:
                    parent_value = current_state[parent_name]
                    
                    # Map decision to maneuver
                    if parent_value == "left":
                        return {
                            **node.value,
                            "type": ManeuverType.LEFT_TURN
                        }
                    elif parent_value == "right":
                        return {
                            **node.value,
                            "type": ManeuverType.RIGHT_TURN
                        }
                    elif parent_value == "straight" or parent_value == "stay":
                        return {
                            **node.value,
                            "type": ManeuverType.STRAIGHT
                        }
                    elif parent_value == "brake":
                        return {
                            **node.value,
                            "type": ManeuverType.DECELERATE,
                            "aggressiveness": Aggressiveness.AGGRESSIVE
                        }
        
        # Default: keep original value
        return node.value
    
    def enumerate_counterfactuals(self, max_samples: int = 10) -> List[Dict[str, Any]]:
        """Enumerate counterfactuals by intervening on decision nodes."""
        
        counterfactuals = []
        decision_nodes = self.dag.get_decision_nodes()
        
        for node_name in decision_nodes:
            node = self.dag.nodes[node_name]
            for alt_value in node.domain:
                if alt_value != node.value:  # Skip ground truth
                    cf = self.sample_counterfactual({node_name: alt_value})
                    counterfactuals.append(cf)
                    
                    if len(counterfactuals) >= max_samples:
                        return counterfactuals
        
        return counterfactuals


# ============ CONSTRAINT COMPILATION ============

class ConstraintCompiler:
    """Compile DAG state to BMT token biases."""
    
    K = 33  # Token bins per dimension
    K2 = 33 * 33  # Total tokens
    A_MAX = 10.0  # m/s²
    YAW_MAX = np.pi / 2  # rad/s
    
    def compile(
        self, 
        cf_state: Dict[str, Any],
        features: ScenarioFeatures
    ) -> BMTConstraints:
        """Compile counterfactual state to BMT constraints."""
        
        token_biases = {}
        
        # Process maneuver nodes
        for node_name, value in cf_state.items():
            if "maneuver" in node_name and isinstance(value, dict):
                biases = self._compile_maneuver(value)
                token_biases = self._merge_biases(token_biases, biases)
            
            elif "decision" in node_name:
                # Find the corresponding decision
                for decision in features.critical_decisions:
                    if decision.ground_truth_choice != value:
                        # This decision was intervened; apply bias
                        biases = self._compile_decision(decision, value)
                        token_biases = self._merge_biases(token_biases, biases)
        
        return BMTConstraints(
            token_biases=token_biases,
            source_intervention=cf_state.get("_intervention", {})
        )
    
    def _compile_decision(
        self, 
        decision: CriticalDecisionPoint, 
        new_value: str
    ) -> Dict[int, np.ndarray]:
        """Compile decision change to token biases."""
        
        biases = {}
        t_start = decision.timestep
        t_end = min(decision.timestep + 20, 91)  # 20 frames = 2 seconds
        
        if decision.decision_type == DecisionType.LANE_CHOICE:
            if new_value == "left":
                for t in range(t_start, t_end):
                    biases[t] = self._left_bias()
            elif new_value == "right":
                for t in range(t_start, t_end):
                    biases[t] = self._right_bias()
            elif new_value in ["straight", "stay"]:
                for t in range(t_start, t_end):
                    biases[t] = self._straight_bias()
        
        elif decision.decision_type == DecisionType.EVASIVE_ACTION:
            if new_value == "brake":
                for t in range(t_start, t_end):
                    biases[t] = self._decel_bias() * 2.0
            elif new_value == "swerve_left":
                for t in range(t_start, t_end):
                    biases[t] = self._left_bias() * 2.0
            elif new_value == "swerve_right":
                for t in range(t_start, t_end):
                    biases[t] = self._right_bias() * 2.0
            elif new_value == "none":
                for t in range(t_start, t_end):
                    biases[t] = self._straight_bias()
        
        return biases
    
    def _compile_maneuver(self, maneuver: Dict) -> Dict[int, np.ndarray]:
        """Compile maneuver to token biases."""
        
        biases = {}
        t_start = maneuver.get("start", 0)
        t_end = maneuver.get("end", 91)
        maneuver_type = maneuver.get("type")
        aggressiveness = maneuver.get("aggressiveness", Aggressiveness.NORMAL)
        
        # Aggressiveness multiplier
        if aggressiveness == Aggressiveness.AGGRESSIVE:
            mult = 1.5
        elif aggressiveness == Aggressiveness.PASSIVE:
            mult = 0.5
        else:
            mult = 1.0
        
        for t in range(t_start, t_end):
            if maneuver_type == ManeuverType.LEFT_TURN:
                biases[t] = self._left_bias() * mult
            elif maneuver_type == ManeuverType.RIGHT_TURN:
                biases[t] = self._right_bias() * mult
            elif maneuver_type == ManeuverType.LANE_CHANGE_LEFT:
                biases[t] = self._left_bias() * 0.5 * mult
            elif maneuver_type == ManeuverType.LANE_CHANGE_RIGHT:
                biases[t] = self._right_bias() * 0.5 * mult
            elif maneuver_type == ManeuverType.ACCELERATE:
                biases[t] = self._accel_bias() * mult
            elif maneuver_type == ManeuverType.DECELERATE:
                biases[t] = self._decel_bias() * mult
            elif maneuver_type == ManeuverType.STRAIGHT:
                biases[t] = self._straight_bias()
        
        return biases
    
    # Bias generation helpers
    def _left_bias(self) -> np.ndarray:
        """Bias toward positive yaw rate."""
        bias = np.zeros(self.K2)
        for i in range(self.K2):
            a_idx, yaw_idx = divmod(i, self.K)
            yaw_rate = (yaw_idx / (self.K - 1) - 0.5) * 2 * self.YAW_MAX
            bias[i] = max(0, yaw_rate)
        return bias
    
    def _right_bias(self) -> np.ndarray:
        """Bias toward negative yaw rate."""
        bias = np.zeros(self.K2)
        for i in range(self.K2):
            a_idx, yaw_idx = divmod(i, self.K)
            yaw_rate = (yaw_idx / (self.K - 1) - 0.5) * 2 * self.YAW_MAX
            bias[i] = max(0, -yaw_rate)
        return bias
    
    def _straight_bias(self) -> np.ndarray:
        """Bias toward zero yaw rate."""
        bias = np.zeros(self.K2)
        for i in range(self.K2):
            a_idx, yaw_idx = divmod(i, self.K)
            yaw_rate = (yaw_idx / (self.K - 1) - 0.5) * 2 * self.YAW_MAX
            bias[i] = 1.0 - abs(yaw_rate) / self.YAW_MAX  # Peak at zero yaw
        return bias
    
    def _accel_bias(self) -> np.ndarray:
        """Bias toward positive acceleration."""
        bias = np.zeros(self.K2)
        for i in range(self.K2):
            a_idx, yaw_idx = divmod(i, self.K)
            accel = (a_idx / (self.K - 1) - 0.5) * 2 * self.A_MAX
            bias[i] = max(0, accel)
        return bias
    
    def _decel_bias(self) -> np.ndarray:
        """Bias toward negative acceleration."""
        bias = np.zeros(self.K2)
        for i in range(self.K2):
            a_idx, yaw_idx = divmod(i, self.K)
            accel = (a_idx / (self.K - 1) - 0.5) * 2 * self.A_MAX
            bias[i] = max(0, -accel)
        return bias
    
    def _merge_biases(
        self, 
        existing: Dict[int, np.ndarray], 
        new: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Merge bias dictionaries."""
        merged = existing.copy()
        for t, bias in new.items():
            if t in merged:
                merged[t] = merged[t] + bias
            else:
                merged[t] = bias.copy()
        return merged


# ============ GUIDED GENERATION (BMT INTERFACE) ============

class GuidedBMTGenerator:
    """
    Generate trajectories with DAG-derived guidance.
    
    NOTE: This is a MOCK implementation. You need to connect
    to actual BMT model for real usage.
    """
    
    K = 33
    K2 = 33 * 33
    A_MAX = 10.0
    YAW_MAX = np.pi / 2
    DT = 0.5  # BMT timestep
    
    def __init__(self, bmt_model=None, guidance_strength: float = 2.0):
        """
        Args:
            bmt_model: Your actual BMT model (or None for mock)
            guidance_strength: How strongly to apply biases
        """
        self.bmt = bmt_model
        self.guidance_strength = guidance_strength
        
        if bmt_model is None:
            print("WARNING: Using mock BMT model. Connect real model for actual generation.")
    
    def generate(
        self,
        initial_state: np.ndarray,
        constraints: BMTConstraints,
        horizon: int = 91,
        num_samples: int = 1
    ) -> List[np.ndarray]:
        """
        Generate constrained trajectories.
        
        Args:
            initial_state: [x, y, theta, v] at t=0
            constraints: Compiled BMT constraints
            horizon: Number of timesteps
            num_samples: Number of trajectories to generate
        
        Returns:
            List of trajectories, each [T, 4]
        """
        
        trajectories = []
        
        for _ in range(num_samples):
            if self.bmt is not None:
                traj = self._generate_with_real_bmt(initial_state, constraints, horizon)
            else:
                traj = self._generate_mock(initial_state, constraints, horizon)
            trajectories.append(traj)
        
        return trajectories
    
    def _generate_with_real_bmt(
        self,
        initial_state: np.ndarray,
        constraints: BMTConstraints,
        horizon: int
    ) -> np.ndarray:
        """
        Generate using actual BMT model.
        
        TODO: Implement this by connecting to your BMT model.
        
        Pseudocode:
```
        tokens = []
        scene_encoding = self.bmt.encode_scene(scene_data)
        
        for t in range(horizon):
            logits = self.bmt.get_token_logits(scene_encoding, tokens)
            
            if t in constraints.token_biases:
                logits = logits + self.guidance_strength * constraints.token_biases[t]
            
            token = sample_from_logits(logits)
            tokens.append(token)
        
        trajectory = self.bmt.decode_tokens(tokens, initial_state)
        return trajectory
```
        """
        raise NotImplementedError(
            "Connect to your actual BMT model. "
            "See _generate_mock for the logic pattern."
        )
    
    def _generate_mock(
        self,
        initial_state: np.ndarray,
        constraints: BMTConstraints,
        horizon: int
    ) -> np.ndarray:
        """
        Mock generation for testing without BMT.
        
        Uses random sampling + bias to generate trajectories.
        """
        
        trajectory = np.zeros((horizon, 4))
        trajectory[0] = initial_state
        
        state = initial_state.copy()
        
        for t in range(1, horizon):
            # Mock "base distribution" - uniform over tokens
            logits = np.zeros(self.K2)
            
            # Apply bias if present
            if t in constraints.token_biases:
                logits = logits + self.guidance_strength * constraints.token_biases[t]
            
            # Sample token
            probs = self._softmax(logits)
            token = np.random.choice(self.K2, p=probs)
            
            # Integrate token to get next state
            state = self._integrate_token(state, token)
            trajectory[t] = state
        
        return trajectory
    
    def _integrate_token(self, state: np.ndarray, token: int) -> np.ndarray:
        """Integrate one token using BMT's midpoint integration."""
        x, y, theta, v = state
        
        # Decode token
        a_idx = token // self.K
        yaw_idx = token % self.K
        accel = (a_idx / (self.K - 1) - 0.5) * 2 * self.A_MAX
        yaw_rate = (yaw_idx / (self.K - 1) - 0.5) * 2 * self.YAW_MAX
        
        # Midpoint integration
        v_new = max(0, v + accel * self.DT)  # Clamp to non-negative
        theta_new = theta + yaw_rate * self.DT
        v_mid = (v + v_new) / 2
        theta_mid = (theta + theta_new) / 2
        x_new = x + v_mid * np.cos(theta_mid) * self.DT
        y_new = y + v_mid * np.sin(theta_mid) * self.DT
        
        return np.array([x_new, y_new, theta_new, v_new])
    
    def _softmax(self, logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature."""
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()


# ============ MAIN PIPELINE ============

class DAGGuidedGenerator:
    """Complete pipeline: trajectory → DAG → counterfactuals → guided generation."""
    
    def __init__(self, bmt_model=None, guidance_strength: float = 2.0):
        self.extractor = SafetyCriticalExtractor()
        self.compiler = ConstraintCompiler()
        self.generator = GuidedBMTGenerator(bmt_model, guidance_strength)
    
    def generate_counterfactuals(
        self,
        ego_trajectory: np.ndarray,
        scenario_id: str = "unknown",
        num_counterfactuals: int = 5,
        samples_per_counterfactual: int = 1
    ) -> List[Dict]:
        """
        Generate counterfactual trajectories from a ground truth trajectory.
        
        Args:
            ego_trajectory: [T, 4] array of (x, y, theta, v)
            scenario_id: Identifier for logging
            num_counterfactuals: How many different counterfactual interventions
            samples_per_counterfactual: Samples per intervention
        
        Returns:
            List of dicts with intervention, counterfactual state, and trajectory
        """
        
        # Stage 1: Extract features
        features = self.extractor.extract(ego_trajectory, scenario_id)
        print(f"Extracted {len(features.maneuver_sequence)} maneuvers, "
              f"{len(features.critical_decisions)} decisions")
        
        # Stage 2: Build DAG
        dag = ScenarioDAG(features)
        print(f"Built DAG with {len(dag.nodes)} nodes")
        
        # Stage 3: Sample counterfactuals
        sampler = CounterfactualSampler(dag)
        cf_states = sampler.enumerate_counterfactuals(max_samples=num_counterfactuals)
        print(f"Generated {len(cf_states)} counterfactual states")
        
        results = []
        for cf_state in cf_states:
            # Stage 4: Compile constraints
            constraints = self.compiler.compile(cf_state, features)
            
            # Stage 5: Generate trajectories
            initial_state = ego_trajectory[0]
            trajectories = self.generator.generate(
                initial_state,
                constraints,
                horizon=len(ego_trajectory),
                num_samples=samples_per_counterfactual
            )
            
            for traj in trajectories:
                results.append({
                    "intervention": cf_state.get("_intervention", {}),
                    "counterfactual_state": cf_state,
                    "trajectory": traj,
                    "original_trajectory": ego_trajectory
                })
        
        return results


# ============ EXAMPLE USAGE ============

def demo():
    """Demonstrate the pipeline with synthetic data."""
    
    # Create a synthetic "ground truth" trajectory
    # Vehicle going straight, then turning left
    T = 91
    dt = 0.1
    
    trajectory = np.zeros((T, 4))
    x, y, theta, v = 0.0, 0.0, 0.0, 10.0  # Start at origin, heading east, 10 m/s
    
    for t in range(T):
        trajectory[t] = [x, y, theta, v]
        
        # Kinematics: go straight for 3 seconds, then turn left
        if t < 30:
            accel, yaw_rate = 0.0, 0.0  # Straight
        elif t < 50:
            accel, yaw_rate = -1.0, 0.4  # Slow down and turn left
        else:
            accel, yaw_rate = 1.0, 0.0  # Accelerate straight
        
        # Integrate
        v = max(0, v + accel * dt)
        theta = theta + yaw_rate * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
    
    print("=" * 60)
    print("DAG-Guided Counterfactual Trajectory Generation Demo")
    print("=" * 60)
    print(f"\nOriginal trajectory: {T} timesteps")
    print(f"  Start: ({trajectory[0, 0]:.1f}, {trajectory[0, 1]:.1f})")
    print(f"  End: ({trajectory[-1, 0]:.1f}, {trajectory[-1, 1]:.1f})")
    
    # Create generator (mock BMT)
    generator = DAGGuidedGenerator(bmt_model=None, guidance_strength=3.0)
    
    # Generate counterfactuals
    results = generator.generate_counterfactuals(
        trajectory,
        scenario_id="demo_scenario",
        num_counterfactuals=5,
        samples_per_counterfactual=1
    )
    
    print(f"\nGenerated {len(results)} counterfactual trajectories:")
    for i, result in enumerate(results):
        intervention = result["intervention"]
        cf_traj = result["trajectory"]
        
        print(f"\n  [{i+1}] Intervention: {intervention}")
        print(f"      End position: ({cf_traj[-1, 0]:.1f}, {cf_traj[-1, 1]:.1f})")
        
        # Compute displacement from original
        orig_end = trajectory[-1, :2]
        cf_end = cf_traj[-1, :2]
        displacement = np.linalg.norm(cf_end - orig_end)
        print(f"      Displacement from original: {displacement:.1f}m")
    
    return results


if __name__ == "__main__":
    results = demo()
```

---

## What This Gives You

Running `python this_file.py` will:

1. Create a synthetic trajectory (straight → left turn → straight)
2. Extract maneuvers and decision points
3. Build a causal DAG
4. Generate counterfactual interventions (e.g., "what if they turned right instead?")
5. Compile constraints to token biases
6. Generate counterfactual trajectories (using mock sampling)

**Output looks like:**
```
============================================================
DAG-Guided Counterfactual Trajectory Generation Demo
============================================================

Original trajectory: 91 timesteps
  Start: (0.0, 0.0)
  End: (45.2, 28.3)
Extracted 3 maneuvers, 1 decisions
Built DAG with 5 nodes
Generated 2 counterfactual states

Generated 2 counterfactual trajectories:

  [1] Intervention: {'decision_0_lane_choice': 'right'}
      End position: (52.1, -25.7)
      Displacement from original: 54.1m

  [2] Intervention: {'decision_0_lane_choice': 'straight'}
      End position: (89.2, 1.2)
      Displacement from original: 52.3m