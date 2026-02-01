"""
vlm_extractor.py

Self-contained VLM-based safety-critical feature extraction.
Designed for simulator screenshots with arbitrary timestamps.

Usage:
    # Test individual components
    python vlm_extractor.py --test-parsing
    python vlm_extractor.py --test-grounding
    python vlm_extractor.py --demo
    
    # With real API
    python vlm_extractor.py --extract /path/to/screenshots --api-key YOUR_KEY
"""

import os
import re
import json
import base64
import logging
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
import numpy as np

# Optional OpenAI import
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ManeuverType(Enum):
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STOP = "stop"
    REVERSE = "reverse"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, s: str) -> "ManeuverType":
        """Parse string to ManeuverType, with fuzzy matching."""
        s = s.lower().strip().replace(" ", "_").replace("-", "_")
        
        # Direct match
        for member in cls:
            if member.value == s:
                return member
        
        # Fuzzy matching
        if "left" in s and "turn" in s:
            return cls.LEFT_TURN
        if "right" in s and "turn" in s:
            return cls.RIGHT_TURN
        if "left" in s and ("lane" in s or "change" in s):
            return cls.LANE_CHANGE_LEFT
        if "right" in s and ("lane" in s or "change" in s):
            return cls.LANE_CHANGE_RIGHT
        if "accel" in s or "speed" in s and "up" in s:
            return cls.ACCELERATE
        if "decel" in s or "slow" in s or "brak" in s:
            return cls.DECELERATE
        if "stop" in s:
            return cls.STOP
        if "straight" in s or "forward" in s:
            return cls.STRAIGHT
        
        return cls.UNKNOWN


class DecisionType(Enum):
    PROCEED_OR_YIELD = "proceed_or_yield"
    LANE_CHOICE = "lane_choice"
    EVASIVE_ACTION = "evasive_action"
    GAP_ACCEPTANCE = "gap_acceptance"
    SPEED_CHOICE = "speed_choice"
    
    @classmethod
    def from_string(cls, s: str) -> "DecisionType":
        """Parse string to DecisionType."""
        s = s.lower().strip().replace(" ", "_").replace("-", "_")
        
        for member in cls:
            if member.value == s:
                return member
        
        # Fuzzy matching
        if "yield" in s or "proceed" in s:
            return cls.PROCEED_OR_YIELD
        if "lane" in s or "direction" in s or "turn" in s:
            return cls.LANE_CHOICE
        if "evasive" in s or "emergency" in s or "avoid" in s:
            return cls.EVASIVE_ACTION
        if "gap" in s or "merge" in s:
            return cls.GAP_ACCEPTANCE
        if "speed" in s:
            return cls.SPEED_CHOICE
        
        return cls.LANE_CHOICE  # Default


class Aggressiveness(Enum):
    PASSIVE = "passive"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    
    @classmethod
    def from_string(cls, s: str) -> "Aggressiveness":
        """Parse string to Aggressiveness."""
        s = s.lower().strip()
        
        if "passive" in s or "gentle" in s or "slow" in s or "cautious" in s:
            return cls.PASSIVE
        if "aggressive" in s or "fast" in s or "hard" in s or "sharp" in s:
            return cls.AGGRESSIVE
        
        return cls.NORMAL


@dataclass
class TimestampedImage:
    """An image with its timestamp."""
    path: str
    timestamp: float  # Seconds from scenario start
    base64_data: Optional[str] = None  # Loaded image data
    
    def load(self) -> str:
        """Load image as base64."""
        if self.base64_data is None:
            with open(self.path, "rb") as f:
                self.base64_data = base64.b64encode(f.read()).decode()
        return self.base64_data


@dataclass
class CriticalDecisionPoint:
    """A safety-critical decision in the scenario."""
    timestep: int  # Trajectory timestep (if grounded) or frame index
    timestamp: float  # Time in seconds
    decision_type: DecisionType
    ground_truth_choice: str
    alternatives: List[str]
    description: str = ""
    reasoning: str = ""
    confidence: float = 1.0


@dataclass
class ManeuverSegment:
    """A contiguous maneuver segment."""
    maneuver_type: ManeuverType
    start_timestep: int  # Trajectory timestep or frame index
    end_timestep: int
    start_timestamp: float  # Time in seconds
    end_timestamp: float
    aggressiveness: Aggressiveness
    description: str = ""
    reasoning: str = ""
    confidence: float = 1.0


@dataclass 
class ScenarioFeatures:
    """Complete extraction result."""
    scenario_id: str
    maneuver_sequence: List[ManeuverSegment]
    critical_decisions: List[CriticalDecisionPoint]
    
    # Optional trajectory data
    ego_trajectory: Optional[np.ndarray] = None
    
    # Metadata for debugging
    vlm_raw_responses: Dict[str, str] = field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            "scenario_id": self.scenario_id,
            "maneuvers": [
                {
                    "type": m.maneuver_type.value,
                    "start_timestep": m.start_timestep,
                    "end_timestep": m.end_timestep,
                    "start_timestamp": m.start_timestamp,
                    "end_timestamp": m.end_timestamp,
                    "aggressiveness": m.aggressiveness.value,
                    "description": m.description,
                    "reasoning": m.reasoning,
                    "confidence": m.confidence
                }
                for m in self.maneuver_sequence
            ],
            "decisions": [
                {
                    "timestep": d.timestep,
                    "timestamp": d.timestamp,
                    "type": d.decision_type.value,
                    "choice": d.ground_truth_choice,
                    "alternatives": d.alternatives,
                    "description": d.description,
                    "reasoning": d.reasoning,
                    "confidence": d.confidence
                }
                for d in self.critical_decisions
            ],
            "metadata": self.extraction_metadata
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Scenario: {self.scenario_id}",
            f"Maneuvers: {len(self.maneuver_sequence)}",
            f"Decisions: {len(self.critical_decisions)}",
            ""
        ]
        
        for i, m in enumerate(self.maneuver_sequence):
            lines.append(f"  M{i+1}: {m.maneuver_type.value} "
                        f"({m.start_timestamp:.1f}s - {m.end_timestamp:.1f}s) "
                        f"[{m.aggressiveness.value}]")
        
        lines.append("")
        for i, d in enumerate(self.critical_decisions):
            lines.append(f"  D{i+1}: {d.decision_type.value} -> '{d.ground_truth_choice}' "
                        f"@ {d.timestamp:.1f}s")
        
        return "\n".join(lines)


# =============================================================================
# GPT-4o CLIENT
# =============================================================================

class GPT4oClient:
    """Client for GPT-4o API calls."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        if not HAS_OPENAI:
            raise ImportError("openai package required: pip install openai")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.call_count = 0
        self.call_log = []
    
    def complete(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """
        Call GPT-4o with text and optional images.
        
        Args:
            prompt: Text prompt
            images: List of base64-encoded images
            temperature: Sampling temperature
            max_tokens: Max response tokens
        
        Returns:
            Response text
        """
        content = []
        
        # Add images first
        if images:
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "high"
                    }
                })
        
        # Add text prompt
        content.append({"type": "text", "text": prompt})
        
        self.call_count += 1
        call_id = self.call_count
        
        logger.debug(f"GPT-4o call #{call_id}: {len(images) if images else 0} images")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            
            self.call_log.append({
                "call_id": call_id,
                "n_images": len(images) if images else 0,
                "prompt_len": len(prompt),
                "response_len": len(result),
                "success": True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"GPT-4o call #{call_id} failed: {e}")
            self.call_log.append({
                "call_id": call_id,
                "error": str(e),
                "success": False
            })
            raise
    
    def get_call_log(self) -> List[Dict]:
        return self.call_log


class MockGPT4oClient:
    """Mock client for testing without API."""
    
    def __init__(self):
        self.call_count = 0
        self.call_log = []
        self.mock_responses = {}
    
    def set_mock_response(self, prompt_contains: str, response: str):
        """Set a mock response for prompts containing given string."""
        self.mock_responses[prompt_contains] = response
    
    def complete(
        self, 
        prompt: str, 
        images: Optional[List[str]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        self.call_count += 1
        
        # Check for matching mock
        for key, response in self.mock_responses.items():
            if key.lower() in prompt.lower():
                self.call_log.append({
                    "call_id": self.call_count,
                    "matched": key,
                    "n_images": len(images) if images else 0
                })
                return response
        
        # Default mock response
        if "maneuver" in prompt.lower():
            return json.dumps({
                "maneuvers": [
                    {
                        "type": "straight",
                        "start_time": 0.0,
                        "end_time": 2.5,
                        "aggressiveness": "normal",
                        "description": "Vehicle traveling straight"
                    }
                ],
                "overall_description": "Mock maneuver extraction"
            })
        elif "decision" in prompt.lower():
            return json.dumps({
                "decisions": [],
                "risk_level": "low",
                "risk_explanation": "Mock decision extraction"
            })
        
        return "{}"
    
    def get_call_log(self) -> List[Dict]:
        return self.call_log


# =============================================================================
# VLM EXTRACTOR
# =============================================================================

class VLMSafetyCriticalExtractor:
    """
    Extract safety-critical features from simulator screenshots using VLM.
    
    Designed for:
    - Arbitrary number of screenshots
    - Screenshots with timestamps
    - Easy prompt iteration
    - Comprehensive debugging
    """
    
    # =========================================================================
    # PROMPTS - Edit these to tune extraction
    # =========================================================================
    
    MANEUVER_PROMPT_TEMPLATE = """Analyze these simulator screenshots showing a driving scenario.
The images are in chronological order with the following timestamps: {timestamps}

For each image, I'll tell you its timestamp in seconds from the start of the scenario.

The ego vehicle is the GREEN car. There will only ever be ONE green car in any scene.
Only describe actions of the green car; ignore all other vehicles.
If you are unsure about the ego vehicle in a frame, state that in reasoning and skip the maneuver.
Identify ALL MANEUVERS the ego vehicle (the green car) performs throughout this sequence.

Ego state per frame (ground truth from simulation):
{ego_state}

Maneuver types to look for:
- straight: Vehicle maintaining lane and direction
- left_turn: Vehicle turning left (e.g., at intersection)
- right_turn: Vehicle turning right
- lane_change_left: Vehicle changing to left lane
- lane_change_right: Vehicle changing to right lane
- accelerate: Vehicle noticeably speeding up
- decelerate: Vehicle noticeably slowing down
- stop: Vehicle coming to a stop

For each maneuver, estimate:
1. Start time (seconds) - when this maneuver begins
2. End time (seconds) - when this maneuver ends
3. Aggressiveness: passive (gentle/slow), normal, or aggressive (fast/sharp)
4. Reasoning: reference the frames (timestamps), where the green car is, and
   what motion between those frames indicates the maneuver

Respond ONLY with valid JSON:
{{
    "maneuvers": [
        {{
            "type": "<maneuver_type>",
            "start_time": <float>,
            "end_time": <float>,
            "aggressiveness": "passive" | "normal" | "aggressive",
            "description": "<brief description of what you observe>",
            "reasoning": "<which frames, where the green car is, what motion indicates the maneuver>"
        }}
    ],
    "overall_description": "<one sentence summary of the entire sequence>"
}}

Be precise with timestamps. If a maneuver spans multiple images, estimate the actual start/end times.
If unsure about exact timing, use the timestamps of the images as reference points."""

    DECISION_PROMPT_TEMPLATE = """Analyze these simulator screenshots showing a driving scenario.
The images are in chronological order with the following timestamps: {timestamps}

The ego vehicle is the GREEN car. There will only ever be ONE green car in any scene.
Only describe decisions made by the green car; ignore other vehicles.
If you are unsure about the ego vehicle in a frame, state that in reasoning and skip the decision.
Identify any CRITICAL SAFETY DECISIONS the ego driver made or should have made.

Ego state per frame (ground truth from simulation):
{ego_state}

Decision types to look for:
- proceed_or_yield: Choosing to proceed vs yield/wait (at intersections, crossings)
- lane_choice: Choosing which lane or direction to take
- evasive_action: Taking or not taking emergency action (hard braking, swerving)
- gap_acceptance: Accepting or rejecting a gap when merging/crossing
- speed_choice: Choosing to speed up, maintain, or slow down

For each decision:
1. Time (seconds) - when this decision point occurs
2. What choice was made
3. What alternatives existed
4. Why this was safety-critical
5. Reasoning: reference the frames (timestamps), where the green car is, and
   what motion/context indicates the decision point

Respond ONLY with valid JSON:
{{
    "decisions": [
        {{
            "type": "<decision_type>",
            "time": <float>,
            "choice_made": "<what the driver did>",
            "alternatives": ["<other>", "<options>", "<available>"],
            "description": "<why this was a critical decision>",
            "reasoning": "<which frames, where the green car is, what indicates the decision>",
            "confidence": <0.0-1.0>
        }}
    ],
    "risk_level": "low" | "medium" | "high",
    "risk_explanation": "<brief explanation of overall risk in this scenario>"
}}

Only include decisions that are genuinely safety-relevant. Not every action is a critical decision."""

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(
        self, 
        client: Union[GPT4oClient, MockGPT4oClient],
        debug: bool = True,
        max_images_per_call: int = 10,
        debug_output_dir: Optional[str] = None
    ):
        """
        Args:
            client: GPT4o client (real or mock)
            debug: Enable verbose logging
            max_images_per_call: Max images to send in one API call
        """
        self.client = client
        self.debug = debug
        self.max_images_per_call = max_images_per_call
        self.extraction_log = []
        self.debug_output_dir = debug_output_dir
    
    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================
    
    def extract(
        self,
        images: List[TimestampedImage],
        scenario_id: str = "unknown",
        trajectory: Optional[np.ndarray] = None
    ) -> ScenarioFeatures:
        """
        Extract safety-critical features from timestamped images.
        
        Args:
            images: List of TimestampedImage objects
            scenario_id: Identifier for logging
            trajectory: Optional [T, 4] trajectory for timestep grounding
        
        Returns:
            ScenarioFeatures with extracted maneuvers and decisions
        """
        
        if not images:
            logger.warning(f"No images provided for {scenario_id}")
            return self._empty_features(scenario_id, "No images provided")
        
        # Sort by timestamp
        images = sorted(images, key=lambda x: x.timestamp)
        
        if self.debug:
            logger.info(f"Extracting from {len(images)} images for {scenario_id}")
            logger.info(f"Time range: {images[0].timestamp:.2f}s - {images[-1].timestamp:.2f}s")
        
        # Load images
        for img in images:
            img.load()

        ego_state_summary = self._build_ego_state_summary(images, trajectory)

        # Extract maneuvers
        maneuvers, maneuver_raw, maneuver_prompt = self._extract_maneuvers(images, ego_state_summary)
        
        # Extract decisions
        decisions, decision_raw, decision_prompt = self._extract_decisions(images, ego_state_summary)
        
        # Ground to trajectory timesteps if provided
        if trajectory is not None:
            maneuvers = self._ground_maneuvers_to_trajectory(maneuvers, trajectory)
            decisions = self._ground_decisions_to_trajectory(decisions, trajectory)
        
        # Build result
        features = ScenarioFeatures(
            scenario_id=scenario_id,
            maneuver_sequence=maneuvers,
            critical_decisions=decisions,
            ego_trajectory=trajectory,
            vlm_raw_responses={
                "maneuvers": maneuver_raw,
                "decisions": decision_raw
            },
            extraction_metadata={
                "n_images": len(images),
                "time_range": (images[0].timestamp, images[-1].timestamp),
                "timestamps": [img.timestamp for img in images],
                "has_trajectory": trajectory is not None
            }
        )

        # Optional: save prompt/response logs to disk
        if self.debug_output_dir:
            self._save_debug_log(
                output_dir=self.debug_output_dir,
                scenario_id=scenario_id,
                images=images,
                ego_state_summary=ego_state_summary,
                maneuver_prompt=maneuver_prompt,
                maneuver_response=maneuver_raw,
                decision_prompt=decision_prompt,
                decision_response=decision_raw,
            )
        
        # Log extraction
        self.extraction_log.append({
            "scenario_id": scenario_id,
            "n_images": len(images),
            "n_maneuvers": len(maneuvers),
            "n_decisions": len(decisions)
        })
        
        if self.debug:
            logger.info(f"Extracted {len(maneuvers)} maneuvers, {len(decisions)} decisions")
            logger.info("\n" + features.summary())
        
        return features
    
    def extract_from_directory(
        self,
        directory: str,
        scenario_id: Optional[str] = None,
        trajectory: Optional[np.ndarray] = None,
        timestamp_pattern: str = r"(\d+\.?\d*)"
    ) -> ScenarioFeatures:
        """
        Extract from a directory of screenshots.
        
        Expects filenames like: frame_0.5.png, screenshot_1.0.jpg, 001.500.png
        Will extract timestamp from filename using regex pattern.
        
        Args:
            directory: Path to directory with screenshots
            scenario_id: Optional ID (defaults to directory name)
            trajectory: Optional trajectory for grounding
            timestamp_pattern: Regex pattern to extract timestamp from filename
        
        Returns:
            ScenarioFeatures
        """
        
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        if scenario_id is None:
            scenario_id = dir_path.name
        
        # Find image files
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        images = []
        
        for p in sorted(dir_path.iterdir()):
            if p.suffix.lower() in image_extensions:
                # Try to extract timestamp from filename
                timestamp = self._extract_timestamp_from_filename(p.name, timestamp_pattern)
                
                if timestamp is not None:
                    images.append(TimestampedImage(
                        path=str(p),
                        timestamp=timestamp
                    ))
                else:
                    logger.warning(f"Could not extract timestamp from: {p.name}")
        
        if not images:
            logger.warning(f"No valid images found in {directory}")
            return self._empty_features(scenario_id, f"No images in {directory}")
        
        if self.debug:
            logger.info(f"Found {len(images)} images in {directory}")
        
        return self.extract(images, scenario_id, trajectory)
    
    # =========================================================================
    # MANEUVER EXTRACTION
    # =========================================================================
    
    def _extract_maneuvers(
        self, 
        images: List[TimestampedImage],
        ego_state_summary: str
    ) -> Tuple[List[ManeuverSegment], str, str]:
        """Extract maneuvers using VLM."""
        
        # Build timestamp string for prompt
        timestamps_str = ", ".join([f"{img.timestamp:.2f}s" for img in images])
        prompt = self.MANEUVER_PROMPT_TEMPLATE.format(
            timestamps=timestamps_str,
            ego_state=ego_state_summary
        )
        
        # Get base64 images (limit if too many)
        image_data = [img.base64_data for img in images[:self.max_images_per_call]]
        
        if len(images) > self.max_images_per_call:
            logger.warning(f"Truncating to {self.max_images_per_call} images")
        
        try:
            response = self.client.complete(prompt, images=image_data)
            
            if self.debug:
                logger.debug(f"Maneuver VLM response:\n{response[:500]}...")
            
            parsed = self._parse_json_response(response)
            
            if "error" in parsed:
                logger.warning(f"Maneuver parse error: {parsed['error']}")
                return [], response, prompt
            
            maneuvers = []
            for m in parsed.get("maneuvers", []):
                maneuver = self._parse_maneuver_dict(m)
                if maneuver:
                    maneuvers.append(maneuver)
            
            return maneuvers, response, prompt
            
        except Exception as e:
            logger.error(f"Maneuver extraction failed: {e}")
            return [], str(e), prompt
    
    def _parse_maneuver_dict(self, m: Dict) -> Optional[ManeuverSegment]:
        """Parse a maneuver dict from VLM response."""
        try:
            maneuver_type = ManeuverType.from_string(m.get("type", "unknown"))
            aggressiveness = Aggressiveness.from_string(m.get("aggressiveness", "normal"))
            
            start_time = float(m.get("start_time", 0))
            end_time = float(m.get("end_time", start_time + 1))
            
            return ManeuverSegment(
                maneuver_type=maneuver_type,
                start_timestep=0,  # Will be grounded later
                end_timestep=0,
                start_timestamp=start_time,
                end_timestamp=end_time,
                aggressiveness=aggressiveness,
                description=m.get("description", ""),
                reasoning=m.get("reasoning", ""),
                confidence=float(m.get("confidence", 1.0))
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse maneuver {m}: {e}")
            return None
    
    # =========================================================================
    # DECISION EXTRACTION
    # =========================================================================
    
    def _extract_decisions(
        self, 
        images: List[TimestampedImage],
        ego_state_summary: str
    ) -> Tuple[List[CriticalDecisionPoint], str, str]:
        """Extract decisions using VLM."""
        
        timestamps_str = ", ".join([f"{img.timestamp:.2f}s" for img in images])
        prompt = self.DECISION_PROMPT_TEMPLATE.format(
            timestamps=timestamps_str,
            ego_state=ego_state_summary
        )
        
        image_data = [img.base64_data for img in images[:self.max_images_per_call]]
        
        try:
            response = self.client.complete(prompt, images=image_data)
            
            if self.debug:
                logger.debug(f"Decision VLM response:\n{response[:500]}...")
            
            parsed = self._parse_json_response(response)
            
            if "error" in parsed:
                logger.warning(f"Decision parse error: {parsed['error']}")
                return [], response, prompt
            
            decisions = []
            for d in parsed.get("decisions", []):
                decision = self._parse_decision_dict(d)
                if decision:
                    decisions.append(decision)
            
            return decisions, response, prompt
            
        except Exception as e:
            logger.error(f"Decision extraction failed: {e}")
            return [], str(e), prompt

    def _save_debug_log(
        self,
        output_dir: str,
        scenario_id: str,
        images: List[TimestampedImage],
        ego_state_summary: Optional[str],
        maneuver_prompt: str,
        maneuver_response: str,
        decision_prompt: str,
        decision_response: str,
    ) -> None:
        """Save VLM prompt/response logs to disk."""
        from pathlib import Path

        try:
            debug_dir = Path(output_dir) / "vlm_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = debug_dir / f"{scenario_id}_vlm_{timestamp}.json"

            debug_data = {
                "scenario_id": scenario_id,
                "timestamps": [img.timestamp for img in images],
                "image_paths": [img.path for img in images],
                "ego_state": ego_state_summary,
                "maneuver_prompt": maneuver_prompt,
                "maneuver_response": maneuver_response,
                "decision_prompt": decision_prompt,
                "decision_response": decision_response,
            }

            with open(log_path, "w") as f:
                json.dump(debug_data, f, indent=2)

            logger.debug(f"Saved VLM debug log to {log_path}")
        except Exception as e:
            logger.warning(f"Failed to save VLM debug log: {e}")
    
    def _parse_decision_dict(self, d: Dict) -> Optional[CriticalDecisionPoint]:
        """Parse a decision dict from VLM response."""
        try:
            decision_type = DecisionType.from_string(d.get("type", "lane_choice"))
            
            timestamp = float(d.get("time", 0))
            choice = d.get("choice_made", "unknown")
            alternatives = d.get("alternatives", [choice])
            
            if choice not in alternatives:
                alternatives = [choice] + alternatives
            
            return CriticalDecisionPoint(
                timestep=0,  # Will be grounded later
                timestamp=timestamp,
                decision_type=decision_type,
                ground_truth_choice=choice,
                alternatives=alternatives,
                description=d.get("description", ""),
                reasoning=d.get("reasoning", ""),
                confidence=float(d.get("confidence", 1.0))
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse decision {d}: {e}")
            return None

    def _build_ego_state_summary(
        self,
        images: List[TimestampedImage],
        trajectory: Optional[np.ndarray],
        dt: float = 0.1,
    ) -> str:
        """Build a per-frame ego state summary aligned to image timestamps."""
        if trajectory is None or len(trajectory) == 0:
            return "Ego state unavailable."

        traj = np.asarray(trajectory)
        if traj.shape[1] < 4:
            return "Ego state unavailable."

        headings = np.unwrap(traj[:, 2].astype(float))
        speeds = traj[:, 3].astype(float)
        n = len(traj)

        def idx_from_time(t: float) -> int:
            return max(0, min(n - 1, int(round(t / dt))))

        lines = []
        for img in images:
            idx = idx_from_time(img.timestamp)
            pos = (float(traj[idx, 0]), float(traj[idx, 1]))
            heading = float(headings[idx])
            speed = float(speeds[idx])

            # Acceleration (finite difference on speed)
            if 0 < idx < n - 1:
                acc = (speeds[idx + 1] - speeds[idx - 1]) / (2 * dt)
            elif idx == 0 and n > 1:
                acc = (speeds[1] - speeds[0]) / dt
            elif n > 1:
                acc = (speeds[-1] - speeds[-2]) / dt
            else:
                acc = 0.0

            # Yaw rate (finite difference on heading)
            if 0 < idx < n - 1:
                yaw_rate = (headings[idx + 1] - headings[idx - 1]) / (2 * dt)
            elif idx == 0 and n > 1:
                yaw_rate = (headings[1] - headings[0]) / dt
            elif n > 1:
                yaw_rate = (headings[-1] - headings[-2]) / dt
            else:
                yaw_rate = 0.0

            lines.append(
                f"- t={img.timestamp:.2f}s: pos=({pos[0]:.2f}, {pos[1]:.2f}), "
                f"heading={heading:.3f} rad, speed={speed:.2f} m/s, "
                f"accel={acc:.2f} m/s^2, yaw_rate={yaw_rate:.3f} rad/s"
            )

        return "\n".join(lines)
    
    # =========================================================================
    # TRAJECTORY GROUNDING
    # =========================================================================
    
    def _ground_maneuvers_to_trajectory(
        self,
        maneuvers: List[ManeuverSegment],
        trajectory: np.ndarray
    ) -> List[ManeuverSegment]:
        """
        Convert maneuver timestamps to trajectory timesteps.
        
        Assumes trajectory is sampled at 10Hz (0.1s per step).
        """
        dt = 0.1  # Trajectory timestep
        T = len(trajectory)
        max_time = T * dt
        
        grounded = []
        for m in maneuvers:
            start_step = int(m.start_timestamp / dt)
            end_step = int(m.end_timestamp / dt)
            
            # Clamp to trajectory bounds
            start_step = max(0, min(start_step, T - 1))
            end_step = max(start_step + 1, min(end_step, T))
            
            grounded.append(ManeuverSegment(
                maneuver_type=m.maneuver_type,
                start_timestep=start_step,
                end_timestep=end_step,
                start_timestamp=m.start_timestamp,
                end_timestamp=m.end_timestamp,
                aggressiveness=m.aggressiveness,
                description=m.description,
                confidence=m.confidence
            ))
        
        return grounded
    
    def _ground_decisions_to_trajectory(
        self,
        decisions: List[CriticalDecisionPoint],
        trajectory: np.ndarray
    ) -> List[CriticalDecisionPoint]:
        """Convert decision timestamps to trajectory timesteps."""
        dt = 0.1
        T = len(trajectory)
        
        grounded = []
        for d in decisions:
            timestep = int(d.timestamp / dt)
            timestep = max(0, min(timestep, T - 1))
            
            grounded.append(CriticalDecisionPoint(
                timestep=timestep,
                timestamp=d.timestamp,
                decision_type=d.decision_type,
                ground_truth_choice=d.ground_truth_choice,
                alternatives=d.alternatives,
                description=d.description,
                confidence=d.confidence
            ))
        
        return grounded
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from VLM response, handling markdown code blocks."""
        try:
            resp = response.strip()
            
            # Remove markdown code blocks
            if resp.startswith("```"):
                lines = resp.split("\n")
                # Find end of code block
                end_idx = len(lines)
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == "```":
                        end_idx = i
                        break
                resp = "\n".join(lines[1:end_idx])
            
            return json.loads(resp)
            
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse failed: {e}", "raw": response[:300]}
    
    def _extract_timestamp_from_filename(
        self, 
        filename: str, 
        pattern: str
    ) -> Optional[float]:
        """Extract timestamp from filename using regex."""
        try:
            # Try the provided pattern
            match = re.search(pattern, filename)
            if match:
                return float(match.group(1))
            
            # Try common patterns
            common_patterns = [
                r"_(\d+\.?\d*)s?\.",     # frame_1.5.png, frame_1.5s.png
                r"(\d+\.?\d*)_",          # 1.5_frame.png
                r"^(\d+\.?\d*)\.",        # 1.5.png
                r"_t(\d+\.?\d*)",          # frame_t1.5.png
            ]
            
            for pat in common_patterns:
                match = re.search(pat, filename)
                if match:
                    return float(match.group(1))
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def _empty_features(self, scenario_id: str, error: str) -> ScenarioFeatures:
        """Return empty features on error."""
        return ScenarioFeatures(
            scenario_id=scenario_id,
            maneuver_sequence=[],
            critical_decisions=[],
            extraction_metadata={"error": error}
        )
    
    def get_extraction_log(self) -> List[Dict]:
        """Return log of all extractions."""
        return self.extraction_log


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_enum_parsing():
    """Test enum parsing from strings."""
    print("=" * 60)
    print("Testing Enum Parsing")
    print("=" * 60)
    
    # ManeuverType tests
    test_cases = [
        ("straight", ManeuverType.STRAIGHT),
        ("left_turn", ManeuverType.LEFT_TURN),
        ("LEFT TURN", ManeuverType.LEFT_TURN),
        ("turning left", ManeuverType.LEFT_TURN),
        ("lane change right", ManeuverType.LANE_CHANGE_RIGHT),
        ("slowing down", ManeuverType.DECELERATE),
        ("braking", ManeuverType.DECELERATE),
        ("speeding up", ManeuverType.ACCELERATE),
        ("gibberish", ManeuverType.UNKNOWN),
    ]
    
    print("\nManeuverType parsing:")
    for input_str, expected in test_cases:
        result = ManeuverType.from_string(input_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_str}' -> {result.value} (expected: {expected.value})")
    
    # Aggressiveness tests
    agg_cases = [
        ("passive", Aggressiveness.PASSIVE),
        ("AGGRESSIVE", Aggressiveness.AGGRESSIVE),
        ("gentle", Aggressiveness.PASSIVE),
        ("sharp turn", Aggressiveness.AGGRESSIVE),
        ("normal", Aggressiveness.NORMAL),
    ]
    
    print("\nAggressiveness parsing:")
    for input_str, expected in agg_cases:
        result = Aggressiveness.from_string(input_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_str}' -> {result.value} (expected: {expected.value})")
    
    print("\n✓ Enum parsing tests complete")


def test_json_parsing():
    """Test JSON response parsing."""
    print("=" * 60)
    print("Testing JSON Parsing")
    print("=" * 60)
    
    extractor = VLMSafetyCriticalExtractor(MockGPT4oClient(), debug=False)
    
    # Test cases
    test_cases = [
        # Clean JSON
        ('{"maneuvers": []}', True),
        
        # Markdown code block
        ('```json\n{"maneuvers": []}\n```', True),
        
        # Markdown with language tag
        ('```\n{"maneuvers": []}\n```', True),
        
        # Invalid JSON
        ('{"maneuvers": [}', False),
        
        # Text before JSON
        ('Here is the response:\n{"maneuvers": []}', False),  # This will fail
    ]
    
    for input_str, should_succeed in test_cases:
        result = extractor._parse_json_response(input_str)
        has_error = "error" in result
        status = "✓" if (not has_error) == should_succeed else "✗"
        print(f"  {status} Parse {'succeeded' if not has_error else 'failed'}: "
              f"{input_str[:40]}...")
    
    print("\n✓ JSON parsing tests complete")


def test_timestamp_extraction():
    """Test timestamp extraction from filenames."""
    print("=" * 60)
    print("Testing Timestamp Extraction")
    print("=" * 60)
    
    extractor = VLMSafetyCriticalExtractor(MockGPT4oClient(), debug=False)
    
    test_cases = [
        ("frame_1.5.png", 1.5),
        ("screenshot_0.0.jpg", 0.0),
        ("2.5_capture.png", 2.5),
        ("3.75.png", 3.75),
        ("frame_t4.2.bmp", 4.2),
        ("image.png", None),  # No timestamp
        ("frame_abc.png", None),  # Invalid
    ]
    
    for filename, expected in test_cases:
        result = extractor._extract_timestamp_from_filename(filename, r"(\d+\.?\d*)")
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{filename}' -> {result} (expected: {expected})")
    
    print("\n✓ Timestamp extraction tests complete")


def test_trajectory_grounding():
    """Test grounding timestamps to trajectory timesteps."""
    print("=" * 60)
    print("Testing Trajectory Grounding")
    print("=" * 60)
    
    extractor = VLMSafetyCriticalExtractor(MockGPT4oClient(), debug=False)
    
    # Create fake trajectory: 10 seconds at 10Hz = 100 timesteps
    T = 100
    trajectory = np.zeros((T, 4))
    
    # Create test maneuvers with known timestamps
    maneuvers = [
        ManeuverSegment(
            maneuver_type=ManeuverType.STRAIGHT,
            start_timestep=0,
            end_timestep=0,
            start_timestamp=0.0,
            end_timestamp=2.5,
            aggressiveness=Aggressiveness.NORMAL
        ),
        ManeuverSegment(
            maneuver_type=ManeuverType.LEFT_TURN,
            start_timestep=0,
            end_timestep=0,
            start_timestamp=2.5,
            end_timestamp=5.0,
            aggressiveness=Aggressiveness.AGGRESSIVE
        ),
    ]
    
    grounded = extractor._ground_maneuvers_to_trajectory(maneuvers, trajectory)
    
    print("\nManeuver grounding (10Hz trajectory, 100 timesteps):")
    for orig, gnd in zip(maneuvers, grounded):
        print(f"  {orig.maneuver_type.value}:")
        print(f"    Time: {orig.start_timestamp}s - {orig.end_timestamp}s")
        print(f"    Steps: {gnd.start_timestep} - {gnd.end_timestep}")
        
        # Verify
        expected_start = int(orig.start_timestamp / 0.1)
        expected_end = int(orig.end_timestamp / 0.1)
        assert gnd.start_timestep == expected_start, f"Start mismatch: {gnd.start_timestep} vs {expected_start}"
        assert gnd.end_timestep == expected_end, f"End mismatch: {gnd.end_timestep} vs {expected_end}"
    
    print("\n✓ Trajectory grounding tests complete")


def test_mock_extraction():
    """Test full extraction with mock client."""
    print("=" * 60)
    print("Testing Mock Extraction")
    print("=" * 60)
    
    # Setup mock client with custom responses
    client = MockGPT4oClient()
    
    client.set_mock_response("MANEUVERS", json.dumps({
        "maneuvers": [
            {
                "type": "straight",
                "start_time": 0.0,
                "end_time": 2.0,
                "aggressiveness": "normal",
                "description": "Vehicle going straight"
            },
            {
                "type": "left_turn",
                "start_time": 2.0,
                "end_time": 4.0,
                "aggressiveness": "aggressive",
                "description": "Sharp left turn at intersection"
            }
        ],
        "overall_description": "Vehicle approaches and turns left"
    }))
    
    client.set_mock_response("CRITICAL", json.dumps({
        "decisions": [
            {
                "type": "proceed_or_yield",
                "time": 1.5,
                "choice_made": "proceed",
                "alternatives": ["proceed", "yield"],
                "description": "Chose to enter intersection",
                "confidence": 0.9
            }
        ],
        "risk_level": "medium",
        "risk_explanation": "Intersection crossing"
    }))
    
    # Create fake images
    images = [
        TimestampedImage(path="fake1.png", timestamp=0.0, base64_data="fake"),
        TimestampedImage(path="fake2.png", timestamp=1.0, base64_data="fake"),
        TimestampedImage(path="fake3.png", timestamp=2.0, base64_data="fake"),
        TimestampedImage(path="fake4.png", timestamp=3.0, base64_data="fake"),
    ]
    
    # Create trajectory for grounding
    trajectory = np.zeros((50, 4))  # 5 seconds at 10Hz
    
    # Extract
    extractor = VLMSafetyCriticalExtractor(client, debug=True)
    features = extractor.extract(images, "test_scenario", trajectory)
    
    print("\n" + "=" * 40)
    print("Extraction Result:")
    print("=" * 40)
    print(features.summary())
    
    print("\nAs dict:")
    print(json.dumps(features.to_dict(), indent=2))
    
    # Verify
    assert len(features.maneuver_sequence) == 2, f"Expected 2 maneuvers, got {len(features.maneuver_sequence)}"
    assert len(features.critical_decisions) == 1, f"Expected 1 decision, got {len(features.critical_decisions)}"
    
    # Check grounding
    assert features.maneuver_sequence[0].start_timestep == 0
    assert features.maneuver_sequence[0].end_timestep == 20  # 2.0s at 10Hz
    
    print("\n✓ Mock extraction tests complete")


def demo():
    """Demo the full extraction pipeline."""
    print("=" * 60)
    print("VLM Safety-Critical Extraction Demo")
    print("=" * 60)
    
    # Run all tests
    test_enum_parsing()
    print()
    test_json_parsing()
    print()
    test_timestamp_extraction()
    print()
    test_trajectory_grounding()
    print()
    test_mock_extraction()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VLM-based safety-critical feature extraction"
    )
    
    parser.add_argument(
        "--test-parsing", 
        action="store_true",
        help="Test enum and JSON parsing"
    )
    parser.add_argument(
        "--test-grounding",
        action="store_true", 
        help="Test trajectory grounding"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demo with mock client"
    )
    parser.add_argument(
        "--extract",
        type=str,
        help="Extract from directory of screenshots"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test_parsing:
        test_enum_parsing()
        test_json_parsing()
    elif args.test_grounding:
        test_timestamp_extraction()
        test_trajectory_grounding()
    elif args.demo:
        demo()
    elif args.extract:
        # Real extraction
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        
        try:
            client = GPT4oClient()
        except (ImportError, ValueError) as e:
            print(f"Error: {e}")
            print("Using mock client instead")
            client = MockGPT4oClient()
        
        extractor = VLMSafetyCriticalExtractor(client, debug=True)
        features = extractor.extract_from_directory(args.extract)
        
        print("\n" + features.summary())
        
        # Save result
        output_path = Path(args.extract) / "extraction_result.json"
        with open(output_path, "w") as f:
            json.dump(features.to_dict(), f, indent=2)
        print(f"\nSaved to: {output_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()