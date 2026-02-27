"""Core typed contracts for CounterBMT v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class ManeuverType(str, Enum):
    STRAIGHT = "straight"
    LEFT_TURN = "left_turn"
    RIGHT_TURN = "right_turn"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STOP = "stop"
    UNKNOWN = "unknown"


class DecisionType(str, Enum):
    PROCEED_OR_YIELD = "proceed_or_yield"
    LANE_CHOICE = "lane_choice"
    EVASIVE_ACTION = "evasive_action"
    GAP_ACCEPTANCE = "gap_acceptance"
    SPEED_CHOICE = "speed_choice"
    UNKNOWN = "unknown"


@dataclass
class TimestampedFrame:
    path: str
    timestamp_s: float


@dataclass
class ScenarioInput:
    scenario_id: str
    frames: List[TimestampedFrame] = field(default_factory=list)
    ego_trajectory_xy: Optional[np.ndarray] = None  # [T, 2]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ManeuverSegment:
    maneuver_type: ManeuverType
    start_s: float
    end_s: float
    aggressiveness: str = "normal"
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class DecisionPoint:
    decision_type: DecisionType
    timestamp_s: float
    choice: str
    alternatives: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class VLMFeatures:
    scenario_id: str
    maneuvers: List[ManeuverSegment] = field(default_factory=list)
    decisions: List[DecisionPoint] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGNode:
    node_id: str
    node_type: str
    value: Any
    timestamp_s: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGEdge:
    parent_id: str
    child_id: str
    confidence: float = 1.0
    mechanism: str = ""


@dataclass
class BayesianDAG:
    scenario_id: str
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    edges: List[DAGEdge] = field(default_factory=list)
    cpts: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Intervention:
    variable: str
    value: Any
    original_value: Any = None
    timestamp_s: Optional[float] = None
    aggressiveness: str = "normal"
    description: str = ""


@dataclass
class ConditioningSignal:
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryRollout:
    trajectory_xy: np.ndarray
    conditioning: ConditioningSignal
    sample_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    reward: float
    matched: bool
    explanation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardBreakdown:
    alignment: float
    safety: float
    realism: float
    total: float
    novelty: float = 0.0
    consensus: float = 0.0
    total_env: float = 0.0
    total_augmented: float = 0.0


@dataclass
class RLBatchDiagnostics:
    entropy: float
    cluster_hist: Dict[str, int] = field(default_factory=dict)
    thermostat_eta: float = 0.0
    thermostat_alpha: float = 0.0


@dataclass
class PipelineResult:
    scenario_id: str
    features: VLMFeatures
    dag: BayesianDAG
    intervention: Intervention
    rollouts: List[TrajectoryRollout]
    judge_results: List[JudgeResult]
    rewards: List[RewardBreakdown]

    def to_dict(self) -> Dict[str, Any]:
        # Keep this JSON-friendly for quick experiment logging.
        return _jsonify(asdict(self))


def make_demo_frames(prefix: str, timestamps: Sequence[float]) -> List[TimestampedFrame]:
    return [TimestampedFrame(path=f"{prefix}_{i:03d}.png", timestamp_s=t) for i, t in enumerate(timestamps)]


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj
