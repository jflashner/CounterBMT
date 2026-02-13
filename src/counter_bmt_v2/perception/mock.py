"""Mock perception implementation for fast vertical-slice execution."""

from __future__ import annotations

from counter_bmt_v2.contracts import (
    DecisionPoint,
    DecisionType,
    ManeuverSegment,
    ManeuverType,
    ScenarioInput,
    VLMFeatures,
)
from counter_bmt_v2.perception.base import PerceptionModel


class MockPerceptionModel(PerceptionModel):
    def extract(self, scene: ScenarioInput) -> VLMFeatures:
        maneuvers = [
            ManeuverSegment(
                maneuver_type=ManeuverType.STRAIGHT,
                start_s=0.0,
                end_s=2.0,
                aggressiveness="normal",
                confidence=0.9,
                reasoning="Mock: ego remains lane-centered in initial frames.",
            ),
            ManeuverSegment(
                maneuver_type=ManeuverType.LANE_CHANGE_RIGHT,
                start_s=2.0,
                end_s=4.0,
                aggressiveness="normal",
                confidence=0.75,
                reasoning="Mock: slight rightward lateral displacement.",
            ),
        ]
        decisions = [
            DecisionPoint(
                decision_type=DecisionType.PROCEED_OR_YIELD,
                timestamp_s=1.8,
                choice="proceed",
                alternatives=["proceed", "yield"],
                confidence=0.8,
                reasoning="Mock: no close conflict vehicle in intersection zone.",
            )
        ]
        return VLMFeatures(
            scenario_id=scene.scenario_id,
            maneuvers=maneuvers,
            decisions=decisions,
            raw={"backend": "mock"},
        )
