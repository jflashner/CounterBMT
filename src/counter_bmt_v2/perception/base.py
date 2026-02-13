"""Perception stage interfaces."""

from __future__ import annotations

from typing import Protocol

from counter_bmt_v2.contracts import ScenarioInput, VLMFeatures


class PerceptionModel(Protocol):
    def extract(self, scene: ScenarioInput) -> VLMFeatures:
        """Extract structured features from scene frames."""
