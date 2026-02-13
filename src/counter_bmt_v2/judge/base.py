"""Trajectory judging interfaces."""

from __future__ import annotations

from typing import Protocol

from counter_bmt_v2.contracts import Intervention, JudgeResult, TrajectoryRollout


class TrajectoryJudge(Protocol):
    def evaluate(self, intervention: Intervention, rollout: TrajectoryRollout) -> JudgeResult:
        """Score whether trajectory aligns with intervention intent."""
