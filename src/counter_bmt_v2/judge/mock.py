"""Mock judge for alignment scoring."""

from __future__ import annotations

import numpy as np

from counter_bmt_v2.contracts import Intervention, JudgeResult, TrajectoryRollout
from counter_bmt_v2.judge.base import TrajectoryJudge


class MockTrajectoryJudge(TrajectoryJudge):
    def evaluate(self, intervention: Intervention, rollout: TrajectoryRollout) -> JudgeResult:
        traj = rollout.trajectory_xy
        if len(traj) < 2:
            return JudgeResult(reward=0.0, matched=False, explanation="trajectory too short")

        dx = float(traj[-1, 0] - traj[0, 0])
        dy = float(traj[-1, 1] - traj[0, 1])

        val = str(intervention.value)
        score = 0.3

        if "left" in val:
            score = 0.5 + float(np.clip(dy, -5.0, 5.0)) / 10.0
        elif "right" in val:
            score = 0.5 - float(np.clip(dy, -5.0, 5.0)) / 10.0
        elif "stop" in val:
            speed_like = np.mean(np.linalg.norm(np.diff(traj, axis=0), axis=1))
            score = 1.0 - float(np.clip(speed_like, 0.0, 1.0))
        elif "accelerate" in val:
            score = 0.5 + float(np.clip(dx, -10.0, 10.0)) / 20.0

        score = float(np.clip(score, 0.0, 1.0))
        return JudgeResult(
            reward=score,
            matched=score >= 0.6,
            explanation=f"mock judge score from displacement dx={dx:.2f}, dy={dy:.2f}",
            details={"dx": dx, "dy": dy},
        )
