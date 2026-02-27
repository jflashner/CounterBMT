"""Reward composition for RL rollouts."""

from __future__ import annotations

import numpy as np

from counter_bmt_v2.config import RewardConfig
from counter_bmt_v2.contracts import JudgeResult, RewardBreakdown, TrajectoryRollout


def _estimate_realism(rollout: TrajectoryRollout) -> float:
    traj = rollout.trajectory_xy
    if len(traj) < 3:
        return 0.0

    vel = np.diff(traj, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)

    jerk_mag = float(np.mean(np.linalg.norm(jerk, axis=1))) if len(jerk) else 0.0
    return float(np.clip(1.0 - jerk_mag, 0.0, 1.0))


def _estimate_safety(rollout: TrajectoryRollout) -> float:
    # Placeholder: neutral safety prior until collision checks are integrated.
    return 0.8


def compose_reward(judge: JudgeResult, rollout: TrajectoryRollout, cfg: RewardConfig) -> RewardBreakdown:
    alignment = float(np.clip(judge.reward, 0.0, 1.0))
    safety = _estimate_safety(rollout)
    realism = _estimate_realism(rollout)
    total_env = (
        cfg.w_alignment * alignment
        + cfg.w_safety * safety
        + cfg.w_realism * realism
    )
    novelty = float(rollout.metadata.get("novelty_score", 0.0))
    consensus = float(rollout.metadata.get("consensus_score", 0.0))
    total_augmented = float(
        total_env
        + cfg.w_novelty * novelty
        + cfg.w_consensus * consensus
    )
    return RewardBreakdown(
        alignment=alignment,
        safety=safety,
        realism=realism,
        total=float(total_augmented),
        novelty=novelty,
        consensus=consensus,
        total_env=float(total_env),
        total_augmented=float(total_augmented),
    )
