"""Minimal RL loop building blocks for grouped rollouts (GRPO-style collection)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from counter_bmt_v2.contracts import PipelineResult, RewardBreakdown


@dataclass
class GroupedRolloutBatch:
    results: List[PipelineResult]

    @property
    def rewards(self) -> np.ndarray:
        vals = [r.total for result in self.results for r in result.rewards]
        return np.asarray(vals, dtype=np.float32)


@dataclass
class GRPOTrainerStub:
    """Placeholder trainer API to preserve clean RL module boundaries."""

    def update(self, batch: GroupedRolloutBatch) -> dict:
        rewards = batch.rewards
        if rewards.size == 0:
            return {"loss": 0.0, "reward_mean": 0.0, "reward_std": 0.0}
        return {
            "loss": float(-np.mean(rewards)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "note": "stub update - replace with real GRPO optimizer",
        }


def summarize_reward_breakdown(rewards: List[RewardBreakdown]) -> dict:
    if not rewards:
        return {"n": 0}
    arr_total = np.asarray([r.total for r in rewards], dtype=np.float32)
    arr_align = np.asarray([r.alignment for r in rewards], dtype=np.float32)
    return {
        "n": int(arr_total.size),
        "total_mean": float(np.mean(arr_total)),
        "alignment_mean": float(np.mean(arr_align)),
    }
