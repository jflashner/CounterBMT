"""Lightweight GRPO-style statistics/update helpers.

This module intentionally keeps the optimizer side minimal for the current
codebase stage where policy/value heads are still under active refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def compute_group_advantages(rewards: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / (std + eps)).astype(np.float32)


@dataclass
class GRPOState:
    step: int = 0
    ema_reward: float = 0.0


@dataclass
class GRPOTrainer:
    """Statistics-only GRPO update scaffold.

    Once a trainable policy head is wired in, this class can be extended to
    optimize the clipped objective and KL/entropy terms directly.
    """

    reward_ema_decay: float = 0.99
    state: GRPOState = field(default_factory=GRPOState)

    def update(
        self,
        *,
        rewards: np.ndarray,
        advantages: np.ndarray,
        entropy: float,
        alpha: float,
        eta: Optional[float] = None,
    ) -> Dict[str, float]:
        r = np.asarray(rewards, dtype=np.float32).reshape(-1)
        a = np.asarray(advantages, dtype=np.float32).reshape(-1)
        if r.size == 0:
            return {
                "loss": 0.0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "adv_mean": 0.0,
                "adv_std": 0.0,
                "entropy": float(entropy),
                "alpha": float(alpha),
                "eta": float(0.0 if eta is None else eta),
                "step": float(self.state.step),
            }

        surrogate = float(np.mean(a * r))
        entropy_bonus = float(alpha) * float(entropy)
        # Sign convention: minimize loss.
        loss = float(-(surrogate + entropy_bonus))

        r_mean = float(np.mean(r))
        self.state.ema_reward = (
            self.reward_ema_decay * self.state.ema_reward + (1.0 - self.reward_ema_decay) * r_mean
        )
        self.state.step += 1

        return {
            "loss": loss,
            "surrogate": surrogate,
            "entropy_bonus": entropy_bonus,
            "reward_mean": r_mean,
            "reward_std": float(np.std(r)),
            "reward_ema": float(self.state.ema_reward),
            "adv_mean": float(np.mean(a)),
            "adv_std": float(np.std(a)),
            "entropy": float(entropy),
            "alpha": float(alpha),
            "eta": float(0.0 if eta is None else eta),
            "step": float(self.state.step),
        }
