"""GRPO/PPO-style update helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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


def clipped_surrogate_stats(
    *,
    old_logprob: np.ndarray,
    new_logprob: np.ndarray,
    advantages: np.ndarray,
    clip_eps: float,
) -> Dict[str, np.ndarray | float]:
    old_lp = np.asarray(old_logprob, dtype=np.float32).reshape(-1)
    new_lp = np.asarray(new_logprob, dtype=np.float32).reshape(-1)
    adv = np.asarray(advantages, dtype=np.float32).reshape(-1)
    ratio = np.exp(new_lp - old_lp).astype(np.float32)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)).astype(np.float32) * adv
    surrogate_terms = np.minimum(unclipped, clipped).astype(np.float32)
    return {
        "ratio": ratio,
        "surrogate_terms": surrogate_terms,
        "surrogate": float(np.mean(surrogate_terms)) if surrogate_terms.size else 0.0,
        "clip_fraction": float(np.mean((np.abs(ratio - 1.0) > float(clip_eps)).astype(np.float32))) if ratio.size else 0.0,
    }


def categorical_kl_from_log_probs(log_probs: np.ndarray, ref_log_probs: np.ndarray) -> float:
    lp = np.asarray(log_probs, dtype=np.float32)
    ref_lp = np.asarray(ref_log_probs, dtype=np.float32)
    probs = np.exp(lp)
    return float(np.mean(np.sum(probs * (lp - ref_lp), axis=-1)))


@dataclass
class GRPOState:
    step: int = 0
    ema_reward: float = 0.0


@dataclass
class GRPOTrainer:
    reward_ema_decay: float = 0.99
    policy_backend: Any = None
    state: GRPOState = field(default_factory=GRPOState)

    def update(
        self,
        *,
        rewards: np.ndarray,
        advantages: np.ndarray,
        entropy: float,
        alpha: float,
        eta: Optional[float] = None,
        policy_batch: Any = None,
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

        out = {
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
        if self.policy_backend is not None and policy_batch is not None and r.size > 0:
            out.update(
                self.policy_backend.update(
                    batch=policy_batch,
                    advantages=np.asarray(advantages, dtype=np.float32),
                    alpha=float(alpha),
                )
            )
        return out
