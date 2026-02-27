"""Entropy thermostat for novelty/consensus weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from counter_bmt_v2.config import RLTrainConfig


def cluster_entropy(cluster_ids: np.ndarray) -> float:
    ids = np.asarray(cluster_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return 0.0
    _, counts = np.unique(ids, return_counts=True)
    p = counts.astype(np.float64) / float(ids.size)
    h = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)))
    return float(h)


@dataclass
class EntropyThermostat:
    """Adaptive η/α updates from group entropy."""

    entropy_target: float
    eta0: float
    alpha0: float
    k_eta: float
    k_alpha: float
    eta_min: float = 0.0
    alpha_min: float = 0.0
    eta_max: float = 4.0
    alpha_max: float = 4.0

    @classmethod
    def from_config(cls, cfg: RLTrainConfig) -> "EntropyThermostat":
        return cls(
            entropy_target=float(cfg.entropy_target),
            eta0=float(cfg.eta0),
            alpha0=float(cfg.alpha0),
            k_eta=float(cfg.k_eta),
            k_alpha=float(cfg.k_alpha),
        )

    def compute(self, cluster_ids: np.ndarray) -> Tuple[float, float, float]:
        h = cluster_entropy(cluster_ids)
        eta = self.eta0 + self.k_eta * (self.entropy_target - h)
        alpha = self.alpha0 + self.k_alpha * (h - self.entropy_target)
        eta = float(np.clip(eta, self.eta_min, self.eta_max))
        alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
        return eta, alpha, h

