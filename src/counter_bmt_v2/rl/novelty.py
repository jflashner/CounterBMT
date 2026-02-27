"""Novelty scoring in behavior-manifold space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from counter_bmt_v2.config import NoveltyConfig


class NoveltyEstimator(Protocol):
    def score_batch(self, embeddings: np.ndarray, *, update: bool = True) -> np.ndarray:
        """Return per-embedding surprisal-like novelty scores."""


@dataclass
class EMAGaussianNovelty:
    """EMA Gaussian density estimator for online novelty scoring."""

    dim: int
    ema_decay: float = 0.99
    eps: float = 1e-6
    mean: np.ndarray = field(init=False)
    var: np.ndarray = field(init=False)
    initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.mean = np.zeros((int(self.dim),), dtype=np.float32)
        self.var = np.ones((int(self.dim),), dtype=np.float32)

    def _update_stats(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        if not self.initialized:
            self.mean = batch_mean.astype(np.float32)
            self.var = np.maximum(batch_var, self.eps).astype(np.float32)
            self.initialized = True
            return

        d = float(self.ema_decay)
        self.mean = (d * self.mean + (1.0 - d) * batch_mean).astype(np.float32)
        self.var = (d * self.var + (1.0 - d) * np.maximum(batch_var, self.eps)).astype(np.float32)

    def score_batch(self, embeddings: np.ndarray, *, update: bool = True) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        if update:
            self._update_stats(x)
        mu = self.mean[None, :]
        var = np.maximum(self.var[None, :], self.eps)

        # -log p(x) for diagonal Gaussian (up to additive constant).
        z = ((x - mu) ** 2) / var
        nll = 0.5 * np.sum(z + np.log(var), axis=1)
        d = float(x.shape[1])
        nll = nll / max(1.0, d)
        return nll.astype(np.float32)


@dataclass
class KNNNovelty:
    """Memory-bank KNN surprisal proxy."""

    dim: int
    k: int = 8
    max_bank: int = 20000
    _bank: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._bank = np.zeros((0, int(self.dim)), dtype=np.float32)

    def score_batch(self, embeddings: np.ndarray, *, update: bool = True) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if x.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        if self._bank.shape[0] == 0:
            out = np.full((x.shape[0],), 1.0, dtype=np.float32)
        else:
            dist = np.linalg.norm(x[:, None, :] - self._bank[None, :, :], axis=-1)
            k = min(int(self.k), self._bank.shape[0])
            nn = np.partition(dist, kth=k - 1, axis=1)[:, :k]
            out = np.mean(nn, axis=1).astype(np.float32)

        if update:
            self._bank = np.concatenate([self._bank, x], axis=0)
            if self._bank.shape[0] > int(self.max_bank):
                self._bank = self._bank[-int(self.max_bank) :]
        return out


def build_novelty_estimator(cfg: NoveltyConfig, *, dim: int) -> NoveltyEstimator:
    mode = str(cfg.density).lower()
    if mode == "knn":
        return KNNNovelty(dim=dim)
    return EMAGaussianNovelty(dim=dim, ema_decay=float(cfg.ema_decay))

