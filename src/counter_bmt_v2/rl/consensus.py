"""Cluster/consensus utilities for behavior-manifold rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from counter_bmt_v2.config import ConsensusConfig


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _kmeans_assign(x: np.ndarray, k: int, seed: int = 0, iters: int = 20) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if k <= 1 or n == 1:
        return np.zeros((n,), dtype=np.int32)
    k = min(int(k), n)

    rng = np.random.default_rng(seed)
    centers = x[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros((n,), dtype=np.int32)

    for _ in range(int(iters)):
        dist = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        labels = np.argmin(dist, axis=1).astype(np.int32)
        for j in range(k):
            idx = np.where(labels == j)[0]
            if idx.size == 0:
                centers[j] = x[int(rng.integers(0, n))]
            else:
                centers[j] = np.mean(x[idx], axis=0)
    return labels


@dataclass
class ConsensusScorer:
    cfg: ConsensusConfig

    def _cluster(self, psi: np.ndarray, *, seed: int) -> np.ndarray:
        mode = str(self.cfg.clusterer).lower()
        k = max(1, int(self.cfg.k_clusters))

        if mode == "hdbscan":
            try:
                import hdbscan  # type: ignore

                # Min cluster size picks a conservative fraction of group size.
                min_cluster_size = max(2, min(8, psi.shape[0] // 2))
                labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(psi)
                # Map noise to its own bucket for reward bookkeeping.
                if np.any(labels < 0):
                    labels = labels.copy()
                    labels[labels < 0] = labels.max(initial=-1) + 1
                return labels.astype(np.int32)
            except Exception:
                pass

        return _kmeans_assign(psi, k=k, seed=seed)

    @staticmethod
    def _quality_scores(risk_features: Sequence[Dict[str, float]]) -> np.ndarray:
        if not risk_features:
            return np.zeros((0,), dtype=np.float32)
        progress = np.asarray([float(r.get("progress_delta", 0.0)) for r in risk_features], dtype=np.float32)
        collision_risk = np.asarray(
            [float(r.get("collision_risk_proxy", 0.0)) for r in risk_features], dtype=np.float32
        )
        violations = np.asarray([float(r.get("rule_violation_proxy", 0.0)) for r in risk_features], dtype=np.float32)

        # Standardize progress within the group before quality fusion.
        p_mu = float(np.mean(progress))
        p_std = float(np.std(progress) + 1e-6)
        p_norm = (progress - p_mu) / p_std

        # Q(C) proxy at rollout level before cluster pooling.
        q_i = _sigmoid(1.2 * p_norm - 1.0 * collision_risk - 0.8 * violations)
        return np.asarray(q_i, dtype=np.float32)

    def score(
        self,
        psi: np.ndarray,
        risk_features: Sequence[Dict[str, float]],
        *,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], np.ndarray]:
        """Return (cluster_ids, consensus_scores, cluster_hist, quality_per_rollout)."""
        x = np.asarray(psi, dtype=np.float32)
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        n = x.shape[0]
        if n == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                {},
                np.zeros((0,), dtype=np.float32),
            )

        cluster_ids = self._cluster(x, seed=seed)
        q_i = self._quality_scores(risk_features)
        if q_i.shape[0] != n:
            q_i = np.zeros((n,), dtype=np.float32)

        unique, counts = np.unique(cluster_ids, return_counts=True)
        hist = {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        rho = {int(k): float(v) / float(n) for k, v in zip(unique.tolist(), counts.tolist())}

        q_cluster: Dict[int, float] = {}
        for k in unique.tolist():
            idx = np.where(cluster_ids == int(k))[0]
            q_cluster[int(k)] = float(np.mean(q_i[idx])) if idx.size else 0.0

        consensus = np.asarray(
            [float(rho.get(int(cid), 0.0) * q_cluster.get(int(cid), 0.0)) for cid in cluster_ids],
            dtype=np.float32,
        )
        return cluster_ids.astype(np.int32), consensus, hist, q_i

