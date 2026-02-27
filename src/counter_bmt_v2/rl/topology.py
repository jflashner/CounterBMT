"""Topology-oriented behavior embedding utilities.

This module provides a practical interface for temporal topology embeddings in
the RL loop. The current implementation is dependency-light:
- `zigzag` backend: optional external backend hook (if available later).
- `ph_fallback` backend: deterministic handcrafted topological summary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Protocol, Tuple

import numpy as np

from counter_bmt_v2.contracts import TrajectoryRollout


def _stable_project(vec: np.ndarray, out_dim: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(max(1, vec.size)), size=(vec.size, out_dim)).astype(np.float32)
    out = np.asarray(vec, dtype=np.float32) @ w
    n = float(np.linalg.norm(out))
    if n > 0.0:
        out = out / n
    return out.astype(np.float32)


@dataclass
class BehaviorImageBuilder:
    """Construct a simple time-image tensor from a rollout trajectory.

    Output shape: [T, H, W, C], with channels:
    - occupancy (binary)
    - speed proxy
    - curvature proxy
    """

    grid_size: int = 32

    def build(self, rollout: TrajectoryRollout) -> np.ndarray:
        traj = np.asarray(rollout.trajectory_xy, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            return np.zeros((1, self.grid_size, self.grid_size, 3), dtype=np.float32)

        t_steps = traj.shape[0]
        lo = np.min(traj, axis=0)
        hi = np.max(traj, axis=0)
        span = np.maximum(hi - lo, 1e-4)
        norm_xy = (traj - lo[None, :]) / span[None, :]
        norm_xy = np.clip(norm_xy, 0.0, 1.0)
        pix = np.clip((norm_xy * float(self.grid_size - 1)).astype(np.int32), 0, self.grid_size - 1)

        vel = np.diff(traj, axis=0, prepend=traj[:1])
        speed = np.linalg.norm(vel, axis=1)
        speed = speed / (float(np.max(speed)) + 1e-6)
        acc = np.diff(vel, axis=0, prepend=vel[:1])
        curvature = np.linalg.norm(acc, axis=1)
        curvature = curvature / (float(np.max(curvature)) + 1e-6)

        img = np.zeros((t_steps, self.grid_size, self.grid_size, 3), dtype=np.float32)
        for t in range(t_steps):
            y, x = int(pix[t, 1]), int(pix[t, 0])
            img[t, y, x, 0] = 1.0
            img[t, y, x, 1] = float(speed[t])
            img[t, y, x, 2] = float(curvature[t])
        return img


class TopologyEncoder(Protocol):
    def encode(self, image_seq: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        """Encode time-image sequence into a fixed-size topology embedding."""


@dataclass
class PHPersistenceFallbackEncoder:
    """Dependency-light proxy for temporal topology descriptors.

    We compute stable summaries of occupancy dynamics and project them to a
    fixed embedding size. This is not a full PH implementation; it is a robust
    fallback that preserves the same API as future zigzag/PH backends.
    """

    out_dim: int = 32

    def encode(self, image_seq: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        x = np.asarray(image_seq, dtype=np.float32)
        if x.ndim != 4:
            x = np.zeros((1, 8, 8, 3), dtype=np.float32)

        occ = x[..., 0] > 0.5
        occ_count_t = np.sum(occ, axis=(1, 2)).astype(np.float32)
        # First/second temporal differences mimic birth/death style events.
        d1 = np.diff(occ_count_t, prepend=occ_count_t[:1])
        d2 = np.diff(d1, prepend=d1[:1])
        speed_ch = np.mean(x[..., 1], axis=(1, 2))
        curv_ch = np.mean(x[..., 2], axis=(1, 2))

        raw = np.asarray(
            [
                float(np.mean(occ_count_t)),
                float(np.std(occ_count_t)),
                float(np.mean(np.abs(d1))),
                float(np.mean(np.abs(d2))),
                float(np.max(occ_count_t)),
                float(np.min(occ_count_t)),
                float(np.mean(speed_ch)),
                float(np.std(speed_ch)),
                float(np.mean(curv_ch)),
                float(np.std(curv_ch)),
            ],
            dtype=np.float32,
        )
        emb = _stable_project(raw, out_dim=self.out_dim, seed=73)
        return emb, {"backend": "ph_fallback"}


@dataclass
class ZigzagTopologyEncoder:
    """Optional zigzag backend wrapper with graceful fallback.

    If a true zigzag backend is unavailable, we return the PH fallback vector.
    """

    out_dim: int = 32
    fallback: PHPersistenceFallbackEncoder = field(
        default_factory=lambda: PHPersistenceFallbackEncoder(out_dim=32)
    )

    def __post_init__(self) -> None:
        # Keep hook for future real backend integration.
        self._backend_available = False

    def encode(self, image_seq: np.ndarray) -> Tuple[np.ndarray, Dict[str, str]]:
        if self._backend_available:
            # Reserved for future real zigzag implementation.
            pass
        emb, meta = self.fallback.encode(image_seq)
        meta = dict(meta)
        meta["backend"] = "zigzag_unavailable_fallback"
        return emb, meta


@dataclass
class TopologyEmbeddingRunner:
    """Cache-aware topology embedding wrapper for RL loops."""

    out_dim: int = 32
    cache_dir: str = "outputs/topology_cache"
    prefer_zigzag: bool = True
    image_builder: BehaviorImageBuilder = field(default_factory=BehaviorImageBuilder)

    def __post_init__(self) -> None:
        self._cache = Path(self.cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        self._ph = PHPersistenceFallbackEncoder(out_dim=self.out_dim)
        self._zz = ZigzagTopologyEncoder(out_dim=self.out_dim, fallback=self._ph)

    def _payload_hash(self, rollout: TrajectoryRollout) -> str:
        arr = np.asarray(rollout.trajectory_xy, dtype=np.float32)
        payload = arr.tobytes() + str(arr.shape).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def _cache_paths(self, scenario_id: str, rollout_id: str) -> Tuple[Path, Path]:
        base = self._cache / str(scenario_id) / str(rollout_id)
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def encode(
        self,
        *,
        scenario_id: str,
        rollout_id: str,
        rollout: TrajectoryRollout,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, str]]:
        npz_path, json_path = self._cache_paths(scenario_id, rollout_id)
        sig = self._payload_hash(rollout)

        if use_cache and npz_path.is_file() and json_path.is_file():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8"))
                if meta.get("payload_hash") == sig:
                    arr = np.load(npz_path, allow_pickle=False)["embedding"].astype(np.float32)
                    return arr, {"backend": str(meta.get("backend", "cache")), "cache_hit": "1"}
            except Exception:
                pass

        image_seq = self.image_builder.build(rollout)
        if self.prefer_zigzag:
            emb, meta = self._zz.encode(image_seq)
        else:
            emb, meta = self._ph.encode(image_seq)

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, embedding=np.asarray(emb, dtype=np.float32))
        json_path.write_text(
            json.dumps(
                {
                    "scenario_id": scenario_id,
                    "rollout_id": rollout_id,
                    "payload_hash": sig,
                    "backend": meta.get("backend", "unknown"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return np.asarray(emb, dtype=np.float32), meta
