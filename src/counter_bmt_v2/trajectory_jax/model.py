"""JAX-first trajectory generator with simple autoregressive dynamics.

This is intentionally lightweight for rapid iteration.
It is designed to evolve into a fuller NNX implementation and eventually
into a unified LLM+trajectory backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from counter_bmt_v2.config import TrajectoryModelConfig
from counter_bmt_v2.contracts import ConditioningSignal, ScenarioInput, TrajectoryRollout


class TrajectoryGenerator(Protocol):
    def generate(
        self,
        scene: ScenarioInput,
        conditioning: ConditioningSignal,
        *,
        n_samples: int = 1,
        seed: int = 0,
    ) -> List[TrajectoryRollout]:
        """Generate one or more trajectory samples conditioned on intervention signal."""


@dataclass
class JaxTrajectoryGenerator(TrajectoryGenerator):
    config: TrajectoryModelConfig = field(default_factory=TrajectoryModelConfig)

    def __post_init__(self) -> None:
        self._jax = None
        self._jnp = None
        self._params: Optional[Dict[str, Any]] = None

        try:
            import jax
            import jax.numpy as jnp

            self._jax = jax
            self._jnp = jnp
        except Exception:
            # Keep a NumPy fallback so vertical slice always runs.
            self._jax = None
            self._jnp = None

    def _init_params(self, signal_dim: int, seed: int = 0) -> None:
        input_dim = signal_dim + 3  # prev_x, prev_y, t_norm + conditioning signal
        hidden = self.config.hidden_dim

        if self._jax is not None:
            key = self._jax.random.PRNGKey(seed)
            k1, k2 = self._jax.random.split(key)
            self._params = {
                "w1": self._jax.random.normal(k1, (input_dim, hidden)) * 0.2,
                "b1": self._jnp.zeros((hidden,)),
                "w2": self._jax.random.normal(k2, (hidden, 2)) * 0.1,
                "b2": self._jnp.zeros((2,)),
            }
        else:
            rng = np.random.default_rng(seed)
            self._params = {
                "w1": rng.normal(0.0, 0.2, size=(input_dim, hidden)).astype(np.float32),
                "b1": np.zeros((hidden,), dtype=np.float32),
                "w2": rng.normal(0.0, 0.1, size=(hidden, 2)).astype(np.float32),
                "b2": np.zeros((2,), dtype=np.float32),
            }

    def _step_np(self, prev_xy: np.ndarray, cond: np.ndarray, t_norm: float) -> np.ndarray:
        p = self._params
        x = np.concatenate([prev_xy, np.array([t_norm], dtype=np.float32), cond], axis=0)
        h = np.tanh(x @ p["w1"] + p["b1"])
        delta = np.tanh(h @ p["w2"] + p["b2"]) * 0.4
        return prev_xy + delta

    def _step_jax(self, prev_xy: Any, cond: Any, t_norm: float) -> Any:
        p = self._params
        x = self._jnp.concatenate([prev_xy, self._jnp.array([t_norm]), cond], axis=0)
        h = self._jnp.tanh(x @ p["w1"] + p["b1"])
        delta = self._jnp.tanh(h @ p["w2"] + p["b2"]) * 0.4
        return prev_xy + delta

    def generate(
        self,
        scene: ScenarioInput,
        conditioning: ConditioningSignal,
        *,
        n_samples: int = 1,
        seed: int = 0,
    ) -> List[TrajectoryRollout]:
        cond = np.asarray(conditioning.vector, dtype=np.float32)

        if self._params is None:
            self._init_params(signal_dim=len(cond), seed=seed)

        if scene.ego_trajectory_xy is not None and len(scene.ego_trajectory_xy) > 0:
            start_xy = np.asarray(scene.ego_trajectory_xy[0], dtype=np.float32)
        else:
            start_xy = np.zeros((2,), dtype=np.float32)

        results: List[TrajectoryRollout] = []
        for sample_idx in range(n_samples):
            rng = np.random.default_rng(seed + sample_idx)
            traj = np.zeros((self.config.horizon_steps, 2), dtype=np.float32)
            prev = start_xy.copy()

            for t in range(self.config.horizon_steps):
                t_norm = float(t) / float(max(1, self.config.horizon_steps - 1))
                if self._jax is not None:
                    prev = np.asarray(
                        self._step_jax(
                            self._jnp.asarray(prev),
                            self._jnp.asarray(cond),
                            t_norm,
                        )
                    )
                else:
                    prev = self._step_np(prev, cond, t_norm)

                # Small stochasticity for multi-sample diversity.
                prev = prev + rng.normal(0.0, 0.02, size=(2,)).astype(np.float32)
                traj[t] = prev

            results.append(
                TrajectoryRollout(
                    trajectory_xy=traj,
                    conditioning=conditioning,
                    sample_index=sample_idx,
                    metadata={
                        "backend": "jax" if self._jax is not None else "numpy_fallback",
                        "horizon_steps": self.config.horizon_steps,
                    },
                )
            )

        return results
