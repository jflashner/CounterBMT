"""NNX Fourier embedding matching legacy Adv-BMT intent.

This is a JAX/NNX equivalent of the legacy PyTorch FourierEmbedding module used
for relation and motion feature conditioning.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    from flax import nnx

    HAS_NNX = True
except Exception:  # pragma: no cover
    nnx = None
    HAS_NNX = False


if HAS_NNX:

    class _Dense(nnx.Module):
        def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs, scale: float = 0.02):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (d_in, d_out)) * scale)
            self.b = nnx.Param(jnp.zeros((d_out,), dtype=jnp.float32))

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("...d,df->...f", x, self.w.value) + self.b.value


    class _LayerNorm(nnx.Module):
        def __init__(self, dim: int, *, eps: float = 1e-6):
            self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))
            self.bias = nnx.Param(jnp.zeros((dim,), dtype=jnp.float32))
            self.eps = eps

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            x_hat = (x - mean) / jnp.sqrt(var + self.eps)
            return x_hat * self.scale.value + self.bias.value


    class _PerDimMLP(nnx.Module):
        def __init__(self, in_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
            self.fc1 = _Dense(in_dim, hidden_dim, rngs=rngs)
            self.norm = _LayerNorm(hidden_dim)
            self.fc2 = _Dense(hidden_dim, hidden_dim, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            x = self.fc1(x)
            x = self.norm(x)
            x = jax.nn.relu(x)
            x = self.fc2(x)
            return x


    class FourierEmbeddingNNX(nnx.Module):
        """Learnable Fourier embedding compatible with legacy module semantics."""

        def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int, *, rngs: nnx.Rngs):
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_freq_bands = int(num_freq_bands)

            if self.input_dim > 0:
                self.freqs = nnx.Param(jax.random.normal(rngs.params(), (self.input_dim, self.num_freq_bands)) * 0.02)
                in_dim = self.num_freq_bands * 2 + 1
                self.mlps = tuple(_PerDimMLP(in_dim, self.hidden_dim, rngs=rngs) for _ in range(self.input_dim))
            else:
                self.freqs = None
                self.mlps = tuple()

            self.out_norm = _LayerNorm(self.hidden_dim)
            self.out_proj = _Dense(self.hidden_dim, self.hidden_dim, rngs=rngs)

        def __call__(
            self,
            continuous_inputs: Optional[jnp.ndarray] = None,
            categorical_embs: Optional[Iterable[jnp.ndarray]] = None,
        ) -> jnp.ndarray:
            if continuous_inputs is None:
                if categorical_embs is None:
                    raise ValueError("Both continuous_inputs and categorical_embs are None")
                stacked = [jnp.asarray(v) for v in categorical_embs]
                if not stacked:
                    raise ValueError("categorical_embs is empty")
                x = jnp.sum(jnp.stack(stacked, axis=0), axis=0)
            else:
                continuous_inputs = jnp.asarray(continuous_inputs, dtype=jnp.float32)
                if continuous_inputs.shape[-1] != self.input_dim:
                    raise ValueError(
                        f"continuous_inputs last dim mismatch: expected {self.input_dim}, got {continuous_inputs.shape[-1]}"
                    )

                # [..., Din, F]
                scaled = continuous_inputs[..., :, None] * self.freqs.value * (2.0 * math.pi)
                feat = jnp.concatenate([jnp.cos(scaled), jnp.sin(scaled), continuous_inputs[..., :, None]], axis=-1)

                per_dim = []
                for i in range(self.input_dim):
                    per_dim.append(self.mlps[i](feat[..., i, :]))
                x = jnp.sum(jnp.stack(per_dim, axis=0), axis=0)

                if categorical_embs is not None:
                    stacked = [jnp.asarray(v) for v in categorical_embs]
                    if stacked:
                        x = x + jnp.sum(jnp.stack(stacked, axis=0), axis=0)

            x = self.out_norm(x)
            x = jax.nn.relu(x)
            x = self.out_proj(x)
            return x


else:  # HAS_NNX == False

    class FourierEmbeddingNNX:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            raise RuntimeError("flax.nnx + jax are required for FourierEmbeddingNNX")
