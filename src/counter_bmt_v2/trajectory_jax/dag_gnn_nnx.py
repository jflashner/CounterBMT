"""JAX/NNX DAG graph encoder for latent conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

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


@dataclass
class NNXDAGEncoderConfig:
    enabled: bool = False
    d_node_in: int = 24
    d_edge_in: int = 8
    d_hidden: int = 128
    n_layers: int = 3
    dropout: float = 0.0
    max_nodes: int = 64
    max_edges: int = 256


if HAS_NNX:

    class Linear(nnx.Module):
        def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs, scale: float = 0.02):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (d_in, d_out)) * scale)
            self.b = nnx.Param(jnp.zeros((d_out,), dtype=jnp.float32))

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("...d,df->...f", x, self.w.value) + self.b.value


    class RMSNorm(nnx.Module):
        def __init__(self, d_model: int, *, eps: float = 1e-6):
            self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
            self.eps = float(eps)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
            return (x / rms) * self.scale.value


    class MLP(nnx.Module):
        def __init__(self, d_in: int, d_hidden: int, d_out: int, *, rngs: nnx.Rngs):
            self.l1 = Linear(d_in, d_hidden, rngs=rngs)
            self.l2 = Linear(d_hidden, d_out, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return self.l2(jax.nn.gelu(self.l1(x)))


    class GraphMessageLayer(nnx.Module):
        def __init__(self, d_hidden: int, d_edge: int, *, rngs: nnx.Rngs):
            self.msg_mlp = MLP(2 * d_hidden + d_edge, d_hidden, d_hidden, rngs=rngs)
            self.upd_mlp = MLP(2 * d_hidden, d_hidden, d_hidden, rngs=rngs)
            self.norm = RMSNorm(d_hidden)

        @staticmethod
        def _segment_sum(
            messages: jnp.ndarray,
            dst: jnp.ndarray,
            valid_e: jnp.ndarray,
            n_nodes: int,
        ) -> jnp.ndarray:
            m = jnp.where(valid_e[:, None], messages, jnp.zeros_like(messages))
            return jax.ops.segment_sum(m, dst, num_segments=int(n_nodes))

        def __call__(
            self,
            h: jnp.ndarray,  # [G,D]
            node_mask: jnp.ndarray,  # [G]
            edge_src: jnp.ndarray,  # [E]
            edge_dst: jnp.ndarray,  # [E]
            edge_feat: jnp.ndarray,  # [E,Fe]
            edge_mask: jnp.ndarray,  # [E]
        ) -> jnp.ndarray:
            n_nodes = int(h.shape[0])
            src_h = h[edge_src]
            dst_h = h[edge_dst]
            msg_in = jnp.concatenate([src_h, dst_h, edge_feat], axis=-1)
            msg = self.msg_mlp(msg_in)  # [E,D]
            agg = self._segment_sum(msg, edge_dst, edge_mask.astype(bool), n_nodes=n_nodes)
            upd = self.upd_mlp(jnp.concatenate([h, agg], axis=-1))
            out = self.norm(h + upd)
            out = jnp.where(node_mask[:, None], out, jnp.zeros_like(out))
            return out


    class NNXDAGGraphEncoder(nnx.Module):
        """Small message-passing DAG encoder returning node and pooled latent."""

        def __init__(self, cfg: NNXDAGEncoderConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_h = int(cfg.d_hidden)
            self.node_in = Linear(int(cfg.d_node_in), d_h, rngs=rngs)
            self.edge_in = Linear(int(cfg.d_edge_in), d_h, rngs=rngs)
            self.layers = tuple(
                GraphMessageLayer(d_hidden=d_h, d_edge=d_h, rngs=rngs) for _ in range(max(1, int(cfg.n_layers)))
            )
            self.out_norm = RMSNorm(d_h)

        def _encode_one(
            self,
            node_feat: jnp.ndarray,  # [G,Fn]
            node_mask: jnp.ndarray,  # [G]
            edge_src: jnp.ndarray,  # [E]
            edge_dst: jnp.ndarray,  # [E]
            edge_feat: jnp.ndarray,  # [E,Fe]
            edge_mask: jnp.ndarray,  # [E]
            global_feat: Optional[jnp.ndarray],  # [Fg] or None
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            h = self.node_in(node_feat)
            e = self.edge_in(edge_feat)

            # Global feature support is intentionally light-touch for now:
            # add mean-projected bias if provided.
            if global_feat is not None and global_feat.size > 0:
                g = jnp.mean(global_feat).astype(h.dtype)
                h = h + g

            h = jnp.where(node_mask[:, None], h, jnp.zeros_like(h))
            for layer in self.layers:
                h = layer(
                    h=h,
                    node_mask=node_mask,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_feat=e,
                    edge_mask=edge_mask,
                )
            h = self.out_norm(h)
            h = jnp.where(node_mask[:, None], h, jnp.zeros_like(h))

            mask_f = node_mask.astype(jnp.float32)[:, None]
            denom = jnp.maximum(1.0, jnp.sum(mask_f, axis=0))
            mean_pool = jnp.sum(h * mask_f, axis=0) / denom
            max_pool = jnp.max(
                jnp.where(node_mask[:, None], h, jnp.full_like(h, -1e9)),
                axis=0,
            )
            max_pool = jnp.where(jnp.isfinite(max_pool), max_pool, jnp.zeros_like(max_pool))
            z = jnp.concatenate([mean_pool, max_pool], axis=-1)  # [2D]
            return h, z

        def __call__(
            self,
            *,
            dag_node_feat: jnp.ndarray,  # [B,G,Fn]
            dag_node_mask: jnp.ndarray,  # [B,G]
            dag_edge_src: jnp.ndarray,  # [B,E]
            dag_edge_dst: jnp.ndarray,  # [B,E]
            dag_edge_feat: jnp.ndarray,  # [B,E,Fe]
            dag_edge_mask: jnp.ndarray,  # [B,E]
            dag_global_feat: Optional[jnp.ndarray] = None,  # [B,Fg]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            bsz = int(dag_node_feat.shape[0])
            if dag_global_feat is None:
                dag_global_feat = jnp.zeros((bsz, 0), dtype=jnp.float32)

            encode = jax.vmap(self._encode_one, in_axes=(0, 0, 0, 0, 0, 0, 0))
            node_latent, z_cat = encode(
                dag_node_feat,
                dag_node_mask.astype(bool),
                dag_edge_src.astype(jnp.int32),
                dag_edge_dst.astype(jnp.int32),
                dag_edge_feat,
                dag_edge_mask.astype(bool),
                dag_global_feat,
            )
            return node_latent, z_cat


else:  # pragma: no cover
    Linear = None
    RMSNorm = None
    MLP = None
    GraphMessageLayer = None
    NNXDAGGraphEncoder = None

