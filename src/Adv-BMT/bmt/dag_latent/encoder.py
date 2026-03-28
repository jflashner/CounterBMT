"""Minimal Torch DAG encoder for legacy Adv-BMT latent control.

The implementation mirrors the spirit of the v2 DAG-latent path, but stays
small and dependency-light:
- project node/edge features
- run a few directed message-passing layers
- pool a single graph latent per batch item
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DAGLatentConfig:
    """Configuration for the additive DAG-latent wrapper.

    Defaults are chosen to align with the v2 DAG tensor contract without
    requiring any legacy config-file edits.
    """

    enabled: bool = True
    use_graph_encoder: bool = True

    d_node_in: int = 24
    d_edge_in: int = 8
    d_hidden: int = 128
    n_layers: int = 3
    dropout: float = 0.0

    max_nodes: int = 64
    max_edges: int = 256

    # If you want to bypass graph encoding and provide `dag_latent` directly,
    # set `use_graph_encoder=False` and optionally choose a custom latent width.
    latent_dim: Optional[int] = None

    injection_mode: str = "global_gated_residual"
    dag_dropout_prob: float = 0.0
    use_null_latent: bool = False
    null_latent_init_std: float = 0.02

    use_time_guidance: bool = False
    time_guidance_feature_dim: int = 18
    time_guidance_mode: str = "gated"
    time_guidance_use_global: bool = False
    time_guidance_init_gate_bias: float = -2.0

    use_maneuver_tokens: bool = False
    maneuver_token_feature_dim: int = 20
    maneuver_token_use_global: bool = False
    maneuver_token_init_gate_bias: float = -2.0


class TorchRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.scale


class TorchMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchGraphMessageLayer(nn.Module):
    def __init__(self, d_hidden: int, d_edge: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.msg_mlp = TorchMLP(2 * d_hidden + d_edge, d_hidden, d_hidden, dropout=dropout)
        self.upd_mlp = TorchMLP(2 * d_hidden, d_hidden, d_hidden, dropout=dropout)
        self.norm = TorchRMSNorm(d_hidden)

    def forward(
        self,
        h: torch.Tensor,
        *,
        node_mask: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = int(h.shape[0])
        valid_nodes = node_mask.bool()
        valid_edges = edge_mask.bool()

        agg = h.new_zeros((n_nodes, h.shape[-1]))
        if n_nodes > 0 and bool(valid_edges.any()):
            src = edge_src[valid_edges].long().clamp(min=0, max=max(0, n_nodes - 1))
            dst = edge_dst[valid_edges].long().clamp(min=0, max=max(0, n_nodes - 1))
            msg_in = torch.cat([h.index_select(0, src), h.index_select(0, dst), edge_feat[valid_edges]], dim=-1)
            msg = self.msg_mlp(msg_in)
            agg.index_add_(0, dst, msg)

        out = self.upd_mlp(torch.cat([h, agg], dim=-1))
        out = self.norm(h + out)
        out = torch.where(valid_nodes[:, None], out, torch.zeros_like(out))
        return out


class TorchDAGGraphEncoder(nn.Module):
    """Small directed message-passing encoder returning node and pooled latents."""

    def __init__(self, cfg: DAGLatentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_hidden = int(cfg.d_hidden)
        self.node_in = nn.Linear(int(cfg.d_node_in), d_hidden)
        self.edge_in = nn.Linear(int(cfg.d_edge_in), d_hidden)
        self.layers = nn.ModuleList(
            [TorchGraphMessageLayer(d_hidden=d_hidden, d_edge=d_hidden, dropout=float(cfg.dropout)) for _ in range(max(1, int(cfg.n_layers)))]
        )
        self.out_norm = TorchRMSNorm(d_hidden)

    def _encode_one(
        self,
        *,
        node_feat: torch.Tensor,
        node_mask: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_mask: torch.Tensor,
        global_feat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.node_in(node_feat)
        e = self.edge_in(edge_feat)

        valid_nodes = node_mask.bool()
        if global_feat is not None and global_feat.numel() > 0:
            h = h + global_feat.float().mean().to(h.dtype)
        h = torch.where(valid_nodes[:, None], h, torch.zeros_like(h))

        for layer in self.layers:
            h = layer(
                h,
                node_mask=valid_nodes,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_feat=e,
                edge_mask=edge_mask,
            )
        h = self.out_norm(h)
        h = torch.where(valid_nodes[:, None], h, torch.zeros_like(h))

        if bool(valid_nodes.any()):
            pooled_nodes = h[valid_nodes]
            mean_pool = pooled_nodes.mean(dim=0)
            max_pool = pooled_nodes.amax(dim=0)
        else:
            mean_pool = h.new_zeros((h.shape[-1],))
            max_pool = h.new_zeros((h.shape[-1],))
        z = torch.cat([mean_pool, max_pool], dim=-1)
        return h, z

    def forward(
        self,
        *,
        dag_node_feat: torch.Tensor,
        dag_node_mask: torch.Tensor,
        dag_edge_src: torch.Tensor,
        dag_edge_dst: torch.Tensor,
        dag_edge_feat: torch.Tensor,
        dag_edge_mask: torch.Tensor,
        dag_global_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dag_node_feat.ndim != 3:
            raise ValueError(f"dag_node_feat must be [B,G,F], got shape={tuple(dag_node_feat.shape)}")

        bsz = int(dag_node_feat.shape[0])
        if dag_global_feat is None:
            dag_global_feat = dag_node_feat.new_zeros((bsz, 0))

        node_latents = []
        pooled_latents = []
        for batch_idx in range(bsz):
            node_h, z = self._encode_one(
                node_feat=dag_node_feat[batch_idx],
                node_mask=dag_node_mask[batch_idx],
                edge_src=dag_edge_src[batch_idx],
                edge_dst=dag_edge_dst[batch_idx],
                edge_feat=dag_edge_feat[batch_idx],
                edge_mask=dag_edge_mask[batch_idx],
                global_feat=dag_global_feat[batch_idx],
            )
            node_latents.append(node_h)
            pooled_latents.append(z)
        return torch.stack(node_latents, dim=0), torch.stack(pooled_latents, dim=0)
