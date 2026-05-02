from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32)
    mask_f = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    pooled = (values * mask_f).sum(dim=1) / denom
    valid = mask_f.sum(dim=1) > 0.0
    return torch.where(valid, pooled, torch.zeros_like(pooled))


class RolloutBehaviorEncoder(nn.Module):
    """Small temporal encoder for detached SDC rollout features."""

    def __init__(self, *, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(features.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = F.silu(self.input_proj(x))
        x, _ = self.gru(x)
        pooled = _masked_mean(x, valid_mask)
        embedding = self.output_proj(pooled)
        embedding = self.output_norm(embedding)
        return F.normalize(embedding, dim=-1)


class CausalDAGContextEncoder(nn.Module):
    """Encodes the fixed local-intervention DAG using the existing control tokens."""

    def __init__(self, *, path_dim: int, compliance_dim: int, timing_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.path_proj = nn.Linear(path_dim, hidden_dim)
        self.signal_proj = nn.Linear(compliance_dim, hidden_dim)
        self.compliance_proj = nn.Linear(compliance_dim, hidden_dim)
        self.conflict_proj = nn.Linear(timing_dim, hidden_dim)
        self.timing_proj = nn.Linear(timing_dim, hidden_dim)
        self.node_type_embed = nn.Embedding(5, hidden_dim)
        self.self_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.self_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        adjacency = torch.zeros((5, 5), dtype=torch.float32)
        adjacency[1, 4] = 1.0  # conflict_eta -> entry_timing
        adjacency[2, 3] = 1.0  # path_choice -> compliance context coupling
        adjacency[0, 3] = 1.0  # signal_state -> compliance
        adjacency[3, 4] = 1.0  # compliance -> timing
        adjacency = adjacency + adjacency.T
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj_norm", adjacency / degree)
        self.register_buffer("node_type_ids", torch.arange(5, dtype=torch.long))

    def _message_pass(self, node_state: torch.Tensor, *, self_fc: nn.Linear, neighbor_fc: nn.Linear) -> torch.Tensor:
        neighbor = torch.einsum("ij,bjh->bih", self.adj_norm, node_state)
        updated = self_fc(node_state) + neighbor_fc(neighbor)
        return F.silu(updated + node_state)

    def forward(
        self,
        *,
        path_token: torch.Tensor,
        compliance_token: torch.Tensor,
        timing_token: torch.Tensor,
    ) -> torch.Tensor:
        dtype = torch.float32
        device = path_token.device
        path_token = torch.nan_to_num(path_token.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        compliance_token = torch.nan_to_num(compliance_token.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)
        timing_token = torch.nan_to_num(timing_token.to(dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)

        signal_node = compliance_token
        compliance_node = compliance_token
        conflict_node = timing_token
        timing_node = timing_token
        path_node = path_token
        node_state = torch.stack(
            [
                self.signal_proj(signal_node),
                self.conflict_proj(conflict_node),
                self.path_proj(path_node),
                self.compliance_proj(compliance_node),
                self.timing_proj(timing_node),
            ],
            dim=1,
        )
        node_state = node_state + self.node_type_embed(self.node_type_ids.to(device=device))[None, :, :]
        node_state = self._message_pass(node_state, self_fc=self.self_fc1, neighbor_fc=self.neighbor_fc1)
        node_state = self._message_pass(node_state, self_fc=self.self_fc2, neighbor_fc=self.neighbor_fc2)
        pooled = node_state.mean(dim=1)
        embedding = self.out_proj(pooled)
        embedding = self.out_norm(embedding)
        return F.normalize(embedding, dim=-1)


@dataclass
class EMAGaussianState:
    mean: torch.Tensor
    var: torch.Tensor
    count: int


class EMAGaussianNoveltyBank:
    """Rank-local EMA Gaussian novelty estimator keyed by a discrete DAG bucket."""

    def __init__(self, *, decay: float = 0.995, min_var: float = 1e-3):
        self.decay = float(decay)
        self.min_var = float(min_var)
        self._states: Dict[Tuple[int, ...], EMAGaussianState] = {}

    def score(self, embeddings: torch.Tensor, bucket_keys: Sequence[Tuple[int, ...]], *, warmup_count: int) -> torch.Tensor:
        scores = embeddings.new_zeros((embeddings.shape[0],), dtype=torch.float32)
        if embeddings.numel() <= 0:
            return scores
        with torch.no_grad():
            cpu_embeddings = embeddings.detach().to(dtype=torch.float32, device="cpu")
            for idx, key in enumerate(bucket_keys):
                state = self._states.get(tuple(key))
                if state is None or int(state.count) < int(warmup_count):
                    continue
                diff = cpu_embeddings[idx] - state.mean
                var = state.var.clamp_min(self.min_var)
                log_det = torch.log(2.0 * math.pi * var).sum()
                maha = (diff.square() / var).sum()
                scores[idx] = 0.5 * (log_det + maha).to(device=embeddings.device)
        return scores

    def update(self, embeddings: torch.Tensor, bucket_keys: Sequence[Tuple[int, ...]]) -> None:
        if embeddings.numel() <= 0:
            return
        with torch.no_grad():
            cpu_embeddings = embeddings.detach().to(dtype=torch.float32, device="cpu")
            for idx, key in enumerate(bucket_keys):
                value = cpu_embeddings[idx]
                state = self._states.get(tuple(key))
                if state is None:
                    self._states[tuple(key)] = EMAGaussianState(
                        mean=value.clone(),
                        var=torch.ones_like(value),
                        count=1,
                    )
                    continue
                prev_mean = state.mean
                decay = float(self.decay)
                new_mean = decay * prev_mean + (1.0 - decay) * value
                centered = value - prev_mean
                new_var = decay * state.var + (1.0 - decay) * centered.square()
                self._states[tuple(key)] = EMAGaussianState(
                    mean=new_mean,
                    var=new_var.clamp_min(self.min_var),
                    count=int(state.count) + 1,
                )

    def state_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for key, state in self._states.items():
            payload[str(tuple(int(v) for v in key))] = {
                "mean": state.mean.clone(),
                "var": state.var.clone(),
                "count": int(state.count),
            }
        return {
            "decay": float(self.decay),
            "min_var": float(self.min_var),
            "states": payload,
        }

    def load_state_dict(self, state_dict: Dict[str, object] | None) -> None:
        self._states = {}
        if not isinstance(state_dict, dict):
            return
        self.decay = float(state_dict.get("decay", self.decay))
        self.min_var = float(state_dict.get("min_var", self.min_var))
        raw_states = state_dict.get("states", {})
        if not isinstance(raw_states, dict):
            return
        for raw_key, raw_value in raw_states.items():
            if not isinstance(raw_value, dict):
                continue
            try:
                key_tuple = tuple(int(part.strip()) for part in str(raw_key).strip("() ").split(",") if part.strip())
            except Exception:
                continue
            mean = raw_value.get("mean")
            var = raw_value.get("var")
            count = int(raw_value.get("count", 0) or 0)
            if not torch.is_tensor(mean) or not torch.is_tensor(var):
                continue
            self._states[key_tuple] = EMAGaussianState(
                mean=mean.detach().to(dtype=torch.float32, device="cpu").clone(),
                var=var.detach().to(dtype=torch.float32, device="cpu").clone().clamp_min(self.min_var),
                count=count,
            )
