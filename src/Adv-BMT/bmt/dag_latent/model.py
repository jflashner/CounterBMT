"""Additive DAG-latent wrapper for the legacy Adv-BMT MotionLM.

Design goal:
- do not modify legacy `MotionLM`, scene encoder, or decoder code
- add a narrow DAG latent control path through inheritance only

To stay minimally invasive, the DAG latent is injected as a global gated
residual on `encoder/scenario_token` just before the legacy decoder runs.
That keeps all existing tensor contracts intact while still letting the decoder
attend to DAG-conditioned scene context.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from bmt.models.motionlm import MotionLM

from .encoder import DAGLatentConfig, TorchDAGGraphEncoder


class MotionLMDAGLatent(MotionLM):
    """Legacy `MotionLM` plus an opt-in DAG latent control path."""

    _APPLIED_FLAG = "dag/conditioning_applied"

    def __init__(self, config: Any, dag_config: Optional[DAGLatentConfig] = None) -> None:
        super().__init__(config=config)
        self.dag_config = dag_config or DAGLatentConfig()
        self.d_model = int(self.config.MODEL.D_MODEL)

        if not bool(self.dag_config.enabled):
            self.dag_encoder = None
            self.dag_latent_in = 0
            self.dag_latent_proj = None
            self.dag_gate_proj = None
            self.null_dag_latent = None
            return

        if bool(self.dag_config.use_graph_encoder):
            self.dag_encoder = TorchDAGGraphEncoder(self.dag_config)
            self.dag_latent_in = 2 * int(self.dag_config.d_hidden)
        else:
            self.dag_encoder = None
            self.dag_latent_in = int(self.dag_config.latent_dim or self.d_model)

        self.dag_latent_proj = nn.Linear(self.dag_latent_in, self.d_model)
        self.dag_gate_proj = nn.Linear(self.dag_latent_in, self.d_model)
        self.null_dag_latent = nn.Parameter(
            torch.randn(self.dag_latent_in) * float(self.dag_config.null_latent_init_std)
        )

    @staticmethod
    def _lookup(batch: Dict[str, Any], *names: str) -> Any:
        for name in names:
            if name in batch:
                return batch[name]
        return None

    def _resolve_precomputed_latent(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        latent = self._lookup(batch, "dag/latent", "dag_latent")
        if latent is None:
            return None
        if not torch.is_tensor(latent):
            latent = torch.as_tensor(latent)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[-1] != self.dag_latent_in:
            raise ValueError(
                f"Precomputed DAG latent width mismatch: expected {self.dag_latent_in}, got {latent.shape[-1]}"
            )
        return latent

    def _resolve_graph_latent(self, batch: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        meta: Dict[str, torch.Tensor] = {}
        if self.dag_encoder is None:
            return None, meta

        dag_node_feat = self._lookup(batch, "dag/node_feat", "dag_node_feat")
        dag_node_mask = self._lookup(batch, "dag/node_mask", "dag_node_mask")
        dag_edge_src = self._lookup(batch, "dag/edge_src", "dag_edge_src")
        dag_edge_dst = self._lookup(batch, "dag/edge_dst", "dag_edge_dst")
        dag_edge_feat = self._lookup(batch, "dag/edge_feat", "dag_edge_feat")
        dag_edge_mask = self._lookup(batch, "dag/edge_mask", "dag_edge_mask")
        dag_global_feat = self._lookup(batch, "dag/global_feat", "dag_global_feat")

        has_graph = all(x is not None for x in (dag_node_feat, dag_node_mask, dag_edge_src, dag_edge_dst, dag_edge_feat, dag_edge_mask))
        if not has_graph:
            return None, meta

        if not torch.is_tensor(dag_node_feat):
            dag_node_feat = torch.as_tensor(dag_node_feat)
        if not torch.is_tensor(dag_node_mask):
            dag_node_mask = torch.as_tensor(dag_node_mask)
        if not torch.is_tensor(dag_edge_src):
            dag_edge_src = torch.as_tensor(dag_edge_src)
        if not torch.is_tensor(dag_edge_dst):
            dag_edge_dst = torch.as_tensor(dag_edge_dst)
        if not torch.is_tensor(dag_edge_feat):
            dag_edge_feat = torch.as_tensor(dag_edge_feat)
        if not torch.is_tensor(dag_edge_mask):
            dag_edge_mask = torch.as_tensor(dag_edge_mask)
        if dag_global_feat is not None and not torch.is_tensor(dag_global_feat):
            dag_global_feat = torch.as_tensor(dag_global_feat)

        _node_latent, z_dag = self.dag_encoder(
            dag_node_feat=dag_node_feat.float(),
            dag_node_mask=dag_node_mask.bool(),
            dag_edge_src=dag_edge_src.long(),
            dag_edge_dst=dag_edge_dst.long(),
            dag_edge_feat=dag_edge_feat.float(),
            dag_edge_mask=dag_edge_mask.bool(),
            dag_global_feat=(None if dag_global_feat is None else dag_global_feat.float()),
        )
        source_used = self._lookup(batch, "dag/source_used", "dag_source_used")
        if source_used is None:
            meta["source_used"] = dag_node_mask.bool().any(dim=1).float()
        else:
            if not torch.is_tensor(source_used):
                source_used = torch.as_tensor(source_used)
            meta["source_used"] = source_used.float()
        return z_dag, meta

    def resolve_dag_latent(self, batch: Dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        meta: Dict[str, torch.Tensor] = {}
        if not bool(self.dag_config.enabled):
            return None, meta

        z_dag = self._resolve_precomputed_latent(batch)
        if z_dag is None:
            z_dag, meta = self._resolve_graph_latent(batch)

        if z_dag is not None:
            z_dag = z_dag.to(device=device, dtype=dtype)
            if bool(self.dag_config.use_null_latent):
                present = meta.get("source_used")
                if present is not None:
                    z_null = self.null_dag_latent.to(device=device, dtype=dtype).unsqueeze(0).expand(z_dag.shape[0], -1)
                    z_dag = torch.where(present[:, None].bool(), z_dag, z_null)
            return z_dag, meta

        if bool(self.dag_config.use_null_latent):
            z_null = self.null_dag_latent.to(device=device, dtype=dtype).unsqueeze(0)
            scene_token = self._lookup(batch, "encoder/scenario_token")
            batch_size = int(scene_token.shape[0]) if torch.is_tensor(scene_token) else 1
            meta["source_used"] = torch.zeros((batch_size,), device=device, dtype=dtype)
            return z_null.expand(batch_size, -1), meta

        return None, meta

    def apply_dag_conditioning(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self.dag_config.enabled):
            return batch
        if bool(batch.get(self._APPLIED_FLAG, False)):
            return batch

        scene_token = self._lookup(batch, "encoder/scenario_token")
        if scene_token is None:
            raise KeyError("Legacy scene encoding must run before DAG conditioning; missing `encoder/scenario_token`.")
        if not torch.is_tensor(scene_token):
            raise TypeError("`encoder/scenario_token` must be a torch.Tensor before DAG conditioning.")

        z_dag, meta = self.resolve_dag_latent(batch, device=scene_token.device, dtype=scene_token.dtype)
        if z_dag is None:
            return batch

        present = meta.get("source_used")
        if present is not None:
            present = present.to(device=scene_token.device, dtype=scene_token.dtype)

        p_drop = float(self.dag_config.dag_dropout_prob)
        if self.training and p_drop >= 1.0:
            batch[self._APPLIED_FLAG] = True
            batch["dag/latent_norm"] = torch.zeros((scene_token.shape[0],), device=scene_token.device, dtype=scene_token.dtype)
            batch["dag/gate_mean"] = torch.zeros((scene_token.shape[0],), device=scene_token.device, dtype=scene_token.dtype)
            for key, value in meta.items():
                batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
            return batch
        if self.training and p_drop > 0.0:
            keep = (torch.rand((z_dag.shape[0], 1), device=z_dag.device) >= p_drop).to(z_dag.dtype)
            z_dag = z_dag * keep

        if str(self.dag_config.injection_mode) != "global_gated_residual":
            raise ValueError(
                f"Unsupported DAG injection mode {self.dag_config.injection_mode!r}; "
                "this minimal wrapper only implements `global_gated_residual`."
            )

        bias = self.dag_latent_proj(z_dag)
        gate = torch.sigmoid(self.dag_gate_proj(z_dag))
        dag_bias = gate * bias
        if present is not None:
            dag_bias = dag_bias * present[:, None]
            effective_z = z_dag * present[:, None]
            effective_gate = gate * present[:, None]
        else:
            effective_z = z_dag
            effective_gate = gate

        batch["encoder/scenario_token"] = scene_token + dag_bias[:, None, :]
        batch[self._APPLIED_FLAG] = True
        batch["dag/latent"] = z_dag
        batch["dag/latent_norm"] = torch.linalg.norm(effective_z, dim=-1)
        batch["dag/gate_mean"] = effective_gate.mean(dim=-1)
        for key, value in meta.items():
            batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
        return batch

    def decode_motion(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if args and isinstance(args[0], dict):
            batch = self.apply_dag_conditioning(args[0])
            args = (batch,) + args[1:]
        return super().decode_motion(*args, **kwargs)
