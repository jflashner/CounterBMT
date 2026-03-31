"""Additive DAG-latent wrapper for the legacy Adv-BMT MotionLM.

Design goal:
- do not modify legacy `MotionLM`, scene encoder, or decoder code
- add a narrow DAG latent control path through inheritance only

To stay minimally invasive, the DAG latent is injected as a global gated
residual on `encoder/scenario_token` just before the legacy decoder runs.
That keeps all existing tensor contracts intact while still letting the decoder
attend to DAG-conditioned scene context.

This wrapper also supports an optional timestep-aligned DAG guidance path:
- cache-backed maneuver intervals are projected into a per-step control tensor
- the decoder can add that tensor to motion tokens using `decoder/input_step`
- this gives the model access to "what maneuver is active when" rather than
  only a single pooled graph latent
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
            self.dag_time_in = 0
            self.dag_time_proj = None
            self.dag_time_gate_proj = None
            self.dag_maneuver_in = 0
            self.dag_maneuver_token_proj = None
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
        self.dag_time_in = int(self.dag_config.time_guidance_feature_dim)
        if bool(self.dag_config.use_time_guidance) and bool(self.dag_config.time_guidance_use_global):
            self.dag_time_in += int(self.dag_latent_in)
        if bool(self.dag_config.use_time_guidance):
            self.dag_time_proj = nn.Linear(self.dag_time_in, self.d_model)
            self.dag_time_gate_proj = nn.Linear(self.dag_time_in, self.d_model)
            nn.init.constant_(
                self.dag_time_gate_proj.bias,
                float(self.dag_config.time_guidance_init_gate_bias),
            )
        else:
            self.dag_time_proj = None
            self.dag_time_gate_proj = None
        self.dag_maneuver_in = int(self.dag_config.maneuver_token_feature_dim)
        if bool(self.dag_config.use_maneuver_tokens) and bool(self.dag_config.maneuver_token_use_global):
            self.dag_maneuver_in += int(self.dag_latent_in)
        if bool(self.dag_config.use_maneuver_tokens):
            self.dag_maneuver_token_proj = nn.Linear(self.dag_maneuver_in, self.d_model)
        else:
            self.dag_maneuver_token_proj = None

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

    def resolve_dag_time_guidance(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        z_dag: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        meta: Dict[str, torch.Tensor] = {}
        if not bool(self.dag_config.enabled) or not bool(self.dag_config.use_time_guidance):
            return None, meta

        dag_time_feat = self._lookup(batch, "dag/time_feat", "dag_time_feat")
        dag_time_mask = self._lookup(batch, "dag/time_mask", "dag_time_mask")
        if dag_time_feat is None or dag_time_mask is None:
            return None, meta

        if not torch.is_tensor(dag_time_feat):
            dag_time_feat = torch.as_tensor(dag_time_feat)
        if not torch.is_tensor(dag_time_mask):
            dag_time_mask = torch.as_tensor(dag_time_mask)

        dag_time_feat = dag_time_feat.to(device=device, dtype=dtype)
        dag_time_mask = dag_time_mask.to(device=device).bool()

        if bool(self.dag_config.time_guidance_use_global):
            if z_dag is None:
                z_expand = dag_time_feat.new_zeros(
                    (dag_time_feat.shape[0], dag_time_feat.shape[1], self.dag_latent_in)
                )
            else:
                z_expand = z_dag.to(device=device, dtype=dtype)[:, None, :].expand(
                    -1,
                    dag_time_feat.shape[1],
                    -1,
                )
            dag_time_feat = torch.cat([dag_time_feat, z_expand], dim=-1)

        meta["time_mask"] = dag_time_mask
        return dag_time_feat, meta

    def resolve_dag_maneuver_tokens(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        z_dag: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        meta: Dict[str, torch.Tensor] = {}
        if not bool(self.dag_config.enabled) or not bool(self.dag_config.use_maneuver_tokens):
            return None, meta

        dag_maneuver_feat = self._lookup(batch, "dag/maneuver_feat", "dag_maneuver_feat")
        dag_maneuver_mask = self._lookup(batch, "dag/maneuver_mask", "dag_maneuver_mask")
        if dag_maneuver_feat is None or dag_maneuver_mask is None:
            return None, meta

        if not torch.is_tensor(dag_maneuver_feat):
            dag_maneuver_feat = torch.as_tensor(dag_maneuver_feat)
        if not torch.is_tensor(dag_maneuver_mask):
            dag_maneuver_mask = torch.as_tensor(dag_maneuver_mask)

        dag_maneuver_feat = dag_maneuver_feat.to(device=device, dtype=dtype)
        dag_maneuver_mask = dag_maneuver_mask.to(device=device).bool()

        if bool(self.dag_config.maneuver_token_use_global):
            if z_dag is None:
                z_expand = dag_maneuver_feat.new_zeros(
                    (dag_maneuver_feat.shape[0], dag_maneuver_feat.shape[1], self.dag_latent_in)
                )
            else:
                z_expand = z_dag.to(device=device, dtype=dtype)[:, None, :].expand(
                    -1,
                    dag_maneuver_feat.shape[1],
                    -1,
                )
            dag_maneuver_feat = torch.cat([dag_maneuver_feat, z_expand], dim=-1)

        meta["maneuver_mask"] = dag_maneuver_mask
        return dag_maneuver_feat, meta

    def apply_dag_conditioning(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if bool(self.config.MODEL.get("LOCAL_CONTROL_FORWARD_ENABLED", False)):
            return batch
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
        dag_time_feat, time_meta = self.resolve_dag_time_guidance(
            batch,
            device=scene_token.device,
            dtype=scene_token.dtype,
            z_dag=z_dag,
        )
        dag_maneuver_feat, maneuver_meta = self.resolve_dag_maneuver_tokens(
            batch,
            device=scene_token.device,
            dtype=scene_token.dtype,
            z_dag=z_dag,
        )
        if z_dag is None and dag_time_feat is None and dag_maneuver_feat is None:
            return batch

        present = meta.get("source_used")
        if present is not None:
            present = present.to(device=scene_token.device, dtype=scene_token.dtype)

        p_drop = float(self.dag_config.dag_dropout_prob)
        if self.training and p_drop >= 1.0:
            batch[self._APPLIED_FLAG] = True
            batch["dag/latent_norm"] = torch.zeros((scene_token.shape[0],), device=scene_token.device, dtype=scene_token.dtype)
            batch["dag/gate_mean"] = torch.zeros((scene_token.shape[0],), device=scene_token.device, dtype=scene_token.dtype)
            if dag_time_feat is not None:
                batch["dag/time_control"] = torch.zeros(
                    (dag_time_feat.shape[0], dag_time_feat.shape[1], self.d_model),
                    device=scene_token.device,
                    dtype=scene_token.dtype,
                )
                batch["dag/time_mask"] = torch.zeros(
                    (dag_time_feat.shape[0], dag_time_feat.shape[1]),
                    device=scene_token.device,
                    dtype=torch.bool,
                )
                batch["dag/time_gate_mean"] = torch.zeros(
                    (dag_time_feat.shape[0],),
                    device=scene_token.device,
                    dtype=scene_token.dtype,
                )
            if dag_maneuver_feat is not None:
                batch["dag/maneuver_token"] = torch.zeros(
                    (dag_maneuver_feat.shape[0], dag_maneuver_feat.shape[1], self.d_model),
                    device=scene_token.device,
                    dtype=scene_token.dtype,
                )
                batch["dag/maneuver_mask"] = torch.zeros(
                    (dag_maneuver_feat.shape[0], dag_maneuver_feat.shape[1]),
                    device=scene_token.device,
                    dtype=torch.bool,
                )
            for key, value in meta.items():
                batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
            for key, value in time_meta.items():
                batch[f"dag/{key}"] = value.to(device=scene_token.device)
            for key, value in maneuver_meta.items():
                batch[f"dag/{key}"] = value.to(device=scene_token.device)
            return batch

        keep = None
        if self.training and p_drop > 0.0:
            batch_size = (
                int(z_dag.shape[0])
                if z_dag is not None
                else int(dag_time_feat.shape[0])
            )
            keep_device = z_dag.device if z_dag is not None else dag_time_feat.device
            keep_dtype = z_dag.dtype if z_dag is not None else dag_time_feat.dtype
            keep = (torch.rand((batch_size, 1), device=keep_device) >= p_drop).to(keep_dtype)
            if z_dag is not None:
                z_dag = z_dag * keep
        elif z_dag is not None:
            keep = torch.ones((z_dag.shape[0], 1), device=z_dag.device, dtype=z_dag.dtype)
        elif dag_time_feat is not None:
            keep = torch.ones((dag_time_feat.shape[0], 1), device=dag_time_feat.device, dtype=dag_time_feat.dtype)

        if z_dag is not None:
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
            batch["dag/latent"] = z_dag
            batch["dag/latent_norm"] = torch.linalg.norm(effective_z, dim=-1)
            batch["dag/gate_mean"] = effective_gate.mean(dim=-1)
        else:
            batch["dag/latent_norm"] = torch.zeros(
                (scene_token.shape[0],),
                device=scene_token.device,
                dtype=scene_token.dtype,
            )
            batch["dag/gate_mean"] = torch.zeros(
                (scene_token.shape[0],),
                device=scene_token.device,
                dtype=scene_token.dtype,
            )

        if dag_time_feat is not None:
            time_mask = time_meta.get("time_mask")
            if time_mask is None:
                time_mask = torch.ones(
                    dag_time_feat.shape[:2],
                    device=scene_token.device,
                    dtype=torch.bool,
                )
            if keep is not None:
                dag_time_feat = dag_time_feat * keep[:, None, :]

            mode = str(self.dag_config.time_guidance_mode).strip().lower()
            time_bias = self.dag_time_proj(dag_time_feat)
            if mode == "gated":
                time_gate = torch.sigmoid(self.dag_time_gate_proj(dag_time_feat))
                time_control = time_gate * time_bias
            elif mode == "additive":
                time_gate = torch.ones_like(time_bias)
                time_control = time_bias
            else:
                raise ValueError(
                    "Unsupported DAG time guidance mode "
                    f"{self.dag_config.time_guidance_mode!r}; expected `gated` or `additive`."
                )

            time_control = time_control * time_mask[:, :, None].to(dtype=scene_token.dtype)
            if present is not None:
                time_control = time_control * present[:, None, None]
                effective_time_gate = time_gate * present[:, None, None]
            else:
                effective_time_gate = time_gate
            batch["dag/time_control"] = time_control
            batch["dag/time_mask"] = time_mask
            denom = time_mask.float().sum(dim=1).clamp_min(1.0).to(scene_token.device, dtype=scene_token.dtype)
            batch["dag/time_gate_mean"] = (
                effective_time_gate.mean(dim=-1) * time_mask.float().to(scene_token.device)
            ).sum(dim=1).to(scene_token.dtype) / denom

        if dag_maneuver_feat is not None:
            maneuver_mask = maneuver_meta.get("maneuver_mask")
            if maneuver_mask is None:
                maneuver_mask = torch.ones(
                    dag_maneuver_feat.shape[:2],
                    device=scene_token.device,
                    dtype=torch.bool,
                )
            if keep is not None:
                dag_maneuver_feat = dag_maneuver_feat * keep[:, None, :]
            maneuver_token = self.dag_maneuver_token_proj(dag_maneuver_feat)
            maneuver_token = maneuver_token * maneuver_mask[:, :, None].to(dtype=scene_token.dtype)
            if present is not None:
                maneuver_token = maneuver_token * present[:, None, None]
                maneuver_mask = maneuver_mask & present[:, None].bool()
            batch["dag/maneuver_token"] = maneuver_token
            batch["dag/maneuver_mask"] = maneuver_mask

        batch[self._APPLIED_FLAG] = True
        for key, value in meta.items():
            batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
        for key, value in time_meta.items():
            if value.dtype == torch.bool:
                batch[f"dag/{key}"] = value.to(device=scene_token.device)
            else:
                batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
        for key, value in maneuver_meta.items():
            if value.dtype == torch.bool:
                batch[f"dag/{key}"] = value.to(device=scene_token.device)
            else:
                batch[f"dag/{key}"] = value.to(device=scene_token.device, dtype=scene_token.dtype)
        return batch

    def decode_motion(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if args and isinstance(args[0], dict):
            batch = self.apply_dag_conditioning(args[0])
            args = (batch,) + args[1:]
        return super().decode_motion(*args, **kwargs)
