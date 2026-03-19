"""Adv-BMT aligned NNX implementation.

Paper alignment:
- Adv-BMT (NeurIPS 2025) uses a shared bidirectional motion-token space over
  acceleration and yaw-rate controls. We implement that tokenization directly.
- The decoder path is relation-aware with explicit A2A / A2T / A2S components,
  matching the architecture intent described in the paper and appendix.
- Scene context is represented as map + traffic-light tokens so A2S attention
  can condition on actual scene structure instead of pooled hidden-state fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

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

from .fourier_embedding_nnx import FourierEmbeddingNNX
from .dag_gnn_nnx import NNXDAGEncoderConfig, NNXDAGGraphEncoder
from .relation_parity import compute_scene_relation_simple_jax


@dataclass
class BMTTokenSpaceConfig:
    """Token-space configuration from Adv-BMT paper.

    Shared forward/reverse token grid over acceleration and yaw-rate controls.
    """

    n_acc_bins: int = 33
    n_yaw_bins: int = 33
    acc_min: float = -10.0
    acc_max: float = 10.0
    yaw_min: float = -float(np.pi / 2.0)
    yaw_max: float = float(np.pi / 2.0)
    dt_s: float = 0.5

    @property
    def n_tokens(self) -> int:
        return self.n_acc_bins * self.n_yaw_bins


@dataclass
class NNXSceneEncoderConfig:
    """Scene token encoder configuration.

    Adv-BMT scene encoding uses map vectors and traffic-light state channels as
    structured scene context. We keep those feature dimensions first-class.
    """

    map_feature_dim: int = 27
    traffic_light_feature_dim: int = 7
    max_scene_tokens: int = 576
    map_encoder_style: Literal["mean_pool", "legacy_pointnet"] = "mean_pool"
    legacy_polyline_hidden_dim: int = 64
    legacy_polyline_num_layers: int = 2
    legacy_polyline_num_pre_layers: int = 1
    norm_style: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    use_post_proj_head: bool = False


@dataclass
class NNXRelationParityConfig:
    """Parity toggles mapped from legacy Adv-BMT config fields."""

    enabled: bool = False
    simple_relation: bool = True
    simple_relation_factor: int = 1
    remove_traffic_light_state: bool = True
    per_contour_point_relation: bool = False
    add_relation_to_v: bool = False
    remove_rel_norm: bool = False
    update_relation: bool = False

    s2s_knn: Optional[int] = 128
    s2s_distance: Optional[float] = None
    a2s_knn: Optional[int] = 128
    a2s_distance: Optional[float] = None
    a2a_knn: Optional[int] = 64
    a2a_distance: Optional[float] = 50.0

    heading_placeholder: float = -100.0
    scene_num_layers: int = 3


@dataclass
class NNXDecoderParityConfig:
    """Decoder parity toggles aligned to legacy motion_decoder_gpt semantics."""

    enabled: bool = False
    use_legacy_motion_embed: bool = False
    add_pe_for_token: bool = True
    randomize_agent_id: bool = False
    use_backward_indicator_embed: bool = False
    dense_masked_relation_attn: bool = True


@dataclass
class NNXDAGConditioningConfig:
    enabled: bool = False
    injection_mode: Literal["global_gated_residual"] = "global_gated_residual"
    dag_dropout_prob: float = 0.0
    use_null_latent: bool = True
    null_latent_init_std: float = 0.02


@dataclass
class NNXBMTConfig:
    """Model configuration for relation-aware NNX BMT."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4

    n_agent_types: int = 8
    n_special_tokens: int = 4
    max_agent_id: int = 512

    # Relation feature dims for A2A / A2T / A2S.
    a2a_rel_dim: int = 8
    a2t_rel_dim: int = 4
    a2s_rel_dim: int = 8

    scene_encoder: NNXSceneEncoderConfig = field(default_factory=NNXSceneEncoderConfig)
    relation: NNXRelationParityConfig = field(default_factory=NNXRelationParityConfig)
    decoder: NNXDecoderParityConfig = field(default_factory=NNXDecoderParityConfig)
    dag_encoder: NNXDAGEncoderConfig = field(default_factory=NNXDAGEncoderConfig)
    dag_conditioning: NNXDAGConditioningConfig = field(default_factory=NNXDAGConditioningConfig)
    token_space: BMTTokenSpaceConfig = field(default_factory=BMTTokenSpaceConfig)


class BidirectionalMotionTokenizer:
    """Bidirectional motion tokenizer from Adv-BMT details.

    Implements shared tokenization for forward/reverse with midpoint integration.
    """

    def __init__(self, cfg: Optional[BMTTokenSpaceConfig] = None):
        self.cfg = cfg or BMTTokenSpaceConfig()

        self.acc_edges = np.linspace(self.cfg.acc_min, self.cfg.acc_max, self.cfg.n_acc_bins + 1)
        self.yaw_edges = np.linspace(self.cfg.yaw_min, self.cfg.yaw_max, self.cfg.n_yaw_bins + 1)

        self.acc_centers = (self.acc_edges[:-1] + self.acc_edges[1:]) / 2.0
        self.yaw_centers = (self.yaw_edges[:-1] + self.yaw_edges[1:]) / 2.0

        # [V,2] lookup table: token -> (acc, yaw)
        acc_grid, yaw_grid = np.meshgrid(self.acc_centers, self.yaw_centers, indexing="ij")
        self._action_table = np.stack([acc_grid, yaw_grid], axis=-1).reshape(-1, 2).astype(np.float32)

    def action_table_np(self) -> np.ndarray:
        return self._action_table.copy()

    def token_to_action(self, token_id: int) -> Tuple[float, float]:
        a, y = self._action_table[int(token_id)]
        return float(a), float(y)

    def action_to_token(self, acceleration: float, yaw_rate: float) -> int:
        a_bin = int(np.clip(np.searchsorted(self.acc_edges[1:], acceleration), 0, self.cfg.n_acc_bins - 1))
        y_bin = int(np.clip(np.searchsorted(self.yaw_edges[1:], yaw_rate), 0, self.cfg.n_yaw_bins - 1))
        return a_bin * self.cfg.n_yaw_bins + y_bin

    def token_ids_to_actions_np(self, token_ids: np.ndarray) -> np.ndarray:
        return self._action_table[np.asarray(token_ids, dtype=np.int32)]

    def token_ids_to_actions_jax(self, token_ids: Any) -> Any:
        if jnp is None:
            raise RuntimeError("jax is required for token_ids_to_actions_jax")
        table = jnp.asarray(self._action_table)
        return jnp.take(table, token_ids, axis=0)

    def step_forward(self, state_xyhv: np.ndarray, token_id: int) -> np.ndarray:
        """Forward dynamics using midpoint integration.

        state format: [x, y, heading, speed]
        """
        x_t, y_t, heading_t, speed_t = [float(v) for v in state_xyhv]
        acc, yaw = self.token_to_action(token_id)
        dt = self.cfg.dt_s

        speed_next = speed_t + acc * dt
        heading_next = heading_t + yaw * dt

        speed_mid = 0.5 * (speed_t + speed_next)
        heading_mid = 0.5 * (heading_t + heading_next)

        x_next = x_t + speed_mid * np.cos(heading_mid) * dt
        y_next = y_t + speed_mid * np.sin(heading_mid) * dt

        return np.array([x_next, y_next, heading_next, speed_next], dtype=np.float32)

    def step_reverse(self, next_state_xyhv: np.ndarray, token_id: int) -> np.ndarray:
        """Reverse dynamics from future state and token.

        Implements inverse integration as described in Adv-BMT appendix.
        """
        x_next, y_next, heading_next, speed_next = [float(v) for v in next_state_xyhv]
        acc, yaw = self.token_to_action(token_id)
        dt = self.cfg.dt_s

        speed_t = speed_next - acc * dt
        heading_t = heading_next - yaw * dt

        speed_mid = 0.5 * (speed_t + speed_next)
        heading_mid = 0.5 * (heading_t + heading_next)

        x_t = x_next - speed_mid * np.cos(heading_mid) * dt
        y_t = y_next - speed_mid * np.sin(heading_mid) * dt

        return np.array([x_t, y_t, heading_t, speed_t], dtype=np.float32)

    def reconstruct_forward(self, initial_state_xyhv: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        states = [np.asarray(initial_state_xyhv, dtype=np.float32)]
        cur = states[0]
        for tok in token_ids:
            cur = self.step_forward(cur, int(tok))
            states.append(cur)
        return np.stack(states, axis=0)

    def reconstruct_reverse(self, final_state_xyhv: np.ndarray, reverse_token_ids: np.ndarray) -> np.ndarray:
        """Reverse reconstruct previous states from known final state.

        reverse_token_ids are assumed ordered from t=T-1 ... 0.
        """
        states = [np.asarray(final_state_xyhv, dtype=np.float32)]
        cur = states[0]
        for tok in reverse_token_ids:
            cur = self.step_reverse(cur, int(tok))
            states.append(cur)
        states = states[::-1]
        return np.stack(states, axis=0)


if HAS_NNX:

    class Linear(nnx.Module):
        def __init__(
            self,
            d_in: int,
            d_out: int,
            *,
            rngs: nnx.Rngs,
            scale: float = 0.02,
            use_bias: bool = True,
        ):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (d_in, d_out)) * scale)
            self.use_bias = bool(use_bias)
            self.b = nnx.Param(jnp.zeros((d_out,), dtype=jnp.float32)) if self.use_bias else None

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            out = jnp.einsum("...d,df->...f", x, self.w.value)
            if self.use_bias and self.b is not None:
                out = out + self.b.value
            return out


    class LayerNorm(nnx.Module):
        def __init__(self, d_model: int, *, eps: float = 1e-5):
            self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
            self.bias = nnx.Param(jnp.zeros((d_model,), dtype=jnp.float32))
            self.eps = eps

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            normed = (x - mean) / jnp.sqrt(var + self.eps)
            return normed * self.scale.value + self.bias.value


    class RMSNorm(nnx.Module):
        def __init__(self, d_model: int, *, eps: float = 1e-6):
            self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
            self.eps = eps

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
            return (x / rms) * self.scale.value


    class LegacyMLPBlock(nnx.Module):
        def __init__(
            self,
            d_in: int,
            d_out: int,
            *,
            rngs: nnx.Rngs,
            ret_before_act: bool = False,
            without_norm: bool = False,
        ):
            if ret_before_act:
                self.linear = Linear(d_in, d_out, rngs=rngs, use_bias=True)
                self.norm = None
                self.apply_relu = False
            elif without_norm:
                self.linear = Linear(d_in, d_out, rngs=rngs, use_bias=True)
                self.norm = None
                self.apply_relu = True
            else:
                self.linear = Linear(d_in, d_out, rngs=rngs, use_bias=False)
                self.norm = LayerNorm(d_out)
                self.apply_relu = True

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            out = self.linear(x)
            if self.norm is not None:
                out = self.norm(out)
            if self.apply_relu:
                out = jax.nn.relu(out)
            return out


    class LegacyMLP(nnx.Module):
        """NNX port of legacy `build_mlps` used by Adv-BMT scene encoding."""

        def __init__(
            self,
            c_in: int,
            mlp_channels: Tuple[int, ...],
            *,
            rngs: nnx.Rngs,
            ret_before_act: bool = False,
            without_norm: bool = False,
        ):
            blocks = []
            cur_in = int(c_in)
            num_layers = len(mlp_channels)
            for idx, c_out in enumerate(mlp_channels):
                is_last = idx + 1 == num_layers
                blocks.append(
                    LegacyMLPBlock(
                        cur_in,
                        int(c_out),
                        rngs=rngs,
                        ret_before_act=bool(is_last and ret_before_act),
                        without_norm=bool(without_norm and not (is_last and ret_before_act)),
                    )
                )
                cur_in = int(c_out)
            self.blocks = tuple(blocks)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            out = x
            for block in self.blocks:
                out = block(out)
            return out


    class LegacyPointNetPolylineEncoder(nnx.Module):
        """NNX port of Adv-BMT's PointNetPolylineEncoder."""

        def __init__(
            self,
            *,
            in_channels: int,
            hidden_dim: int,
            num_layers: int,
            num_pre_layers: int,
            out_channels: Optional[int],
            rngs: nnx.Rngs,
        ):
            self.pre_mlps = LegacyMLP(
                int(in_channels),
                tuple([int(hidden_dim)] * int(num_pre_layers)),
                rngs=rngs,
                ret_before_act=False,
                without_norm=False,
            )
            self.mlps = LegacyMLP(
                int(hidden_dim) * 2,
                tuple([int(hidden_dim)] * max(0, int(num_layers) - int(num_pre_layers))),
                rngs=rngs,
                ret_before_act=False,
                without_norm=False,
            )
            self.out_mlps = (
                LegacyMLP(
                    int(hidden_dim),
                    (int(hidden_dim), int(out_channels)),
                    rngs=rngs,
                    ret_before_act=True,
                    without_norm=True,
                )
                if out_channels is not None
                else None
            )

        @staticmethod
        def _mask_points(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(mask[..., None], x, jnp.zeros_like(x))

        def __call__(self, polylines: jnp.ndarray, polylines_mask: jnp.ndarray) -> jnp.ndarray:
            point_feat = self.pre_mlps(polylines)
            point_feat = self._mask_points(point_feat, polylines_mask.astype(bool))

            pooled_feature = jnp.max(point_feat, axis=2)
            pooled_expanded = jnp.repeat(pooled_feature[:, :, None, :], point_feat.shape[2], axis=2)
            point_feat = jnp.concatenate([point_feat, pooled_expanded], axis=-1)

            point_feat = self.mlps(point_feat)
            point_feat = self._mask_points(point_feat, polylines_mask.astype(bool))

            polyline_feat = jnp.max(point_feat, axis=2)
            valid_mask = jnp.any(polylines_mask.astype(bool), axis=-1)

            if self.out_mlps is not None:
                polyline_feat = self.out_mlps(polyline_feat)
            polyline_feat = jnp.where(valid_mask[..., None], polyline_feat, jnp.zeros_like(polyline_feat))
            return polyline_feat


    class SceneRelationSelfAttentionBlock(nnx.Module):
        """Scene self-attention block with legacy-style simple relation terms."""

        def __init__(self, cfg: NNXBMTConfig, rel_dim: int, *, rngs: nnx.Rngs):
            self.cfg = cfg
            self.d_model = int(cfg.d_model)
            self.n_heads = int(cfg.n_heads)
            if self.d_model % self.n_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
            self.head_dim = self.d_model // self.n_heads
            self.norm_style = str(cfg.scene_encoder.norm_style)

            self.token_norm = LayerNorm(self.d_model) if self.norm_style == "layernorm" else RMSNorm(self.d_model)
            self.rel_norm = None
            if not cfg.relation.remove_rel_norm:
                self.rel_norm = LayerNorm(rel_dim) if self.norm_style == "layernorm" else RMSNorm(rel_dim)

            self.q_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.k_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.v_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.q_rel_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.rel_k_proj = Linear(rel_dim, self.d_model, rngs=rngs)
            self.rel_v_proj = Linear(rel_dim, self.d_model, rngs=rngs)
            self.o_proj = Linear(self.d_model, self.d_model, rngs=rngs)

            ff_hidden = self.d_model * int(cfg.ff_mult)
            self.ff_norm = LayerNorm(self.d_model) if self.norm_style == "layernorm" else RMSNorm(self.d_model)
            self.ff_in = Linear(self.d_model, ff_hidden, rngs=rngs)
            self.ff_out = Linear(ff_hidden, self.d_model, rngs=rngs)

        @staticmethod
        def _gather_kv(x: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
            """Gather [B,H,S,D] by [B,S,K] into [B,H,S,K,D]."""

            def gather_batch(x_b: jnp.ndarray, idx_b: jnp.ndarray) -> jnp.ndarray:
                def gather_head(x_h: jnp.ndarray) -> jnp.ndarray:
                    return x_h[idx_b]

                return jax.vmap(gather_head, in_axes=0)(x_b)

            return jax.vmap(gather_batch, in_axes=(0, 0))(x, indices)

        def __call__(
            self,
            *,
            scene_tokens: jnp.ndarray,  # [B,S,D]
            rel_emb: jnp.ndarray,  # [B,S,K,R]
            rel_mask: jnp.ndarray,  # [B,S,K]
            rel_indices: jnp.ndarray,  # [B,S,K]
        ) -> jnp.ndarray:
            bsz, s_len, _ = scene_tokens.shape
            k_len = rel_emb.shape[2]

            h = self.token_norm(scene_tokens)
            rel_in = rel_emb if self.rel_norm is None else self.rel_norm(rel_emb)

            q = self.q_proj(h).reshape(bsz, s_len, self.n_heads, self.head_dim)
            k = self.k_proj(h).reshape(bsz, s_len, self.n_heads, self.head_dim)
            v = self.v_proj(h).reshape(bsz, s_len, self.n_heads, self.head_dim)
            q_rel = self.q_rel_proj(h).reshape(bsz, s_len, self.n_heads, self.head_dim)

            q = jnp.transpose(q, (0, 2, 1, 3))  # [B,H,S,Dh]
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))
            q_rel = jnp.transpose(q_rel, (0, 2, 1, 3))

            k_g = self._gather_kv(k, rel_indices.astype(jnp.int32))  # [B,H,S,K,Dh]
            v_g = self._gather_kv(v, rel_indices.astype(jnp.int32))

            rel_k = self.rel_k_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                bsz, s_len, k_len, self.n_heads, self.head_dim
            )
            rel_k = jnp.transpose(rel_k, (0, 3, 1, 2, 4))

            if self.cfg.relation.add_relation_to_v:
                rel_v = self.rel_v_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                    bsz, s_len, k_len, self.n_heads, self.head_dim
                )
                rel_v = jnp.transpose(rel_v, (0, 3, 1, 2, 4))
            else:
                rel_v = rel_k

            scale = 1.0 / np.sqrt(float(self.head_dim))
            score_main = jnp.sum(q[:, :, :, None, :] * k_g, axis=-1) * scale
            score_rel = jnp.sum(q_rel[:, :, :, None, :] * rel_k, axis=-1) * scale
            scores = score_main + score_rel

            scores = jnp.where(rel_mask[:, None, :, :], scores, jnp.full_like(scores, -1e9))
            attn = jax.nn.softmax(scores, axis=-1)

            value = v_g + rel_v
            out = jnp.sum(attn[..., None] * value, axis=-2)  # [B,H,S,Dh]
            out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bsz, s_len, self.d_model)
            scene_tokens = scene_tokens + self.o_proj(out)

            ff_hidden = self.ff_in(self.ff_norm(scene_tokens))
            ff = self.ff_out(jax.nn.gelu(ff_hidden, approximate=True))
            scene_tokens = scene_tokens + ff
            return scene_tokens


    class NNXSceneTokenEncoder(nnx.Module):
        """Encodes map + traffic-light scene tensors into scene tokens."""

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = int(cfg.d_model)
            self.map_encoder_style = str(cfg.scene_encoder.map_encoder_style)
            self.norm_style = str(cfg.scene_encoder.norm_style)
            # Adv-BMT hard-codes 11 scene-history steps for traffic-light context.
            self.history_steps = 11
            if self.map_encoder_style == "legacy_pointnet":
                self.map_vector_proj = None
                self.map_token_proj = None
                self.map_polyline_encoder = LegacyPointNetPolylineEncoder(
                    in_channels=int(cfg.scene_encoder.map_feature_dim),
                    hidden_dim=int(cfg.scene_encoder.legacy_polyline_hidden_dim),
                    num_layers=int(cfg.scene_encoder.legacy_polyline_num_layers),
                    num_pre_layers=int(cfg.scene_encoder.legacy_polyline_num_pre_layers),
                    out_channels=d_model,
                    rngs=rngs,
                )
            else:
                self.map_vector_proj = Linear(cfg.scene_encoder.map_feature_dim, d_model, rngs=rngs)
                self.map_token_proj = Linear(d_model, d_model, rngs=rngs)
                self.map_polyline_encoder = None
            self.position_proj = Linear(3, d_model, rngs=rngs)
            if bool(cfg.relation.remove_traffic_light_state):
                # Legacy MidGPT consumes one pre-collapsed 7-D traffic-light feature
                # per token. Those 7 dimensions already contain the stop-point xyz, so
                # we must not add a second learned position embedding here.
                self.light_mlps = LegacyMLP(
                    int(cfg.scene_encoder.traffic_light_feature_dim),
                    (d_model,),
                    rngs=rngs,
                    ret_before_act=True,
                    without_norm=False,
                )
            else:
                # The non-collapsed legacy path flattens 11 history steps of 7-D light
                # state/features and runs a 3-layer MLP.
                self.light_mlps = LegacyMLP(
                    int(cfg.scene_encoder.traffic_light_feature_dim) * self.history_steps,
                    (d_model, d_model, d_model),
                    rngs=rngs,
                    ret_before_act=True,
                    without_norm=False,
                )
            self.out_norm = LayerNorm(d_model) if self.norm_style == "layernorm" else RMSNorm(d_model)
            self.post_norm = (
                LayerNorm(d_model) if bool(cfg.scene_encoder.use_post_proj_head) else None
            )
            self.post_proj = (
                Linear(d_model, d_model, rngs=rngs) if bool(cfg.scene_encoder.use_post_proj_head) else None
            )

            self.relation_enabled = bool(cfg.relation.enabled)
            if self.relation_enabled:
                if not cfg.relation.simple_relation:
                    raise ValueError("P1 runtime currently supports simple_relation=True only")
                rel_hidden = max(1, int(d_model // max(1, cfg.relation.simple_relation_factor)))
                self.scene_relation_embed = FourierEmbeddingNNX(
                    input_dim=3,
                    hidden_dim=rel_hidden,
                    num_freq_bands=64,
                    rngs=rngs,
                )
                n_scene_layers = max(1, int(cfg.relation.scene_num_layers))
                self.scene_relation_layers = tuple(
                    SceneRelationSelfAttentionBlock(cfg, rel_hidden, rngs=rngs) for _ in range(n_scene_layers)
                )
            else:
                self.scene_relation_embed = None
                self.scene_relation_layers = tuple()

        @staticmethod
        def _masked_circular_mean(angles: jnp.ndarray, mask: jnp.ndarray, *, axis: int) -> jnp.ndarray:
            mask_f = mask.astype(jnp.float32)
            sin_sum = jnp.sum(jnp.sin(angles) * mask_f, axis=axis)
            cos_sum = jnp.sum(jnp.cos(angles) * mask_f, axis=axis)
            valid = jnp.sum(mask_f, axis=axis) > 0
            mean = jnp.arctan2(sin_sum, cos_sum)
            return jnp.where(valid, mean, jnp.zeros_like(mean))

        def _encode_map(
            self,
            *,
            map_feature: jnp.ndarray,  # [B,M,V,Fm]
            map_feature_valid_mask: jnp.ndarray,  # [B,M,V] bool
            map_position: jnp.ndarray,  # [B,M,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            map_token_mask = jnp.any(map_feature_valid_mask, axis=-1)
            if self.map_encoder_style == "legacy_pointnet":
                map_tokens = self.map_polyline_encoder(map_feature, map_feature_valid_mask.astype(bool))
            else:
                bsz, n_map, n_vec, feat_dim = map_feature.shape
                map_vec = self.map_vector_proj(map_feature.reshape(-1, feat_dim)).reshape(
                    bsz, n_map, n_vec, self.cfg.d_model
                )
                valid = map_feature_valid_mask.astype(jnp.float32)[..., None]
                num = jnp.sum(map_vec * valid, axis=2)
                den = jnp.maximum(1.0, jnp.sum(valid, axis=2))
                pooled = num / den
                pos_e = self.position_proj(map_position.reshape(-1, 3)).reshape(bsz, n_map, self.cfg.d_model)
                map_tokens = self.map_token_proj(pooled + pos_e)
            map_tokens = jnp.where(map_token_mask[..., None], map_tokens, jnp.zeros_like(map_tokens))

            # Adv-BMT map heading is derived from polyline segment headings.
            map_heading = self._masked_circular_mean(map_feature[..., 9], map_feature_valid_mask, axis=2)
            map_heading = jnp.where(map_token_mask, map_heading, jnp.zeros_like(map_heading))
            return map_tokens, map_token_mask, map_position, map_heading

        def _encode_traffic_lights(
            self,
            *,
            traffic_light_feature: jnp.ndarray,
            traffic_light_valid_mask: jnp.ndarray,
            traffic_light_position: jnp.ndarray,  # [B,L,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            remove_tl_state = bool(self.cfg.relation.remove_traffic_light_state)

            if traffic_light_feature.ndim == 4:
                # [B,T,L,7]
                if remove_tl_state:
                    # Legacy preprocessing collapses each light to a single 7-D feature:
                    # [stop_point_xyz, one_hot(light_state)] using the majority state.
                    light_mask = jnp.any(traffic_light_valid_mask, axis=1)
                    valid = traffic_light_valid_mask.astype(jnp.float32)
                    pos_num = jnp.sum(traffic_light_feature[..., :3] * valid[..., None], axis=1)
                    pos_den = jnp.maximum(1.0, jnp.sum(valid[..., None], axis=1))
                    stop_point = pos_num / pos_den
                    state_scores = jnp.sum(traffic_light_feature[..., 3:7] * valid[..., None], axis=1)
                    cls = jnp.argmax(state_scores, axis=-1)
                    onehot = jax.nn.one_hot(cls, 4, dtype=jnp.float32)
                    light_feat = jnp.concatenate([stop_point, onehot], axis=-1)
                else:
                    valid = traffic_light_valid_mask[:, : self.history_steps].astype(jnp.float32)
                    feature = traffic_light_feature[:, : self.history_steps] * valid[..., None]
                    feature = jnp.transpose(feature, (0, 2, 1, 3))
                    light_feat = feature.reshape(feature.shape[0], feature.shape[1], -1)
                    light_mask = jnp.any(traffic_light_valid_mask, axis=1)
            else:
                # [B,L,7]
                light_feat = traffic_light_feature
                light_mask = traffic_light_valid_mask

            bsz, n_light, feat_dim = light_feat.shape
            light_tokens = self.light_mlps(light_feat.reshape(-1, feat_dim)).reshape(
                bsz, n_light, self.cfg.d_model
            )
            light_tokens = jnp.where(light_mask[..., None], light_tokens, jnp.zeros_like(light_tokens))
            light_heading = jnp.full(
                (bsz, n_light),
                float(self.cfg.relation.heading_placeholder),
                dtype=jnp.float32,
            )
            return light_tokens, light_mask, traffic_light_position, light_heading

        def __call__(
            self,
            *,
            map_feature: jnp.ndarray,  # [B,M,V,Fm]
            map_feature_valid_mask: jnp.ndarray,  # [B,M,V]
            map_position: jnp.ndarray,  # [B,M,3]
            traffic_light_feature: Optional[jnp.ndarray] = None,  # [B,T,L,Fl] or [B,L,Fl]
            traffic_light_valid_mask: Optional[jnp.ndarray] = None,  # [B,T,L] or [B,L]
            traffic_light_position: Optional[jnp.ndarray] = None,  # [B,L,3]
            return_metadata: bool = False,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
            debug_meta: Dict[str, jnp.ndarray] = {}

            map_tokens, map_mask, map_pos, map_heading = self._encode_map(
                map_feature=map_feature,
                map_feature_valid_mask=map_feature_valid_mask,
                map_position=map_position,
            )

            if (
                traffic_light_feature is not None
                and traffic_light_valid_mask is not None
                and traffic_light_position is not None
            ):
                light_tokens, light_mask, light_pos, light_heading = self._encode_traffic_lights(
                    traffic_light_feature=traffic_light_feature,
                    traffic_light_valid_mask=traffic_light_valid_mask,
                    traffic_light_position=traffic_light_position,
                )
                scene_tokens = jnp.concatenate([map_tokens, light_tokens], axis=1)
                scene_mask = jnp.concatenate([map_mask, light_mask], axis=1)
                scene_position = jnp.concatenate([map_pos, light_pos], axis=1)
                scene_heading = jnp.concatenate([map_heading, light_heading], axis=1)
            else:
                scene_tokens, scene_mask, scene_position, scene_heading = map_tokens, map_mask, map_pos, map_heading

            # Keep bounded token count for predictable compute in early-stage training.
            max_tokens = int(self.cfg.scene_encoder.max_scene_tokens)
            if max_tokens > 0 and scene_tokens.shape[1] > max_tokens:
                scene_tokens = scene_tokens[:, :max_tokens, :]
                scene_mask = scene_mask[:, :max_tokens]
                scene_position = scene_position[:, :max_tokens, :]
                scene_heading = scene_heading[:, :max_tokens]

            if self.relation_enabled and scene_tokens.shape[1] > 0:
                knn = self.cfg.relation.s2s_knn
                if knn is None:
                    knn = int(scene_tokens.shape[1])

                rel_feat, rel_mask, rel_indices = compute_scene_relation_simple_jax(
                    query_pos=scene_position,
                    query_heading=scene_heading,
                    query_valid_mask=scene_mask,
                    key_pos=scene_position,
                    key_heading=scene_heading,
                    key_valid_mask=scene_mask,
                    heading_placeholder=float(self.cfg.relation.heading_placeholder),
                    knn=knn,
                    max_distance=self.cfg.relation.s2s_distance,
                    gather=True,
                )

                rel_emb = self.scene_relation_embed(
                    rel_feat.reshape(-1, rel_feat.shape[-1])
                ).reshape(rel_feat.shape[0], rel_feat.shape[1], rel_feat.shape[2], -1)

                for layer in self.scene_relation_layers:
                    scene_tokens = layer(
                        scene_tokens=scene_tokens,
                        rel_emb=rel_emb,
                        rel_mask=rel_mask,
                        rel_indices=rel_indices,
                    )

                if return_metadata:
                    debug_meta["scene_s2s_rel_feat"] = rel_feat
                    debug_meta["scene_s2s_mask"] = rel_mask
                    if rel_indices is not None:
                        debug_meta["scene_s2s_indices"] = rel_indices

            scene_tokens = self.out_norm(scene_tokens)
            if self.post_norm is not None and self.post_proj is not None:
                scene_tokens = self.post_proj(self.post_norm(scene_tokens))
                scene_tokens = jnp.where(scene_mask[..., None], scene_tokens, jnp.zeros_like(scene_tokens))
            if return_metadata:
                return scene_tokens, scene_mask, scene_position, debug_meta
            return scene_tokens, scene_mask, scene_position


    class MultiHeadAttention(nnx.Module):
        """Dense masked MHA with optional legacy-style relation score/value terms."""

        def __init__(
            self,
            d_model: int,
            n_heads: int,
            *,
            relation_dim: Optional[int] = None,
            add_relation_to_v: bool = False,
            remove_rel_norm: bool = False,
            rngs: nnx.Rngs,
        ):
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.relation_dim = relation_dim
            self.add_relation_to_v = bool(add_relation_to_v)

            self.q_proj = Linear(d_model, d_model, rngs=rngs)
            self.k_proj = Linear(d_model, d_model, rngs=rngs)
            self.v_proj = Linear(d_model, d_model, rngs=rngs)
            self.o_proj = Linear(d_model, d_model, rngs=rngs)
            if relation_dim is not None and int(relation_dim) > 0:
                rel_dim = int(relation_dim)
                self.rel_norm = None if remove_rel_norm else RMSNorm(rel_dim)
                self.q_rel_proj = Linear(d_model, d_model, rngs=rngs)
                self.rel_k_proj = Linear(rel_dim, d_model, rngs=rngs)
                self.rel_v_proj = Linear(rel_dim, d_model, rngs=rngs)
            else:
                self.rel_norm = None
                self.q_rel_proj = None
                self.rel_k_proj = None
                self.rel_v_proj = None

        @staticmethod
        def _gather_kv(x: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
            """Gather [B,H,K,D] by [B,Q,R] into [B,H,Q,R,D]."""

            def gather_batch(x_b: jnp.ndarray, idx_b: jnp.ndarray) -> jnp.ndarray:
                def gather_head(x_h: jnp.ndarray) -> jnp.ndarray:
                    return x_h[idx_b]

                return jax.vmap(gather_head, in_axes=0)(x_b)

            return jax.vmap(gather_batch, in_axes=(0, 0))(x, indices)

        @staticmethod
        def _gather_pairwise(x: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
            """Gather pairwise [B,Q,K,...] by [B,Q,R] into [B,Q,R,...]."""

            def gather_batch(x_b: jnp.ndarray, idx_b: jnp.ndarray) -> jnp.ndarray:
                return jax.vmap(lambda row, row_idx: row[row_idx], in_axes=(0, 0))(x_b, idx_b)

            return jax.vmap(gather_batch, in_axes=(0, 0))(x, indices)

        @staticmethod
        def _full_relation_indices(*, bsz: int, q_len: int, k_len: int) -> jnp.ndarray:
            base = jnp.arange(k_len, dtype=jnp.int32)[None, None, :]
            return jnp.broadcast_to(base, (bsz, q_len, k_len))

        @staticmethod
        def _normalize_relation_indices(
            rel_indices: Optional[jnp.ndarray],
            *,
            bsz: int,
            q_len: int,
            k_len: int,
            rel_feat: Optional[jnp.ndarray],
            mask: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
        ) -> jnp.ndarray:
            if rel_indices is not None and rel_indices.shape[-1] > 0:
                return rel_indices.astype(jnp.int32)

            for candidate, name in ((rel_feat, "rel_feat"), (mask, "mask"), (rel_mask, "rel_mask")):
                if candidate is None:
                    continue
                if candidate.shape[2] != k_len:
                    raise ValueError(
                        f"{name} has gathered width {candidate.shape[2]} but rel_indices are missing; "
                        f"expected full key width {k_len}"
                    )

            return MultiHeadAttention._full_relation_indices(bsz=bsz, q_len=q_len, k_len=k_len)

        @staticmethod
        def _align_pairwise_tensor(
            tensor: Optional[jnp.ndarray],
            *,
            rel_indices: jnp.ndarray,
            key_len: int,
            kind: str,
        ) -> Optional[jnp.ndarray]:
            if tensor is None:
                return None
            if tensor.shape[2] == rel_indices.shape[2]:
                return tensor
            if tensor.shape[2] != key_len:
                raise ValueError(
                    f"{kind} width mismatch: got {tensor.shape[2]}, expected gathered width "
                    f"{rel_indices.shape[2]} or key width {key_len}"
                )
            return MultiHeadAttention._gather_pairwise(tensor, rel_indices)

        @staticmethod
        def _apply_attention_mask(scores: jnp.ndarray, mask: Optional[jnp.ndarray]) -> jnp.ndarray:
            if mask is None:
                return scores
            keep = mask.astype(bool)
            any_valid = jnp.any(keep, axis=-1, keepdims=True)
            first_slot = jax.nn.one_hot(
                jnp.zeros(keep.shape[:2], dtype=jnp.int32),
                keep.shape[-1],
                dtype=jnp.bool_,
            )
            safe_keep = jnp.where(any_valid, keep, first_slot)
            return jnp.where(safe_keep[:, None, :, :], scores, jnp.full_like(scores, -1e9))

        def __call__(
            self,
            query: jnp.ndarray,
            key_value: jnp.ndarray,
            *,
            mask: Optional[jnp.ndarray] = None,   # [B,Lq,Lk] bool
            rel_feat: Optional[jnp.ndarray] = None,  # [B,Lq,Lk,R]
            rel_mask: Optional[jnp.ndarray] = None,  # [B,Lq,Lk] bool
            rel_indices: Optional[jnp.ndarray] = None,  # [B,Lq,R] int
        ) -> jnp.ndarray:
            bsz, q_len, _ = query.shape
            _, k_len, _ = key_value.shape

            q = self.q_proj(query).reshape(bsz, q_len, self.n_heads, self.head_dim)
            k = self.k_proj(key_value).reshape(bsz, k_len, self.n_heads, self.head_dim)
            v = self.v_proj(key_value).reshape(bsz, k_len, self.n_heads, self.head_dim)

            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            rel_indices = self._normalize_relation_indices(
                rel_indices,
                bsz=bsz,
                q_len=q_len,
                k_len=k_len,
                rel_feat=rel_feat,
                mask=mask,
                rel_mask=rel_mask,
            )
            k_g = self._gather_kv(k, rel_indices)
            v_g = self._gather_kv(v, rel_indices)
            gathered_mask = self._align_pairwise_tensor(mask, rel_indices=rel_indices, key_len=k_len, kind="mask")
            gathered_rel_mask = self._align_pairwise_tensor(
                rel_mask,
                rel_indices=rel_indices,
                key_len=k_len,
                kind="rel_mask",
            )

            scale = 1.0 / np.sqrt(float(self.head_dim))
            scores = jnp.sum(q[:, :, :, None, :] * k_g, axis=-1) * scale

            rel_v = None
            if rel_feat is not None and self.relation_dim is not None:
                rel_feat = self._align_pairwise_tensor(
                    rel_feat,
                    rel_indices=rel_indices,
                    key_len=k_len,
                    kind="rel_feat",
                )
                rel_in = rel_feat if self.rel_norm is None else self.rel_norm(rel_feat)
                q_rel = self.q_rel_proj(query).reshape(bsz, q_len, self.n_heads, self.head_dim)
                q_rel = jnp.transpose(q_rel, (0, 2, 1, 3))  # [B,H,Q,D]

                rel_k = self.rel_k_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                    bsz, q_len, rel_in.shape[2], self.n_heads, self.head_dim
                )
                rel_k = jnp.transpose(rel_k, (0, 3, 1, 2, 4))  # [B,H,Q,K,D]
                score_rel = jnp.sum(q_rel[:, :, :, None, :] * rel_k, axis=-1) * scale
                scores = scores + score_rel

                if self.add_relation_to_v:
                    rel_v = self.rel_v_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                        bsz, q_len, rel_in.shape[2], self.n_heads, self.head_dim
                    )
                    rel_v = jnp.transpose(rel_v, (0, 3, 1, 2, 4))
                else:
                    rel_v = rel_k

            combined_mask = gathered_mask
            if gathered_rel_mask is not None:
                gathered_rel_mask = gathered_rel_mask.astype(bool)
                combined_mask = (
                    gathered_rel_mask
                    if combined_mask is None
                    else jnp.logical_and(combined_mask.astype(bool), gathered_rel_mask)
                )
            scores = self._apply_attention_mask(scores, combined_mask)

            attn = jax.nn.softmax(scores, axis=-1)

            if rel_v is None:
                out = jnp.sum(attn[..., None] * v_g, axis=-2)
            else:
                value = v_g + rel_v
                out = jnp.sum(attn[..., None] * value, axis=-2)

            out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bsz, q_len, self.d_model)
            return self.o_proj(out)


    class RelationAwareDecoderBlock(nnx.Module):
        """One decoder block with explicit A2A, A2T, A2S attention."""

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = cfg.d_model
            self.legacy_parity_mode = bool(cfg.decoder.enabled) and bool(cfg.decoder.use_legacy_motion_embed)
            self.use_sparse_relation_attn = self.legacy_parity_mode and (not bool(cfg.decoder.dense_masked_relation_attn))
            rel_factor = max(1, int(cfg.relation.simple_relation_factor))
            legacy_rel_dim = max(1, int(d_model // rel_factor))
            self.a2a_relation_dim = legacy_rel_dim if self.legacy_parity_mode else int(cfg.a2a_rel_dim)
            self.a2t_relation_dim = legacy_rel_dim if self.legacy_parity_mode else int(cfg.a2t_rel_dim)
            self.a2s_relation_dim = legacy_rel_dim if self.legacy_parity_mode else int(cfg.a2s_rel_dim)
            if self.legacy_parity_mode:
                self.a2t_norm = LayerNorm(d_model)
                self.a2a_norm = LayerNorm(d_model)
                self.a2s_norm = LayerNorm(d_model)
                self.mlp_prenorm = LayerNorm(d_model)
                if bool(cfg.relation.remove_rel_norm):
                    self.a2t_rel_norm = None
                    self.a2a_rel_norm = None
                    self.a2s_rel_norm = None
                else:
                    self.a2t_rel_norm = LayerNorm(self.a2t_relation_dim)
                    self.a2a_rel_norm = LayerNorm(self.a2a_relation_dim)
                    self.a2s_rel_norm = LayerNorm(self.a2s_relation_dim)
                self.norm1 = None
                self.norm2 = None
            else:
                self.norm1 = RMSNorm(d_model)
                self.norm2 = RMSNorm(d_model)
                self.a2t_norm = None
                self.a2a_norm = None
                self.a2s_norm = None
                self.mlp_prenorm = None
                self.a2t_rel_norm = None
                self.a2a_rel_norm = None
                self.a2s_rel_norm = None

            self.a2a_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=self.a2a_relation_dim,
                add_relation_to_v=bool(cfg.relation.add_relation_to_v),
                remove_rel_norm=bool(cfg.relation.remove_rel_norm),
                rngs=rngs,
            )
            self.a2t_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=self.a2t_relation_dim,
                add_relation_to_v=bool(cfg.relation.add_relation_to_v),
                remove_rel_norm=bool(cfg.relation.remove_rel_norm),
                rngs=rngs,
            )
            self.a2s_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=self.a2s_relation_dim,
                add_relation_to_v=bool(cfg.relation.add_relation_to_v),
                remove_rel_norm=bool(cfg.relation.remove_rel_norm),
                rngs=rngs,
            )

            ff_hidden = d_model * cfg.ff_mult
            self.ff_in = Linear(d_model, ff_hidden, rngs=rngs)
            self.ff_out = Linear(ff_hidden, d_model, rngs=rngs)

        def _a2a(
            self,
            h: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
            rel_indices: Optional[jnp.ndarray],
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*T,N,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz * t_steps, n_agents, d_model)
            rel_bt = None
            mask_bt = None
            indices_bt = None
            if rel is not None:
                rel_bt = rel.reshape(bsz * t_steps, n_agents, rel.shape[3], rel.shape[-1])
            if rel_mask is not None:
                mask_bt = rel_mask.reshape(bsz * t_steps, n_agents, rel_mask.shape[3]).astype(bool)
            if rel_indices is not None:
                indices_bt = rel_indices.reshape(bsz * t_steps, n_agents, rel_indices.shape[3]).astype(jnp.int32)
            out = self.a2a_attn(
                q,
                q,
                rel_feat=rel_bt,
                rel_mask=mask_bt,
                rel_indices=indices_bt if self.use_sparse_relation_attn else None,
            )
            return out.reshape(bsz, t_steps, n_agents, d_model)

        def _a2t(
            self,
            h: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
            rel_indices: Optional[jnp.ndarray],
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*N,T,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = jnp.transpose(h, (0, 2, 1, 3)).reshape(bsz * n_agents, t_steps, d_model)
            rel_bn = None
            indices_bn = None
            if rel is not None:
                rel_bn = rel.reshape(bsz * n_agents, t_steps, rel.shape[3], rel.shape[-1])
            if rel_indices is not None:
                indices_bn = rel_indices.reshape(bsz * n_agents, t_steps, rel_indices.shape[3]).astype(jnp.int32)

            # Temporal causality mask:
            # token at step t can only attend to [0..t]. This prevents future
            # leakage during teacher-forcing and aligns training with rollout.
            causal = jnp.tril(jnp.ones((t_steps, t_steps), dtype=bool))
            causal = jnp.broadcast_to(causal[None, :, :], (bsz * n_agents, t_steps, t_steps))
            rel_mask_bn = None
            if rel_mask is not None:
                rel_mask_bn = rel_mask.reshape(bsz * n_agents, t_steps, rel_mask.shape[3]).astype(bool)

            out = self.a2t_attn(
                q,
                q,
                mask=causal,
                rel_feat=rel_bn,
                rel_mask=rel_mask_bn,
                rel_indices=indices_bn if self.use_sparse_relation_attn else None,
            )
            out = out.reshape(bsz, n_agents, t_steps, d_model)
            return jnp.transpose(out, (0, 2, 1, 3))

        def _a2s(
            self,
            h: jnp.ndarray,
            scene_tokens: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
            rel_indices: Optional[jnp.ndarray],
            scene_token_mask: Optional[jnp.ndarray] = None,  # [B,S]
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B,T*N,D], scene_tokens: [B,S,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz, t_steps * n_agents, d_model)

            rel_qs = None
            indices_qs = None
            if rel is not None:
                rel_qs = rel.reshape(bsz, t_steps * n_agents, rel.shape[3], rel.shape[-1])
            if rel_indices is not None:
                indices_qs = rel_indices.reshape(bsz, t_steps * n_agents, rel_indices.shape[3]).astype(jnp.int32)

            attn_mask = None
            if scene_token_mask is not None:
                attn_mask = jnp.broadcast_to(
                    scene_token_mask[:, None, :], (bsz, t_steps * n_agents, scene_tokens.shape[1])
                )
            rel_mask_qs = None
            if rel_mask is not None:
                rel_mask_qs = rel_mask.reshape(bsz, t_steps * n_agents, rel_mask.shape[3]).astype(bool)

            out = self.a2s_attn(
                q,
                scene_tokens,
                mask=attn_mask,
                rel_feat=rel_qs,
                rel_mask=rel_mask_qs,
                rel_indices=indices_qs if self.use_sparse_relation_attn else None,
            )
            return out.reshape(bsz, t_steps, n_agents, d_model)

        def __call__(
            self,
            x: jnp.ndarray,  # [B,T,N,D]
            *,
            scene_tokens: jnp.ndarray,  # [B,S,D]
            scene_token_mask: Optional[jnp.ndarray] = None,  # [B,S]
            a2a_rel: Optional[jnp.ndarray] = None,  # [B,T,N,N,Ra2a]
            a2t_rel: Optional[jnp.ndarray] = None,  # [B,N,T,T,Ra2t]
            a2s_rel: Optional[jnp.ndarray] = None,  # [B,T,N,S,Ra2s]
            a2a_mask: Optional[jnp.ndarray] = None,  # [B,T,N,N]
            a2t_mask: Optional[jnp.ndarray] = None,  # [B,N,T,T]
            a2s_mask: Optional[jnp.ndarray] = None,  # [B,T,N,S]
            a2a_indices: Optional[jnp.ndarray] = None,  # [B,T,N,K]
            a2t_indices: Optional[jnp.ndarray] = None,  # [B,N,T,K]
            a2s_indices: Optional[jnp.ndarray] = None,  # [B,T,N,K]
        ) -> jnp.ndarray:
            if self.legacy_parity_mode:
                h_t = self.a2t_norm(x)
                rel_t = a2t_rel if self.a2t_rel_norm is None or a2t_rel is None else self.a2t_rel_norm(a2t_rel)
                x = x + self._a2t(h_t, rel_t, a2t_mask, a2t_indices)

                h_a = self.a2a_norm(x)
                rel_a = a2a_rel if self.a2a_rel_norm is None or a2a_rel is None else self.a2a_rel_norm(a2a_rel)
                x = x + self._a2a(h_a, rel_a, a2a_mask, a2a_indices)

                h_s = self.a2s_norm(x)
                rel_s = a2s_rel if self.a2s_rel_norm is None or a2s_rel is None else self.a2s_rel_norm(a2s_rel)
                x = x + self._a2s(
                    h_s,
                    scene_tokens,
                    rel_s,
                    a2s_mask,
                    a2s_indices,
                    scene_token_mask=scene_token_mask,
                )

                y = self.ff_in(self.mlp_prenorm(x))
                y = self.ff_out(jax.nn.gelu(y, approximate=True))
                x = x + y
                return x

            h = self.norm1(x)

            a2a_out = self._a2a(h, a2a_rel, a2a_mask, None)
            a2t_out = self._a2t(h, a2t_rel, a2t_mask, None)
            a2s_out = self._a2s(h, scene_tokens, a2s_rel, a2s_mask, None, scene_token_mask=scene_token_mask)

            x = x + (a2a_out + a2t_out + a2s_out) / 3.0

            y = self.norm2(x)
            y = self.ff_out(jax.nn.gelu(self.ff_in(y)))
            x = x + y
            return x


    class NNXBidirectionalMotionTransformer(nnx.Module):
        """Relation-aware NNX BMT decoder.

        This module models motion-token predictions conditioned on:
        - previous token sequence
        - agent descriptors (type/shape/id)
        - continuous motion features (acc,yaw)
        - optional scene tokens
        - optional relation tensors for A2A/A2T/A2S
        """

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = cfg.d_model
            n_tokens = cfg.token_space.n_tokens
            self.legacy_parity_mode = bool(cfg.decoder.enabled) and bool(cfg.decoder.use_legacy_motion_embed)

            # Non-parity/default embedding path.
            if self.legacy_parity_mode:
                self.motion_token_embed = None
                self.continuous_motion_proj = None
            else:
                self.motion_token_embed = nnx.Param(
                    jax.random.normal(rngs.params(), (n_tokens + cfg.n_special_tokens, d_model)) * 0.02
                )
                self.continuous_motion_proj = Linear(2, d_model, rngs=rngs)
            self.agent_type_embed = nnx.Param(
                jax.random.normal(rngs.params(), (cfg.n_agent_types, d_model)) * 0.02
            )
            self.agent_id_embed = nnx.Param(
                jax.random.normal(rngs.params(), (cfg.max_agent_id, d_model)) * 0.02
            )
            self.reverse_indicator_embed = (
                nnx.Param(jax.random.normal(rngs.params(), (2, d_model)) * 0.02)
                if (not self.legacy_parity_mode) or bool(cfg.decoder.use_backward_indicator_embed)
                else None
            )

            if self.legacy_parity_mode:
                self.agent_shape_proj = LegacyMLP(
                    3,
                    (d_model, d_model),
                    rngs=rngs,
                    ret_before_act=True,
                    without_norm=False,
                )
            else:
                self.agent_shape_proj = Linear(3, d_model, rngs=rngs)

            # MidGPT parity decoder-input composition path.
            self.action_embed = nnx.Param(
                jax.random.normal(rngs.params(), (n_tokens + 1, d_model)) * 0.02
            )
            self.special_token_embed = nnx.Param(
                jax.random.normal(rngs.params(), (4, d_model)) * 0.02
            )
            self.motion_embed = FourierEmbeddingNNX(
                input_dim=6,
                hidden_dim=d_model,
                num_freq_bands=64,
                rngs=rngs,
            )
            tok = BidirectionalMotionTokenizer(cfg.token_space)
            motion_feat = tok.action_table_np()  # [V,2] => (acc, yaw)
            motion_dist = np.linalg.norm(motion_feat, axis=-1, keepdims=True).astype(np.float32)
            motion_heading = np.arctan2(motion_feat[:, 1], motion_feat[:, 0]).reshape(-1, 1).astype(np.float32)
            motion_feat = np.concatenate([motion_feat, motion_dist, motion_heading], axis=-1).astype(np.float32)
            motion_feat = np.concatenate([motion_feat, np.zeros((1, 4), dtype=np.float32)], axis=0)
            self.motion_feature_table = jnp.asarray(motion_feat, dtype=jnp.float32)
            rel_factor = max(1, int(cfg.relation.simple_relation_factor))
            decoder_relation_dim = (
                max(1, int(d_model // rel_factor)) if bool(cfg.relation.simple_relation) else int(d_model)
            )
            if self.legacy_parity_mode:
                # Legacy MotionDecoderGPT learns separate Fourier embedders for raw
                # A2A/A2T/A2S relation features before attention. The raw bundle
                # dimensions are 12/12/3, while the decoder attends over the embedded
                # hidden width (128 for the released MidGPT recipe).
                self.relation_embed_a2a = FourierEmbeddingNNX(
                    input_dim=int(cfg.a2a_rel_dim),
                    hidden_dim=decoder_relation_dim,
                    num_freq_bands=64,
                    rngs=rngs,
                )
                self.relation_embed_a2t = FourierEmbeddingNNX(
                    input_dim=int(cfg.a2t_rel_dim),
                    hidden_dim=decoder_relation_dim,
                    num_freq_bands=64,
                    rngs=rngs,
                )
                self.relation_embed_a2s = FourierEmbeddingNNX(
                    input_dim=int(cfg.a2s_rel_dim),
                    hidden_dim=decoder_relation_dim,
                    num_freq_bands=64,
                    rngs=rngs,
                )
                self.decoder_relation_hidden_dim = decoder_relation_dim
            else:
                self.relation_embed_a2a = None
                self.relation_embed_a2t = None
                self.relation_embed_a2s = None
                self.decoder_relation_hidden_dim = None

            self.scene_encoder = NNXSceneTokenEncoder(cfg, rngs=rngs)

            # Optional DAG latent conditioning path.
            self.dag_conditioning_enabled = bool(cfg.dag_conditioning.enabled)
            self.dag_encoder_enabled = bool(cfg.dag_encoder.enabled)
            if self.dag_conditioning_enabled:
                if self.dag_encoder_enabled:
                    self.dag_encoder = NNXDAGGraphEncoder(cfg.dag_encoder, rngs=rngs)
                    dag_latent_in = int(cfg.dag_encoder.d_hidden) * 2
                else:
                    self.dag_encoder = None
                    dag_latent_in = int(cfg.d_model)
                self.dag_latent_proj = Linear(dag_latent_in, d_model, rngs=rngs)
                self.dag_gate_proj = Linear(dag_latent_in, d_model, rngs=rngs)
                self.null_dag_latent = nnx.Param(
                    jax.random.normal(rngs.params(), (dag_latent_in,))
                    * float(cfg.dag_conditioning.null_latent_init_std)
                )
            else:
                self.dag_encoder = None
                self.dag_latent_proj = None
                self.dag_gate_proj = None
                self.null_dag_latent = None

            self.decoder_blocks = tuple(
                RelationAwareDecoderBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)
            )
            self.final_norm = None if self.legacy_parity_mode else RMSNorm(d_model)
            self.prediction_prenorm = LayerNorm(d_model) if self.legacy_parity_mode else None
            if self.legacy_parity_mode:
                self.token_head = LegacyMLP(
                    d_model,
                    (d_model, n_tokens),
                    rngs=rngs,
                    ret_before_act=True,
                    without_norm=False,
                )
            else:
                self.token_head = Linear(d_model, n_tokens, rngs=rngs)

        def _apply_dag_conditioning(
            self,
            h: jnp.ndarray,
            *,
            dag_node_feat: Optional[jnp.ndarray],
            dag_node_mask: Optional[jnp.ndarray],
            dag_edge_src: Optional[jnp.ndarray],
            dag_edge_dst: Optional[jnp.ndarray],
            dag_edge_feat: Optional[jnp.ndarray],
            dag_edge_mask: Optional[jnp.ndarray],
            dag_global_feat: Optional[jnp.ndarray],
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            meta: Dict[str, jnp.ndarray] = {}
            if not self.dag_conditioning_enabled:
                return h, meta

            bsz = int(h.shape[0])
            z_dag: Optional[jnp.ndarray] = None
            has_dag = (
                dag_node_feat is not None
                and dag_node_mask is not None
                and dag_edge_src is not None
                and dag_edge_dst is not None
                and dag_edge_feat is not None
                and dag_edge_mask is not None
            )
            if has_dag and self.dag_encoder_enabled and self.dag_encoder is not None:
                _, z_dag = self.dag_encoder(
                    dag_node_feat=dag_node_feat,
                    dag_node_mask=dag_node_mask,
                    dag_edge_src=dag_edge_src,
                    dag_edge_dst=dag_edge_dst,
                    dag_edge_feat=dag_edge_feat,
                    dag_edge_mask=dag_edge_mask,
                    dag_global_feat=dag_global_feat,
                )
                dag_present = jnp.any(dag_node_mask.astype(bool), axis=1)
                if bool(self.cfg.dag_conditioning.use_null_latent):
                    z0 = self.null_dag_latent.value
                    z_null = jnp.broadcast_to(z0[None, :], (bsz, z0.shape[0]))
                    z_dag = jnp.where(dag_present[:, None], z_dag, z_null)
                meta["dag_source_used"] = dag_present.astype(jnp.float32)
            elif bool(self.cfg.dag_conditioning.use_null_latent):
                z0 = self.null_dag_latent.value
                z_dag = jnp.broadcast_to(z0[None, :], (bsz, z0.shape[0]))
                meta["dag_source_used"] = jnp.zeros((bsz,), dtype=jnp.float32)
            else:
                return h, meta

            if z_dag is None:
                return h, meta

            p_drop = float(np.clip(self.cfg.dag_conditioning.dag_dropout_prob, 0.0, 1.0))
            if p_drop >= 1.0:
                # Stage-A pretraining uses full DAG dropout and should reduce to the
                # exact no-DAG baseline. Returning early here avoids a constant
                # residual from linear-layer biases when z_dag is zeroed.
                zeros = jnp.zeros((bsz,), dtype=jnp.float32)
                meta["dag_latent_norm"] = zeros
                meta["dag_gate_mean"] = zeros
                return h, meta
            if p_drop > 0.0:
                keep = 1.0 - p_drop
                # Stochastic without explicit rng threading: deterministic-ish hash by batch index.
                idx = jnp.arange(z_dag.shape[0], dtype=jnp.float32)
                pseudo = jnp.mod(jnp.sin(idx * 12.9898 + 78.233) * 43758.5453, 1.0)
                keep_mask = (pseudo < keep).astype(z_dag.dtype)[:, None]
                z_dag = z_dag * keep_mask

            if str(self.cfg.dag_conditioning.injection_mode) != "global_gated_residual":
                return h, meta

            bias = self.dag_latent_proj(z_dag)  # [B,D]
            gate = jax.nn.sigmoid(self.dag_gate_proj(z_dag))  # [B,D]
            dag_bias = gate * bias
            h = h + dag_bias[:, None, None, :]

            meta["dag_latent_norm"] = jnp.linalg.norm(z_dag, axis=-1)
            meta["dag_gate_mean"] = jnp.mean(gate, axis=-1)
            return h, meta

        def encode_scene_tokens(
            self,
            *,
            map_feature: jnp.ndarray,  # [B,M,V,27]
            map_feature_valid_mask: jnp.ndarray,  # [B,M,V]
            map_position: jnp.ndarray,  # [B,M,3]
            traffic_light_feature: Optional[jnp.ndarray] = None,  # [B,T,L,7] or [B,L,7]
            traffic_light_valid_mask: Optional[jnp.ndarray] = None,  # [B,T,L] or [B,L]
            traffic_light_position: Optional[jnp.ndarray] = None,  # [B,L,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Public scene-token encoder entrypoint.

            Paper note:
            This path exposes Adv-BMT-style map + traffic-light scene channels so
            A2S attention can condition on explicit scene structure.
            """
            return self.scene_encoder(
                map_feature=map_feature,
                map_feature_valid_mask=map_feature_valid_mask,
                map_position=map_position,
                traffic_light_feature=traffic_light_feature,
                traffic_light_valid_mask=traffic_light_valid_mask,
                traffic_light_position=traffic_light_position,
            )

        def _build_scene_tokens(
            self,
            h: jnp.ndarray,
            scene_tokens: Optional[jnp.ndarray],
            scene_token_mask: Optional[jnp.ndarray],
            *,
            map_feature: Optional[jnp.ndarray],
            map_feature_valid_mask: Optional[jnp.ndarray],
            map_position: Optional[jnp.ndarray],
            traffic_light_feature: Optional[jnp.ndarray],
            traffic_light_valid_mask: Optional[jnp.ndarray],
            traffic_light_position: Optional[jnp.ndarray],
            return_metadata: bool = False,
        ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
            metadata: Dict[str, jnp.ndarray] = {}
            if scene_tokens is not None:
                if scene_token_mask is None:
                    scene_token_mask = jnp.ones(scene_tokens.shape[:2], dtype=bool)
                if return_metadata:
                    return scene_tokens, scene_token_mask, metadata
                return scene_tokens, scene_token_mask

            if (
                map_feature is not None
                and map_feature_valid_mask is not None
                and map_position is not None
            ):
                if return_metadata:
                    encoded_scene, encoded_mask, _, scene_meta = self.scene_encoder(
                        map_feature=map_feature,
                        map_feature_valid_mask=map_feature_valid_mask,
                        map_position=map_position,
                        traffic_light_feature=traffic_light_feature,
                        traffic_light_valid_mask=traffic_light_valid_mask,
                        traffic_light_position=traffic_light_position,
                        return_metadata=True,
                    )
                    metadata.update(scene_meta)
                    return encoded_scene, encoded_mask, metadata
                encoded_scene, encoded_mask, _ = self.encode_scene_tokens(
                    map_feature=map_feature,
                    map_feature_valid_mask=map_feature_valid_mask,
                    map_position=map_position,
                    traffic_light_feature=traffic_light_feature,
                    traffic_light_valid_mask=traffic_light_valid_mask,
                    traffic_light_position=traffic_light_position,
                )
                return encoded_scene, encoded_mask

            # Fallback scene context from hidden states if external scene encoder is absent.
            mean_tok = jnp.mean(h, axis=(1, 2), keepdims=True)  # [B,1,D]
            max_tok = jnp.max(h, axis=(1, 2), keepdims=True)    # [B,1,D]
            fallback = jnp.concatenate([mean_tok, max_tok], axis=1)  # [B,2,D]
            fallback_mask = jnp.ones(fallback.shape[:2], dtype=bool)
            if return_metadata:
                return fallback, fallback_mask, metadata
            return fallback, fallback_mask

        def _sanitize_agent_ids(self, agent_ids: jnp.ndarray) -> jnp.ndarray:
            """Match legacy mode_agent_id behavior for embedding lookups."""
            if bool(self.cfg.decoder.enabled):
                max_id = int(self.cfg.max_agent_id)
                invalid = (agent_ids < 0) | (agent_ids >= max_id)
                return jnp.where(invalid, max_id - 1, agent_ids).astype(jnp.int32)
            return jnp.mod(agent_ids, self.cfg.max_agent_id).astype(jnp.int32)

        def _embed_decoder_relation_tensor(
            self,
            rel: Optional[jnp.ndarray],
            mask: Optional[jnp.ndarray],
            *,
            embedder: Optional[FourierEmbeddingNNX],
            expected_raw_dim: int,
            expected_emb_dim: int,
            label: str,
        ) -> Optional[jnp.ndarray]:
            """Apply legacy decoder relation Fourier embedding to raw relation tensors."""
            if rel is None or embedder is None:
                return rel

            if rel.shape[-1] == expected_emb_dim:
                rel_emb = rel
            else:
                if rel.shape[-1] != expected_raw_dim:
                    raise ValueError(
                        f"{label} last dim mismatch: expected raw dim {expected_raw_dim} or embedded dim "
                        f"{expected_emb_dim}, got {rel.shape[-1]}"
                    )
                flat = rel.reshape(-1, rel.shape[-1])
                rel_emb = embedder(flat).reshape(*rel.shape[:-1], expected_emb_dim)

            # Legacy unwrap/wrap logic leaves invalid edges at zero. We mirror that
            # explicitly so masked relation slots cannot leak arbitrary values.
            if mask is not None:
                rel_emb = jnp.where(mask[..., None].astype(bool), rel_emb, jnp.zeros_like(rel_emb))
            return rel_emb

        def _compose_decoder_tokens_parity(
            self,
            *,
            prev_token_ids: jnp.ndarray,  # [B,T,N]
            input_action_valid_mask: Optional[jnp.ndarray],  # [B,T,N]
            modeled_agent_delta: Optional[jnp.ndarray],  # [B,T,N,2]
            agent_type_ids: jnp.ndarray,  # [B,N]
            agent_shape: jnp.ndarray,  # [B,N,3]
            agent_ids: jnp.ndarray,  # [B,N]
            reverse_indicator: jnp.ndarray,  # [B]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Legacy-style decoder input assembly from action/special/motion features."""
            bsz, t_steps, n_agents = prev_token_ids.shape
            d_model = int(self.cfg.d_model)
            n_tokens = int(self.cfg.token_space.n_tokens)

            if input_action_valid_mask is None:
                valid_mask = jnp.ones_like(prev_token_ids, dtype=bool)
            else:
                valid_mask = jnp.asarray(input_action_valid_mask, dtype=bool)

            if modeled_agent_delta is None:
                modeled_delta = jnp.zeros((bsz, t_steps, n_agents, 2), dtype=jnp.float32)
            else:
                modeled_delta = jnp.asarray(modeled_agent_delta, dtype=jnp.float32)

            is_action = prev_token_ids < n_tokens
            is_start = prev_token_ids == n_tokens
            is_end = prev_token_ids == (n_tokens + 1)
            is_pad = prev_token_ids == (n_tokens + 2)
            is_mask = prev_token_ids == (n_tokens + 3)

            special_cls = jnp.zeros_like(prev_token_ids, dtype=jnp.int32)  # normal
            special_cls = jnp.where(is_start, 1, special_cls)
            special_cls = jnp.where(is_end, 2, special_cls)
            special_cls = jnp.where(is_pad | is_mask, 3, special_cls)

            action_idx = jnp.where(is_action, prev_token_ids, n_tokens).astype(jnp.int32)
            action_emb = self.action_embed.value[action_idx]  # [B,T,N,D]
            special_emb = self.special_token_embed.value[special_cls]  # [B,T,N,D]

            type_ids = jnp.mod(agent_type_ids, self.cfg.n_agent_types)
            type_emb = self.agent_type_embed.value[type_ids][:, None, :, :]
            type_emb = jnp.broadcast_to(type_emb, (bsz, t_steps, n_agents, d_model))

            safe_ids = self._sanitize_agent_ids(agent_ids)
            id_emb = self.agent_id_embed.value[safe_ids][:, None, :, :]
            id_emb = jnp.broadcast_to(id_emb, (bsz, t_steps, n_agents, d_model))

            shape_emb = self.agent_shape_proj(agent_shape.reshape(-1, 3)).reshape(
                bsz, 1, n_agents, d_model
            )
            shape_emb = jnp.broadcast_to(shape_emb, (bsz, t_steps, n_agents, d_model))

            motion4 = jnp.take(self.motion_feature_table, action_idx, axis=0)  # [B,T,N,4]
            motion6 = jnp.concatenate([motion4, modeled_delta], axis=-1)

            categorical = [special_emb, id_emb, type_emb, shape_emb, action_emb]
            if bool(self.cfg.decoder.use_backward_indicator_embed):
                if self.reverse_indicator_embed is None:
                    raise RuntimeError("reverse_indicator_embed is required when use_backward_indicator_embed=True")
                rev_ids = jnp.clip(reverse_indicator, 0, 1)
                rev_emb = self.reverse_indicator_embed.value[rev_ids][:, None, None, :]
                rev_emb = jnp.broadcast_to(rev_emb, (bsz, t_steps, n_agents, d_model))
                categorical.append(rev_emb)

            token = self.motion_embed(
                continuous_inputs=motion6,
                categorical_embs=categorical,
            )

            token = jnp.where(valid_mask[..., None], token, jnp.zeros_like(token))
            return token, valid_mask

        def _compose_decoder_tokens_simple(
            self,
            *,
            prev_token_ids: jnp.ndarray,
            agent_type_ids: jnp.ndarray,
            agent_shape: jnp.ndarray,
            agent_ids: jnp.ndarray,
            continuous_motion: jnp.ndarray,
            reverse_indicator: jnp.ndarray,
        ) -> jnp.ndarray:
            if self.motion_token_embed is None or self.continuous_motion_proj is None or self.reverse_indicator_embed is None:
                raise RuntimeError("Simple decoder token path requires simple-path embeddings to be initialized")
            token_e = self.motion_token_embed.value[prev_token_ids]  # [B,T,N,D]

            type_ids = jnp.mod(agent_type_ids, self.cfg.n_agent_types)
            type_e = self.agent_type_embed.value[type_ids][:, None, :, :]  # [B,1,N,D]

            id_ids = jnp.mod(agent_ids, self.cfg.max_agent_id)
            id_e = self.agent_id_embed.value[id_ids][:, None, :, :]  # [B,1,N,D]

            shp = self.agent_shape_proj(agent_shape.reshape(-1, 3)).reshape(
                agent_shape.shape[0], 1, agent_shape.shape[1], self.cfg.d_model
            )
            mot = self.continuous_motion_proj(continuous_motion.reshape(-1, 2)).reshape(
                continuous_motion.shape[0],
                continuous_motion.shape[1],
                continuous_motion.shape[2],
                self.cfg.d_model,
            )

            rev_ids = jnp.clip(reverse_indicator, 0, 1)
            rev_e = self.reverse_indicator_embed.value[rev_ids][:, None, None, :]
            return token_e + type_e + id_e + shp + mot + rev_e

        def __call__(
            self,
            *,
            prev_token_ids: jnp.ndarray,  # [B,T,N]
            agent_type_ids: jnp.ndarray,  # [B,N]
            agent_shape: jnp.ndarray,  # [B,N,3]
            agent_ids: jnp.ndarray,  # [B,N]
            continuous_motion: jnp.ndarray,  # [B,T,N,2] (acc,yaw)
            reverse_indicator: jnp.ndarray,  # [B], 0=forward,1=reverse
            input_action_valid_mask: Optional[jnp.ndarray] = None,  # [B,T,N]
            modeled_agent_delta: Optional[jnp.ndarray] = None,  # [B,T,N,2]
            scene_tokens: Optional[jnp.ndarray] = None,  # [B,S,D]
            scene_token_mask: Optional[jnp.ndarray] = None,  # [B,S]
            scene_map_feature: Optional[jnp.ndarray] = None,  # [B,M,V,27]
            scene_map_valid_mask: Optional[jnp.ndarray] = None,  # [B,M,V]
            scene_map_position: Optional[jnp.ndarray] = None,  # [B,M,3]
            scene_tl_feature: Optional[jnp.ndarray] = None,  # [B,T,L,7] or [B,L,7]
            scene_tl_valid_mask: Optional[jnp.ndarray] = None,  # [B,T,L] or [B,L]
            scene_tl_position: Optional[jnp.ndarray] = None,  # [B,L,3]
            a2a_rel: Optional[jnp.ndarray] = None,  # [B,T,N,N,Ra2a]
            a2t_rel: Optional[jnp.ndarray] = None,  # [B,N,T,T,Ra2t]
            a2s_rel: Optional[jnp.ndarray] = None,  # [B,T,N,S,Ra2s]
            a2a_mask: Optional[jnp.ndarray] = None,  # [B,T,N,N]
            a2t_mask: Optional[jnp.ndarray] = None,  # [B,N,T,T]
            a2s_mask: Optional[jnp.ndarray] = None,  # [B,T,N,S]
            a2a_indices: Optional[jnp.ndarray] = None,  # [B,T,N,K]
            a2t_indices: Optional[jnp.ndarray] = None,  # [B,N,T,K]
            a2s_indices: Optional[jnp.ndarray] = None,  # [B,T,N,K]
            dag_node_feat: Optional[jnp.ndarray] = None,  # [B,G,Fn]
            dag_node_mask: Optional[jnp.ndarray] = None,  # [B,G]
            dag_edge_src: Optional[jnp.ndarray] = None,  # [B,E]
            dag_edge_dst: Optional[jnp.ndarray] = None,  # [B,E]
            dag_edge_feat: Optional[jnp.ndarray] = None,  # [B,E,Fe]
            dag_edge_mask: Optional[jnp.ndarray] = None,  # [B,E]
            dag_global_feat: Optional[jnp.ndarray] = None,  # [B,Fg]
            return_metadata: bool = False,
        ) -> jnp.ndarray | Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            if bool(self.cfg.decoder.enabled) and bool(self.cfg.decoder.use_legacy_motion_embed):
                h, token_valid_mask = self._compose_decoder_tokens_parity(
                    prev_token_ids=prev_token_ids,
                    input_action_valid_mask=input_action_valid_mask,
                    modeled_agent_delta=modeled_agent_delta,
                    agent_type_ids=agent_type_ids,
                    agent_shape=agent_shape,
                    agent_ids=agent_ids,
                    reverse_indicator=reverse_indicator,
                )
            else:
                h = self._compose_decoder_tokens_simple(
                    prev_token_ids=prev_token_ids,
                    agent_type_ids=agent_type_ids,
                    agent_shape=agent_shape,
                    agent_ids=agent_ids,
                    continuous_motion=continuous_motion,
                    reverse_indicator=reverse_indicator,
                )
                token_valid_mask = jnp.ones(prev_token_ids.shape, dtype=bool)
            scene_meta: Dict[str, jnp.ndarray] = {}
            if return_metadata:
                scene, scene_mask, scene_meta = self._build_scene_tokens(
                    h,
                    scene_tokens,
                    scene_token_mask,
                    map_feature=scene_map_feature,
                    map_feature_valid_mask=scene_map_valid_mask,
                    map_position=scene_map_position,
                    traffic_light_feature=scene_tl_feature,
                    traffic_light_valid_mask=scene_tl_valid_mask,
                    traffic_light_position=scene_tl_position,
                    return_metadata=True,
                )
            else:
                scene, scene_mask = self._build_scene_tokens(
                    h,
                    scene_tokens,
                    scene_token_mask,
                    map_feature=scene_map_feature,
                    map_feature_valid_mask=scene_map_valid_mask,
                    map_position=scene_map_position,
                    traffic_light_feature=scene_tl_feature,
                    traffic_light_valid_mask=scene_tl_valid_mask,
                    traffic_light_position=scene_tl_position,
                    return_metadata=False,
                )

            dag_meta: Dict[str, jnp.ndarray] = {}
            h, dag_meta = self._apply_dag_conditioning(
                h,
                dag_node_feat=dag_node_feat,
                dag_node_mask=dag_node_mask,
                dag_edge_src=dag_edge_src,
                dag_edge_dst=dag_edge_dst,
                dag_edge_feat=dag_edge_feat,
                dag_edge_mask=dag_edge_mask,
                dag_global_feat=dag_global_feat,
            )

            if self.legacy_parity_mode:
                a2a_rel = self._embed_decoder_relation_tensor(
                    a2a_rel,
                    a2a_mask,
                    embedder=self.relation_embed_a2a,
                    expected_raw_dim=int(self.cfg.a2a_rel_dim),
                    expected_emb_dim=int(self.decoder_relation_hidden_dim),
                    label="a2a_rel",
                )
                a2t_rel = self._embed_decoder_relation_tensor(
                    a2t_rel,
                    a2t_mask,
                    embedder=self.relation_embed_a2t,
                    expected_raw_dim=int(self.cfg.a2t_rel_dim),
                    expected_emb_dim=int(self.decoder_relation_hidden_dim),
                    label="a2t_rel",
                )
                a2s_rel = self._embed_decoder_relation_tensor(
                    a2s_rel,
                    a2s_mask,
                    embedder=self.relation_embed_a2s,
                    expected_raw_dim=int(self.cfg.a2s_rel_dim),
                    expected_emb_dim=int(self.decoder_relation_hidden_dim),
                    label="a2s_rel",
                )

            for block in self.decoder_blocks:
                h = block(
                    h,
                    scene_tokens=scene,
                    scene_token_mask=scene_mask,
                    a2a_rel=a2a_rel,
                    a2t_rel=a2t_rel,
                    a2s_rel=a2s_rel,
                    a2a_mask=a2a_mask,
                    a2t_mask=a2t_mask,
                    a2s_mask=a2s_mask,
                    a2a_indices=a2a_indices,
                    a2t_indices=a2t_indices,
                    a2s_indices=a2s_indices,
                )

            h = jnp.where(token_valid_mask[..., None], h, jnp.zeros_like(h))
            if self.legacy_parity_mode:
                if self.prediction_prenorm is None:
                    raise RuntimeError("prediction_prenorm is required in legacy parity mode")
                h = self.prediction_prenorm(h)
            else:
                h = self.final_norm(h)
            logits = self.token_head(h)  # [B,T,N,|A|]
            if return_metadata:
                md: Dict[str, Dict[str, jnp.ndarray]] = {"scene": scene_meta}
                if dag_meta:
                    md["dag"] = dag_meta
                return logits, md
            return logits


else:  # HAS_NNX == False
    # Keep symbols available for import paths.
    Linear = None
    RMSNorm = None
    SceneRelationSelfAttentionBlock = None
    NNXSceneTokenEncoder = None
    RelationBiasProjector = None
    MultiHeadAttention = None
    RelationAwareDecoderBlock = None
    NNXBidirectionalMotionTransformer = None


def cross_entropy_token_loss(logits: Any, targets: Any, mask: Optional[Any] = None) -> Any:
    """Cross-entropy token loss matching Adv-BMT training objective."""
    if jnp is None:
        raise RuntimeError("jax is required for cross_entropy_token_loss")

    if mask is None:
        mask = jnp.ones_like(targets, dtype=jnp.float32)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, axis=-1), axis=-1).squeeze(-1)
    num = -jnp.sum(mask * picked)
    den = jnp.maximum(1.0, jnp.sum(mask))
    return num / den


def masked_token_accuracy(logits: Any, targets: Any, mask: Optional[Any] = None) -> Any:
    """Masked token accuracy helper."""
    if jnp is None:
        raise RuntimeError("jax is required for masked_token_accuracy")

    if mask is None:
        mask = jnp.ones_like(targets, dtype=jnp.float32)

    pred = jnp.argmax(logits, axis=-1)
    correct = (pred == targets).astype(jnp.float32)
    num = jnp.sum(mask * correct)
    den = jnp.maximum(1.0, jnp.sum(mask))
    return num / den


def sample_motion_tokens(
    logits: Any,
    key: Any,
    *,
    sampling_method: str = "topp",
    temperature: float = 1.0,
    topp: float = 0.95,
    topk: int = 5,
) -> Any:
    """Sample token ids from logits [B,N,V]."""
    if jnp is None:
        raise RuntimeError("jax is required for sample_motion_tokens")

    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    bsz, n_agents, vocab = logits.shape
    flat = (logits / temperature).reshape(bsz * n_agents, vocab)

    method = sampling_method.lower()
    if method == "argmax":
        sampled = jnp.argmax(flat, axis=-1)

    elif method == "softmax":
        sampled = jax.random.categorical(key, flat, axis=-1)

    elif method == "topk":
        k = max(1, min(int(topk), vocab))
        # Highest-k columns.
        top_idx = jnp.argsort(flat, axis=-1)[:, -k:]
        top_logits = jnp.take_along_axis(flat, top_idx, axis=-1)
        sel = jax.random.categorical(key, top_logits, axis=-1)
        sampled = jnp.take_along_axis(top_idx, sel[:, None], axis=-1).squeeze(-1)

    elif method == "topp":
        p = float(np.clip(topp, 1e-6, 1.0))
        sort_idx = jnp.argsort(flat, axis=-1)[:, ::-1]
        sort_logits = jnp.take_along_axis(flat, sort_idx, axis=-1)
        sort_probs = jax.nn.softmax(sort_logits, axis=-1)
        cum_probs = jnp.cumsum(sort_probs, axis=-1)

        keep = cum_probs <= p
        keep = keep.at[:, 0].set(True)

        filtered_logits = jnp.where(keep, sort_logits, jnp.full_like(sort_logits, -1e9))
        sampled_sorted = jax.random.categorical(key, filtered_logits, axis=-1)
        sampled = jnp.take_along_axis(sort_idx, sampled_sorted[:, None], axis=-1).squeeze(-1)

    else:
        raise ValueError(f"Unknown sampling_method: {sampling_method}")

    return sampled.reshape(bsz, n_agents)


def autoregressive_token_rollout(
    model: Any,
    *,
    start_token_ids: Any,  # [B,N]
    agent_type_ids: Any,  # [B,N]
    agent_shape: Any,  # [B,N,3]
    agent_ids: Any,  # [B,N]
    reverse_indicator: Any,  # [B]
    horizon_steps: int,
    tokenizer: Optional[BidirectionalMotionTokenizer] = None,
    input_action_valid_mask: Optional[Any] = None,  # [B,T0,N]
    modeled_agent_delta: Optional[Any] = None,  # [B,T0,N,2]
    scene_tokens: Optional[Any] = None,
    scene_token_mask: Optional[Any] = None,
    scene_map_feature: Optional[Any] = None,
    scene_map_valid_mask: Optional[Any] = None,
    scene_map_position: Optional[Any] = None,
    scene_tl_feature: Optional[Any] = None,
    scene_tl_valid_mask: Optional[Any] = None,
    scene_tl_position: Optional[Any] = None,
    a2a_rel: Optional[Any] = None,
    a2t_rel: Optional[Any] = None,
    a2s_rel: Optional[Any] = None,
    a2a_mask: Optional[Any] = None,
    a2t_mask: Optional[Any] = None,
    a2s_mask: Optional[Any] = None,
    a2a_indices: Optional[Any] = None,
    a2t_indices: Optional[Any] = None,
    a2s_indices: Optional[Any] = None,
    sampling_method: str = "topp",
    temperature: float = 1.0,
    topp: float = 0.95,
    topk: int = 5,
    key: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Autoregressive token rollout using model next-token logits.

    Returns:
        predicted_tokens: [B,H,N]
        predicted_continuous_motion: [B,H,N,2]
    """
    if jnp is None:
        raise RuntimeError("jax is required for autoregressive_token_rollout")

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    if key is None:
        key = jax.random.PRNGKey(0)

    start_token_ids = jnp.asarray(start_token_ids)
    agent_type_ids = jnp.asarray(agent_type_ids)
    agent_shape = jnp.asarray(agent_shape)
    agent_ids = jnp.asarray(agent_ids)
    reverse_indicator = jnp.asarray(reverse_indicator)

    # Seed sequences with one token frame.
    token_seq = start_token_ids[:, None, :]  # [B,1,N]
    if input_action_valid_mask is None:
        valid_seq = jnp.ones_like(token_seq, dtype=bool)
    else:
        valid_seq = jnp.asarray(input_action_valid_mask, dtype=bool)
        if valid_seq.shape[1] != token_seq.shape[1]:
            valid_seq = valid_seq[:, : token_seq.shape[1], :]

    if tokenizer is not None:
        action_table = jnp.asarray(tokenizer.action_table_np())
        motion_seq = jnp.take(action_table, token_seq, axis=0)  # [B,1,N,2]
    else:
        bsz, _, n_agents = token_seq.shape
        motion_seq = jnp.zeros((bsz, 1, n_agents, 2), dtype=jnp.float32)
    if modeled_agent_delta is None:
        delta_seq = jnp.zeros_like(motion_seq)
    else:
        delta_seq = jnp.asarray(modeled_agent_delta, dtype=jnp.float32)
        if delta_seq.shape[1] != token_seq.shape[1]:
            delta_seq = delta_seq[:, : token_seq.shape[1], :, :]

    for _ in range(horizon_steps):
        cur_t = int(token_seq.shape[1])
        model_kwargs = {
            "prev_token_ids": token_seq,
            "agent_type_ids": agent_type_ids,
            "agent_shape": agent_shape,
            "agent_ids": agent_ids,
            "continuous_motion": motion_seq,
            "reverse_indicator": reverse_indicator,
            "input_action_valid_mask": valid_seq,
            "modeled_agent_delta": delta_seq,
            "scene_tokens": scene_tokens,
        }
        if a2a_rel is not None:
            model_kwargs["a2a_rel"] = a2a_rel[:, :cur_t, ...]
        if a2t_rel is not None:
            model_kwargs["a2t_rel"] = a2t_rel[:, :, :cur_t, :cur_t, ...]
        if a2s_rel is not None:
            model_kwargs["a2s_rel"] = a2s_rel[:, :cur_t, ...]
        if a2a_mask is not None:
            model_kwargs["a2a_mask"] = a2a_mask[:, :cur_t, ...]
        if a2t_mask is not None:
            model_kwargs["a2t_mask"] = a2t_mask[:, :, :cur_t, :cur_t]
        if a2s_mask is not None:
            model_kwargs["a2s_mask"] = a2s_mask[:, :cur_t, ...]
        if a2a_indices is not None:
            model_kwargs["a2a_indices"] = a2a_indices[:, :cur_t, ...]
        if a2t_indices is not None:
            model_kwargs["a2t_indices"] = a2t_indices[:, :, :cur_t, ...]
        if a2s_indices is not None:
            model_kwargs["a2s_indices"] = a2s_indices[:, :cur_t, ...]
        if scene_token_mask is not None:
            model_kwargs["scene_token_mask"] = scene_token_mask
        if scene_map_feature is not None:
            model_kwargs["scene_map_feature"] = scene_map_feature
        if scene_map_valid_mask is not None:
            model_kwargs["scene_map_valid_mask"] = scene_map_valid_mask
        if scene_map_position is not None:
            model_kwargs["scene_map_position"] = scene_map_position
        if scene_tl_feature is not None:
            model_kwargs["scene_tl_feature"] = scene_tl_feature
        if scene_tl_valid_mask is not None:
            model_kwargs["scene_tl_valid_mask"] = scene_tl_valid_mask
        if scene_tl_position is not None:
            model_kwargs["scene_tl_position"] = scene_tl_position

        logits = model(**model_kwargs)

        step_logits = logits[:, -1, :, :]  # [B,N,V]
        key, sub = jax.random.split(key)
        next_tok = sample_motion_tokens(
            step_logits,
            sub,
            sampling_method=sampling_method,
            temperature=temperature,
            topp=topp,
            topk=topk,
        )

        token_seq = jnp.concatenate([token_seq, next_tok[:, None, :]], axis=1)
        valid_seq = jnp.concatenate([valid_seq, jnp.ones_like(next_tok[:, None, :], dtype=bool)], axis=1)

        if tokenizer is not None:
            next_motion = jnp.take(action_table, next_tok, axis=0)  # [B,N,2]
        else:
            bsz, n_agents = next_tok.shape
            next_motion = jnp.zeros((bsz, n_agents, 2), dtype=jnp.float32)

        motion_seq = jnp.concatenate([motion_seq, next_motion[:, None, :, :]], axis=1)
        delta_seq = jnp.concatenate([delta_seq, next_motion[:, None, :, :]], axis=1)

    # Remove seed frame.
    return token_seq[:, 1:, :], motion_seq[:, 1:, :, :]
