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
from typing import Any, Dict, Optional, Tuple

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
        def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs, scale: float = 0.02):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (d_in, d_out)) * scale)
            self.b = nnx.Param(jnp.zeros((d_out,), dtype=jnp.float32))

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return jnp.einsum("...d,df->...f", x, self.w.value) + self.b.value


    class RMSNorm(nnx.Module):
        def __init__(self, d_model: int, *, eps: float = 1e-6):
            self.scale = nnx.Param(jnp.ones((d_model,), dtype=jnp.float32))
            self.eps = eps

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
            return (x / rms) * self.scale.value


    class SceneRelationSelfAttentionBlock(nnx.Module):
        """Scene self-attention block with legacy-style simple relation terms."""

        def __init__(self, cfg: NNXBMTConfig, rel_dim: int, *, rngs: nnx.Rngs):
            self.cfg = cfg
            self.d_model = int(cfg.d_model)
            self.n_heads = int(cfg.n_heads)
            if self.d_model % self.n_heads != 0:
                raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
            self.head_dim = self.d_model // self.n_heads

            self.token_norm = RMSNorm(self.d_model)
            self.rel_norm = None if cfg.relation.remove_rel_norm else RMSNorm(rel_dim)

            self.q_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.k_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.v_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.q_rel_proj = Linear(self.d_model, self.d_model, rngs=rngs)
            self.rel_k_proj = Linear(rel_dim, self.d_model, rngs=rngs)
            self.rel_v_proj = Linear(rel_dim, self.d_model, rngs=rngs)
            self.o_proj = Linear(self.d_model, self.d_model, rngs=rngs)

            ff_hidden = self.d_model * int(cfg.ff_mult)
            self.ff_norm = RMSNorm(self.d_model)
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

            ff = self.ff_out(jax.nn.gelu(self.ff_in(self.ff_norm(scene_tokens))))
            scene_tokens = scene_tokens + ff
            return scene_tokens


    class NNXSceneTokenEncoder(nnx.Module):
        """Encodes map + traffic-light scene tensors into scene tokens."""

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = int(cfg.d_model)
            self.map_vector_proj = Linear(cfg.scene_encoder.map_feature_dim, d_model, rngs=rngs)
            self.map_token_proj = Linear(d_model, d_model, rngs=rngs)
            self.traffic_light_proj = Linear(cfg.scene_encoder.traffic_light_feature_dim, d_model, rngs=rngs)
            self.position_proj = Linear(3, d_model, rngs=rngs)
            self.out_norm = RMSNorm(d_model)

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

            map_token_mask = jnp.any(map_feature_valid_mask, axis=-1)
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
                    # MidGPT parity: represent one static state per light using majority vote.
                    light_mask = jnp.any(traffic_light_valid_mask, axis=1)
                    valid = traffic_light_valid_mask.astype(jnp.float32)
                    state_scores = jnp.sum(traffic_light_feature[..., 3:7] * valid[..., None], axis=1)
                    cls = jnp.argmax(state_scores, axis=-1)
                    onehot = jax.nn.one_hot(cls, 4, dtype=jnp.float32)
                    light_feat = jnp.concatenate([traffic_light_position, onehot], axis=-1)
                else:
                    valid = traffic_light_valid_mask.astype(jnp.float32)
                    num = jnp.sum(traffic_light_feature * valid[..., None], axis=1)
                    den = jnp.maximum(1.0, jnp.sum(valid[..., None], axis=1))
                    light_feat = num / den
                    light_mask = jnp.any(traffic_light_valid_mask, axis=1)
            else:
                # [B,L,7]
                light_feat = traffic_light_feature
                light_mask = traffic_light_valid_mask

            bsz, n_light, feat_dim = light_feat.shape
            light_tokens = self.traffic_light_proj(light_feat.reshape(-1, feat_dim)).reshape(
                bsz, n_light, self.cfg.d_model
            )
            pos_e = self.position_proj(traffic_light_position.reshape(-1, 3)).reshape(
                bsz, n_light, self.cfg.d_model
            )
            light_tokens = light_tokens + pos_e
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

        def __call__(
            self,
            query: jnp.ndarray,
            key_value: jnp.ndarray,
            *,
            mask: Optional[jnp.ndarray] = None,   # [B,Lq,Lk] bool
            rel_feat: Optional[jnp.ndarray] = None,  # [B,Lq,Lk,R]
            rel_mask: Optional[jnp.ndarray] = None,  # [B,Lq,Lk] bool
        ) -> jnp.ndarray:
            bsz, q_len, _ = query.shape
            _, k_len, _ = key_value.shape

            q = self.q_proj(query).reshape(bsz, q_len, self.n_heads, self.head_dim)
            k = self.k_proj(key_value).reshape(bsz, k_len, self.n_heads, self.head_dim)
            v = self.v_proj(key_value).reshape(bsz, k_len, self.n_heads, self.head_dim)

            q = jnp.transpose(q, (0, 2, 1, 3))
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            scale = 1.0 / np.sqrt(float(self.head_dim))
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

            rel_v = None
            if rel_feat is not None and self.relation_dim is not None:
                rel_in = rel_feat if self.rel_norm is None else self.rel_norm(rel_feat)
                q_rel = self.q_rel_proj(query).reshape(bsz, q_len, self.n_heads, self.head_dim)
                q_rel = jnp.transpose(q_rel, (0, 2, 1, 3))  # [B,H,Q,D]

                rel_k = self.rel_k_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                    bsz, q_len, k_len, self.n_heads, self.head_dim
                )
                rel_k = jnp.transpose(rel_k, (0, 3, 1, 2, 4))  # [B,H,Q,K,D]
                score_rel = jnp.sum(q_rel[:, :, :, None, :] * rel_k, axis=-1) * scale
                scores = scores + score_rel

                if self.add_relation_to_v:
                    rel_v = self.rel_v_proj(rel_in.reshape(-1, rel_in.shape[-1])).reshape(
                        bsz, q_len, k_len, self.n_heads, self.head_dim
                    )
                    rel_v = jnp.transpose(rel_v, (0, 3, 1, 2, 4))
                else:
                    rel_v = rel_k

            combined_mask = mask
            if rel_mask is not None:
                rel_mask = rel_mask.astype(bool)
                combined_mask = rel_mask if combined_mask is None else jnp.logical_and(combined_mask, rel_mask)
            if combined_mask is not None:
                # mask True means keep; False means masked out.
                scores = jnp.where(combined_mask[:, None, :, :], scores, jnp.full_like(scores, -1e9))

            attn = jax.nn.softmax(scores, axis=-1)

            if rel_v is None:
                out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
            else:
                value = v[:, :, None, :, :] + rel_v
                out = jnp.sum(attn[..., None] * value, axis=-2)

            out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bsz, q_len, self.d_model)
            return self.o_proj(out)


    class RelationAwareDecoderBlock(nnx.Module):
        """One decoder block with explicit A2A, A2T, A2S attention."""

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = cfg.d_model

            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)

            self.a2a_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=int(cfg.a2a_rel_dim),
                add_relation_to_v=bool(cfg.relation.add_relation_to_v),
                remove_rel_norm=bool(cfg.relation.remove_rel_norm),
                rngs=rngs,
            )
            self.a2t_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=int(cfg.a2t_rel_dim),
                add_relation_to_v=bool(cfg.relation.add_relation_to_v),
                remove_rel_norm=bool(cfg.relation.remove_rel_norm),
                rngs=rngs,
            )
            self.a2s_attn = MultiHeadAttention(
                d_model,
                cfg.n_heads,
                relation_dim=int(cfg.a2s_rel_dim),
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
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*T,N,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz * t_steps, n_agents, d_model)
            rel_bt = None
            mask_bt = None
            if rel is not None:
                # rel: [B,T,N,N,R] => [B*T,N,N,R]
                rel_bt = rel.reshape(bsz * t_steps, n_agents, n_agents, rel.shape[-1])
            if rel_mask is not None:
                mask_bt = rel_mask.reshape(bsz * t_steps, n_agents, n_agents).astype(bool)
            out = self.a2a_attn(q, q, rel_feat=rel_bt, rel_mask=mask_bt)
            return out.reshape(bsz, t_steps, n_agents, d_model)

        def _a2t(
            self,
            h: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*N,T,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = jnp.transpose(h, (0, 2, 1, 3)).reshape(bsz * n_agents, t_steps, d_model)
            rel_bn = None
            if rel is not None:
                # rel: [B,N,T,T,R] => [B*N,T,T,R]
                rel_bn = rel.reshape(bsz * n_agents, t_steps, t_steps, rel.shape[-1])

            # Temporal causality mask:
            # token at step t can only attend to [0..t]. This prevents future
            # leakage during teacher-forcing and aligns training with rollout.
            causal = jnp.tril(jnp.ones((t_steps, t_steps), dtype=bool))
            causal = jnp.broadcast_to(causal[None, :, :], (bsz * n_agents, t_steps, t_steps))
            rel_mask_bn = None
            if rel_mask is not None:
                rel_mask_bn = rel_mask.reshape(bsz * n_agents, t_steps, t_steps).astype(bool)

            out = self.a2t_attn(q, q, mask=causal, rel_feat=rel_bn, rel_mask=rel_mask_bn)
            out = out.reshape(bsz, n_agents, t_steps, d_model)
            return jnp.transpose(out, (0, 2, 1, 3))

        def _a2s(
            self,
            h: jnp.ndarray,
            scene_tokens: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            rel_mask: Optional[jnp.ndarray],
            scene_token_mask: Optional[jnp.ndarray] = None,  # [B,S]
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B,T*N,D], scene_tokens: [B,S,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz, t_steps * n_agents, d_model)

            rel_qs = None
            if rel is not None:
                # rel: [B,T,N,S,R] => [B,T*N,S,R]
                rel_qs = rel.reshape(bsz, t_steps * n_agents, scene_tokens.shape[1], rel.shape[-1])

            attn_mask = None
            if scene_token_mask is not None:
                attn_mask = jnp.broadcast_to(
                    scene_token_mask[:, None, :], (bsz, t_steps * n_agents, scene_tokens.shape[1])
                )
            rel_mask_qs = None
            if rel_mask is not None:
                rel_mask_qs = rel_mask.reshape(bsz, t_steps * n_agents, scene_tokens.shape[1]).astype(bool)

            out = self.a2s_attn(q, scene_tokens, mask=attn_mask, rel_feat=rel_qs, rel_mask=rel_mask_qs)
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
        ) -> jnp.ndarray:
            h = self.norm1(x)

            a2a_out = self._a2a(h, a2a_rel, a2a_mask)
            a2t_out = self._a2t(h, a2t_rel, a2t_mask)
            a2s_out = self._a2s(h, scene_tokens, a2s_rel, a2s_mask, scene_token_mask=scene_token_mask)

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

            # Non-parity/default embedding path.
            self.motion_token_embed = nnx.Param(
                jax.random.normal(rngs.params(), (n_tokens + cfg.n_special_tokens, d_model)) * 0.02
            )
            self.agent_type_embed = nnx.Param(
                jax.random.normal(rngs.params(), (cfg.n_agent_types, d_model)) * 0.02
            )
            self.agent_id_embed = nnx.Param(
                jax.random.normal(rngs.params(), (cfg.max_agent_id, d_model)) * 0.02
            )
            self.reverse_indicator_embed = nnx.Param(
                jax.random.normal(rngs.params(), (2, d_model)) * 0.02
            )

            self.agent_shape_proj = Linear(3, d_model, rngs=rngs)
            self.continuous_motion_proj = Linear(2, d_model, rngs=rngs)

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
            self.step_embed = nnx.Param(
                jax.random.normal(rngs.params(), (512, d_model)) * 0.02
            )
            tok = BidirectionalMotionTokenizer(cfg.token_space)
            motion_feat = tok.action_table_np()  # [V,2] => (acc, yaw)
            motion_dist = np.linalg.norm(motion_feat, axis=-1, keepdims=True).astype(np.float32)
            motion_heading = np.arctan2(motion_feat[:, 1], motion_feat[:, 0]).reshape(-1, 1).astype(np.float32)
            motion_feat = np.concatenate([motion_feat, motion_dist, motion_heading], axis=-1).astype(np.float32)
            motion_feat = np.concatenate([motion_feat, np.zeros((1, 4), dtype=np.float32)], axis=0)
            self.motion_feature_table = jnp.asarray(motion_feat, dtype=jnp.float32)

            self.scene_encoder = NNXSceneTokenEncoder(cfg, rngs=rngs)

            self.decoder_blocks = tuple(
                RelationAwareDecoderBlock(cfg, rngs=rngs) for _ in range(cfg.n_layers)
            )
            self.final_norm = RMSNorm(d_model)
            self.token_head = Linear(d_model, n_tokens, rngs=rngs)

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
                rev_ids = jnp.clip(reverse_indicator, 0, 1)
                rev_emb = self.reverse_indicator_embed.value[rev_ids][:, None, None, :]
                rev_emb = jnp.broadcast_to(rev_emb, (bsz, t_steps, n_agents, d_model))
                categorical.append(rev_emb)

            token = self.motion_embed(
                continuous_inputs=motion6,
                categorical_embs=categorical,
            )

            if bool(self.cfg.decoder.add_pe_for_token):
                step_ids = jnp.arange(t_steps, dtype=jnp.int32)
                step_ids = jnp.clip(step_ids, 0, self.step_embed.value.shape[0] - 1)
                step_emb = self.step_embed.value[step_ids][None, :, None, :]
                token = token + step_emb

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
                )

            h = jnp.where(token_valid_mask[..., None], h, jnp.zeros_like(h))
            h = self.final_norm(h)
            logits = self.token_head(h)  # [B,T,N,|A|]
            if return_metadata:
                return logits, {"scene": scene_meta}
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
