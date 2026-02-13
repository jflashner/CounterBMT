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
from typing import Any, Optional, Tuple

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


    class NNXSceneTokenEncoder(nnx.Module):
        """Encodes map + traffic-light scene tensors into scene tokens.

        Paper note:
        Adv-BMT scene encoding is built from map vectors and dynamic traffic-light
        signals. This module keeps the same channel split for A2S conditioning.
        """

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = cfg.d_model
            self.map_vector_proj = Linear(cfg.scene_encoder.map_feature_dim, d_model, rngs=rngs)
            self.map_token_proj = Linear(d_model, d_model, rngs=rngs)
            self.traffic_light_proj = Linear(cfg.scene_encoder.traffic_light_feature_dim, d_model, rngs=rngs)
            self.position_proj = Linear(3, d_model, rngs=rngs)
            self.out_norm = RMSNorm(d_model)

        def _encode_map(
            self,
            *,
            map_feature: jnp.ndarray,  # [B,M,V,Fm]
            map_feature_valid_mask: jnp.ndarray,  # [B,M,V] bool
            map_position: jnp.ndarray,  # [B,M,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
            return map_tokens, map_token_mask, map_position

        def _encode_traffic_lights(
            self,
            *,
            traffic_light_feature: jnp.ndarray,
            traffic_light_valid_mask: jnp.ndarray,
            traffic_light_position: jnp.ndarray,  # [B,L,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if traffic_light_feature.ndim == 4:
                # Collapse temporal light states to one token per light.
                # This mirrors the "compact scene token" requirement while retaining
                # state history contribution.
                valid = traffic_light_valid_mask.astype(jnp.float32)
                num = jnp.sum(traffic_light_feature * valid[..., None], axis=1)
                den = jnp.maximum(1.0, jnp.sum(valid[..., None], axis=1))
                light_feat = num / den  # [B,L,Fl]
                light_mask = jnp.any(traffic_light_valid_mask, axis=1)  # [B,L]
            else:
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
            return light_tokens, light_mask, traffic_light_position

        def __call__(
            self,
            *,
            map_feature: jnp.ndarray,  # [B,M,V,Fm]
            map_feature_valid_mask: jnp.ndarray,  # [B,M,V]
            map_position: jnp.ndarray,  # [B,M,3]
            traffic_light_feature: Optional[jnp.ndarray] = None,  # [B,T,L,Fl] or [B,L,Fl]
            traffic_light_valid_mask: Optional[jnp.ndarray] = None,  # [B,T,L] or [B,L]
            traffic_light_position: Optional[jnp.ndarray] = None,  # [B,L,3]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            map_tokens, map_mask, map_pos = self._encode_map(
                map_feature=map_feature,
                map_feature_valid_mask=map_feature_valid_mask,
                map_position=map_position,
            )

            if (
                traffic_light_feature is not None
                and traffic_light_valid_mask is not None
                and traffic_light_position is not None
            ):
                light_tokens, light_mask, light_pos = self._encode_traffic_lights(
                    traffic_light_feature=traffic_light_feature,
                    traffic_light_valid_mask=traffic_light_valid_mask,
                    traffic_light_position=traffic_light_position,
                )
                scene_tokens = jnp.concatenate([map_tokens, light_tokens], axis=1)
                scene_mask = jnp.concatenate([map_mask, light_mask], axis=1)
                scene_position = jnp.concatenate([map_pos, light_pos], axis=1)
            else:
                scene_tokens, scene_mask, scene_position = map_tokens, map_mask, map_pos

            # Keep bounded token count for predictable compute in early-stage training.
            max_tokens = int(self.cfg.scene_encoder.max_scene_tokens)
            if max_tokens > 0 and scene_tokens.shape[1] > max_tokens:
                scene_tokens = scene_tokens[:, :max_tokens, :]
                scene_mask = scene_mask[:, :max_tokens]
                scene_position = scene_position[:, :max_tokens, :]

            scene_tokens = self.out_norm(scene_tokens)
            return scene_tokens, scene_mask, scene_position


    class RelationBiasProjector(nnx.Module):
        """Projects relation features [B,Lq,Lk,R] -> [B,H,Lq,Lk] bias."""

        def __init__(self, rel_dim: int, n_heads: int, *, rngs: nnx.Rngs):
            self.w = nnx.Param(jax.random.normal(rngs.params(), (rel_dim, n_heads)) * 0.02)
            self.b = nnx.Param(jnp.zeros((n_heads,), dtype=jnp.float32))

        def __call__(self, rel_feat: jnp.ndarray) -> jnp.ndarray:
            # [B,Lq,Lk,R] x [R,H] -> [B,Lq,Lk,H] -> [B,H,Lq,Lk]
            bias = jnp.einsum("bqkr,rh->bqkh", rel_feat, self.w.value) + self.b.value
            return jnp.transpose(bias, (0, 3, 1, 2))


    class MultiHeadAttention(nnx.Module):
        def __init__(self, d_model: int, n_heads: int, *, rngs: nnx.Rngs):
            if d_model % n_heads != 0:
                raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads

            self.q_proj = Linear(d_model, d_model, rngs=rngs)
            self.k_proj = Linear(d_model, d_model, rngs=rngs)
            self.v_proj = Linear(d_model, d_model, rngs=rngs)
            self.o_proj = Linear(d_model, d_model, rngs=rngs)

        def __call__(
            self,
            query: jnp.ndarray,
            key_value: jnp.ndarray,
            *,
            mask: Optional[jnp.ndarray] = None,   # [B,Lq,Lk] bool
            rel_bias: Optional[jnp.ndarray] = None,  # [B,H,Lq,Lk]
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

            if rel_bias is not None:
                scores = scores + rel_bias

            if mask is not None:
                # mask True means keep; False means masked out.
                scores = jnp.where(mask[:, None, :, :], scores, jnp.full_like(scores, -1e9))

            attn = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
            out = jnp.transpose(out, (0, 2, 1, 3)).reshape(bsz, q_len, self.d_model)
            return self.o_proj(out)


    class RelationAwareDecoderBlock(nnx.Module):
        """One decoder block with explicit A2A, A2T, A2S attention."""

        def __init__(self, cfg: NNXBMTConfig, *, rngs: nnx.Rngs):
            self.cfg = cfg
            d_model = cfg.d_model

            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)

            self.a2a_attn = MultiHeadAttention(d_model, cfg.n_heads, rngs=rngs)
            self.a2t_attn = MultiHeadAttention(d_model, cfg.n_heads, rngs=rngs)
            self.a2s_attn = MultiHeadAttention(d_model, cfg.n_heads, rngs=rngs)

            self.a2a_rel_proj = RelationBiasProjector(cfg.a2a_rel_dim, cfg.n_heads, rngs=rngs)
            self.a2t_rel_proj = RelationBiasProjector(cfg.a2t_rel_dim, cfg.n_heads, rngs=rngs)
            self.a2s_rel_proj = RelationBiasProjector(cfg.a2s_rel_dim, cfg.n_heads, rngs=rngs)

            ff_hidden = d_model * cfg.ff_mult
            self.ff_in = Linear(d_model, ff_hidden, rngs=rngs)
            self.ff_out = Linear(ff_hidden, d_model, rngs=rngs)

        def _a2a(self, h: jnp.ndarray, rel: Optional[jnp.ndarray]) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*T,N,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz * t_steps, n_agents, d_model)
            rel_bias = None
            if rel is not None:
                # rel: [B,T,N,N,R] => [B*T,N,N,R]
                rel_bt = rel.reshape(bsz * t_steps, n_agents, n_agents, rel.shape[-1])
                rel_bias = self.a2a_rel_proj(rel_bt)
            out = self.a2a_attn(q, q, rel_bias=rel_bias)
            return out.reshape(bsz, t_steps, n_agents, d_model)

        def _a2t(self, h: jnp.ndarray, rel: Optional[jnp.ndarray]) -> jnp.ndarray:
            # h: [B,T,N,D] => [B*N,T,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = jnp.transpose(h, (0, 2, 1, 3)).reshape(bsz * n_agents, t_steps, d_model)
            rel_bias = None
            if rel is not None:
                # rel: [B,N,T,T,R] => [B*N,T,T,R]
                rel_bn = rel.reshape(bsz * n_agents, t_steps, t_steps, rel.shape[-1])
                rel_bias = self.a2t_rel_proj(rel_bn)
            out = self.a2t_attn(q, q, rel_bias=rel_bias)
            out = out.reshape(bsz, n_agents, t_steps, d_model)
            return jnp.transpose(out, (0, 2, 1, 3))

        def _a2s(
            self,
            h: jnp.ndarray,
            scene_tokens: jnp.ndarray,
            rel: Optional[jnp.ndarray],
            scene_token_mask: Optional[jnp.ndarray] = None,  # [B,S]
        ) -> jnp.ndarray:
            # h: [B,T,N,D] => [B,T*N,D], scene_tokens: [B,S,D]
            bsz, t_steps, n_agents, d_model = h.shape
            q = h.reshape(bsz, t_steps * n_agents, d_model)

            rel_bias = None
            if rel is not None:
                # rel: [B,T,N,S,R] => [B,T*N,S,R]
                rel_qs = rel.reshape(bsz, t_steps * n_agents, scene_tokens.shape[1], rel.shape[-1])
                rel_bias = self.a2s_rel_proj(rel_qs)

            attn_mask = None
            if scene_token_mask is not None:
                attn_mask = jnp.broadcast_to(
                    scene_token_mask[:, None, :], (bsz, t_steps * n_agents, scene_tokens.shape[1])
                )

            out = self.a2s_attn(q, scene_tokens, mask=attn_mask, rel_bias=rel_bias)
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
        ) -> jnp.ndarray:
            h = self.norm1(x)

            a2a_out = self._a2a(h, a2a_rel)
            a2t_out = self._a2t(h, a2t_rel)
            a2s_out = self._a2s(h, scene_tokens, a2s_rel, scene_token_mask=scene_token_mask)

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

            # Embeddings from Adv-BMT appendix.
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
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            if scene_tokens is not None:
                if scene_token_mask is None:
                    scene_token_mask = jnp.ones(scene_tokens.shape[:2], dtype=bool)
                return scene_tokens, scene_token_mask

            if (
                map_feature is not None
                and map_feature_valid_mask is not None
                and map_position is not None
            ):
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
            return fallback, fallback_mask

        def __call__(
            self,
            *,
            prev_token_ids: jnp.ndarray,  # [B,T,N]
            agent_type_ids: jnp.ndarray,  # [B,N]
            agent_shape: jnp.ndarray,  # [B,N,3]
            agent_ids: jnp.ndarray,  # [B,N]
            continuous_motion: jnp.ndarray,  # [B,T,N,2] (acc,yaw)
            reverse_indicator: jnp.ndarray,  # [B], 0=forward,1=reverse
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

            h = token_e + type_e + id_e + shp + mot + rev_e
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
            )

            for block in self.decoder_blocks:
                h = block(
                    h,
                    scene_tokens=scene,
                    scene_token_mask=scene_mask,
                    a2a_rel=a2a_rel,
                    a2t_rel=a2t_rel,
                    a2s_rel=a2s_rel,
                )

            h = self.final_norm(h)
            logits = self.token_head(h)  # [B,T,N,|A|]
            return logits


else:  # HAS_NNX == False
    # Keep symbols available for import paths.
    Linear = None
    RMSNorm = None
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

    if tokenizer is not None:
        action_table = jnp.asarray(tokenizer.action_table_np())
        motion_seq = jnp.take(action_table, token_seq, axis=0)  # [B,1,N,2]
    else:
        bsz, _, n_agents = token_seq.shape
        motion_seq = jnp.zeros((bsz, 1, n_agents, 2), dtype=jnp.float32)

    for _ in range(horizon_steps):
        model_kwargs = {
            "prev_token_ids": token_seq,
            "agent_type_ids": agent_type_ids,
            "agent_shape": agent_shape,
            "agent_ids": agent_ids,
            "continuous_motion": motion_seq,
            "reverse_indicator": reverse_indicator,
            "scene_tokens": scene_tokens,
            "a2a_rel": a2a_rel,
            "a2t_rel": a2t_rel,
            "a2s_rel": a2s_rel,
        }
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

        if tokenizer is not None:
            next_motion = jnp.take(action_table, next_tok, axis=0)  # [B,N,2]
        else:
            bsz, n_agents = next_tok.shape
            next_motion = jnp.zeros((bsz, n_agents, 2), dtype=jnp.float32)

        motion_seq = jnp.concatenate([motion_seq, next_motion[:, None, :, :]], axis=1)

    # Remove seed frame.
    return token_seq[:, 1:, :], motion_seq[:, 1:, :, :]
