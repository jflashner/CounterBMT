"""Legacy-aligned relation feature builders for Adv-BMT parity.

This module provides two paths:
- NumPy parity builders used by offline parity scripts and debug export.
- JAX relation builder used by the NNX runtime scene encoder path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None


HEADING_PLACEHOLDER_DEFAULT = -100.0


def _torch_argsort_last_axis(values: np.ndarray) -> Optional[np.ndarray]:
    """Use torch argsort when available to match legacy tie behavior."""
    try:
        import torch
    except Exception:
        return None

    t = torch.from_numpy(np.asarray(values))
    idx = torch.argsort(t, dim=-1)
    return idx.detach().cpu().numpy()


def _argsort_last_axis(values: np.ndarray) -> np.ndarray:
    """Legacy-like argsort helper.

    Torch `argsort` can produce a different tie order than NumPy for equal
    distances. We prefer torch to improve side-by-side parity with legacy
    relation builders. If torch is unavailable, fall back to NumPy.
    """
    torch_idx = _torch_argsort_last_axis(values)
    if torch_idx is not None:
        return torch_idx
    return np.argsort(values, axis=-1)


def _compute_relation_non_agent_with_torch(
    *,
    query_pos: np.ndarray,
    query_heading: np.ndarray,
    query_valid_mask: np.ndarray,
    key_pos: np.ndarray,
    key_heading: np.ndarray,
    key_valid_mask: np.ndarray,
    causal_valid_mask: Optional[np.ndarray],
    knn: Optional[int],
    max_distance: Optional[float],
    gather: bool,
    heading_placeholder: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Torch-backed non-agent relation path for strict legacy parity."""
    try:
        import torch
    except Exception:
        return None

    q_pos = torch.from_numpy(np.asarray(query_pos, dtype=np.float32))
    k_pos = torch.from_numpy(np.asarray(key_pos, dtype=np.float32))
    q_heading = torch.from_numpy(np.asarray(query_heading, dtype=np.float32))
    k_heading = torch.from_numpy(np.asarray(key_heading, dtype=np.float32))
    q_mask = torch.from_numpy(np.asarray(query_valid_mask, dtype=bool))
    k_mask = torch.from_numpy(np.asarray(key_valid_mask, dtype=bool))

    B, Q = q_heading.shape
    K = k_heading.shape[1]

    pairwise_heading = k_heading.unsqueeze(1) - q_heading.unsqueeze(2)
    heading_fill_0 = torch.logical_and(
        q_heading.eq(float(heading_placeholder)).unsqueeze(-1),
        k_heading.eq(float(heading_placeholder)).unsqueeze(-2),
    )
    pairwise_heading[heading_fill_0] = 0.0

    rel_pos = k_pos[:, None, :, :2] - q_pos[:, :, None, :2]
    i_local_x = q_heading.reshape(B, Q, 1).expand(B, Q, K) - (np.pi / 2.0)
    angle = -i_local_x
    rotated_pos = torch.stack(
        (
            torch.cos(angle) * rel_pos[..., 0] - torch.sin(angle) * rel_pos[..., 1],
            torch.cos(angle) * rel_pos[..., 1] + torch.sin(angle) * rel_pos[..., 0],
        ),
        dim=-1,
    )

    valid_mask = torch.logical_and(q_mask.unsqueeze(-1), k_mask.unsqueeze(-2))
    dist = torch.norm(rel_pos, p=2, dim=-1)

    if causal_valid_mask is not None:
        causal = np.asarray(causal_valid_mask, dtype=bool)
        if causal.ndim == 2:
            causal_t = torch.from_numpy(causal).unsqueeze(0).expand(B, -1, -1)
        elif causal.ndim == 3:
            causal_t = torch.from_numpy(causal)
        else:
            raise ValueError(f"Unsupported causal_valid_mask shape: {causal.shape}")
        dist = dist.masked_fill(~causal_t, float("+inf"))
        valid_mask = torch.logical_and(valid_mask, causal_t)

    max_distance = _as_optional_float(max_distance)
    if max_distance is not None:
        within_dist = dist < float(max_distance)
        force_k = min(8, K)
        closest = dist.argsort(dim=-1)[..., :force_k]
        b_idx = torch.arange(B, dtype=torch.long).view(B, 1, 1)
        q_idx = torch.arange(Q, dtype=torch.long).view(1, Q, 1)
        within_dist[b_idx, q_idx, closest] = True
        valid_mask = torch.logical_and(valid_mask, within_dist)

    indices = None
    if knn:
        k = max(1, min(int(knn), K))
        dist_masked = dist.masked_fill(~valid_mask, float("+inf"))
        indices = dist_masked.argsort(dim=-1)[..., :k]
        if gather:
            gather_idx_2 = indices.unsqueeze(-1).expand(-1, -1, -1, 2)
            rotated_pos = torch.gather(rotated_pos, dim=-2, index=gather_idx_2)
            pairwise_heading = torch.gather(pairwise_heading, dim=-1, index=indices)
            valid_mask = torch.gather(valid_mask, dim=-1, index=indices)
        else:
            original_valid = torch.zeros_like(valid_mask, dtype=torch.bool)
            b_idx = torch.arange(B, dtype=torch.long).view(B, 1, 1)
            q_idx = torch.arange(Q, dtype=torch.long).view(1, Q, 1)
            original_valid[b_idx, q_idx, indices] = True
            valid_mask = torch.logical_and(valid_mask, original_valid)

    distance = torch.norm(rotated_pos, p=2, dim=-1)
    rel_dir = torch.atan2(rotated_pos[..., 1], rotated_pos[..., 0])
    rel_feat = torch.cat([rel_dir[..., None], distance[..., None], pairwise_heading[..., None]], dim=-1)
    rel_feat = rel_feat.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    return (
        rel_feat.detach().cpu().numpy().astype(np.float32),
        valid_mask.detach().cpu().numpy().astype(bool),
        None if indices is None else indices.detach().cpu().numpy().astype(np.int32),
    )


@dataclass
class RelationBundleConfig:
    """Config for building parity relation bundles."""

    simple_relation: bool = True
    per_contour_point_relation: bool = False
    include_contour: bool = True

    heading_placeholder: float = HEADING_PLACEHOLDER_DEFAULT

    s2s_knn: Optional[int] = 128
    s2s_distance: Optional[float] = None

    a2s_knn: Optional[int] = 128
    a2s_distance: Optional[float] = None

    a2a_knn: Optional[int] = 64
    a2a_distance: Optional[float] = 50.0

    remove_traffic_light_state: bool = True
    # Legacy torch path is exact for non-agent tie behavior but slower.
    strict_non_agent_relation: bool = True


def pairwise_mask(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Pairwise validity mask from [B,Q] and [B,K] -> [B,Q,K]."""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if mask_a.ndim != 2 or mask_b.ndim != 2:
        raise ValueError(f"pairwise_mask expects [B,Q]/[B,K], got {mask_a.shape} and {mask_b.shape}")
    if mask_a.shape[0] != mask_b.shape[0]:
        raise ValueError(f"batch mismatch in pairwise_mask: {mask_a.shape} vs {mask_b.shape}")
    return np.logical_and(mask_a[:, :, None], mask_b[:, None, :])


def pairwise_relative_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise difference b-a over the second axis.

    For [B,Q,*] and [B,K,*], output is [B,Q,K,*].
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != b.ndim:
        raise ValueError(f"rank mismatch in pairwise_relative_diff: {a.shape} vs {b.shape}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch mismatch in pairwise_relative_diff: {a.shape} vs {b.shape}")
    return b[:, None, ...] - a[:, :, None, ...]


def rotate_local(x: np.ndarray, y: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate vectors by `angle` in radians."""
    x = np.asarray(x)
    y = np.asarray(y)
    angle = np.asarray(angle)
    xr = np.cos(angle) * x - np.sin(angle) * y
    yr = np.cos(angle) * y + np.sin(angle) * x
    return np.stack([xr, yr], axis=-1)


def cal_polygon_contour(
    x: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    width: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Compute oriented box contours with shape [...,4,2]."""
    x = np.asarray(x)
    y = np.asarray(y)
    theta = np.asarray(theta)
    width = np.asarray(width)
    length = np.asarray(length)

    left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)

    right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)

    right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
    right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)

    left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
    left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)

    return np.stack(
        [
            np.stack([left_front_x, left_front_y], axis=-1),
            np.stack([right_front_x, right_front_y], axis=-1),
            np.stack([right_back_x, right_back_y], axis=-1),
            np.stack([left_back_x, left_back_y], axis=-1),
        ],
        axis=-2,
    )


def _as_optional_float(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    vf = float(v)
    if not np.isfinite(vf):
        return None
    return vf


def compute_relation_simple_parity(
    *,
    query_pos: np.ndarray,
    query_heading: np.ndarray,
    query_valid_mask: np.ndarray,
    key_pos: np.ndarray,
    key_heading: np.ndarray,
    key_valid_mask: np.ndarray,
    hidden_dim: int = 0,
    causal_valid_mask: Optional[np.ndarray] = None,
    knn: Optional[int] = 128,
    max_distance: Optional[float] = None,
    gather: bool = True,
    return_pe: bool = False,
    query_step: Optional[np.ndarray] = None,
    key_step: Optional[np.ndarray] = None,
    include_contour: bool = False,
    query_width: Optional[np.ndarray] = None,
    query_length: Optional[np.ndarray] = None,
    key_width: Optional[np.ndarray] = None,
    key_length: Optional[np.ndarray] = None,
    non_agent_relation: bool = False,
    per_contour_point_relation: Optional[bool] = None,
    heading_placeholder: float = HEADING_PLACEHOLDER_DEFAULT,
    prefer_torch_non_agent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """NumPy parity port of legacy `compute_relation_simple_relation`.

    This returns raw relation features (not Fourier PE), relation mask, and optional
    KNN indices.
    """
    del hidden_dim
    del return_pe

    if per_contour_point_relation is None:
        per_contour_point_relation = False

    query_pos = np.asarray(query_pos, dtype=np.float32)
    key_pos = np.asarray(key_pos, dtype=np.float32)
    query_heading = np.asarray(query_heading, dtype=np.float32)
    key_heading = np.asarray(key_heading, dtype=np.float32)
    query_valid_mask = np.asarray(query_valid_mask, dtype=bool)
    key_valid_mask = np.asarray(key_valid_mask, dtype=bool)

    if query_pos.ndim != 3 or key_pos.ndim != 3:
        raise ValueError(f"query/key pos must be [B,Q,D]/[B,K,D], got {query_pos.shape} {key_pos.shape}")

    if non_agent_relation and prefer_torch_non_agent:
        torch_ret = _compute_relation_non_agent_with_torch(
            query_pos=query_pos,
            query_heading=query_heading,
            query_valid_mask=query_valid_mask,
            key_pos=key_pos,
            key_heading=key_heading,
            key_valid_mask=key_valid_mask,
            causal_valid_mask=causal_valid_mask,
            knn=knn,
            max_distance=max_distance,
            gather=gather,
            heading_placeholder=heading_placeholder,
        )
        if torch_ret is not None:
            return torch_ret

    B, Q = query_heading.shape
    K = key_heading.shape[1]

    pairwise_heading = pairwise_relative_diff(query_heading, key_heading)
    heading_fill_0 = pairwise_mask(query_heading == heading_placeholder, key_heading == heading_placeholder)
    pairwise_heading[heading_fill_0] = 0.0

    rel_pos = pairwise_relative_diff(query_pos[..., :2], key_pos[..., :2])

    # Query local x-axis wrt global is heading - pi/2 in legacy.
    i_local_x = query_heading[:, :, None] - np.pi / 2.0
    rotated_pos = rotate_local(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x)

    valid_mask = pairwise_mask(query_valid_mask, key_valid_mask)

    contour_info = None
    if include_contour and (not non_agent_relation):
        if query_width is None:
            query_width = np.zeros((B, Q), dtype=np.float32)
        if query_length is None:
            query_length = np.zeros((B, Q), dtype=np.float32)
        if key_width is None:
            key_width = np.zeros((B, K), dtype=np.float32)
        if key_length is None:
            key_length = np.zeros((B, K), dtype=np.float32)

        query_width = np.asarray(query_width, dtype=np.float32)
        query_length = np.asarray(query_length, dtype=np.float32)
        key_width = np.asarray(key_width, dtype=np.float32)
        key_length = np.asarray(key_length, dtype=np.float32)

        contour_q_center = cal_polygon_contour(
            x=query_pos[..., 0],
            y=query_pos[..., 1],
            theta=query_heading,
            width=np.zeros_like(query_pos[..., 0]),
            length=np.zeros_like(query_pos[..., 0]),
        )
        contour_k = cal_polygon_contour(
            x=key_pos[..., 0],
            y=key_pos[..., 1],
            theta=key_heading,
            width=key_width,
            length=key_length,
        )

        if per_contour_point_relation:
            contour_q = cal_polygon_contour(
                x=query_pos[..., 0],
                y=query_pos[..., 1],
                theta=query_heading,
                width=query_width,
                length=query_length,
            )
            contour_k_center = cal_polygon_contour(
                x=key_pos[..., 0],
                y=key_pos[..., 1],
                theta=key_heading,
                width=np.zeros_like(key_pos[..., 0]),
                length=np.zeros_like(key_pos[..., 0]),
            )

            contour_diff_in_q = pairwise_relative_diff(contour_q_center, contour_k)  # [B,Q,K,4,2]
            contour_q_min = contour_diff_in_q.min(axis=-2)
            contour_q_max = contour_diff_in_q.max(axis=-2)

            contour_diff_in_k = pairwise_relative_diff(contour_k_center, contour_q)  # [B,K,Q,4,2]
            key_local_x = key_heading[:, :, None] - np.pi / 2.0
            contour_diff_in_k = rotate_local(
                contour_diff_in_k[..., 0],
                contour_diff_in_k[..., 1],
                angle=-key_local_x[:, :, :, None],
            )
            contour_k_min = contour_diff_in_k.min(axis=-2).transpose(0, 2, 1, 3)
            contour_k_max = contour_diff_in_k.max(axis=-2).transpose(0, 2, 1, 3)
            contour_info = np.concatenate([contour_q_min, contour_q_max, contour_k_min, contour_k_max], axis=-1)
        else:
            contour_diff_in_q = pairwise_relative_diff(contour_q_center, contour_k)
            contour_diff_in_q = rotate_local(
                contour_diff_in_q[..., 0],
                contour_diff_in_q[..., 1],
                angle=-i_local_x[:, :, :, None],
            )
            contour_info = contour_diff_in_q.reshape(B, Q, K, 8)

    dist = np.linalg.norm(rel_pos, axis=-1)

    if causal_valid_mask is not None:
        causal_valid_mask = np.asarray(causal_valid_mask, dtype=bool)
        if causal_valid_mask.ndim == 2:
            causal = np.broadcast_to(causal_valid_mask[None, :, :], (B, Q, K))
            dist = np.where(causal, dist, np.inf)
            valid_mask = np.logical_and(valid_mask, causal)
        elif causal_valid_mask.ndim == 3:
            dist = np.where(causal_valid_mask, dist, np.inf)
            valid_mask = np.logical_and(valid_mask, causal_valid_mask)
        else:
            raise ValueError(f"Unsupported causal_valid_mask shape: {causal_valid_mask.shape}")

    max_distance = _as_optional_float(max_distance)
    if max_distance is not None:
        within_dist = dist < max_distance
        force_k = min(8, K)
        closest = _argsort_last_axis(dist)[..., :force_k]
        b_idx = np.arange(B)[:, None, None]
        q_idx = np.arange(Q)[None, :, None]
        within_dist[b_idx, q_idx, closest] = True
        valid_mask = np.logical_and(valid_mask, within_dist)

    if (not non_agent_relation) and (query_step is not None):
        if key_step is None:
            raise ValueError("key_step is required when query_step is provided")
        step_diff = pairwise_relative_diff(np.asarray(query_step, dtype=np.float32), np.asarray(key_step, dtype=np.float32))
    else:
        step_diff = None

    indices = None
    if knn:
        k = max(1, min(int(knn), K))
        dist_masked = np.where(valid_mask, dist, np.inf)
        indices = _argsort_last_axis(dist_masked)[..., :k].astype(np.int32)

        if gather:
            rotated_pos = np.take_along_axis(rotated_pos, indices[..., None], axis=-2)
            pairwise_heading = np.take_along_axis(pairwise_heading, indices, axis=-1)
            valid_mask = np.take_along_axis(valid_mask, indices, axis=-1)
            if step_diff is not None:
                step_diff = np.take_along_axis(step_diff, indices, axis=-1)
            if contour_info is not None:
                contour_info = np.take_along_axis(contour_info, indices[..., None], axis=-2)
        else:
            original_valid_mask = np.zeros_like(valid_mask, dtype=bool)
            b_idx = np.arange(B)[:, None, None]
            q_idx = np.arange(Q)[None, :, None]
            original_valid_mask[b_idx, q_idx, indices] = True
            valid_mask = np.logical_and(valid_mask, original_valid_mask)

    distance = np.linalg.norm(rotated_pos, axis=-1)
    relative_direction = np.arctan2(rotated_pos[..., 1], rotated_pos[..., 0])

    ret = [relative_direction[..., None], distance[..., None], pairwise_heading[..., None]]
    if (not non_agent_relation) and (step_diff is not None):
        ret.append(step_diff[..., None])
    if (not non_agent_relation) and (contour_info is not None):
        ret.append(contour_info)

    rel_feat = np.concatenate(ret, axis=-1).astype(np.float32)
    rel_feat[~valid_mask] = 0.0
    return rel_feat, valid_mask.astype(bool), indices


def compute_relation_parity(*, simple_relation: bool = True, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Public parity relation entrypoint.

    Non-simple mode is intentionally deferred; we keep a compatible fallback for
    P1 so callers can invoke a single API.
    """
    if simple_relation:
        return compute_relation_simple_parity(**kwargs)
    # P1 fallback: route to simple path until P2 full non-simple parity lands.
    return compute_relation_simple_parity(**kwargs)


def _masked_circular_mean_np(angles: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float32)
    sin_sum = np.sum(np.sin(angles) * m, axis=axis)
    cos_sum = np.sum(np.cos(angles) * m, axis=axis)
    valid = np.sum(m, axis=axis) > 0
    out = np.arctan2(sin_sum, cos_sum).astype(np.float32)
    out[~valid] = 0.0
    return out


def build_scene_token_relation_inputs_np(
    *,
    map_feature: np.ndarray,
    map_feature_valid_mask: np.ndarray,
    map_position: np.ndarray,
    traffic_light_feature: Optional[np.ndarray],
    traffic_light_valid_mask: Optional[np.ndarray],
    traffic_light_position: Optional[np.ndarray],
    remove_traffic_light_state: bool = True,
    heading_placeholder: float = HEADING_PLACEHOLDER_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Build scene token relation inputs (position/heading/mask) from collated tensors."""
    map_feature = np.asarray(map_feature, dtype=np.float32)
    map_feature_valid_mask = np.asarray(map_feature_valid_mask, dtype=bool)
    map_position = np.asarray(map_position, dtype=np.float32)

    map_mask = np.any(map_feature_valid_mask, axis=-1)  # [B,M]
    map_heading = _masked_circular_mean_np(map_feature[..., 9], map_feature_valid_mask, axis=2)  # [B,M]

    scene_pos = map_position
    scene_heading = map_heading
    scene_mask = map_mask

    if (
        traffic_light_feature is not None
        and traffic_light_valid_mask is not None
        and traffic_light_position is not None
    ):
        tl_feat = np.asarray(traffic_light_feature, dtype=np.float32)
        tl_valid = np.asarray(traffic_light_valid_mask, dtype=bool)
        tl_pos = np.asarray(traffic_light_position, dtype=np.float32)

        if tl_feat.ndim == 4:
            # [B,T,L,7]
            if remove_traffic_light_state:
                light_mask = np.any(tl_valid, axis=1)  # [B,L]
                valid_f = tl_valid.astype(np.float32)[..., None]
                pos_num = np.sum(tl_feat[..., :3] * valid_f, axis=1)
                pos_den = np.maximum(1.0, np.sum(valid_f, axis=1))
                stop_point = pos_num / pos_den
                state_scores = np.sum(tl_feat[..., 3:7] * valid_f, axis=1)  # [B,L,4]
                cls = np.argmax(state_scores, axis=-1)
                onehot = np.eye(4, dtype=np.float32)[cls]
                light_feat = np.concatenate([stop_point, onehot], axis=-1)
            else:
                light_mask = np.any(tl_valid, axis=1)
                valid_f = tl_valid.astype(np.float32)
                num = np.sum(tl_feat * valid_f[..., None], axis=1)
                den = np.maximum(1.0, np.sum(valid_f[..., None], axis=1))
                light_feat = num / den
        else:
            # [B,L,7]
            light_feat = tl_feat
            light_mask = tl_valid

        light_heading = np.full(light_mask.shape, float(heading_placeholder), dtype=np.float32)
        scene_pos = np.concatenate([scene_pos, tl_pos], axis=1)
        scene_heading = np.concatenate([scene_heading, light_heading], axis=1)
        scene_mask = np.concatenate([scene_mask, light_mask], axis=1)

        return {
            "scene_position": scene_pos.astype(np.float32),
            "scene_heading": scene_heading.astype(np.float32),
            "scene_valid_mask": scene_mask.astype(bool),
            "map_token_mask": map_mask.astype(bool),
            "traffic_light_token_mask": light_mask.astype(bool),
            "traffic_light_feature_collapsed": light_feat.astype(np.float32),
        }

    return {
        "scene_position": scene_pos.astype(np.float32),
        "scene_heading": scene_heading.astype(np.float32),
        "scene_valid_mask": scene_mask.astype(bool),
        "map_token_mask": map_mask.astype(bool),
        "traffic_light_token_mask": np.zeros((scene_pos.shape[0], 0), dtype=bool),
        "traffic_light_feature_collapsed": np.zeros((scene_pos.shape[0], 0, 7), dtype=np.float32),
    }


def build_relation_bundle(
    *,
    agent_position_xy: np.ndarray,
    agent_heading: np.ndarray,
    agent_valid_mask: np.ndarray,
    decoder_valid_mask: Optional[np.ndarray] = None,
    agent_shape: np.ndarray,
    sample_steps: Optional[np.ndarray] = None,
    scene_position: Optional[np.ndarray] = None,
    scene_heading: Optional[np.ndarray] = None,
    scene_valid_mask: Optional[np.ndarray] = None,
    map_feature: Optional[np.ndarray] = None,
    map_feature_valid_mask: Optional[np.ndarray] = None,
    map_position: Optional[np.ndarray] = None,
    traffic_light_feature: Optional[np.ndarray] = None,
    traffic_light_valid_mask: Optional[np.ndarray] = None,
    traffic_light_position: Optional[np.ndarray] = None,
    cfg: Optional[RelationBundleConfig] = None,
) -> Dict[str, np.ndarray]:
    """Build scene + decoder-ready relation tensors for parity debugging."""
    cfg = cfg or RelationBundleConfig()

    agent_position_xy = np.asarray(agent_position_xy, dtype=np.float32)
    agent_heading = np.asarray(agent_heading, dtype=np.float32)
    agent_valid_mask = np.asarray(agent_valid_mask, dtype=bool)
    agent_shape = np.asarray(agent_shape, dtype=np.float32)

    if sample_steps is not None:
        steps = np.asarray(sample_steps, dtype=np.int32)
        agent_position_xy = agent_position_xy[:, steps]
        agent_heading = agent_heading[:, steps]
        agent_valid_mask = agent_valid_mask[:, steps]
        step_vals = steps.astype(np.float32)
    else:
        step_vals = np.arange(agent_position_xy.shape[1], dtype=np.float32)

    if decoder_valid_mask is not None:
        # Decoder relations should be masked by tokenizer input validity
        # (`decoder/input_action_valid_mask`) to match legacy motion_decoder_gpt.
        dec_mask = np.asarray(decoder_valid_mask, dtype=bool)
        if dec_mask.ndim != 3:
            raise ValueError(f"decoder_valid_mask must be [B,T,N], got shape {dec_mask.shape}")
        if sample_steps is not None and dec_mask.shape[1] != len(steps):
            # Allow callers to pass full-rate masks; subsample to decoder steps.
            dec_mask = dec_mask[:, steps]
        if dec_mask.shape != agent_valid_mask.shape:
            raise ValueError(
                "decoder_valid_mask shape mismatch: "
                f"decoder={dec_mask.shape} vs sampled_agent_valid={agent_valid_mask.shape}"
            )
        agent_valid_mask = dec_mask

    B, T, N, _ = agent_position_xy.shape

    if scene_position is None or scene_heading is None or scene_valid_mask is None:
        if map_feature is None or map_feature_valid_mask is None or map_position is None:
            raise ValueError("scene inputs missing: provide scene_* or map/tl tensors")
        scene_inputs = build_scene_token_relation_inputs_np(
            map_feature=np.asarray(map_feature, dtype=np.float32),
            map_feature_valid_mask=np.asarray(map_feature_valid_mask, dtype=bool),
            map_position=np.asarray(map_position, dtype=np.float32),
            traffic_light_feature=None if traffic_light_feature is None else np.asarray(traffic_light_feature, dtype=np.float32),
            traffic_light_valid_mask=None if traffic_light_valid_mask is None else np.asarray(traffic_light_valid_mask, dtype=bool),
            traffic_light_position=None if traffic_light_position is None else np.asarray(traffic_light_position, dtype=np.float32),
            remove_traffic_light_state=cfg.remove_traffic_light_state,
            heading_placeholder=cfg.heading_placeholder,
        )
        scene_position = scene_inputs["scene_position"]
        scene_heading = scene_inputs["scene_heading"]
        scene_valid_mask = scene_inputs["scene_valid_mask"]
    else:
        scene_position = np.asarray(scene_position, dtype=np.float32)
        scene_heading = np.asarray(scene_heading, dtype=np.float32)
        scene_valid_mask = np.asarray(scene_valid_mask, dtype=bool)

    # Scene self relation (S2S): non-agent relation returns 3 dims.
    scene_rel, scene_mask, scene_idx = compute_relation_parity(
        simple_relation=cfg.simple_relation,
        query_pos=scene_position,
        query_heading=scene_heading,
        query_valid_mask=scene_valid_mask,
        key_pos=scene_position,
        key_heading=scene_heading,
        key_valid_mask=scene_valid_mask,
        causal_valid_mask=None,
        knn=cfg.s2s_knn,
        max_distance=cfg.s2s_distance,
        gather=False,
        return_pe=False,
        non_agent_relation=True,
        per_contour_point_relation=cfg.per_contour_point_relation,
        heading_placeholder=cfg.heading_placeholder,
        prefer_torch_non_agent=bool(cfg.strict_non_agent_relation),
    )

    agent_length = agent_shape[..., 0]
    agent_width = agent_shape[..., 1]

    # A2A: legacy MotionDecoderGPT uses the default relation gather path here.
    # That means the raw relation tensor is already reduced to KNN width before
    # Fourier embedding and sparse edge extraction. Keeping this gathered in the
    # parity path is important both for shape fidelity and to avoid paying full
    # [N x N] memory when the legacy model is only operating on the KNN subset.
    #
    # Legacy reference:
    #   motion_decoder_gpt.py -> relation_func(...) for a2a without gather=False
    #   relation.py -> compute_relation(..., gather=True by default)
    a2a_q_pos = agent_position_xy.reshape(B * T, N, 2)
    a2a_q_heading = agent_heading.reshape(B * T, N)
    a2a_q_mask = agent_valid_mask.reshape(B * T, N)

    w_bt = np.broadcast_to(agent_width[:, None, :], (B, T, N)).reshape(B * T, N)
    l_bt = np.broadcast_to(agent_length[:, None, :], (B, T, N)).reshape(B * T, N)
    s_bt = np.broadcast_to(step_vals[None, :, None], (B, T, N)).reshape(B * T, N)

    a2a_rel_bt, a2a_mask_bt, a2a_idx_bt = compute_relation_parity(
        simple_relation=cfg.simple_relation,
        query_pos=a2a_q_pos,
        query_heading=a2a_q_heading,
        query_valid_mask=a2a_q_mask,
        key_pos=a2a_q_pos,
        key_heading=a2a_q_heading,
        key_valid_mask=a2a_q_mask,
        query_step=s_bt,
        key_step=s_bt,
        include_contour=cfg.include_contour,
        query_width=w_bt,
        query_length=l_bt,
        key_width=w_bt,
        key_length=l_bt,
        causal_valid_mask=None,
        knn=cfg.a2a_knn,
        max_distance=cfg.a2a_distance,
        gather=True,
        return_pe=False,
        non_agent_relation=False,
        per_contour_point_relation=cfg.per_contour_point_relation,
        heading_placeholder=cfg.heading_placeholder,
    )
    a2a_rel = a2a_rel_bt.reshape(B, T, N, a2a_rel_bt.shape[2], a2a_rel_bt.shape[3])
    a2a_mask = a2a_mask_bt.reshape(B, T, N, a2a_mask_bt.shape[2])

    # A2T: legacy uses knn=None, so this remains full temporal width. The
    # gather flag is therefore irrelevant for parity in this branch.
    a2t_q_pos = np.transpose(agent_position_xy, (0, 2, 1, 3)).reshape(B * N, T, 2)
    a2t_q_heading = np.transpose(agent_heading, (0, 2, 1)).reshape(B * N, T)
    a2t_q_mask = np.transpose(agent_valid_mask, (0, 2, 1)).reshape(B * N, T)

    w_bn = np.broadcast_to(agent_width[:, :, None], (B, N, T)).reshape(B * N, T)
    l_bn = np.broadcast_to(agent_length[:, :, None], (B, N, T)).reshape(B * N, T)
    s_bn = np.broadcast_to(step_vals[None, None, :], (B, N, T)).reshape(B * N, T)

    a2t_rel_bn, a2t_mask_bn, a2t_idx_bn = compute_relation_parity(
        simple_relation=cfg.simple_relation,
        query_pos=a2t_q_pos,
        query_heading=a2t_q_heading,
        query_valid_mask=a2t_q_mask,
        key_pos=a2t_q_pos,
        key_heading=a2t_q_heading,
        key_valid_mask=a2t_q_mask,
        query_step=s_bn,
        key_step=s_bn,
        include_contour=cfg.include_contour,
        query_width=w_bn,
        query_length=l_bn,
        key_width=w_bn,
        key_length=l_bn,
        causal_valid_mask=None,
        knn=None,
        max_distance=None,
        gather=False,
        return_pe=False,
        non_agent_relation=False,
        per_contour_point_relation=cfg.per_contour_point_relation,
        heading_placeholder=cfg.heading_placeholder,
    )
    a2t_rel = a2t_rel_bn.reshape(B, N, T, a2t_rel_bn.shape[2], a2t_rel_bn.shape[3])
    a2t_mask = a2t_mask_bn.reshape(B, N, T, a2t_mask_bn.shape[2])

    # A2S: legacy explicitly passes gather=False, then converts the masked dense
    # [query x scene] relation tensor into sparse edge lists. We keep that exact
    # shape behavior here for parity, even though it is still a major memory hot
    # spot in both implementations.
    S = scene_position.shape[1]
    a2s_q_pos = agent_position_xy.reshape(B, T * N, 2)
    a2s_q_heading = agent_heading.reshape(B, T * N)
    a2s_q_mask = agent_valid_mask.reshape(B, T * N)
    s_tn = np.broadcast_to(step_vals[None, :, None], (B, T, N)).reshape(B, T * N)

    a2s_rel, a2s_mask, a2s_idx = compute_relation_parity(
        simple_relation=cfg.simple_relation,
        query_pos=a2s_q_pos,
        query_heading=a2s_q_heading,
        query_valid_mask=a2s_q_mask,
        key_pos=scene_position,
        key_heading=scene_heading,
        key_valid_mask=scene_valid_mask,
        query_step=s_tn,
        key_step=np.zeros((B, S), dtype=np.float32),
        include_contour=cfg.include_contour,
        query_width=np.broadcast_to(agent_width[:, None, :], (B, T, N)).reshape(B, T * N),
        query_length=np.broadcast_to(agent_length[:, None, :], (B, T, N)).reshape(B, T * N),
        key_width=np.zeros((B, S), dtype=np.float32),
        key_length=np.zeros((B, S), dtype=np.float32),
        causal_valid_mask=None,
        knn=cfg.a2s_knn,
        max_distance=cfg.a2s_distance,
        gather=False,
        return_pe=False,
        non_agent_relation=True,
        per_contour_point_relation=cfg.per_contour_point_relation,
        heading_placeholder=cfg.heading_placeholder,
        prefer_torch_non_agent=bool(cfg.strict_non_agent_relation),
    )
    a2s_rel = a2s_rel.reshape(B, T, N, a2s_rel.shape[2], a2s_rel.shape[3])
    a2s_mask = a2s_mask.reshape(B, T, N, a2s_mask.shape[2])

    return {
        "scene_s2s_rel_feat": scene_rel.astype(np.float32),
        "scene_s2s_mask": scene_mask.astype(bool),
        "scene_s2s_indices": (
            np.zeros((scene_rel.shape[0], scene_rel.shape[1], 0), dtype=np.int32)
            if scene_idx is None
            else scene_idx.astype(np.int32)
        ),
        "a2a_rel_feat": a2a_rel.astype(np.float32),
        "a2a_mask": a2a_mask.astype(bool),
        "a2a_indices": (
            np.zeros((B, T, N, 0), dtype=np.int32)
            if a2a_idx_bt is None
            else a2a_idx_bt.reshape(B, T, N, -1).astype(np.int32)
        ),
        "a2t_rel_feat": a2t_rel.astype(np.float32),
        "a2t_mask": a2t_mask.astype(bool),
        "a2t_indices": (
            np.zeros((B, N, T, 0), dtype=np.int32)
            if a2t_idx_bn is None
            else a2t_idx_bn.reshape(B, N, T, -1).astype(np.int32)
        ),
        "a2s_rel_feat": a2s_rel.astype(np.float32),
        "a2s_mask": a2s_mask.astype(bool),
        "a2s_indices": (
            np.zeros((B, T, N, 0), dtype=np.int32)
            if a2s_idx is None
            else a2s_idx.reshape(B, T, N, -1).astype(np.int32)
        ),
        "sample_steps": step_vals.astype(np.float32),
    }


def compute_scene_relation_simple_jax(
    *,
    query_pos: Any,
    query_heading: Any,
    query_valid_mask: Any,
    key_pos: Any,
    key_heading: Any,
    key_valid_mask: Any,
    heading_placeholder: float = HEADING_PLACEHOLDER_DEFAULT,
    knn: Optional[int] = 128,
    max_distance: Optional[float] = None,
    gather: bool = True,
) -> Tuple[Any, Any, Optional[Any]]:
    """JAX simple relation path for runtime scene S2S attention."""
    if jnp is None or jax is None:
        raise RuntimeError("jax is required for compute_scene_relation_simple_jax")

    query_pos = jnp.asarray(query_pos, dtype=jnp.float32)
    key_pos = jnp.asarray(key_pos, dtype=jnp.float32)
    query_heading = jnp.asarray(query_heading, dtype=jnp.float32)
    key_heading = jnp.asarray(key_heading, dtype=jnp.float32)
    query_valid_mask = jnp.asarray(query_valid_mask, dtype=bool)
    key_valid_mask = jnp.asarray(key_valid_mask, dtype=bool)

    B, Q = query_heading.shape
    K = key_heading.shape[1]

    pairwise_heading = key_heading[:, None, :] - query_heading[:, :, None]
    heading_fill_0 = jnp.logical_and(
        (query_heading == float(heading_placeholder))[:, :, None],
        (key_heading == float(heading_placeholder))[:, None, :],
    )
    pairwise_heading = jnp.where(heading_fill_0, 0.0, pairwise_heading)

    rel_pos = key_pos[:, None, :, :2] - query_pos[:, :, None, :2]
    i_local_x = query_heading[:, :, None] - (np.pi / 2.0)
    angle = -i_local_x
    rx = jnp.cos(angle) * rel_pos[..., 0] - jnp.sin(angle) * rel_pos[..., 1]
    ry = jnp.cos(angle) * rel_pos[..., 1] + jnp.sin(angle) * rel_pos[..., 0]
    rotated_pos = jnp.stack([rx, ry], axis=-1)

    valid_mask = jnp.logical_and(query_valid_mask[:, :, None], key_valid_mask[:, None, :])
    dist = jnp.linalg.norm(rel_pos, axis=-1)

    max_distance = _as_optional_float(max_distance)
    if max_distance is not None:
        within_dist = dist < float(max_distance)
        force_k = min(8, int(K))
        closest = jnp.argsort(dist, axis=-1)[..., :force_k]
        closest_mask = jnp.any(jax.nn.one_hot(closest, K, dtype=jnp.bool_), axis=-2)
        within_dist = jnp.logical_or(within_dist, closest_mask)
        valid_mask = jnp.logical_and(valid_mask, within_dist)

    indices = None
    if knn:
        k = max(1, min(int(knn), int(K)))
        dist_masked = jnp.where(valid_mask, dist, jnp.inf)
        indices = jnp.argsort(dist_masked, axis=-1)[..., :k]
        if gather:
            rotated_pos = jnp.take_along_axis(rotated_pos, indices[..., None], axis=-2)
            pairwise_heading = jnp.take_along_axis(pairwise_heading, indices, axis=-1)
            valid_mask = jnp.take_along_axis(valid_mask, indices, axis=-1)
        else:
            knn_mask = jnp.any(jax.nn.one_hot(indices, K, dtype=jnp.bool_), axis=-2)
            valid_mask = jnp.logical_and(valid_mask, knn_mask)

    distance = jnp.linalg.norm(rotated_pos, axis=-1)
    rel_dir = jnp.arctan2(rotated_pos[..., 1], rotated_pos[..., 0])
    rel_feat = jnp.stack([rel_dir, distance, pairwise_heading], axis=-1)
    rel_feat = jnp.where(valid_mask[..., None], rel_feat, 0.0)

    return rel_feat.astype(jnp.float32), valid_mask, indices
