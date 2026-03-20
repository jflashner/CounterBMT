"""Day 2 supervised training loop for NNX Adv-BMT rewrite.

Paper alignment notes:
- Training objective mirrors Adv-BMT intent: masked cross-entropy over discrete
  motion tokens with top-level token-space configuration fixed at 33x33 bins.
- Optimizer/schedule defaults follow the released training setup direction:
  AdamW + cosine schedule + warmup + gradient clipping.
- Forward/backward mixed training is supported to match the bidirectional
  supervision intent, with explicit reverse indicators passed to the model.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import os
import pickle
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from counter_bmt_v2.data import NNXBMTSceneSample, ScenarioNetNNXLoader, collate_nnx_scene_samples
from .forward_metrics import (
    ForwardPassEvalConfig,
    compute_forward_pass_metrics_for_batch,
    nanmean_metrics,
)
from .tensorboard_logging import (
    create_tb_writer,
    tb_close,
    tb_write_scalar,
    tb_write_scalars,
    tb_write_text,
)
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    BidirectionalMotionTokenizer,
    NNXBMTConfig,
    NNXBidirectionalMotionTransformer,
    ParityTokenizerConfig,
    RelationBundleConfig,
    build_relation_bundle,
    build_scene_token_relation_inputs_np,
    cross_entropy_token_loss,
    midgpt_parity_config,
    masked_token_accuracy,
    paper_like_full_config,
    paper_like_small_config,
)


ModeType = Literal["forward", "reverse", "mixed"]
PresetType = Literal["paper_like_small", "paper_like_full", "midgpt_parity"]
TokenizerModeType = Literal["paper_simple", "adv_bmt_parity"]
DistributedBackendType = Literal["none", "pmap"]
PrecisionType = Literal["fp32", "bf16-mixed"]
LRScheduleModeType = Literal["v2_cosine_minlr", "legacy_cosine_zero"]
CollatePaddingModeType = Literal["fixed", "batch_local", "bucketed"]


@dataclass
class SupervisedTrainConfig:
    """Configuration for NNX supervised motion-token training."""

    data_dir: str = ""
    train_data_dir: str = ""
    val_data_dir: str = ""
    output_dir: str = "outputs/counter_bmt_v2_training"

    model_preset: PresetType = "paper_like_small"
    seed: int = 0

    num_epochs: int = 3
    batch_size: int = 4
    max_steps: Optional[int] = None

    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 200
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    lr_schedule_mode: LRScheduleModeType = "v2_cosine_minlr"
    distributed_backend: DistributedBackendType = "none"
    precision: PrecisionType = "fp32"
    save_rng_state: bool = True
    resume_strict_determinism: bool = True

    mode: ModeType = "mixed"
    reverse_probability: float = 0.5
    tokenizer_mode: TokenizerModeType = "paper_simple"

    # Raw ScenarioNet is typically 10Hz. Adv-BMT token chunks are 0.5s by default.
    # Using skip_steps=5 approximates the same temporal chunking.
    skip_steps: int = 5

    train_fraction: float = 0.95
    sample_interval_training: int = 1
    sample_interval_test: int = 1
    num_train_scenarios: Optional[int] = None
    num_val_scenarios: Optional[int] = None
    strict_91_steps: bool = False
    prescan_log_every: int = 5000
    prescan_workers: int = 0
    use_prescan_cache: bool = True

    eval_every_steps: int = 100
    eval_batches: int = 10
    log_every_steps: int = 10
    checkpoint_every_steps: int = 200
    enable_tensorboard: bool = True
    tensorboard_subdir: str = "tensorboard"
    tensorboard_flush_secs: int = 30
    tensorboard_log_run_config: bool = True

    # Loader ceilings are still applied per-sample before batching. The collate
    # mode controls whether batches are padded all the way to those ceilings
    # (`fixed`) or only to the batch-local maxima under the same ceilings
    # (`batch_local`), which matches legacy Adv-BMT more closely.
    max_time_steps: int = 91
    max_agents: int = 128
    max_map_features: int = 512
    max_vectors_per_map_feature: int = 128
    max_traffic_lights: int = 64
    collate_padding_mode: CollatePaddingModeType = "fixed"
    # Bucketed padding is a compromise between legacy-like batch-local memory
    # efficiency and JAX-friendly static shapes. We round each batch up to the
    # next coarse bucket so many batches can share the same compiled step.
    collate_agent_buckets: Tuple[int, ...] = (16, 24, 32, 48, 64, 80, 96, 112, 128)
    collate_map_feature_buckets: Tuple[int, ...] = (128, 192, 256, 320, 384, 448, 512)
    collate_traffic_light_buckets: Tuple[int, ...] = (8, 16, 24, 32, 48, 64)

    center_to_map: bool = True
    resume_checkpoint: str = ""
    relation_debug_dump_dir: str = ""
    relation_debug_dump_every_steps: int = 0
    relation_debug_max_batches: int = 1
    decoder_edge_sparse_attn: bool = False
    runtime_preset: str = "none"
    runtime_resolved_overrides: Dict[str, Any] = field(default_factory=dict)

    # Scenario-level forward-pass evaluator (Adv-BMT-style metrics).
    forward_eval: ForwardPassEvalConfig = field(default_factory=ForwardPassEvalConfig)


def _resolve_model_preset(name: PresetType) -> NNXBMTConfig:
    if name == "midgpt_parity":
        return midgpt_parity_config()
    if name == "paper_like_full":
        return paper_like_full_config()
    return paper_like_small_config()


def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _masked_mean_jax(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    num = jnp.sum(values * mask)
    den = jnp.maximum(1.0, jnp.sum(mask))
    return num / den


def _compute_reverse_indicator(mode: ModeType, batch_size: int, rng: np.random.Generator, reverse_prob: float) -> np.ndarray:
    if mode == "forward":
        return np.zeros((batch_size,), dtype=np.int32)
    if mode == "reverse":
        return np.ones((batch_size,), dtype=np.int32)

    p = float(np.clip(reverse_prob, 0.0, 1.0))
    return (rng.random(batch_size) < p).astype(np.int32)


def _prepare_modeled_agent_ids(
    *,
    agent_ids: np.ndarray,
    max_agent_id: int,
    randomize: bool,
    is_training: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Legacy-like modeled-agent ID preprocessing for embedding lookup."""
    ids = np.asarray(agent_ids, dtype=np.int32).copy()
    max_id = max(2, int(max_agent_id))

    # Clip invalid/out-of-range IDs to final bucket.
    invalid = (ids < 0) | (ids >= max_id)
    ids[invalid] = max_id - 1

    if not randomize or not is_training:
        return ids

    bsz, n_agents = ids.shape
    out = np.full_like(ids, fill_value=max_id - 1, dtype=np.int32)
    num_samples = min(n_agents, max_id)
    for b in range(bsz):
        perm = rng.choice(max_id, size=num_samples, replace=False).astype(np.int32)
        out[b, :num_samples] = perm

    out[invalid] = max_id - 1
    return out


def _tokenize_motion_targets_simple(
    batch: Dict[str, Any],
    *,
    tokenizer: BidirectionalMotionTokenizer,
    skip_steps: int,
    mode: ModeType,
    reverse_probability: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Build teacher-forced token sequences and masks from collated trajectories.

    This function converts continuous trajectory channels into the discrete motion
    token space used in Adv-BMT-style training.
    """
    if skip_steps <= 0:
        raise ValueError(f"skip_steps must be > 0, got: {skip_steps}")

    velocity = np.asarray(batch["agent_velocity_xy"], dtype=np.float32)  # [B,T,N,2]
    heading = np.asarray(batch["agent_heading"], dtype=np.float32)  # [B,T,N]
    valid = np.asarray(batch["agent_valid_mask"], dtype=bool)  # [B,T,N]
    dt_s = np.asarray(batch["dt_s"], dtype=np.float32)  # [B]

    bsz, t_raw, n_agents, _ = velocity.shape
    sample_steps = np.arange(0, t_raw, int(skip_steps), dtype=np.int32)
    if sample_steps.shape[0] < 2:
        raise ValueError(
            f"Not enough sampled steps after skip_steps={skip_steps}: t_raw={t_raw}, sampled={sample_steps.shape[0]}"
        )

    speed = np.linalg.norm(velocity, axis=-1)  # [B,T,N]
    speed_s = speed[:, sample_steps, :]  # [B,Ts,N]
    heading_s = heading[:, sample_steps, :]  # [B,Ts,N]
    valid_s = valid[:, sample_steps, :]  # [B,Ts,N]

    # Transition-level controls between sampled states.
    dt_chunk = dt_s[:, None, None] * float(skip_steps)
    dt_chunk = np.maximum(dt_chunk, 1e-6)

    acc = (speed_s[:, 1:, :] - speed_s[:, :-1, :]) / dt_chunk
    yaw_rate = _wrap_to_pi_np(heading_s[:, 1:, :] - heading_s[:, :-1, :]) / dt_chunk
    action_valid = valid_s[:, 1:, :] & valid_s[:, :-1, :]

    a_bin = np.searchsorted(tokenizer.acc_edges[1:], acc)
    y_bin = np.searchsorted(tokenizer.yaw_edges[1:], yaw_rate)
    a_bin = np.clip(a_bin, 0, tokenizer.cfg.n_acc_bins - 1)
    y_bin = np.clip(y_bin, 0, tokenizer.cfg.n_yaw_bins - 1)

    targets = (a_bin * tokenizer.cfg.n_yaw_bins + y_bin).astype(np.int32)
    targets[~action_valid] = 0

    reverse_indicator = _compute_reverse_indicator(
        mode=mode,
        batch_size=bsz,
        rng=rng,
        reverse_prob=reverse_probability,
    )

    # Mixed forward/backward mode: reverse sequence order per selected sample.
    if np.any(reverse_indicator == 1):
        rev_idx = np.where(reverse_indicator == 1)[0]
        targets[rev_idx] = targets[rev_idx, ::-1, :]
        action_valid[rev_idx] = action_valid[rev_idx, ::-1, :]

    seq_len = targets.shape[1]
    start_token = tokenizer.cfg.n_tokens  # first special token slot

    prev = np.full((bsz, seq_len, n_agents), fill_value=start_token, dtype=np.int32)
    if seq_len > 1:
        prev[:, 1:, :] = targets[:, :-1, :]

    # Convert previous token ids to continuous (acc, yaw-rate) features.
    action_table = tokenizer.action_table_np()
    continuous_motion = np.zeros((bsz, seq_len, n_agents, 2), dtype=np.float32)
    valid_prev = prev < tokenizer.cfg.n_tokens
    if np.any(valid_prev):
        continuous_motion[valid_prev] = action_table[prev[valid_prev]]

    return {
        "prev_token_ids": prev,
        "targets": targets,
        "target_mask": action_valid.astype(np.float32),
        "continuous_motion": continuous_motion,
        "reverse_indicator": reverse_indicator.astype(np.int32),
        "sample_steps": sample_steps.astype(np.int32),
    }


def _slice_batch_dict(batch: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (batch["agent_ids"].shape[0],):
            out[k] = v[indices]
        elif isinstance(v, list) and len(v) == batch["agent_ids"].shape[0]:
            out[k] = [v[int(i)] for i in indices.tolist()]
        else:
            out[k] = v
    return out


def _tokenize_motion_targets_parity(
    batch: Dict[str, Any],
    *,
    tokenizer: AdvBMTParityTokenizer,
    mode: ModeType,
    reverse_probability: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    bsz = int(np.asarray(batch["agent_ids"]).shape[0])
    reverse_indicator = _compute_reverse_indicator(
        mode=mode,
        batch_size=bsz,
        rng=rng,
        reverse_prob=reverse_probability,
    ).astype(np.int32)

    if bsz == 0:
        raise ValueError("Empty batch in parity tokenization")

    seq_len = max(1, int(np.ceil(batch["agent_position_xy"].shape[1] / float(tokenizer.parity_cfg.num_skipped_steps))))
    n_agents = int(batch["agent_position_xy"].shape[2])
    sample_steps = np.arange(
        0,
        int(batch["agent_position_xy"].shape[1]),
        int(tokenizer.parity_cfg.num_skipped_steps),
        dtype=np.int32,
    )

    prev = np.full((bsz, seq_len, n_agents), tokenizer.PAD_MODEL_ID, dtype=np.int32)
    targets = np.full((bsz, seq_len, n_agents), tokenizer.default_token_id, dtype=np.int32)
    target_mask = np.zeros((bsz, seq_len, n_agents), dtype=np.float32)
    motion = np.zeros((bsz, seq_len, n_agents, 2), dtype=np.float32)
    input_mask = np.zeros((bsz, seq_len, n_agents), dtype=bool)
    modeled_agent_delta = np.zeros((bsz, seq_len, n_agents, 2), dtype=np.float32)

    forward_idx = np.where(reverse_indicator == 0)[0]
    reverse_idx = np.where(reverse_indicator == 1)[0]

    def _assign(indices: np.ndarray, backward: bool) -> None:
        nonlocal sample_steps
        if indices.size == 0:
            return
        sub = _slice_batch_dict(batch, indices)
        tok = tokenizer.tokenize_batch(sub, backward_prediction=backward)
        sample_steps = tok.sample_steps
        seq = tok.prev_token_ids.shape[1]
        prev[indices, :seq] = tok.prev_token_ids
        targets[indices, :seq] = tok.targets
        target_mask[indices, :seq] = tok.target_mask
        motion[indices, :seq] = tok.continuous_motion
        input_mask[indices, :seq] = tok.input_mask
        modeled_agent_delta[indices, :seq] = tok.modeled_agent_delta

    _assign(forward_idx, backward=False)
    _assign(reverse_idx, backward=True)

    return {
        "prev_token_ids": prev.astype(np.int32),
        "targets": targets.astype(np.int32),
        "target_mask": target_mask.astype(np.float32),
        "continuous_motion": motion.astype(np.float32),
        "input_mask": input_mask.astype(bool),
        "modeled_agent_delta": modeled_agent_delta.astype(np.float32),
        "reverse_indicator": reverse_indicator.astype(np.int32),
        "sample_steps": sample_steps.astype(np.int32),
    }


def _tokenize_motion_targets(
    batch: Dict[str, Any],
    *,
    tokenizer: Any,
    tokenizer_mode: TokenizerModeType,
    skip_steps: int,
    mode: ModeType,
    reverse_probability: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if tokenizer_mode == "adv_bmt_parity":
        if not isinstance(tokenizer, AdvBMTParityTokenizer):
            raise TypeError("adv_bmt_parity mode requires AdvBMTParityTokenizer")
        return _tokenize_motion_targets_parity(
            batch,
            tokenizer=tokenizer,
            mode=mode,
            reverse_probability=reverse_probability,
            rng=rng,
        )
    if not isinstance(tokenizer, BidirectionalMotionTokenizer):
        raise TypeError("paper_simple mode requires BidirectionalMotionTokenizer")
    return _tokenize_motion_targets_simple(
        batch,
        tokenizer=tokenizer,
        skip_steps=skip_steps,
        mode=mode,
        reverse_probability=reverse_probability,
        rng=rng,
    )


def _prepare_supervised_batch(
    samples: Sequence[NNXBMTSceneSample],
    *,
    train_cfg: SupervisedTrainConfig,
    model_cfg: NNXBMTConfig,
    tokenizer: Any,
    rng: np.random.Generator,
    is_training: bool,
) -> Dict[str, Any]:
    collate_limits = _resolve_collate_padding_limits(train_cfg, samples=samples)
    batch = collate_nnx_scene_samples(
        samples,
        max_time_steps=collate_limits["max_time_steps"],
        max_agents=collate_limits["max_agents"],
        max_map_features=collate_limits["max_map_features"],
        max_vectors_per_map_feature=collate_limits["max_vectors_per_map_feature"],
        max_traffic_lights=collate_limits["max_traffic_lights"],
    )

    token_batch = _tokenize_motion_targets(
        batch,
        tokenizer=tokenizer,
        tokenizer_mode=train_cfg.tokenizer_mode,
        skip_steps=train_cfg.skip_steps,
        mode=train_cfg.mode,
        reverse_probability=train_cfg.reverse_probability,
        rng=rng,
    )

    modeled_agent_ids = _prepare_modeled_agent_ids(
        agent_ids=np.asarray(batch["agent_ids"], dtype=np.int32),
        max_agent_id=int(model_cfg.max_agent_id),
        randomize=bool(model_cfg.decoder.randomize_agent_id),
        is_training=bool(is_training),
        rng=rng,
    )

    model_inputs = {
        "prev_token_ids": jnp.asarray(token_batch["prev_token_ids"], dtype=jnp.int32),
        "agent_type_ids": jnp.asarray(batch["agent_type_ids"], dtype=jnp.int32),
        "agent_shape": jnp.asarray(batch["agent_shape"], dtype=jnp.float32),
        "agent_ids": jnp.asarray(modeled_agent_ids, dtype=jnp.int32),
        "continuous_motion": jnp.asarray(token_batch["continuous_motion"], dtype=jnp.float32),
        "reverse_indicator": jnp.asarray(token_batch["reverse_indicator"], dtype=jnp.int32),
        "scene_map_feature": jnp.asarray(batch["map_feature"], dtype=jnp.float32),
        "scene_map_valid_mask": jnp.asarray(batch["map_feature_valid_mask"], dtype=bool),
        "scene_map_position": jnp.asarray(batch["map_position"], dtype=jnp.float32),
        "scene_tl_feature": jnp.asarray(batch["traffic_light_feature"], dtype=jnp.float32),
        "scene_tl_valid_mask": jnp.asarray(batch["traffic_light_valid_mask"], dtype=bool),
        "scene_tl_position": jnp.asarray(batch["traffic_light_position"], dtype=jnp.float32),
    }
    decoder_valid_mask = None
    relation_sample_steps = np.asarray(token_batch["sample_steps"], dtype=np.int32)
    if "input_mask" in token_batch:
        decoder_valid_mask = np.asarray(token_batch["input_mask"], dtype=bool)
        model_inputs["input_action_valid_mask"] = jnp.asarray(decoder_valid_mask, dtype=bool)
    else:
        relation_sample_steps = relation_sample_steps[:-1]
        decoder_valid_mask = np.asarray(batch["agent_valid_mask"], dtype=bool)[:, relation_sample_steps, :]
    if "modeled_agent_delta" in token_batch:
        model_inputs["modeled_agent_delta"] = jnp.asarray(token_batch["modeled_agent_delta"], dtype=jnp.float32)

    if bool(model_cfg.decoder.enabled):
        scene_inputs = build_scene_token_relation_inputs_np(
            map_feature=np.asarray(batch["map_feature"], dtype=np.float32),
            map_feature_valid_mask=np.asarray(batch["map_feature_valid_mask"], dtype=bool),
            map_position=np.asarray(batch["map_position"], dtype=np.float32),
            traffic_light_feature=np.asarray(batch["traffic_light_feature"], dtype=np.float32),
            traffic_light_valid_mask=np.asarray(batch["traffic_light_valid_mask"], dtype=bool),
            traffic_light_position=np.asarray(batch["traffic_light_position"], dtype=np.float32),
            remove_traffic_light_state=bool(model_cfg.relation.remove_traffic_light_state),
            heading_placeholder=float(model_cfg.relation.heading_placeholder),
        )
        relation_cfg = RelationBundleConfig(
            simple_relation=bool(model_cfg.relation.simple_relation),
            per_contour_point_relation=bool(model_cfg.relation.per_contour_point_relation),
            include_contour=True,
            heading_placeholder=float(model_cfg.relation.heading_placeholder),
            s2s_knn=model_cfg.relation.s2s_knn,
            s2s_distance=model_cfg.relation.s2s_distance,
            a2s_knn=model_cfg.relation.a2s_knn,
            a2s_distance=model_cfg.relation.a2s_distance,
            a2a_knn=model_cfg.relation.a2a_knn,
            a2a_distance=model_cfg.relation.a2a_distance,
            remove_traffic_light_state=bool(model_cfg.relation.remove_traffic_light_state),
            strict_non_agent_relation=False,
        )
        relation_bundle = build_relation_bundle(
            agent_position_xy=np.asarray(batch["agent_position_xy"], dtype=np.float32),
            agent_heading=np.asarray(batch["agent_heading"], dtype=np.float32),
            agent_valid_mask=np.asarray(batch["agent_valid_mask"], dtype=bool),
            decoder_valid_mask=decoder_valid_mask,
            agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
            sample_steps=relation_sample_steps,
            scene_position=scene_inputs["scene_position"],
            scene_heading=scene_inputs["scene_heading"],
            scene_valid_mask=scene_inputs["scene_valid_mask"],
            cfg=relation_cfg,
        )
        model_inputs.update(
            {
                "a2a_rel": jnp.asarray(relation_bundle["a2a_rel_feat"], dtype=jnp.float32),
                "a2t_rel": jnp.asarray(relation_bundle["a2t_rel_feat"], dtype=jnp.float32),
                "a2s_rel": jnp.asarray(relation_bundle["a2s_rel_feat"], dtype=jnp.float32),
                "a2a_mask": jnp.asarray(relation_bundle["a2a_mask"], dtype=bool),
                "a2t_mask": jnp.asarray(relation_bundle["a2t_mask"], dtype=bool),
                "a2s_mask": jnp.asarray(relation_bundle["a2s_mask"], dtype=bool),
                "a2a_indices": jnp.asarray(relation_bundle["a2a_indices"], dtype=jnp.int32),
                "a2t_indices": jnp.asarray(relation_bundle["a2t_indices"], dtype=jnp.int32),
                "a2s_indices": jnp.asarray(relation_bundle["a2s_indices"], dtype=jnp.int32),
            }
        )

    model_inputs = _cast_tree_precision(model_inputs, precision=train_cfg.precision)
    targets = jnp.asarray(token_batch["targets"], dtype=jnp.int32)
    target_mask = jnp.asarray(token_batch["target_mask"], dtype=jnp.float32)
    reverse_indicator = model_inputs["reverse_indicator"]

    return {
        "model_inputs": model_inputs,
        "targets": targets,
        "target_mask": target_mask,
        "reverse_indicator": reverse_indicator,
        # Keep raw tensors for forward-pass metric computation.
        "raw_batch": batch,
        "sample_steps": token_batch["sample_steps"],
    }


def _bucket_up(value: int, buckets: Tuple[int, ...], *, ceiling: int) -> int:
    value_i = int(value)
    if value_i <= 0:
        return 0
    for bucket in sorted(int(x) for x in buckets if int(x) > 0):
        if value_i <= bucket <= int(ceiling):
            return int(bucket)
    return int(ceiling)


def _resolve_collate_padding_limits(
    train_cfg: SupervisedTrainConfig,
    *,
    samples: Optional[Sequence[NNXBMTSceneSample]] = None,
) -> Dict[str, Optional[int]]:
    """Resolve batch padding limits from the training config.

    Legacy Adv-BMT keeps `MAX_*` values as per-sample ceilings, but with
    `PADDING_TO_MAX=false` it only pads each batch to its local maxima. The v2
    parity path needs the same behavior to keep attention/relation activation
    memory in line with the legacy stack. We still keep a `fixed` mode because
    it is useful when minimizing recompilation churn matters more than memory.
    """

    mode = str(train_cfg.collate_padding_mode).strip().lower()
    if mode not in {"fixed", "batch_local", "bucketed"}:
        raise ValueError(
            "Unsupported collate_padding_mode: "
            f"{train_cfg.collate_padding_mode!r}. Expected 'fixed', 'batch_local', or 'bucketed'."
        )

    if mode == "fixed":
        return {
            "max_time_steps": int(train_cfg.max_time_steps),
            "max_agents": int(train_cfg.max_agents),
            "max_map_features": int(train_cfg.max_map_features),
            "max_vectors_per_map_feature": int(train_cfg.max_vectors_per_map_feature),
            "max_traffic_lights": int(train_cfg.max_traffic_lights),
        }

    # In batch-local mode we still keep the time horizon fixed to preserve the
    # intended 91-step training window, and we keep the per-polyline vector
    # budget fixed because individual map tokens are already extracted at that
    # shape. The large memory savings come from letting agent/map/light counts
    # follow the batch-local maxima instead of the global ceilings.
    if mode == "batch_local":
        return {
            "max_time_steps": int(train_cfg.max_time_steps),
            "max_agents": None,
            "max_map_features": None,
            "max_vectors_per_map_feature": int(train_cfg.max_vectors_per_map_feature),
            "max_traffic_lights": None,
        }

    # Bucketed mode is the speed-oriented compromise for JAX training. It uses
    # the batch-local maxima, but rounds them up to a small set of reusable
    # shapes so XLA can amortize compilation across many steps.
    if not samples:
        return {
            "max_time_steps": int(train_cfg.max_time_steps),
            "max_agents": int(train_cfg.max_agents),
            "max_map_features": int(train_cfg.max_map_features),
            "max_vectors_per_map_feature": int(train_cfg.max_vectors_per_map_feature),
            "max_traffic_lights": int(train_cfg.max_traffic_lights),
        }

    inferred_n = max(int(s.agent_position_xy.shape[1]) for s in samples)
    inferred_m = max(int(s.map_feature.shape[0]) for s in samples)
    inferred_l = max(int(s.traffic_light_feature.shape[1]) for s in samples)
    return {
        "max_time_steps": int(train_cfg.max_time_steps),
        "max_agents": _bucket_up(
            inferred_n,
            tuple(train_cfg.collate_agent_buckets),
            ceiling=int(train_cfg.max_agents),
        ),
        "max_map_features": _bucket_up(
            inferred_m,
            tuple(train_cfg.collate_map_feature_buckets),
            ceiling=int(train_cfg.max_map_features),
        ),
        "max_vectors_per_map_feature": int(train_cfg.max_vectors_per_map_feature),
        "max_traffic_lights": _bucket_up(
            inferred_l,
            tuple(train_cfg.collate_traffic_light_buckets),
            ceiling=int(train_cfg.max_traffic_lights),
        ),
    }


def _maybe_dump_relation_debug(
    *,
    prepared: Dict[str, Any],
    train_cfg: SupervisedTrainConfig,
    model_cfg: NNXBMTConfig,
    step: int,
    phase: str,
    dump_counter: int,
) -> int:
    dump_dir_cfg = str(train_cfg.relation_debug_dump_dir or "").strip()
    dump_every = int(train_cfg.relation_debug_dump_every_steps)
    dump_max = max(0, int(train_cfg.relation_debug_max_batches))
    if not dump_dir_cfg or dump_every <= 0 or dump_max <= 0:
        return dump_counter
    if dump_counter >= dump_max:
        return dump_counter
    if int(step) % dump_every != 0:
        return dump_counter

    raw = prepared["raw_batch"]
    scene_inputs = build_scene_token_relation_inputs_np(
        map_feature=np.asarray(raw["map_feature"], dtype=np.float32),
        map_feature_valid_mask=np.asarray(raw["map_feature_valid_mask"], dtype=bool),
        map_position=np.asarray(raw["map_position"], dtype=np.float32),
        traffic_light_feature=np.asarray(raw["traffic_light_feature"], dtype=np.float32),
        traffic_light_valid_mask=np.asarray(raw["traffic_light_valid_mask"], dtype=bool),
        traffic_light_position=np.asarray(raw["traffic_light_position"], dtype=np.float32),
        remove_traffic_light_state=bool(model_cfg.relation.remove_traffic_light_state),
        heading_placeholder=float(model_cfg.relation.heading_placeholder),
    )

    bundle_cfg = RelationBundleConfig(
        simple_relation=bool(model_cfg.relation.simple_relation),
        per_contour_point_relation=bool(model_cfg.relation.per_contour_point_relation),
        include_contour=True,
        heading_placeholder=float(model_cfg.relation.heading_placeholder),
        s2s_knn=model_cfg.relation.s2s_knn,
        s2s_distance=model_cfg.relation.s2s_distance,
        a2s_knn=model_cfg.relation.a2s_knn,
        a2s_distance=model_cfg.relation.a2s_distance,
        a2a_knn=model_cfg.relation.a2a_knn,
        a2a_distance=model_cfg.relation.a2a_distance,
        remove_traffic_light_state=bool(model_cfg.relation.remove_traffic_light_state),
    )

    bundle = build_relation_bundle(
        agent_position_xy=np.asarray(raw["agent_position_xy"], dtype=np.float32),
        agent_heading=np.asarray(raw["agent_heading"], dtype=np.float32),
        agent_valid_mask=np.asarray(raw["agent_valid_mask"], dtype=bool),
        decoder_valid_mask=np.asarray(prepared["model_inputs"]["input_action_valid_mask"], dtype=bool),
        agent_shape=np.asarray(raw["agent_shape"], dtype=np.float32),
        sample_steps=np.asarray(prepared["sample_steps"], dtype=np.int32),
        scene_position=scene_inputs["scene_position"],
        scene_heading=scene_inputs["scene_heading"],
        scene_valid_mask=scene_inputs["scene_valid_mask"],
        cfg=bundle_cfg,
    )

    dump_dir = Path(dump_dir_cfg)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{phase}_step_{int(step):07d}_dump_{int(dump_counter):03d}"
    npz_path = dump_dir / f"{stem}.npz"
    json_path = dump_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        scenario_ids=np.asarray(raw["scenario_ids"], dtype=object),
        sample_steps=np.asarray(prepared["sample_steps"], dtype=np.int32),
        scene_position=scene_inputs["scene_position"],
        scene_heading=scene_inputs["scene_heading"],
        scene_valid_mask=scene_inputs["scene_valid_mask"],
        **bundle,
    )
    meta = {
        "phase": str(phase),
        "step": int(step),
        "scenario_ids": [str(s) for s in raw["scenario_ids"]],
        "tokenizer_mode": str(train_cfg.tokenizer_mode),
        "model_preset": str(train_cfg.model_preset),
        "paths": {"npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved relation debug dump: {npz_path}")
    return dump_counter + 1


def _compute_metric_dict(
    *,
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
) -> Dict[str, jnp.ndarray]:
    loss = cross_entropy_token_loss(logits, targets, target_mask)
    accuracy = masked_token_accuracy(logits, targets, target_mask)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    token_entropy = -jnp.sum(probs * log_probs, axis=-1)
    entropy = _masked_mean_jax(token_entropy, target_mask)
    perplexity = jnp.exp(entropy)

    pred = jnp.argmax(logits, axis=-1)
    rate_default_gt = _masked_mean_jax((targets == default_token_id).astype(jnp.float32), target_mask)
    rate_default_pred = _masked_mean_jax((pred == default_token_id).astype(jnp.float32), target_mask)

    # Token-usage diagnostics akin to the original Adv-BMT logging style.
    onehot_pred = jax.nn.one_hot(pred, logits.shape[-1])
    weighted_pred = onehot_pred * target_mask[..., None]
    sum_pred = jnp.sum(weighted_pred, axis=(0, 1, 2))
    denom = jnp.maximum(1.0, jnp.sum(target_mask))
    pred_probs = sum_pred / denom
    pred_perplexity = jnp.exp(-jnp.sum(pred_probs * jnp.log(pred_probs + 1e-10)))
    cluster_use = jnp.sum(pred_probs > 0)

    onehot_gt = jax.nn.one_hot(targets, logits.shape[-1])
    weighted_gt = onehot_gt * target_mask[..., None]
    sum_gt = jnp.sum(weighted_gt, axis=(0, 1, 2))
    gt_probs = sum_gt / denom
    gt_perplexity = jnp.exp(-jnp.sum(gt_probs * jnp.log(gt_probs + 1e-10)))
    gt_cluster_use = jnp.sum(gt_probs > 0)

    # Forward/backward split diagnostics.
    rev = reverse_indicator[:, None, None].astype(bool)
    rev_f = rev.astype(jnp.float32)
    fwd_f = (~rev).astype(jnp.float32)

    mask_rev = target_mask * rev_f
    mask_fwd = target_mask * fwd_f

    acc_rev = _masked_mean_jax((pred == targets).astype(jnp.float32), mask_rev)
    acc_fwd = _masked_mean_jax((pred == targets).astype(jnp.float32), mask_fwd)

    picked = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, axis=-1), axis=-1).squeeze(-1)
    token_ce = -picked
    loss_rev = _masked_mean_jax(token_ce, mask_rev)
    loss_fwd = _masked_mean_jax(token_ce, mask_fwd)

    entropy_rev = _masked_mean_jax(token_entropy, mask_rev)
    entropy_fwd = _masked_mean_jax(token_entropy, mask_fwd)

    return {
        "total_loss": loss,
        "accuracy": accuracy,
        "entropy": entropy,
        "perplexity": pred_perplexity,
        "gt_perplexity": gt_perplexity,
        "cluster_use": cluster_use.astype(jnp.float32),
        "gt_cluster_use": gt_cluster_use.astype(jnp.float32),
        "rate_default_gt": rate_default_gt,
        "rate_default_pred": rate_default_pred,
        "num_trained_tokens": jnp.sum(target_mask),
        "accuracy_in_backward": acc_rev,
        "accuracy_in_forward": acc_fwd,
        "loss_in_backward": loss_rev,
        "loss_in_forward": loss_fwd,
        "entropy_in_backward": entropy_rev,
        "entropy_in_forward": entropy_fwd,
        "backward_ratio": jnp.mean(reverse_indicator.astype(jnp.float32)),
    }


@nnx.jit
def _train_step(
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
) -> Dict[str, jnp.ndarray]:
    def loss_fn(m: NNXBidirectionalMotionTransformer) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits = m(**model_inputs).astype(jnp.float32)
        metrics = _compute_metric_dict(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
            reverse_indicator=reverse_indicator,
            default_token_id=default_token_id,
        )
        return metrics["total_loss"], metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return metrics


@nnx.jit
def _eval_step(
    model: NNXBidirectionalMotionTransformer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
) -> Dict[str, jnp.ndarray]:
    logits = model(**model_inputs).astype(jnp.float32)
    return _compute_metric_dict(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
        reverse_indicator=reverse_indicator,
        default_token_id=default_token_id,
    )


@nnx.pmap(axis_name="data", in_axes=(None, None, 0, 0, 0, 0, None), out_axes=0)
def _train_step_pmap(
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
) -> Dict[str, jnp.ndarray]:
    def loss_fn(m: NNXBidirectionalMotionTransformer) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits = m(**model_inputs).astype(jnp.float32)
        metrics = _compute_metric_dict(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
            reverse_indicator=reverse_indicator,
            default_token_id=default_token_id,
        )
        return metrics["total_loss"], metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.lax.pmean(grads, axis_name="data")
    optimizer.update(grads)
    return jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="data"), metrics)


@nnx.pmap(axis_name="data", in_axes=(None, 0, 0, 0, 0, None), out_axes=0)
def _eval_step_pmap(
    model: NNXBidirectionalMotionTransformer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
) -> Dict[str, jnp.ndarray]:
    logits = model(**model_inputs).astype(jnp.float32)
    metrics = _compute_metric_dict(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
        reverse_indicator=reverse_indicator,
        default_token_id=default_token_id,
    )
    return jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="data"), metrics)


def _as_float_metrics(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        arr = np.asarray(jax.device_get(v))
        out[k] = float(np.mean(arr))
    return out


def _mean_metrics(list_of_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not list_of_metrics:
        return {}
    keys = list(list_of_metrics[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in list_of_metrics]))
    return out


def _legacy_cosine_zero_multiplier(
    step: jnp.ndarray | int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> jnp.ndarray:
    step_f = jnp.asarray(step, dtype=jnp.float32)
    warmup = float(max(1, int(warmup_steps)))
    total = float(max(int(total_steps), int(warmup_steps) + 1))
    warmup_mult = step_f / warmup
    progress = (step_f - warmup) / max(1.0, total - warmup)
    progress = jnp.clip(progress, 0.0, 1.0)
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step_f < warmup, warmup_mult, cosine_mult)


def _build_lr_schedule(
    train_cfg: SupervisedTrainConfig,
    total_steps_target: int,
) -> Tuple[Any, Dict[str, Any]]:
    mode = str(train_cfg.lr_schedule_mode)
    if mode == "legacy_cosine_zero":
        warmup_steps = max(1, int(train_cfg.warmup_steps))
        total_steps = max(int(total_steps_target), warmup_steps + 1)

        def _schedule(step: jnp.ndarray | int) -> jnp.ndarray:
            mult = _legacy_cosine_zero_multiplier(
                step,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
            )
            return jnp.asarray(float(train_cfg.learning_rate), dtype=jnp.float32) * mult

        meta = {
            "mode": "legacy_cosine_zero",
            "learning_rate": float(train_cfg.learning_rate),
            "warmup_steps": int(warmup_steps),
            "total_steps": int(total_steps),
            "min_learning_rate": 0.0,
        }
        return _schedule, meta

    decay_steps = int(max(int(total_steps_target), int(train_cfg.warmup_steps) + 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(train_cfg.learning_rate),
        warmup_steps=int(max(1, train_cfg.warmup_steps)),
        decay_steps=decay_steps,
        end_value=float(train_cfg.min_learning_rate),
    )
    meta = {
        "mode": "v2_cosine_minlr",
        "learning_rate": float(train_cfg.learning_rate),
        "warmup_steps": int(max(1, train_cfg.warmup_steps)),
        "total_steps": int(decay_steps),
        "min_learning_rate": float(train_cfg.min_learning_rate),
    }
    return schedule, meta


def _cast_tree_precision(tree: Any, *, precision: PrecisionType) -> Any:
    if precision != "bf16-mixed":
        return tree

    def _cast_leaf(x: Any) -> Any:
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x

    return jax.tree.map(_cast_leaf, tree)


def _shard_tree_for_pmap(tree: Any, *, num_devices: int) -> Any:
    def _reshape_leaf(x: Any) -> Any:
        if not isinstance(x, jnp.ndarray):
            return x
        if x.ndim == 0:
            raise ValueError("Cannot shard scalar input for pmap")
        batch = int(x.shape[0])
        if batch % int(num_devices) != 0:
            raise ValueError(
                f"Batch dimension {batch} must be divisible by num_devices={num_devices} for pmap training"
            )
        per_device = batch // int(num_devices)
        return x.reshape((int(num_devices), per_device) + x.shape[1:])

    return jax.tree.map(_reshape_leaf, tree)


def _hash_indices(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _prescan_cache_path(output_dir: Path) -> Path:
    return output_dir / "manifests" / "prescan_cache.pkl"


def _prescan_cache_store_root() -> Path:
    env = str(os.environ.get("COUNTER_BMT_PRESCAN_CACHE_DIR", "")).strip()
    if env:
        return Path(env)
    return Path("outputs") / "_prescan_cache"


def _prescan_cache_key_hash(cache_key: Dict[str, Any]) -> str:
    encoded = json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prescan_global_cache_path(cache_key: Dict[str, Any]) -> Path:
    return _prescan_cache_store_root() / f"{_prescan_cache_key_hash(cache_key)}.pkl"


def _build_prescan_cache_key(
    *,
    split_mode: str,
    resolved_dirs: Dict[str, str],
    train_indices_pre: np.ndarray,
    val_indices_pre: np.ndarray,
    strict_91_steps: bool,
    max_time_steps: int,
) -> Dict[str, Any]:
    return {
        "version": 1,
        "split_mode": str(split_mode),
        "resolved_dirs": dict(resolved_dirs),
        "train_indices_pre_hash": _hash_indices(np.asarray(train_indices_pre, dtype=np.int32)),
        "val_indices_pre_hash": _hash_indices(np.asarray(val_indices_pre, dtype=np.int32)),
        "train_indices_pre_len": int(len(train_indices_pre)),
        "val_indices_pre_len": int(len(val_indices_pre)),
        "strict_91_steps": bool(strict_91_steps),
        "max_time_steps": int(max_time_steps),
    }


def _extract_prescan_cache_data(payload: Any, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if payload.get("cache_key") != cache_key:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    required = (
        "train_indices",
        "val_indices",
        "train_manifest",
        "val_manifest",
        "skipped_records",
        "train_trunc_candidates",
        "val_trunc_candidates",
    )
    if any(k not in data for k in required):
        return None
    return data


def _load_prescan_cache_from_path(*, cache_path: Path, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
    return _extract_prescan_cache_data(payload, cache_key)


def _load_prescan_cache(*, output_dir: Path, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Prefer run-local cache for compatibility, then dataset-keyed global cache.
    local_path = _prescan_cache_path(output_dir)
    cached = _load_prescan_cache_from_path(cache_path=local_path, cache_key=cache_key)
    if cached is not None:
        return cached
    global_path = _prescan_global_cache_path(cache_key)
    return _load_prescan_cache_from_path(cache_path=global_path, cache_key=cache_key)


def _save_prescan_cache(*, output_dir: Path, cache_key: Dict[str, Any], data: Dict[str, Any]) -> None:
    payload = {"cache_key": cache_key, "data": data}
    # Write both run-local and global dataset-keyed caches.
    for cache_path in (_prescan_cache_path(output_dir), _prescan_global_cache_path(cache_key)):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(payload, f)
        except Exception:
            continue


def _assert_finite_metrics(metrics: Dict[str, float], *, phase: str, step: int) -> None:
    bad = [k for k, v in metrics.items() if not np.isfinite(float(v))]
    if bad:
        raise FloatingPointError(f"Non-finite metrics in {phase} step={step}: {bad}")


def _iter_minibatches(indices: np.ndarray, batch_size: int) -> Sequence[np.ndarray]:
    batches: List[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        batches.append(indices[start:start + batch_size])
    return batches


def _apply_interval(indices: np.ndarray, interval: int) -> np.ndarray:
    interval = int(interval)
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got: {interval}")
    return np.asarray(indices, dtype=np.int32)[::interval]


def _resolve_data_sources(
    train_cfg: SupervisedTrainConfig,
) -> Tuple[str, ScenarioNetNNXLoader, np.ndarray, ScenarioNetNNXLoader, np.ndarray, Dict[str, str]]:
    common_loader_kwargs = dict(
        max_agents=train_cfg.max_agents,
        max_map_features=train_cfg.max_map_features,
        max_vectors_per_map_feature=train_cfg.max_vectors_per_map_feature,
        max_traffic_lights=train_cfg.max_traffic_lights,
        center_to_map=train_cfg.center_to_map,
    )

    train_data_dir = str(train_cfg.train_data_dir).strip()
    val_data_dir = str(train_cfg.val_data_dir).strip()
    data_dir = str(train_cfg.data_dir).strip()

    use_explicit_split = bool(train_data_dir or val_data_dir)
    if use_explicit_split:
        if not train_data_dir or not val_data_dir:
            raise ValueError("Both train_data_dir and val_data_dir must be set when using explicit split mode")
        train_loader = ScenarioNetNNXLoader(data_dir=train_data_dir, **common_loader_kwargs)
        val_loader = ScenarioNetNNXLoader(data_dir=val_data_dir, **common_loader_kwargs)
        train_indices = np.arange(len(train_loader), dtype=np.int32)
        val_indices = np.arange(len(val_loader), dtype=np.int32)
        split_mode = "explicit_split"
    else:
        if not data_dir:
            raise ValueError("Provide data_dir (fallback split) or both train_data_dir and val_data_dir")
        loader = ScenarioNetNNXLoader(data_dir=data_dir, **common_loader_kwargs)
        all_indices = np.arange(len(loader), dtype=np.int32)

        split_rng = np.random.default_rng(train_cfg.seed)
        split_rng.shuffle(all_indices)

        split = int(round(len(all_indices) * float(np.clip(train_cfg.train_fraction, 0.0, 1.0))))
        split = max(1, min(split, len(all_indices)))
        train_indices = all_indices[:split]
        val_indices = all_indices[split:]

        train_loader = loader
        val_loader = loader
        split_mode = "fallback_split"

    train_indices = _apply_interval(train_indices, int(train_cfg.sample_interval_training))
    val_indices = _apply_interval(val_indices, int(train_cfg.sample_interval_test))

    if train_cfg.num_train_scenarios is not None:
        train_indices = train_indices[: int(train_cfg.num_train_scenarios)]
    if train_cfg.num_val_scenarios is not None:
        val_indices = val_indices[: int(train_cfg.num_val_scenarios)]

    resolved_dirs = {
        "train_data_dir": str(train_loader.data_dir),
        "val_data_dir": str(val_loader.data_dir),
        "fallback_data_dir": data_dir,
    }
    return split_mode, train_loader, train_indices, val_loader, val_indices, resolved_dirs


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


class _PrescanProgressBar:
    """Animated terminal progress for slow dataset prescan stages."""

    _SPINNER = "|/-\\"

    def __init__(self, *, split_name: str, total: int, enabled: bool, min_interval_s: float = 0.2) -> None:
        self.split_name = str(split_name)
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.min_interval_s = max(0.05, float(min_interval_s))
        self._is_tty = bool(sys.stdout.isatty())
        self._start_t = time.time()
        self._last_render_t = 0.0
        self._frame = 0
        self._last_len = 0

    def update(self, *, done: int, kept: int, skipped: int, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_render_t) < self.min_interval_s:
            return

        self._last_render_t = now
        elapsed = max(1e-6, float(now - self._start_t))
        done_i = min(max(0, int(done)), self.total)
        pct = 100.0 * float(done_i) / float(self.total)
        rate = float(done_i) / elapsed
        eta_s = (self.total - done_i) / max(1e-6, rate)

        bar_w = 28
        fill = int(bar_w * done_i / float(self.total))
        if fill <= 0:
            bar = ">" + "." * (bar_w - 1)
        elif fill >= bar_w:
            bar = "=" * bar_w
        else:
            bar = "=" * (fill - 1) + ">" + "." * (bar_w - fill)

        spin = self._SPINNER[self._frame % len(self._SPINNER)]
        self._frame += 1
        msg = (
            f"[prescan:{self.split_name}] {spin} [{bar}] {done_i}/{self.total} ({pct:5.1f}%) "
            f"kept={int(kept)} skipped={int(skipped)} "
            f"rate={rate:6.1f}/s eta={eta_s:7.1f}s"
        )
        if self._is_tty:
            padded = msg.ljust(self._last_len)
            self._last_len = len(padded)
            print(f"\r{padded}", end="", flush=True)
        elif force or done_i >= self.total:
            print(msg, flush=True)

    def close(self) -> None:
        if self.enabled and self._is_tty:
            print("", flush=True)


def _prescan_indices(
    *,
    loader: ScenarioNetNNXLoader,
    indices: np.ndarray,
    split_name: str,
    strict_91_steps: bool,
    max_time_steps: int,
    log_every: int = 0,
    workers: int = 0,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[int] = []
    manifests: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    trunc_candidates: List[Dict[str, Any]] = []

    root = Path(loader.data_dir)
    total = int(len(indices))
    log_interval = max(0, int(log_every))
    n_workers = max(0, int(workers))
    if total == 0:
        return np.asarray(kept, dtype=np.int32), manifests, skipped, trunc_candidates

    progress = _PrescanProgressBar(
        split_name=split_name,
        total=total,
        enabled=(log_interval > 0),
    )
    if log_interval > 0:
        print(f"[prescan:{split_name}] start total={total} workers={max(1, n_workers)}", flush=True)

    def _inspect_index(i: int) -> Dict[str, Any]:
        i = int(i)
        file_path = loader.files[i]
        file_name = file_path.name
        rel_path = _safe_relative_path(file_path, root)
        base_record = {
            "split": split_name,
            "loader_index": i,
            "file_name": file_name,
            "relative_path": rel_path,
        }

        try:
            sample = loader.load(i)
        except Exception as exc:
            return {
                "loader_index": i,
                "base": base_record,
                "ok": False,
                "scenario_id": "",
                "raw_time_steps": -1,
                "reason": "load_error",
                "error": str(exc),
            }

        scenario_id = str(sample.scenario_id)
        raw_t = int(sample.agent_position_xy.shape[0])
        has_map = (
            sample.map_feature.shape[0] > 0
            and sample.map_feature_valid_mask.size > 0
            and bool(np.any(sample.map_feature_valid_mask))
        )
        return {
            "loader_index": i,
            "base": base_record,
            "ok": True,
            "scenario_id": scenario_id,
            "raw_time_steps": raw_t,
            "has_map": bool(has_map),
        }

    indices_arr = np.asarray(indices, dtype=np.int32)
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            probe_iter = pool.map(_inspect_index, indices_arr.tolist())
            for n_done, probe in enumerate(probe_iter, start=1):
                base_record = dict(probe["base"])
                scenario_id = str(probe.get("scenario_id", ""))
                raw_t = int(probe.get("raw_time_steps", -1))
                i = int(probe.get("loader_index", -1))

                if not bool(probe.get("ok", False)):
                    rec = dict(base_record)
                    rec["scenario_id"] = scenario_id
                    rec["raw_time_steps"] = raw_t
                    rec["reason"] = str(probe.get("reason", "load_error"))
                    if "error" in probe:
                        rec["error"] = str(probe.get("error"))
                    skipped.append(rec)
                elif not bool(probe.get("has_map", False)):
                    rec = dict(base_record)
                    rec["scenario_id"] = scenario_id
                    rec["raw_time_steps"] = raw_t
                    rec["reason"] = "no_map_feature"
                    skipped.append(rec)
                elif strict_91_steps and raw_t != 91:
                    rec = dict(base_record)
                    rec["scenario_id"] = scenario_id
                    rec["raw_time_steps"] = raw_t
                    rec["reason"] = "strict_91_mismatch"
                    rec["expected_time_steps"] = 91
                    skipped.append(rec)
                else:
                    kept.append(i)
                    manifests.append(
                        {
                            "rank": len(kept) - 1,
                            "loader_index": i,
                            "scenario_id": scenario_id,
                            "file_name": str(base_record["file_name"]),
                            "relative_path": str(base_record["relative_path"]),
                            "raw_time_steps": raw_t,
                        }
                    )
                    if raw_t > int(max_time_steps):
                        trunc_candidates.append(
                            {
                                "split": split_name,
                                "loader_index": i,
                                "scenario_id": scenario_id,
                                "file_name": str(base_record["file_name"]),
                                "relative_path": str(base_record["relative_path"]),
                                "raw_time_steps": raw_t,
                                "max_time_steps": int(max_time_steps),
                            }
                        )
                progress.update(
                    done=n_done,
                    kept=len(kept),
                    skipped=len(skipped),
                    force=(n_done % max(1, log_interval) == 0 or n_done == total),
                )
    else:
        for n_done, i in enumerate(indices_arr.tolist(), start=1):
            probe = _inspect_index(int(i))
            base_record = dict(probe["base"])
            scenario_id = str(probe.get("scenario_id", ""))
            raw_t = int(probe.get("raw_time_steps", -1))

            if not bool(probe.get("ok", False)):
                rec = dict(base_record)
                rec["scenario_id"] = scenario_id
                rec["raw_time_steps"] = raw_t
                rec["reason"] = str(probe.get("reason", "load_error"))
                if "error" in probe:
                    rec["error"] = str(probe.get("error"))
                skipped.append(rec)
            elif not bool(probe.get("has_map", False)):
                rec = dict(base_record)
                rec["scenario_id"] = scenario_id
                rec["raw_time_steps"] = raw_t
                rec["reason"] = "no_map_feature"
                skipped.append(rec)
            elif strict_91_steps and raw_t != 91:
                rec = dict(base_record)
                rec["scenario_id"] = scenario_id
                rec["raw_time_steps"] = raw_t
                rec["reason"] = "strict_91_mismatch"
                rec["expected_time_steps"] = 91
                skipped.append(rec)
            else:
                kept.append(int(i))
                manifests.append(
                    {
                        "rank": len(kept) - 1,
                        "loader_index": int(i),
                        "scenario_id": scenario_id,
                        "file_name": str(base_record["file_name"]),
                        "relative_path": str(base_record["relative_path"]),
                        "raw_time_steps": raw_t,
                    }
                )
                if raw_t > int(max_time_steps):
                    trunc_candidates.append(
                        {
                            "split": split_name,
                            "loader_index": int(i),
                            "scenario_id": scenario_id,
                            "file_name": str(base_record["file_name"]),
                            "relative_path": str(base_record["relative_path"]),
                            "raw_time_steps": raw_t,
                            "max_time_steps": int(max_time_steps),
                        }
                    )
            progress.update(
                done=n_done,
                kept=len(kept),
                skipped=len(skipped),
                force=(n_done % max(1, log_interval) == 0 or n_done == total),
            )

    progress.close()

    return np.asarray(kept, dtype=np.int32), manifests, skipped, trunc_candidates


def _write_split_artifacts(
    *,
    output_dir: Path,
    train_manifest: Sequence[Dict[str, Any]],
    val_manifest: Sequence[Dict[str, Any]],
    skipped_records: Sequence[Dict[str, Any]],
    truncation_report: Dict[str, Any],
) -> Dict[str, str]:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_path = manifest_dir / "train_manifest.json"
    val_manifest_path = manifest_dir / "val_manifest.json"
    train_ids_path = manifest_dir / "train_ids.txt"
    val_ids_path = manifest_dir / "val_ids.txt"
    skipped_path = manifest_dir / "skipped_scenarios.json"
    trunc_path = manifest_dir / "truncation_report.json"

    train_manifest_path.write_text(json.dumps(list(train_manifest), indent=2), encoding="utf-8")
    val_manifest_path.write_text(json.dumps(list(val_manifest), indent=2), encoding="utf-8")
    train_ids_path.write_text("\n".join(str(x["scenario_id"]) for x in train_manifest) + ("\n" if train_manifest else ""), encoding="utf-8")
    val_ids_path.write_text("\n".join(str(x["scenario_id"]) for x in val_manifest) + ("\n" if val_manifest else ""), encoding="utf-8")
    skipped_path.write_text(json.dumps(list(skipped_records), indent=2), encoding="utf-8")
    trunc_path.write_text(json.dumps(truncation_report, indent=2), encoding="utf-8")

    return {
        "manifest_dir": str(manifest_dir),
        "train_manifest": str(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "train_ids": str(train_ids_path),
        "val_ids": str(val_ids_path),
        "skipped_scenarios": str(skipped_path),
        "truncation_report": str(trunc_path),
    }


def _save_checkpoint(
    *,
    output_dir: Path,
    train_step: int,
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
    train_cfg: SupervisedTrainConfig,
    model_cfg: NNXBMTConfig,
    latest_metrics: Dict[str, float],
    runtime_state: Optional[Dict[str, Any]] = None,
) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "train_step": int(train_step),
        "model_state": jax.device_get(nnx.state(model)),
        "opt_state": jax.device_get(optimizer.opt_state),
        "optimizer_step": int(np.asarray(jax.device_get(optimizer.step.value))),
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "latest_metrics": latest_metrics,
        "runtime_state": runtime_state or {},
    }

    step_path = ckpt_dir / f"step_{train_step:07d}.pkl"
    with step_path.open("wb") as f:
        pickle.dump(payload, f)

    # Keep a stable latest pointer for simple resume semantics.
    last_path = ckpt_dir / "last.pkl"
    with last_path.open("wb") as f:
        pickle.dump(payload, f)

    return step_path


def _load_checkpoint(
    *,
    checkpoint_path: Path,
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
    with checkpoint_path.open("rb") as f:
        payload = pickle.load(f)

    nnx.update(model, payload["model_state"])
    optimizer.opt_state = payload["opt_state"]
    optimizer.step.value = jnp.asarray(payload["optimizer_step"], dtype=optimizer.step.value.dtype)

    runtime_state = payload.get("runtime_state", {})
    if not isinstance(runtime_state, dict):
        runtime_state = {}
    return int(payload["train_step"]), runtime_state, payload


def _validate_resume_compatibility(
    *,
    train_cfg: SupervisedTrainConfig,
    split_hashes: Dict[str, str],
    resume_runtime_state: Dict[str, Any],
    resume_payload: Dict[str, Any],
) -> None:
    ckpt_split_hashes = resume_runtime_state.get("split_hashes", {})
    if ckpt_split_hashes and ckpt_split_hashes != split_hashes:
        raise ValueError(
            "Resume strict determinism failed: split hashes differ between checkpoint and current run "
            f"({ckpt_split_hashes} vs {split_hashes})"
        )

    ckpt_train_cfg = resume_payload.get("train_cfg", {})
    for k in (
        "model_preset",
        "tokenizer_mode",
        "skip_steps",
        "precision",
        "collate_padding_mode",
        "decoder_edge_sparse_attn",
    ):
        if k in ckpt_train_cfg and ckpt_train_cfg[k] != getattr(train_cfg, k):
            raise ValueError(
                f"Resume strict determinism failed: train_cfg[{k}] mismatch "
                f"({ckpt_train_cfg[k]} vs {getattr(train_cfg, k)})"
            )


def _evaluate(
    *,
    model: NNXBidirectionalMotionTransformer,
    model_cfg: NNXBMTConfig,
    loader: ScenarioNetNNXLoader,
    val_indices: np.ndarray,
    train_cfg: SupervisedTrainConfig,
    tokenizer: Any,
    default_token_id: int,
    rng: np.random.Generator,
    output_dir: Path | None = None,
    global_step: int = 0,
) -> Dict[str, float]:
    if len(val_indices) == 0:
        return {}

    val_batches = _iter_minibatches(val_indices, train_cfg.batch_size)
    if train_cfg.eval_batches > 0:
        val_batches = val_batches[: train_cfg.eval_batches]

    metrics_list: List[Dict[str, float]] = []
    forward_metrics_list: List[Dict[str, float]] = []
    forward_viz_saved = 0
    forward_artifact_saved = 0
    viz_remaining = max(0, int(train_cfg.forward_eval.viz_max_scenarios))
    artifact_remaining = max(0, int(train_cfg.forward_eval.artifact_max_scenarios_per_eval))
    for idx_batch in val_batches:
        samples = [loader.load(int(i)) for i in idx_batch]
        prepared = _prepare_supervised_batch(
            samples,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            rng=rng,
            is_training=False,
        )

        metrics = _eval_step(
            model,
            prepared["model_inputs"],
            prepared["targets"],
            prepared["target_mask"],
            prepared["reverse_indicator"],
            default_token_id,
        )
        metrics_list.append(_as_float_metrics(metrics))

        if train_cfg.forward_eval.enabled:
            seed = int(rng.integers(low=0, high=2**31 - 1))
            batch_forward_metrics, batch_viz_saved, batch_artifact_saved = compute_forward_pass_metrics_for_batch(
                model=model,
                prepared_batch=prepared,
                tokenizer=tokenizer,
                skip_steps=train_cfg.skip_steps,
                eval_cfg=train_cfg.forward_eval,
                seed=seed,
                output_dir=output_dir,
                global_step=global_step,
                max_visualizations=viz_remaining,
                max_artifacts=artifact_remaining,
            )
            forward_metrics_list.extend(batch_forward_metrics)
            forward_viz_saved += int(batch_viz_saved)
            forward_artifact_saved += int(batch_artifact_saved)
            viz_remaining = max(0, viz_remaining - int(batch_viz_saved))
            artifact_remaining = max(0, artifact_remaining - int(batch_artifact_saved))

    merged = _mean_metrics(metrics_list)
    if forward_metrics_list:
        forward_avg = nanmean_metrics(forward_metrics_list)
        merged.update({f"forward_approx/{k}": v for k, v in forward_avg.items()})
        merged["forward_approx/scenario_count"] = float(len(forward_metrics_list))
        merged["forward_approx/visualizations_saved"] = float(forward_viz_saved)
        merged["forward_approx/artifacts_saved"] = float(forward_artifact_saved)

    return merged


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _print_metrics(prefix: str, step: int, metrics: Dict[str, float], lr: float, elapsed_s: float) -> None:
    forward_sfde = metrics.get("forward_approx/sfde_min", float("nan"))
    forward_fdd = metrics.get("forward_approx/fdd", float("nan"))
    forward_vel_jsd = metrics.get("forward_approx/vel_jsd", float("nan"))
    msg = (
        f"{prefix} step={step} "
        f"loss={metrics.get('total_loss', float('nan')):.4f} "
        f"acc={metrics.get('accuracy', float('nan')):.4f} "
        f"entropy={metrics.get('entropy', float('nan')):.4f} "
        f"ppl={metrics.get('perplexity', float('nan')):.2f} "
        f"sfde={forward_sfde:.3f} "
        f"fdd={forward_fdd:.3f} "
        f"vel_jsd={forward_vel_jsd:.3f} "
        f"tokens={metrics.get('num_trained_tokens', 0.0):.1f} "
        f"sps={metrics.get('train/steps_per_sec', float('nan')):.2f} "
        f"tps={metrics.get('train/tokens_per_sec', float('nan')):.1f} "
        f"lr={lr:.6f} "
        f"t={elapsed_s:.2f}s"
    )
    print(msg, flush=True)


def train_supervised(train_cfg: SupervisedTrainConfig) -> Dict[str, Any]:
    """Run supervised NNX motion-token training.

    Returns a summary dict with final metrics and paths.
    """
    if train_cfg.mode not in ("forward", "reverse", "mixed"):
        raise ValueError(f"Unsupported mode: {train_cfg.mode}")
    if train_cfg.tokenizer_mode not in ("paper_simple", "adv_bmt_parity"):
        raise ValueError(f"Unsupported tokenizer_mode: {train_cfg.tokenizer_mode}")
    if train_cfg.distributed_backend not in ("none", "pmap"):
        raise ValueError(f"Unsupported distributed_backend: {train_cfg.distributed_backend}")
    if train_cfg.precision not in ("fp32", "bf16-mixed"):
        raise ValueError(f"Unsupported precision: {train_cfg.precision}")
    if train_cfg.lr_schedule_mode not in ("v2_cosine_minlr", "legacy_cosine_zero"):
        raise ValueError(f"Unsupported lr_schedule_mode: {train_cfg.lr_schedule_mode}")

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(train_cfg.sample_interval_training) < 1:
        raise ValueError(f"sample_interval_training must be >= 1, got {train_cfg.sample_interval_training}")
    if int(train_cfg.sample_interval_test) < 1:
        raise ValueError(f"sample_interval_test must be >= 1, got {train_cfg.sample_interval_test}")

    model_cfg = _resolve_model_preset(train_cfg.model_preset)
    if bool(getattr(train_cfg, "decoder_edge_sparse_attn", False)):
        model_cfg.decoder.edge_sparse_relation_attn = True
    if train_cfg.tokenizer_mode == "adv_bmt_parity":
        tokenizer = AdvBMTParityTokenizer(
            ParityTokenizerConfig(num_skipped_steps=int(train_cfg.skip_steps))
        )
        default_token_id = int(tokenizer.default_token_id)
    else:
        tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)
        default_token_id = int(tokenizer.action_to_token(0.0, 0.0))

    split_mode, train_loader, train_indices, val_loader, val_indices, resolved_dirs = _resolve_data_sources(train_cfg)
    train_size_pre_filter = int(len(train_indices))
    val_size_pre_filter = int(len(val_indices))
    train_indices_pre = np.asarray(train_indices, dtype=np.int32).copy()
    val_indices_pre = np.asarray(val_indices, dtype=np.int32).copy()
    prescan_cache_key = _build_prescan_cache_key(
        split_mode=split_mode,
        resolved_dirs=resolved_dirs,
        train_indices_pre=train_indices_pre,
        val_indices_pre=val_indices_pre,
        strict_91_steps=bool(train_cfg.strict_91_steps),
        max_time_steps=int(train_cfg.max_time_steps),
    )

    cached_prescan = None
    if bool(train_cfg.use_prescan_cache):
        cached_prescan = _load_prescan_cache(output_dir=output_dir, cache_key=prescan_cache_key)

    if cached_prescan is not None:
        train_indices = np.asarray(cached_prescan["train_indices"], dtype=np.int32)
        val_indices = np.asarray(cached_prescan["val_indices"], dtype=np.int32)
        train_manifest = list(cached_prescan["train_manifest"])
        val_manifest = list(cached_prescan["val_manifest"])
        skipped_records = list(cached_prescan["skipped_records"])
        train_trunc_candidates = list(cached_prescan["train_trunc_candidates"])
        val_trunc_candidates = list(cached_prescan["val_trunc_candidates"])
        print(
            "[prescan] loaded cache "
            f"train={len(train_indices)} val={len(val_indices)} skipped={len(skipped_records)}"
        )
    else:
        train_indices, train_manifest, train_skipped, train_trunc_candidates = _prescan_indices(
            loader=train_loader,
            indices=train_indices,
            split_name="train",
            strict_91_steps=bool(train_cfg.strict_91_steps),
            max_time_steps=int(train_cfg.max_time_steps),
            log_every=int(train_cfg.prescan_log_every),
            workers=int(train_cfg.prescan_workers),
        )
        val_indices, val_manifest, val_skipped, val_trunc_candidates = _prescan_indices(
            loader=val_loader,
            indices=val_indices,
            split_name="val",
            strict_91_steps=bool(train_cfg.strict_91_steps),
            max_time_steps=int(train_cfg.max_time_steps),
            log_every=int(train_cfg.prescan_log_every),
            workers=int(train_cfg.prescan_workers),
        )
        skipped_records = [*train_skipped, *val_skipped]

        if bool(train_cfg.use_prescan_cache):
            _save_prescan_cache(
                output_dir=output_dir,
                cache_key=prescan_cache_key,
                data={
                    "train_indices": np.asarray(train_indices, dtype=np.int32).tolist(),
                    "val_indices": np.asarray(val_indices, dtype=np.int32).tolist(),
                    "train_manifest": list(train_manifest),
                    "val_manifest": list(val_manifest),
                    "skipped_records": list(skipped_records),
                    "train_trunc_candidates": list(train_trunc_candidates),
                    "val_trunc_candidates": list(val_trunc_candidates),
                },
            )
            print(
                "[prescan] saved cache "
                f"train={len(train_indices)} val={len(val_indices)} skipped={len(skipped_records)}"
            )

    strict_violations = [x for x in skipped_records if str(x.get("reason")) == "strict_91_mismatch"]
    skip_reason_counts = dict(Counter(str(x.get("reason", "unknown")) for x in skipped_records))

    truncation_report = {
        "strict_91_steps": bool(train_cfg.strict_91_steps),
        "max_time_steps": int(train_cfg.max_time_steps),
        "train_selected_pre_filter": train_size_pre_filter,
        "val_selected_pre_filter": val_size_pre_filter,
        "train_selected_post_filter": int(len(train_indices)),
        "val_selected_post_filter": int(len(val_indices)),
        "num_skipped_total": int(len(skipped_records)),
        "skip_reason_counts": skip_reason_counts,
        "num_strict_91_mismatch": int(len(strict_violations)),
        "num_truncated_candidates": int(len(train_trunc_candidates) + len(val_trunc_candidates)),
        "truncated_candidates_examples": [*train_trunc_candidates, *val_trunc_candidates][:100],
    }
    artifact_paths = _write_split_artifacts(
        output_dir=output_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        skipped_records=skipped_records,
        truncation_report=truncation_report,
    )

    if bool(train_cfg.strict_91_steps) and strict_violations:
        raise ValueError(
            "strict_91_steps enabled but non-91 scenarios were found. "
            f"See report: {artifact_paths['truncation_report']}"
        )

    if len(train_indices) == 0:
        raise ValueError("No training scenarios available after split/config filtering")

    split_hashes = {
        "train": _hash_indices(train_indices),
        "val": _hash_indices(val_indices),
    }

    num_devices = max(1, len(jax.local_devices()))
    if train_cfg.distributed_backend == "pmap":
        if train_cfg.batch_size % num_devices != 0:
            raise ValueError(
                f"batch_size ({train_cfg.batch_size}) must be divisible by num_devices ({num_devices}) for pmap"
            )

    steps_per_epoch = int(math.ceil(len(train_indices) / float(train_cfg.batch_size)))
    total_steps_target = (
        int(train_cfg.max_steps)
        if train_cfg.max_steps is not None
        else int(train_cfg.num_epochs) * steps_per_epoch
    )
    total_steps_target = max(1, total_steps_target)

    lr_schedule, lr_schedule_meta = _build_lr_schedule(train_cfg, total_steps_target)

    adamw_kwargs: Dict[str, Any] = {
        "learning_rate": lr_schedule,
        "weight_decay": float(train_cfg.weight_decay),
        "b1": 0.9,
        "b2": 0.95,
        "eps": 1e-5,
    }
    if train_cfg.precision == "bf16-mixed":
        adamw_kwargs["mu_dtype"] = jnp.float32

    tx = optax.chain(
        optax.clip_by_global_norm(float(train_cfg.grad_clip_norm)),
        optax.adamw(**adamw_kwargs),
    )

    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
    optimizer = nnx.Optimizer(model, tx)

    start_step = 0
    resume_runtime_state: Dict[str, Any] = {}
    resume_payload: Dict[str, Any] = {}
    if train_cfg.resume_checkpoint:
        ckpt_path = Path(train_cfg.resume_checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.pkl"
        if ckpt_path.is_file():
            start_step, resume_runtime_state, resume_payload = _load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
            )
            print(f"Resumed checkpoint: {ckpt_path} (step={start_step})")

            if bool(train_cfg.resume_strict_determinism):
                _validate_resume_compatibility(
                    train_cfg=train_cfg,
                    split_hashes=split_hashes,
                    resume_runtime_state=resume_runtime_state,
                    resume_payload=resume_payload,
                )
            else:
                print(
                    "Warning: resume strict determinism disabled; allowing split/config mismatch for this resumed run."
                )

    run_meta = {
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "runtime_preset": str(train_cfg.runtime_preset),
        "runtime_resolved_overrides": dict(train_cfg.runtime_resolved_overrides),
        "data_source_mode": split_mode,
        "resolved_data_dirs": resolved_dirs,
        "split_hashes": split_hashes,
        "distributed": {
            "backend": str(train_cfg.distributed_backend),
            "num_devices": int(num_devices),
        },
        "precision": str(train_cfg.precision),
        "lr_schedule": lr_schedule_meta,
        "split_settings": {
            "train_fraction": float(train_cfg.train_fraction),
            "sample_interval_training": int(train_cfg.sample_interval_training),
            "sample_interval_test": int(train_cfg.sample_interval_test),
            "strict_91_steps": bool(train_cfg.strict_91_steps),
            "prescan_log_every": int(train_cfg.prescan_log_every),
            "prescan_workers": int(train_cfg.prescan_workers),
            "use_prescan_cache": bool(train_cfg.use_prescan_cache),
        },
        "prescan_cache": {
            "enabled": bool(train_cfg.use_prescan_cache),
            "cache_hit": bool(cached_prescan is not None),
            "cache_local_path": str(_prescan_cache_path(output_dir)),
            "cache_global_path": str(_prescan_global_cache_path(prescan_cache_key)),
            "cache_key_hash": str(_prescan_cache_key_hash(prescan_cache_key)),
            "cache_key": prescan_cache_key,
        },
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "train_size_pre_filter": train_size_pre_filter,
        "val_size_pre_filter": val_size_pre_filter,
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps_target": int(total_steps_target),
        "start_step": int(start_step),
        "artifacts": artifact_paths,
        "skip_reason_counts": skip_reason_counts,
        "num_strict_91_mismatch": int(len(strict_violations)),
        "num_truncated_candidates": int(len(train_trunc_candidates) + len(val_trunc_candidates)),
        "forward_metric_namespaces": ["forward_approx"],
        "forward_artifact_export": {
            "enabled": bool(train_cfg.forward_eval.export_artifacts),
            "subdir": str(train_cfg.forward_eval.artifact_output_subdir),
            "max_scenarios_per_eval": int(train_cfg.forward_eval.artifact_max_scenarios_per_eval),
            "metric_scope": str(train_cfg.forward_eval.metric_scope),
        },
        "tensorboard": {
            "enabled": bool(train_cfg.enable_tensorboard),
            "log_dir": str(output_dir / str(train_cfg.tensorboard_subdir)),
            "flush_secs": int(train_cfg.tensorboard_flush_secs),
            "log_run_config": bool(train_cfg.tensorboard_log_run_config),
        },
        "created_at": int(time.time()),
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    tb_writer = create_tb_writer(
        output_dir=output_dir,
        subdir=str(train_cfg.tensorboard_subdir),
        enabled=bool(train_cfg.enable_tensorboard),
        flush_secs=int(train_cfg.tensorboard_flush_secs),
    )
    if tb_writer is not None:
        atexit.register(tb_close, tb_writer)
    if bool(train_cfg.tensorboard_log_run_config):
        tb_write_text(tb_writer, "run/config", json.dumps(run_meta, indent=2), step=0)

    metrics_log_path = output_dir / "metrics.jsonl"

    global_step = int(start_step)
    train_rng = np.random.default_rng(train_cfg.seed + 1)
    if bool(train_cfg.save_rng_state) and resume_runtime_state.get("train_rng_state") is not None:
        train_rng.bit_generator.state = resume_runtime_state["train_rng_state"]

    best_eval_loss = float("inf")
    best_eval_step = -1
    relation_dump_counter = 0
    epoch = int(resume_runtime_state.get("epoch", 0))
    batch_cursor = int(resume_runtime_state.get("batch_cursor_in_epoch", 0))
    epoch_indices_saved = resume_runtime_state.get("epoch_indices", None)
    epoch_indices: Optional[np.ndarray]
    if epoch_indices_saved is None:
        epoch_indices = None
    else:
        epoch_indices = np.asarray(epoch_indices_saved, dtype=np.int32)
        if len(epoch_indices) != len(train_indices):
            if bool(train_cfg.resume_strict_determinism):
                raise ValueError(
                    "Resume strict determinism failed: checkpoint epoch_indices length does not match current train split"
                )
            epoch_indices = None
            batch_cursor = 0

    t0 = time.time()
    def _runtime_state() -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_cursor_in_epoch": int(batch_cursor),
            "epoch_indices": None if epoch_indices is None else np.asarray(epoch_indices, dtype=np.int32).tolist(),
            "train_rng_state": train_rng.bit_generator.state if bool(train_cfg.save_rng_state) else None,
            "split_hashes": dict(split_hashes),
            "split_sizes": {"train": int(len(train_indices)), "val": int(len(val_indices))},
            "distributed_backend": str(train_cfg.distributed_backend),
            "precision": str(train_cfg.precision),
            "lr_schedule_mode": str(train_cfg.lr_schedule_mode),
            "runtime_preset": str(train_cfg.runtime_preset),
        }

    while epoch < int(train_cfg.num_epochs) and global_step < total_steps_target:
        if epoch_indices is None:
            epoch_indices = train_indices.copy()
            train_rng.shuffle(epoch_indices)
            batch_cursor = 0

        if batch_cursor >= len(epoch_indices):
            epoch += 1
            epoch_indices = None
            batch_cursor = 0
            continue

        idx_batch = epoch_indices[batch_cursor: batch_cursor + int(train_cfg.batch_size)]
        batch_cursor += int(len(idx_batch))

        samples = [train_loader.load(int(i)) for i in idx_batch]
        prepared = _prepare_supervised_batch(
            samples,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            rng=train_rng,
            is_training=True,
        )
        relation_dump_counter = _maybe_dump_relation_debug(
            prepared=prepared,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            step=global_step + 1,
            phase="train",
            dump_counter=relation_dump_counter,
        )

        step_start = time.time()
        if train_cfg.distributed_backend == "pmap":
            metrics = _train_step_pmap(
                model,
                optimizer,
                _shard_tree_for_pmap(prepared["model_inputs"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["targets"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["target_mask"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["reverse_indicator"], num_devices=num_devices),
                default_token_id,
            )
        else:
            metrics = _train_step(
                model,
                optimizer,
                prepared["model_inputs"],
                prepared["targets"],
                prepared["target_mask"],
                prepared["reverse_indicator"],
                default_token_id,
            )

        global_step += 1
        lr_now = float(np.asarray(jax.device_get(lr_schedule(global_step))))
        metrics_f = _as_float_metrics(metrics)
        if train_cfg.distributed_backend == "pmap":
            metrics_f["num_trained_tokens"] = float(metrics_f.get("num_trained_tokens", 0.0) * num_devices)
        step_dt = max(1e-6, float(time.time() - step_start))
        metrics_f["train/steps_per_sec"] = float(1.0 / step_dt)
        metrics_f["train/tokens_per_sec"] = float(metrics_f.get("num_trained_tokens", 0.0) / step_dt)
        metrics_f["train/global_batch_size"] = float(train_cfg.batch_size)
        metrics_f["train/num_devices"] = float(num_devices)
        _assert_finite_metrics(metrics_f, phase="train", step=global_step)

        _write_jsonl(
            metrics_log_path,
            {
                "phase": "train",
                "step": global_step,
                "epoch": epoch,
                "batch_cursor_in_epoch": int(batch_cursor),
                "lr": lr_now,
                "metrics": metrics_f,
            },
        )
        tb_write_scalar(tb_writer, "train/lr", lr_now, global_step)
        tb_write_scalars(tb_writer, "train", metrics_f, global_step)

        if global_step % max(1, train_cfg.log_every_steps) == 0:
            _print_metrics(
                prefix="train",
                step=global_step,
                metrics=metrics_f,
                lr=lr_now,
                elapsed_s=time.time() - t0,
            )

        if len(val_indices) > 0 and global_step % max(1, train_cfg.eval_every_steps) == 0:
            eval_metrics = _evaluate(
                model=model,
                model_cfg=model_cfg,
                loader=val_loader,
                val_indices=val_indices,
                train_cfg=train_cfg,
                tokenizer=tokenizer,
                default_token_id=default_token_id,
                rng=train_rng,
                output_dir=output_dir,
                global_step=global_step,
            )
            _assert_finite_metrics(eval_metrics, phase="eval", step=global_step)

            _write_jsonl(
                metrics_log_path,
                {
                    "phase": "eval",
                    "step": global_step,
                    "epoch": epoch,
                    "batch_cursor_in_epoch": int(batch_cursor),
                    "lr": lr_now,
                    "metrics": eval_metrics,
                },
            )
            tb_write_scalars(tb_writer, "eval", eval_metrics, global_step)
            _print_metrics(
                prefix="eval ",
                step=global_step,
                metrics=eval_metrics,
                lr=lr_now,
                elapsed_s=time.time() - t0,
            )

            eval_loss = float(eval_metrics.get("total_loss", float("inf")))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = global_step
                best_path = _save_checkpoint(
                    output_dir=output_dir,
                    train_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    latest_metrics=eval_metrics,
                    runtime_state=_runtime_state(),
                )
                print(f"Saved improved checkpoint: {best_path}")
                tb_write_scalar(tb_writer, "events/checkpoint_saved", 1.0, global_step)

        if global_step % max(1, train_cfg.checkpoint_every_steps) == 0:
            ckpt_path = _save_checkpoint(
                output_dir=output_dir,
                train_step=global_step,
                model=model,
                optimizer=optimizer,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                latest_metrics=metrics_f,
                runtime_state=_runtime_state(),
            )
            print(f"Saved checkpoint: {ckpt_path}")
            tb_write_scalar(tb_writer, "events/checkpoint_saved", 1.0, global_step)

    # Final checkpoint and summary.
    final_metrics = _evaluate(
        model=model,
        model_cfg=model_cfg,
        loader=val_loader,
        val_indices=val_indices,
        train_cfg=train_cfg,
        tokenizer=tokenizer,
        default_token_id=default_token_id,
        rng=train_rng,
        output_dir=output_dir,
        global_step=global_step,
    ) if len(val_indices) > 0 else {}
    _assert_finite_metrics(final_metrics, phase="final_eval", step=global_step)
    tb_write_scalars(tb_writer, "final_eval", final_metrics, global_step)

    final_ckpt = _save_checkpoint(
        output_dir=output_dir,
        train_step=global_step,
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        latest_metrics=final_metrics,
        runtime_state=_runtime_state(),
    )
    tb_write_scalar(tb_writer, "events/checkpoint_saved", 1.0, global_step)

    summary = {
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_ckpt),
        "total_steps": int(global_step),
        "distributed_backend": str(train_cfg.distributed_backend),
        "num_devices": int(num_devices),
        "precision": str(train_cfg.precision),
        "lr_schedule_mode": str(train_cfg.lr_schedule_mode),
        "data_source_mode": split_mode,
        "train_size": int(len(train_indices)),
        "val_size": int(len(val_indices)),
        "best_eval_loss": float(best_eval_loss),
        "best_eval_step": int(best_eval_step),
        "final_eval_metrics": final_metrics,
        "artifacts": artifact_paths,
        "elapsed_seconds": float(time.time() - t0),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if bool(train_cfg.tensorboard_log_run_config):
        tb_write_text(tb_writer, "run/summary", json.dumps(summary, indent=2), step=global_step)
    tb_close(tb_writer)

    return summary
