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

import json
import math
import pickle
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

    eval_every_steps: int = 100
    eval_batches: int = 10
    log_every_steps: int = 10
    checkpoint_every_steps: int = 200

    # Fixed collate shapes prevent recompilation churn and keep JIT stable.
    max_time_steps: int = 91
    max_agents: int = 128
    max_map_features: int = 512
    max_vectors_per_map_feature: int = 128
    max_traffic_lights: int = 64

    center_to_map: bool = True
    resume_checkpoint: str = ""
    relation_debug_dump_dir: str = ""
    relation_debug_dump_every_steps: int = 0
    relation_debug_max_batches: int = 1

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
    batch = collate_nnx_scene_samples(
        samples,
        max_time_steps=train_cfg.max_time_steps,
        max_agents=train_cfg.max_agents,
        max_map_features=train_cfg.max_map_features,
        max_vectors_per_map_feature=train_cfg.max_vectors_per_map_feature,
        max_traffic_lights=train_cfg.max_traffic_lights,
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
    if "input_mask" in token_batch:
        model_inputs["input_action_valid_mask"] = jnp.asarray(token_batch["input_mask"], dtype=bool)
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
            decoder_valid_mask=np.asarray(token_batch["input_mask"], dtype=bool),
            agent_shape=np.asarray(batch["agent_shape"], dtype=np.float32),
            sample_steps=np.asarray(token_batch["sample_steps"], dtype=np.int32),
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
            }
        )

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
        logits = m(**model_inputs)
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
    logits = model(**model_inputs)
    return _compute_metric_dict(
        logits=logits,
        targets=targets,
        target_mask=target_mask,
        reverse_indicator=reverse_indicator,
        default_token_id=default_token_id,
    )


def _as_float_metrics(metrics: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        out[k] = float(np.asarray(jax.device_get(v)))
    return out


def _mean_metrics(list_of_metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not list_of_metrics:
        return {}
    keys = list(list_of_metrics[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in list_of_metrics]))
    return out


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


def _prescan_indices(
    *,
    loader: ScenarioNetNNXLoader,
    indices: np.ndarray,
    split_name: str,
    strict_91_steps: bool,
    max_time_steps: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[int] = []
    manifests: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    trunc_candidates: List[Dict[str, Any]] = []

    root = Path(loader.data_dir)
    for idx in np.asarray(indices, dtype=np.int32):
        i = int(idx)
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
            rec = dict(base_record)
            rec["scenario_id"] = ""
            rec["raw_time_steps"] = -1
            rec["reason"] = "load_error"
            rec["error"] = str(exc)
            skipped.append(rec)
            continue

        scenario_id = str(sample.scenario_id)
        raw_t = int(sample.agent_position_xy.shape[0])
        has_map = (
            sample.map_feature.shape[0] > 0
            and sample.map_feature_valid_mask.size > 0
            and bool(np.any(sample.map_feature_valid_mask))
        )

        if not has_map:
            rec = dict(base_record)
            rec["scenario_id"] = scenario_id
            rec["raw_time_steps"] = raw_t
            rec["reason"] = "no_map_feature"
            skipped.append(rec)
            continue

        if strict_91_steps and raw_t != 91:
            rec = dict(base_record)
            rec["scenario_id"] = scenario_id
            rec["raw_time_steps"] = raw_t
            rec["reason"] = "strict_91_mismatch"
            rec["expected_time_steps"] = 91
            skipped.append(rec)
            continue

        kept.append(i)
        manifests.append(
            {
                "rank": len(kept) - 1,
                "loader_index": i,
                "scenario_id": scenario_id,
                "file_name": file_name,
                "relative_path": rel_path,
                "raw_time_steps": raw_t,
            }
        )

        if raw_t > int(max_time_steps):
            trunc_candidates.append(
                {
                    "split": split_name,
                    "loader_index": i,
                    "scenario_id": scenario_id,
                    "file_name": file_name,
                    "relative_path": rel_path,
                    "raw_time_steps": raw_t,
                    "max_time_steps": int(max_time_steps),
                }
            )

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
) -> int:
    with checkpoint_path.open("rb") as f:
        payload = pickle.load(f)

    nnx.update(model, payload["model_state"])
    optimizer.opt_state = payload["opt_state"]
    optimizer.step.value = jnp.asarray(payload["optimizer_step"], dtype=optimizer.step.value.dtype)

    return int(payload["train_step"])


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
        f"lr={lr:.6f} "
        f"t={elapsed_s:.2f}s"
    )
    print(msg)


def train_supervised(train_cfg: SupervisedTrainConfig) -> Dict[str, Any]:
    """Run supervised NNX motion-token training.

    Returns a summary dict with final metrics and paths.
    """
    if train_cfg.mode not in ("forward", "reverse", "mixed"):
        raise ValueError(f"Unsupported mode: {train_cfg.mode}")
    if train_cfg.tokenizer_mode not in ("paper_simple", "adv_bmt_parity"):
        raise ValueError(f"Unsupported tokenizer_mode: {train_cfg.tokenizer_mode}")

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if int(train_cfg.sample_interval_training) < 1:
        raise ValueError(f"sample_interval_training must be >= 1, got {train_cfg.sample_interval_training}")
    if int(train_cfg.sample_interval_test) < 1:
        raise ValueError(f"sample_interval_test must be >= 1, got {train_cfg.sample_interval_test}")

    model_cfg = _resolve_model_preset(train_cfg.model_preset)
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

    train_indices, train_manifest, train_skipped, train_trunc_candidates = _prescan_indices(
        loader=train_loader,
        indices=train_indices,
        split_name="train",
        strict_91_steps=bool(train_cfg.strict_91_steps),
        max_time_steps=int(train_cfg.max_time_steps),
    )
    val_indices, val_manifest, val_skipped, val_trunc_candidates = _prescan_indices(
        loader=val_loader,
        indices=val_indices,
        split_name="val",
        strict_91_steps=bool(train_cfg.strict_91_steps),
        max_time_steps=int(train_cfg.max_time_steps),
    )

    skipped_records = [*train_skipped, *val_skipped]
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

    steps_per_epoch = int(math.ceil(len(train_indices) / float(train_cfg.batch_size)))
    total_steps_target = (
        int(train_cfg.max_steps)
        if train_cfg.max_steps is not None
        else int(train_cfg.num_epochs) * steps_per_epoch
    )
    total_steps_target = max(1, total_steps_target)

    decay_steps = int(max(total_steps_target, train_cfg.warmup_steps + 1))
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(train_cfg.learning_rate),
        warmup_steps=int(max(1, train_cfg.warmup_steps)),
        decay_steps=decay_steps,
        end_value=float(train_cfg.min_learning_rate),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(float(train_cfg.grad_clip_norm)),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=float(train_cfg.weight_decay),
            b1=0.9,
            b2=0.95,
            eps=1e-5,
        ),
    )

    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
    optimizer = nnx.Optimizer(model, tx)

    start_step = 0
    if train_cfg.resume_checkpoint:
        ckpt_path = Path(train_cfg.resume_checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.pkl"
        if ckpt_path.is_file():
            start_step = _load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
            )
            print(f"Resumed checkpoint: {ckpt_path} (step={start_step})")

    run_meta = {
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "data_source_mode": split_mode,
        "resolved_data_dirs": resolved_dirs,
        "split_settings": {
            "train_fraction": float(train_cfg.train_fraction),
            "sample_interval_training": int(train_cfg.sample_interval_training),
            "sample_interval_test": int(train_cfg.sample_interval_test),
            "strict_91_steps": bool(train_cfg.strict_91_steps),
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
        "created_at": int(time.time()),
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    metrics_log_path = output_dir / "metrics.jsonl"

    global_step = int(start_step)
    train_rng = np.random.default_rng(train_cfg.seed + 1)

    best_eval_loss = float("inf")
    best_eval_step = -1
    relation_dump_counter = 0

    t0 = time.time()
    stop = False

    for epoch in range(train_cfg.num_epochs):
        # Deterministic but epoch-varying shuffle.
        epoch_indices = train_indices.copy()
        train_rng.shuffle(epoch_indices)

        for idx_batch in _iter_minibatches(epoch_indices, train_cfg.batch_size):
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

            _write_jsonl(
                metrics_log_path,
                {
                    "phase": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "lr": lr_now,
                    "metrics": metrics_f,
                },
            )

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

                _write_jsonl(
                    metrics_log_path,
                    {
                        "phase": "eval",
                        "step": global_step,
                        "epoch": epoch,
                        "lr": lr_now,
                        "metrics": eval_metrics,
                    },
                )
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
                    )
                    print(f"Saved improved checkpoint: {best_path}")

            if global_step % max(1, train_cfg.checkpoint_every_steps) == 0:
                ckpt_path = _save_checkpoint(
                    output_dir=output_dir,
                    train_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    latest_metrics=metrics_f,
                )
                print(f"Saved checkpoint: {ckpt_path}")

            if global_step >= total_steps_target:
                stop = True
                break

        if stop:
            break

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

    final_ckpt = _save_checkpoint(
        output_dir=output_dir,
        train_step=global_step,
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        latest_metrics=final_metrics,
    )

    summary = {
        "output_dir": str(output_dir),
        "final_checkpoint": str(final_ckpt),
        "total_steps": int(global_step),
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

    return summary
