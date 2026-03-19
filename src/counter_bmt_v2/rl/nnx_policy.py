"""Checkpoint-backed NNX policy backend for Topo-MCPO RL."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from counter_bmt_v2.config import RLPolicyConfig
from counter_bmt_v2.contracts import BayesianDAG, ConditioningSignal, ScenarioInput, TrajectoryRollout
from counter_bmt_v2.data import NNXBMTSceneSample
from counter_bmt_v2.training.dag_tensorize import tensorize_dag_batch
from counter_bmt_v2.training.forward_metrics import _reconstruct_rollout_states
from counter_bmt_v2.training.supervised import SupervisedTrainConfig, _prepare_supervised_batch, _resolve_model_preset
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    BidirectionalMotionTokenizer,
    NNXBidirectionalMotionTransformer,
    ParityTokenizerConfig,
    midgpt_dag_latent_config,
    sample_motion_tokens,
)


_DYNAMIC_INPUT_KEYS = {
    "prev_token_ids",
    "continuous_motion",
    "input_action_valid_mask",
    "modeled_agent_delta",
}


@dataclass
class PreparedPolicyScene:
    scene: ScenarioInput
    raw_batch: Dict[str, Any]
    static_model_inputs: Dict[str, jnp.ndarray]
    horizon_steps: int
    start_token_ids: np.ndarray  # [1,N]
    init_pos_bn2: np.ndarray  # [1,N,2]
    init_heading_bn: np.ndarray  # [1,N]
    init_speed_bn: np.ndarray  # [1,N]
    dt_chunk_b: np.ndarray  # [1]
    sampled_dag: BayesianDAG
    sampled_dag_payload: Dict[str, Any]


@dataclass
class PolicyRolloutData:
    prepared_scene: PreparedPolicyScene
    token_ids: np.ndarray  # [K,H,N]
    old_logprob_sum: np.ndarray  # [K]
    entropy_mean: np.ndarray  # [K]
    feasibility_mask_rate: np.ndarray  # [K]
    old_logprob_steps: np.ndarray | None = None  # [K,H] (ego only)
    entropy_steps: np.ndarray | None = None  # [K,H] (ego only)
    feasibility_mask_rate_steps: np.ndarray | None = None  # [K,H] (ego only)
    ego_motion_deltas: np.ndarray | None = None  # [K,H,2]


@dataclass
class PolicyCandidatePool:
    prepared_scene: PreparedPolicyScene
    rollout_data: PolicyRolloutData
    rollouts: List[TrajectoryRollout]
    trajectory_all_xy: np.ndarray  # [K,H,N,2]


def _slice_optional_array(arr: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr[idx])


def _empty_step_matrix(batch_size: int, horizon_steps: int) -> jnp.ndarray:
    return jnp.zeros((batch_size, horizon_steps), dtype=jnp.float32)


def _resolve_model_cfg_for_policy(preset_name: str):
    if str(preset_name) == "midgpt_dag_latent":
        return midgpt_dag_latent_config()
    return _resolve_model_preset(str(preset_name))  # type: ignore[arg-type]


def _filter_supervised_cfg(payload: Mapping[str, Any], *, model_preset: str, tokenizer_mode: str, skip_steps: int) -> SupervisedTrainConfig:
    cfg = SupervisedTrainConfig()
    valid = {f.name for f in fields(SupervisedTrainConfig)}
    for key, value in dict(payload).items():
        if key in valid:
            setattr(cfg, key, value)
    cfg.model_preset = str(model_preset)
    cfg.tokenizer_mode = str(tokenizer_mode)
    cfg.skip_steps = int(skip_steps)
    cfg.mode = "forward"
    cfg.reverse_probability = 0.0
    cfg.batch_size = 1
    cfg.precision = "fp32"
    return cfg


def _path_to_str(path: Sequence[Any]) -> str:
    segs: List[str] = []
    for p in path:
        if hasattr(p, "key"):
            segs.append(str(getattr(p, "key")))
        elif hasattr(p, "name"):
            segs.append(str(getattr(p, "name")))
        elif hasattr(p, "idx"):
            segs.append(str(getattr(p, "idx")))
        else:
            segs.append(str(p))
    return ".".join(segs)


def _is_trainable_path(path_str: str, *, scope: str) -> bool:
    if str(scope) == "all":
        return True
    dag_roots = ("dag_encoder", "dag_latent_proj", "dag_gate_proj", "null_dag_latent")
    decoder_roots = ("decoder_blocks", "final_norm", "token_head")
    return any(part in path_str for part in (*dag_roots, *decoder_roots))


def _build_grad_scale_tree(model: NNXBidirectionalMotionTransformer, *, scope: str) -> Any:
    state = nnx.state(model, nnx.Param)
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    scales: List[jnp.ndarray] = []
    for path, _leaf in path_leaves:
        p = _path_to_str(path)
        s = 1.0 if _is_trainable_path(p, scope=str(scope)) else 0.0
        scales.append(jnp.asarray(float(s), dtype=jnp.float32))
    return jax.tree_util.tree_unflatten(treedef, scales)


def _build_trainable_label_tree(model: NNXBidirectionalMotionTransformer, *, scope: str) -> Any:
    state = nnx.state(model, nnx.Param)
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    labels: List[str] = []
    for path, _leaf in path_leaves:
        p = _path_to_str(path)
        labels.append("train" if _is_trainable_path(p, scope=str(scope)) else "freeze")
    return jax.tree_util.tree_unflatten(treedef, labels)


def _build_policy_optimizer(
    model: NNXBidirectionalMotionTransformer,
    *,
    cfg: RLPolicyConfig,
) -> nnx.Optimizer:
    train_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=float(cfg.policy_lr),
            weight_decay=0.0,
            b1=0.9,
            b2=0.95,
            eps=1e-5,
        ),
    )
    tx = optax.multi_transform(
        {
            "train": train_tx,
            "freeze": optax.set_to_zero(),
        },
        _build_trainable_label_tree(model, scope=str(cfg.trainable_scope)),
    )
    return nnx.Optimizer(model, tx)


def _repeat_first_axis(arr: jnp.ndarray, repeats: int) -> jnp.ndarray:
    if int(repeats) <= 1:
        return arr
    return jnp.repeat(arr, int(repeats), axis=0)


def _repeat_model_inputs(model_inputs: Dict[str, jnp.ndarray], repeats: int) -> Dict[str, jnp.ndarray]:
    return {k: _repeat_first_axis(v, repeats) for k, v in model_inputs.items()}


def _build_feasibility_mask(
    *,
    current_speed_bn: jnp.ndarray,  # [B,N]
    prev_action_bn2: jnp.ndarray,  # [B,N,2]
    action_table_v2: jnp.ndarray,  # [V,2]
    dt_chunk_b: jnp.ndarray,  # [B]
    cfg: RLPolicyConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    bsz, n_agents = current_speed_bn.shape
    vocab = int(action_table_v2.shape[0])
    acc = action_table_v2[:, 0][None, None, :]  # [1,1,V]
    yaw = action_table_v2[:, 1][None, None, :]  # [1,1,V]
    dt = dt_chunk_b[:, None, None]
    speed_next = current_speed_bn[:, :, None] + acc * dt

    valid = speed_next >= 0.0
    valid = jnp.logical_and(valid, speed_next <= float(cfg.feasible_max_speed_mps))
    valid = jnp.logical_and(valid, jnp.abs(acc - prev_action_bn2[:, :, 0:1]) <= float(cfg.feasible_max_accel_delta))
    valid = jnp.logical_and(valid, jnp.abs(yaw - prev_action_bn2[:, :, 1:2]) <= float(cfg.feasible_max_yaw_delta))

    any_valid = jnp.any(valid, axis=-1, keepdims=True)
    valid = jnp.where(any_valid, valid, jnp.ones((bsz, n_agents, vocab), dtype=bool))
    invalid_rate = 1.0 - jnp.mean(valid.astype(jnp.float32), axis=(1, 2))
    return valid, invalid_rate


def _build_rollout_masks(
    static_inputs: Dict[str, jnp.ndarray],
    *,
    batch_size: int,
    horizon_steps: int,
    step_index: int,
    n_agents: int,
) -> Dict[str, jnp.ndarray]:
    out: Dict[str, jnp.ndarray] = {}
    prefix_t = (jnp.arange(horizon_steps, dtype=jnp.int32) <= int(step_index))[None, :, None]
    input_action_valid_mask = jnp.broadcast_to(prefix_t, (batch_size, horizon_steps, n_agents))
    out["input_action_valid_mask"] = input_action_valid_mask

    if "a2a_mask" in static_inputs:
        prefix_a2a = input_action_valid_mask[:, :, :, None]
        out["a2a_mask"] = jnp.logical_and(static_inputs["a2a_mask"], prefix_a2a)
    if "a2s_mask" in static_inputs:
        prefix_a2s = input_action_valid_mask[:, :, :, None]
        out["a2s_mask"] = jnp.logical_and(static_inputs["a2s_mask"], prefix_a2s)
    if "a2t_mask" in static_inputs:
        tmask = (jnp.arange(horizon_steps, dtype=jnp.int32) <= int(step_index))
        causal_t = jnp.logical_and(tmask[None, None, :, None], tmask[None, None, None, :])
        out["a2t_mask"] = jnp.logical_and(static_inputs["a2t_mask"], causal_t)
    return out


class NNXPolicyBackend:
    def __init__(
        self,
        *,
        cfg: RLPolicyConfig,
        model_cfg: Any,
        prep_train_cfg: SupervisedTrainConfig,
        tokenizer: Any,
        model: NNXBidirectionalMotionTransformer,
        reference_model: NNXBidirectionalMotionTransformer,
        optimizer: nnx.Optimizer,
    ) -> None:
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.prep_train_cfg = prep_train_cfg
        self.tokenizer = tokenizer
        self.model = model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.grad_scale_tree = _build_grad_scale_tree(model, scope=str(cfg.trainable_scope))
        self.action_table_np = np.asarray(tokenizer.action_table_np(), dtype=np.float32)
        self.action_table_jnp = jnp.asarray(self.action_table_np, dtype=jnp.float32)
        self.step = 0

    @classmethod
    def from_checkpoint(
        cls,
        *,
        cfg: RLPolicyConfig,
        seed: int = 0,
    ) -> "NNXPolicyBackend":
        ckpt_path = Path(str(cfg.checkpoint))
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.pkl"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"RL policy checkpoint not found: {ckpt_path}")
        with ckpt_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "model_state" not in payload:
            raise ValueError(f"Invalid policy checkpoint payload: {ckpt_path}")

        ckpt_train_cfg = payload.get("train_cfg", {}) if isinstance(payload.get("train_cfg"), dict) else {}
        model_preset = str(cfg.model_preset or ckpt_train_cfg.get("model_preset", "midgpt_dag_latent"))
        tokenizer_mode = str(cfg.tokenizer_mode or ckpt_train_cfg.get("tokenizer_mode", "adv_bmt_parity"))
        skip_steps = int(cfg.skip_steps or ckpt_train_cfg.get("skip_steps", 5))

        model_cfg = _resolve_model_cfg_for_policy(model_preset)
        if not bool(getattr(model_cfg, "dag_encoder", None) and model_cfg.dag_encoder.enabled):
            raise ValueError(
                "NNX RL backend expects a DAG-latent capable checkpoint/model preset. "
                f"Got model_preset={model_preset} with dag_encoder.enabled={getattr(model_cfg.dag_encoder, 'enabled', False)}"
            )
        prep_train_cfg = _filter_supervised_cfg(
            ckpt_train_cfg,
            model_preset=model_preset,
            tokenizer_mode=tokenizer_mode,
            skip_steps=skip_steps,
        )

        if tokenizer_mode == "adv_bmt_parity":
            tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=skip_steps))
        else:
            tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)

        model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(seed))
        reference_model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(seed + 1))
        nnx.update(model, payload["model_state"])
        nnx.update(reference_model, payload["model_state"])

        optimizer = _build_policy_optimizer(model, cfg=cfg)
        return cls(
            cfg=cfg,
            model_cfg=model_cfg,
            prep_train_cfg=prep_train_cfg,
            tokenizer=tokenizer,
            model=model,
            reference_model=reference_model,
            optimizer=optimizer,
        )

    def _prepare_scene(
        self,
        *,
        scene: ScenarioInput,
        sampled_dag: BayesianDAG,
        sampled_dag_payload: Dict[str, Any],
        seed: int,
    ) -> PreparedPolicyScene:
        sample = scene.metadata.get("nnx_sample") if isinstance(scene.metadata, dict) else None
        if not isinstance(sample, NNXBMTSceneSample):
            raise ValueError(
                "NNX RL backend requires ScenarioInput.metadata['nnx_sample'] with an NNXBMTSceneSample"
            )
        rng = np.random.default_rng(int(seed))
        prepared = _prepare_supervised_batch(
            [sample],
            train_cfg=self.prep_train_cfg,
            model_cfg=self.model_cfg,
            tokenizer=self.tokenizer,
            rng=rng,
            is_training=False,
        )

        dag_t = tensorize_dag_batch(
            [sampled_dag_payload],
            max_nodes=int(self.model_cfg.dag_encoder.max_nodes),
            max_edges=int(self.model_cfg.dag_encoder.max_edges),
            d_node_in=int(self.model_cfg.dag_encoder.d_node_in),
            d_edge_in=int(self.model_cfg.dag_encoder.d_edge_in),
        )
        static_inputs = {
            k: v for k, v in prepared["model_inputs"].items()
            if k not in _DYNAMIC_INPUT_KEYS
        }
        static_inputs.update(
            {
                "dag_node_feat": jnp.asarray(dag_t["dag_node_feat"], dtype=jnp.float32),
                "dag_node_mask": jnp.asarray(dag_t["dag_node_mask"], dtype=bool),
                "dag_edge_src": jnp.asarray(dag_t["dag_edge_src"], dtype=jnp.int32),
                "dag_edge_dst": jnp.asarray(dag_t["dag_edge_dst"], dtype=jnp.int32),
                "dag_edge_feat": jnp.asarray(dag_t["dag_edge_feat"], dtype=jnp.float32),
                "dag_edge_mask": jnp.asarray(dag_t["dag_edge_mask"], dtype=bool),
                "dag_global_feat": jnp.asarray(dag_t["dag_global_feat"], dtype=jnp.float32),
            }
        )

        raw = prepared["raw_batch"]
        sample_steps = np.asarray(prepared["sample_steps"], dtype=np.int32)
        init_t = int(sample_steps[0]) if sample_steps.size > 0 else 0
        dt_chunk_b = np.asarray(raw["dt_s"], dtype=np.float32) * float(self.prep_train_cfg.skip_steps)
        init_pos_bn2 = np.asarray(raw["agent_position_xy"], dtype=np.float32)[:, init_t, :, :]
        init_heading_bn = np.asarray(raw["agent_heading"], dtype=np.float32)[:, init_t, :]
        init_speed_bn = np.linalg.norm(np.asarray(raw["agent_velocity_xy"], dtype=np.float32)[:, init_t, :, :], axis=-1)
        start_token_ids = np.asarray(jax.device_get(prepared["model_inputs"]["prev_token_ids"]), dtype=np.int32)[:, 0, :]

        return PreparedPolicyScene(
            scene=scene,
            raw_batch=raw,
            static_model_inputs=static_inputs,
            horizon_steps=int(np.asarray(prepared["targets"]).shape[1]),
            start_token_ids=start_token_ids,
            init_pos_bn2=init_pos_bn2.astype(np.float32),
            init_heading_bn=init_heading_bn.astype(np.float32),
            init_speed_bn=init_speed_bn.astype(np.float32),
            dt_chunk_b=np.asarray(dt_chunk_b, dtype=np.float32),
            sampled_dag=sampled_dag,
            sampled_dag_payload=sampled_dag_payload,
        )

    def sample_candidate_pool(
        self,
        *,
        scene: ScenarioInput,
        sampled_dag: BayesianDAG,
        sampled_dag_payload: Dict[str, Any],
        n_samples: int,
        seed: int,
        conditioning_metadata: Dict[str, Any],
    ) -> PolicyCandidatePool:
        prepared = self._prepare_scene(
            scene=scene,
            sampled_dag=sampled_dag,
            sampled_dag_payload=sampled_dag_payload,
            seed=int(seed),
        )
        bsz = int(max(1, n_samples))
        horizon_steps = int(prepared.horizon_steps)
        n_agents = int(prepared.start_token_ids.shape[1])
        static_inputs = _repeat_model_inputs(prepared.static_model_inputs, bsz)
        token_seq = jnp.broadcast_to(
            jnp.asarray(prepared.start_token_ids, dtype=jnp.int32)[:, None, :],
            (bsz, horizon_steps, n_agents),
        )
        motion_seq = jnp.zeros((bsz, horizon_steps, n_agents, 2), dtype=jnp.float32)
        modeled_delta_seq = jnp.zeros_like(motion_seq)
        current_speed = jnp.asarray(np.repeat(prepared.init_speed_bn, bsz, axis=0), dtype=jnp.float32)
        prev_action = jnp.zeros((bsz, n_agents, 2), dtype=jnp.float32)
        dt_chunk_b = jnp.asarray(np.repeat(prepared.dt_chunk_b, bsz, axis=0), dtype=jnp.float32)
        old_logprob_sum = jnp.zeros((bsz,), dtype=jnp.float32)
        entropy_sum = jnp.zeros((bsz,), dtype=jnp.float32)
        mask_rate_sum = jnp.zeros((bsz,), dtype=jnp.float32)
        old_logprob_steps = jnp.zeros((bsz, horizon_steps), dtype=jnp.float32) if bool(self.cfg.store_rollout_traces) else None
        entropy_steps = jnp.zeros((bsz, horizon_steps), dtype=jnp.float32) if bool(self.cfg.store_rollout_traces) else None
        mask_rate_steps = jnp.zeros((bsz, horizon_steps), dtype=jnp.float32) if bool(self.cfg.store_rollout_traces) else None
        ego_motion_deltas = jnp.zeros((bsz, horizon_steps, 2), dtype=jnp.float32) if bool(self.cfg.store_rollout_traces) else None
        key = jax.random.PRNGKey(int(seed))

        for t in range(horizon_steps):
            rollout_masks = _build_rollout_masks(
                static_inputs,
                batch_size=bsz,
                horizon_steps=horizon_steps,
                step_index=t,
                n_agents=n_agents,
            )
            model_inputs = dict(static_inputs)
            model_inputs.update(rollout_masks)
            model_inputs["prev_token_ids"] = token_seq
            model_inputs["continuous_motion"] = motion_seq
            model_inputs["modeled_agent_delta"] = modeled_delta_seq

            logits = self.model(**model_inputs).astype(jnp.float32)
            step_logits = logits[:, t, :, :]

            ego_invalid_rate = jnp.zeros((bsz,), dtype=jnp.float32)
            if bool(self.cfg.enable_feasibility_mask):
                feasible_mask, invalid_rate = _build_feasibility_mask(
                    current_speed_bn=current_speed,
                    prev_action_bn2=prev_action,
                    action_table_v2=self.action_table_jnp,
                    dt_chunk_b=dt_chunk_b,
                    cfg=self.cfg,
                )
                step_logits = jnp.where(feasible_mask, step_logits, jnp.full_like(step_logits, -1e9))
                mask_rate_sum = mask_rate_sum + invalid_rate
                ego_invalid_rate = 1.0 - jnp.mean(feasible_mask[:, 0, :].astype(jnp.float32), axis=-1)

            log_probs = jax.nn.log_softmax(step_logits, axis=-1)
            probs = jnp.exp(log_probs)
            entropy_agents = -jnp.sum(probs * log_probs, axis=-1)

            key, sub = jax.random.split(key)
            next_tok = sample_motion_tokens(
                step_logits,
                sub,
                sampling_method=str(self.cfg.sampling_method),
                temperature=float(self.cfg.sampling_temperature),
                topp=float(self.cfg.sampling_topp),
                topk=int(self.cfg.sampling_topk),
            )
            next_lp = jnp.take_along_axis(log_probs, next_tok[..., None], axis=-1).squeeze(-1)
            old_logprob_sum = old_logprob_sum + next_lp[:, 0]
            entropy_sum = entropy_sum + entropy_agents[:, 0]

            next_motion = jnp.take(self.action_table_jnp, next_tok, axis=0)
            if old_logprob_steps is not None and entropy_steps is not None and mask_rate_steps is not None and ego_motion_deltas is not None:
                old_logprob_steps = old_logprob_steps.at[:, t].set(next_lp[:, 0])
                entropy_steps = entropy_steps.at[:, t].set(entropy_agents[:, 0])
                mask_rate_steps = mask_rate_steps.at[:, t].set(ego_invalid_rate)
                ego_motion_deltas = ego_motion_deltas.at[:, t, :].set(next_motion[:, 0, :])
            token_seq = token_seq.at[:, t, :].set(next_tok)
            motion_seq = motion_seq.at[:, t, :, :].set(next_motion)
            modeled_delta_seq = modeled_delta_seq.at[:, t, :, :].set(next_motion)
            current_speed = jnp.maximum(0.0, current_speed + next_motion[:, :, 0] * dt_chunk_b[:, None])
            prev_action = next_motion

        token_np = np.asarray(jax.device_get(token_seq), dtype=np.int32)
        pos, _, _, _ = _reconstruct_rollout_states(
            predicted_tokens_kbtn=token_np[None, ...],
            action_table=self.action_table_np,
            init_pos_bn2=np.repeat(prepared.init_pos_bn2, bsz, axis=0),
            init_heading_bn=np.repeat(prepared.init_heading_bn, bsz, axis=0),
            init_speed_bn=np.repeat(prepared.init_speed_bn, bsz, axis=0),
            dt_chunk_b=np.repeat(prepared.dt_chunk_b, bsz, axis=0),
        )
        all_xy = pos[0]
        ego_xy = all_xy[:, :, 0, :]
        old_logprob_np = np.asarray(jax.device_get(old_logprob_sum), dtype=np.float32)
        entropy_mean_np = (np.asarray(jax.device_get(entropy_sum), dtype=np.float32) / max(1, horizon_steps)).astype(np.float32)
        feasibility_mask_rate_np = (np.asarray(jax.device_get(mask_rate_sum), dtype=np.float32) / max(1, horizon_steps)).astype(np.float32)
        old_logprob_steps_np = np.asarray(jax.device_get(old_logprob_steps), dtype=np.float32) if old_logprob_steps is not None else None
        entropy_steps_np = np.asarray(jax.device_get(entropy_steps), dtype=np.float32) if entropy_steps is not None else None
        mask_rate_steps_np = np.asarray(jax.device_get(mask_rate_steps), dtype=np.float32) if mask_rate_steps is not None else None
        ego_motion_deltas_np = np.asarray(jax.device_get(ego_motion_deltas), dtype=np.float32) if ego_motion_deltas is not None else None

        rollouts: List[TrajectoryRollout] = []
        for i in range(bsz):
            conditioning = ConditioningSignal(
                vector=np.zeros((0,), dtype=np.float32),
                metadata=dict(conditioning_metadata),
            )
            metadata = {
                "backend": "nnx_checkpoint",
                "trajectory_all_xy": np.asarray(all_xy[i], dtype=np.float32).tolist(),
                "predicted_tokens": token_np[i].tolist(),
                "old_logprob_sum": float(old_logprob_np[i]),
                "entropy_mean": float(entropy_mean_np[i]),
                "feasibility_mask_rate": float(feasibility_mask_rate_np[i]),
                "conditioning_assignments": dict(conditioning_metadata.get("assignments", {})),
            }
            if (
                old_logprob_steps_np is not None
                and entropy_steps_np is not None
                and mask_rate_steps_np is not None
                and ego_motion_deltas_np is not None
            ):
                metadata["rollout_trace"] = {
                    "ego_old_logprob_steps": old_logprob_steps_np[i].tolist(),
                    "ego_entropy_steps": entropy_steps_np[i].tolist(),
                    "ego_feasibility_mask_rate_steps": mask_rate_steps_np[i].tolist(),
                    "ego_motion_deltas": ego_motion_deltas_np[i].tolist(),
                }
            rollouts.append(
                TrajectoryRollout(
                    trajectory_xy=np.asarray(ego_xy[i], dtype=np.float32),
                    conditioning=conditioning,
                    sample_index=i,
                    metadata=metadata,
                )
            )

        rollout_data = PolicyRolloutData(
            prepared_scene=prepared,
            token_ids=token_np,
            old_logprob_sum=old_logprob_np,
            entropy_mean=entropy_mean_np,
            feasibility_mask_rate=feasibility_mask_rate_np,
            old_logprob_steps=old_logprob_steps_np,
            entropy_steps=entropy_steps_np,
            feasibility_mask_rate_steps=mask_rate_steps_np,
            ego_motion_deltas=ego_motion_deltas_np,
        )
        return PolicyCandidatePool(
            prepared_scene=prepared,
            rollout_data=rollout_data,
            rollouts=rollouts,
            trajectory_all_xy=np.asarray(all_xy, dtype=np.float32),
        )

    def select_rollout_data(self, pool: PolicyCandidatePool, indices: Sequence[int]) -> PolicyRolloutData:
        idx = np.asarray(list(indices), dtype=np.int32)
        data = pool.rollout_data
        return PolicyRolloutData(
            prepared_scene=data.prepared_scene,
            token_ids=data.token_ids[idx],
            old_logprob_sum=data.old_logprob_sum[idx],
            entropy_mean=data.entropy_mean[idx],
            feasibility_mask_rate=data.feasibility_mask_rate[idx],
            old_logprob_steps=_slice_optional_array(data.old_logprob_steps, idx),
            entropy_steps=_slice_optional_array(data.entropy_steps, idx),
            feasibility_mask_rate_steps=_slice_optional_array(data.feasibility_mask_rate_steps, idx),
            ego_motion_deltas=_slice_optional_array(data.ego_motion_deltas, idx),
        )

    def _recompute_rollout_stats(
        self,
        *,
        model: NNXBidirectionalMotionTransformer,
        batch: PolicyRolloutData,
        reference_model: NNXBidirectionalMotionTransformer | None = None,
    ) -> Dict[str, jnp.ndarray]:
        tokens = jnp.asarray(batch.token_ids, dtype=jnp.int32)
        if tokens.ndim != 3:
            raise ValueError(f"Expected token_ids with shape [K,H,N], got {tokens.shape}")

        k, horizon_steps, n_agents = tokens.shape
        prepared = batch.prepared_scene
        static_inputs = _repeat_model_inputs(prepared.static_model_inputs, int(k))
        token_seq = jnp.broadcast_to(
            jnp.asarray(prepared.start_token_ids, dtype=jnp.int32)[:, None, :],
            (int(k), int(horizon_steps), int(n_agents)),
        )
        motion_seq = jnp.zeros((int(k), int(horizon_steps), int(n_agents), 2), dtype=jnp.float32)
        modeled_delta_seq = jnp.zeros_like(motion_seq)
        current_speed = jnp.asarray(np.repeat(prepared.init_speed_bn, int(k), axis=0), dtype=jnp.float32)
        prev_action = jnp.zeros((int(k), int(n_agents), 2), dtype=jnp.float32)
        dt_chunk_b = jnp.asarray(np.repeat(prepared.dt_chunk_b, int(k), axis=0), dtype=jnp.float32)

        logprob_steps = _empty_step_matrix(int(k), int(horizon_steps))
        entropy_steps = _empty_step_matrix(int(k), int(horizon_steps))
        mask_rate_steps = _empty_step_matrix(int(k), int(horizon_steps))
        kl_steps = _empty_step_matrix(int(k), int(horizon_steps)) if reference_model is not None else None

        for t in range(int(horizon_steps)):
            rollout_masks = _build_rollout_masks(
                static_inputs,
                batch_size=int(k),
                horizon_steps=int(horizon_steps),
                step_index=int(t),
                n_agents=int(n_agents),
            )
            model_inputs = dict(static_inputs)
            model_inputs.update(rollout_masks)
            model_inputs["prev_token_ids"] = token_seq
            model_inputs["continuous_motion"] = motion_seq
            model_inputs["modeled_agent_delta"] = modeled_delta_seq

            logits = model(**model_inputs).astype(jnp.float32)
            step_logits = logits[:, t, :, :]
            ego_invalid_rate = jnp.zeros((int(k),), dtype=jnp.float32)
            feasible_mask = None
            if bool(self.cfg.enable_feasibility_mask):
                feasible_mask, _invalid_rate = _build_feasibility_mask(
                    current_speed_bn=current_speed,
                    prev_action_bn2=prev_action,
                    action_table_v2=self.action_table_jnp,
                    dt_chunk_b=dt_chunk_b,
                    cfg=self.cfg,
                )
                step_logits = jnp.where(feasible_mask, step_logits, jnp.full_like(step_logits, -1e9))
                ego_invalid_rate = 1.0 - jnp.mean(feasible_mask[:, 0, :].astype(jnp.float32), axis=-1)

            log_probs = jax.nn.log_softmax(step_logits, axis=-1)
            probs = jnp.exp(log_probs)
            entropy_agents = -jnp.sum(probs * log_probs, axis=-1)
            next_tok = tokens[:, t, :]
            next_lp = jnp.take_along_axis(log_probs, next_tok[..., None], axis=-1).squeeze(-1)

            logprob_steps = logprob_steps.at[:, t].set(next_lp[:, 0])
            entropy_steps = entropy_steps.at[:, t].set(entropy_agents[:, 0])
            mask_rate_steps = mask_rate_steps.at[:, t].set(ego_invalid_rate)

            if reference_model is not None and kl_steps is not None:
                ref_logits = reference_model(**model_inputs).astype(jnp.float32)
                ref_step_logits = ref_logits[:, t, :, :]
                if feasible_mask is not None:
                    ref_step_logits = jnp.where(feasible_mask, ref_step_logits, jnp.full_like(ref_step_logits, -1e9))
                ref_log_probs = jax.nn.log_softmax(ref_step_logits, axis=-1)
                ego_kl = jnp.sum(
                    probs[:, 0, :] * (log_probs[:, 0, :] - ref_log_probs[:, 0, :]),
                    axis=-1,
                )
                kl_steps = kl_steps.at[:, t].set(ego_kl)

            next_motion = jnp.take(self.action_table_jnp, next_tok, axis=0)
            token_seq = token_seq.at[:, t, :].set(next_tok)
            motion_seq = motion_seq.at[:, t, :, :].set(next_motion)
            modeled_delta_seq = modeled_delta_seq.at[:, t, :, :].set(next_motion)
            current_speed = jnp.maximum(0.0, current_speed + next_motion[:, :, 0] * dt_chunk_b[:, None])
            prev_action = next_motion

        stats = {
            "logprob_steps": logprob_steps,
            "logprob_sum": jnp.sum(logprob_steps, axis=1),
            "entropy_steps": entropy_steps,
            "entropy_mean": jnp.mean(entropy_steps, axis=1) if int(horizon_steps) > 0 else jnp.zeros((int(k),), dtype=jnp.float32),
            "feasibility_mask_rate_steps": mask_rate_steps,
            "feasibility_mask_rate": jnp.mean(mask_rate_steps, axis=1) if int(horizon_steps) > 0 else jnp.zeros((int(k),), dtype=jnp.float32),
        }
        if kl_steps is not None:
            stats["kl_steps"] = kl_steps
            stats["kl_mean"] = jnp.mean(kl_steps) if int(horizon_steps) > 0 else jnp.asarray(0.0, dtype=jnp.float32)
        return stats

    def _build_teacher_forced_inputs(self, batch: PolicyRolloutData) -> Dict[str, jnp.ndarray]:
        tokens = np.asarray(batch.token_ids, dtype=np.int32)
        k, horizon_steps, n_agents = tokens.shape
        start = np.repeat(batch.prepared_scene.start_token_ids, k, axis=0)
        prev = np.concatenate([start[:, None, :], tokens[:, :-1, :]], axis=1).astype(np.int32)
        motion = self.action_table_np[tokens]

        static_inputs = _repeat_model_inputs(batch.prepared_scene.static_model_inputs, k)
        model_inputs = dict(static_inputs)
        model_inputs["prev_token_ids"] = jnp.asarray(prev, dtype=jnp.int32)
        model_inputs["continuous_motion"] = jnp.asarray(motion, dtype=jnp.float32)
        model_inputs["modeled_agent_delta"] = jnp.asarray(motion, dtype=jnp.float32)
        model_inputs["input_action_valid_mask"] = jnp.ones((k, horizon_steps, n_agents), dtype=bool)
        if "a2t_mask" in static_inputs:
            causal = jnp.tril(jnp.ones((horizon_steps, horizon_steps), dtype=bool))[None, None, :, :]
            model_inputs["a2t_mask"] = jnp.logical_and(static_inputs["a2t_mask"], causal)
        return model_inputs

    def update(
        self,
        *,
        batch: PolicyRolloutData,
        advantages: np.ndarray,
        alpha: float,
    ) -> Dict[str, float]:
        adv = jnp.asarray(np.asarray(advantages, dtype=np.float32).reshape(-1), dtype=jnp.float32)
        old_logprob = jnp.asarray(np.asarray(batch.old_logprob_sum, dtype=np.float32).reshape(-1), dtype=jnp.float32)
        tokens = jnp.asarray(batch.token_ids, dtype=jnp.int32)
        if tokens.ndim != 3:
            raise ValueError(f"Expected token_ids with shape [K,H,N], got {tokens.shape}")
        if int(tokens.shape[0]) != int(adv.shape[0]):
            raise ValueError("Advantages length must match number of selected rollouts")

        inputs = self._build_teacher_forced_inputs(batch)
        clip_eps = float(self.cfg.clip_eps)
        kl_beta = float(self.cfg.kl_beta)

        last_metrics: Dict[str, float] = {}
        for _ in range(max(1, int(self.cfg.ppo_epochs))):
            def loss_fn(model: NNXBidirectionalMotionTransformer) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
                stats = self._recompute_rollout_stats(
                    model=model,
                    batch=batch,
                    reference_model=self.reference_model,
                )
                new_logprob = stats["logprob_sum"]
                ratio = jnp.exp(new_logprob - old_logprob)
                unclipped = ratio * adv
                clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                surrogate = jnp.mean(jnp.minimum(unclipped, clipped))
                entropy = jnp.mean(stats["entropy_steps"])
                kl_ref = stats.get("kl_mean", jnp.asarray(0.0, dtype=jnp.float32))

                loss = -(surrogate + float(alpha) * entropy - kl_beta * kl_ref)
                clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
                return loss, {
                    "policy/loss": loss,
                    "policy/surrogate": surrogate,
                    "policy/clip_fraction": clip_fraction,
                    "policy/kl_ref": kl_ref,
                    "policy/entropy": entropy,
                    "policy/logprob_old_mean": jnp.mean(old_logprob),
                    "policy/logprob_new_mean": jnp.mean(new_logprob),
                }

            (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(self.model)
            grads = jax.tree.map(lambda g, s: g * s, grads, self.grad_scale_tree)
            self.optimizer.update(grads)
            last_metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}

        self.step += 1
        last_metrics["policy/step"] = float(self.step)
        last_metrics["policy/feasibility_mask_rate"] = float(np.mean(batch.feasibility_mask_rate)) if batch.feasibility_mask_rate.size else 0.0
        return last_metrics
