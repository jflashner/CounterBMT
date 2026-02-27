"""Opt-in staged supervised training with DAG latent conditioning."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from counter_bmt_v2.data import NNXBMTSceneSample
from counter_bmt_v2.training.dag_sources import DAGSourceResolver
from counter_bmt_v2.training.dag_tensorize import tensorize_dag_batch
from counter_bmt_v2.trajectory_jax import (
    AdvBMTParityTokenizer,
    NNXBMTConfig,
    NNXBidirectionalMotionTransformer,
    ParityTokenizerConfig,
    BidirectionalMotionTokenizer,
    midgpt_dag_latent_config,
)
from .supervised import (
    ForwardPassEvalConfig,
    SupervisedTrainConfig,
    _as_float_metrics,
    _assert_finite_metrics,
    _build_lr_schedule,
    _cast_tree_precision,
    _eval_step,
    _eval_step_pmap,
    _hash_indices,
    _load_checkpoint,
    _prepare_supervised_batch,
    _prescan_indices,
    _print_metrics,
    _resolve_data_sources,
    _resolve_model_preset,
    _save_checkpoint,
    _shard_tree_for_pmap,
    _write_jsonl,
    _write_split_artifacts,
    _compute_metric_dict,
)


DAGSourceModeType = Literal["dual", "cache", "scene_derived"]
StageType = Literal["A", "B", "C", "A_B_C"]


@dataclass
class DAGLatentTrainConfig(SupervisedTrainConfig):
    dag_source_mode: DAGSourceModeType = "dual"
    dag_cache_dir: str = ""
    dag_cache_strict: bool = False
    stage: StageType = "A_B_C"
    stage_a_steps: int = 200
    stage_b_steps: int = 200
    stage_c_steps: int = 200
    stage_a_dag_dropout_prob: float = 1.0
    stage_b_dag_dropout_prob: float = 0.3
    stage_c_dag_dropout_prob: float = 0.1
    stage_b_freeze_non_dag: bool = True
    stage_c_decoder_lr_scale: float = 0.1
    stage_c_dag_lr_scale: float = 1.0


def _resolve_model_preset_dag(name: str) -> NNXBMTConfig:
    if str(name) == "midgpt_dag_latent":
        return midgpt_dag_latent_config()
    return _resolve_model_preset(name)  # type: ignore[arg-type]


def _slice_raw_for_sample(raw: Dict[str, Any], b: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bsz = int(np.asarray(raw["agent_ids"]).shape[0])
    for k, v in raw.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (bsz,):
            out[k] = v[b]
        elif isinstance(v, list) and len(v) == bsz:
            out[k] = v[b]
        else:
            out[k] = v
    return out


def _empty_dag_payload(scenario_id: str) -> Dict[str, Any]:
    return {
        "schema_version": "counter_bmt_v2_dag_cache_v1",
        "scenario_id": str(scenario_id),
        "nodes": [],
        "edges": [],
        "cpts": {},
        "metadata": {"source": "null"},
    }


def _attach_dag_inputs(
    prepared: Dict[str, Any],
    *,
    resolver: DAGSourceResolver,
    model_cfg: NNXBMTConfig,
    train_cfg: DAGLatentTrainConfig,
) -> Dict[str, float]:
    raw = prepared["raw_batch"]
    scenario_ids = list(raw["scenario_ids"])

    dags: List[Dict[str, Any]] = []
    source_labels: List[str] = []
    for b, sid in enumerate(scenario_ids):
        batch_slice = _slice_raw_for_sample(raw, b)
        dag, source = resolver.resolve_one(
            scenario_id=str(sid),
            batch_slice=batch_slice,
            sample_index=b,
        )
        if source == "cache_miss_strict":
            raise ValueError(
                f"DAG cache strict mode enabled and cache miss for scenario_id={sid}. "
                f"dag_cache_dir={train_cfg.dag_cache_dir}"
            )
        if dag is None:
            dag = _empty_dag_payload(str(sid))
            source = "null"
        dags.append(dag)
        source_labels.append(source)

    dag_t = tensorize_dag_batch(
        dags,
        max_nodes=int(model_cfg.dag_encoder.max_nodes),
        max_edges=int(model_cfg.dag_encoder.max_edges),
        d_node_in=int(model_cfg.dag_encoder.d_node_in),
        d_edge_in=int(model_cfg.dag_encoder.d_edge_in),
    )
    dag_inputs = {
        "dag_node_feat": jnp.asarray(dag_t["dag_node_feat"], dtype=jnp.float32),
        "dag_node_mask": jnp.asarray(dag_t["dag_node_mask"], dtype=bool),
        "dag_edge_src": jnp.asarray(dag_t["dag_edge_src"], dtype=jnp.int32),
        "dag_edge_dst": jnp.asarray(dag_t["dag_edge_dst"], dtype=jnp.int32),
        "dag_edge_feat": jnp.asarray(dag_t["dag_edge_feat"], dtype=jnp.float32),
        "dag_edge_mask": jnp.asarray(dag_t["dag_edge_mask"], dtype=bool),
        "dag_global_feat": jnp.asarray(dag_t["dag_global_feat"], dtype=jnp.float32),
    }
    dag_inputs = _cast_tree_precision(dag_inputs, precision=train_cfg.precision)
    prepared["model_inputs"].update(dag_inputs)

    total = float(max(1, len(source_labels)))
    hits = float(sum(1 for s in source_labels if s == "cache"))
    fallback = float(sum(1 for s in source_labels if s == "scene_derived"))
    nulls = float(sum(1 for s in source_labels if s == "null"))
    return {
        "dag_source/cache_hit_rate": hits / total,
        "dag_source/fallback_rate": fallback / total,
        "dag_source/null_rate": nulls / total,
    }


def _path_to_str(path: Tuple[Any, ...]) -> str:
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


def _is_dag_param_path(path_str: str) -> bool:
    dag_roots = ("dag_encoder", "dag_latent_proj", "dag_gate_proj", "null_dag_latent")
    return any(part in path_str for part in dag_roots)


def _build_grad_scale_tree(
    model: NNXBidirectionalMotionTransformer,
    *,
    stage: str,
    stage_b_freeze_non_dag: bool,
    stage_c_decoder_lr_scale: float,
    stage_c_dag_lr_scale: float,
) -> Any:
    # Build scale factors over the same tree as gradients (Param state only).
    state = nnx.state(model, nnx.Param)
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(state)
    scales: List[jnp.ndarray] = []
    for path, _leaf in path_leaves:
        p = _path_to_str(path)
        is_dag = _is_dag_param_path(p)
        if stage == "B" and bool(stage_b_freeze_non_dag):
            s = float(1.0 if is_dag else 0.0)
        elif stage == "C":
            s = float(stage_c_dag_lr_scale if is_dag else stage_c_decoder_lr_scale)
        else:
            s = 1.0
        scales.append(jnp.asarray(s, dtype=jnp.float32))
    return jax.tree_util.tree_unflatten(treedef, scales)


@nnx.jit
def _train_step_scaled(
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
    grad_scale_tree: Any,
) -> Dict[str, jnp.ndarray]:
    def loss_fn(m: NNXBidirectionalMotionTransformer) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits, meta = m(**model_inputs, return_metadata=True)
        logits = logits.astype(jnp.float32)
        metrics = _compute_metric_dict(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
            reverse_indicator=reverse_indicator,
            default_token_id=default_token_id,
        )
        dag_meta = meta.get("dag", {})
        if isinstance(dag_meta, dict):
            if "dag_latent_norm" in dag_meta:
                metrics["dag_latent/norm"] = jnp.mean(dag_meta["dag_latent_norm"])
            if "dag_gate_mean" in dag_meta:
                metrics["dag_latent/gate_mean"] = jnp.mean(dag_meta["dag_gate_mean"])
        return metrics["total_loss"], metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.tree.map(lambda g, s: g * s, grads, grad_scale_tree)
    optimizer.update(grads)
    return metrics


@nnx.pmap(axis_name="data", in_axes=(None, None, 0, 0, 0, 0, None, None), out_axes=0)
def _train_step_scaled_pmap(
    model: NNXBidirectionalMotionTransformer,
    optimizer: nnx.Optimizer,
    model_inputs: Dict[str, jnp.ndarray],
    targets: jnp.ndarray,
    target_mask: jnp.ndarray,
    reverse_indicator: jnp.ndarray,
    default_token_id: int,
    grad_scale_tree: Any,
) -> Dict[str, jnp.ndarray]:
    def loss_fn(m: NNXBidirectionalMotionTransformer) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits, meta = m(**model_inputs, return_metadata=True)
        logits = logits.astype(jnp.float32)
        metrics = _compute_metric_dict(
            logits=logits,
            targets=targets,
            target_mask=target_mask,
            reverse_indicator=reverse_indicator,
            default_token_id=default_token_id,
        )
        dag_meta = meta.get("dag", {})
        if isinstance(dag_meta, dict):
            if "dag_latent_norm" in dag_meta:
                metrics["dag_latent/norm"] = jnp.mean(dag_meta["dag_latent_norm"])
            if "dag_gate_mean" in dag_meta:
                metrics["dag_latent/gate_mean"] = jnp.mean(dag_meta["dag_gate_mean"])
        return metrics["total_loss"], metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grads = jax.lax.pmean(grads, axis_name="data")
    grads = jax.tree.map(lambda g, s: g * s, grads, grad_scale_tree)
    optimizer.update(grads)
    metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="data"), metrics)
    return metrics


def _resolve_stage(train_cfg: DAGLatentTrainConfig, step: int) -> str:
    stage = str(train_cfg.stage)
    if stage in ("A", "B", "C"):
        return stage
    a = max(0, int(train_cfg.stage_a_steps))
    b = max(0, int(train_cfg.stage_b_steps))
    if step <= a:
        return "A"
    if step <= (a + b):
        return "B"
    return "C"


def _stage_total_steps(train_cfg: DAGLatentTrainConfig) -> int:
    stage = str(train_cfg.stage)
    if stage == "A":
        total = int(train_cfg.stage_a_steps)
    elif stage == "B":
        total = int(train_cfg.stage_b_steps)
    elif stage == "C":
        total = int(train_cfg.stage_c_steps)
    else:
        total = int(train_cfg.stage_a_steps) + int(train_cfg.stage_b_steps) + int(train_cfg.stage_c_steps)
    total = max(1, total)
    if train_cfg.max_steps is not None and int(train_cfg.max_steps) > 0:
        total = min(total, int(train_cfg.max_steps))
    return int(total)


def _apply_stage_dropout(model: NNXBidirectionalMotionTransformer, *, train_cfg: DAGLatentTrainConfig, stage: str) -> None:
    if stage == "A":
        p = float(train_cfg.stage_a_dag_dropout_prob)
    elif stage == "B":
        p = float(train_cfg.stage_b_dag_dropout_prob)
    else:
        p = float(train_cfg.stage_c_dag_dropout_prob)
    model.cfg.dag_conditioning.dag_dropout_prob = float(np.clip(p, 0.0, 1.0))


def _evaluate_dag(
    *,
    model: NNXBidirectionalMotionTransformer,
    loader: Any,
    val_indices: np.ndarray,
    train_cfg: DAGLatentTrainConfig,
    model_cfg: NNXBMTConfig,
    tokenizer: Any,
    default_token_id: int,
    rng: np.random.Generator,
    resolver: DAGSourceResolver,
    num_devices: int,
) -> Dict[str, float]:
    if len(val_indices) == 0:
        return {}
    val_batches = [val_indices[i : i + int(train_cfg.batch_size)] for i in range(0, len(val_indices), int(train_cfg.batch_size))]
    if int(train_cfg.eval_batches) > 0:
        val_batches = val_batches[: int(train_cfg.eval_batches)]
    all_metrics: List[Dict[str, float]] = []
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
        dag_stats = _attach_dag_inputs(prepared, resolver=resolver, model_cfg=model_cfg, train_cfg=train_cfg)
        if train_cfg.distributed_backend == "pmap":
            metrics = _eval_step_pmap(
                model,
                _shard_tree_for_pmap(prepared["model_inputs"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["targets"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["target_mask"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["reverse_indicator"], num_devices=num_devices),
                default_token_id,
            )
        else:
            metrics = _eval_step(
                model,
                prepared["model_inputs"],
                prepared["targets"],
                prepared["target_mask"],
                prepared["reverse_indicator"],
                default_token_id,
            )
        mf = _as_float_metrics(metrics)
        mf.update({k: float(v) for k, v in dag_stats.items()})
        all_metrics.append(mf)

    if not all_metrics:
        return {}
    keys = list(all_metrics[0].keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m[k] for m in all_metrics]))
    return out


def train_supervised_dag_latent(train_cfg: DAGLatentTrainConfig) -> Dict[str, Any]:
    if train_cfg.distributed_backend not in ("none", "pmap"):
        raise ValueError(f"Unsupported distributed_backend: {train_cfg.distributed_backend}")
    if str(train_cfg.stage) not in ("A", "B", "C", "A_B_C"):
        raise ValueError(f"Unsupported stage: {train_cfg.stage}")
    if str(train_cfg.dag_source_mode) not in ("dual", "cache", "scene_derived"):
        raise ValueError(f"Unsupported dag_source_mode: {train_cfg.dag_source_mode}")

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = _resolve_model_preset_dag(str(train_cfg.model_preset))
    model_cfg.dag_encoder.enabled = True
    model_cfg.dag_conditioning.enabled = True

    if train_cfg.tokenizer_mode == "adv_bmt_parity":
        tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(train_cfg.skip_steps)))
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
    skip_reason_counts = {}
    for s in skipped_records:
        r = str(s.get("reason", "unknown"))
        skip_reason_counts[r] = int(skip_reason_counts.get(r, 0) + 1)
    strict_violations = [x for x in skipped_records if str(x.get("reason")) == "strict_91_mismatch"]

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
        raise ValueError("No training scenarios available after filtering")

    num_devices = max(1, len(jax.local_devices()))
    if train_cfg.distributed_backend == "pmap" and int(train_cfg.batch_size) % int(num_devices) != 0:
        raise ValueError(
            f"batch_size ({train_cfg.batch_size}) must be divisible by num_devices ({num_devices}) for pmap"
        )

    total_steps_target = _stage_total_steps(train_cfg)
    lr_schedule, lr_meta = _build_lr_schedule(train_cfg, total_steps_target)
    tx = optax.chain(
        optax.clip_by_global_norm(float(train_cfg.grad_clip_norm)),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=float(train_cfg.weight_decay),
            b1=0.9,
            b2=0.95,
            eps=1e-5,
            mu_dtype=(jnp.float32 if train_cfg.precision == "bf16-mixed" else None),
        ),
    )

    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
    optimizer = nnx.Optimizer(model, tx)

    start_step = 0
    resume_runtime_state: Dict[str, Any] = {}
    if train_cfg.resume_checkpoint:
        ckpt_path = Path(train_cfg.resume_checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "last.pkl"
        if ckpt_path.is_file():
            start_step, resume_runtime_state, _ = _load_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=optimizer,
            )
            print(f"Resumed checkpoint: {ckpt_path} (step={start_step})")

    split_hashes = {"train": _hash_indices(train_indices), "val": _hash_indices(val_indices)}
    run_meta = {
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "data_source_mode": split_mode,
        "resolved_data_dirs": resolved_dirs,
        "split_hashes": split_hashes,
        "distributed": {"backend": str(train_cfg.distributed_backend), "num_devices": int(num_devices)},
        "precision": str(train_cfg.precision),
        "lr_schedule": lr_meta,
        "total_steps_target": int(total_steps_target),
        "dag": {
            "source_mode": str(train_cfg.dag_source_mode),
            "cache_dir": str(train_cfg.dag_cache_dir),
            "cache_strict": bool(train_cfg.dag_cache_strict),
            "stage": str(train_cfg.stage),
        },
        "artifacts": artifact_paths,
        "created_at": int(time.time()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    resolver = DAGSourceResolver(
        mode=str(train_cfg.dag_source_mode),
        cache_dir=str(train_cfg.dag_cache_dir),
        cache_strict=bool(train_cfg.dag_cache_strict),
    )
    metrics_log_path = output_dir / "metrics.jsonl"
    train_rng = np.random.default_rng(train_cfg.seed + 7)

    global_step = int(start_step)
    epoch = int(resume_runtime_state.get("epoch", 0))
    best_eval_loss = float("inf")
    best_eval_step = -1
    stage_scale_trees: Dict[str, Any] = {}
    stage_prev = ""
    epoch_indices: Optional[np.ndarray] = None
    batch_cursor = 0
    t0 = time.time()

    def _runtime_state(stage_now: str) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "batch_cursor_in_epoch": int(batch_cursor),
            "split_hashes": dict(split_hashes),
            "active_stage": str(stage_now),
            "dag_source_mode": str(train_cfg.dag_source_mode),
            "dag_cache_strict": bool(train_cfg.dag_cache_strict),
        }

    while global_step < int(total_steps_target):
        if epoch_indices is None:
            epoch_indices = np.asarray(train_indices, dtype=np.int32).copy()
            train_rng.shuffle(epoch_indices)
            batch_cursor = 0
        if batch_cursor >= len(epoch_indices):
            epoch += 1
            epoch_indices = None
            batch_cursor = 0
            continue

        stage_now = _resolve_stage(train_cfg, global_step + 1)
        _apply_stage_dropout(model, train_cfg=train_cfg, stage=stage_now)
        if stage_now != stage_prev:
            if stage_now not in stage_scale_trees:
                stage_scale_trees[stage_now] = _build_grad_scale_tree(
                    model,
                    stage=stage_now,
                    stage_b_freeze_non_dag=bool(train_cfg.stage_b_freeze_non_dag),
                    stage_c_decoder_lr_scale=float(train_cfg.stage_c_decoder_lr_scale),
                    stage_c_dag_lr_scale=float(train_cfg.stage_c_dag_lr_scale),
                )
            stage_prev = stage_now

        idx_batch = epoch_indices[batch_cursor : batch_cursor + int(train_cfg.batch_size)]
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
        dag_stats = _attach_dag_inputs(prepared, resolver=resolver, model_cfg=model_cfg, train_cfg=train_cfg)

        step_start = time.time()
        grad_scale = stage_scale_trees[stage_now]
        if train_cfg.distributed_backend == "pmap":
            metrics = _train_step_scaled_pmap(
                model,
                optimizer,
                _shard_tree_for_pmap(prepared["model_inputs"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["targets"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["target_mask"], num_devices=num_devices),
                _shard_tree_for_pmap(prepared["reverse_indicator"], num_devices=num_devices),
                default_token_id,
                grad_scale,
            )
        else:
            metrics = _train_step_scaled(
                model,
                optimizer,
                prepared["model_inputs"],
                prepared["targets"],
                prepared["target_mask"],
                prepared["reverse_indicator"],
                default_token_id,
                grad_scale,
            )

        global_step += 1
        lr_now = float(np.asarray(jax.device_get(lr_schedule(global_step))))
        mf = _as_float_metrics(metrics)
        if train_cfg.distributed_backend == "pmap":
            mf["num_trained_tokens"] = float(mf.get("num_trained_tokens", 0.0) * num_devices)
        step_dt = max(1e-6, float(time.time() - step_start))
        mf["train/steps_per_sec"] = float(1.0 / step_dt)
        mf["train/tokens_per_sec"] = float(mf.get("num_trained_tokens", 0.0) / step_dt)
        mf["train/global_batch_size"] = float(train_cfg.batch_size)
        mf["train/num_devices"] = float(num_devices)
        mf["train/active_stage"] = float({"A": 0, "B": 1, "C": 2}.get(stage_now, -1))
        mf.update({k: float(v) for k, v in dag_stats.items()})
        _assert_finite_metrics(mf, phase="train", step=global_step)

        _write_jsonl(
            metrics_log_path,
            {
                "phase": "train",
                "step": int(global_step),
                "epoch": int(epoch),
                "batch_cursor_in_epoch": int(batch_cursor),
                "lr": float(lr_now),
                "stage": str(stage_now),
                "metrics": mf,
            },
        )
        if global_step % max(1, int(train_cfg.log_every_steps)) == 0:
            _print_metrics(
                prefix=f"train[{stage_now}]",
                step=global_step,
                metrics=mf,
                lr=lr_now,
                elapsed_s=time.time() - t0,
            )

        if len(val_indices) > 0 and global_step % max(1, int(train_cfg.eval_every_steps)) == 0:
            eval_metrics = _evaluate_dag(
                model=model,
                loader=val_loader,
                val_indices=val_indices,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                default_token_id=default_token_id,
                rng=train_rng,
                resolver=resolver,
                num_devices=num_devices,
            )
            _assert_finite_metrics(eval_metrics, phase="eval", step=global_step)
            _write_jsonl(
                metrics_log_path,
                {
                    "phase": "eval",
                    "step": int(global_step),
                    "epoch": int(epoch),
                    "lr": float(lr_now),
                    "stage": str(stage_now),
                    "metrics": eval_metrics,
                },
            )
            _print_metrics(
                prefix=f"eval [{stage_now}]",
                step=global_step,
                metrics=eval_metrics,
                lr=lr_now,
                elapsed_s=time.time() - t0,
            )
            eval_loss = float(eval_metrics.get("total_loss", float("inf")))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_step = int(global_step)
                path = _save_checkpoint(
                    output_dir=output_dir,
                    train_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    latest_metrics=eval_metrics,
                    runtime_state=_runtime_state(stage_now),
                )
                print(f"Saved improved checkpoint: {path}")

        if global_step % max(1, int(train_cfg.checkpoint_every_steps)) == 0:
            path = _save_checkpoint(
                output_dir=output_dir,
                train_step=global_step,
                model=model,
                optimizer=optimizer,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                latest_metrics=mf,
                runtime_state=_runtime_state(stage_now),
            )
            print(f"Saved checkpoint: {path}")

    final_stage = _resolve_stage(train_cfg, global_step)
    final_eval = (
        _evaluate_dag(
            model=model,
            loader=val_loader,
            val_indices=val_indices,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            default_token_id=default_token_id,
            rng=train_rng,
            resolver=resolver,
            num_devices=num_devices,
        )
        if len(val_indices) > 0
        else {}
    )
    _assert_finite_metrics(final_eval, phase="final_eval", step=global_step)
    final_ckpt = _save_checkpoint(
        output_dir=output_dir,
        train_step=global_step,
        model=model,
        optimizer=optimizer,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        latest_metrics=final_eval,
        runtime_state=_runtime_state(final_stage),
    )

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
        "final_eval_metrics": final_eval,
        "active_stage_end": str(final_stage),
        "dag_source_mode": str(train_cfg.dag_source_mode),
        "artifacts": artifact_paths,
        "elapsed_seconds": float(time.time() - t0),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
