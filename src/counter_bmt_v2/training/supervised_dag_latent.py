"""Opt-in staged supervised training with DAG latent conditioning."""

from __future__ import annotations

import atexit
import json
import math
import pickle
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
    _build_prescan_cache_key,
    _build_lr_schedule,
    _cast_tree_precision,
    _eval_step,
    _eval_step_pmap,
    _hash_indices,
    _load_prescan_cache,
    _load_checkpoint,
    _prepare_supervised_batch,
    _prescan_cache_path,
    _prescan_indices,
    _print_metrics,
    _resolve_data_sources,
    _resolve_model_preset,
    _save_prescan_cache,
    _save_checkpoint,
    _shard_tree_for_pmap,
    _write_jsonl,
    _write_split_artifacts,
    _compute_metric_dict,
)
from .forward_metrics import compute_forward_pass_metrics_for_batch, nanmean_metrics
from .tensorboard_logging import (
    create_tb_writer,
    tb_close,
    tb_write_scalar,
    tb_write_scalars,
    tb_write_text,
)


DAGSourceModeType = Literal["dual", "cache", "scene_derived"]
StageType = Literal["A", "B", "C", "A_B_C"]


@dataclass
class DAGLatentTrainConfig(SupervisedTrainConfig):
    # Optional override path to reuse a compatible prescan cache file from
    # another run (for example the non-DAG WOMD full training run).
    prescan_cache_source: str = ""
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
    # Keep resolved DAG payloads for eval-time qualitative export.
    prepared["_dag_payloads"] = dags
    prepared["_dag_sources"] = source_labels

    total = float(max(1, len(source_labels)))
    hits = float(sum(1 for s in source_labels if s == "cache"))
    fallback = float(sum(1 for s in source_labels if s == "scene_derived"))
    nulls = float(sum(1 for s in source_labels if s == "null"))
    return {
        "dag_source/cache_hit_rate": hits / total,
        "dag_source/fallback_rate": fallback / total,
        "dag_source/null_rate": nulls / total,
    }


def _sanitize_name(name: str) -> str:
    safe = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "scenario"


def _save_scene_snapshot_plot(
    *,
    out_file: Path,
    scenario_id: str,
    raw_batch: Dict[str, Any],
    sample_index: int,
    max_agents: int,
) -> bool:
    """Save a compact BEV scene snapshot for qualitative checkpoint tracking."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    try:
        pos_tn2 = np.asarray(raw_batch["agent_position_xy"][sample_index], dtype=np.float32)
        valid_tn = np.asarray(raw_batch["agent_valid_mask"][sample_index], dtype=bool)
        map_pos_m3 = np.asarray(raw_batch["map_position"][sample_index], dtype=np.float32)
        map_valid_mvn = np.asarray(raw_batch["map_feature_valid_mask"][sample_index], dtype=bool)
        tl_pos_ln2 = np.asarray(raw_batch["traffic_light_position"][sample_index], dtype=np.float32)
        tl_valid_tl = np.asarray(raw_batch["traffic_light_valid_mask"][sample_index], dtype=bool)
    except Exception:
        return False

    t_steps = int(pos_tn2.shape[0]) if pos_tn2.ndim == 3 else 0
    n_agents = int(pos_tn2.shape[1]) if pos_tn2.ndim == 3 else 0
    if t_steps <= 0 or n_agents <= 0:
        return False

    if np.any(valid_tn):
        valid_t = np.where(np.any(valid_tn, axis=1))[0]
        t0 = int(valid_t[0]) if valid_t.size > 0 else 0
    else:
        t0 = 0

    fig = plt.figure(figsize=(7.0, 7.0))
    ax = fig.add_subplot(1, 1, 1)

    # Static map feature centers.
    if map_pos_m3.ndim == 2 and map_pos_m3.shape[0] == map_valid_mvn.shape[0]:
        map_valid_m = np.any(map_valid_mvn, axis=1)
        map_xy = map_pos_m3[map_valid_m, :2]
        if map_xy.size > 0:
            keep = int(min(map_xy.shape[0], 4000))
            ax.scatter(map_xy[:keep, 0], map_xy[:keep, 1], s=1.0, c="#b0b0b0", alpha=0.35)

    # Traffic lights.
    if tl_valid_tl.ndim == 2 and t0 < tl_valid_tl.shape[0]:
        tl_valid_l = tl_valid_tl[t0]
    elif tl_valid_tl.ndim == 1:
        tl_valid_l = tl_valid_tl
    else:
        tl_valid_l = np.zeros((tl_pos_ln2.shape[0],), dtype=bool)

    if tl_pos_ln2.ndim == 2 and tl_pos_ln2.shape[0] == tl_valid_l.shape[0]:
        tl_xy = tl_pos_ln2[tl_valid_l, :2]
        if tl_xy.size > 0:
            ax.scatter(tl_xy[:, 0], tl_xy[:, 1], s=10, marker="s", c="#f39c12", alpha=0.9, label="traffic_light")

    # Agents at snapshot timestep + short history.
    valid_counts = np.sum(valid_tn, axis=0)
    order = np.argsort(-valid_counts)
    picked: List[int] = []
    if n_agents > 0:
        picked.append(0)  # SDC first.
    for idx in order.tolist():
        if idx in picked or valid_counts[idx] <= 0:
            continue
        picked.append(int(idx))
        if len(picked) >= int(max(1, max_agents)):
            break

    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(picked):
        color = cmap(i % 20)
        hist_start = max(0, t0 - 6)
        hist_mask = valid_tn[hist_start : t0 + 1, n]
        if np.any(hist_mask):
            hist_xy = pos_tn2[hist_start : t0 + 1, n][hist_mask]
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color=color, linewidth=1.0, alpha=0.8)
        if valid_tn[t0, n]:
            ax.scatter(pos_tn2[t0, n, 0], pos_tn2[t0, n, 1], s=18 if n == 0 else 10, color=color, alpha=0.95)

    ax.set_title(f"Scene Snapshot | {scenario_id} | t={t0}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=140)
    plt.close(fig)
    return True


def _export_eval_dag_context(
    *,
    output_dir: Path | None,
    global_step: int | None,
    prepared: Dict[str, Any],
    max_scenarios: int,
    max_agents: int,
) -> int:
    """Export DAG JSON + scene snapshot for a few eval scenarios."""
    if output_dir is None or global_step is None or int(max_scenarios) <= 0:
        return 0

    raw = prepared["raw_batch"]
    scenario_ids = list(raw.get("scenario_ids", []))
    dags: List[Dict[str, Any]] = list(prepared.get("_dag_payloads", []))
    sources: List[str] = list(prepared.get("_dag_sources", []))
    if not scenario_ids:
        return 0

    step_dir = Path(output_dir) / "forward_eval_context" / f"step_{int(global_step):07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for b, sid in enumerate(scenario_ids):
        if saved >= int(max_scenarios):
            break
        sid_s = str(sid)
        sid_safe = _sanitize_name(sid_s)
        scenario_dir = step_dir / sid_safe
        scenario_dir.mkdir(parents=True, exist_ok=True)

        dag_payload: Dict[str, Any]
        if b < len(dags) and isinstance(dags[b], dict):
            dag_payload = dict(dags[b])
        else:
            dag_payload = _empty_dag_payload(sid_s)
        dag_meta = dag_payload.get("metadata", {})
        if not isinstance(dag_meta, dict):
            dag_meta = {"metadata_raw": str(dag_meta)}
        dag_meta["resolved_source"] = str(sources[b]) if b < len(sources) else "unknown"
        dag_payload["metadata"] = dag_meta
        dag_payload["scenario_id"] = sid_s

        (scenario_dir / "dag.json").write_text(json.dumps(dag_payload, indent=2), encoding="utf-8")
        _save_scene_snapshot_plot(
            out_file=scenario_dir / "scene_snapshot.png",
            scenario_id=sid_s,
            raw_batch=raw,
            sample_index=b,
            max_agents=max(1, int(max_agents)),
        )
        saved += 1
    return saved


def _load_prescan_cache_from_file(*, cache_file: Path, cache_key: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not cache_file.is_file():
        return None
    try:
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
    except Exception:
        return None
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
    output_dir: Path | None = None,
    global_step: int | None = None,
) -> Dict[str, float]:
    if len(val_indices) == 0:
        return {}
    val_batches = [val_indices[i : i + int(train_cfg.batch_size)] for i in range(0, len(val_indices), int(train_cfg.batch_size))]
    if int(train_cfg.eval_batches) > 0:
        val_batches = val_batches[: int(train_cfg.eval_batches)]
    all_metrics: List[Dict[str, float]] = []
    forward_metrics_list: List[Dict[str, float]] = []
    forward_viz_saved = 0
    forward_artifact_saved = 0
    forward_context_saved = 0
    viz_remaining = max(0, int(train_cfg.forward_eval.viz_max_scenarios))
    artifact_remaining = max(0, int(train_cfg.forward_eval.artifact_max_scenarios_per_eval))
    context_remaining = max(viz_remaining, artifact_remaining)
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

        if bool(train_cfg.forward_eval.enabled):
            batch_forward_metrics, batch_viz_saved, batch_artifact_saved = compute_forward_pass_metrics_for_batch(
                model=model,
                prepared_batch=prepared,
                tokenizer=tokenizer,
                skip_steps=int(train_cfg.skip_steps),
                eval_cfg=train_cfg.forward_eval,
                seed=int(train_cfg.seed + int(global_step or 0) + int(idx_batch[0] if len(idx_batch) > 0 else 0)),
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

            # Export scenario snapshots + resolved DAGs aligned with rollout exports.
            context_budget = max(int(batch_viz_saved), int(batch_artifact_saved))
            if context_budget > 0 and context_remaining > 0:
                saved_now = _export_eval_dag_context(
                    output_dir=output_dir,
                    global_step=global_step,
                    prepared=prepared,
                    max_scenarios=min(context_remaining, context_budget),
                    max_agents=max(1, int(train_cfg.forward_eval.viz_max_agents)),
                )
                forward_context_saved += int(saved_now)
                context_remaining = max(0, context_remaining - int(saved_now))

    out: Dict[str, float] = {}
    if all_metrics:
        keys = list(all_metrics[0].keys())
        for k in keys:
            out[k] = float(np.mean([m[k] for m in all_metrics]))
    if forward_metrics_list:
        forward_avg = nanmean_metrics(forward_metrics_list)
        out.update({f"forward_approx/{k}": float(v) for k, v in forward_avg.items()})
        out["forward_approx/scenario_count"] = float(len(forward_metrics_list))
        out["forward_approx/visualizations_saved"] = float(forward_viz_saved)
        out["forward_approx/artifacts_saved"] = float(forward_artifact_saved)
        out["forward_approx/context_saved"] = float(forward_context_saved)
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

    cached_prescan: Optional[Dict[str, Any]] = None
    if bool(train_cfg.use_prescan_cache):
        cached_prescan = _load_prescan_cache(output_dir=output_dir, cache_key=prescan_cache_key)
        if cached_prescan is None:
            external = str(train_cfg.prescan_cache_source or "").strip()
            if external:
                cached_prescan = _load_prescan_cache_from_file(cache_file=Path(external), cache_key=prescan_cache_key)
                if cached_prescan is not None:
                    print(f"[prescan] loaded external cache: {external}")
                    _save_prescan_cache(output_dir=output_dir, cache_key=prescan_cache_key, data=cached_prescan)
                    print(f"[prescan] copied cache -> {_prescan_cache_path(output_dir)}")

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
        "split_settings": {
            "train_fraction": float(train_cfg.train_fraction),
            "sample_interval_training": int(train_cfg.sample_interval_training),
            "sample_interval_test": int(train_cfg.sample_interval_test),
            "strict_91_steps": bool(train_cfg.strict_91_steps),
            "prescan_log_every": int(train_cfg.prescan_log_every),
            "prescan_workers": int(train_cfg.prescan_workers),
            "use_prescan_cache": bool(train_cfg.use_prescan_cache),
            "prescan_cache_source": str(train_cfg.prescan_cache_source or ""),
        },
        "prescan_cache": {
            "enabled": bool(train_cfg.use_prescan_cache),
            "cache_hit": bool(cached_prescan is not None),
            "cache_path": str(_prescan_cache_path(output_dir)),
            "cache_key": prescan_cache_key,
        },
        "forward_metric_namespaces": ["forward_approx"],
        "forward_artifact_export": {
            "enabled": bool(train_cfg.forward_eval.export_artifacts),
            "subdir": str(train_cfg.forward_eval.artifact_output_subdir),
            "max_scenarios_per_eval": int(train_cfg.forward_eval.artifact_max_scenarios_per_eval),
            "metric_scope": str(train_cfg.forward_eval.metric_scope),
            "context_subdir": "forward_eval_context",
        },
        "tensorboard": {
            "enabled": bool(train_cfg.enable_tensorboard),
            "log_dir": str(output_dir / str(train_cfg.tensorboard_subdir)),
            "flush_secs": int(train_cfg.tensorboard_flush_secs),
            "log_run_config": bool(train_cfg.tensorboard_log_run_config),
        },
        "artifacts": artifact_paths,
        "created_at": int(time.time()),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
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
        tb_write_scalar(tb_writer, "train/lr", lr_now, global_step)
        tb_write_scalars(tb_writer, "train", mf, global_step)
        if global_step % max(1, int(train_cfg.log_every_steps)) == 0:
            _print_metrics(
                prefix=f"train[{stage_now}]",
                step=global_step,
                metrics=mf,
                lr=lr_now,
                elapsed_s=time.time() - t0,
            )

        did_eval_this_step = False
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
                output_dir=output_dir,
                global_step=global_step,
            )
            did_eval_this_step = True
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
            tb_write_scalars(tb_writer, "eval", eval_metrics, global_step)
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
                tb_write_scalar(tb_writer, "events/checkpoint_saved", 1.0, global_step)

        if global_step % max(1, int(train_cfg.checkpoint_every_steps)) == 0:
            # Ensure checkpoint steps have qualitative rollout exports even when
            # checkpoint cadence differs from eval cadence.
            if len(val_indices) > 0 and bool(train_cfg.forward_eval.enabled) and (not did_eval_this_step):
                ckpt_eval_metrics = _evaluate_dag(
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
                    output_dir=output_dir,
                    global_step=global_step,
                )
                _assert_finite_metrics(ckpt_eval_metrics, phase="checkpoint_eval", step=global_step)
                _write_jsonl(
                    metrics_log_path,
                    {
                        "phase": "checkpoint_eval",
                        "step": int(global_step),
                        "epoch": int(epoch),
                        "lr": float(lr_now),
                        "stage": str(stage_now),
                        "metrics": ckpt_eval_metrics,
                    },
                )
                tb_write_scalars(tb_writer, "checkpoint_eval", ckpt_eval_metrics, global_step)

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
            tb_write_scalar(tb_writer, "events/checkpoint_saved", 1.0, global_step)

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
            output_dir=output_dir,
            global_step=global_step,
        )
        if len(val_indices) > 0
        else {}
    )
    _assert_finite_metrics(final_eval, phase="final_eval", step=global_step)
    tb_write_scalars(tb_writer, "final_eval", final_eval, global_step)
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
        "final_eval_metrics": final_eval,
        "active_stage_end": str(final_stage),
        "dag_source_mode": str(train_cfg.dag_source_mode),
        "artifacts": artifact_paths,
        "elapsed_seconds": float(time.time() - t0),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if bool(train_cfg.tensorboard_log_run_config):
        tb_write_text(tb_writer, "run/summary", json.dumps(summary, indent=2), step=global_step)
    tb_close(tb_writer)
    return summary
