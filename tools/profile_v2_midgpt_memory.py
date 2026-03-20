#!/usr/bin/env python3
"""Profile v2 MidGPT-style supervised training memory on a short controlled run.

This is the v2 companion to `tools/profile_legacy_midgpt_memory.py`.

The goal is deliberately narrow:

    "How much GPU memory does the v2 parity stack use on real training batches,
    and what batch-local shapes explain that footprint?"

Why a dedicated helper exists:
- The learning probe brings along extra orchestration that makes it harder to
  isolate raw training memory behavior.
- JAX/XLA memory is dominated by compiled step activations and temporaries,
  rather than the ~5M model parameters, so we want a short reproducible run
  that captures actual train-step peaks.
- The legacy comparison only makes sense if we can record the same kind of
  batch-local context on the v2 side.

Implementation notes:
- This script builds the same v2 model, tokenizer, loader, optimizer, and train
  step used by `train_nnx_bmt.py`.
- GPU peaks are captured with a lightweight `nvidia-smi` poller because JAX does
  not expose stable per-batch allocator peaks in the same way PyTorch does.
- By default we disable XLA preallocation for profiling runs so the measured
  peaks reflect actual demand rather than a reserved allocator pool. Override
  with `--xla-preallocate keep` if you want to inspect the current environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile v2 MidGPT GPU memory on short supervised training runs.")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--train-data-dir", type=str, default="")
    p.add_argument("--val-data-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs/v2_midgpt_memory_profile")

    p.add_argument("--runtime-preset", type=str, default="legacy_midgpt_recipe")
    p.add_argument("--model-preset", type=str, default=None, choices=["paper_like_small", "paper_like_full", "midgpt_parity"])
    p.add_argument("--tokenizer-mode", type=str, default=None, choices=["paper_simple", "adv_bmt_parity"])
    p.add_argument("--skip-steps", type=int, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--limit-train-batches", type=int, default=5)
    p.add_argument("--profile-batches", type=int, default=5)
    p.add_argument("--warmup-batches", type=int, default=0, help="run initial train batches before starting detailed profiling")
    p.add_argument("--precision", type=str, default="bf16-mixed", choices=["fp32", "bf16-mixed"])
    p.add_argument("--distributed-backend", type=str, default="none", choices=["none", "pmap"])

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lr-schedule-mode", type=str, default=None, choices=["v2_cosine_minlr", "legacy_cosine_zero"])

    p.add_argument("--mode", type=str, default=None, choices=["forward", "reverse", "mixed"])
    p.add_argument("--reverse-prob", type=float, default=None)

    p.add_argument("--train-fraction", type=float, default=0.95)
    p.add_argument("--sample-interval-training", type=int, default=1)
    p.add_argument("--num-train-scenarios", type=int, default=-1)

    p.add_argument("--strict-91-steps", action="store_true")
    p.add_argument("--max-time-steps", type=int, default=91)
    p.add_argument("--max-agents", type=int, default=128)
    p.add_argument("--max-map-features", type=int, default=512)
    p.add_argument("--max-vectors", type=int, default=128)
    p.add_argument("--max-traffic-lights", type=int, default=64)
    p.add_argument("--collate-padding-mode", type=str, default=None, choices=["fixed", "batch_local"])
    p.add_argument("--no-center-to-map", action="store_true")

    p.add_argument("--poll-interval-ms", type=int, default=50)
    p.add_argument(
        "--gpu-indices",
        type=str,
        default="",
        help="comma-separated physical GPU indices for nvidia-smi polling; defaults to detected visible devices",
    )
    p.add_argument(
        "--xla-preallocate",
        type=str,
        default="false",
        choices=["keep", "true", "false"],
        help="set XLA_PYTHON_CLIENT_PREALLOCATE before importing JAX (default: false for demand-style profiling)",
    )
    p.add_argument(
        "--xla-mem-fraction",
        type=str,
        default="",
        help="optional XLA_PYTHON_CLIENT_MEM_FRACTION value to export before importing JAX",
    )
    return p.parse_args()


def _configure_xla_env(args: argparse.Namespace) -> None:
    if str(args.xla_preallocate) != "keep":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if str(args.xla_preallocate) == "true" else "false"
    if str(args.xla_mem_fraction).strip():
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction).strip()


def _ensure_src_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


class NvidiaSmiPoller:
    """Capture coarse but reliable GPU-memory peaks during JAX train steps."""

    def __init__(self, *, gpu_indices: Sequence[int], poll_interval_s: float) -> None:
        if not gpu_indices:
            raise ValueError("gpu_indices must be non-empty for NvidiaSmiPoller")
        self.gpu_indices = [int(x) for x in gpu_indices]
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._window_active = False
        self._last_sample: Dict[str, Any] = {}
        self._peak_sample: Dict[str, Any] = {}
        self._peak_total_used_mib: int = 0

    def _query(self) -> Dict[str, Any]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        rows: List[Dict[str, int]] = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[0])
                used = int(parts[1])
                total = int(parts[2])
            except Exception:
                continue
            if idx in self.gpu_indices:
                rows.append({"index": idx, "used_mib": used, "total_mib": total})
        rows.sort(key=lambda x: x["index"])
        total_used = int(sum(x["used_mib"] for x in rows))
        total_capacity = int(sum(x["total_mib"] for x in rows))
        return {
            "timestamp_s": float(time.time()),
            "gpus": rows,
            "total_used_mib": total_used,
            "total_capacity_mib": total_capacity,
            "total_used_bytes": int(total_used * 1024 * 1024),
            "total_capacity_bytes": int(total_capacity * 1024 * 1024),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._query()
            except Exception:
                time.sleep(self.poll_interval_s)
                continue
            with self._lock:
                self._last_sample = sample
                if self._window_active and int(sample.get("total_used_mib", 0)) >= self._peak_total_used_mib:
                    self._peak_total_used_mib = int(sample.get("total_used_mib", 0))
                    self._peak_sample = sample
            time.sleep(self.poll_interval_s)

    def start(self) -> None:
        if self._thread is not None:
            return
        # Fail fast before spawning the thread if nvidia-smi is unavailable.
        _ = self._query()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="nvidia-smi-poller")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 5.0 * self.poll_interval_s))
            self._thread = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self._last_sample:
                return dict(self._last_sample)
        return self._query()

    def begin_window(self) -> Dict[str, Any]:
        sample = self.snapshot()
        with self._lock:
            self._window_active = True
            self._peak_total_used_mib = int(sample.get("total_used_mib", 0))
            self._peak_sample = dict(sample)
        return sample

    def end_window(self) -> Dict[str, Any]:
        sample = self.snapshot()
        with self._lock:
            peak = dict(self._peak_sample) if self._peak_sample else dict(sample)
            self._window_active = False
        return {"end": sample, "peak": peak}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _collect_provided_flags(argv: Sequence[str]) -> set[str]:
    provided: set[str] = set()
    for tok in argv:
        if not tok.startswith("--"):
            continue
        if "=" in tok:
            provided.add(tok.split("=", 1)[0])
        else:
            provided.add(tok)
    return provided


def _resolve_runtime_defaults(args: argparse.Namespace, provided_flags: set[str]) -> Dict[str, object]:
    from counter_bmt_v2.trajectory_jax import get_runtime_preset

    base_defaults: Dict[str, object] = {
        "model_preset": "paper_like_small",
        "tokenizer_mode": "paper_simple",
        "learning_rate": 3e-4,
        "warmup_steps": 200,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "skip_steps": 5,
        "lr_schedule_mode": "v2_cosine_minlr",
        "num_epochs": 3,
        "mode": "mixed",
        "reverse_probability": 0.5,
        "collate_padding_mode": "fixed",
    }
    resolved = dict(base_defaults)
    resolved.update(get_runtime_preset(str(args.runtime_preset)))

    explicit_map = {
        "--model-preset": ("model_preset", args.model_preset),
        "--tokenizer-mode": ("tokenizer_mode", args.tokenizer_mode),
        "--lr": ("learning_rate", args.lr),
        "--warmup-steps": ("warmup_steps", args.warmup_steps),
        "--weight-decay": ("weight_decay", args.weight_decay),
        "--grad-clip": ("grad_clip_norm", args.grad_clip),
        "--skip-steps": ("skip_steps", args.skip_steps),
        "--lr-schedule-mode": ("lr_schedule_mode", args.lr_schedule_mode),
        "--mode": ("mode", args.mode),
        "--reverse-prob": ("reverse_probability", args.reverse_prob),
        "--collate-padding-mode": ("collate_padding_mode", args.collate_padding_mode),
    }
    for flag, (key, value) in explicit_map.items():
        if flag in provided_flags and value is not None:
            resolved[key] = value
    return resolved


def _resolve_gpu_indices(*, args: argparse.Namespace, jax_module: Any) -> List[int]:
    raw = str(args.gpu_indices).strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    gpu_devices = [d for d in jax_module.local_devices() if getattr(d, "platform", "") == "gpu"]
    local_ids: List[int] = []
    for dev in gpu_devices:
        hardware_id = getattr(dev, "local_hardware_id", None)
        if hardware_id is not None:
            local_ids.append(int(hardware_id))
    local_ids = sorted(set(local_ids))

    if local_ids:
        if str(args.distributed_backend) == "pmap":
            return local_ids
        return [local_ids[0]]

    env = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if env:
        try:
            parsed = [int(x.strip()) for x in env.split(",") if x.strip()]
            if parsed:
                return parsed if str(args.distributed_backend) == "pmap" else [parsed[0]]
        except Exception:
            pass
    return [0]


def _build_batch_summary(prepared: Dict[str, Any]) -> Dict[str, Any]:
    raw = prepared["raw_batch"]
    targets = np.asarray(prepared["targets"])
    target_mask = np.asarray(prepared["target_mask"])
    reverse_indicator = np.asarray(prepared["reverse_indicator"])

    agent_valid = np.asarray(raw["agent_valid_mask"], dtype=bool)
    map_valid = np.asarray(raw["map_feature_valid_mask"], dtype=bool)
    tl_valid = np.asarray(raw["traffic_light_valid_mask"], dtype=bool)

    collate_shape = dict(raw.get("collate_shape", {}))
    summary: Dict[str, Any] = {
        "scenario_ids": [str(x) for x in raw.get("scenario_ids", [])],
        "collate_shape": collate_shape,
        "targets_shape": list(targets.shape),
        "target_mask_shape": list(target_mask.shape),
        "sample_steps": np.asarray(prepared["sample_steps"], dtype=np.int32).tolist(),
        "reverse_indicator": reverse_indicator.astype(np.int32).tolist(),
        "active_agents_per_sample": agent_valid.any(axis=1).sum(axis=1).astype(np.int32).tolist(),
        "agent_valid_cells_per_sample": agent_valid.sum(axis=(1, 2)).astype(np.int32).tolist(),
        "active_modeled_agents_per_sample": target_mask.astype(bool).any(axis=1).sum(axis=1).astype(np.int32).tolist(),
        "decoder_valid_cells_per_sample": target_mask.sum(axis=(1, 2)).astype(np.int32).tolist(),
        # Count both map tokens and valid vectors because the parity path uses a
        # token-per-polyline representation whose internal vector budget is fixed.
        "valid_map_tokens_per_sample": map_valid.any(axis=-1).sum(axis=-1).astype(np.int32).tolist(),
        "valid_map_vectors_per_sample": map_valid.sum(axis=(1, 2)).astype(np.int32).tolist(),
        "active_traffic_lights_per_sample": tl_valid.any(axis=1).sum(axis=1).astype(np.int32).tolist(),
    }
    return summary


def _count_model_parameters(state: Any) -> Dict[str, int]:
    import jax.numpy as jnp
    import jax

    total = 0
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, jnp.ndarray):
            total += int(leaf.size)
    return {"total": int(total), "trainable": int(total)}


def _block_tree(tree: Any, *, jax_module: Any) -> None:
    for leaf in jax_module.tree.leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def main() -> int:
    args = _parse_args()
    _configure_xla_env(args)
    _ensure_src_import_path()

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx

    from counter_bmt_v2.training.supervised import (
        SupervisedTrainConfig,
        _as_float_metrics,
        _build_lr_schedule,
        _iter_minibatches,
        _prepare_supervised_batch,
        _resolve_data_sources,
        _resolve_model_preset,
        _shard_tree_for_pmap,
        _train_step,
        _train_step_pmap,
    )
    from counter_bmt_v2.trajectory_jax import (
        AdvBMTParityTokenizer,
        BidirectionalMotionTokenizer,
        NNXBidirectionalMotionTransformer,
        ParityTokenizerConfig,
    )

    provided_flags = _collect_provided_flags(sys.argv[1:])
    resolved_runtime = _resolve_runtime_defaults(args, provided_flags)

    if not any(getattr(d, "platform", "") == "gpu" for d in jax.local_devices()):
        raise RuntimeError("GPU-backed JAX device required for v2 MidGPT memory profiling.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = SupervisedTrainConfig(
        data_dir=str(args.data_dir).strip(),
        train_data_dir=str(args.train_data_dir).strip(),
        val_data_dir=str(args.val_data_dir).strip(),
        output_dir=str(output_dir),
        model_preset=str(resolved_runtime["model_preset"]),
        seed=int(args.seed),
        num_epochs=1,
        batch_size=int(args.batch_size),
        max_steps=None,
        learning_rate=float(resolved_runtime["learning_rate"]),
        min_learning_rate=float(args.min_lr),
        warmup_steps=int(resolved_runtime["warmup_steps"]),
        weight_decay=float(resolved_runtime["weight_decay"]),
        grad_clip_norm=float(resolved_runtime["grad_clip_norm"]),
        lr_schedule_mode=str(resolved_runtime["lr_schedule_mode"]),
        distributed_backend=str(args.distributed_backend),
        precision=str(args.precision),
        mode=str(resolved_runtime["mode"]),
        reverse_probability=float(resolved_runtime["reverse_probability"]),
        tokenizer_mode=str(resolved_runtime["tokenizer_mode"]),
        skip_steps=int(resolved_runtime["skip_steps"]),
        train_fraction=float(args.train_fraction),
        sample_interval_training=int(args.sample_interval_training),
        sample_interval_test=1,
        num_train_scenarios=(None if int(args.num_train_scenarios) <= 0 else int(args.num_train_scenarios)),
        num_val_scenarios=1,
        strict_91_steps=bool(args.strict_91_steps),
        eval_every_steps=10**9,
        eval_batches=0,
        log_every_steps=1,
        checkpoint_every_steps=10**9,
        enable_tensorboard=False,
        tensorboard_log_run_config=False,
        max_time_steps=int(args.max_time_steps),
        max_agents=int(args.max_agents),
        max_map_features=int(args.max_map_features),
        max_vectors_per_map_feature=int(args.max_vectors),
        max_traffic_lights=int(args.max_traffic_lights),
        collate_padding_mode=str(resolved_runtime["collate_padding_mode"]),
        center_to_map=(not bool(args.no_center_to_map)),
        runtime_preset=str(args.runtime_preset),
        runtime_resolved_overrides={k: v for k, v in resolved_runtime.items()},
    )

    model_cfg = _resolve_model_preset(train_cfg.model_preset)
    if train_cfg.tokenizer_mode == "adv_bmt_parity":
        tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(train_cfg.skip_steps)))
        default_token_id = int(tokenizer.default_token_id)
    else:
        tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)
        default_token_id = int(tokenizer.action_to_token(0.0, 0.0))

    split_mode, train_loader, train_indices, _val_loader, _val_indices, resolved_dirs = _resolve_data_sources(train_cfg)
    if len(train_indices) == 0:
        raise ValueError("No training scenarios available for profiling.")

    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
    total_steps_target = max(1, int(args.warmup_batches) + int(args.limit_train_batches))
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
    optimizer = nnx.Optimizer(model, tx)

    num_devices = max(1, len(jax.local_devices()))
    if train_cfg.distributed_backend == "pmap" and train_cfg.batch_size % num_devices != 0:
        raise ValueError(
            f"batch_size ({train_cfg.batch_size}) must be divisible by num_devices ({num_devices}) for pmap profiling"
        )

    gpu_indices = _resolve_gpu_indices(args=args, jax_module=jax)
    poller = NvidiaSmiPoller(
        gpu_indices=gpu_indices,
        poll_interval_s=float(int(args.poll_interval_ms)) / 1000.0,
    )
    poller.start()

    try:
        model_state = nnx.state(model)
        fit_start = {
            "gpu_indices": list(gpu_indices),
            "jax_devices": [str(d) for d in jax.local_devices()],
            "jax_platform": str(jax.default_backend()),
            "model_after_init": poller.snapshot(),
            "parameter_count": _count_model_parameters(model_state),
        }
        (output_dir / "fit_start.json").write_text(json.dumps(_jsonable(fit_start), indent=2), encoding="utf-8")
        print(
            "[v2-memory] fit_start "
            f"gpus={gpu_indices} total_params={fit_start['parameter_count']['total']} "
            f"used_bytes={fit_start['model_after_init'].get('total_used_bytes', 0)}",
            flush=True,
        )

        run_meta = {
            "output_dir": str(output_dir),
            "runtime_preset": str(args.runtime_preset),
            "resolved_runtime": dict(resolved_runtime),
            "train_cfg": {
                "model_preset": train_cfg.model_preset,
                "tokenizer_mode": train_cfg.tokenizer_mode,
                "skip_steps": int(train_cfg.skip_steps),
                "precision": str(train_cfg.precision),
                "distributed_backend": str(train_cfg.distributed_backend),
                "batch_size": int(train_cfg.batch_size),
                "collate_padding_mode": str(train_cfg.collate_padding_mode),
                "max_time_steps": int(train_cfg.max_time_steps),
                "max_agents": int(train_cfg.max_agents),
                "max_map_features": int(train_cfg.max_map_features),
                "max_vectors_per_map_feature": int(train_cfg.max_vectors_per_map_feature),
                "max_traffic_lights": int(train_cfg.max_traffic_lights),
                "strict_91_steps": bool(train_cfg.strict_91_steps),
            },
            "resolved_data_dirs": resolved_dirs,
            "data_source_mode": split_mode,
            "num_train_indices": int(len(train_indices)),
            "num_devices": int(num_devices),
            "xla_env": {
                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
            },
            "lr_schedule": lr_schedule_meta,
        }
        (output_dir / "run_meta.json").write_text(json.dumps(_jsonable(run_meta), indent=2), encoding="utf-8")

        train_rng = np.random.default_rng(train_cfg.seed + 1)
        epoch_indices = np.asarray(train_indices, dtype=np.int32).copy()
        train_rng.shuffle(epoch_indices)
        minibatches = _iter_minibatches(epoch_indices, train_cfg.batch_size)

        warmup_batches = max(0, int(args.warmup_batches))
        limit_train_batches = max(1, int(args.limit_train_batches))
        profile_batches = max(1, int(args.profile_batches))
        if warmup_batches + limit_train_batches > len(minibatches):
            limit_train_batches = max(0, len(minibatches) - warmup_batches)
        if limit_train_batches <= 0:
            raise ValueError("Not enough training minibatches available after warmup for profiling.")

        records: List[Dict[str, Any]] = []
        step_counter = 0

        def _run_step(prepared: Dict[str, Any]) -> Dict[str, Any]:
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
            _block_tree(metrics, jax_module=jax)
            return metrics

        for warm_idx in range(warmup_batches):
            idx_batch = minibatches[warm_idx]
            samples = [train_loader.load(int(i)) for i in idx_batch]
            prepared = _prepare_supervised_batch(
                samples,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                rng=train_rng,
                is_training=True,
            )
            _ = _run_step(prepared)
            step_counter += 1
            print(f"[v2-memory] warmup_step idx={warm_idx} done", flush=True)

        for local_idx in range(limit_train_batches):
            batch_idx = warmup_batches + local_idx
            idx_batch = minibatches[batch_idx]
            samples = [train_loader.load(int(i)) for i in idx_batch]
            prepared = _prepare_supervised_batch(
                samples,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                rng=train_rng,
                is_training=True,
            )

            batch_summary = _build_batch_summary(prepared)
            record = {
                "batch_idx": int(local_idx),
                "global_minibatch_idx": int(batch_idx),
                "batch_summary": batch_summary,
                "before_step": poller.begin_window(),
            }
            print(
                "[v2-memory] batch_start "
                f"idx={local_idx} "
                f"agents={batch_summary.get('collate_shape', {}).get('agents')} "
                f"map_tokens={batch_summary.get('collate_shape', {}).get('map_features')} "
                f"traffic_lights={batch_summary.get('collate_shape', {}).get('traffic_lights')} "
                f"token_steps={batch_summary.get('targets_shape', [0, 0, 0])[1] if batch_summary.get('targets_shape') else 0} "
                f"used_bytes={record['before_step'].get('total_used_bytes', 0)}",
                flush=True,
            )

            step_start = time.time()
            metrics = _run_step(prepared)
            step_dt = max(1e-6, float(time.time() - step_start))
            peak_window = poller.end_window()
            metrics_f = _as_float_metrics(metrics)
            if train_cfg.distributed_backend == "pmap":
                metrics_f["num_trained_tokens"] = float(metrics_f.get("num_trained_tokens", 0.0) * num_devices)
            metrics_f["train/steps_per_sec"] = float(1.0 / step_dt)
            metrics_f["train/tokens_per_sec"] = float(metrics_f.get("num_trained_tokens", 0.0) / step_dt)
            metrics_f["train/global_batch_size"] = float(train_cfg.batch_size)
            metrics_f["train/num_devices"] = float(num_devices)
            step_counter += 1
            lr_now = float(np.asarray(jax.device_get(lr_schedule(step_counter))))

            record.update(
                {
                    "duration_s": step_dt,
                    "after_step": peak_window["end"],
                    "peak_during_step": peak_window["peak"],
                    "metrics": metrics_f,
                    "lr": lr_now,
                }
            )
            if local_idx < profile_batches:
                records.append(record)
            print(
                "[v2-memory] batch_end "
                f"idx={local_idx} peak_used_bytes={record['peak_during_step'].get('total_used_bytes', 0)} "
                f"peak_used_mib={record['peak_during_step'].get('total_used_mib', 0)}",
                flush=True,
            )

        fit_end = poller.snapshot()
        peak_record = max(records, key=lambda x: int(x.get("peak_during_step", {}).get("total_used_bytes", 0)), default=None)
        summary = {
            "profiled_batches": records,
            "peak_batch_total_used_bytes": int(
                max((r.get("peak_during_step", {}).get("total_used_bytes", 0) for r in records), default=0)
            ),
            "peak_batch_total_used_mib": int(
                max((r.get("peak_during_step", {}).get("total_used_mib", 0) for r in records), default=0)
            ),
            "peak_batch_idx": None if peak_record is None else int(peak_record["batch_idx"]),
            "fit_end": fit_end,
        }
        (output_dir / "memory_profile.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print(
            "[v2-memory] fit_end "
            f"peak_used_bytes={summary['peak_batch_total_used_bytes']} "
            f"peak_used_mib={summary['peak_batch_total_used_mib']}",
            flush=True,
        )
    finally:
        poller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
