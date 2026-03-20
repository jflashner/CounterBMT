#!/usr/bin/env python3
"""Benchmark v2 MidGPT supervised train-step throughput.

This helper is intentionally speed-focused rather than memory-focused. It runs
the real v2 supervised train step on a short controlled batch stream and
reports warmup time, steady-state steps/sec, and tokens/sec.

The main use case is comparing the default decoder attention path against the
opt-in edge-sparse decoder path on the exact same data batches.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark v2 MidGPT train-step throughput.")
    p.add_argument("--data-dir", type=str, default="")
    p.add_argument("--train-data-dir", type=str, default="")
    p.add_argument("--val-data-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs/v2_midgpt_speed_benchmark")

    p.add_argument("--runtime-preset", type=str, default="legacy_midgpt_speed_recipe")
    p.add_argument("--model-preset", type=str, default=None, choices=["paper_like_small", "paper_like_full", "midgpt_parity"])
    p.add_argument("--tokenizer-mode", type=str, default=None, choices=["paper_simple", "adv_bmt_parity"])
    p.add_argument("--skip-steps", type=int, default=None)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--warmup-batches", type=int, default=2)
    p.add_argument("--benchmark-batches", type=int, default=5)
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
    p.add_argument("--collate-padding-mode", type=str, default=None, choices=["fixed", "batch_local", "bucketed"])
    p.add_argument("--no-center-to-map", action="store_true")

    p.add_argument(
        "--benchmark-mode",
        type=str,
        default="compare",
        choices=["baseline", "sparse_edge", "compare"],
        help="run only the baseline path, only the sparse-edge path, or both back to back",
    )
    p.add_argument(
        "--xla-preallocate",
        type=str,
        default="keep",
        choices=["keep", "true", "false"],
        help="optionally override XLA_PYTHON_CLIENT_PREALLOCATE before importing JAX",
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


def _block_tree(tree: Any, *, jax_module: Any) -> None:
    for leaf in jax_module.tree.leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if callable(block):
            block()


def _resolve_gpu_ids(jax_module: Any) -> List[int]:
    local_ids: List[int] = []
    for dev in jax_module.local_devices():
        if getattr(dev, "platform", "") != "gpu":
            continue
        hardware_id = getattr(dev, "local_hardware_id", None)
        if hardware_id is not None:
            local_ids.append(int(hardware_id))
    return sorted(set(local_ids))


def main() -> int:
    args = _parse_args()
    _configure_xla_env(args)
    _ensure_src_import_path()

    import jax
    import jax.numpy as jnp
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

    if not any(getattr(d, "platform", "") == "gpu" for d in jax.local_devices()):
        raise RuntimeError("GPU-backed JAX device required for speed benchmarking.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provided_flags = _collect_provided_flags(sys.argv[1:])
    resolved_runtime = _resolve_runtime_defaults(args, provided_flags)

    base_cfg = SupervisedTrainConfig(
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

    split_mode, train_loader, train_indices, _val_loader, _val_indices, resolved_dirs = _resolve_data_sources(base_cfg)
    if len(train_indices) == 0:
        raise ValueError("No training scenarios available for benchmarking.")

    rng_np = np.random.default_rng(base_cfg.seed + 1)
    epoch_indices = np.asarray(train_indices, dtype=np.int32).copy()
    rng_np.shuffle(epoch_indices)
    minibatches = _iter_minibatches(epoch_indices, base_cfg.batch_size)

    total_batches = int(args.warmup_batches) + int(args.benchmark_batches)
    if total_batches <= 0:
        raise ValueError("Need at least one warmup or benchmark batch.")
    if total_batches > len(minibatches):
        raise ValueError(
            f"Requested {total_batches} batches but only {len(minibatches)} are available from the selected split."
        )

    selected_minibatches = minibatches[:total_batches]
    selected_samples = [[train_loader.load(int(i)) for i in batch_ids] for batch_ids in selected_minibatches]

    mode_names: List[str]
    if str(args.benchmark_mode) == "compare":
        mode_names = ["baseline", "sparse_edge"]
    elif str(args.benchmark_mode) == "sparse_edge":
        mode_names = ["sparse_edge"]
    else:
        mode_names = ["baseline"]

    summary: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "resolved_runtime": dict(resolved_runtime),
        "resolved_data_dirs": resolved_dirs,
        "data_source_mode": split_mode,
        "gpu_ids": _resolve_gpu_ids(jax),
        "jax_devices": [str(d) for d in jax.local_devices()],
        "xla_env": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
            "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        },
        "batches": {
            "warmup": int(args.warmup_batches),
            "benchmark": int(args.benchmark_batches),
            "global_batch_size": int(args.batch_size),
        },
        "results": {},
    }

    for mode_name in mode_names:
        edge_sparse = bool(mode_name == "sparse_edge")
        train_cfg = SupervisedTrainConfig(**{**base_cfg.__dict__, "decoder_edge_sparse_attn": edge_sparse})

        model_cfg = _resolve_model_preset(train_cfg.model_preset)
        if train_cfg.decoder_edge_sparse_attn:
            model_cfg.decoder.edge_sparse_relation_attn = True

        if train_cfg.tokenizer_mode == "adv_bmt_parity":
            tokenizer = AdvBMTParityTokenizer(ParityTokenizerConfig(num_skipped_steps=int(train_cfg.skip_steps)))
            default_token_id = int(tokenizer.default_token_id)
        else:
            tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)
            default_token_id = int(tokenizer.action_to_token(0.0, 0.0))

        model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(train_cfg.seed))
        total_steps_target = max(1, total_batches)
        lr_schedule, _lr_schedule_meta = _build_lr_schedule(train_cfg, total_steps_target)

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
                f"batch_size ({train_cfg.batch_size}) must be divisible by num_devices ({num_devices}) for pmap benchmarking"
            )

        run_rng = np.random.default_rng(train_cfg.seed + 123)
        warmup_total_s = 0.0
        benchmark_total_s = 0.0
        benchmark_tokens = 0.0
        benchmark_metrics: List[Dict[str, float]] = []

        for batch_idx, samples in enumerate(selected_samples):
            prepared = _prepare_supervised_batch(
                samples,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                tokenizer=tokenizer,
                rng=run_rng,
                is_training=True,
            )

            start = time.time()
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
            dt = max(1e-6, float(time.time() - start))

            if batch_idx < int(args.warmup_batches):
                warmup_total_s += dt
                print(f"[v2-speed] {mode_name} warmup_batch={batch_idx} dt_s={dt:.4f}", flush=True)
                continue

            metrics_f = _as_float_metrics(metrics)
            if train_cfg.distributed_backend == "pmap":
                metrics_f["num_trained_tokens"] = float(metrics_f.get("num_trained_tokens", 0.0) * num_devices)
            benchmark_total_s += dt
            benchmark_tokens += float(metrics_f.get("num_trained_tokens", 0.0))
            benchmark_metrics.append(metrics_f)
            bench_idx = batch_idx - int(args.warmup_batches)
            print(
                f"[v2-speed] {mode_name} benchmark_batch={bench_idx} dt_s={dt:.4f} "
                f"tokens={metrics_f.get('num_trained_tokens', 0.0):.1f}",
                flush=True,
            )

        bench_steps = max(1, int(args.benchmark_batches))
        result = {
            "decoder_edge_sparse_attn": bool(edge_sparse),
            "warmup_total_s": float(warmup_total_s),
            "benchmark_total_s": float(benchmark_total_s),
            "benchmark_batches": int(args.benchmark_batches),
            "mean_step_s": float(benchmark_total_s / bench_steps),
            "steps_per_s": float(bench_steps / max(1e-6, benchmark_total_s)),
            "tokens_per_s": float(benchmark_tokens / max(1e-6, benchmark_total_s)),
            "mean_num_trained_tokens": float(benchmark_tokens / bench_steps),
            "mean_metrics": (
                {}
                if not benchmark_metrics
                else {
                    key: float(np.mean([m[key] for m in benchmark_metrics]))
                    for key in benchmark_metrics[0].keys()
                }
            ),
        }
        summary["results"][mode_name] = result

    if "baseline" in summary["results"] and "sparse_edge" in summary["results"]:
        base = summary["results"]["baseline"]
        sparse = summary["results"]["sparse_edge"]
        summary["comparison"] = {
            "speedup_steps_per_s": float(sparse["steps_per_s"] / max(1e-9, base["steps_per_s"])),
            "speedup_tokens_per_s": float(sparse["tokens_per_s"] / max(1e-9, base["tokens_per_s"])),
            "delta_mean_step_s": float(sparse["mean_step_s"] - base["mean_step_s"]),
        }

    out_path = output_dir / "speed_benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[v2-speed] wrote benchmark summary to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
