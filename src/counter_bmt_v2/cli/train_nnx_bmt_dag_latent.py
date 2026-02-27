"""CLI entrypoint for opt-in staged DAG-latent supervised training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Set

# Allow script-style execution without editable install.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.training import DAGLatentTrainConfig, train_supervised_dag_latent
from counter_bmt_v2.trajectory_jax import get_runtime_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NNX BMT with opt-in DAG latent conditioning")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Fallback ScenarioNet dataset directory (used when explicit split dirs are not provided)",
    )
    parser.add_argument("--train-data-dir", type=str, default="", help="Explicit ScenarioNet train dataset directory")
    parser.add_argument("--val-data-dir", type=str, default="", help="Explicit ScenarioNet val dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/counter_bmt_v2_training_dag_latent")

    parser.add_argument(
        "--model-preset",
        type=str,
        default=None,
        choices=["paper_like_small", "paper_like_full", "midgpt_parity", "midgpt_dag_latent"],
        help="model preset",
    )
    parser.add_argument(
        "--runtime-preset",
        type=str,
        default="none",
        choices=["none", "adv_bmt_runtime_parity"],
        help="training/runtime defaults; explicit CLI flags override these values",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--lr-schedule-mode",
        type=str,
        default=None,
        choices=["v2_cosine_minlr", "legacy_cosine_zero"],
        help="learning-rate schedule mode",
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default="none",
        choices=["none", "pmap"],
        help="single-host distributed backend",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16-mixed"],
        help="training precision mode",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["forward", "reverse", "mixed"],
        help="token sequence direction mode",
    )
    parser.add_argument("--reverse-prob", type=float, default=0.5)
    parser.add_argument("--skip-steps", type=int, default=None)
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default=None,
        choices=["paper_simple", "adv_bmt_parity"],
        help="tokenizer implementation mode",
    )

    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--sample-interval-training", type=int, default=1)
    parser.add_argument("--sample-interval-test", type=int, default=1)
    parser.add_argument("--num-train-scenarios", type=int, default=-1)
    parser.add_argument("--num-val-scenarios", type=int, default=-1)
    parser.add_argument(
        "--strict-91-steps",
        action="store_true",
        help="fail fast if any selected scenario horizon is not exactly 91 steps",
    )

    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=200)

    parser.add_argument("--max-time-steps", type=int, default=91)
    parser.add_argument("--max-agents", type=int, default=128)
    parser.add_argument("--max-map-features", type=int, default=512)
    parser.add_argument("--max-vectors", type=int, default=128)
    parser.add_argument("--max-traffic-lights", type=int, default=64)
    parser.add_argument("--no-center-to-map", action="store_true")

    parser.add_argument("--resume-checkpoint", type=str, default="")

    # DAG source controls.
    parser.add_argument(
        "--dag-source-mode",
        type=str,
        default="dual",
        choices=["dual", "cache", "scene_derived"],
    )
    parser.add_argument("--dag-cache-dir", type=str, default="")
    parser.add_argument("--dag-cache-strict", action="store_true")

    # Stage schedule.
    parser.add_argument(
        "--stage",
        type=str,
        default="A_B_C",
        choices=["A", "B", "C", "A_B_C"],
    )
    parser.add_argument("--stage-a-steps", type=int, default=200)
    parser.add_argument("--stage-b-steps", type=int, default=200)
    parser.add_argument("--stage-c-steps", type=int, default=200)
    parser.add_argument("--stage-a-dag-dropout", type=float, default=1.0)
    parser.add_argument("--stage-b-dag-dropout", type=float, default=0.3)
    parser.add_argument("--stage-c-dag-dropout", type=float, default=0.1)
    parser.add_argument(
        "--stage-b-freeze-non-dag",
        dest="stage_b_freeze_non_dag",
        action="store_true",
        help="freeze non-DAG parameters during stage B",
    )
    parser.add_argument(
        "--no-stage-b-freeze-non-dag",
        dest="stage_b_freeze_non_dag",
        action="store_false",
        help="do not freeze non-DAG parameters during stage B",
    )
    parser.set_defaults(stage_b_freeze_non_dag=True)
    parser.add_argument("--stage-c-decoder-lr-scale", type=float, default=0.1)
    parser.add_argument("--stage-c-dag-lr-scale", type=float, default=1.0)

    return parser.parse_args()


def _collect_provided_flags(argv: list[str]) -> Set[str]:
    provided: Set[str] = set()
    for tok in argv:
        if not tok.startswith("--"):
            continue
        if "=" in tok:
            provided.add(tok.split("=", 1)[0])
        else:
            provided.add(tok)
    return provided


def _resolve_runtime_defaults(args: argparse.Namespace, provided_flags: Set[str]) -> Dict[str, object]:
    base_defaults: Dict[str, object] = {
        "model_preset": "midgpt_dag_latent",
        "tokenizer_mode": "adv_bmt_parity",
        "learning_rate": 3e-4,
        "warmup_steps": 200,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "skip_steps": 5,
        "lr_schedule_mode": "v2_cosine_minlr",
    }
    runtime_defaults = get_runtime_preset(str(args.runtime_preset))
    resolved: Dict[str, object] = dict(base_defaults)
    resolved.update(runtime_defaults)

    explicit_map = {
        "--model-preset": ("model_preset", args.model_preset),
        "--tokenizer-mode": ("tokenizer_mode", args.tokenizer_mode),
        "--lr": ("learning_rate", args.lr),
        "--warmup-steps": ("warmup_steps", args.warmup_steps),
        "--weight-decay": ("weight_decay", args.weight_decay),
        "--grad-clip": ("grad_clip_norm", args.grad_clip),
        "--skip-steps": ("skip_steps", args.skip_steps),
        "--lr-schedule-mode": ("lr_schedule_mode", args.lr_schedule_mode),
    }
    for flag, (key, value) in explicit_map.items():
        if flag in provided_flags and value is not None:
            resolved[key] = value
    return resolved


def main() -> int:
    argv = sys.argv[1:]
    args = parse_args()
    provided_flags = _collect_provided_flags(argv)
    resolved_runtime = _resolve_runtime_defaults(args, provided_flags)

    train_data_dir = str(args.train_data_dir).strip()
    val_data_dir = str(args.val_data_dir).strip()
    data_dir = str(args.data_dir).strip()
    has_explicit_split = bool(train_data_dir or val_data_dir)
    if has_explicit_split:
        if not train_data_dir or not val_data_dir:
            raise ValueError("Both --train-data-dir and --val-data-dir must be provided when using explicit split dirs")
    elif not data_dir:
        raise ValueError("Provide --data-dir (fallback) or both --train-data-dir and --val-data-dir")

    if int(args.sample_interval_training) < 1:
        raise ValueError(f"--sample-interval-training must be >= 1, got {args.sample_interval_training}")
    if int(args.sample_interval_test) < 1:
        raise ValueError(f"--sample-interval-test must be >= 1, got {args.sample_interval_test}")

    cfg = DAGLatentTrainConfig(
        data_dir=data_dir,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=str(args.output_dir),
        model_preset=str(resolved_runtime["model_preset"]),
        seed=int(args.seed),
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_steps=(None if int(args.max_steps) <= 0 else int(args.max_steps)),
        learning_rate=float(resolved_runtime["learning_rate"]),
        min_learning_rate=float(args.min_lr),
        warmup_steps=int(resolved_runtime["warmup_steps"]),
        weight_decay=float(resolved_runtime["weight_decay"]),
        grad_clip_norm=float(resolved_runtime["grad_clip_norm"]),
        lr_schedule_mode=str(resolved_runtime["lr_schedule_mode"]),
        distributed_backend=str(args.distributed_backend),
        precision=str(args.precision),
        mode=str(args.mode),
        reverse_probability=float(args.reverse_prob),
        skip_steps=int(resolved_runtime["skip_steps"]),
        tokenizer_mode=str(resolved_runtime["tokenizer_mode"]),
        train_fraction=float(args.train_fraction),
        sample_interval_training=int(args.sample_interval_training),
        sample_interval_test=int(args.sample_interval_test),
        num_train_scenarios=(None if int(args.num_train_scenarios) <= 0 else int(args.num_train_scenarios)),
        num_val_scenarios=(None if int(args.num_val_scenarios) <= 0 else int(args.num_val_scenarios)),
        strict_91_steps=bool(args.strict_91_steps),
        eval_every_steps=int(args.eval_every),
        eval_batches=int(args.eval_batches),
        log_every_steps=int(args.log_every),
        checkpoint_every_steps=int(args.checkpoint_every),
        max_time_steps=int(args.max_time_steps),
        max_agents=int(args.max_agents),
        max_map_features=int(args.max_map_features),
        max_vectors_per_map_feature=int(args.max_vectors),
        max_traffic_lights=int(args.max_traffic_lights),
        center_to_map=(not bool(args.no_center_to_map)),
        resume_checkpoint=str(args.resume_checkpoint),
        runtime_preset=str(args.runtime_preset),
        runtime_resolved_overrides={k: v for k, v in resolved_runtime.items()},
        dag_source_mode=str(args.dag_source_mode),
        dag_cache_dir=str(args.dag_cache_dir),
        dag_cache_strict=bool(args.dag_cache_strict),
        stage=str(args.stage),
        stage_a_steps=int(args.stage_a_steps),
        stage_b_steps=int(args.stage_b_steps),
        stage_c_steps=int(args.stage_c_steps),
        stage_a_dag_dropout_prob=float(args.stage_a_dag_dropout),
        stage_b_dag_dropout_prob=float(args.stage_b_dag_dropout),
        stage_c_dag_dropout_prob=float(args.stage_c_dag_dropout),
        stage_b_freeze_non_dag=bool(args.stage_b_freeze_non_dag),
        stage_c_decoder_lr_scale=float(args.stage_c_decoder_lr_scale),
        stage_c_dag_lr_scale=float(args.stage_c_dag_lr_scale),
    )

    summary = train_supervised_dag_latent(cfg)
    print("Training complete")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved: {Path(cfg.output_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

