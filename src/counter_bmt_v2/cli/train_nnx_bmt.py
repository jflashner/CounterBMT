"""CLI entrypoint for Day 2 NNX supervised training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Set

# Allow `python src/counter_bmt_v2/cli/train_nnx_bmt.py ...` without requiring
# editable install; this keeps local no-admin workflows simple.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.training import ForwardPassEvalConfig, SupervisedTrainConfig, train_supervised
from counter_bmt_v2.trajectory_jax import get_runtime_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NNX Bidirectional Motion Transformer (CounterBMT v2)")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Fallback ScenarioNet dataset directory (used when explicit split dirs are not provided)",
    )
    parser.add_argument("--train-data-dir", type=str, default="", help="Explicit ScenarioNet train dataset directory")
    parser.add_argument("--val-data-dir", type=str, default="", help="Explicit ScenarioNet val dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/counter_bmt_v2_training")

    parser.add_argument(
        "--model-preset",
        type=str,
        default=None,
        choices=["paper_like_small", "paper_like_full", "midgpt_parity"],
        help="model preset",
    )
    parser.add_argument(
        "--runtime-preset",
        type=str,
        default="none",
        choices=["none", "adv_bmt_runtime_parity", "legacy_midgpt_recipe", "legacy_midgpt_speed_recipe"],
        help="training/runtime defaults; explicit CLI flags override these values",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument(
        "--prescan-log-every",
        type=int,
        default=5000,
        help="print dataset prescan progress every N scenarios (0 disables)",
    )
    parser.add_argument(
        "--prescan-workers",
        type=int,
        default=0,
        help="number of worker threads for startup prescan (0/1 = sequential)",
    )
    parser.add_argument(
        "--prescan-cache",
        dest="prescan_cache",
        action="store_true",
        help="reuse/save startup prescan cache when compatible with current split/config",
    )
    parser.add_argument(
        "--no-prescan-cache",
        dest="prescan_cache",
        action="store_false",
        help="disable startup prescan cache reuse",
    )
    parser.set_defaults(prescan_cache=True)
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
    parser.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        help="enable TensorBoard scalar logging",
    )
    parser.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="disable TensorBoard scalar logging",
    )
    parser.set_defaults(tensorboard=True)
    parser.add_argument(
        "--tensorboard-subdir",
        type=str,
        default="tensorboard",
        help="TensorBoard log subdirectory under --output-dir",
    )
    parser.add_argument(
        "--tensorboard-flush-secs",
        type=int,
        default=30,
        help="TensorBoard SummaryWriter flush interval in seconds",
    )
    parser.add_argument(
        "--no-tensorboard-log-run-config",
        dest="tensorboard_log_run_config",
        action="store_false",
        help="do not write run config/summary text entries to TensorBoard",
    )
    parser.set_defaults(tensorboard_log_run_config=True)

    parser.add_argument("--max-time-steps", type=int, default=91)
    parser.add_argument("--max-agents", type=int, default=128)
    parser.add_argument("--max-map-features", type=int, default=512)
    parser.add_argument("--max-vectors", type=int, default=128)
    parser.add_argument("--max-traffic-lights", type=int, default=64)
    parser.add_argument(
        "--collate-padding-mode",
        type=str,
        default=None,
        choices=["fixed", "batch_local", "bucketed"],
        help=(
            "batch padding policy: 'fixed' pads to configured ceilings, "
            "'batch_local' pads to per-batch maxima under those ceilings, "
            "'bucketed' rounds per-batch maxima up to reusable compile-friendly buckets"
        ),
    )
    parser.add_argument(
        "--decoder-edge-sparse-attn",
        dest="decoder_edge_sparse_attn",
        action="store_true",
        help="enable the opt-in edge-sparse decoder attention path for speed experiments",
    )
    parser.add_argument(
        "--no-decoder-edge-sparse-attn",
        dest="decoder_edge_sparse_attn",
        action="store_false",
        help="disable the opt-in edge-sparse decoder attention path",
    )
    parser.set_defaults(decoder_edge_sparse_attn=False)

    parser.add_argument(
        "--no-center-to-map",
        action="store_true",
        help="disable map-centering during data extraction",
    )

    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default="",
        help="checkpoint .pkl path or checkpoint directory containing last.pkl",
    )
    parser.add_argument(
        "--resume-strict-determinism",
        dest="resume_strict_determinism",
        action="store_true",
        help="require split/config hash match when resuming from checkpoint",
    )
    parser.add_argument(
        "--no-resume-strict-determinism",
        dest="resume_strict_determinism",
        action="store_false",
        help="allow resuming from checkpoint even if split/config hashes differ (useful for eval-only runs)",
    )
    parser.set_defaults(resume_strict_determinism=True)
    parser.add_argument(
        "--relation-debug-dump-dir",
        type=str,
        default="",
        help="optional directory for parity relation debug dumps",
    )
    parser.add_argument(
        "--relation-debug-dump-every",
        type=int,
        default=0,
        help="save relation debug bundle every N train steps (0 disables)",
    )
    parser.add_argument(
        "--relation-debug-max-batches",
        type=int,
        default=1,
        help="max number of relation debug bundles to save",
    )

    parser.add_argument(
        "--no-forward-eval",
        action="store_true",
        help="disable Adv-BMT-style forward-pass validation metrics",
    )
    parser.add_argument("--forward-eval-modes", type=int, default=6)
    parser.add_argument(
        "--forward-eval-sampling",
        type=str,
        default="topp",
        choices=["topp", "topk", "softmax", "argmax"],
    )
    parser.add_argument("--forward-eval-temperature", type=float, default=1.0)
    parser.add_argument("--forward-eval-topp", type=float, default=0.95)
    parser.add_argument("--forward-eval-topk", type=int, default=5)
    parser.add_argument(
        "--no-forward-viz",
        action="store_true",
        help="disable saving rollout-vs-GT visualizations during eval",
    )
    parser.add_argument("--forward-viz-max-scenarios", type=int, default=2)
    parser.add_argument("--forward-viz-max-agents", type=int, default=10)
    parser.add_argument(
        "--forward-export-artifacts",
        dest="forward_export_artifacts",
        action="store_true",
        help="export per-scenario forward-eval artifacts for offline strict parity checks",
    )
    parser.add_argument(
        "--no-forward-export-artifacts",
        dest="forward_export_artifacts",
        action="store_false",
        help="disable forward-eval artifact export",
    )
    parser.set_defaults(forward_export_artifacts=True)
    parser.add_argument(
        "--forward-artifact-max-scenarios",
        type=int,
        default=32,
        help="maximum number of scenarios to export per eval call",
    )
    parser.add_argument(
        "--forward-artifact-subdir",
        type=str,
        default="forward_eval_artifacts",
        help="output subdirectory for forward-eval artifact exports",
    )

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
        "--epochs": ("num_epochs", args.epochs),
        "--mode": ("mode", args.mode),
        "--reverse-prob": ("reverse_probability", args.reverse_prob),
        "--collate-padding-mode": ("collate_padding_mode", args.collate_padding_mode),
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

    cfg = SupervisedTrainConfig(
        data_dir=data_dir,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        output_dir=args.output_dir,
        model_preset=str(resolved_runtime["model_preset"]),
        seed=args.seed,
        num_epochs=int(resolved_runtime["num_epochs"]),
        batch_size=args.batch_size,
        max_steps=(None if args.max_steps <= 0 else args.max_steps),
        learning_rate=float(resolved_runtime["learning_rate"]),
        min_learning_rate=args.min_lr,
        warmup_steps=int(resolved_runtime["warmup_steps"]),
        weight_decay=float(resolved_runtime["weight_decay"]),
        grad_clip_norm=float(resolved_runtime["grad_clip_norm"]),
        lr_schedule_mode=str(resolved_runtime["lr_schedule_mode"]),
        distributed_backend=str(args.distributed_backend),
        precision=str(args.precision),
        mode=str(resolved_runtime["mode"]),
        reverse_probability=float(resolved_runtime["reverse_probability"]),
        skip_steps=int(resolved_runtime["skip_steps"]),
        tokenizer_mode=str(resolved_runtime["tokenizer_mode"]),
        train_fraction=args.train_fraction,
        sample_interval_training=int(args.sample_interval_training),
        sample_interval_test=int(args.sample_interval_test),
        prescan_log_every=max(0, int(args.prescan_log_every)),
        prescan_workers=max(0, int(args.prescan_workers)),
        use_prescan_cache=bool(args.prescan_cache),
        num_train_scenarios=(None if args.num_train_scenarios <= 0 else args.num_train_scenarios),
        num_val_scenarios=(None if args.num_val_scenarios <= 0 else args.num_val_scenarios),
        strict_91_steps=bool(args.strict_91_steps),
        eval_every_steps=args.eval_every,
        eval_batches=args.eval_batches,
        log_every_steps=args.log_every,
        checkpoint_every_steps=args.checkpoint_every,
        enable_tensorboard=bool(args.tensorboard),
        tensorboard_subdir=str(args.tensorboard_subdir),
        tensorboard_flush_secs=max(1, int(args.tensorboard_flush_secs)),
        tensorboard_log_run_config=bool(args.tensorboard_log_run_config),
        max_time_steps=args.max_time_steps,
        max_agents=args.max_agents,
        max_map_features=args.max_map_features,
        max_vectors_per_map_feature=args.max_vectors,
        max_traffic_lights=args.max_traffic_lights,
        collate_padding_mode=str(resolved_runtime["collate_padding_mode"]),
        decoder_edge_sparse_attn=bool(args.decoder_edge_sparse_attn),
        center_to_map=(not args.no_center_to_map),
        resume_checkpoint=args.resume_checkpoint,
        resume_strict_determinism=bool(args.resume_strict_determinism),
        relation_debug_dump_dir=args.relation_debug_dump_dir,
        relation_debug_dump_every_steps=max(0, int(args.relation_debug_dump_every)),
        relation_debug_max_batches=max(0, int(args.relation_debug_max_batches)),
        runtime_preset=str(args.runtime_preset),
        runtime_resolved_overrides={k: v for k, v in resolved_runtime.items()},
        forward_eval=ForwardPassEvalConfig(
            enabled=(not args.no_forward_eval),
            num_modes=max(1, int(args.forward_eval_modes)),
            sampling_method=args.forward_eval_sampling,
            temperature=float(args.forward_eval_temperature),
            topp=float(args.forward_eval_topp),
            topk=max(1, int(args.forward_eval_topk)),
            metric_scope="core_realism",
            export_artifacts=bool(args.forward_export_artifacts),
            artifact_output_subdir=str(args.forward_artifact_subdir),
            artifact_max_scenarios_per_eval=max(0, int(args.forward_artifact_max_scenarios)),
            save_visualizations=(not args.no_forward_viz),
            viz_max_scenarios=max(0, int(args.forward_viz_max_scenarios)),
            viz_max_agents=max(1, int(args.forward_viz_max_agents)),
        ),
    )

    summary = train_supervised(cfg)

    print("Training complete")
    print(json.dumps(summary, indent=2))
    print(f"Summary saved: {Path(cfg.output_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
