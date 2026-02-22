"""CLI entrypoint for Day 2 NNX supervised training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Allow `python src/counter_bmt_v2/cli/train_nnx_bmt.py ...` without requiring
# editable install; this keeps local no-admin workflows simple.
if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.training import ForwardPassEvalConfig, SupervisedTrainConfig, train_supervised


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
        default="paper_like_small",
        choices=["paper_like_small", "paper_like_full", "midgpt_parity"],
        help="model preset",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["forward", "reverse", "mixed"],
        help="token sequence direction mode",
    )
    parser.add_argument("--reverse-prob", type=float, default=0.5)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="paper_simple",
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


def main() -> int:
    args = parse_args()

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
        model_preset=args.model_preset,
        seed=args.seed,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_steps=(None if args.max_steps <= 0 else args.max_steps),
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        mode=args.mode,
        reverse_probability=args.reverse_prob,
        skip_steps=args.skip_steps,
        tokenizer_mode=args.tokenizer_mode,
        train_fraction=args.train_fraction,
        sample_interval_training=int(args.sample_interval_training),
        sample_interval_test=int(args.sample_interval_test),
        num_train_scenarios=(None if args.num_train_scenarios <= 0 else args.num_train_scenarios),
        num_val_scenarios=(None if args.num_val_scenarios <= 0 else args.num_val_scenarios),
        strict_91_steps=bool(args.strict_91_steps),
        eval_every_steps=args.eval_every,
        eval_batches=args.eval_batches,
        log_every_steps=args.log_every,
        checkpoint_every_steps=args.checkpoint_every,
        max_time_steps=args.max_time_steps,
        max_agents=args.max_agents,
        max_map_features=args.max_map_features,
        max_vectors_per_map_feature=args.max_vectors,
        max_traffic_lights=args.max_traffic_lights,
        center_to_map=(not args.no_center_to_map),
        resume_checkpoint=args.resume_checkpoint,
        relation_debug_dump_dir=args.relation_debug_dump_dir,
        relation_debug_dump_every_steps=max(0, int(args.relation_debug_dump_every)),
        relation_debug_max_batches=max(0, int(args.relation_debug_max_batches)),
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
