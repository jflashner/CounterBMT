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

    parser.add_argument("--data-dir", type=str, required=True, help="ScenarioNet dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/counter_bmt_v2_training")

    parser.add_argument(
        "--model-preset",
        type=str,
        default="paper_like_small",
        choices=["paper_like_small", "paper_like_full"],
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

    parser.add_argument("--train-fraction", type=float, default=0.95)
    parser.add_argument("--num-train-scenarios", type=int, default=-1)
    parser.add_argument("--num-val-scenarios", type=int, default=-1)

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

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cfg = SupervisedTrainConfig(
        data_dir=args.data_dir,
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
        train_fraction=args.train_fraction,
        num_train_scenarios=(None if args.num_train_scenarios <= 0 else args.num_train_scenarios),
        num_val_scenarios=(None if args.num_val_scenarios <= 0 else args.num_val_scenarios),
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
        forward_eval=ForwardPassEvalConfig(
            enabled=(not args.no_forward_eval),
            num_modes=max(1, int(args.forward_eval_modes)),
            sampling_method=args.forward_eval_sampling,
            temperature=float(args.forward_eval_temperature),
            topp=float(args.forward_eval_topp),
            topk=max(1, int(args.forward_eval_topk)),
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
