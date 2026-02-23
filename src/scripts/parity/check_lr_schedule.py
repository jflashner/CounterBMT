"""Check LR schedule parity against closed-form expectations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from counter_bmt_v2.training.supervised import SupervisedTrainConfig, _build_lr_schedule


def _parse_steps(raw: str) -> List[int]:
    vals: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("steps list is empty")
    return vals


def _legacy_closed_form(step: int, *, lr: float, warmup_steps: int, total_steps: int) -> float:
    warm = max(1, int(warmup_steps))
    total = max(int(total_steps), warm + 1)
    if step < warm:
        return float(lr) * (float(step) / float(warm))
    progress = (float(step) - float(warm)) / float(max(1, total - warm))
    progress = max(0.0, min(progress, 1.0))
    mult = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(lr) * mult


def _v2_closed_form(step: int, *, lr: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    warm = max(1, int(warmup_steps))
    total = max(int(total_steps), warm + 1)
    if step < warm:
        return float(step) / float(warm) * float(lr)
    progress = (float(step) - float(warm)) / float(max(1, total - warm))
    progress = max(0.0, min(progress, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(lr) - float(min_lr)) * cosine


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LR schedule parity values")
    parser.add_argument("--steps", type=str, required=True, help="comma-separated step list")
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--mode", type=str, default="legacy_cosine_zero", choices=["legacy_cosine_zero", "v2_cosine_minlr"])
    parser.add_argument("--max-abs-error", type=float, default=1e-9)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    steps = _parse_steps(args.steps)
    cfg = SupervisedTrainConfig(
        learning_rate=float(args.lr),
        min_learning_rate=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        lr_schedule_mode=str(args.mode),
    )
    schedule, schedule_meta = _build_lr_schedule(cfg, int(args.total_steps))

    rows: List[Dict[str, float]] = []
    abs_errs: List[float] = []
    for step in steps:
        actual = float(np.asarray(schedule(step)))
        if args.mode == "legacy_cosine_zero":
            expected = _legacy_closed_form(
                step,
                lr=float(args.lr),
                warmup_steps=int(args.warmup_steps),
                total_steps=int(args.total_steps),
            )
        else:
            expected = _v2_closed_form(
                step,
                lr=float(args.lr),
                warmup_steps=int(args.warmup_steps),
                total_steps=int(args.total_steps),
                min_lr=float(args.min_lr),
            )
        abs_err = abs(actual - expected)
        abs_errs.append(abs_err)
        rows.append(
            {
                "step": float(step),
                "actual": actual,
                "expected": expected,
                "abs_error": abs_err,
            }
        )

    max_abs_error = float(max(abs_errs) if abs_errs else 0.0)
    report = {
        "config": {
            "mode": str(args.mode),
            "lr": float(args.lr),
            "warmup_steps": int(args.warmup_steps),
            "total_steps": int(args.total_steps),
            "min_lr": float(args.min_lr),
            "max_abs_error_threshold": float(args.max_abs_error),
        },
        "schedule_meta": schedule_meta,
        "rows": rows,
        "max_abs_error": max_abs_error,
        "passed": bool(max_abs_error <= float(args.max_abs_error)),
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")
    if max_abs_error > float(args.max_abs_error):
        print("FAILED: LR schedule parity check failed")
        return 1
    print("PASSED: LR schedule parity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
