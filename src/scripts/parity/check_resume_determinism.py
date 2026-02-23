"""Run resume determinism check (uninterrupted vs split+resume)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _run(cmd: List[str], env: Dict[str, str]) -> None:
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _load_train_losses(path: Path) -> Dict[int, float]:
    out: Dict[int, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") != "train":
                continue
            step = int(rec.get("step"))
            loss = float(rec.get("metrics", {}).get("total_loss"))
            out[step] = loss
    return out


def _mad(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(statistics.mean(abs(x) for x in xs))


def _build_base_cmd(
    *,
    data_dir: str,
    output_dir: str,
    steps: int,
    batch_size: int,
    seed: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "counter_bmt_v2.cli.train_nnx_bmt",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--runtime-preset",
        "adv_bmt_runtime_parity",
        "--batch-size",
        str(int(batch_size)),
        "--max-steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--train-fraction",
        "1.0",
        "--eval-every",
        "1000000",
        "--eval-batches",
        "0",
        "--checkpoint-every",
        str(max(1, int(steps))),
        "--log-every",
        "20",
        "--no-forward-eval",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume determinism parity check")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--steps-total", type=int, default=200)
    parser.add_argument("--split-step", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-mad", type=float, default=1e-6)
    args = parser.parse_args()

    if int(args.split_step) <= 0 or int(args.steps_total) <= int(args.split_step):
        raise ValueError("Require 0 < split-step < steps-total")

    root = Path(args.output_dir)
    full_dir = root / "full"
    split_dir = root / "split_first"
    resume_dir = root / "split_resume"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")

    cmd_full = _build_base_cmd(
        data_dir=str(args.data_dir),
        output_dir=str(full_dir),
        steps=int(args.steps_total),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    _run(cmd_full, env)

    cmd_split = _build_base_cmd(
        data_dir=str(args.data_dir),
        output_dir=str(split_dir),
        steps=int(args.split_step),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    _run(cmd_split, env)

    resume_ckpt = split_dir / "checkpoints" / "last.pkl"
    cmd_resume = _build_base_cmd(
        data_dir=str(args.data_dir),
        output_dir=str(resume_dir),
        steps=int(args.steps_total),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )
    cmd_resume.extend(["--resume-checkpoint", str(resume_ckpt)])
    _run(cmd_resume, env)

    full_losses = _load_train_losses(full_dir / "metrics.jsonl")
    resume_losses = _load_train_losses(resume_dir / "metrics.jsonl")

    common_steps = sorted(set(full_losses).intersection(resume_losses))
    common_steps = [s for s in common_steps if s > int(args.split_step)]
    tail_steps = common_steps[-100:] if len(common_steps) > 100 else common_steps

    diffs = [float(full_losses[s] - resume_losses[s]) for s in tail_steps]
    mad = _mad(diffs)

    report = {
        "config": {
            "data_dir": str(args.data_dir),
            "output_dir": str(root),
            "steps_total": int(args.steps_total),
            "split_step": int(args.split_step),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "max_mad_threshold": float(args.max_mad),
        },
        "summary": {
            "num_common_steps": int(len(common_steps)),
            "num_tail_steps_compared": int(len(tail_steps)),
            "mad": float(mad),
        },
        "paths": {
            "full_metrics": str(full_dir / "metrics.jsonl"),
            "resume_metrics": str(resume_dir / "metrics.jsonl"),
            "resume_checkpoint": str(resume_ckpt),
        },
        "passed": bool(np.isfinite(mad) and mad <= float(args.max_mad)),
    }
    print(json.dumps(report, indent=2))
    if not np.isfinite(mad) or mad > float(args.max_mad):
        print("FAILED: resume determinism check failed")
        return 1
    print("PASSED: resume determinism check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
