"""Run a short training benchmark and report throughput metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_train_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metrics_path.is_file():
        return rows
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") == "train":
                rows.append(rec)
    return rows


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CounterBMT throughput with a short train run")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--distributed-backend", type=str, default="none", choices=["none", "pmap"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--model-preset", type=str, default="midgpt_parity")
    parser.add_argument("--tokenizer-mode", type=str, default="adv_bmt_parity")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16-mixed"])
    parser.add_argument("--runtime-preset", type=str, default="adv_bmt_runtime_parity", choices=["none", "adv_bmt_runtime_parity"])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "counter_bmt_v2.cli.train_nnx_bmt",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(out_dir),
        "--runtime-preset",
        str(args.runtime_preset),
        "--model-preset",
        str(args.model_preset),
        "--tokenizer-mode",
        str(args.tokenizer_mode),
        "--distributed-backend",
        str(args.distributed_backend),
        "--precision",
        str(args.precision),
        "--batch-size",
        str(int(args.batch_size)),
        "--max-steps",
        str(int(args.max_steps)),
        "--eval-every",
        str(max(int(args.max_steps), 1)),
        "--eval-batches",
        "1",
        "--log-every",
        "10",
        "--checkpoint-every",
        str(max(int(args.max_steps), 1)),
    ]

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")
    proc = subprocess.run(cmd, env=env, check=False)
    if proc.returncode != 0:
        print(f"FAILED: benchmark training command exited {proc.returncode}")
        return int(proc.returncode)

    metrics_rows = _read_train_rows(out_dir / "metrics.jsonl")
    steps_per_sec = [
        float(row.get("metrics", {}).get("train/steps_per_sec"))
        for row in metrics_rows
        if row.get("metrics", {}).get("train/steps_per_sec") is not None
    ]
    tokens_per_sec = [
        float(row.get("metrics", {}).get("train/tokens_per_sec"))
        for row in metrics_rows
        if row.get("metrics", {}).get("train/tokens_per_sec") is not None
    ]
    report = {
        "config": {
            "data_dir": str(args.data_dir),
            "output_dir": str(out_dir),
            "distributed_backend": str(args.distributed_backend),
            "batch_size": int(args.batch_size),
            "max_steps": int(args.max_steps),
            "model_preset": str(args.model_preset),
            "tokenizer_mode": str(args.tokenizer_mode),
            "precision": str(args.precision),
            "runtime_preset": str(args.runtime_preset),
        },
        "summary": {
            "num_train_rows": int(len(metrics_rows)),
            "mean_steps_per_sec": _mean(steps_per_sec),
            "mean_tokens_per_sec": _mean(tokens_per_sec),
            "max_steps_per_sec": float(max(steps_per_sec) if steps_per_sec else float("nan")),
            "max_tokens_per_sec": float(max(tokens_per_sec) if tokens_per_sec else float("nan")),
        },
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        out_json = Path(args.json_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote benchmark report: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

