#!/usr/bin/env python3
"""Short paired-training probe for v2 vs legacy Adv-BMT MidGPT.

This script is meant to answer one practical question:

    "If we train both models for a short, controlled budget, do they start
    learning and behaving in similar ways?"

The probe is intentionally conservative:
- both trainers see the exact same symlinked subset directories
- both use forward-only MidGPT settings
- both write their native training artifacts
- then we reuse the existing head-to-head evaluator on the resulting checkpoints

The output is a compact summary JSON containing:
- shared subset metadata
- commands that were run
- checkpoint paths
- short training-curve summaries (when logs are available)
- head-to-head report path
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from counter_bmt_v2.data.scenarionet import ScenarioNetNNXLoader


def _run(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=so, stderr=se, check=False)
    return int(proc.returncode)


def _pick_subset_files(data_dir: Path, count: int, seed: int) -> List[Path]:
    loader = ScenarioNetNNXLoader(data_dir=data_dir)
    files = list(loader.files)
    if count < 0 or count >= len(files):
        return files

    import numpy as np

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(files), dtype=np.int32)
    rng.shuffle(idx)
    chosen = sorted((files[int(i)] for i in idx[: int(count)].tolist()), key=lambda p: p.relative_to(data_dir).as_posix())
    return chosen


def _materialize_symlink_subset(*, src_root: Path, dst_root: Path, files: Sequence[Path]) -> List[str]:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    rel_paths: List[str] = []
    for src in files:
        rel = src.relative_to(src_root)
        rel_paths.append(rel.as_posix())
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    return rel_paths


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} found under {root}")
    return matches[-1]


def _load_v2_training_summary(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    out: Dict[str, Any] = {"metrics_path": str(metrics_path)}
    if not metrics_path.is_file():
        out["available"] = False
        return out

    train_losses: List[float] = []
    eval_losses: List[float] = []
    train_acc: List[float] = []
    forward_sfde: List[float] = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            phase = str(rec.get("phase", ""))
            metrics = rec.get("metrics", {}) if isinstance(rec.get("metrics"), dict) else {}
            if phase == "train":
                if "total_loss" in metrics:
                    train_losses.append(float(metrics["total_loss"]))
                if "accuracy" in metrics:
                    train_acc.append(float(metrics["accuracy"]))
            elif phase == "eval":
                if "total_loss" in metrics:
                    eval_losses.append(float(metrics["total_loss"]))
                if "forward_approx/sfde_min" in metrics:
                    forward_sfde.append(float(metrics["forward_approx/sfde_min"]))

    out.update(
        {
            "available": True,
            "train_total_loss_first": (train_losses[0] if train_losses else None),
            "train_total_loss_last": (train_losses[-1] if train_losses else None),
            "train_accuracy_first": (train_acc[0] if train_acc else None),
            "train_accuracy_last": (train_acc[-1] if train_acc else None),
            "eval_total_loss_last": (eval_losses[-1] if eval_losses else None),
            "eval_forward_sfde_last": (forward_sfde[-1] if forward_sfde else None),
            "num_train_points": len(train_losses),
            "num_eval_points": len(eval_losses),
        }
    )
    return out


def _load_legacy_training_summary(run_root: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_root": str(run_root)}
    event_files = sorted(run_root.rglob("events.out.tfevents.*"))
    if not event_files:
        out["available"] = False
        return out

    out["event_file"] = str(event_files[-1])

    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        out["available"] = False
        out["reason"] = f"tensorboard event loader unavailable: {exc}"
        return out

    ea = event_accumulator.EventAccumulator(str(event_files[-1]))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    def _last_scalar(tag: str) -> float | None:
        if tag not in tags:
            return None
        vals = ea.Scalars(tag)
        if not vals:
            return None
        return float(vals[-1].value)

    def _first_scalar(tag: str) -> float | None:
        if tag not in tags:
            return None
        vals = ea.Scalars(tag)
        if not vals:
            return None
        return float(vals[0].value)

    out.update(
        {
            "available": True,
            "scalar_tags": sorted(tags),
            "train_total_loss_first": _first_scalar("train/total_loss"),
            "train_total_loss_last": _last_scalar("train/total_loss"),
            "train_accuracy_first": _first_scalar("train/accuracy"),
            "train_accuracy_last": _last_scalar("train/accuracy"),
            "train_entropy_last": _last_scalar("train/entropy"),
            "train_perplexity_last": _last_scalar("train/perplexity"),
        }
    )
    return out


def _write_head2head_registry(
    *,
    path: Path,
    dataset_dir: Path,
    output_dir: Path,
    n_scenarios: int,
    seed: int,
    v2_ckpt: Path,
    legacy_ckpt: Path,
    legacy_python_bin: str,
    legacy_root: Path,
    skip_steps: int,
) -> None:
    payload = {
        "run": {
            "dataset_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "n_scenarios": int(n_scenarios),
            "seed": int(seed),
            "metrics": {"mode": "approx"},
            "visualization": {"max_scenarios": 4, "max_agents": 10},
            "replay_export": {"enabled": True, "max_scenarios": 4, "mode_index": 0, "include_ground_truth": False},
            "legacy_policy": "required",
            "reuse_artifacts": False,
            "max_parallel_models": 1,
        },
        "models": [
            {
                "id": "v2_probe",
                "backend": "v2",
                "checkpoint": str(v2_ckpt),
                "runtime": {
                    "model_preset": "midgpt_parity",
                    "tokenizer_mode": "adv_bmt_parity",
                    "skip_steps": int(skip_steps),
                    "num_modes": 6,
                    "sampling_method": "topp",
                    "topp": 0.95,
                    "temperature": 1.0,
                },
            },
            {
                "id": "legacy_probe",
                "backend": "legacy_adv_bmt",
                "checkpoint": str(legacy_ckpt),
                "runtime": {
                    "skip_steps": int(skip_steps),
                    "num_modes": 6,
                    "sampling_method": "topp",
                    "topp": 0.95,
                    "temperature": 1.0,
                    "python_bin": str(legacy_python_bin),
                    "legacy_root": str(legacy_root),
                },
            },
        ],
    }
    # JSON is valid YAML, so this stays dependency-free while remaining compatible
    # with the existing head-to-head registry loader.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Short paired-learning probe for v2 vs legacy MidGPT.")
    p.add_argument("--train-data-dir", type=str, required=True)
    p.add_argument("--val-data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--python-bin",
        type=str,
        default="",
        help="deprecated shared python path; if set, it becomes the default for --v2-python-bin and --legacy-python-bin",
    )
    p.add_argument(
        "--v2-python-bin",
        type=str,
        default="",
        help="python executable for the v2 JAX environment; defaults to --python-bin or the current interpreter",
    )
    p.add_argument(
        "--legacy-python-bin",
        type=str,
        default="",
        help="python executable for the separate legacy Adv-BMT environment; defaults to --python-bin or the current interpreter",
    )
    p.add_argument(
        "--head2head-python-bin",
        type=str,
        default="",
        help="python executable for the head-to-head wrapper; defaults to the v2 python",
    )
    p.add_argument("--legacy-root", type=str, default="src/Adv-BMT")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-scenarios", type=int, default=512)
    p.add_argument("--val-scenarios", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--skip-steps", type=int, default=5)
    p.add_argument("--v2-max-steps", type=int, default=200)
    p.add_argument(
        "--v2-eval-batches",
        type=int,
        default=0,
        help="cap the number of v2 validation minibatches per eval; 0 means use the full shared validation subset",
    )
    p.add_argument("--legacy-epochs", type=int, default=4)
    p.add_argument("--legacy-limit-train-batches", type=int, default=50)
    p.add_argument(
        "--legacy-limit-val-batches",
        type=int,
        default=0,
        help="legacy Lightning validation batch budget; 0 means no validation batches when combined with EVAL_MOTION=False",
    )
    p.add_argument("--distributed-backend", choices=["none", "pmap"], default="none")
    p.add_argument("--precision", choices=["fp32", "bf16-mixed"], default="fp32")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    shared_python = str(args.python_bin).strip()
    v2_python_bin = str(args.v2_python_bin).strip() or shared_python or sys.executable
    legacy_python_bin = str(args.legacy_python_bin).strip() or shared_python or sys.executable
    head2head_python_bin = str(args.head2head_python_bin).strip() or v2_python_bin

    train_data_dir = Path(args.train_data_dir).resolve()
    val_data_dir = Path(args.val_data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    legacy_root = (REPO_ROOT / args.legacy_root).resolve() if not Path(args.legacy_root).is_absolute() else Path(args.legacy_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_dir = output_dir / "shared_subset"
    subset_train_dir = shared_dir / "train"
    subset_val_dir = shared_dir / "val"
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_files = _pick_subset_files(train_data_dir, int(args.train_scenarios), int(args.seed))
    val_files = _pick_subset_files(val_data_dir, int(args.val_scenarios), int(args.seed) + 1)
    train_rel = _materialize_symlink_subset(src_root=train_data_dir, dst_root=subset_train_dir, files=train_files)
    val_rel = _materialize_symlink_subset(src_root=val_data_dir, dst_root=subset_val_dir, files=val_files)

    subset_manifest = {
        "train_data_dir": str(train_data_dir),
        "val_data_dir": str(val_data_dir),
        "subset_train_dir": str(subset_train_dir),
        "subset_val_dir": str(subset_val_dir),
        "seed": int(args.seed),
        "train_files": train_rel,
        "val_files": val_rel,
    }
    (output_dir / "subset_manifest.json").write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = "src" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")

    v2_run_dir = output_dir / "v2_probe"
    legacy_run_dir = output_dir / "legacy_probe"
    head2head_dir = output_dir / "head2head"
    registry_path = output_dir / "head2head_registry.yaml"

    v2_cmd = [
        str(v2_python_bin),
        "src/counter_bmt_v2/cli/train_nnx_bmt.py",
        "--train-data-dir",
        str(subset_train_dir),
        "--val-data-dir",
        str(subset_val_dir),
        "--output-dir",
        str(v2_run_dir),
        "--runtime-preset",
        "legacy_midgpt_recipe",
        "--model-preset",
        "midgpt_parity",
        "--tokenizer-mode",
        "adv_bmt_parity",
        "--distributed-backend",
        str(args.distributed_backend),
        "--precision",
        str(args.precision),
        "--batch-size",
        str(int(args.batch_size)),
        "--epochs",
        str(int(args.legacy_epochs)),
        "--max-steps",
        str(int(args.v2_max_steps)),
        "--strict-91-steps",
        "--sample-interval-training",
        "1",
        "--sample-interval-test",
        "1",
        "--prescan-workers",
        str(int(args.num_workers)),
        "--eval-every",
        str(int(args.v2_max_steps)),
        "--eval-batches",
        str(int(args.v2_eval_batches)),
        "--checkpoint-every",
        str(int(args.v2_max_steps)),
        "--log-every",
        "10",
        "--forward-eval-modes",
        "6",
        "--forward-eval-sampling",
        "topp",
        "--forward-eval-temperature",
        "1.0",
        "--forward-eval-topp",
        "0.95",
        "--forward-eval-topk",
        "5",
        "--forward-export-artifacts",
        "--forward-artifact-max-scenarios",
        str(max(1, int(args.val_scenarios))),
        "--forward-viz-max-scenarios",
        "2",
    ]

    legacy_cmd = [
        str(legacy_python_bin),
        "src/Adv-BMT/bmt/train_motion.py",
        "--config-name",
        "0202_midgpt",
        "hydra.job.chdir=False",
        f"exp_name=midgpt_probe",
        f"seed={int(args.seed)}",
        "wandb=False",
        f"log_dir={str(legacy_run_dir)}",
        f"epochs={int(args.legacy_epochs)}",
        f"batch_size={int(args.batch_size)}",
        f"val_batch_size={int(args.val_batch_size)}",
        f"num_workers={int(args.num_workers)}",
        f"val_num_workers={int(args.num_workers)}",
        "num_sanity_val_steps=0",
        "EVAL_MOTION=False",
        f"limit_train_batches={int(args.legacy_limit_train_batches)}",
        f"limit_val_batches={int(args.legacy_limit_val_batches)}",
        f"DATA.TRAINING_DATA_DIR={str(subset_train_dir)}",
        f"DATA.TEST_DATA_DIR={str(subset_val_dir)}",
    ]

    print("Running v2 probe...")
    rc_v2 = _run(
        v2_cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout_path=logs_dir / "v2.stdout.log",
        stderr_path=logs_dir / "v2.stderr.log",
    )
    if rc_v2 != 0:
        print(f"v2 probe failed with rc={rc_v2}. See {logs_dir / 'v2.stderr.log'}", file=sys.stderr)
        return rc_v2

    print("Running legacy probe...")
    rc_legacy = _run(
        legacy_cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout_path=logs_dir / "legacy.stdout.log",
        stderr_path=logs_dir / "legacy.stderr.log",
    )
    if rc_legacy != 0:
        print(f"legacy probe failed with rc={rc_legacy}. See {logs_dir / 'legacy.stderr.log'}", file=sys.stderr)
        return rc_legacy

    v2_ckpt = _find_one(v2_run_dir / "checkpoints", "last.pkl")
    legacy_ckpt = _find_one(legacy_run_dir, "last.ckpt")

    _write_head2head_registry(
        path=registry_path,
        dataset_dir=subset_val_dir,
        output_dir=head2head_dir,
        n_scenarios=len(val_rel),
        seed=int(args.seed),
        v2_ckpt=v2_ckpt,
        legacy_ckpt=legacy_ckpt,
        legacy_python_bin=str(legacy_python_bin),
        legacy_root=legacy_root,
        skip_steps=int(args.skip_steps),
    )

    head2head_cmd = [
        str(head2head_python_bin),
        "src/scripts/eval/compare_models_head2head.py",
        "--registry",
        str(registry_path),
        "--output-dir",
        str(head2head_dir),
        "--no-reuse-artifacts",
    ]
    print("Running head-to-head eval...")
    rc_h2h = _run(
        head2head_cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout_path=logs_dir / "head2head.stdout.log",
        stderr_path=logs_dir / "head2head.stderr.log",
    )
    if rc_h2h != 0:
        print(f"head2head eval failed with rc={rc_h2h}. See {logs_dir / 'head2head.stderr.log'}", file=sys.stderr)
        return rc_h2h

    report_path = head2head_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}

    summary = {
        "subset_manifest": str(output_dir / "subset_manifest.json"),
        "commands": {
            "v2": v2_cmd,
            "legacy": legacy_cmd,
            "head2head": head2head_cmd,
        },
        "python_bins": {
            "v2": str(v2_python_bin),
            "legacy": str(legacy_python_bin),
            "head2head": str(head2head_python_bin),
        },
        "paths": {
            "v2_run_dir": str(v2_run_dir),
            "legacy_run_dir": str(legacy_run_dir),
            "v2_checkpoint": str(v2_ckpt),
            "legacy_checkpoint": str(legacy_ckpt),
            "head2head_registry": str(registry_path),
            "head2head_report": str(report_path),
        },
        "v2_training": _load_v2_training_summary(v2_run_dir),
        "legacy_training": _load_legacy_training_summary(legacy_run_dir),
        "head2head": {
            "report_path": str(report_path),
            "rankings": report.get("rankings", []),
            "aggregate": report.get("aggregate", []),
            "pairwise": report.get("pairwise", []),
        },
    }

    summary_path = output_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote probe summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
