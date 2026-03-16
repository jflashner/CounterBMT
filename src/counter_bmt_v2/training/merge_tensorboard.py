"""Rebuild a single TensorBoard run from one or more metrics.jsonl logs."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .tensorboard_logging import create_tb_writer, tb_close, tb_write_scalar, tb_write_scalars, tb_write_text

_PHASE_PRIORITY = {
    "train": 0,
    "eval": 1,
    "checkpoint_eval": 2,
    "final_eval": 3,
}


def _phase_priority(phase: str) -> int:
    return int(_PHASE_PRIORITY.get(str(phase), 99))


def load_metric_records(run_dir: Path) -> List[Dict[str, Any]]:
    """Load metrics.jsonl rows from one run directory."""
    metrics_path = Path(run_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found in {run_dir}")

    records: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            row["_line_index"] = int(line_index)
            row["_run_dir"] = str(Path(run_dir))
            records.append(row)
    return records


def dedupe_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the latest record for each (phase, step) pair."""
    latest: Dict[tuple[str, int], Dict[str, Any]] = {}
    for record in records:
        phase = str(record.get("phase", ""))
        step = int(record.get("step", -1))
        latest[(phase, step)] = dict(record)
    return list(latest.values())


def sort_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort records into the order expected by TensorBoard."""
    return sorted(
        (dict(record) for record in records),
        key=lambda row: (
            int(row.get("step", -1)),
            _phase_priority(str(row.get("phase", ""))),
            int(row.get("_source_index", 0)),
            int(row.get("_line_index", 0)),
        ),
    )


def write_record(writer: Any, record: Dict[str, Any]) -> None:
    """Write one metrics record into a SummaryWriter-like object."""
    phase = str(record.get("phase", "")).strip()
    if not phase:
        return
    step = int(record.get("step", 0))
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return
    if phase == "train" and "lr" in record:
        tb_write_scalar(writer, "train/lr", record.get("lr"), step)
    tb_write_scalars(writer, phase, metrics, step)


def merge_metric_runs(
    run_dirs: Sequence[Path],
    output_dir: Path,
    *,
    tensorboard_subdir: str = "tensorboard",
    flush_secs: int = 30,
    overwrite: bool = False,
    dedupe_by_phase_step: bool = True,
) -> Dict[str, Any]:
    """Merge multiple run directories into one TensorBoard logdir."""
    if len(run_dirs) < 1:
        raise ValueError("at least one --run-dir is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / str(tensorboard_subdir)
    merged_metrics_path = output_dir / "merged_metrics.jsonl"
    manifest_path = output_dir / "merge_manifest.json"

    if overwrite:
        if tb_dir.exists():
            shutil.rmtree(tb_dir)
        if merged_metrics_path.exists():
            merged_metrics_path.unlink()
        if manifest_path.exists():
            manifest_path.unlink()
    elif tb_dir.exists() and any(tb_dir.iterdir()):
        raise FileExistsError(f"TensorBoard output already exists: {tb_dir} (use --overwrite)")

    all_records: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()

    for source_index, run_dir in enumerate(run_dirs):
        rows = load_metric_records(Path(run_dir))
        for row in rows:
            row["_source_index"] = int(source_index)
        all_records.extend(rows)
        phase_counter = Counter(str(row.get("phase", "")) for row in rows)
        phase_counts.update(phase_counter)
        source_summaries.append(
            {
                "run_dir": str(Path(run_dir)),
                "records": int(len(rows)),
                "phase_counts": dict(sorted(phase_counter.items())),
            }
        )

    raw_record_count = len(all_records)
    if dedupe_by_phase_step:
        all_records = dedupe_records(all_records)
    sorted_records = sort_records(all_records)

    writer = create_tb_writer(
        output_dir=output_dir,
        subdir=str(tensorboard_subdir),
        enabled=True,
        flush_secs=int(flush_secs),
    )
    try:
        with merged_metrics_path.open("w", encoding="utf-8") as merged_f:
            for record in sorted_records:
                write_record(writer, record)
                clean_record = {
                    k: v
                    for k, v in record.items()
                    if not str(k).startswith("_")
                }
                merged_f.write(json.dumps(clean_record, sort_keys=True) + "\n")

        manifest = {
            "run_dirs": [str(Path(p)) for p in run_dirs],
            "tensorboard_subdir": str(tensorboard_subdir),
            "raw_record_count": int(raw_record_count),
            "merged_record_count": int(len(sorted_records)),
            "dedupe_by_phase_step": bool(dedupe_by_phase_step),
            "source_summaries": source_summaries,
            "phase_counts_before_dedupe": dict(sorted(phase_counts.items())),
            "step_min": int(min((int(r.get("step", 0)) for r in sorted_records), default=0)),
            "step_max": int(max((int(r.get("step", 0)) for r in sorted_records), default=0)),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tb_write_text(writer, "run/merge_manifest", json.dumps(manifest, indent=2), step=0)
    finally:
        tb_close(writer)

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="training run directory containing metrics.jsonl; pass multiple times in chronological order",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="output directory for merged TensorBoard files",
    )
    parser.add_argument(
        "--tensorboard-subdir",
        default="tensorboard",
        help="subdirectory name under output-dir for TensorBoard events",
    )
    parser.add_argument(
        "--flush-secs",
        type=int,
        default=30,
        help="SummaryWriter flush interval in seconds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing merged tensorboard output directory",
    )
    parser.add_argument(
        "--no-dedupe-by-phase-step",
        dest="dedupe_by_phase_step",
        action="store_false",
        help="keep duplicate (phase, step) rows instead of preferring later runs",
    )
    parser.set_defaults(dedupe_by_phase_step=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    manifest = merge_metric_runs(
        run_dirs=[Path(p) for p in args.run_dir],
        output_dir=Path(args.output_dir),
        tensorboard_subdir=str(args.tensorboard_subdir),
        flush_secs=max(1, int(args.flush_secs)),
        overwrite=bool(args.overwrite),
        dedupe_by_phase_step=bool(args.dedupe_by_phase_step),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
