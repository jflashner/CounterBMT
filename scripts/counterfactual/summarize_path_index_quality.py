from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.path_corpus import histogram, load_jsonl_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize curated path-index quality metrics.")
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl_rows(args.path_index)

    decision_time_rel = [_maybe_float(row.get("decision_time_idx")) - _maybe_float(row.get("current_time_index_global")) for row in rows]
    window_lengths = [
        int(row.get("window_end_idx", 0)) - int(row.get("window_start_idx", 0)) + 1
        for row in rows
    ]
    branch_margin = [_maybe_float(row.get("branch_margin")) for row in rows]
    downstream_progress = [_maybe_float(row.get("downstream_progress_along_branch_m")) for row in rows]
    stopline_progress = [_maybe_float(row.get("signed_stopline_progress_m")) for row in rows]
    heading_error = [_maybe_float(row.get("final_heading_error_rad")) for row in rows]

    summary = {
        "num_rows": int(len(rows)),
        "num_scenarios": int(len({str(row.get("scenario_id")) for row in rows})),
        "branch_label_histogram": histogram(row.get("branch_label") for row in rows),
        "agent_role_histogram": histogram(row.get("agent_role") for row in rows),
        "signal_state_histogram": histogram(row.get("signal_state_at_decision") for row in rows),
        "compliance_histogram": histogram(row.get("compliance_label", row.get("compliance_token", {}).get("compliance_label")) for row in rows),
        "decision_time_rel_to_current_histogram": _numeric_histogram(decision_time_rel, bin_size=5.0),
        "window_length_histogram": _integer_histogram(window_lengths),
        "branch_margin_percentiles": _percentiles(branch_margin),
        "downstream_progress_percentiles": _percentiles(downstream_progress),
        "signed_stopline_progress_percentiles": _percentiles(stopline_progress),
        "final_heading_error_percentiles": _percentiles(heading_error),
        "cluster_size_histogram": histogram(row.get("cluster_size") for row in rows),
        "light_group_size_histogram": histogram(row.get("light_group_size") for row in rows),
    }
    histograms = {
        "decision_time_rel_to_current_histogram": summary["decision_time_rel_to_current_histogram"],
        "window_length_histogram": summary["window_length_histogram"],
        "branch_label_histogram": summary["branch_label_histogram"],
        "agent_role_histogram": summary["agent_role_histogram"],
        "signal_state_histogram": summary["signal_state_histogram"],
        "compliance_histogram": summary["compliance_histogram"],
        "cluster_size_histogram": summary["cluster_size_histogram"],
        "light_group_size_histogram": summary["light_group_size_histogram"],
    }
    write_json(outdir / "curated_quality_summary.json", summary)
    write_json(outdir / "curated_quality_histograms.json", histograms)
    return 0


def _percentiles(values: Sequence[float]) -> Dict[str, float | None]:
    finite = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float32)
    if finite.size == 0:
        return {key: None for key in ("p0", "p5", "p25", "p50", "p75", "p95", "p100")}
    return {
        "p0": float(np.percentile(finite, 0)),
        "p5": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "p100": float(np.percentile(finite, 100)),
    }


def _maybe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _numeric_histogram(values: Iterable[float], *, bin_size: float) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if not math.isfinite(float(value)):
            key = "null"
        else:
            lower = math.floor(float(value) / float(bin_size)) * float(bin_size)
            upper = lower + float(bin_size)
            key = f"{int(lower)}:{int(upper)}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _integer_histogram(values: Iterable[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


if __name__ == "__main__":
    raise SystemExit(main())
