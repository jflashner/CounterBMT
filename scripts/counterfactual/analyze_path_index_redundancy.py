from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.path_corpus import (
    DedupConfig,
    analyze_path_index_redundancy,
    load_jsonl_rows,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze redundancy in a path-control index.")
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl_rows(args.path_index)
    analysis = analyze_path_index_redundancy(rows, dedup_config=DedupConfig())

    write_json(outdir / "redundancy_summary.json", analysis["redundancy_summary"])
    write_jsonl(outdir / "duplicated_groups.jsonl", analysis["duplicated_groups"])
    write_json(outdir / "window_overlap_histograms.json", analysis["window_overlap_histograms"])
    write_json(outdir / "light_duplication_summary.json", analysis["light_duplication_summary"])
    write_json(outdir / "compliance_histogram.json", analysis["compliance_histogram"])
    write_json(outdir / "scenario_agent_branch_counts.json", analysis["scenario_agent_branch_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
