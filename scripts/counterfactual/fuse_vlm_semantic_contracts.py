from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.path_eval_bundle import load_jsonl, write_json, write_jsonl
from bmt.counterfactual.vlm_semantics.fuse import fuse_geometry_and_vlm_contracts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse raw VLM semantic contracts with existing geometry semantics.")
    parser.add_argument("--contracts", type=str, required=True)
    parser.add_argument("--path-index", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    raw_contract_rows = load_jsonl(args.contracts)
    path_index_rows: List[Dict[str, Any]] = load_jsonl(args.path_index) if args.path_index else []
    fused_rows, disagreement_report, training_eligibility_report = fuse_geometry_and_vlm_contracts(
        raw_contract_rows=raw_contract_rows,
        path_index_rows=path_index_rows,
        confidence_threshold=float(args.confidence_threshold),
    )
    write_jsonl(outdir / "fused_semantic_contract.jsonl", fused_rows)
    write_json(outdir / "semantic_disagreement_report.json", disagreement_report)
    write_json(outdir / "training_eligibility_report.json", training_eligibility_report)
    summary = {
        "num_raw_contract_rows": int(len(raw_contract_rows)),
        "num_path_index_rows": int(len(path_index_rows)),
        "num_fused_rows": int(len(fused_rows)),
        "fused_semantic_contract_jsonl": str((outdir / "fused_semantic_contract.jsonl").resolve()),
        "semantic_disagreement_report_json": str((outdir / "semantic_disagreement_report.json").resolve()),
        "training_eligibility_report_json": str((outdir / "training_eligibility_report.json").resolve()),
    }
    write_json(outdir / "fuse_summary.json", summary)
    print(summary["fused_semantic_contract_jsonl"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
