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

from bmt.counterfactual.path_corpus import load_jsonl_rows, split_rows_by_scenario, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a curated path index by scenario.")
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl_rows(args.path_index)
    train_rows, val_rows, summary = split_rows_by_scenario(
        rows,
        seed=int(args.seed),
        val_fraction=float(args.val_fraction),
    )

    write_jsonl(outdir / "path_index_curated_train.jsonl", train_rows)
    write_jsonl(outdir / "path_index_curated_val.jsonl", val_rows)
    write_json(outdir / "split_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
