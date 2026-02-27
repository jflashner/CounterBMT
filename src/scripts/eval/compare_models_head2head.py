"""Run head-to-head evaluation across v2 and legacy trajectory models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.eval.head2head import run_head2head


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple trajectory models head-to-head.")
    p.add_argument("--registry", type=str, required=True, help="YAML model registry config path")
    p.add_argument("--output-dir", type=str, default="", help="Optional output dir override")
    p.add_argument("--reuse-artifacts", dest="reuse_artifacts", action="store_true")
    p.add_argument("--no-reuse-artifacts", dest="reuse_artifacts", action="store_false")
    p.set_defaults(reuse_artifacts=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = run_head2head(
        registry_path=str(args.registry),
        output_dir=str(args.output_dir),
        reuse_artifacts=args.reuse_artifacts,
    )
    print(json.dumps(report, indent=2))
    print(f"Saved report: {Path(report['output_dir']) / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
