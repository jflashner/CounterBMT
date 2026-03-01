"""Inspect and summarize generated DAG cache examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _summarize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    nodes = list(payload.get("nodes", []))
    edges = list(payload.get("edges", []))
    cpts = payload.get("cpts", {})
    node_type_counts: Dict[str, int] = {}
    for node in nodes:
        t = str(node.get("node_type", "unknown"))
        node_type_counts[t] = int(node_type_counts.get(t, 0) + 1)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    contract_report = metadata.get("contract_report", {})
    if not isinstance(contract_report, dict):
        contract_report = {}
    return {
        "scenario_id": str(payload.get("scenario_id", "unknown")),
        "n_nodes": int(len(nodes)),
        "n_edges": int(len(edges)),
        "node_type_counts": node_type_counts,
        "cpt_nodes": sorted(list(cpts.keys())) if isinstance(cpts, dict) else [],
        "contract_name": str(metadata.get("contract_name", "")),
        "contract_version": str(metadata.get("contract_version", "")),
        "contract_passed": bool(contract_report.get("passed", False)),
    }


def _pick_files(cache_dir: Path, n: int, seed: int) -> List[Path]:
    files = sorted(cache_dir.glob("*.json"))
    if not files:
        return []
    k = min(int(n), len(files))
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(files), size=k, replace=False).tolist()
    return [files[int(i)] for i in idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect DAG cache examples.")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--examples-dir", type=str, default="")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-md", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")
    examples_dir = Path(args.examples_dir) if args.examples_dir else None

    chosen = _pick_files(cache_dir, n=int(args.n), seed=int(args.seed))
    if not chosen:
        print("No cache files found.")
        return 1

    lines: List[str] = []
    lines.append("# DAG Inspection")
    lines.append("")
    for fp in chosen:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        s = _summarize_payload(payload)
        sid = s["scenario_id"]
        print(
            f"{sid}: nodes={s['n_nodes']} edges={s['n_edges']} "
            f"types={s['node_type_counts']} cpt_nodes={len(s['cpt_nodes'])} "
            f"contract={s['contract_name']}@{s['contract_version']} passed={s['contract_passed']}"
        )
        lines.append(f"## {sid}")
        lines.append(f"- cache: `{fp}`")
        lines.append(f"- nodes: `{s['n_nodes']}`")
        lines.append(f"- edges: `{s['n_edges']}`")
        lines.append(f"- node_type_counts: `{json.dumps(s['node_type_counts'], sort_keys=True)}`")
        lines.append(f"- cpt_nodes: `{', '.join(s['cpt_nodes']) if s['cpt_nodes'] else '(none)'}`")
        lines.append(
            f"- contract: `{s['contract_name']}@{s['contract_version']}` passed=`{s['contract_passed']}`"
        )
        if examples_dir is not None:
            ex = examples_dir / sid
            lines.append(f"- example_dir: `{ex}`")
            lines.append(f"- dag_summary: `{ex / 'dag_summary.txt'}`")
            lines.append(f"- frames_raw: `{ex / 'frames_raw'}`")
            lines.append(f"- frames_vlm: `{ex / 'frames_vlm'}`")
            lines.append(f"- frame_manifest: `{ex / 'frame_manifest.json'}`")
        lines.append("")

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved markdown: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
