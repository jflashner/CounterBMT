"""Import legacy pipeline DAG JSON outputs into v2 DAG cache format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache_schema import SCHEMA_VERSION, validate_cache_payload


def _normalize_node(n: Dict[str, Any]) -> Dict[str, Any]:
    ts = n.get("timestamp_s")
    if ts is None:
        ts = n.get("timestamp")
    if ts is None:
        ts = n.get("time")
    md = n.get("metadata", {})
    if not isinstance(md, dict):
        md = {}
    return {
        "node_id": str(n.get("node_id") or n.get("id") or n.get("name") or "unknown_node"),
        "node_type": str(n.get("node_type") or n.get("type") or "unknown"),
        "value": n.get("value"),
        "timestamp_s": ts,
        "metadata": dict(md),
    }


def _normalize_edge(e: Dict[str, Any]) -> Dict[str, Any]:
    conf = e.get("confidence")
    if conf is None:
        conf = e.get("weight")
    if conf is None:
        conf = e.get("probability")
    mech = e.get("mechanism")
    if mech is None:
        mech = e.get("type")
    return {
        "parent_id": str(e.get("parent_id") or e.get("parent") or e.get("from") or e.get("src") or ""),
        "child_id": str(e.get("child_id") or e.get("child") or e.get("to") or e.get("dst") or ""),
        "confidence": float(conf if conf is not None else 0.7),
        "mechanism": str(mech or ""),
    }


def _infer_payload(raw: Dict[str, Any], scenario_id_hint: str) -> Dict[str, Any]:
    scenario_id = str(
        raw.get("scenario_id")
        or raw.get("id")
        or raw.get("scenario", {}).get("scenario_id")
        or scenario_id_hint
    )
    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])
    cpts = raw.get("cpts", {})

    # Handle nested payloads like {"dag": {...}}.
    if isinstance(raw.get("dag"), dict):
        dag = raw["dag"]
        nodes = dag.get("nodes", nodes)
        edges = dag.get("edges", edges)
        cpts = dag.get("cpts", cpts)
        scenario_id = str(dag.get("scenario_id", scenario_id))

    nodes_n = [_normalize_node(n) for n in (nodes if isinstance(nodes, list) else [])]
    edges_n = [_normalize_edge(e) for e in (edges if isinstance(edges, list) else [])]

    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "nodes": nodes_n,
        "edges": edges_n,
        "cpts": cpts if isinstance(cpts, dict) else {},
        "metadata": {
            "source": "legacy_import",
            "contract_name": "compact10",
            "contract_version": "1",
            "contract_report": {"passed": False, "reason": "not_checked"},
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Import legacy DAG JSONs into v2 cache format")
    p.add_argument("--legacy-root", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--dag-contract", type=str, default="compact10", choices=["compact10"])
    p.add_argument("--dag-contract-mode", type=str, default="hard", choices=["hard"])
    args = p.parse_args()

    legacy_root = Path(args.legacy_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(legacy_root.rglob("dag.json"))
    if not files:
        print(f"No dag.json files found under: {legacy_root}")
        return 1

    converted = 0
    skipped = 0
    fail_reasons: Dict[str, int] = {}
    contract_cfg = DAGContractConfig(name=str(args.dag_contract), mode=str(args.dag_contract_mode))
    for fp in files:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            fail_reasons["invalid_json"] = int(fail_reasons.get("invalid_json", 0) + 1)
            continue
        sid_hint = fp.parent.name
        payload = _infer_payload(raw, sid_hint)
        sid = str(payload.get("scenario_id", "")).strip()
        if not sid:
            skipped += 1
            fail_reasons["missing_scenario_id"] = int(fail_reasons.get("missing_scenario_id", 0) + 1)
            continue
        ok, canonical, _report = enforce_dag_contract(payload, config=contract_cfg)
        if not ok:
            skipped += 1
            fail_reasons["contract_failed"] = int(fail_reasons.get("contract_failed", 0) + 1)
            continue
        canonical["schema_version"] = SCHEMA_VERSION
        canonical["scenario_id"] = sid
        if not validate_cache_payload(canonical):
            skipped += 1
            fail_reasons["schema_failed"] = int(fail_reasons.get("schema_failed", 0) + 1)
            continue
        out_path = out_dir / f"{sid}.json"
        out_path.write_text(json.dumps(canonical, indent=2), encoding="utf-8")
        converted += 1

    print(
        json.dumps(
            {
                "converted": converted,
                "skipped": skipped,
                "out_dir": str(out_dir),
                "contract": {"name": str(args.dag_contract), "mode": str(args.dag_contract_mode)},
                "fail_reasons": fail_reasons,
            },
            indent=2,
        )
    )
    return 0 if converted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
