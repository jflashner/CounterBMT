"""Validate compact v2 DAG cache payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import sys

if __package__ is None or __package__ == "":
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache_schema import (
    schema_version_for_contract,
    validate_cache_payload,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate DAG cache payloads against contract + schema")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--dag-contract", type=str, default="maneuver_outcome_v1", choices=["compact10", "maneuver_outcome_v1"])
    p.add_argument("--dag-contract-mode", type=str, default="hard", choices=["hard"])
    p.add_argument("--stop-on-first-failure", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.cache_dir)
    files = sorted(root.glob("*.json"))
    if not files:
        print(json.dumps({"cache_dir": str(root), "count": 0, "error": "no_cache_files"}, indent=2))
        return 1

    cfg = DAGContractConfig(name=str(args.dag_contract), mode=str(args.dag_contract_mode))
    expected_schema = schema_version_for_contract(str(args.dag_contract))
    ok_count = 0
    fail_count = 0
    fail_reasons: Dict[str, int] = {}

    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            fail_count += 1
            fail_reasons["invalid_json"] = int(fail_reasons.get("invalid_json", 0) + 1)
            if args.stop_on_first_failure:
                break
            continue

        if str(payload.get("schema_version", "")) != expected_schema:
            fail_count += 1
            fail_reasons["wrong_schema_version"] = int(fail_reasons.get("wrong_schema_version", 0) + 1)
            if args.stop_on_first_failure:
                break
            continue
        if not validate_cache_payload(payload, allowed_schema_versions=(expected_schema,)):
            fail_count += 1
            fail_reasons["schema_validation_failed"] = int(fail_reasons.get("schema_validation_failed", 0) + 1)
            if args.stop_on_first_failure:
                break
            continue

        passed, _, report = enforce_dag_contract(payload, config=cfg)
        if not passed:
            fail_count += 1
            fail_reasons["contract_validation_failed"] = int(fail_reasons.get("contract_validation_failed", 0) + 1)
            if args.stop_on_first_failure:
                break
            continue

        ok_count += 1

    summary = {
        "cache_dir": str(root),
        "schema_version_expected": expected_schema,
        "checked": int(ok_count + fail_count),
        "passed": int(ok_count),
        "failed": int(fail_count),
        "fail_reasons": fail_reasons,
    }
    print(json.dumps(summary, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
