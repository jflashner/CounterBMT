from __future__ import annotations

import argparse
import json
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

from bmt.counterfactual import (
    compile_alternative_control_codes_from_local_intervention,
    compile_control_code_from_local_intervention,
    load_and_normalize_scenario,
    validate_control_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile local intervention train-view artifacts into control_code_v1 JSON files.")
    parser.add_argument("--input", type=str, required=True, help="A local_intervention_train_view.json/local_intervention.json file or a directory containing them.")
    parser.add_argument("--outdir", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    intervention_files = _discover_local_intervention_files(input_path)
    if not intervention_files:
        print(json.dumps({"error": "no_local_intervention_files_found", "input": str(input_path)}, indent=2))
        return 1

    compiled: List[Dict[str, Any]] = []
    scenario_to_paths: Dict[str, List[str]] = {}
    for intervention_path in intervention_files:
        payload = json.loads(intervention_path.read_text(encoding="utf-8"))
        scenario_pkl = payload.get("debug", {}).get("candidate", {}).get("scenario_pkl", "")
        if not scenario_pkl:
            raise ValueError(f"Missing debug.candidate.scenario_pkl in {intervention_path}")
        canonical = load_and_normalize_scenario(scenario_pkl)
        factual_control_code = compile_control_code_from_local_intervention(
            payload,
            canonical=canonical,
            source_path=str(intervention_path),
        )
        alternative_control_codes = compile_alternative_control_codes_from_local_intervention(
            payload,
            canonical=canonical,
            source_path=str(intervention_path),
        )

        relative_parent = Path(".") if input_path.is_file() else intervention_path.parent.relative_to(input_path)
        example_dir = outdir / relative_parent
        example_dir.mkdir(parents=True, exist_ok=True)
        factual_path = example_dir / "factual_control_code.json"
        compatibility_path = example_dir / "control_code.json"
        alternatives_path = example_dir / "alternative_control_codes.json"

        factual_payload = factual_control_code.to_dict()
        factual_path.write_text(json.dumps(factual_payload, indent=2, sort_keys=True), encoding="utf-8")
        compatibility_path.write_text(json.dumps(factual_payload, indent=2, sort_keys=True), encoding="utf-8")
        alternatives_path.write_text(json.dumps(alternative_control_codes, indent=2, sort_keys=True), encoding="utf-8")

        errors = validate_control_code(factual_payload)
        record = {
            "scenario_id": factual_control_code.scenario_id,
            "agent_id": factual_control_code.agent_id,
            "decision_time_idx": factual_control_code.decision_time_idx,
            "source_local_intervention": str(intervention_path),
            "factual_control_code_path": str(factual_path),
            "compatibility_control_code_path": str(compatibility_path),
            "alternative_control_codes_path": str(alternatives_path),
            "num_alternatives": len(alternative_control_codes),
            "validation_errors": errors,
        }
        compiled.append(record)
        scenario_to_paths.setdefault(factual_control_code.scenario_id, []).append(str(factual_path))

    summary = {
        "input": str(input_path),
        "outdir": str(outdir),
        "num_local_interventions": len(intervention_files),
        "num_factual_control_codes": len(compiled),
        "num_unique_scenarios": len(scenario_to_paths),
        "scenarios_with_multiple_control_codes": {
            scenario_id: sorted(paths)
            for scenario_id, paths in sorted(scenario_to_paths.items())
            if len(paths) > 1
        },
        "num_validation_failures": sum(1 for item in compiled if item["validation_errors"]),
        "num_alternative_control_codes": int(sum(item["num_alternatives"] for item in compiled)),
    }

    (outdir / "control_code_index.json").write_text(
        json.dumps({scenario_id: sorted(paths) for scenario_id, paths in sorted(scenario_to_paths.items())}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (outdir / "control_code_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (outdir / "control_code_manifest.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in compiled),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _discover_local_intervention_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    preferred = sorted(input_path.rglob("local_intervention_train_view.json"))
    if preferred:
        return preferred
    return sorted(input_path.rglob("local_intervention.json"))


if __name__ == "__main__":
    raise SystemExit(main())
