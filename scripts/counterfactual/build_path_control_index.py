from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import discover_scenario_pickles, load_motion_config, select_signalized_candidates_for_scenario
from scripts.counterfactual.mine_local_interventions import _mine_candidate_for_trainable_agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a path-only control index from mined local interventions.")
    parser.add_argument("--scenario-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-scenarios", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--write-examples-manifest", action="store_true")
    parser.add_argument("--write-histograms", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--max-agents-per-candidate", type=int, default=0)
    parser.add_argument("--max-candidates-per-scenario", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts_root = outdir / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    scenario_paths = discover_scenario_pickles(args.scenario_root)
    selected_paths = _select_scenarios(scenario_paths, max_scenarios=int(args.max_scenarios), seed=int(args.seed))
    worker_args = [
        {
            "scenario_pkl": str(path),
            "artifacts_root": str(artifacts_root),
            "config_path": str(args.config or ""),
            "max_agents_per_candidate": int(args.max_agents_per_candidate),
            "max_candidates_per_scenario": int(args.max_candidates_per_scenario),
        }
        for path in selected_paths
    ]

    if int(args.num_workers) > 1 and len(worker_args) > 1:
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as executor:
            scenario_results = list(executor.map(_process_one_scenario, worker_args))
    else:
        scenario_results = [_process_one_scenario(item) for item in worker_args]

    entries: List[Dict[str, Any]] = []
    filter_counts: Counter[str] = Counter()
    signalized_drop_counts: Counter[str] = Counter()
    for result in scenario_results:
        entries.extend(result["entries"])
        filter_counts.update(result["path_filter_drop_counts"])
        if result["signalized_primary_drop_reason"] is not None:
            signalized_drop_counts[str(result["signalized_primary_drop_reason"])] += 1

    entries = sorted(
        entries,
        key=lambda item: (
            str(item["scenario_id"]),
            str(item["branch_label"]),
            str(item["agent_id"]),
            int(item["decision_time_idx"]),
        ),
    )

    path_index_path = outdir / "path_index.jsonl"
    path_index_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )

    label_hist = _histogram(entry["branch_label"] for entry in entries)
    support = {label: int(label_hist.get(label, 0)) for label in ("left", "straight", "right")}
    at_least_100 = {label: bool(count >= 100) for label, count in support.items()}
    below_300 = {label: bool(count < 300) for label, count in support.items()}
    warnings = [f"class_{label}_below_300" for label, flag in below_300.items() if flag]

    summary = {
        "scenario_root": str(Path(args.scenario_root).expanduser()),
        "outdir": str(outdir),
        "max_scenarios": int(args.max_scenarios),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "max_agents_per_candidate": int(args.max_agents_per_candidate),
        "max_candidates_per_scenario": int(args.max_candidates_per_scenario),
        "num_scenarios_discovered": len(scenario_paths),
        "num_scenarios_scanned": len(selected_paths),
        "num_scenarios_with_path_examples": int(sum(bool(item["entries"]) for item in scenario_results)),
        "num_path_examples": len(entries),
        "signalized_drop_reasons": dict(sorted(signalized_drop_counts.items())),
        "path_filter_drop_reasons": dict(sorted(filter_counts.items())),
        "class_support": support,
        "class_support_at_least_100": at_least_100,
        "class_support_warn_below_300": below_300,
        "warnings": warnings,
        "path_index_jsonl": str(path_index_path),
    }
    histograms = {
        "branch_label_histogram": label_hist,
        "agent_role_histogram": _histogram(entry["agent_role"] for entry in entries),
        "decision_state_histogram": _histogram(entry["decision_state"] for entry in entries),
        "signal_state_histogram": _histogram(entry["signal_state"] for entry in entries),
        "target_is_sdc_histogram": _histogram(entry["is_sdc_target"] for entry in entries),
    }
    manifest = {
        "examples": entries,
    }

    if args.write_summary:
        (outdir / "path_support_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.write_histograms:
        (outdir / "path_label_histograms.json").write_text(json.dumps(histograms, indent=2, sort_keys=True), encoding="utf-8")
    if args.write_examples_manifest:
        (outdir / "path_examples_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _select_scenarios(paths: List[Path], *, max_scenarios: int, seed: int) -> List[Path]:
    if max_scenarios <= 0 or len(paths) <= max_scenarios:
        return list(paths)
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(paths), size=int(max_scenarios), replace=False))
    return [paths[int(idx)] for idx in indices.tolist()]


def _process_one_scenario(args: Dict[str, Any]) -> Dict[str, Any]:
    scenario_pkl = str(args["scenario_pkl"])
    artifacts_root = Path(args["artifacts_root"]).expanduser()
    config = load_motion_config(config_path=args.get("config_path") or None)

    signalized_result = select_signalized_candidates_for_scenario(scenario_pkl)
    path_filter_drop_counts: Counter[str] = Counter()
    entries: List[Dict[str, Any]] = []
    candidates = list(signalized_result.candidates)
    max_candidates = int(args.get("max_candidates_per_scenario", 0))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    for candidate in candidates:
        records = _mine_candidate_for_trainable_agents(
            candidate,
            outdir=artifacts_root,
            config=config,
            artifact_mode="index_minimal",
            max_agents=int(args.get("max_agents_per_candidate", 0)) or None,
        )
        for record in records:
            train_view = _load_json(record["train_view_path"])
            keep, drop_reason = _is_path_train_view_eligible(train_view)
            if not keep:
                path_filter_drop_counts[str(drop_reason)] += 1
                continue
            factual_path = str(record["factual_control_code_path"])
            entries.append(
                {
                    "scenario_id": str(train_view["scenario_id"]),
                    "scenario_pkl": scenario_pkl,
                    "scenario_file_name": Path(scenario_pkl).name,
                    "agent_id": str(train_view["agent_id"]),
                    "agent_role": str(train_view["provenance"]["agent_role"]),
                    "decision_time_idx": int(train_view["decision_time_idx"]),
                    "branch_label": str(train_view["supervised_decision"]["branch_label"]),
                    "train_view_path": str(record["train_view_path"]),
                    "factual_control_code_path": factual_path,
                    "alternative_control_codes_path": str(record["alternative_control_codes_path"]),
                    "signal_state": train_view["context"].get("signal_state_at_decision"),
                    "decision_state": train_view["supervision"].get("decision_state"),
                    "is_sdc_target": bool(str(train_view["agent_id"]) == str(train_view["context"].get("sdc_id"))),
                    "conditioning_eligible": bool(train_view.get("conditioning_eligible")),
                    "target_is_trainable": bool(train_view.get("target_is_trainable")),
                    "path_choice_supervisable": bool(train_view.get("supervision", {}).get("path_choice_supervisable")),
                    "control_available_at_current": bool(train_view.get("control_available_at_current")),
                }
            )
    return {
        "scenario_pkl": scenario_pkl,
        "signalized_primary_drop_reason": signalized_result.primary_drop_reason,
        "path_filter_drop_counts": dict(path_filter_drop_counts),
        "entries": entries,
    }


def _is_path_train_view_eligible(train_view: Dict[str, Any]) -> Tuple[bool, str]:
    supervised_decision = dict(train_view.get("supervised_decision", {}))
    branch_label = supervised_decision.get("branch_label")
    terminal_pose = supervised_decision.get("terminal_pose")
    if not bool(train_view.get("conditioning_eligible")):
        return False, "conditioning_ineligible"
    if not bool(train_view.get("target_is_trainable")):
        return False, "non_trainable_target"
    if not bool(train_view.get("control_available_at_current")):
        return False, "control_unavailable"
    if not bool(train_view.get("supervision", {}).get("path_choice_supervisable")):
        return False, "path_not_supervisable"
    if branch_label is None:
        return False, "branch_label_null"
    if str(branch_label) == "u_turn":
        return False, "u_turn_excluded"
    if str(branch_label) not in {"left", "straight", "right"}:
        return False, "unsupported_branch_label"
    if terminal_pose is None:
        return False, "missing_terminal_pose"
    return True, "kept"


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _histogram(values: Iterable[Any]) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: item[0]))


if __name__ == "__main__":
    raise SystemExit(main())
