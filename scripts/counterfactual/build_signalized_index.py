from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import build_signalized_index, write_signal_qc_artifacts_for_candidate, write_signalized_index_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a signalized intersection index over ScenarioNet pickles.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--artifact-examples", type=int, default=10)
    parser.add_argument("--distance-threshold-m", type=float, default=35.0)
    parser.add_argument("--local-patch-radius-m", type=float, default=30.0)
    parser.add_argument("--min-valid-sdc-steps", type=int, default=10)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.45)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_result = build_signalized_index(
        args.data_dir,
        max_scenarios=args.max_scenarios,
        distance_threshold_m=args.distance_threshold_m,
        local_patch_radius_m=args.local_patch_radius_m,
        min_valid_sdc_steps=args.min_valid_sdc_steps,
        ambiguity_threshold=args.ambiguity_threshold,
    )
    paths = write_signalized_index_outputs(build_result, data_dir=args.data_dir, outdir=args.outdir)

    sampled = []
    examples_root = Path(args.outdir).expanduser() / "examples"
    for candidate in build_result.candidates[: max(0, int(args.artifact_examples))]:
        example_dir = examples_root / candidate.scenario_id / candidate.light_id
        artifact_paths = write_signal_qc_artifacts_for_candidate(candidate, outdir=example_dir)
        sampled.append(
            {
                "scenario_id": candidate.scenario_id,
                "light_id": candidate.light_id,
                "artifact_dir": str(example_dir),
                "artifacts": {key: str(value) for key, value in artifact_paths.items()},
            }
        )

    print(
        json.dumps(
            {
                "outdir": str(Path(args.outdir).expanduser()),
                "summary": {key: str(value) for key, value in paths.items()},
                "scanned_scenarios": len(build_result.scenario_results),
                "candidate_windows": len(build_result.candidates),
                "sampled_examples": sampled,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
