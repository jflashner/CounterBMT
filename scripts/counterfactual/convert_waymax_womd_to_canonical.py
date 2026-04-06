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

from bmt.counterfactual.waymax_adapter import iter_waymax_simulator_states, raw_scenario_from_waymax_state, resolve_waymax_config, waymax_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Waymax WOMD 1.3.1 scenes into canonical CounterBMT pickles.")
    parser.add_argument("--path", type=str, required=True, help="Waymax/WOMD path or shard pattern.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--current-time-index", type=int, default=-1)
    parser.add_argument("--num-paths", type=int, default=16)
    parser.add_argument("--num-points-per-path", type=int, default=128)
    parser.add_argument("--include-sdc-paths", dest="include_sdc_paths", action="store_true")
    parser.add_argument("--no-include-sdc-paths", dest="include_sdc_paths", action="store_false")
    parser.set_defaults(include_sdc_paths=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not waymax_available():
        raise SystemExit("waymax is not installed. Install waymax before running this adapter.")

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    config = resolve_waymax_config(
        config_name=str(args.config_name),
        path=str(args.path),
        include_sdc_paths=bool(args.include_sdc_paths),
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )

    manifest = []
    for index, state in enumerate(iter_waymax_simulator_states(config)):
        if int(args.max_examples) > 0 and index >= int(args.max_examples):
            break
        fallback_id = f"waymax_{index:06d}"
        raw = raw_scenario_from_waymax_state(
            state,
            scenario_id=fallback_id,
            current_time_index=(None if int(args.current_time_index) < 0 else int(args.current_time_index)),
        )
        scenario_id = str(raw.get("id") or fallback_id)
        file_name = f"sd_waymo_v1.3.1_{scenario_id}.pkl"
        output_path = outdir / file_name
        with output_path.open("wb") as f:
            import pickle

            pickle.dump(raw, f)
        manifest.append(
            {
                "scenario_id": scenario_id,
                "output_pkl": str(output_path),
                "num_tracks": int(len(raw.get("tracks", {}))),
                "num_map_features": int(len(raw.get("map_features", {}))),
                "num_traffic_lights": int(len(raw.get("dynamic_map_states", {}))),
                "num_sdc_paths": int(len(raw.get("sdc_paths", {}))),
            }
        )

    summary = {
        "config_name": str(args.config_name),
        "path": str(args.path),
        "num_examples_written": int(len(manifest)),
        "include_sdc_paths": bool(args.include_sdc_paths),
        "num_paths": int(args.num_paths),
        "num_points_per_path": int(args.num_points_per_path),
        "examples": manifest,
    }
    (outdir / "waymax_conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
