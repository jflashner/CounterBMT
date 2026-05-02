from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_KEYS = [
    "episode_reward_mean",
    "cost_mean",
    "route_completion_mean",
    "crash_mean",
    "arrive_dest_mean",
    "out_of_road_mean",
    "episode_length_mean",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fixed val-100 TD3 evaluation JSON files.")
    parser.add_argument("eval_json", nargs="+")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    runs = []
    for path_str in args.eval_json:
        path = Path(path_str).expanduser().resolve()
        with path.open("r") as f:
            payload = json.load(f)
        payload["eval_json"] = str(path)
        runs.append(payload)

    aggregate: dict[str, Any] = {
        "num_runs": len(runs),
        "runs": [
            {
                "eval_json": run.get("eval_json"),
                "eval_data_dir": run.get("eval_data_dir"),
                "eval_ep": run.get("eval_ep"),
                **{key: run.get(key) for key in METRIC_KEYS if key in run},
            }
            for run in runs
        ],
        "mean_std_across_val100_samples": {},
    }

    for key in METRIC_KEYS:
        values = [float(run[key]) for run in runs if key in run and run[key] is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        aggregate["mean_std_across_val100_samples"][key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "values": values,
        }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as f:
        json.dump(aggregate, f, indent=2, default=_json_default)
    print(json.dumps(aggregate, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
