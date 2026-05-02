from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


TABLE_RE = re.compile(r"\|\s+([A-Za-z0-9_./-]+)\s+\|\s+([-+0-9.eE]+)\s+\|")
STEP_RE = re.compile(r"Eval num_timesteps=([0-9]+), episode_reward=([-+0-9.eE]+)\s+\+/-\s+([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final TD3 eval metrics from one or more run directories.")
    parser.add_argument("run_dir", nargs="+", help="TD3 run directory containing train.log and/or evaluations.npz.")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--csv-out", type=str, default="")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _summarize_npz(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    summary: Dict[str, Any] = {"evaluations_npz": str(path)}
    if "timesteps" in data and len(data["timesteps"]) > 0:
        summary["timestep"] = int(np.asarray(data["timesteps"])[-1])
    if "results" in data and len(data["results"]) > 0:
        rewards = np.asarray(data["results"][-1], dtype=np.float64)
        summary["episode_reward_mean"] = float(np.mean(rewards))
        summary["episode_reward_std"] = float(np.std(rewards))
    if "ep_lengths" in data and len(data["ep_lengths"]) > 0:
        lengths = np.asarray(data["ep_lengths"][-1], dtype=np.float64)
        summary["episode_length_mean"] = float(np.mean(lengths))
        summary["episode_length_std"] = float(np.std(lengths))
    for key in ("cost", "route_completion", "crash", "arrive_dest", "out_of_road"):
        if key in data and len(data[key]) > 0:
            raw_values = np.asarray(data[key])
            # Stable-Baselines stores rewards/lengths as (num_evals, episodes),
            # but our custom info buffers are overwritten each eval and end up as
            # just the final eval's per-episode vector. Support both shapes.
            values = np.asarray(raw_values[-1] if raw_values.ndim >= 2 else raw_values, dtype=np.float64)
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def _summarize_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    eval_matches = STEP_RE.findall(text)
    summary: Dict[str, Any] = {"train_log": str(path)}
    if eval_matches:
        step, mean, std = eval_matches[-1]
        summary["timestep"] = int(step)
        summary["episode_reward_mean"] = float(mean)
        summary["episode_reward_std"] = float(std)

    table_values: Dict[str, float] = {}
    for key, value in TABLE_RE.findall(text):
        parsed = _safe_float(value)
        if parsed is not None:
            table_values[key] = parsed
    for source_key, dest_key in (
        ("eval/avg_rewards", "avg_rewards"),
        ("eval/avg_costs", "avg_costs"),
        ("eval/avg_completion", "avg_completion"),
        ("eval/avg_collisions", "avg_collisions"),
        ("eval/avg_length", "avg_length"),
        ("eval/cost", "cost"),
        ("eval/crash", "crash"),
        ("eval/route_completion", "route_completion"),
        ("eval/out_of_road", "out_of_road"),
        ("eval/arrive_dest", "arrive_dest"),
    ):
        if source_key in table_values:
            summary[dest_key] = table_values[source_key]
    if "time/total_timesteps" in table_values:
        summary["timestep"] = int(table_values["time/total_timesteps"])
    return summary


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary = {"run_dir": str(run_dir), "name": run_dir.name}
    # Prefer the npz for exact per-episode means/stds, then fill any missing
    # table fields from the human-readable log.
    log_summary = _summarize_log(run_dir / "train.log")
    npz_summary = _summarize_npz(run_dir / "evaluations.npz")
    summary.update(log_summary)
    summary.update(npz_summary)
    return summary


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if any(ch in text for ch in [",", "\n", '"']):
        text = '"' + text.replace('"', '""') + '"'
    return text


def write_csv(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    preferred = [
        "name",
        "timestep",
        "episode_reward_mean",
        "episode_reward_std",
        "cost_mean",
        "cost_std",
        "route_completion_mean",
        "route_completion_std",
        "crash_mean",
        "crash_std",
        "arrive_dest_mean",
        "out_of_road_mean",
        "episode_length_mean",
        "episode_length_std",
        "run_dir",
    ]
    all_keys = sorted({key for row in rows for key in row})
    columns = [key for key in preferred if key in all_keys] + [key for key in all_keys if key not in preferred]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(_csv_value(row.get(col)) for col in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = [summarize_run(Path(path)) for path in args.run_dir]
    print(json.dumps(rows, indent=2, sort_keys=True))
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    if args.csv_out:
        write_csv(rows, Path(args.csv_out).expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
