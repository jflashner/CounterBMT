from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect weighted path sampling over a curated train split.")
    parser.add_argument("--control-index", type=str, required=True)
    parser.add_argument("--num-draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(Path(args.control_index).expanduser())
    labels = [str(row.get("branch_label", "none")) for row in rows]
    counts = Counter(label for label in labels if label in {"left", "straight", "right"})
    if not counts:
        summary = {
            "num_draws": int(args.num_draws),
            "sampled_branch_histogram": {},
            "sampled_scenario_count": 0,
            "sampled_agent_count": 0,
            "sampled_example_ids_head": [],
            "reason": "no_supported_branch_labels",
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    weights = np.asarray(
        [1.0 / float(counts[label]) if label in counts else 0.0 for label in labels],
        dtype=np.float64,
    )
    weights = weights / weights.sum()

    rng = np.random.default_rng(int(args.seed))
    draw_indices = rng.choice(len(rows), size=int(args.num_draws), replace=True, p=weights)
    drawn_rows = [rows[int(index)] for index in draw_indices]

    summary = {
        "num_draws": int(args.num_draws),
        "sampled_branch_histogram": {
            key: int(value)
            for key, value in sorted(Counter(str(row.get("branch_label", "none")) for row in drawn_rows).items())
        },
        "sampled_scenario_count": int(len({str(row.get("scenario_id", "")) for row in drawn_rows})),
        "sampled_agent_count": int(len({(str(row.get("scenario_id", "")), str(row.get("agent_id", ""))) for row in drawn_rows})),
        "sampled_example_ids_head": [str(row.get("example_id", "")) for row in drawn_rows[:10]],
        "weight_histogram": {key: int(value) for key, value in sorted(counts.items())},
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
