from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import load_motion_config
from scripts.counterfactual.mine_local_interventions import materialize_candidate_debug_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize heavy debug bundles from a lightweight path index.")
    parser.add_argument("--path-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--scenario-ids", type=str, default="")
    parser.add_argument("--example-ids", type=str, default="")
    parser.add_argument("--sample-total", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--include-pngs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_jsonl(Path(args.path_index).expanduser())
    selected = _select_rows(
        rows,
        scenario_ids=_split_csv(args.scenario_ids),
        example_ids=_split_csv(args.example_ids),
        sample_total=int(args.sample_total),
        seed=int(args.seed),
    )
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    config = load_motion_config(config_path=args.config or None)

    manifest_rows: List[Dict[str, Any]] = []
    for row in selected:
        result = materialize_candidate_debug_bundle(
            scenario_pkl=str(row["scenario_pkl"]),
            light_id=str(row["light_id"]),
            agent_id=str(row["agent_id"]),
            outdir=outdir,
            config=config,
            include_pngs=bool(args.include_pngs),
        )
        manifest_rows.append(
            {
                "example_id": row.get("example_id"),
                "scenario_id": row.get("scenario_id"),
                "scenario_pkl": row.get("scenario_pkl"),
                "light_id": row.get("light_id"),
                "agent_id": row.get("agent_id"),
                "decision_time_idx": row.get("decision_time_idx"),
                "materialized": result is not None,
                "artifact_dir": None if result is None else result.get("artifact_dir"),
                "train_view_path": None if result is None else result.get("train_view_path"),
                "factual_control_code_path": None if result is None else result.get("factual_control_code_path"),
            }
        )

    manifest_path = outdir / "debug_bundle_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows), encoding="utf-8")
    summary = {
        "path_index": str(Path(args.path_index).expanduser()),
        "outdir": str(outdir),
        "num_rows_available": len(rows),
        "num_rows_selected": len(selected),
        "num_rows_materialized": int(sum(bool(row.get("materialized")) for row in manifest_rows)),
        "debug_bundle_manifest_jsonl": str(manifest_path),
    }
    (outdir / "materialization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _select_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    scenario_ids: Sequence[str],
    example_ids: Sequence[str],
    sample_total: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected = list(rows)
    if scenario_ids:
        scenario_id_set = {str(value) for value in scenario_ids}
        selected = [row for row in selected if str(row.get("scenario_id")) in scenario_id_set]
    if example_ids:
        example_id_set = {str(value) for value in example_ids}
        selected = [row for row in selected if str(row.get("example_id")) in example_id_set]
    if not scenario_ids and not example_ids and sample_total <= 0:
        raise ValueError("Provide --scenario-ids, --example-ids, or --sample-total")
    if sample_total > 0 and len(selected) > sample_total:
        rng = random.Random(int(seed))
        selected = list(selected)
        rng.shuffle(selected)
        selected = selected[:sample_total]
    return sorted(
        selected,
        key=lambda row: (
            str(row.get("scenario_id")),
            str(row.get("agent_id")),
            int(row.get("decision_time_idx", 0)),
            str(row.get("light_id")),
        ),
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _split_csv(text: str) -> List[str]:
    return [chunk.strip() for chunk in str(text or "").split(",") if chunk.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
