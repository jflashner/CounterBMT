from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a semantic-only training index that mixes alternative SDC intervention rows "
            "with duplicated factual no_intervention CE anchors."
        )
    )
    parser.add_argument("--input-index", type=str, required=True)
    parser.add_argument("--output-index", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument(
        "--scenario-root",
        type=str,
        default="",
        help=(
            "Optional ScenarioNet root used to rewrite scenario_pkl paths and add scenario_file_name. "
            "Useful when the source semantic index was built on a different machine."
        ),
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=1.0,
        help="Number of no_intervention anchor rows per alternative row. Default 1.0 gives a 50/50 mix.",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        choices=("mixed", "anchors-only", "alternatives-only"),
        default="mixed",
        help="Write the 50/50 mixed index, only no_intervention GT anchors, or only alternatives.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(dict(json.loads(text)))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True))
            f.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _is_factual_gt(row: Mapping[str, Any]) -> bool:
    source_kind = str(row.get("source_kind") or "").strip()
    slot_id = str(row.get("selected_slot_id") or row.get("slot_id") or "").strip()
    return source_kind == "factual_gt" or slot_id == "gt"


def _make_no_intervention_anchor(row: Mapping[str, Any], *, repeat_index: int) -> Dict[str, Any]:
    anchor = copy.deepcopy(dict(row))
    anchor["source_kind"] = "factual_gt"
    anchor["requested_semantic_label"] = "no_intervention"
    anchor["semantic_label"] = "no_intervention"
    anchor["requested_semantic_confidence"] = 1.0
    anchor["use_for_training"] = True
    anchor["no_intervention_anchor"] = True
    anchor["no_intervention_anchor_repeat_index"] = int(repeat_index)
    anchor["counterfactual_objective"] = "ce_only"
    anchor.setdefault("selected_slot_id", "gt")
    return anchor


def _rewrite_scenario_path(row: Mapping[str, Any], *, scenario_root: Path | None) -> Dict[str, Any]:
    output = dict(row)
    if scenario_root is None:
        return output
    scenario_file_name = str(output.get("scenario_file_name") or "").strip()
    if not scenario_file_name:
        scenario_pkl = str(output.get("scenario_pkl") or "").strip()
        if scenario_pkl:
            scenario_file_name = Path(scenario_pkl).name
    if not scenario_file_name:
        scenario_id = str(output.get("scenario_id") or "").strip()
        if scenario_id:
            scenario_file_name = f"sd_waymo_v1.3.1_{scenario_id}.pkl"
    if scenario_file_name:
        output["scenario_file_name"] = scenario_file_name
        output["scenario_pkl"] = str((scenario_root / scenario_file_name).resolve())
    return output


def _cycle_rows(rows: Sequence[Mapping[str, Any]], count: int) -> Iterable[Dict[str, Any]]:
    if not rows:
        return
    for idx in range(int(count)):
        yield _make_no_intervention_anchor(rows[idx % len(rows)], repeat_index=idx // len(rows))


def _interleave(primary: Sequence[Mapping[str, Any]], secondary: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    max_len = max(len(primary), len(secondary))
    for idx in range(max_len):
        if idx < len(primary):
            output.append(dict(primary[idx]))
        if idx < len(secondary):
            output.append(dict(secondary[idx]))
    return output


def main() -> None:
    args = parse_args()
    input_index = Path(args.input_index).expanduser()
    output_index = Path(args.output_index).expanduser()
    scenario_root = Path(args.scenario_root).expanduser() if args.scenario_root else None
    rows = [_rewrite_scenario_path(row, scenario_root=scenario_root) for row in _read_jsonl(input_index)]
    factual_rows = [row for row in rows if _is_factual_gt(row)]
    alternative_rows = [row for row in rows if not _is_factual_gt(row)]
    if not alternative_rows:
        raise RuntimeError(f"No alternative rows found in {input_index}")
    if not factual_rows:
        raise RuntimeError(f"No factual GT rows found in {input_index}")

    if str(args.output_mode) == "alternatives-only":
        anchor_rows = []
        output_rows = [dict(row) for row in alternative_rows]
    elif str(args.output_mode) == "anchors-only":
        anchor_rows = [
            _make_no_intervention_anchor(row, repeat_index=0)
            for row in factual_rows
        ]
        output_rows = [dict(row) for row in anchor_rows]
    else:
        anchor_count = int(round(float(args.anchor_ratio) * float(len(alternative_rows))))
        anchor_rows = list(_cycle_rows(factual_rows, anchor_count))
        output_rows = _interleave(alternative_rows, anchor_rows)
    _write_jsonl(output_index, output_rows)

    summary = {
        "input_index": str(input_index),
        "output_index": str(output_index),
        "input_rows": int(len(rows)),
        "alternative_rows": int(len(alternative_rows)),
        "factual_gt_rows": int(len(factual_rows)),
        "no_intervention_anchor_rows": int(len(anchor_rows)),
        "output_rows": int(len(output_rows)),
        "anchor_ratio": float(args.anchor_ratio),
        "output_mode": str(args.output_mode),
    }
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else output_index.with_suffix(".summary.json")
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
