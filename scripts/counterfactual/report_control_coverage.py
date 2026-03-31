from __future__ import annotations

import argparse
import json
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report control coverage over mined train-view interventions.")
    parser.add_argument("--input", type=str, required=True, help="A control_index.jsonl file or root directory containing it.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    index_path = _resolve_index_path(Path(args.input).expanduser())
    records = _load_jsonl(index_path)
    train_views = [_load_json(Path(record["train_view_path"])) for record in records]

    summary = {
        "total_mined_interventions": len(train_views),
        "trainable_interventions": int(sum(bool(item.get("target_is_trainable")) for item in train_views)),
        "path_choice_supervisable_interventions": int(sum(bool(item.get("supervision", {}).get("path_choice_supervisable")) for item in train_views)),
        "compliance_supervisable_interventions": int(sum(bool(item.get("supervision", {}).get("compliance_supervisable")) for item in train_views)),
        "timing_supervisable_interventions": int(sum(bool(item.get("supervision", {}).get("timing_supervisable")) for item in train_views)),
        "fraction_target_is_sdc": _fraction(sum(str(item.get("agent_id")) == str(item.get("context", {}).get("sdc_id")) for item in train_views), len(train_views)),
        "fraction_target_is_tracks_to_predict": _fraction(sum(_is_track_to_predict(item) for item in train_views), len(train_views)),
        "drop_reasons": _histogram(item.get("supervision", {}).get("drop_reason") for item in train_views),
    }
    histograms = {
        "branch_label_histogram": _histogram(item.get("supervised_decision", {}).get("branch_label") for item in train_views),
        "compliance_histogram": _histogram(item.get("supervised_decision", {}).get("compliance_label") for item in train_views),
        "decision_time_rel_to_current_histogram": _histogram(item.get("provenance", {}).get("decision_time_index_rel_to_current") for item in train_views),
        "decision_state_histogram": _histogram(item.get("supervision", {}).get("decision_state") for item in train_views),
    }

    (outdir / "control_coverage_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "control_coverage_histograms.json").write_text(json.dumps(histograms, indent=2, sort_keys=True), encoding="utf-8")
    _copy_sample_bundles(records, outdir=outdir / "sampled_train_view_bundles", limit=int(args.sample_count))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_index_path(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    candidate = input_path / "control_index.jsonl"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find control_index.jsonl under {input_path}")


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _histogram(values) -> Dict[str, int]:
    histogram: Dict[str, int] = {}
    for value in values:
        key = "null" if value is None else str(value)
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda item: item[0]))


def _fraction(num: int, denom: int) -> float:
    return float(num / denom) if denom > 0 else 0.0


def _is_track_to_predict(payload: Dict[str, Any]) -> bool:
    forward_supervision = payload.get("debug", {}).get("forward_supervision", {})
    tracks_to_predict = set(str(value) for value in forward_supervision.get("tracks_to_predict_ids", []))
    return str(payload.get("agent_id")) in tracks_to_predict


def _copy_sample_bundles(records: List[Dict[str, Any]], *, outdir: Path, limit: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for sample_idx, record in enumerate(records[: max(0, int(limit))]):
        train_view_path = Path(record["train_view_path"]).expanduser()
        source_dir = train_view_path.parent
        target_dir = outdir / f"{sample_idx:03d}_{record['scenario_id']}_{record['agent_id']}_{record['decision_time_idx']}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "local_intervention_train_view.json",
            "local_intervention_raw.json",
            "branch_candidates.json",
            "branch_candidates.png",
            "factual_control_code.json",
            "alternative_control_codes.json",
            "mining_report.json",
        ):
            src = source_dir / filename
            if src.is_file():
                shutil.copy2(src, target_dir / filename)


if __name__ == "__main__":
    raise SystemExit(main())
