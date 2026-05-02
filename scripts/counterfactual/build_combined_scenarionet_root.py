from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a lightweight ScenarioNet root whose summary/mapping span multiple existing roots. "
            "Scenario files are not copied; mapping entries point at the original absolute directories."
        )
    )
    parser.add_argument("--input-root", action="append", required=True, help="Existing ScenarioNet root. May be passed multiple times.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="")
    return parser.parse_args()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _resolve_file_dir(root: Path, mapping_value: Any) -> Path:
    mapping_text = str(mapping_value or "")
    mapping_path = Path(mapping_text)
    if mapping_path.is_absolute():
        return mapping_path
    return (root / mapping_path).resolve()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    combined_summary: Dict[str, Any] = {}
    combined_mapping: Dict[str, str] = {}
    per_root_counts: Dict[str, int] = {}
    duplicate_files = []

    for input_text in args.input_root:
        root = Path(input_text).expanduser().resolve()
        summary_path = root / "dataset_summary.pkl"
        mapping_path = root / "dataset_mapping.pkl"
        if not summary_path.is_file() or not mapping_path.is_file():
            raise FileNotFoundError(f"Missing dataset_summary.pkl or dataset_mapping.pkl under {root}")
        summary = dict(_load_pickle(summary_path))
        mapping = dict(_load_pickle(mapping_path))
        kept = 0
        for file_name, metadata in summary.items():
            file_name = str(file_name)
            if file_name in combined_summary:
                duplicate_files.append(file_name)
                continue
            file_dir = _resolve_file_dir(root, mapping.get(file_name, ""))
            scenario_path = file_dir / file_name
            if not scenario_path.is_file():
                continue
            combined_summary[file_name] = metadata
            combined_mapping[file_name] = str(file_dir)
            kept += 1
        per_root_counts[str(root)] = kept

    _write_pickle(output_root / "dataset_summary.pkl", combined_summary)
    _write_pickle(output_root / "dataset_mapping.pkl", combined_mapping)

    summary_payload = {
        "output_root": str(output_root),
        "input_roots": [str(Path(p).expanduser().resolve()) for p in args.input_root],
        "per_root_counts": per_root_counts,
        "combined_rows": int(len(combined_summary)),
        "duplicate_files_skipped": duplicate_files[:50],
        "duplicate_file_count": int(len(duplicate_files)),
    }
    summary_json = Path(args.summary_json).expanduser() if args.summary_json else output_root / "combined_summary.json"
    _write_json(summary_json, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
