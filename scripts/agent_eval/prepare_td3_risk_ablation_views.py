from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


SCENE_RE = re.compile(r"waymax_scene_\d+")
RISK_RANK = {"unknown": -1, "": -1, "low": 0, "medium": 1, "high": 2}
RISK_FROM_RANK = {-1: "unknown", 0: "low", 1: "medium", 2: "high"}
DEFAULT_RISK_GROUPS = ("low=low", "medium=medium", "high=high", "medium_high=medium,high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare TD3-ready CounterBMT training views where only the synthetic "
            "training scenarios are filtered by VLM-assessed maneuver risk."
        )
    )
    parser.add_argument("--base-views-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--vlm-bundle-root",
        type=str,
        default="",
        help="Bundle root containing examples/*/contract_normalized_*.json.",
    )
    parser.add_argument(
        "--vlm-risk-index",
        type=str,
        default="",
        help="Optional compact risk index JSON previously produced by this script.",
    )
    parser.add_argument(
        "--risk-index-out",
        type=str,
        default="",
        help="Optional path to write the compact risk index JSON.",
    )
    parser.add_argument(
        "--risk-mode",
        type=str,
        default="scene_max",
        choices=("scene_max", "gt", "label_match"),
        help=(
            "How to assign one risk level to a synthetic scene. scene_max uses the "
            "maximum risk over all VLM-labeled paths in the source scene; gt uses "
            "the factual SDC route risk; label_match uses paths whose VLM semantic "
            "label matches the synthetic intervention label."
        ),
    )
    parser.add_argument(
        "--label-match-fallback",
        type=str,
        default="scene_max",
        choices=("scene_max", "gt", "skip"),
        help="Fallback when --risk-mode label_match finds no matching semantic label.",
    )
    parser.add_argument(
        "--risk-group",
        action="append",
        default=[],
        help=(
            "Risk group spec as name=level[,level...]. May be repeated. "
            "Defaults to low, medium, high, and medium_high."
        ),
    )
    parser.add_argument(
        "--natural-mode",
        type=str,
        default="all",
        choices=("all", "matching"),
        help=(
            "all keeps the full original/Waymo training set fixed and filters only "
            "synthetic scenarios; matching keeps only originals paired with the "
            "selected synthetic scenes, preserving a 1:1 ratio."
        ),
    )
    parser.add_argument("--target-pairs", type=int, default=-1, help="Optional max synthetic scenes per risk group.")
    parser.add_argument("--balance-to-min", action="store_true", help="Downsample all groups to the smallest group size.")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--shuffle-scenes", action="store_true")
    parser.add_argument("--link-mode", type=str, default="symlink", choices=("symlink", "hardlink", "copy"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _normalize_risk(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return text if text in RISK_RANK else "unknown"


def _max_risk(risks: Sequence[str]) -> str:
    if not risks:
        return "unknown"
    return RISK_FROM_RANK[max(RISK_RANK.get(_normalize_risk(risk), -1) for risk in risks)]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _extract_scene_id(filename: str, metadata: Mapping[str, Any]) -> str:
    for key in ("source_scenario", "scenario_id", "id"):
        value = str(metadata.get(key) or "").strip()
        match = SCENE_RE.search(value)
        if match:
            return match.group(0)
    match = SCENE_RE.search(str(filename))
    if match:
        return match.group(0)
    raise ValueError(f"Could not extract waymax scene id from {filename!r}")


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_summary(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    summary_path = data_dir / "dataset_summary.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"dataset_summary.pkl not found under {data_dir}")
    summary = _read_pickle(summary_path)
    if not isinstance(summary, dict):
        raise TypeError(f"dataset_summary.pkl under {data_dir} must contain dict, got {type(summary)}")
    return summary


def _build_scene_to_file_map(
    data_dir: Path,
    *,
    counterfactual: bool | None = None,
) -> Dict[str, Path]:
    summary = _load_summary(data_dir)
    scene_to_path: Dict[str, Path] = {}
    for filename, metadata in summary.items():
        filename = str(filename)
        if not filename.startswith("sd_") or not filename.endswith(".pkl"):
            continue
        meta_dict = metadata if isinstance(metadata, dict) else {}
        if counterfactual is not None and bool(meta_dict.get("counterfactual", False)) is not bool(counterfactual):
            continue
        scene_id = _extract_scene_id(filename, meta_dict)
        source_path = data_dir / filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing scenario referenced by dataset_summary: {source_path}")
        if scene_id in scene_to_path:
            raise ValueError(f"Duplicate scene id {scene_id} in {data_dir} for counterfactual={counterfactual}")
        scene_to_path[scene_id] = source_path
    return scene_to_path


def _load_metadata_for_paths(paths: Mapping[str, Path]) -> Dict[str, Dict[str, Any]]:
    metadata_by_scene: Dict[str, Dict[str, Any]] = {}
    for scene_id, path in paths.items():
        scenario = _read_pickle(path)
        metadata_by_scene[scene_id] = dict(scenario.get("metadata", {}))
    return metadata_by_scene


def _strip_intervention_suffix(value: str) -> str:
    text = str(value or "").strip().lower()
    for suffix in ("_semantic_probe", "_ground_truth", "_probe", "_gt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _extract_synthetic_label(metadata: Mapping[str, Any], filename: str) -> str:
    label = _strip_intervention_suffix(str(metadata.get("intervention") or ""))
    if label:
        return label
    stem = Path(filename).stem.lower()
    for candidate in ("left_lane_change", "right_lane_change", "left", "right", "straight", "stop"):
        if f"_{candidate}" in stem or stem.endswith(candidate):
            return candidate
    return ""


def _parse_risk_groups(raw_groups: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for raw in raw_groups or DEFAULT_RISK_GROUPS:
        if "=" not in raw:
            raise ValueError(f"Invalid --risk-group {raw!r}; expected name=level[,level...]")
        name, levels_text = raw.split("=", 1)
        clean_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
        levels = [_normalize_risk(part) for part in levels_text.split(",") if part.strip()]
        if not clean_name or not levels:
            raise ValueError(f"Invalid --risk-group {raw!r}; expected non-empty name and levels")
        groups[clean_name] = sorted(set(levels), key=lambda risk: RISK_RANK.get(risk, -1))
    return groups


def build_vlm_risk_index_from_bundle(bundle_root: Path) -> Dict[str, Any]:
    examples_root = bundle_root / "examples"
    if not examples_root.is_dir():
        raise FileNotFoundError(f"Could not find VLM examples directory: {examples_root}")

    scenes: Dict[str, Dict[str, Any]] = {}
    path_counts: Dict[str, int] = {}
    scene_max_counts: Dict[str, int] = {}

    for example_dir in sorted(examples_root.glob("waymax_scene_*")):
        normalized_paths = [
            path
            for path in sorted(example_dir.glob("contract_normalized_*.json"))
            if path.name != "contract_normalized.json"
        ]
        if not normalized_paths:
            continue

        path_records: List[Dict[str, Any]] = []
        scenario_id = ""
        sdc_id = ""
        for contract_path in normalized_paths:
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            scenario_id = scenario_id or str(contract.get("scenario_id") or "")
            sdc_id = sdc_id or str(contract.get("sdc_id") or "")
            for highlighted in contract.get("highlighted_paths", []) or []:
                risk = _normalize_risk(highlighted.get("risk_level"))
                record = {
                    "slot_id": str(highlighted.get("slot_id") or ""),
                    "path_id": highlighted.get("path_id"),
                    "source_kind": str(highlighted.get("source_kind") or ""),
                    "semantic_label": str(highlighted.get("semantic_label") or "").strip().lower(),
                    "risk_level": risk,
                    "risk_rationale_short": str(highlighted.get("risk_rationale_short") or ""),
                    "confidence": highlighted.get("confidence"),
                    "is_valid_target": bool(highlighted.get("is_valid_target", True)),
                    "contract_path": str(contract_path),
                }
                path_records.append(record)
                path_counts[risk] = int(path_counts.get(risk, 0)) + 1

        if not scenario_id:
            match = SCENE_RE.search(example_dir.name)
            scenario_id = match.group(0) if match else example_dir.name
        if not path_records:
            continue

        risks = [str(record["risk_level"]) for record in path_records]
        gt_records = [record for record in path_records if record.get("slot_id") == "gt"]
        label_to_risks: Dict[str, List[str]] = {}
        for record in path_records:
            label = str(record.get("semantic_label") or "")
            if label:
                label_to_risks.setdefault(label, []).append(str(record["risk_level"]))

        scene_max_risk = _max_risk(risks)
        scene_max_counts[scene_max_risk] = int(scene_max_counts.get(scene_max_risk, 0)) + 1
        scenes[scenario_id] = {
            "scenario_id": scenario_id,
            "sdc_id": sdc_id,
            "scene_max_risk": scene_max_risk,
            "gt_risk": _max_risk([str(record["risk_level"]) for record in gt_records]),
            "any_risks": sorted(set(risks), key=lambda risk: RISK_RANK.get(risk, -1)),
            "label_to_max_risk": {
                label: _max_risk(values)
                for label, values in sorted(label_to_risks.items())
            },
            "path_records": path_records,
        }

    return {
        "schema_version": "td3_vlm_risk_index_v1",
        "source_bundle_root": str(bundle_root),
        "num_scenes": len(scenes),
        "path_risk_counts": path_counts,
        "scene_max_risk_counts": scene_max_counts,
        "scenes": scenes,
    }


def load_or_build_risk_index(*, vlm_risk_index: str, vlm_bundle_root: str) -> Dict[str, Any]:
    if str(vlm_risk_index).strip():
        path = Path(vlm_risk_index).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VLM risk index not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    if str(vlm_bundle_root).strip():
        return build_vlm_risk_index_from_bundle(Path(vlm_bundle_root).expanduser().resolve())
    raise ValueError("Provide either --vlm-risk-index or --vlm-bundle-root")


def _assign_scene_risk(
    *,
    scene_id: str,
    synthetic_label: str,
    risk_record: Mapping[str, Any] | None,
    risk_mode: str,
    label_match_fallback: str,
) -> tuple[str, str]:
    if not risk_record:
        return "unknown", "missing_risk_record"

    if risk_mode == "scene_max":
        return _normalize_risk(risk_record.get("scene_max_risk")), "scene_max"
    if risk_mode == "gt":
        return _normalize_risk(risk_record.get("gt_risk")), "gt"
    if risk_mode != "label_match":
        raise ValueError(f"Unsupported risk mode: {risk_mode}")

    label = str(synthetic_label or "").strip().lower()
    label_to_max = risk_record.get("label_to_max_risk", {}) if isinstance(risk_record, Mapping) else {}
    if label and label in label_to_max:
        return _normalize_risk(label_to_max.get(label)), f"label_match:{label}"

    if label_match_fallback == "skip":
        return "unknown", f"label_match_missing:{label}:skip"
    if label_match_fallback == "gt":
        return _normalize_risk(risk_record.get("gt_risk")), f"label_match_missing:{label}:gt"
    return _normalize_risk(risk_record.get("scene_max_risk")), f"label_match_missing:{label}:scene_max"


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_file(source: Path, dest: Path, *, link_mode: str) -> None:
    source = source.resolve()
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if link_mode == "symlink":
        os.symlink(source, dest)
    elif link_mode == "hardlink":
        os.link(source, dest)
    elif link_mode == "copy":
        shutil.copy2(source, dest)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def create_dataset_summary(scenario_paths: Sequence[Path], output_dir: Path) -> Path | None:
    output_dir = Path(output_dir)
    summary_dict: Dict[str, Dict[str, Any]] = {}

    for scenario_path in sorted(Path(path) for path in scenario_paths):
        if not scenario_path.exists() or not scenario_path.name.startswith("sd_") or not scenario_path.name.endswith(".pkl"):
            continue
        scenario = _read_pickle(scenario_path)
        filename = scenario_path.name
        metadata = dict(scenario.get("metadata", {}))
        summary_entry = dict(metadata)
        summary_entry.update(
            {
                "scenario_id": scenario.get("id", filename.replace(".pkl", "")),
                "sdc_id": metadata.get("sdc_id", ""),
                "dataset": metadata.get("dataset", "counterfactual"),
                "counterfactual": metadata.get("counterfactual", True),
                "intervention": metadata.get("intervention", ""),
            }
        )
        summary_dict[filename] = summary_entry

    if not summary_dict:
        return None

    summary_path = output_dir / "dataset_summary.pkl"
    with summary_path.open("wb") as f:
        pickle.dump(summary_dict, f)

    mapping_path = output_dir / "dataset_mapping.pkl"
    with mapping_path.open("wb") as f:
        pickle.dump({filename: "" for filename in summary_dict}, f)

    return summary_path


def _materialize_dataset_view(
    *,
    dest_dir: Path,
    scenario_paths: Sequence[Path],
    link_mode: str,
) -> Dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    linked_paths: List[Path] = []
    for source_path in scenario_paths:
        dest_path = dest_dir / source_path.name
        _materialize_file(source_path, dest_path, link_mode=link_mode)
        linked_paths.append(dest_path)
    summary_path = create_dataset_summary(linked_paths, dest_dir)
    return {
        "num_scenarios": len(linked_paths),
        "dataset_summary_pkl": None if summary_path is None else str(summary_path),
        "dataset_mapping_pkl": str(dest_dir / "dataset_mapping.pkl"),
    }


def _select_scene_ids(scene_ids: Sequence[str], *, seed: int, shuffle: bool, limit: int) -> List[str]:
    selected = list(scene_ids)
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(selected)
    else:
        selected.sort()
    if limit > 0:
        selected = selected[: int(limit)]
    return selected


def main() -> int:
    args = parse_args()

    base_views_dir = Path(args.base_views_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    _prepare_output_dir(outdir, overwrite=bool(args.overwrite))

    train_waymo_dir = base_views_dir / "train_waymo_only"
    train_mixed_dir = base_views_dir / "train_counterbmt_mixed"
    eval_waymo_dir = base_views_dir / "eval_waymo_only"
    eval_adv_dir = base_views_dir / "eval_counterbmt_adversarial"

    for required in (train_waymo_dir, train_mixed_dir, eval_waymo_dir, eval_adv_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"Expected TD3 view directory does not exist: {required}")

    risk_index = load_or_build_risk_index(
        vlm_risk_index=str(args.vlm_risk_index),
        vlm_bundle_root=str(args.vlm_bundle_root),
    )
    if str(args.risk_index_out).strip():
        risk_index_path = Path(args.risk_index_out).expanduser().resolve()
    else:
        risk_index_path = outdir / "vlm_risk_index.json"
    risk_index_path.parent.mkdir(parents=True, exist_ok=True)
    risk_index_path.write_text(json.dumps(risk_index, indent=2, sort_keys=True), encoding="utf-8")

    risk_scenes = risk_index.get("scenes", {})
    if not isinstance(risk_scenes, dict):
        raise TypeError("Risk index must contain a dict field named 'scenes'")

    natural_map = _build_scene_to_file_map(train_waymo_dir, counterfactual=False)
    synthetic_map = _build_scene_to_file_map(train_mixed_dir, counterfactual=True)
    eval_waymo_map = _build_scene_to_file_map(eval_waymo_dir, counterfactual=False)
    eval_adv_map = _build_scene_to_file_map(eval_adv_dir, counterfactual=True)
    synthetic_metadata = _load_metadata_for_paths(synthetic_map)

    assignments: Dict[str, Dict[str, Any]] = {}
    missing_risk_scenes: List[str] = []
    for scene_id, path in sorted(synthetic_map.items()):
        metadata = synthetic_metadata.get(scene_id, {})
        synthetic_label = _extract_synthetic_label(metadata, path.name)
        risk_record = risk_scenes.get(scene_id)
        assigned_risk, assignment_source = _assign_scene_risk(
            scene_id=scene_id,
            synthetic_label=synthetic_label,
            risk_record=risk_record if isinstance(risk_record, Mapping) else None,
            risk_mode=str(args.risk_mode),
            label_match_fallback=str(args.label_match_fallback),
        )
        if risk_record is None:
            missing_risk_scenes.append(scene_id)
        assignments[scene_id] = {
            "scene_id": scene_id,
            "synthetic_path": str(path),
            "synthetic_label": synthetic_label,
            "assigned_risk": assigned_risk,
            "assignment_source": assignment_source,
            "risk_record": risk_record,
        }

    groups = _parse_risk_groups(args.risk_group)
    group_to_candidate_ids: Dict[str, List[str]] = {}
    for group_name, levels in groups.items():
        level_set = set(levels)
        group_to_candidate_ids[group_name] = [
            scene_id
            for scene_id, assignment in assignments.items()
            if str(assignment.get("assigned_risk")) in level_set
        ]

    target_pairs = int(args.target_pairs)
    if args.balance_to_min:
        non_empty_sizes = [len(ids) for ids in group_to_candidate_ids.values() if ids]
        if not non_empty_sizes:
            raise ValueError("No non-empty risk groups available to balance.")
        target_pairs = min(non_empty_sizes) if target_pairs <= 0 else min(target_pairs, min(non_empty_sizes))

    group_to_selected_ids = {
        group_name: _select_scene_ids(
            scene_ids,
            seed=int(args.selection_seed),
            shuffle=bool(args.shuffle_scenes),
            limit=target_pairs,
        )
        for group_name, scene_ids in group_to_candidate_ids.items()
    }

    eval_waymo_meta = _materialize_dataset_view(
        dest_dir=outdir / "eval_waymo_only",
        scenario_paths=[eval_waymo_map[sid] for sid in sorted(eval_waymo_map)],
        link_mode=str(args.link_mode),
    )
    eval_adv_meta = _materialize_dataset_view(
        dest_dir=outdir / "eval_counterbmt_adversarial",
        scenario_paths=[eval_adv_map[sid] for sid in sorted(eval_adv_map)],
        link_mode=str(args.link_mode),
    )
    train_waymo_meta = _materialize_dataset_view(
        dest_dir=outdir / "train_waymo_only",
        scenario_paths=[natural_map[sid] for sid in sorted(natural_map)],
        link_mode=str(args.link_mode),
    )

    view_meta: Dict[str, Dict[str, Any]] = {
        "train_waymo_only": {
            "path": str(outdir / "train_waymo_only"),
            **train_waymo_meta,
        },
        "eval_waymo_only": {
            "path": str(outdir / "eval_waymo_only"),
            **eval_waymo_meta,
        },
        "eval_counterbmt_adversarial": {
            "path": str(outdir / "eval_counterbmt_adversarial"),
            **eval_adv_meta,
        },
    }

    for group_name, selected_ids in group_to_selected_ids.items():
        if str(args.natural_mode) == "matching":
            natural_ids = [sid for sid in selected_ids if sid in natural_map]
        else:
            natural_ids = sorted(natural_map)
        synthetic_ids = [sid for sid in selected_ids if sid in synthetic_map]
        view_name = f"train_counterbmt_mixed_risk_{group_name}"
        scenario_paths = [natural_map[sid] for sid in natural_ids] + [synthetic_map[sid] for sid in synthetic_ids]
        meta = _materialize_dataset_view(
            dest_dir=outdir / view_name,
            scenario_paths=scenario_paths,
            link_mode=str(args.link_mode),
        )
        view_meta[view_name] = {
            "path": str(outdir / view_name),
            "risk_group": group_name,
            "risk_levels": groups[group_name],
            "natural_mode": str(args.natural_mode),
            "num_natural_scenarios": len(natural_ids),
            "num_synthetic_scenarios": len(synthetic_ids),
            **meta,
        }

    manifest = {
        "schema_version": "td3_risk_ablation_views_v1",
        "base_views_dir": str(base_views_dir),
        "vlm_bundle_root": str(args.vlm_bundle_root),
        "vlm_risk_index": str(risk_index_path),
        "risk_mode": str(args.risk_mode),
        "label_match_fallback": str(args.label_match_fallback),
        "natural_mode": str(args.natural_mode),
        "selection": {
            "selection_seed": int(args.selection_seed),
            "shuffle_scenes": bool(args.shuffle_scenes),
            "target_pairs": int(target_pairs),
            "balance_to_min": bool(args.balance_to_min),
            "link_mode": str(args.link_mode),
        },
        "base_counts": {
            "train_natural": len(natural_map),
            "train_synthetic": len(synthetic_map),
            "eval_natural": len(eval_waymo_map),
            "eval_synthetic": len(eval_adv_map),
            "missing_risk_scenes": len(missing_risk_scenes),
        },
        "risk_index_summary": {
            "num_scenes": risk_index.get("num_scenes"),
            "path_risk_counts": risk_index.get("path_risk_counts"),
            "scene_max_risk_counts": risk_index.get("scene_max_risk_counts"),
        },
        "risk_groups": {
            group_name: {
                "levels": levels,
                "available_synthetic_scenes": len(group_to_candidate_ids[group_name]),
                "selected_synthetic_scenes": len(group_to_selected_ids[group_name]),
                "selected_scene_ids": group_to_selected_ids[group_name],
            }
            for group_name, levels in groups.items()
        },
        "views": view_meta,
        "assignments": assignments,
        "missing_risk_scenes": missing_risk_scenes,
        "suggested_td3_env": {
            f"risk_{group_name}_natural_eval": {
                "DATA_DIR": str(outdir / f"train_counterbmt_mixed_risk_{group_name}"),
                "EVAL_DATA_DIR": str(outdir / "eval_waymo_only"),
            }
            for group_name in groups
        },
    }

    manifest_path = outdir / "td3_risk_ablation_views_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
