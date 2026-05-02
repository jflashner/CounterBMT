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
from typing import Any, Dict, Iterable, List, Mapping, Sequence


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    modern_src = repo_root / "src"
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (modern_src, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


WAYMAX_SCENE_RE = re.compile(r"waymax_scene_\d+")
COUNTERFACTUAL_SCENE_RE = re.compile(r"sd_counterfactual_1\.0_([^_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare TD3-ready ScenarioNet directory views for Table 4 style victim-centric evaluation."
    )
    parser.add_argument("--train-natural-dir", type=str, required=True)
    parser.add_argument("--train-adversarial-dir", type=str, required=True)
    parser.add_argument("--val-natural-dir", type=str, required=True)
    parser.add_argument("--val-adversarial-dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--target-train-pairs", type=int, default=500)
    parser.add_argument("--target-val-pairs", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--shuffle-scenes", action="store_true")
    parser.add_argument(
        "--disjoint-val-from-train",
        action="store_true",
        help="When train/val inputs share source scenes, remove selected train scenes before selecting validation scenes.",
    )
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "hardlink", "copy"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _extract_scene_id(filename: str, metadata: Mapping[str, Any]) -> str:
    for key in ("source_scenario", "scenario_id", "id"):
        value = str(metadata.get(key) or "").strip()
        if key == "source_scenario" and value:
            return value
        match = WAYMAX_SCENE_RE.search(value)
        if match:
            return match.group(0)
    match = WAYMAX_SCENE_RE.search(str(filename))
    if match:
        return match.group(0)
    match = COUNTERFACTUAL_SCENE_RE.search(str(filename))
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract scene id from {filename!r}")


def _load_summary(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    summary_path = data_dir / "dataset_summary.pkl"
    if not summary_path.exists():
        raise FileNotFoundError(f"dataset_summary.pkl not found under {data_dir}")
    with summary_path.open("rb") as f:
        summary = pickle.load(f)
    if not isinstance(summary, dict):
        raise TypeError(f"dataset_summary.pkl under {data_dir} must contain dict, got {type(summary)}")
    return summary


def _build_scene_to_file_map(data_dir: Path) -> Dict[str, Path]:
    summary = _load_summary(data_dir)
    scene_to_path: Dict[str, Path] = {}
    for filename, metadata in summary.items():
        if not str(filename).startswith("sd_") or not str(filename).endswith(".pkl"):
            continue
        meta_dict = metadata if isinstance(metadata, dict) else {}
        scene_id = _extract_scene_id(str(filename), meta_dict)
        source_path = data_dir / str(filename)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing scenario referenced by dataset_summary: {source_path}")
        if scene_id in scene_to_path:
            raise ValueError(f"Duplicate scene id {scene_id} in {data_dir}")
        scene_to_path[scene_id] = source_path
    return scene_to_path


def _select_scene_ids(
    natural_map: Mapping[str, Path],
    adversarial_map: Mapping[str, Path],
    *,
    target_pairs: int,
    shuffle_scenes: bool,
    selection_seed: int,
) -> List[str]:
    scene_ids = sorted(set(natural_map) & set(adversarial_map))
    if shuffle_scenes:
        rng = random.Random(selection_seed)
        rng.shuffle(scene_ids)
    return scene_ids[: min(int(target_pairs), len(scene_ids))]


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _materialize_file(source: Path, dest: Path, *, link_mode: str) -> None:
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


def _materialize_dataset_view(
    *,
    dest_dir: Path,
    scenario_paths: Sequence[Path],
    link_mode: str,
) -> Dict[str, Any]:
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


def create_dataset_summary(scenario_paths: Sequence[Path], output_dir: Path) -> Path | None:
    output_dir = Path(output_dir)
    summary_dict: Dict[str, Dict[str, Any]] = {}
    all_scenario_files = set()

    for path in scenario_paths:
        path = Path(path)
        if path.exists() and path.name.startswith("sd_") and path.name.endswith(".pkl"):
            all_scenario_files.add(path)

    if output_dir.exists():
        for scenario_path in output_dir.glob("sd_*.pkl"):
            if scenario_path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}:
                all_scenario_files.add(scenario_path)

    for scenario_path in sorted(all_scenario_files):
        if not scenario_path.exists():
            continue
        with scenario_path.open("rb") as f:
            scenario = pickle.load(f)

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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def main() -> None:
    args = parse_args()

    train_nat_dir = Path(args.train_natural_dir).expanduser().resolve()
    train_adv_dir = Path(args.train_adversarial_dir).expanduser().resolve()
    val_nat_dir = Path(args.val_natural_dir).expanduser().resolve()
    val_adv_dir = Path(args.val_adversarial_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    train_nat_map = _build_scene_to_file_map(train_nat_dir)
    train_adv_map = _build_scene_to_file_map(train_adv_dir)
    val_nat_map = _build_scene_to_file_map(val_nat_dir)
    val_adv_map = _build_scene_to_file_map(val_adv_dir)

    selected_train_ids = _select_scene_ids(
        train_nat_map,
        train_adv_map,
        target_pairs=args.target_train_pairs,
        shuffle_scenes=args.shuffle_scenes,
        selection_seed=args.selection_seed,
    )
    if bool(args.disjoint_val_from_train):
        train_id_set = set(selected_train_ids)
        val_nat_candidates = {sid: path for sid, path in val_nat_map.items() if sid not in train_id_set}
        val_adv_candidates = {sid: path for sid, path in val_adv_map.items() if sid not in train_id_set}
    else:
        val_nat_candidates = val_nat_map
        val_adv_candidates = val_adv_map
    selected_val_ids = _select_scene_ids(
        val_nat_candidates,
        val_adv_candidates,
        target_pairs=args.target_val_pairs,
        shuffle_scenes=args.shuffle_scenes,
        selection_seed=args.selection_seed,
    )

    train_waymo_dir = outdir / "train_waymo_only"
    train_mixed_dir = outdir / "train_counterbmt_mixed"
    eval_waymo_dir = outdir / "eval_waymo_only"
    eval_adv_dir = outdir / "eval_counterbmt_adversarial"

    _prepare_output_dir(outdir, overwrite=args.overwrite)
    for path in (train_waymo_dir, train_mixed_dir, eval_waymo_dir, eval_adv_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_nat_paths = [train_nat_map[sid] for sid in selected_train_ids]
    train_adv_paths = [train_adv_map[sid] for sid in selected_train_ids]
    val_nat_paths = [val_nat_map[sid] for sid in selected_val_ids]
    val_adv_paths = [val_adv_map[sid] for sid in selected_val_ids]

    train_waymo_meta = _materialize_dataset_view(
        dest_dir=train_waymo_dir,
        scenario_paths=train_nat_paths,
        link_mode=args.link_mode,
    )
    train_mixed_meta = _materialize_dataset_view(
        dest_dir=train_mixed_dir,
        scenario_paths=[*train_nat_paths, *train_adv_paths],
        link_mode=args.link_mode,
    )
    eval_waymo_meta = _materialize_dataset_view(
        dest_dir=eval_waymo_dir,
        scenario_paths=val_nat_paths,
        link_mode=args.link_mode,
    )
    eval_adv_meta = _materialize_dataset_view(
        dest_dir=eval_adv_dir,
        scenario_paths=val_adv_paths,
        link_mode=args.link_mode,
    )

    manifest = {
        "train_inputs": {
            "natural_dir": str(train_nat_dir),
            "adversarial_dir": str(train_adv_dir),
            "available_paired_scenes": len(set(train_nat_map) & set(train_adv_map)),
            "requested_pairs": int(args.target_train_pairs),
            "selected_pairs": len(selected_train_ids),
        },
        "val_inputs": {
            "natural_dir": str(val_nat_dir),
            "adversarial_dir": str(val_adv_dir),
            "available_paired_scenes": len(set(val_nat_map) & set(val_adv_map)),
            "available_paired_scenes_after_train_exclusion": len(set(val_nat_candidates) & set(val_adv_candidates)),
            "requested_pairs": int(args.target_val_pairs),
            "selected_pairs": len(selected_val_ids),
        },
        "selection": {
            "shuffle_scenes": bool(args.shuffle_scenes),
            "selection_seed": int(args.selection_seed),
            "link_mode": str(args.link_mode),
            "disjoint_val_from_train": bool(args.disjoint_val_from_train),
        },
        "views": {
            "train_waymo_only": {
                "path": str(train_waymo_dir),
                **train_waymo_meta,
            },
            "train_counterbmt_mixed": {
                "path": str(train_mixed_dir),
                **train_mixed_meta,
            },
            "eval_waymo_only": {
                "path": str(eval_waymo_dir),
                **eval_waymo_meta,
            },
            "eval_counterbmt_adversarial": {
                "path": str(eval_adv_dir),
                **eval_adv_meta,
            },
        },
        "selected_train_scene_ids": selected_train_ids,
        "selected_val_scene_ids": selected_val_ids,
        "suggested_td3_args": {
            "waymo_row": {
                "data_dir": str(train_waymo_dir),
                "eval_data_dir": str(eval_waymo_dir),
            },
            "counterbmt_row": {
                "data_dir": str(train_mixed_dir),
                "eval_data_dir": str(eval_waymo_dir),
            },
            "posthoc_adversarial_eval": {
                "eval_data_dir": str(eval_adv_dir),
            },
        },
    }

    manifest_path = outdir / "td3_table4_views_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=_json_default)

    print(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
