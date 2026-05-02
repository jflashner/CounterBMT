from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    modern_src = repo_root / "src"
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, modern_src, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from counter_bmt.scenario_export import (  # noqa: E402
    create_dataset_summary,
    normalize_scenario_for_metadrive,
)
from bmt.counterfactual.normalize import load_raw_scenario  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-faithful Adv-BMT/SCGEN offline bank for Table 4 style TD3 evaluation."
    )
    parser.add_argument("--selection-manifest", type=str, required=True)
    parser.add_argument("--split", type=str, choices=("train", "val"), required=True)
    parser.add_argument("--source-builder-manifest", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--scgen-ckpt", type=str, default="src/Adv-BMT/bmt/ckpt/last.ckpt")
    parser.add_argument("--model-name", type=str, default="0202_midgpt")
    parser.add_argument("--tf-mode", type=str, default="all_TF_except_adv")
    parser.add_argument("--track-length", type=int, default=91)
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_default(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _load_selected_scene_ids(path: Path, *, split: str) -> List[str]:
    payload = _load_json(path)
    key = "selected_train_scene_ids" if split == "train" else "selected_val_scene_ids"
    raw = payload.get(key, [])
    if not isinstance(raw, list):
        raise TypeError(f"{path} missing list field {key!r}")
    return [str(scene_id).strip() for scene_id in raw if str(scene_id).strip()]


def _load_source_records(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    scenes = payload.get("scenes", [])
    if not isinstance(scenes, list):
        raise TypeError(f"{path} missing list field 'scenes'")
    out: Dict[str, Dict[str, Any]] = {}
    for item in scenes:
        if not isinstance(item, dict):
            continue
        scenario_id = str(item.get("scenario_id") or "").strip()
        if not scenario_id:
            continue
        out[scenario_id] = dict(item)
    return out


def _resolve_source_scenario_path(record: Mapping[str, Any]) -> Path:
    for key in ("scenario_pkl", "export_source_pkl"):
        raw = str(record.get(key) or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve source scenario path from record keys scenario_pkl/export_source_pkl: {record}"
    )


def _annotate_natural_scenario(*, scenario: Dict[str, Any], scene_id: str, source_path: Path) -> Dict[str, Any]:
    normalized = normalize_scenario_for_metadrive(scenario, original_file_path=source_path)
    normalized["id"] = str(scene_id)
    metadata = normalized.setdefault("metadata", {})
    metadata["source_scenario"] = str(scene_id)
    metadata["counterfactual"] = False
    metadata["intervention"] = "ground_truth"
    metadata["dataset"] = "waymo_advbmt_table4"
    metadata["advbmt_paper_faithful"] = True
    return normalized


def _annotate_generated_scenario(
    *,
    scenario: Dict[str, Any],
    scene_id: str,
    source_path: Path,
    generator_info: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = normalize_scenario_for_metadrive(scenario, original_file_path=source_path)
    normalized["id"] = f"{scene_id}_advbmt_scgen"
    metadata = normalized.setdefault("metadata", {})
    metadata["source_scenario"] = str(scene_id)
    metadata["counterfactual"] = True
    metadata["intervention"] = "advbmt_scgen"
    metadata["dataset"] = "waymo_advbmt_table4"
    metadata["advbmt_paper_faithful"] = True
    metadata["generator_name"] = "SCGEN"
    metadata["generator_model_name"] = str(generator_info.get("model_name") or "")
    metadata["generator_tf_mode"] = str(generator_info.get("tf_mode") or "")
    metadata["generator_ckpt"] = str(generator_info.get("ckpt") or "")
    return normalized


def _save_scenario(path: Path, scenario: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(dict(scenario), fp)
    return path


def _slice_scene_ids(scene_ids: Sequence[str], *, scene_offset: int, num_scenes: int) -> List[str]:
    selected = list(scene_ids[int(max(0, scene_offset)) :])
    if int(num_scenes) > 0:
        selected = selected[: int(num_scenes)]
    return selected


def main() -> int:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    selection_manifest = Path(args.selection_manifest).expanduser()
    source_builder_manifest = Path(args.source_builder_manifest).expanduser()
    outdir = Path(args.outdir).expanduser()
    if not selection_manifest.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {selection_manifest}")
    if not source_builder_manifest.is_file():
        raise FileNotFoundError(f"Source builder manifest not found: {source_builder_manifest}")
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    scene_ids = _slice_scene_ids(
        _load_selected_scene_ids(selection_manifest, split=str(args.split)),
        scene_offset=int(args.scene_offset),
        num_scenes=int(args.num_scenes),
    )
    source_records = _load_source_records(source_builder_manifest)

    natural_dir = outdir / "natural_scenarios"
    adversarial_dir = outdir / "adversarial_scenarios"
    analysis_dir = outdir / "scene_analysis"
    natural_dir.mkdir(parents=True, exist_ok=True)
    adversarial_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    from bmt.rl_train.train.scgen_generator import SCGEN_Generator  # noqa: E402

    generator = SCGEN_Generator(
        model_name=str(args.model_name),
        TF_mode=str(args.tf_mode),
        ckpt_path=str(Path(args.scgen_ckpt).expanduser()),
    )
    generator_info = {
        "model_name": str(args.model_name),
        "tf_mode": str(args.tf_mode),
        "ckpt": str(Path(args.scgen_ckpt).expanduser()),
    }

    natural_paths: List[Path] = []
    adversarial_paths: List[Path] = []
    scene_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for scene_index, scene_id in enumerate(scene_ids):
        scene_dir = analysis_dir / _safe_name(scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        try:
            if scene_id not in source_records:
                raise KeyError(f"Scene {scene_id} not found in source builder manifest")
            record = source_records[scene_id]
            source_path = _resolve_source_scenario_path(record)
            original_scenario = load_raw_scenario(source_path)

            natural_scenario = _annotate_natural_scenario(
                scenario=original_scenario,
                scene_id=scene_id,
                source_path=source_path,
            )
            generated_raw = generator.generate_from_raw_SD(
                scenario_data=pickle.loads(pickle.dumps(original_scenario)),
                track_length=int(args.track_length),
            )
            if generated_raw is None:
                raise RuntimeError("SCGEN generator returned None")
            adversarial_scenario = _annotate_generated_scenario(
                scenario=generated_raw,
                scene_id=scene_id,
                source_path=source_path,
                generator_info=generator_info,
            )

            natural_path = _save_scenario(
                natural_dir / f"sd_{scene_id}_ground_truth.pkl",
                natural_scenario,
            )
            adversarial_path = _save_scenario(
                adversarial_dir / f"sd_{scene_id}_advbmt_scgen.pkl",
                adversarial_scenario,
            )
            natural_paths.append(natural_path)
            adversarial_paths.append(adversarial_path)

            scene_summary = {
                "scene_index": int(scene_index),
                "scenario_id": str(scene_id),
                "source_scenario_pkl": str(source_path),
                "natural_scenario_pkl": str(natural_path),
                "adversarial_scenario_pkl": str(adversarial_path),
                "generator": dict(generator_info),
                "source_record": {
                    "scenario_pkl": str(record.get("scenario_pkl") or ""),
                    "export_source_pkl": str(record.get("export_source_pkl") or ""),
                },
            }
            scene_results.append(scene_summary)
            _write_json(scene_dir / "scene_summary.json", scene_summary)
        except Exception as exc:
            failure = {
                "scene_index": int(scene_index),
                "scenario_id": str(scene_id),
                "reason": "exception",
                "error": repr(exc),
            }
            skipped.append(failure)
            _write_json(scene_dir / "scene_skip.json", failure)

    natural_summary_path = create_dataset_summary(natural_paths, natural_dir) if natural_paths else None
    adversarial_summary_path = create_dataset_summary(adversarial_paths, adversarial_dir) if adversarial_paths else None

    builder_summary = {
        "selection_manifest": str(selection_manifest),
        "source_builder_manifest": str(source_builder_manifest),
        "split": str(args.split),
        "num_requested_scenes": int(len(scene_ids)),
        "num_exported_pairs": int(len(scene_results)),
        "num_skipped_scenes": int(len(skipped)),
        "natural_dir": str(natural_dir),
        "adversarial_dir": str(adversarial_dir),
        "natural_dataset_summary_pkl": None if natural_summary_path is None else str(natural_summary_path),
        "adversarial_dataset_summary_pkl": None if adversarial_summary_path is None else str(adversarial_summary_path),
        "generator": dict(generator_info),
        "track_length": int(args.track_length),
        "seed": int(args.seed),
    }

    _write_json(outdir / "builder_summary.json", builder_summary)
    _write_json(outdir / "dataset_manifest.json", {"scenes": scene_results, "skipped_scenes": skipped})
    print(json.dumps(_json_default(builder_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
