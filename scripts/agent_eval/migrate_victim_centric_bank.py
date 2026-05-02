from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


POLYGON_FEATURE_TYPES = {"CROSSWALK", "SPEED_BUMP", "DRIVEWAY"}


def _normalize_track_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    return "" if value is None else str(value)


def _infer_track_length(scenario: Dict[str, Any]) -> int:
    tracks = dict(scenario.get("tracks", {}))
    if not tracks:
        return 0
    first_track = next(iter(tracks.values()))
    position = np.asarray(first_track.get("state", {}).get("position", []))
    if position.ndim == 0:
        return 0
    return int(position.shape[0])


def _is_waymax_reconstructed_scenario(scenario: Dict[str, Any]) -> bool:
    metadata = dict(scenario.get("metadata", {}))
    if str(metadata.get("source_format", "")).strip() == "waymax_womd":
        return True
    scenario_id = str(scenario.get("id") or metadata.get("scenario_id") or "").strip()
    return scenario_id.startswith("waymax_scene_")


def _normalize_waymax_map_features(map_features: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for feature_id, feature in dict(map_features).items():
        if not isinstance(feature, dict):
            normalized[str(feature_id)] = feature
            continue
        item = pickle.loads(pickle.dumps(feature))
        item.pop("metadata", None)
        feature_type = str(item.get("type") or "").strip()
        polyline = item.get("polyline")
        polyline_np = None
        if polyline is not None:
            polyline_np = np.asarray(polyline, dtype=np.float32)
            item["polyline"] = polyline_np
        if feature_type == "ROADGRAPH_TYPE_20":
            item["type"] = "DRIVEWAY"
            feature_type = "DRIVEWAY"
        if feature_type in POLYGON_FEATURE_TYPES and polyline_np is not None and polyline_np.size > 0:
            item["polygon"] = np.asarray(polyline_np, dtype=np.float32)
            item.pop("polyline", None)
        if feature_type == "STOP_SIGN" and polyline_np is not None and polyline_np.size > 0:
            item["position"] = np.asarray(polyline_np[0], dtype=np.float32)
            item.setdefault("lane", [])
            item.pop("polyline", None)
        normalized[str(feature_id)] = item
    return normalized


def _is_waymax_placeholder_track(track_id: Any, track: Any) -> bool:
    if not isinstance(track, dict):
        return False
    track_name = _normalize_track_name(track_id)
    metadata = dict(track.get("metadata", {}))
    raw_object_type = metadata.get("raw_object_type_id", None)
    state = dict(track.get("state", {}))
    valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)

    try:
        if int(track_name) < 0:
            return True
    except Exception:
        pass
    try:
        if raw_object_type is not None and int(raw_object_type) < 0:
            return True
    except Exception:
        pass
    if valid.size == 0 or not bool(np.any(valid)):
        return True
    return False


def _normalize_waymax_tracks(tracks: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for track_id, track in dict(tracks).items():
        if _is_waymax_placeholder_track(track_id, track):
            continue
        normalized[_normalize_track_name(track_id)] = pickle.loads(pickle.dumps(track))
    return normalized


def normalize_scenario_for_metadrive(
    scenario: Dict[str, Any],
    *,
    original_file_path: Path | None = None,
) -> Dict[str, Any]:
    scenario_copy = pickle.loads(pickle.dumps(scenario))
    if not _is_waymax_reconstructed_scenario(scenario_copy):
        return scenario_copy
    track_length = _infer_track_length(scenario_copy)
    scenario_id = str(scenario_copy.get("id") or scenario_copy.get("metadata", {}).get("scenario_id") or "").strip()
    scenario_copy["id"] = scenario_id
    scenario_copy["version"] = str(scenario_copy.get("version") or "v1.2")
    scenario_copy["length"] = int(scenario_copy.get("length") or track_length)
    scenario_copy["tracks"] = _normalize_waymax_tracks(dict(scenario_copy.get("tracks", {})))
    scenario_copy["map_features"] = _normalize_waymax_map_features(dict(scenario_copy.get("map_features", {})))

    metadata = scenario_copy.setdefault("metadata", {})
    metadata.setdefault("id", scenario_id)
    metadata.setdefault("coordinate", "waymo")
    metadata.setdefault("dataset", "waymo")
    metadata.setdefault("scenario_id", scenario_id)
    metadata.setdefault("track_length", track_length)
    metadata.setdefault("metadrive_processed", False)
    metadata.setdefault("objects_of_interest", [])
    metadata.setdefault("tracks_to_predict", {})
    if original_file_path is not None:
        metadata.setdefault("source_file", str(original_file_path.expanduser()))
    else:
        metadata.setdefault("source_file", "raw_reconstructed")
    return scenario_copy


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
        with scenario_path.open("rb") as fp:
            scenario = pickle.load(fp)
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
    with summary_path.open("wb") as fp:
        pickle.dump(summary_dict, fp)

    mapping_path = output_dir / "dataset_mapping.pkl"
    with mapping_path.open("wb") as fp:
        pickle.dump({filename: "" for filename in summary_dict}, fp)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate an existing victim-centric bank into MetaDrive-compatible ScenarioNet schema."
    )
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--natural-subdir", type=str, default="natural_scenarios")
    parser.add_argument("--adversarial-subdir", type=str, default="adversarial_scenarios")
    parser.add_argument("--max-scenarios-per-dir", type=int, default=0)
    parser.add_argument("--copy-scene-analysis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _scenario_files(data_dir: Path, *, limit: int) -> List[Path]:
    files = sorted(
        path
        for path in data_dir.glob("sd_*.pkl")
        if path.name not in {"dataset_summary.pkl", "dataset_mapping.pkl"}
    )
    if limit > 0:
        return files[:limit]
    return files


def _count_polygon_features(scenario: Dict[str, Any]) -> int:
    return sum(
        1
        for value in dict(scenario.get("map_features", {})).values()
        if isinstance(value, dict) and "polygon" in value
    )


def _count_polyline_features(scenario: Dict[str, Any]) -> int:
    return sum(
        1
        for value in dict(scenario.get("map_features", {})).values()
        if isinstance(value, dict) and "polyline" in value
    )


def _migrate_directory(*, source_dir: Path, dest_dir: Path, limit: int) -> Dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_files = _scenario_files(source_dir, limit=limit)
    migrated_paths: List[Path] = []
    changed_files = 0
    polygon_before = 0
    polygon_after = 0
    polyline_before = 0
    polyline_after = 0
    version_missing_before = 0
    version_missing_after = 0

    for source_path in source_files:
        with source_path.open("rb") as fp:
            scenario = pickle.load(fp)
        polygon_before += _count_polygon_features(scenario)
        polyline_before += _count_polyline_features(scenario)
        if scenario.get("version") in (None, ""):
            version_missing_before += 1

        migrated = normalize_scenario_for_metadrive(scenario, original_file_path=source_path)

        polygon_after += _count_polygon_features(migrated)
        polyline_after += _count_polyline_features(migrated)
        if migrated.get("version") in (None, ""):
            version_missing_after += 1
        if pickle.dumps(migrated, protocol=pickle.HIGHEST_PROTOCOL) != pickle.dumps(
            scenario,
            protocol=pickle.HIGHEST_PROTOCOL,
        ):
            changed_files += 1

        dest_path = dest_dir / source_path.name
        with dest_path.open("wb") as fp:
            pickle.dump(migrated, fp)
        migrated_paths.append(dest_path)

    summary_path = create_dataset_summary(migrated_paths, dest_dir) if migrated_paths else None
    return {
        "source_dir": source_dir,
        "dest_dir": dest_dir,
        "num_scenarios": len(migrated_paths),
        "num_changed_files": int(changed_files),
        "polygon_before": int(polygon_before),
        "polygon_after": int(polygon_after),
        "polyline_before": int(polyline_before),
        "polyline_after": int(polyline_after),
        "version_missing_before": int(version_missing_before),
        "version_missing_after": int(version_missing_after),
        "dataset_summary_pkl": summary_path,
        "dataset_mapping_pkl": dest_dir / "dataset_mapping.pkl" if migrated_paths else None,
    }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if outdir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{outdir} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    natural_source = source_root / str(args.natural_subdir)
    adversarial_source = source_root / str(args.adversarial_subdir)
    if not natural_source.is_dir():
        raise FileNotFoundError(f"natural scenario dir not found: {natural_source}")
    if not adversarial_source.is_dir():
        raise FileNotFoundError(f"adversarial scenario dir not found: {adversarial_source}")

    natural_report = _migrate_directory(
        source_dir=natural_source,
        dest_dir=outdir / str(args.natural_subdir),
        limit=int(args.max_scenarios_per_dir),
    )
    adversarial_report = _migrate_directory(
        source_dir=adversarial_source,
        dest_dir=outdir / str(args.adversarial_subdir),
        limit=int(args.max_scenarios_per_dir),
    )

    if args.copy_scene_analysis:
        source_scene_analysis = source_root / "scene_analysis"
        if source_scene_analysis.is_dir():
            shutil.copytree(source_scene_analysis, outdir / "scene_analysis")

    report = {
        "source_root": source_root,
        "outdir": outdir,
        "natural": natural_report,
        "adversarial": adversarial_report,
        "copied_scene_analysis": bool(args.copy_scene_analysis and (source_root / "scene_analysis").is_dir()),
        "max_scenarios_per_dir": int(args.max_scenarios_per_dir),
    }
    report_path = outdir / "migration_report.json"
    report_path.write_text(json.dumps(_json_default(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_default(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
