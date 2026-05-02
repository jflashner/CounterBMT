from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import struct
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (repo_root, repo_root / "src", repo_root / "src" / "Adv-BMT", repo_root / "scenarionet"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    # We only need ScenarioNet converter modules here. Some editable checkouts
    # import heavier MetaDrive engine modules from scenarionet.__init__, which
    # can trigger local circular imports. Pre-seeding a light package object lets
    # Python resolve scenarionet.converter.* without executing that initializer.
    scenarionet_pkg = repo_root / "scenarionet" / "scenarionet"
    if scenarionet_pkg.exists() and "scenarionet" not in sys.modules:
        module = types.ModuleType("scenarionet")
        module.__path__ = [str(scenarionet_pkg)]  # type: ignore[attr-defined]
        sys.modules["scenarionet"] = module


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _list_tfrecords(raw_data_dir: Path) -> list[Path]:
    files = [
        path
        for path in raw_data_dir.iterdir()
        if path.is_file() and "tfrecord" in path.name and not path.name.endswith("_.gstmp")
    ]
    return sorted(files, key=lambda p: p.name)


def _iter_tfrecord_payloads(file_path: Path):
    """Yield uncompressed TFRecord payload bytes.

    The WOMD files used here are plain TFRecord files. TensorFlow is convenient
    but not required for reading the container format, and avoiding that import
    makes local conversion possible on lightweight environments.
    """

    with file_path.open("rb") as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise EOFError(f"Truncated TFRecord length header in {file_path}")
            (length,) = struct.unpack("<Q", length_bytes)
            length_crc = f.read(4)
            if len(length_crc) != 4:
                raise EOFError(f"Truncated TFRecord length CRC in {file_path}")
            data = f.read(length)
            if len(data) != length:
                raise EOFError(f"Truncated TFRecord payload in {file_path}")
            data_crc = f.read(4)
            if len(data_crc) != 4:
                raise EOFError(f"Truncated TFRecord data CRC in {file_path}")
            yield data


def _scan_records(raw_data_dir: Path) -> list[dict[str, Any]]:
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for file_path in _list_tfrecords(raw_data_dir):
        for record_index, data in enumerate(_iter_tfrecord_payloads(file_path)):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data)
            scenario_id = str(scenario.scenario_id)
            if scenario_id in seen:
                raise ValueError(f"Duplicate Waymo scenario_id found while scanning validation set: {scenario_id}")
            seen.add(scenario_id)
            records.append(
                {
                    "scenario_id": scenario_id,
                    "source_file": str(file_path),
                    "record_index": int(record_index),
                }
            )
    return records


def _write_scenarionet_samples(
    *,
    raw_data_dir: Path,
    out_root: Path,
    records: list[dict[str, Any]],
    seed_to_selected_ids: dict[int, list[str]],
    dataset_name: str,
    version: str,
    overwrite: bool,
) -> dict[str, Any]:
    from bmt.counterfactual.scenarionet_waymo_export_source import SPLIT_KEY, _convert_waymo_scenario
    from scenarionet.converter.waymo.waymo_protos import scenario_pb2

    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    id_to_record = {str(row["scenario_id"]): row for row in records}
    requested_ids = sorted({sid for ids in seed_to_selected_ids.values() for sid in ids})
    requested_by_file: dict[str, set[str]] = defaultdict(set)
    for scenario_id in requested_ids:
        requested_by_file[str(id_to_record[scenario_id]["source_file"])].add(scenario_id)

    converted_root = out_root / "_converted_union"
    converted_root.mkdir(parents=True, exist_ok=True)
    converted_paths: dict[str, Path] = {}

    for file_path in _list_tfrecords(raw_data_dir):
        wanted = requested_by_file.get(str(file_path), set())
        if not wanted:
            continue
        for data in _iter_tfrecord_payloads(file_path):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data)
            scenario_id = str(scenario.scenario_id)
            if scenario_id not in wanted:
                continue
            scenario.scenario_id = scenario.scenario_id + SPLIT_KEY + str(file_path)
            scenario_dict = _convert_waymo_scenario(
                scenario,
                version=version,
                source_file=str(file_path),
            )
            export_file_name = f"sd_{dataset_name}_{version}_{scenario_id}.pkl"
            out_path = converted_root / export_file_name
            with out_path.open("wb") as f:
                pickle.dump(scenario_dict, f)
            converted_paths[scenario_id] = out_path

    missing = sorted(set(requested_ids) - set(converted_paths))
    if missing:
        raise RuntimeError(f"Failed to convert {len(missing)} requested scenarios: {missing[:10]}")

    seed_manifests: dict[str, Any] = {}
    for seed, selected_ids in sorted(seed_to_selected_ids.items()):
        seed_dir = out_root / f"val100_seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, dict[str, Any]] = {}
        mapping: dict[str, str] = {}
        for scenario_id in selected_ids:
            source = converted_paths[scenario_id]
            dest = seed_dir / source.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(Path(os.path.relpath(source, start=seed_dir)))
            with source.open("rb") as f:
                scenario = pickle.load(f)
            metadata = dict(scenario.get("metadata", {}))
            metadata.update(
                {
                    "scenario_id": scenario.get("id", scenario_id),
                    "dataset": metadata.get("dataset", "waymo"),
                    "sample_seed": int(seed),
                    "sample_source": "womd_validation",
                }
            )
            summary[source.name] = metadata
            mapping[source.name] = ""
        with (seed_dir / "dataset_summary.pkl").open("wb") as f:
            pickle.dump(summary, f)
        with (seed_dir / "dataset_mapping.pkl").open("wb") as f:
            pickle.dump(mapping, f)

        ids_path = out_root / f"val100_seed{seed}_scene_ids.txt"
        ids_path.write_text("\n".join(selected_ids) + "\n")
        seed_manifests[str(seed)] = {
            "seed": int(seed),
            "num_scenarios": len(selected_ids),
            "scenario_dir": str(seed_dir),
            "scene_ids_txt": str(ids_path),
            "scene_ids": selected_ids,
        }

    return {
        "raw_data_dir": str(raw_data_dir),
        "out_root": str(out_root),
        "dataset_name": dataset_name,
        "version": version,
        "num_raw_tfrecord_files": len(_list_tfrecords(raw_data_dir)),
        "num_validation_records": len(records),
        "num_converted_union": len(converted_paths),
        "samples": seed_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample fixed WOMD validation val-100 ScenarioNet dirs from raw Waymo validation TFRecords."
    )
    parser.add_argument("--raw-data-dir", required=True, help="Directory containing validation.tfrecord-* files.")
    parser.add_argument("--out-root", required=True, help="Output root for val100_seed* ScenarioNet dirs.")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", action="append", type=int, default=[], help="Random seed. May be repeated.")
    parser.add_argument("--dataset-name", default="waymo")
    parser.add_argument("--version", default="v1.2")
    parser.add_argument("--scan-cache", default=None, help="Optional JSON cache of scanned scenario ids/source files.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    seeds = args.seed or [0, 1, 2]

    if not raw_data_dir.exists():
        raise FileNotFoundError(raw_data_dir)

    scan_cache = Path(args.scan_cache).expanduser().resolve() if args.scan_cache else out_root / "validation_scan_cache.json"
    if scan_cache.exists():
        with scan_cache.open("r") as f:
            records = json.load(f)
    else:
        records = _scan_records(raw_data_dir)
        scan_cache.parent.mkdir(parents=True, exist_ok=True)
        with scan_cache.open("w") as f:
            json.dump(records, f, indent=2)

    if len(records) < int(args.sample_size):
        raise ValueError(f"Need at least {args.sample_size} validation scenarios, found {len(records)}")

    seed_to_selected_ids: dict[int, list[str]] = {}
    population_ids = [str(row["scenario_id"]) for row in records]
    for seed in seeds:
        rng = random.Random(seed)
        seed_to_selected_ids[int(seed)] = rng.sample(population_ids, int(args.sample_size))

    manifest = _write_scenarionet_samples(
        raw_data_dir=raw_data_dir,
        out_root=out_root,
        records=records,
        seed_to_selected_ids=seed_to_selected_ids,
        dataset_name=str(args.dataset_name),
        version=str(args.version),
        overwrite=bool(args.overwrite),
    )
    manifest["scan_cache"] = str(scan_cache)
    manifest_path = out_root / "waymo_val100_samples_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    print(json.dumps(manifest, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
