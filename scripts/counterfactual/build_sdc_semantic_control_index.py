from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.normalize import load_raw_scenario
from bmt.counterfactual.sdc_path_control import split_polyline_on_discontinuities
from bmt.counterfactual.sdc_semantic_control import (
    DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
    DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    DEFAULT_FAMILY_DIVERGENCE_MIN_RUN,
    DEFAULT_FAMILY_DIVERGENCE_THRESHOLD,
    DEFAULT_RESAMPLE_SPACING_M,
    DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    DEFAULT_SEPARABILITY_SCALE_M,
    SDC_SEMANTIC_CONTROL_SCHEMA_VERSION,
    build_resampled_world_path,
    build_world_paths_for_contract,
    compute_family_divergence_profile,
    first_divergence_onset_m,
    iter_highlighted_slots,
    extract_model_frame,
    world_direction_to_model_frame,
    world_xy_to_model_frame,
)
from bmt.counterfactual.waymax_adapter import raw_scenario_from_waymax_state, resolve_waymax_config
from waymax.dataloader import womd_dataloader

DEFAULT_WOD_131_TRAIN_PATH = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic-only SDC control rows with privileged path-family supervision.")
    parser.add_argument("--semantics-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="sdc_semantic_control_index.jsonl")
    parser.add_argument("--path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--config-name", type=str, default="WOD_1_3_1_TRAINING")
    parser.add_argument("--num-paths", type=int, default=45)
    parser.add_argument("--num-points-per-path", type=int, default=800)
    parser.add_argument("--resample-spacing-m", type=float, default=DEFAULT_RESAMPLE_SPACING_M)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument("--separability-heading-weight-m", type=float, default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M)
    parser.add_argument("--divergence-threshold", type=float, default=DEFAULT_FAMILY_DIVERGENCE_THRESHOLD)
    parser.add_argument("--divergence-min-run", type=int, default=DEFAULT_FAMILY_DIVERGENCE_MIN_RUN)
    parser.add_argument("--stitch-discontinuities", action="store_true")
    parser.add_argument("--stitch-radius-m", type=float, default=DEFAULT_DISCONTINUITY_STITCH_RADIUS_M)
    parser.add_argument("--stitch-jump-threshold-m", type=float, default=DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--debug-max-rows", type=int, default=4)
    parser.add_argument("--include-stop", action="store_true")
    parser.add_argument("--stage-vlm-artifacts", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("rt", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(dict(json.loads(text)))
    return rows


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")
    return path


def _find_example_dir(row: Mapping[str, Any]) -> Path:
    for mapping_key in ("prompt_paths", "request_jsons", "images"):
        mapping = dict(row.get(mapping_key, {}) or {})
        for value in mapping.values():
            path = Path(str(value)).expanduser()
            if path.exists():
                return path.parent
    raise FileNotFoundError(f"Could not locate example directory for row: {row.get('example_id')}")


def _find_scenario_pkl(example_dir: Path, *, scenario_id: str) -> Path:
    pkls = sorted(example_dir.glob("*.pkl"))
    if pkls:
        return pkls[0]
    repo_root = Path(__file__).resolve().parents[2]
    outputs_root = repo_root / "outputs"
    pattern = f"sd_waymo_v1.3.1_{scenario_id}.pkl"
    matches = sorted(outputs_root.rglob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No scenario .pkl found in {example_dir} or elsewhere under outputs/ for {scenario_id}")


def _parse_scene_index_from_scenario_id(scenario_id: str) -> int:
    text = str(scenario_id or "").strip()
    if not text.startswith("waymax_scene_"):
        raise ValueError(f"Unsupported scenario_id for Waymax reconstruction: {scenario_id!r}")
    return int(text.rsplit("_", 1)[-1])


def _materialize_missing_waymax_pkls(
    *,
    semantics_rows: Sequence[Mapping[str, Any]],
    outdir: Path,
    path: str,
    config_name: str,
    num_paths: int,
    num_points_per_path: int,
) -> Dict[str, Path]:
    requested: Dict[int, Dict[str, Any]] = {}
    for row in semantics_rows:
        scenario_id = str(row.get("scenario_id") or dict(row.get("contract") or {}).get("scenario_id") or "").strip()
        if not scenario_id:
            continue
        current_time_index = int(row.get("current_time_index") or dict(row.get("contract") or {}).get("current_time_index") or 0)
        try:
            scene_index = _parse_scene_index_from_scenario_id(scenario_id)
        except Exception:
            continue
        requested.setdefault(
            scene_index,
            {
                "scenario_id": scenario_id,
                "current_time_index": current_time_index,
            },
        )
    if not requested:
        return {}

    cache_root = outdir / "_reconstructed_waymax_pkls"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached: Dict[str, Path] = {}
    missing_scene_indices: List[int] = []
    for scene_index, meta in sorted(requested.items()):
        scenario_id = str(meta["scenario_id"])
        pkl_path = cache_root / f"sd_waymo_v1.3.1_{scenario_id}.pkl"
        if pkl_path.is_file():
            cached[scenario_id] = pkl_path
        else:
            missing_scene_indices.append(int(scene_index))
    if not missing_scene_indices:
        return cached

    config = resolve_waymax_config(
        config_name=str(config_name),
        path=str(path),
        include_sdc_paths=True,
        num_paths=int(num_paths),
        num_points_per_path=int(num_points_per_path),
    )
    if dataclasses.is_dataclass(config) and hasattr(config, "num_shards"):
        config = dataclasses.replace(config, num_shards=1, deterministic=True)

    wanted = set(int(idx) for idx in missing_scene_indices)
    max_scene_index = max(wanted)
    scene_iter = womd_dataloader.simulator_state_generator(config=config)
    for local_idx, state in enumerate(scene_iter):
        if local_idx > max_scene_index:
            break
        if local_idx not in wanted:
            continue
        meta = requested[int(local_idx)]
        scenario_id = str(meta["scenario_id"])
        current_time_index = int(meta["current_time_index"])
        raw_scenario = raw_scenario_from_waymax_state(
            state,
            scenario_id=scenario_id,
            current_time_index=current_time_index,
        )
        pkl_path = cache_root / f"sd_waymo_v1.3.1_{scenario_id}.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump(raw_scenario, f)
        cached[scenario_id] = pkl_path
        wanted.remove(int(local_idx))
        if not wanted:
            break
    if wanted:
        missing_ids = [str(requested[idx]["scenario_id"]) for idx in sorted(wanted)]
        raise FileNotFoundError(f"Unable to reconstruct Waymax scenarios for: {missing_ids}")
    return cached


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _stage_vlm_artifacts(example_row: Mapping[str, Any], *, example_dir: Path, outdir: Path) -> Dict[str, Any]:
    example_id = str(example_row.get("example_id") or example_dir.name)
    artifact_root = outdir / "vlm_artifacts" / example_id
    staged: Dict[str, Any] = {
        "example_id": example_id,
        "artifact_root": str(artifact_root),
        "images": {},
        "prompt_paths": {},
        "request_jsons": {},
        "contract_raw_jsons": {},
        "contract_normalized_jsons": {},
        "render_metadata_json": "",
        "all_sdc_paths_grid_png": "",
    }

    for field_name in ("images", "prompt_paths", "request_jsons"):
        mapping = dict(example_row.get(field_name, {}) or {})
        for slot_id, value in mapping.items():
            src = Path(str(value)).expanduser()
            if not src.exists():
                continue
            relative = src.relative_to(example_dir) if example_dir in src.parents else Path(src.name)
            dst = artifact_root / relative
            _copy_file(src, dst)
            staged[field_name][str(slot_id)] = str(dst)

    for src in sorted(example_dir.glob("contract_raw*.json")):
        dst = artifact_root / src.name
        _copy_file(src, dst)
        suffix = src.stem.removeprefix("contract_raw")
        slot_id = suffix.lstrip("_") or "aggregate"
        staged["contract_raw_jsons"][str(slot_id)] = str(dst)

    for src in sorted(example_dir.glob("contract_normalized*.json")):
        dst = artifact_root / src.name
        _copy_file(src, dst)
        suffix = src.stem.removeprefix("contract_normalized")
        slot_id = suffix.lstrip("_") or "aggregate"
        staged["contract_normalized_jsons"][str(slot_id)] = str(dst)

    for name in ("render_metadata.json", "all_sdc_paths_grid.png"):
        src = example_dir / name
        if src.exists():
            dst = artifact_root / src.name
            _copy_file(src, dst)
            if name == "render_metadata.json":
                staged["render_metadata_json"] = str(dst)
            else:
                staged["all_sdc_paths_grid_png"] = str(dst)
    return staged


def _build_dataset_summary_entry(raw_scenario: Mapping[str, Any], *, scenario_pkl_name: str) -> Dict[str, Any]:
    metadata = dict(raw_scenario.get("metadata", {}) or {})
    ts = list(metadata.get("ts", []) or [])
    return {
        "id": str(raw_scenario.get("id") or metadata.get("scenario_id") or scenario_pkl_name),
        "scenario_id": str(raw_scenario.get("id") or metadata.get("scenario_id") or scenario_pkl_name),
        "coordinate": "waymo",
        "dataset": "waymo",
        "source_format": str(metadata.get("source_format") or "waymax_womd"),
        "ts": ts,
        "sdc_id": str(metadata.get("sdc_id") or ""),
        "track_length": int(len(ts)),
        "current_time_index": int(metadata.get("current_time_index") or 0),
        "num_sdc_paths": int(metadata.get("num_sdc_paths") or len(dict(raw_scenario.get("sdc_paths", {})))),
    }


def _default_extent_by_track_type(track_type: str) -> tuple[float, float, float]:
    track_type_name = str(track_type).upper()
    if track_type_name == "PEDESTRIAN":
        return (0.8, 0.8, 1.7)
    if track_type_name == "CYCLIST":
        return (1.8, 0.6, 1.5)
    return (4.5, 1.8, 1.5)


def _normalize_traffic_light_state(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        state_id = int(value)
    except Exception:
        return str(value)
    mapping = {
        -1: "LANE_STATE_UNKNOWN",
        0: "LANE_STATE_UNKNOWN",
        1: "LANE_STATE_ARROW_STOP",
        2: "LANE_STATE_ARROW_CAUTION",
        3: "LANE_STATE_ARROW_GO",
        4: "LANE_STATE_STOP",
        5: "LANE_STATE_CAUTION",
        6: "LANE_STATE_GO",
        7: "LANE_STATE_FLASHING_STOP",
        8: "LANE_STATE_FLASHING_CAUTION",
    }
    return mapping.get(state_id, "LANE_STATE_UNKNOWN")


def _normalize_traffic_light_state_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=object)
    if isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    return np.asarray([_normalize_traffic_light_state(item) for item in items], dtype=object)


def _coerce_training_scenario_arrays(raw_scenario: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(raw_scenario)

    tracks_out: Dict[str, Any] = {}
    for track_id, track in dict(raw.get("tracks", {}) or {}).items():
        track_dict = dict(track)
        state = dict(track_dict.get("state", {}) or {})
        position = np.asarray(state.get("position", []), dtype=np.float32)
        heading = np.asarray(state.get("heading", []), dtype=np.float32).reshape(-1)
        velocity = np.asarray(state.get("velocity", []), dtype=np.float32)
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
        num_steps = int(position.shape[0]) if position.ndim >= 1 else int(heading.shape[0])
        default_length, default_width, default_height = _default_extent_by_track_type(str(track_dict.get("type") or "VEHICLE"))
        length = np.asarray(state.get("length", np.full((num_steps,), default_length, dtype=np.float32)), dtype=np.float32).reshape(-1)
        width = np.asarray(state.get("width", np.full((num_steps,), default_width, dtype=np.float32)), dtype=np.float32).reshape(-1)
        height = np.asarray(state.get("height", np.full((num_steps,), default_height, dtype=np.float32)), dtype=np.float32).reshape(-1)
        if length.size == 0:
            length = np.full((num_steps,), default_length, dtype=np.float32)
        if width.size == 0:
            width = np.full((num_steps,), default_width, dtype=np.float32)
        if height.size == 0:
            height = np.full((num_steps,), default_height, dtype=np.float32)
        state["position"] = position.astype(np.float32)
        state["heading"] = heading.astype(np.float32)
        state["velocity"] = velocity.astype(np.float32)
        state["valid"] = valid.astype(bool)
        state["length"] = length.astype(np.float32)
        state["width"] = width.astype(np.float32)
        state["height"] = height.astype(np.float32)
        track_dict["state"] = state
        tracks_out[str(track_id)] = track_dict
    raw["tracks"] = tracks_out

    map_features_out: Dict[str, Any] = {}
    for feature_id, feature in dict(raw.get("map_features", {}) or {}).items():
        feature_dict = dict(feature)
        for key in ("polyline", "position", "polygon", "location"):
            if key in feature_dict:
                feature_dict[key] = np.asarray(feature_dict[key], dtype=np.float32)
        map_features_out[str(feature_id)] = feature_dict
    raw["map_features"] = map_features_out

    dynamic_map_states_out: Dict[str, Any] = {}
    for light_id, light in dict(raw.get("dynamic_map_states", {}) or {}).items():
        light_dict = dict(light)
        light_dict["type"] = light_dict.get("type") or "TRAFFIC_LIGHT"
        if "stop_point" in light_dict:
            light_dict["stop_point"] = np.asarray(light_dict["stop_point"], dtype=np.float32)
        state = dict(light_dict.get("state", {}) or {})
        if "object_state" in state:
            state["object_state"] = _normalize_traffic_light_state_array(state["object_state"])
        light_dict["state"] = state
        dynamic_map_states_out[str(light_id)] = light_dict
    raw["dynamic_map_states"] = dynamic_map_states_out

    sdc_paths_out: Dict[str, Any] = {}
    for path_id, path in dict(raw.get("sdc_paths", {}) or {}).items():
        path_dict = dict(path)
        if "polyline_xyz" in path_dict:
            path_dict["polyline_xyz"] = np.asarray(path_dict["polyline_xyz"], dtype=np.float32)
        if "valid" in path_dict:
            path_dict["valid"] = np.asarray(path_dict["valid"], dtype=bool)
        sdc_paths_out[str(path_id)] = path_dict
    raw["sdc_paths"] = sdc_paths_out
    return raw


def _augment_raw_scenario_for_training(raw_scenario: Mapping[str, Any]) -> Dict[str, Any]:
    raw = _coerce_training_scenario_arrays(raw_scenario)
    metadata = dict(raw.get("metadata", {}) or {})
    tracks = dict(raw.get("tracks", {}) or {})
    track_ids = [str(track_id) for track_id in tracks.keys()]
    sdc_id = str(metadata.get("sdc_id") or (track_ids[0] if track_ids else "0"))
    ts = np.asarray(metadata.get("ts", []), dtype=np.float32).reshape(-1)
    length = int(raw.get("length") or ts.shape[0] or 0)
    if length <= 0:
        for track in tracks.values():
            state = dict(dict(track).get("state", {}) or {})
            position = np.asarray(state.get("position", []), dtype=np.float32)
            if position.ndim == 2 and position.shape[0] > length:
                length = int(position.shape[0])
    if not track_ids:
        track_index = 0
    else:
        try:
            track_index = track_ids.index(sdc_id)
        except ValueError:
            track_index = 0
    metadata.setdefault("tracks_to_predict", {str(sdc_id): {"track_index": int(track_index)}})
    metadata.setdefault("objects_of_interest", [str(sdc_id)])
    metadata["sdc_id"] = str(sdc_id)
    metadata["current_time_index"] = int(metadata.get("current_time_index") or 0)
    raw["metadata"] = metadata
    raw["length"] = int(length)
    return raw


def _stage_scenario_root(rows: Sequence[Mapping[str, Any]], *, outdir: Path) -> Path:
    scenario_root = outdir / "scenario_root"
    scenario_root.mkdir(parents=True, exist_ok=True)
    dataset_summary: Dict[str, Any] = {}
    dataset_mapping: Dict[str, str] = {}
    seen: Dict[str, Path] = {}
    for row in rows:
        scenario_pkl = Path(str(row["scenario_pkl"])).expanduser()
        if not scenario_pkl.exists():
            continue
        dst = scenario_root / scenario_pkl.name
        if scenario_pkl.name not in seen:
            raw = load_raw_scenario(scenario_pkl)
            staged_raw = _augment_raw_scenario_for_training(raw)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            with dst.open("wb") as f:
                pickle.dump(staged_raw, f)
            dataset_summary[scenario_pkl.name] = _build_dataset_summary_entry(staged_raw, scenario_pkl_name=scenario_pkl.name)
            dataset_mapping[scenario_pkl.name] = "."
            seen[scenario_pkl.name] = scenario_pkl
    with (scenario_root / "dataset_summary.pkl").open("wb") as f:
        pickle.dump(dataset_summary, f)
    with (scenario_root / "dataset_mapping.pkl").open("wb") as f:
        pickle.dump(dataset_mapping, f)
    return scenario_root


def _plot_segmented(ax, points_xy: np.ndarray, *, label: Optional[str] = None, **kwargs) -> None:
    for idx, segment in enumerate(split_polyline_on_discontinuities(points_xy)):
        if segment.shape[0] < 2:
            continue
        ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)


def _plot_family_overlay(
    *,
    out_path: Path,
    row: Mapping[str, Any],
    family_paths: Sequence[Mapping[str, Any]],
    competing_paths: Sequence[Mapping[str, Any]],
    divergence_onsets_m: Sequence[float],
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.set_facecolor("#f8fafc")
    for idx, payload in enumerate(competing_paths):
        xy = np.asarray(payload["waypoints_xy_world"], dtype=np.float32)
        _plot_segmented(ax, xy, color="#cbd5e1", linewidth=1.0, alpha=0.75, label="other-label paths" if idx == 0 else None)
    family_colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#dc2626"]
    for idx, payload in enumerate(family_paths):
        xy = np.asarray(payload["waypoints_xy_world"], dtype=np.float32)
        color = family_colors[idx % len(family_colors)]
        label = f"family path {idx + 1}: {payload.get('slot_id')}"
        _plot_segmented(ax, xy, color=color, linewidth=2.2, alpha=0.95, label=label)
        arc = np.asarray(payload["arc_lengths_m"], dtype=np.float32).reshape(-1)
        onset = float(divergence_onsets_m[idx]) if idx < len(divergence_onsets_m) else float("inf")
        if np.isfinite(onset) and arc.size > 0:
            onset_idx = int(np.argmin(np.abs(arc - onset)))
            onset_idx = int(np.clip(onset_idx, 0, max(0, xy.shape[0] - 1)))
            ax.scatter([xy[onset_idx, 0]], [xy[onset_idx, 1]], s=24, color=color, edgecolors="#111827", linewidths=0.5)
    ax.scatter([0.0], [0.0], s=46, color="#f43f5e", label="SDC current pose")
    ax.arrow(0.0, 0.0, 0.0, 7.0, width=0.16, head_width=0.9, head_length=1.3, color="#111827", length_includes_head=True)
    ax.set_title(
        f"Semantic Family Overlay\n{row['scenario_id']} | slot={dict(row.get('metadata', {}) or {}).get('slot_id')} | "
        f"requested={row['requested_semantic_label']}"
    )
    ax.set_xlabel("Local x (m)")
    ax.set_ylabel("Local y (m, forward)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_family_separability_profile(
    *,
    out_path: Path,
    family_paths: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#dc2626"]
    for idx, (payload, profile) in enumerate(zip(family_paths, profiles)):
        arc = np.asarray(profile.get("arc_lengths_m", []), dtype=np.float32).reshape(-1)
        sep = np.asarray(profile.get("separability", []), dtype=np.float32).reshape(-1)
        if arc.size == 0 or sep.size == 0:
            continue
        ax.plot(arc, sep, color=colors[idx % len(colors)], linewidth=1.9, label=f"{payload.get('slot_id')} ({payload.get('semantic_label')})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Family separability")
    ax.set_title("Semantic-Family Separability vs Other Labels")
    ax.grid(alpha=0.2, linewidth=0.4)
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _build_row(
    *,
    example_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    slot: Mapping[str, Any],
    world_paths: Mapping[str, Any],
    scenario_pkl: Path,
    raw_scenario: Mapping[str, Any],
    scale_m: float,
    heading_weight_m: float,
    divergence_threshold: float,
    divergence_min_run: int,
    debug_root: Optional[Path],
    staged_vlm_artifacts: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    scenario_id = str(contract.get("scenario_id") or example_row.get("scenario_id") or "")
    sdc_id = str(contract.get("sdc_id") or example_row.get("sdc_id") or "")
    current_time_index = int(contract.get("current_time_index") or example_row.get("current_time_index") or 0)
    requested_label = str(slot.get("semantic_label") or "straight")
    requested_confidence = float(slot.get("confidence") or 0.0)
    slots = iter_highlighted_slots(contract, include_stop=True)
    family_slots = [entry for entry in slots if str(entry.get("semantic_label")) == requested_label and bool(entry.get("is_valid_target", True))]
    if not family_slots:
        family_slots = [dict(slot)]
    family_paths = [world_paths[str(entry["slot_id"])] for entry in family_slots if str(entry.get("slot_id")) in world_paths]
    competing_slots = [entry for entry in slots if str(entry.get("semantic_label")) != requested_label and bool(entry.get("is_valid_target", True))]
    competing_paths = [world_paths[str(entry["slot_id"])] for entry in competing_slots if str(entry.get("slot_id")) in world_paths]

    profiles = []
    divergence_onsets = []
    family_payloads = []
    for family_path in family_paths:
        profile = compute_family_divergence_profile(
            family_path=family_path,
            competing_other_label_paths=competing_paths,
            scale_m=float(scale_m),
            heading_weight_m=float(heading_weight_m),
        )
        arc = np.asarray(family_path.arc_lengths_m, dtype=np.float32).reshape(-1)
        sep = np.asarray(profile.get("separability", []), dtype=np.float32).reshape(-1)
        onset = first_divergence_onset_m(
            arc,
            sep,
            threshold=float(divergence_threshold),
            min_run=int(divergence_min_run),
        )
        divergence_onsets.append(float(onset))
        profiles.append(
            {
                "slot_id": family_path.slot_id,
                "path_id": family_path.path_id,
                "arc_lengths_m": arc.tolist(),
                "separability": sep.tolist(),
                "min_distance_m": np.asarray(profile.get("min_distance_m", []), dtype=np.float32).reshape(-1).tolist(),
                "heading_delta_rad": np.asarray(profile.get("heading_delta_rad", []), dtype=np.float32).reshape(-1).tolist(),
                "nearest_competing_path_id": list(profile.get("nearest_competing_path_id", [])),
                "divergence_onset_m": float(onset),
            }
        )
        family_payloads.append(
            {
                "slot_id": family_path.slot_id,
                "path_id": family_path.path_id,
                "semantic_label": family_path.semantic_label,
                "confidence": float(family_path.confidence),
                "waypoints_xy_world": np.asarray(family_path.waypoints_xy_world, dtype=np.float32),
                "tangents_world": np.asarray(family_path.tangents_world, dtype=np.float32),
                "arc_lengths_m": np.asarray(family_path.arc_lengths_m, dtype=np.float32),
            }
        )

    metadata = {
        "example_id": str(example_row.get("example_id") or ""),
        "slot_id": str(slot.get("slot_id") or ""),
        "selected_path_id": None if slot.get("path_id") is None else str(slot.get("path_id")),
        "family_slot_ids": [str(path.slot_id) for path in family_paths],
        "other_label_slot_ids": [str(path.slot_id) for path in competing_paths],
        "other_label_semantics": [str(path.semantic_label) for path in competing_paths],
        "family_size": int(len(family_paths)),
        "other_label_count": int(len(competing_paths)),
    }
    if staged_vlm_artifacts is not None:
        slot_id = str(slot.get("slot_id") or "")
        metadata.update(
            {
                "vlm_image_png": dict(staged_vlm_artifacts.get("images", {})).get(slot_id),
                "vlm_prompt_txt": dict(staged_vlm_artifacts.get("prompt_paths", {})).get(slot_id),
                "vlm_request_json": dict(staged_vlm_artifacts.get("request_jsons", {})).get(slot_id),
                "vlm_contract_raw_json": dict(staged_vlm_artifacts.get("contract_raw_jsons", {})).get(slot_id),
                "vlm_contract_normalized_json": dict(staged_vlm_artifacts.get("contract_normalized_jsons", {})).get(slot_id),
                "vlm_render_metadata_json": staged_vlm_artifacts.get("render_metadata_json"),
                "vlm_all_sdc_paths_grid_png": staged_vlm_artifacts.get("all_sdc_paths_grid_png"),
            }
        )

    row = {
        "schema_version": SDC_SEMANTIC_CONTROL_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_pkl": str(scenario_pkl.resolve()),
        "sdc_id": sdc_id,
        "current_time_index": int(current_time_index),
        "requested_semantic_label": requested_label,
        "requested_semantic_confidence": float(requested_confidence),
        "use_for_training": bool(contract.get("use_for_training", True)) and bool(slot.get("use_for_training", True)),
        "source_kind": str(slot.get("row_source_kind") or "alternative_sdc_path"),
        "selected_slot_id": str(slot.get("slot_id") or ""),
        "selected_path_id": None if slot.get("path_id") is None else str(slot.get("path_id")),
        "candidate_family_path_ids": [None if path.path_id is None else str(path.path_id) for path in family_paths],
        "candidate_family_slot_ids": [str(path.slot_id) for path in family_paths],
        "candidate_family_confidences": [float(path.confidence) for path in family_paths],
        "candidate_family_resampled_paths_world": [np.asarray(path.waypoints_xy_world, dtype=np.float32).tolist() for path in family_paths],
        "candidate_family_resampled_path_tangents_world": [np.asarray(path.tangents_world, dtype=np.float32).tolist() for path in family_paths],
        "candidate_family_arc_lengths_m": [np.asarray(path.arc_lengths_m, dtype=np.float32).tolist() for path in family_paths],
        "candidate_family_divergence_onsets_m": [float(value) for value in divergence_onsets],
        "metadata": metadata,
    }
    map_center, map_heading = extract_model_frame(raw_scenario)
    row["candidate_family_frame"] = "model_map_centered"
    row["candidate_family_map_center"] = np.asarray(map_center, dtype=np.float32).tolist()
    row["candidate_family_map_heading"] = float(map_heading)
    row["candidate_family_resampled_paths_model"] = [
        world_xy_to_model_frame(path.waypoints_xy_world, map_center=map_center, map_heading=map_heading).tolist()
        for path in family_paths
    ]
    row["candidate_family_resampled_path_tangents_model"] = [
        world_direction_to_model_frame(path.tangents_world, map_heading=map_heading).tolist()
        for path in family_paths
    ]

    if debug_root is not None:
        local_family_paths = []
        local_competing_paths = []
        current_xy_world, current_heading_world = load_raw_scenario(scenario_pkl)["tracks"][str(sdc_id)]["state"]["position"][int(current_time_index)][:2], float(load_raw_scenario(scenario_pkl)["tracks"][str(sdc_id)]["state"]["heading"][int(current_time_index)])
        current_xy_world = np.asarray(current_xy_world, dtype=np.float32)
        rot = (np.pi / 2.0) - float(current_heading_world)
        c = float(np.cos(rot))
        s = float(np.sin(rot))

        def _to_local(xy_world: np.ndarray) -> np.ndarray:
            centered = np.asarray(xy_world, dtype=np.float32) - current_xy_world.reshape(1, 2)
            x_new = c * centered[:, 0] - s * centered[:, 1]
            y_new = s * centered[:, 0] + c * centered[:, 1]
            return np.stack([x_new, y_new], axis=-1).astype(np.float32)

        for payload in family_payloads:
            local_family_paths.append(
                {
                    **payload,
                    "waypoints_xy_world": _to_local(np.asarray(payload["waypoints_xy_world"], dtype=np.float32)),
                }
            )
        for path in competing_paths:
            local_competing_paths.append(
                {
                    "slot_id": path.slot_id,
                    "path_id": path.path_id,
                    "semantic_label": path.semantic_label,
                    "waypoints_xy_world": _to_local(np.asarray(path.waypoints_xy_world, dtype=np.float32)),
                }
            )
        overlay_path = debug_root / "semantic_family_overlay.png"
        profile_path = debug_root / "family_separability_profile.png"
        debug_json_path = debug_root / "semantic_family_debug.json"
        _plot_family_overlay(
            out_path=overlay_path,
            row=row,
            family_paths=local_family_paths,
            competing_paths=local_competing_paths,
            divergence_onsets_m=divergence_onsets,
        )
        _plot_family_separability_profile(
            out_path=profile_path,
            family_paths=family_payloads,
            profiles=profiles,
        )
        _write_json(
            debug_json_path,
            {
                "scenario_id": scenario_id,
                "selected_slot_id": str(slot.get("slot_id") or ""),
                "requested_semantic_label": requested_label,
                "family_profiles": profiles,
            },
        )
        row["debug_artifacts"] = {
            "semantic_family_overlay_png": str(overlay_path),
            "family_separability_profile_png": str(profile_path),
            "semantic_family_debug_json": str(debug_json_path),
        }
    return row


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    semantics_rows = _read_jsonl(Path(args.semantics_index).expanduser())
    if int(args.max_examples) > 0:
        semantics_rows = semantics_rows[: int(args.max_examples)]

    built_rows: List[Dict[str, Any]] = []
    debug_records: List[Dict[str, Any]] = []
    family_group_audit: List[Dict[str, Any]] = []
    staged_manifests: List[Dict[str, Any]] = []
    reconstructed_pkls = _materialize_missing_waymax_pkls(
        semantics_rows=semantics_rows,
        outdir=outdir,
        path=str(args.path),
        config_name=str(args.config_name),
        num_paths=int(args.num_paths),
        num_points_per_path=int(args.num_points_per_path),
    )

    for example_idx, example_row in enumerate(semantics_rows):
        example_dir = _find_example_dir(example_row)
        contract = dict(example_row.get("contract") or {})
        if not contract:
            contract_path = example_dir / "contract_normalized.json"
            contract = dict(json.loads(contract_path.read_text(encoding="utf-8")))
        scenario_id = str(contract.get("scenario_id") or example_row.get("scenario_id") or "")
        try:
            scenario_pkl = _find_scenario_pkl(example_dir, scenario_id=scenario_id)
        except FileNotFoundError:
            scenario_pkl = reconstructed_pkls.get(str(scenario_id), None)
            if scenario_pkl is None:
                raise
        raw_scenario = load_raw_scenario(scenario_pkl)
        sdc_id = str(contract.get("sdc_id") or example_row.get("sdc_id") or "")
        current_time_index = int(contract.get("current_time_index") or example_row.get("current_time_index") or 0)
        world_paths = build_world_paths_for_contract(
            raw_scenario=raw_scenario,
            contract=contract,
            sdc_id=str(sdc_id),
            current_time_index=int(current_time_index),
            spacing_m=float(args.resample_spacing_m),
            include_stop=True,
            stitch_discontinuities=bool(args.stitch_discontinuities),
            stitch_radius_m=float(args.stitch_radius_m),
            stitch_jump_threshold_m=float(args.stitch_jump_threshold_m),
        )
        staged_vlm_artifacts = None
        if bool(args.stage_vlm_artifacts):
            staged_vlm_artifacts = _stage_vlm_artifacts(example_row, example_dir=example_dir, outdir=outdir)
            staged_manifests.append(dict(staged_vlm_artifacts))

        for slot in iter_highlighted_slots(contract, include_stop=True):
            debug_root = None
            if len(debug_records) < int(args.debug_max_rows):
                debug_root = outdir / "debug" / str(example_row.get("example_id") or example_dir.name) / str(slot.get("slot_id") or "")
            row = _build_row(
                example_row=example_row,
                contract=contract,
                slot=slot,
                world_paths=world_paths,
                scenario_pkl=scenario_pkl,
                raw_scenario=raw_scenario,
                scale_m=float(args.separability_scale_m),
                heading_weight_m=float(args.separability_heading_weight_m),
                divergence_threshold=float(args.divergence_threshold),
                divergence_min_run=int(args.divergence_min_run),
                debug_root=debug_root,
                staged_vlm_artifacts=staged_vlm_artifacts,
            )
            built_rows.append(row)
            family_group_audit.append(
                {
                    "example_id": str(example_row.get("example_id") or example_dir.name),
                    "scenario_id": scenario_id,
                    "slot_id": str(slot.get("slot_id") or ""),
                    "requested_semantic_label": str(row["requested_semantic_label"]),
                    "family_size": int(len(list(row.get("candidate_family_path_ids", []) or []))),
                    "source_kind": str(row["source_kind"]),
                    "use_for_training": bool(row["use_for_training"]),
                }
            )
            if debug_root is not None and "debug_artifacts" in row:
                debug_records.append(
                    {
                        "example_id": str(example_row.get("example_id") or example_dir.name),
                        "scenario_id": scenario_id,
                        "slot_id": str(slot.get("slot_id") or ""),
                        **dict(row["debug_artifacts"]),
                    }
                )

    scenario_root = _stage_scenario_root(built_rows, outdir=outdir)
    output_path = _write_jsonl(outdir / args.output_name, built_rows)
    _write_json(outdir / "family_group_audit.json", family_group_audit)
    _write_json(outdir / "debug_manifest.json", debug_records)
    if staged_manifests:
        _write_json(outdir / "vlm_artifact_manifest.json", staged_manifests)
    summary = {
        "schema_version": SDC_SEMANTIC_CONTROL_SCHEMA_VERSION,
        "num_examples": int(len(semantics_rows)),
        "num_rows": int(len(built_rows)),
        "num_factual_rows": int(sum(1 for row in built_rows if str(row.get("source_kind")) == "factual_gt")),
        "num_alternative_rows": int(sum(1 for row in built_rows if str(row.get("source_kind")) == "alternative_sdc_path")),
        "num_use_for_training_rows": int(sum(1 for row in built_rows if bool(row.get("use_for_training", True)))),
        "output_jsonl": str(output_path),
        "scenario_root": str(scenario_root),
        "debug_manifest": str(outdir / "debug_manifest.json"),
        "family_group_audit": str(outdir / "family_group_audit.json"),
    }
    if staged_manifests:
        summary["vlm_artifact_manifest"] = str(outdir / "vlm_artifact_manifest.json")
    _write_json(outdir / "build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
