from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
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

from bmt.counterfactual.sdc_path_control import (
    DEFAULT_RESAMPLE_SPACING_M,
    DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    DEFAULT_SEPARABILITY_SCALE_M,
    SDC_PATH_CONTROL_SCHEMA_VERSION,
    build_local_competing_paths,
    build_local_selected_path,
    compute_path_separability_profile,
    is_sdc_path_control_row,
    list_on_route_candidate_path_ids,
    load_raw_scenario_from_row,
    normalize_semantic_label,
    polyline_length_m,
    split_polyline_on_discontinuities,
)
from bmt.counterfactual.normalize import load_raw_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical SDC-path control rows from VLM-labeled Waymax SDC path artifacts.")
    parser.add_argument("--semantics-index", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="sdc_path_control_index.jsonl")
    parser.add_argument("--resample-spacing-m", type=float, default=DEFAULT_RESAMPLE_SPACING_M)
    parser.add_argument("--separability-scale-m", type=float, default=DEFAULT_SEPARABILITY_SCALE_M)
    parser.add_argument("--separability-heading-weight-m", type=float, default=DEFAULT_SEPARABILITY_HEADING_WEIGHT_M)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--debug-max-rows", type=int, default=4)
    parser.add_argument("--include-stop", action="store_true")
    parser.add_argument("--max-alternative-paths-per-example", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=0)
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


def _find_scenario_pkl(example_dir: Path) -> Path:
    pkls = sorted(example_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No scenario .pkl found in {example_dir}")
    return pkls[0]


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

    render_metadata = example_dir / "render_metadata.json"
    if render_metadata.exists():
        dst = artifact_root / render_metadata.name
        _copy_file(render_metadata, dst)
        staged["render_metadata_json"] = str(dst)

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


def _augment_raw_scenario_for_training(raw_scenario: Mapping[str, Any]) -> Dict[str, Any]:
    raw = dict(raw_scenario)
    metadata = dict(raw.get("metadata", {}) or {})
    tracks = dict(raw.get("tracks", {}) or {})
    map_features = dict(raw.get("map_features", {}) or {})
    dynamic_map_states = dict(raw.get("dynamic_map_states", {}) or {})
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

    for track_id, track in tracks.items():
        track_dict = dict(track or {})
        state = dict(track_dict.get("state", {}) or {})
        track_type = str(track_dict.get("type") or dict(track_dict.get("metadata", {}) or {}).get("type") or "VEHICLE").upper()
        if "PEDESTRIAN" in track_type:
            defaults = (0.8, 0.8, 1.7)
        elif "CYCLIST" in track_type or "BICYCLE" in track_type:
            defaults = (1.8, 0.7, 1.6)
        else:
            defaults = (4.8, 1.8, 1.6)
        if "position" in state:
            state["position"] = np.asarray(state["position"], dtype=np.float32)
        if "heading" in state:
            state["heading"] = np.asarray(state["heading"], dtype=np.float32)
        if "velocity" in state:
            state["velocity"] = np.asarray(state["velocity"], dtype=np.float32)
        if "valid" in state:
            state["valid"] = np.asarray(state["valid"], dtype=bool)
        time_len = int(np.asarray(state.get("heading", [])).reshape(-1).shape[0] or length)
        state.setdefault("length", np.full((time_len,), defaults[0], dtype=np.float32))
        state.setdefault("width", np.full((time_len,), defaults[1], dtype=np.float32))
        state.setdefault("height", np.full((time_len,), defaults[2], dtype=np.float32))
        state["length"] = np.asarray(state["length"], dtype=np.float32)
        state["width"] = np.asarray(state["width"], dtype=np.float32)
        state["height"] = np.asarray(state["height"], dtype=np.float32)
        track_dict["state"] = state
        tracks[str(track_id)] = track_dict
    raw["tracks"] = tracks

    for feature_id, feature in map_features.items():
        feature_dict = dict(feature or {})
        if "polyline" in feature_dict:
            feature_dict["polyline"] = np.asarray(feature_dict["polyline"], dtype=np.float32)
        if "polygon" in feature_dict:
            feature_dict["polygon"] = np.asarray(feature_dict["polygon"], dtype=np.float32)
        map_features[str(feature_id)] = feature_dict
    raw["map_features"] = map_features

    for light_id, light in dynamic_map_states.items():
        light_dict = dict(light or {})
        state = dict(light_dict.get("state", {}) or {})
        if "object_state" in state:
            state["object_state"] = np.asarray(state["object_state"], dtype=np.int64)
        light_dict["state"] = state
        if "stop_point" in light_dict:
            light_dict["stop_point"] = np.asarray(light_dict["stop_point"], dtype=np.float32)
        dynamic_map_states[str(light_id)] = light_dict
    raw["dynamic_map_states"] = dynamic_map_states
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


def _plot_selected_path_overlay(
    *,
    out_path: Path,
    selected_xy: np.ndarray,
    competing_paths: Mapping[str, Any],
    separability: np.ndarray,
    semantic_label: str,
    source_kind: str,
) -> Path:
    def _plot_segmented(ax, points_xy: np.ndarray, *, label: Optional[str] = None, **kwargs) -> None:
        segments = split_polyline_on_discontinuities(points_xy)
        for idx, segment in enumerate(segments):
            ax.plot(segment[:, 0], segment[:, 1], label=label if idx == 0 else None, **kwargs)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.set_facecolor("#f8fafc")
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0.0, vmax=1.0)
    for idx, (path_id, path) in enumerate(competing_paths.items()):
        xy = np.asarray(path.waypoints_xy, dtype=np.float32)
        if xy.shape[0] < 2:
            continue
        _plot_segmented(
            ax,
            xy,
            color="#cbd5e1",
            linewidth=1.2,
            alpha=0.9,
            label="competing on-route paths" if idx == 0 else None,
        )
    if selected_xy.shape[0] >= 2:
        pts = np.asarray(selected_xy, dtype=np.float32)
        sep = np.asarray(separability, dtype=np.float32).reshape(-1)
        _plot_segmented(
            ax,
            pts,
            color="#0f172a",
            linewidth=4.2,
            alpha=0.15,
        )
        for idx in range(1, int(pts.shape[0])):
            color = cmap(norm(float(np.clip(sep[min(idx, sep.shape[0] - 1)], 0.0, 1.0))))
            ax.plot(pts[idx - 1 : idx + 1, 0], pts[idx - 1 : idx + 1, 1], color=color, linewidth=3.0)
    ax.scatter([0.0], [0.0], s=50, c="#f43f5e", marker="o", label="SDC current pose")
    ax.arrow(0.0, 0.0, 0.0, 7.5, width=0.18, head_width=1.0, head_length=1.5, color="#111827", length_includes_head=True)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlabel("Local x (m)")
    ax.set_ylabel("Local y (m, forward)")
    ax.set_title(f"Selected SDC Path Overlay\n{source_kind} | semantic={semantic_label} | path colored by separability")
    legend_handles = [
        Line2D([0], [0], color="#cbd5e1", linewidth=1.6, label="competing on-route paths"),
        Line2D([0], [0], color="#334155", linewidth=3.0, label="selected path (see colorbar)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f43f5e", markeredgecolor="#f43f5e", markersize=6, label="SDC current pose"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Separability / soft split strength", rotation=90)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["shared", "distinct"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_separability_profile(*, out_path: Path, arc_lengths_m: np.ndarray, separability: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3), dpi=180)
    arc = np.asarray(arc_lengths_m, dtype=np.float32).reshape(-1)
    sep = np.asarray(separability, dtype=np.float32).reshape(-1)
    if arc.size > 0 and sep.size > 0:
        ax.plot(arc, sep, color="#2563eb", linewidth=2.0)
        ax.fill_between(arc, 0.0, sep, color="#93c5fd", alpha=0.35)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Arc length (m)")
    ax.set_ylabel("Separability")
    ax.set_title("Selected Path Separability Profile")
    ax.grid(alpha=0.25, linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _slot_record_lookup(example_contract: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {
        str(slot.get("slot_id")): dict(slot)
        for slot in list(example_contract.get("highlighted_paths", []) or [])
        if slot.get("slot_id") is not None
    }


def _select_slot_contracts(
    example_contract: Mapping[str, Any],
    *,
    max_alternative_paths: int,
    rng: random.Random,
) -> List[Mapping[str, Any]]:
    slots = [dict(slot) for slot in list(example_contract.get("highlighted_paths", []) or [])]
    factual_slots = [
        slot for slot in slots if str(slot.get("source_kind")) == "ground_truth" or str(slot.get("slot_id")) == "gt"
    ]
    alternative_slots = [
        slot for slot in slots if not (str(slot.get("source_kind")) == "ground_truth" or str(slot.get("slot_id")) == "gt")
    ]
    alternative_slots = sorted(alternative_slots, key=lambda slot: str(slot.get("slot_id") or slot.get("path_id") or ""))
    if max_alternative_paths >= 0 and len(alternative_slots) > int(max_alternative_paths):
        keep_indices = sorted(rng.sample(list(range(len(alternative_slots))), int(max_alternative_paths)))
        alternative_slots = [alternative_slots[idx] for idx in keep_indices]
    return factual_slots + alternative_slots


def _build_row(
    *,
    example_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    slot_contract: Mapping[str, Any],
    scenario_pkl: Path,
    raw_scenario: Mapping[str, Any],
    spacing_m: float,
    separability_scale_m: float,
    separability_heading_weight_m: float,
    include_stop: bool,
    debug_root: Optional[Path],
    staged_vlm_artifacts: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    scenario_id = str(contract.get("scenario_id") or example_row.get("scenario_id") or "")
    sdc_id = str(contract.get("sdc_id") or example_row.get("sdc_id") or "")
    current_time_index = int(contract.get("current_time_index") or example_row.get("current_time_index") or 0)
    semantic_label = normalize_semantic_label(slot_contract.get("semantic_label"))
    source_kind = "factual_gt" if str(slot_contract.get("source_kind")) == "ground_truth" else "alternative_sdc_path"
    selected_path_id = "gt" if source_kind == "factual_gt" else str(slot_contract.get("path_id") or "")
    selected_local = build_local_selected_path(
        raw_scenario=raw_scenario,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
        source_kind=source_kind,
        selected_path_id=None if selected_path_id == "gt" else selected_path_id,
        spacing_m=spacing_m,
    )
    competing = build_local_competing_paths(
        raw_scenario=raw_scenario,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
        selected_path_id=None if selected_path_id == "gt" else selected_path_id,
        spacing_m=spacing_m,
    )
    separability_debug = compute_path_separability_profile(
        selected_local,
        competing,
        scale_m=separability_scale_m,
        heading_weight_m=separability_heading_weight_m,
    )
    use_for_training = bool(contract.get("use_for_training", True)) and bool(slot_contract.get("is_valid_target", True))
    if semantic_label == "stop" and not include_stop and source_kind != "factual_gt":
        use_for_training = False

    row: Dict[str, Any] = {
        "schema_version": SDC_PATH_CONTROL_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "scenario_pkl": str(scenario_pkl.resolve()),
        "sdc_id": sdc_id,
        "current_time_index": current_time_index,
        "selected_path_id": selected_path_id,
        "semantic_label": semantic_label,
        "semantic_confidence": float(slot_contract.get("confidence") or 0.0),
        "use_for_training": bool(use_for_training),
        "source_kind": source_kind,
        "selected_path_waypoints_local_xy": np.asarray(selected_local.waypoints_xy, dtype=np.float32).tolist(),
        "selected_path_waypoints_local_heading": np.asarray(selected_local.headings, dtype=np.float32).tolist(),
        "selected_path_arc_lengths_m": np.asarray(selected_local.arc_lengths_m, dtype=np.float32).tolist(),
        "selected_path_separability": np.asarray(separability_debug["separability"], dtype=np.float32).tolist(),
        "candidate_path_ids": list_on_route_candidate_path_ids(raw_scenario),
        "candidate_count": int(len(list_on_route_candidate_path_ids(raw_scenario))),
        "metadata": {
            "example_id": str(contract.get("example_id") or example_row.get("example_id") or ""),
            "slot_id": str(slot_contract.get("slot_id") or ""),
            "model_name": str(contract.get("model_name") or ""),
            "scene_ambiguity_level": str(dict(contract.get("scene_ambiguity", {}) or {}).get("level") or ""),
            "selected_path_length_m": float(polyline_length_m(selected_local.waypoints_xy)),
            "selected_path_point_count": int(np.asarray(selected_local.waypoints_xy).shape[0]),
            "min_competing_distance_m": np.asarray(separability_debug["min_distance_m"], dtype=np.float32).tolist(),
            "nearest_competing_path_id": list(separability_debug["nearest_competing_path_id"]),
        },
    }
    if staged_vlm_artifacts is not None:
        slot_id = str(slot_contract.get("slot_id") or "")
        row["metadata"]["vlm_artifact_root"] = str(staged_vlm_artifacts.get("artifact_root") or "")
        row["metadata"]["vlm_image_png"] = str(dict(staged_vlm_artifacts.get("images", {}) or {}).get(slot_id) or "")
        row["metadata"]["vlm_prompt_txt"] = str(dict(staged_vlm_artifacts.get("prompt_paths", {}) or {}).get(slot_id) or "")
        row["metadata"]["vlm_request_json"] = str(dict(staged_vlm_artifacts.get("request_jsons", {}) or {}).get(slot_id) or "")
        row["metadata"]["vlm_contract_raw_json"] = str(dict(staged_vlm_artifacts.get("contract_raw_jsons", {}) or {}).get(slot_id) or "")
        row["metadata"]["vlm_contract_normalized_json"] = str(
            dict(staged_vlm_artifacts.get("contract_normalized_jsons", {}) or {}).get(slot_id) or ""
        )
        row["metadata"]["vlm_render_metadata_json"] = str(staged_vlm_artifacts.get("render_metadata_json") or "")
    if debug_root is not None:
        row_dir = debug_root / str(contract.get("example_id") or row["selected_path_id"]).replace("/", "_") / str(slot_contract.get("slot_id") or "slot")
        overlay_path = _plot_selected_path_overlay(
            out_path=row_dir / "selected_path_overlay.png",
            selected_xy=np.asarray(selected_local.waypoints_xy, dtype=np.float32),
            competing_paths=competing,
            separability=np.asarray(separability_debug["separability"], dtype=np.float32),
            semantic_label=semantic_label,
            source_kind=source_kind,
        )
        profile_path = _plot_separability_profile(
            out_path=row_dir / "separability_profile_plot.png",
            arc_lengths_m=np.asarray(selected_local.arc_lengths_m, dtype=np.float32),
            separability=np.asarray(separability_debug["separability"], dtype=np.float32),
        )
        debug_json_path = _write_json(
            row_dir / "path_separability_debug.json",
            {
                "schema_version": SDC_PATH_CONTROL_SCHEMA_VERSION,
                "selected_path_id": selected_path_id,
                "semantic_label": semantic_label,
                "source_kind": source_kind,
                "selected_path_waypoints_local_xy": row["selected_path_waypoints_local_xy"],
                "selected_path_waypoints_local_heading": row["selected_path_waypoints_local_heading"],
                "selected_path_arc_lengths_m": row["selected_path_arc_lengths_m"],
                "selected_path_separability": row["selected_path_separability"],
                "min_competing_distance_m": row["metadata"]["min_competing_distance_m"],
                "nearest_competing_path_id": row["metadata"]["nearest_competing_path_id"],
            },
        )
        row["metadata"]["selected_path_overlay_png"] = str(overlay_path)
        row["metadata"]["separability_profile_plot_png"] = str(profile_path)
        row["metadata"]["path_separability_debug_json"] = str(debug_json_path)
    return row


def main() -> int:
    args = parse_args()
    semantics_index_path = Path(args.semantics_index).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(semantics_index_path)
    if args.max_examples and int(args.max_examples) > 0:
        rows = rows[: int(args.max_examples)]
    rng = random.Random(int(args.random_seed))

    built_rows: List[Dict[str, Any]] = []
    vlm_artifact_manifest: List[Dict[str, Any]] = []
    debug_budget = int(args.debug_max_rows)
    for example_row in rows:
        contract = dict(example_row.get("contract", {}) or {})
        example_dir = _find_example_dir(example_row)
        scenario_pkl = _find_scenario_pkl(example_dir)
        raw_scenario = load_raw_scenario(scenario_pkl)
        staged_vlm_artifacts = None
        if bool(args.stage_vlm_artifacts):
            staged_vlm_artifacts = _stage_vlm_artifacts(example_row, example_dir=example_dir, outdir=outdir)
            vlm_artifact_manifest.append(dict(staged_vlm_artifacts))
        for slot_contract in _select_slot_contracts(
            contract,
            max_alternative_paths=int(args.max_alternative_paths_per_example),
            rng=rng,
        ):
            debug_root = outdir / "debug" if len(built_rows) < debug_budget else None
            built_rows.append(
                _build_row(
                    example_row=example_row,
                    contract=contract,
                    slot_contract=slot_contract,
                    scenario_pkl=scenario_pkl,
                    raw_scenario=raw_scenario,
                    spacing_m=float(args.resample_spacing_m),
                    separability_scale_m=float(args.separability_scale_m),
                    separability_heading_weight_m=float(args.separability_heading_weight_m),
                    include_stop=bool(args.include_stop),
                    debug_root=debug_root,
                    staged_vlm_artifacts=staged_vlm_artifacts,
                )
            )

    output_index = outdir / str(args.output_name)
    _write_jsonl(output_index, built_rows)
    scenario_root = _stage_scenario_root(built_rows, outdir=outdir)
    split_audit = {
        "schema_version": SDC_PATH_CONTROL_SCHEMA_VERSION,
        "num_rows": int(len(built_rows)),
        "num_factual_rows": int(sum(1 for row in built_rows if str(row.get("source_kind")) == "factual_gt")),
        "num_alternative_rows": int(sum(1 for row in built_rows if str(row.get("source_kind")) == "alternative_sdc_path")),
        "num_use_for_training_true": int(sum(1 for row in built_rows if bool(row.get("use_for_training")))),
    }
    gt_match_audit = {
        "schema_version": SDC_PATH_CONTROL_SCHEMA_VERSION,
        "num_rows": int(len(built_rows)),
        "semantic_label_histogram": {
            label: int(sum(1 for row in built_rows if str(row.get("semantic_label")) == label))
            for label in sorted({str(row.get("semantic_label")) for row in built_rows})
        },
    }
    _write_json(outdir / "split_point_audit.json", split_audit)
    _write_json(outdir / "gt_branch_match_audit.json", gt_match_audit)
    vlm_artifact_manifest_path = outdir / "vlm_artifact_manifest.json"
    if bool(args.stage_vlm_artifacts):
        _write_json(vlm_artifact_manifest_path, vlm_artifact_manifest)
    _write_json(
        outdir / "build_summary.json",
        {
            "semantics_index": str(semantics_index_path),
            "output_index": str(output_index),
            "scenario_root": str(scenario_root),
            "num_examples_input": int(len(rows)),
            "num_rows_output": int(len(built_rows)),
            "include_stop": bool(args.include_stop),
            "max_alternative_paths_per_example": int(args.max_alternative_paths_per_example),
            "random_seed": int(args.random_seed),
            "stage_vlm_artifacts": bool(args.stage_vlm_artifacts),
            "vlm_artifact_manifest": str(vlm_artifact_manifest_path) if bool(args.stage_vlm_artifacts) else "",
            "resample_spacing_m": float(args.resample_spacing_m),
            "separability_scale_m": float(args.separability_scale_m),
            "separability_heading_weight_m": float(args.separability_heading_weight_m),
        },
    )
    print(json.dumps({"output_index": str(output_index), "num_rows": int(len(built_rows)), "scenario_root": str(scenario_root)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
