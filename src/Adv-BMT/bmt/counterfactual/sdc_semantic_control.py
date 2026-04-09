from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .normalize import load_raw_scenario
from .sdc_path_control import (
    DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
    DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    DEFAULT_RESAMPLE_SPACING_M,
    DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
    DEFAULT_SEPARABILITY_SCALE_M,
    SDC_PATH_SEMANTIC_LABEL_ORDER,
    ResampledLocalPath,
    _extract_valid_sdc_path_xy,
    build_selected_path_world,
    compute_path_separability_profile,
    extract_ground_truth_sdc_route_xy,
    extract_sdc_current_pose,
    heading_from_points,
    normalize_semantic_label,
    polyline_arc_lengths,
    polyline_headings,
    polyline_segment_valid_mask,
    resample_polyline_xy,
    semantic_label_to_id,
    split_polyline_on_discontinuities,
    stitch_polyline_discontinuities,
    trim_polyline_from_point,
)

SDC_SEMANTIC_CONTROL_SCHEMA_VERSION = "sdc_semantic_control_v1"
DEFAULT_FAMILY_DIVERGENCE_THRESHOLD = 0.25
DEFAULT_FAMILY_DIVERGENCE_MIN_RUN = 2
DEFAULT_FAMILY_GUIDE_BANDWIDTH_M = 6.0
DEFAULT_FAMILY_PATH_DEADBAND_M = 1.0
DEFAULT_FAMILY_HEADING_DEADBAND_RAD = 0.25
DEFAULT_FAMILY_HEADING_BETA_RAD = 0.35
DEFAULT_FAMILY_BACKWARD_SLACK_M = 0.25
DEFAULT_FAMILY_TEACHER_TEMPERATURE = 1.0
DEFAULT_FAMILY_POSITION_WEIGHT = 1.0
DEFAULT_FAMILY_HEADING_WEIGHT = 0.75
DEFAULT_FAMILY_BACKWARD_WEIGHT = 0.5


def _iter_map_feature_locations(raw_scenario: Mapping[str, Any]) -> Iterable[np.ndarray]:
    for feature in dict(raw_scenario.get("map_features", {}) or {}).values():
        payload = dict(feature or {})
        for key in ("polyline", "polygon", "position"):
            value = payload.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] >= 2 and arr.shape[0] > 0:
                yield arr
                break


def extract_model_frame(raw_scenario: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    metadata = dict(raw_scenario.get("metadata", {}) or {})
    map_center_value = metadata.get("map_center")
    if map_center_value is not None:
        map_center = np.asarray(map_center_value, dtype=np.float32).reshape(-1)
        if map_center.shape[0] >= 3 and np.isfinite(map_center[:3]).all():
            return map_center[:3].astype(np.float32), float(metadata.get("map_heading") or 0.0)

    # Match the current dataset preprocessor exactly. That code overwrites the running
    # map bounds on each feature instead of taking a global min/max, so the effective
    # center comes from the final iterated map feature rather than the true scene bounds.
    # We intentionally mirror that behavior here so privileged semantic supervision and
    # decoder states share the same coordinate frame.
    max_x = max_y = max_z = float("-inf")
    min_x = min_y = min_z = float("inf")
    found = False
    for locations in _iter_map_feature_locations(raw_scenario):
        loc = np.asarray(locations, dtype=np.float32).reshape(-1, locations.shape[-1])
        max_boundary = np.nanmax(loc, axis=0)
        min_boundary = np.nanmin(loc, axis=0)
        max_x = float(max_boundary[0])
        max_y = float(max_boundary[1])
        min_x = float(min_boundary[0])
        min_y = float(min_boundary[1])
        if loc.shape[-1] >= 3:
            max_z = float(max_boundary[2])
            min_z = float(min_boundary[2])
        found = True
    if not found:
        return np.zeros((3,), dtype=np.float32), float(metadata.get("map_heading") or 0.0)
    if max_z == float("-inf"):
        max_z = 0.0
    if min_z == float("inf"):
        min_z = 0.0
    map_center = np.stack(
        [
            np.asarray([max_x, max_y, max_z], dtype=np.float32),
            np.asarray([min_x, min_y, min_z], dtype=np.float32),
        ],
        axis=0,
    ).mean(axis=0).astype(np.float32)
    return map_center, float(metadata.get("map_heading") or 0.0)


def world_xy_to_model_frame(points_xy_world: Any, *, map_center: np.ndarray, map_heading: float) -> np.ndarray:
    xy_world = np.asarray(points_xy_world, dtype=np.float32).reshape(-1, 2)
    if xy_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    centered = xy_world - np.asarray(map_center, dtype=np.float32).reshape(1, 3)[:, :2]
    if float(map_heading) == 0.0:
        return centered.astype(np.float32)
    c = float(np.cos(-float(map_heading)))
    s = float(np.sin(-float(map_heading)))
    return np.stack(
        [c * centered[:, 0] - s * centered[:, 1], s * centered[:, 0] + c * centered[:, 1]],
        axis=-1,
    ).astype(np.float32)


def world_direction_to_model_frame(vectors_world_xy: Any, *, map_heading: float) -> np.ndarray:
    vec = np.asarray(vectors_world_xy, dtype=np.float32).reshape(-1, 2)
    if vec.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if float(map_heading) == 0.0:
        return vec.astype(np.float32)
    c = float(np.cos(-float(map_heading)))
    s = float(np.sin(-float(map_heading)))
    return np.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1]], axis=-1).astype(np.float32)


@dataclass
class ResampledWorldPath:
    path_id: Optional[str]
    slot_id: str
    source_kind: str
    semantic_label: str
    confidence: float
    use_for_training: bool
    waypoints_xy_world: np.ndarray
    headings_world: np.ndarray
    tangents_world: np.ndarray
    arc_lengths_m: np.ndarray


def tangents_from_headings(headings_world: Any) -> np.ndarray:
    headings = np.asarray(headings_world, dtype=np.float32).reshape(-1)
    if headings.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([np.cos(headings), np.sin(headings)], axis=-1).astype(np.float32)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _slot_semantic_label(slot: Mapping[str, Any]) -> str:
    return normalize_semantic_label(slot.get("semantic_label"), default="straight")


def _slot_confidence(slot: Mapping[str, Any]) -> float:
    return float(np.clip(_safe_float(slot.get("confidence"), default=0.0), 0.0, 1.0))


def _slot_is_trainable(slot: Mapping[str, Any], *, include_stop: bool = True) -> bool:
    valid = bool(slot.get("is_valid_target", True))
    if not valid:
        return False
    if _slot_semantic_label(slot) == "stop" and not include_stop:
        return False
    return True


def _source_kind_to_row(slot: Mapping[str, Any]) -> str:
    return "factual_gt" if str(slot.get("source_kind") or "") == "ground_truth" else "alternative_sdc_path"


def iter_highlighted_slots(contract: Mapping[str, Any], *, include_stop: bool = True) -> List[Dict[str, Any]]:
    slots: List[Dict[str, Any]] = []
    for slot in list(contract.get("highlighted_paths", []) or []):
        row = dict(slot or {})
        row["semantic_label"] = _slot_semantic_label(row)
        row["confidence"] = _slot_confidence(row)
        row["use_for_training"] = _slot_is_trainable(row, include_stop=include_stop)
        row["row_source_kind"] = _source_kind_to_row(row)
        row["slot_id"] = str(row.get("slot_id") or "")
        row["path_id"] = None if row.get("path_id") is None else str(row.get("path_id"))
        slots.append(row)
    return slots


def group_slots_by_semantic_label(contract: Mapping[str, Any], *, include_stop: bool = True) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for slot in iter_highlighted_slots(contract, include_stop=include_stop):
        grouped.setdefault(str(slot["semantic_label"]), []).append(slot)
    return grouped


def build_resampled_world_path(
    *,
    raw_scenario: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    slot: Mapping[str, Any],
    spacing_m: float = DEFAULT_RESAMPLE_SPACING_M,
    stitch_discontinuities: bool = False,
    stitch_radius_m: float = DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    stitch_jump_threshold_m: float = DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
) -> ResampledWorldPath:
    current_xy_world, _ = extract_sdc_current_pose(
        raw_scenario,
        sdc_id=str(sdc_id),
        current_time_index=int(current_time_index),
    )
    source_kind = _source_kind_to_row(slot)
    if source_kind == "factual_gt":
        world_xy = extract_ground_truth_sdc_route_xy(
            raw_scenario,
            sdc_id=str(sdc_id),
            current_time_index=int(current_time_index),
        )
    else:
        candidate_xy = _extract_valid_sdc_path_xy(raw_scenario, str(slot.get("path_id") or ""))
        world_xy = trim_polyline_from_point(candidate_xy, current_xy_world, prepend_point=True)
    if bool(stitch_discontinuities):
        world_xy = stitch_polyline_discontinuities(
            world_xy,
            handoff_radius_m=float(stitch_radius_m),
            jump_threshold_m=float(stitch_jump_threshold_m),
        )
    resampled_world = resample_polyline_xy(world_xy, spacing_m=float(spacing_m))
    headings_world = polyline_headings(resampled_world)
    tangents_world = tangents_from_headings(headings_world)
    arc_lengths_m = polyline_arc_lengths(resampled_world)
    return ResampledWorldPath(
        path_id=None if slot.get("path_id") is None else str(slot.get("path_id")),
        slot_id=str(slot.get("slot_id") or ""),
        source_kind=source_kind,
        semantic_label=_slot_semantic_label(slot),
        confidence=_slot_confidence(slot),
        use_for_training=_slot_is_trainable(slot),
        waypoints_xy_world=np.asarray(resampled_world, dtype=np.float32),
        headings_world=np.asarray(headings_world, dtype=np.float32),
        tangents_world=np.asarray(tangents_world, dtype=np.float32),
        arc_lengths_m=np.asarray(arc_lengths_m, dtype=np.float32),
    )


def build_world_paths_for_contract(
    *,
    raw_scenario: Mapping[str, Any],
    contract: Mapping[str, Any],
    sdc_id: str,
    current_time_index: int,
    spacing_m: float = DEFAULT_RESAMPLE_SPACING_M,
    include_stop: bool = True,
    stitch_discontinuities: bool = False,
    stitch_radius_m: float = DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    stitch_jump_threshold_m: float = DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
) -> Dict[str, ResampledWorldPath]:
    out: Dict[str, ResampledWorldPath] = {}
    for slot in iter_highlighted_slots(contract, include_stop=include_stop):
        slot_id = str(slot.get("slot_id") or "")
        if not slot_id:
            continue
        out[slot_id] = build_resampled_world_path(
            raw_scenario=raw_scenario,
            sdc_id=str(sdc_id),
            current_time_index=int(current_time_index),
            slot=slot,
            spacing_m=float(spacing_m),
            stitch_discontinuities=bool(stitch_discontinuities),
            stitch_radius_m=float(stitch_radius_m),
            stitch_jump_threshold_m=float(stitch_jump_threshold_m),
        )
    return out


def _to_local_path(path: ResampledWorldPath) -> ResampledLocalPath:
    return ResampledLocalPath(
        waypoints_xy=np.asarray(path.waypoints_xy_world, dtype=np.float32),
        headings=np.asarray(path.headings_world, dtype=np.float32),
        arc_lengths_m=np.asarray(path.arc_lengths_m, dtype=np.float32),
    )


def compute_family_divergence_profile(
    *,
    family_path: ResampledWorldPath,
    competing_other_label_paths: Sequence[ResampledWorldPath],
    scale_m: float = DEFAULT_SEPARABILITY_SCALE_M,
    heading_weight_m: float = DEFAULT_SEPARABILITY_HEADING_WEIGHT_M,
) -> Dict[str, Any]:
    competitor_map = {
        f"{path.slot_id}:{path.path_id or 'gt'}": _to_local_path(path)
        for path in competing_other_label_paths
        if np.asarray(path.waypoints_xy_world, dtype=np.float32).shape[0] >= 2
    }
    return compute_path_separability_profile(
        _to_local_path(family_path),
        competitor_map,
        scale_m=float(scale_m),
        heading_weight_m=float(heading_weight_m),
    )


def first_divergence_onset_m(
    arc_lengths_m: Sequence[float],
    separability: Sequence[float],
    *,
    threshold: float = DEFAULT_FAMILY_DIVERGENCE_THRESHOLD,
    min_run: int = DEFAULT_FAMILY_DIVERGENCE_MIN_RUN,
) -> float:
    arc = np.asarray(arc_lengths_m, dtype=np.float32).reshape(-1)
    sep = np.asarray(separability, dtype=np.float32).reshape(-1)
    if arc.size == 0 or sep.size == 0:
        return float("inf")
    meets = sep >= float(threshold)
    if not bool(np.any(meets)):
        return float("inf")
    run = 0
    for idx, flag in enumerate(meets.tolist()):
        run = run + 1 if flag else 0
        if run >= int(max(min_run, 1)):
            start_idx = idx - run + 1
            return float(arc[max(start_idx, 0)])
    return float("inf")


def load_sdc_semantic_control_row(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("rt", encoding="utf-8") as f:
        return dict(json.load(f))


def is_sdc_semantic_control_row(row: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    if str(row.get("schema_version") or "").strip() == SDC_SEMANTIC_CONTROL_SCHEMA_VERSION:
        return True
    required = (
        "requested_semantic_label",
        "candidate_family_path_ids",
        "candidate_family_resampled_paths_world",
        "candidate_family_arc_lengths_m",
        "candidate_family_divergence_onsets_m",
    )
    return all(key in row for key in required)


def build_sdc_semantic_dataset_fields(
    *,
    scenario_id: str,
    decoder_track_names: Sequence[Any],
    horizon: int,
    row: Mapping[str, Any],
    require_trainable: bool,
    include_stop: bool = True,
    stitch_discontinuities: bool = False,
    stitch_radius_m: float = DEFAULT_DISCONTINUITY_STITCH_RADIUS_M,
    stitch_jump_threshold_m: float = DEFAULT_DISCONTINUITY_STITCH_JUMP_THRESHOLD_M,
) -> Dict[str, Any]:
    debug_meta = {
        "schema_version": str(row.get("schema_version") or SDC_SEMANTIC_CONTROL_SCHEMA_VERSION),
        "scenario_id": str(scenario_id),
        "sdc_id": str(row.get("sdc_id") or ""),
        "scenario_pkl": str(row.get("scenario_pkl") or ""),
        "current_time_index": int(row.get("current_time_index") or 0),
        "selected_slot_id": str(row.get("selected_slot_id") or ""),
        "selected_path_id": (None if row.get("selected_path_id") is None else str(row.get("selected_path_id"))),
        "requested_semantic_label": normalize_semantic_label(row.get("requested_semantic_label")),
        "source_kind": str(row.get("source_kind") or ""),
        "candidate_family_path_ids": [None if value is None else str(value) for value in list(row.get("candidate_family_path_ids", []) or [])],
        "family_size": int(len(list(row.get("candidate_family_path_ids", []) or []))),
        "family_stitch_discontinuities": bool(stitch_discontinuities),
        "family_stitch_radius_m": float(stitch_radius_m),
        "family_stitch_jump_threshold_m": float(stitch_jump_threshold_m),
    }
    sdc_id = str(row.get("sdc_id") or "")
    decoder_track_names = [str(value) for value in np.asarray(decoder_track_names, dtype=object).reshape(-1).tolist()]
    decision_agent_mask = np.zeros((len(decoder_track_names),), dtype=np.float32)
    if sdc_id and sdc_id in decoder_track_names:
        decision_agent_mask[decoder_track_names.index(sdc_id)] = 1.0
    time_window_mask = np.ones((int(horizon),), dtype=np.float32)

    semantic_label = normalize_semantic_label(row.get("requested_semantic_label"))
    semantic_confidence = float(row.get("requested_semantic_confidence") or 0.0)
    use_for_training = bool(row.get("use_for_training", True))
    if semantic_label == "stop" and not include_stop:
        use_for_training = False
        debug_meta["drop_reason"] = "stop_row_disabled"
    control_available = bool(decision_agent_mask.sum() > 0) and use_for_training
    if require_trainable and not control_available:
        debug_meta.setdefault("drop_reason", "sdc_not_modeled_or_row_disabled")

    raw_family_paths = list(row.get("candidate_family_resampled_paths_model", []) or [])
    raw_family_tangents = list(row.get("candidate_family_resampled_path_tangents_model", []) or [])
    if not raw_family_paths or not raw_family_tangents:
        world_family_paths = list(row.get("candidate_family_resampled_paths_world", []) or [])
        world_family_tangents = list(row.get("candidate_family_resampled_path_tangents_world", []) or [])
        if world_family_paths and world_family_tangents:
            scenario_pkl = str(row.get("scenario_pkl") or "").strip()
            if scenario_pkl:
                raw_scenario = load_raw_scenario_from_row(row)
                map_center, map_heading = extract_model_frame(raw_scenario)
                raw_family_paths = [
                    world_xy_to_model_frame(path_xy, map_center=map_center, map_heading=map_heading).tolist()
                    for path_xy in world_family_paths
                ]
                raw_family_tangents = [
                    world_direction_to_model_frame(tangent_xy, map_heading=map_heading).tolist()
                    for tangent_xy in world_family_tangents
                ]
                debug_meta["family_path_frame"] = "model_map_centered_fallback"
                debug_meta["map_center"] = np.asarray(map_center, dtype=np.float32).tolist()
                debug_meta["map_heading"] = float(map_heading)
            else:
                raw_family_paths = world_family_paths
                raw_family_tangents = world_family_tangents
                debug_meta["family_path_frame"] = "world_passthrough"
    else:
        debug_meta["family_path_frame"] = str(row.get("candidate_family_frame") or "model_map_centered")
    raw_family_arc = list(row.get("candidate_family_arc_lengths_m", []) or [])
    family_onsets = np.asarray(list(row.get("candidate_family_divergence_onsets_m", []) or []), dtype=np.float32).reshape(-1)
    family_confidences = np.asarray(list(row.get("candidate_family_confidences", []) or []), dtype=np.float32).reshape(-1)

    family_mask = np.zeros((0, 0), dtype=np.float32)
    family_paths = np.zeros((0, 0, 2), dtype=np.float32)
    family_tangents = np.zeros((0, 0, 2), dtype=np.float32)
    family_arc = np.zeros((0, 0), dtype=np.float32)
    selected_raw_path_world = np.zeros((0, 2), dtype=np.float32)
    selected_raw_path_model = np.zeros((0, 2), dtype=np.float32)
    selected_raw_path_segment_mask = np.zeros((0,), dtype=np.float32)
    selected_raw_path_mask = np.zeros((0,), dtype=np.float32)
    scenario_pkl = str(row.get("scenario_pkl") or "").strip()
    if scenario_pkl and sdc_id:
        raw_scenario = load_raw_scenario_from_row(row)
        map_center, map_heading = extract_model_frame(raw_scenario)
        selected_raw_path_world = np.asarray(
            build_selected_path_world(
                raw_scenario=raw_scenario,
                sdc_id=sdc_id,
                current_time_index=int(row.get("current_time_index") or 0),
                source_kind=str(row.get("source_kind") or ""),
                selected_path_id=None if row.get("selected_path_id") is None else str(row.get("selected_path_id")),
            ),
            dtype=np.float32,
        ).reshape(-1, 2)
        selected_raw_path_model = world_xy_to_model_frame(
            selected_raw_path_world,
            map_center=map_center,
            map_heading=map_heading,
        ).astype(np.float32)
        selected_raw_path_segment_mask = polyline_segment_valid_mask(
            selected_raw_path_model,
            jump_threshold_m=float(stitch_jump_threshold_m),
        ).astype(np.float32)
        selected_raw_path_mask = np.ones((selected_raw_path_model.shape[0],), dtype=np.float32)
        debug_meta["selected_raw_path_num_points"] = int(selected_raw_path_world.shape[0])
        debug_meta["selected_raw_path_num_segments"] = int(
            np.maximum(selected_raw_path_segment_mask.sum(), 0.0)
        )
        debug_meta["selected_raw_path_frame"] = "model_map_centered"
        debug_meta["selected_raw_path_map_center"] = np.asarray(map_center, dtype=np.float32).tolist()
        debug_meta["selected_raw_path_map_heading"] = float(map_heading)
    num_paths = min(
        len(raw_family_paths),
        len(raw_family_tangents),
        len(raw_family_arc),
        int(family_onsets.shape[0]) if family_onsets.size > 0 else len(raw_family_paths),
        int(family_confidences.shape[0]) if family_confidences.size > 0 else len(raw_family_paths),
    )
    if num_paths > 0:
        if bool(stitch_discontinuities):
            stitched_paths = []
            stitched_tangents = []
            stitched_arcs = []
            discontinuities_before: List[int] = []
            discontinuities_after: List[int] = []
            for idx in range(num_paths):
                original_path = np.asarray(raw_family_paths[idx], dtype=np.float32).reshape(-1, 2)
                discontinuities_before.append(
                    int(
                        max(
                            0,
                            len(
                                split_polyline_on_discontinuities(
                                    original_path,
                                    jump_threshold_m=float(stitch_jump_threshold_m),
                                )
                            )
                            - 1,
                        )
                    )
                )
                stitched_path = stitch_polyline_discontinuities(
                    original_path,
                    handoff_radius_m=float(stitch_radius_m),
                    jump_threshold_m=float(stitch_jump_threshold_m),
                )
                discontinuities_after.append(
                    int(
                        max(
                            0,
                            len(
                                split_polyline_on_discontinuities(
                                    stitched_path,
                                    jump_threshold_m=float(stitch_jump_threshold_m),
                                )
                            )
                            - 1,
                        )
                    )
                )
                stitched_paths.append(stitched_path.tolist())
                stitched_tangents.append(tangents_from_headings(polyline_headings(stitched_path)).tolist())
                stitched_arcs.append(polyline_arc_lengths(stitched_path).tolist())
            raw_family_paths = stitched_paths
            raw_family_tangents = stitched_tangents
            raw_family_arc = stitched_arcs
            debug_meta["family_discontinuities_before"] = discontinuities_before
            debug_meta["family_discontinuities_after"] = discontinuities_after
        path_arrays = [np.asarray(raw_family_paths[idx], dtype=np.float32).reshape(-1, 2) for idx in range(num_paths)]
        tangent_arrays = [np.asarray(raw_family_tangents[idx], dtype=np.float32).reshape(-1, 2) for idx in range(num_paths)]
        arc_arrays = [np.asarray(raw_family_arc[idx], dtype=np.float32).reshape(-1) for idx in range(num_paths)]
        max_waypoints = int(
            max(
                min(path_arrays[idx].shape[0], tangent_arrays[idx].shape[0], arc_arrays[idx].shape[0])
                for idx in range(num_paths)
            )
        )
        family_paths = np.zeros((num_paths, max_waypoints, 2), dtype=np.float32)
        family_tangents = np.zeros((num_paths, max_waypoints, 2), dtype=np.float32)
        family_arc = np.zeros((num_paths, max_waypoints), dtype=np.float32)
        family_mask = np.zeros((num_paths, max_waypoints), dtype=np.float32)
        for idx in range(num_paths):
            path_xy = np.asarray(path_arrays[idx], dtype=np.float32)
            tangents_xy = np.asarray(tangent_arrays[idx], dtype=np.float32)
            arc_lengths = np.asarray(arc_arrays[idx], dtype=np.float32)
            length = int(min(path_xy.shape[0], tangents_xy.shape[0], arc_lengths.shape[0], max_waypoints))
            if length <= 0:
                continue
            family_paths[idx, :length] = path_xy[:length]
            family_tangents[idx, :length] = tangents_xy[:length]
            family_arc[idx, :length] = arc_lengths[:length]
            family_mask[idx, :length] = 1.0
        family_onsets = np.asarray(family_onsets[:num_paths], dtype=np.float32)
        family_confidences = np.asarray(family_confidences[:num_paths], dtype=np.float32)
    if family_mask.shape[0] == 0:
        control_available = False
        debug_meta.setdefault("drop_reason", "missing_family_paths")

    is_factual = str(row.get("source_kind") or "") == "factual_gt"
    fields = {
        "cf/sdc_semantic_label_id": int(semantic_label_to_id(semantic_label)),
        "cf/sdc_semantic_confidence": np.float32(semantic_confidence),
        "cf/sdc_family_path_polylines_world": family_paths.astype(np.float32),
        "cf/sdc_family_path_tangents_world": family_tangents.astype(np.float32),
        "cf/sdc_family_arc_lengths": family_arc.astype(np.float32),
        "cf/sdc_family_divergence_onsets": family_onsets.astype(np.float32),
        "cf/sdc_family_path_mask": family_mask.astype(np.float32),
        "cf/sdc_family_confidences": family_confidences.astype(np.float32),
        "cf/sdc_selected_raw_path_world": selected_raw_path_world.astype(np.float32),
        "cf/sdc_selected_raw_path_model": selected_raw_path_model.astype(np.float32),
        "cf/sdc_selected_raw_path_mask": selected_raw_path_mask.astype(np.float32),
        "cf/sdc_selected_raw_path_segment_mask": selected_raw_path_segment_mask.astype(np.float32),
        "cf/sdc_is_factual": int(is_factual),
        "cf/sdc_control_available": int(control_available),
        "cf/sdc_debug_meta": dict(debug_meta),
        "cf/time_window_mask": time_window_mask.astype(np.float32),
        "cf/decision_agent_mask": decision_agent_mask.astype(np.float32),
        "cf/conditioning_eligible": int(control_available),
        "cf/control_available": int(control_available),
        "cf/path_supervision_mask": int(control_available),
        "cf/compliance_supervision_mask": 0,
        "cf/timing_supervision_mask": 0,
        "cf/debug_meta": dict(debug_meta),
    }
    return fields


def load_raw_scenario_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    scenario_pkl = row.get("scenario_pkl")
    if not scenario_pkl:
        raise ValueError("sdc_semantic_control row is missing scenario_pkl")
    return load_raw_scenario(Path(str(scenario_pkl)).expanduser())


def project_points_to_family_paths_torch(
    points_world_xy: torch.Tensor,
    *,
    family_path_polylines_world: torch.Tensor,
    family_path_mask: torch.Tensor,
    family_path_tangents_world: torch.Tensor,
    family_path_arc_lengths: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if points_world_xy.ndim < 3 or points_world_xy.shape[-1] != 2:
        raise ValueError(f"points_world_xy must end with [...,2], got {tuple(points_world_xy.shape)}")
    if family_path_polylines_world.ndim != 4 or family_path_polylines_world.shape[-1] != 2:
        raise ValueError(
            f"family_path_polylines_world must be [B,K,M,2], got {tuple(family_path_polylines_world.shape)}"
        )
    B = points_world_xy.shape[0]
    K = family_path_polylines_world.shape[1]
    flat_shape = points_world_xy.shape[1:-1]
    num_points = int(np.prod(flat_shape)) if flat_shape else 1
    points_flat = points_world_xy.reshape(B, num_points, 2)
    diff = points_flat[:, :, None, None, :] - family_path_polylines_world[:, None, :, :, :]
    d = torch.linalg.norm(diff, dim=-1)
    large = torch.full_like(d, 1e6)
    mask = family_path_mask[:, None, :, :] > 0
    d = torch.where(mask, d, large)
    nearest_idx = torch.argmin(d, dim=-1)
    nearest_distance = torch.gather(d, dim=-1, index=nearest_idx.unsqueeze(-1)).squeeze(-1)

    gather_idx = nearest_idx.unsqueeze(-1)
    tangent = torch.gather(
        family_path_tangents_world[:, None, :, :, :].expand(-1, num_points, -1, -1, -1),
        dim=3,
        index=gather_idx[..., None].expand(-1, -1, -1, 1, 2),
    ).squeeze(3)
    nearest_arc = torch.gather(
        family_path_arc_lengths[:, None, :, :].expand(-1, num_points, -1, -1),
        dim=3,
        index=gather_idx,
    ).squeeze(-1)
    nearest_heading = torch.atan2(tangent[..., 1], tangent[..., 0])

    out_shape = tuple(points_world_xy.shape[:-1]) + (K,)
    return {
        "nearest_idx": nearest_idx.reshape(out_shape),
        "nearest_distance": nearest_distance.reshape(out_shape),
        "nearest_arc": nearest_arc.reshape(out_shape),
        "nearest_heading": nearest_heading.reshape(out_shape),
    }


def project_points_to_segment_tube_torch(
    points_world_xy: torch.Tensor,
    *,
    path_points_world: torch.Tensor,
    path_point_mask: torch.Tensor,
    path_segment_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if points_world_xy.ndim < 3 or points_world_xy.shape[-1] != 2:
        raise ValueError(f"points_world_xy must end with [...,2], got {tuple(points_world_xy.shape)}")
    if path_points_world.ndim != 3 or path_points_world.shape[-1] != 2:
        raise ValueError(f"path_points_world must be [B,M,2], got {tuple(path_points_world.shape)}")

    B = points_world_xy.shape[0]
    flat_shape = points_world_xy.shape[1:-1]
    num_points = int(np.prod(flat_shape)) if flat_shape else 1
    points_flat = points_world_xy.reshape(B, num_points, 2)
    path_points_world = path_points_world.to(device=points_world_xy.device, dtype=points_world_xy.dtype)
    path_point_mask = path_point_mask.to(device=points_world_xy.device) > 0
    path_segment_mask = path_segment_mask.to(device=points_world_xy.device) > 0

    if path_points_world.shape[1] == 0:
        nearest_distance = torch.full(
            (B, num_points),
            1e6,
            device=points_world_xy.device,
            dtype=points_world_xy.dtype,
        )
        out_shape = tuple(points_world_xy.shape[:-1])
        return {
            "nearest_distance": nearest_distance.reshape(out_shape),
            "nearest_arc": torch.zeros((B, num_points), device=points_world_xy.device, dtype=points_world_xy.dtype).reshape(out_shape),
            "path_total_arc": torch.zeros((B,), device=points_world_xy.device, dtype=points_world_xy.dtype),
            "has_valid_segment": torch.zeros((B,), device=points_world_xy.device, dtype=torch.bool),
        }

    point_large = torch.full(
        (B, num_points, path_points_world.shape[1]),
        1e6,
        device=points_world_xy.device,
        dtype=points_world_xy.dtype,
    )
    point_distance = torch.linalg.norm(points_flat[:, :, None, :] - path_points_world[:, None, :, :], dim=-1)
    point_distance = torch.where(path_point_mask[:, None, :], point_distance, point_large)
    nearest_point = point_distance.min(dim=-1)
    nearest_point_distance = nearest_point.values
    nearest_point_idx = nearest_point.indices

    if path_points_world.shape[1] < 2:
        out_shape = tuple(points_world_xy.shape[:-1])
        return {
            "nearest_distance": nearest_point_distance.reshape(out_shape),
            "nearest_arc": torch.zeros((B, num_points), device=points_world_xy.device, dtype=points_world_xy.dtype).reshape(out_shape),
            "path_total_arc": torch.zeros((B,), device=points_world_xy.device, dtype=points_world_xy.dtype),
            "has_valid_segment": torch.zeros((B,), device=points_world_xy.device, dtype=torch.bool),
        }

    segment_valid = path_point_mask[:, :-1] & path_point_mask[:, 1:]
    if path_segment_mask.shape[1] == path_points_world.shape[1]:
        segment_valid = segment_valid & path_segment_mask[:, :-1]
    else:
        segment_valid = segment_valid & path_segment_mask[:, : segment_valid.shape[1]]

    seg_start = path_points_world[:, :-1, :]
    seg_end = path_points_world[:, 1:, :]
    seg_delta = seg_end - seg_start
    seg_length = torch.linalg.norm(seg_delta, dim=-1)
    seg_denom = (seg_delta * seg_delta).sum(dim=-1).clamp_min(1e-6)
    seg_step = torch.where(segment_valid, seg_length, torch.zeros_like(seg_length))
    path_arc = torch.cat(
        [
            torch.zeros((B, 1), device=points_world_xy.device, dtype=points_world_xy.dtype),
            torch.cumsum(seg_step, dim=-1),
        ],
        dim=-1,
    )
    path_total_arc = path_arc[:, -1]
    rel = points_flat[:, :, None, :] - seg_start[:, None, :, :]
    t = ((rel * seg_delta[:, None, :, :]).sum(dim=-1) / seg_denom[:, None, :]).clamp_(0.0, 1.0)
    seg_projection = seg_start[:, None, :, :] + t[..., None] * seg_delta[:, None, :, :]
    seg_distance = torch.linalg.norm(points_flat[:, :, None, :] - seg_projection, dim=-1)
    seg_large = torch.full_like(seg_distance, 1e6)
    seg_distance = torch.where(segment_valid[:, None, :], seg_distance, seg_large)
    nearest_segment = seg_distance.min(dim=-1)
    nearest_segment_distance = nearest_segment.values
    nearest_segment_idx = nearest_segment.indices
    nearest_segment_t = torch.gather(t, dim=-1, index=nearest_segment_idx.unsqueeze(-1)).squeeze(-1)
    seg_arc_start = path_arc[:, :-1]
    nearest_segment_arc_start = torch.gather(
        seg_arc_start,
        dim=-1,
        index=nearest_segment_idx,
    )
    nearest_segment_length = torch.gather(seg_length, dim=-1, index=nearest_segment_idx)
    nearest_segment_arc = nearest_segment_arc_start + nearest_segment_t * nearest_segment_length
    nearest_point_arc = torch.gather(path_arc, dim=-1, index=nearest_point_idx)

    has_valid_segment = segment_valid.any(dim=-1)
    nearest_distance = torch.where(
        has_valid_segment[:, None],
        nearest_segment_distance,
        nearest_point_distance,
    )
    nearest_arc = torch.where(
        has_valid_segment[:, None],
        nearest_segment_arc,
        nearest_point_arc,
    )
    out_shape = tuple(points_world_xy.shape[:-1])
    return {
        "nearest_distance": nearest_distance.reshape(out_shape),
        "nearest_arc": nearest_arc.reshape(out_shape),
        "path_total_arc": path_total_arc,
        "has_valid_segment": has_valid_segment,
    }


def compute_family_gate_torch(
    projected_arc_m: torch.Tensor,
    divergence_onsets_m: torch.Tensor,
    *,
    bandwidth_m: float = DEFAULT_FAMILY_GUIDE_BANDWIDTH_M,
) -> torch.Tensor:
    bandwidth = max(float(bandwidth_m), 1e-3)
    onset = divergence_onsets_m.to(dtype=projected_arc_m.dtype)
    finite = torch.isfinite(onset)
    onset = torch.where(finite, onset, torch.full_like(onset, 1e9))
    delta = projected_arc_m - onset[:, None, :]
    gate = torch.sigmoid(delta / bandwidth)
    gate = torch.where(finite[:, None, :], gate, torch.zeros_like(gate))
    return gate


def family_confidence_weights_torch(
    family_confidences: torch.Tensor,
    *,
    family_path_mask: torch.Tensor,
    confidence_weighted: bool = False,
) -> torch.Tensor:
    valid_paths = (family_path_mask.sum(dim=-1) > 0).to(dtype=family_confidences.dtype)
    if not bool(confidence_weighted):
        raw = valid_paths
    else:
        raw = torch.clamp(family_confidences, min=1e-3) * valid_paths
    denom = raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return raw / denom
