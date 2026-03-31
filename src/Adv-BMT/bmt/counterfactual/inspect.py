from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .types import CanonicalMapFeature, CanonicalScenario, CanonicalTrack, stable_string_sort_key
from .visualize import render_bev_overview

INSPECTION_FILENAMES = {
    "scenario_summary": "scenario_summary.json",
    "track_table": "track_table.csv",
    "traffic_light_table": "traffic_light_table.csv",
    "map_feature_inventory": "map_feature_inventory.json",
    "unique_map_feature_types": "unique_map_feature_types.json",
    "sdc_track": "sdc_track.npz",
    "canonical_scenario": "canonical_scenario.pkl",
    "bev_overview": "bev_overview.png",
}


def inspection_output_paths(outdir: str | Path) -> Dict[str, Path]:
    root = Path(outdir).expanduser()
    return {name: root / filename for name, filename in INSPECTION_FILENAMES.items()}


def write_inspection_artifacts(canonical: CanonicalScenario, outdir: str | Path) -> Dict[str, Path]:
    paths = inspection_output_paths(outdir)
    root = Path(outdir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    summary = build_scenario_summary(canonical)
    track_rows = build_track_table_rows(canonical)
    light_rows = build_traffic_light_table_rows(canonical)
    map_inventory = build_map_feature_inventory(canonical)
    unique_types = sorted({feature.feature_type for feature in canonical.map_features.values()})

    _write_json(paths["scenario_summary"], summary)
    _write_csv(
        paths["track_table"],
        track_rows,
        fieldnames=[
            "track_id",
            "object_type",
            "is_sdc",
            "is_object_of_interest",
            "num_valid_steps",
            "valid_fraction",
            "first_valid_index",
            "last_valid_index",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "distance_m",
            "mean_speed_mps",
            "max_speed_mps",
        ],
    )
    _write_csv(
        paths["traffic_light_table"],
        light_rows,
        fieldnames=[
            "light_id",
            "lane_ref",
            "stop_x",
            "stop_y",
            "num_known_states",
            "first_known_state",
            "state_transition_count",
            "fraction_unknown",
        ],
    )
    _write_json(paths["map_feature_inventory"], map_inventory)
    _write_json(paths["unique_map_feature_types"], unique_types)
    _write_sdc_track_npz(paths["sdc_track"], canonical)
    with paths["canonical_scenario"].open("wb") as f:
        pickle.dump(canonical, f)
    render_bev_overview(canonical, paths["bev_overview"])
    return paths


def build_scenario_summary(canonical: CanonicalScenario) -> Dict[str, Any]:
    map_feature_type_counts = Counter(feature.feature_type for feature in canonical.map_features.values())
    sdc_track = canonical.tracks.get(canonical.sdc_id)
    sdc_stats = basic_track_motion_stats(sdc_track) if sdc_track is not None else _empty_motion_stats()
    return {
        "scenario_id": canonical.scenario_id,
        "length": int(canonical.length),
        "sdc_id": canonical.sdc_id,
        "num_tracks": int(len(canonical.tracks)),
        "num_traffic_lights": int(len(canonical.traffic_lights)),
        "map_feature_type_counts": dict(sorted(map_feature_type_counts.items())),
        "objects_of_interest": list(canonical.objects_of_interest),
        "current_time_index": int(canonical.current_time_index),
        "sdc_motion_stats": sdc_stats,
        "metadata_summary": canonical.metadata_summary,
    }


def build_track_table_rows(canonical: CanonicalScenario) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for track_id in sorted(canonical.tracks.keys(), key=lambda value: (0 if value == canonical.sdc_id else 1, *stable_string_sort_key(value))):
        track = canonical.tracks[track_id]
        stats = basic_track_motion_stats(track)
        rows.append(
            {
                "track_id": track_id,
                "object_type": track.object_type,
                "is_sdc": track_id == canonical.sdc_id,
                "is_object_of_interest": track_id in set(canonical.objects_of_interest),
                "num_valid_steps": stats["num_valid_steps"],
                "valid_fraction": stats["valid_fraction"],
                "first_valid_index": stats["first_valid_index"],
                "last_valid_index": stats["last_valid_index"],
                "start_x": stats["start_x"],
                "start_y": stats["start_y"],
                "end_x": stats["end_x"],
                "end_y": stats["end_y"],
                "distance_m": stats["distance_m"],
                "mean_speed_mps": stats["mean_speed_mps"],
                "max_speed_mps": stats["max_speed_mps"],
            }
        )
    return rows


def build_traffic_light_table_rows(canonical: CanonicalScenario) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for light_id in sorted(canonical.traffic_lights.keys(), key=stable_string_sort_key):
        light = canonical.traffic_lights[light_id]
        known_states = [state for state in light.object_state if not _is_unknown_light_state(state)]
        transition_count = 0
        if known_states:
            prev = known_states[0]
            for state in known_states[1:]:
                if state != prev:
                    transition_count += 1
                prev = state

        unknown_count = sum(1 for state in light.object_state if _is_unknown_light_state(state))
        length = max(len(light.object_state), 1)
        rows.append(
            {
                "light_id": light_id,
                "lane_ref": light.lane_ref or "",
                "stop_x": "" if light.stop_point_xy is None else float(light.stop_point_xy[0]),
                "stop_y": "" if light.stop_point_xy is None else float(light.stop_point_xy[1]),
                "num_known_states": int(len(known_states)),
                "first_known_state": known_states[0] if known_states else "",
                "state_transition_count": int(transition_count),
                "fraction_unknown": float(unknown_count / float(length)),
            }
        )
    return rows


def build_map_feature_inventory(canonical: CanonicalScenario) -> Dict[str, Any]:
    features = []
    for feature_id in sorted(canonical.map_features.keys(), key=stable_string_sort_key):
        feature = canonical.map_features[feature_id]
        bbox = _feature_bbox(feature)
        features.append(
            {
                "feature_id": feature_id,
                "type": feature.feature_type,
                "num_polyline_points": int(feature.polyline_xy.shape[0]),
                "num_polygon_points": int(0 if feature.polygon_xy is None else feature.polygon_xy.shape[0]),
                "has_polyline": bool(feature.polyline_xy.shape[0] > 0),
                "has_polygon": bool(feature.polygon_xy is not None and feature.polygon_xy.shape[0] > 0),
                "bbox_xy": bbox,
                "metadata": feature.metadata,
            }
        )
    return {
        "scenario_id": canonical.scenario_id,
        "num_map_features": int(len(features)),
        "features": features,
    }


def basic_track_motion_stats(track: CanonicalTrack | None) -> Dict[str, Any]:
    if track is None:
        return _empty_motion_stats()

    valid_mask = np.asarray(track.valid, dtype=bool)
    finite_mask = np.isfinite(track.position_xy).all(axis=-1)
    mask = valid_mask & finite_mask
    valid_idx = np.flatnonzero(mask)
    if valid_idx.size == 0:
        return _empty_motion_stats()

    xy = np.asarray(track.position_xy, dtype=np.float32)[valid_idx]
    speed = np.linalg.norm(np.asarray(track.velocity_xy, dtype=np.float32)[valid_idx], axis=-1)
    diffs = np.diff(xy, axis=0)
    distance_m = float(np.linalg.norm(diffs, axis=-1).sum()) if diffs.size > 0 else 0.0
    start = xy[0]
    end = xy[-1]
    return {
        "num_valid_steps": int(valid_idx.size),
        "valid_fraction": float(valid_idx.size / max(track.valid.shape[0], 1)),
        "first_valid_index": int(valid_idx[0]),
        "last_valid_index": int(valid_idx[-1]),
        "start_x": float(start[0]),
        "start_y": float(start[1]),
        "end_x": float(end[0]),
        "end_y": float(end[1]),
        "distance_m": distance_m,
        "displacement_m": float(np.linalg.norm(end - start)),
        "mean_speed_mps": float(np.nanmean(speed)) if speed.size > 0 else 0.0,
        "max_speed_mps": float(np.nanmax(speed)) if speed.size > 0 else 0.0,
    }


def _empty_motion_stats() -> Dict[str, Any]:
    return {
        "num_valid_steps": 0,
        "valid_fraction": 0.0,
        "first_valid_index": None,
        "last_valid_index": None,
        "start_x": None,
        "start_y": None,
        "end_x": None,
        "end_y": None,
        "distance_m": 0.0,
        "displacement_m": 0.0,
        "mean_speed_mps": 0.0,
        "max_speed_mps": 0.0,
    }


def _feature_bbox(feature: CanonicalMapFeature) -> Dict[str, float] | None:
    xy = feature.polyline_xy
    if feature.polygon_xy is not None and feature.polygon_xy.shape[0] > 0:
        xy = np.concatenate([xy, feature.polygon_xy], axis=0) if xy.shape[0] > 0 else feature.polygon_xy
    if xy.shape[0] == 0:
        return None
    return {
        "min_x": float(np.nanmin(xy[:, 0])),
        "min_y": float(np.nanmin(xy[:, 1])),
        "max_x": float(np.nanmax(xy[:, 0])),
        "max_y": float(np.nanmax(xy[:, 1])),
    }


def _is_unknown_light_state(state: str | None) -> bool:
    return state in {None, "", "LANE_STATE_UNKNOWN"}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], *, fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_sdc_track_npz(path: Path, canonical: CanonicalScenario) -> None:
    track = canonical.tracks.get(canonical.sdc_id)
    if track is None:
        np.savez(
            path,
            scenario_id=np.asarray(canonical.scenario_id),
            sdc_id=np.asarray(canonical.sdc_id),
            ts=canonical.ts,
            position_xy=np.zeros((0, 2), dtype=np.float32),
            position_xyz=np.zeros((0, 3), dtype=np.float32),
            heading=np.zeros((0,), dtype=np.float32),
            velocity_xy=np.zeros((0, 2), dtype=np.float32),
            valid=np.zeros((0,), dtype=bool),
        )
        return

    np.savez(
        path,
        scenario_id=np.asarray(canonical.scenario_id),
        sdc_id=np.asarray(canonical.sdc_id),
        ts=canonical.ts,
        position_xy=track.position_xy,
        position_xyz=track.position_xyz,
        heading=track.heading,
        velocity_xy=track.velocity_xy,
        valid=track.valid,
    )

