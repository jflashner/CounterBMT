from __future__ import annotations

import dataclasses
import importlib
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .normalize import normalize_scenario


def waymax_available() -> bool:
    return importlib.util.find_spec("waymax") is not None


def require_waymax() -> None:
    if not waymax_available():
        raise ImportError("waymax is not installed in this environment. Install waymax to use the WOMD 1.3.1 adapter.")


def resolve_waymax_config(
    *,
    config_name: str = "WOD_1_3_1_TRAINING",
    path: str = "",
    include_sdc_paths: bool = True,
    num_paths: int = 16,
    num_points_per_path: int = 128,
) -> Any:
    require_waymax()
    from waymax import config as waymax_config

    if not hasattr(waymax_config, str(config_name)):
        raise AttributeError(f"waymax.config has no attribute {config_name!r}")
    base_config = getattr(waymax_config, str(config_name))
    replace_kwargs = {}
    if path:
        replace_kwargs["path"] = str(path)
    if include_sdc_paths:
        replace_kwargs["include_sdc_paths"] = True
        replace_kwargs["num_paths"] = int(num_paths)
        replace_kwargs["num_points_per_path"] = int(num_points_per_path)

    if dataclasses.is_dataclass(base_config) and hasattr(base_config, "path"):
        return dataclasses.replace(base_config, **replace_kwargs)
    if dataclasses.is_dataclass(base_config) and hasattr(base_config, "dataset_config"):
        dataset_config = getattr(base_config, "dataset_config")
        if dataclasses.is_dataclass(dataset_config):
            replaced_dataset = dataclasses.replace(dataset_config, **replace_kwargs)
            return dataclasses.replace(base_config, dataset_config=replaced_dataset)
    raise TypeError(f"Unsupported Waymax config object for {config_name!r}: {type(base_config)!r}")


def iter_waymax_simulator_states(config: Any) -> Iterator[Any]:
    require_waymax()
    from waymax.dataloader import womd_dataloader

    yield from womd_dataloader.simulator_state_generator(config=config)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _get_any(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
    for name in names:
        if hasattr(value, name):
            attr = getattr(value, name)
            if attr is not None:
                return attr
    return default


def _to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=(dtype or np.float32))
    array = np.asarray(value, dtype=dtype)
    return np.asarray(array)


def _squeeze_leading_dims(array: np.ndarray, *, max_ndim: int) -> np.ndarray:
    squeezed = np.asarray(array)
    while squeezed.ndim > max_ndim and squeezed.shape[0] == 1:
        squeezed = squeezed[0]
    return np.asarray(squeezed)


def _align_object_time(array: np.ndarray, *, num_objects: int | None = None) -> np.ndarray:
    arr = _squeeze_leading_dims(np.asarray(array), max_ndim=2)
    if arr.ndim == 1:
        arr = arr[None, :]
    if num_objects is not None and arr.shape[0] != num_objects and arr.shape[1] == num_objects:
        arr = arr.T
    return np.asarray(arr)


def _align_path_points(array: np.ndarray, *, num_paths: int | None = None) -> np.ndarray:
    arr = _squeeze_leading_dims(np.asarray(array), max_ndim=2)
    if arr.ndim == 1:
        arr = arr[None, :]
    if num_paths is not None and arr.shape[0] != num_paths and arr.shape[1] == num_paths:
        arr = arr.T
    return np.asarray(arr)


def _feature_type_name(type_id: Any) -> str:
    try:
        import waymax.datatypes as waymax_datatypes  # type: ignore

        enum_cls = getattr(waymax_datatypes, "MapElementIds", None)
        if enum_cls is not None:
            try:
                return str(enum_cls(int(type_id)).name)
            except Exception:
                pass
    except Exception:
        pass
    try:
        scalar = int(type_id)
    except Exception:
        return "UNKNOWN"
    return f"ROADGRAPH_TYPE_{scalar}"


def _object_type_name(type_id: Any) -> str:
    try:
        import waymax.datatypes as waymax_datatypes  # type: ignore

        enum_cls = getattr(waymax_datatypes, "ObjectTypeIds", None)
        if enum_cls is not None:
            try:
                return str(enum_cls(int(type_id)).name)
            except Exception:
                pass
    except Exception:
        pass
    mapping = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST"}
    try:
        return mapping.get(int(type_id), f"TYPE_{int(type_id)}")
    except Exception:
        return "UNKNOWN"


def _default_extent_by_object_type(type_name: str) -> tuple[float, float, float]:
    object_type = str(type_name).upper()
    if object_type == "PEDESTRIAN":
        return (0.8, 0.8, 1.7)
    if object_type == "CYCLIST":
        return (1.8, 0.6, 1.5)
    return (4.5, 1.8, 1.5)


def _normalize_waymax_traffic_light_state(value: Any) -> str | None:
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


def _extract_scenario_id(source: Any, *, fallback: str) -> str:
    for key in ("scenario_id", "id", "scenario/name", "scenario/id"):
        value = _get_any(source, (key,), default=None)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            if value.size == 0:
                continue
            value = value.reshape(-1)[0]
        text = str(value).strip()
        if text:
            return text
    return str(fallback)


def _extract_current_time_index(state: Any) -> int:
    value = _get_any(state, ("timestep", "current_time_index"), default=None)
    if value is None:
        return 10
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return 10
    return int(arr[0])


def _extract_track_dict(state: Any, *, fallback_scenario_id: str) -> tuple[Dict[str, Any], str, np.ndarray]:
    trajectory = _get_any(state, ("log_trajectory", "trajectory"), default=None)
    metadata = _get_any(state, ("object_metadata", "metadata"), default=None)
    if trajectory is None or metadata is None:
        raise ValueError("Waymax state is missing trajectory or object metadata")

    x = _align_object_time(_to_numpy(_get_any(trajectory, ("x",), default=None), dtype=np.float32))
    y = _align_object_time(_to_numpy(_get_any(trajectory, ("y",), default=None), dtype=np.float32), num_objects=x.shape[0])
    z = _align_object_time(_to_numpy(_get_any(trajectory, ("z",), default=np.zeros_like(x)), dtype=np.float32), num_objects=x.shape[0])
    yaw = _align_object_time(_to_numpy(_get_any(trajectory, ("yaw", "heading"), default=np.zeros_like(x)), dtype=np.float32), num_objects=x.shape[0])
    vel_x = _align_object_time(_to_numpy(_get_any(trajectory, ("vel_x", "velocity_x"), default=np.zeros_like(x)), dtype=np.float32), num_objects=x.shape[0])
    vel_y = _align_object_time(_to_numpy(_get_any(trajectory, ("vel_y", "velocity_y"), default=np.zeros_like(x)), dtype=np.float32), num_objects=x.shape[0])
    length = _align_object_time(
        _to_numpy(_get_any(trajectory, ("length", "lengths"), default=np.zeros_like(x)), dtype=np.float32),
        num_objects=x.shape[0],
    )
    width = _align_object_time(
        _to_numpy(_get_any(trajectory, ("width", "widths"), default=np.zeros_like(x)), dtype=np.float32),
        num_objects=x.shape[0],
    )
    height = _align_object_time(
        _to_numpy(_get_any(trajectory, ("height", "heights"), default=np.zeros_like(x)), dtype=np.float32),
        num_objects=x.shape[0],
    )
    valid = _align_object_time(_to_numpy(_get_any(trajectory, ("valid",), default=np.ones_like(x, dtype=bool)), dtype=bool), num_objects=x.shape[0])

    ids = _to_numpy(_get_any(metadata, ("ids", "id"), default=np.arange(x.shape[0])), dtype=np.int64).reshape(-1)
    object_types = _to_numpy(_get_any(metadata, ("object_types", "object_type"), default=np.zeros((ids.shape[0],), dtype=np.int64)), dtype=np.int64).reshape(-1)
    is_sdc = _to_numpy(_get_any(metadata, ("is_sdc", "is_self_driving_car"), default=np.zeros((ids.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    scenario_id = _extract_scenario_id(state, fallback=fallback_scenario_id)

    tracks: Dict[str, Any] = {}
    sdc_id = str(ids[0]) if ids.size > 0 else "0"
    if is_sdc.size == ids.size and np.any(is_sdc):
        sdc_id = str(ids[int(np.flatnonzero(is_sdc)[0])])

    for index in range(int(ids.shape[0])):
        track_id = str(ids[index])
        type_name = _object_type_name(object_types[index] if index < object_types.shape[0] else 0)
        default_length, default_width, default_height = _default_extent_by_object_type(type_name)
        track_length = np.asarray(length[index], dtype=np.float32).reshape(-1)
        track_width = np.asarray(width[index], dtype=np.float32).reshape(-1)
        track_height = np.asarray(height[index], dtype=np.float32).reshape(-1)
        if track_length.size == 0 or not np.any(np.isfinite(track_length)) or float(np.nanmax(np.abs(track_length))) <= 1e-6:
            track_length = np.full((x.shape[1],), float(default_length), dtype=np.float32)
        if track_width.size == 0 or not np.any(np.isfinite(track_width)) or float(np.nanmax(np.abs(track_width))) <= 1e-6:
            track_width = np.full((x.shape[1],), float(default_width), dtype=np.float32)
        if track_height.size == 0 or not np.any(np.isfinite(track_height)) or float(np.nanmax(np.abs(track_height))) <= 1e-6:
            track_height = np.full((x.shape[1],), float(default_height), dtype=np.float32)
        tracks[track_id] = {
            "type": type_name,
            "state": {
                "position": np.stack([x[index], y[index], z[index]], axis=-1).astype(np.float32).tolist(),
                "heading": yaw[index].astype(np.float32).tolist(),
                "velocity": np.stack([vel_x[index], vel_y[index]], axis=-1).astype(np.float32).tolist(),
                "valid": valid[index].astype(bool).tolist(),
                "length": track_length.astype(np.float32).tolist(),
                "width": track_width.astype(np.float32).tolist(),
                "height": track_height.astype(np.float32).tolist(),
            },
            "metadata": {
                "raw_object_type_id": int(object_types[index]) if index < object_types.shape[0] else 0,
                "is_sdc": bool(is_sdc[index]) if index < is_sdc.shape[0] else False,
            },
        }

    ts = np.arange(x.shape[1], dtype=np.float32) * np.float32(0.1)
    return tracks, sdc_id, ts


def _extract_dynamic_map_states(state: Any, *, horizon: int) -> Dict[str, Any]:
    lights = _get_any(state, ("log_traffic_light", "traffic_lights"), default=None)
    if lights is None:
        return {}
    state_arr = _align_object_time(_to_numpy(_get_any(lights, ("state",), default=None), dtype=np.int64))
    if state_arr.size == 0:
        return {}

    x = _align_object_time(_to_numpy(_get_any(lights, ("x",), default=np.zeros_like(state_arr)), dtype=np.float32), num_objects=state_arr.shape[0])
    y = _align_object_time(_to_numpy(_get_any(lights, ("y",), default=np.zeros_like(state_arr)), dtype=np.float32), num_objects=state_arr.shape[0])
    z = _align_object_time(_to_numpy(_get_any(lights, ("z",), default=np.zeros_like(state_arr)), dtype=np.float32), num_objects=state_arr.shape[0])
    valid = _align_object_time(_to_numpy(_get_any(lights, ("valid",), default=np.ones_like(state_arr, dtype=bool)), dtype=bool), num_objects=state_arr.shape[0])
    ids = _to_numpy(_get_any(lights, ("ids", "id"), default=np.arange(state_arr.shape[0])), dtype=np.int64).reshape(-1)
    lane_ids = _to_numpy(_get_any(lights, ("lane_ids", "lane_id"), default=np.full((ids.shape[0],), -1, dtype=np.int64)), dtype=np.int64).reshape(-1)

    rows: Dict[str, Any] = {}
    for index in range(int(ids.shape[0])):
        light_id = str(ids[index])
        stop_xyz = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        coords = np.stack([x[index], y[index], z[index]], axis=-1)
        valid_mask = valid[index] if index < valid.shape[0] else np.ones((coords.shape[0],), dtype=bool)
        valid_rows = np.flatnonzero(valid_mask)
        if valid_rows.size > 0:
            stop_xyz = np.asarray(coords[int(valid_rows[0])], dtype=np.float32)
        rows[light_id] = {
            "type": "TRAFFIC_LIGHT",
            "lane": None if index >= lane_ids.shape[0] or int(lane_ids[index]) < 0 else str(int(lane_ids[index])),
            "stop_point": stop_xyz.astype(np.float32).tolist(),
            "state": {
                "object_state": [
                    _normalize_waymax_traffic_light_state(item)
                    for item in state_arr[index, : min(state_arr.shape[1], horizon)].reshape(-1).tolist()
                ],
            },
            "metadata": {
                "raw_light_id": int(ids[index]),
            },
        }
    return rows


def _extract_map_features(state: Any) -> Dict[str, Any]:
    roadgraph = _get_any(state, ("roadgraph_points", "roadgraph"), default=None)
    if roadgraph is None:
        return {}

    x = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("x",), default=None), dtype=np.float32), max_ndim=1)
    y = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("y",), default=np.zeros_like(x)), dtype=np.float32), max_ndim=1)
    z = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("z",), default=np.zeros_like(x)), dtype=np.float32), max_ndim=1)
    valid = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("valid",), default=np.ones_like(x, dtype=bool)), dtype=bool), max_ndim=1)
    ids = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("ids", "id"), default=np.arange(x.shape[0])), dtype=np.int64), max_ndim=1)
    types = _squeeze_leading_dims(_to_numpy(_get_any(roadgraph, ("types", "type"), default=np.zeros((x.shape[0],), dtype=np.int64)), dtype=np.int64), max_ndim=1)
    if x.size == 0:
        return {}

    grouped: MutableMapping[str, List[int]] = {}
    for index in range(int(x.shape[0])):
        if index >= valid.shape[0] or not bool(valid[index]):
            continue
        feature_id = str(int(ids[index])) if index < ids.shape[0] else str(index)
        grouped.setdefault(feature_id, []).append(index)

    features: Dict[str, Any] = {}
    for feature_id, indices in grouped.items():
        coords = np.stack([x[indices], y[indices], z[indices]], axis=-1).astype(np.float32)
        type_id = int(types[indices[0]]) if indices and indices[0] < types.shape[0] else 0
        features[feature_id] = {
            "type": _feature_type_name(type_id),
            "polyline": coords.tolist(),
            "metadata": {
                "raw_type_id": int(type_id),
                "num_points": int(coords.shape[0]),
            },
        }
    return features


def _extract_sdc_paths(state: Any) -> Dict[str, Any]:
    paths = _get_any(state, ("sdc_paths", "paths"), default=None)
    if paths is None:
        return {}
    x = _align_path_points(_to_numpy(_get_any(paths, ("x",), default=None), dtype=np.float32))
    if x.size == 0:
        return {}
    y = _align_path_points(_to_numpy(_get_any(paths, ("y",), default=np.zeros_like(x)), dtype=np.float32), num_paths=x.shape[0])
    z = _align_path_points(_to_numpy(_get_any(paths, ("z",), default=np.zeros_like(x)), dtype=np.float32), num_paths=x.shape[0])
    valid = _align_path_points(_to_numpy(_get_any(paths, ("valid",), default=np.ones_like(x, dtype=bool)), dtype=bool), num_paths=x.shape[0])
    on_route = _to_numpy(_get_any(paths, ("on_route",), default=np.zeros((x.shape[0],), dtype=bool)), dtype=bool).reshape(-1)
    road_part_ids = _align_path_points(
        _to_numpy(_get_any(paths, ("ids", "id"), default=np.full_like(x, -1, dtype=np.int64)), dtype=np.int64),
        num_paths=x.shape[0],
    )

    sdc_paths: Dict[str, Any] = {}
    for index in range(int(x.shape[0])):
        # Waymax `Paths.ids` is per-point road-part membership, not a unique per-path id.
        # Use the row index as the stable path id so padded rows never overwrite each other.
        path_id = f"sdc_path_{index}"
        coords = np.stack([x[index], y[index], z[index]], axis=-1).astype(np.float32)
        valid_mask = valid[index].astype(bool) if index < valid.shape[0] else np.ones((coords.shape[0],), dtype=bool)
        path_road_part_ids = road_part_ids[index].reshape(-1).astype(np.int64) if index < road_part_ids.shape[0] else np.full((coords.shape[0],), -1, dtype=np.int64)
        unique_road_part_ids = [int(v) for v in np.unique(path_road_part_ids[valid_mask]) if int(v) >= 0]
        sdc_paths[path_id] = {
            "path_id": path_id,
            "polyline_xyz": coords.tolist(),
            "valid": valid_mask.tolist(),
            "metadata": {
                "source": "waymax_sdc_paths",
                "path_index": int(index),
                "on_route": bool(on_route[index]) if index < on_route.shape[0] else False,
                "road_part_ids": unique_road_part_ids,
                "point_road_part_ids": path_road_part_ids.tolist(),
            },
        }
    return sdc_paths


def raw_scenario_from_waymax_state(
    state: Any,
    *,
    scenario_id: str | None = None,
    current_time_index: int | None = None,
) -> Dict[str, Any]:
    fallback_id = str(scenario_id or "waymax_scene")
    tracks, sdc_id, ts = _extract_track_dict(state, fallback_scenario_id=fallback_id)
    scenario_name = _extract_scenario_id(state, fallback=fallback_id)
    current_idx = _extract_current_time_index(state) if current_time_index is None else int(current_time_index)
    raw = {
        "id": str(scenario_name),
        "metadata": {
            "scenario_id": str(scenario_name),
            "sdc_id": str(sdc_id),
            "current_time_index": int(current_idx),
            "ts": ts.astype(np.float32).tolist(),
            "source_format": "waymax_womd",
        },
        "tracks": tracks,
        "dynamic_map_states": _extract_dynamic_map_states(state, horizon=int(ts.shape[0])),
        "map_features": _extract_map_features(state),
        "sdc_paths": _extract_sdc_paths(state),
    }
    raw["metadata"]["num_sdc_paths"] = int(len(raw["sdc_paths"]))
    return raw


def canonical_scenario_from_waymax_state(state: Any, *, scenario_id: str | None = None, current_time_index: int | None = None) -> Any:
    return normalize_scenario(raw_scenario_from_waymax_state(state, scenario_id=scenario_id, current_time_index=current_time_index))


def save_raw_waymax_scenario_pickle(state: Any, *, out_path: str | Path, scenario_id: str | None = None, current_time_index: int | None = None) -> Path:
    payload = raw_scenario_from_waymax_state(state, scenario_id=scenario_id, current_time_index=current_time_index)
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path
