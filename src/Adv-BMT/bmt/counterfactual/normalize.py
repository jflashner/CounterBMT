from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from .types import (
    CanonicalMapFeature,
    CanonicalScenario,
    CanonicalSDCPath,
    CanonicalTrack,
    CanonicalTrafficLight,
    JsonValue,
    stable_string_sort_key,
)

_DEFAULT_DT_S = 0.1


def _install_numpy_pickle_compat_aliases() -> None:
    """Alias NumPy 2.x pickle module paths for older NumPy envs.

    Some scenario pickles were created in environments that reference
    ``numpy._core`` modules. Older NumPy builds expose the same objects under
    ``numpy.core`` only, which makes plain ``pickle.load`` fail with
    ``ModuleNotFoundError``. Registering these aliases is harmless when the
    newer paths already exist and keeps old environments able to read the same
    artifacts.
    """

    try:
        import numpy.core as np_core
    except Exception:
        return

    sys.modules.setdefault("numpy._core", np_core)

    alias_pairs = (
        ("multiarray", "multiarray"),
        ("numeric", "numeric"),
        ("numerictypes", "numerictypes"),
        ("umath", "umath"),
        ("_multiarray_umath", "_multiarray_umath"),
    )
    for alias_name, attr_name in alias_pairs:
        target = getattr(np_core, attr_name, None)
        if target is not None:
            sys.modules.setdefault(f"numpy._core.{alias_name}", target)


def load_raw_scenario(path: str | Path) -> Any:
    _install_numpy_pickle_compat_aliases()
    with Path(path).expanduser().open("rb") as f:
        return pickle.load(f)


def load_and_normalize_scenario(path: str | Path) -> CanonicalScenario:
    return normalize_scenario(load_raw_scenario(path))


def normalize_scenario(raw_scenario: Mapping[str, Any]) -> CanonicalScenario:
    raw = dict(raw_scenario or {})
    metadata = _as_mapping(raw.get("metadata"))
    tracks_raw = _as_mapping(raw.get("tracks"))
    traffic_lights_raw = _as_mapping(raw.get("dynamic_map_states"))
    map_features_raw = _as_mapping(raw.get("map_features"))

    length = _infer_length(raw, metadata, tracks_raw, traffic_lights_raw)
    ts = _normalize_timestamps(metadata.get("ts"), length)
    scenario_id = _normalize_id(_first_not_none(raw.get("id"), metadata.get("scenario_id"), metadata.get("id")), default="unknown")

    tracks: Dict[str, CanonicalTrack] = {}
    for track_id in sorted(tracks_raw.keys(), key=stable_string_sort_key):
        key = _normalize_id(track_id)
        tracks[key] = _normalize_track(tracks_raw[track_id], length=length)

    sdc_id = _infer_sdc_id(metadata=metadata, tracks=tracks)
    current_time_index = _normalize_current_time_index(metadata.get("current_time_index"), length=length)

    traffic_lights: Dict[str, CanonicalTrafficLight] = {}
    for light_id in sorted(traffic_lights_raw.keys(), key=stable_string_sort_key):
        key = _normalize_id(light_id)
        traffic_lights[key] = _normalize_traffic_light(traffic_lights_raw[light_id], length=length)

    map_features: Dict[str, CanonicalMapFeature] = {}
    for feature_id in sorted(map_features_raw.keys(), key=stable_string_sort_key):
        key = _normalize_id(feature_id)
        map_features[key] = _normalize_map_feature(map_features_raw[feature_id])

    sdc_paths = _normalize_sdc_paths(raw.get("sdc_paths"))

    return CanonicalScenario(
        scenario_id=scenario_id,
        length=length,
        ts=ts,
        sdc_id=sdc_id,
        current_time_index=current_time_index,
        tracks=tracks,
        traffic_lights=traffic_lights,
        map_features=map_features,
        sdc_paths=sdc_paths,
        metadata_summary=_summarize_metadata(metadata),
        objects_of_interest=_normalize_id_list(metadata.get("objects_of_interest")),
    )


def _as_mapping(value: Any) -> Mapping[Any, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_id(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_id_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_id(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _infer_length(
    raw: Mapping[str, Any],
    metadata: Mapping[str, Any],
    tracks: Mapping[Any, Any],
    traffic_lights: Mapping[Any, Any],
) -> int:
    candidates: list[int] = []

    try:
        base_length = int(raw.get("length", 0))
    except Exception:
        base_length = 0
    if base_length > 0:
        candidates.append(base_length)

    ts = metadata.get("ts")
    if ts is not None:
        try:
            candidates.append(int(np.asarray(ts).reshape(-1).shape[0]))
        except Exception:
            pass

    for track in tracks.values():
        state = _as_mapping(_as_mapping(track).get("state"))
        for key in ("position", "heading", "velocity", "valid"):
            length = _time_axis_length(state.get(key))
            if length > 0:
                candidates.append(length)

    for light in traffic_lights.values():
        state = _as_mapping(_as_mapping(light).get("state"))
        length = _time_axis_length(state.get("object_state"))
        if length > 0:
            candidates.append(length)

    return max(candidates) if candidates else 0


def _time_axis_length(value: Any) -> int:
    if value is None:
        return 0
    try:
        arr = np.asarray(value)
    except Exception:
        return 0
    if arr.ndim == 0:
        return 0
    return int(arr.shape[0])


def _normalize_timestamps(ts_value: Any, length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)

    arr = np.zeros((0,), dtype=np.float32)
    if ts_value is not None:
        try:
            arr = np.asarray(ts_value, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.zeros((0,), dtype=np.float32)

    if arr.shape[0] == length:
        return arr.astype(np.float32, copy=False)

    dt = _DEFAULT_DT_S
    if arr.shape[0] >= 2:
        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size > 0:
            dt = float(np.median(diffs))

    start = float(arr[0]) if arr.shape[0] > 0 and np.isfinite(arr[0]) else 0.0
    out = start + np.arange(length, dtype=np.float32) * np.float32(dt)
    if arr.shape[0] > 0:
        prefix = min(arr.shape[0], length)
        out[:prefix] = arr[:prefix]
        if prefix < length:
            out[prefix:] = out[prefix - 1] + np.arange(1, length - prefix + 1, dtype=np.float32) * np.float32(dt)
    return out.astype(np.float32, copy=False)


def _normalize_current_time_index(value: Any, *, length: int) -> int:
    try:
        current = int(value)
    except Exception:
        current = 0
    if length <= 0:
        return 0
    return int(np.clip(current, 0, max(0, length - 1)))


def _infer_sdc_id(*, metadata: Mapping[str, Any], tracks: Mapping[str, CanonicalTrack]) -> str:
    sdc_id = _normalize_id(metadata.get("sdc_id"))
    if sdc_id and sdc_id in tracks:
        return sdc_id
    if tracks:
        return sorted(tracks.keys(), key=stable_string_sort_key)[0]
    return sdc_id


def _normalize_track(track_value: Any, *, length: int) -> CanonicalTrack:
    track = _as_mapping(track_value)
    state = _as_mapping(track.get("state"))
    metadata = _as_mapping(track.get("metadata"))

    position_xyz = _coerce_time_series_matrix(state.get("position"), length=length, width=3, fill=np.nan)
    position_xy = position_xyz[:, :2].copy()
    velocity_xy = _coerce_time_series_matrix(state.get("velocity"), length=length, width=2, fill=0.0)
    heading = _coerce_time_series_vector(state.get("heading"), length=length, fill=np.nan)
    valid = _coerce_valid_mask(state.get("valid"), length=length, fallback_xy=position_xy)
    object_type = _normalize_id(track.get("type") or metadata.get("type"), default="UNKNOWN")

    return CanonicalTrack(
        object_type=object_type,
        position_xy=position_xy,
        position_xyz=position_xyz,
        heading=heading,
        velocity_xy=velocity_xy,
        valid=valid,
        metadata=_summarize_small_mapping(metadata),
    )


def _normalize_traffic_light(light_value: Any, *, length: int) -> CanonicalTrafficLight:
    light = _as_mapping(light_value)
    state = _as_mapping(light.get("state"))
    metadata = _as_mapping(light.get("metadata"))

    stop_point_xyz = _coerce_point(light.get("stop_point"))
    stop_point_xy = None if stop_point_xyz is None else (stop_point_xyz[0], stop_point_xyz[1])

    return CanonicalTrafficLight(
        object_state=_coerce_traffic_light_states(state.get("object_state"), length=length),
        lane_ref=_normalize_optional_id(_first_not_none(light.get("lane"), light.get("lane_ref"))),
        stop_point_xy=stop_point_xy,
        stop_point_xyz=stop_point_xyz,
        metadata=_summarize_small_mapping(metadata),
    )


def _normalize_map_feature(feature_value: Any) -> CanonicalMapFeature:
    feature = _as_mapping(feature_value)
    polyline_xyz = _coerce_polyline(feature.get("polyline"))
    polygon_xyz = _coerce_polyline(feature.get("polygon"))
    polygon_xy = None if polygon_xyz.shape[0] == 0 else polygon_xyz[:, :2].copy()

    metadata: Dict[str, JsonValue] = {"raw_keys": sorted(str(k) for k in feature.keys())}
    for key, value in feature.items():
        if key in {"type", "polyline", "polygon"}:
            continue
        metadata[str(key)] = _summarize_value(value)

    return CanonicalMapFeature(
        feature_type=_normalize_id(feature.get("type"), default="UNKNOWN"),
        polyline_xy=polyline_xyz[:, :2].copy(),
        polyline_xyz=polyline_xyz,
        polygon_xy=polygon_xy,
        polygon_xyz=(None if polygon_xyz.shape[0] == 0 else polygon_xyz),
        metadata=metadata,
    )


def _normalize_sdc_paths(paths_value: Any) -> Dict[str, CanonicalSDCPath]:
    if isinstance(paths_value, Mapping):
        items = list(paths_value.items())
    elif isinstance(paths_value, Sequence) and not isinstance(paths_value, (str, bytes)):
        items = [(str(index), value) for index, value in enumerate(paths_value)]
    else:
        return {}

    normalized: Dict[str, CanonicalSDCPath] = {}
    for raw_key, raw_value in items:
        path_key = _normalize_id(_first_not_none(_as_mapping(raw_value).get("path_id"), raw_key))
        if not path_key:
            continue
        path = _normalize_sdc_path(raw_value, default_path_id=path_key)
        if path is not None:
            normalized[path_key] = path
    return normalized


def _normalize_sdc_path(path_value: Any, *, default_path_id: str) -> Optional[CanonicalSDCPath]:
    path = _as_mapping(path_value)
    path_id = _normalize_id(_first_not_none(path.get("path_id"), default_path_id), default=default_path_id)
    polyline_xyz = _coerce_polyline(
        _first_not_none(
            path.get("polyline_xyz"),
            path.get("polyline"),
            path.get("points_xyz"),
            path.get("points"),
            path.get("xyz"),
        )
    )
    if polyline_xyz.shape[0] < 2:
        return None
    valid_raw = path.get("valid")
    if valid_raw is None:
        valid = np.ones((polyline_xyz.shape[0],), dtype=bool)
    else:
        valid = np.asarray(valid_raw, dtype=bool).reshape(-1)
        if valid.shape[0] != polyline_xyz.shape[0]:
            adjusted = np.zeros((polyline_xyz.shape[0],), dtype=bool)
            prefix = min(valid.shape[0], polyline_xyz.shape[0])
            adjusted[:prefix] = valid[:prefix]
            if prefix < polyline_xyz.shape[0]:
                adjusted[prefix:] = True
            valid = adjusted

    metadata: Dict[str, JsonValue] = {"raw_keys": sorted(str(k) for k in path.keys())}
    for key, value in path.items():
        if key in {"path_id", "polyline_xyz", "polyline", "points_xyz", "points", "xyz", "valid"}:
            continue
        metadata[str(key)] = _summarize_value(value)

    return CanonicalSDCPath(
        path_id=path_id,
        polyline_xy=polyline_xyz[:, :2].copy(),
        polyline_xyz=polyline_xyz,
        valid=valid,
        metadata=metadata,
    )


def _coerce_time_series_matrix(value: Any, *, length: int, width: int, fill: float) -> np.ndarray:
    out = np.full((length, width), fill_value=np.float32(fill), dtype=np.float32)
    if length <= 0 or value is None:
        return out

    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return out

    if arr.ndim == 1:
        if width == 1:
            arr = arr.reshape(-1, 1)
        elif arr.size % width == 0 and arr.size > 0:
            arr = arr.reshape(-1, width)
        else:
            return out
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2:
        return out

    rows = min(length, int(arr.shape[0]))
    cols = min(width, int(arr.shape[1]))
    if rows > 0 and cols > 0:
        out[:rows, :cols] = arr[:rows, :cols]
    return out


def _coerce_time_series_vector(value: Any, *, length: int, fill: float) -> np.ndarray:
    return _coerce_time_series_matrix(value, length=length, width=1, fill=fill).reshape(length)


def _coerce_valid_mask(value: Any, *, length: int, fallback_xy: np.ndarray) -> np.ndarray:
    if value is not None:
        try:
            arr = np.asarray(value).reshape(-1)
            out = np.zeros((length,), dtype=bool)
            rows = min(length, int(arr.shape[0]))
            if rows > 0:
                out[:rows] = arr[:rows].astype(bool)
            return out
        except Exception:
            pass
    if fallback_xy.shape[0] == 0:
        return np.zeros((length,), dtype=bool)
    return np.isfinite(fallback_xy).all(axis=-1)


def _normalize_traffic_light_state(value: Any) -> Optional[str]:
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


def _coerce_traffic_light_states(value: Any, *, length: int) -> tuple[Optional[str], ...]:
    out: list[Optional[str]] = [None] * max(length, 0)
    if length <= 0 or value is None:
        return tuple(out)

    if isinstance(value, np.ndarray):
        items: Iterable[Any] = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        items = [value]

    for idx, item in enumerate(items):
        if idx >= length:
            break
        out[idx] = _normalize_traffic_light_state(item)
    return tuple(out)


def _coerce_point(value: Any) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] < 2:
        return None
    xyz = np.full((3,), np.nan, dtype=np.float32)
    xyz[: min(3, arr.shape[0])] = arr[: min(3, arr.shape[0])]
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


def _coerce_polyline(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)

    if arr.ndim == 1:
        if arr.size % 3 == 0 and arr.size > 0:
            arr = arr.reshape(-1, 3)
        elif arr.size % 2 == 0 and arr.size > 0:
            arr = arr.reshape(-1, 2)
        else:
            return np.zeros((0, 3), dtype=np.float32)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    out = np.full((int(arr.shape[0]), 3), np.nan, dtype=np.float32)
    cols = min(3, int(arr.shape[1]))
    out[:, :cols] = arr[:, :cols]
    return out


def _normalize_optional_id(value: Any) -> Optional[str]:
    text = _normalize_id(value)
    return text or None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _summarize_metadata(metadata: Mapping[str, Any]) -> Dict[str, JsonValue]:
    tracks_to_predict = _as_mapping(metadata.get("tracks_to_predict"))
    return {
        "metadata_keys": sorted(str(k) for k in metadata.keys()),
        "dataset": _normalize_optional_id(metadata.get("dataset")),
        "coordinate": _normalize_optional_id(metadata.get("coordinate")),
        "source_file": _normalize_optional_id(metadata.get("source_file")),
        "metadrive_processed": bool(metadata.get("metadrive_processed", False)),
        "track_length": _safe_int(metadata.get("track_length")),
        "sdc_track_index": _safe_int(metadata.get("sdc_track_index")),
        "num_tracks_to_predict": int(len(tracks_to_predict)),
        "tracks_to_predict_ids": sorted((_normalize_id(k) for k in tracks_to_predict.keys() if _normalize_id(k)), key=stable_string_sort_key),
        "number_summary": _summarize_value(metadata.get("number_summary")),
        "object_summary_size": int(len(metadata.get("object_summary", {}))) if isinstance(metadata.get("object_summary"), Mapping) else 0,
    }


def _summarize_small_mapping(mapping: Mapping[str, Any]) -> Dict[str, JsonValue]:
    out: Dict[str, JsonValue] = {}
    for key, value in mapping.items():
        if key == "ts":
            continue
        out[str(key)] = _summarize_value(value)
    return out


def _summarize_value(value: Any) -> JsonValue:
    if isinstance(value, np.ndarray):
        return {
            "shape": [int(v) for v in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _summarize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_summarize_value(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_summarize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None
