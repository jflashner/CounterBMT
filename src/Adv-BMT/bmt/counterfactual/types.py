from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, List["JsonValue"], Dict[str, "JsonValue"]]


def jsonify_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify_value(v) for v in value]
    return value


def stable_string_sort_key(value: Any) -> tuple[int, int, str]:
    text = str(value).strip()
    try:
        return (0, int(text), text)
    except Exception:
        return (1, 0, text)


@dataclass
class CanonicalTrack:
    object_type: str
    position_xy: np.ndarray
    position_xyz: np.ndarray
    heading: np.ndarray
    velocity_xy: np.ndarray
    valid: np.ndarray
    metadata: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class CanonicalTrafficLight:
    object_state: Tuple[Optional[str], ...]
    lane_ref: Optional[str]
    stop_point_xy: Optional[Tuple[float, float]]
    stop_point_xyz: Optional[Tuple[float, float, float]]
    metadata: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class CanonicalMapFeature:
    feature_type: str
    polyline_xy: np.ndarray
    polyline_xyz: np.ndarray
    polygon_xy: Optional[np.ndarray] = None
    polygon_xyz: Optional[np.ndarray] = None
    metadata: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class CanonicalSDCPath:
    path_id: str
    polyline_xy: np.ndarray
    polyline_xyz: np.ndarray
    valid: np.ndarray
    metadata: Dict[str, JsonValue] = field(default_factory=dict)


@dataclass
class CanonicalScenario:
    scenario_id: str
    length: int
    ts: np.ndarray
    sdc_id: str
    current_time_index: int
    tracks: Dict[str, CanonicalTrack]
    traffic_lights: Dict[str, CanonicalTrafficLight]
    map_features: Dict[str, CanonicalMapFeature]
    sdc_paths: Dict[str, CanonicalSDCPath] = field(default_factory=dict)
    metadata_summary: Dict[str, JsonValue] = field(default_factory=dict)
    objects_of_interest: List[str] = field(default_factory=list)
