#!/usr/bin/env python3
"""Render paper-ready top-down Waymax/ScenarioNet scene graphs.

The default path renders logged trajectories directly from raw ScenarioNet-style
pickle files. Passing ``--rollout-source checkpoint`` additionally runs a BMT
checkpoint autoregressively and overlays the generated trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml"
CHECKPOINT_ALIAS_PATHS = {
    "prog": REPO_ROOT / "outputs" / "remote_checkpoints" / "paper_defaults" / "prog.ckpt",
    "topo": REPO_ROOT / "outputs" / "remote_checkpoints" / "paper_defaults" / "topo.ckpt",
}
SEMANTIC_INTERVENTION_LABELS = (
    "left",
    "right",
    "left_lane_change",
    "right_lane_change",
    "straight",
    "stop",
)
TD3_ADVERSARY_SEMANTIC_LABELS = (
    "left",
    "right",
    "left_lane_change",
    "right_lane_change",
)

LANE_COLOR = "#CBD5E1"
ROAD_COLOR = "#334155"
CROSSWALK_COLOR = "#E2E8F0"
BACKGROUND_COLOR = "#F8FAFC"
AGENT_FILL = "#D1D5DB"
AGENT_EDGE = "#334155"
AGENT_HISTORY = "#94A3B8"
EGO_COLOR = "#16A34A"
ADVERSARY_COLOR = "#DC2626"
HIGHLIGHT_COLOR = "#2563EB"
LOGGED_TRAJ_COLOR = "#16A34A"
GENERATED_TRAJ_COLOR = "#111827"
INTERVENTION_TRAJ_COLOR = EGO_COLOR
SIGNAL_RED = "#DC2626"
SIGNAL_YELLOW = "#F59E0B"
SIGNAL_GREEN = "#16A34A"
SIGNAL_UNKNOWN = "#64748B"

np = None
plt = None
MplPolygon = None


@dataclass(frozen=True)
class SceneSelection:
    requested: str
    path: Path
    scenario_id: str
    raw: Mapping[str, Any]


@dataclass
class CheckpointRunner:
    config: Any
    module: Any
    tokenizer: Any
    extract_model_frame: Any
    build_control_sample: Any
    build_time_window_mask: Any
    normalize_semantic_label: Any
    preprocess_raw_scenario_for_forward_supervision: Any
    summarize_forward_supervision_for_sample: Any
    load_report: Mapping[str, Any]


def _ensure_repo_imports() -> None:
    paths = [
        REPO_ROOT / "scenarionet",
        REPO_ROOT / "metadrive",
        REPO_ROOT / "src",
        REPO_ROOT,
        REPO_ROOT / "src" / "Adv-BMT",
    ]
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _require_plot_deps() -> None:
    global np, plt, MplPolygon
    if np is not None and plt is not None and MplPolygon is not None:
        return
    import numpy as _np
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as _plt
    from matplotlib.patches import Polygon as _Polygon

    np = _np
    plt = _plt
    MplPolygon = _Polygon


def _expand(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _resolve_checkpoint_path(value: str) -> Tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "", ""
    alias_path = CHECKPOINT_ALIAS_PATHS.get(text.lower())
    if alias_path is not None:
        return text.lower(), str(alias_path.resolve())
    return "", str(_expand(text))


def _safe_slug(text: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return slug.strip("_") or "scene"


def _read_json(path: str | Path) -> Any:
    return json.loads(_expand(path).read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_default(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    output = _expand(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_default(payload), indent=2, sort_keys=True), encoding="utf-8")


def _install_numpy_pickle_compat_aliases() -> None:
    """Allow NumPy-2-written scenario pickles to load in NumPy-1 envs."""

    try:
        import numpy._core as np_core
    except Exception:
        try:
            import numpy.core as np_core
        except Exception:
            return

    sys.modules.setdefault("numpy._core", np_core)
    for alias_name in ("multiarray", "numeric", "numerictypes", "umath", "_multiarray_umath"):
        target = getattr(np_core, alias_name, None)
        if target is not None:
            sys.modules.setdefault(f"numpy._core.{alias_name}", target)


def _load_pickle(path: Path) -> Mapping[str, Any]:
    _install_numpy_pickle_compat_aliases()
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a mapping in {path}, got {type(payload).__name__}")
    return payload


def _scenario_id_from_path(path: Path, raw: Optional[Mapping[str, Any]] = None) -> str:
    for text in [path.stem, str(path)]:
        match = re.search(r"waymax_scene_\d+", text)
        if match:
            return match.group(0)
    if raw is not None:
        metadata = dict(raw.get("metadata", {}) or {})
        for value in (metadata.get("scenario_id"), raw.get("id")):
            if value:
                return str(value)
    return path.stem.removeprefix("sd_waymo_v1.3.1_").removeprefix("sd_waymo_v1.2_")


def _scene_name_candidates(scene: str) -> List[str]:
    text = str(scene).strip()
    if text.isdigit():
        return [f"waymax_scene_{int(text):05d}"]
    candidates = [text]
    match = re.search(r"waymax_scene_\d+", text)
    if match:
        candidates.append(match.group(0))
    return list(dict.fromkeys(candidates))


def _resolve_scene_path(scene: str, roots: Sequence[Path]) -> Path:
    direct = Path(scene).expanduser()
    if direct.exists():
        direct = direct.resolve()
        if direct.is_dir():
            candidates = sorted(p for p in direct.rglob("sd_*.pkl") if p.name != "dataset_summary.pkl")
            if not candidates:
                raise FileNotFoundError(f"No sd_*.pkl scenario file found under {direct}")
            return candidates[0]
        return direct

    search_roots = [root for root in roots if root.exists()]
    if not search_roots:
        raise FileNotFoundError(
            f"Could not resolve scene {scene!r}; none of the search roots exist: {[str(r) for r in roots]}"
        )

    name_candidates = _scene_name_candidates(scene)
    matches: List[Path] = []
    for root in search_roots:
        for name in name_candidates:
            patterns = [
                f"sd_*{name}*.pkl",
                f"*{name}*/sd_*.pkl",
                f"*{name}*.pkl",
            ]
            for pattern in patterns:
                matches.extend(
                    path
                    for path in root.rglob(pattern)
                    if path.is_file() and path.suffix == ".pkl" and path.name != "dataset_summary.pkl"
                )
    unique = sorted(set(path.resolve() for path in matches), key=lambda p: (len(str(p)), str(p)))
    if not unique:
        raise FileNotFoundError(f"Could not resolve scene {scene!r} from roots {[str(r) for r in search_roots]}")
    return unique[0]


def _load_scene(scene: str, roots: Sequence[Path]) -> SceneSelection:
    path = _resolve_scene_path(scene, roots)
    raw = _load_pickle(path)
    return SceneSelection(requested=str(scene), path=path, scenario_id=_scenario_id_from_path(path, raw), raw=raw)


def _state_array(track: Mapping[str, Any], key: str, *, dtype: Any = None):
    _require_plot_deps()
    state = dict(track.get("state", {}) or {})
    value = state.get(key, [])
    return np.asarray(value, dtype=dtype)


def _first_metadata_value(metadata: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            return metadata[key]
    return default


def _track_valid(raw: Mapping[str, Any], track_id: str):
    track = dict(raw.get("tracks", {}).get(str(track_id), {}) or {})
    valid = _state_array(track, "valid", dtype=bool).reshape(-1)
    return valid


def _infer_num_steps(raw: Mapping[str, Any]) -> int:
    _require_plot_deps()
    max_steps = 0
    for track in dict(raw.get("tracks", {}) or {}).values():
        valid = np.asarray(dict(track.get("state", {}) or {}).get("valid", []), dtype=bool).reshape(-1)
        max_steps = max(max_steps, int(valid.shape[0]))
    metadata = dict(raw.get("metadata", {}) or {})
    ts = _first_metadata_value(metadata, ("ts", "timestamps_seconds"), [])
    try:
        max_steps = max(max_steps, int(len(ts)))
    except Exception:
        pass
    return max(max_steps, 1)


def _timestamps(raw: Mapping[str, Any], num_steps: int):
    _require_plot_deps()
    metadata = dict(raw.get("metadata", {}) or {})
    ts = _first_metadata_value(metadata, ("ts", "timestamps_seconds"), [])
    array = np.asarray(ts, dtype=np.float64).reshape(-1)
    if array.size >= num_steps:
        return array[:num_steps]
    if array.size >= 2:
        dt = float(np.nanmedian(np.diff(array)))
        if not math.isfinite(dt) or dt <= 0:
            dt = 0.1
    else:
        dt = float(metadata.get("dt_s", 0.1) or 0.1)
    return np.arange(num_steps, dtype=np.float64) * dt


def _parse_until(raw: Mapping[str, Any], *, until: str, until_step: Optional[int], until_s: Optional[float]) -> int:
    _require_plot_deps()
    num_steps = _infer_num_steps(raw)
    ts = _timestamps(raw, num_steps)
    if until_step is not None:
        return int(np.clip(int(until_step), 0, num_steps - 1))
    if until_s is not None:
        return int(np.argmin(np.abs(ts - float(until_s))))
    text = str(until or "last").strip().lower()
    if text in {"last", "end", "final"}:
        return num_steps - 1
    if text.endswith("s"):
        seconds = float(text[:-1])
        return int(np.argmin(np.abs(ts - seconds)))
    if "." in text:
        seconds = float(text)
        return int(np.argmin(np.abs(ts - seconds)))
    return int(np.clip(int(text), 0, num_steps - 1))


def _find_sdc_id(raw: Mapping[str, Any], explicit: str = "") -> str:
    if explicit and explicit.lower() not in {"sdc", "ego"}:
        return str(explicit)
    metadata = dict(raw.get("metadata", {}) or {})
    if metadata.get("sdc_id") is not None:
        return str(metadata["sdc_id"])
    for track_id, track in dict(raw.get("tracks", {}) or {}).items():
        track_meta = dict(track.get("metadata", {}) or {})
        if bool(track_meta.get("is_sdc", False)):
            return str(track_id)
    tracks = list(dict(raw.get("tracks", {}) or {}).keys())
    if not tracks:
        raise ValueError("Raw scenario has no tracks.")
    return str(tracks[0])


def _nearest_valid_index(valid, requested_idx: int) -> Optional[int]:
    _require_plot_deps()
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if mask.size == 0:
        return None
    idx = int(np.clip(int(requested_idx), 0, mask.size - 1))
    if bool(mask[idx]):
        return idx
    before = np.flatnonzero(mask[: idx + 1])
    if before.size:
        return int(before[-1])
    after = np.flatnonzero(mask[idx:])
    if after.size:
        return int(idx + after[0])
    return None


def _track_pose(raw: Mapping[str, Any], track_id: str, time_idx: int) -> Optional[Dict[str, float]]:
    _require_plot_deps()
    track = dict(raw.get("tracks", {}).get(str(track_id), {}) or {})
    if not track:
        return None
    position = _state_array(track, "position", dtype=np.float64)
    heading = _state_array(track, "heading", dtype=np.float64).reshape(-1)
    valid = _state_array(track, "valid", dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[0] == 0:
        return None
    idx = _nearest_valid_index(valid, time_idx)
    if idx is None or idx >= position.shape[0]:
        return None
    h = float(heading[idx]) if idx < heading.shape[0] and math.isfinite(float(heading[idx])) else 0.0
    return {"x": float(position[idx, 0]), "y": float(position[idx, 1]), "heading": h, "index": int(idx)}


def _track_dimensions(track: Mapping[str, Any], idx: int) -> Tuple[float, float]:
    _require_plot_deps()
    state = dict(track.get("state", {}) or {})
    for length_key, width_key in (("length", "width"), ("bbox_length", "bbox_width")):
        length = np.asarray(state.get(length_key, []), dtype=np.float64).reshape(-1)
        width = np.asarray(state.get(width_key, []), dtype=np.float64).reshape(-1)
        if idx < length.shape[0] and idx < width.shape[0] and length[idx] > 0 and width[idx] > 0:
            return float(length[idx]), float(width[idx])
    track_type = str(track.get("type", "VEHICLE")).upper()
    if "PEDESTRIAN" in track_type:
        return 0.9, 0.9
    if "CYCL" in track_type or "BIKE" in track_type:
        return 1.9, 0.8
    return 4.7, 2.1


def _world_to_sdc_up(points_xy: Any, *, center_xy: Any, heading_rad: float):
    _require_plot_deps()
    xy = np.asarray(points_xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)
    if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    xy = xy[:, :2]
    mask = np.isfinite(xy).all(axis=-1)
    xy = xy[mask]
    if xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = xy - np.asarray(center_xy, dtype=np.float64).reshape(1, 2)
    rot = (math.pi / 2.0) - float(heading_rad)
    c = math.cos(rot)
    s = math.sin(rot)
    x_new = c * centered[:, 0] - s * centered[:, 1]
    y_new = s * centered[:, 0] + c * centered[:, 1]
    return np.stack([x_new, y_new], axis=-1)


def _world_to_view(points_xy: Any, *, view: str, center_xy: Any, heading_rad: float):
    _require_plot_deps()
    if str(view) == "world":
        xy = np.asarray(points_xy, dtype=np.float64)
        if xy.ndim == 1:
            xy = xy.reshape(1, -1)
        if xy.ndim != 2 or xy.shape[0] == 0 or xy.shape[1] < 2:
            return np.zeros((0, 2), dtype=np.float64)
        xy = xy[:, :2]
        return xy[np.isfinite(xy).all(axis=-1)]
    return _world_to_sdc_up(points_xy, center_xy=center_xy, heading_rad=heading_rad)


def _heading_to_view(heading_rad: float, *, view: str, center_xy: Any, heading_ref_rad: float) -> float:
    _require_plot_deps()
    if str(view) == "world":
        return float(heading_rad)
    origin = np.asarray(center_xy, dtype=np.float64).reshape(2)
    tip = origin + np.asarray([math.cos(float(heading_rad)), math.sin(float(heading_rad))], dtype=np.float64)
    local = _world_to_sdc_up(np.stack([origin, tip], axis=0), center_xy=origin, heading_rad=heading_ref_rad)
    delta = local[1] - local[0]
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _finite_xy(points_xy: Any):
    _require_plot_deps()
    xy = np.asarray(points_xy, dtype=np.float64)
    if xy.ndim == 1:
        xy = xy.reshape(1, -1)
    if xy.ndim != 2 or xy.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    xy = xy[:, :2]
    return xy[np.isfinite(xy).all(axis=-1)]


def _feature_xy(feature: Mapping[str, Any]):
    if "polyline" in feature:
        return _finite_xy(feature["polyline"])
    if "polygon" in feature:
        return _finite_xy(feature["polygon"])
    return _finite_xy([])


def _draw_map(ax: Any, raw: Mapping[str, Any], *, view: str, center_xy: Any, heading_rad: float, radius_m: float) -> None:
    _require_plot_deps()
    view_limit = float(radius_m) * 1.35
    for feature_id, feature in sorted(dict(raw.get("map_features", {}) or {}).items(), key=lambda item: str(item[0])):
        feature = dict(feature or {})
        xy_world = _feature_xy(feature)
        if xy_world.shape[0] < 2:
            continue
        local = _world_to_view(xy_world, view=view, center_xy=center_xy, heading_rad=heading_rad)
        if local.shape[0] < 2:
            continue
        if str(view) != "world" and not bool(np.any((np.abs(local[:, 0]) <= view_limit) & (np.abs(local[:, 1]) <= view_limit))):
            continue
        feature_type = str(feature.get("type", "")).upper()
        if feature_type == "CROSSWALK" or "CROSSWALK" in feature_type:
            if local.shape[0] >= 3:
                ax.fill(local[:, 0], local[:, 1], color=CROSSWALK_COLOR, alpha=0.36, zorder=1)
        elif feature_type.startswith("ROAD_EDGE") or feature_type.startswith("ROAD_LINE"):
            ax.plot(local[:, 0], local[:, 1], color=ROAD_COLOR, linewidth=2.2, alpha=0.82, zorder=2)
        elif feature_type.startswith("LANE") or feature_type == "DRIVEWAY":
            ax.plot(local[:, 0], local[:, 1], color=LANE_COLOR, linewidth=1.0, alpha=0.42, zorder=3)
        else:
            ax.plot(local[:, 0], local[:, 1], color="#94A3B8", linewidth=0.8, alpha=0.25, zorder=2)


def _traffic_signal_state(dynamic_state: Mapping[str, Any], time_idx: int) -> str:
    state_payload = dynamic_state.get("state", {})
    state_sources = []
    if isinstance(state_payload, Mapping):
        state_sources.extend(
            state_payload.get(key)
            for key in ("object_state", "state", "lane_state", "traffic_light_state")
        )
    state_sources.extend(dynamic_state.get(key) for key in ("object_state", "lane_state", "traffic_light_state"))
    for values in state_sources:
        if values is None:
            continue
        arr = np.asarray(values).reshape(-1)
        if arr.size <= 0:
            continue
        idx = int(np.clip(int(time_idx), 0, arr.size - 1))
        value = arr[idx]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        return str(value)
    return "LANE_STATE_UNKNOWN"


def _traffic_signal_color(state: str) -> str:
    text = str(state or "").upper()
    if "STOP" in text or "RED" in text:
        return SIGNAL_RED
    if "CAUTION" in text or "YELLOW" in text:
        return SIGNAL_YELLOW
    if "GO" in text or "GREEN" in text:
        return SIGNAL_GREEN
    return SIGNAL_UNKNOWN


def _draw_traffic_signals(
    ax: Any,
    raw: Mapping[str, Any],
    *,
    time_idx: int,
    view: str,
    center_xy: Any,
    heading_rad: float,
    radius_m: float,
    label: bool = False,
) -> None:
    _require_plot_deps()
    view_limit = float(radius_m) * 1.35
    for light_id, dynamic_state in sorted(dict(raw.get("dynamic_map_states", {}) or {}).items(), key=lambda item: str(item[0])):
        dynamic_state = dict(dynamic_state or {})
        if "TRAFFIC_LIGHT" not in str(dynamic_state.get("type", "TRAFFIC_LIGHT")).upper():
            continue
        stop_point = dynamic_state.get("stop_point", dynamic_state.get("stop_point_xy", None))
        if stop_point is None:
            continue
        point_world = _finite_xy([stop_point])
        if point_world.shape[0] == 0:
            continue
        point_view = _world_to_view(point_world, view=view, center_xy=center_xy, heading_rad=heading_rad)
        if point_view.shape[0] == 0:
            continue
        x, y = float(point_view[0, 0]), float(point_view[0, 1])
        if str(view) != "world" and (abs(x) > view_limit or abs(y) > view_limit):
            continue
        state = _traffic_signal_state(dynamic_state, int(time_idx))
        color = _traffic_signal_color(state)
        ax.scatter(
            [x],
            [y],
            s=54,
            marker="o",
            facecolor=color,
            edgecolor="#F8FAFC",
            linewidth=1.2,
            alpha=0.96,
            zorder=12,
        )
        ax.scatter(
            [x],
            [y],
            s=74,
            marker="o",
            facecolor="none",
            edgecolor="#111827",
            linewidth=0.7,
            alpha=0.52,
            zorder=11,
        )
        if label:
            lane = dynamic_state.get("lane", "")
            label_text = str(lane or light_id)
            ax.text(
                x + 1.2,
                y + 1.2,
                label_text,
                fontsize=5.5,
                color="#111827",
                alpha=0.78,
                zorder=13,
            )


def _box_world(center_xy: Any, heading_rad: float, length_m: float, width_m: float):
    _require_plot_deps()
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    forward = np.asarray([math.cos(float(heading_rad)), math.sin(float(heading_rad))], dtype=np.float64)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    half_l = 0.5 * max(0.4, float(length_m))
    half_w = 0.5 * max(0.3, float(width_m))
    return np.stack(
        [
            center + half_l * forward + half_w * left,
            center + half_l * forward - half_w * left,
            center - half_l * forward - half_w * left,
            center - half_l * forward + half_w * left,
        ],
        axis=0,
    )


def _draw_agent_box(
    ax: Any,
    *,
    raw: Mapping[str, Any],
    track_id: str,
    time_idx: int,
    view: str,
    center_xy: Any,
    heading_rad: float,
    fill_color: str,
    edge_color: str,
    alpha: float,
    linewidth: float,
    zorder: int,
    label: str = "",
    pose_override: Optional[Mapping[str, float]] = None,
) -> bool:
    _require_plot_deps()
    track = dict(raw.get("tracks", {}).get(str(track_id), {}) or {})
    pose = dict(pose_override) if pose_override is not None else _track_pose(raw, str(track_id), time_idx)
    if pose is None:
        return False
    logged_pose = _track_pose(raw, str(track_id), time_idx)
    dim_idx = int(logged_pose["index"]) if logged_pose is not None else int(pose.get("index", time_idx))
    length_m, width_m = _track_dimensions(track, dim_idx)
    polygon_world = _box_world([pose["x"], pose["y"]], float(pose["heading"]), length_m, width_m)
    polygon_view = _world_to_view(polygon_world, view=view, center_xy=center_xy, heading_rad=heading_rad)
    if polygon_view.shape[0] < 3:
        return False
    patch = MplPolygon(
        polygon_view,
        closed=True,
        facecolor=fill_color,
        edgecolor=edge_color,
        linewidth=float(linewidth),
        alpha=float(alpha),
        zorder=int(zorder),
    )
    ax.add_patch(patch)
    heading_view = _heading_to_view(float(pose["heading"]), view=view, center_xy=center_xy, heading_ref_rad=heading_rad)
    box_center = _world_to_view([[pose["x"], pose["y"]]], view=view, center_xy=center_xy, heading_rad=heading_rad)
    if box_center.shape[0]:
        arrow_len = min(3.8, max(1.2, 0.55 * float(length_m)))
        ax.plot(
            [box_center[0, 0], box_center[0, 0] + arrow_len * math.cos(heading_view)],
            [box_center[0, 1], box_center[0, 1] + arrow_len * math.sin(heading_view)],
            color=edge_color,
            linewidth=max(1.0, float(linewidth) * 0.75),
            alpha=min(1.0, float(alpha) + 0.12),
            zorder=int(zorder) + 1,
        )
        if label:
            ax.text(
                float(box_center[0, 0]) + 0.8,
                float(box_center[0, 1]) + 0.8,
                str(label),
                fontsize=7.0,
                color=edge_color,
                zorder=int(zorder) + 2,
            )
    return True


def _track_xy(raw: Mapping[str, Any], track_id: str, *, end_idx: int, start_idx: int = 0):
    _require_plot_deps()
    track = dict(raw.get("tracks", {}).get(str(track_id), {}) or {})
    if not track:
        return np.zeros((0, 2), dtype=np.float64)
    position = _state_array(track, "position", dtype=np.float64)
    valid = _state_array(track, "valid", dtype=bool).reshape(-1)
    if position.ndim != 2 or position.shape[0] == 0 or valid.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    lo = int(np.clip(int(start_idx), 0, position.shape[0] - 1))
    hi = int(np.clip(int(end_idx), lo, position.shape[0] - 1))
    valid_slice = valid[lo : hi + 1]
    xy = position[lo : hi + 1, :2]
    return _finite_xy(xy[valid_slice[: xy.shape[0]]])


def _draw_polyline(ax: Any, xy: Any, *, color: str, linewidth: float, alpha: float, linestyle: Any, zorder: int, label: str = "") -> None:
    _require_plot_deps()
    points = _finite_xy(xy)
    if points.shape[0] < 2:
        return
    jumps = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    split_after = np.flatnonzero(jumps > 8.0).tolist()
    start = 0
    first = True
    for split_idx in split_after + [points.shape[0] - 1]:
        end = int(split_idx) + 1
        seg = points[start:end]
        if seg.shape[0] >= 2:
            ax.plot(
                seg[:, 0],
                seg[:, 1],
                color=color,
                linewidth=float(linewidth),
                alpha=float(alpha),
                linestyle=linestyle,
                zorder=int(zorder),
                label=label if first and label else None,
                solid_capstyle="round",
            )
            first = False
        start = end


def _visible_track_ids(
    raw: Mapping[str, Any],
    *,
    center_xy: Any,
    time_idx: int,
    radius_m: float,
    agent_limit: int,
    required_ids: Iterable[str],
) -> List[str]:
    _require_plot_deps()
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    rows: List[Tuple[float, str]] = []
    for track_id in dict(raw.get("tracks", {}) or {}).keys():
        pose = _track_pose(raw, str(track_id), time_idx)
        if pose is None:
            continue
        dist = float(np.linalg.norm(np.asarray([pose["x"], pose["y"]], dtype=np.float64) - center))
        if dist <= float(radius_m):
            rows.append((dist, str(track_id)))
    rows.sort(key=lambda item: (item[0], item[1]))
    ordered = [track_id for _, track_id in rows[: max(1, int(agent_limit))]]
    for required in required_ids:
        if required and required not in ordered and _track_pose(raw, str(required), time_idx) is not None:
            ordered.append(str(required))
    return ordered


def _set_fitted_limits(ax: Any, points_list: Sequence[Any], *, padding_m: float) -> None:
    _require_plot_deps()
    finite_points = []
    for points in points_list:
        xy = _finite_xy(points)
        if xy.shape[0] > 0:
            finite_points.append(xy)
    if not finite_points:
        return
    merged = np.concatenate(finite_points, axis=0)
    x_min, y_min = np.min(merged, axis=0)
    x_max, y_max = np.max(merged, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 12.0)
    pad = max(float(padding_m), 0.08 * span)
    x_mid = 0.5 * float(x_min + x_max)
    y_mid = 0.5 * float(y_min + y_max)
    half = 0.5 * span + pad
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)


def _extract_json_trajectory(payload: Any, *, scene_id: str, agent_id: str):
    _require_plot_deps()
    if payload is None:
        return None
    candidate = payload
    if isinstance(payload, Mapping):
        candidate = payload.get(scene_id, payload.get(str(agent_id), payload))
        if isinstance(candidate, Mapping):
            candidate = candidate.get(str(agent_id), candidate.get("trajectory", candidate.get("xy", candidate)))
    points = _finite_xy(candidate)
    return None if points.shape[0] < 2 else points


def _build_checkpoint_runner(args: argparse.Namespace) -> CheckpointRunner:
    _ensure_repo_imports()
    try:
        from scripts.counterfactual.probe_agent_semantic_rollout import (
            _build_control_sample,
            _build_eval_module,
            _build_time_window_mask,
            _load_config,
            _load_model,
            _optional_positive_float,
            _resolve_device,
        )
        from bmt.counterfactual.forward_supervision import (
            preprocess_raw_scenario_for_forward_supervision,
            summarize_forward_supervision_for_sample,
        )
        from bmt.counterfactual.sdc_semantic_control import extract_model_frame
        from bmt.counterfactual.sdc_path_control import normalize_semantic_label
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or exc)
        raise RuntimeError(
            "--rollout-source checkpoint requires the full BMT evaluation environment "
            f"(missing module: {missing}). Use --rollout-source logged for renderer-only figures, "
            "or run checkpoint overlays in the environment used for model evaluation."
        ) from exc

    if not args.checkpoint:
        raise ValueError("--checkpoint is required when --rollout-source checkpoint")
    if not Path(str(args.checkpoint)).expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    config_args = argparse.Namespace(
        config=args.config,
        ckpt=args.checkpoint,
        teacher_ckpt=args.teacher_checkpoint or args.checkpoint,
    )
    config = _load_config(config_args)
    device = _resolve_device(args.device)
    model, load_report = _load_model(config=config, ckpt_path=args.checkpoint, load_mode=args.load_mode)
    model = model.to(device)
    module, tokenizer = _build_eval_module(
        config=config,
        ckpt_path=args.checkpoint,
        device=device,
        save_path=Path(args.outdir).expanduser().resolve() / "unused_eval_metrics",
        model=model,
    )
    if str(args.sampling_method).strip():
        module.config.SAMPLING.SAMPLING_METHOD = str(args.sampling_method)
    temperature = _optional_positive_float(float(args.temperature))
    topp = _optional_positive_float(float(args.topp))
    if temperature is not None:
        module.config.SAMPLING.TEMPERATURE = float(temperature)
    if topp is not None:
        module.config.SAMPLING.TOPP = float(topp)
    return CheckpointRunner(
        config=config,
        module=module,
        tokenizer=tokenizer,
        extract_model_frame=extract_model_frame,
        build_control_sample=_build_control_sample,
        build_time_window_mask=_build_time_window_mask,
        normalize_semantic_label=normalize_semantic_label,
        preprocess_raw_scenario_for_forward_supervision=preprocess_raw_scenario_for_forward_supervision,
        summarize_forward_supervision_for_sample=summarize_forward_supervision_for_sample,
        load_report=load_report,
    )


def _model_to_world(points_model_xy: Any, *, map_center_world: Any, map_heading_world: float):
    _require_plot_deps()
    xy = _finite_xy(points_model_xy)
    if xy.shape[0] == 0:
        return xy
    center = np.asarray(map_center_world, dtype=np.float64).reshape(-1)
    if float(map_heading_world) == 0.0:
        return xy + center[:2].reshape(1, 2)
    c = math.cos(float(map_heading_world))
    s = math.sin(float(map_heading_world))
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(center[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(center[1])
    return np.stack([x_world, y_world], axis=-1)


def _normalize_intervention_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"", "none", "off", "false", "baseline", "logged"}:
        return ""
    aliases = {
        "left_lane": "left_lane_change",
        "right_lane": "right_lane_change",
        "lane_left": "left_lane_change",
        "lane_right": "right_lane_change",
        "leftchange": "left_lane_change",
        "rightchange": "right_lane_change",
    }
    text = aliases.get(text, text)
    if text == "all":
        return "all"
    if text not in SEMANTIC_INTERVENTION_LABELS:
        allowed = ", ".join(SEMANTIC_INTERVENTION_LABELS)
        raise ValueError(f"Unknown intervention label {value!r}. Expected one of: {allowed}, none, all.")
    return text


def _requested_intervention_labels(args: argparse.Namespace) -> List[str]:
    raw_values = [str(value) for value in (getattr(args, "intervention", []) or [])]
    labels: List[str] = []
    use_all = bool(getattr(args, "all_interventions", False))
    for value in raw_values:
        normalized = _normalize_intervention_label(value)
        if normalized == "all":
            use_all = True
        elif normalized:
            labels.append(normalized)
    if use_all:
        return list(SEMANTIC_INTERVENTION_LABELS)
    if not labels:
        return [""]
    seen = set()
    unique: List[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return unique


def _adversary_semantic_labels(args: argparse.Namespace) -> List[str]:
    raw_values = [str(value) for value in (getattr(args, "adversary_semantic_labels", []) or [])]
    if not raw_values:
        raw_values = [str(value) for value in (getattr(args, "intervention", []) or []) if str(value or "").strip()]
    labels: List[str] = []
    use_all = bool(getattr(args, "all_interventions", False))
    for value in raw_values:
        normalized = _normalize_intervention_label(value)
        if normalized == "all":
            use_all = True
        elif normalized:
            labels.append(normalized)
    if use_all:
        return list(TD3_ADVERSARY_SEMANTIC_LABELS)
    if not labels:
        return list(TD3_ADVERSARY_SEMANTIC_LABELS)
    seen = set()
    unique: List[str] = []
    for label in labels:
        if label in seen:
            continue
        seen.add(label)
        unique.append(label)
    return unique


def _resolve_intervention_agent_id(
    *,
    request: Any,
    sdc_id: Any,
    modeled_agent_ids: Sequence[Any],
    summary: Any,
) -> Tuple[str, int]:
    requested = str(request or "ego").strip()
    target_agent_id = str(sdc_id) if requested.lower() in {"", "ego", "sdc"} else requested
    modeled = [str(value) for value in modeled_agent_ids]
    if target_agent_id in modeled:
        return target_agent_id, int(modeled.index(target_agent_id))
    for row in list(getattr(summary, "agents", []) or []):
        raw_track_id = str(getattr(row, "raw_track_id", ""))
        if raw_track_id == target_agent_id:
            slot = int(getattr(row, "model_agent_slot"))
            if 0 <= slot < len(modeled):
                return raw_track_id, slot
    available = ", ".join(modeled[:24])
    raise ValueError(
        f"Intervention agent {target_agent_id!r} is not a modeled decoder agent. "
        f"Available modeled agents: {available}"
    )


def _intervention_output_suffix(label: str) -> str:
    return f"__intervention_{_safe_slug(label)}" if str(label or "").strip() else ""


def _ground_truth_output_suffix() -> str:
    return "__ground_truth"


def _select_td3_style_adversary_intervention(
    runner: CheckpointRunner,
    *,
    raw: Mapping[str, Any],
    scenario_id: str,
    labels: Sequence[str],
    semantic_confidence: float,
    start_step: int,
    end_step: int,
    min_moving_speed_mps: float,
    max_distance_to_sdc_m: float,
    max_candidates: int,
    min_final_position_delta_m: float,
    min_changed_action_steps: int,
) -> Dict[str, Any]:
    _ensure_repo_imports()
    try:
        from scripts.agent_eval.build_victim_centric_table4_dataset import (
            _evaluate_candidate_label,
            _rank_adversary_candidates,
            _score_intervention,
        )
        from scripts.counterfactual.probe_agent_semantic_rollout import (
            _extract_all_reference_world_from_sample,
            _run_rollout,
        )
    except ModuleNotFoundError as exc:
        missing = str(getattr(exc, "name", "") or exc)
        raise RuntimeError(f"adversary mode requires victim-centric TD3 helpers (missing module: {missing})") from exc

    base_sample = runner.preprocess_raw_scenario_for_forward_supervision(
        dict(raw),
        config=runner.config,
        in_evaluation=True,
    )
    base_sample["metadata/scenario_id"] = str(raw.get("id") or scenario_id)
    forward_summary = runner.summarize_forward_supervision_for_sample(base_sample, raw_scenario=dict(raw))
    map_center_world, map_heading_world = runner.extract_model_frame(dict(raw))
    all_reference_world = _extract_all_reference_world_from_sample(
        base_sample,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
        modeled_agent_ids=forward_summary.modeled_agent_ids,
    )
    candidates = _rank_adversary_candidates(
        base_sample=base_sample,
        forward_summary=forward_summary,
        all_reference_world=all_reference_world,
        min_moving_speed_mps=float(min_moving_speed_mps),
        max_distance_to_sdc_m=float(max_distance_to_sdc_m),
        max_candidates=int(max_candidates),
    )
    if not candidates:
        return {
            "selected_intervention": None,
            "reason": "no_candidate_adversaries",
            "sdc_id": str(getattr(forward_summary, "sdc_id", "")),
            "candidate_adversaries": [],
            "evaluated_interventions": [],
        }

    baseline = _run_rollout(runner.module, runner.tokenizer, raw_sample=base_sample)
    horizon = int(np.asarray(base_sample["decoder/target_action_valid_mask"]).shape[0])
    time_window_mask = runner.build_time_window_mask(
        horizon=horizon,
        start_step=int(start_step),
        end_step=int(end_step),
    )
    normalized_labels = [str(runner.normalize_semantic_label(label)) for label in labels]
    candidate_records: List[Dict[str, Any]] = []
    best_record: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        for label in normalized_labels:
            record = _evaluate_candidate_label(
                module=runner.module,
                tokenizer=runner.tokenizer,
                raw_scenario=dict(raw),
                base_sample=base_sample,
                forward_summary=forward_summary,
                candidate=candidate,
                semantic_label=label,
                semantic_confidence=float(semantic_confidence),
                time_window_mask=time_window_mask,
                map_center_world=np.asarray(map_center_world, dtype=np.float32),
                map_heading_world=float(map_heading_world),
                baseline_output_np=baseline["output_np"],
                all_reference_world=all_reference_world,
            )
            candidate_records.append(record)
            passes = (
                float(record["effect"]["final_position_delta_m"]) >= float(min_final_position_delta_m)
                and int(record["effect"]["num_changed_action_steps"]) >= int(min_changed_action_steps)
                and math.isfinite(float(record["victim_min_distance_m"]))
            )
            if not passes:
                continue
            if best_record is None or _score_intervention(record) < _score_intervention(best_record):
                best_record = record

    sorted_records = sorted(candidate_records, key=_score_intervention)
    return {
        "selected_intervention": best_record,
        "reason": None if best_record is not None else "no_intervention_passed_filters",
        "sdc_id": str(getattr(forward_summary, "sdc_id", "")),
        "candidate_adversaries": [
            {
                "agent_id": str(candidate["agent_id"]),
                "slot": int(candidate["slot"]),
                "speed_mps": float(candidate["speed_mps"]),
                "distance_to_sdc_m": float(candidate["distance_to_sdc_m"]),
                "num_loss_steps": int(candidate["num_loss_steps"]),
            }
            for candidate in candidates
        ],
        "evaluated_interventions": [
            {
                "agent_id": str(record["agent_id"]),
                "semantic_label": str(record["semantic_label"]),
                "victim_agent_id": str(record["victim_agent_id"]),
                "victim_min_distance_m": float(record["victim_min_distance_m"]),
                "victim_min_distance_step": int(record["victim_min_distance_step"]),
                "effect": dict(record["effect"]),
            }
            for record in sorted_records
        ],
    }


def _run_checkpoint_rollout(
    runner: CheckpointRunner,
    *,
    raw: Mapping[str, Any],
    scenario_id: str,
    intervention_label: str = "",
    intervention_agent: str = "ego",
    intervention_start_step: int = 0,
    intervention_end_step: int = -1,
    intervention_confidence: float = 1.0,
) -> Dict[str, Any]:
    _require_plot_deps()
    from scripts.counterfactual.probe_agent_semantic_rollout import _run_rollout

    sample = runner.preprocess_raw_scenario_for_forward_supervision(
        dict(raw),
        config=runner.config,
        in_evaluation=True,
    )
    sample["metadata/scenario_id"] = str(raw.get("id") or scenario_id)
    summary = runner.summarize_forward_supervision_for_sample(sample, raw_scenario=dict(raw))
    modeled_agent_ids = [str(value) for value in list(getattr(summary, "modeled_agent_ids", []))]
    intervention_control: Dict[str, Any] = {}
    if str(intervention_label or "").strip():
        semantic_label = _normalize_intervention_label(intervention_label)
        semantic_label = str(runner.normalize_semantic_label(semantic_label))
        target_agent_id, target_slot = _resolve_intervention_agent_id(
            request=intervention_agent,
            sdc_id=getattr(summary, "sdc_id", ""),
            modeled_agent_ids=modeled_agent_ids,
            summary=summary,
        )
        valid_mask_source = sample.get("decoder/target_action_valid_mask", sample.get("decoder/agent_valid_mask"))
        valid_mask_array = np.asarray(valid_mask_source)
        if valid_mask_array.ndim < 1:
            raise ValueError("Could not infer decoder horizon for semantic intervention control.")
        horizon = int(valid_mask_array.shape[0])
        decision_agent_mask = np.zeros((len(modeled_agent_ids),), dtype=np.float32)
        decision_agent_mask[int(target_slot)] = 1.0
        time_window_mask = runner.build_time_window_mask(
            horizon=horizon,
            start_step=int(intervention_start_step),
            end_step=int(intervention_end_step),
        )
        sample = runner.build_control_sample(
            base_sample=sample,
            semantic_label=semantic_label,
            semantic_confidence=float(intervention_confidence),
            time_window_mask=time_window_mask,
            decision_agent_mask=decision_agent_mask,
        )
        active_steps = np.flatnonzero(np.asarray(time_window_mask) > 0.0)
        intervention_control = {
            "semantic_label": str(semantic_label),
            "target_agent_id": str(target_agent_id),
            "target_slot": int(target_slot),
            "semantic_confidence": float(intervention_confidence),
            "start_step": int(active_steps[0]) if active_steps.size else None,
            "end_step": int(active_steps[-1]) if active_steps.size else None,
            "num_active_steps": int(active_steps.size),
            "control_available": bool(np.asarray(decision_agent_mask).sum() > 0.0 and active_steps.size > 0),
        }
    rollout = _run_rollout(runner.module, runner.tokenizer, raw_sample=sample)
    output_np = rollout["output_np"]
    pos_model = np.asarray(output_np.get("decoder/reconstructed_position", []), dtype=np.float64)
    valid_mask = np.asarray(output_np.get("decoder/reconstructed_valid_mask", []), dtype=bool)
    heading_model = np.asarray(output_np.get("decoder/reconstructed_heading", []), dtype=np.float64)
    if pos_model.ndim != 3 or pos_model.shape[-1] < 2:
        return {
            "world_pos": np.zeros((0, 0, 2), dtype=np.float64),
            "world_heading": np.zeros((0, 0), dtype=np.float64),
            "valid_mask": np.zeros((0, 0), dtype=bool),
            "agent_id_to_slot": {},
            "intervention_control": intervention_control,
        }
    map_center_world, map_heading_world = runner.extract_model_frame(dict(raw))
    world_flat = _model_to_world(
        pos_model[..., :2].reshape(-1, 2),
        map_center_world=map_center_world,
        map_heading_world=float(map_heading_world),
    )
    world_pos = world_flat.reshape(pos_model.shape[0], pos_model.shape[1], 2)
    if heading_model.ndim == 3 and heading_model.shape[-1] == 1:
        heading_model = heading_model[..., 0]
    if heading_model.ndim == 2 and heading_model.shape[:2] == pos_model.shape[:2]:
        world_heading = heading_model + float(map_heading_world)
    else:
        world_heading = np.zeros((0, 0), dtype=np.float64)
    agent_id_to_slot = {agent_id: idx for idx, agent_id in enumerate(modeled_agent_ids)}
    return {
        "world_pos": world_pos,
        "world_heading": world_heading,
        "valid_mask": valid_mask,
        "agent_id_to_slot": agent_id_to_slot,
        "intervention_control": intervention_control,
        "forward_summary": {
            "scenario_id": str(getattr(summary, "scenario_id", scenario_id)),
            "sdc_id": str(getattr(summary, "sdc_id", "")),
            "num_modeled_agents": int(len(modeled_agent_ids)),
        },
    }


def _rollout_agent_xy(rollout: Optional[Mapping[str, Any]], *, agent_id: str, until_idx: int):
    _require_plot_deps()
    if not rollout:
        return None
    slot = dict(rollout.get("agent_id_to_slot", {}) or {}).get(str(agent_id))
    world_pos = np.asarray(rollout.get("world_pos", []), dtype=np.float64)
    valid_mask = np.asarray(rollout.get("valid_mask", []), dtype=bool)
    if slot is None or world_pos.ndim != 3 or valid_mask.ndim != 2:
        return None
    if int(slot) < 0 or int(slot) >= world_pos.shape[1]:
        return None
    hi = int(np.clip(int(until_idx), 0, max(0, world_pos.shape[0] - 1)))
    mask = valid_mask[: hi + 1, int(slot)]
    points = world_pos[: hi + 1, int(slot), :2]
    points = _finite_xy(points[mask[: points.shape[0]]])
    return None if points.shape[0] < 2 else points


def _rollout_agent_pose(rollout: Optional[Mapping[str, Any]], *, agent_id: str, until_idx: int) -> Optional[Dict[str, float]]:
    _require_plot_deps()
    if not rollout:
        return None
    slot = dict(rollout.get("agent_id_to_slot", {}) or {}).get(str(agent_id))
    world_pos = np.asarray(rollout.get("world_pos", []), dtype=np.float64)
    valid_mask = np.asarray(rollout.get("valid_mask", []), dtype=bool)
    if slot is None or world_pos.ndim != 3 or valid_mask.ndim != 2:
        return None
    slot = int(slot)
    if slot < 0 or slot >= world_pos.shape[1]:
        return None
    hi = int(np.clip(int(until_idx), 0, max(0, world_pos.shape[0] - 1)))
    valid_indices = np.flatnonzero(valid_mask[: hi + 1, slot])
    if valid_indices.size == 0:
        return None
    idx = int(valid_indices[-1])
    xy = np.asarray(world_pos[idx, slot, :2], dtype=np.float64)
    if xy.shape[0] < 2 or not np.isfinite(xy[:2]).all():
        return None

    heading = None
    world_heading = np.asarray(rollout.get("world_heading", []), dtype=np.float64)
    if world_heading.ndim == 2 and idx < world_heading.shape[0] and slot < world_heading.shape[1]:
        candidate = float(world_heading[idx, slot])
        if math.isfinite(candidate):
            heading = candidate
    if heading is None and valid_indices.size >= 2:
        prev_idx = int(valid_indices[-2])
        prev_xy = np.asarray(world_pos[prev_idx, slot, :2], dtype=np.float64)
        delta = xy[:2] - prev_xy[:2]
        if np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 1e-3:
            heading = float(math.atan2(float(delta[1]), float(delta[0])))
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "heading": float(heading if heading is not None else 0.0),
        "index": int(idx),
    }


def _rollout_agent_pose_at(
    rollout: Optional[Mapping[str, Any]],
    *,
    agent_id: str,
    time_idx: int,
) -> Optional[Dict[str, float]]:
    _require_plot_deps()
    if not rollout:
        return None
    slot = dict(rollout.get("agent_id_to_slot", {}) or {}).get(str(agent_id))
    world_pos = np.asarray(rollout.get("world_pos", []), dtype=np.float64)
    valid_mask = np.asarray(rollout.get("valid_mask", []), dtype=bool)
    if slot is None or world_pos.ndim != 3 or valid_mask.ndim != 2:
        return None
    slot = int(slot)
    idx = int(time_idx)
    if idx < 0 or idx >= world_pos.shape[0] or slot < 0 or slot >= world_pos.shape[1]:
        return None
    if idx >= valid_mask.shape[0] or not bool(valid_mask[idx, slot]):
        return None
    xy = np.asarray(world_pos[idx, slot, :2], dtype=np.float64)
    if xy.shape[0] < 2 or not np.isfinite(xy[:2]).all():
        return None
    heading = None
    world_heading = np.asarray(rollout.get("world_heading", []), dtype=np.float64)
    if world_heading.ndim == 2 and idx < world_heading.shape[0] and slot < world_heading.shape[1]:
        candidate = float(world_heading[idx, slot])
        if math.isfinite(candidate):
            heading = candidate
    if heading is None:
        for prev_idx in range(idx - 1, -1, -1):
            if prev_idx < valid_mask.shape[0] and bool(valid_mask[prev_idx, slot]):
                prev_xy = np.asarray(world_pos[prev_idx, slot, :2], dtype=np.float64)
                delta = xy[:2] - prev_xy[:2]
                if np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 1e-3:
                    heading = float(math.atan2(float(delta[1]), float(delta[0])))
                    break
    return {
        "x": float(xy[0]),
        "y": float(xy[1]),
        "heading": float(heading if heading is not None else 0.0),
        "index": int(idx),
    }


def _polygons_overlap_sat(poly_a: Any, poly_b: Any) -> bool:
    _require_plot_deps()
    a = np.asarray(poly_a, dtype=np.float64)
    b = np.asarray(poly_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 3 or b.shape[0] < 3:
        return False
    for poly in (a, b):
        for i in range(poly.shape[0]):
            edge = poly[(i + 1) % poly.shape[0]] - poly[i]
            if not np.isfinite(edge).all() or float(np.linalg.norm(edge)) < 1e-6:
                continue
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
            proj_a = a @ axis
            proj_b = b @ axis
            if float(np.max(proj_a)) < float(np.min(proj_b)) or float(np.max(proj_b)) < float(np.min(proj_a)):
                return False
    return True


def _first_sdc_adversary_collision_step(
    raw: Mapping[str, Any],
    rollout: Optional[Mapping[str, Any]],
    *,
    sdc_id: str,
    adversary_id: str,
    until_idx: int,
    padding_m: float = 0.0,
) -> Optional[int]:
    _require_plot_deps()
    if not rollout or not adversary_id:
        return None
    sdc_track = dict(raw.get("tracks", {}).get(str(sdc_id), {}) or {})
    adv_track = dict(raw.get("tracks", {}).get(str(adversary_id), {}) or {})
    if not sdc_track or not adv_track:
        return None
    max_step = max(0, int(until_idx))
    for step in range(max_step + 1):
        sdc_pose = _track_pose(raw, str(sdc_id), step)
        adv_pose = _rollout_agent_pose_at(rollout, agent_id=str(adversary_id), time_idx=step)
        if sdc_pose is None or adv_pose is None:
            continue
        sdc_length, sdc_width = _track_dimensions(sdc_track, int(sdc_pose.get("index", step)))
        adv_length, adv_width = _track_dimensions(adv_track, int(step))
        pad = max(0.0, float(padding_m))
        sdc_poly = _box_world(
            [sdc_pose["x"], sdc_pose["y"]],
            float(sdc_pose["heading"]),
            float(sdc_length) + 2.0 * pad,
            float(sdc_width) + 2.0 * pad,
        )
        adv_poly = _box_world(
            [adv_pose["x"], adv_pose["y"]],
            float(adv_pose["heading"]),
            float(adv_length) + 2.0 * pad,
            float(adv_width) + 2.0 * pad,
        )
        if _polygons_overlap_sat(sdc_poly, adv_poly):
            return int(step)
    return None


def _visible_rollout_track_ids(
    rollout: Optional[Mapping[str, Any]],
    *,
    center_xy: Any,
    until_idx: int,
    radius_m: float,
    agent_limit: int,
    required_ids: Iterable[str],
) -> List[str]:
    _require_plot_deps()
    if not rollout:
        return []
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    rows: List[Tuple[float, str]] = []
    for agent_id in dict(rollout.get("agent_id_to_slot", {}) or {}).keys():
        pose = _rollout_agent_pose(rollout, agent_id=str(agent_id), until_idx=until_idx)
        if pose is None:
            continue
        dist = float(np.linalg.norm(np.asarray([pose["x"], pose["y"]], dtype=np.float64) - center))
        if dist <= float(radius_m):
            rows.append((dist, str(agent_id)))
    rows.sort(key=lambda item: (item[0], item[1]))
    ordered = [track_id for _, track_id in rows[: max(1, int(agent_limit))]]
    for required in required_ids:
        required = str(required)
        if required and required not in ordered and _rollout_agent_pose(rollout, agent_id=required, until_idx=until_idx) is not None:
            ordered.append(required)
    return ordered


def _render_scene_graph(
    *,
    selection: SceneSelection,
    args: argparse.Namespace,
    out_path: Path,
    checkpoint_rollout: Optional[Mapping[str, Any]],
    trajectory_json_payload: Any,
) -> Dict[str, Any]:
    _require_plot_deps()
    raw = selection.raw
    sdc_id = _find_sdc_id(raw, explicit=str(args.ego_id or ""))
    until_idx = _parse_until(raw, until=str(args.until), until_step=args.until_step, until_s=args.until_s)
    active_intervention_label = str(getattr(args, "active_intervention_label", "") or "").strip()
    active_intervention_agent_id = str(getattr(args, "active_intervention_agent_id", "") or "").strip()
    active_ground_truth_panel = bool(getattr(args, "active_ground_truth_panel", False))
    active_ground_truth_agent_id = str(getattr(args, "active_ground_truth_agent_id", "") or "").strip()
    active_adversary_mode = bool(getattr(args, "active_adversary_mode", False))
    if active_intervention_agent_id.lower() in {"ego", "sdc"}:
        active_intervention_agent_id = sdc_id
    if active_ground_truth_agent_id.lower() in {"", "ego", "sdc"}:
        active_ground_truth_agent_id = sdc_id
    if active_intervention_label and not active_intervention_agent_id:
        active_intervention_agent_id = sdc_id
    collision_step = None
    plot_until_idx = int(until_idx)
    if (
        active_adversary_mode
        and active_intervention_label
        and active_intervention_agent_id
        and checkpoint_rollout is not None
        and not bool(getattr(args, "no_truncate_at_collision", False))
    ):
        collision_step = _first_sdc_adversary_collision_step(
            raw,
            checkpoint_rollout,
            sdc_id=sdc_id,
            adversary_id=active_intervention_agent_id,
            until_idx=until_idx,
            padding_m=float(getattr(args, "collision_padding_m", 0.0) or 0.0),
        )
        if collision_step is not None:
            plot_until_idx = int(collision_step)

    ego_pose = _track_pose(raw, sdc_id, plot_until_idx)
    if ego_pose is None:
        raise ValueError(f"Could not find valid ego/SDC pose for track {sdc_id!r} at step {plot_until_idx}.")
    center_agent = str(args.center_agent_id or "ego")
    if center_agent.lower() in {"ego", "sdc"}:
        center_agent = sdc_id
    center_pose = _track_pose(raw, center_agent, plot_until_idx) or ego_pose
    lock_view = bool(getattr(args, "lock_view", False))
    view_anchor_source = "logged"
    if active_intervention_label and checkpoint_rollout is not None and not lock_view and not active_adversary_mode:
        rollout_center_agent = sdc_id if str(args.center_agent_id or "ego").lower() in {"ego", "sdc"} else center_agent
        rollout_center_pose = _rollout_agent_pose(checkpoint_rollout, agent_id=rollout_center_agent, until_idx=plot_until_idx)
        if rollout_center_pose is not None:
            center_pose = rollout_center_pose
            view_anchor_source = "rollout"
    center_xy = np.asarray([center_pose["x"], center_pose["y"]], dtype=np.float64)
    heading_rad = float(center_pose["heading"]) if str(args.view) == "ego" else 0.0

    if active_adversary_mode and active_intervention_label:
        trajectory_agent_ids = [sdc_id, active_intervention_agent_id]
    elif active_intervention_label:
        trajectory_agent_ids = [active_intervention_agent_id or sdc_id]
    elif active_ground_truth_panel:
        trajectory_agent_ids = [active_ground_truth_agent_id or sdc_id]
    else:
        trajectory_agent_ids = [str(agent) for agent in (args.trajectory_agent or [])]
        if not trajectory_agent_ids:
            trajectory_agent_ids = [sdc_id]
        trajectory_agent_ids = [sdc_id if item.lower() in {"ego", "sdc"} else item for item in trajectory_agent_ids]

    highlighted_ids = set(str(item) for item in (args.highlight_agent or []))
    adversary_id = str(args.adversary_id or "")
    if adversary_id.lower() in {"", "none"}:
        adversary_id = ""
    if active_adversary_mode and active_intervention_agent_id:
        adversary_id = active_intervention_agent_id
    highlighted_ids.update(trajectory_agent_ids)
    if adversary_id:
        highlighted_ids.add(adversary_id)
    if active_intervention_agent_id:
        highlighted_ids.add(active_intervention_agent_id)
    if active_ground_truth_agent_id:
        highlighted_ids.add(active_ground_truth_agent_id)
    highlighted_ids.add(sdc_id)

    fig, ax = plt.subplots(figsize=(float(args.figsize), float(args.figsize)), dpi=int(args.dpi))
    ax.set_facecolor(BACKGROUND_COLOR)
    fixed_radius_m = float(args.radius_m) if float(args.radius_m) > 0.0 else None
    context_radius_m = float(fixed_radius_m if fixed_radius_m is not None else args.auto_context_radius_m)
    extent_points_view: List[Any] = [
        _world_to_view(center_xy.reshape(1, 2), view=str(args.view), center_xy=center_xy, heading_rad=heading_rad)
    ]
    _draw_map(ax, raw, view=str(args.view), center_xy=center_xy, heading_rad=heading_rad, radius_m=context_radius_m)
    if bool(getattr(args, "show_traffic_signals", False)):
        _draw_traffic_signals(
            ax,
            raw,
            time_idx=plot_until_idx,
            view=str(args.view),
            center_xy=center_xy,
            heading_rad=heading_rad,
            radius_m=context_radius_m,
            label=bool(getattr(args, "label_traffic_signals", False)),
        )

    visible_ids = []
    if active_intervention_label and checkpoint_rollout is not None and not active_adversary_mode:
        visible_ids = _visible_rollout_track_ids(
            checkpoint_rollout,
            center_xy=center_xy,
            until_idx=plot_until_idx,
            radius_m=context_radius_m * 1.25,
            agent_limit=int(args.agent_limit),
            required_ids=highlighted_ids,
        )
    if not visible_ids:
        visible_ids = _visible_track_ids(
            raw,
            center_xy=center_xy,
            time_idx=plot_until_idx,
            radius_m=context_radius_m * 1.25,
            agent_limit=int(args.agent_limit),
            required_ids=highlighted_ids,
        )

    history_start = 0 if int(args.history_window) <= 0 else max(0, plot_until_idx - int(args.history_window))
    show_context_history = (
        (not active_intervention_label and not active_ground_truth_panel)
        or bool(getattr(args, "show_context_history", False))
    )
    if show_context_history:
        for track_id in visible_ids:
            xy_world = _track_xy(raw, track_id, end_idx=plot_until_idx, start_idx=history_start)
            xy_view = _world_to_view(xy_world, view=str(args.view), center_xy=center_xy, heading_rad=heading_rad)
            if xy_view.shape[0] >= 2:
                extent_points_view.append(xy_view)
                is_focus = str(track_id) in highlighted_ids
                color = EGO_COLOR if str(track_id) == sdc_id else (ADVERSARY_COLOR if str(track_id) == adversary_id else AGENT_HISTORY)
                _draw_polyline(
                    ax,
                    xy_view,
                    color=color,
                    linewidth=2.4 if is_focus else 0.9,
                    alpha=0.82 if is_focus else 0.34,
                    linestyle="-",
                    zorder=6 if is_focus else 4,
                )

    for track_id in visible_ids:
        is_sdc = str(track_id) == sdc_id
        is_adv = bool(adversary_id and str(track_id) == adversary_id)
        is_highlight = str(track_id) in highlighted_ids
        fill = EGO_COLOR if is_sdc else (ADVERSARY_COLOR if is_adv else (HIGHLIGHT_COLOR if is_highlight else AGENT_FILL))
        edge = "#064E3B" if is_sdc else ("#7F1D1D" if is_adv else ("#1E3A8A" if is_highlight else AGENT_EDGE))
        label = "ego" if is_sdc and args.label_agents else ("adv" if is_adv and args.label_agents else "")
        box_zorder = (
            18
            if (active_intervention_label or active_ground_truth_panel) and (is_sdc or is_adv)
            else (
                17
                if (active_intervention_label or active_ground_truth_panel) and is_highlight
                else (16 if (active_intervention_label or active_ground_truth_panel) else (11 if (is_sdc or is_adv) else (9 if is_highlight else 7)))
            )
        )
        _draw_agent_box(
            ax,
            raw=raw,
            track_id=str(track_id),
            time_idx=plot_until_idx,
            view=str(args.view),
            center_xy=center_xy,
            heading_rad=heading_rad,
            fill_color=fill,
            edge_color=edge,
            alpha=0.86 if (is_sdc or is_adv or is_highlight) else 0.62,
            linewidth=2.0 if (is_sdc or is_adv or is_highlight) else 1.0,
            zorder=box_zorder,
            label=label,
            pose_override=(
                _rollout_agent_pose(checkpoint_rollout, agent_id=str(track_id), until_idx=plot_until_idx)
                if active_intervention_label
                and checkpoint_rollout is not None
                and (not active_adversary_mode or str(track_id) == active_intervention_agent_id)
                else None
            ),
        )

    plotted_rollout_agents: List[str] = []
    show_logged_trajectory = (
        not bool(getattr(args, "hide_logged_trajectory", False))
        and (
            bool(active_ground_truth_panel)
            or (not active_intervention_label and not active_ground_truth_panel)
            or bool(getattr(args, "show_logged_trajectory", False))
        )
    )
    if active_adversary_mode and active_intervention_label and active_intervention_agent_id:
        sdc_world = _track_xy(raw, sdc_id, end_idx=plot_until_idx, start_idx=0)
        sdc_view = _world_to_view(sdc_world, view=str(args.view), center_xy=center_xy, heading_rad=heading_rad)
        if sdc_view.shape[0] > 0:
            extent_points_view.append(sdc_view)
        _draw_polyline(
            ax,
            sdc_view,
            color=EGO_COLOR,
            linewidth=3.8,
            alpha=0.94,
            linestyle="-",
            zorder=15,
            label=f"SDC ({sdc_id})",
        )
        adversary_world = _rollout_agent_xy(checkpoint_rollout, agent_id=active_intervention_agent_id, until_idx=plot_until_idx)
        if adversary_world is not None:
            adversary_view = _world_to_view(
                adversary_world,
                view=str(args.view),
                center_xy=center_xy,
                heading_rad=heading_rad,
            )
            if adversary_view.shape[0] > 0:
                extent_points_view.append(adversary_view)
            _draw_polyline(
                ax,
                adversary_view,
                color=ADVERSARY_COLOR,
                linewidth=3.8,
                alpha=0.94,
                linestyle="-",
                zorder=16,
                label=f"adversary ({active_intervention_agent_id})",
            )
            if adversary_view.shape[0] > 0:
                ax.scatter(
                    [adversary_view[-1, 0]],
                    [adversary_view[-1, 1]],
                    c=ADVERSARY_COLOR,
                    s=58,
                    edgecolors="white",
                    linewidths=0.9,
                    zorder=17,
                )
            plotted_rollout_agents.append(str(active_intervention_agent_id))
        if collision_step is not None:
            crash_pose = _track_pose(raw, sdc_id, int(collision_step))
            if crash_pose is not None:
                crash_view = _world_to_view(
                    [[crash_pose["x"], crash_pose["y"]]],
                    view=str(args.view),
                    center_xy=center_xy,
                    heading_rad=heading_rad,
                )
                if crash_view.shape[0]:
                    ax.scatter(
                        [crash_view[0, 0]],
                        [crash_view[0, 1]],
                        marker="x",
                        c="#111827",
                        s=95,
                        linewidths=2.2,
                        zorder=18,
                        label=f"first contact t={collision_step}",
                    )
    else:
        for agent_id in trajectory_agent_ids:
            logged_world = _track_xy(raw, agent_id, end_idx=plot_until_idx, start_idx=0)
            logged_view = _world_to_view(logged_world, view=str(args.view), center_xy=center_xy, heading_rad=heading_rad)
            if show_logged_trajectory:
                if logged_view.shape[0] > 0:
                    extent_points_view.append(logged_view)
                _draw_polyline(
                    ax,
                    logged_view,
                    color=INTERVENTION_TRAJ_COLOR if active_ground_truth_panel else (LOGGED_TRAJ_COLOR if agent_id == sdc_id else HIGHLIGHT_COLOR),
                    linewidth=3.8 if active_ground_truth_panel else 3.2,
                    alpha=0.92,
                    linestyle="-",
                    zorder=14 if active_ground_truth_panel else 13,
                    label=f"ground truth {agent_id}" if active_ground_truth_panel else f"logged {agent_id}",
                )

            json_traj_world = _extract_json_trajectory(trajectory_json_payload, scene_id=selection.scenario_id, agent_id=agent_id)
            rollout_world = _rollout_agent_xy(checkpoint_rollout, agent_id=agent_id, until_idx=plot_until_idx)
            generated_world = json_traj_world if json_traj_world is not None else rollout_world
            if generated_world is not None:
                generated_view = _world_to_view(
                    generated_world,
                    view=str(args.view),
                    center_xy=center_xy,
                    heading_rad=heading_rad,
                )
                generated_color = INTERVENTION_TRAJ_COLOR if active_intervention_label else GENERATED_TRAJ_COLOR
                _draw_polyline(
                    ax,
                    generated_view,
                    color=generated_color,
                    linewidth=3.8 if active_intervention_label else 3.0,
                    alpha=0.92,
                    linestyle="-" if active_intervention_label else (0, (5, 3)),
                    zorder=14,
                    label=(
                        f"{active_intervention_label} intervention {agent_id}"
                        if active_intervention_label
                        else f"generated {agent_id}"
                    ),
                )
                if generated_view.shape[0] > 0:
                    extent_points_view.append(generated_view)
                    ax.scatter(
                        [generated_view[-1, 0]],
                        [generated_view[-1, 1]],
                        c=generated_color,
                        s=54,
                        edgecolors="white",
                        linewidths=0.9,
                        zorder=15,
                    )
                plotted_rollout_agents.append(str(agent_id))

    if bool(args.show_tube):
        ax.text(
            0.02,
            0.08,
            "tube overlay requested\n(pass --tube-json in a future extension)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            color="#92400E",
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#F59E0B"},
            zorder=30,
        )
    if bool(args.show_paths):
        ax.text(
            0.02,
            0.16,
            "path overlay requested\n(pass --paths-json in a future extension)",
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            color="#1E3A8A",
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "#60A5FA"},
            zorder=30,
        )

    if fixed_radius_m is not None and str(args.view) == "ego":
        half = float(fixed_radius_m)
        ax.set_xlim(-half, half)
        if bool(getattr(args, "forward_biased_ego_view", False)):
            vertical_span = 2.0 * half
            y_min = -float(args.ego_vertical_fraction) * vertical_span
            ax.set_ylim(y_min, y_min + vertical_span)
        else:
            ax.set_ylim(-half, half)
    elif fixed_radius_m is not None:
        half = float(fixed_radius_m)
        ax.set_xlim(float(center_xy[0] - half), float(center_xy[0] + half))
        ax.set_ylim(float(center_xy[1] - half), float(center_xy[1] + half))
    else:
        _set_fitted_limits(ax, extent_points_view, padding_m=float(args.fit_padding_m))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    timestamps = _timestamps(raw, _infer_num_steps(raw))
    timestamp_s = float(timestamps[plot_until_idx]) if plot_until_idx < timestamps.shape[0] else float(plot_until_idx) * 0.1
    title_lines = [
        f"{selection.scenario_id}",
        f"t={plot_until_idx} ({timestamp_s:.2f}s)  view={args.view}",
        f"ego={sdc_id}" + (f"  adv={adversary_id}" if adversary_id else ""),
    ]
    if args.checkpoint:
        if str(getattr(args, "checkpoint_alias", "") or ""):
            title_lines.append(f"ckpt={args.checkpoint_alias} ({Path(str(args.checkpoint)).name})")
        else:
            title_lines.append(f"ckpt={Path(str(args.checkpoint)).name}")
    if checkpoint_rollout is not None:
        title_lines.append(f"rollout={args.rollout_source}")
    if active_intervention_label:
        target_text = active_intervention_agent_id or str(getattr(args, "intervention_agent", "ego") or "ego")
        title_lines.append(f"intervention={active_intervention_label}  target={target_text}")
    if active_adversary_mode and collision_step is not None:
        title_lines.append(f"first contact at t={collision_step}")
    if active_ground_truth_panel:
        title_lines.append(f"ground truth  target={active_ground_truth_agent_id or sdc_id}")
    info = "\n".join(title_lines)
    show_info_box = bool(getattr(args, "show_info_box", False)) or not bool(getattr(args, "hide_legend", False))
    if show_info_box:
        ax.text(
            0.02,
            0.975,
            info,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#CBD5E1"},
            zorder=30,
        )

    scale_x = float(ax.get_xlim()[0] + 0.08 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    scale_y = float(ax.get_ylim()[0] + 0.08 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.plot([scale_x, scale_x + 10.0], [scale_y, scale_y], color="#111827", linewidth=2.0, zorder=31)
    ax.text(scale_x + 5.0, scale_y + 1.2, "10m", ha="center", va="bottom", fontsize=8, color="#111827", zorder=31)
    if not bool(getattr(args, "hide_legend", False)):
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="lower right", fontsize=7, framealpha=0.86)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return {
        "requested_scene": selection.requested,
        "scenario_id": selection.scenario_id,
        "scenario_pkl": str(selection.path),
        "output_png": str(out_path),
        "until_index": int(until_idx),
        "render_until_index": int(plot_until_idx),
        "timestamp_s": float(timestamp_s),
        "sdc_id": str(sdc_id),
        "center_agent_id": str(center_agent),
        "view_anchor_source": str(view_anchor_source),
        "lock_view": bool(lock_view),
        "view_center_xy": [float(center_xy[0]), float(center_xy[1])],
        "view_heading_rad": float(heading_rad),
        "trajectory_agent_ids": trajectory_agent_ids,
        "plotted_generated_agent_ids": plotted_rollout_agents,
        "rollout_source": str(args.rollout_source),
        "panel_type": (
            "ground_truth"
            if active_ground_truth_panel
            else ("adversary" if active_adversary_mode and active_intervention_label else ("intervention" if active_intervention_label else "baseline"))
        ),
        "intervention_label": active_intervention_label,
        "intervention_agent_id": active_intervention_agent_id,
        "adversary_mode": bool(active_adversary_mode),
        "adversary_id": str(adversary_id),
        "collision_step": None if collision_step is None else int(collision_step),
        "truncated_at_collision": bool(collision_step is not None and plot_until_idx == int(collision_step)),
        "ground_truth_panel": bool(active_ground_truth_panel),
        "ground_truth_agent_id": active_ground_truth_agent_id if active_ground_truth_panel else "",
        "show_logged_trajectory": bool(show_logged_trajectory),
        "show_context_history": bool(show_context_history),
        "hide_legend": bool(getattr(args, "hide_legend", False)),
        "show_info_box": bool(show_info_box),
        "fixed_radius_centered": bool(fixed_radius_m is not None and str(args.view) == "ego" and not bool(getattr(args, "forward_biased_ego_view", False))),
        "figsize": float(args.figsize),
        "dpi": int(args.dpi),
        "checkpoint_alias": str(getattr(args, "checkpoint_alias", "") or ""),
        "checkpoint": str(args.checkpoint or ""),
        "fixed_radius_m": None if fixed_radius_m is None else float(fixed_radius_m),
        "auto_context_radius_m": float(context_radius_m),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render paper-ready Waymax scene context graphs with agent rectangles and trajectory overlays. "
            "Use --scene multiple times or pass multiple values after one --scene. "
            "Numeric shortcuts are accepted, e.g. --scene 2 resolves to waymax_scene_00002."
        )
    )
    parser.add_argument(
        "--scene",
        nargs="+",
        action="append",
        default=[],
        help="Scene id(s), numeric shortcut(s), or raw scenario .pkl path(s). Example: 2 -> waymax_scene_00002.",
    )
    parser.add_argument(
        "--scene-root",
        action="append",
        default=[],
        help="Root directory to search when --scene is a scene id. Can be repeated.",
    )
    parser.add_argument("--outdir", default="outputs/paper_graphs", help="Directory for PNGs and manifest.json.")
    parser.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        default="",
        help="Checkpoint path or alias. Built-in aliases: prog, topo.",
    )
    parser.add_argument("--list-checkpoints", action="store_true", help="List built-in checkpoint aliases and exit.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Model config used when --rollout-source checkpoint.")
    parser.add_argument("--teacher-checkpoint", "--teacher-ckpt", dest="teacher_checkpoint", default="")
    parser.add_argument(
        "--rollout-source",
        choices=("logged", "checkpoint"),
        default="logged",
        help="logged renders only raw logged trajectories; checkpoint overlays an autoregressive model rollout.",
    )
    parser.add_argument("--load-mode", default="forgiving_state_dict", choices=("forgiving_state_dict", "strict_state_dict"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling-method", default="argmax")
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--topp", type=float, default=-1.0)
    parser.add_argument(
        "--intervention",
        "--semantic-label",
        dest="intervention",
        action="append",
        default=[],
        help=(
            "Semantic intervention label for checkpoint rollout. Repeatable. "
            "Use left, right, left_lane_change, right_lane_change, straight, stop, none, or all."
        ),
    )
    parser.add_argument(
        "--all-interventions",
        action="store_true",
        help="Render one checkpoint rollout per semantic intervention label.",
    )
    parser.add_argument(
        "--adversary-mode",
        action="store_true",
        help=(
            "Select a non-SDC adversary with the same victim-centric TD3 bank logic, "
            "then draw SDC in green and adversary rollout in red."
        ),
    )
    parser.add_argument(
        "--adversary-semantic-label",
        action="append",
        dest="adversary_semantic_labels",
        default=[],
        help=(
            "Semantic label candidate for adversary-mode selection. Repeatable. "
            "Defaults to the TD3 bank labels: left, right, left_lane_change, right_lane_change."
        ),
    )
    parser.add_argument("--adversary-max-candidates", type=int, default=2)
    parser.add_argument("--adversary-min-moving-speed-mps", type=float, default=0.5)
    parser.add_argument("--adversary-max-distance-to-sdc-m", type=float, default=40.0)
    parser.add_argument("--adversary-min-final-position-delta-m", type=float, default=1.0)
    parser.add_argument("--adversary-min-changed-action-steps", type=int, default=1)
    parser.add_argument("--no-truncate-at-collision", action="store_true")
    parser.add_argument("--collision-padding-m", type=float, default=0.0)
    parser.add_argument(
        "--include-ground-truth",
        action="store_true",
        help="Also render a ground-truth panel with the same view settings and green logged trajectory.",
    )
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="Do not add the default ground-truth panel when using --all-interventions.",
    )
    parser.add_argument(
        "--intervention-agent",
        default="ego",
        help="Track id to control with --intervention. Defaults to ego/SDC; accepts ego or sdc shortcuts.",
    )
    parser.add_argument("--intervention-start-step", type=int, default=0, help="First decoder step where semantic control is active.")
    parser.add_argument("--intervention-end-step", type=int, default=-1, help="Last active decoder step. Default -1 means full horizon.")
    parser.add_argument("--intervention-confidence", type=float, default=1.0, help="Semantic control confidence passed to the model.")
    parser.add_argument("--until", default="last", help="End timestamp/step: 'last', integer step, or seconds like '4.5s'.")
    parser.add_argument("--until-step", type=int, default=None, help="Explicit integer end step; overrides --until.")
    parser.add_argument("--until-s", type=float, default=None, help="Explicit end time in seconds; overrides --until.")
    parser.add_argument("--ego-id", default="", help="Track id for ego/SDC. Defaults to scenario metadata sdc_id.")
    parser.add_argument("--center-agent-id", default="ego", help="Track id used as plot center in ego view.")
    parser.add_argument("--adversary-id", default="", help="Optional adversary/other focal track id to color red.")
    parser.add_argument("--trajectory-agent", action="append", default=[], help="Track id whose full trajectory should be emphasized. Repeatable.")
    parser.add_argument("--highlight-agent", action="append", default=[], help="Additional track ids to highlight. Repeatable.")
    parser.add_argument("--trajectory-json", default="", help="Optional JSON trajectory overlay, keyed by scenario id and/or agent id.")
    parser.add_argument(
        "--show-logged-trajectory",
        action="store_true",
        help="Also draw the factual logged trajectory in intervention figures. Default hides it for intervention renders.",
    )
    parser.add_argument(
        "--hide-logged-trajectory",
        "--no-logged-trajectory",
        action="store_true",
        help="Never draw the factual logged trajectory overlay.",
    )
    parser.add_argument(
        "--show-context-history",
        action="store_true",
        help="Draw logged agent history trails behind context boxes. Default hides these in intervention figures.",
    )
    parser.add_argument("--view", choices=("ego", "world"), default="ego")
    parser.add_argument(
        "--lock-view",
        action="store_true",
        help=(
            "Keep the crop center and ego-view orientation locked to the logged center-agent pose. "
            "Use with --all-interventions and --radius-m for directly comparable images."
        ),
    )
    parser.add_argument(
        "--radius-m",
        "--scene-radius-m",
        dest="radius_m",
        type=float,
        default=0.0,
        help="Fixed crop radius in meters. Default 0 auto-fits the visible scene to the image window.",
    )
    parser.add_argument(
        "--auto-context-radius-m",
        type=float,
        default=120.0,
        help="Search radius for actors/map when auto-fitting. Ignored when --radius-m is positive.",
    )
    parser.add_argument("--fit-padding-m", type=float, default=8.0, help="Padding around auto-fitted content in meters.")
    parser.add_argument(
        "--forward-biased-ego-view",
        action="store_true",
        help="With --radius-m in ego view, place ego near the bottom instead of centered.",
    )
    parser.add_argument(
        "--ego-vertical-fraction",
        type=float,
        default=0.10,
        help="Vertical ego placement used only with --forward-biased-ego-view.",
    )
    parser.add_argument("--agent-limit", type=int, default=48)
    parser.add_argument("--history-window", type=int, default=18, help="Recent context trail length. Use <=0 for full history.")
    parser.add_argument("--figsize", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--label-agents", action="store_true")
    parser.add_argument("--show-traffic-signals", action="store_true", help="Draw traffic-light stop points colored by state at the rendered timestep.")
    parser.add_argument("--label-traffic-signals", action="store_true", help="Label traffic-light stop points by lane/light id.")
    parser.add_argument("--hide-legend", "--no-legend", action="store_true", help="Do not render the legend or top-left info box.")
    parser.add_argument("--show-info-box", action="store_true", help="Show the top-left scene info box even with --hide-legend.")
    parser.add_argument("--show-tube", action="store_true", help="Reserved placeholder for future valid-tube overlays.")
    parser.add_argument("--show-paths", action="store_true", help="Reserved placeholder for future path-family overlays.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_plot_deps()

    if bool(args.list_checkpoints):
        rows = {
            alias: {
                "path": str(path.resolve()),
                "exists": bool(path.is_file()),
                "size_bytes": int(path.stat().st_size) if path.is_file() else None,
            }
            for alias, path in CHECKPOINT_ALIAS_PATHS.items()
        }
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if not args.scene:
        print("ERROR: at least one --scene is required unless --list-checkpoints is used.", file=sys.stderr)
        return 2

    checkpoint_alias, checkpoint_path = _resolve_checkpoint_path(str(args.checkpoint or ""))
    args.checkpoint_alias = checkpoint_alias
    args.checkpoint = checkpoint_path
    try:
        intervention_labels = _requested_intervention_labels(args)
        adversary_labels = _adversary_semantic_labels(args) if bool(getattr(args, "adversary_mode", False)) else []
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    has_interventions = any(str(label or "").strip() for label in intervention_labels)
    if bool(getattr(args, "adversary_mode", False)):
        has_interventions = True
    include_ground_truth = (
        (bool(args.all_interventions) or bool(getattr(args, "include_ground_truth", False)))
        and not bool(getattr(args, "no_ground_truth", False))
    )
    render_specs: List[Tuple[str, str]] = []
    if include_ground_truth:
        render_specs.append(("ground_truth", ""))
    if has_interventions:
        render_specs.extend(("intervention", str(label)) for label in intervention_labels if str(label or "").strip())
    elif not include_ground_truth:
        render_specs.append(("baseline", ""))

    if bool(getattr(args, "adversary_mode", False)) and not args.checkpoint:
        print("ERROR: --adversary-mode requires --checkpoint.", file=sys.stderr)
        return 2
    if has_interventions and str(args.rollout_source) != "checkpoint":
        if args.checkpoint:
            args.rollout_source = "checkpoint"
            print(
                "INFO: semantic interventions require checkpoint rollouts; using --rollout-source checkpoint.",
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: --intervention/--all-interventions requires --checkpoint and checkpoint rollouts.",
                file=sys.stderr,
            )
            return 2
    if bool(getattr(args, "lock_view", False)) and float(args.radius_m) <= 0.0:
        print(
            "INFO: --lock-view locks center/orientation; add --radius-m for identical crop limits across images.",
            file=sys.stderr,
        )

    outdir = _expand(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scene_args = [item for group in args.scene for item in group]
    roots = [_expand(root) for root in args.scene_root]
    if not roots:
        roots = [REPO_ROOT / "outputs", REPO_ROOT / "data", REPO_ROOT]

    trajectory_json_payload = _read_json(args.trajectory_json) if args.trajectory_json else None
    try:
        checkpoint_runner = _build_checkpoint_runner(args) if str(args.rollout_source) == "checkpoint" else None
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest: List[Dict[str, Any]] = []
    for scene_text in scene_args:
        selection = _load_scene(scene_text, roots)
        adversary_plan: Optional[Dict[str, Any]] = None
        scene_render_specs: List[Tuple[str, str, str]] = [
            (panel_type, intervention_label, str(args.intervention_agent or "ego"))
            for panel_type, intervention_label in render_specs
        ]
        if bool(getattr(args, "adversary_mode", False)):
            if checkpoint_runner is None:
                print("ERROR: --adversary-mode requires checkpoint runner.", file=sys.stderr)
                return 2
            adversary_plan = _select_td3_style_adversary_intervention(
                checkpoint_runner,
                raw=selection.raw,
                scenario_id=selection.scenario_id,
                labels=adversary_labels,
                semantic_confidence=float(args.intervention_confidence),
                start_step=int(args.intervention_start_step),
                end_step=int(args.intervention_end_step),
                min_moving_speed_mps=float(args.adversary_min_moving_speed_mps),
                max_distance_to_sdc_m=float(args.adversary_max_distance_to_sdc_m),
                max_candidates=int(args.adversary_max_candidates),
                min_final_position_delta_m=float(args.adversary_min_final_position_delta_m),
                min_changed_action_steps=int(args.adversary_min_changed_action_steps),
            )
            selected = adversary_plan.get("selected_intervention")
            scene_render_specs = []
            if include_ground_truth:
                scene_render_specs.append(("ground_truth", "", str(args.intervention_agent or "ego")))
            if selected is None:
                manifest.append(
                    {
                        "requested_scene": str(scene_text),
                        "scenario_id": selection.scenario_id,
                        "scenario_pkl": str(selection.path),
                        "panel_type": "adversary_skip",
                        "adversary_mode": True,
                        "reason": str(adversary_plan.get("reason") or "no_selected_adversary"),
                        "candidate_adversaries": list(adversary_plan.get("candidate_adversaries", []) or []),
                        "evaluated_interventions": list(adversary_plan.get("evaluated_interventions", []) or []),
                    }
                )
            else:
                selected_agent = str(selected["agent_id"])
                labels_to_render = list(adversary_labels) if bool(args.all_interventions) else [str(selected["semantic_label"])]
                for label in labels_to_render:
                    scene_render_specs.append(("adversary", str(label), selected_agent))

        for panel_type, intervention_label, intervention_agent in scene_render_specs:
            checkpoint_rollout = None
            args.active_ground_truth_panel = bool(panel_type == "ground_truth")
            args.active_ground_truth_agent_id = str(args.intervention_agent or "ego")
            args.active_adversary_mode = bool(panel_type == "adversary")
            args.active_intervention_label = str(intervention_label or "") if panel_type in {"intervention", "adversary"} else ""
            args.active_intervention_agent_id = ""
            if checkpoint_runner is not None and panel_type != "ground_truth":
                checkpoint_rollout = _run_checkpoint_rollout(
                    checkpoint_runner,
                    raw=selection.raw,
                    scenario_id=selection.scenario_id,
                    intervention_label=str(intervention_label or ""),
                    intervention_agent=str(intervention_agent or "ego"),
                    intervention_start_step=int(args.intervention_start_step),
                    intervention_end_step=int(args.intervention_end_step),
                    intervention_confidence=float(args.intervention_confidence),
                )
                control = dict(checkpoint_rollout.get("intervention_control", {}) or {})
                args.active_intervention_agent_id = str(control.get("target_agent_id", "") or "")
            suffix = (
                _ground_truth_output_suffix()
                if panel_type == "ground_truth"
                else (
                    f"__adversary_{_safe_slug(str(intervention_agent or 'agent'))}_{_safe_slug(str(intervention_label or 'intervention'))}"
                    if panel_type == "adversary"
                    else _intervention_output_suffix(str(intervention_label or ""))
                )
            )
            output_name = (
                f"{_safe_slug(selection.scenario_id)}__t{_safe_slug(args.until)}"
                f"{suffix}__paper_graph.png"
            )
            output_path = outdir / output_name
            row = _render_scene_graph(
                selection=selection,
                args=args,
                out_path=output_path,
                checkpoint_rollout=checkpoint_rollout,
                trajectory_json_payload=trajectory_json_payload,
            )
            if checkpoint_runner is not None:
                row["checkpoint_load_report"] = dict(checkpoint_runner.load_report)
                if checkpoint_rollout is not None:
                    row["checkpoint_forward_summary"] = dict(checkpoint_rollout.get("forward_summary", {}) or {})
                    row["intervention_control"] = dict(checkpoint_rollout.get("intervention_control", {}) or {})
            if adversary_plan is not None:
                selected_for_manifest = adversary_plan.get("selected_intervention")
                if selected_for_manifest is not None:
                    selected_for_manifest = {
                        "agent_id": str(selected_for_manifest["agent_id"]),
                        "semantic_label": str(selected_for_manifest["semantic_label"]),
                        "victim_agent_id": str(selected_for_manifest["victim_agent_id"]),
                        "victim_min_distance_m": float(selected_for_manifest["victim_min_distance_m"]),
                        "victim_min_distance_step": int(selected_for_manifest["victim_min_distance_step"]),
                        "effect": dict(selected_for_manifest["effect"]),
                    }
                row["adversary_selection"] = {
                    "selected_intervention": selected_for_manifest,
                    "reason": adversary_plan.get("reason"),
                    "candidate_adversaries": adversary_plan.get("candidate_adversaries", []),
                    "evaluated_interventions": adversary_plan.get("evaluated_interventions", []),
                    "rendered_semantic_labels": [str(label) for label in (adversary_labels if bool(args.all_interventions) else [intervention_label]) if str(label or "").strip()],
                }
            manifest.append(row)

    manifest_path = outdir / "manifest.json"
    manifest_summary = {
        "num_graphs": len(manifest),
        "num_scenes": len(manifest),
        "num_requested_scenes": len(scene_args),
        "intervention_labels": [str(label) for label in intervention_labels if str(label or "").strip()],
        "adversary_mode": bool(getattr(args, "adversary_mode", False)),
        "adversary_semantic_labels": [str(label) for label in adversary_labels],
        "include_ground_truth": bool(include_ground_truth),
        "render_specs": [{"panel_type": panel_type, "intervention_label": label} for panel_type, label in render_specs],
        "one_image_per_intervention": bool(any(str(label or "").strip() for label in intervention_labels)),
        "scenes": manifest,
    }
    _write_json(manifest_path, manifest_summary)
    print(
        json.dumps(
            {
                "num_graphs": len(manifest),
                "num_requested_scenes": len(scene_args),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
