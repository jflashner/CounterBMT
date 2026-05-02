from __future__ import annotations

import json
import pickle
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

if __package__ is None or __package__ == "":
    script_dir = str(Path(__file__).resolve().parent)
    while script_dir in sys.path:
        sys.path.remove(script_dir)
    repo_root = Path(__file__).resolve().parents[4]
    vendored_scenarionet = repo_root / "scenarionet"
    for path in (vendored_scenarionet, repo_root, repo_root / "src", repo_root / "src" / "Adv-BMT"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


DEFAULT_WOD_131_TRAIN_PATH = (
    "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/"
    "training_tfexample.tfrecord-00000-of-01000"
)

COMMON_LOCAL_WOD_131_TRAIN_PATHS = (
    "/home/grads/jflashner/CounterBMT/data/training_full/training/training.tfrecord-00000-of-01000",
    "/data/home/grads/jflashner/CounterBMT/data/training_full/training/training.tfrecord-00000-of-01000",
    "/data/home/grads/jflashner/CounterBMT_run/data/training_full/training/training.tfrecord-00000-of-01000",
)
SUBPROCESS_ENV_FLAG = "COUNTERBMT_SCENARIONET_EXPORT_SUBPROCESS"
COORDINATE_WAYMO = "waymo"
TYPE_TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
TYPE_STOP_SIGN = "STOP_SIGN"
TYPE_CROSSWALK = "CROSSWALK"
TYPE_SPEED_BUMP = "SPEED_BUMP"
TYPE_DRIVEWAY = "DRIVEWAY"
SPLIT_KEY = "|"


class WaymoLaneType:
    ENUM_TO_STR = {
        0: "LANE_UNKNOWN",
        1: "LANE_FREEWAY",
        2: "LANE_SURFACE_STREET",
        3: "LANE_BIKE_LANE",
    }

    @classmethod
    def from_waymo(cls, item: int) -> str:
        return cls.ENUM_TO_STR.get(int(item), "LANE_UNKNOWN")


class WaymoRoadLineType:
    ENUM_TO_STR = {
        0: "UNKNOWN",
        1: "ROAD_LINE_BROKEN_SINGLE_WHITE",
        2: "ROAD_LINE_SOLID_SINGLE_WHITE",
        3: "ROAD_LINE_SOLID_DOUBLE_WHITE",
        4: "ROAD_LINE_BROKEN_SINGLE_YELLOW",
        5: "ROAD_LINE_BROKEN_DOUBLE_YELLOW",
        6: "ROAD_LINE_SOLID_SINGLE_YELLOW",
        7: "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        8: "ROAD_LINE_PASSING_DOUBLE_YELLOW",
    }

    @classmethod
    def from_waymo(cls, item: int) -> str:
        return cls.ENUM_TO_STR.get(int(item), "UNKNOWN")


class WaymoRoadEdgeType:
    ENUM_TO_STR = {
        0: "UNKNOWN",
        1: "ROAD_EDGE_BOUNDARY",
        2: "ROAD_EDGE_MEDIAN",
    }

    @classmethod
    def from_waymo(cls, item: int) -> str:
        return cls.ENUM_TO_STR.get(int(item), "UNKNOWN")


class WaymoAgentType:
    ENUM_TO_STR = {
        0: "UNSET",
        1: "VEHICLE",
        2: "PEDESTRIAN",
        3: "CYCLIST",
        4: "OTHER",
    }

    @classmethod
    def from_waymo(cls, item: int) -> str:
        return cls.ENUM_TO_STR.get(int(item), "UNSET")


def _mph_to_kmh(value: float) -> float:
    return float(value) * 1.609344


def _extract_poly(message) -> np.ndarray:
    coord = np.stack(
        [[point.x, point.y, point.z] for point in message],
        axis=0,
    ).astype(np.float32)
    return coord


def _extract_boundaries(boundaries) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for boundary in boundaries:
        record = {
            "lane_start_index": str(int(boundary.lane_start_index)),
            "lane_end_index": str(int(boundary.lane_end_index)),
            "boundary_type": WaymoRoadLineType.from_waymo(int(boundary.boundary_type)),
            "boundary_feature_id": str(int(boundary.boundary_feature_id)),
        }
        items.append(record)
    return items


def _extract_neighbors(neighbors) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for neighbor in neighbors:
        items.append(
            {
                "feature_id": str(int(neighbor.feature_id)),
                "self_start_index": str(int(neighbor.self_start_index)),
                "self_end_index": str(int(neighbor.self_end_index)),
                "neighbor_start_index": str(int(neighbor.neighbor_start_index)),
                "neighbor_end_index": str(int(neighbor.neighbor_end_index)),
                "boundaries": _extract_boundaries(neighbor.boundaries),
            }
        )
    return items


def _extract_center(feature) -> dict[str, object]:
    lane = feature.lane
    return {
        "speed_limit_mph": float(lane.speed_limit_mph),
        "speed_limit_kmh": _mph_to_kmh(float(lane.speed_limit_mph)),
        "type": WaymoLaneType.from_waymo(int(lane.type)),
        "polyline": _extract_poly(lane.polyline),
        "interpolating": bool(lane.interpolating),
        "entry_lanes": [int(x) for x in lane.entry_lanes],
        "exit_lanes": [int(x) for x in lane.exit_lanes],
        "left_boundaries": _extract_boundaries(lane.left_boundaries),
        "right_boundaries": _extract_boundaries(lane.right_boundaries),
        "left_neighbor": _extract_neighbors(lane.left_neighbors),
        "right_neighbor": _extract_neighbors(lane.right_neighbors),
    }


def _extract_line(feature) -> dict[str, object]:
    road_line = feature.road_line
    return {
        "type": WaymoRoadLineType.from_waymo(int(road_line.type)),
        "polyline": _extract_poly(road_line.polyline),
    }


def _extract_edge(feature) -> dict[str, object]:
    road_edge = feature.road_edge
    return {
        "type": WaymoRoadEdgeType.from_waymo(int(road_edge.type)),
        "polyline": _extract_poly(road_edge.polyline),
    }


def _extract_stop(feature) -> dict[str, object]:
    stop_sign = feature.stop_sign
    return {
        "type": TYPE_STOP_SIGN,
        "lane": [int(x) for x in stop_sign.lane],
        "position": np.asarray([stop_sign.position.x, stop_sign.position.y, stop_sign.position.z], dtype=np.float32),
    }


def _extract_crosswalk(feature) -> dict[str, object]:
    return {
        "type": TYPE_CROSSWALK,
        "polygon": _extract_poly(feature.crosswalk.polygon),
    }


def _extract_bump(feature) -> dict[str, object]:
    return {
        "type": TYPE_SPEED_BUMP,
        "polygon": _extract_poly(feature.speed_bump.polygon),
    }


def _extract_driveway(feature) -> dict[str, object]:
    return {
        "type": TYPE_DRIVEWAY,
        "polygon": _extract_poly(feature.driveway.polygon),
    }


def _extract_tracks(tracks, sdc_track_index: int, track_length: int) -> tuple[dict[str, object], str]:
    processed: dict[str, object] = {}
    for obj in tracks:
        object_id = str(int(obj.id))
        obj_type = WaymoAgentType.from_waymo(int(obj.object_type))
        state = {
            "position": np.zeros((track_length, 3), dtype=np.float32),
            "length": np.zeros((track_length,), dtype=np.float32),
            "width": np.zeros((track_length,), dtype=np.float32),
            "height": np.zeros((track_length,), dtype=np.float32),
            "heading": np.zeros((track_length,), dtype=np.float32),
            "velocity": np.zeros((track_length, 2), dtype=np.float32),
            "valid": np.zeros((track_length,), dtype=bool),
        }
        for step_count, track_state in enumerate(obj.states):
            if step_count >= track_length:
                break
            state["position"][step_count] = np.asarray(
                [track_state.center_x, track_state.center_y, track_state.center_z],
                dtype=np.float32,
            )
            state["length"][step_count] = float(track_state.length)
            state["width"][step_count] = float(track_state.width)
            state["height"][step_count] = float(track_state.height)
            state["heading"][step_count] = float(track_state.heading)
            state["velocity"][step_count] = np.asarray(
                [track_state.velocity_x, track_state.velocity_y],
                dtype=np.float32,
            )
            state["valid"][step_count] = bool(track_state.valid)

        processed[object_id] = {
            "type": obj_type,
            "state": state,
            "metadata": {
                "track_length": int(track_length),
                "type": obj_type,
                "object_id": object_id,
                "dataset": "waymo",
            },
        }

    sdc_id = str(int(tracks[int(sdc_track_index)].id))
    return processed, sdc_id


def _extract_map_features(map_features) -> dict[str, object]:
    processed: dict[str, object] = {}
    for feature in map_features:
        feature_id = str(int(feature.id))
        if feature.HasField("lane"):
            processed[feature_id] = _extract_center(feature)
        elif feature.HasField("road_line"):
            processed[feature_id] = _extract_line(feature)
        elif feature.HasField("road_edge"):
            processed[feature_id] = _extract_edge(feature)
        elif feature.HasField("stop_sign"):
            processed[feature_id] = _extract_stop(feature)
        elif feature.HasField("crosswalk"):
            processed[feature_id] = _extract_crosswalk(feature)
        elif feature.HasField("speed_bump"):
            processed[feature_id] = _extract_bump(feature)
        elif feature.HasField("driveway"):
            processed[feature_id] = _extract_driveway(feature)
    return processed


def _extract_dynamic_map_states(dynamic_map_states, track_length: int) -> dict[str, object]:
    processed: dict[str, object] = {}
    for step_count, step_states in enumerate(dynamic_map_states):
        if step_count >= track_length:
            break
        for lane_state in step_states.lane_states:
            object_id = str(int(lane_state.lane))
            if object_id not in processed:
                processed[object_id] = {
                    "type": TYPE_TRAFFIC_LIGHT,
                    "state": {"object_state": [None] * track_length},
                    "lane": int(lane_state.lane),
                    "stop_point": np.zeros((3,), dtype=np.float32),
                    "metadata": {
                        "track_length": int(track_length),
                        "type": TYPE_TRAFFIC_LIGHT,
                        "object_id": object_id,
                        "dataset": "waymo",
                    },
                }
            processed[object_id]["state"]["object_state"][step_count] = lane_state.State.Name(lane_state.state)
            processed[object_id]["stop_point"][:] = np.asarray(
                [lane_state.stop_point.x, lane_state.stop_point.y, lane_state.stop_point.z],
                dtype=np.float32,
            )
    return processed


def _nearest_point(point: np.ndarray, line: np.ndarray) -> int:
    dist = np.square(line - point)
    dist = np.sqrt(dist[:, 0] + dist[:, 1])
    return int(np.argmin(dist))


def _extract_width(map_features: dict[str, object], polyline: np.ndarray, boundaries: list[dict[str, str]]) -> np.ndarray:
    width = np.zeros((polyline.shape[0],), dtype=np.float32)
    for boundary in boundaries:
        boundary_feature_id = str(boundary["boundary_feature_id"])
        if boundary_feature_id not in map_features:
            continue
        boundary_feature = map_features[boundary_feature_id]
        boundary_polyline = np.asarray(boundary_feature.get("polyline", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)[:, :2]
        if boundary_polyline.shape[0] == 0:
            continue
        lane_start = int(boundary["lane_start_index"])
        lane_end = int(boundary["lane_end_index"])
        start_point = polyline[lane_start]
        start_index = _nearest_point(start_point, boundary_polyline)
        seg_len = lane_end - lane_start
        end_index = min(start_index + seg_len, int(boundary_polyline.shape[0] - 1))
        length = min(end_index - start_index, seg_len) + 1
        self_range = range(lane_start, lane_start + length)
        boundary_range = range(start_index, start_index + length)
        center_lane = polyline[list(self_range)]
        boundary_points = boundary_polyline[list(boundary_range)]
        dist = np.square(center_lane - boundary_points)
        dist = np.sqrt(dist[:, 0] + dist[:, 1])
        width[list(self_range)] = dist
    return width


def _compute_width(map_features: dict[str, object]) -> None:
    for lane_id, lane in map_features.items():
        lane_type = str(lane.get("type", ""))
        if "LANE" not in lane_type or "polyline" not in lane:
            continue
        polyline = np.asarray(lane["polyline"], dtype=np.float32)[:, :2]
        width = np.zeros((polyline.shape[0], 2), dtype=np.float32)
        width[:, 0] = _extract_width(map_features, polyline, list(lane.get("left_boundaries", [])))
        width[:, 1] = _extract_width(map_features, polyline, list(lane.get("right_boundaries", [])))
        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]
        lane["width"] = width


def _split_scenario_id(scenario_id: str, *, source_file_hint: str) -> tuple[str, str]:
    text = str(scenario_id or "").strip()
    if SPLIT_KEY in text:
        primary, source = text.split(SPLIT_KEY, 1)
        return primary, source
    return text, str(source_file_hint)


def _convert_waymo_scenario(scenario, *, version: str, source_file: str) -> dict[str, object]:
    scenario_id, source_file_text = _split_scenario_id(str(scenario.scenario_id), source_file_hint=source_file)
    track_length = int(len(list(scenario.timestamps_seconds)))
    tracks, sdc_id = _extract_tracks(scenario.tracks, int(scenario.sdc_track_index), track_length)
    dynamic_states = _extract_dynamic_map_states(scenario.dynamic_map_states, track_length)
    map_features = _extract_map_features(scenario.map_features)
    _compute_width(map_features)

    converted = {
        "id": scenario_id,
        "version": str(version),
        "length": track_length,
        "tracks": tracks,
        "dynamic_map_states": dynamic_states,
        "map_features": map_features,
        "metadata": {
            "id": scenario_id,
            "coordinate": COORDINATE_WAYMO,
            "ts": np.asarray(list(scenario.timestamps_seconds), dtype=np.float32),
            "metadrive_processed": False,
            "sdc_id": str(sdc_id),
            "dataset": "waymo",
            "scenario_id": scenario_id,
            "source_file": str(source_file_text),
            "track_length": track_length,
            "current_time_index": int(scenario.current_time_index),
            "sdc_track_index": int(scenario.sdc_track_index),
            "objects_of_interest": [str(int(obj)) for obj in scenario.objects_of_interest],
        },
    }

    track_index = [int(obj.track_index) for obj in scenario.tracks_to_predict]
    track_id = [str(int(scenario.tracks[idx].id)) for idx in track_index]
    track_difficulty = [int(obj.difficulty) for obj in scenario.tracks_to_predict]
    track_obj_type = [tracks[obj_id]["type"] for obj_id in track_id]
    converted["metadata"]["tracks_to_predict"] = {
        obj_id: {
            "track_index": track_index[count],
            "track_id": obj_id,
            "difficulty": track_difficulty[count],
            "object_type": track_obj_type[count],
        }
        for count, obj_id in enumerate(track_id)
    }
    return converted


def parse_waymax_scene_index(scenario_id: str) -> int:
    text = str(scenario_id or "").strip()
    if not text.startswith("waymax_scene_"):
        raise ValueError(f"Unsupported scenario_id for Waymo reconstruction: {scenario_id!r}")
    return int(text.rsplit("_", 1)[-1])


def expected_scenarionet_source_path(
    *,
    cache_root: str | Path,
    scenario_id: str,
    version: str = "v1.2",
) -> Path:
    safe_version = str(version).strip() or "v1.2"
    return Path(cache_root).expanduser() / f"sd_waymo_{safe_version}_{str(scenario_id).strip()}.pkl"


def resolve_waymo_raw_path(path: str = DEFAULT_WOD_131_TRAIN_PATH) -> str:
    requested = str(path or "").strip()
    env_override = str(os.environ.get("COUNTERBMT_WAYMO_RAW_PATH", "")).strip()
    if env_override and Path(env_override).expanduser().is_file():
        return str(Path(env_override).expanduser())

    if requested and Path(requested).expanduser().is_file():
        return str(Path(requested).expanduser())

    if requested in {"", DEFAULT_WOD_131_TRAIN_PATH} or requested.startswith("gs://"):
        for candidate in COMMON_LOCAL_WOD_131_TRAIN_PATHS:
            if Path(candidate).expanduser().is_file():
                return str(Path(candidate).expanduser())

    return requested


def materialize_scenarionet_waymo_sources(
    *,
    scenario_ids: Iterable[str],
    cache_root: str | Path,
    waymo_raw_path: str = DEFAULT_WOD_131_TRAIN_PATH,
    version: str = "v1.2",
) -> Dict[str, Path]:
    if os.environ.get(SUBPROCESS_ENV_FLAG) != "1":
        payload = {
            "scenario_ids": [str(item).strip() for item in scenario_ids if str(item).strip()],
            "cache_root": str(Path(cache_root).expanduser()),
            "waymo_raw_path": str(waymo_raw_path),
            "version": str(version),
        }
        repo_root = Path(__file__).resolve().parents[4]
        py_paths = [
            str(repo_root / "scenarionet"),
            str(repo_root),
            str(repo_root / "src"),
            str(repo_root / "src" / "Adv-BMT"),
        ]
        worker_code = (
            "import json, sys; "
            "from bmt.counterfactual.scenarionet_waymo_export_source import "
            "_materialize_scenarionet_waymo_sources_impl; "
            "payload=json.loads(sys.argv[1]); "
            "resolved=_materialize_scenarionet_waymo_sources_impl("
            "scenario_ids=payload.get('scenario_ids', []), "
            "cache_root=payload.get('cache_root', ''), "
            "waymo_raw_path=payload.get('waymo_raw_path', ''), "
            "version=payload.get('version', 'v1.2')); "
            "print(json.dumps({str(k): str(v) for k, v in resolved.items()}, sort_keys=True))"
        )
        cmd = [
            sys.executable,
            "-c",
            worker_code,
            json.dumps(payload, sort_keys=True),
        ]
        env = dict(os.environ)
        env[SUBPROCESS_ENV_FLAG] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [path for path in py_paths + [str(env.get("PYTHONPATH", "")).strip()] if path]
        )
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr_text = str(exc.stderr or "").strip()
            stdout_text = str(exc.stdout or "").strip()
            details = stderr_text or stdout_text or f"exit code {exc.returncode}"
            raise RuntimeError(f"ScenarioNet export source subprocess failed: {details}") from exc
        stdout_text = str(completed.stdout or "").strip()
        if not stdout_text:
            raise RuntimeError("ScenarioNet export source subprocess returned no output.")
        result = json.loads(stdout_text.splitlines()[-1])
        return {str(key): Path(value) for key, value in dict(result).items()}

    return _materialize_scenarionet_waymo_sources_impl(
        scenario_ids=scenario_ids,
        cache_root=cache_root,
        waymo_raw_path=waymo_raw_path,
        version=version,
    )


def _materialize_scenarionet_waymo_sources_impl(
    *,
    scenario_ids: Iterable[str],
    cache_root: str | Path,
    waymo_raw_path: str = DEFAULT_WOD_131_TRAIN_PATH,
    version: str = "v1.2",
) -> Dict[str, Path]:
    scenario_ids = [str(item).strip() for item in scenario_ids if str(item).strip()]
    cache_root = Path(cache_root).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved: Dict[str, Path] = {}
    missing: Dict[int, str] = {}
    for scenario_id in scenario_ids:
        out_path = expected_scenarionet_source_path(
            cache_root=cache_root,
            scenario_id=scenario_id,
            version=version,
        )
        if out_path.is_file():
            resolved[scenario_id] = out_path
            continue
        missing[parse_waymax_scene_index(scenario_id)] = scenario_id

    if not missing:
        return resolved

    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    resolved_waymo_raw_path = resolve_waymo_raw_path(waymo_raw_path)

    try:
        tf.config.experimental.set_visible_devices([], "GPU")
    except Exception:
        pass

    max_scene_index = max(missing)
    dataset = tf.data.TFRecordDataset(str(resolved_waymo_raw_path), compression_type="")
    remaining = set(missing.keys())
    for local_index, data in enumerate(dataset.as_numpy_iterator()):
        if local_index > max_scene_index:
            break
        if local_index not in remaining:
            continue
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data)
        converted = _convert_waymo_scenario(
            scenario,
            version=str(version),
            source_file=str(resolved_waymo_raw_path),
        )
        scenario_id = missing[int(local_index)]
        out_path = expected_scenarionet_source_path(
            cache_root=cache_root,
            scenario_id=scenario_id,
            version=version,
        )
        with out_path.open("wb") as f:
            pickle.dump(converted, f)
        resolved[scenario_id] = out_path
        remaining.remove(int(local_index))
        if not remaining:
            break

    if remaining:
        missing_ids = [missing[idx] for idx in sorted(remaining)]
        raise FileNotFoundError(
            f"Unable to materialize ScenarioNet Waymo sources for: {missing_ids} "
            f"from raw path {resolved_waymo_raw_path!r}"
        )

    return resolved


def _main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "--materialize-json":
        payload = json.loads(argv[2])
        resolved = _materialize_scenarionet_waymo_sources_impl(
            scenario_ids=payload.get("scenario_ids", []),
            cache_root=payload.get("cache_root", ""),
            waymo_raw_path=payload.get("waymo_raw_path", DEFAULT_WOD_131_TRAIN_PATH),
            version=payload.get("version", "v1.2"),
        )
        print(json.dumps({str(key): str(value) for key, value in resolved.items()}, sort_keys=True))
        return 0
    raise SystemExit("Usage: scenarionet_waymo_export_source.py --materialize-json '<json-payload>'")


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
