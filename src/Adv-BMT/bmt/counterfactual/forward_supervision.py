from __future__ import annotations

import copy
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml
from easydict import EasyDict

from .types import stable_string_sort_key


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _normalize_track_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    text = str(value)
    return text


def _tracks_to_predict_ids(raw_scenario: Optional[Mapping[str, Any]], sample: Mapping[str, Any]) -> List[str]:
    if raw_scenario is not None:
        metadata = raw_scenario.get("metadata", {})
        tracks_to_predict = metadata.get("tracks_to_predict", {})
        if isinstance(tracks_to_predict, Mapping):
            return sorted((_normalize_track_id(key) for key in tracks_to_predict.keys()), key=stable_string_sort_key)
    original_sd = sample.get("original_SD")
    if isinstance(original_sd, Mapping):
        metadata = original_sd.get("metadata", {})
        tracks_to_predict = metadata.get("tracks_to_predict", {})
        if isinstance(tracks_to_predict, Mapping):
            return sorted((_normalize_track_id(key) for key in tracks_to_predict.keys()), key=stable_string_sort_key)
    return []


@dataclass
class ForwardLossAgentSummary:
    model_agent_slot: int
    raw_track_id: str
    receives_motion_loss: bool
    num_loss_steps: int
    motion_loss_mask: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class ForwardSupervisionExample:
    scenario_id: str
    sdc_id: str
    tracks_to_predict_ids: List[str]
    modeled_agent_ids: List[str]
    trainable_track_ids: List[str]
    sdc_receives_forward_loss: bool
    motion_loss_mask_shape: List[int]
    slot_to_raw_track_id: Dict[str, str]
    agents: List[ForwardLossAgentSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


def load_motion_config(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    control_code_dir: str | Path | None = None,
) -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    resolved_config_path = Path(config_path).expanduser() if config_path else repo_root / "cfgs" / "motion_default.yaml"
    with resolved_config_path.open("r", encoding="utf-8") as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config.ROOT_DIR = repo_root
    config.LOCAL_RANK = 0
    if data_dir:
        data_dir_text = str(Path(data_dir).expanduser())
        config.DATA.TRAINING_DATA_DIR = data_dir_text
        config.DATA.TEST_DATA_DIR = data_dir_text
    if control_code_dir is not None:
        config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = str(Path(control_code_dir).expanduser())
    return config


def _ensure_runtime_imports() -> None:
    project_root = Path(__file__).resolve().parents[4]
    vendored_scenarionet = project_root / "scenarionet"
    vendored_metadrive = project_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        import hydra as _hydra  # noqa: F401
    except Exception:
        hydra_stub = types.ModuleType("hydra")

        def _hydra_main(*args, **kwargs):
            def decorator(fn):
                return fn

            return decorator

        hydra_stub.main = _hydra_main
        sys.modules["hydra"] = hydra_stub

    try:
        from omegaconf import OmegaConf as _OmegaConf  # noqa: F401
    except Exception:
        omegaconf_stub = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def create(value):
                return value

            @staticmethod
            def merge(*values):
                return values[-1] if values else None

            @staticmethod
            def to_container(value, **kwargs):
                return value

        omegaconf_stub.OmegaConf = _OmegaConf
        omegaconf_stub.DictConfig = dict
        sys.modules["omegaconf"] = omegaconf_stub

    try:
        import lightning  # noqa: F401
        import lightning.pytorch  # noqa: F401
        from lightning.fabric.utilities.cloud_io import _load as _pl_load  # noqa: F401
    except Exception:
        lightning_stub = types.ModuleType("lightning")
        lightning_pytorch_stub = types.ModuleType("lightning.pytorch")
        lightning_fabric_stub = types.ModuleType("lightning.fabric")
        lightning_fabric_utilities_stub = types.ModuleType("lightning.fabric.utilities")
        lightning_fabric_cloud_io_stub = types.ModuleType("lightning.fabric.utilities.cloud_io")
        lightning_fabric_types_stub = types.ModuleType("lightning.fabric.utilities.types")
        lightning_pytorch_utilities_stub = types.ModuleType("lightning.pytorch.utilities")
        lightning_pytorch_migration_stub = types.ModuleType("lightning.pytorch.utilities.migration")
        lightning_pytorch_migration_utils_stub = types.ModuleType("lightning.pytorch.utilities.migration.utils")

        def _identity(value=None, *args, **kwargs):
            return value

        def _legacy_patch():
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

        lightning_fabric_cloud_io_stub._load = _identity
        lightning_fabric_types_stub._MAP_LOCATION_TYPE = object
        lightning_fabric_types_stub._PATH = object
        lightning_pytorch_utilities_stub.rank_zero_only = _identity
        lightning_pytorch_migration_stub.pl_legacy_patch = _legacy_patch
        lightning_pytorch_migration_utils_stub._pl_migrate_checkpoint = _identity

        sys.modules["lightning"] = lightning_stub
        sys.modules["lightning.pytorch"] = lightning_pytorch_stub
        sys.modules["lightning.fabric"] = lightning_fabric_stub
        sys.modules["lightning.fabric.utilities"] = lightning_fabric_utilities_stub
        sys.modules["lightning.fabric.utilities.cloud_io"] = lightning_fabric_cloud_io_stub
        sys.modules["lightning.fabric.utilities.types"] = lightning_fabric_types_stub
        sys.modules["lightning.pytorch.utilities"] = lightning_pytorch_utilities_stub
        sys.modules["lightning.pytorch.utilities.migration"] = lightning_pytorch_migration_stub
        sys.modules["lightning.pytorch.utilities.migration.utils"] = lightning_pytorch_migration_utils_stub

    try:
        from metadrive.scenario.scenario_description import ScenarioDescription as _SD  # type: ignore
        from metadrive.scenario.scenario_description import MetaDriveType as _MetaDriveType  # type: ignore
        for attr, value in {
            "UNSET": "UNSET",
            "VEHICLE": "VEHICLE",
            "PEDESTRIAN": "PEDESTRIAN",
            "CYCLIST": "CYCLIST",
            "OTHER": "OTHER",
            "TRAFFIC_LIGHT": "TRAFFIC_LIGHT",
        }.items():
            if not hasattr(_MetaDriveType, attr):
                setattr(_MetaDriveType, attr, value)
    except Exception:
        metadrive_stub = types.ModuleType("metadrive")
        metadrive_scenario_stub = types.ModuleType("metadrive.scenario")
        metadrive_sd_stub = types.ModuleType("metadrive.scenario.scenario_description")

        class _ScenarioDescription:
            STATE = "state"
            METADATA = "metadata"
            MAP_FEATURES = "map_features"
            DYNAMIC_MAP_STATES = "dynamic_map_states"
            LENGTH = "length"
            TRACKS = "tracks"
            ID = "id"

        class _MetaDriveType:
            UNSET = "UNSET"
            VEHICLE = "VEHICLE"
            PEDESTRIAN = "PEDESTRIAN"
            CYCLIST = "CYCLIST"
            OTHER = "OTHER"
            TRAFFIC_LIGHT = "TRAFFIC_LIGHT"

            @staticmethod
            def is_participant(_value):
                return True

            @staticmethod
            def is_vehicle(value):
                return str(value) == "VEHICLE"

            @staticmethod
            def is_pedestrian(value):
                return str(value) == "PEDESTRIAN"

            @staticmethod
            def is_cyclist(value):
                return str(value) == "CYCLIST"

            @staticmethod
            def is_lane(value):
                return str(value).startswith("LANE_")

            @staticmethod
            def is_sidewalk(_value):
                return False

            @staticmethod
            def is_road_boundary_line(_value):
                return False

            @staticmethod
            def is_road_line(_value):
                return False

            @staticmethod
            def is_broken_line(_value):
                return False

            @staticmethod
            def is_solid_line(_value):
                return False

            @staticmethod
            def is_yellow_line(_value):
                return False

            @staticmethod
            def is_white_line(_value):
                return False

            @staticmethod
            def is_driveway(_value):
                return False

            @staticmethod
            def is_crosswalk(_value):
                return False

            @staticmethod
            def is_speed_bump(_value):
                return False

            @staticmethod
            def is_stop_sign(_value):
                return False

            @staticmethod
            def is_traffic_light_in_green(value):
                return "GO" in str(value).upper()

            @staticmethod
            def is_traffic_light_in_yellow(value):
                upper = str(value).upper()
                return "CAUTION" in upper or "YELLOW" in upper

            @staticmethod
            def is_traffic_light_in_red(value):
                upper = str(value).upper()
                return "STOP" in upper or "RED" in upper

            @staticmethod
            def is_traffic_light_unknown(value):
                return value is None or "UNKNOWN" in str(value).upper()

        metadrive_sd_stub.ScenarioDescription = _ScenarioDescription
        metadrive_sd_stub.MetaDriveType = _MetaDriveType
        sys.modules["metadrive"] = metadrive_stub
        sys.modules["metadrive.scenario"] = metadrive_scenario_stub
        sys.modules["metadrive.scenario.scenario_description"] = metadrive_sd_stub


def preprocess_raw_scenario_for_forward_supervision(
    raw_scenario: Mapping[str, Any],
    *,
    config: Any,
) -> Dict[str, Any]:
    _ensure_runtime_imports()
    from bmt.dataset.preprocessor import preprocess_scenario_description
    from bmt.tokenization import get_tokenizer

    scenario_copy = copy.deepcopy(dict(raw_scenario))
    tracks = scenario_copy.get("tracks", {})
    if isinstance(tracks, Mapping):
        for track in tracks.values():
            if not isinstance(track, dict):
                continue
            state = track.get("state", {})
            if not isinstance(state, dict):
                continue
            num_steps = 0
            if "position" in state:
                num_steps = int(np.asarray(state["position"]).shape[0])
            defaults = {
                "length": 4.5,
                "width": 1.8,
                "height": 1.5,
            }
            for key, default_value in defaults.items():
                if key not in state:
                    state[key] = np.full((num_steps,), float(default_value), dtype=np.float32)
    dynamic_map_states = scenario_copy.get("dynamic_map_states", {})
    if isinstance(dynamic_map_states, Mapping):
        for state in dynamic_map_states.values():
            if not isinstance(state, dict):
                continue
            stop_point = state.get("stop_point")
            if stop_point is not None and not isinstance(stop_point, np.ndarray):
                state["stop_point"] = np.asarray(stop_point, dtype=np.float32)
    tokenizer = get_tokenizer(config=config)
    return preprocess_scenario_description(
        scenario=scenario_copy,
        config=copy.deepcopy(config),
        in_evaluation=False,
        keep_all_data=True,
        backward_prediction=False,
        tokenizer=tokenizer,
    )


def summarize_forward_supervision_for_sample(
    sample: Mapping[str, Any],
    *,
    raw_scenario: Optional[Mapping[str, Any]] = None,
) -> ForwardSupervisionExample:
    target_action_valid_mask = np.asarray(sample["decoder/target_action_valid_mask"], dtype=bool)
    if target_action_valid_mask.ndim != 2:
        raise ValueError(
            "decoder/target_action_valid_mask must be [T, N] for a single sample, "
            f"got shape={target_action_valid_mask.shape}"
        )

    scenario_id = _normalize_track_id(sample.get("metadata/scenario_id", sample.get("scenario_id", "")))
    sdc_id = _normalize_track_id(sample.get("metadata/sdc_name", ""))
    modeled_agent_ids = [_normalize_track_id(value) for value in np.asarray(sample.get("decoder/track_name", []), dtype=object).reshape(-1)]
    if not modeled_agent_ids and "encoder/modeled_agent_id" in sample and "encoder/track_name" in sample:
        modeled_ids = np.asarray(sample["encoder/modeled_agent_id"], dtype=int).reshape(-1)
        encoder_track_names = np.asarray(sample["encoder/track_name"], dtype=object).reshape(-1)
        modeled_agent_ids = []
        for idx in modeled_ids.tolist():
            if 0 <= int(idx) < encoder_track_names.shape[0]:
                modeled_agent_ids.append(_normalize_track_id(encoder_track_names[int(idx)]))
            else:
                modeled_agent_ids.append("")

    num_agents = int(target_action_valid_mask.shape[1])
    if len(modeled_agent_ids) < num_agents:
        modeled_agent_ids = modeled_agent_ids + [""] * (num_agents - len(modeled_agent_ids))
    modeled_agent_ids = modeled_agent_ids[:num_agents]

    agent_rows: List[ForwardLossAgentSummary] = []
    slot_to_track: Dict[str, str] = {}
    trainable_track_ids: List[str] = []
    for agent_slot in range(num_agents):
        track_id = _normalize_track_id(modeled_agent_ids[agent_slot])
        mask = target_action_valid_mask[:, agent_slot]
        receives_loss = bool(mask.any())
        if receives_loss and track_id:
            trainable_track_ids.append(track_id)
        slot_to_track[str(agent_slot)] = track_id
        agent_rows.append(
            ForwardLossAgentSummary(
                model_agent_slot=int(agent_slot),
                raw_track_id=track_id,
                receives_motion_loss=receives_loss,
                num_loss_steps=int(mask.sum()),
                motion_loss_mask=mask.astype(np.int64).tolist(),
            )
        )

    return ForwardSupervisionExample(
        scenario_id=scenario_id,
        sdc_id=sdc_id,
        tracks_to_predict_ids=_tracks_to_predict_ids(raw_scenario, sample),
        modeled_agent_ids=modeled_agent_ids,
        trainable_track_ids=sorted(set(trainable_track_ids), key=stable_string_sort_key),
        sdc_receives_forward_loss=bool(
            any(row.raw_track_id == sdc_id and row.receives_motion_loss for row in agent_rows)
        ),
        motion_loss_mask_shape=[int(target_action_valid_mask.shape[0]), int(target_action_valid_mask.shape[1])],
        slot_to_raw_track_id=slot_to_track,
        agents=agent_rows,
    )


def summarize_forward_supervision_for_raw_scenario(
    raw_scenario: Mapping[str, Any],
    *,
    config: Any,
) -> ForwardSupervisionExample:
    sample = preprocess_raw_scenario_for_forward_supervision(raw_scenario, config=config)
    return summarize_forward_supervision_for_sample(sample, raw_scenario=raw_scenario)


def summarize_forward_supervision_for_batch(batch: Mapping[str, Any]) -> List[ForwardSupervisionExample]:
    scenario_ids = np.asarray(batch["metadata/scenario_id"], dtype=object).reshape(-1)
    if "metadata/sdc_name" in batch:
        sdc_ids = np.asarray(batch["metadata/sdc_name"], dtype=object).reshape(-1)
    else:
        sdc_ids = np.asarray([""] * int(scenario_ids.shape[0]), dtype=object)
    target_action_valid_mask = np.asarray(batch["decoder/target_action_valid_mask"], dtype=bool)
    decoder_track_names = batch.get("decoder/track_name", [])
    original_sd = batch.get("original_SD", [])
    encoder_track_names = batch.get("encoder/track_name", [])
    encoded_modeled_ids = batch.get("encoder/modeled_agent_id")

    examples: List[ForwardSupervisionExample] = []
    for batch_idx in range(int(target_action_valid_mask.shape[0])):
        sample_like: Dict[str, Any] = {
            "metadata/scenario_id": scenario_ids[batch_idx],
            "metadata/sdc_name": sdc_ids[batch_idx],
            "decoder/target_action_valid_mask": target_action_valid_mask[batch_idx],
        }
        if batch_idx < len(decoder_track_names):
            sample_like["decoder/track_name"] = decoder_track_names[batch_idx]
        elif encoded_modeled_ids is not None and batch_idx < len(encoder_track_names):
            modeled_ids = np.asarray(encoded_modeled_ids[batch_idx], dtype=int).reshape(-1)
            track_names = np.asarray(encoder_track_names[batch_idx], dtype=object).reshape(-1)
            sample_like["decoder/track_name"] = [
                track_names[int(track_idx)]
                for track_idx in modeled_ids.tolist()
                if 0 <= int(track_idx) < track_names.shape[0]
            ]
        if batch_idx < len(original_sd):
            sample_like["original_SD"] = original_sd[batch_idx]
        raw_scenario = original_sd[batch_idx] if batch_idx < len(original_sd) and isinstance(original_sd[batch_idx], Mapping) else None
        if sample_like["metadata/sdc_name"] == "" and raw_scenario is not None:
            sample_like["metadata/sdc_name"] = raw_scenario.get("metadata", {}).get("sdc_id", "")
        examples.append(summarize_forward_supervision_for_sample(sample_like, raw_scenario=raw_scenario))
    return examples


def build_forward_supervision_summary_payload(examples: Sequence[ForwardSupervisionExample]) -> Dict[str, Any]:
    total_examples = len(examples)
    total_agents = int(sum(len(example.agents) for example in examples))
    total_trainable_agents = int(sum(sum(1 for row in example.agents if row.receives_motion_loss) for example in examples))
    total_sdc_with_loss = int(sum(1 for example in examples if example.sdc_receives_forward_loss))
    return {
        "num_examples": int(total_examples),
        "num_modeled_agents": int(total_agents),
        "num_trainable_agents": int(total_trainable_agents),
        "num_examples_with_sdc_forward_loss": int(total_sdc_with_loss),
        "fraction_examples_with_sdc_forward_loss": float(total_sdc_with_loss / total_examples) if total_examples > 0 else 0.0,
        "scenario_ids": [example.scenario_id for example in examples],
        "trainable_track_id_counts": {
            track_id: count_forward_supervision_track_hits(examples, track_id)
            for track_id in sorted(
                {track_id for example in examples for track_id in example.trainable_track_ids},
                key=stable_string_sort_key,
            )
        },
    }


def count_forward_supervision_track_hits(examples: Sequence[ForwardSupervisionExample], track_id: str) -> int:
    track_id = _normalize_track_id(track_id)
    return int(sum(track_id in example.trainable_track_ids for example in examples))


def get_forward_loss_track_ids(raw_scenario: Mapping[str, Any], *, config: Any) -> List[str]:
    example = summarize_forward_supervision_for_raw_scenario(raw_scenario, config=config)
    return list(example.trainable_track_ids)
