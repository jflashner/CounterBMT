from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    matplotlib = None
    plt = None

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    modern_src = repo_root / "src"
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, modern_src, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual.forward_supervision import (
    preprocess_raw_scenario_for_forward_supervision,
    summarize_forward_supervision_for_sample,
)
from bmt.counterfactual.normalize import load_raw_scenario
from bmt.counterfactual.sdc_path_control import normalize_semantic_label, semantic_label_to_id
from bmt.counterfactual.scenarionet_waymo_export_source import (
    DEFAULT_WOD_131_TRAIN_PATH,
    materialize_scenarionet_waymo_sources,
)
from bmt.counterfactual.sdc_semantic_control import extract_model_frame
from bmt.eval.evaluate_scenario_metrics import EvaluationLightningModule, _load_eval_model
from bmt.eval.scenario_evaluator import Evaluator
from bmt.tokenization import get_tokenizer
from bmt.utils import utils
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config
from counter_bmt.scenario_export import (
    create_replay_script,
    export_victim_centric_ground_truth_scenario,
    export_victim_centric_scenario,
)
from scripts.counterfactual.render_sdc_semantic_eval_examples import _to_numpy_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe generation-only semantic-only control on an arbitrary modeled agent."
    )
    parser.add_argument("--scenario-pkl", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--agent-id", type=str, required=True)
    parser.add_argument("--victim-agent-id", type=str, default="sdc")
    parser.add_argument("--semantic-label", type=str, required=True)
    parser.add_argument("--semantic-confidence", type=float, default=1.0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=-1)
    parser.add_argument("--rollout-sampling-method", type=str, default="argmax")
    parser.add_argument("--rollout-temperature", type=float, default=-1.0)
    parser.add_argument("--rollout-topp", type=float, default=-1.0)
    parser.add_argument("--load-mode", type=str, default="forgiving_state_dict")
    parser.add_argument("--teacher-ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-label", action="append", default=[])
    parser.add_argument("--debug-trace", action="store_true")
    parser.add_argument("--debug-compare-sdc", action="store_true")
    parser.add_argument("--export-victim-centric", action="store_true")
    parser.add_argument("--export-scenario-dir", type=str, default="")
    parser.add_argument("--intervention-name", type=str, default="")
    parser.add_argument(
        "--export-source-mode",
        type=str,
        choices=("auto", "raw_scenario_pkl", "scenarionet_waymo"),
        default="auto",
    )
    parser.add_argument("--waymo-raw-path", type=str, default=DEFAULT_WOD_131_TRAIN_PATH)
    parser.add_argument("--waymo-source-version", type=str, default="v1.2")
    parser.add_argument("--export-source-cache-dir", type=str, default="")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    config = cfg_from_yaml_file(cfg_path, config)
    config.DATA.COUNTERFACTUAL_MODE = "sdc_semantic_only"
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    teacher_ckpt_text = str(args.teacher_ckpt or args.ckpt).strip()
    teacher_ckpt_path = Path(teacher_ckpt_text).expanduser()
    if not teacher_ckpt_path.is_absolute():
        teacher_ckpt_path = (Path.cwd() / teacher_ckpt_path).resolve()
    config.MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT = str(teacher_ckpt_path)
    return config


def _load_model(*, config: Any, ckpt_path: str, load_mode: str):
    from bmt.models.motionlm_lightning import MotionLMLightning

    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs/motion_default.yaml", global_config)
    resolved_ckpt = Path(ckpt_path).expanduser()
    if not resolved_ckpt.is_absolute():
        resolved_ckpt = (Path.cwd() / resolved_ckpt).resolve()
    model, load_report = load_model_from_checkpoint_forgiving(
        config=config,
        ckpt_path=str(resolved_ckpt),
        load_mode=load_mode,
        strict_state_dict=(load_mode == "strict_state_dict"),
        map_location="cpu",
        checkpoint_surgery_func=utils.checkpoint_surgery_func,
    )
    assert isinstance(model, MotionLMLightning)
    model.eval()
    return model, load_report


def _build_eval_module(
    *,
    config: Any,
    ckpt_path: str,
    device: torch.device,
    save_path: str | Path,
    model: Any | None = None,
) -> tuple[EvaluationLightningModule, Any]:
    if model is None:
        resolved_ckpt = Path(ckpt_path).expanduser()
        if not resolved_ckpt.is_absolute():
            resolved_ckpt = (Path.cwd() / resolved_ckpt).resolve()
        model = _load_eval_model(config, str(resolved_ckpt))
    tokenizer = get_tokenizer(config)
    module = EvaluationLightningModule(
        model=model,
        evaluator=Evaluator(key_metrics_only=True),
        tokenizer=tokenizer,
        config=config,
        dataset=None,
        eval_mode="GPTmodel",
        multi_mode=False,
        num_modes=1,
        save_path=str(Path(save_path).expanduser().resolve()),
    )
    module.eval()
    module.model.eval()
    module.model.to(device)
    return module, tokenizer


def _resolve_device(requested: str) -> torch.device:
    text = str(requested).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def _normalize_track_id(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        value = value.item()
    return "" if value is None else str(value)


def _safe_name(value: Any) -> str:
    text = _normalize_track_id(value).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _single_sample_to_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
    object_keys = {
        "raw_scenario_description",
        "original_SD",
        "encoder/track_name",
        "decoder/track_name",
        "eval/track_name",
        "cf/debug_meta",
    }
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in object_keys:
            batch[key] = [value]
        elif isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                batch[key] = np.asarray([value])
            else:
                batch[key] = utils.numpy_to_torch(value[None])
        elif isinstance(value, (int, float, bool, np.integer, np.floating)):
            batch[key] = utils.numpy_to_torch(np.asarray([value]))
        elif isinstance(value, str):
            batch[key] = np.asarray([value])
        else:
            batch[key] = [value]
    return batch


def _to_torch_device(value: Any, *, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_torch_device(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch_device(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch_device(item, device=device) for item in value)
    return value


def _optional_positive_float(value: float) -> float | None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _wrap_angle_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.arctan2(np.sin(array), np.cos(array)).astype(np.float32)


def _prepare_batch_for_autoregressive_rollout(
    batch_torch: Dict[str, Any],
    *,
    raw_scenario: Mapping[str, Any],
) -> Dict[str, Any]:
    rollout_batch = copy.deepcopy(batch_torch)
    required = (
        "decoder/agent_position",
        "decoder/agent_velocity",
        "decoder/agent_heading",
        "decoder/agent_valid_mask",
    )
    if all(key in rollout_batch for key in required):
        return rollout_batch

    modeled_agent_id = rollout_batch.get("encoder/modeled_agent_id")
    if modeled_agent_id is None:
        raise KeyError("encoder/modeled_agent_id is required to reconstruct decoder/agent_* tensors for rollout.")
    if modeled_agent_id.ndim != 2:
        raise ValueError(f"Expected encoder/modeled_agent_id to be [B,N], got {tuple(modeled_agent_id.shape)}")

    batch_size, num_agents = [int(dim) for dim in modeled_agent_id.shape]
    if batch_size != 1:
        raise ValueError(f"Autoregressive rollout probe expects batch_size=1, got {batch_size}")

    map_center, map_heading = extract_model_frame(raw_scenario)
    tracks = dict(raw_scenario.get("tracks", {}) or {})
    if not tracks:
        raise ValueError("Raw scenario does not contain any tracks for rollout reconstruction.")

    horizon = 0
    for track_payload in tracks.values():
        state = dict(dict(track_payload).get("state") or {})
        horizon = max(horizon, int(np.asarray(state.get("position", []), dtype=np.float32).shape[0]))
    if horizon <= 0:
        raise ValueError("Unable to infer rollout horizon from raw scenario track positions.")

    agent_position = np.zeros((batch_size, horizon, num_agents, 3), dtype=np.float32)
    agent_velocity = np.zeros((batch_size, horizon, num_agents, 2), dtype=np.float32)
    agent_heading = np.zeros((batch_size, horizon, num_agents), dtype=np.float32)
    agent_valid_mask = np.zeros((batch_size, horizon, num_agents), dtype=bool)

    decoder_track_names = rollout_batch.get("decoder/track_name")
    track_names_arr = None if decoder_track_names is None else np.asarray(decoder_track_names)
    if track_names_arr is not None and track_names_arr.ndim == 2:
        raw_track_names = track_names_arr[0].astype(str)
    else:
        encoder_track_names = rollout_batch.get("encoder/track_name")
        if encoder_track_names is None:
            raise KeyError("encoder/track_name is required to reconstruct decoder/agent_* tensors for rollout.")
        encoder_track_names_np = np.asarray(encoder_track_names[0]).astype(str)
        modeled_agent_ids_np = modeled_agent_id.detach().cpu().numpy().astype(np.int64)
        raw_track_names = np.asarray(
            [encoder_track_names_np[int(agent_id)] for agent_id in modeled_agent_ids_np[0].tolist()],
            dtype=str,
        )

    for agent_idx, raw_track_name in enumerate(raw_track_names.tolist()):
        track = tracks.get(str(raw_track_name))
        if track is None:
            continue
        state = dict(dict(track).get("state") or {})
        position_world = np.asarray(state.get("position", []), dtype=np.float32)
        heading_world = np.asarray(state.get("heading", []), dtype=np.float32).reshape(-1)
        velocity_world = np.asarray(state.get("velocity", []), dtype=np.float32)
        valid = np.asarray(state.get("valid", []), dtype=bool).reshape(-1)
        if position_world.ndim != 2 or position_world.shape[0] == 0:
            continue
        if velocity_world.ndim != 2 or velocity_world.shape[1] < 2:
            velocity_world = np.zeros((position_world.shape[0], 2), dtype=np.float32)
        length = min(horizon, position_world.shape[0], heading_world.shape[0], velocity_world.shape[0], valid.shape[0])
        if length <= 0:
            continue

        xy_world = position_world[:length, :2]
        centered = xy_world - np.asarray(map_center, dtype=np.float32).reshape(1, 3)[:, :2]
        if float(map_heading) == 0.0:
            xy_model = centered.astype(np.float32)
            vel_model = velocity_world[:length, :2].astype(np.float32)
        else:
            c = float(np.cos(-float(map_heading)))
            s = float(np.sin(-float(map_heading)))
            xy_model = np.stack(
                [c * centered[:, 0] - s * centered[:, 1], s * centered[:, 0] + c * centered[:, 1]],
                axis=-1,
            ).astype(np.float32)
            vel_xy = velocity_world[:length, :2].astype(np.float32)
            vel_model = np.stack(
                [c * vel_xy[:, 0] - s * vel_xy[:, 1], s * vel_xy[:, 0] + c * vel_xy[:, 1]],
                axis=-1,
            ).astype(np.float32)
        z_model = (
            position_world[:length, 2].astype(np.float32) - float(map_center[2])
            if position_world.shape[1] >= 3
            else np.zeros((length,), dtype=np.float32)
        )
        heading_model = _wrap_angle_array(heading_world[:length] - float(map_heading))

        agent_position[0, :length, agent_idx, :2] = xy_model
        agent_position[0, :length, agent_idx, 2] = z_model
        agent_velocity[0, :length, agent_idx, :] = vel_model
        agent_heading[0, :length, agent_idx] = heading_model
        agent_valid_mask[0, :length, agent_idx] = valid[:length]

    device = modeled_agent_id.device
    rollout_batch["decoder/agent_position"] = torch.as_tensor(agent_position, device=device, dtype=torch.float32)
    rollout_batch["decoder/agent_velocity"] = torch.as_tensor(agent_velocity, device=device, dtype=torch.float32)
    rollout_batch["decoder/agent_heading"] = torch.as_tensor(agent_heading, device=device, dtype=torch.float32)
    rollout_batch["decoder/agent_valid_mask"] = torch.as_tensor(agent_valid_mask, device=device, dtype=torch.bool)
    if "in_evaluation" not in rollout_batch:
        rollout_batch["in_evaluation"] = torch.ones((batch_size,), dtype=torch.bool, device=device)
    return rollout_batch


def _adapt_rollout_output_for_probe(
    *,
    base_batch: Dict[str, Any],
    rollout_output: Dict[str, Any],
) -> Dict[str, Any]:
    eval_output = copy.deepcopy(rollout_output)
    rollout_logits = rollout_output["decoder/output_logit"]
    horizon = int(rollout_logits.shape[1])
    initial_pos = base_batch["decoder/modeled_agent_position"][:, :1]
    initial_heading = base_batch["decoder/modeled_agent_heading"][:, :1]
    initial_velocity = base_batch["decoder/modeled_agent_velocity"][:, :1]
    rollout_next_pos = rollout_output["decoder/debug_ar_pos"][:, :horizon]
    rollout_next_heading = rollout_output["decoder/debug_ar_head"][:, :horizon]
    rollout_next_velocity = rollout_output["decoder/debug_ar_vel"][:, :horizon]
    if horizon > 1:
        current_pos = torch.cat([initial_pos, rollout_next_pos[:, :-1]], dim=1)
        current_heading = torch.cat([initial_heading, rollout_next_heading[:, :-1]], dim=1)
        current_velocity = torch.cat([initial_velocity, rollout_next_velocity[:, :-1]], dim=1)
    else:
        current_pos = initial_pos.clone()
        current_heading = initial_heading.clone()
        current_velocity = initial_velocity.clone()

    eval_output["decoder/modeled_agent_position"] = current_pos
    eval_output["decoder/modeled_agent_heading"] = current_heading
    eval_output["decoder/modeled_agent_velocity"] = current_velocity
    eval_output["decoder/modeled_agent_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
    eval_output["decoder/modeled_agent_delta"] = rollout_next_pos[..., :2] - current_pos[..., :2]
    eval_output["decoder/input_action"] = rollout_output["decoder/output_action"][:, :horizon]
    eval_output["decoder/input_action_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
    eval_output["decoder/target_action_valid_mask"] = rollout_output["decoder/input_action_valid_mask"][:, :horizon]
    eval_output["decoder/input_step"] = torch.arange(horizon, device=rollout_logits.device, dtype=torch.long)
    eval_output["decoder/rollout_next_position"] = rollout_next_pos
    eval_output["decoder/rollout_next_heading"] = rollout_next_heading
    eval_output["decoder/rollout_next_velocity"] = rollout_next_velocity
    return eval_output


def _model_to_world(points_model_xy: np.ndarray, *, map_center_world: np.ndarray, map_heading_world: float) -> np.ndarray:
    xy = np.asarray(points_model_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if float(map_heading_world) == 0.0:
        return (xy + np.asarray(map_center_world, dtype=np.float32).reshape(1, 3)[:, :2]).astype(np.float32)
    c = math.cos(float(map_heading_world))
    s = math.sin(float(map_heading_world))
    x_world = c * xy[:, 0] - s * xy[:, 1] + float(map_center_world[0])
    y_world = s * xy[:, 0] + c * xy[:, 1] + float(map_center_world[1])
    return np.stack([x_world, y_world], axis=-1).astype(np.float32)


def _extract_target_rollout_world(
    eval_output: Dict[str, Any],
    *,
    target_slot: int,
    map_center_world: np.ndarray,
    map_heading_world: float,
) -> np.ndarray:
    current_model_xy = np.asarray(
        eval_output["decoder/modeled_agent_position"][0, :, target_slot, :2].detach().cpu(),
        dtype=np.float32,
    )
    rollout_next_model_xy = np.asarray(
        eval_output["decoder/rollout_next_position"][0, :, target_slot, :2].detach().cpu(),
        dtype=np.float32,
    )
    rollout_traj_model_xy = np.concatenate([current_model_xy[:1], rollout_next_model_xy], axis=0).astype(np.float32)
    return _model_to_world(
        rollout_traj_model_xy,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )


def _extract_target_rollout_world_from_output_np(
    output_np: Mapping[str, Any],
    *,
    target_slot: int,
    map_center_world: np.ndarray,
    map_heading_world: float,
) -> np.ndarray:
    rollout_model_xy = np.asarray(output_np["decoder/reconstructed_position"], dtype=np.float32)[:, target_slot, :2]
    valid_mask = np.asarray(output_np["decoder/reconstructed_valid_mask"], dtype=bool)[:, target_slot]
    return _model_to_world(
        rollout_model_xy[valid_mask],
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )


def _extract_target_reference_world(
    batch_torch: Dict[str, Any],
    *,
    target_slot: int,
    map_center_world: np.ndarray,
    map_heading_world: float,
) -> np.ndarray:
    current_model_xy = np.asarray(
        batch_torch["decoder/modeled_agent_position"][0, :, target_slot, :2].detach().cpu(),
        dtype=np.float32,
    )
    return _model_to_world(
        current_model_xy,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )


def _extract_target_reference_world_from_sample(
    sample: Mapping[str, Any],
    *,
    target_slot: int,
    map_center_world: np.ndarray,
    map_heading_world: float,
) -> np.ndarray:
    current_model_xy = np.asarray(sample["decoder/agent_position"], dtype=np.float32)[:, target_slot, :2]
    valid_mask = np.asarray(sample["decoder/agent_valid_mask"], dtype=bool)[:, target_slot]
    return _model_to_world(
        current_model_xy[valid_mask],
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )


def _extract_all_reference_world(
    batch_torch: Dict[str, Any],
    *,
    map_center_world: np.ndarray,
    map_heading_world: float,
    modeled_agent_ids: Sequence[Any],
) -> Dict[str, np.ndarray]:
    modeled_xy = np.asarray(
        batch_torch["decoder/modeled_agent_position"][0, :, :, :2].detach().cpu(),
        dtype=np.float32,
    )
    all_reference_world: Dict[str, np.ndarray] = {}
    for slot, agent_id in enumerate(modeled_agent_ids):
        track_id = _normalize_track_id(agent_id)
        if not track_id:
            continue
        all_reference_world[track_id] = _model_to_world(
            modeled_xy[:, slot, :],
            map_center_world=map_center_world,
            map_heading_world=map_heading_world,
        )
    return all_reference_world


def _extract_all_reference_world_from_sample(
    sample: Mapping[str, Any],
    *,
    map_center_world: np.ndarray,
    map_heading_world: float,
    modeled_agent_ids: Sequence[Any],
) -> Dict[str, np.ndarray]:
    modeled_xy = np.asarray(sample["decoder/agent_position"], dtype=np.float32)[..., :2]
    valid_mask = np.asarray(sample["decoder/agent_valid_mask"], dtype=bool)
    all_reference_world: Dict[str, np.ndarray] = {}
    for slot, agent_id in enumerate(modeled_agent_ids):
        track_id = _normalize_track_id(agent_id)
        if not track_id:
            continue
        slot_xy = modeled_xy[:, slot, :]
        slot_valid = valid_mask[:, slot]
        all_reference_world[track_id] = _model_to_world(
            slot_xy[slot_valid],
            map_center_world=map_center_world,
            map_heading_world=map_heading_world,
        )
    return all_reference_world


def _rank_victim_candidates(
    *,
    modeled_agent_ids: Sequence[Any],
    all_reference_world: Mapping[str, np.ndarray],
    adversary_track_id: str,
    adversary_controlled_world_xy: np.ndarray,
) -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    adversary_xy = np.asarray(adversary_controlled_world_xy, dtype=np.float32)
    for slot, agent_id in enumerate(modeled_agent_ids):
        track_id = _normalize_track_id(agent_id)
        if not track_id or track_id == str(adversary_track_id):
            continue
        victim_xy = np.asarray(all_reference_world.get(track_id, np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        compare_len = int(min(adversary_xy.shape[0], victim_xy.shape[0]))
        if compare_len <= 0:
            continue
        pairwise_distance = np.linalg.norm(adversary_xy[:compare_len] - victim_xy[:compare_len], axis=-1)
        min_idx = int(pairwise_distance.argmin())
        candidates.append(
            {
                "agent_id": track_id,
                "slot": int(slot),
                "compare_len": int(compare_len),
                "min_distance_m": float(pairwise_distance[min_idx]),
                "min_distance_step": int(min_idx),
                "mean_distance_m": float(pairwise_distance.mean()),
            }
        )
    candidates.sort(key=lambda item: (item["min_distance_m"], item["mean_distance_m"], item["agent_id"]))
    return candidates


def _resolve_victim_selection(
    *,
    victim_agent_id_arg: str,
    modeled_agent_ids: Sequence[Any],
    all_reference_world: Mapping[str, np.ndarray],
    adversary_track_id: str,
    adversary_controlled_world_xy: np.ndarray,
    sdc_track_id: str,
) -> Dict[str, Any]:
    ranked = _rank_victim_candidates(
        modeled_agent_ids=modeled_agent_ids,
        all_reference_world=all_reference_world,
        adversary_track_id=adversary_track_id,
        adversary_controlled_world_xy=adversary_controlled_world_xy,
    )
    if not ranked:
        raise ValueError(
            f"Unable to select a victim for adversary '{adversary_track_id}': no other modeled agents have usable reference trajectories."
        )

    requested = _normalize_track_id(victim_agent_id_arg)
    if not requested or requested.lower() == "auto":
        selected = dict(ranked[0])
        selected["auto_selected"] = True
        selected["selection_mode"] = "auto"
    elif requested.lower() == "sdc":
        match = next((row for row in ranked if row["agent_id"] == str(sdc_track_id)), None)
        if match is None:
            available = [row["agent_id"] for row in ranked[:12]]
            raise ValueError(
                f"Requested victim agent 'sdc' resolved to '{sdc_track_id}', which is not an eligible modeled victim. "
                f"Available candidates include: {available}"
            )
        selected = dict(match)
        selected["auto_selected"] = False
        selected["selection_mode"] = "sdc"
    else:
        if requested == str(adversary_track_id):
            raise ValueError("victim-agent-id must differ from the adversary agent-id.")
        match = next((row for row in ranked if row["agent_id"] == requested), None)
        if match is None:
            available = [row["agent_id"] for row in ranked[:12]]
            raise ValueError(
                f"Requested victim agent '{requested}' is not an eligible modeled victim. Available candidates include: {available}"
            )
        selected = dict(match)
        selected["auto_selected"] = False
        selected["selection_mode"] = "manual"

    selected["ranked_candidates"] = ranked
    selected["sdc_is_closest"] = bool(ranked[0]["agent_id"] == str(sdc_track_id))
    selected["closest_candidate"] = dict(ranked[0])
    return selected


def _extract_target_action_tokens(eval_output: Dict[str, Any], *, target_slot: int) -> np.ndarray:
    action_tokens = np.asarray(eval_output["decoder/input_action"][0, :, target_slot].detach().cpu(), dtype=np.int64)
    return action_tokens.reshape(-1)


def _extract_target_action_tokens_from_output_np(output_np: Mapping[str, Any], *, target_slot: int) -> np.ndarray:
    action_tokens = np.asarray(output_np["decoder/output_action"], dtype=np.int64)[:, target_slot]
    return action_tokens.reshape(-1)


def _path_length_m(points_xy: np.ndarray) -> float:
    xy = np.asarray(points_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(xy[1:] - xy[:-1], axis=-1).sum())


def _json_ready(value: Any) -> Any:
    if torch.is_tensor(value):
        return _json_ready(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _tensor_summary(value: Any, *, max_entries: int = 16) -> Dict[str, Any]:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    summary: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size == 0:
        summary["values"] = []
        return summary
    flat = arr.reshape(-1)
    summary["numel"] = int(arr.size)
    if np.issubdtype(arr.dtype, np.number):
        summary["min"] = float(flat.min())
        summary["max"] = float(flat.max())
        summary["mean"] = float(flat.mean())
    summary["values"] = _json_ready(flat[: max_entries])
    return summary


def _summarize_control_fields(sample: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "cf/semantic_label_id",
        "cf/sdc_semantic_label_id",
        "cf/semantic_confidence",
        "cf/sdc_semantic_confidence",
        "cf/time_window_mask",
        "cf/decision_agent_mask",
        "cf/conditioning_eligible",
        "cf/control_available",
        "cf/sdc_control_available",
        "cf/is_factual",
        "cf/sdc_is_factual",
        "cf/runtime_probe_enabled",
    )
    summary: Dict[str, Any] = {}
    for key in keys:
        if key not in sample:
            continue
        value = sample[key]
        if key == "cf/time_window_mask":
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            summary[key] = {
                "shape": list(arr.shape),
                "active_steps": [int(x) for x in np.flatnonzero(arr > 0.0).tolist()],
                "sum": float(arr.sum()),
            }
        elif key == "cf/decision_agent_mask":
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            summary[key] = {
                "shape": list(arr.shape),
                "active_slots": [int(x) for x in np.flatnonzero(arr > 0.0).tolist()],
                "sum": float(arr.sum()),
            }
        else:
            summary[key] = _tensor_summary(value)
    return summary


def _build_time_window_mask(*, horizon: int, start_step: int, end_step: int) -> np.ndarray:
    start = int(max(0, start_step))
    end = int(horizon - 1 if end_step < 0 else min(end_step, horizon - 1))
    if start > end:
        raise ValueError(f"Invalid control window: start_step={start} is after end_step={end}.")
    mask = np.zeros((horizon,), dtype=np.float32)
    mask[start : end + 1] = 1.0
    return mask


def _build_control_sample(
    *,
    base_sample: Mapping[str, Any],
    semantic_label: str,
    semantic_confidence: float,
    time_window_mask: np.ndarray,
    decision_agent_mask: np.ndarray,
) -> Dict[str, Any]:
    sample = dict(base_sample)
    label_id = int(semantic_label_to_id(semantic_label))
    control_available = bool(decision_agent_mask.sum() > 0 and time_window_mask.sum() > 0)
    factual = False
    sample.update(
        {
            "cf/semantic_label_id": int(label_id),
            "cf/sdc_semantic_label_id": int(label_id),
            "cf/semantic_confidence": float(semantic_confidence),
            "cf/sdc_semantic_confidence": float(semantic_confidence),
            "cf/time_window_mask": np.asarray(time_window_mask, dtype=np.float32),
            "cf/decision_agent_mask": np.asarray(decision_agent_mask, dtype=np.float32),
            "cf/conditioning_eligible": int(control_available),
            "cf/control_available": int(control_available),
            "cf/sdc_control_available": int(control_available),
            "cf/is_factual": int(factual),
            "cf/sdc_is_factual": int(factual),
            "cf/debug_meta": {
                "source": "probe_agent_semantic_rollout",
                "semantic_label": str(semantic_label),
                "semantic_label_id": int(label_id),
                "semantic_confidence": float(semantic_confidence),
                "control_available": bool(control_available),
            },
        }
    )
    return sample


def _build_runtime_probe_sample(*, sample: Mapping[str, Any]) -> Dict[str, Any]:
    traced = dict(sample)
    traced["cf/runtime_probe_enabled"] = 1
    return traced


def _run_rollout(
    module: EvaluationLightningModule,
    tokenizer: Any,
    *,
    raw_sample: Mapping[str, Any],
) -> Dict[str, Any]:
    with torch.no_grad():
        input_data = module.preprocess_GPTmodel(copy.deepcopy(dict(raw_sample)), backward_prediction=False)
        rollout_output = module.GPT_AR(input_data, backward_prediction=False, teacher_forcing=False)
    detok_output = tokenizer.detokenize(
        rollout_output,
        detokenizing_gt=False,
        backward_prediction=False,
        teacher_forcing=False,
    )
    output_np = _to_numpy_output(detok_output)
    return {
        "rollout_output": rollout_output,
        "eval_output": detok_output,
        "output_np": output_np,
    }


def _run_one_step_runtime_probe(
    module: EvaluationLightningModule,
    *,
    raw_sample: Mapping[str, Any],
) -> Dict[str, Any]:
    with torch.no_grad():
        input_data = module.preprocess_GPTmodel(copy.deepcopy(dict(raw_sample)), backward_prediction=False)
        rollout_output = module.model.model.autoregressive_rollout_GPT(
            input_data,
            num_decode_steps=1,
            sampling_method=module.config.SAMPLING.SAMPLING_METHOD,
            temperature=module.config.SAMPLING.TEMPERATURE,
            topp=module.config.SAMPLING.TOPP,
            backward_prediction=False,
        )
    return {
        "input_data": input_data,
        "rollout_output": rollout_output,
    }


def _run_rollout_direct(
    model: Any,
    *,
    batch_torch: Dict[str, Any],
    raw_scenario: Mapping[str, Any],
    sampling_method: str,
    temperature: float | None,
    topp: float | None,
) -> Dict[str, Any]:
    rollout_batch = _prepare_batch_for_autoregressive_rollout(batch_torch, raw_scenario=raw_scenario)
    motion_decoder = getattr(model.model, "motion_decoder", None)
    previous_null_dropout_prob = getattr(motion_decoder, "cf_null_dropout_prob", None)
    if motion_decoder is not None and previous_null_dropout_prob is not None:
        motion_decoder.cf_null_dropout_prob = 0.0
    try:
        with torch.no_grad():
            rollout_output = model.model.autoregressive_rollout(
                copy.deepcopy(rollout_batch),
                num_decode_steps=None,
                sampling_method=str(sampling_method),
                temperature=temperature,
                topp=topp,
                autoregressive_start_step=0,
            )
    finally:
        if motion_decoder is not None and previous_null_dropout_prob is not None:
            motion_decoder.cf_null_dropout_prob = previous_null_dropout_prob
    eval_output = _adapt_rollout_output_for_probe(base_batch=batch_torch, rollout_output=rollout_output)
    return {
        "rollout_output": rollout_output,
        "eval_output": eval_output,
    }


def _extract_topk_logits(
    rollout_output: Mapping[str, Any],
    *,
    target_slot: int,
    topk: int = 8,
) -> Dict[str, Any]:
    logits = np.asarray(rollout_output["decoder/output_logit"][0, 0, target_slot].detach().cpu(), dtype=np.float32)
    order = np.argsort(logits)[::-1]
    top_indices = order[:topk]
    return {
        "target_slot": int(target_slot),
        "top_indices": [int(x) for x in top_indices.tolist()],
        "top_logits": [float(logits[idx]) for idx in top_indices.tolist()],
        "argmax_index": int(order[0]),
        "argmax_logit": float(logits[order[0]]),
    }


def _extract_runtime_probe_summary(
    probe_output: Mapping[str, Any],
    *,
    target_slot: int,
) -> Dict[str, Any]:
    input_data = probe_output["input_data"]
    rollout_output = probe_output["rollout_output"]
    randomized_agent_id = rollout_output.get(
        "decoder/randomized_modeled_agent_id",
        input_data.get("decoder/randomized_modeled_agent_id"),
    )
    return {
        "preprocess_cf": _summarize_control_fields(input_data),
        "randomized_modeled_agent_id": _json_ready(
            randomized_agent_id[0] if torch.is_tensor(randomized_agent_id) else randomized_agent_id
        ),
        "control_forward_mode": str(rollout_output.get("decoder/control_forward_mode", "")),
        "control_kind": str(rollout_output.get("decoder/control_probe_control_kind", "")),
        "control_available_mask": _json_ready(rollout_output.get("decoder/control_probe_available_mask")),
        "supervision_pos_mask_summary": _tensor_summary(
            rollout_output.get("decoder/control_probe_supervision_pos_mask", np.zeros((0,), dtype=np.float32))
        ),
        "control_pos_mask_summary": _tensor_summary(
            rollout_output.get("decoder/control_probe_control_pos_mask", np.zeros((0,), dtype=np.float32))
        ),
        "local_bias_summary": _tensor_summary(
            rollout_output.get("decoder/control_probe_local_bias", np.zeros((0,), dtype=np.float32))
        ),
        "target_hidden_valid_mask": _json_ready(rollout_output.get("decoder/control_target_valid_mask")),
        "first_step_logits": _extract_topk_logits(rollout_output, target_slot=target_slot),
        "first_step_output_action": _json_ready(rollout_output.get("decoder/output_action")),
    }


def _plot_world_map(ax, raw_scenario: Mapping[str, Any], *, center_xy: np.ndarray, radius_m: float) -> None:
    for feature in dict(raw_scenario.get("map_features", {})).values():
        polyline = np.asarray(dict(feature).get("polyline", []), dtype=np.float32)
        if polyline.ndim != 2 or polyline.shape[1] < 2:
            continue
        xy = polyline[:, :2]
        finite_mask = np.isfinite(xy).all(axis=-1)
        xy = xy[finite_mask]
        if xy.shape[0] < 2:
            continue
        if np.max(np.linalg.norm(xy - center_xy.reshape(1, 2), axis=-1)) > radius_m * 2.0:
            continue
        ax.plot(xy[:, 0], xy[:, 1], color="#cbd5e1", linewidth=0.7, alpha=0.8, zorder=1)


def _save_overlay_plot(
    *,
    out_path: Path,
    raw_scenario: Mapping[str, Any],
    reference_world_xy: np.ndarray,
    baseline_world_xy: np.ndarray,
    controlled_world_xy: np.ndarray,
    scenario_id: str,
    agent_id: str,
    semantic_label: str,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=150)
    stacked = np.concatenate(
        [
            arr
            for arr in (reference_world_xy, baseline_world_xy, controlled_world_xy)
            if np.asarray(arr).ndim == 2 and np.asarray(arr).shape[0] > 0
        ],
        axis=0,
    )
    center_xy = np.asarray(stacked[0], dtype=np.float32) if stacked.size > 0 else np.zeros((2,), dtype=np.float32)
    radius_m = max(
        20.0,
        float(np.max(np.linalg.norm(stacked - center_xy.reshape(1, 2), axis=-1))) + 8.0 if stacked.size > 0 else 20.0,
    )
    _plot_world_map(ax, raw_scenario, center_xy=center_xy, radius_m=radius_m)
    if reference_world_xy.shape[0] >= 2:
        ax.plot(
            reference_world_xy[:, 0],
            reference_world_xy[:, 1],
            color="#6b7280",
            linewidth=2.0,
            alpha=0.9,
            label="reference",
            zorder=3,
        )
    if baseline_world_xy.shape[0] >= 2:
        ax.plot(
            baseline_world_xy[:, 0],
            baseline_world_xy[:, 1],
            color="#2563eb",
            linewidth=2.4,
            alpha=0.95,
            label="baseline rollout",
            zorder=4,
        )
    if controlled_world_xy.shape[0] >= 2:
        ax.plot(
            controlled_world_xy[:, 0],
            controlled_world_xy[:, 1],
            color="#dc2626",
            linewidth=2.6,
            alpha=0.95,
            label="controlled rollout",
            zorder=5,
        )
    if baseline_world_xy.shape[0] > 0:
        ax.scatter(
            [baseline_world_xy[0, 0]],
            [baseline_world_xy[0, 1]],
            c="#111827",
            s=36,
            label="start",
            zorder=6,
        )
    ax.set_title(f"{scenario_id} | agent {agent_id} | label={semantic_label}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_victim_centric_overlay_plot(
    *,
    out_path: Path,
    raw_scenario: Mapping[str, Any],
    all_reference_world: Mapping[str, np.ndarray],
    victim_reference_world_xy: np.ndarray,
    adversary_reference_world_xy: np.ndarray,
    adversary_baseline_world_xy: np.ndarray,
    adversary_controlled_world_xy: np.ndarray,
    victim_agent_id: str,
    adversary_agent_id: str,
    semantic_label: str,
    scenario_id: str,
) -> None:
    if plt is None:
        return

    arrays = [
        np.asarray(victim_reference_world_xy, dtype=np.float32),
        np.asarray(adversary_reference_world_xy, dtype=np.float32),
        np.asarray(adversary_baseline_world_xy, dtype=np.float32),
        np.asarray(adversary_controlled_world_xy, dtype=np.float32),
    ]
    arrays.extend(
        np.asarray(xy, dtype=np.float32)
        for track_id, xy in all_reference_world.items()
        if track_id not in {victim_agent_id, adversary_agent_id}
    )
    valid_arrays = [arr for arr in arrays if arr.ndim == 2 and arr.shape[0] > 0]

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=160)
    if valid_arrays:
        stacked = np.concatenate(valid_arrays, axis=0)
        center_xy = np.asarray(stacked.mean(axis=0), dtype=np.float32)
        radius_m = max(25.0, float(np.max(np.linalg.norm(stacked - center_xy.reshape(1, 2), axis=-1))) + 10.0)
    else:
        center_xy = np.zeros((2,), dtype=np.float32)
        radius_m = 25.0
    _plot_world_map(ax, raw_scenario, center_xy=center_xy, radius_m=radius_m)

    for track_id, xy in sorted(all_reference_world.items()):
        arr = np.asarray(xy, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0 or track_id in {victim_agent_id, adversary_agent_id}:
            continue
        if arr.shape[0] >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="#cbd5e1", linewidth=1.0, alpha=0.7, zorder=2)
        ax.scatter(
            [arr[0, 0]],
            [arr[0, 1]],
            c="#94a3b8",
            s=14,
            marker="o",
            alpha=0.9,
            zorder=3,
        )

    if victim_reference_world_xy.shape[0] >= 2:
        ax.plot(
            victim_reference_world_xy[:, 0],
            victim_reference_world_xy[:, 1],
            color="#16a34a",
            linewidth=2.6,
            alpha=0.95,
            label=f"victim ref ({victim_agent_id})",
            zorder=4,
        )
    if victim_reference_world_xy.shape[0] > 0:
        ax.scatter(
            [victim_reference_world_xy[0, 0]],
            [victim_reference_world_xy[0, 1]],
            c="#166534",
            s=52,
            marker="o",
            edgecolors="white",
            linewidths=0.6,
            zorder=7,
        )
    if adversary_reference_world_xy.shape[0] >= 2:
        ax.plot(
            adversary_reference_world_xy[:, 0],
            adversary_reference_world_xy[:, 1],
            color="#64748b",
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
            label=f"adversary ref ({adversary_agent_id})",
            zorder=4,
        )
    if adversary_reference_world_xy.shape[0] > 0:
        ax.scatter(
            [adversary_reference_world_xy[0, 0]],
            [adversary_reference_world_xy[0, 1]],
            c="#475569",
            s=42,
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            zorder=7,
        )
    if adversary_baseline_world_xy.shape[0] >= 2:
        ax.plot(
            adversary_baseline_world_xy[:, 0],
            adversary_baseline_world_xy[:, 1],
            color="#2563eb",
            linewidth=2.2,
            alpha=0.95,
            label="adversary baseline",
            zorder=5,
        )
    if adversary_controlled_world_xy.shape[0] >= 2:
        ax.plot(
            adversary_controlled_world_xy[:, 0],
            adversary_controlled_world_xy[:, 1],
            color="#dc2626",
            linewidth=2.8,
            alpha=0.98,
            label=f"adversary controlled ({semantic_label})",
            zorder=6,
        )

    if adversary_controlled_world_xy.shape[0] > 0:
        ax.scatter(
            [adversary_controlled_world_xy[0, 0]],
            [adversary_controlled_world_xy[0, 1]],
            c="#7f1d1d",
            s=52,
            marker="s",
            edgecolors="white",
            linewidths=0.6,
            zorder=7,
        )

    ax.set_title(f"{scenario_id} | adv={adversary_agent_id} | victim={victim_agent_id} | label={semantic_label}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _run_label_diagnostic(
    *,
    module: EvaluationLightningModule,
    tokenizer: Any,
    base_sample: Mapping[str, Any],
    semantic_label: str,
    semantic_confidence: float,
    time_window_mask: np.ndarray,
    decision_agent_mask: np.ndarray,
    target_slot: int,
    map_center_world: np.ndarray,
    map_heading_world: float,
) -> Dict[str, Any]:
    controlled_sample = _build_control_sample(
        base_sample=base_sample,
        semantic_label=semantic_label,
        semantic_confidence=float(semantic_confidence),
        time_window_mask=time_window_mask,
        decision_agent_mask=decision_agent_mask,
    )
    runtime_probe_sample = _build_runtime_probe_sample(sample=controlled_sample)
    runtime_probe = _run_one_step_runtime_probe(module, raw_sample=runtime_probe_sample)
    controlled = _run_rollout(module, tokenizer, raw_sample=controlled_sample)
    baseline = _run_rollout(module, tokenizer, raw_sample=base_sample)
    baseline_world_xy = _extract_target_rollout_world_from_output_np(
        baseline["output_np"],
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    controlled_world_xy = _extract_target_rollout_world_from_output_np(
        controlled["output_np"],
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    baseline_tokens = _extract_target_action_tokens_from_output_np(baseline["output_np"], target_slot=target_slot)
    controlled_tokens = _extract_target_action_tokens_from_output_np(controlled["output_np"], target_slot=target_slot)
    compare_len = int(min(baseline_world_xy.shape[0], controlled_world_xy.shape[0]))
    rollout_delta = (
        np.linalg.norm(controlled_world_xy[:compare_len] - baseline_world_xy[:compare_len], axis=-1)
        if compare_len > 0
        else np.zeros((0,), dtype=np.float32)
    )
    return {
        "semantic_label": str(semantic_label),
        "controlled_sample_cf": _summarize_control_fields(controlled_sample),
        "runtime_probe": _extract_runtime_probe_summary(runtime_probe, target_slot=target_slot),
        "effect": {
            "compare_len": int(compare_len),
            "num_changed_action_steps": int(
                (baseline_tokens[: min(len(baseline_tokens), len(controlled_tokens))] != controlled_tokens[: min(len(baseline_tokens), len(controlled_tokens))]).sum()
            ),
            "final_position_delta_m": float(
                np.linalg.norm(
                    controlled_world_xy[min(compare_len - 1, controlled_world_xy.shape[0] - 1)]
                    - baseline_world_xy[min(compare_len - 1, baseline_world_xy.shape[0] - 1)]
                )
            )
            if compare_len > 0
            else 0.0,
            "mean_position_delta_m": float(rollout_delta.mean()) if rollout_delta.size > 0 else 0.0,
            "max_position_delta_m": float(rollout_delta.max()) if rollout_delta.size > 0 else 0.0,
            "baseline_action_tokens": baseline_tokens.tolist(),
            "controlled_action_tokens": controlled_tokens.tolist(),
        },
    }


def _resolve_export_source_pkl(args: argparse.Namespace, *, scenario_id: str, scenario_pkl: str) -> Path:
    mode = str(args.export_source_mode).strip()
    if mode == "raw_scenario_pkl":
        return Path(str(scenario_pkl)).expanduser()
    if mode == "scenarionet_waymo":
        cache_dir = (
            Path(args.export_source_cache_dir).expanduser()
            if str(args.export_source_cache_dir).strip()
            else Path(args.outdir).expanduser() / "_scenarionet_export_sources"
        )
        resolved = materialize_scenarionet_waymo_sources(
            scenario_ids=[str(scenario_id)],
            cache_root=cache_dir,
            waymo_raw_path=str(args.waymo_raw_path),
            version=str(args.waymo_source_version),
        )
        return resolved[str(scenario_id)]
    return Path(str(scenario_pkl)).expanduser()


def main() -> int:
    args = parse_args()
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    raw_scenario = load_raw_scenario(args.scenario_pkl)
    config = _load_config(args)
    model, load_report = _load_model(config=config, ckpt_path=args.ckpt, load_mode=args.load_mode)
    device = _resolve_device(args.device)
    model = model.to(device)
    module, tokenizer = _build_eval_module(
        config=config,
        ckpt_path=args.ckpt,
        device=device,
        save_path=outdir / "unused_eval_metrics",
        model=model,
    )

    base_sample = preprocess_raw_scenario_for_forward_supervision(raw_scenario, config=config, in_evaluation=True)
    base_sample["metadata/scenario_id"] = str(raw_scenario.get("id") or base_sample.get("metadata/scenario_id", ""))
    forward_summary = summarize_forward_supervision_for_sample(base_sample, raw_scenario=raw_scenario)
    agent_id = _normalize_track_id(args.agent_id)
    target_agent_summary = next((row for row in forward_summary.agents if row.raw_track_id == agent_id), None)
    if target_agent_summary is None:
        available = sorted(set(forward_summary.modeled_agent_ids))
        raise ValueError(
            f"Agent '{agent_id}' is not a modeled decoder agent in scenario '{forward_summary.scenario_id}'. "
            f"Available modeled agents: {available[:24]}"
        )

    target_slot = int(target_agent_summary.model_agent_slot)
    try:
        sdc_slot = int(list(forward_summary.modeled_agent_ids).index(str(forward_summary.sdc_id)))
    except ValueError:
        sdc_slot = 0
    horizon = int(np.asarray(base_sample["decoder/target_action_valid_mask"]).shape[0])
    decision_agent_mask = np.zeros((len(forward_summary.modeled_agent_ids),), dtype=np.float32)
    decision_agent_mask[target_slot] = 1.0
    time_window_mask = _build_time_window_mask(
        horizon=horizon,
        start_step=int(args.start_step),
        end_step=int(args.end_step),
    )
    semantic_label = normalize_semantic_label(args.semantic_label)

    controlled_sample = _build_control_sample(
        base_sample=base_sample,
        semantic_label=semantic_label,
        semantic_confidence=float(args.semantic_confidence),
        time_window_mask=time_window_mask,
        decision_agent_mask=decision_agent_mask,
    )
    if str(args.rollout_sampling_method).strip():
        module.config.SAMPLING.SAMPLING_METHOD = str(args.rollout_sampling_method)
    rollout_temperature = _optional_positive_float(float(args.rollout_temperature))
    rollout_topp = _optional_positive_float(float(args.rollout_topp))
    if rollout_temperature is not None:
        module.config.SAMPLING.TEMPERATURE = float(rollout_temperature)
    if rollout_topp is not None:
        module.config.SAMPLING.TOPP = float(rollout_topp)
    baseline = _run_rollout(module, tokenizer, raw_sample=base_sample)
    controlled = _run_rollout(module, tokenizer, raw_sample=controlled_sample)

    map_center_world, map_heading_world = extract_model_frame(raw_scenario)
    reference_world_xy = _extract_target_reference_world_from_sample(
        base_sample,
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    baseline_world_xy = _extract_target_rollout_world_from_output_np(
        baseline["output_np"],
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    controlled_world_xy = _extract_target_rollout_world_from_output_np(
        controlled["output_np"],
        target_slot=target_slot,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
    )
    all_reference_world = _extract_all_reference_world_from_sample(
        base_sample,
        map_center_world=map_center_world,
        map_heading_world=map_heading_world,
        modeled_agent_ids=forward_summary.modeled_agent_ids,
    )
    victim_selection = _resolve_victim_selection(
        victim_agent_id_arg=str(args.victim_agent_id),
        modeled_agent_ids=forward_summary.modeled_agent_ids,
        all_reference_world=all_reference_world,
        adversary_track_id=agent_id,
        adversary_controlled_world_xy=controlled_world_xy,
        sdc_track_id=str(forward_summary.sdc_id),
    )
    victim_agent_id = str(victim_selection["agent_id"])
    victim_reference_world_xy = np.asarray(
        all_reference_world.get(victim_agent_id, np.zeros((0, 2), dtype=np.float32)),
        dtype=np.float32,
    )
    adversary_reference_world_xy = np.asarray(
        all_reference_world.get(agent_id, reference_world_xy),
        dtype=np.float32,
    )
    baseline_tokens = _extract_target_action_tokens_from_output_np(baseline["output_np"], target_slot=target_slot)
    controlled_tokens = _extract_target_action_tokens_from_output_np(controlled["output_np"], target_slot=target_slot)

    compare_len = int(min(baseline_world_xy.shape[0], controlled_world_xy.shape[0]))
    rollout_delta = (
        np.linalg.norm(controlled_world_xy[:compare_len] - baseline_world_xy[:compare_len], axis=-1)
        if compare_len > 0
        else np.zeros((0,), dtype=np.float32)
    )

    summary = {
        "scenario_id": str(forward_summary.scenario_id),
        "sdc_id": str(forward_summary.sdc_id),
        "scenario_pkl": str(Path(args.scenario_pkl).expanduser()),
        "ckpt": str(Path(args.ckpt).expanduser()),
        "config": str(Path(args.config).expanduser()),
        "device": str(device),
        "load_mode": str(args.load_mode),
        "checkpoint_load_report": load_report,
        "agent_id": str(agent_id),
        "target_agent_slot": int(target_slot),
        "victim_agent_id": str(victim_agent_id),
        "victim_agent_slot": int(victim_selection["slot"]),
        "victim_auto_selected": bool(victim_selection["auto_selected"]),
        "victim_selection_mode": str(victim_selection["selection_mode"]),
        "sdc_id_for_victim_selection": str(forward_summary.sdc_id),
        "sdc_is_closest_victim": bool(victim_selection["sdc_is_closest"]),
        "closest_victim_candidate": {
            "agent_id": str(victim_selection["closest_candidate"]["agent_id"]),
            "slot": int(victim_selection["closest_candidate"]["slot"]),
            "min_distance_m": float(victim_selection["closest_candidate"]["min_distance_m"]),
            "min_distance_step": int(victim_selection["closest_candidate"]["min_distance_step"]),
            "mean_distance_m": float(victim_selection["closest_candidate"]["mean_distance_m"]),
        },
        "victim_min_distance_m": float(victim_selection["min_distance_m"]),
        "victim_min_distance_step": int(victim_selection["min_distance_step"]),
        "victim_candidates_ranked": [
            {
                "agent_id": str(row["agent_id"]),
                "slot": int(row["slot"]),
                "min_distance_m": float(row["min_distance_m"]),
                "min_distance_step": int(row["min_distance_step"]),
                "mean_distance_m": float(row["mean_distance_m"]),
            }
            for row in victim_selection["ranked_candidates"][:8]
        ],
        "agent_receives_forward_loss": bool(target_agent_summary.receives_motion_loss),
        "agent_num_loss_steps": int(target_agent_summary.num_loss_steps),
        "modeled_agent_ids": list(forward_summary.modeled_agent_ids),
        "trainable_track_ids": list(forward_summary.trainable_track_ids),
        "semantic_label": str(semantic_label),
        "semantic_label_id": int(semantic_label_to_id(semantic_label)),
        "semantic_confidence": float(args.semantic_confidence),
        "control_window": {
            "start_step": int(max(0, args.start_step)),
            "end_step": int(np.flatnonzero(time_window_mask > 0.0)[-1]),
            "num_active_steps": int((time_window_mask > 0.0).sum()),
        },
        "rollout": {
            "sampling_method": str(args.rollout_sampling_method),
            "temperature": float(module.config.SAMPLING.TEMPERATURE),
            "topp": float(module.config.SAMPLING.TOPP),
        },
        "baseline": {
            "num_points": int(baseline_world_xy.shape[0]),
            "path_length_m": _path_length_m(baseline_world_xy),
            "net_displacement_m": float(np.linalg.norm(baseline_world_xy[-1] - baseline_world_xy[0])) if baseline_world_xy.shape[0] >= 2 else 0.0,
            "final_world_xy": baseline_world_xy[-1].tolist() if baseline_world_xy.shape[0] > 0 else None,
            "action_tokens": baseline_tokens.tolist(),
        },
        "controlled": {
            "num_points": int(controlled_world_xy.shape[0]),
            "path_length_m": _path_length_m(controlled_world_xy),
            "net_displacement_m": float(np.linalg.norm(controlled_world_xy[-1] - controlled_world_xy[0])) if controlled_world_xy.shape[0] >= 2 else 0.0,
            "final_world_xy": controlled_world_xy[-1].tolist() if controlled_world_xy.shape[0] > 0 else None,
            "action_tokens": controlled_tokens.tolist(),
        },
        "effect": {
            "compare_len": int(compare_len),
            "final_position_delta_m": float(np.linalg.norm(controlled_world_xy[min(compare_len - 1, controlled_world_xy.shape[0] - 1)] - baseline_world_xy[min(compare_len - 1, baseline_world_xy.shape[0] - 1)])) if compare_len > 0 else 0.0,
            "mean_position_delta_m": float(rollout_delta.mean()) if rollout_delta.size > 0 else 0.0,
            "max_position_delta_m": float(rollout_delta.max()) if rollout_delta.size > 0 else 0.0,
            "num_changed_action_steps": int((baseline_tokens[: min(len(baseline_tokens), len(controlled_tokens))] != controlled_tokens[: min(len(baseline_tokens), len(controlled_tokens))]).sum()),
        },
        "artifacts": {
            "summary_json": str(outdir / "summary.json"),
            "trajectories_npz": str(outdir / "trajectories.npz"),
            "overlay_png": (None if args.skip_plot else str(outdir / "overlay.png")),
            "victim_centric_overlay_png": (None if args.skip_plot else str(outdir / "victim_centric_overlay.png")),
            "victim_centric_export_dir": None,
            "victim_centric_replay_script": None,
            "victim_centric_manifest_json": None,
        },
    }

    if args.debug_trace:
        compare_labels = [str(semantic_label)]
        for extra_label in args.compare_label:
            normalized = normalize_semantic_label(extra_label)
            if normalized not in compare_labels:
                compare_labels.append(normalized)

        map_center_world_arr = np.asarray(map_center_world, dtype=np.float32)
        debug_trace = {
            "scenario_id": str(forward_summary.scenario_id),
            "target_agent_id": str(agent_id),
            "target_agent_slot": int(target_slot),
            "sdc_id": str(forward_summary.sdc_id),
            "sdc_slot": int(sdc_slot),
            "modeled_agent_ids": list(forward_summary.modeled_agent_ids),
            "trainable_track_ids": list(forward_summary.trainable_track_ids),
            "base_sample_cf": _summarize_control_fields(base_sample),
            "time_window_mask": _json_ready(time_window_mask),
            "target_decision_agent_mask": _json_ready(decision_agent_mask),
            "labels": {},
        }

        for label_name in compare_labels:
            debug_trace["labels"][label_name] = _run_label_diagnostic(
                module=module,
                tokenizer=tokenizer,
                base_sample=base_sample,
                semantic_label=label_name,
                semantic_confidence=float(args.semantic_confidence),
                time_window_mask=time_window_mask,
                decision_agent_mask=decision_agent_mask,
                target_slot=target_slot,
                map_center_world=map_center_world_arr,
                map_heading_world=map_heading_world,
            )

        if args.debug_compare_sdc:
            sdc_decision_agent_mask = np.zeros_like(decision_agent_mask)
            sdc_decision_agent_mask[sdc_slot] = 1.0
            debug_trace["sdc_target_comparison"] = {}
            for label_name in compare_labels:
                debug_trace["sdc_target_comparison"][label_name] = _run_label_diagnostic(
                    module=module,
                    tokenizer=tokenizer,
                    base_sample=base_sample,
                    semantic_label=label_name,
                    semantic_confidence=float(args.semantic_confidence),
                    time_window_mask=time_window_mask,
                    decision_agent_mask=sdc_decision_agent_mask,
                    target_slot=sdc_slot,
                    map_center_world=map_center_world_arr,
                    map_heading_world=map_heading_world,
                )

        debug_trace_path = outdir / "debug_trace.json"
        debug_trace_path.write_text(json.dumps(_json_ready(debug_trace), indent=2), encoding="utf-8")
        summary.setdefault("artifacts", {})["debug_trace_json"] = str(debug_trace_path)

    np.savez(
        outdir / "trajectories.npz",
        reference_world_xy=reference_world_xy.astype(np.float32),
        baseline_world_xy=baseline_world_xy.astype(np.float32),
        controlled_world_xy=controlled_world_xy.astype(np.float32),
        adversary_reference_world_xy=adversary_reference_world_xy.astype(np.float32),
        victim_reference_world_xy=victim_reference_world_xy.astype(np.float32),
        baseline_action_tokens=baseline_tokens.astype(np.int64),
        controlled_action_tokens=controlled_tokens.astype(np.int64),
        time_window_mask=time_window_mask.astype(np.float32),
        decision_agent_mask=decision_agent_mask.astype(np.float32),
    )

    if not args.skip_plot:
        _save_overlay_plot(
            out_path=outdir / "overlay.png",
            raw_scenario=raw_scenario,
            reference_world_xy=reference_world_xy,
            baseline_world_xy=baseline_world_xy,
            controlled_world_xy=controlled_world_xy,
            scenario_id=str(forward_summary.scenario_id),
            agent_id=agent_id,
            semantic_label=semantic_label,
        )
        _save_victim_centric_overlay_plot(
            out_path=outdir / "victim_centric_overlay.png",
            raw_scenario=raw_scenario,
            all_reference_world=all_reference_world,
            victim_reference_world_xy=victim_reference_world_xy,
            adversary_reference_world_xy=adversary_reference_world_xy,
            adversary_baseline_world_xy=baseline_world_xy,
            adversary_controlled_world_xy=controlled_world_xy,
            victim_agent_id=victim_agent_id,
            adversary_agent_id=agent_id,
            semantic_label=semantic_label,
            scenario_id=str(forward_summary.scenario_id),
        )

    if args.export_victim_centric:
        export_source_pkl = _resolve_export_source_pkl(
            args,
            scenario_id=str(forward_summary.scenario_id),
            scenario_pkl=str(args.scenario_pkl),
        )
        export_dir = (
            Path(args.export_scenario_dir).expanduser()
            if str(args.export_scenario_dir).strip()
            else outdir / "victim_centric_export"
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        intervention_name = (
            str(args.intervention_name).strip()
            if str(args.intervention_name).strip()
            else f"{semantic_label}_semantic_probe"
        )
        scenario_safe = _safe_name(forward_summary.scenario_id)
        victim_safe = _safe_name(victim_agent_id)
        adversary_safe = _safe_name(agent_id)
        intervention_safe = _safe_name(intervention_name)
        ground_truth_path = export_victim_centric_ground_truth_scenario(
            raw_scenario,
            export_dir / f"sd_counterfactual_1.0_{scenario_safe}_victim_{victim_safe}_ground_truth.pkl",
            victim_track_id=victim_agent_id,
            adversary_track_id=agent_id,
            original_file_path=export_source_pkl,
            intervention_name=f"{intervention_name}_ground_truth",
        )
        counterfactual_path = export_victim_centric_scenario(
            raw_scenario,
            export_dir / (
                f"sd_counterfactual_1.0_{scenario_safe}_victim_{victim_safe}_adv_{adversary_safe}_{intervention_safe}.pkl"
            ),
            victim_track_id=victim_agent_id,
            adversary_track_id=agent_id,
            adversary_trajectory_world_xy=controlled_world_xy,
            intervention_name=intervention_name,
            original_file_path=export_source_pkl,
        )
        replay_script_path = create_replay_script(
            [ground_truth_path, counterfactual_path],
            export_dir / "replay_victim_centric.py",
        )
        export_manifest = {
            "scenario_id": str(forward_summary.scenario_id),
            "victim_agent_id": str(victim_agent_id),
            "adversary_agent_id": str(agent_id),
            "semantic_label": str(semantic_label),
            "intervention_name": str(intervention_name),
            "ground_truth_scenario_pkl": str(ground_truth_path),
            "counterfactual_scenario_pkl": str(counterfactual_path),
            "replay_script": str(replay_script_path),
            "export_source_pkl": str(export_source_pkl),
        }
        manifest_path = export_dir / "victim_centric_export_manifest.json"
        manifest_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True), encoding="utf-8")
        summary["artifacts"]["victim_centric_export_dir"] = str(export_dir)
        summary["artifacts"]["victim_centric_replay_script"] = str(replay_script_path)
        summary["artifacts"]["victim_centric_manifest_json"] = str(manifest_path)

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
