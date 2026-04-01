from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    vendored_scenarionet = repo_root / "scenarionet"
    vendored_metadrive = repo_root / "metadrive"
    for path in (vendored_scenarionet, vendored_metadrive, repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
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
    from scenarionet import read_dataset_summary as _scenarionet_read_dataset_summary  # type: ignore
    from scenarionet import read_scenario as _scenarionet_read_scenario  # type: ignore
except Exception:
    scenarionet_stub = types.ModuleType("scenarionet")

    def _read_dataset_summary(dataset_path):
        dataset_root = Path(dataset_path).expanduser()
        files = sorted(path.name for path in dataset_root.glob("*.pkl"))
        summary = {name: {} for name in files}
        mapping = {name: str(dataset_root) for name in files}
        return summary, files, mapping

    def _read_scenario(dataset_path, mapping, scenario_file_name):
        dataset_root = Path(dataset_path).expanduser()
        base_dir = Path(mapping.get(scenario_file_name, dataset_root))
        scenario_path = base_dir / scenario_file_name
        with scenario_path.open("rb") as f:
            return pickle.load(f)

    scenarionet_stub.read_dataset_summary = _read_dataset_summary
    scenarionet_stub.read_scenario = _read_scenario
    sys.modules["scenarionet"] = scenarionet_stub

try:
    from metadrive.scenario.scenario_description import ScenarioDescription as _SD  # type: ignore
    from metadrive.scenario.scenario_description import MetaDriveType as _MetaDriveType  # type: ignore
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

from bmt.counterfactual import (
    decode_compliance_token_tensor,
    decode_decision_agent_mask,
    decode_path_token_tensor,
    decode_terminal_anchor_tensor,
    decode_time_window_mask,
    decode_timing_token_tensor,
)
from bmt.counterfactual.runtime_probe import (
    classify_probe_behavior,
    summarize_probe_stages,
)
from bmt.dataset.dataset import InfgenDataset
from bmt.utils.checkpoint_loading import (
    load_model_from_checkpoint_forgiving,
    summarize_load_report_by_module,
)
from bmt.utils import utils
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect counterfactual control-code tensors in a collated dataset batch.")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--control-code-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="training", choices=("training", "test"))
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-forward", action="store_true", help="Run a real model forward pass and compare controlled vs baseline outputs.")
    parser.add_argument(
        "--forward-control-mode",
        type=str,
        default="interactive",
        choices=("interactive", "strict_local"),
        help="Forward-only local control mode to use during the forward smoke.",
    )
    parser.add_argument(
        "--load-mode",
        type=str,
        default="forgiving_state_dict",
        choices=("forgiving_state_dict", "strict_state_dict", "legacy_merge"),
        help="Checkpoint load mode for forward/runtime smoke.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    dataset = InfgenDataset(config=config, mode=args.mode)
    samples = [dataset[idx] for idx in range(int(args.offset), int(args.offset) + int(args.batch_size))]
    batch = dataset.collate_batch(samples)

    debug_meta = batch["cf/debug_meta"]
    available_flags = [bool(item.get("available")) for item in debug_meta]
    selected_example_idx = available_flags.index(True) if any(available_flags) else 0
    decoder_track_names = None
    if "decoder/track_name" in batch:
        decoder_track_names = batch["decoder/track_name"][selected_example_idx]

    summary = {
        "mode": args.mode,
        "batch_size": len(samples),
        "selected_example_idx": selected_example_idx,
        "scenario_ids": _to_numpy(batch["metadata/scenario_id"]).astype(str).tolist(),
        "available_control_codes": int(sum(available_flags)),
        "tensor_shapes": {
            "cf/path_token": list(_to_numpy(batch["cf/path_token"]).shape),
            "cf/compliance_token": list(_to_numpy(batch["cf/compliance_token"]).shape),
            "cf/timing_token": list(_to_numpy(batch["cf/timing_token"]).shape),
            "cf/terminal_anchor": list(_to_numpy(batch["cf/terminal_anchor"]).shape),
            "cf/time_window_mask": list(_to_numpy(batch["cf/time_window_mask"]).shape),
            "cf/decision_agent_mask": list(_to_numpy(batch["cf/decision_agent_mask"]).shape),
        },
        "selected_debug_meta": debug_meta[selected_example_idx],
    }

    decoded = {
        "decoded_cf_path_token.json": decode_path_token_tensor(_to_numpy(batch["cf/path_token"])[selected_example_idx]),
        "decoded_cf_compliance_token.json": decode_compliance_token_tensor(_to_numpy(batch["cf/compliance_token"])[selected_example_idx]),
        "decoded_cf_timing_token.json": decode_timing_token_tensor(_to_numpy(batch["cf/timing_token"])[selected_example_idx]),
        "decoded_cf_terminal_anchor.json": decode_terminal_anchor_tensor(_to_numpy(batch["cf/terminal_anchor"])[selected_example_idx]),
        "decoded_cf_time_window_mask.json": decode_time_window_mask(_to_numpy(batch["cf/time_window_mask"])[selected_example_idx]),
        "decoded_cf_decision_agent_mask.json": decode_decision_agent_mask(
            _to_numpy(batch["cf/decision_agent_mask"])[selected_example_idx],
            decoder_track_names=decoder_track_names,
        ),
        "decoded_cf_debug_meta.json": debug_meta[selected_example_idx],
    }
    selected_control = {
        "scenario_id": summary["scenario_ids"][selected_example_idx],
        "debug_meta": debug_meta[selected_example_idx],
        "path_token": decoded["decoded_cf_path_token.json"],
        "compliance_token": decoded["decoded_cf_compliance_token.json"],
        "timing_token": decoded["decoded_cf_timing_token.json"],
        "terminal_anchor": decoded["decoded_cf_terminal_anchor.json"],
        "time_window": decoded["decoded_cf_time_window_mask.json"],
        "target_agent_slot": decoded["decoded_cf_decision_agent_mask.json"],
    }
    selected_tokens = np.concatenate(
        [
            _to_numpy(batch["cf/path_token"])[selected_example_idx].reshape(-1),
            _to_numpy(batch["cf/compliance_token"])[selected_example_idx].reshape(-1),
            _to_numpy(batch["cf/timing_token"])[selected_example_idx].reshape(-1),
            _to_numpy(batch["cf/terminal_anchor"])[selected_example_idx].reshape(-1),
        ],
        axis=0,
    )
    control_pos_mask = (
        _to_numpy(batch["cf/time_window_mask"])[selected_example_idx][:, None]
        * _to_numpy(batch["cf/decision_agent_mask"])[selected_example_idx][None, :]
    )
    predicted_branch_eval = {
        "selected_branch_label": decoded["decoded_cf_path_token.json"]["branch_label"],
        "selected_compliance_label": decoded["decoded_cf_compliance_token.json"]["compliance_label"],
        "selected_timing_label": decoded["decoded_cf_timing_token.json"]["timing_label"],
        "model_forward_ran": False,
    }
    forward_runtime_summary = None
    if args.run_forward:
        forward_runtime_summary = _run_forward_smoke(
            config=config,
            batch=batch,
            selected_example_idx=selected_example_idx,
            ckpt_path=args.ckpt,
            load_mode=args.load_mode,
        )
        predicted_branch_eval.update(
            {
                "model_forward_ran": True,
                "control_forward_mode": forward_runtime_summary["control_forward_mode"],
                "conditioning_eligible": bool(forward_runtime_summary["conditioning_eligible"]),
                "control_memory_shape": forward_runtime_summary["control_memory_shape"],
                "control_pos_mask_nonzero_fraction": forward_runtime_summary["control_pos_mask_nonzero_fraction"],
                "selected_agent_slot": forward_runtime_summary["selected_agent_slot"],
                "residual_reached_non_target_positions": forward_runtime_summary["residual_reached_non_target_positions"],
                "direct_leakage_bug": forward_runtime_summary["direct_leakage_bug"],
                "propagated_to_non_target_positions": forward_runtime_summary["propagated_to_non_target_positions"],
            }
        )
        summary["forward_runtime"] = forward_runtime_summary

    (outdir / "batch_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for filename, payload in decoded.items():
        (outdir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    np.savez_compressed(
        outdir / "batch_debug.npz",
        scenario_ids=_to_numpy(batch["metadata/scenario_id"]).astype(str),
        path_token=_to_numpy(batch["cf/path_token"]),
        compliance_token=_to_numpy(batch["cf/compliance_token"]),
        timing_token=_to_numpy(batch["cf/timing_token"]),
        terminal_anchor=_to_numpy(batch["cf/terminal_anchor"]),
        time_window_mask=_to_numpy(batch["cf/time_window_mask"]),
        decision_agent_mask=_to_numpy(batch["cf/decision_agent_mask"]),
        conditioning_eligible=_to_numpy(batch.get("cf/conditioning_eligible", np.zeros((len(samples),), dtype=np.float32))),
    )
    (outdir / "selected_control.json").write_text(json.dumps(selected_control, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "target_agent_slot.json").write_text(json.dumps(decoded["decoded_cf_decision_agent_mask.json"], indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "predicted_branch_eval.json").write_text(json.dumps(predicted_branch_eval, indent=2, sort_keys=True), encoding="utf-8")
    if forward_runtime_summary is not None:
        (outdir / "forward_control_runtime.json").write_text(
            json.dumps(forward_runtime_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    np.save(outdir / "selected_control_tokens.npy", selected_tokens.astype(np.float32))
    np.save(outdir / "control_pos_mask.npy", control_pos_mask.astype(np.float32))
    control_code_path = str(debug_meta[selected_example_idx].get("control_code_path", ""))
    if control_code_path:
        source_overlay = Path(control_code_path).expanduser().parent / "branch_candidates.png"
        if source_overlay.is_file():
            shutil.copy2(source_overlay, outdir / "controlled_vs_baseline_bev.png")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_config(args: argparse.Namespace):
    config = copy.deepcopy(global_config)
    default_forward_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_forward_cfg.is_file():
        config = cfg_from_yaml_file(default_forward_cfg, config)
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (REPO_ROOT / cfg_path).resolve()
        config = cfg_from_yaml_file(cfg_path, config)
    config.DATA.COUNTERFACTUAL_CONTROL_CODE_DIR = str(Path(args.control_code_dir).expanduser())
    if args.data_dir:
        if args.mode == "training":
            config.DATA.TRAINING_DATA_DIR = args.data_dir
        else:
            config.DATA.TEST_DATA_DIR = args.data_dir
    if args.run_forward:
        config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
        config.MODEL.LOCAL_CONTROL_FORWARD_MODE = args.forward_control_mode
    return config


def _run_forward_smoke(
    *,
    config: Any,
    batch: Dict[str, Any],
    selected_example_idx: int,
    ckpt_path: str = "",
    load_mode: str = "forgiving_state_dict",
) -> Dict[str, Any]:

    if str(config.MODEL.NAME) != "gpt":
        raise ValueError("Forward smoke requires a GPT decoder config so local control can be injected.")

    model, load_report, loaded_module_summary = _load_motion_model_for_smoke(
        config=config,
        ckpt_path=ckpt_path,
        load_mode=load_mode,
    )
    model.eval()

    controlled_batch = copy.deepcopy(batch)
    baseline_batch = copy.deepcopy(batch)
    device = torch.device("cpu")
    controlled_batch = _to_torch_device(controlled_batch, device=device)
    baseline_batch = _to_torch_device(baseline_batch, device=device)

    controlled_batch["in_evaluation"] = _force_eval_flag(controlled_batch.get("in_evaluation"), device=device)
    baseline_batch["in_evaluation"] = _force_eval_flag(baseline_batch.get("in_evaluation"), device=device)
    controlled_batch["cf/runtime_probe_enabled"] = torch.ones((1,), device=device, dtype=torch.bool)
    baseline_batch["cf/runtime_probe_enabled"] = torch.ones((1,), device=device, dtype=torch.bool)

    if bool(getattr(config, "REMOVE_AGENT_FROM_SCENE_ENCODER", False)):
        randomized_modeled_agent_id = model.motion_decoder.randomize_modeled_agent_id(controlled_batch, clip_agent_id=False)
        controlled_batch["decoder/randomized_modeled_agent_id"] = randomized_modeled_agent_id
        baseline_batch["decoder/randomized_modeled_agent_id"] = randomized_modeled_agent_id.clone()

    if "cf/conditioning_eligible" in baseline_batch:
        baseline_batch["cf/conditioning_eligible"] = torch.zeros_like(baseline_batch["cf/conditioning_eligible"])
    if "cf/control_available" in baseline_batch:
        baseline_batch["cf/control_available"] = torch.zeros_like(baseline_batch["cf/control_available"])

    with torch.no_grad():
        controlled_out = model(copy.deepcopy(controlled_batch))
        baseline_out = model(copy.deepcopy(baseline_batch))

    control_memory = _to_numpy(controlled_out.get("cf/control_memory", np.zeros((0,), dtype=np.float32)))
    control_pos_mask = _to_numpy(controlled_out.get("cf/control_pos_mask", np.zeros((0,), dtype=np.float32)))
    supervision_pos_mask = _to_numpy(controlled_out.get("cf/supervision_pos_mask", np.zeros((0,), dtype=np.float32)))
    decoded_tokens_controlled = _to_numpy(controlled_out["decoder/decoded_tokens"])
    decoded_tokens_baseline = _to_numpy(baseline_out["decoder/decoded_tokens"])
    valid_mask = _to_numpy(controlled_out["decoder/input_action_valid_mask"]).astype(bool)
    decision_agent_mask = _to_numpy(controlled_out["cf/decision_agent_mask"]).astype(bool)
    conditioning_eligible = bool(
        _to_numpy(controlled_out.get("cf/conditioning_eligible", controlled_out.get("cf/control_available", np.zeros((1,), dtype=np.float32))))[selected_example_idx]
    )

    selected_control_pos_mask = control_pos_mask[selected_example_idx] > 0
    selected_supervision_pos_mask = supervision_pos_mask[selected_example_idx] > 0
    selected_valid_mask = valid_mask[selected_example_idx]
    token_delta = decoded_tokens_controlled[selected_example_idx] - decoded_tokens_baseline[selected_example_idx]
    token_delta_norm = np.linalg.norm(token_delta, axis=-1)
    target_mask = selected_control_pos_mask & selected_valid_mask
    non_target_mask = (~selected_control_pos_mask) & selected_valid_mask
    supervision_only_mask = selected_supervision_pos_mask & (~selected_control_pos_mask) & selected_valid_mask
    selected_agent_slots = np.flatnonzero(decision_agent_mask[selected_example_idx]).astype(int).tolist()
    tolerance = 1e-6

    probe_keys = {
        "after_control_add": "decoder/control_probe_after_control_add",
        "after_next_shared_block": "decoder/control_probe_after_next_shared_block",
        "after_full_decoder": "decoder/control_probe_after_full_decoder",
    }
    controlled_probes = {
        stage_name: _to_numpy(controlled_out.get(key, controlled_out["decoder/decoded_tokens"]))
        for stage_name, key in probe_keys.items()
    }
    baseline_probes = {
        stage_name: _to_numpy(baseline_out.get(key, baseline_out["decoder/decoded_tokens"]))
        for stage_name, key in probe_keys.items()
    }
    stage_summaries = summarize_probe_stages(
        controlled_probes=controlled_probes,
        baseline_probes=baseline_probes,
        selected_example_idx=selected_example_idx,
        inside_mask=target_mask,
        outside_mask=non_target_mask,
        tolerance=tolerance,
    )
    probe_behavior = classify_probe_behavior(stage_summaries)

    return {
        "checkpoint_load_report": load_report,
        "loaded_module_summary": loaded_module_summary,
        "control_forward_mode": str(
            controlled_out.get("decoder/control_forward_mode", getattr(config.MODEL, "LOCAL_CONTROL_FORWARD_MODE", "interactive"))
        ),
        "conditioning_eligible": conditioning_eligible,
        "control_memory_shape": list(control_memory.shape),
        "control_pos_mask_nonzero_fraction": float(selected_control_pos_mask.mean()) if selected_control_pos_mask.size > 0 else 0.0,
        "control_pos_mask_nonzero_fraction_over_valid": (
            float(selected_control_pos_mask[selected_valid_mask].mean()) if bool(selected_valid_mask.any()) else 0.0
        ),
        "selected_agent_slot": (selected_agent_slots[0] if selected_agent_slots else None),
        "selected_agent_slots": selected_agent_slots,
        "target_position_count": int(target_mask.sum()),
        "non_target_position_count": int(non_target_mask.sum()),
        "supervision_only_position_count": int(supervision_only_mask.sum()),
        "direct_leakage_bug": probe_behavior["direct_leakage_bug"],
        "propagated_to_non_target_positions": probe_behavior["propagated_to_non_target_positions"],
        "stage_deltas": stage_summaries,
        "residual_reached_target_positions": bool(np.any(token_delta_norm[target_mask] > tolerance)) if bool(target_mask.any()) else False,
        "residual_reached_non_target_positions": bool(np.any(token_delta_norm[non_target_mask] > tolerance)) if bool(non_target_mask.any()) else False,
        "residual_reached_supervision_only_positions": bool(np.any(token_delta_norm[supervision_only_mask] > tolerance)) if bool(supervision_only_mask.any()) else False,
        "target_delta_max_abs": float(np.max(np.abs(token_delta[target_mask]))) if bool(target_mask.any()) else 0.0,
        "non_target_delta_max_abs": float(np.max(np.abs(token_delta[non_target_mask]))) if bool(non_target_mask.any()) else 0.0,
        "supervision_only_delta_max_abs": float(np.max(np.abs(token_delta[supervision_only_mask]))) if bool(supervision_only_mask.any()) else 0.0,
    }


def _load_motion_model_for_smoke(
    *,
    config: Any,
    ckpt_path: str = "",
    load_mode: str = "forgiving_state_dict",
):
    if not ckpt_path:
        from bmt.models.motionlm import MotionLM

        return MotionLM(config), {"load_mode": "none", "ckpt_path": "", "num_loaded_keys": 0}, {}

    from bmt.models.motionlm_lightning import MotionLMLightning

    default_config = cfg_from_yaml_file(REPO_ROOT / "cfgs/motion_default.yaml", global_config)
    resolved_ckpt = Path(ckpt_path).expanduser()
    if not resolved_ckpt.is_absolute():
        resolved_ckpt = (REPO_ROOT / resolved_ckpt).resolve()
    if load_mode == "legacy_merge":
        lightning_model = utils.load_from_checkpoint(
            checkpoint_path=str(resolved_ckpt),
            cls=MotionLMLightning,
            config=config,
            default_config=default_config,
            strict=False,
            checkpoint_surgery_func=utils.checkpoint_surgery_func,
            map_location="cpu",
        )
        load_report = {
            "ckpt_path": str(resolved_ckpt),
            "load_mode": "legacy_merge",
            "num_ckpt_state_dict_keys": None,
            "num_loaded_keys": None,
            "num_missing_keys": None,
            "num_unexpected_keys": None,
            "num_shape_mismatch_keys": None,
            "strict_state_dict_used": False,
        }
        loaded_module_summary = {}
    else:
        lightning_model, load_report = load_model_from_checkpoint_forgiving(
            config=config,
            ckpt_path=str(resolved_ckpt),
            load_mode=load_mode,
            strict_state_dict=(load_mode == "strict_state_dict"),
            map_location="cpu",
            checkpoint_surgery_func=utils.checkpoint_surgery_func,
        )
        loaded_module_summary = summarize_load_report_by_module(load_report)
    return lightning_model.model, load_report, loaded_module_summary


def _force_eval_flag(value: Any, *, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.ones((1,), device=device, dtype=torch.bool)
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    tensor = tensor.to(device=device)
    return torch.ones_like(tensor, dtype=torch.bool)


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


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


if __name__ == "__main__":
    raise SystemExit(main())
