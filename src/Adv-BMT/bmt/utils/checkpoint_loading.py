from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
from omegaconf import OmegaConf

from bmt.utils.config import REPO_ROOT

EXPECTED_NEW_PATH_CONTROL_PREFIXES = (
    "path_head.",
    "compliance_head.",
    "timing_head.",
    "anchor_head.",
    "sdc_semantic_head.",
    "model.motion_decoder.cf_path_proj.",
    "model.motion_decoder.cf_compliance_proj.",
    "model.motion_decoder.cf_timing_proj.",
    "model.motion_decoder.cf_anchor_proj.",
    "model.motion_decoder.cf_local_bias.",
    "model.motion_decoder.cf_local_residual_gate",
    "model.motion_decoder.cf_sdc_semantic_embed.",
    "model.motion_decoder.cf_sdc_waypoint_proj.",
    "model.motion_decoder.cf_sdc_waypoint_summary.",
    "model.motion_decoder.cf_sdc_local_bias.",
    "model.motion_decoder.cf_sdc_local_residual_gate",
)


def _resolve_checkpoint_path(ckpt_path: str | Path) -> Path:
    resolved = Path(ckpt_path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def _torch_load_checkpoint(path: Path, *, map_location: Any):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _ensure_model_runtime_defaults(config: Any) -> Any:
    prepared = copy.deepcopy(config)
    if not OmegaConf.is_config(prepared):
        prepared = OmegaConf.create(_to_plain_container(prepared))
    model_cfg = getattr(prepared, "MODEL", None)
    if model_cfg is None:
        return prepared
    if "DROPOUT_OF_ATTN" not in model_cfg:
        model_cfg["DROPOUT_OF_ATTN"] = float(model_cfg.get("DROPOUT", 0.0))
    return prepared


def _to_plain_container(value: Any):
    if isinstance(value, dict):
        return {key: _to_plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_container(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {key: _to_plain_container(item) for key, item in value.items()}
        except Exception:
            return value
    return value


def _extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint state_dict must be a mapping, got {type(state_dict)!r}")
    return state_dict


def _is_expected_new_path_control_key(key: str) -> bool:
    if any(key.startswith(prefix) for prefix in EXPECTED_NEW_PATH_CONTROL_PREFIXES):
        return True
    if ".cross_maneuver." in key:
        return True
    if ".maneuver_norm." in key:
        return True
    if key.endswith(".maneuver_residual_gate"):
        return True
    return False


def _is_policy_teacher_key(key: str) -> bool:
    return str(key).startswith("policy_teacher.")


def _parameter_owner_from_key(key: str) -> str:
    if "." not in key:
        return key
    return key.rsplit(".", 1)[0]


def _module_bucket_for_summary(key: str) -> str:
    text = str(key)
    if text.startswith("path_head."):
        return "path_head"
    if text.startswith("anchor_head."):
        return "anchor_head"
    if text.startswith("sdc_semantic_head."):
        return "sdc_semantic_head"
    if text.startswith("compliance_head."):
        return "compliance_head"
    if text.startswith("timing_head."):
        return "timing_head"
    if text.startswith("model.scene_encoder."):
        return "model.scene_encoder"
    if text.startswith("model.motion_decoder."):
        if ".cross_maneuver." in text or ".maneuver_norm." in text or text.endswith(".maneuver_residual_gate"):
            return "model.motion_decoder.decoder_local_control"
        if ".cf_" in text:
            return "model.motion_decoder.control_tokens"
        return "model.motion_decoder"
    return _parameter_owner_from_key(text)


def summarize_load_report_by_module(load_report: Mapping[str, Any]) -> Dict[str, Any]:
    if "loaded_module_prefix_counts" in load_report:
        return {
            "loaded_parameter_prefix_counts": dict(load_report.get("loaded_module_prefix_counts", {})),
            "missing_parameter_prefix_counts": dict(load_report.get("missing_module_prefix_counts", {})),
            "unexpected_parameter_prefix_counts": dict(load_report.get("unexpected_module_prefix_counts", {})),
            "shape_mismatch_parameter_prefix_counts": dict(load_report.get("shape_mismatch_module_prefix_counts", {})),
            "expected_new_path_control_prefix_counts": dict(load_report.get("expected_new_path_control_prefix_counts", {})),
        }

    def _bucket(keys: Iterable[str]) -> Dict[str, int]:
        counts = Counter(_module_bucket_for_summary(str(key)) for key in keys)
        return {key: int(counts[key]) for key in sorted(counts)}

    return {
        "loaded_parameter_prefix_counts": _bucket(load_report.get("loaded_keys", [])),
        "missing_parameter_prefix_counts": _bucket(load_report.get("missing_keys", [])),
        "unexpected_parameter_prefix_counts": _bucket(load_report.get("unexpected_keys", [])),
        "shape_mismatch_parameter_prefix_counts": _bucket(load_report.get("shape_mismatch_keys", [])),
        "expected_new_path_control_prefix_counts": _bucket(load_report.get("expected_new_path_control_keys_missing", [])),
    }


def load_model_from_checkpoint_forgiving(
    config: Any,
    ckpt_path: str | Path,
    *,
    load_mode: str = "forgiving_state_dict",
    strict_state_dict: bool = False,
    ignore_checkpoint_hparams: bool = True,
    map_location: Any = "cpu",
    checkpoint_surgery_func=None,
):
    from bmt.models.motionlm_lightning import MotionLMLightning

    if load_mode not in {"forgiving_state_dict", "strict_state_dict"}:
        raise ValueError(
            f"Unsupported load_mode={load_mode!r}. Expected 'forgiving_state_dict' or 'strict_state_dict'."
        )

    prepared_config = _ensure_model_runtime_defaults(config)
    model = MotionLMLightning(config=prepared_config)

    resolved_ckpt = _resolve_checkpoint_path(ckpt_path)
    checkpoint = _torch_load_checkpoint(resolved_ckpt, map_location=map_location)
    if checkpoint_surgery_func is not None and isinstance(checkpoint, MutableMapping):
        checkpoint = checkpoint_surgery_func(checkpoint, MotionLMLightning, prepared_config)

    checkpoint_state = _extract_state_dict(checkpoint)
    model_state = model.state_dict()

    loaded_state: Dict[str, torch.Tensor] = {}
    unexpected_keys = []
    shape_mismatch_keys = []
    for key, value in checkpoint_state.items():
        if key not in model_state:
            unexpected_keys.append(str(key))
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch_keys.append(str(key))
            continue
        loaded_state[str(key)] = value

    missing_keys = [str(key) for key in model_state.keys() if key not in loaded_state]
    expected_new_path_control_keys_missing = sorted(key for key in missing_keys if _is_expected_new_path_control_key(key))
    expected_policy_teacher_keys_missing = sorted(key for key in missing_keys if _is_policy_teacher_key(key))
    unexpected_missing_keys = sorted(
        key
        for key in missing_keys
        if key not in expected_new_path_control_keys_missing and key not in expected_policy_teacher_keys_missing
    )

    if strict_state_dict or load_mode == "strict_state_dict":
        if missing_keys or unexpected_keys or shape_mismatch_keys:
            report = {
                "ckpt_path": str(resolved_ckpt),
                "load_mode": load_mode,
                "ignore_checkpoint_hparams": bool(ignore_checkpoint_hparams),
                "num_ckpt_state_dict_keys": int(len(checkpoint_state)),
                "num_loaded_keys": int(len(loaded_state)),
                "num_missing_keys": int(len(missing_keys)),
                "num_unexpected_keys": int(len(unexpected_keys)),
                "num_shape_mismatch_keys": int(len(shape_mismatch_keys)),
                "first_50_missing_keys": sorted(missing_keys)[:50],
                "first_50_unexpected_keys": sorted(unexpected_keys)[:50],
                "first_50_shape_mismatch_keys": sorted(shape_mismatch_keys)[:50],
                "expected_new_path_control_keys_missing": expected_new_path_control_keys_missing,
                "expected_policy_teacher_keys_missing": expected_policy_teacher_keys_missing,
                "unexpected_missing_keys": unexpected_missing_keys,
                "strict_state_dict_used": True,
                "loaded_module_prefix_counts": _bucket_module_counts(loaded_state.keys()),
                "missing_module_prefix_counts": _bucket_module_counts(missing_keys),
                "unexpected_module_prefix_counts": _bucket_module_counts(unexpected_keys),
                "shape_mismatch_module_prefix_counts": _bucket_module_counts(shape_mismatch_keys),
                "expected_new_path_control_prefix_counts": _bucket_module_counts(expected_new_path_control_keys_missing),
                "expected_policy_teacher_prefix_counts": _bucket_module_counts(expected_policy_teacher_keys_missing),
            }
            raise RuntimeError(
                "Strict checkpoint load failed: "
                f"{report['num_missing_keys']} missing, "
                f"{report['num_unexpected_keys']} unexpected, "
                f"{report['num_shape_mismatch_keys']} shape-mismatch keys."
            )
        model.load_state_dict(loaded_state, strict=True)
    else:
        model.load_state_dict(loaded_state, strict=False)

    policy_teacher_sync_report = None
    if hasattr(model, "sync_policy_teacher_from_student"):
        policy_teacher_sync_report = model.sync_policy_teacher_from_student()

    load_report = {
        "ckpt_path": str(resolved_ckpt),
        "load_mode": load_mode,
        "ignore_checkpoint_hparams": bool(ignore_checkpoint_hparams),
        "num_ckpt_state_dict_keys": int(len(checkpoint_state)),
        "num_loaded_keys": int(len(loaded_state)),
        "num_missing_keys": int(len(missing_keys)),
        "num_unexpected_keys": int(len(unexpected_keys)),
        "num_shape_mismatch_keys": int(len(shape_mismatch_keys)),
        "first_50_missing_keys": sorted(missing_keys)[:50],
        "first_50_unexpected_keys": sorted(unexpected_keys)[:50],
        "first_50_shape_mismatch_keys": sorted(shape_mismatch_keys)[:50],
        "expected_new_path_control_keys_missing": expected_new_path_control_keys_missing,
        "expected_policy_teacher_keys_missing": expected_policy_teacher_keys_missing,
        "unexpected_missing_keys": unexpected_missing_keys,
        "strict_state_dict_used": bool(strict_state_dict or load_mode == "strict_state_dict"),
        "loaded_module_prefix_counts": _bucket_module_counts(loaded_state.keys()),
        "missing_module_prefix_counts": _bucket_module_counts(missing_keys),
        "unexpected_module_prefix_counts": _bucket_module_counts(unexpected_keys),
        "shape_mismatch_module_prefix_counts": _bucket_module_counts(shape_mismatch_keys),
        "expected_new_path_control_prefix_counts": _bucket_module_counts(expected_new_path_control_keys_missing),
        "expected_policy_teacher_prefix_counts": _bucket_module_counts(expected_policy_teacher_keys_missing),
        "policy_teacher_sync_report": policy_teacher_sync_report,
    }
    return model, load_report


def _bucket_module_counts(keys: Iterable[str]) -> Dict[str, int]:
    counts = Counter(_module_bucket_for_summary(str(key)) for key in keys)
    return {key: int(counts[key]) for key in sorted(counts)}
