from __future__ import annotations

import copy
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from bmt.counterfactual.compile_control_code import BRANCH_LABEL_ORDER, build_counterfactual_dataset_fields, default_counterfactual_dataset_fields
from bmt.dataset.preprocessor import preprocess_scenario_description
from bmt.tokenization import get_tokenizer
from bmt.utils import utils
from bmt.utils.checkpoint_loading import load_model_from_checkpoint_forgiving
from bmt.utils.config import REPO_ROOT, cfg_from_yaml_file, global_config

FRAME_WORLD = "world"
FRAME_AGENT_RELATIVE_AT_DECISION = "agent_relative_at_decision"
FRAME_MODEL_OUTPUT = "model_output_frame"

OBJECT_BATCH_KEYS = {
    "raw_scenario_description",
    "original_SD",
    "encoder/track_name",
    "decoder/track_name",
    "eval/track_name",
    "cf/debug_meta",
}


@dataclass(frozen=True)
class TaggedPose:
    x: float
    y: float
    heading: float
    frame: str


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")


def wrap_angle(value: float) -> float:
    return float(math.atan2(math.sin(value), math.cos(value)))


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return float(np.mean(np.asarray(filtered, dtype=np.float64)))


def percentile_dict(values: Sequence[float], percentiles: Sequence[int] = (50, 90, 95, 99)) -> Dict[str, Optional[float]]:
    if not values:
        return {f"p{int(p)}": None for p in percentiles}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{int(p)}": float(np.percentile(array, p)) for p in percentiles}


def parse_example_id(example_id: str) -> Dict[str, Optional[str]]:
    tokens = str(example_id).split("__")
    result: Dict[str, Optional[str]] = {
        "example_id": str(example_id),
        "scenario_id": None,
        "agent_id": None,
        "light_id": None,
        "decision_time_idx": None,
    }
    if tokens:
        result["scenario_id"] = tokens[0]
    for token in tokens[1:]:
        if token.startswith("agent_"):
            result["agent_id"] = token.removeprefix("agent_")
        elif token.startswith("light_"):
            result["light_id"] = token.removeprefix("light_")
        elif token.startswith("t_"):
            result["decision_time_idx"] = str(int(token.removeprefix("t_")))
    return result


def discover_local_scenario_pkls(bundle_root: str | Path) -> Dict[str, Path]:
    root = Path(bundle_root).expanduser()
    discovered: Dict[str, Path] = {}
    for path in root.rglob("sd_*.pkl"):
        discovered[path.name] = path.resolve()
    return discovered


def find_bundle_checkpoint(bundle_root: str | Path) -> Optional[Path]:
    root = Path(bundle_root).expanduser()
    for path in sorted(root.rglob("last.ckpt")):
        return path.resolve()
    return None


def find_bundle_config_yaml(bundle_root: str | Path) -> Optional[Path]:
    root = Path(bundle_root).expanduser()
    for path in sorted(root.rglob("config.yaml")):
        return path.resolve()
    return None


def find_materialized_eval_dir(
    *,
    bundle_root: str | Path,
    example_id: str,
    scenario_id: str,
    agent_id: str,
    decision_time_idx: int,
    light_id: Optional[str] = None,
) -> Optional[Path]:
    root = Path(bundle_root).expanduser() / "outputs" / "pr6_path_eval_20260401_run1" / "materialized_eval_inputs" / "examples"
    agent_root = root / str(scenario_id) / f"agent_{agent_id}"
    if not agent_root.is_dir():
        return None
    parsed = parse_example_id(example_id)
    candidate_light_ids: List[str] = []
    if light_id:
        candidate_light_ids.append(str(light_id))
    if parsed.get("light_id"):
        candidate_light_ids.append(str(parsed["light_id"]))
    candidate_light_ids.extend(path.name.removeprefix("light_") for path in sorted(agent_root.glob("light_*")))
    seen = set()
    for candidate_light_id in candidate_light_ids:
        if candidate_light_id in seen:
            continue
        seen.add(candidate_light_id)
        candidate_dir = agent_root / f"light_{candidate_light_id}" / f"t_{int(decision_time_idx):03d}"
        if candidate_dir.is_dir():
            return candidate_dir.resolve()
    return None


def build_bundle_inventory(bundle_root: str | Path) -> Dict[str, Any]:
    root = Path(bundle_root).expanduser()
    scenario_pkls = sorted(discover_local_scenario_pkls(root).values())
    control_sweeps = sorted((root / "outputs" / "pr6_path_eval_20260401_run1").glob("control_sweep_*.png"))
    materialized_examples = sorted((root / "outputs" / "pr6_path_eval_20260401_run1" / "materialized_eval_inputs" / "examples").glob("*"))
    return {
        "bundle_root": str(root.resolve()),
        "path_control_eval_summary_json": str((root / "outputs" / "pr6_path_eval_20260401_run1" / "path_control_eval_summary.json").resolve()),
        "path_control_eval_per_example_jsonl": str((root / "outputs" / "pr6_path_eval_20260401_run1" / "path_control_eval_per_example.jsonl").resolve()),
        "path_index_curated_val_jsonl": str((root / "outputs" / "pr6_path_index_5000" / "path_index_curated_val.jsonl").resolve()),
        "split_summary_json": str((root / "outputs" / "pr6_path_index_5000" / "split_summary.json").resolve()),
        "path_support_summary_curated_json": str((root / "outputs" / "pr6_path_index_5000" / "path_support_summary_curated.json").resolve()),
        "num_local_scenario_pkls": int(len(scenario_pkls)),
        "local_scenario_pkl_basenames": [path.name for path in scenario_pkls],
        "num_control_sweep_pngs": int(len(control_sweeps)),
        "control_sweep_pngs": [str(path.resolve()) for path in control_sweeps],
        "num_materialized_example_scenarios": int(len(materialized_examples)),
        "materialized_example_scenarios": [path.name for path in materialized_examples],
        "checkpoint_path": str(find_bundle_checkpoint(root)) if find_bundle_checkpoint(root) is not None else None,
        "saved_config_yaml": str(find_bundle_config_yaml(root)) if find_bundle_config_yaml(root) is not None else None,
    }


def rewrite_path_index_rows_for_bundle(
    rows: Sequence[Dict[str, Any]],
    *,
    bundle_root: str | Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    local_by_basename = discover_local_scenario_pkls(bundle_root)
    rewritten: List[Dict[str, Any]] = []
    rewritten_count = 0
    missing_count = 0
    example_ids_with_local_scenario = 0
    missing_examples: List[str] = []
    for row in rows:
        new_row = dict(row)
        basename = Path(str(row.get("scenario_pkl", ""))).name
        local_path = local_by_basename.get(basename)
        if local_path is not None:
            new_row["scenario_pkl_local"] = str(local_path)
            rewritten_count += 1
            example_ids_with_local_scenario += 1
        else:
            new_row["scenario_pkl_local"] = None
            missing_count += 1
            if len(missing_examples) < 50:
                missing_examples.append(str(row.get("example_id", basename)))
        rewritten.append(new_row)
    report = {
        "num_rows": int(len(rows)),
        "num_unique_local_scenario_pkls": int(len(local_by_basename)),
        "num_rewritten_rows": int(rewritten_count),
        "num_missing_rows": int(missing_count),
        "first_50_missing_examples": missing_examples,
    }
    return rewritten, report


def assert_same_frame(left_frame: str, right_frame: str, *, context: str) -> None:
    if str(left_frame) != str(right_frame):
        raise ValueError(f"{context}: frame mismatch {left_frame!r} vs {right_frame!r}")


def inverse_map_center_xy(xy: np.ndarray, *, map_center: Sequence[float], map_heading: float) -> np.ndarray:
    array = np.asarray(xy, dtype=np.float64)
    c = math.cos(float(map_heading))
    s = math.sin(float(map_heading))
    x = c * array[..., 0] - s * array[..., 1] + float(map_center[0])
    y = s * array[..., 0] + c * array[..., 1] + float(map_center[1])
    return np.stack([x, y], axis=-1)


def restore_world_headings(headings: np.ndarray, *, map_heading: float) -> np.ndarray:
    return np.asarray([wrap_angle(float(value) + float(map_heading)) for value in np.asarray(headings).reshape(-1)], dtype=np.float64)


def tagged_pose_from_xyh(x: float, y: float, heading: float, *, frame: str) -> TaggedPose:
    return TaggedPose(x=float(x), y=float(y), heading=float(heading), frame=str(frame))


def pose_to_dict(pose: TaggedPose) -> Dict[str, Any]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "heading": float(pose.heading),
        "frame": str(pose.frame),
    }


def world_pose_to_agent_relative(pose: TaggedPose, *, agent_pose_world: Mapping[str, Any]) -> TaggedPose:
    assert_same_frame(pose.frame, FRAME_WORLD, context="world_pose_to_agent_relative.pose")
    agent_x = float(agent_pose_world.get("x", 0.0))
    agent_y = float(agent_pose_world.get("y", 0.0))
    agent_heading = float(agent_pose_world.get("heading", 0.0))
    dx = float(pose.x - agent_x)
    dy = float(pose.y - agent_y)
    c = math.cos(agent_heading)
    s = math.sin(agent_heading)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    heading_rel = wrap_angle(float(pose.heading - agent_heading))
    return tagged_pose_from_xyh(x_rel, y_rel, heading_rel, frame=FRAME_AGENT_RELATIVE_AT_DECISION)


def agent_relative_pose_to_world(pose: TaggedPose, *, agent_pose_world: Mapping[str, Any]) -> TaggedPose:
    assert_same_frame(pose.frame, FRAME_AGENT_RELATIVE_AT_DECISION, context="agent_relative_pose_to_world.pose")
    agent_x = float(agent_pose_world.get("x", 0.0))
    agent_y = float(agent_pose_world.get("y", 0.0))
    agent_heading = float(agent_pose_world.get("heading", 0.0))
    c = math.cos(agent_heading)
    s = math.sin(agent_heading)
    world_x = agent_x + c * float(pose.x) - s * float(pose.y)
    world_y = agent_y + s * float(pose.x) + c * float(pose.y)
    world_heading = wrap_angle(float(agent_heading + pose.heading))
    return tagged_pose_from_xyh(world_x, world_y, world_heading, frame=FRAME_WORLD)


def world_xy_to_agent_relative_xy(
    xy_world: np.ndarray,
    *,
    agent_pose_world: Mapping[str, Any],
) -> np.ndarray:
    xy = np.asarray(xy_world, dtype=np.float64)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    dx = xy[..., 0] - float(agent_pose_world.get("x", 0.0))
    dy = xy[..., 1] - float(agent_pose_world.get("y", 0.0))
    agent_heading = float(agent_pose_world.get("heading", 0.0))
    c = math.cos(agent_heading)
    s = math.sin(agent_heading)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    return np.stack([x_rel, y_rel], axis=-1)


def agent_relative_xy_to_world_xy(
    xy_rel: np.ndarray,
    *,
    agent_pose_world: Mapping[str, Any],
) -> np.ndarray:
    xy = np.asarray(xy_rel, dtype=np.float64)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    agent_x = float(agent_pose_world.get("x", 0.0))
    agent_y = float(agent_pose_world.get("y", 0.0))
    agent_heading = float(agent_pose_world.get("heading", 0.0))
    c = math.cos(agent_heading)
    s = math.sin(agent_heading)
    x_world = agent_x + c * xy[..., 0] - s * xy[..., 1]
    y_world = agent_y + s * xy[..., 0] + c * xy[..., 1]
    return np.stack([x_world, y_world], axis=-1)


def anchor_pose_from_control_code(control_code: Mapping[str, Any]) -> Optional[TaggedPose]:
    anchor = dict(control_code.get("terminal_anchor", {}))
    if not anchor:
        return None
    heading = math.atan2(
        float(anchor.get("target_sin_heading_rel", 0.0)),
        float(anchor.get("target_cos_heading_rel", 1.0)),
    )
    return tagged_pose_from_xyh(
        float(anchor.get("target_x_rel", 0.0)),
        float(anchor.get("target_y_rel", 0.0)),
        heading,
        frame=FRAME_AGENT_RELATIVE_AT_DECISION,
    )


def nearest_point_on_polyline(point_xy: Sequence[float], polyline_xy: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    polyline = np.asarray(polyline_xy, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[0] == 0:
        return None, None
    query = np.asarray(point_xy, dtype=np.float64).reshape(2)
    if polyline.shape[0] == 1:
        dist = float(np.linalg.norm(query - polyline[0, :2]))
        return dist, np.asarray(polyline[0, :2], dtype=np.float64)

    best_dist = float("inf")
    best_point: Optional[np.ndarray] = None
    for idx in range(polyline.shape[0] - 1):
        start = np.asarray(polyline[idx, :2], dtype=np.float64)
        end = np.asarray(polyline[idx + 1, :2], dtype=np.float64)
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom <= 1e-12:
            candidate = start
        else:
            t = float(np.clip(np.dot(query - start, segment) / denom, 0.0, 1.0))
            candidate = start + t * segment
        dist = float(np.linalg.norm(query - candidate))
        if dist < best_dist:
            best_dist = dist
            best_point = candidate
    return best_dist, (None if best_point is None else np.asarray(best_point, dtype=np.float64))


def branch_candidates_world(materialized_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(materialized_dir).expanduser() / "branch_candidates.json"
    if not path.is_file():
        return []
    payload = load_json(path)
    return list(payload.get("branch_candidates", []))


def classify_branch_from_world_pose(
    pose_world: TaggedPose,
    branch_candidates: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    assert_same_frame(pose_world.frame, FRAME_WORLD, context="classify_branch_from_world_pose.pose")
    scored: List[Tuple[float, str]] = []
    for candidate in branch_candidates:
        terminal = dict(candidate.get("terminal_pose") or candidate.get("target_terminal_pose") or {})
        if not terminal:
            continue
        dx = float(pose_world.x - float(terminal.get("x", 0.0)))
        dy = float(pose_world.y - float(terminal.get("y", 0.0)))
        d_heading = wrap_angle(float(pose_world.heading - float(terminal.get("heading", 0.0))))
        score = math.hypot(dx, dy) + 2.0 * abs(d_heading)
        scored.append((score, str(candidate.get("branch_label"))))
    if not scored:
        return {"branch_label": None, "score_margin": None, "best_score": None}
    scored.sort(key=lambda item: item[0])
    best_score, best_label = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None
    return {
        "branch_label": str(best_label),
        "score_margin": float((second_score - best_score) if second_score is not None else 0.0),
        "best_score": float(best_score),
    }


def normalize_predicted_branch(value: Any) -> str:
    if value is None:
        return "none"
    label = str(value)
    if label in {"left", "straight", "right"}:
        return label
    if not label or label.lower() == "none":
        return "none"
    return "other"


def build_confusion_and_breakdown(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    requested_labels = ["left", "straight", "right"]
    predicted_labels = ["left", "straight", "right", "none", "other"]
    confusion = {
        requested_label: {predicted_label: 0 for predicted_label in predicted_labels}
        for requested_label in requested_labels
    }
    breakdown: Dict[str, Dict[str, Any]] = {}
    for label in requested_labels:
        label_rows = [row for row in rows if str(row.get("requested_branch_label")) == label]
        factual_rows = [row for row in label_rows if str(row.get("mode_bucket")) == "factual"]
        alternative_rows = [row for row in label_rows if str(row.get("mode_bucket")) == "alternative"]
        for row in label_rows:
            confusion[label][normalize_predicted_branch(row.get("predicted_branch_label"))] += 1
        breakdown[label] = {
            "num_rows": int(len(label_rows)),
            "num_matches": int(sum(bool(row.get("requested_branch_match")) for row in label_rows)),
            "match_rate": float(sum(bool(row.get("requested_branch_match")) for row in label_rows) / len(label_rows)) if label_rows else 0.0,
            "factual_rows": int(len(factual_rows)),
            "factual_match_rate": float(sum(bool(row.get("requested_branch_match")) for row in factual_rows) / len(factual_rows)) if factual_rows else 0.0,
            "alternative_rows": int(len(alternative_rows)),
            "alternative_match_rate": float(sum(bool(row.get("requested_branch_match")) for row in alternative_rows) / len(alternative_rows)) if alternative_rows else 0.0,
            "mean_branch_score_margin": mean_or_none(row.get("branch_score_margin") for row in label_rows),
            "mean_final_pose_to_requested_anchor_m": mean_or_none(row.get("final_pose_to_requested_anchor_m") for row in label_rows),
        }
    return (
        {"labels": requested_labels, "predicted_labels": predicted_labels, "matrix": confusion},
        {"requested_labels": requested_labels, "breakdown_by_requested_class": breakdown},
    )


def load_motion_config_for_bundle(config_path: Optional[str | Path] = None):
    config = copy.deepcopy(global_config)
    default_cfg = REPO_ROOT / "cfgs" / "0202_midgpt.yaml"
    if default_cfg.is_file():
        config = cfg_from_yaml_file(default_cfg, config)
    if config_path:
        resolved = Path(config_path).expanduser()
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / resolved).resolve()
        config = cfg_from_yaml_file(resolved, config)
    else:
        fallback = REPO_ROOT / "cfgs" / "motion_forward_path_control_strict_local.yaml"
        if fallback.is_file():
            config = cfg_from_yaml_file(fallback, config)
    config.MODEL.LOCAL_CONTROL_FORWARD_ENABLED = True
    config.MODEL.LOCAL_CONTROL_FORWARD_MODE = "strict_local"
    config.DATA.COUNTERFACTUAL_MODE = "path_only"
    return config


def load_model_and_tokenizer_for_bundle(
    *,
    ckpt_path: str | Path,
    config_path: Optional[str | Path] = None,
    load_mode: str = "forgiving_state_dict",
):
    config = load_motion_config_for_bundle(config_path)
    model, load_report = load_model_from_checkpoint_forgiving(
        config=config,
        ckpt_path=str(Path(ckpt_path).expanduser().resolve()),
        load_mode=load_mode,
        strict_state_dict=(load_mode == "strict_state_dict"),
        map_location="cpu",
        checkpoint_surgery_func=utils.checkpoint_surgery_func,
    )
    model.eval()
    tokenizer = get_tokenizer(config)
    return config, model, tokenizer, load_report


def load_raw_scenario(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().open("rb") as f:
        return pickle.load(f)


def preprocess_raw_scenario_for_audit(raw_scenario: Mapping[str, Any], *, config: Any, tokenizer: Any) -> Dict[str, Any]:
    # Scenario preprocessing normalizes track/map coordinates in place. Keep the
    # original raw scenario untouched so downstream audit code can still access
    # trustworthy world-frame map features and track states for plotting.
    sample = preprocess_scenario_description(
        scenario=copy.deepcopy(raw_scenario),
        config=copy.deepcopy(config),
        in_evaluation=True,
        keep_all_data=False,
        backward_prediction=False,
        tokenizer=tokenizer,
    )
    sample["metadata/scenario_id"] = raw_scenario["id"]
    return sample


def _single_sample_to_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
    batch: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in OBJECT_BATCH_KEYS:
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


def _clone_for_model(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_for_model(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_for_model(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_for_model(item) for item in value)
    return copy.deepcopy(value)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_sample_with_control(
    *,
    base_sample: Mapping[str, Any],
    scenario_id: str,
    control_code: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    sample_with_cf = dict(base_sample)
    horizon = int(np.asarray(sample_with_cf["decoder/agent_position"]).shape[0])
    decoder_track_names = sample_with_cf.get("decoder/track_name", [])
    if control_code is None:
        cf_fields = default_counterfactual_dataset_fields(
            scenario_id=str(scenario_id),
            decoder_track_names=decoder_track_names,
            horizon=horizon,
        )
    else:
        cf_fields = build_counterfactual_dataset_fields(
            scenario_id=str(scenario_id),
            decoder_track_names=decoder_track_names,
            horizon=horizon,
            control_code=dict(control_code),
            control_code_path="",
            require_trainable=False,
        )
    sample_with_cf.update(cf_fields)
    return sample_with_cf


def restore_world_trajectory(
    xy_model: np.ndarray,
    heading_model: np.ndarray,
    *,
    map_center: Sequence[float],
    map_heading: float,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        inverse_map_center_xy(np.asarray(xy_model, dtype=np.float64), map_center=map_center, map_heading=map_heading),
        restore_world_headings(np.asarray(heading_model, dtype=np.float64), map_heading=map_heading),
    )


def extract_final_pose(xy: np.ndarray, heading: np.ndarray, valid_mask: np.ndarray, *, frame: str) -> TaggedPose:
    valid = np.asarray(valid_mask, dtype=bool)
    if bool(valid.any()):
        idx = int(np.flatnonzero(valid)[-1])
    else:
        idx = int(np.asarray(xy).shape[0] - 1)
    return tagged_pose_from_xyh(
        float(np.asarray(xy)[idx, 0]),
        float(np.asarray(xy)[idx, 1]),
        float(np.asarray(heading)[idx]),
        frame=frame,
    )


def compute_path_head_prediction(model: Any, forward_out: Mapping[str, Any]) -> Dict[str, Any]:
    hidden = forward_out.get("decoder/control_target_hidden")
    valid_mask = forward_out.get("decoder/control_target_valid_mask")
    if hidden is None or not hasattr(model, "path_head"):
        return {"label": None, "margin": None, "logits": None}
    if valid_mask is not None and not bool(_to_numpy(valid_mask).astype(bool).any()):
        return {"label": None, "margin": None, "logits": None}
    with torch.no_grad():
        logits = model.path_head(hidden)
    logits_np = _to_numpy(logits)[0]
    order = np.argsort(logits_np)[::-1]
    best_idx = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else best_idx
    return {
        "label": str(BRANCH_LABEL_ORDER[best_idx]),
        "margin": float(logits_np[best_idx] - logits_np[second_idx]) if len(order) > 1 else 0.0,
        "logits": [float(value) for value in logits_np.tolist()],
    }


def run_control_variant(
    *,
    base_sample: Mapping[str, Any],
    scenario_id: str,
    mode: str,
    control_code: Optional[Mapping[str, Any]],
    model: Any,
    tokenizer: Any,
    sampling_method: Optional[str] = None,
    temperature: Optional[float] = None,
    topp: Optional[float] = None,
    seed: Optional[int] = None,
    deterministic_agent_ids: bool = False,
) -> Dict[str, Any]:
    sample_with_cf = build_sample_with_control(base_sample=base_sample, scenario_id=scenario_id, control_code=control_code)

    device = torch.device("cpu")
    forward_batch = _to_torch_device(_single_sample_to_batch(sample_with_cf), device=device)
    forward_batch["in_evaluation"] = torch.ones((1,), dtype=torch.bool, device=device)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
    if deterministic_agent_ids:
        randomized_modeled_agent_id = forward_batch["decoder/agent_id"].clone()
    else:
        randomized_modeled_agent_id = model.model.motion_decoder.randomize_modeled_agent_id(forward_batch, clip_agent_id=False)
    forward_batch["decoder/randomized_modeled_agent_id"] = randomized_modeled_agent_id
    rollout_batch = _clone_for_model(forward_batch)

    resolved_sampling_method = str(sampling_method or model.config.SAMPLING.get("SAMPLING_METHOD", "topp"))
    resolved_temperature = float(model.config.SAMPLING.TEMPERATURE if temperature is None else temperature)
    resolved_topp = float(model.config.SAMPLING.TOPP if topp is None else topp)

    with torch.no_grad():
        forward_out = model(_clone_for_model(forward_batch))
        rollout = model.model.autoregressive_rollout(
            rollout_batch,
            num_decode_steps=None,
            sampling_method=resolved_sampling_method,
            temperature=resolved_temperature,
            topp=resolved_topp,
        )
        rollout = tokenizer.detokenize(rollout, detokenizing_gt=False, backward_prediction=False)

    decision_mask = np.asarray(sample_with_cf["cf/decision_agent_mask"], dtype=np.float32) > 0
    target_slot = int(np.flatnonzero(decision_mask)[0]) if bool(decision_mask.any()) else 0

    predicted_xy_model = _to_numpy(rollout["decoder/reconstructed_position"])[0]
    predicted_heading_model = _to_numpy(rollout["decoder/reconstructed_heading"])[0]
    predicted_valid_mask = _to_numpy(rollout["decoder/reconstructed_valid_mask"])[0].astype(bool)
    gt_xy_model = np.asarray(sample_with_cf["decoder/agent_position"], dtype=np.float32)[..., :2]
    gt_heading_model = np.asarray(sample_with_cf["decoder/agent_heading"], dtype=np.float32)
    gt_valid_mask = np.asarray(sample_with_cf["decoder/agent_valid_mask"], dtype=bool)

    map_center = np.asarray(sample_with_cf["metadata/map_center"], dtype=np.float64)
    map_heading = float(sample_with_cf["metadata/map_heading"])

    target_xy_model = predicted_xy_model[:, target_slot, :2]
    target_heading_model = predicted_heading_model[:, target_slot]
    target_valid_mask = predicted_valid_mask[:, target_slot]
    gt_target_xy_model = gt_xy_model[:, target_slot, :2]
    gt_target_heading_model = gt_heading_model[:, target_slot]
    gt_target_valid_mask = gt_valid_mask[:, target_slot]

    target_xy_world, target_heading_world = restore_world_trajectory(
        target_xy_model,
        target_heading_model,
        map_center=map_center,
        map_heading=map_heading,
    )
    gt_target_xy_world, gt_target_heading_world = restore_world_trajectory(
        gt_target_xy_model,
        gt_target_heading_model,
        map_center=map_center,
        map_heading=map_heading,
    )

    non_target_mask = np.ones(predicted_xy_model.shape[1], dtype=bool)
    non_target_mask[target_slot] = False
    non_target_xy_model = predicted_xy_model[:, non_target_mask, :2]
    non_target_heading_model = predicted_heading_model[:, non_target_mask]
    non_target_valid_mask = predicted_valid_mask[:, non_target_mask]
    if non_target_xy_model.size > 0:
        non_target_xy_world = inverse_map_center_xy(non_target_xy_model, map_center=map_center, map_heading=map_heading)
    else:
        non_target_xy_world = np.zeros((predicted_xy_model.shape[0], 0, 2), dtype=np.float64)

    requested_branch_label = None
    requested_anchor = None
    if control_code is not None:
        requested_branch_label = str(control_code.get("path_token", {}).get("branch_label"))
        requested_anchor = anchor_pose_from_control_code(control_code)

    return {
        "mode": str(mode),
        "mode_bucket": mode_bucket(str(mode)),
        "control_code": (dict(control_code) if control_code is not None else None),
        "requested_branch_label": requested_branch_label,
        "requested_anchor": requested_anchor,
        "target_slot": int(target_slot),
        "track_name": str(np.asarray(sample_with_cf["decoder/track_name"])[target_slot]),
        "target_positions_model_output": np.asarray(target_xy_model, dtype=np.float64),
        "target_headings_model_output": np.asarray(target_heading_model, dtype=np.float64),
        "target_positions_world": np.asarray(target_xy_world, dtype=np.float64),
        "target_headings_world": np.asarray(target_heading_world, dtype=np.float64),
        "target_valid_mask": np.asarray(target_valid_mask, dtype=bool),
        "gt_target_positions_model_output": np.asarray(gt_target_xy_model, dtype=np.float64),
        "gt_target_headings_model_output": np.asarray(gt_target_heading_model, dtype=np.float64),
        "gt_target_positions_world": np.asarray(gt_target_xy_world, dtype=np.float64),
        "gt_target_headings_world": np.asarray(gt_target_heading_world, dtype=np.float64),
        "gt_target_valid_mask": np.asarray(gt_target_valid_mask, dtype=bool),
        "non_target_positions_model_output": np.asarray(non_target_xy_model, dtype=np.float64),
        "non_target_positions_world": np.asarray(non_target_xy_world, dtype=np.float64),
        "non_target_valid_mask": np.asarray(non_target_valid_mask, dtype=bool),
        "map_center": np.asarray(map_center, dtype=np.float64),
        "map_heading": float(map_heading),
        "path_head": compute_path_head_prediction(model, forward_out),
        "target_final_pose_model_output": extract_final_pose(target_xy_model, target_heading_model, target_valid_mask, frame=FRAME_MODEL_OUTPUT),
        "target_final_pose_world": extract_final_pose(target_xy_world, target_heading_world, target_valid_mask, frame=FRAME_WORLD),
        "gt_final_pose_world": extract_final_pose(gt_target_xy_world, gt_target_heading_world, gt_target_valid_mask, frame=FRAME_WORLD),
        "sampling_method": resolved_sampling_method,
        "temperature": resolved_temperature,
        "topp": resolved_topp,
        "decode_seed": (None if seed is None else int(seed)),
        "deterministic_agent_ids": bool(deterministic_agent_ids),
    }


def mode_bucket(mode: str) -> str:
    if mode == "no_control":
        return "no_control"
    if mode == "factual":
        return "factual"
    if mode.startswith("alternative_"):
        return "alternative"
    return str(mode)


def load_materialized_controls(materialized_dir: str | Path) -> Dict[str, Any]:
    root = Path(materialized_dir).expanduser()
    factual = load_json(root / "factual_control_code.json") if (root / "factual_control_code.json").is_file() else None
    alternatives = load_json(root / "alternative_control_codes.json") if (root / "alternative_control_codes.json").is_file() else []
    branch_candidates = branch_candidates_world(root)
    local_intervention_train_view = load_json(root / "local_intervention_train_view.json") if (root / "local_intervention_train_view.json").is_file() else None
    return {
        "factual_control_code": factual,
        "alternative_control_codes": alternatives,
        "branch_candidates": branch_candidates,
        "local_intervention_train_view": local_intervention_train_view,
    }


def agent_relative_error_to_anchor(
    *,
    pose_world: TaggedPose,
    anchor_rel: TaggedPose,
    agent_pose_world: Mapping[str, Any],
) -> Tuple[float, float, TaggedPose]:
    pose_rel = world_pose_to_agent_relative(pose_world, agent_pose_world=agent_pose_world)
    assert_same_frame(pose_rel.frame, anchor_rel.frame, context="agent_relative_error_to_anchor")
    dx = float(pose_rel.x - anchor_rel.x)
    dy = float(pose_rel.y - anchor_rel.y)
    heading_error = abs(wrap_angle(float(pose_rel.heading - anchor_rel.heading)))
    return float(math.hypot(dx, dy)), float(heading_error), pose_rel


def trajectory_mean_displacement(
    first_xy: np.ndarray,
    first_valid: np.ndarray,
    second_xy: np.ndarray,
    second_valid: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    length = min(
        int(np.asarray(first_xy).shape[0]),
        int(np.asarray(second_xy).shape[0]),
        int(np.asarray(first_valid).shape[0]),
        int(np.asarray(second_valid).shape[0]),
    )
    if length <= 0:
        return None, None
    mask = np.asarray(first_valid[:length], dtype=bool) & np.asarray(second_valid[:length], dtype=bool)
    if not bool(mask.any()):
        return None, None
    displacement = np.linalg.norm(np.asarray(first_xy[:length])[mask] - np.asarray(second_xy[:length])[mask], axis=-1)
    return float(np.mean(displacement)), float(displacement[-1])


def non_target_displacement(
    first_xy: np.ndarray,
    second_xy: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    if first_xy.size == 0 or second_xy.size == 0:
        return 0.0, 0.0
    if first_xy.shape != second_xy.shape:
        return None, None
    displacement = np.linalg.norm(np.asarray(first_xy) - np.asarray(second_xy), axis=-1)
    return float(np.mean(displacement)), float(np.max(displacement))


def ade_fde(
    pred_xy: np.ndarray,
    pred_valid: np.ndarray,
    gt_xy: np.ndarray,
    gt_valid: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    length = min(
        int(np.asarray(pred_xy).shape[0]),
        int(np.asarray(gt_xy).shape[0]),
        int(np.asarray(pred_valid).shape[0]),
        int(np.asarray(gt_valid).shape[0]),
    )
    if length <= 0:
        return None, None
    mask = np.asarray(pred_valid[:length], dtype=bool) & np.asarray(gt_valid[:length], dtype=bool)
    if not bool(mask.any()):
        return None, None
    errors = np.linalg.norm(np.asarray(pred_xy[:length])[mask] - np.asarray(gt_xy[:length])[mask], axis=-1)
    final_idx = int(np.flatnonzero(mask)[-1])
    return (
        float(np.mean(errors)),
        float(np.linalg.norm(np.asarray(pred_xy[:length])[final_idx] - np.asarray(gt_xy[:length])[final_idx])),
    )


def raw_track_world_state(raw_scenario: Mapping[str, Any], *, track_id: str) -> Dict[str, np.ndarray]:
    track = dict(raw_scenario["tracks"][str(track_id)])
    state = dict(track["state"])
    return {
        "position": np.asarray(state["position"], dtype=np.float64),
        "heading": np.asarray(state["heading"], dtype=np.float64),
        "velocity": np.asarray(state["velocity"], dtype=np.float64),
        "valid": np.asarray(state["valid"], dtype=bool),
    }


def gt_final_world_pose_from_raw(raw_scenario: Mapping[str, Any], *, track_id: str) -> TaggedPose:
    state = raw_track_world_state(raw_scenario, track_id=str(track_id))
    valid = np.asarray(state["valid"], dtype=bool)
    if bool(valid.any()):
        idx = int(np.flatnonzero(valid)[-1])
    else:
        idx = int(len(valid) - 1)
    position = np.asarray(state["position"], dtype=np.float64)
    heading = np.asarray(state["heading"], dtype=np.float64)
    return tagged_pose_from_xyh(float(position[idx, 0]), float(position[idx, 1]), float(heading[idx]), frame=FRAME_WORLD)
