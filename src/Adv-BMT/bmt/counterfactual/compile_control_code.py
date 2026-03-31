from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .conflict_analysis import analyze_conflicts
from .geometry import angle_delta
from .signal_qc import (
    is_caution_signal_state,
    is_go_like_signal_state,
    is_stop_like_signal_state,
)
from .types import CanonicalScenario, stable_string_sort_key

CONTROL_CODE_CONTRACT_NAME = "control_code_v1"
CONTROL_CODE_SCHEMA_VERSION = "counter_bmt_v3_control_code_v1"

BRANCH_LABEL_ORDER = ("none", "left", "straight", "right", "u_turn")
COMPLIANCE_LABEL_ORDER = ("none", "obey_signal", "red_light_violation")
SIGNAL_CATEGORY_ORDER = ("unknown", "stop", "go", "caution")
TIMING_LABEL_ORDER = ("none", "before_conflict", "after_conflict")

BRANCH_LABEL_TO_ID = {name: idx for idx, name in enumerate(BRANCH_LABEL_ORDER)}
COMPLIANCE_LABEL_TO_ID = {name: idx for idx, name in enumerate(COMPLIANCE_LABEL_ORDER)}
SIGNAL_CATEGORY_TO_ID = {name: idx for idx, name in enumerate(SIGNAL_CATEGORY_ORDER)}
TIMING_LABEL_TO_ID = {name: idx for idx, name in enumerate(TIMING_LABEL_ORDER)}

ID_TO_BRANCH_LABEL = {idx: name for name, idx in BRANCH_LABEL_TO_ID.items()}
ID_TO_COMPLIANCE_LABEL = {idx: name for name, idx in COMPLIANCE_LABEL_TO_ID.items()}
ID_TO_SIGNAL_CATEGORY = {idx: name for name, idx in SIGNAL_CATEGORY_TO_ID.items()}
ID_TO_TIMING_LABEL = {idx: name for name, idx in TIMING_LABEL_TO_ID.items()}

PATH_TOKEN_DIM = 5
COMPLIANCE_TOKEN_DIM = 4
TIMING_TOKEN_DIM = 3
TERMINAL_ANCHOR_DIM = 4


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


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    raise TypeError(f"Expected mapping-like value, got {type(value)!r}")


def _coerce_optional_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    return _coerce_mapping(value)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _finite_optional_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return str(default)
    return str(value)


def _signal_state_category(signal_state: Optional[str]) -> str:
    if signal_state is None:
        return "unknown"
    if is_stop_like_signal_state(signal_state):
        return "stop"
    if is_go_like_signal_state(signal_state):
        return "go"
    if is_caution_signal_state(signal_state):
        return "caution"
    return "unknown"


def world_xy_to_agent_frame(
    point_xy: Sequence[float],
    *,
    agent_xy: Sequence[float],
    agent_heading: float,
) -> Tuple[float, float]:
    point = np.asarray(point_xy, dtype=np.float32)[:2]
    origin = np.asarray(agent_xy, dtype=np.float32)[:2]
    delta = point - origin
    c = math.cos(float(agent_heading))
    s = math.sin(float(agent_heading))
    x_rel = c * float(delta[0]) + s * float(delta[1])
    y_rel = -s * float(delta[0]) + c * float(delta[1])
    return float(x_rel), float(y_rel)


def heading_to_agent_frame(target_heading: float, *, agent_heading: float) -> float:
    return float(angle_delta(float(target_heading), float(agent_heading)))


@dataclass
class RelativePose2D:
    x_rel: float
    y_rel: float
    heading_rel: float
    sin_heading_rel: float
    cos_heading_rel: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class PathToken:
    branch_label: str
    branch_id: str
    target_terminal_pose: RelativePose2D

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class ComplianceToken:
    signal_state: Optional[str]
    compliance_label: str
    stop_point_xy: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class TimingToken:
    conflict_agent_id: Optional[str]
    delta_t_entry_s: Optional[float]
    timing_label: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class TerminalAnchor:
    target_x_rel: float
    target_y_rel: float
    target_sin_heading_rel: float
    target_cos_heading_rel: float

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class ControlCodeV1:
    scenario_id: str
    agent_id: str
    decision_time_idx: int
    window: Dict[str, int]
    path_token: PathToken
    compliance_token: ComplianceToken
    timing_token: TimingToken
    terminal_anchor: TerminalAnchor
    sparse_time_mask: List[float]
    debug: Dict[str, Any] = field(default_factory=dict)
    contract_name: str = CONTROL_CODE_CONTRACT_NAME
    schema_version: str = CONTROL_CODE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(order=True)
class IndexedControlCode:
    sort_key: Tuple[int, str] = field(init=False, repr=False)
    scenario_id: str
    decision_time_idx: int
    path: str
    available: bool = False
    target_is_trainable: bool = False
    path_choice_supervisable: bool = False
    compliance_supervisable: bool = False
    timing_supervisable: bool = False

    def __post_init__(self) -> None:
        filename = Path(self.path).name
        priority = 0 if filename == "factual_control_code.json" else (1 if filename == "control_code.json" else 2)
        self.sort_key = (
            0 if self.available else 1,
            0 if self.target_is_trainable else 1,
            0 if self.path_choice_supervisable else 1,
            0 if self.compliance_supervisable else 1,
            0 if self.timing_supervisable else 1,
            int(self.decision_time_idx),
            f"{priority}:{self.path}",
        )


def validate_control_code(control_code: Mapping[str, Any]) -> List[str]:
    payload = dict(control_code)
    errors: List[str] = []
    if str(payload.get("schema_version")) != CONTROL_CODE_SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    for key in (
        "scenario_id",
        "agent_id",
        "decision_time_idx",
        "window",
        "path_token",
        "compliance_token",
        "timing_token",
        "terminal_anchor",
        "sparse_time_mask",
        "debug",
    ):
        if key not in payload:
            errors.append(f"missing_{key}")

    window = payload.get("window", {})
    if int(window.get("end_idx", -1)) < int(window.get("start_idx", 0)):
        errors.append("invalid_window_range")

    path_token = payload.get("path_token", {})
    for key in ("branch_label", "branch_id", "target_terminal_pose"):
        if key not in path_token:
            errors.append(f"missing_path_token_{key}")

    compliance_token = payload.get("compliance_token", {})
    for key in ("signal_state", "compliance_label", "stop_point_xy"):
        if key not in compliance_token:
            errors.append(f"missing_compliance_token_{key}")

    timing_token = payload.get("timing_token", {})
    for key in ("conflict_agent_id", "delta_t_entry_s", "timing_label"):
        if key not in timing_token:
            errors.append(f"missing_timing_token_{key}")

    terminal_anchor = payload.get("terminal_anchor", {})
    for key in (
        "target_x_rel",
        "target_y_rel",
        "target_sin_heading_rel",
        "target_cos_heading_rel",
    ):
        if key not in terminal_anchor:
            errors.append(f"missing_terminal_anchor_{key}")

    mask = payload.get("sparse_time_mask", [])
    if not isinstance(mask, list) or not mask:
        errors.append("invalid_sparse_time_mask")
    return errors


def _find_branch_candidate_payload(payload: Mapping[str, Any], branch_id: str) -> Optional[Dict[str, Any]]:
    debug_payload = _coerce_mapping(payload.get("debug", {}))
    branch_candidates = debug_payload.get("branch_candidates", [])
    if not isinstance(branch_candidates, Sequence):
        return None
    for candidate in branch_candidates:
        candidate_payload = _coerce_mapping(candidate)
        if _normalize_string(candidate_payload.get("branch_id")) == str(branch_id):
            return candidate_payload
    return None


def _resolve_terminal_pose(
    *,
    payload: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    has_ground_truth_target: bool,
) -> Tuple[Dict[str, Any], bool]:
    terminal_pose = _coerce_optional_mapping(decision_payload.get("terminal_pose", {}))
    if terminal_pose:
        return terminal_pose, True
    branch_id = _normalize_string(decision_payload.get("branch_id"))
    if not branch_id:
        return {}, False
    if has_ground_truth_target:
        return terminal_pose, True
    branch_payload = _find_branch_candidate_payload(payload, branch_id)
    if branch_payload is not None:
        return _coerce_mapping(branch_payload.get("terminal_pose", {})), True
    raw_recovered = _coerce_mapping(payload.get("raw_recovered_decision", payload.get("gt_decision", {})))
    return _coerce_optional_mapping(raw_recovered.get("terminal_pose", {})), False


def _build_feasibility_flags(
    *,
    payload: Mapping[str, Any],
    decision_payload: Mapping[str, Any],
    terminal_pose_found: bool,
    has_ground_truth_target: bool,
) -> Dict[str, Any]:
    context = _coerce_mapping(payload.get("context", {}))
    signal_state = context.get("signal_state_at_decision")
    signal_category = _signal_state_category(signal_state)
    compliance_label = _normalize_string(decision_payload.get("compliance_label"), default="none")
    timing_label = decision_payload.get("entry_timing")
    path_feasible = bool(terminal_pose_found and _normalize_string(decision_payload.get("branch_label"), default="none") != "none")
    compliance_feasible = True
    if compliance_label == "red_light_violation":
        compliance_feasible = signal_category in {"stop", "caution"}
    timing_feasible = timing_label is None or len(_coerce_mapping(payload.get("context", {})).get("conflict_agents", [])) > 0
    return {
        "path_feasible": bool(path_feasible),
        "compliance_feasible": bool(compliance_feasible),
        "timing_feasible": bool(timing_feasible),
        "overall_feasible": bool(path_feasible and compliance_feasible and timing_feasible),
        "has_ground_truth_target": bool(has_ground_truth_target),
    }


def _compile_control_code(
    payload: Mapping[str, Any],
    *,
    canonical: CanonicalScenario,
    source_path: str,
    decision_payload: Mapping[str, Any],
    has_ground_truth_target: bool,
    control_kind: str,
    alternative_rank: Optional[int] = None,
) -> ControlCodeV1:
    payload = _coerce_mapping(payload)
    scenario_id = _normalize_string(payload.get("scenario_id"))
    agent_id = _normalize_string(payload.get("agent_id"))
    if scenario_id != canonical.scenario_id:
        raise ValueError(
            f"Scenario mismatch between intervention ({scenario_id}) and canonical scenario ({canonical.scenario_id})"
        )
    if agent_id not in canonical.tracks:
        raise KeyError(f"agent_id={agent_id!r} is not present in canonical tracks")

    window = _coerce_mapping(payload.get("window", {}))
    decision_time_idx = int(np.clip(int(payload.get("decision_time_idx", 0)), 0, max(0, canonical.length - 1)))
    window_start = int(np.clip(int(window.get("start_idx", decision_time_idx)), 0, max(0, canonical.length - 1)))
    window_end = int(np.clip(int(window.get("end_idx", decision_time_idx)), window_start, max(0, canonical.length - 1)))

    agent_track = canonical.tracks[agent_id]
    agent_xy = np.asarray(agent_track.position_xy[decision_time_idx], dtype=np.float32)
    agent_heading = _finite_float(agent_track.heading[decision_time_idx], default=0.0)

    terminal_pose, terminal_pose_found = _resolve_terminal_pose(
        payload=payload,
        decision_payload=decision_payload,
        has_ground_truth_target=has_ground_truth_target,
    )
    if terminal_pose_found:
        target_x_rel, target_y_rel = world_xy_to_agent_frame(
            [terminal_pose.get("x", 0.0), terminal_pose.get("y", 0.0)],
            agent_xy=agent_xy,
            agent_heading=agent_heading,
        )
        target_heading_rel = heading_to_agent_frame(
            _finite_float(terminal_pose.get("heading", 0.0), default=0.0),
            agent_heading=agent_heading,
        )
    else:
        target_x_rel = 0.0
        target_y_rel = 0.0
        target_heading_rel = 0.0
    relative_terminal_pose = RelativePose2D(
        x_rel=target_x_rel,
        y_rel=target_y_rel,
        heading_rel=target_heading_rel,
        sin_heading_rel=float(math.sin(target_heading_rel)),
        cos_heading_rel=float(math.cos(target_heading_rel)),
    )

    context = _coerce_mapping(payload.get("context", {}))
    stop_point_xy = context.get("stop_point_xy", [0.0, 0.0])
    stop_x_rel, stop_y_rel = world_xy_to_agent_frame(
        stop_point_xy,
        agent_xy=agent_xy,
        agent_heading=agent_heading,
    )

    stop_point_tuple = (float(stop_point_xy[0]), float(stop_point_xy[1]))
    conflict_result = analyze_conflicts(
        canonical,
        agent_id=agent_id,
        stop_point_xy=stop_point_tuple,
        decision_time_idx=decision_time_idx,
    )
    preferred_conflict = conflict_result.conflict_agents[0] if conflict_result.conflict_agents else None
    delta_t_entry_s: Optional[float] = None
    conflict_agent_id: Optional[str] = None
    if (
        preferred_conflict is not None
        and preferred_conflict.eta_s is not None
        and (conflict_result.target_eta_s is not None or conflict_result.sdc_eta_s is not None)
    ):
        conflict_agent_id = preferred_conflict.track_id
        reference_eta = conflict_result.target_eta_s if conflict_result.target_eta_s is not None else conflict_result.sdc_eta_s
        delta_t_entry_s = float(preferred_conflict.eta_s - reference_eta)

    timing_label = decision_payload.get("entry_timing")
    if timing_label is None and delta_t_entry_s is not None:
        timing_label = "before_conflict" if float(delta_t_entry_s) >= 0.0 else "after_conflict"

    mask = np.zeros((canonical.length,), dtype=np.float32)
    mask[window_start : window_end + 1] = 1.0
    feasibility = _build_feasibility_flags(
        payload=payload,
        decision_payload=decision_payload,
        terminal_pose_found=terminal_pose_found,
        has_ground_truth_target=has_ground_truth_target,
    )

    control_code = ControlCodeV1(
        scenario_id=scenario_id,
        agent_id=agent_id,
        decision_time_idx=decision_time_idx,
        window={"start_idx": window_start, "end_idx": window_end},
        path_token=PathToken(
            branch_label=_normalize_string(decision_payload.get("branch_label"), default="none"),
            branch_id=_normalize_string(decision_payload.get("branch_id"), default=""),
            target_terminal_pose=relative_terminal_pose,
        ),
        compliance_token=ComplianceToken(
            signal_state=context.get("signal_state_at_decision"),
            compliance_label=_normalize_string(decision_payload.get("compliance_label"), default="none"),
            stop_point_xy=(stop_x_rel, stop_y_rel),
        ),
        timing_token=TimingToken(
            conflict_agent_id=conflict_agent_id,
            delta_t_entry_s=delta_t_entry_s,
            timing_label=None if timing_label is None else str(timing_label),
        ),
        terminal_anchor=TerminalAnchor(
            target_x_rel=relative_terminal_pose.x_rel,
            target_y_rel=relative_terminal_pose.y_rel,
            target_sin_heading_rel=relative_terminal_pose.sin_heading_rel,
            target_cos_heading_rel=relative_terminal_pose.cos_heading_rel,
        ),
        sparse_time_mask=mask.tolist(),
        debug={
            "source_contract_name": payload.get("contract_name"),
            "source_schema_version": payload.get("schema_version"),
            "source_view_type": payload.get("view_type"),
            "source_path": str(source_path),
            "num_alternatives": len(payload.get("alternatives", [])),
            "control_kind": control_kind,
            "alternative_rank": alternative_rank,
            "has_ground_truth_target": has_ground_truth_target,
            "feasibility": feasibility,
            "source_provenance": payload.get("provenance"),
            "source_supervision_gates": payload.get("supervision", payload.get("supervision_gates")),
            "source_target_agent_alignment": payload.get("target_alignment", payload.get("target_agent_alignment")),
            "signal_qc_confidence": _finite_optional_float(_coerce_mapping(payload.get("signal_qc", {})).get("confidence_score")),
            "agent_pose_at_decision": {
                "x": float(agent_xy[0]),
                "y": float(agent_xy[1]),
                "heading": float(agent_heading),
            },
            "delta_t_entry_sign_convention": "conflict_eta_minus_agent_eta",
            "selected_conflict_agent_id": conflict_agent_id,
            "control_available_at_current": payload.get("control_available_at_current"),
            "target_is_trainable": payload.get("target_is_trainable"),
            "conditioning_eligible": payload.get("conditioning_eligible"),
        },
    )
    return control_code


def compile_control_code_from_local_intervention(
    local_intervention: Mapping[str, Any] | Any,
    *,
    canonical: CanonicalScenario,
    source_path: str = "",
) -> ControlCodeV1:
    payload = _coerce_mapping(local_intervention)
    gt_decision = _coerce_mapping(payload.get("supervised_decision", payload.get("gt_decision", {})))
    has_ground_truth_target = bool(gt_decision.get("branch_label") is not None and gt_decision.get("terminal_pose") is not None)
    return _compile_control_code(
        payload,
        canonical=canonical,
        source_path=source_path,
        decision_payload=gt_decision,
        has_ground_truth_target=has_ground_truth_target,
        control_kind="factual",
    )


def compile_alternative_control_codes_from_local_intervention(
    local_intervention: Mapping[str, Any] | Any,
    *,
    canonical: CanonicalScenario,
    source_path: str = "",
) -> List[Dict[str, Any]]:
    payload = _coerce_mapping(local_intervention)
    alternatives = payload.get("alternatives", [])
    if not isinstance(alternatives, Sequence):
        return []
    records: List[Dict[str, Any]] = []
    for idx, alternative in enumerate(alternatives):
        alternative_payload = _coerce_mapping(alternative)
        control_code = _compile_control_code(
            payload,
            canonical=canonical,
            source_path=source_path,
            decision_payload=alternative_payload,
            has_ground_truth_target=False,
            control_kind="alternative",
            alternative_rank=idx,
        )
        records.append(
            {
                "alternative_rank": idx,
                "branch_id": alternative_payload.get("branch_id"),
                "branch_label": alternative_payload.get("branch_label"),
                "compliance_label": alternative_payload.get("compliance_label"),
                "entry_timing": alternative_payload.get("entry_timing"),
                "has_ground_truth_target": False,
                "feasibility": _coerce_mapping(control_code.debug.get("feasibility", {})),
                "control_code": control_code.to_dict(),
            }
        )
    return records


def encode_path_token_tensor(control_code: Mapping[str, Any]) -> np.ndarray:
    path_token = _coerce_mapping(control_code.get("path_token", {}))
    target_pose = _coerce_mapping(path_token.get("target_terminal_pose", {}))
    branch_label = _normalize_string(path_token.get("branch_label"), default="none")
    return np.asarray(
        [
            float(BRANCH_LABEL_TO_ID.get(branch_label, 0)),
            _finite_float(target_pose.get("x_rel", 0.0), default=0.0),
            _finite_float(target_pose.get("y_rel", 0.0), default=0.0),
            _finite_float(target_pose.get("sin_heading_rel", 0.0), default=0.0),
            _finite_float(target_pose.get("cos_heading_rel", 0.0), default=0.0),
        ],
        dtype=np.float32,
    )


def encode_compliance_token_tensor(control_code: Mapping[str, Any]) -> np.ndarray:
    compliance_token = _coerce_mapping(control_code.get("compliance_token", {}))
    stop_xy = compliance_token.get("stop_point_xy", [0.0, 0.0])
    signal_category = _signal_state_category(compliance_token.get("signal_state"))
    compliance_label = _normalize_string(compliance_token.get("compliance_label"), default="none")
    return np.asarray(
        [
            float(SIGNAL_CATEGORY_TO_ID.get(signal_category, 0)),
            float(COMPLIANCE_LABEL_TO_ID.get(compliance_label, 0)),
            _finite_float(stop_xy[0], default=0.0),
            _finite_float(stop_xy[1], default=0.0),
        ],
        dtype=np.float32,
    )


def encode_timing_token_tensor(control_code: Mapping[str, Any]) -> np.ndarray:
    timing_token = _coerce_mapping(control_code.get("timing_token", {}))
    conflict_agent_id = timing_token.get("conflict_agent_id")
    timing_label = timing_token.get("timing_label")
    return np.asarray(
        [
            1.0 if conflict_agent_id is not None else 0.0,
            _finite_float(timing_token.get("delta_t_entry_s", 0.0), default=0.0),
            float(TIMING_LABEL_TO_ID.get(_normalize_string(timing_label, default="none"), 0)),
        ],
        dtype=np.float32,
    )


def encode_terminal_anchor_tensor(control_code: Mapping[str, Any]) -> np.ndarray:
    anchor = _coerce_mapping(control_code.get("terminal_anchor", {}))
    return np.asarray(
        [
            _finite_float(anchor.get("target_x_rel", 0.0), default=0.0),
            _finite_float(anchor.get("target_y_rel", 0.0), default=0.0),
            _finite_float(anchor.get("target_sin_heading_rel", 0.0), default=0.0),
            _finite_float(anchor.get("target_cos_heading_rel", 0.0), default=0.0),
        ],
        dtype=np.float32,
    )


def default_counterfactual_dataset_fields(
    *,
    scenario_id: str,
    decoder_track_names: Sequence[Any],
    horizon: int,
) -> Dict[str, Any]:
    return {
        "cf/path_token": np.zeros((PATH_TOKEN_DIM,), dtype=np.float32),
        "cf/compliance_token": np.zeros((COMPLIANCE_TOKEN_DIM,), dtype=np.float32),
        "cf/timing_token": np.zeros((TIMING_TOKEN_DIM,), dtype=np.float32),
        "cf/terminal_anchor": np.zeros((TERMINAL_ANCHOR_DIM,), dtype=np.float32),
        "cf/time_window_mask": np.zeros((int(horizon),), dtype=np.float32),
        "cf/decision_agent_mask": np.zeros((len(decoder_track_names),), dtype=np.float32),
        "cf/conditioning_eligible": 0,
        "cf/control_available": 0,
        "cf/path_supervision_mask": 0,
        "cf/compliance_supervision_mask": 0,
        "cf/timing_supervision_mask": 0,
        "cf/target_is_trainable": 0,
        "cf/debug_meta": {
            "available": False,
            "conditioning_eligible": False,
            "auxiliary_supervision_eligible": False,
            "scenario_id": str(scenario_id),
            "control_code_path": "",
        },
    }


def build_counterfactual_dataset_fields(
    *,
    scenario_id: str,
    decoder_track_names: Sequence[Any],
    horizon: int,
    control_code: Optional[Mapping[str, Any]] = None,
    control_code_path: str = "",
    require_trainable: bool = False,
) -> Dict[str, Any]:
    if control_code is None:
        return default_counterfactual_dataset_fields(
            scenario_id=scenario_id,
            decoder_track_names=decoder_track_names,
            horizon=horizon,
        )

    payload = dict(control_code)
    fields = default_counterfactual_dataset_fields(
        scenario_id=scenario_id,
        decoder_track_names=decoder_track_names,
        horizon=horizon,
    )
    mask = np.zeros((int(horizon),), dtype=np.float32)
    raw_mask = np.asarray(payload.get("sparse_time_mask", []), dtype=np.float32).reshape(-1)
    rows = min(mask.shape[0], raw_mask.shape[0])
    if rows > 0:
        mask[:rows] = raw_mask[:rows]

    decision_agent_mask = np.zeros((len(decoder_track_names),), dtype=np.float32)
    agent_id = _normalize_string(payload.get("agent_id"))
    modeled_agent_index = None
    for idx, track_name in enumerate(decoder_track_names):
        if _normalize_string(track_name) == agent_id:
            decision_agent_mask[idx] = 1.0
            if modeled_agent_index is None:
                modeled_agent_index = int(idx)

    source_alignment = _coerce_mapping(_coerce_mapping(payload.get("debug", {})).get("source_target_agent_alignment", {}))
    source_supervision = _coerce_mapping(_coerce_mapping(payload.get("debug", {})).get("source_supervision_gates", {}))
    path_token_payload = _coerce_mapping(payload.get("path_token", {}))
    compliance_token_payload = _coerce_mapping(payload.get("compliance_token", {}))
    timing_token_payload = _coerce_mapping(payload.get("timing_token", {}))
    decision_agent_is_modeled = modeled_agent_index is not None
    source_trainable = source_alignment.get("target_is_trainable")
    target_is_trainable = bool(decision_agent_is_modeled and (True if source_trainable is None else bool(source_trainable)))
    control_available_at_current = bool(
        _coerce_mapping(payload.get("debug", {})).get("control_available_at_current", True)
    )
    path_choice_supervisable = bool(
        source_supervision.get(
            "path_choice_supervisable",
            _normalize_string(path_token_payload.get("branch_label"), default="none") != "none",
        )
    )
    compliance_supervisable = bool(
        source_supervision.get(
            "compliance_supervisable",
            _normalize_string(compliance_token_payload.get("compliance_label"), default="none") != "none",
        )
    )
    timing_supervisable = bool(
        source_supervision.get(
            "timing_supervisable",
            _normalize_string(timing_token_payload.get("timing_label"), default="none") != "none",
        )
    )
    has_any_supervision = bool(path_choice_supervisable or compliance_supervisable or timing_supervisable)
    conditioning_eligible = bool(
        payload.get(
            "conditioning_eligible",
            _coerce_mapping(payload.get("debug", {})).get(
                "conditioning_eligible",
                target_is_trainable and control_available_at_current and has_any_supervision,
            ),
        )
    )
    auxiliary_supervision_eligible = bool(
        target_is_trainable
        and decision_agent_is_modeled
        and mask.sum() > 0.0
        and decision_agent_mask.sum() > 0.0
        and has_any_supervision
    )
    available = bool(conditioning_eligible and auxiliary_supervision_eligible)
    debug_meta = {
        "available": available,
        "conditioning_eligible": conditioning_eligible,
        "auxiliary_supervision_eligible": auxiliary_supervision_eligible,
        "scenario_id": str(payload.get("scenario_id", scenario_id)),
        "schema_version": payload.get("schema_version"),
        "control_code_path": str(control_code_path),
        "view_type": payload.get("view_type", _coerce_mapping(payload.get("debug", {})).get("source_view_type")),
        "branch_id": path_token_payload.get("branch_id"),
        "branch_label": path_token_payload.get("branch_label"),
        "signal_state": compliance_token_payload.get("signal_state"),
        "conflict_agent_id": timing_token_payload.get("conflict_agent_id"),
        "decision_time_idx": payload.get("decision_time_idx"),
        "decision_agent_is_modeled": bool(decision_agent_is_modeled),
        "modeled_agent_index": modeled_agent_index,
        "target_is_trainable": bool(target_is_trainable),
        "control_kind": _coerce_mapping(payload.get("debug", {})).get("control_kind"),
        "has_ground_truth_target": _coerce_mapping(payload.get("debug", {})).get("has_ground_truth_target"),
        "control_available_at_current": control_available_at_current,
        "path_choice_supervisable": path_choice_supervisable,
        "compliance_supervisable": compliance_supervisable,
        "timing_supervisable": timing_supervisable,
        "decision_state": source_supervision.get("decision_state"),
    }
    if require_trainable and not target_is_trainable:
        fields["cf/debug_meta"].update(debug_meta)
        fields["cf/debug_meta"]["available"] = False
        fields["cf/debug_meta"]["drop_reason"] = "non_trainable_target"
        return fields
    if not auxiliary_supervision_eligible:
        fields["cf/debug_meta"].update(debug_meta)
        fields["cf/debug_meta"]["available"] = False
        if not decision_agent_is_modeled:
            fields["cf/debug_meta"]["drop_reason"] = "decision_agent_not_modeled"
        elif mask.sum() <= 0.0:
            fields["cf/debug_meta"]["drop_reason"] = "empty_time_window"
        elif not has_any_supervision:
            fields["cf/debug_meta"]["drop_reason"] = "no_supervision_labels"
        return fields

    if not conditioning_eligible:
        debug_meta["available"] = False
        if not control_available_at_current:
            debug_meta["drop_reason"] = "control_unavailable_at_current"
        else:
            debug_meta["drop_reason"] = "conditioning_ineligible"

    fields.update(
        {
            "cf/path_token": encode_path_token_tensor(payload),
            "cf/compliance_token": encode_compliance_token_tensor(payload),
            "cf/timing_token": encode_timing_token_tensor(payload),
            "cf/terminal_anchor": encode_terminal_anchor_tensor(payload),
            "cf/time_window_mask": mask,
            "cf/decision_agent_mask": decision_agent_mask,
            "cf/conditioning_eligible": int(conditioning_eligible),
            "cf/control_available": int(conditioning_eligible),
            "cf/path_supervision_mask": int(path_choice_supervisable),
            "cf/compliance_supervision_mask": int(compliance_supervisable),
            "cf/timing_supervision_mask": int(timing_supervisable),
            "cf/target_is_trainable": int(target_is_trainable),
            "cf/debug_meta": debug_meta,
        }
    )
    return fields


def decode_path_token_tensor(path_token: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(path_token, dtype=np.float32).reshape(-1)
    branch_id = int(round(float(values[0]))) if values.size > 0 else 0
    return {
        "branch_label_id": branch_id,
        "branch_label": ID_TO_BRANCH_LABEL.get(branch_id, "none"),
        "target_x_rel": float(values[1]) if values.size > 1 else 0.0,
        "target_y_rel": float(values[2]) if values.size > 2 else 0.0,
        "target_sin_heading_rel": float(values[3]) if values.size > 3 else 0.0,
        "target_cos_heading_rel": float(values[4]) if values.size > 4 else 0.0,
    }


def decode_compliance_token_tensor(compliance_token: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(compliance_token, dtype=np.float32).reshape(-1)
    signal_category_id = int(round(float(values[0]))) if values.size > 0 else 0
    compliance_label_id = int(round(float(values[1]))) if values.size > 1 else 0
    return {
        "signal_category_id": signal_category_id,
        "signal_category": ID_TO_SIGNAL_CATEGORY.get(signal_category_id, "unknown"),
        "compliance_label_id": compliance_label_id,
        "compliance_label": ID_TO_COMPLIANCE_LABEL.get(compliance_label_id, "none"),
        "stop_x_rel": float(values[2]) if values.size > 2 else 0.0,
        "stop_y_rel": float(values[3]) if values.size > 3 else 0.0,
    }


def decode_timing_token_tensor(timing_token: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(timing_token, dtype=np.float32).reshape(-1)
    timing_label_id = int(round(float(values[2]))) if values.size > 2 else 0
    return {
        "has_conflict_agent": bool(round(float(values[0]))) if values.size > 0 else False,
        "delta_t_entry_s": float(values[1]) if values.size > 1 else 0.0,
        "timing_label_id": timing_label_id,
        "timing_label": ID_TO_TIMING_LABEL.get(timing_label_id, "none"),
    }


def decode_terminal_anchor_tensor(anchor: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(anchor, dtype=np.float32).reshape(-1)
    return {
        "target_x_rel": float(values[0]) if values.size > 0 else 0.0,
        "target_y_rel": float(values[1]) if values.size > 1 else 0.0,
        "target_sin_heading_rel": float(values[2]) if values.size > 2 else 0.0,
        "target_cos_heading_rel": float(values[3]) if values.size > 3 else 0.0,
    }


def decode_time_window_mask(mask: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    active_idx = np.flatnonzero(values > 0.0)
    return {
        "num_active_steps": int(active_idx.size),
        "first_active_idx": int(active_idx[0]) if active_idx.size > 0 else None,
        "last_active_idx": int(active_idx[-1]) if active_idx.size > 0 else None,
    }


def decode_decision_agent_mask(mask: Sequence[float], *, decoder_track_names: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    values = np.asarray(mask, dtype=np.float32).reshape(-1)
    active_idx = np.flatnonzero(values > 0.0)
    active_track_names: List[str] = []
    if decoder_track_names is not None:
        for idx in active_idx.tolist():
            if idx < len(decoder_track_names):
                active_track_names.append(_normalize_string(decoder_track_names[idx]))
    return {
        "num_active_agents": int(active_idx.size),
        "active_agent_indices": active_idx.astype(int).tolist(),
        "active_track_names": active_track_names,
    }


def load_control_code(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != CONTROL_CODE_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a {CONTROL_CODE_SCHEMA_VERSION} control code")
    return payload


def load_local_intervention(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def discover_control_code_files(root: str | Path) -> List[Path]:
    root_path = Path(root).expanduser()
    if root_path.is_file():
        return [root_path]
    candidates: List[Path] = []
    for pattern in ("factual_control_code.json", "control_code.json", "*_control_code.json"):
        candidates.extend(root_path.rglob(pattern))
    unique = sorted({path.resolve() for path in candidates})
    return unique


def index_control_code_directory(root: str | Path) -> Dict[str, str]:
    preferred: Dict[str, IndexedControlCode] = {}
    for path in discover_control_code_files(root):
        try:
            payload = load_control_code(path)
        except Exception:
            continue
        scenario_id = _normalize_string(payload.get("scenario_id"))
        entry = IndexedControlCode(
            scenario_id=scenario_id,
            decision_time_idx=int(payload.get("decision_time_idx", 0)),
            path=str(path),
            available=bool(
                _coerce_mapping(payload.get("debug", {})).get(
                    "conditioning_eligible",
                    _coerce_mapping(payload.get("debug", {})).get("control_available_at_current", False),
                )
            ),
            target_is_trainable=bool(_coerce_mapping(payload.get("debug", {})).get("target_is_trainable", False)),
            path_choice_supervisable=bool(_coerce_mapping(_coerce_mapping(payload.get("debug", {})).get("source_supervision_gates", {})).get("path_choice_supervisable", False)),
            compliance_supervisable=bool(_coerce_mapping(_coerce_mapping(payload.get("debug", {})).get("source_supervision_gates", {})).get("compliance_supervisable", False)),
            timing_supervisable=bool(_coerce_mapping(_coerce_mapping(payload.get("debug", {})).get("source_supervision_gates", {})).get("timing_supervisable", False)),
        )
        current = preferred.get(scenario_id)
        if current is None or entry.sort_key < current.sort_key:
            preferred[scenario_id] = entry
    return {scenario_id: entry.path for scenario_id, entry in preferred.items()}


def compile_control_codes_in_directory(
    local_intervention_files: Iterable[str | Path],
    *,
    outdir: str | Path,
    canonical_loader: Any,
) -> Dict[str, Any]:
    output_root = Path(outdir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    for intervention_path in sorted(
        (Path(path).expanduser() for path in local_intervention_files),
        key=lambda path: stable_string_sort_key(path.as_posix()),
    ):
        intervention = load_local_intervention(intervention_path)
        candidate_debug = _coerce_mapping(_coerce_mapping(intervention.get("debug", {})).get("candidate", {}))
        scenario_pkl = candidate_debug.get("scenario_pkl", "")
        canonical = canonical_loader(scenario_pkl)
        factual_control_code = compile_control_code_from_local_intervention(
            intervention,
            canonical=canonical,
            source_path=str(intervention_path),
        )
        alternative_control_codes = compile_alternative_control_codes_from_local_intervention(
            intervention,
            canonical=canonical,
            source_path=str(intervention_path),
        )
        scenario_dir = output_root / factual_control_code.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        factual_path = scenario_dir / "factual_control_code.json"
        compatibility_path = scenario_dir / "control_code.json"
        alternative_path = scenario_dir / "alternative_control_codes.json"
        factual_payload = factual_control_code.to_dict()
        factual_path.write_text(json.dumps(factual_payload, indent=2, sort_keys=True), encoding="utf-8")
        compatibility_path.write_text(json.dumps(factual_payload, indent=2, sort_keys=True), encoding="utf-8")
        alternative_path.write_text(json.dumps(alternative_control_codes, indent=2, sort_keys=True), encoding="utf-8")
        results.append(
            {
                "scenario_id": factual_control_code.scenario_id,
                "agent_id": factual_control_code.agent_id,
                "decision_time_idx": factual_control_code.decision_time_idx,
                "factual_control_code_path": str(factual_path),
                "alternative_control_codes_path": str(alternative_path),
                "validation_errors": validate_control_code(factual_payload),
                "num_alternatives": len(alternative_control_codes),
            }
        )
    return {
        "outdir": str(output_root),
        "compiled_count": len(results),
        "results": results,
    }
