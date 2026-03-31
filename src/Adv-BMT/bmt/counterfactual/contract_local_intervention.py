from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOCAL_INTERVENTION_CONTRACT_NAME = "local_intervention_v1"
LOCAL_INTERVENTION_CONTRACT_VERSION = "1"
LOCAL_INTERVENTION_SCHEMA_VERSION = "counter_bmt_v3_local_intervention_v1"
LOCAL_INTERVENTION_RAW_CONTRACT_NAME = "local_intervention_raw_v1"
LOCAL_INTERVENTION_RAW_SCHEMA_VERSION = "counter_bmt_v3_local_intervention_raw_v1"
LOCAL_INTERVENTION_TRAIN_VIEW_CONTRACT_NAME = "local_intervention_train_view_v1"
LOCAL_INTERVENTION_TRAIN_VIEW_SCHEMA_VERSION = "counter_bmt_v3_local_intervention_train_view_v1"


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    return value


@dataclass
class FilterReport:
    stage: str
    input_count: int
    kept_count: int
    dropped_count: int
    drop_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class WindowSpec:
    start_idx: int
    end_idx: int


@dataclass
class ConflictAgentRef:
    track_id: str
    eta_s: Optional[float]
    eta_gap_s: Optional[float]


@dataclass
class ArtifactProvenance:
    agent_id: str
    agent_role: str
    current_time_index_global: int
    decision_time_index_global: int
    cross_time_index_global: Optional[int]
    branch_commit_index_global: Optional[int]
    decision_time_index_rel_to_current: int
    cross_time_index_rel_to_current: Optional[int]
    branch_commit_index_rel_to_current: Optional[int]
    control_available_at_current: bool


@dataclass
class CommitmentMetrics:
    signed_stopline_progress_m: float
    downstream_progress_along_branch_m: float
    intersection_core_dwell_s: float
    best_branch_score: float
    second_best_branch_score: float
    branch_margin: float
    final_heading_error_rad: float
    mean_lateral_error_to_best_branch_m: float


@dataclass
class SupervisionGates:
    path_choice_supervisable: bool
    compliance_supervisable: bool
    timing_supervisable: bool
    decision_state: str
    drop_reason: Optional[str] = None


@dataclass
class TargetAgentAlignment:
    decision_agent_is_modeled: bool
    modeled_agent_index: Optional[int]
    target_is_trainable: bool


@dataclass
class InterventionContext:
    sdc_id: str
    traffic_light_id: str
    stop_point_xy: Tuple[float, float]
    approach_heading: float
    signal_state_at_decision: Optional[str]
    objects_of_interest: List[str] = field(default_factory=list)
    conflict_agents: List[ConflictAgentRef] = field(default_factory=list)


@dataclass
class TerminalPose:
    x: float
    y: float
    heading: float


@dataclass
class GroundTruthDecision:
    branch_id: str
    branch_label: str
    terminal_pose: TerminalPose
    crossed_stop_region: bool
    compliance_label: str
    entry_timing: Optional[str] = None
    signal_state_at_crossing: Optional[str] = None


@dataclass
class AlternativeDecision:
    branch_id: str
    branch_label: str
    compliance_label: str
    entry_timing: Optional[str] = None
    rank: int = 0
    rationale: str = ""


@dataclass
class RecoveredDecision:
    branch_id: Optional[str]
    branch_label: Optional[str]
    terminal_pose: Optional[TerminalPose]
    crossed_stop_region: bool
    compliance_label: Optional[str]
    entry_timing: Optional[str] = None
    signal_state_at_crossing: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class SupervisedDecision:
    branch_id: Optional[str]
    branch_label: Optional[str]
    terminal_pose: Optional[TerminalPose]
    compliance_label: Optional[str]
    entry_timing: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class LocalInterventionV1:
    scenario_id: str
    agent_id: str
    decision_time_idx: int
    window: WindowSpec
    context: InterventionContext
    gt_decision: GroundTruthDecision
    alternatives: List[AlternativeDecision]
    signal_qc: Dict[str, Any]
    provenance: Optional[ArtifactProvenance] = None
    commitment_metrics: Optional[CommitmentMetrics] = None
    supervision_gates: Optional[SupervisionGates] = None
    target_agent_alignment: Optional[TargetAgentAlignment] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    contract_name: str = LOCAL_INTERVENTION_CONTRACT_NAME
    schema_version: str = LOCAL_INTERVENTION_SCHEMA_VERSION
    contract_version: str = LOCAL_INTERVENTION_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class LocalInterventionRawV1:
    scenario_id: str
    agent_id: str
    decision_time_idx: int
    window: WindowSpec
    context: InterventionContext
    signal_qc: Dict[str, Any]
    recovered_decision: RecoveredDecision
    alternatives: List[AlternativeDecision]
    provenance: Optional[ArtifactProvenance] = None
    commitment_metrics: Optional[CommitmentMetrics] = None
    debug: Dict[str, Any] = field(default_factory=dict)
    view_type: str = "raw"
    contract_name: str = LOCAL_INTERVENTION_RAW_CONTRACT_NAME
    schema_version: str = LOCAL_INTERVENTION_RAW_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass
class LocalInterventionTrainViewV1:
    scenario_id: str
    agent_id: str
    decision_time_idx: int
    window: WindowSpec
    context: InterventionContext
    signal_qc: Dict[str, Any]
    provenance: ArtifactProvenance
    commitment: CommitmentMetrics
    supervision: SupervisionGates
    target_alignment: TargetAgentAlignment
    control_available_at_current: bool
    target_is_trainable: bool
    conditioning_eligible: bool
    raw_recovered_decision: RecoveredDecision
    supervised_decision: SupervisedDecision
    alternatives: List[AlternativeDecision]
    debug: Dict[str, Any] = field(default_factory=dict)
    view_type: str = "train_view"
    contract_name: str = LOCAL_INTERVENTION_TRAIN_VIEW_CONTRACT_NAME
    schema_version: str = LOCAL_INTERVENTION_TRAIN_VIEW_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return _jsonify(asdict(self))


def compute_conditioning_eligible(
    *,
    provenance: Optional[ArtifactProvenance],
    supervision_gates: Optional[SupervisionGates],
    target_alignment: Optional[TargetAgentAlignment],
) -> bool:
    if provenance is None or supervision_gates is None or target_alignment is None:
        return False
    return bool(
        target_alignment.target_is_trainable
        and provenance.control_available_at_current
        and (
            supervision_gates.path_choice_supervisable
            or supervision_gates.compliance_supervisable
            or supervision_gates.timing_supervisable
        )
    )


def build_supervised_decision(
    *,
    recovered_decision: RecoveredDecision,
    supervision_gates: Optional[SupervisionGates],
) -> SupervisedDecision:
    path_choice_supervisable = bool(getattr(supervision_gates, "path_choice_supervisable", False))
    compliance_supervisable = bool(getattr(supervision_gates, "compliance_supervisable", False))
    timing_supervisable = bool(getattr(supervision_gates, "timing_supervisable", False))
    return SupervisedDecision(
        branch_id=(recovered_decision.branch_id if path_choice_supervisable else None),
        branch_label=(recovered_decision.branch_label if path_choice_supervisable else None),
        terminal_pose=(recovered_decision.terminal_pose if path_choice_supervisable else None),
        compliance_label=(recovered_decision.compliance_label if compliance_supervisable else None),
        entry_timing=(recovered_decision.entry_timing if timing_supervisable else None),
    )


def build_local_intervention_raw(
    *,
    scenario_id: str,
    agent_id: str,
    decision_time_idx: int,
    window: WindowSpec,
    context: InterventionContext,
    signal_qc: Dict[str, Any],
    recovered_decision: RecoveredDecision,
    alternatives: List[AlternativeDecision],
    provenance: Optional[ArtifactProvenance] = None,
    commitment_metrics: Optional[CommitmentMetrics] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> LocalInterventionRawV1:
    return LocalInterventionRawV1(
        scenario_id=scenario_id,
        agent_id=agent_id,
        decision_time_idx=int(decision_time_idx),
        window=window,
        context=context,
        signal_qc=dict(signal_qc),
        recovered_decision=recovered_decision,
        alternatives=list(alternatives),
        provenance=provenance,
        commitment_metrics=commitment_metrics,
        debug={} if debug is None else dict(debug),
    )


def build_local_intervention_train_view(
    *,
    scenario_id: str,
    agent_id: str,
    decision_time_idx: int,
    window: WindowSpec,
    context: InterventionContext,
    signal_qc: Dict[str, Any],
    provenance: ArtifactProvenance,
    commitment: CommitmentMetrics,
    supervision: SupervisionGates,
    target_alignment: TargetAgentAlignment,
    raw_recovered_decision: RecoveredDecision,
    alternatives: List[AlternativeDecision],
    debug: Optional[Dict[str, Any]] = None,
) -> LocalInterventionTrainViewV1:
    conditioning_eligible = compute_conditioning_eligible(
        provenance=provenance,
        supervision_gates=supervision,
        target_alignment=target_alignment,
    )
    return LocalInterventionTrainViewV1(
        scenario_id=scenario_id,
        agent_id=agent_id,
        decision_time_idx=int(decision_time_idx),
        window=window,
        context=context,
        signal_qc=dict(signal_qc),
        provenance=provenance,
        commitment=commitment,
        supervision=supervision,
        target_alignment=target_alignment,
        control_available_at_current=bool(provenance.control_available_at_current),
        target_is_trainable=bool(target_alignment.target_is_trainable),
        conditioning_eligible=conditioning_eligible,
        raw_recovered_decision=raw_recovered_decision,
        supervised_decision=build_supervised_decision(
            recovered_decision=raw_recovered_decision,
            supervision_gates=supervision,
        ),
        alternatives=list(alternatives),
        debug={} if debug is None else dict(debug),
    )


def validate_local_intervention(intervention: LocalInterventionV1) -> List[str]:
    payload = intervention.to_dict()
    errors: List[str] = []
    if str(payload.get("contract_name")) != LOCAL_INTERVENTION_CONTRACT_NAME:
        errors.append("invalid_contract_name")
    if str(payload.get("schema_version")) != LOCAL_INTERVENTION_SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    if str(payload.get("contract_version")) != LOCAL_INTERVENTION_CONTRACT_VERSION:
        errors.append("invalid_contract_version")

    required_top = ("scenario_id", "agent_id", "decision_time_idx", "window", "context", "gt_decision", "alternatives", "signal_qc", "debug")
    for key in required_top:
        if key not in payload:
            errors.append(f"missing_{key}")

    context = payload.get("context", {})
    for key in ("sdc_id", "traffic_light_id", "stop_point_xy", "approach_heading", "signal_state_at_decision", "objects_of_interest", "conflict_agents"):
        if key not in context:
            errors.append(f"missing_context_{key}")

    gt = payload.get("gt_decision", {})
    for key in ("branch_id", "branch_label", "terminal_pose", "crossed_stop_region", "compliance_label"):
        if key not in gt:
            errors.append(f"missing_gt_decision_{key}")

    if str(gt.get("branch_label")) not in {"left", "straight", "right", "u_turn"}:
        errors.append("invalid_gt_branch_label")
    if str(gt.get("compliance_label")) not in {"obey_signal", "red_light_violation"}:
        errors.append("invalid_gt_compliance_label")

    window = payload.get("window", {})
    if int(window.get("end_idx", -1)) < int(window.get("start_idx", 0)):
        errors.append("invalid_window_range")

    for idx, alt in enumerate(payload.get("alternatives", [])):
        if str(alt.get("branch_label", "")) not in {"left", "straight", "right"}:
            errors.append(f"alternative_{idx}_invalid_branch_label")
        if str(alt.get("compliance_label", "")) not in {"obey_signal", "red_light_violation"}:
            errors.append(f"alternative_{idx}_invalid_compliance_label")
        entry_timing = alt.get("entry_timing")
        if entry_timing is not None and str(entry_timing) not in {"before_conflict", "after_conflict"}:
            errors.append(f"alternative_{idx}_invalid_entry_timing")

    supervision = payload.get("supervision_gates")
    if supervision is not None and str(supervision.get("decision_state", "")) not in {"waiting", "creeping", "committed"}:
        errors.append("invalid_supervision_decision_state")
    return errors


def build_alternative_decisions(
    *,
    branch_candidates: List[Dict[str, Any]],
    gt_branch_id: str,
    gt_branch_label: str,
    gt_compliance_label: str,
    conflict_agents: List[ConflictAgentRef],
    signal_state_at_decision: Optional[str],
) -> List[AlternativeDecision]:
    feasible_branches = [candidate for candidate in branch_candidates if str(candidate.get("branch_label")) in {"left", "straight", "right"}]
    compliance_options = ["obey_signal"]
    signal_text = "" if signal_state_at_decision is None else str(signal_state_at_decision).upper()
    if "STOP" in signal_text or "RED" in signal_text:
        compliance_options.append("red_light_violation")

    out: List[AlternativeDecision] = []
    rank = 0
    for candidate in feasible_branches:
        branch_id = str(candidate.get("branch_id"))
        branch_label = str(candidate.get("branch_label"))
        timings = [None]
        if conflict_agents:
            timings = ["before_conflict", "after_conflict"]
        for compliance_label in compliance_options:
            for entry_timing in timings:
                if branch_id == gt_branch_id and branch_label == gt_branch_label and compliance_label == gt_compliance_label and entry_timing is None:
                    continue
                out.append(
                    AlternativeDecision(
                        branch_id=branch_id,
                        branch_label=branch_label,
                        compliance_label=compliance_label,
                        entry_timing=entry_timing,
                        rank=rank,
                        rationale="geometry_feasible_counterfactual",
                    )
                )
                rank += 1
    return out
