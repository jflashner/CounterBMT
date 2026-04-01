from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    legacy_src = repo_root / "src" / "Adv-BMT"
    for path in (repo_root, legacy_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

from bmt.counterfactual import (
    ArtifactProvenance,
    CommitmentMetrics,
    ConflictAgentRef,
    InterventionContext,
    RecoveredDecision,
    SignalizedCandidateWindow,
    SupervisionGates,
    TargetAgentAlignment,
    TerminalPose,
    WindowSpec,
    analyze_conflicts,
    build_alternative_decisions,
    build_local_intervention_raw,
    build_local_intervention_train_view,
    choose_decision_window,
    compile_alternative_control_codes_from_local_intervention,
    compile_control_code_from_local_intervention,
    extract_local_patch,
    get_forward_loss_track_ids,
    load_and_normalize_scenario,
    load_motion_config,
    load_raw_scenario,
    local_intervention_to_bayesian_dag,
    recover_ground_truth_branch,
    select_signalized_candidates_for_scenario,
    summarize_forward_supervision_for_raw_scenario,
    write_signal_qc_artifacts_for_candidate,
)
from bmt.counterfactual.visualize import render_branch_candidates, render_conflict_plot, render_local_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine local_intervention_v1 artifacts from indexed signalized scenarios.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario-pkl", type=str, default="")
    source.add_argument("--signalized-index-jsonl", type=str, default="")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--light-id", type=str, default="")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--config", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = _load_candidates(args)
    if not candidates:
        print(json.dumps({"error": "no_candidates_found"}, indent=2))
        return 1

    config = load_motion_config(config_path=args.config or None)
    output_root = Path(args.outdir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    mined_records: List[Dict[str, Any]] = []
    deduped: Dict[tuple[str, str, int], Dict[str, Any]] = {}
    for candidate in candidates[: max(0, int(args.max_candidates))]:
        for record in _mine_candidate_for_trainable_agents(candidate, outdir=output_root, config=config):
            key = (str(record["scenario_id"]), str(record["agent_id"]), int(record["decision_time_idx"]))
            current = deduped.get(key)
            if current is None or tuple(record["score_key"]) < tuple(current["score_key"]):
                deduped[key] = record
            mined_records.append(record)

    kept_records = list(sorted(deduped.values(), key=lambda item: (item["scenario_id"], item["agent_id"], item["decision_time_idx"])))
    control_index_path = output_root / "control_index.jsonl"
    control_index_path.write_text(
        "".join(
            json.dumps(
                {
                    "scenario_id": item["scenario_id"],
                    "agent_id": item["agent_id"],
                    "decision_time_idx": item["decision_time_idx"],
                    "train_view_path": item["train_view_path"],
                    "factual_control_code_path": item["factual_control_code_path"],
                    "alternative_control_codes_path": item["alternative_control_codes_path"],
                },
                sort_keys=True,
            )
            + "\n"
            for item in kept_records
        ),
        encoding="utf-8",
    )

    summary = {
        "outdir": str(output_root),
        "num_light_candidates": len(candidates[: max(0, int(args.max_candidates))]),
        "num_candidate_interventions": len(mined_records),
        "num_kept_interventions": len(kept_records),
        "control_index_jsonl": str(control_index_path),
    }
    (output_root / "mining_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_candidates(args: argparse.Namespace) -> List[SignalizedCandidateWindow]:
    if args.scenario_pkl:
        result = select_signalized_candidates_for_scenario(args.scenario_pkl)
        candidates = result.candidates
        if args.light_id:
            candidates = [candidate for candidate in candidates if candidate.light_id == str(args.light_id)]
        return candidates

    candidates: List[SignalizedCandidateWindow] = []
    with Path(args.signalized_index_jsonl).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if args.light_id and str(payload.get("light_id")) != str(args.light_id):
                continue
            candidates.append(SignalizedCandidateWindow(**payload))
    return candidates


def _mine_candidate_for_trainable_agents(
    candidate: SignalizedCandidateWindow,
    *,
    outdir: Path,
    config: Any,
    artifact_mode: str = "full",
    max_agents: int | None = None,
    include_pngs: bool = True,
) -> List[Dict[str, Any]]:
    raw_scenario = load_raw_scenario(candidate.scenario_pkl)
    canonical = load_and_normalize_scenario(candidate.scenario_pkl)
    light = canonical.traffic_lights[candidate.light_id]
    forward_summary = summarize_forward_supervision_for_raw_scenario(raw_scenario, config=config)
    records: List[Dict[str, Any]] = []
    considered_agents = list(forward_summary.trainable_track_ids)
    if max_agents is not None and int(max_agents) > 0:
        considered_agents = considered_agents[: int(max_agents)]

    for agent_id in considered_agents:
        if agent_id not in canonical.tracks:
            continue
        track = canonical.tracks[agent_id]
        if str(track.object_type) != "VEHICLE":
            continue
        if not _track_comes_within_stop_point(track, candidate.stop_point_xy, threshold_m=35.0):
            continue
        record = _mine_one_agent_candidate(
            candidate,
            canonical=canonical,
            raw_scenario=raw_scenario,
            light=light,
            agent_id=agent_id,
            forward_summary=forward_summary,
            outdir=outdir,
            artifact_mode=artifact_mode,
            include_pngs=include_pngs,
        )
        if record is not None:
            records.append(record)
    return records


def _mine_one_agent_candidate(
    candidate: SignalizedCandidateWindow,
    *,
    canonical: Any,
    raw_scenario: Dict[str, Any],
    light: Any,
    agent_id: str,
    forward_summary: Any,
    outdir: Path,
    artifact_mode: str = "full",
    include_pngs: bool = True,
) -> Optional[Dict[str, Any]]:
    agent_role = "sdc" if str(agent_id) == str(canonical.sdc_id) else "forward_loss_vehicle"
    try:
        decision_window = choose_decision_window(
            canonical,
            agent_id=agent_id,
            agent_role=agent_role,
            stop_point_xy=candidate.stop_point_xy,
        )
    except Exception:
        return None

    example_dir = (
        outdir
        / "examples"
        / candidate.scenario_id
        / f"agent_{agent_id}"
        / f"light_{candidate.light_id}"
        / f"t_{decision_window.decision_time_idx:03d}"
    )
    example_dir.mkdir(parents=True, exist_ok=True)
    write_full_artifacts = str(artifact_mode).strip().lower() != "index_minimal"
    signal_qc_artifacts = {}
    if write_full_artifacts:
        signal_qc_artifacts = write_signal_qc_artifacts_for_candidate(candidate, outdir=example_dir)

    local_patch = extract_local_patch(canonical, stop_point_xy=candidate.stop_point_xy, radius_m=30.0, time_index=decision_window.decision_time_idx)
    branch_candidates = _serialize_branches(
        recover_ground_truth_branch(
            canonical,
            decision_window=decision_window,
            branch_candidates=_enumerate_branches(local_patch, candidate.stop_point_xy, decision_window.approach_heading),
            agent_id=agent_id,
        )
    )
    gt_recovery = branch_candidates["gt_recovery"]
    branch_list = branch_candidates["branch_candidates"]
    provenance = _coerce_dataclass_payload(gt_recovery.get("provenance"), ArtifactProvenance)
    commitment_metrics = _coerce_dataclass_payload(gt_recovery.get("commitment_metrics"), CommitmentMetrics)
    if provenance is None or commitment_metrics is None:
        return None

    conflict_result = analyze_conflicts(
        canonical,
        agent_id=agent_id,
        stop_point_xy=candidate.stop_point_xy,
        decision_time_idx=decision_window.decision_time_idx,
    )
    signal_state_at_crossing = None
    if decision_window.cross_time_idx is not None and decision_window.cross_time_idx < len(light.object_state):
        signal_state_at_crossing = light.object_state[decision_window.cross_time_idx]
    compliance_label = _derive_compliance_label(signal_state_at_crossing or candidate.signal_state_at_time, decision_window.crossed_stop_region)
    entry_timing = _derive_entry_timing(conflict_result)
    target_agent_alignment = _infer_target_agent_alignment(forward_summary=forward_summary, agent_id=agent_id)
    supervision_gates = _build_supervision_gates(
        canonical=canonical,
        candidate=candidate,
        conflict_result=conflict_result,
        provenance=provenance,
        commitment_metrics=commitment_metrics,
        agent_id=agent_id,
        target_agent_alignment=target_agent_alignment,
    )

    context = InterventionContext(
        sdc_id=canonical.sdc_id,
        traffic_light_id=candidate.light_id,
        stop_point_xy=candidate.stop_point_xy,
        approach_heading=float(decision_window.approach_heading),
        signal_state_at_decision=candidate.signal_state_at_time,
        objects_of_interest=list(canonical.objects_of_interest),
        conflict_agents=[
            ConflictAgentRef(track_id=record.track_id, eta_s=record.eta_s, eta_gap_s=record.eta_gap_s)
            for record in conflict_result.conflict_agents
        ],
    )
    recovered_decision = RecoveredDecision(
        branch_id=str(gt_recovery["branch_id"]),
        branch_label=str(gt_recovery["branch_label"]),
        terminal_pose=TerminalPose(
            x=float(gt_recovery["terminal_pose"]["x"]),
            y=float(gt_recovery["terminal_pose"]["y"]),
            heading=float(gt_recovery["terminal_pose"]["heading"]),
        ),
        crossed_stop_region=bool(gt_recovery["crossed_stop_region"]),
        compliance_label=compliance_label,
        entry_timing=entry_timing,
        signal_state_at_crossing=signal_state_at_crossing,
    )
    alternatives = build_alternative_decisions(
        branch_candidates=branch_list,
        gt_branch_id=str(recovered_decision.branch_id or ""),
        gt_branch_label=str(recovered_decision.branch_label or ""),
        gt_compliance_label=str(recovered_decision.compliance_label or "obey_signal"),
        conflict_agents=context.conflict_agents,
        signal_state_at_decision=candidate.signal_state_at_time,
    )
    debug_payload = {
        "candidate": candidate.to_dict(),
        "decision_window": decision_window.to_dict(),
        "gt_branch_recovery": gt_recovery,
        "branch_candidates": branch_list,
        "target_agent_alignment": _maybe_to_dict(target_agent_alignment),
        "supervision_gates": _maybe_to_dict(supervision_gates),
        "num_lane_features": len(local_patch.lane_features),
        "num_branch_candidates": len(branch_list),
        "num_conflict_agents": len(conflict_result.conflict_agents),
        "current_track_state": _current_track_state(canonical, stop_point_xy=candidate.stop_point_xy, agent_id=agent_id),
        "forward_supervision": forward_summary.to_dict(),
    }
    raw_intervention = build_local_intervention_raw(
        scenario_id=candidate.scenario_id,
        agent_id=agent_id,
        decision_time_idx=decision_window.decision_time_idx,
        window=WindowSpec(start_idx=decision_window.window_start_idx, end_idx=decision_window.window_end_idx),
        context=context,
        signal_qc=dict(candidate.signal_qc),
        recovered_decision=recovered_decision,
        alternatives=alternatives,
        provenance=provenance,
        commitment_metrics=commitment_metrics,
        debug=debug_payload,
    )
    train_view = build_local_intervention_train_view(
        scenario_id=candidate.scenario_id,
        agent_id=agent_id,
        decision_time_idx=decision_window.decision_time_idx,
        window=WindowSpec(start_idx=decision_window.window_start_idx, end_idx=decision_window.window_end_idx),
        context=context,
        signal_qc=dict(candidate.signal_qc),
        provenance=provenance,
        commitment=commitment_metrics,
        supervision=supervision_gates,
        target_alignment=target_agent_alignment,
        raw_recovered_decision=recovered_decision,
        alternatives=alternatives,
        debug=debug_payload,
    )

    dag = local_intervention_to_bayesian_dag(train_view.to_dict())
    factual_control_code = compile_control_code_from_local_intervention(
        train_view.to_dict(),
        canonical=canonical,
        source_path=str(example_dir / "local_intervention_train_view.json"),
    )
    alternative_control_codes = compile_alternative_control_codes_from_local_intervention(
        train_view.to_dict(),
        canonical=canonical,
        source_path=str(example_dir / "local_intervention_train_view.json"),
    )

    local_patch_json = example_dir / "local_patch.json"
    local_patch_png = example_dir / "local_patch.png"
    decision_window_json = example_dir / "decision_window.json"
    branch_candidates_json = example_dir / "branch_candidates.json"
    branch_candidates_png = example_dir / "branch_candidates.png"
    conflict_agents_json = example_dir / "conflict_agents.json"
    eta_table_csv = example_dir / "eta_table.csv"
    conflict_plot_png = example_dir / "conflict_plot.png"
    raw_intervention_json = example_dir / "local_intervention_raw.json"
    train_view_json = example_dir / "local_intervention_train_view.json"
    compatibility_intervention_json = example_dir / "local_intervention.json"
    intervention_dag_json = example_dir / "local_intervention_dag.json"
    mining_report_json = example_dir / "mining_report.json"
    factual_control_json = example_dir / "factual_control_code.json"
    compatibility_control_json = example_dir / "control_code.json"
    alternative_control_json = example_dir / "alternative_control_codes.json"

    _write_json(
        branch_candidates_json,
        {
            **branch_candidates,
            "provenance": _maybe_to_dict(provenance),
            "commitment_metrics": _maybe_to_dict(commitment_metrics),
            "supervision_gates": _maybe_to_dict(supervision_gates),
            "target_agent_alignment": _maybe_to_dict(target_agent_alignment),
        },
    )
    _write_json(raw_intervention_json, raw_intervention.to_dict())
    _write_json(train_view_json, train_view.to_dict())
    _write_json(factual_control_json, factual_control_code.to_dict())
    _write_json(alternative_control_json, alternative_control_codes)
    if write_full_artifacts:
        _write_json(
            local_patch_json,
            {
                "provenance": _maybe_to_dict(provenance),
                "target_agent_alignment": _maybe_to_dict(target_agent_alignment),
                "local_patch": local_patch.to_dict(),
            },
        )
        if include_pngs:
            render_local_patch(
                stop_point_xy=candidate.stop_point_xy,
                radius_m=30.0,
                lane_features=[feature.to_dict() for feature in local_patch.lane_features],
                nearby_tracks=[track.to_dict() for track in local_patch.nearby_tracks],
                out_path=local_patch_png,
            )
        _write_json(
            decision_window_json,
            {
                **decision_window.to_dict(),
                "target_agent_alignment": _maybe_to_dict(target_agent_alignment),
            },
        )
        target_track = canonical.tracks[agent_id]
        past_xy = _slice_valid_track_xy(target_track.position_xy, target_track.valid, 0, canonical.current_time_index)
        future_xy = _slice_valid_track_xy(target_track.position_xy, target_track.valid, canonical.current_time_index, target_track.position_xy.shape[0] - 1)
        decision_xy = tuple(float(v) for v in target_track.position_xy[decision_window.decision_time_idx])
        current_xy = tuple(float(v) for v in target_track.position_xy[canonical.current_time_index])
        current_heading = float(target_track.heading[canonical.current_time_index]) if np.isfinite(target_track.heading[canonical.current_time_index]) else float(decision_window.approach_heading)
        if include_pngs:
            render_branch_candidates(
                stop_point_xy=candidate.stop_point_xy,
                lane_features=[feature.to_dict() for feature in local_patch.lane_features],
                branch_candidates=branch_list,
                sdc_past_xy=past_xy,
                sdc_future_xy=future_xy,
                current_xy=current_xy,
                decision_xy=decision_xy,
                approach_heading=float(decision_window.approach_heading),
                current_heading=current_heading,
                current_time_idx=int(canonical.current_time_index),
                decision_time_idx=int(decision_window.decision_time_idx),
                agent_id=str(agent_id),
                gt_branch_id=str(recovered_decision.branch_id or ""),
                out_path=branch_candidates_png,
            )
        _write_json(
            conflict_agents_json,
            {
                "provenance": _maybe_to_dict(provenance),
                "supervision_gates": _maybe_to_dict(supervision_gates),
                "conflict_agents": [record.to_dict() for record in conflict_result.conflict_agents],
            },
        )
        _write_eta_table_csv(eta_table_csv, conflict_result.eta_table)
        if include_pngs:
            render_conflict_plot(
                stop_point_xy=candidate.stop_point_xy,
                core_radius_m=conflict_result.core_radius_m,
                sdc_position_xy=decision_xy,
                eta_table=[record.to_dict() for record in conflict_result.eta_table],
                out_path=conflict_plot_png,
            )
        _write_json(compatibility_intervention_json, train_view.to_dict())
        _write_json(intervention_dag_json, dag.to_dict())
        _write_json(compatibility_control_json, factual_control_code.to_dict())
        _write_json(
            mining_report_json,
            {
                "scenario_id": candidate.scenario_id,
                "agent_id": agent_id,
                "light_id": candidate.light_id,
                "provenance": _maybe_to_dict(provenance),
                "commitment_metrics": _maybe_to_dict(commitment_metrics),
                "supervision": _maybe_to_dict(supervision_gates),
                "target_alignment": _maybe_to_dict(target_agent_alignment),
                "branch_recall_hit": gt_recovery["branch_recall_hit"],
                "recovered_from_existing_candidate": gt_recovery["recovered_from_existing_candidate"],
                "num_branch_candidates": len(branch_list),
                "num_conflict_agents": len(conflict_result.conflict_agents),
                "artifacts": {
                    "signal_qc": {key: str(value) for key, value in signal_qc_artifacts.items()},
                    "local_patch": str(local_patch_json),
                    "branch_candidates": str(branch_candidates_json),
                    "local_intervention_train_view": str(train_view_json),
                    "factual_control_code": str(factual_control_json),
                },
            },
        )
    score_key = _candidate_score_key(
        target_alignment=target_agent_alignment,
        train_view_payload=train_view.to_dict(),
        candidate=candidate,
    )
    return {
        "scenario_id": candidate.scenario_id,
        "agent_id": agent_id,
        "light_id": candidate.light_id,
        "decision_time_idx": int(decision_window.decision_time_idx),
        "artifact_dir": str(example_dir),
        "train_view_path": str(train_view_json),
        "factual_control_code_path": str(factual_control_json),
        "alternative_control_codes_path": str(alternative_control_json),
        "score_key": list(score_key),
    }


def materialize_candidate_debug_bundle(
    *,
    scenario_pkl: str | Path,
    light_id: str,
    agent_id: str,
    outdir: str | Path,
    config: Any,
    include_pngs: bool = True,
) -> Optional[Dict[str, Any]]:
    scenario_pkl = str(Path(scenario_pkl).expanduser())
    candidates = select_signalized_candidates_for_scenario(scenario_pkl).candidates
    selected_candidate = None
    for candidate in candidates:
        if str(candidate.light_id) == str(light_id):
            selected_candidate = candidate
            break
    if selected_candidate is None:
        return None

    raw_scenario = load_raw_scenario(scenario_pkl)
    canonical = load_and_normalize_scenario(scenario_pkl)
    if str(light_id) not in canonical.traffic_lights:
        return None
    light = canonical.traffic_lights[str(light_id)]
    forward_summary = summarize_forward_supervision_for_raw_scenario(raw_scenario, config=config)
    return _mine_one_agent_candidate(
        selected_candidate,
        canonical=canonical,
        raw_scenario=raw_scenario,
        light=light,
        agent_id=str(agent_id),
        forward_summary=forward_summary,
        outdir=Path(outdir).expanduser(),
        artifact_mode="full",
        include_pngs=include_pngs,
    )


def _enumerate_branches(local_patch: Any, stop_point_xy: tuple[float, float], approach_heading: float) -> List[Any]:
    from bmt.counterfactual import enumerate_branch_candidates

    return enumerate_branch_candidates(local_patch.lane_features, stop_point_xy=stop_point_xy, approach_heading=approach_heading)


def _serialize_branches(recovery_output: tuple[Any, List[Any]]) -> Dict[str, Any]:
    gt_recovery, branch_candidates = recovery_output
    return {
        "gt_recovery": gt_recovery.to_dict(),
        "branch_candidates": [branch.to_dict() for branch in branch_candidates],
    }


def _derive_compliance_label(signal_state_at_crossing: Optional[str], crossed_stop_region: bool) -> str:
    text = "" if signal_state_at_crossing is None else str(signal_state_at_crossing).upper()
    if crossed_stop_region and ("STOP" in text or "RED" in text):
        return "red_light_violation"
    return "obey_signal"


def _derive_entry_timing(conflict_result: Any) -> Optional[str]:
    reference_eta = conflict_result.target_eta_s if getattr(conflict_result, "target_eta_s", None) is not None else getattr(conflict_result, "sdc_eta_s", None)
    if not conflict_result.conflict_agents or reference_eta is None:
        return None
    earliest_conflict = min(
        (record.eta_s for record in conflict_result.conflict_agents if record.eta_s is not None),
        default=None,
    )
    if earliest_conflict is None:
        return None
    return "before_conflict" if float(reference_eta) <= float(earliest_conflict) else "after_conflict"


def _infer_target_agent_alignment(*, forward_summary: Any, agent_id: str) -> TargetAgentAlignment:
    modeled_agent_index = None
    for row in getattr(forward_summary, "agents", []):
        if str(row.raw_track_id) == str(agent_id):
            modeled_agent_index = int(row.model_agent_slot)
            break
    decision_agent_is_modeled = modeled_agent_index is not None
    return TargetAgentAlignment(
        decision_agent_is_modeled=decision_agent_is_modeled,
        modeled_agent_index=modeled_agent_index,
        target_is_trainable=bool(str(agent_id) in set(getattr(forward_summary, "trainable_track_ids", []))),
    )


def _build_supervision_gates(
    *,
    canonical: Any,
    candidate: SignalizedCandidateWindow,
    conflict_result: Any,
    provenance: Optional[ArtifactProvenance],
    commitment_metrics: Optional[CommitmentMetrics],
    agent_id: str,
    target_agent_alignment: TargetAgentAlignment,
) -> SupervisionGates:
    target_track = canonical.tracks[agent_id]
    current_idx = int(canonical.current_time_index)
    current_valid = current_idx < target_track.valid.shape[0] and bool(target_track.valid[current_idx]) and np.isfinite(target_track.position_xy[current_idx]).all()
    current_speed = float(np.linalg.norm(np.asarray(target_track.velocity_xy[current_idx], dtype=np.float32))) if current_valid else 0.0
    current_dist = float(np.linalg.norm(np.asarray(target_track.position_xy[current_idx], dtype=np.float32) - np.asarray(candidate.stop_point_xy, dtype=np.float32))) if current_valid else float("inf")

    decision_state = "waiting"
    if provenance is not None and provenance.branch_commit_index_global is not None and provenance.branch_commit_index_global <= provenance.current_time_index_global:
        decision_state = "committed"
    elif current_dist <= 6.0 and current_speed > 0.5:
        decision_state = "creeping"

    path_choice_supervisable = True
    path_drop_reason = None
    if provenance is None or commitment_metrics is None:
        path_choice_supervisable = False
        path_drop_reason = "missing_commitment_metrics"
    elif provenance.decision_time_index_global < provenance.current_time_index_global:
        path_choice_supervisable = False
        path_drop_reason = "decision_before_current"
    elif provenance.cross_time_index_global is None:
        path_choice_supervisable = False
        path_drop_reason = "never_crossed_stopline"
    elif provenance.cross_time_index_global < provenance.current_time_index_global:
        path_choice_supervisable = False
        path_drop_reason = "cross_before_current"
    elif commitment_metrics.signed_stopline_progress_m < 3.0:
        path_choice_supervisable = False
        path_drop_reason = "insufficient_stopline_progress"
    elif commitment_metrics.downstream_progress_along_branch_m < 8.0:
        path_choice_supervisable = False
        path_drop_reason = "insufficient_downstream_progress"
    elif not np.isfinite(commitment_metrics.branch_margin) or commitment_metrics.branch_margin < 0.75:
        path_choice_supervisable = False
        path_drop_reason = "ambiguous_branch_margin"
    elif commitment_metrics.final_heading_error_rad > np.deg2rad(35.0):
        path_choice_supervisable = False
        path_drop_reason = "final_heading_mismatch"
    elif not target_agent_alignment.target_is_trainable:
        path_choice_supervisable = False
        path_drop_reason = "target_not_trainable"

    signal_is_known = candidate.signal_state_at_time is not None and not bool(candidate.signal_qc.get("ambiguous_light_state", False))
    compliance_supervisable = bool(current_valid and signal_is_known and current_dist <= 35.0 and target_agent_alignment.target_is_trainable)
    timing_supervisable = bool(compliance_supervisable and len(conflict_result.conflict_agents) > 0)

    primary_drop_reason = path_drop_reason
    if primary_drop_reason is None and not compliance_supervisable:
        primary_drop_reason = "compliance_not_available_at_current"

    return SupervisionGates(
        path_choice_supervisable=bool(path_choice_supervisable),
        compliance_supervisable=bool(compliance_supervisable),
        timing_supervisable=bool(timing_supervisable),
        decision_state=decision_state,
        drop_reason=primary_drop_reason,
    )


def _current_track_state(canonical: Any, *, stop_point_xy: tuple[float, float], agent_id: str) -> Dict[str, Any]:
    target_track = canonical.tracks[agent_id]
    current_idx = int(canonical.current_time_index)
    if current_idx >= target_track.valid.shape[0] or not bool(target_track.valid[current_idx]) or not np.isfinite(target_track.position_xy[current_idx]).all():
        return {"valid": False, "current_time_index": current_idx}
    pos = np.asarray(target_track.position_xy[current_idx], dtype=np.float32)
    return {
        "valid": True,
        "current_time_index": current_idx,
        "agent_id": str(agent_id),
        "position_xy": [float(pos[0]), float(pos[1])],
        "speed_mps": float(np.linalg.norm(np.asarray(target_track.velocity_xy[current_idx], dtype=np.float32))),
        "distance_to_stop_point_m": float(np.linalg.norm(pos - np.asarray(stop_point_xy, dtype=np.float32))),
    }


def _track_comes_within_stop_point(track: Any, stop_point_xy: tuple[float, float], *, threshold_m: float) -> bool:
    valid_mask = np.asarray(track.valid, dtype=bool)
    position_xy = np.asarray(track.position_xy, dtype=np.float32)
    if position_xy.ndim != 2 or position_xy.shape[-1] < 2:
        return False
    finite_mask = np.isfinite(position_xy).all(axis=-1)
    valid_idx = valid_mask & finite_mask
    if not np.any(valid_idx):
        return False
    distances = np.linalg.norm(position_xy[valid_idx] - np.asarray(stop_point_xy, dtype=np.float32)[None, :], axis=-1)
    return bool(np.min(distances) <= float(threshold_m))


def _candidate_score_key(*, target_alignment: TargetAgentAlignment, train_view_payload: Dict[str, Any], candidate: SignalizedCandidateWindow) -> tuple[int, int, int, int, int, float]:
    supervision = train_view_payload.get("supervision", {})
    return (
        0 if bool(target_alignment.target_is_trainable) else 1,
        0 if bool(train_view_payload.get("control_available_at_current")) else 1,
        0 if bool(supervision.get("path_choice_supervisable")) else 1,
        0 if bool(supervision.get("compliance_supervisable")) else 1,
        0 if bool(supervision.get("timing_supervisable")) else 1,
        float(candidate.min_dist_stop_point_m),
    )


def _slice_valid_track_xy(position_xy: np.ndarray, valid_mask: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if position_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    start_idx = int(np.clip(int(start_idx), 0, max(0, position_xy.shape[0] - 1)))
    end_idx = int(np.clip(int(end_idx), start_idx, max(0, position_xy.shape[0] - 1)))
    idx = [t for t in range(start_idx, end_idx + 1) if bool(valid_mask[t]) and np.isfinite(position_xy[t]).all()]
    if not idx:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(position_xy[idx], dtype=np.float32)


def _coerce_dataclass_payload(value: Any, cls: Any) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    return value


def _maybe_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_eta_table_csv(path: Path, rows: Iterable[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "track_id",
                "object_type",
                "eta_s",
                "eta_gap_s",
                "current_distance_to_core_m",
                "current_speed_mps",
                "will_enter_core",
                "current_x",
                "current_y",
                "is_object_of_interest",
            ],
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(
                {
                    "track_id": record.track_id,
                    "object_type": record.object_type,
                    "eta_s": record.eta_s,
                    "eta_gap_s": record.eta_gap_s,
                    "current_distance_to_core_m": record.current_distance_to_core_m,
                    "current_speed_mps": record.current_speed_mps,
                    "will_enter_core": record.will_enter_core,
                    "current_x": record.current_position_xy[0],
                    "current_y": record.current_position_xy[1],
                    "is_object_of_interest": record.is_object_of_interest,
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
