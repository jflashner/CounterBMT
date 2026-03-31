from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "src" / "Adv-BMT"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from bmt.counterfactual import (
    ConflictAgentRef,
    GroundTruthDecision,
    InterventionContext,
    LocalInterventionV1,
    TerminalPose,
    WindowSpec,
    analyze_conflicts,
    build_alternative_decisions,
    choose_decision_window,
    enumerate_branch_candidates,
    extract_local_patch,
    load_and_normalize_scenario,
    local_intervention_to_bayesian_dag,
    recover_ground_truth_branch,
    select_signalized_candidates_for_scenario,
    validate_local_intervention,
)
from bmt.counterfactual.signal_qc import evaluate_signal_qc


def _synthetic_signalized_scenario() -> dict:
    t = 21
    ts = np.arange(t, dtype=np.float32) * 0.1
    sdc_y = np.linspace(-40.0, 10.0, t, dtype=np.float32)
    sdc_x = np.zeros((t,), dtype=np.float32)
    sdc_pos = np.stack([sdc_x, sdc_y, np.zeros((t,), dtype=np.float32)], axis=-1)
    sdc_vel = np.stack([np.zeros((t,), dtype=np.float32), np.full((t,), 5.0, dtype=np.float32)], axis=-1)

    conflict_x = np.linspace(12.0, -12.0, t, dtype=np.float32)
    conflict_y = np.full((t,), -5.0, dtype=np.float32)
    conflict_pos = np.stack([conflict_x, conflict_y, np.zeros((t,), dtype=np.float32)], axis=-1)
    conflict_vel = np.stack([np.full((t,), -3.0, dtype=np.float32), np.zeros((t,), dtype=np.float32)], axis=-1)

    return {
        "id": "scenario_synthetic_signalized",
        "length": t,
        "tracks": {
            0: {
                "type": "VEHICLE",
                "state": {
                    "position": sdc_pos,
                    "heading": np.full((t,), np.pi / 2.0, dtype=np.float32),
                    "velocity": sdc_vel,
                    "valid": np.ones((t,), dtype=bool),
                },
                "metadata": {"object_id": 0, "type": "VEHICLE"},
            },
            1: {
                "type": "VEHICLE",
                "state": {
                    "position": conflict_pos,
                    "heading": np.full((t,), np.pi, dtype=np.float32),
                    "velocity": conflict_vel,
                    "valid": np.ones((t,), dtype=bool),
                },
                "metadata": {"object_id": 1, "type": "VEHICLE"},
            },
        },
        "dynamic_map_states": {
            100: {
                "type": "TRAFFIC_LIGHT",
                "state": {
                    "object_state": ["LANE_STATE_STOP"] * 8 + ["LANE_STATE_GO"] * (t - 8),
                },
                "lane": 100,
                "stop_point": [0.0, -5.0, 0.0],
                "metadata": {"object_id": 100, "type": "TRAFFIC_LIGHT"},
            }
        },
        "map_features": {
            10: {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[0.0, -25.0, 0.0], [0.0, -5.0, 0.0], [0.0, 15.0, 0.0]], dtype=np.float32),
            },
            11: {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[0.0, -5.0, 0.0], [-10.0, 0.0, 0.0], [-20.0, 0.0, 0.0]], dtype=np.float32),
            },
            12: {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[0.0, -5.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
            },
        },
        "metadata": {
            "scenario_id": "scenario_synthetic_signalized",
            "sdc_id": 0,
            "current_time_index": 5,
            "ts": ts,
            "objects_of_interest": [1],
            "tracks_to_predict": {"0": {"track_id": "0"}},
        },
    }


class CounterfactualSignalAndLocalInterventionTests(unittest.TestCase):
    def test_signal_qc_flags_short_oscillation(self) -> None:
        qc = evaluate_signal_qc(
            ["LANE_STATE_STOP", "LANE_STATE_GO", "LANE_STATE_STOP", "LANE_STATE_STOP"],
            stop_point_present=True,
            reference_time_index=1,
        )
        self.assertTrue(qc.short_oscillation_flag)
        self.assertLess(qc.confidence_score, 1.0)

    def test_signalized_index_finds_candidate_on_synthetic_scene(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sd_synthetic.pkl"
            with path.open("wb") as f:
                pickle.dump(_synthetic_signalized_scenario(), f)
            result = select_signalized_candidates_for_scenario(path)
            self.assertEqual(result.primary_drop_reason, None)
            self.assertEqual(len(result.candidates), 1)
            candidate = result.candidates[0]
            self.assertEqual(candidate.light_id, "100")
            self.assertLess(candidate.min_dist_stop_point_m, 35.0)
            self.assertEqual(candidate.signal_state_at_time, "LANE_STATE_STOP")
            self.assertEqual(candidate.objects_of_interest_overlap, ["1"])

    def test_local_intervention_contract_and_dag_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sd_synthetic.pkl"
            with path.open("wb") as f:
                pickle.dump(_synthetic_signalized_scenario(), f)

            candidate = select_signalized_candidates_for_scenario(path).candidates[0]
            canonical = load_and_normalize_scenario(path)
            decision_window = choose_decision_window(canonical, stop_point_xy=candidate.stop_point_xy)
            local_patch = extract_local_patch(canonical, stop_point_xy=candidate.stop_point_xy, time_index=decision_window.decision_time_idx)
            branches = enumerate_branch_candidates(local_patch.lane_features, stop_point_xy=candidate.stop_point_xy, approach_heading=decision_window.approach_heading)
            gt_recovery, branch_candidates = recover_ground_truth_branch(canonical, decision_window=decision_window, branch_candidates=branches)
            conflict_result = analyze_conflicts(canonical, stop_point_xy=candidate.stop_point_xy, decision_time_idx=decision_window.decision_time_idx)

            context = InterventionContext(
                sdc_id=canonical.sdc_id,
                traffic_light_id=candidate.light_id,
                stop_point_xy=candidate.stop_point_xy,
                approach_heading=decision_window.approach_heading,
                signal_state_at_decision=candidate.signal_state_at_time,
                objects_of_interest=canonical.objects_of_interest,
                conflict_agents=[
                    ConflictAgentRef(track_id=record.track_id, eta_s=record.eta_s, eta_gap_s=record.eta_gap_s)
                    for record in conflict_result.conflict_agents
                ],
            )
            gt_decision = GroundTruthDecision(
                branch_id=gt_recovery.branch_id,
                branch_label=gt_recovery.branch_label,
                terminal_pose=TerminalPose(
                    x=gt_recovery.terminal_pose.x,
                    y=gt_recovery.terminal_pose.y,
                    heading=gt_recovery.terminal_pose.heading,
                ),
                crossed_stop_region=gt_recovery.crossed_stop_region,
                compliance_label="obey_signal",
                entry_timing="before_conflict" if conflict_result.conflict_agents else None,
                signal_state_at_crossing="LANE_STATE_GO",
            )
            alternatives = build_alternative_decisions(
                branch_candidates=[branch.to_dict() for branch in branch_candidates],
                gt_branch_id=gt_decision.branch_id,
                gt_branch_label=gt_decision.branch_label,
                gt_compliance_label=gt_decision.compliance_label,
                conflict_agents=context.conflict_agents,
                signal_state_at_decision=candidate.signal_state_at_time,
            )
            intervention = LocalInterventionV1(
                scenario_id=candidate.scenario_id,
                agent_id=canonical.sdc_id,
                decision_time_idx=decision_window.decision_time_idx,
                window=WindowSpec(start_idx=decision_window.window_start_idx, end_idx=decision_window.window_end_idx),
                context=context,
                gt_decision=gt_decision,
                alternatives=alternatives,
                signal_qc=candidate.signal_qc,
                debug={"num_branch_candidates": len(branch_candidates)},
            )

            self.assertEqual(validate_local_intervention(intervention), [])
            dag = local_intervention_to_bayesian_dag(intervention)
            self.assertEqual([node.node_id for node in dag.nodes], [
                "context/signal_state",
                "context/conflict_eta",
                "decision/path_choice",
                "decision/compliance",
                "decision/entry_timing",
                "outcome/stopline_crossing",
                "outcome/collision",
                "outcome/interaction_order",
            ])
            self.assertEqual(gt_recovery.branch_label, "straight")
            self.assertTrue(len(alternatives) > 0)

    def test_4245_regression_keeps_branches_but_disables_path_supervision(self) -> None:
        scenario_path = ROOT / "data" / "scenarionet_waymo_training_500" / "sd_waymo_v1.2_4245da4b159fa62c.pkl"
        if not scenario_path.is_file():
            self.skipTest(f"missing regression scenario: {scenario_path}")

        with tempfile.TemporaryDirectory() as td:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "counterfactual" / "mine_local_interventions.py"),
                "--scenario-pkl",
                str(scenario_path),
                "--light-id",
                "128",
                "--max-candidates",
                "1",
                "--outdir",
                td,
            ]
            subprocess.run(cmd, check=True, cwd=ROOT)

            train_view_files = sorted(Path(td).rglob("local_intervention_train_view.json"))
            self.assertTrue(train_view_files)
            saw_compliance_supervision = False
            for train_view_path in train_view_files:
                train_view_payload = json.loads(train_view_path.read_text(encoding="utf-8"))
                branch_payload = json.loads((train_view_path.parent / "branch_candidates.json").read_text(encoding="utf-8"))
                self.assertEqual(train_view_payload["view_type"], "train_view")
                self.assertGreaterEqual(len(branch_payload["branch_candidates"]), 3)
                if not bool(train_view_payload["control_available_at_current"]):
                    self.assertFalse(bool(train_view_payload["conditioning_eligible"]))
                    self.assertIsNone(train_view_payload["supervised_decision"]["branch_label"])
                if bool(train_view_payload["supervision"]["compliance_supervisable"]):
                    saw_compliance_supervision = True
            self.assertTrue(saw_compliance_supervision)


if __name__ == "__main__":
    unittest.main()
