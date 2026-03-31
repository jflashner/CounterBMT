from __future__ import annotations

import copy
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "src" / "Adv-BMT"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

if "hydra" not in sys.modules:
    hydra_stub = types.ModuleType("hydra")

    def _main(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    hydra_stub.main = _main
    sys.modules["hydra"] = hydra_stub

if "scenarionet" not in sys.modules:
    scenarionet_stub = types.ModuleType("scenarionet")

    def _read_dataset_summary(*args, **kwargs):
        raise RuntimeError("read_dataset_summary should not be called in this unit test")

    def _read_scenario(*args, **kwargs):
        raise RuntimeError("read_scenario should not be called in this unit test")

    scenarionet_stub.read_dataset_summary = _read_dataset_summary
    scenarionet_stub.read_scenario = _read_scenario
    sys.modules["scenarionet"] = scenarionet_stub

if "metadrive.scenario.scenario_description" not in sys.modules:
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
            return "CAUTION" in str(value).upper() or "YELLOW" in str(value).upper()

        @staticmethod
        def is_traffic_light_in_red(value):
            return "STOP" in str(value).upper() or "RED" in str(value).upper()

        @staticmethod
        def is_traffic_light_unknown(value):
            return value is None or "UNKNOWN" in str(value).upper()

    metadrive_sd_stub.ScenarioDescription = _ScenarioDescription
    metadrive_sd_stub.MetaDriveType = _MetaDriveType
    sys.modules["metadrive"] = metadrive_stub
    sys.modules["metadrive.scenario"] = metadrive_scenario_stub
    sys.modules["metadrive.scenario.scenario_description"] = metadrive_sd_stub

if "lightning" not in sys.modules:
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

from bmt.counterfactual import (
    COMPLIANCE_TOKEN_DIM,
    PATH_TOKEN_DIM,
    TERMINAL_ANCHOR_DIM,
    TIMING_TOKEN_DIM,
    ConflictAgentRef,
    CommitmentMetrics,
    GroundTruthDecision,
    InterventionContext,
    LocalInterventionV1,
    RecoveredDecision,
    SupervisionGates,
    TerminalPose,
    TargetAgentAlignment,
    WindowSpec,
    analyze_conflicts,
    build_alternative_decisions,
    build_counterfactual_dataset_fields,
    build_local_intervention_train_view,
    choose_decision_window,
    compile_control_code_from_local_intervention,
    default_counterfactual_dataset_fields,
    enumerate_branch_candidates,
    extract_local_patch,
    load_and_normalize_scenario,
    load_motion_config,
    recover_ground_truth_branch,
    select_signalized_candidates_for_scenario,
    summarize_forward_supervision_for_raw_scenario,
    validate_control_code,
)
from bmt.dataset.dataset import InfgenDataset
from scripts.counterfactual.build_path_control_index import _is_path_train_view_eligible


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
            "tracks_to_predict": {"0": {"track_id": "0", "track_index": 0}},
        },
    }


class CounterfactualControlCodeAndDatasetTests(unittest.TestCase):
    def test_compile_control_code_from_local_intervention(self) -> None:
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
            intervention = LocalInterventionV1(
                scenario_id=candidate.scenario_id,
                agent_id=canonical.sdc_id,
                decision_time_idx=decision_window.decision_time_idx,
                window=WindowSpec(start_idx=decision_window.window_start_idx, end_idx=decision_window.window_end_idx),
                context=context,
                gt_decision=gt_decision,
                alternatives=build_alternative_decisions(
                    branch_candidates=[branch.to_dict() for branch in branch_candidates],
                    gt_branch_id=gt_decision.branch_id,
                    gt_branch_label=gt_decision.branch_label,
                    gt_compliance_label=gt_decision.compliance_label,
                    conflict_agents=context.conflict_agents,
                    signal_state_at_decision=candidate.signal_state_at_time,
                ),
                signal_qc=candidate.signal_qc,
                debug={"candidate": candidate.to_dict()},
            )

            control_code = compile_control_code_from_local_intervention(intervention, canonical=canonical)
            payload = control_code.to_dict()
            self.assertEqual(validate_control_code(payload), [])
            self.assertEqual(payload["schema_version"], "counter_bmt_v3_control_code_v1")
            self.assertEqual(len(payload["sparse_time_mask"]), canonical.length)
            active = np.flatnonzero(np.asarray(payload["sparse_time_mask"], dtype=np.float32) > 0.0)
            self.assertEqual(int(active[0]), payload["window"]["start_idx"])
            self.assertEqual(int(active[-1]), payload["window"]["end_idx"])
            self.assertIn(payload["path_token"]["branch_label"], {"left", "straight", "right", "u_turn"})
            self.assertTrue(np.isfinite(payload["terminal_anchor"]["target_x_rel"]))

    def test_counterfactual_dataset_fields_encode_expected_masks(self) -> None:
        control_code = {
            "schema_version": "counter_bmt_v3_control_code_v1",
            "scenario_id": "scenario_a",
            "agent_id": "ego",
            "decision_time_idx": 2,
            "window": {"start_idx": 1, "end_idx": 4},
            "path_token": {
                "branch_label": "left",
                "branch_id": "branch_01",
                "target_terminal_pose": {
                    "x_rel": 8.0,
                    "y_rel": 2.5,
                    "heading_rel": 0.3,
                    "sin_heading_rel": float(np.sin(0.3)),
                    "cos_heading_rel": float(np.cos(0.3)),
                },
            },
            "compliance_token": {
                "signal_state": "LANE_STATE_STOP",
                "compliance_label": "obey_signal",
                "stop_point_xy": [4.0, 0.5],
            },
            "timing_token": {
                "conflict_agent_id": "veh_1",
                "delta_t_entry_s": 1.2,
                "timing_label": "before_conflict",
            },
            "terminal_anchor": {
                "target_x_rel": 8.0,
                "target_y_rel": 2.5,
                "target_sin_heading_rel": float(np.sin(0.3)),
                "target_cos_heading_rel": float(np.cos(0.3)),
            },
            "sparse_time_mask": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "debug": {"source": "unit_test"},
        }

        sample_a = {
            "encoder/map_feature": np.zeros((1, 1, 27), dtype=np.float32),
            "metadata/scenario_id": "scenario_a",
            **build_counterfactual_dataset_fields(
                scenario_id="scenario_a",
                decoder_track_names=np.array(["ego", "veh_1"], dtype=str),
                horizon=6,
                control_code=control_code,
                control_code_path="/tmp/control_code.json",
            ),
        }
        sample_b = {
            "encoder/map_feature": np.zeros((1, 1, 27), dtype=np.float32),
            "metadata/scenario_id": "scenario_b",
            **default_counterfactual_dataset_fields(
                scenario_id="scenario_b",
                decoder_track_names=np.array(["ego", "veh_2", "veh_3"], dtype=str),
                horizon=4,
            ),
        }
        self.assertEqual(tuple(sample_a["cf/path_token"].shape), (PATH_TOKEN_DIM,))
        self.assertEqual(tuple(sample_a["cf/compliance_token"].shape), (COMPLIANCE_TOKEN_DIM,))
        self.assertEqual(tuple(sample_a["cf/timing_token"].shape), (TIMING_TOKEN_DIM,))
        self.assertEqual(tuple(sample_a["cf/terminal_anchor"].shape), (TERMINAL_ANCHOR_DIM,))
        self.assertEqual(tuple(sample_a["cf/time_window_mask"].shape), (6,))
        self.assertEqual(sample_a["cf/decision_agent_mask"].tolist(), [1.0, 0.0])
        self.assertEqual(sample_a["cf/conditioning_eligible"], 1)
        self.assertTrue(sample_a["cf/debug_meta"]["available"])
        self.assertEqual(sample_b["cf/time_window_mask"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(sample_b["cf/decision_agent_mask"].tolist(), [0.0, 0.0, 0.0])
        self.assertFalse(sample_b["cf/debug_meta"]["available"])

    def test_aux_supervision_can_remain_when_conditioning_is_disabled(self) -> None:
        control_code = {
            "schema_version": "counter_bmt_v3_control_code_v1",
            "scenario_id": "scenario_a",
            "agent_id": "ego",
            "decision_time_idx": 2,
            "window": {"start_idx": 1, "end_idx": 4},
            "path_token": {
                "branch_label": "none",
                "branch_id": "",
                "target_terminal_pose": {
                    "x_rel": 0.0,
                    "y_rel": 0.0,
                    "heading_rel": 0.0,
                    "sin_heading_rel": 0.0,
                    "cos_heading_rel": 1.0,
                },
            },
            "compliance_token": {
                "signal_state": "LANE_STATE_STOP",
                "compliance_label": "obey_signal",
                "stop_point_xy": [4.0, 0.5],
            },
            "timing_token": {
                "conflict_agent_id": None,
                "delta_t_entry_s": None,
                "timing_label": None,
            },
            "terminal_anchor": {
                "target_x_rel": 0.0,
                "target_y_rel": 0.0,
                "target_sin_heading_rel": 0.0,
                "target_cos_heading_rel": 1.0,
            },
            "sparse_time_mask": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "debug": {
                "control_available_at_current": False,
                "source_target_agent_alignment": {
                    "decision_agent_is_modeled": True,
                    "modeled_agent_index": 0,
                    "target_is_trainable": True,
                },
                "source_supervision_gates": {
                    "path_choice_supervisable": False,
                    "compliance_supervisable": True,
                    "timing_supervisable": False,
                    "decision_state": "waiting",
                },
            },
        }

        sample = build_counterfactual_dataset_fields(
            scenario_id="scenario_a",
            decoder_track_names=np.array(["ego", "veh_1"], dtype=str),
            horizon=6,
            control_code=control_code,
            control_code_path="/tmp/factual_control_code.json",
        )
        self.assertEqual(sample["cf/conditioning_eligible"], 0)
        self.assertEqual(sample["cf/control_available"], 0)
        self.assertEqual(sample["cf/compliance_supervision_mask"], 1)
        self.assertEqual(sample["cf/path_supervision_mask"], 0)
        self.assertEqual(sample["cf/decision_agent_mask"].tolist(), [1.0, 0.0])
        self.assertEqual(sample["cf/time_window_mask"].tolist(), [0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        self.assertFalse(sample["cf/debug_meta"]["available"])
        self.assertTrue(sample["cf/debug_meta"]["auxiliary_supervision_eligible"])
        self.assertEqual(sample["cf/debug_meta"]["drop_reason"], "control_unavailable_at_current")

    def test_counterfactual_dataset_fields_drop_non_trainable_targets(self) -> None:
        control_code = {
            "schema_version": "counter_bmt_v3_control_code_v1",
            "scenario_id": "scenario_a",
            "agent_id": "ego",
            "decision_time_idx": 2,
            "window": {"start_idx": 1, "end_idx": 4},
            "path_token": {
                "branch_label": "straight",
                "branch_id": "branch_03",
                "target_terminal_pose": {
                    "x_rel": 8.0,
                    "y_rel": 0.0,
                    "heading_rel": 0.0,
                    "sin_heading_rel": 0.0,
                    "cos_heading_rel": 1.0,
                },
            },
            "compliance_token": {
                "signal_state": "LANE_STATE_STOP",
                "compliance_label": "obey_signal",
                "stop_point_xy": [4.0, 0.5],
            },
            "timing_token": {
                "conflict_agent_id": None,
                "delta_t_entry_s": None,
                "timing_label": None,
            },
            "terminal_anchor": {
                "target_x_rel": 8.0,
                "target_y_rel": 0.0,
                "target_sin_heading_rel": 0.0,
                "target_cos_heading_rel": 1.0,
            },
            "sparse_time_mask": [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "debug": {
                "source_target_agent_alignment": {
                    "decision_agent_is_modeled": False,
                    "modeled_agent_index": None,
                    "target_is_trainable": False,
                },
                "source_supervision_gates": {
                    "path_choice_supervisable": False,
                    "compliance_supervisable": True,
                    "timing_supervisable": False,
                    "decision_state": "waiting",
                },
            },
        }

        sample = build_counterfactual_dataset_fields(
            scenario_id="scenario_a",
            decoder_track_names=np.array(["ego", "veh_1"], dtype=str),
            horizon=6,
            control_code=control_code,
            control_code_path="/tmp/factual_control_code.json",
            require_trainable=True,
        )
        self.assertFalse(sample["cf/debug_meta"]["available"])
        self.assertEqual(sample["cf/debug_meta"]["drop_reason"], "non_trainable_target")
        self.assertEqual(sample["cf/path_token"].tolist(), [0.0] * PATH_TOKEN_DIM)
        self.assertEqual(sample["cf/decision_agent_mask"].tolist(), [0.0, 0.0])

    def test_forward_supervision_summary_uses_target_action_mask(self) -> None:
        config = load_motion_config()
        scenario = _synthetic_signalized_scenario()
        summary = summarize_forward_supervision_for_raw_scenario(scenario, config=config)
        self.assertEqual(summary.scenario_id, "scenario_synthetic_signalized")
        self.assertEqual(summary.sdc_id, "0")
        self.assertIn("0", summary.trainable_track_ids)
        self.assertTrue(summary.sdc_receives_forward_loss)
        self.assertTrue(any(agent.receives_motion_loss for agent in summary.agents))

    def test_train_view_control_code_nulls_unsupervisable_path(self) -> None:
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
            recovered = RecoveredDecision(
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
            train_view = build_local_intervention_train_view(
                scenario_id=candidate.scenario_id,
                agent_id=canonical.sdc_id,
                decision_time_idx=decision_window.decision_time_idx,
                window=WindowSpec(start_idx=decision_window.window_start_idx, end_idx=decision_window.window_end_idx),
                context=context,
                signal_qc=candidate.signal_qc,
                provenance=gt_recovery.provenance,
                commitment=gt_recovery.commitment_metrics,
                supervision=SupervisionGates(
                    path_choice_supervisable=False,
                    compliance_supervisable=True,
                    timing_supervisable=False,
                    decision_state="waiting",
                    drop_reason="unit_test",
                ),
                target_alignment=TargetAgentAlignment(
                    decision_agent_is_modeled=True,
                    modeled_agent_index=0,
                    target_is_trainable=True,
                ),
                raw_recovered_decision=recovered,
                alternatives=build_alternative_decisions(
                    branch_candidates=[branch.to_dict() for branch in branch_candidates],
                    gt_branch_id=recovered.branch_id or "",
                    gt_branch_label=recovered.branch_label or "",
                    gt_compliance_label=recovered.compliance_label or "obey_signal",
                    conflict_agents=context.conflict_agents,
                    signal_state_at_decision=candidate.signal_state_at_time,
                ),
                debug={"candidate": candidate.to_dict()},
            )
            control_code = compile_control_code_from_local_intervention(train_view.to_dict(), canonical=canonical)
            payload = control_code.to_dict()
            train_view_payload = train_view.to_dict()
            self.assertEqual(train_view.to_dict()["view_type"], "train_view")
            self.assertEqual(
                train_view_payload["conditioning_eligible"],
                bool(train_view_payload["control_available_at_current"] and train_view_payload["supervision"]["compliance_supervisable"]),
            )
            self.assertEqual(payload["path_token"]["branch_label"], "none")
            self.assertEqual(payload["terminal_anchor"]["target_x_rel"], 0.0)
            self.assertEqual(payload["compliance_token"]["compliance_label"], "obey_signal")

    def test_path_only_mode_zeroes_compliance_and_timing(self) -> None:
        dataset = InfgenDataset.__new__(InfgenDataset)
        dataset.counterfactual_mode = "path_only"
        sample = {
            "cf/path_token": np.asarray([2.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "cf/compliance_token": np.asarray([1.0, 1.0, 3.0, 4.0], dtype=np.float32),
            "cf/timing_token": np.asarray([1.0, 0.5, 1.0], dtype=np.float32),
            "cf/path_supervision_mask": 1,
            "cf/compliance_supervision_mask": 1,
            "cf/timing_supervision_mask": 1,
            "cf/conditioning_eligible": 1,
            "cf/control_available": 1,
            "cf/debug_meta": {"available": True},
        }
        result = InfgenDataset._apply_counterfactual_mode(dataset, sample)
        self.assertEqual(result["cf/compliance_token"].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["cf/timing_token"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(result["cf/compliance_supervision_mask"], 0)
        self.assertEqual(result["cf/timing_supervision_mask"], 0)
        self.assertEqual(result["cf/conditioning_eligible"], 1)
        self.assertEqual(result["cf/debug_meta"]["counterfactual_mode"], "path_only")

    def test_path_control_filter_rejects_compliance_only_case(self) -> None:
        train_view = {
            "conditioning_eligible": True,
            "target_is_trainable": True,
            "control_available_at_current": True,
            "supervision": {"path_choice_supervisable": False},
            "supervised_decision": {
                "branch_label": None,
                "terminal_pose": None,
            },
        }
        keep, drop_reason = _is_path_train_view_eligible(train_view)
        self.assertFalse(keep)
        self.assertEqual(drop_reason, "path_not_supervisable")


if __name__ == "__main__":
    unittest.main()
