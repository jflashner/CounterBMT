from __future__ import annotations

import unittest

import numpy as np

from bmt.counterfactual.normalize import normalize_scenario
from bmt.counterfactual.sdc_path_branches import enumerate_branch_candidates_from_sdc_paths
from bmt.counterfactual.waymax_adapter import raw_scenario_from_waymax_state


class _DummyTrajectory:
    def __init__(self) -> None:
        self.x = np.asarray([[0.0, 1.0, 2.0], [10.0, 10.5, 11.0]], dtype=np.float32)
        self.y = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.2, 0.3]], dtype=np.float32)
        self.z = np.zeros_like(self.x)
        self.yaw = np.zeros_like(self.x)
        self.vel_x = np.ones_like(self.x)
        self.vel_y = np.zeros_like(self.x)
        self.valid = np.ones_like(self.x, dtype=bool)


class _DummyMetadata:
    def __init__(self) -> None:
        self.ids = np.asarray([101, 202], dtype=np.int64)
        self.object_types = np.asarray([1, 1], dtype=np.int64)
        self.is_sdc = np.asarray([True, False], dtype=bool)


class _DummyLights:
    def __init__(self) -> None:
        self.ids = np.asarray([900], dtype=np.int64)
        self.lane_ids = np.asarray([55], dtype=np.int64)
        self.state = np.asarray([[3, 3, 3]], dtype=np.int64)
        self.x = np.asarray([[4.0, 4.0, 4.0]], dtype=np.float32)
        self.y = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
        self.z = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.valid = np.asarray([[True, True, True]], dtype=bool)


class _DummyRoadgraph:
    def __init__(self) -> None:
        self.ids = np.asarray([10, 10, 10, 20, 20], dtype=np.int64)
        self.types = np.asarray([1, 1, 1, 1, 1], dtype=np.int64)
        self.x = np.asarray([0.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.y = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
        self.z = np.zeros((5,), dtype=np.float32)
        self.valid = np.asarray([True, True, True, True, True], dtype=bool)


class _DummyPaths:
    def __init__(self) -> None:
        self.ids = np.asarray([0, 1], dtype=np.int64)
        self.x = np.asarray([[0.0, 5.0, 10.0], [0.0, 5.0, 5.0]], dtype=np.float32)
        self.y = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float32)
        self.z = np.zeros_like(self.x)
        self.valid = np.asarray([[True, True, True], [True, True, True]], dtype=bool)
        self.on_route = np.asarray([True, False], dtype=bool)


class _DummyState:
    def __init__(self) -> None:
        self.id = "synthetic_waymax_scene"
        self.current_time_index = 1
        self.log_trajectory = _DummyTrajectory()
        self.object_metadata = _DummyMetadata()
        self.log_traffic_light = _DummyLights()
        self.roadgraph_points = _DummyRoadgraph()
        self.sdc_paths = _DummyPaths()


class WaymaxAdapterTest(unittest.TestCase):
    def test_raw_conversion_preserves_sdc_paths(self) -> None:
        raw = raw_scenario_from_waymax_state(_DummyState(), current_time_index=1)
        self.assertEqual(raw["metadata"]["sdc_id"], "101")
        self.assertEqual(len(raw["sdc_paths"]), 2)
        self.assertIn("sdc_path_0", raw["sdc_paths"])
        self.assertIn("sdc_path_1", raw["sdc_paths"])

    def test_sdc_path_branches_produce_semantic_families(self) -> None:
        canonical = normalize_scenario(raw_scenario_from_waymax_state(_DummyState(), current_time_index=1))
        branches = enumerate_branch_candidates_from_sdc_paths(
            canonical,
            agent_id=str(canonical.sdc_id),
            decision_time_idx=1,
            approach_heading=0.0,
        )
        labels = [str(branch.branch_label) for branch in branches]
        self.assertIn("straight", labels)
        self.assertIn("left", labels)


if __name__ == "__main__":
    unittest.main()
