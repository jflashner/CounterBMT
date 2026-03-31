from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SRC = ROOT / "src" / "Adv-BMT"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from bmt.counterfactual.inspect import INSPECTION_FILENAMES, build_scenario_summary, build_traffic_light_table_rows, inspection_output_paths
from bmt.counterfactual.normalize import normalize_scenario


def _mock_scenario() -> dict:
    return {
        "id": "scenario_mock",
        "length": 5,
        "tracks": {
            2: {
                "type": "VEHICLE",
                "state": {
                    "position": np.array(
                        [
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 1.0],
                            [2.0, 0.0, 1.0],
                            [3.0, 0.0, 1.0],
                            [4.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ),
                    "heading": np.zeros((5,), dtype=np.float32),
                    "velocity": np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (5, 1)),
                    "valid": np.array([True, True, True, True, True]),
                },
                "metadata": {"object_id": 2, "type": "VEHICLE"},
            },
            "7": {
                "type": "PEDESTRIAN",
                "state": {
                    "position": np.array(
                        [
                            [10.0, 10.0, 0.0],
                            [10.0, 10.5, 0.0],
                            [10.0, 11.0, 0.0],
                            [10.0, 11.5, 0.0],
                            [10.0, 12.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    "heading": np.zeros((5,), dtype=np.float32),
                    "velocity": np.tile(np.array([[0.0, 0.5]], dtype=np.float32), (5, 1)),
                    "valid": np.array([True, True, True, False, False]),
                },
                "metadata": {"object_id": "7", "type": "PEDESTRIAN"},
            },
        },
        "dynamic_map_states": {
            900: {
                "type": "TRAFFIC_LIGHT",
                "state": {
                    "object_state": [
                        None,
                        "LANE_STATE_STOP",
                        "LANE_STATE_STOP",
                        "LANE_STATE_GO",
                        "LANE_STATE_GO",
                    ]
                },
                "lane": 55,
                "stop_point": [3.0, 4.0, 0.0],
                "metadata": {"object_id": 900, "type": "TRAFFIC_LIGHT"},
            }
        },
        "map_features": {
            11: {
                "type": "LANE_SURFACE_STREET",
                "polyline": np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32),
            }
        },
        "metadata": {
            "scenario_id": "scenario_mock",
            "sdc_id": 2,
            "current_time_index": 2,
            "ts": np.arange(5, dtype=np.float32) * 0.2,
            "objects_of_interest": [7],
            "tracks_to_predict": {"2": {"track_id": "2"}},
            "number_summary": {"num_objects": 2},
        },
    }


class CounterfactualPR1Tests(unittest.TestCase):
    def test_normalization_on_mocked_mini_scenario(self) -> None:
        canonical = normalize_scenario(_mock_scenario())
        self.assertEqual(canonical.scenario_id, "scenario_mock")
        self.assertEqual(canonical.length, 5)
        self.assertEqual(canonical.sdc_id, "2")
        self.assertEqual(canonical.current_time_index, 2)
        self.assertEqual(sorted(canonical.tracks.keys()), ["2", "7"])
        self.assertEqual(canonical.traffic_lights["900"].lane_ref, "55")
        self.assertEqual(canonical.traffic_lights["900"].object_state[1], "LANE_STATE_STOP")
        self.assertEqual(canonical.map_features["11"].feature_type, "LANE_SURFACE_STREET")

    def test_json_serialization_of_summary(self) -> None:
        canonical = normalize_scenario(_mock_scenario())
        summary = build_scenario_summary(canonical)
        payload = json.loads(json.dumps(summary))
        self.assertEqual(payload["scenario_id"], "scenario_mock")
        self.assertEqual(payload["sdc_motion_stats"]["num_valid_steps"], 5)

    def test_no_crash_on_missing_lane_ref_or_stop_point(self) -> None:
        raw = _mock_scenario()
        raw["dynamic_map_states"][900].pop("lane", None)
        raw["dynamic_map_states"][900].pop("stop_point", None)
        canonical = normalize_scenario(raw)
        rows = build_traffic_light_table_rows(canonical)
        self.assertEqual(rows[0]["lane_ref"], "")
        self.assertEqual(rows[0]["stop_x"], "")
        self.assertEqual(rows[0]["stop_y"], "")

    def test_deterministic_output_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paths = inspection_output_paths(td)
            self.assertEqual({key: path.name for key, path in paths.items()}, INSPECTION_FILENAMES)


if __name__ == "__main__":
    unittest.main()
