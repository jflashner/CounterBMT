from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache import DAGCacheReader
from counter_bmt_v2.training.dag_cache_schema import SCHEMA_VERSION, validate_cache_payload


def _base_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": "scenario_unit",
        "nodes": [
            {
                "node_id": "ego_initial_speed",
                "node_type": "ego_state",
                "value": 8.5,
                "timestamp_s": 0.0,
                "metadata": {},
            },
            {
                "node_id": "maneuver_raw",
                "node_type": "maneuver",
                "value": "changing lane left quickly",
                "timestamp_s": 0.5,
                "metadata": {},
            },
            {
                "node_id": "decision_raw",
                "node_type": "decision",
                "value": "keep speed",
                "timestamp_s": 1.0,
                "metadata": {},
            },
            {
                "node_id": "risk_raw",
                "node_type": "risk",
                "value": 0.2,
                "timestamp_s": 1.5,
                "metadata": {},
            },
            {
                "node_id": "collision_outcome",
                "node_type": "outcome",
                "value": "safe",
                "timestamp_s": None,
                "metadata": {},
            },
        ],
        "edges": [
            {
                "parent_id": "ego_initial_speed",
                "child_id": "maneuver_raw",
                "confidence": 0.8,
                "mechanism": "speed to maneuver",
            },
            {
                "parent_id": "maneuver_raw",
                "child_id": "decision_raw",
                "confidence": 0.7,
                "mechanism": "maneuver to decision",
            },
            {
                "parent_id": "risk_raw",
                "child_id": "collision_outcome",
                "confidence": 0.9,
                "mechanism": "risk to outcome",
            },
            {
                "parent_id": "decision_raw",
                "child_id": "collision_outcome",
                "confidence": 0.6,
                "mechanism": "decision to outcome",
            },
        ],
        "cpts": {
            "collision_outcome": {
                "values": ["collision_avoided", "collision_possible"],
                "parents": ["decision_raw", "risk_raw"],
                "cpt": {
                    "*": {
                        "collision_avoided": 2.0,
                        "collision_possible": 1.0,
                    }
                },
            }
        },
        "metadata": {"source": "unit_test"},
    }


class DAGContractV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DAGContractConfig(name="compact10", mode="hard")

    def test_canonicalization_determinism(self) -> None:
        p = _base_payload()
        ok1, c1, _ = enforce_dag_contract(p, config=self.cfg)
        ok2, c2, _ = enforce_dag_contract(p, config=self.cfg)
        self.assertTrue(ok1 and ok2)
        self.assertEqual(
            json.dumps(c1, sort_keys=True, separators=(",", ":")),
            json.dumps(c2, sort_keys=True, separators=(",", ":")),
        )

    def test_tier_violation_rejection(self) -> None:
        p = _base_payload()
        p["edges"].append(
            {
                "parent_id": "collision_outcome",
                "child_id": "decision_raw",
                "confidence": 0.5,
                "mechanism": "invalid reverse",
            }
        )
        ok, _canonical, report = enforce_dag_contract(p, config=self.cfg)
        self.assertFalse(ok)
        self.assertIn("invalid_tier_edge", report.violation_counts)

    def test_parent_cap_rejection(self) -> None:
        p = _base_payload()
        p["nodes"].extend(
            [
                {"node_id": "context_0", "node_type": "context", "value": "urban", "timestamp_s": 0.0, "metadata": {}},
                {"node_id": "context_1", "node_type": "context", "value": "signalized", "timestamp_s": 0.0, "metadata": {}},
                {"node_id": "context_2", "node_type": "context", "value": "dense", "timestamp_s": 0.0, "metadata": {}},
            ]
        )
        p["edges"].extend(
            [
                {"parent_id": "context_0", "child_id": "collision_outcome", "confidence": 0.5, "mechanism": "context"},
                {"parent_id": "context_1", "child_id": "collision_outcome", "confidence": 0.5, "mechanism": "context"},
                {"parent_id": "context_2", "child_id": "collision_outcome", "confidence": 0.5, "mechanism": "context"},
            ]
        )
        ok, _canonical, report = enforce_dag_contract(p, config=self.cfg)
        self.assertFalse(ok)
        self.assertIn("max_parents_exceeded", report.violation_counts)

    def test_behavior_normalization(self) -> None:
        p = _base_payload()
        ok, canonical, _report = enforce_dag_contract(p, config=self.cfg)
        self.assertTrue(ok)
        by_type = {n["node_type"]: n for n in canonical["nodes"]}
        self.assertEqual(by_type["maneuver"]["value"], "lane_change_left")
        self.assertEqual(by_type["decision"]["value"], "maintain_speed")

    def test_cpt_rows_normalized(self) -> None:
        p = _base_payload()
        ok, canonical, _ = enforce_dag_contract(p, config=self.cfg)
        self.assertTrue(ok)
        outcome_cpt = canonical["cpts"]["collision_outcome"]["cpt"]["*"]
        total = float(sum(float(x) for x in outcome_cpt.values()))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_cycle_rejection(self) -> None:
        p = _base_payload()
        p["edges"].append(
            {
                "parent_id": "decision_raw",
                "child_id": "maneuver_raw",
                "confidence": 0.4,
                "mechanism": "invalid_back",
            }
        )
        ok, _canonical, report = enforce_dag_contract(p, config=self.cfg)
        self.assertFalse(ok)
        self.assertIn("invalid_tier_edge", report.violation_counts)

    def test_schema_reader_rejects_v1_accepts_v2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sid = "scenario_unit"
            v1 = _base_payload()
            v1["schema_version"] = "counter_bmt_v2_dag_cache_v1"
            (root / f"{sid}.json").write_text(json.dumps(v1), encoding="utf-8")
            reader = DAGCacheReader(str(root))
            self.assertIsNone(reader.get(sid))

            ok, canonical, _ = enforce_dag_contract(_base_payload(), config=self.cfg)
            self.assertTrue(ok)
            self.assertTrue(validate_cache_payload(canonical))
            (root / f"{sid}.json").write_text(json.dumps(canonical), encoding="utf-8")
            loaded = reader.get(sid)
            self.assertIsNotNone(loaded)
            self.assertEqual(str(loaded["schema_version"]), SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
