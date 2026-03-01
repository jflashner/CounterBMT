from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from counter_bmt_v2.causal.dag_contract import DAGContractConfig, enforce_dag_contract
from counter_bmt_v2.training.dag_cache import DAGCacheReader
from counter_bmt_v2.training.dag_cache_schema import (
    SCHEMA_VERSION_V2_COMPACT10,
    SCHEMA_VERSION_V3_MANEUVER_OUTCOME,
    validate_cache_payload,
)
from counter_bmt_v2.training.dag_tensorize import tensorize_dag_batch


def _base_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION_V2_COMPACT10,
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
            self.assertTrue(validate_cache_payload(canonical, allowed_schema_versions=(SCHEMA_VERSION_V2_COMPACT10,)))
            (root / f"{sid}.json").write_text(json.dumps(canonical), encoding="utf-8")
            loaded = reader.get(sid)
            self.assertIsNotNone(loaded)
            self.assertEqual(str(loaded["schema_version"]), SCHEMA_VERSION_V2_COMPACT10)


def _mo_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION_V3_MANEUVER_OUTCOME,
        "scenario_id": "scenario_mo",
        "nodes": [
            {
                "node_id": "segment_a",
                "node_type": "maneuver",
                "value": "changing lane left",
                "timestamp_s": 1.0,
                "metadata": {"start_s": 0.5, "end_s": 1.5},
            },
            {
                "node_id": "collision",
                "node_type": "outcome",
                "value": "possible collision",
                "metadata": {},
            },
            {
                "node_id": "progress",
                "node_type": "outcome",
                "value": "good progress",
                "metadata": {},
            },
            {
                "node_id": "compliance",
                "node_type": "outcome",
                "value": "compliant",
                "metadata": {},
            },
        ],
        "edges": [
            {"parent_id": "segment_a", "child_id": "collision", "confidence": 0.8, "mechanism": "maneuver_to_outcome"},
            {"parent_id": "segment_a", "child_id": "progress", "confidence": 0.7, "mechanism": "maneuver_to_outcome"},
            {"parent_id": "segment_a", "child_id": "compliance", "confidence": 0.7, "mechanism": "maneuver_to_outcome"},
        ],
        "cpts": {},
        "metadata": {"source": "unit_test"},
    }


class DAGContractV3ManeuverOutcomeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DAGContractConfig(name="maneuver_outcome_v1", mode="hard")

    def test_maneuver_outcome_contract_passes(self) -> None:
        ok, canonical, report = enforce_dag_contract(_mo_payload(), config=self.cfg)
        self.assertTrue(ok)
        self.assertTrue(report.passed)
        self.assertEqual(canonical["schema_version"], SCHEMA_VERSION_V3_MANEUVER_OUTCOME)
        self.assertTrue(
            validate_cache_payload(canonical, allowed_schema_versions=(SCHEMA_VERSION_V3_MANEUVER_OUTCOME,))
        )

    def test_rejects_invalid_node_types(self) -> None:
        p = _mo_payload()
        p["nodes"].append(
            {"node_id": "decision_0", "node_type": "decision", "value": "accelerate", "metadata": {}}
        )
        ok, _canonical, report = enforce_dag_contract(p, config=self.cfg)
        self.assertFalse(ok)
        self.assertIn("invalid_node_type", report.violation_counts)

    def test_rejects_missing_interval_metadata(self) -> None:
        p = _mo_payload()
        p["nodes"][0]["metadata"] = {"start_s": 0.5}
        ok, _canonical, report = enforce_dag_contract(p, config=self.cfg)
        self.assertFalse(ok)
        self.assertIn("invalid_interval", report.violation_counts)

    def test_tensorize_v3_shape_and_interval_features(self) -> None:
        ok, canonical, _ = enforce_dag_contract(_mo_payload(), config=self.cfg)
        self.assertTrue(ok)
        batch = tensorize_dag_batch([canonical], max_nodes=16, max_edges=32, d_node_in=24, d_edge_in=8)
        self.assertEqual(batch["dag_node_feat"].shape[-1], 24)
        self.assertEqual(batch["dag_node_feat"].shape[0], 1)
        # first maneuver node should have non-zero interval features (slice 18:22 in v3 layout)
        feat = batch["dag_node_feat"][0, 0]
        self.assertGreater(float(np.sum(np.abs(feat[18:22]))), 0.0)

    def test_dual_read_and_strict_schema_modes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sid = "scenario_mo"
            ok, canonical, _ = enforce_dag_contract(_mo_payload(), config=self.cfg)
            self.assertTrue(ok)
            (root / f"{sid}.json").write_text(json.dumps(canonical), encoding="utf-8")

            reader_any = DAGCacheReader(str(root))
            self.assertIsNotNone(reader_any.get(sid))

            reader_v3 = DAGCacheReader(str(root), allowed_schema_versions=(SCHEMA_VERSION_V3_MANEUVER_OUTCOME,))
            self.assertIsNotNone(reader_v3.get(sid))

            reader_v2_only = DAGCacheReader(str(root), allowed_schema_versions=(SCHEMA_VERSION_V2_COMPACT10,))
            self.assertIsNone(reader_v2_only.get(sid))


if __name__ == "__main__":
    unittest.main()
