from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from counter_bmt_v2.config import VLMAlignmentConfig
from counter_bmt_v2.contracts import BayesianDAG, ConditioningSignal, DAGEdge, DAGNode, Intervention, ScenarioInput, TrajectoryRollout
from counter_bmt_v2.rl.vlm_alignment import VLMAlignmentVerifier
from counter_bmt_v2.rl.vlm_alignment_evidence import build_alignment_cache_context
from counter_bmt_v2.rl.vlm_alignment_prompt import parse_alignment_response


class _TestVerifier(VLMAlignmentVerifier):
    def _select_indices(self, *, step: int, scenario_id: str, total: int) -> list[int]:
        _ = (step, scenario_id)
        return [0] if total > 0 else []

    def _score_one(
        self,
        *,
        scenario: ScenarioInput,
        dag: BayesianDAG,
        intervention: Intervention,
        rollout: TrajectoryRollout,
        rollout_index: int,
        step: int,
        evidence_dir: Path,
    ) -> Tuple[Optional[float], Dict[str, object]]:
        _ = (scenario, dag, intervention, rollout, rollout_index, step, evidence_dir)
        return 0.8, {"latency_ms": 1.0}


def _dummy_scene() -> ScenarioInput:
    t = np.linspace(0.0, 1.0, num=8, dtype=np.float32)
    traj = np.stack([t * 5.0, np.zeros_like(t)], axis=1)
    return ScenarioInput(scenario_id="scene_unit", ego_trajectory_xy=traj)


def _dummy_dag() -> BayesianDAG:
    dag = BayesianDAG(scenario_id="scene_unit")
    dag.nodes["ego_initial_speed"] = DAGNode(node_id="ego_initial_speed", node_type="ego_state", value=8.0)
    dag.nodes["decision_0"] = DAGNode(node_id="decision_0", node_type="decision", value="maintain_speed")
    dag.nodes["collision_outcome"] = DAGNode(node_id="collision_outcome", node_type="outcome", value="collision_avoided")
    dag.edges.append(DAGEdge(parent_id="decision_0", child_id="collision_outcome", confidence=0.8, mechanism="effect"))
    return dag


def _dummy_rollouts(k: int) -> list[TrajectoryRollout]:
    out = []
    for i in range(k):
        t = np.linspace(0.0, 1.0, num=10, dtype=np.float32)
        traj = np.stack([t * (3.0 + i), np.zeros_like(t)], axis=1)
        out.append(
            TrajectoryRollout(
                trajectory_xy=traj,
                conditioning=ConditioningSignal(vector=np.zeros((4,), dtype=np.float32), metadata={}),
                sample_index=i,
                metadata={},
            )
        )
    return out


class VLMAlignmentTests(unittest.TestCase):
    def test_parse_response(self) -> None:
        text = """```json
{"conformance_score": 0.73, "confidence": 0.61, "matched_factors": ["lane"], "violations": [], "reason": "ok"}
```"""
        parsed = parse_alignment_response(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertAlmostEqual(parsed.score, 0.73, places=6)
        self.assertEqual(parsed.matched_factors, ["lane"])

    def test_step_mean_fill_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = VLMAlignmentConfig(
                enabled=True,
                source_mode="vlm_replace",
                backend="mock",
                cache_dir=str(Path(td) / "cache"),
                save_evidence_artifacts=False,
            )
            verifier = _TestVerifier(cfg=cfg, output_dir=Path(td) / "out")
            res = verifier.score_rollouts(
                step=5,
                scenario=_dummy_scene(),
                dag=_dummy_dag(),
                intervention=Intervention(variable="decision_0", value="maintain_speed"),
                rollouts=_dummy_rollouts(3),
            )
            self.assertEqual(res.scores.shape[0], 3)
            self.assertTrue(np.allclose(res.scores, np.asarray([0.8, 0.8, 0.8], dtype=np.float32), atol=1e-6))
            self.assertTrue(bool(res.scored_mask[0]))
            self.assertFalse(bool(res.scored_mask[1]))
            self.assertFalse(bool(res.scored_mask[2]))
            self.assertAlmostEqual(float(res.diagnostics.get("calls_attempted", 0.0)), 1.0, places=6)

    def test_disabled_mode_is_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = VLMAlignmentConfig(
                enabled=False,
                source_mode="judge",
                backend="mock",
                neutral_score=0.2,
                cache_dir=str(Path(td) / "cache"),
            )
            verifier = VLMAlignmentVerifier(cfg=cfg, output_dir=Path(td) / "out")
            res = verifier.score_rollouts(
                step=1,
                scenario=_dummy_scene(),
                dag=_dummy_dag(),
                intervention=Intervention(variable="decision_0", value="maintain_speed"),
                rollouts=_dummy_rollouts(2),
            )
            self.assertTrue(np.allclose(res.scores, np.asarray([0.2, 0.2], dtype=np.float32), atol=1e-6))
            self.assertFalse(np.any(res.scored_mask))
            self.assertEqual(float(res.diagnostics.get("step_skipped", 0.0)), 1.0)

    def test_cache_key_uses_full_assignment_and_dag_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = VLMAlignmentConfig(
                enabled=True,
                source_mode="vlm_replace",
                backend="mock",
                cache_dir=str(Path(td) / "cache"),
                save_evidence_artifacts=False,
            )
            verifier = VLMAlignmentVerifier(cfg=cfg, output_dir=Path(td) / "out")
            rollout = _dummy_rollouts(1)[0]
            dag = _dummy_dag()
            intervention_a = Intervention(
                variable="decision_0",
                value="maintain_speed",
                assignments={"decision_0": "maintain_speed", "collision_outcome": "collision_avoided"},
                assignment_order=["decision_0", "collision_outcome"],
                source_dag_schema="counter_bmt_v2_dag_cache_v3_maneuver_outcome",
                is_counterfactual=True,
            )
            intervention_b = Intervention(
                variable="decision_0",
                value="maintain_speed",
                assignments={"decision_0": "maintain_speed", "collision_outcome": "collision_possible"},
                assignment_order=["decision_0", "collision_outcome"],
                source_dag_schema="counter_bmt_v2_dag_cache_v3_maneuver_outcome",
                is_counterfactual=True,
            )
            dag_variant = _dummy_dag()
            dag_variant.nodes["collision_outcome"].value = "collision_possible"

            key_a = verifier._cache_key(
                scenario_id=str(dag.scenario_id),
                rollout=rollout,
                cache_context=build_alignment_cache_context(dag, intervention_a),
            )
            key_b = verifier._cache_key(
                scenario_id=str(dag.scenario_id),
                rollout=rollout,
                cache_context=build_alignment_cache_context(dag, intervention_b),
            )
            key_c = verifier._cache_key(
                scenario_id=str(dag_variant.scenario_id),
                rollout=rollout,
                cache_context=build_alignment_cache_context(dag_variant, intervention_a),
            )

            self.assertNotEqual(key_a, key_b)
            self.assertNotEqual(key_a, key_c)


if __name__ == "__main__":
    unittest.main()
