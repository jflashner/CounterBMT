from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from counter_bmt_v2.causal import TopologicalDAGAssignmentSampler, payload_to_bayesian_dag
from counter_bmt_v2.config import RLPolicyConfig
from counter_bmt_v2.contracts import BayesianDAG, DAGNode, DAGEdge, ScenarioInput
from counter_bmt_v2.data.scenarionet import NNXBMTSceneSample
from counter_bmt_v2.rl.grpo import categorical_kl_from_log_probs, clipped_surrogate_stats
from counter_bmt_v2.rl.nnx_policy import NNXPolicyBackend, _build_feasibility_mask
from counter_bmt_v2.training.supervised import SupervisedTrainConfig
from counter_bmt_v2.trajectory_jax import (
    BidirectionalMotionTokenizer,
    NNXBMTConfig,
    NNXBidirectionalMotionTransformer,
    NNXDAGEncoderConfig,
)


def _dummy_dag() -> BayesianDAG:
    dag = BayesianDAG(scenario_id="scene_rl")
    dag.nodes["maneuver_0"] = DAGNode(
        node_id="maneuver_0",
        node_type="maneuver",
        value="straight",
        timestamp_s=0.5,
        metadata={"alternatives": ["straight", "stop", "accelerate"]},
    )
    dag.nodes["collision_outcome"] = DAGNode(
        node_id="collision_outcome",
        node_type="outcome",
        value="collision_avoided",
        metadata={"alternatives": ["collision_avoided", "collision_possible"]},
    )
    dag.edges.append(DAGEdge(parent_id="maneuver_0", child_id="collision_outcome", confidence=1.0, mechanism="maneuver_to_outcome"))
    dag.cpts = {
        "collision_outcome": {
            "values": ["collision_avoided", "collision_possible"],
            "parents": ["maneuver_0"],
            "cpt": {
                "maneuver_0=stop": {"collision_avoided": 0.0, "collision_possible": 1.0},
                "*": {"collision_avoided": 1.0, "collision_possible": 0.0},
            },
        }
    }
    return dag


def _dummy_dag_payload() -> dict:
    return {
        "schema_version": "counter_bmt_v2_dag_cache_v3_maneuver_outcome",
        "scenario_id": "scene_rl",
        "nodes": [
            {
                "node_id": "maneuver_0",
                "node_type": "maneuver",
                "value": "straight",
                "timestamp_s": 0.5,
                "metadata": {
                    "alternatives": ["straight", "stop", "accelerate"],
                    "start_s": 0.0,
                    "end_s": 1.0,
                    "duration_s": 1.0,
                    "mid_s": 0.5,
                    "observed": True,
                },
            },
            {
                "node_id": "collision_outcome",
                "node_type": "outcome",
                "value": "collision_avoided",
                "timestamp_s": None,
                "metadata": {"alternatives": ["collision_avoided", "collision_possible"], "observed": True},
            },
            {
                "node_id": "progress_outcome",
                "node_type": "outcome",
                "value": "progress_good",
                "timestamp_s": None,
                "metadata": {"alternatives": ["progress_good", "progress_limited"], "observed": True},
            },
            {
                "node_id": "compliance_outcome",
                "node_type": "outcome",
                "value": "compliant",
                "timestamp_s": None,
                "metadata": {"alternatives": ["compliant", "violation_possible"], "observed": True},
            },
        ],
        "edges": [
            {"parent_id": "maneuver_0", "child_id": "collision_outcome", "confidence": 1.0, "mechanism": "maneuver_to_outcome"},
            {"parent_id": "maneuver_0", "child_id": "progress_outcome", "confidence": 1.0, "mechanism": "maneuver_to_outcome"},
            {"parent_id": "maneuver_0", "child_id": "compliance_outcome", "confidence": 1.0, "mechanism": "maneuver_to_outcome"},
        ],
        "cpts": {
            "collision_outcome": {
                "values": ["collision_avoided", "collision_possible"],
                "parents": ["maneuver_0"],
                "cpt": {
                    "maneuver_0=stop": {"collision_avoided": 0.0, "collision_possible": 1.0},
                    "*": {"collision_avoided": 1.0, "collision_possible": 0.0},
                },
            },
            "progress_outcome": {
                "values": ["progress_good", "progress_limited"],
                "parents": ["maneuver_0"],
                "cpt": {"*": {"progress_good": 1.0, "progress_limited": 0.0}},
            },
            "compliance_outcome": {
                "values": ["compliant", "violation_possible"],
                "parents": ["maneuver_0"],
                "cpt": {"*": {"compliant": 1.0, "violation_possible": 0.0}},
            },
        },
        "metadata": {
            "contract_name": "maneuver_outcome_v1",
            "contract_version": "1",
            "contract_report": {"passed": True},
        },
    }


def _dummy_scene_sample() -> NNXBMTSceneSample:
    t = 6
    n = 2
    time = np.arange(t, dtype=np.float32)[:, None]
    ego_x = 0.5 * time
    other_x = 0.4 * time + 1.0
    pos = np.zeros((t, n, 2), dtype=np.float32)
    pos[:, 0, 0] = ego_x[:, 0]
    pos[:, 1, 0] = other_x[:, 0]
    vel = np.zeros_like(pos)
    vel[:, 0, 0] = 1.0
    vel[:, 1, 0] = 0.8
    heading = np.zeros((t, n), dtype=np.float32)
    valid = np.ones((t, n), dtype=bool)
    map_feature = np.zeros((1, 1, 27), dtype=np.float32)
    map_feature[0, 0, 0] = 1.0
    map_valid = np.ones((1, 1), dtype=bool)
    map_pos = np.zeros((1, 3), dtype=np.float32)
    tl_feat = np.zeros((t, 1, 7), dtype=np.float32)
    tl_valid = np.zeros((t, 1), dtype=bool)
    tl_pos = np.zeros((1, 3), dtype=np.float32)
    return NNXBMTSceneSample(
        scenario_id="scene_rl",
        current_time_index=0,
        dt_s=0.1,
        map_center_xyz=np.zeros((3,), dtype=np.float32),
        agent_ids=np.asarray([0, 1], dtype=np.int32),
        agent_type_ids=np.asarray([1, 1], dtype=np.int32),
        agent_shape=np.asarray([[4.5, 2.0, 1.5], [4.0, 1.8, 1.5]], dtype=np.float32),
        agent_position_xy=pos,
        agent_heading=heading,
        agent_velocity_xy=vel,
        agent_valid_mask=valid,
        map_feature=map_feature,
        map_feature_valid_mask=map_valid,
        map_position=map_pos,
        traffic_light_feature=tl_feat,
        traffic_light_valid_mask=tl_valid,
        traffic_light_position=tl_pos,
    )


class TopoMCPONNXTests(unittest.TestCase):
    def test_topological_sampler_respects_cpt_rows(self) -> None:
        dag = _dummy_dag()
        dag.nodes["maneuver_0"].value = "stop"
        dag.nodes["maneuver_0"].metadata["alternatives"] = ["stop"]
        sampler = TopologicalDAGAssignmentSampler()
        intervention = sampler.sample(dag, rare=False, seed=0)
        self.assertEqual(intervention.assignment_order[0], "maneuver_0")
        self.assertEqual(intervention.assignments["maneuver_0"], "stop")
        self.assertEqual(intervention.assignments["collision_outcome"], "collision_possible")

    def test_rare_sampling_biases_toward_lower_probability_values(self) -> None:
        dag = _dummy_dag()
        dag.nodes["maneuver_0"].metadata["alternatives"] = ["straight", "stop"]
        dag.cpts["maneuver_0"] = {
            "values": ["straight", "stop"],
            "parents": [],
            "cpt": {"*": {"straight": 0.95, "stop": 0.05}},
        }
        sampler = TopologicalDAGAssignmentSampler()
        normal_stop = 0
        rare_stop = 0
        for seed in range(200):
            normal_stop += int(sampler.sample(dag, rare=False, seed=seed).assignments["maneuver_0"] == "stop")
            rare_stop += int(sampler.sample(dag, rare=True, seed=seed).assignments["maneuver_0"] == "stop")
        self.assertGreater(rare_stop, normal_stop)

    def test_feasibility_mask_filters_invalid_actions(self) -> None:
        cfg = RLPolicyConfig(
            feasible_max_speed_mps=1.0,
            feasible_max_accel_delta=0.25,
            feasible_max_yaw_delta=0.25,
            enable_feasibility_mask=True,
        )
        action_table = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        valid, invalid_rate = _build_feasibility_mask(
            current_speed_bn=jnp.asarray([[0.5]], dtype=jnp.float32),
            prev_action_bn2=jnp.asarray([[[0.0, 0.0]]], dtype=jnp.float32),
            action_table_v2=action_table,
            dt_chunk_b=jnp.asarray([1.0], dtype=jnp.float32),
            cfg=cfg,
        )
        valid_np = np.asarray(valid)
        self.assertTrue(bool(valid_np[0, 0, 0]))
        self.assertFalse(bool(valid_np[0, 0, 1]))
        self.assertFalse(bool(valid_np[0, 0, 2]))
        self.assertGreater(float(np.asarray(invalid_rate)[0]), 0.0)

    def test_nnx_policy_backend_prepare_scene_smoke(self) -> None:
        model_cfg = NNXBMTConfig(d_model=16, n_layers=1, n_heads=4, ff_mult=2)
        model_cfg.dag_encoder = NNXDAGEncoderConfig(
            enabled=True,
            d_node_in=24,
            d_edge_in=8,
            d_hidden=16,
            n_layers=1,
            max_nodes=8,
            max_edges=16,
        )
        model_cfg.dag_conditioning.enabled = True
        train_cfg = SupervisedTrainConfig(
            model_preset="paper_like_small",
            tokenizer_mode="paper_simple",
            skip_steps=1,
            batch_size=1,
            mode="forward",
            reverse_probability=0.0,
            precision="fp32",
        )
        tokenizer = BidirectionalMotionTokenizer(model_cfg.token_space)
        model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(0))
        backend = NNXPolicyBackend(
            cfg=RLPolicyConfig(tokenizer_mode="paper_simple", skip_steps=1, enable_feasibility_mask=True),
            model_cfg=model_cfg,
            prep_train_cfg=train_cfg,
            tokenizer=tokenizer,
            model=model,
            reference_model=model,
            optimizer=nnx.Optimizer(model, optax.adamw(1e-4)),
        )
        sample = _dummy_scene_sample()
        scene = ScenarioInput(
            scenario_id="scene_rl",
            ego_trajectory_xy=sample.agent_position_xy[:, 0, :],
            metadata={"nnx_sample": sample},
        )
        dag_payload = _dummy_dag_payload()
        dag = payload_to_bayesian_dag(dag_payload)
        prepared = backend._prepare_scene(
            scene=scene,
            sampled_dag=dag,
            sampled_dag_payload=dag_payload,
            seed=0,
        )
        self.assertGreater(prepared.horizon_steps, 0)
        self.assertEqual(prepared.start_token_ids.shape[0], 1)
        self.assertIn("dag_node_feat", prepared.static_model_inputs)
        self.assertTrue(np.isfinite(prepared.init_speed_bn).all())

    def test_clipped_surrogate_and_kl_helpers(self) -> None:
        stats = clipped_surrogate_stats(
            old_logprob=np.asarray([0.0, 0.0], dtype=np.float32),
            new_logprob=np.asarray([np.log(1.4), np.log(0.7)], dtype=np.float32),
            advantages=np.asarray([1.0, -1.0], dtype=np.float32),
            clip_eps=0.2,
        )
        self.assertGreater(float(stats["clip_fraction"]), 0.0)
        self.assertTrue(np.isfinite(float(stats["surrogate"])))

        log_probs = np.log(np.asarray([[[0.7, 0.3]], [[0.5, 0.5]]], dtype=np.float32))
        self.assertAlmostEqual(categorical_kl_from_log_probs(log_probs, log_probs), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
