from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import jax
import numpy as np
from flax import nnx

from counter_bmt_v2.causal import TopologicalDAGAssignmentSampler
from counter_bmt_v2.cli.train_rl_topo_mcpo import _build_parser as rl_build_parser
from counter_bmt_v2.cli.train_rl_topo_mcpo import main as rl_main
from counter_bmt_v2.config import ConsensusConfig, PipelineConfig
from counter_bmt_v2.contracts import ConditioningSignal, ScenarioInput, TrajectoryRollout
from counter_bmt_v2.orchestration import CounterBMTPipeline
from counter_bmt_v2.rl.consensus import ConsensusScorer, mean_cluster_quality
from counter_bmt_v2.rl.loop import _collect_group_rollouts_nnx, _top_surprisal_resample
from counter_bmt_v2.rl.nnx_policy import PolicyCandidatePool, PolicyRolloutData
from counter_bmt_v2.trajectory_jax import NNXBMTConfig, NNXBidirectionalMotionTransformer, NNXDAGEncoderConfig

from tests.test_topo_mcpo_nnx import _dummy_dag_payload, _dummy_scene_sample


def _dummy_scene_input() -> ScenarioInput:
    sample = _dummy_scene_sample()
    return ScenarioInput(
        scenario_id="scene_rl",
        ego_trajectory_xy=sample.agent_position_xy[:, 0, :],
        metadata={"nnx_sample": sample},
    )


def _raw_scenario_from_sample() -> dict:
    sample = _dummy_scene_sample()
    horizon = int(sample.agent_position_xy.shape[0])
    tracks: dict[int, dict] = {}
    for j, agent_id in enumerate(np.asarray(sample.agent_ids).tolist()):
        pos3 = np.concatenate(
            [
                np.asarray(sample.agent_position_xy[:, j, :], dtype=np.float32),
                np.zeros((horizon, 1), dtype=np.float32),
            ],
            axis=1,
        )
        vel = np.asarray(sample.agent_velocity_xy[:, j, :], dtype=np.float32)
        heading = np.asarray(sample.agent_heading[:, j], dtype=np.float32)
        valid = np.asarray(sample.agent_valid_mask[:, j], dtype=bool)
        shape = np.asarray(sample.agent_shape[j], dtype=np.float32)
        tracks[int(agent_id)] = {
            "type": "VEHICLE",
            "metadata": {"object_id": int(agent_id), "type": "VEHICLE"},
            "state": {
                "position": pos3,
                "velocity": vel,
                "heading": heading,
                "valid": valid,
                "length": np.full((horizon,), float(shape[0]), dtype=np.float32),
                "width": np.full((horizon,), float(shape[1]), dtype=np.float32),
                "height": np.full((horizon,), float(shape[2]), dtype=np.float32),
            },
        }
    return {
        "id": "scene_rl",
        "metadata": {
            "scenario_id": "scene_rl",
            "sdc_id": int(sample.agent_ids[0]),
            "current_time_index": int(sample.current_time_index),
            "ts": np.arange(horizon, dtype=np.float32) * float(sample.dt_s),
        },
        "tracks": tracks,
        "map_features": {},
        "dynamic_map_states": {},
    }


def _write_synthetic_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "sd_scene_rl.pkl").open("wb") as f:
        pickle.dump(_raw_scenario_from_sample(), f)
    return root


def _tiny_dag_model_cfg() -> NNXBMTConfig:
    model_cfg = NNXBMTConfig(d_model=16, n_layers=1, n_heads=4, ff_mult=2)
    model_cfg.max_agents = 8
    model_cfg.max_map_objects = 16
    model_cfg.max_tl_objects = 4
    model_cfg.dag_encoder = NNXDAGEncoderConfig(
        enabled=True,
        d_node_in=24,
        d_edge_in=8,
        d_hidden=16,
        n_layers=1,
        max_nodes=16,
        max_edges=32,
    )
    model_cfg.dag_conditioning.enabled = True
    return model_cfg


def _write_synthetic_checkpoint(path: Path, *, model_cfg: NNXBMTConfig | None = None) -> Path:
    model_cfg = model_cfg or _tiny_dag_model_cfg()
    model = NNXBidirectionalMotionTransformer(model_cfg, rngs=nnx.Rngs(0))
    payload = {
        "model_state": jax.device_get(nnx.state(model)),
        "train_cfg": {
            "model_preset": "midgpt_dag_latent",
            "tokenizer_mode": "paper_simple",
            "skip_steps": 1,
            "precision": "fp32",
            "max_time_steps": 6,
            "max_agents": 8,
            "max_map_features": 16,
            "max_vectors_per_map_feature": 128,
            "max_traffic_lights": 4,
        },
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _RecordingNoveltyEstimator:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, np.ndarray]] = []

    def score_batch(self, embeddings: np.ndarray, *, update: bool = True) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32).reshape(len(embeddings), -1)
        self.calls.append((bool(update), arr.copy()))
        return arr[:, 0].astype(np.float32)


class _FakeEncoder:
    def encode(self, *, dag, intervention, rollout, scenario_id: str, rollout_id: str):
        _ = (dag, intervention, scenario_id, rollout_id)
        return (
            np.asarray(rollout.metadata["psi"], dtype=np.float32),
            dict(rollout.metadata["risk"]),
            {},
        )


class _FakeConsensusScorer:
    def score(self, psi: np.ndarray, risk_features, *, seed: int = 0):
        _ = seed
        arr = np.asarray(psi, dtype=np.float32).reshape(len(psi), -1)
        cluster_ids = (arr[:, 0] >= 2.0).astype(np.int32) if arr.size else np.zeros((0,), dtype=np.int32)
        q_i = np.asarray([float(r.get("progress_delta", 0.0)) for r in risk_features], dtype=np.float32)
        hist = {str(int(k)): int(v) for k, v in zip(*np.unique(cluster_ids, return_counts=True))} if cluster_ids.size else {}
        consensus = np.zeros_like(q_i, dtype=np.float32)
        return cluster_ids, consensus, hist, q_i


class _FixedThermostat:
    def __init__(self, eta: float, alpha: float = 0.0, entropy: float = 0.0) -> None:
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.entropy = float(entropy)

    def compute(self, cluster_ids: np.ndarray):
        _ = cluster_ids
        return self.eta, self.alpha, self.entropy


class _FakeDAGResolver:
    def resolve_one(self, *, scenario_id: str, batch_slice: dict, sample_index: int):
        _ = (scenario_id, batch_slice, sample_index)
        return _dummy_dag_payload(), "scene_derived"


class _FakePolicyBackend:
    def __init__(self) -> None:
        rollouts = []
        for i, psi_val in enumerate([0.1, 0.3, 2.4, 2.8]):
            traj = np.stack(
                [
                    np.linspace(0.0, 1.0 + 0.1 * i, num=6, dtype=np.float32),
                    np.zeros((6,), dtype=np.float32),
                ],
                axis=1,
            )
            rollouts.append(
                TrajectoryRollout(
                    trajectory_xy=traj,
                    conditioning=ConditioningSignal(vector=np.zeros((0,), dtype=np.float32), metadata={}),
                    sample_index=i,
                    metadata={
                        "psi": [psi_val],
                        "risk": {
                            "progress_delta": float(i + 1),
                            "collision_risk_proxy": 0.0,
                            "rule_violation_proxy": 0.0,
                        },
                    },
                )
            )
        rollout_data = PolicyRolloutData(
            prepared_scene=None,  # type: ignore[arg-type]
            token_ids=np.zeros((len(rollouts), 1, 1), dtype=np.int32),
            old_logprob_sum=np.zeros((len(rollouts),), dtype=np.float32),
            entropy_mean=np.zeros((len(rollouts),), dtype=np.float32),
            feasibility_mask_rate=np.zeros((len(rollouts),), dtype=np.float32),
        )
        self.pool = PolicyCandidatePool(
            prepared_scene=None,  # type: ignore[arg-type]
            rollout_data=rollout_data,
            rollouts=rollouts,
            trajectory_all_xy=np.zeros((len(rollouts), 1, 1, 2), dtype=np.float32),
        )

    def sample_candidate_pool(self, **kwargs):
        _ = kwargs
        return self.pool

    def select_rollout_data(self, pool: PolicyCandidatePool, indices):
        idx = np.asarray(indices, dtype=np.int32)
        return PolicyRolloutData(
            prepared_scene=pool.rollout_data.prepared_scene,
            token_ids=pool.rollout_data.token_ids[idx],
            old_logprob_sum=pool.rollout_data.old_logprob_sum[idx],
            entropy_mean=pool.rollout_data.entropy_mean[idx],
            feasibility_mask_rate=pool.rollout_data.feasibility_mask_rate[idx],
        )


class RLTopoMCPOLoopTests(unittest.TestCase):
    def test_top_surprisal_resample_prefers_high_surprisal_with_higher_eta(self) -> None:
        surprisal = np.asarray([0.1, 0.2, 1.2, 2.0], dtype=np.float32)
        low = 0
        high = 0
        for seed in range(400):
            idx_low, _, _ = _top_surprisal_resample(surprisal=surprisal, group_size=1, eta=0.0, seed=seed)
            idx_high, _, _ = _top_surprisal_resample(surprisal=surprisal, group_size=1, eta=2.5, seed=seed)
            low += int(int(idx_low[0]) == 3)
            high += int(int(idx_high[0]) == 3)
        self.assertGreater(high, low)

    def test_novelty_updates_only_after_final_group_selection(self) -> None:
        cfg = PipelineConfig()
        pipeline = CounterBMTPipeline.default(cfg)
        pipeline.sampler = TopologicalDAGAssignmentSampler()
        novelty = _RecordingNoveltyEstimator()
        policy_backend = _FakePolicyBackend()
        batch = _collect_group_rollouts_nnx(
            pipeline,
            _dummy_scene_input(),
            step=1,
            encoder=_FakeEncoder(),
            novelty_estimator=novelty,
            consensus_scorer=_FakeConsensusScorer(),
            thermostat=_FixedThermostat(eta=1.5),
            group_size=2,
            dag_resolver=_FakeDAGResolver(),
            policy_backend=policy_backend,
            vlm_aligner=None,
            seed=0,
            rare=False,
            update_novelty=True,
        )
        self.assertEqual(len(batch.rollouts), 2)
        self.assertEqual(len(novelty.calls), 2)
        self.assertFalse(novelty.calls[0][0])
        self.assertTrue(novelty.calls[1][0])
        all_embeddings = novelty.calls[0][1]
        selected_idx, _, _ = _top_surprisal_resample(
            surprisal=all_embeddings[:, 0],
            group_size=2,
            eta=1.5,
            seed=17,
        )
        self.assertTrue(np.allclose(novelty.calls[1][1], all_embeddings[selected_idx], atol=1e-6))

    def test_consensus_score_matches_cluster_mass_times_mean_quality(self) -> None:
        scorer = ConsensusScorer(cfg=ConsensusConfig(clusterer="kmeans", k_clusters=2))
        psi = np.asarray([[0.0], [0.2], [10.0], [10.2]], dtype=np.float32)
        risk_features = [
            {"progress_delta": 5.0, "collision_risk_proxy": 0.0, "rule_violation_proxy": 0.0},
            {"progress_delta": 4.5, "collision_risk_proxy": 0.1, "rule_violation_proxy": 0.0},
            {"progress_delta": -1.0, "collision_risk_proxy": 0.7, "rule_violation_proxy": 0.4},
            {"progress_delta": -1.5, "collision_risk_proxy": 0.6, "rule_violation_proxy": 0.5},
        ]
        cluster_ids, consensus, hist, q_i = scorer.score(psi, risk_features, seed=0)
        _ = hist

        unique = np.unique(cluster_ids)
        cluster_means = []
        for label in unique.tolist():
            idx = np.where(cluster_ids == int(label))[0]
            expected = (float(idx.size) / float(cluster_ids.size)) * float(np.mean(q_i[idx]))
            self.assertTrue(np.allclose(consensus[idx], expected, atol=1e-6))
            cluster_means.append(float(np.mean(q_i[idx])))
        self.assertGreater(max(cluster_means), min(cluster_means))
        self.assertAlmostEqual(
            mean_cluster_quality(cluster_ids, q_i),
            float(np.mean(np.asarray(cluster_means, dtype=np.float32))),
            places=6,
        )


class RLTopoMCPOCLISmokeTests(unittest.TestCase):
    def _run_cli(self, argv: list[str]) -> int:
        with patch.object(sys, "argv", argv):
            return rl_main()

    def test_nnx_checkpoint_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = _write_synthetic_dataset(root / "data")
            model_cfg = _tiny_dag_model_cfg()
            ckpt = _write_synthetic_checkpoint(root / "policy.pkl", model_cfg=model_cfg)
            out_dir = root / "out_mock"
            with patch("counter_bmt_v2.rl.nnx_policy._resolve_model_cfg_for_policy", return_value=model_cfg):
                rc = self._run_cli(
                    [
                        "train_rl_topo_mcpo",
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(out_dir),
                        "--steps",
                        "1",
                        "--log-every",
                        "1",
                        "--max-scenes",
                        "1",
                        "--group-size",
                        "2",
                        "--embedding-mode",
                        "risk_vector",
                        "--policy-backend",
                        "nnx_checkpoint",
                        "--policy-checkpoint",
                        str(ckpt),
                        "--policy-model-preset",
                        "midgpt_dag_latent",
                        "--policy-tokenizer-mode",
                        "paper_simple",
                        "--policy-skip-steps",
                        "1",
                        "--dag-source-mode",
                        "scene_derived",
                        "--allow-debug-fallbacks",
                        "--perception-backend",
                        "mock",
                        "--dag-backend",
                        "simple",
                        "--alignment-source-mode",
                        "judge",
                        "--no-vlm-alignment-enabled",
                    ]
                )
            self.assertEqual(rc, 0)
            records = _read_metrics(out_dir / "metrics.jsonl")
            self.assertEqual(len(records), 1)
            metrics = records[-1]["metrics"]
            self.assertTrue(np.isfinite(float(metrics["policy/loss"])))
            self.assertTrue(np.isfinite(float(metrics["policy/kl_ref"])))
            self.assertEqual(float(metrics["sampling/candidate_pool_size"]), 4.0)
            self.assertEqual(float(metrics["policy/step"]), 1.0)

    def test_nnx_checkpoint_vlm_replace_mock_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = _write_synthetic_dataset(root / "data")
            model_cfg = _tiny_dag_model_cfg()
            ckpt = _write_synthetic_checkpoint(root / "policy.pkl", model_cfg=model_cfg)
            out_dir = root / "out_vlm"
            with patch("counter_bmt_v2.rl.nnx_policy._resolve_model_cfg_for_policy", return_value=model_cfg):
                rc = self._run_cli(
                    [
                        "train_rl_topo_mcpo",
                        "--data-dir",
                        str(data_dir),
                        "--output-dir",
                        str(out_dir),
                        "--steps",
                        "1",
                        "--log-every",
                        "1",
                        "--max-scenes",
                        "1",
                        "--group-size",
                        "2",
                        "--embedding-mode",
                        "risk_vector",
                        "--policy-backend",
                        "nnx_checkpoint",
                        "--policy-checkpoint",
                        str(ckpt),
                        "--policy-model-preset",
                        "midgpt_dag_latent",
                        "--policy-tokenizer-mode",
                        "paper_simple",
                        "--policy-skip-steps",
                        "1",
                        "--dag-source-mode",
                        "scene_derived",
                        "--allow-debug-fallbacks",
                        "--perception-backend",
                        "mock",
                        "--dag-backend",
                        "simple",
                        "--alignment-source-mode",
                        "vlm_replace",
                        "--vlm-alignment-enabled",
                        "--vlm-alignment-backend",
                        "mock",
                        "--vlm-alignment-sample-rate",
                        "1.0",
                        "--vlm-alignment-every-n-steps",
                        "1",
                        "--vlm-alignment-max-calls-per-step",
                        "1",
                        "--no-vlm-alignment-save-evidence",
                    ]
                )
            self.assertEqual(rc, 0)
            records = _read_metrics(out_dir / "metrics.jsonl")
            metrics = records[-1]["metrics"]
            self.assertTrue(np.isfinite(float(metrics["reward/total_mean"])))
            self.assertGreater(float(metrics["alignment/vlm_scored_fraction"]), 0.0)
            self.assertEqual(float(metrics["alignment/source_mode_vlm_replace"]), 1.0)

    def test_scaffold_backend_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out_scaffold"
            rc = self._run_cli(
                [
                    "train_rl_topo_mcpo",
                    "--output-dir",
                    str(out_dir),
                    "--steps",
                    "1",
                    "--log-every",
                    "1",
                    "--group-size",
                    "2",
                    "--embedding-mode",
                    "risk_vector",
                    "--allow-debug-fallbacks",
                    "--perception-backend",
                    "mock",
                    "--dag-backend",
                    "simple",
                    "--alignment-source-mode",
                    "judge",
                    "--no-vlm-alignment-enabled",
                    "--policy-backend",
                    "scaffold",
                ]
            )
            self.assertEqual(rc, 0)
            records = _read_metrics(out_dir / "metrics.jsonl")
            self.assertEqual(len(records), 1)
            self.assertTrue(np.isfinite(float(records[-1]["metrics"]["reward/total_mean"])))

    def test_parser_defaults_are_real_run_defaults(self) -> None:
        args = rl_build_parser().parse_args([])
        self.assertEqual(str(args.alignment_source_mode), "vlm_replace")
        self.assertTrue(bool(args.vlm_alignment_enabled))
        self.assertEqual(str(args.vlm_alignment_backend), "openai")
        self.assertEqual(str(args.vlm_alignment_model), "gpt-5-mini")
        self.assertEqual(str(args.perception_backend), "openai")
        self.assertEqual(str(args.dag_backend), "promptbn")
        self.assertEqual(str(args.llm_model), "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
