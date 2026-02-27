"""Grouped rollout collection + GRPO utilities for CounterBMT v2 RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from counter_bmt_v2.contracts import (
    BayesianDAG,
    Intervention,
    JudgeResult,
    PipelineResult,
    RLBatchDiagnostics,
    RewardBreakdown,
    ScenarioInput,
    TrajectoryRollout,
    VLMFeatures,
)
from counter_bmt_v2.rl.behavior_embedding import BehaviorManifoldEncoder
from counter_bmt_v2.rl.consensus import ConsensusScorer
from counter_bmt_v2.rl.grpo import GRPOTrainer, compute_group_advantages as _compute_group_advantages
from counter_bmt_v2.rl.novelty import NoveltyEstimator
from counter_bmt_v2.rl.reward import compose_reward
from counter_bmt_v2.rl.thermostat import EntropyThermostat

if TYPE_CHECKING:
    from counter_bmt_v2.orchestration import CounterBMTPipeline


@dataclass
class GroupedRolloutBatch:
    scenario_id: str
    features: VLMFeatures
    dag: BayesianDAG
    intervention: Intervention
    rollouts: List[TrajectoryRollout]
    judge_results: List[JudgeResult]
    rewards: List[RewardBreakdown]
    behavior_embeddings: np.ndarray
    risk_features: List[Dict[str, float]]
    novelty_surprisal: np.ndarray
    novelty_scores: np.ndarray
    cluster_ids: np.ndarray
    consensus_scores: np.ndarray
    diagnostics: RLBatchDiagnostics
    pipeline_result: PipelineResult

    @property
    def reward_array(self) -> np.ndarray:
        vals = [float(r.total) for r in self.rewards]
        return np.asarray(vals, dtype=np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20.0, 20.0)))


def _update_rollout_metadata(
    rollout: TrajectoryRollout,
    *,
    risk_features: Dict[str, float],
    behavior_embedding: np.ndarray,
    novelty_surprisal: float,
    novelty_score: float,
    cluster_id: int,
    consensus_score: float,
) -> None:
    rollout.metadata["risk_features"] = {k: float(v) for k, v in risk_features.items()}
    rollout.metadata["behavior_embedding"] = np.asarray(behavior_embedding, dtype=np.float32).tolist()
    rollout.metadata["novelty_surprisal"] = float(novelty_surprisal)
    rollout.metadata["novelty_score"] = float(novelty_score)
    rollout.metadata["cluster_id"] = int(cluster_id)
    rollout.metadata["consensus_score"] = float(consensus_score)


def collect_group_rollouts(
    pipeline: "CounterBMTPipeline",
    scene: ScenarioInput,
    *,
    encoder: BehaviorManifoldEncoder,
    novelty_estimator: NoveltyEstimator,
    consensus_scorer: ConsensusScorer,
    thermostat: EntropyThermostat,
    group_size: int,
    seed: int = 0,
    rare: bool = False,
    update_novelty: bool = True,
) -> GroupedRolloutBatch:
    """Collect one grouped rollout batch and annotate manifold statistics."""
    result = pipeline.run(scene, n_samples=int(group_size), seed=int(seed), rare=bool(rare))

    embeddings: List[np.ndarray] = []
    risk_features: List[Dict[str, float]] = []
    for i, rollout in enumerate(result.rollouts):
        emb, risk, _ = encoder.encode(
            dag=result.dag,
            intervention=result.intervention,
            rollout=rollout,
            scenario_id=result.scenario_id,
            rollout_id=f"seed{seed}_sample{i}",
        )
        embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        risk_features.append({k: float(v) for k, v in risk.items()})

    if not embeddings:
        diag = RLBatchDiagnostics(entropy=0.0, cluster_hist={}, thermostat_eta=0.0, thermostat_alpha=0.0)
        return GroupedRolloutBatch(
            scenario_id=result.scenario_id,
            features=result.features,
            dag=result.dag,
            intervention=result.intervention,
            rollouts=result.rollouts,
            judge_results=result.judge_results,
            rewards=result.rewards,
            behavior_embeddings=np.zeros((0, int(encoder.cfg.dim)), dtype=np.float32),
            risk_features=[],
            novelty_surprisal=np.zeros((0,), dtype=np.float32),
            novelty_scores=np.zeros((0,), dtype=np.float32),
            cluster_ids=np.zeros((0,), dtype=np.int32),
            consensus_scores=np.zeros((0,), dtype=np.float32),
            diagnostics=diag,
            pipeline_result=result,
        )

    psi = np.stack(embeddings, axis=0).astype(np.float32)
    surprisal = novelty_estimator.score_batch(psi, update=bool(update_novelty)).astype(np.float32)

    cluster_ids, consensus_base, cluster_hist, _ = consensus_scorer.score(
        psi,
        risk_features,
        seed=int(seed),
    )
    eta, alpha, entropy = thermostat.compute(cluster_ids)

    s_mu = float(np.mean(surprisal))
    s_std = float(np.std(surprisal) + 1e-6)
    surprisal_norm = (surprisal - s_mu) / s_std
    novelty_scores = _softplus(float(eta) * surprisal_norm).astype(np.float32)
    # Consensus tilt with alpha (higher when entropy is above target).
    consensus_scores = (consensus_base * (1.0 + float(alpha))).astype(np.float32)

    for i, rollout in enumerate(result.rollouts):
        _update_rollout_metadata(
            rollout,
            risk_features=risk_features[i],
            behavior_embedding=psi[i],
            novelty_surprisal=float(surprisal[i]),
            novelty_score=float(novelty_scores[i]),
            cluster_id=int(cluster_ids[i]),
            consensus_score=float(consensus_scores[i]),
        )

    rewards = [
        compose_reward(j, r, pipeline.config.reward)
        for j, r in zip(result.judge_results, result.rollouts)
    ]
    result.rewards = rewards
    diag = RLBatchDiagnostics(
        entropy=float(entropy),
        cluster_hist=cluster_hist,
        thermostat_eta=float(eta),
        thermostat_alpha=float(alpha),
    )

    return GroupedRolloutBatch(
        scenario_id=result.scenario_id,
        features=result.features,
        dag=result.dag,
        intervention=result.intervention,
        rollouts=result.rollouts,
        judge_results=result.judge_results,
        rewards=rewards,
        behavior_embeddings=psi,
        risk_features=risk_features,
        novelty_surprisal=surprisal,
        novelty_scores=novelty_scores,
        cluster_ids=cluster_ids,
        consensus_scores=consensus_scores,
        diagnostics=diag,
        pipeline_result=result,
    )


def compute_group_advantages(batch: GroupedRolloutBatch) -> np.ndarray:
    return _compute_group_advantages(batch.reward_array)


def grpo_update(
    trainer: GRPOTrainer,
    batch: GroupedRolloutBatch,
    advantages: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if advantages is None:
        advantages = compute_group_advantages(batch)
    stats = trainer.update(
        rewards=batch.reward_array,
        advantages=np.asarray(advantages, dtype=np.float32),
        entropy=float(batch.diagnostics.entropy),
        alpha=float(batch.diagnostics.thermostat_alpha),
        eta=float(batch.diagnostics.thermostat_eta),
    )
    return stats


def summarize_reward_breakdown(rewards: Sequence[RewardBreakdown]) -> Dict[str, float]:
    if not rewards:
        return {"n": 0}
    total = np.asarray([float(r.total) for r in rewards], dtype=np.float32)
    env_total = np.asarray([float(r.total_env) for r in rewards], dtype=np.float32)
    novelty = np.asarray([float(r.novelty) for r in rewards], dtype=np.float32)
    consensus = np.asarray([float(r.consensus) for r in rewards], dtype=np.float32)
    return {
        "n": int(total.size),
        "total_mean": float(np.mean(total)),
        "total_std": float(np.std(total)),
        "total_env_mean": float(np.mean(env_total)),
        "novelty_mean": float(np.mean(novelty)),
        "consensus_mean": float(np.mean(consensus)),
    }
