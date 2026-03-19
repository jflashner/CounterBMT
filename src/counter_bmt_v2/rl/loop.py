"""Grouped rollout collection + GRPO utilities for CounterBMT v2 RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from counter_bmt_v2.causal import apply_intervention_assignments, payload_to_bayesian_dag
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
from counter_bmt_v2.rl.consensus import ConsensusScorer, mean_cluster_quality
from counter_bmt_v2.rl.grpo import GRPOTrainer, compute_group_advantages as _compute_group_advantages
from counter_bmt_v2.rl.nnx_policy import NNXPolicyBackend
from counter_bmt_v2.rl.novelty import NoveltyEstimator
from counter_bmt_v2.rl.reward import compose_reward
from counter_bmt_v2.rl.thermostat import EntropyThermostat
from counter_bmt_v2.rl.vlm_alignment import VLMAlignmentVerifier
from counter_bmt_v2.training.dag_cache_schema import dag_to_cache_payload
from counter_bmt_v2.training.dag_sources import DAGSourceResolver

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
    alignment_scores: np.ndarray
    alignment_scored_mask: np.ndarray
    alignment_diagnostics: Dict[str, float]
    diagnostics: RLBatchDiagnostics
    pipeline_result: PipelineResult
    policy_rollout_data: Any | None = None

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


def _top_surprisal_resample(
    *,
    surprisal: np.ndarray,
    group_size: int,
    eta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(surprisal.shape[0])
    if n <= group_size:
        idx = np.arange(n, dtype=np.int32)
        weights = np.ones((n,), dtype=np.float32) / max(1.0, float(n))
        ranks = np.argsort(np.argsort(-surprisal)).astype(np.int32)
        return idx, weights, ranks

    mu = float(np.mean(surprisal))
    std = float(np.std(surprisal) + 1e-6)
    norm = (surprisal - mu) / std
    weights = np.exp(float(eta) * norm).astype(np.float32)
    weights_sum = float(np.sum(weights))
    if weights_sum <= 0.0:
        weights = np.ones((n,), dtype=np.float32) / float(n)
    else:
        weights = weights / weights_sum
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(n, size=int(group_size), replace=False, p=weights).astype(np.int32)
    ranks = np.argsort(np.argsort(-surprisal)).astype(np.int32)
    return np.sort(selected), weights.astype(np.float32), ranks


def _resolve_dag_payload_for_scene(
    *,
    scene: ScenarioInput,
    dag_resolver: DAGSourceResolver,
) -> tuple[Dict[str, Any], str]:
    sample = scene.metadata.get("nnx_sample") if isinstance(scene.metadata, dict) else None
    if sample is None:
        raise ValueError("NNX RL backend requires ScenarioInput.metadata['nnx_sample']")
    batch_slice = {
        "scenario_id": str(getattr(sample, "scenario_id", scene.scenario_id)),
        "dt_s": float(getattr(sample, "dt_s", 0.1)),
        "agent_ids": np.asarray(sample.agent_ids)[None, ...],
        "agent_type_ids": np.asarray(sample.agent_type_ids)[None, ...],
        "agent_shape": np.asarray(sample.agent_shape)[None, ...],
        "agent_position_xy": np.asarray(sample.agent_position_xy)[None, ...],
        "agent_heading": np.asarray(sample.agent_heading)[None, ...],
        "agent_velocity_xy": np.asarray(sample.agent_velocity_xy)[None, ...],
        "agent_valid_mask": np.asarray(sample.agent_valid_mask)[None, ...],
    }
    dag_payload, source = dag_resolver.resolve_one(
        scenario_id=str(scene.scenario_id),
        batch_slice=batch_slice,
        sample_index=0,
    )
    if dag_payload is None:
        raise ValueError(f"DAG resolution failed for RL scene={scene.scenario_id} source={source}")
    return dict(dag_payload), str(source)


def _score_alignment_and_rewards(
    *,
    pipeline: "CounterBMTPipeline",
    scene: ScenarioInput,
    dag: BayesianDAG,
    intervention: Intervention,
    rollouts: Sequence[TrajectoryRollout],
    judge_results: Sequence[JudgeResult],
    step: int,
    vlm_aligner: Optional[VLMAlignmentVerifier],
) -> tuple[List[JudgeResult], List[RewardBreakdown], np.ndarray, np.ndarray, Dict[str, float]]:
    alignment_scores = np.asarray([float(j.reward) for j in judge_results], dtype=np.float32)
    alignment_scored_mask = np.zeros((len(rollouts),), dtype=bool)
    alignment_diag: Dict[str, float] = {"source_mode_vlm_replace": 0.0, "step_skipped": 1.0}
    replaced = list(judge_results)
    if (
        vlm_aligner is not None
        and bool(pipeline.config.rl.vlm_alignment.enabled)
        and str(pipeline.config.rl.vlm_alignment.source_mode) == "vlm_replace"
    ):
        alignment_result = vlm_aligner.score_rollouts(
            step=int(step),
            scenario=scene,
            dag=dag,
            intervention=intervention,
            rollouts=list(rollouts),
        )
        alignment_scores = np.asarray(alignment_result.scores, dtype=np.float32)
        alignment_scored_mask = np.asarray(alignment_result.scored_mask, dtype=bool)
        alignment_diag = dict(alignment_result.diagnostics)
        alignment_diag["source_mode_vlm_replace"] = 1.0
        alignment_diag.setdefault("step_skipped", 0.0)
        threshold = float(pipeline.config.rl.vlm_alignment.match_threshold)
        replaced = []
        for i, j in enumerate(judge_results):
            sc = float(alignment_scores[i]) if i < alignment_scores.shape[0] else 0.0
            details = dict(j.details) if isinstance(j.details, dict) else {}
            details["judge_reward_original"] = float(j.reward)
            details["vlm_alignment_score"] = float(sc)
            details["vlm_alignment_scored"] = bool(alignment_scored_mask[i]) if i < alignment_scored_mask.shape[0] else False
            replaced.append(
                JudgeResult(
                    reward=float(sc),
                    matched=bool(sc >= threshold),
                    explanation="vlm_replace alignment",
                    details=details,
                )
            )
        for i, rollout in enumerate(rollouts):
            rollout.metadata["vlm_dag_conformance_score"] = float(alignment_scores[i]) if i < alignment_scores.shape[0] else 0.0
            rollout.metadata["vlm_dag_conformance_scored"] = bool(alignment_scored_mask[i]) if i < alignment_scored_mask.shape[0] else False
    rewards = [compose_reward(j, r, pipeline.config.reward) for j, r in zip(replaced, rollouts)]
    return replaced, rewards, alignment_scores, alignment_scored_mask, alignment_diag


def _collect_group_rollouts_nnx(
    pipeline: "CounterBMTPipeline",
    scene: ScenarioInput,
    *,
    step: int,
    encoder: BehaviorManifoldEncoder,
    novelty_estimator: NoveltyEstimator,
    consensus_scorer: ConsensusScorer,
    thermostat: EntropyThermostat,
    group_size: int,
    dag_resolver: DAGSourceResolver,
    policy_backend: NNXPolicyBackend,
    vlm_aligner: Optional[VLMAlignmentVerifier],
    seed: int,
    rare: bool,
    update_novelty: bool,
) -> GroupedRolloutBatch:
    base_payload, dag_source = _resolve_dag_payload_for_scene(scene=scene, dag_resolver=dag_resolver)
    base_dag = payload_to_bayesian_dag(base_payload)
    intervention = pipeline.sampler.sample(base_dag, rare=bool(rare), seed=int(seed))
    sampled_dag = apply_intervention_assignments(base_dag, intervention)
    sampled_dag_payload = dag_to_cache_payload(sampled_dag)

    candidate_multiplier = max(1, int(pipeline.config.rl.policy.candidate_multiplier))
    candidate_count = max(int(group_size), candidate_multiplier * int(group_size))
    pool = policy_backend.sample_candidate_pool(
        scene=scene,
        sampled_dag=sampled_dag,
        sampled_dag_payload=sampled_dag_payload,
        n_samples=int(candidate_count),
        seed=int(seed),
        conditioning_metadata={
            "assignments": dict(intervention.assignments),
            "assignment_order": list(intervention.assignment_order),
            "source_dag_schema": str(intervention.source_dag_schema),
            "variable": intervention.variable,
            "value": intervention.value,
        },
    )

    embeddings: List[np.ndarray] = []
    risk_features: List[Dict[str, float]] = []
    for i, rollout in enumerate(pool.rollouts):
        emb, risk, _ = encoder.encode(
            dag=sampled_dag,
            intervention=intervention,
            rollout=rollout,
            scenario_id=scene.scenario_id,
            rollout_id=f"step{step}_candidate{i}",
        )
        embeddings.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        risk_features.append({k: float(v) for k, v in risk.items()})
    psi_all = np.stack(embeddings, axis=0).astype(np.float32)
    surprisal_all = novelty_estimator.score_batch(psi_all, update=False).astype(np.float32)

    candidate_cluster_ids, _, _, _ = consensus_scorer.score(psi_all, risk_features, seed=int(seed))
    sampling_eta, _, _ = thermostat.compute(candidate_cluster_ids)
    selected_idx, novelty_weights, novelty_ranks = _top_surprisal_resample(
        surprisal=surprisal_all,
        group_size=int(group_size),
        eta=float(sampling_eta),
        seed=int(seed) + 17,
    )

    rollouts = [pool.rollouts[int(i)] for i in selected_idx.tolist()]
    psi = psi_all[selected_idx]
    surprisal = surprisal_all[selected_idx]
    risk_features_sel = [risk_features[int(i)] for i in selected_idx.tolist()]
    policy_rollout_data = policy_backend.select_rollout_data(pool, selected_idx.tolist())

    if bool(update_novelty) and psi.shape[0] > 0:
        novelty_estimator.score_batch(psi, update=True)

    cluster_ids, consensus_base, cluster_hist, quality_scores = consensus_scorer.score(
        psi,
        risk_features_sel,
        seed=int(seed),
    )
    eta, alpha, entropy = thermostat.compute(cluster_ids)
    s_mu = float(np.mean(surprisal))
    s_std = float(np.std(surprisal) + 1e-6)
    surprisal_norm = (surprisal - s_mu) / s_std
    novelty_scores = _softplus(float(eta) * surprisal_norm).astype(np.float32)
    consensus_scores = (consensus_base * (1.0 + float(alpha))).astype(np.float32)

    judge_results = [pipeline.judge.evaluate(intervention, rollout) for rollout in rollouts]
    judge_original = [float(j.reward) for j in judge_results]
    for i, rollout in enumerate(rollouts):
        _update_rollout_metadata(
            rollout,
            risk_features=risk_features_sel[i],
            behavior_embedding=psi[i],
            novelty_surprisal=float(surprisal[i]),
            novelty_score=float(novelty_scores[i]),
            cluster_id=int(cluster_ids[i]),
            consensus_score=float(consensus_scores[i]),
        )
        rollout.metadata["judge_reward_original"] = float(judge_original[i])

    judge_results, rewards, alignment_scores, alignment_scored_mask, alignment_diag = _score_alignment_and_rewards(
        pipeline=pipeline,
        scene=scene,
        dag=sampled_dag,
        intervention=intervention,
        rollouts=rollouts,
        judge_results=judge_results,
        step=int(step),
        vlm_aligner=vlm_aligner,
    )

    extra_metrics = {
        "sampling/candidate_pool_size": float(candidate_count),
        "sampling/selected_rank_mean": float(np.mean(novelty_ranks[selected_idx])) if selected_idx.size else 0.0,
        "sampling/novelty_weight_mean": float(np.mean(novelty_weights[selected_idx])) if selected_idx.size else 0.0,
        "sampling/feasibility_mask_rate": float(np.mean(policy_rollout_data.feasibility_mask_rate)) if policy_rollout_data.feasibility_mask_rate.size else 0.0,
        "consensus/mean_cluster_quality": mean_cluster_quality(cluster_ids, quality_scores),
        "novelty/surprisal_mean": float(np.mean(surprisal)) if surprisal.size else 0.0,
        "dag/source_cache": 1.0 if dag_source == "cache" else 0.0,
        "dag/source_scene_derived": 1.0 if dag_source == "scene_derived" else 0.0,
    }
    diag = RLBatchDiagnostics(
        entropy=float(entropy),
        cluster_hist=cluster_hist,
        thermostat_eta=float(eta),
        thermostat_alpha=float(alpha),
        extra_metrics=extra_metrics,
    )
    features = VLMFeatures(scenario_id=scene.scenario_id, raw={"dag_source": dag_source, "policy_backend": "nnx_checkpoint"})
    result = PipelineResult(
        scenario_id=scene.scenario_id,
        features=features,
        dag=sampled_dag,
        intervention=intervention,
        rollouts=list(rollouts),
        judge_results=list(judge_results),
        rewards=list(rewards),
    )
    return GroupedRolloutBatch(
        scenario_id=scene.scenario_id,
        features=features,
        dag=sampled_dag,
        intervention=intervention,
        rollouts=list(rollouts),
        judge_results=list(judge_results),
        rewards=list(rewards),
        behavior_embeddings=psi,
        risk_features=risk_features_sel,
        novelty_surprisal=surprisal,
        novelty_scores=novelty_scores,
        cluster_ids=cluster_ids,
        consensus_scores=consensus_scores,
        alignment_scores=alignment_scores,
        alignment_scored_mask=alignment_scored_mask,
        alignment_diagnostics=alignment_diag,
        diagnostics=diag,
        pipeline_result=result,
        policy_rollout_data=policy_rollout_data,
    )


def collect_group_rollouts(
    pipeline: "CounterBMTPipeline",
    scene: ScenarioInput,
    *,
    step: int,
    encoder: BehaviorManifoldEncoder,
    novelty_estimator: NoveltyEstimator,
    consensus_scorer: ConsensusScorer,
    thermostat: EntropyThermostat,
    group_size: int,
    dag_resolver: Optional[DAGSourceResolver] = None,
    policy_backend: Optional[NNXPolicyBackend] = None,
    vlm_aligner: Optional[VLMAlignmentVerifier] = None,
    seed: int = 0,
    rare: bool = False,
    update_novelty: bool = True,
) -> GroupedRolloutBatch:
    """Collect one grouped rollout batch and annotate manifold statistics."""
    if policy_backend is not None:
        if dag_resolver is None:
            raise ValueError("NNX RL backend requires dag_resolver")
        return _collect_group_rollouts_nnx(
            pipeline,
            scene,
            step=int(step),
            encoder=encoder,
            novelty_estimator=novelty_estimator,
            consensus_scorer=consensus_scorer,
            thermostat=thermostat,
            group_size=int(group_size),
            dag_resolver=dag_resolver,
            policy_backend=policy_backend,
            vlm_aligner=vlm_aligner,
            seed=int(seed),
            rare=bool(rare),
            update_novelty=bool(update_novelty),
        )

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
            alignment_scores=np.zeros((0,), dtype=np.float32),
            alignment_scored_mask=np.zeros((0,), dtype=bool),
            alignment_diagnostics={"step_skipped": 1.0},
            diagnostics=diag,
            pipeline_result=result,
        )

    psi = np.stack(embeddings, axis=0).astype(np.float32)
    surprisal = novelty_estimator.score_batch(psi, update=bool(update_novelty)).astype(np.float32)

    cluster_ids, consensus_base, cluster_hist, quality_scores = consensus_scorer.score(
        psi,
        risk_features,
        seed=int(seed),
    )
    eta, alpha, entropy = thermostat.compute(cluster_ids)

    s_mu = float(np.mean(surprisal))
    s_std = float(np.std(surprisal) + 1e-6)
    surprisal_norm = (surprisal - s_mu) / s_std
    novelty_scores = _softplus(float(eta) * surprisal_norm).astype(np.float32)
    consensus_scores = (consensus_base * (1.0 + float(alpha))).astype(np.float32)

    judge_original = [float(j.reward) for j in result.judge_results]
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
        rollout.metadata["judge_reward_original"] = float(judge_original[i]) if i < len(judge_original) else 0.0

    judge_results, rewards, alignment_scores, alignment_scored_mask, alignment_diag = _score_alignment_and_rewards(
        pipeline=pipeline,
        scene=scene,
        dag=result.dag,
        intervention=result.intervention,
        rollouts=result.rollouts,
        judge_results=result.judge_results,
        step=int(step),
        vlm_aligner=vlm_aligner,
    )
    result.judge_results = judge_results
    result.rewards = rewards
    diag = RLBatchDiagnostics(
        entropy=float(entropy),
        cluster_hist=cluster_hist,
        thermostat_eta=float(eta),
        thermostat_alpha=float(alpha),
        extra_metrics={
            "sampling/candidate_pool_size": float(group_size),
            "sampling/selected_rank_mean": 0.0,
            "sampling/novelty_weight_mean": 0.0,
            "sampling/feasibility_mask_rate": 0.0,
            "consensus/mean_cluster_quality": mean_cluster_quality(cluster_ids, quality_scores),
            "novelty/surprisal_mean": float(np.mean(surprisal)) if surprisal.size else 0.0,
        },
    )

    return GroupedRolloutBatch(
        scenario_id=result.scenario_id,
        features=result.features,
        dag=result.dag,
        intervention=result.intervention,
        rollouts=result.rollouts,
        judge_results=judge_results,
        rewards=rewards,
        behavior_embeddings=psi,
        risk_features=risk_features,
        novelty_surprisal=surprisal,
        novelty_scores=novelty_scores,
        cluster_ids=cluster_ids,
        consensus_scores=consensus_scores,
        alignment_scores=alignment_scores,
        alignment_scored_mask=alignment_scored_mask,
        alignment_diagnostics=alignment_diag,
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
        policy_batch=batch.policy_rollout_data,
    )
    return stats


def summarize_reward_breakdown(rewards: Sequence[RewardBreakdown]) -> Dict[str, float]:
    if not rewards:
        return {"n": 0}
    total = np.asarray([float(r.total) for r in rewards], dtype=np.float32)
    env_total = np.asarray([float(r.total_env) for r in rewards], dtype=np.float32)
    alignment = np.asarray([float(r.alignment) for r in rewards], dtype=np.float32)
    safety = np.asarray([float(r.safety) for r in rewards], dtype=np.float32)
    realism = np.asarray([float(r.realism) for r in rewards], dtype=np.float32)
    novelty = np.asarray([float(r.novelty) for r in rewards], dtype=np.float32)
    consensus = np.asarray([float(r.consensus) for r in rewards], dtype=np.float32)
    vlm_align = np.asarray([float(r.vlm_dag_conformance) for r in rewards], dtype=np.float32)
    return {
        "n": int(total.size),
        "total_mean": float(np.mean(total)),
        "total_std": float(np.std(total)),
        "total_env_mean": float(np.mean(env_total)),
        "alignment_mean": float(np.mean(alignment)),
        "safety_mean": float(np.mean(safety)),
        "realism_mean": float(np.mean(realism)),
        "novelty_mean": float(np.mean(novelty)),
        "consensus_mean": float(np.mean(consensus)),
        "vlm_dag_conformance_mean": float(np.mean(vlm_align)),
    }
