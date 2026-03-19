"""Train/evaluate RL behavior-manifold loop (Topo-MCPO mainline)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np

from counter_bmt_v2.causal import TopologicalDAGAssignmentSampler
from counter_bmt_v2.config import PipelineConfig
from counter_bmt_v2.contracts import ScenarioInput
from counter_bmt_v2.data import ScenarioNetNNXLoader
from counter_bmt_v2.orchestration import CounterBMTPipeline
from counter_bmt_v2.rl import (
    BehaviorManifoldEncoder,
    ConsensusScorer,
    EntropyThermostat,
    GRPOTrainer,
    NNXPolicyBackend,
    TopologyEmbeddingRunner,
    VLMAlignmentVerifier,
    build_novelty_estimator,
    collect_group_rollouts,
    compute_group_advantages,
    grpo_update,
    summarize_reward_breakdown,
)
from counter_bmt_v2.runtime_guards import collect_debug_violations, normalize_openai_backend, require_debug_fallbacks
from counter_bmt_v2.training.dag_sources import DAGSourceResolver


def _build_demo_scene(scenario_id: str, seed: int) -> ScenarioInput:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num=40, dtype=np.float32)
    x = 6.0 * t
    y = 0.3 * np.sin(2.0 * np.pi * t + float(rng.uniform(0.0, np.pi)))
    traj = np.stack([x, y], axis=1).astype(np.float32)
    return ScenarioInput(scenario_id=scenario_id, ego_trajectory_xy=traj, metadata={"source": "demo"})


def _scene_from_loader(loader: ScenarioNetNNXLoader, index: int) -> ScenarioInput:
    sample = loader.load(int(index))
    ego_xy = np.zeros((0, 2), dtype=np.float32)
    if sample.agent_position_xy.size > 0 and sample.agent_valid_mask.size > 0:
        pos = sample.agent_position_xy[:, 0, :]
        valid = sample.agent_valid_mask[:, 0]
        ego_xy = pos[valid].astype(np.float32)
        if ego_xy.shape[0] == 0:
            ego_xy = pos.astype(np.float32)

    return ScenarioInput(
        scenario_id=str(sample.scenario_id),
        ego_trajectory_xy=ego_xy,
        metadata={
            "source": "scenarionet",
            "loader_index": int(index),
            "dt_s": float(sample.dt_s),
            "nnx_sample": sample,
        },
    )


def _resolve_scene_indices(loader: ScenarioNetNNXLoader, *, max_scenes: int, seed: int) -> List[int]:
    indices = np.arange(len(loader), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    if max_scenes > 0:
        indices = indices[: int(max_scenes)]
    if indices.size == 0:
        indices = np.asarray([0], dtype=np.int64)
    return [int(i) for i in indices.tolist()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RL manifold loop for CounterBMT v2")
    p.add_argument("--data-dir", type=str, default="", help="ScenarioNet directory; if empty, uses demo scenes")
    p.add_argument("--output-dir", type=str, default="outputs/counter_bmt_v2_rl_topo_mcpo")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-scenes", type=int, default=512, help="scene pool size when using --data-dir")
    p.add_argument("--rare-prob", type=float, default=0.0)

    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--entropy-target", type=float, default=1.2)
    p.add_argument("--eta0", type=float, default=0.2)
    p.add_argument("--alpha0", type=float, default=0.3)
    p.add_argument("--k-eta", type=float, default=0.1)
    p.add_argument("--k-alpha", type=float, default=0.1)

    p.add_argument(
        "--embedding-mode",
        type=str,
        default="dag_gnn",
        choices=["risk_vector", "dag_gnn", "topology_zpi", "hybrid"],
    )
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--use-topology-branch", action="store_true")
    p.add_argument("--topology-cache-dir", type=str, default="outputs/topology_cache")
    p.add_argument("--novelty-density", type=str, default="ema_gaussian", choices=["ema_gaussian", "knn"])
    p.add_argument("--novelty-ema-decay", type=float, default=0.99)

    p.add_argument("--clusterer", type=str, default="kmeans", choices=["kmeans", "hdbscan"])
    p.add_argument("--k-clusters", type=int, default=4)

    p.add_argument("--w-alignment", type=float, default=0.7)
    p.add_argument("--w-safety", type=float, default=0.2)
    p.add_argument("--w-realism", type=float, default=0.1)
    p.add_argument("--w-novelty", type=float, default=0.05)
    p.add_argument("--w-consensus", type=float, default=0.10)

    p.add_argument("--alignment-source-mode", type=str, default="vlm_replace", choices=["judge", "vlm_replace"])
    p.add_argument("--vlm-alignment-enabled", action="store_true")
    p.add_argument("--no-vlm-alignment-enabled", dest="vlm_alignment_enabled", action="store_false")
    p.set_defaults(vlm_alignment_enabled=True)
    p.add_argument("--vlm-alignment-backend", type=str, default="openai", choices=["openai", "gpt4o", "mock"])
    p.add_argument("--vlm-alignment-model", type=str, default="gpt-5-mini")
    p.add_argument("--vlm-alignment-api-key", type=str, default=None)
    p.add_argument("--vlm-alignment-sample-rate", type=float, default=0.15)
    p.add_argument("--vlm-alignment-every-n-steps", type=int, default=5)
    p.add_argument("--vlm-alignment-max-calls-per-step", type=int, default=2)
    p.add_argument("--vlm-alignment-max-concurrency", type=int, default=2)
    p.add_argument("--vlm-alignment-timeout-sec", type=float, default=8.0)
    p.add_argument("--vlm-alignment-step-wait-budget-sec", type=float, default=6.0)
    p.add_argument("--vlm-alignment-neutral-score", type=float, default=0.0)
    p.add_argument("--vlm-alignment-cache-dir", type=str, default="outputs/rl_vlm_alignment_cache")
    p.add_argument("--vlm-alignment-save-evidence", action="store_true")
    p.add_argument("--no-vlm-alignment-save-evidence", dest="vlm_alignment_save_evidence", action="store_false")
    p.set_defaults(vlm_alignment_save_evidence=True)
    p.add_argument("--vlm-alignment-num-frames", type=int, default=6)
    p.add_argument("--vlm-alignment-max-agents-render", type=int, default=48)

    p.add_argument("--policy-backend", type=str, default="nnx_checkpoint", choices=["nnx_checkpoint", "scaffold"])
    p.add_argument("--policy-checkpoint", type=str, default="")
    p.add_argument("--policy-model-preset", type=str, default="")
    p.add_argument("--policy-tokenizer-mode", type=str, default="adv_bmt_parity")
    p.add_argument("--policy-skip-steps", type=int, default=5)
    p.add_argument("--dag-source-mode", type=str, default="dual", choices=["dual", "cache", "scene_derived"])
    p.add_argument("--dag-cache-dir", type=str, default="")
    p.add_argument("--dag-cache-strict", action="store_true")
    p.add_argument("--dag-expected-schema", type=str, default="any")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--policy-lr", type=float, default=1e-5)
    p.add_argument("--trainable-scope", type=str, default="decoder_dag", choices=["decoder_dag", "all"])
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--candidate-multiplier", type=int, default=2)
    p.add_argument("--feasible-max-speed-mps", type=float, default=40.0)
    p.add_argument("--feasible-max-accel-delta", type=float, default=4.0)
    p.add_argument("--feasible-max-yaw-delta", type=float, default=0.75)
    p.add_argument("--enable-feasibility-mask", action="store_true")
    p.add_argument("--no-enable-feasibility-mask", dest="enable_feasibility_mask", action="store_false")
    p.set_defaults(enable_feasibility_mask=True)
    p.add_argument("--store-rollout-traces", action="store_true")
    p.add_argument("--no-store-rollout-traces", dest="store_rollout_traces", action="store_false")
    p.set_defaults(store_rollout_traces=False)

    p.add_argument("--perception-backend", type=str, default="openai", choices=["mock", "openai", "gpt4o"])
    p.add_argument("--dag-backend", type=str, default="promptbn", choices=["simple", "promptbn"])
    p.add_argument("--llm-model", type=str, default="gpt-5-mini")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--dag-retries", type=int, default=4)
    p.add_argument("--allow-debug-fallbacks", action="store_true")
    p.add_argument("--no-allow-debug-fallbacks", dest="allow_debug_fallbacks", action="store_false")
    p.set_defaults(allow_debug_fallbacks=False)
    return p


def _validate_runtime_args(args: argparse.Namespace) -> None:
    args.perception_backend = normalize_openai_backend(
        str(args.perception_backend),
        field_name="perception_backend",
    )
    args.vlm_alignment_backend = normalize_openai_backend(
        str(args.vlm_alignment_backend),
        field_name="vlm_alignment_backend",
    )
    violations = collect_debug_violations(
        [
            ("policy_backend", str(args.policy_backend), str(args.policy_backend) == "scaffold"),
            ("alignment_source_mode", str(args.alignment_source_mode), str(args.alignment_source_mode) == "judge"),
            ("vlm_alignment_backend", str(args.vlm_alignment_backend), str(args.vlm_alignment_backend) == "mock"),
            ("perception_backend", str(args.perception_backend), str(args.perception_backend) == "mock"),
            ("dag_backend", str(args.dag_backend), str(args.dag_backend) == "simple"),
            ("vlm_alignment_enabled", "false", not bool(args.vlm_alignment_enabled)),
        ]
    )
    require_debug_fallbacks(
        allow_debug_fallbacks=bool(args.allow_debug_fallbacks),
        violations=violations,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_runtime_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    run_config_path = out_dir / "run_config.json"

    cfg = PipelineConfig()
    cfg.allow_debug_fallbacks = bool(args.allow_debug_fallbacks)
    cfg.reward.w_alignment = float(args.w_alignment)
    cfg.reward.w_safety = float(args.w_safety)
    cfg.reward.w_realism = float(args.w_realism)
    cfg.reward.w_novelty = float(args.w_novelty)
    cfg.reward.w_consensus = float(args.w_consensus)

    cfg.rl.train.group_size = int(args.group_size)
    cfg.rl.train.entropy_target = float(args.entropy_target)
    cfg.rl.train.eta0 = float(args.eta0)
    cfg.rl.train.alpha0 = float(args.alpha0)
    cfg.rl.train.k_eta = float(args.k_eta)
    cfg.rl.train.k_alpha = float(args.k_alpha)

    cfg.rl.embedding.mode = str(args.embedding_mode)
    cfg.rl.embedding.dim = int(args.embedding_dim)
    cfg.rl.embedding.use_topology_branch = bool(args.use_topology_branch)
    cfg.rl.novelty.density = str(args.novelty_density)
    cfg.rl.novelty.ema_decay = float(args.novelty_ema_decay)
    cfg.rl.consensus.clusterer = str(args.clusterer)
    cfg.rl.consensus.k_clusters = int(args.k_clusters)

    cfg.rl.policy.backend = str(args.policy_backend)  # type: ignore[assignment]
    cfg.rl.policy.checkpoint = str(args.policy_checkpoint)
    cfg.rl.policy.model_preset = str(args.policy_model_preset)
    cfg.rl.policy.tokenizer_mode = str(args.policy_tokenizer_mode)
    cfg.rl.policy.skip_steps = int(args.policy_skip_steps)
    cfg.rl.policy.dag_source_mode = str(args.dag_source_mode)  # type: ignore[assignment]
    cfg.rl.policy.dag_cache_dir = str(args.dag_cache_dir)
    cfg.rl.policy.dag_cache_strict = bool(args.dag_cache_strict)
    cfg.rl.policy.dag_expected_schema = str(args.dag_expected_schema)
    cfg.rl.policy.clip_eps = float(args.clip_eps)
    cfg.rl.policy.kl_beta = float(args.kl_beta)
    cfg.rl.policy.policy_lr = float(args.policy_lr)
    cfg.rl.policy.trainable_scope = str(args.trainable_scope)  # type: ignore[assignment]
    cfg.rl.policy.ppo_epochs = int(args.ppo_epochs)
    cfg.rl.policy.candidate_multiplier = int(args.candidate_multiplier)
    cfg.rl.policy.feasible_max_speed_mps = float(args.feasible_max_speed_mps)
    cfg.rl.policy.feasible_max_accel_delta = float(args.feasible_max_accel_delta)
    cfg.rl.policy.feasible_max_yaw_delta = float(args.feasible_max_yaw_delta)
    cfg.rl.policy.enable_feasibility_mask = bool(args.enable_feasibility_mask)
    cfg.rl.policy.store_rollout_traces = bool(args.store_rollout_traces)

    cfg.rl.vlm_alignment.enabled = bool(args.vlm_alignment_enabled)
    cfg.rl.vlm_alignment.source_mode = str(args.alignment_source_mode)  # type: ignore[assignment]
    cfg.rl.vlm_alignment.backend = str(args.vlm_alignment_backend)  # type: ignore[assignment]
    cfg.rl.vlm_alignment.model = str(args.vlm_alignment_model)
    cfg.rl.vlm_alignment.api_key = args.vlm_alignment_api_key
    cfg.rl.vlm_alignment.sample_rate = float(args.vlm_alignment_sample_rate)
    cfg.rl.vlm_alignment.every_n_steps = int(args.vlm_alignment_every_n_steps)
    cfg.rl.vlm_alignment.max_calls_per_step = int(args.vlm_alignment_max_calls_per_step)
    cfg.rl.vlm_alignment.max_concurrency = int(args.vlm_alignment_max_concurrency)
    cfg.rl.vlm_alignment.per_call_timeout_s = float(args.vlm_alignment_timeout_sec)
    cfg.rl.vlm_alignment.step_wait_budget_s = float(args.vlm_alignment_step_wait_budget_sec)
    cfg.rl.vlm_alignment.neutral_score = float(args.vlm_alignment_neutral_score)
    cfg.rl.vlm_alignment.cache_dir = str(args.vlm_alignment_cache_dir)
    cfg.rl.vlm_alignment.save_evidence_artifacts = bool(args.vlm_alignment_save_evidence)
    cfg.rl.vlm_alignment.num_frames = int(args.vlm_alignment_num_frames)
    cfg.rl.vlm_alignment.max_agents_render = int(args.vlm_alignment_max_agents_render)

    pipeline = CounterBMTPipeline.from_backends(
        config=cfg,
        perception_backend=args.perception_backend,
        dag_backend=args.dag_backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        dag_retries=int(args.dag_retries),
    )

    dag_resolver = DAGSourceResolver(
        mode=str(cfg.rl.policy.dag_source_mode),
        cache_dir=str(cfg.rl.policy.dag_cache_dir),
        cache_strict=bool(cfg.rl.policy.dag_cache_strict),
        expected_schema=str(cfg.rl.policy.dag_expected_schema),
    )
    policy_backend = None
    if str(cfg.rl.policy.backend) == "nnx_checkpoint":
        if not str(cfg.rl.policy.checkpoint).strip():
            raise ValueError("--policy-checkpoint is required for --policy-backend nnx_checkpoint")
        policy_backend = NNXPolicyBackend.from_checkpoint(cfg=cfg.rl.policy, seed=int(args.seed))
        pipeline.sampler = TopologicalDAGAssignmentSampler()

    topology_runner = TopologyEmbeddingRunner(
        out_dim=max(8, int(cfg.rl.embedding.dim) // 2),
        cache_dir=str(args.topology_cache_dir),
        prefer_zigzag=bool(args.use_topology_branch),
    )
    encoder = BehaviorManifoldEncoder(
        cfg=cfg.rl.embedding,
        topology_runner=topology_runner,
        dag_encoder_model=(policy_backend.model.dag_encoder if policy_backend is not None and getattr(policy_backend.model, "dag_encoder", None) is not None else None),
        dag_encoder_cfg=(policy_backend.model_cfg.dag_encoder if policy_backend is not None else None),
    )
    novelty = build_novelty_estimator(cfg.rl.novelty, dim=int(cfg.rl.embedding.dim))
    consensus = ConsensusScorer(cfg=cfg.rl.consensus)
    thermostat = EntropyThermostat.from_config(cfg.rl.train)
    trainer = GRPOTrainer(policy_backend=policy_backend)
    vlm_aligner = VLMAlignmentVerifier(
        cfg=cfg.rl.vlm_alignment,
        output_dir=out_dir,
        allow_debug_fallbacks=bool(cfg.allow_debug_fallbacks),
    )

    loader = None
    scene_indices: Sequence[int] = []
    if args.data_dir:
        loader = ScenarioNetNNXLoader(args.data_dir)
        scene_indices = _resolve_scene_indices(loader, max_scenes=int(args.max_scenes), seed=int(args.seed))
    elif policy_backend is not None:
        raise ValueError("NNX policy backend requires --data-dir so scenes have NNX tensors")

    vlm_alignment_cfg = asdict(cfg.rl.vlm_alignment)
    if vlm_alignment_cfg.get("api_key"):
        vlm_alignment_cfg["api_key"] = "***redacted***"

    run_config = {
        "args": vars(args),
        "resolved": {
            "scene_source": "scenarionet" if loader is not None else "demo",
            "scene_pool_size": int(len(scene_indices)) if loader is not None else 0,
            "allow_debug_fallbacks": bool(cfg.allow_debug_fallbacks),
            "vlm_alignment": vlm_alignment_cfg,
            "policy": asdict(cfg.rl.policy),
        },
    }
    if run_config["resolved"]["policy"].get("checkpoint"):
        run_config["resolved"]["policy"]["checkpoint"] = str(cfg.rl.policy.checkpoint)
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    rng = np.random.default_rng(int(args.seed))
    start = time.time()
    step_records: List[dict] = []
    with metrics_path.open("w", encoding="utf-8") as f:
        for step in range(1, int(args.steps) + 1):
            if loader is not None:
                idx = scene_indices[(step - 1) % len(scene_indices)]
                scene = _scene_from_loader(loader, idx)
            else:
                scene = _build_demo_scene(f"demo_{step:06d}", seed=int(args.seed) + step)

            rare = bool(rng.random() < float(args.rare_prob))
            batch = collect_group_rollouts(
                pipeline,
                scene,
                step=int(step),
                encoder=encoder,
                novelty_estimator=novelty,
                consensus_scorer=consensus,
                thermostat=thermostat,
                group_size=int(cfg.rl.train.group_size),
                dag_resolver=dag_resolver if policy_backend is not None else None,
                policy_backend=policy_backend,
                vlm_aligner=vlm_aligner,
                seed=int(args.seed) + step,
                rare=rare,
                update_novelty=True,
            )
            advantages = compute_group_advantages(batch)
            grpo_stats = grpo_update(trainer, batch, advantages)
            reward_stats = summarize_reward_breakdown(batch.rewards)
            policy_metrics = {str(k): float(v) for k, v in grpo_stats.items() if str(k).startswith("policy/")}
            grpo_only_metrics = {f"grpo/{k}": float(v) for k, v in grpo_stats.items() if not str(k).startswith("policy/")}

            rec = {
                "step": int(step),
                "scenario_id": str(batch.scenario_id),
                "rare": bool(rare),
                "metrics": {
                    **{f"reward/{k}": v for k, v in reward_stats.items()},
                    **grpo_only_metrics,
                    **policy_metrics,
                    "diagnostics/entropy": float(batch.diagnostics.entropy),
                    "diagnostics/eta": float(batch.diagnostics.thermostat_eta),
                    "diagnostics/alpha": float(batch.diagnostics.thermostat_alpha),
                    "diagnostics/num_clusters": float(len(batch.diagnostics.cluster_hist)),
                    **{k: float(v) for k, v in batch.diagnostics.extra_metrics.items()},
                    "alignment/vlm_mean": (
                        float(np.mean(batch.alignment_scores))
                        if batch.alignment_scores.size and float(batch.alignment_diagnostics.get("source_mode_vlm_replace", 0.0)) > 0.5
                        else 0.0
                    ),
                    "alignment/vlm_scored_fraction": float(np.mean(batch.alignment_scored_mask.astype(np.float32)))
                    if batch.alignment_scored_mask.size
                    else 0.0,
                    "alignment/judge_original_mean": float(
                        np.mean(
                            np.asarray(
                                [float(r.metadata.get("judge_reward_original", 0.0)) for r in batch.rollouts],
                                dtype=np.float32,
                            )
                        )
                    )
                    if batch.rollouts
                    else 0.0,
                    "alignment/source_mode_vlm_replace": float(batch.alignment_diagnostics.get("source_mode_vlm_replace", 0.0)),
                    "alignment/vlm_calls_attempted": float(batch.alignment_diagnostics.get("calls_attempted", 0.0)),
                    "alignment/vlm_calls_success": float(batch.alignment_diagnostics.get("calls_success", 0.0)),
                    "alignment/vlm_cache_hits": float(batch.alignment_diagnostics.get("cache_hits", 0.0)),
                    "alignment/vlm_timeouts": float(batch.alignment_diagnostics.get("timeouts", 0.0)),
                    "alignment/vlm_errors": float(batch.alignment_diagnostics.get("errors", 0.0)),
                    "alignment/vlm_latency_ms_mean": float(batch.alignment_diagnostics.get("latency_ms_mean", 0.0)),
                    "alignment/vlm_step_skipped": float(batch.alignment_diagnostics.get("step_skipped", 0.0)),
                },
                "cluster_hist": batch.diagnostics.cluster_hist,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            step_records.append(rec)

            if step % int(args.log_every) == 0 or step == 1:
                m = rec["metrics"]
                print(
                    "step={step} reward={reward:.4f} env={env:.4f} novelty={nov:.4f} "
                    "consensus={cons:.4f} entropy={h:.4f} eta={eta:.4f} alpha={alpha:.4f} "
                    "policy_loss={pl:.4f} kl={kl:.4f}".format(
                        step=step,
                        reward=float(m.get("reward/total_mean", 0.0)),
                        env=float(m.get("reward/total_env_mean", 0.0)),
                        nov=float(m.get("reward/novelty_mean", 0.0)),
                        cons=float(m.get("reward/consensus_mean", 0.0)),
                        h=float(m.get("diagnostics/entropy", 0.0)),
                        eta=float(m.get("diagnostics/eta", 0.0)),
                        alpha=float(m.get("diagnostics/alpha", 0.0)),
                        pl=float(m.get("policy/loss", 0.0)),
                        kl=float(m.get("policy/kl_ref", 0.0)),
                    )
                )

    elapsed = float(time.time() - start)
    summary = {
        "output_dir": str(out_dir),
        "steps": int(args.steps),
        "elapsed_seconds": elapsed,
        "metrics_path": str(metrics_path),
        "run_config_path": str(run_config_path),
        "last_record": step_records[-1] if step_records else {},
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
