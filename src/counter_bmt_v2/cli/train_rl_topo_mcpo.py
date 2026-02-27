"""Train/evaluate RL behavior-manifold loop (Topo-MCPO style scaffolding)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from counter_bmt_v2.config import PipelineConfig
from counter_bmt_v2.contracts import ScenarioInput
from counter_bmt_v2.data import ScenarioNetNNXLoader
from counter_bmt_v2.orchestration import CounterBMTPipeline
from counter_bmt_v2.rl import (
    BehaviorManifoldEncoder,
    ConsensusScorer,
    EntropyThermostat,
    GRPOTrainer,
    TopologyEmbeddingRunner,
    build_novelty_estimator,
    collect_group_rollouts,
    compute_group_advantages,
    grpo_update,
    summarize_reward_breakdown,
)


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

    p.add_argument("--perception-backend", type=str, default="mock", choices=["mock", "gpt4o"])
    p.add_argument("--dag-backend", type=str, default="simple", choices=["simple", "promptbn"])
    p.add_argument("--llm-model", type=str, default="gpt-4o")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--dag-retries", type=int, default=4)
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    run_config_path = out_dir / "run_config.json"

    cfg = PipelineConfig()
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

    pipeline = CounterBMTPipeline.from_backends(
        config=cfg,
        perception_backend=args.perception_backend,
        dag_backend=args.dag_backend,
        llm_model=args.llm_model,
        api_key=args.api_key,
        dag_retries=int(args.dag_retries),
    )

    topology_runner = TopologyEmbeddingRunner(
        out_dim=max(8, int(cfg.rl.embedding.dim) // 2),
        cache_dir=str(args.topology_cache_dir),
        prefer_zigzag=bool(args.use_topology_branch),
    )
    encoder = BehaviorManifoldEncoder(cfg=cfg.rl.embedding, topology_runner=topology_runner)
    novelty = build_novelty_estimator(cfg.rl.novelty, dim=int(cfg.rl.embedding.dim))
    consensus = ConsensusScorer(cfg=cfg.rl.consensus)
    thermostat = EntropyThermostat.from_config(cfg.rl.train)
    trainer = GRPOTrainer()

    loader = None
    scene_indices: Sequence[int] = []
    if args.data_dir:
        loader = ScenarioNetNNXLoader(args.data_dir)
        scene_indices = _resolve_scene_indices(loader, max_scenes=int(args.max_scenes), seed=int(args.seed))

    run_config = {
        "args": vars(args),
        "resolved": {
            "scene_source": "scenarionet" if loader is not None else "demo",
            "scene_pool_size": int(len(scene_indices)) if loader is not None else 0,
        },
    }
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
                encoder=encoder,
                novelty_estimator=novelty,
                consensus_scorer=consensus,
                thermostat=thermostat,
                group_size=int(cfg.rl.train.group_size),
                seed=int(args.seed) + step,
                rare=rare,
                update_novelty=True,
            )
            advantages = compute_group_advantages(batch)
            grpo_stats = grpo_update(trainer, batch, advantages)
            reward_stats = summarize_reward_breakdown(batch.rewards)

            rec = {
                "step": int(step),
                "scenario_id": str(batch.scenario_id),
                "rare": bool(rare),
                "metrics": {
                    **{f"reward/{k}": v for k, v in reward_stats.items()},
                    **{f"grpo/{k}": float(v) for k, v in grpo_stats.items()},
                    "diagnostics/entropy": float(batch.diagnostics.entropy),
                    "diagnostics/eta": float(batch.diagnostics.thermostat_eta),
                    "diagnostics/alpha": float(batch.diagnostics.thermostat_alpha),
                    "diagnostics/num_clusters": float(len(batch.diagnostics.cluster_hist)),
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
                    "consensus={cons:.4f} entropy={h:.4f} eta={eta:.4f} alpha={alpha:.4f}".format(
                        step=step,
                        reward=float(m.get("reward/total_mean", 0.0)),
                        env=float(m.get("reward/total_env_mean", 0.0)),
                        nov=float(m.get("reward/novelty_mean", 0.0)),
                        cons=float(m.get("reward/consensus_mean", 0.0)),
                        h=float(m.get("diagnostics/entropy", 0.0)),
                        eta=float(m.get("diagnostics/eta", 0.0)),
                        alpha=float(m.get("diagnostics/alpha", 0.0)),
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

