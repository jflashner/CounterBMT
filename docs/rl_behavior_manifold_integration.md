# RL Behavior-Manifold Integration (CounterBMT v2)

## Purpose
Add a principled RL layer for trajectories that are both realistic and reliably diverse by shaping rewards in a learned behavior manifold `psi`.

## Integration Points
- Scene/DAG/rollout orchestration: `src/counter_bmt_v2/orchestration/pipeline.py`
- RL loop API: `src/counter_bmt_v2/rl/loop.py`
- Behavior manifold encoder: `src/counter_bmt_v2/rl/behavior_embedding.py`
- Novelty estimator: `src/counter_bmt_v2/rl/novelty.py`
- Consensus/cluster quality: `src/counter_bmt_v2/rl/consensus.py`
- Entropy thermostat: `src/counter_bmt_v2/rl/thermostat.py`
- GRPO update scaffold: `src/counter_bmt_v2/rl/grpo.py`
- Topology branch + cache: `src/counter_bmt_v2/rl/topology.py`
- RL CLI: `src/counter_bmt_v2/cli/train_rl_topo_mcpo.py`

## Psi Options
1. `risk_vector`
- What: scalar rollout risk/progress features projected to `psi`.
- Pros: fastest, interpretable, robust baseline.
- Cons: weak structural expressiveness.

2. `dag_gnn` (recommended default)
- What: attributed DAG + intervention + rollout risk fused into graph embedding.
- Pros: aligns with causal pipeline, captures decision structure.
- Cons: sensitive to DAG quality; moderate tuning required.

3. `topology_zpi`
- What: time-image behavior tensor encoded by zigzag-like/topology fallback descriptors.
- Pros: captures temporal shape/mode evolution; perturbation-robust signals.
- Cons: heaviest runtime + hardest debugging.

4. `hybrid`
- What: concat(`dag_gnn`, topology vector, risk vector) then projection.
- Pros: strongest representation capacity.
- Cons: highest complexity and more failure modes.

## Zigzag/ZPI Notes
- Current topology branch uses `BehaviorImageBuilder -> TopologyEmbeddingRunner`.
- `ZigzagTopologyEncoder` is a stable interface with fallback to PH-style proxy descriptors, so training remains dependency-light.
- Recommended rollout:
1. Run `dag_gnn` first to stabilize RL signals.
2. Enable `--use-topology-branch` and cached embeddings once reward trends are stable.
3. Compare `dag_gnn` vs `hybrid` with fixed seed and same scene pool.

## Reward Shape
- Environment reward: alignment + safety + realism.
- Augmented reward: env + novelty + consensus.
- Novelty uses manifold surprisal (`-log p_hat(psi)` proxy).
- Consensus uses cluster occupancy and quality proxy:
  - quality increases with progress and decreases with risk/violations.
- Thermostat adapts `eta`/`alpha` from behavior entropy toward target.

## Suggested Run Order
1. Baseline:
```bash
PYTHONPATH=src python -m counter_bmt_v2.cli.train_rl_topo_mcpo --steps 50 --embedding-mode risk_vector
```
2. Mainline:
```bash
PYTHONPATH=src python -m counter_bmt_v2.cli.train_rl_topo_mcpo --data-dir data/scenarionet_waymo_training_500 --embedding-mode dag_gnn --group-size 8
```
3. Topology ablation:
```bash
PYTHONPATH=src python -m counter_bmt_v2.cli.train_rl_topo_mcpo --data-dir data/scenarionet_waymo_training_500 --embedding-mode hybrid --use-topology-branch --topology-cache-dir outputs/topology_cache
```

