# RL Behavior-Manifold Implementation Details (`counter_bmt_v2`)

## Scope
This document explains what was implemented for the RL behavior-manifold layer, how each piece maps to your provided papers, and what remains for full algorithmic parity.

Primary references:
- `NeurIPS_2026_Auto_Driving_RL.pdf` (Topo-MCPO framing: novelty tilt, quality-weighted consensus, entropy thermostat, group-relative optimization).
- `16261_TEN_DM_Topology_Enhanced (1).pdf` (graph + topology representation learning; zigzag persistence image / ZPI ideas).

---

## 1) What was implemented

### 1.1 Config and contract surface
- RL config blocks added in `src/counter_bmt_v2/config.py`:
  - `RLTrainConfig`, `BehaviorEmbeddingConfig`, `NoveltyConfig`, `ConsensusConfig`, `RLConfig`.
  - `PipelineConfig.rl` now carries RL settings.
- Reward terms extended in `src/counter_bmt_v2/config.py`:
  - `RewardConfig` now has `w_novelty` and `w_consensus`.
- RL outputs/contracts extended in `src/counter_bmt_v2/contracts/core.py`:
  - `RewardBreakdown` now includes `novelty`, `consensus`, `total_env`, `total_augmented`.
  - `RLBatchDiagnostics` added (`entropy`, `cluster_hist`, `thermostat_eta`, `thermostat_alpha`).
- Rollout metadata defaults extended in `src/counter_bmt_v2/trajectory_jax/model.py`:
  - `risk_features`, `behavior_embedding`, `novelty_score`, `cluster_id`, `consensus_score`.

### 1.2 RL modules
- Behavior embedding core: `src/counter_bmt_v2/rl/behavior_embedding.py`
  - Modes: `risk_vector`, `dag_gnn`, `topology_zpi`, `hybrid`.
- Topology branch/caching: `src/counter_bmt_v2/rl/topology.py`
  - `BehaviorImageBuilder`, `TopologyEmbeddingRunner`, zigzag backend slot with fallback.
- Novelty estimators: `src/counter_bmt_v2/rl/novelty.py`
  - `EMAGaussianNovelty`, `KNNNovelty`.
- Consensus scoring: `src/counter_bmt_v2/rl/consensus.py`
  - KMeans default, optional HDBSCAN.
- Entropy thermostat: `src/counter_bmt_v2/rl/thermostat.py`
  - Adaptive `eta` and `alpha`.
- GRPO helper: `src/counter_bmt_v2/rl/grpo.py`
  - Group-standardized advantages + statistics update scaffold.
- RL loop orchestrator: `src/counter_bmt_v2/rl/loop.py`
  - `collect_group_rollouts`, `compute_group_advantages`, `grpo_update`.

### 1.3 Reward composition and CLI
- Reward composition updated in `src/counter_bmt_v2/rl/reward.py`:
  - `total_augmented = total_env + w_novelty*novelty + w_consensus*consensus`.
- New RL training entrypoint:
  - `src/counter_bmt_v2/cli/train_rl_topo_mcpo.py`
  - Supports embedding mode, group size, thermostat params, clustering, novelty mode, topology cache, and reward weights.

---

## 2) Paper-to-code mapping

## 2.1 Mapping to Topo-MCPO paper (`NeurIPS_2026_Auto_Driving_RL.pdf`)

### A) Group sampling and behavior embedding
Paper concept:
- Sample grouped rollouts per scenario and embed behavior into `z(τ)=Ψ(...)`.
Code mapping:
- `collect_group_rollouts(...)` in `src/counter_bmt_v2/rl/loop.py` gathers `group_size` rollouts.
- `BehaviorManifoldEncoder.encode(...)` in `src/counter_bmt_v2/rl/behavior_embedding.py` computes `psi` per rollout.

### B) Novelty tilt
Paper concept:
- Novelty factor uses `-log p_hat(z)` and coefficient `eta`.
Code mapping:
- `EMAGaussianNovelty.score_batch(...)` in `src/counter_bmt_v2/rl/novelty.py` computes a diagonal-Gaussian surprisal proxy.
- `loop.py` normalizes surprisal per group and applies `softplus(eta * surprisal_norm)` as `novelty_score`.

### C) Quality-weighted consensus
Paper concept:
- Cluster mass times quality score (`rho(C|s) * Q(C)`).
Code mapping:
- `ConsensusScorer.score(...)` in `src/counter_bmt_v2/rl/consensus.py`:
  - clusters `psi`,
  - computes cluster mass (`rho`) and quality proxy from progress/risk/violations,
  - returns per-rollout consensus score.

### D) Entropy thermostat
Paper concept:
- `eta = eta0 + k_eta (H* - H)`, `alpha = alpha0 + k_alpha (H - H*)`.
Code mapping:
- Implemented directly in `EntropyThermostat.compute(...)` in `src/counter_bmt_v2/rl/thermostat.py`.

### E) Group-relative advantages
Paper concept:
- Standardized within-group returns.
Code mapping:
- Implemented in `compute_group_advantages(...)` in `src/counter_bmt_v2/rl/grpo.py`.

### F) PPO/GRPO clipped optimization
Paper concept:
- Clipped surrogate + optional KL regularization.
Code status:
- Not fully implemented yet.
- `GRPOTrainer.update(...)` currently logs surrogate-style stats and entropy term but does not update a learned policy head.

### G) Feasibility-constrained sampling
Paper concept:
- Hard feasibility support `F(x_t,u_t)` during sampling.
Code status:
- Not implemented yet as hard constraints in trajectory sampling.
- Current path uses reward shaping and rollout scoring, not constrained action support.

---

## 2.2 Mapping to TEN-DM topology paper (`16261_TEN_DM_Topology_Enhanced (1).pdf`)

Paper themes used:
- Graph abstraction for structural dependencies.
- Time-series image conversion for topology.
- Zigzag persistence and vectorization (ZPI).
- Multi-scale topological fusion.

Code mapping:
- Graph branch: approximated in `dag_gnn` encoder (`behavior_embedding.py`).
- Topology branch: interface-ready path in `topology.py`:
  - time-image builder,
  - topology encoder protocol,
  - optional zigzag backend hook,
  - caching for expensive topology features.

Important distinction:
- TEN-DM describes full cubical zigzag persistence + ZPI vectorization.
- Current code provides a production-safe placeholder/fallback (`PHPersistenceFallbackEncoder`) and a drop-in slot (`ZigzagTopologyEncoder`) for full ZPI integration.

---

## 3) Deep dive: `dag_gnn` implementation

Implemented in `src/counter_bmt_v2/rl/behavior_embedding.py` (`_dag_graph_embedding`).

### 3.1 Inputs
- Sampled DAG:
  - nodes, edges, node types, node values, timestamps.
- Intervention:
  - selected variable/value.
- Rollout risk vector:
  - progress/path length/speed/acc/jerk/turn-rate/stop ratio/risk proxies.

### 3.2 Node features
Each node gets an 11D feature:
- 4D node type one-hot: `ego_state`, `maneuver`, `decision`, `outcome`.
- 2D normalized in-degree/out-degree.
- 1D timestamp.
- 4D hashed text/value embedding.

### 3.3 Graph propagation
- Directed adjacency from DAG edges.
- Self-loop added.
- Row-normalized adjacency.
- Two lightweight message-passing blends:
  - linear blend with neighbors,
  - tanh blend with neighbors.
- Mean pooling over nodes.

### 3.4 Fusion into final `psi`
- Pooled DAG features + risk vector + intervention hash are concatenated.
- Deterministic random projection (`_stable_project`) maps to configured embedding dim.

### 3.5 Why this design
Pros:
- Works with current DAG contracts and no heavy dependencies.
- Stable and deterministic for reproducibility.
- Encodes causal structure and intervention context, not only trajectory geometry.

Limitations:
- Not a trainable GNN yet (projection is fixed random).
- No edge feature learning.
- No batching optimizations for large graphs.

Recommended next upgrade:
- Replace `_stable_project` with trainable MLP/GAT layers in JAX/NNX or PyTorch.
- Add edge attributes (mechanism/confidence/CPT-derived statistics).

---

## 4) Deep dive: ZPI/topology branch

Implemented mainly in `src/counter_bmt_v2/rl/topology.py`.

### 4.1 Current data path
1. `BehaviorImageBuilder.build(...)` converts rollout trajectory into `[T,H,W,C]` image sequence:
   - occupancy channel,
   - speed proxy,
   - curvature proxy.
2. `TopologyEmbeddingRunner.encode(...)`:
   - computes payload hash,
   - loads cached embedding if unchanged,
   - otherwise runs selected topology encoder and saves `.npz + .json`.

### 4.2 Current encoder behavior
- `ZigzagTopologyEncoder` currently acts as interface gate.
- If true zigzag backend unavailable, it falls back to `PHPersistenceFallbackEncoder`.
- Fallback encoder computes temporal shape summaries (occupancy count trajectory, first/second differences, speed/curvature moments), then deterministic projection.

### 4.3 Relation to ZPI concept in paper
Paper concept:
- time-series images -> cubical zigzag persistence diagrams -> vectorization (ZPI) -> learned downstream features.

Current approximation:
- time-series images are implemented.
- persistence-computation and ZPI vectorization are not yet fully implemented.
- API and cache path are already structured so true ZPI can replace fallback with minimal changes to RL loop.

### 4.4 Exact implementation path to true ZPI
To match the paper more closely:
1. Build multi-scale windows/patches of time-image sequence.
2. For each scale, compute cubical zigzag persistence diagrams (0D and 1D).
3. Vectorize diagrams into persistence images (ZPI).
4. Aggregate multi-scale ZPIs (`beta_q` weighted mix).
5. Encode via CNN/MLP into topology embedding.
6. Return same interface shape so `BehaviorManifoldEncoder` hybrid path remains unchanged.

---

## 5) End-to-end RL flow in this repo

1. Pipeline generates rollouts (`CounterBMTPipeline.run`, `n_samples=group_size`).
2. Each rollout gets `psi` embedding + risk features.
3. Novelty estimator computes surprisal in `psi` space.
4. Consensus scorer clusters `psi` and computes quality-gated consensus score.
5. Thermostat computes adaptive `eta`/`alpha` from group entropy.
6. Rollout metadata updated:
   - `risk_features`, `behavior_embedding`, `novelty_score`, `cluster_id`, `consensus_score`.
7. Reward recomputed with augmented terms.
8. Group advantages computed and GRPO statistics updated.
9. Metrics logged by `train_rl_topo_mcpo.py` (`metrics.jsonl`, `summary.json`).

---

## 6) What is implemented vs. deferred

Implemented:
- Behavior-manifold scaffolding and APIs.
- DAG-aware embedding branch.
- Topology branch with cache + fallback.
- Novelty/consensus/thermostat mechanics.
- Group-relative advantage computation.
- RL training CLI and diagnostics.

Deferred:
- True clipped PPO/GRPO policy parameter updates.
- Hard feasibility support in sampling.
- Full cubical zigzag persistence + ZPI vectorization backend.
- Unified LLM+trajectory conditioning head.

---

## 7) Practical usage

Baseline (`dag_gnn`):
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_dag \
  --steps 200 --group-size 8 --embedding-mode dag_gnn
```

Hybrid with topology branch:
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_hybrid \
  --steps 200 --group-size 8 \
  --embedding-mode hybrid --use-topology-branch \
  --topology-cache-dir outputs/topology_cache
```

Ablation suggestions:
- `risk_vector` vs `dag_gnn` vs `hybrid` at fixed seed/scene pool.
- novelty estimator: `ema_gaussian` vs `knn`.
- consensus clusterer: `kmeans` vs `hdbscan` (if available).

---

## 8) Recommended next engineering steps

1. Replace `GRPOTrainer` scaffold with real policy optimization on trajectory/LLM policy parameters.
2. Add feasibility checks directly into rollout/action proposal layer.
3. Implement a real zigzag/ZPI backend behind `ZigzagTopologyEncoder`.
4. Add unit tests for:
   - entropy thermostat directionality,
   - consensus quality score monotonicity,
   - novelty estimator update behavior,
   - cache correctness in topology runner.
5. Add offline evaluation script comparing diversity-realism tradeoff curves across embedding modes.

