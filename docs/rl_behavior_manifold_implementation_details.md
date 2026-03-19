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
- NNX checkpoint policy backend: `src/counter_bmt_v2/rl/nnx_policy.py`
  - Loads supervised DAG-latent checkpoints, samples rollouts, applies feasibility masking, and performs clipped PPO-style updates with frozen-reference KL.
- RL loop orchestrator: `src/counter_bmt_v2/rl/loop.py`
  - candidate oversampling, novelty-weighted resampling, consensus scoring, alignment replacement, and reward assembly.

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
Code mapping:
- `NNXPolicyBackend.update(...)` in `src/counter_bmt_v2/rl/nnx_policy.py`:
  - replays sampled ego token sequences,
  - recomputes rollout log-probs on the current policy,
  - applies clipped PPO-style surrogate,
  - adds entropy bonus weighted by `alpha`,
  - adds frozen-reference KL penalty weighted by `kl_beta`,
  - updates only the configured trainable scope.

### G) Feasibility-constrained sampling
Paper concept:
- Hard feasibility support `F(x_t,u_t)` during sampling.
Code mapping:
- `NNXPolicyBackend.sample_candidate_pool(...)` applies a minimal token-level feasibility mask before sampling.
- `_build_feasibility_mask(...)` in `src/counter_bmt_v2/rl/nnx_policy.py` masks:
  - negative next speed,
  - speed above configured maximum,
  - excessive accel jump,
  - excessive yaw-rate jump.
- All-masked rows fall back safely to the unmasked support.

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

1. Resolve or build a DAG for the scenario, then sample a full counterfactual DAG assignment in topological order.
2. Apply the sampled assignment back onto the DAG payload and tensorize it for the policy backend.
3. `NNXPolicyBackend.sample_candidate_pool(...)` loads the supervised DAG-latent checkpoint path, prepares scene tensors through `_prepare_supervised_batch(...)`, and samples a candidate pool of rollouts.
4. During token sampling, the backend logs old ego log-probs, entropy, optional rollout traces, and feasibility-mask statistics.
5. Each candidate rollout gets `psi` embedding + risk features.
6. Novelty estimator computes surprisal in `psi` space before any novelty-model update.
7. Candidate oversampling + novelty-weighted resampling produce the final group.
8. Consensus scorer clusters only the final group and computes `rho(C|s) * mean_Q(C)`.
9. Thermostat computes adaptive `eta`/`alpha` from final-group entropy.
10. Rollout metadata updated:
   - `risk_features`, `behavior_embedding`, `novelty_score`, `cluster_id`, `consensus_score`.
11. Reward is recomputed with environment, novelty, consensus, and optional `vlm_replace` alignment terms.
12. Group advantages are computed on the selected group.
13. `NNXPolicyBackend.update(...)` performs the clipped PPO-style parameter update against the collected trajectories.
14. Metrics are logged by `train_rl_topo_mcpo.py` (`metrics.jsonl`, `summary.json`).

---

## 6) What is implemented vs. deferred

Implemented:
- Behavior-manifold scaffolding and APIs.
- Real NNX checkpoint policy backend for the mainline RL path.
- Full DAG-assignment intervention sampling and DAG-conditioned rollout prep.
- Candidate oversampling + novelty-weighted resampling.
- Consensus on final-group cluster mass times mean cluster quality.
- Clipped PPO-style policy updates with frozen-reference KL and trainable-scope masking.
- Minimal token-level feasibility masking during rollout sampling.
- DAG-aware embedding branch.
- Topology branch with cache + fallback.
- Novelty/consensus/thermostat mechanics.
- Group-relative advantage computation.
- `vlm_replace` alignment on sampled DAG assignments with cache keyed by the full assignment-visible context.
- RL training CLI and diagnostics.

Deferred:
- Full cubical zigzag persistence + ZPI vectorization backend.
- Unified LLM+trajectory conditioning head.
- Any feasibility engine beyond the current minimal token mask.

---

## 7) Practical usage

Checkpoint-backed mock smoke:
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_nnx_smoke \
  --steps 5 --group-size 4 \
  --embedding-mode risk_vector \
  --policy-backend nnx_checkpoint \
  --policy-checkpoint outputs/dag_latent_stage_c/checkpoints/last.pkl \
  --policy-model-preset midgpt_dag_latent \
  --policy-tokenizer-mode paper_simple \
  --policy-skip-steps 1 \
  --dag-source-mode scene_derived
```

Recommended real-run alignment setup:
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_vlm \
  --steps 200 --group-size 8 \
  --embedding-mode risk_vector \
  --policy-backend nnx_checkpoint \
  --policy-checkpoint outputs/dag_latent_stage_c/checkpoints/last.pkl \
  --policy-model-preset midgpt_dag_latent \
  --policy-tokenizer-mode paper_simple \
  --policy-skip-steps 1 \
  --dag-source-mode cache \
  --dag-cache-dir outputs/dag_cache_v3_mo/cache \
  --dag-cache-strict \
  --dag-expected-schema v3_maneuver_outcome \
  --alignment-source-mode vlm_replace \
  --vlm-alignment-enabled \
  --vlm-alignment-backend gpt4o
```

Ablation suggestions:
- `risk_vector` vs `dag_gnn` vs `hybrid` at fixed seed/scene pool.
- novelty estimator: `ema_gaussian` vs `knn`.
- consensus clusterer: `kmeans` vs `hdbscan` (if available).

---

## 8) Recommended next engineering steps

1. Implement a real zigzag/ZPI backend behind `ZigzagTopologyEncoder`.
2. Add richer integration coverage for larger checkpoint-backed runs and multi-step resume behavior.
3. Expand rollout tracing and offline diagnostics for long-run PPO debugging.
4. Add offline evaluation comparing diversity-realism tradeoff curves across embedding modes and policy backends.
5. Keep the unified LLM+trajectory backbone deferred until the current DAG-latent RL path is stable.
