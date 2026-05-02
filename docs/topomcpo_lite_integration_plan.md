# Topo-MCPO-Lite Integration Plan For Adv-BMT Counterfactual Training

## Purpose
This note captures the shortest realistic path for bringing Topo-MCPO-style ideas into the current Adv-BMT semantic counterfactual rollout trainer without migrating the whole project to the separate `counter_bmt_v2` RL stack.

The guiding constraint is:

- preserve the current Adv-BMT semantic-control pipeline,
- preserve the counterfactual dataset and actual-right-wall progress geometry,
- reuse only the smallest useful pieces of Topo-MCPO,
- and postpone invasive PPO/backend changes until the simpler behavior-manifold additions are proven useful.

---

## 1) Compatibility Assessment

## Current Adv-BMT setup already has

- grouped rollout sampling in `src/Adv-BMT/bmt/models/motionlm_lightning.py`
- group-relative optimization via normalized rollout advantages
- a behavior-level RL signal rather than only token-level imitation
- a valid-region/tube reward
- a topology-aware progress definition via the actual right-wall contour trace
- semantic decision families
- causal intervention contracts and DAG projections

## Topo-MCPO adds

- a behavior embedding `z(tau)`
- novelty tilt over the behavior manifold
- quality-weighted consensus over rollout clusters
- an entropy thermostat controlling diversity vs commitment
- PPO/GRPO-style clipped updates and optional reference KL

## Bottom line

The current Adv-BMT trainer is already strongly compatible with Topo-MCPO in spirit. The biggest missing pieces are:

- explicit behavior-manifold embeddings inside the rollout training loop,
- novelty reward,
- consensus reward,
- thermostat adaptation,
- and, only later, clipped PPO-style updates.

That means the shortest path is not a full migration. It is a local augmentation of the existing rollout tube objective.

---

## 2) Design Principle

Do not replace the current semantic counterfactual objective.

Instead:

- keep the current tube reward and progress reward,
- keep the current semantic-family and DAG infrastructure,
- and add Topo-MCPO-style behavior-manifold terms on top of the existing rollout group.

This makes the integration:

- scientifically coherent,
- implementation-light,
- easy to ablate,
- and easy to back out if it does not help.

---

## 3) Recommended Scope

## What to reuse from `counter_bmt_v2`

These components are good candidates for reuse or light porting:

- `src/counter_bmt_v2/rl/consensus.py`
- `src/counter_bmt_v2/rl/novelty.py`
- `src/counter_bmt_v2/rl/thermostat.py`
- `src/counter_bmt_v2/rl/grpo.py`
- `src/counter_bmt_v2/rl/behavior_embedding.py`

## What not to reuse initially

Do not try to pull in the full NNX RL backend yet:

- `src/counter_bmt_v2/rl/nnx_policy.py`
- `src/counter_bmt_v2/cli/train_rl_topo_mcpo.py`

Those are useful references, but they imply a much bigger policy/trainer stack transition than we need right now.

---

## 4) Proposed Integration Path

## Phase 0: Documentation and instrumentation only

Goal:

- make no training-behavior changes,
- add enough hooks and metrics so later phases are easy to debug.

Additions:

- record a simple per-rollout embedding vector in the existing rollout tube bundle
- log per-group entropy, cluster histogram, novelty score, and consensus score
- keep all new reward weights at `0.0`

Suggested files:

- `src/Adv-BMT/bmt/models/motionlm_lightning.py`
- `src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml`

Why this phase matters:

- it lets us verify that the behavior embedding is stable and meaningful before it starts affecting optimization

---

## Phase 1: Topo-MCPO-lite consensus reward

Goal:

- add only the consensus half of Topo-MCPO first
- keep novelty and thermostat off

### 1. Rollout embedding

Inside `_build_sdc_semantic_rollout_tube_policy_objective(...)`, define a lightweight rollout embedding `z(tau)` using statistics we already compute.

First-pass embedding candidates:

- final frontier arc
- inside fraction
- mean tube distance
- progress reward mean
- total return
- average speed of SDC rollout
- average yaw-rate magnitude
- average jerk or acceleration magnitude if available

This should be cheap, deterministic, and built entirely from tensors already in the rollout loop.

### 2. Cluster rollouts within the current group

For each scenario/group:

- cluster the `group_size` rollout embeddings
- compute cluster mass
- compute a quality score per cluster

Quality should be derived from current counterfactual signals:

- more frontier progress is better
- smaller tube distance is better
- higher inside fraction is better
- violations or off-tube behavior are worse

### 3. Add consensus reward

Define:

- `r_consensus(tau) = rho(C(tau)|s) * Q(C(tau))`

Then add:

- `R_total = R_tube + w_consensus * r_consensus`

before return-to-go and advantage normalization.

### Why consensus first

Consensus is the safest first Topo-MCPO term because:

- it is scenario-local
- it needs no persistent density model
- it aligns naturally with the existing grouped rollout objective
- it rewards dominant safe-progress modes rather than arbitrary diversity

---

## Phase 2: Add novelty

Goal:

- encourage the rollout sampler to keep covering multiple feasible behavior modes

### 1. Maintain an EMA density over rollout embeddings

Use a lightweight version of the novelty estimator from:

- `src/counter_bmt_v2/rl/novelty.py`

Simplest first pass:

- diagonal Gaussian EMA over embedding dimensions

### 2. Compute novelty score

For each rollout embedding:

- `novelty(tau) ~= -log p_hat(z(tau))`

Normalize within the group before use.

### 3. Add novelty reward or novelty tilt

The shortest route inside the current trainer is:

- add novelty as an intrinsic reward term

not:

- rebuild the rollout sampler to draw from a fully novelty-tilted proposal distribution

So first use:

- `R_total = R_tube + w_consensus * r_consensus + w_novelty * r_novelty`

This is already Topo-MCPO-like in spirit even if it is not the exact sampling law from the paper.

---

## Phase 3: Add entropy thermostat

Goal:

- adapt diversity pressure vs commitment pressure per scenario/group

Use the existing entropy idea from:

- `src/counter_bmt_v2/rl/thermostat.py`

Compute:

- group cluster entropy
- target entropy `H*`
- adaptive novelty multiplier `eta`
- adaptive consensus multiplier `alpha`

Then use:

- `w_novelty_eff = base_w_novelty * f(eta)`
- `w_consensus_eff = base_w_consensus * f(alpha)`

This is better than fixed manual weights once the consensus/novelty signals are known to be stable.

---

## Phase 4: Optional PPO/GRPO upgrade

Goal:

- move from the current REINFORCE-style surrogate toward a clipped Topo-MCPO-style update

This is the most invasive phase and should come last.

Needed changes:

- save old rollout log-probs
- recompute current rollout log-probs
- build importance ratios
- add PPO clipping
- optionally add reference-policy KL

This phase is valuable, but it is not required to get something meaningfully Topo-MCPO-like.

The current recommendation is:

- do not start here

---

## 5) Shortest Realistic MVP

If we want the smallest implementation that is clearly “in the spirit of Topo-MCPO,” the MVP is:

1. Add a cheap rollout embedding in the existing Adv-BMT rollout group.
2. Cluster the group and compute quality-weighted consensus.
3. Add a small consensus reward term before advantage normalization.
4. Log entropy and cluster diagnostics.

This would already give:

- grouped behavior embeddings
- quality-weighted mode preference
- behavior-manifold-aware relative optimization

without changing the base policy backend.

That is the best “80/20” version.

---

## 6) Suggested First Implementation Details

## New config knobs

Add to:

- `src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml`

Suggested first knobs:

- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_BEHAVIOR_EMBEDDING_ENABLED`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_CONSENSUS_WEIGHT`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_CONSENSUS_K`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_CONSENSUS_PROGRESS_WEIGHT`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_CONSENSUS_DISTANCE_WEIGHT`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_NOVELTY_WEIGHT`
- `MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TOPO_THERMOSTAT_ENABLED`

Default all new reward weights to `0.0`.

## New stats to log

In `motionlm_lightning.py`, log:

- `cf/sdc_rollout_topo_entropy`
- `cf/sdc_rollout_topo_consensus_mean`
- `cf/sdc_rollout_topo_consensus_std`
- `cf/sdc_rollout_topo_cluster_count`
- `cf/sdc_rollout_topo_novelty_mean`
- `cf/sdc_rollout_topo_eta`
- `cf/sdc_rollout_topo_alpha`

These should be visible in W&B before we let them influence training.

## First-pass rollout embedding

Recommended first-pass feature vector:

- `return_total`
- `inside_fraction`
- `frontier_arc_final`
- `progress_reward_mean`
- `tube_distance_mean`
- `speed_mean`
- `speed_std`
- `yaw_rate_mean_abs`

This is enough to test the idea. We do not need DAG-GNN or ZPI on day one.

---

## 7) Where DAGs Fit

The clean integration sequence is:

- first use rollout behavior summaries only
- then, if it helps, extend the rollout embedding with DAG features

Later rollout embedding versions can include:

- compact DAG summary from `dag_adapter.py`
- maneuver/outcome node histograms
- simple graph pooled features

This would let novelty/consensus operate in a space that mixes:

- realized behavior,
- intervention semantics,
- and causal context.

That is a stronger scientific story, but it is not required for the first implementation.

---

## 8) Where TEN-DM / topology features fit

The actual shortest path does not require full TEN-DM-style topology encoding.

A practical sequence is:

1. start with cheap rollout summary embeddings
2. then add DAG-aware embeddings
3. only later, if useful, add time-image/ZPI-style topology embeddings

The existing modules in `counter_bmt_v2` already provide a natural landing zone for that later step:

- `src/counter_bmt_v2/rl/topology.py`
- `src/counter_bmt_v2/rl/behavior_embedding.py`

So topology should be treated as:

- a future strengthening of the rollout embedding,
- not a prerequisite for a useful Topo-MCPO-lite implementation.

---

## 9) Risks And Failure Modes

## Main risks

- consensus reward may over-favor majority but mediocre modes
- novelty reward may encourage noisy but unhelpful diversity
- clustering may be unstable for very small `group_size`
- extra reward terms may obscure the interpretation of the current tube/progress signals

## Mitigations

- start with consensus only
- use small weights
- keep all new metrics logged separately
- ablate with weights `0.0`, `0.05`, `0.1`
- do all early testing on scene `00008` and a small top-N subset before wider runs

---

## 10) Recommended Experiments

## Stage A: no-op instrumentation

- add behavior embedding and topology metrics
- no new reward effect

Success criterion:

- stable metrics and sensible cluster separation

## Stage B: consensus-only small-scene tests

- scene `00008`
- group size `8`
- low consensus weight

Success criterion:

- faster commitment to the correct branch
- less waffling among sampled rollouts

## Stage C: top-50 or top-100 subset

- verify that consensus remains stable across diverse scenes

Success criterion:

- rollout quality improves without collapse to stagnant or trivial modes

## Stage D: novelty + thermostat

- only after consensus-only behavior is understood

---

## 11) Recommendation

When we return to this, the best next implementation is:

- **Phase 1 only: consensus reward inside the current Adv-BMT rollout trainer**

That is the shortest, safest, and most scientifically defensible route to getting something genuinely Topo-MCPO-like into the current counterfactual training loop.

Everything beyond that should be layered on top only if Phase 1 helps.
