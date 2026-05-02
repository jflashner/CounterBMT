# Additive DAG-Conditioned Topo-MCPO Extension

This document describes the additive semantic-RL extension that adds a DAG-conditioned behavior manifold, novelty shaping, and a full entropy thermostat on top of the existing `topomcpo_lite` semantic-control setup.

The important design constraint is unchanged:

- the legacy semantic RL path stays intact by default
- the earlier `strict_local` and `topomcpo_lite` setups remain available for posterity
- the new code only activates when the new preset or new feature flags are enabled

## What Was Added

The new additive extension introduces four new ideas:

1. A small SDC rollout behavior encoder
   - file: [semantic_ext_topomcpo.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/semantic_ext_topomcpo.py)
   - class: `RolloutBehaviorEncoder`
   - input: detached per-step SDC rollout features
   - output: normalized behavior embedding `z_behavior`

2. A causal-DAG context encoder
   - file: [semantic_ext_topomcpo.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/semantic_ext_topomcpo.py)
   - class: `CausalDAGContextEncoder`
   - input: existing `cf/path_token`, `cf/compliance_token`, `cf/timing_token`
   - output: normalized DAG context embedding `z_dag`

3. A label- and DAG-conditioned EMA novelty bank
   - file: [semantic_ext_topomcpo.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/semantic_ext_topomcpo.py)
   - class: `EMAGaussianNoveltyBank`
   - role: maintain a rank-local EMA Gaussian density over behavior embeddings keyed by a discrete DAG bucket

4. A per-scene entropy thermostat
   - file: [motionlm_lightning.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/motionlm_lightning.py)
   - role: adapt novelty weight `eta(s)` and consensus weight `alpha(s)` from observed cluster entropy versus a DAG-aware target entropy

There is also one additional additive rollout-shaping hook now available:

5. A nearby-traffic speed floor
   - file: [motionlm_lightning.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/motionlm_lightning.py)
   - role: discourage the SDC from creeping by comparing it against the median speed of nearby moving non-SDC agents
   - note: in the `...speedfloor` preset, the SDC-side speed now comes from monotone frontier-arc progress on the selected progress centerline, which is the right-wall contour trace used by the rollout reward geometry

## Core Code Map

Main trainer wiring lives in:

- [motionlm_lightning.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/motionlm_lightning.py)

The essential new hooks are:

- `_init_semantic_ext_topomcpo_modules`
  - builds the optional behavior encoder, DAG encoder, prediction heads, and novelty bank

- `_semantic_ext_bucket_keys`
  - converts semantic label plus control-token fields into a discrete causal-context bucket

- `_semantic_ext_target_entropy`
  - computes the thermostat target entropy `H*(s)` from the causal decision context

- `_build_semantic_ext_rollout_features`
  - builds the per-step SDC rollout feature tensor used by the behavior encoder

- `_compute_semantic_ext_behavior_aux_loss`
  - trains the behavior manifold with semantic prediction, DAG-token prediction, and DAG-alignment losses

- `_compute_rollout_topomcpo_regularization_bundle`
  - computes:
    - detached behavior embeddings
    - detached DAG embeddings
    - DAG-bucket novelty scores
    - rollout clustering
    - cluster entropy
    - thermostat-adjusted consensus and novelty bonuses
    - behavior-model auxiliary losses

- `_build_sdc_semantic_rollout_tube_policy_objective`
  - now consumes the new regularization bundle
  - still falls back to the old consensus-only path when the new behavior model is disabled

- `_compute_semantic_ext_traffic_speed_floor_bundle`
  - computes a one-sided rollout penalty from nearby moving-agent speed
  - uses realized per-step displacement from `decoder/modeled_agent_position -> decoder/rollout_next_position`
  - this keeps the speed-floor tied to the same visible rollout motion shown in training debug artifacts
  - stays fully disabled unless its config gate is turned on

Checkpoint persistence:

- `on_save_checkpoint`
- `on_load_checkpoint`
  - these persist the novelty bank so resumed runs do not cold-start the density estimate

## How The Causal DAG Is Used

We do not ship raw intervention JSON through the batch. Instead, we use the existing compiled control tokens, which already encode the fixed local intervention DAG:

- `cf/path_token`
  - path-choice node
- `cf/compliance_token`
  - signal-state context plus compliance node
- `cf/timing_token`
  - conflict context plus entry-timing node

This matches the causal structure defined in:

- [dag_adapter.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/dag_adapter.py)
- [contract_local_intervention.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/contract_local_intervention.py)

The DAG now affects training in three places:

1. Novelty bucket definition
   - novelty is measured within a causal context bucket rather than globally

2. Thermostat target entropy
   - conflict-heavy or permissive contexts get a higher target diversity
   - stop-like or tightly constrained contexts get a lower target diversity

3. Manifold auxiliary supervision
   - the rollout behavior embedding is trained to predict causal decision labels and align with the DAG context embedding

## Behavior Manifold

The rollout behavior encoder currently uses detached SDC-only temporal features:

- forward arc
- forward-arc increment
- tube distance
- inside-tube indicator
- speed
- acceleration
- jerk
- absolute yaw rate

This is intentionally narrow and safe for the first implementation. It is meant to capture:

- progress profile
- comfort
- path adherence

while the DAG encoder captures:

- interaction regime
- compliance regime
- timing regime

## Novelty

Novelty is implemented as:

- compute detached behavior embedding `z_behavior`
- score it under an EMA Gaussian density for its causal bucket
- convert novelty score to a positive terminal bonus only when the rollout is above the group mean novelty

The novelty bank is currently:

- rank-local during training
- checkpoint-persistent
- disabled unless explicitly enabled in config

This is deliberate. It keeps the implementation light and additive while still making the novelty component real.

## Nearby-Traffic Speed Floor

The speed-floor hook is meant to address a specific failure mode of the anti-stall term:

- with anti-stall alone, the SDC can learn to move only just enough to satisfy the threshold

The new shaping term therefore:

- looks at rollout-predicted non-SDC agents
- computes agent speed from realized rollout displacement instead of trusting detokenized velocity state alone
- filters to nearby agents within a configurable radius
- expands that radius geometrically if the initial neighborhood is empty
- falls back to scene-wide moving agents if a local neighborhood still cannot be formed
- keeps only moving agents above a configurable speed threshold
- uses the median of those speeds as the reference
- and if there are no moving non-SDC agents at all, falls back to the average GT SDC speed from the original batch
- applies a one-sided penalty only when the SDC is slower than a fixed fraction of that reference

This is intentionally:

- local rather than whole-scene
- median-based rather than mean-based
- one-sided rather than forcing the SDC to exactly match traffic

The main config knobs are:

- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_FLOOR_ENABLED`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_FLOOR_WEIGHT`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_NEARBY_RADIUS_M`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_EXPAND_FACTOR`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MAX_EXPANSION_STEPS`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MIN_MOVING_SPEED_MPS`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_REFERENCE_RATIO`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MAX_REFERENCE_MPS`
- `LOCAL_CONTROL_SDC_SEMANTIC_EXT_TRAFFIC_SPEED_MIN_NEIGHBORS`

Logged metrics:

- `cf/sdc_rollout_tube_traffic_speed_penalty_mean`
- `cf/sdc_rollout_tube_traffic_speed_reference_mean`
- `cf/sdc_rollout_tube_traffic_speed_sdc_mean`
- `cf/sdc_rollout_tube_traffic_speed_neighbor_count_mean`

Rollout debug artifacts now also store per-rollout traffic-speed details:

- `traffic_speed_penalty_t`
- `traffic_speed_reference_t`
- `traffic_speed_sdc_t`
- `traffic_speed_neighbor_count_t`

## Entropy Thermostat

The thermostat implements the Topo-MCPO-style adaptive balance between diversity and commitment.

For each scenario group:

- cluster rollout behavior embeddings
- compute observed cluster entropy `H(s)`
- compute DAG-aware target entropy `H*(s)`
- adapt:
  - `alpha(s)` for consensus weight
  - `eta(s)` for novelty weight

Current intuition:

- if entropy is too low, increase novelty pressure
- if entropy is too high, increase consensus pressure

Logged thermostat metrics:

- `cf/sdc_rollout_tube_thermostat_target_entropy_mean`
- `cf/sdc_rollout_tube_thermostat_alpha_mean`
- `cf/sdc_rollout_tube_thermostat_eta_mean`

## Config Surface

Defaults remain off in:

- [motion_default.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/motion_default.yaml)

The new preset that enables the full DAG-manifold + novelty + thermostat stack is:

- [motion_forward_sdc_semantic_only_topomcpo_dag_thermostat.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_topomcpo_dag_thermostat.yaml)

An additive preset that also enables the nearby-traffic speed floor is:

- [motion_forward_sdc_semantic_only_topomcpo_dag_thermostat_speedfloor.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_topomcpo_dag_thermostat_speedfloor.yaml)

The previous lighter additive preset remains:

- [motion_forward_sdc_semantic_only_topomcpo_lite.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_topomcpo_lite.yaml)

## Future Tilted Sampling Hook

This implementation does not yet modify the rollout sampler itself.

That is intentional. We want the novelty and thermostat logic to be stable first.

The future tilted-sampling path should reuse the quantities already exposed by the new code:

- detached behavior embedding `z_behavior`
- novelty score per rollout
- per-scenario `eta(s)`
- per-scenario `alpha(s)`
- per-scenario cluster entropy

The natural insertion point for future tilted sampling is the rollout generation path in:

- [motionlm_lightning.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/motionlm_lightning.py)
  - `_build_sdc_semantic_rollout_tube_policy_objective`

Specifically, future work can let:

- `eta(s)` modulate the sampling distribution before rollout completion
- or modulate sampling temperature / logit tilts per scenario

without changing the meaning of the current reward-side novelty and thermostat code.

## Practical Reading Order

If you come back to this later, read in this order:

1. [motion_forward_sdc_semantic_only_topomcpo_dag_thermostat.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_topomcpo_dag_thermostat.yaml)
2. [semantic_ext_topomcpo.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/semantic_ext_topomcpo.py)
3. [motionlm_lightning.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/models/motionlm_lightning.py)
   - `_init_semantic_ext_topomcpo_modules`
   - `_semantic_ext_target_entropy`
   - `_compute_rollout_topomcpo_regularization_bundle`
   - `_build_sdc_semantic_rollout_tube_policy_objective`
