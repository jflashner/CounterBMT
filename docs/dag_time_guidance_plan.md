# DAG Time Guidance Plan

Last updated: 2026-03-28

This note captures the current design direction for a more explicit DAG-conditioned trajectory model.

## Goal

The current additive legacy DAG path compresses the full graph into a single pooled latent and injects it once into `encoder/scenario_token`.

That is useful for global scene intent, but it is likely too coarse for:
- timestep-level trajectory guidance
- maneuver timing
- outcome-aware control over the rollout horizon

The next design direction is to build a per-timestep DAG guidance tensor and inject it alongside motion decoding.

## High-Level Vision

Build a per-timestep DAG guidance tensor.

For each decoder timestep:
- aggregate active maneuver nodes
- attach connected outcome context
- summarize confidence / CPT information
- inject the result as a control signal into the decoder token stream

In short:
- current path: global scene-level DAG bias
- target path: time-local DAG control

## Why This May Help

The current global pooled latent loses:
- which maneuver is active when
- whether multiple maneuvers overlap or occur sequentially
- which outcome links apply during different parts of the horizon

The cache already contains temporal maneuver information:
- `start_s`
- `end_s`
- `duration_s`
- `mid_s`

So the DAG already says something close to "what should happen when"; the model just is not consuming it in an explicitly time-aligned way.

## Proposed Representation

For each scenario, build:

- `dag_time_feat: [B, T, D_dag_time]`
- `dag_time_mask: [B, T]`

Where:
- `B` = batch size
- `T` = decoder horizon or motion-token horizon
- `D_dag_time` = a compact guidance width

Each timestep feature should summarize:
- active maneuver class indicators
- maneuver timing features
  - active flag
  - normalized time-to-start
  - normalized time-to-end
  - normalized distance from `mid_s`
- edge-weighted outcome indicators
  - collision outcome context
  - progress outcome context
  - compliance outcome context
- optional confidence summaries
  - maneuver confidence
  - outgoing edge confidence
  - CPT entropy or CPT row count summaries
- optional appended global DAG latent

## Initial Maneuver Encoding

Start with the existing maneuver classes already used by the perception/DAG path:
- `straight`
- `left_turn`
- `right_turn`
- `lane_change_left`
- `lane_change_right`
- `accelerate`
- `decelerate`
- `stop`

For each timestep:
- mark each maneuver as active if `start_s <= t <= end_s`
- if multiple maneuvers are active, aggregate with sum or max
- include a binary "any maneuver active" channel

## Outcome Encoding

For each active maneuver at timestep `t`:
- inspect its outgoing edges to outcome nodes
- attach the connected outcome values
- weight them by edge confidence when available

Suggested outcome channels:
- collision: `collision_avoided` vs `collision_possible`
- progress: `progress_good` vs `progress_limited`
- compliance: `compliant` vs `violation_possible`

## Injection Strategy

Ranked from least invasive to most ambitious:

### 1. Per-step additive bias

- project `dag_time_feat[t] -> d_model`
- add it to the decoder token embedding at timestep `t`

Pros:
- simple
- low risk
- easiest additive implementation

### 2. Per-step gated residual

- project `dag_time_feat[t]` to a bias and a gate
- apply `token_t = token_t + sigmoid(gate_t) * bias_t`

Pros:
- matches the current global gated-residual design
- allows weak initial conditioning
- likely the best first implementation

### 3. Cross-attention to DAG timestep tokens

- treat the time-guidance sequence as a set of auxiliary control tokens
- let the decoder attend to them directly

Pros:
- more expressive
- keeps multiple maneuvers/outcomes separate

Cons:
- more invasive
- larger implementation jump

Recommended first implementation:
- per-step gated residual

## Proposed Rollout Plan

### Phase 1: Time-guidance tensor only

1. Add a DAG-to-time-grid conversion utility.
2. Attach `dag_time_feat` and `dag_time_mask` in the cache-backed legacy batch path.
3. Add a small projection module in the additive DAG model wrapper.
4. Inject per-step guidance into the decoder input tokens.
5. Keep the existing global latent path available behind a config flag.

### Phase 2: Ablations

Compare:
- no DAG
- current global latent only
- time guidance only
- global latent + time guidance

Primary metrics:
- `loss_gain_vs_without_dag`
- `loss_gain_vs_shuffled_dag`
- `accuracy_gain_vs_without_dag`
- `accuracy_gain_vs_shuffled_dag`

### Phase 3: If needed

If per-step gated residual helps but still plateaus:
- move to decoder cross-attention over DAG timestep tokens

## Config Knobs To Add

Suggested new config entries:

- `DAG_LATENT.USE_TIME_GUIDANCE`
- `DAG_LATENT.TIME_GUIDANCE_DIM`
- `DAG_LATENT.TIME_GUIDANCE_MODE`
  - `additive`
  - `gated`
- `DAG_LATENT.TIME_GUIDANCE_USE_GLOBAL`
- `DAG_LATENT.TIME_GUIDANCE_ALIGN`
  - `decoder_steps`
  - `token_steps`
- `DAG_LATENT.TIME_GUIDANCE_ACTIVE_AGG`
  - `sum`
  - `max`
  - `mean`

## Risks

### 1. Noisy maneuver intervals

If VLM maneuver timing is noisy, time-local conditioning may amplify bad labels.

Mitigation:
- start with weak gates
- log active-rate diagnostics
- compare against global-only DAG path

### 2. Overlapping maneuvers

Multiple active maneuvers at the same timestep may blur semantics.

Mitigation:
- begin with simple sum/max aggregation
- add overlap diagnostics

### 3. Over-conditioning

Too-strong stepwise guidance may overpower the learned motion prior.

Mitigation:
- initialize gates near weak influence
- keep no-DAG and shuffled-DAG alignment checks active

## Suggested Diagnostics

If implemented, add:
- `dag_time/active_rate`
- `dag_time/mean_gate`
- `dag_time/class_coverage/*`
- `dag_time/outcome_coverage/*`
- `dag_time/avg_active_maneuvers`

These should be logged for both train and val if possible.

## Current Recommendation

If the existing global latent Stage B path continues to show:
- positive `loss_gain_vs_shuffled_dag`
- but negative `loss_gain_vs_without_dag`

then this time-aligned guidance path should be the first architectural upgrade to try before abandoning DAG conditioning.

## Implementation Notes

Target code areas for a future implementation:
- legacy additive path:
  - `src/Adv-BMT/bmt/dag_latent/dag_cache.py`
  - `src/Adv-BMT/bmt/dag_latent/model.py`
  - `src/Adv-BMT/bmt/dag_latent/lightning.py`
- schema / tensorization references:
  - `src/counter_bmt_v2/training/dag_tensorize.py`
- docs:
  - `docs/dag_latent_training.md`

