# CounterBMT v2 Roadmap (Fast Build + Unified Model Intent)

## 3-day scope
1. Fresh modular stack (`counter_bmt_v2`) with no legacy compatibility burden.
2. End-to-end runnable path: perception -> DAG -> sampled intervention -> conditioning -> trajectory -> judge -> reward.
3. JAX-first trajectory generator with conditioning input contract.

## Day 1 progress
1. Added `counter_bmt_v2/data/scenarionet.py`:
   - minimal ScenarioNet loader that emits only Adv-BMT-relevant tensors (agents, map vectors, traffic lights).
   - map features use the 27-d vector layout from Adv-BMT preprocessing for scene-token parity.
2. Extended `trajectory_jax/nnx_bmt.py`:
   - added a scene token encoder for map + traffic-light channels.
   - wired A2S attention to consume real encoded scene tokens and masks when provided.
3. Added model presets in `trajectory_jax/presets.py`:
   - `paper_like_small_config()`
   - `paper_like_full_config()`

## Day 2 progress
1. Added `counter_bmt_v2/training/supervised.py`:
   - Optax training loop with AdamW + warmup cosine schedule + grad clipping.
   - masked cross-entropy token objective and token-level diagnostics.
   - forward/reverse/mixed supervision modes with reverse-indicator conditioning.
   - checkpoint save/resume support (`checkpoints/step_*.pkl`, `checkpoints/last.pkl`).
2. Added fixed-shape collation support in `data/scenarionet.py`:
   - optional `max_time_steps/max_agents/max_map_features/max_vectors_per_map_feature/max_traffic_lights`
   - helps keep JIT-friendly stable tensor shapes across batches.
3. Added training CLI:
   - `counter_bmt_v2/cli/train_nnx_bmt.py`
   - exposes all core training and data-shape controls for rapid experiments.
4. Added Adv-BMT-style forward-pass validation metrics:
   - new `training/forward_metrics.py` evaluator integrated into eval loop.
   - logs scenario-level `sfde/sade/ssde`, `fdd/add/sdd`, `vel_jsd/acc_jsd/ttc_jsd`,
     plus collision and SDC comfort summaries under `forward/*` metric keys.
5. Added supervised parity checklist:
   - `ADV_BMT_PARITY_CHECKLIST.md` tracks remaining non-RL alignment work with
     concrete tasks and acceptance tests.

## Planned near-term upgrades
1. Replace mock perception with frontier VLM parser and strict JSON schema validation.
2. Replace simple DAG builder with full Bayesian DAG + CPT inference and posterior sampling.
3. Replace mock judge with VLM verification aligned with intervention target.
4. Replace simple reward with weighted alignment/safety/realism reward and grouped rollout collection.

## Day 3 progress (P1 parity foundation)
1. Added relation parity core and MidGPT relation bundle builder:
   - `trajectory_jax/relation_parity.py`
   - scene S2S + decoder-ready A2A/A2T/A2S relation tensors/masks.
2. Added scene relation Fourier embedding and relation-aware scene self-attention:
   - `trajectory_jax/fourier_embedding_nnx.py`
   - `trajectory_jax/nnx_bmt.py` (`NNXRelationParityConfig`, scene relation stack).
3. Added MidGPT parity preset and relation parity scripts:
   - `trajectory_jax/presets.py` (`midgpt_parity_config`)
   - `scripts/parity/compare_relations.py`
   - `scripts/parity/export_relation_batch.py`

## Unified LLM + trajectory direction (not blocked by day-3 scope)
1. Keep `ConditioningSignal` as stable contract now.
2. Add learned LLM conditioning head that maps hidden states to trajectory-control vectors.
3. Migrate from two-stage pipeline to shared latent backbone:
   - shared encoder for scene + language context
   - trajectory decoder head for rollout
   - language/planning head for intervention reasoning
4. Final objective: joint training where language/planning outputs and trajectory outputs are optimized together under alignment rewards.

## Implementation note
Current `JaxTrajectoryGenerator` is intentionally lightweight; it is the landing pad for the upcoming NNX model rewrite and later unified backbone.
