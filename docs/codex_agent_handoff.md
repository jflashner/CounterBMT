# CounterBMT Codex Agent Handoff

Last updated: 2026-03-07

This is the living handoff document for new Codex agents working on this repo.

Maintenance rule:
- Update this file whenever a noteworthy change lands in any of these areas:
  - training entrypoints or configs
  - DAG cache schema/contracts
  - evaluation or replay tooling
  - environment/bootstrap assumptions
  - local/remote machine workflow
  - theoretical framing that changes implementation choices

Use this document as the first read before making changes.

## 1. Project Goal

CounterBMT is building a trajectory-generation stack for safety-critical autonomous driving scenarios.

The current project has three linked goals:

1. Rebuild the supervised Adv-BMT motion model in JAX/Flax NNX with strong parity to the legacy implementation.
2. Add DAG-latent conditioning so the trajectory model can use causal/behavioral scene structure.
3. Evolve the system toward a topology-aware RL pipeline (Topo-MCPO style), while keeping the supervised path stable and usable now.

In practice, the actively developed path is:

`ScenarioNet -> VLM perception -> PromptBN DAG -> DAG cache -> NNX BMT / DAG-latent supervised training -> forward evaluation -> head-to-head comparison -> replay/GIF inspection`

The repo also contains:
- the legacy Adv-BMT reference implementation
- an older `counter_bmt` pipeline
- an RL scaffold that is only partially realized

## 2. What Is Active vs Legacy

### Actively developed

- `src/counter_bmt_v2/`
- `src/scripts/dag_cache/`
- `src/scripts/eval/`
- `src/scripts/parity/`
- `src/scripts/replay/`
- `docs/`
- `configs/eval/`
- `tests/`

### Legacy but still operationally important

- `src/Adv-BMT/`
  - This is the legacy paper/reference implementation.
  - It is still used for parity checks, paper-number reproduction, and head-to-head comparison.
- `src/counter_bmt/`
  - Older CounterBMT code.
  - Still used for some utilities, especially replay export and ScenarioNet visualization.

### Historical / not the current development center

- `src/dag_models/`
- `src/scripts/run_full_pipeline.py`
- notebooks and one-off diagnostics scripts

Do not start new feature work in `src/Adv-BMT/` or `src/counter_bmt/` unless the task is explicitly about legacy parity or keeping old utilities working.

## 3. Theory Background From Repo Papers

### `Adv-BMT.pdf`

This is the main supervised modeling reference.

Key ideas that directly drive the v2 implementation:
- bidirectional motion tokenization over acceleration and yaw-rate bins
- relation-aware scene encoding and decoding
- forward and reverse supervision
- top-p style rollout sampling for diversity
- forward realism/diversity metrics used in evaluation

Where it maps in code:
- tokenizer and action grid:
  - `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
  - `src/counter_bmt_v2/trajectory_jax/tokenizer_parity.py`
- relation parity:
  - `src/counter_bmt_v2/trajectory_jax/relation_parity.py`
- training loop:
  - `src/counter_bmt_v2/training/supervised.py`
- parity checklist:
  - `src/counter_bmt_v2/ADV_BMT_PARITY_CHECKLIST.md`

### `PromptBN.pdf`

This is the main DAG-generation reference.

Key ideas used here:
- query the LLM with structured variable metadata
- ask for both graph structure and CPTs
- validate structure strictly after generation
- retry bounded times when validation fails
- keep the LLM in the loop rather than using it only as a pre/post-processor

Where it maps in code:
- GPT-4o perception:
  - `src/counter_bmt_v2/perception/gpt4o.py`
- PromptBN DAG builder:
  - `src/counter_bmt_v2/causal/promptbn.py`
- DAG contract enforcement:
  - `src/counter_bmt_v2/causal/dag_contract.py`
- DAG cache building:
  - `src/scripts/dag_cache/build_dag_cache_v2.py`

### `NeurIPS_2026_Auto_Driving_RL.pdf`

This is the Topo-MCPO-style RL reference for the future RL direction.

Core equations from the paper:
- novelty-tilted sampling distribution:
  - `q_theta(tau) propto [prod_t pi_theta(u_t | x_t) exp(lambda Phi(x_t,u_t))] exp(eta * (-log p_hat(z(tau))))`
- quality-gated consensus reward:
  - `r_mc(tau) = rho(C(tau)|s) * Q(C(tau))`
- entropy thermostat:
  - `eta = eta0 + k_eta (H* - H)`
  - `alpha = alpha0 + k_alpha (H - H*)`
- group-relative advantages and PPO/GRPO-style clipping

What is implemented:
- behavior manifold scaffolding
- novelty / consensus / thermostat
- RL VLM alignment replacement mode

What is not fully implemented:
- real policy optimization on the NNX trajectory model
- true clipped PPO/GRPO update over a trainable policy
- hard feasibility-constrained sampling

Key code:
- `src/counter_bmt_v2/rl/`
- `src/counter_bmt_v2/cli/train_rl_topo_mcpo.py`
- `docs/rl_behavior_manifold_integration.md`
- `docs/rl_behavior_manifold_implementation_details.md`
- `docs/rl_vlm_alignment.md`

### `16261_TEN_DM_Topology_Enhanced (1).pdf`

This is the topology/graph representation inspiration.

Key ideas that matter here:
- graph abstraction helps encode higher-order dependencies
- time-series image representations can capture temporal shape
- zigzag persistence / topological descriptors are intended to provide robust structure-aware features

How it maps here:
- supervised DAG-latent path uses a real trainable DAG encoder in JAX/NNX
- RL topology path has a pluggable topology interface and cache, but still uses fallback descriptors rather than a full zigzag backend

Key code:
- supervised DAG encoder:
  - `src/counter_bmt_v2/trajectory_jax/dag_gnn_nnx.py`
- RL topology scaffold:
  - `src/counter_bmt_v2/rl/topology.py`

### `LLADA.pdf`

This is not part of the production path today. It is relevant as background for the eventual unified language + trajectory model direction.

Current status:
- the unified model is explicitly a stub
- see `src/counter_bmt_v2/trajectory_jax/unified_stub.py`

## 4. Current Architecture, End to End

### Supervised base path

1. Load ScenarioNet scenes with `ScenarioNetNNXLoader`.
2. Convert scenes into fixed-shape training tensors.
3. Tokenize trajectories into Adv-BMT-style motion tokens.
4. Train `NNXBidirectionalMotionTransformer`.
5. Evaluate with forward rollout metrics and optional artifact export.

Main files:
- `src/counter_bmt_v2/data/scenarionet.py`
- `src/counter_bmt_v2/training/supervised.py`
- `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
- `src/counter_bmt_v2/trajectory_jax/presets.py`
- `src/counter_bmt_v2/training/forward_metrics.py`
- `src/counter_bmt_v2/cli/train_nnx_bmt.py`

Important runtime presets:
- `adv_bmt_runtime_parity`
  - historical v2 parity preset
  - matches major optimizer/tokenizer/schedule knobs
  - does **not** force legacy forward-only supervision
- `legacy_midgpt_recipe`
  - use this when the task is "match `src/Adv-BMT/cfgs/0202_midgpt.yaml` as closely as possible"
  - locks:
    - `model_preset=midgpt_parity`
    - `tokenizer_mode=adv_bmt_parity`
    - `lr_schedule_mode=legacy_cosine_zero`
    - `num_epochs=30`
    - `mode=forward`
    - `reverse_probability=0.0`
  - does not set batch size because legacy Lightning DDP used per-process batch size while v2 `pmap` uses global batch size

Tokenizer parity note:
- For the active legacy target `src/Adv-BMT/cfgs/0202_midgpt.yaml`, the v2 parity tokenizer path is now spot-checked to exact-match the legacy bicycle tokenizer on both forward and backward tokenization (`n=20`, `batch_size=4`).
- This reduces tokenizer risk for MidGPT-style supervised runs significantly.
- Remaining parity risk is more about training-runtime semantics and full-run evaluation than missing tokenization mechanics.

### DAG-latent supervised path

1. Build DAG caches from ScenarioNet + VLM + PromptBN.
2. Read DAG cache or use deterministic scene-derived fallback.
3. Tensorize DAG into fixed-size node/edge tensors.
4. Encode DAG with the NNX DAG graph encoder.
5. Inject the latent into the transformer through a global gated residual path.
6. Train with staged schedule A/B/C.
7. Verify DAG usefulness via real/null/shuffled DAG ablations at eval time.

Main files:
- `src/scripts/dag_cache/build_dag_cache_v2.py`
- `src/counter_bmt_v2/training/dag_cache.py`
- `src/counter_bmt_v2/training/dag_sources.py`
- `src/counter_bmt_v2/training/dag_tensorize.py`
- `src/counter_bmt_v2/trajectory_jax/dag_gnn_nnx.py`
- `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
- `src/counter_bmt_v2/training/supervised_dag_latent.py`
- `src/counter_bmt_v2/cli/train_nnx_bmt_dag_latent.py`

### Head-to-head evaluation path

1. Resolve deterministic scenario subset.
2. Run each model on the same subset.
3. Save canonical forward rollout artifacts.
4. Compute aggregate and per-scenario metrics.
5. Build overlay plots, replay exports, and comparison reports.

Main files:
- `src/counter_bmt_v2/eval/head2head.py`
- `src/counter_bmt_v2/eval/v2_runner.py`
- `src/counter_bmt_v2/eval/legacy_runner.py`
- `src/counter_bmt_v2/eval/compare.py`
- `src/counter_bmt_v2/eval/visualize.py`
- `src/scripts/eval/compare_models_head2head.py`
- `src/scripts/eval/export_head2head_samples.py`

### Replay / visualization path

- `src/scripts/replay/export_forward_artifact_to_scenario.py`
- `src/scripts/replay/export_forward_artifacts_batch.py`
- `src/scripts/replay/make_scenario_gif.py`
- `src/counter_bmt/scenario_export.py`
- `src/counter_bmt/scenarionet_visualizer.py`

## 5. Key Directory and File Map

### Top level

| Path | Purpose | Notes |
|---|---|---|
| `README.md` | Short doc index | Minimal; not a full guide |
| `requirements.txt` | Pinned v2 stack | JAX/Flax/OpenAI/MetaDrive/ScenarioNet |
| `requirements-legacy.txt` | Pinned legacy stack | Torch + TensorFlow/Waymo for legacy Adv-BMT |
| `tools/bootstrap_linux.sh` | Reproducible env bootstrap | Profiles: `v2`, `legacy`, `full` |
| `tools/verify_environment.py` | Env verifier | Exact package/version/import checks |
| `tools/run_parity_suite.sh` | One-command parity runner | Wraps P6 parity report |
| `configs/eval/` | Evaluation registries | Used by head-to-head runner |
| `docs/` | Current human docs | Keep synchronized with code |

### `src/counter_bmt_v2/`

| Path | Purpose | Notes |
|---|---|---|
| `config.py` | Top-level dataclasses for pipeline and RL | Includes `VLMAlignmentConfig` |
| `contracts/core.py` | Core typed contracts | Shared DAG, rollout, reward, VLM dataclasses |
| `data/scenarionet.py` | ScenarioNet loader for v2 training | Fixed-shape tensors, stable file ordering |
| `data/frame_render.py` | Frame rendering helpers | Used for DAG/VLM evidence |
| `data/vlm_frame_prep.py` | VLM frame preparation and annotation | Global/ego views, manifests, context text |
| `causal/dag.py` | DAG helpers and conversions | Low-level DAG definitions/utilities |
| `causal/promptbn.py` | PromptBN DAG builder | Core VLM-to-DAG path |
| `causal/dag_contract.py` | DAG contract enforcement | Compact and maneuver-outcome contracts |
| `causal/sampler.py` | Intervention sampling | Picks interventions from DAGs |
| `conditioning/signal.py` | Conditioning vector builder | Bridges intervention/DAG to rollout conditioning |
| `trajectory_jax/nnx_bmt.py` | Main NNX trajectory model | Active supervised model |
| `trajectory_jax/dag_gnn_nnx.py` | Real trainable DAG encoder | Used by supervised DAG-latent path |
| `trajectory_jax/presets.py` | Model/runtime presets | `midgpt_parity`, `midgpt_dag_latent` |
| `trajectory_jax/relation_parity.py` | Relation features and masks | Used for Adv-BMT parity |
| `trajectory_jax/tokenizer_parity.py` | Legacy-aligned tokenization | P0 parity path |
| `trajectory_jax/model.py` | Lightweight RL trajectory generator | Not the main supervised model |
| `trajectory_jax/unified_stub.py` | Future unified backbone stub | Explicit placeholder |
| `training/supervised.py` | Base supervised training loop | Prescan cache, eval, checkpoints, TensorBoard |
| `training/supervised_dag_latent.py` | DAG-latent staged training loop | Stage A/B/C, DAG alignment metrics |
| `training/dag_cache_schema.py` | DAG cache schema versions and validation | v2 compact10, v3 maneuver-outcome |
| `training/dag_cache.py` | DAG cache reader | Dual-read by default during migration |
| `training/dag_sources.py` | Cache/scene-derived DAG resolver | Training-time DAG source policy |
| `training/dag_tensorize.py` | DAG tensorization | Fixed `d_node_in=24` path for DAG encoder |
| `training/forward_metrics.py` | Forward realism/diversity metrics | Shared eval kernels |
| `training/tensorboard_logging.py` | TensorBoard helpers | Supports torch or tensorboard backend |
| `eval/head2head.py` | Multi-model comparison orchestrator | Main evaluation harness |
| `eval/v2_runner.py` | In-process v2 model inference | Supports DAG cache runtime inputs |
| `eval/legacy_runner.py` | Legacy subprocess execution | Avoids env conflicts |
| `eval/replay_export.py` | Replay export helpers | Batch artifact export |
| `eval/compare.py` | Aggregation, rankings, pairwise deltas | CSV/JSON report generation |
| `eval/visualize.py` | Overlay plots | GT + model comparisons |
| `perception/gpt4o.py` | GPT-4o perception model | Uses scene metadata, prompt grounding |
| `perception/mock.py` | Mock perception fallback | Useful for offline scaffolding |
| `llm/openai_client.py` | OpenAI client wrapper | Shared API access |
| `orchestration/pipeline.py` | End-to-end scene -> DAG -> rollout pipeline | Used mainly by RL/scaffold path |
| `judge/mock.py` | Mock trajectory judge | RL placeholder alignment path |
| `rl/behavior_embedding.py` | RL behavior manifold embedding | Partially placeholder for `dag_gnn` |
| `rl/novelty.py` | Novelty estimators | EMA Gaussian and KNN |
| `rl/consensus.py` | Cluster consensus logic | Quality-weighted consensus |
| `rl/thermostat.py` | Entropy thermostat | Adaptive novelty/consensus weights |
| `rl/reward.py` | RL reward composition | Environment + augmented terms |
| `rl/loop.py` | RL rollout collection logic | Includes VLM alignment replace mode |
| `rl/grpo.py` | GRPO stats/update scaffold | Not yet a true policy optimizer |
| `rl/topology.py` | RL topology branch | Interface + fallback descriptors |
| `rl/vlm_alignment.py` | RL VLM DAG-conformance scorer | Cost-bounded, cached, RL-only |
| `cli/train_nnx_bmt.py` | Base training CLI | Most common training entrypoint |
| `cli/train_nnx_bmt_dag_latent.py` | DAG-latent training CLI | Opt-in DAG path |
| `cli/train_rl_topo_mcpo.py` | RL CLI | Experimental/scaffold |
| `ADV_BMT_PARITY_CHECKLIST.md` | P0-P6 parity status | Source of truth for parity work |
| `ROADMAP.md` | Early roadmap notes | Useful historical context |

### `src/scripts/`

| Path | Purpose | Notes |
|---|---|---|
| `src/scripts/dag_cache/build_dag_cache_v2.py` | Build DAG caches from ScenarioNet + VLM + PromptBN | Main DAG cache entrypoint |
| `src/scripts/dag_cache/validate_cache_contract.py` | Validate DAG cache against contract/schema | Use after sharded builds |
| `src/scripts/dag_cache/inspect_dag_examples.py` | Human-readable DAG summaries | Quick inspection |
| `src/scripts/dag_cache/visualize_dag_json.py` | DAG graph visualization | Draw a `dag.json` or cache JSON |
| `src/scripts/dag_cache/import_legacy_dag_json.py` | Import legacy DAG outputs into cache schema | Migration tool |
| `src/scripts/eval/compare_models_head2head.py` | Main multi-model comparison entrypoint | Uses YAML registry |
| `src/scripts/eval/export_head2head_samples.py` | Curate explore/exploit samples from a head-to-head run | Supports paired scenes |
| `src/scripts/eval/run_legacy_model_worker.py` | Legacy worker subprocess | Called by legacy runner |
| `src/scripts/eval/run_legacy_forward_paper_eval.py` | Legacy paper-style forward evaluation | Useful for reported-number checks |
| `src/scripts/replay/export_forward_artifact_to_scenario.py` | Convert one forward artifact to replayable ScenarioNet data | Good for checkpoint inspection |
| `src/scripts/replay/export_forward_artifacts_batch.py` | Batch replay export | Used by head-to-head |
| `src/scripts/replay/make_scenario_gif.py` | Make GIFs from frames or direct ScenarioNet scene IDs | Current convenient qualitative tool |
| `src/scripts/parity/parity_report.py` | P6 parity harness | Writes JSON + Markdown |
| `src/scripts/parity/*` | P0-P5 parity checks | Tokenizer, relations, masks, forward metrics, resume, throughput |

### `src/counter_bmt/`

Treat this as utility legacy code, not the main training stack.

Important files still used:
- `src/counter_bmt/scenario_export.py`
- `src/counter_bmt/scenarionet_visualizer.py`
- `src/counter_bmt/dag_visualization.py`

### `src/Adv-BMT/`

This is the legacy reference implementation.

Important areas:
- configs:
  - `src/Adv-BMT/cfgs/motion_default.yaml`
  - `src/Adv-BMT/cfgs/0202_midgpt.yaml`
- tokenizer:
  - `src/Adv-BMT/bmt/tokenization/motion_tokenizers.py`
- model:
  - `src/Adv-BMT/bmt/models/`
- eval:
  - `src/Adv-BMT/bmt/eval/`

## 6. Training Paths That Matter

### Base supervised training

Entry point:
- `python -m counter_bmt_v2.cli.train_nnx_bmt`

What is stable:
- explicit or fallback train/val splits
- prescan cache with animated progress bar
- pmap single-host multi-GPU
- bf16 mixed precision
- forward eval metrics
- checkpoint resume with strict determinism checks
- TensorBoard logging

Main implementation:
- `src/counter_bmt_v2/training/supervised.py`

### DAG-latent staged training

Entry point:
- `python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent`

Stages:
- Stage A:
  - decoder pretraining with null latent / full DAG dropout
- Stage B:
  - fit DAG encoder and conditioning adapters
  - usually freeze non-DAG parameters
- Stage C:
  - joint finetuning with lower LR on decoder and higher LR on DAG modules

Critical implementation points:
- DAG source resolution:
  - `cache`, `scene_derived`, or `dual`
- DAG latent injection:
  - global gated residual in `NNXBidirectionalMotionTransformer`
- actual DAG encoder:
  - `NNXDAGGraphEncoder` in `src/counter_bmt_v2/trajectory_jax/dag_gnn_nnx.py`

### How DAG alignment is currently verified

This is one of the most important recent additions.

At eval time, `src/counter_bmt_v2/training/supervised_dag_latent.py` now evaluates the same batch three ways:
- with the real DAG
- with DAG inputs zeroed/null
- with DAG inputs shuffled across the batch

This yields `dag_alignment/*` metrics such as:
- `dag_alignment/loss_gain_vs_without_dag`
- `dag_alignment/loss_gain_vs_shuffled_dag`
- `dag_alignment/accuracy_gain_vs_without_dag`
- `dag_alignment/accuracy_gain_vs_shuffled_dag`

Interpretation:
- positive gain vs null DAG means the model is using some latent information
- positive gain vs shuffled DAG means it is using the correct DAG semantics rather than just any side channel

This is the main verifiable signal that Stage B/C is working.

## 7. DAG Contracts, Schemas, and Caches

### Current schema versions

- `counter_bmt_v2_dag_cache_v2_compact10`
- `counter_bmt_v2_dag_cache_v3_maneuver_outcome`

Current preferred contract:
- `maneuver_outcome_v1`

Core files:
- `src/counter_bmt_v2/causal/dag_contract.py`
- `src/counter_bmt_v2/training/dag_cache_schema.py`
- `src/counter_bmt_v2/training/dag_cache.py`

### Current DAG design direction

The current preferred DAG contract is intentionally simple:
- maneuver nodes
- outcome nodes
- maneuver interval metadata
  - `start_s`
  - `end_s`
  - `duration_s`
  - `mid_s`

Why:
- the latent should capture high-signal behavior structure
- unnecessary node-type complexity makes embeddings noisy
- interval features are useful for trajectory reconstruction

### DAG cache builder behavior

Entry point:
- `src/scripts/dag_cache/build_dag_cache_v2.py`

Important defaults:
- contract default: `maneuver_outcome_v1`
- hard contract enforcement
- frame renderer default: `scenarionet`
- dual-view default: off
- ego inset default: on
- frame annotation default: on

Outputs:
- `cache/<scenario_id>.json`
- `examples/<scenario_id>/dag.json`
- `examples/<scenario_id>/features.json`
- `examples/<scenario_id>/frame_manifest.json`
- `examples/<scenario_id>/dag_contract_report.json`
- `manifest.json`
- `results.jsonl`

### Training-time cache behavior

`DAGSourceResolver` handles this during DAG-latent training.

Modes:
- `cache`
  - use only cached DAGs
- `scene_derived`
  - deterministic fallback builder
- `dual`
  - try cache, then fallback

If you need a strict experiment where only real cached DAGs are used:
- use `--dag-source-mode cache`
- use `--dag-cache-strict`
- set `--dag-expected-schema`

## 8. Evaluation, Comparison, and Visualization

### Head-to-head comparison

Main entrypoint:
- `src/scripts/eval/compare_models_head2head.py`

This is the standard way to compare:
- v2 checkpoints against each other
- v2 against legacy Adv-BMT

It writes:
- `report.json`
- `report.md`
- `metrics/*.csv`
- `artifacts/<model_id>/step_eval/*.npz`
- overlay PNGs
- optional replay packages

### Curated sample export

Main entrypoint:
- `src/scripts/eval/export_head2head_samples.py`

Use this when you want:
- explore-prioritized examples
- exploit/consensus-prioritized examples
- replay packages
- GIFs

Important feature:
- `--paired-scenes`

That lets you export explore vs exploit using the same scenario IDs for direct qualitative comparison.

### Replay and GIF generation

Replay export:
- `src/scripts/replay/export_forward_artifact_to_scenario.py`
- `src/scripts/replay/export_forward_artifacts_batch.py`

GIF generation:
- `src/scripts/replay/make_scenario_gif.py`

`make_scenario_gif.py` now supports:
- frame directory mode
- direct ScenarioNet rendering mode
- scenario ID lookup, not just raw scenario index

### DAG visualization

Use:
- `src/scripts/dag_cache/visualize_dag_json.py`

This is the current way to render `dag.json` or cached DAG payloads into a viewable graph.

## 9. Parity Work

Parity scope is tracked in:
- `src/counter_bmt_v2/ADV_BMT_PARITY_CHECKLIST.md`

Current status:
- P0-P6 are implemented
- some gates are accepted with documented waivers rather than perfect pass
- parity harness is in place and reproducible

Useful docs:
- `docs/parity_p0.md`
- `docs/parity_p1.md`
- `docs/parity_p2.md`
- `docs/parity_p3.md`
- `docs/parity_p4.md`
- `docs/parity_p5.md`
- `docs/parity_p6.md`

One-command runner:
- `tools/run_parity_suite.sh`

## 10. RL Status: What Exists and What Does Not

This is a critical distinction for new agents.

### Real and usable today

- RL config surface in `src/counter_bmt_v2/config.py`
- reward shaping in `src/counter_bmt_v2/rl/reward.py`
- novelty estimator in `src/counter_bmt_v2/rl/novelty.py`
- consensus in `src/counter_bmt_v2/rl/consensus.py`
- thermostat in `src/counter_bmt_v2/rl/thermostat.py`
- VLM DAG-conformance replacement mode in:
  - `src/counter_bmt_v2/rl/vlm_alignment.py`
  - `src/counter_bmt_v2/rl/vlm_alignment_prompt.py`
  - `src/counter_bmt_v2/rl/vlm_alignment_evidence.py`

### Still scaffold / placeholder

- `src/counter_bmt_v2/trajectory_jax/model.py`
  - lightweight autoregressive trajectory generator for RL scaffolding
  - not the supervised NNX motion transformer
- `src/counter_bmt_v2/rl/grpo.py`
  - statistics/update scaffold, not true clipped policy optimization
- `src/counter_bmt_v2/rl/behavior_embedding.py`
  - RL `dag_gnn` path is not the same as the supervised NNX DAG encoder
  - it is still a handcrafted/fixed embedding path
- `src/counter_bmt_v2/rl/topology.py`
  - topology branch is interface-ready but still uses fallback descriptors rather than a real zigzag persistence backend
- `src/counter_bmt_v2/trajectory_jax/unified_stub.py`
  - explicit placeholder for future unified backbone

Bottom line:
- the supervised DAG-latent path is real and trainable
- the RL path is still an experimental scaffold

## 11. Environments and Machine Layout

### Local machine

Observed local repo path:
- `/mnt/d/Projects/CounterBMT`

This is a WSL/Linux view of a Windows drive checkout.

Observed local virtual environments in this checkout:
- `.venv`
- `.venv-legacy-paper`
- `.venv-waymo-convert`

Practical meaning:
- `.venv` is the main local development env used for most v2 work
- `.venv-legacy-paper` is a local legacy/paper-style env
- `.venv-waymo-convert` is a utility env and not the main training env

Local filesystem guidance:
- always prefer repo-relative paths in code and docs
- avoid hardcoding `/mnt/d/...` into scripts
- remember WSL path semantics differ from pure Linux paths

### Remote machine

Observed operational remote paths from project history:
- `~/CounterBMT`
- `/data/home/grads/jflashner/CounterBMT`

Observed main remote training environment:
- `.venv-v2`

Observed remote hardware/workflow:
- 4x H200 GPUs
- typical launch pattern:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - `XLA_PYTHON_CLIENT_PREALLOCATE=false`

Remote filesystem guidance:
- keep outputs on the remote machine for heavy training/eval
- move only the artifacts you need back locally
- do not write machine-specific absolute paths into committed configs unless the tool explicitly requires it

### Bootstrap profiles

Official bootstrap script:
- `tools/bootstrap_linux.sh`

Profiles:
- `v2`
  - default venv name: `.venv-v2`
- `legacy`
  - default venv name: `.venv-legacy`
- `full`
  - default venv name: `.venv-full`

These are the canonical environment profiles even if a given checkout also has custom env names.

### TensorBoard caveat

TensorBoard packaging on the remote machine has been finicky due `pkg_resources`, NumPy 2, and related dependency issues.

If plain `tensorboard` fails, use the pinned `uvx` workflow documented in:
- `docs/command_reference.md`
- `docs/training_tensorboard.md`

## 12. Datasets and Output Conventions

### Main datasets

- `data/scenarionet_waymo_training_500`
  - small stable dataset used for smoke tests, eval, DAG prototyping, and fast comparisons
- `data/_scenarionet_waymo_training_full_v12`
  - full training dataset
  - observed raw count from project history: `486995`
  - common batch-aligned count used in training: `486992`

### Output conventions

Most long-running jobs write under `outputs/`.

Common substructures:
- training run:
  - `outputs/<run>/checkpoints/`
  - `outputs/<run>/metrics.jsonl`
  - `outputs/<run>/run_config.json`
  - `outputs/<run>/summary.json`
  - `outputs/<run>/forward_eval_artifacts/`
  - `outputs/<run>/forward_eval_viz/`
  - `outputs/<run>/tensorboard/`
- DAG cache run:
  - `outputs/<dag_cache_run>/cache/`
  - `outputs/<dag_cache_run>/examples/`
  - `outputs/<dag_cache_run>/manifest.json`
- head-to-head eval:
  - `outputs/<head2head_run>/artifacts/`
  - `outputs/<head2head_run>/metrics/`
  - `outputs/<head2head_run>/viz/`
  - `outputs/<head2head_run>/replay/`

### Prescan cache

The slow training startup prescan is cached in two places:
- run-local:
  - `outputs/<run>/manifests/prescan_cache.pkl`
- dataset-keyed global cache:
  - `outputs/_prescan_cache/<hash>.pkl`

This matters because:
- startup should not rescan the full dataset every run
- compatible base and DAG-latent runs can share the same dataset-keyed prescan cache

## 13. Known Sharp Edges

1. `pmap` requires batch divisibility.
   - train and eval batch sizes must be divisible by number of devices.
   - otherwise `_shard_tree_for_pmap` will fail.

2. Resume strict determinism is real and enforced.
   - if split hashes differ, resume will fail unless you use `--no-resume-strict-determinism`.
   - this is expected for eval-only runs or intentionally changed datasets.

3. DAG-latent training only uses real cached DAGs if you force it to.
   - `dual` mode can silently fall back to scene-derived DAGs.
   - use cache-only strict mode if you want a real DAG-only experiment.

4. RL docs can look more complete than the actual RL training path.
   - always distinguish between the supervised DAG encoder and the RL scaffold.

5. The repo is large and mixed-purpose.
   - not every script under `src/scripts/` is part of the active mainline.
   - prefer current docs and `counter_bmt_v2/` entrypoints before reviving old scripts.

## 14. Recommended Reading Order For New Agents

If you are brand new to the repo, read in this order:

1. `docs/codex_agent_handoff.md`
2. `docs/command_reference.md`
3. `src/counter_bmt_v2/ADV_BMT_PARITY_CHECKLIST.md`
4. `docs/dag_latent_training.md`
5. `docs/head2head_eval.md`
6. `docs/dag_contract_maneuver_outcome_v1.md`
7. `docs/training_tensorboard.md`
8. `docs/rl_behavior_manifold_integration.md`
9. `docs/rl_vlm_alignment.md`

Then open the specific codepath you plan to edit.

## 15. Recommended First Code Reads By Task

### If working on supervised training

- `src/counter_bmt_v2/cli/train_nnx_bmt.py`
- `src/counter_bmt_v2/training/supervised.py`
- `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
- `src/counter_bmt_v2/trajectory_jax/presets.py`

### If working on DAG-latent conditioning

- `src/counter_bmt_v2/cli/train_nnx_bmt_dag_latent.py`
- `src/counter_bmt_v2/training/supervised_dag_latent.py`
- `src/counter_bmt_v2/trajectory_jax/dag_gnn_nnx.py`
- `src/counter_bmt_v2/training/dag_tensorize.py`
- `src/counter_bmt_v2/training/dag_sources.py`

### If working on DAG generation / VLM perception

- `src/scripts/dag_cache/build_dag_cache_v2.py`
- `src/counter_bmt_v2/perception/gpt4o.py`
- `src/counter_bmt_v2/causal/promptbn.py`
- `src/counter_bmt_v2/causal/dag_contract.py`
- `src/counter_bmt_v2/data/vlm_frame_prep.py`

### If working on evaluation / comparison / replay

- `src/scripts/eval/compare_models_head2head.py`
- `src/counter_bmt_v2/eval/head2head.py`
- `src/counter_bmt_v2/eval/v2_runner.py`
- `src/scripts/eval/export_head2head_samples.py`
- `src/scripts/replay/make_scenario_gif.py`

### If working on parity

- `src/counter_bmt_v2/ADV_BMT_PARITY_CHECKLIST.md`
- `src/scripts/parity/parity_report.py`
- the relevant `docs/parity_p*.md`

### If working on RL

Read these first and treat them as experimental:
- `src/counter_bmt_v2/cli/train_rl_topo_mcpo.py`
- `src/counter_bmt_v2/orchestration/pipeline.py`
- `src/counter_bmt_v2/rl/loop.py`
- `src/counter_bmt_v2/rl/reward.py`
- `src/counter_bmt_v2/rl/vlm_alignment.py`

## 16. Current Tests and Their Purpose

| Path | Purpose |
|---|---|
| `tests/test_dag_contract_v2.py` | DAG contract determinism and validation |
| `tests/test_vlm_alignment.py` | RL VLM alignment replacement behavior |
| `tests/test_dag_latent_alignment.py` | DAG-latent ablation helpers and real DAG encoder forward sanity |

If you change DAG contracts, DAG cache schema, VLM alignment, or DAG-latent eval semantics, these tests are the first ones to update and rerun.

## 17. Current Snapshot Of Important Completed Work

As of this document update, noteworthy implemented work includes:

1. Adv-BMT supervised parity path P0-P6, including one-command parity harness.
2. TensorBoard logging for both supervised CLIs.
3. Dataset-keyed prescan cache with animated startup progress bar.
4. DAG-latent opt-in training path with staged A/B/C schedule.
5. Real trainable supervised DAG encoder in JAX/NNX.
6. DAG alignment verification metrics using real/null/shuffled DAG eval ablations.
7. V2-native DAG cache builder using GPT-4o perception + PromptBN.
8. Hard DAG contract enforcement with maneuver-outcome cache schema support.
9. Head-to-head multi-model evaluation harness for v2 and legacy models.
10. Explore/exploit sample export with replay packages and GIF generation.
11. `make_scenario_gif.py` support for direct ScenarioNet scene IDs.
12. RL VLM DAG-conformance replacement mode, but only inside the RL scaffold.

## 18. Practical Rule Of Thumb

When deciding where to work:

- If the task is about training the actual model used in experiments, stay in `src/counter_bmt_v2/training/` and `src/counter_bmt_v2/trajectory_jax/`.
- If the task is about DAG generation quality, stay in `src/scripts/dag_cache/`, `src/counter_bmt_v2/perception/`, and `src/counter_bmt_v2/causal/`.
- If the task is about comparing models or producing figures, stay in `src/counter_bmt_v2/eval/` and `src/scripts/replay/`.
- If the task is about legacy parity or reproducing paper numbers, use `src/Adv-BMT/` only as a reference or subprocess target.
- If the task is about RL, assume you are improving an experimental scaffold unless you explicitly wire the supervised NNX model into it.
