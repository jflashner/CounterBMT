# Adv-BMT Parity Checklist (v2, Non-RL)

This checklist defines what is still required to align `counter_bmt_v2` with the supervised Adv-BMT stack.

Scope:
- In scope: tokenizer, scene encoder, decoder behavior, data protocol, training/eval parity.
- Out of scope: RL parity (`src/counter_bmt_v2/rl`), which is tracked separately.

Target references:
- `src/Adv-BMT/cfgs/motion_default.yaml`
- `src/Adv-BMT/cfgs/0202_midgpt.yaml`
- `src/Adv-BMT/bmt/tokenization/motion_tokenizers.py`
- `src/Adv-BMT/bmt/models/gpt_scene_encoder.py`
- `src/Adv-BMT/bmt/models/motion_decoder_gpt.py`
- `src/Adv-BMT/bmt/eval/scenario_evaluator.py`

## P0. Tokenizer + Bidirectional Supervision Parity

Status: [x] Implemented (validation in progress)

Dependency note:
- P0 parity path in `counter_bmt_v2` does **not** import TensorFlow/Waymo stacks at runtime.
- Optional legacy side-by-side checks (`--legacy-check`) are isolated and require a separate legacy-compatible env.

Code tasks:
- [x] Add a dedicated parity tokenizer module in `src/counter_bmt_v2/trajectory_jax/tokenizer_parity.py` that mirrors `BicycleModelTokenizerFixed0124` behavior from legacy.
- [x] Implement forward tokenization path with hole-filling, skip-step handling, and add/remove agent semantics (including validity transitions).
- [x] Implement backward tokenization path with explicit reverse dynamics and backward-specific token/mask construction (not sequence reversal approximation).
- [x] Add special token handling parity in the v2 train batch prep (`src/counter_bmt_v2/training/supervised.py`), including start/end semantics where applicable.
- [x] Add a CLI switch in `src/counter_bmt_v2/cli/train_nnx_bmt.py` to select tokenizer mode (`paper_simple`, `adv_bmt_parity`).

Acceptance tests:
- [x] Smoke: `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py --data-dir data/scenarionet_waymo_training_500 --mode forward --n 2 --batch-size 1 --skip-steps 5` runs with no invalid token IDs.
- [x] Smoke: `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py --data-dir data/scenarionet_waymo_training_500 --mode backward --n 2 --batch-size 1 --skip-steps 5` runs with no invalid token IDs.
- [x] Smoke: `PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt ... --tokenizer-mode adv_bmt_parity --max-steps 1` completes training/eval and writes checkpoints.
- [ ] Legacy parity gate: `--legacy-check` forward token ID exact-match >= 99.9%.
- [ ] Legacy parity gate: `--legacy-check` backward token ID exact-match >= 99.5% and valid-mask exact-match >= 99.9%.
- [ ] For a fixed seed and fixed scenario subset, target sequence lengths and counts of special tokens match legacy exactly.

## P1. Scene Encoder + Relation Graph Parity

Status: [x] Implemented (legacy side-by-side gate in progress)

Code tasks:
- [x] Extend relation feature construction in v2 to match legacy relation definitions and dimensions used by scene/decoder attention.
  Notes: `src/counter_bmt_v2/trajectory_jax/relation_parity.py` now builds `scene_s2s (3)`, `a2a (12)`, `a2t (12)`, `a2s (3)` in MidGPT simple-relation mode.
- [x] Implement KNN/sparsification behavior parity and relation masks in `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`.
  Notes: scene S2S relation attention path is enabled via `NNXRelationParityConfig`; relation mask/indices are emitted for debug metadata.
- [x] Add Fourier relation embeddings compatible with the legacy relation pipeline.
  Notes: `src/counter_bmt_v2/trajectory_jax/fourier_embedding_nnx.py` and scene-relation stack integration in `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`.
- [x] Add support for parity config toggles from `0202_midgpt` relevant to scene encoding (`REMOVE_TRAFFIC_LIGHT_STATE`, `SIMPLE_RELATION`, `PER_CONTOUR_POINT_RELATION`, etc.).
  Notes: parity toggles are defined in `NNXRelationParityConfig` and exposed via `midgpt_parity_config()`.
- [x] Add an intermediate dump path for relation tensors and masks for side-by-side debugging.
  Notes: `src/scripts/parity/export_relation_batch.py` and train-time dump hook in `src/counter_bmt_v2/training/supervised.py`.

Acceptance tests:
- [x] Smoke: `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_relations.py --data-dir data/scenarionet_waymo_training_500 --target scene_s2s --mode simple --n 20 --batch-size 4 --skip-steps 5` runs with no NaNs/shape errors.
- [x] Export sanity: `PYTHONPATH=src .venv/bin/python src/scripts/parity/export_relation_batch.py --data-dir data/scenarionet_waymo_training_500 --index 0 --out outputs/parity/relation_batch_0` writes `.npz` + `.json`.
- [x] Training smoke: `PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt --data-dir data/scenarionet_waymo_training_500 --output-dir outputs/p1_smoke_scene_rel --model-preset midgpt_parity --tokenizer-mode adv_bmt_parity --max-steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --log-every 1 --checkpoint-every 2 --num-train-scenarios 4 --num-val-scenarios 2` completes with checkpoint save.
- [x] Legacy parity gate: `compare_relations.py --legacy-check` passes with max abs diff <= 1e-5 and exact mask match (validated on `n=100`, `batch_size=4`).
- [ ] Relation edge count statistics (mean neighbors, sparsity) are within 1% of legacy on a 500-scenario sample.

## P2. Decoder Input/Attention Semantics Parity

Status: [x] Implemented (validation in progress)

Code tasks:
- [x] Add parity motion-feature embedding path in v2 decoder input assembly (matching legacy token-to-motion feature conditioning intent).
  Notes: `NNXDecoderParityConfig` + `_compose_decoder_tokens_parity` in `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py` now uses legacy-style motion feature composition (`[acc,yaw,dist,heading] + modeled_agent_delta`).
- [x] Add parity special-token embedding behavior and backward-indicator embedding controls.
  Notes: special token classes (normal/start/end/mask) are mapped from model token IDs and passed through parity composer with optional backward-indicator embedding.
- [x] Align temporal/agent causal mask semantics with legacy block-causal behavior.
  Notes: decoder now consumes explicit `a2a_mask/a2t_mask/a2s_mask`; A2T combines relation-valid and lower-triangular causal masks.
- [x] Add config toggles for parity-critical decoder options (`ADD_CONTOUR_RELATION`, `ADD_RELATION_TO_V`, `REMOVE_REL_NORM`, `IS_V7` equivalents where feasible).
  Notes: relation toggles are wired through dense masked relation attention (`q·k + q_rel·rel_k`, value path `v + rel_v|rel_k`) with `ADD_RELATION_TO_V` and `REMOVE_REL_NORM`.
- [x] Keep existing clean defaults; parity mode should be opt-in.
  Notes: parity decoder path is only active under `midgpt_parity`/decoder parity config and does not change `paper_like_*` defaults.

Acceptance tests:
- [x] `PYTHONPATH=src .venv/bin/python src/scripts/parity/inspect_decoder_masks.py --data-dir data/scenarionet_waymo_training_500 --n 20 --batch-size 4 --skip-steps 5` confirms exact A2T causal-valid mask parity.
- [x] `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_decoder_inputs.py --data-dir data/scenarionet_waymo_training_500 --n 50 --batch-size 4 --skip-steps 5 --legacy-check --legacy-root src/Adv-BMT --max-embedding-diff 2e-4 --min-mask-match 0.9995` passes gates.
- [ ] `PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt --data-dir data/scenarionet_waymo_training_500 --output-dir outputs/p2_smoke_decoder_parity --model-preset midgpt_parity --tokenizer-mode adv_bmt_parity --max-steps 80 --batch-size 2 --mode mixed --reverse-prob 0.5 --skip-steps 5 --eval-every 40 --eval-batches 2 --log-every 10` completes with checkpoint and no parity runtime errors.

## P3. Data Protocol + Split/Loader Parity

Status: [x] Implemented (order parity check remains optional)

Code tasks:
- [x] Add explicit `--train-data-dir` and `--val-data-dir` flags in `src/counter_bmt_v2/cli/train_nnx_bmt.py` (keep `--data-dir` as fallback).
- [x] Add `sample_interval_training` and `sample_interval_test` controls to align with legacy datamodule behavior.
- [x] Ensure loader skip behavior for invalid/no-map scenarios matches legacy expectations.
- [x] Add optional strict 91-step mode and explicit truncation reporting for 20s WOMD conversion inputs.
- [x] Emit train/val scenario ID manifests into output dir for reproducibility.
- [x] Harden deterministic discovery ordering in `src/counter_bmt_v2/data/scenarionet.py` using stable relative-path sorting.

Acceptance tests:
- [x] `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_dataset_index.py --train data/scenarionet_waymo_training_500 --val data/scenarionet_waymo_training_500 --sample-interval-training 2 --sample-interval-test 3` passes count-parity checks.
- [x] Explicit split smoke run writes manifests and records `data_source_mode=explicit_split` in `run_config.json`.
- [x] Fallback split smoke run writes manifests and records `data_source_mode=fallback_split` in `run_config.json`.
- [x] Scenario ID manifests are stable across reruns with same seed/config (`train_ids.txt` and `val_ids.txt` exact match across two fallback runs).
- [x] Strict 91 fail-fast (`--strict-91-steps`) aborts before train loop and writes `truncation_report.json` + `skipped_scenarios.json`.
- [x] Non-strict mode reports truncation candidates in `truncation_report.json`.
- [ ] Optional exact filename/order match against legacy summary list (`compare_dataset_index.py --check-order`) for all dataset variants.

## P4. Evaluation Metric Parity (Forward Pass)

Status: [x]

Code tasks:
- [x] Add a "strict parity evaluator" workflow using offline strict comparison in `src/scripts/parity/compare_forward_metrics.py` (with Waymo TTC operator path when available).
- [x] Keep current NumPy/JAX approximations as fallback mode, and separate outputs clearly (`forward_approx/*` in training/eval logs, `forward_parity/*` in strict report JSON).
- [x] Align core+realism metric histogram bins/ranges and aggregation conventions with legacy-style evaluator semantics for strict comparison scope.
- [x] Add tooling to evaluate the same prediction artifacts in both approximate and strict evaluators.

Acceptance tests:
- [x] `PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_forward_metrics.py --artifact-dir outputs/p4_smoke_approx/forward_eval_artifacts --output-json outputs/p4_forward_parity_report.json --max-rel-error 0.01 --min-corr 0.99` passes the strict comparison gate.
- [x] Scenario-level metric correlation gate (`>= 0.99`) is enforced and reported by `compare_forward_metrics.py`.
- [x] Existing training/eval runs still work with approximate mode when strict operators are unavailable (`outputs/p4_no_strict_deps` smoke run).
- [x] Approx path regression smoke writes `forward_approx/*` metrics and per-scenario artifacts under `forward_eval_artifacts/`.
- [x] Artifact schema sanity check passes for exported `.npz` keys.

## P5. Training Runtime + Config Parity

Status: [x] Implemented (validation in progress)

Code tasks:
- [x] Add a parity runtime preset in `src/counter_bmt_v2/trajectory_jax/presets.py` that maps directly to Adv-BMT-supervised defaults (`LR`, `warmup`, heads/layers/d_model, skip steps).
- [x] Add distributed training path for single-host multi-GPU nodes in `src/counter_bmt_v2/training/supervised.py` (`pmap` data parallel path).
- [x] Add bf16 mixed precision toggle for parity with production-scale training throughput.
- [x] Add exact learning-rate schedule parity checks against legacy schedule formulas.
- [x] Ensure checkpoint resume persists deterministic runtime state (epoch cursor, epoch permutation, RNG state, split hash) and enforces strict resume checks.
- [x] Add parity scripts:
  - `src/scripts/parity/check_lr_schedule.py`
  - `src/scripts/parity/check_resume_determinism.py`
  - `src/scripts/parity/benchmark_throughput.py`
- [x] Add P5 workflow docs in `docs/parity_p5.md`.

Acceptance tests:
- [ ] `python src/scripts/parity/check_lr_schedule.py --steps 0,1,100,2000,10000` matches legacy LR values to <= 1e-9 absolute error.
- [ ] 4-GPU training launch runs end-to-end and shows >= 3.0x throughput vs single-GPU baseline on same global batch.
- [ ] Resume test (`--resume-checkpoint`) reproduces next-100-step loss curve with <= 1e-6 mean absolute deviation.

## P6. Parity Harness + Gate

Status: [ ]

Code tasks:
- [ ] Add `src/scripts/parity/` suite for repeatable side-by-side checks:
- [ ] `export_legacy_batch.py`
- [ ] `compare_tokenization.py`
- [ ] `compare_relations.py`
- [ ] `compare_decoder_inputs.py`
- [ ] `compare_forward_metrics.py`
- [ ] Add a single aggregation script `parity_report.py` that writes JSON + Markdown summary.
- [ ] Add a one-command runner `tools/run_parity_suite.sh`.

Acceptance tests:
- [ ] `bash tools/run_parity_suite.sh` exits 0 and writes `outputs/parity_report/latest.json`.
- [ ] Report includes pass/fail for all P0-P5 gates and artifact links for failures.
- [ ] Parity suite is documented with exact setup and expected runtime.

## Definition of Done

Project is considered "Adv-BMT aligned (non-RL)" when:
- [ ] P0-P4 acceptance tests pass.
- [ ] P5 has either full pass, or a documented waiver for features intentionally deferred.
- [ ] Parity report artifact is reproducible on a second machine with the same code/data.
