# P1 Scene/Relation Parity (MidGPT-First)

This document summarizes the P1 implementation for scene encoder + relation graph parity in `counter_bmt_v2`.

## What was implemented

- Added relation parity core:
  - `src/counter_bmt_v2/trajectory_jax/relation_parity.py`
  - Legacy-style primitives (`pairwise_mask`, `pairwise_relative_diff`, `rotate_local`, `cal_polygon_contour`)
  - `compute_relation_simple_parity(...)` and `build_relation_bundle(...)`
  - Scene relation input builder (`build_scene_token_relation_inputs_np(...)`)
- Added NNX Fourier relation embedding:
  - `src/counter_bmt_v2/trajectory_jax/fourier_embedding_nnx.py`
- Integrated scene relation parity into NNX model:
  - `src/counter_bmt_v2/trajectory_jax/nnx_bmt.py`
  - `NNXRelationParityConfig`
  - map heading derivation + traffic-light placeholder heading behavior
  - scene self-attention relation stack with Fourier embedding
  - optional scene relation metadata return path
- Added MidGPT parity preset:
  - `src/counter_bmt_v2/trajectory_jax/presets.py` (`midgpt_parity_config`)
- Added parity scripts:
  - `src/scripts/parity/compare_relations.py`
  - `src/scripts/parity/export_relation_batch.py`
- Added train-time relation debug dump hook:
  - `src/counter_bmt_v2/training/supervised.py`
  - CLI wiring in `src/counter_bmt_v2/cli/train_nnx_bmt.py`

## Dependency policy

- v2 runtime/training path remains TensorFlow/Waymo-runtime independent.
- Optional legacy checks (`--legacy-check`) are isolated to parity scripts.

## Commands

### 1) Relation smoke (no legacy env required)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_relations.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --target scene_s2s \
  --mode simple \
  --n 20 \
  --batch-size 4 \
  --skip-steps 5
```

### 2) Legacy side-by-side relation check (optional)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_relations.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --target scene_s2s \
  --mode simple \
  --n 100 \
  --batch-size 4 \
  --skip-steps 5 \
  --legacy-check \
  --legacy-root src/Adv-BMT \
  --max-feat-diff 1e-5 \
  --min-mask-match 1.0
```

Current validation snapshot:
- `n=100`, `batch_size=4` scene S2S run passes with:
  - `feat_abs_max = 0.0`
  - `mask_exact_match_rate = 1.0`
  - `index_exact_match_rate = 1.0`

### 3) Export relation bundle for one scenario

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/export_relation_batch.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --index 0 \
  --skip-steps 5 \
  --out outputs/parity/relation_batch_0
```

### 4) MidGPT parity training smoke

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p1_smoke_scene_rel \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --max-steps 80 \
  --batch-size 2 \
  --eval-every 40 \
  --eval-batches 2 \
  --log-every 10
```

### 5) Optional train-time relation dumps

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p1_with_relation_dumps \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --max-steps 80 \
  --batch-size 2 \
  --relation-debug-dump-dir outputs/parity/relation_debug \
  --relation-debug-dump-every 20 \
  --relation-debug-max-batches 4
```
