# P0 Tokenizer Parity (Vendored-Lite)

This document summarizes the P0 tokenizer parity implementation for `counter_bmt_v2`.

## What was implemented

- Added vendored, NumPy-only parity tokenizer:
  - `src/counter_bmt_v2/trajectory_jax/tokenizer_parity.py`
  - forward/backward tokenization with 33x33 action bins
  - hole-filling for rare valid-mask gaps
  - GPT-style start/end token semantics
  - add/remove agent handling with `ALLOW_SKIP_STEP` semantics
- Added training integration and mode switch:
  - `SupervisedTrainConfig.tokenizer_mode` in `src/counter_bmt_v2/training/supervised.py`
  - `--tokenizer-mode {paper_simple,adv_bmt_parity}` in `src/counter_bmt_v2/cli/train_nnx_bmt.py`
- Added forward-eval horizon safety clamp:
  - `horizon_eval = min(target_mask_len, sample_steps_len - 1)` in `src/counter_bmt_v2/training/forward_metrics.py`
- Added parity scripts:
  - `src/scripts/parity/compare_tokenization.py`
  - `src/scripts/parity/export_legacy_batch.py`

## Dependency policy

- The v2 parity training path does not import TensorFlow/Waymo stacks.
- Optional legacy side-by-side checks use `--legacy-check` and require a separate legacy-compatible environment.

## Commands

### 1) Forward smoke (no legacy env needed)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --mode forward \
  --n 20 \
  --batch-size 4 \
  --skip-steps 5
```

### 2) Backward smoke (no legacy env needed)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --mode backward \
  --n 20 \
  --batch-size 4 \
  --skip-steps 5
```

### 3) Training smoke with parity tokenizer

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p0_smoke_parity \
  --tokenizer-mode adv_bmt_parity \
  --model-preset paper_like_small \
  --max-steps 80 \
  --batch-size 2 \
  --mode mixed \
  --reverse-prob 0.5 \
  --skip-steps 5 \
  --eval-every 40 \
  --eval-batches 2 \
  --log-every 10
```

### 4) Optional legacy side-by-side check (legacy env required)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --mode forward \
  --n 100 \
  --batch-size 4 \
  --skip-steps 5 \
  --legacy-check \
  --legacy-root src/Adv-BMT \
  --min-token-match 0.999 \
  --min-valid-mask-match 0.999
```

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_tokenization.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --mode backward \
  --n 100 \
  --batch-size 4 \
  --skip-steps 5 \
  --legacy-check \
  --legacy-root src/Adv-BMT \
  --min-token-match 0.995 \
  --min-valid-mask-match 0.999
```

### 5) Export a debug bundle for one scenario

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/export_legacy_batch.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --index 0 \
  --mode forward \
  --skip-steps 5 \
  --out outputs/parity/export_batch_forward
```

