# P3 Data Protocol + Split/Loader Parity

This document tracks P3 parity checks for dataset source resolution, split/interval behavior, strict horizon validation, and reproducible manifests.

## Scope
- Explicit split mode: `--train-data-dir` + `--val-data-dir`.
- Fallback mode: `--data-dir` with seeded train/val split.
- Interval parity: split first, then apply `sample_interval_training/test`.
- Pre-filter invalid scenarios once, with skip/truncation reports.
- Strict 91-step validation is opt-in (`--strict-91-steps`).

## Artifacts
Each run writes deterministic manifests to:
- `output_dir/manifests/train_manifest.json`
- `output_dir/manifests/val_manifest.json`
- `output_dir/manifests/train_ids.txt`
- `output_dir/manifests/val_ids.txt`
- `output_dir/manifests/skipped_scenarios.json`
- `output_dir/manifests/truncation_report.json`

Run metadata in `output_dir/run_config.json` includes:
- `data_source_mode` (`explicit_split` or `fallback_split`)
- `resolved_data_dirs`
- `split_settings`
- `artifacts`
- skip/truncation counters

## Commands

1. Dataset index parity (legacy summary count semantics)
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_dataset_index.py \
  --train data/scenarionet_waymo_training_500 \
  --val data/scenarionet_waymo_training_500 \
  --sample-interval-training 2 \
  --sample-interval-test 3
```

2. Explicit split smoke
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir data/scenarionet_waymo_training_500 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p3_smoke_explicit \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --sample-interval-training 2 \
  --sample-interval-test 3 \
  --max-steps 1 --batch-size 1 \
  --eval-every 1 --eval-batches 1 --log-every 1 \
  --no-forward-eval
```

3. Fallback split + interval smoke
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p3_smoke_fallback_a \
  --train-fraction 0.8 \
  --sample-interval-training 2 \
  --sample-interval-test 3 \
  --seed 7 \
  --max-steps 1 --batch-size 1 \
  --eval-every 1 --eval-batches 1 --log-every 1 \
  --no-forward-eval
```

4. Determinism check for fallback manifests
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p3_smoke_fallback_b \
  --train-fraction 0.8 \
  --sample-interval-training 2 \
  --sample-interval-test 3 \
  --seed 7 \
  --max-steps 1 --batch-size 1 \
  --eval-every 1 --eval-batches 1 --log-every 1 \
  --no-forward-eval

cmp outputs/p3_smoke_fallback_a/manifests/train_ids.txt outputs/p3_smoke_fallback_b/manifests/train_ids.txt
cmp outputs/p3_smoke_fallback_a/manifests/val_ids.txt outputs/p3_smoke_fallback_b/manifests/val_ids.txt
```

5. Strict 91 fail-fast check (using a 20s conversion directory)
```bash
FULL_DIR=$(ls -d data/_scenarionet_waymo_training_full_v12_* | head -n 1)

PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir "$FULL_DIR" \
  --val-data-dir "$FULL_DIR" \
  --strict-91-steps \
  --num-train-scenarios 5 \
  --num-val-scenarios 5 \
  --output-dir outputs/p3_strict91_check \
  --max-steps 1 --batch-size 1 \
  --no-forward-eval
```

6. Non-strict truncation reporting sanity
```bash
FULL_DIR=$(ls -d data/_scenarionet_waymo_training_full_v12_* | head -n 1)

PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir "$FULL_DIR" \
  --val-data-dir "$FULL_DIR" \
  --num-train-scenarios 5 \
  --num-val-scenarios 5 \
  --output-dir outputs/p3_non_strict_check \
  --max-steps 1 --batch-size 1 \
  --no-forward-eval
```

## Expected Gates
- `compare_dataset_index.py`: train/val counts match legacy summary count semantics after interval slicing.
- Explicit/fallback smoke runs:
  - complete train/eval + checkpoint save,
  - generate all manifest files,
  - write correct `data_source_mode` in `run_config.json`.
- Fallback determinism: `train_ids.txt` and `val_ids.txt` are identical across same-seed reruns.
- Strict mode: run fails before train loop when non-91 horizons exist, with clear reports.
- Non-strict mode: run continues and records nonzero truncation candidates when horizons exceed `max_time_steps`.
