# Counterfactual Path Remote Runbook

## Local smoke checklist
- Build only tiny local slices.
- Treat `path_index_raw.jsonl` as analysis-only.
- Treat `path_index_curated_train.jsonl` and `path_index_curated_val.jsonl` as the official training-facing artifacts.
- Use `CKPT_LOAD_MODE=forgiving_state_dict` when warm-starting path-control from the older forward-only checkpoint.
- Expect missing path-control-only keys during warm-start; those are reported, not treated as load failure.
- Do not run large indexing, sweeps, or training locally.

Local smoke commands:
```bash
export REPO_ROOT=/Users/joshuaflashner/Projects/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"

python "$REPO_ROOT/scripts/counterfactual/build_path_control_index.py" \
  --scenario-root "$REPO_ROOT/data/scenarionet_waymo_training_500" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_curated" \
  --max-scenarios 25 \
  --seed 0 \
  --num-workers 1 \
  --artifact-mode minimal \
  --dedup-mode overlap_cluster \
  --decision-time-merge-frames 5 \
  --window-overlap-threshold 0.5 \
  --anchor-dist-threshold 5.0 \
  --anchor-heading-threshold-rad 0.35 \
  --path-only-safe-mode \
  --write-examples-manifest \
  --write-histograms \
  --write-summary

python "$REPO_ROOT/scripts/counterfactual/split_path_index_by_scenario.py" \
  --path-index "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_curated/path_index_curated.jsonl" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_curated" \
  --seed 0 \
  --val-fraction 0.2

python "$REPO_ROOT/scripts/counterfactual/inspect_path_only_batch.py" \
  --out "$REPO_ROOT/outputs/counterfactual_path_smoke/path_only_batch_smoke.json" \
  --control-source "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_curated/path_index_curated_train.jsonl" \
  --data-dir "$REPO_ROOT/data/scenarionet_waymo_training_500" \
  --mode training \
  --batch-size 2 \
  --require-nonzero

python "$REPO_ROOT/scripts/counterfactual/summarize_path_index_quality.py" \
  --path-index "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_curated/path_index_curated.jsonl" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/quality"
```

Local artifact mapping:
- Curated path-index build corresponds to the remote path index build.
- `path_only_batch_smoke.json` corresponds to remote training input wiring.
- Curated quality summary corresponds to the remote pre-training label audit.

## Remote path index build
Prerequisites:
- A machine with the full repo checkout.
- Python environment that can import `metadrive`, `scenarionet`, and `src/Adv-BMT`.
- Read access to the scenario root.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export SCENARIO_ROOT=/path/to/scenario_root
export OUTDIR="$REPO_ROOT/outputs/pr6_path_index_5000"
export MAX_SCENARIOS=5000
export NUM_WORKERS=8
export SEED=0
export VAL_FRACTION=0.2
```

Exact copy-paste command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr6_path_index.sh"
```

If class support is weak, rerun at 20k:
```bash
export OUTDIR="$REPO_ROOT/outputs/pr6_path_index_20000"
export MAX_SCENARIOS=20000
bash "$REPO_ROOT/scripts/remote/run_pr6_path_index.sh"
```

Expected output files:
- `$OUTDIR/path_index_raw.jsonl`
- `$OUTDIR/path_index_curated.jsonl`
- `$OUTDIR/path_index_curated_train.jsonl`
- `$OUTDIR/path_index_curated_val.jsonl`
- `$OUTDIR/path_support_summary_curated.json`
- `$OUTDIR/path_label_histograms_curated.json`
- `$OUTDIR/split_summary.json`

How to resume/restart:
- Prefer a fresh `OUTDIR` for each run.
- Reuse the same `SEED` and `VAL_FRACTION` when comparing support across runs.

Common failure modes:
- Very slow runs: reduce `NUM_WORKERS` if memory pressure is high.
- Weak class support: inspect `path_support_summary_curated.json`.
- Unexpectedly low retained rows: inspect `curated_filter_summary.json` and `light_canonicalization_summary.json`.

What artifacts to inspect first:
- `path_support_summary_curated.json`
- `path_label_histograms_curated.json`
- `split_summary.json`
- `curated_filter_summary.json`

## Remote training run
Prerequisites:
- A forward-only checkpoint to finetune from.
- A curated split directory from the previous step.
- GPU machine with enough memory for the chosen batch size.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export DATA_ROOT=/path/to/scenario_root
export CONTROL_INDEX_DIR=/path/to/pr6_path_index_5000
export CONTROL_INDEX_TRAIN="$CONTROL_INDEX_DIR/path_index_curated_train.jsonl"
export CONTROL_INDEX_VAL="$CONTROL_INDEX_DIR/path_index_curated_val.jsonl"
export FORWARD_CKPT=/path/to/forward_only.ckpt
export OUTDIR="$REPO_ROOT/logs/pr6_path_control"
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=8
export BATCH_SIZE=8
export MAX_STEPS=5000
export VAL_INTERVAL=500
export SEED=0
export CKPT_LOAD_MODE=forgiving_state_dict
export WANDB_ENABLED=true
export WANDB_PROJECT=infgen
export WANDB_ENTITY=your_wandb_entity
export WANDB_GROUP=pr6_path_control
# Prefer one of:
export WANDB_API_KEY=your_wandb_api_key
# or:
# export WANDB_API_KEY_FILE=$HOME/wandb_api_key_file.txt
```

Exact copy-paste command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr6_train_path_strict_local.sh"
```

Optional local-only micro smoke if the environment can import the full model path:
```bash
python "$REPO_ROOT/src/Adv-BMT/bmt/train_motion.py" \
  --config-name motion_forward_path_control_strict_local.yaml \
  DATA.TRAINING_DATA_DIR="$REPO_ROOT/data/scenarionet_waymo_training_500" \
  DATA.TEST_DATA_DIR="$REPO_ROOT/data/scenarionet_waymo_training_500" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN="$REPO_ROOT/outputs/counterfactual_path_index_pr62_curated_100/splits/path_index_curated_train.jsonl" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$REPO_ROOT/outputs/counterfactual_path_index_pr62_curated_100/splits/path_index_curated_val.jsonl" \
  limit_train_batches=1 \
  limit_val_batches=1 \
  max_steps=20 \
  val_interval=10 \
  batch_size=2 \
  val_batch_size=2 \
  num_workers=0 \
  val_num_workers=0 \
  wandb=false
```

Expected output files:
- `$OUTDIR/lightning_logs/.../last.ckpt`
- `$OUTDIR/lightning_logs/.../config.yaml`

How to resume/restart:
- Keep train and val indexes fixed for a given run.
- Reuse `OUTDIR` only if overwriting or resuming is intended.

Common failure modes:
- Missing checkpoint path.
- Mismatched curated split and dataset root.
- Expected missing path-control keys during warm-start; inspect the checkpoint load report before treating this as a failure.
- WandB auth missing; set `WANDB_API_KEY` or `WANDB_API_KEY_FILE`.
- OOM with large `BATCH_SIZE`; reduce batch size or workers.

What artifacts to inspect first:
- saved `config.yaml`
- path-head and anchor-loss curves
- batch-level control availability on the curated train split

## Remote eval sweep
Prerequisites:
- A trained path-control checkpoint.
- A curated validation split.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export CKPT=/path/to/path_control.ckpt
export CONTROL_INDEX_DIR=/path/to/pr6_path_index_5000
export CONTROL_INDEX_VAL="$CONTROL_INDEX_DIR/path_index_curated_val.jsonl"
export OUTDIR="$REPO_ROOT/outputs/pr6_path_eval"
export NUM_EXAMPLES=200
export SEED=0
export BATCH_SIZE=1
export NUM_WORKERS=0
export CKPT_LOAD_MODE=forgiving_state_dict
```

Exact copy-paste command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr6_eval_path_control.sh"
```

Expected output files:
- `$OUTDIR/eval_smoke_summary.json`
- `$OUTDIR/eval_smoke_examples.jsonl`
- `$OUTDIR/materialized_eval_inputs/...` for lazily materialized debug inputs when needed

How to resume/restart:
- Use a fresh `OUTDIR` per checkpoint or eval split.
- Rerun with the same `SEED` for reproducible sampling.

Common failure modes:
- Missing checkpoint.
- Scenario paths in the curated split not accessible on the eval machine.
- Eval materialization failing for a selected row; inspect `materialized_eval_inputs`.
- Checkpoint warm-start reporting expected missing path-control keys from an older forward-only checkpoint.

What artifacts to inspect first:
- `eval_smoke_summary.json`
- a handful of rows from `eval_smoke_examples.jsonl`

## Remote post-train runtime probe
Prerequisites:
- A trained strict-local checkpoint.
- The curated validation split.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export CKPT=/path/to/path_control.ckpt
export CONTROL_INDEX_DIR=/path/to/pr6_path_index_5000
export CONTROL_INDEX_VAL="$CONTROL_INDEX_DIR/path_index_curated_val.jsonl"
export DATA_ROOT=/path/to/scenario_root
export OUTDIR="$REPO_ROOT/outputs/pr6_posttrain_probe"
export BATCH_SIZE=2
export CKPT_LOAD_MODE=forgiving_state_dict
```

Exact copy-paste command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr6_posttrain_probe.sh"
```

Expected output files:
- `$OUTDIR/batch_control_summary.json`
- `$OUTDIR/forward_control_runtime.json`
- `$OUTDIR/selected_control.json`
- `$OUTDIR/control_pos_mask.npy`

How to resume/restart:
- Reuse `OUTDIR` only if overwriting is acceptable.
- Keep `BATCH_SIZE` small so the selected example is easy to inspect.

Common failure modes:
- `CONTROL_INDEX_VAL` pointing at the raw index instead of the curated val split.
- `DATA_ROOT` not matching the scenarios referenced by the curated split.
- Checkpoint config mismatch; use `CKPT_LOAD_MODE=forgiving_state_dict` for forward-only warm-start compatibility.

What artifacts to inspect first:
- `forward_control_runtime.json`
- `batch_control_summary.json`

## Artifact checklist to send back
- Curated path index build:
  `path_support_summary_curated.json`
  `path_label_histograms_curated.json`
  `curated_filter_summary.json`
  `path_index_curated_train.jsonl`
  `path_index_curated_val.jsonl`
  `split_summary.json`
- Training:
  final checkpoint path
  saved `config.yaml`
  learning curves for motion/path/anchor
- Eval:
  `eval_smoke_summary.json`
  `eval_smoke_examples.jsonl`
- Post-train probe:
  `forward_control_runtime.json`
  `batch_control_summary.json`

Raw index note:
- `path_index_raw.jsonl` is for redundancy analysis and corpus debugging only.
- Do not use the raw index as the default training or eval source.
