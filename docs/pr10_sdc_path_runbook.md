# PR10 SDC-Path Runbook

## Scope
- Use `sdc_path_control_v1` rows only.
- Control only the SDC.
- Start control at `current_time_index`.
- Use semantic label + selected path geometry + separability.
- Do not use path IDs as learned inputs.
- Do not rely on anchors or exit gates for PR10.

## Local smoke checklist
- Build only a tiny local index slice.
- Run the dataset batch smoke before any training.
- Run the no-grad loss smoke against the warm-start checkpoint.
- Inspect the local/world BEV plots and separability profile plots.
- Treat large missing-key reports for the new SDC-path modules as expected during warm-start.

Local smoke commands:
```bash
export REPO_ROOT=/Users/joshuaflashner/Projects/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"

.venv-mac/bin/python "$REPO_ROOT/scripts/counterfactual/build_sdc_path_control_index.py" \
  --semantics-index "$REPO_ROOT/outputs/waymax_sdc_path_semantics_vlm_examples_10_30_v19_gpt54_original_no_unknown/sdc_path_semantics_index.jsonl" \
  --outdir "$REPO_ROOT/outputs/pr10_sdc_path_smoke" \
  --output-name sdc_path_control_index_smoke.jsonl \
  --max-examples 2 \
  --debug-max-rows 2

.venv-mac/bin/python "$REPO_ROOT/scripts/counterfactual/smoke_sdc_path_control.py" \
  --control-index "$REPO_ROOT/outputs/pr10_sdc_path_smoke/sdc_path_control_index_smoke.jsonl" \
  --data-dir "$REPO_ROOT/outputs/pr10_sdc_path_smoke/scenario_root" \
  --ckpt "$REPO_ROOT/outputs/pr6_eval_debug_bundle_20260403/logs/pr6_path_control_5000_wandb_20260401_run1/lightning_logs/infgen/pr6_path_control_5000_20260401_run1/checkpoints/last.ckpt" \
  --outdir "$REPO_ROOT/outputs/pr10_sdc_path_smoke"
```

Local artifacts to inspect first:
- `sdc_path_batch_smoke.json`
- `sdc_path_loss_smoke.json`
- `debug_manifest.json`
- `debug_examples/*/contact_sheet.png`

## Preparing a remote input index
Recommended shape:
- `sdc_path_control_index_train.jsonl`
- `sdc_path_control_index_val.jsonl`

The remote wrapper can also accept:
- a directory containing those two files
- a single `.jsonl` file, in which case train and val both point at that file
- a `_train.jsonl` file, in which case `_val.jsonl` is inferred if it exists

## Remote training prerequisites
- GPU machine with the full repo checkout.
- Python environment that can import `metadrive`, `scenarionet`, and `src/Adv-BMT`.
- A scenario root aligned with the `scenario_pkl` paths referenced by the PR10 index.
- A warm-start checkpoint for the original GT-trained policy.

Required environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export DATA_ROOT=/path/to/scenario_root
export INPUT_INDEX=/path/to/sdc_path_control_index_train.jsonl
export FORWARD_CKPT=/path/to/forward_only.ckpt
export OUTDIR="$REPO_ROOT/logs/pr10_sdc_path_control"
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=8
export BATCH_SIZE=8
export MAX_STEPS=5000
export VAL_INTERVAL=500
export CKPT_LOAD_MODE=forgiving_state_dict
```

Optional environment variables:
```bash
export VAL_INDEX=/path/to/sdc_path_control_index_val.jsonl
export TEACHER_CKPT="$FORWARD_CKPT"
export SEED=0
export WANDB_ENABLED=true
export WANDB_PROJECT=infgen
export WANDB_ENTITY=your_wandb_entity
export WANDB_GROUP=pr10_sdc_path_control
export WANDB_RUN_NAME=pr10_sdc_path_control_run1
```

Exact remote training command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr10_sdc_path_train.sh"
```

## Expected outputs
- `$OUTDIR/lightning_logs/.../last.ckpt`
- `$OUTDIR/lightning_logs/.../config.yaml`
- training curves including:
  - `cf/sdc_semantic_acc`
  - `cf/sdc_path_prox_loss`
  - `cf/sdc_path_heading_loss`
  - `cf/sdc_path_progress_loss`
  - `cf/sdc_policy_kl`
  - `cf/sdc_motion_gt_weight_mean`

## What to inspect first on remote
- Warm-start checkpoint load report.
- Batch-level `cf/sdc_control_available`.
- `cf/sdc_semantic_acc`.
- `cf/sdc_nearest_path_distance_mean`.
- `cf/sdc_policy_kl`.
- Non-SDC drift relative to the frozen policy on held-out smokes.

## Common failure modes
- Scenario root does not match `scenario_pkl` paths stored in the index.
- Train/val index resolution from `INPUT_INDEX` is wrong.
- OOM from large `BATCH_SIZE` or `NUM_WORKERS`.
- Very long alternative paths creating large waypoint tensors; reduce resample density or batch size if needed.
- Warm-start reporting missing SDC-path-only keys. This is expected for the new modules.

## Current assumptions
- The Waymax-staged smoke root uses approximate agent dimensions for compatibility with the existing preprocessing stack.
- PR10 metrics currently live in the smoke script and are not yet wired into the full scenario evaluator.
- Stop rows remain supported in the schema/config, but can be excluded by data filtering if they prove noisy in the first large run.
