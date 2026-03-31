# Counterfactual Path Remote Runbook

## Local smoke checklist
- Verify path-index build on a tiny slice only.
- Verify `path_only` dataset batching on 1-2 batches only.
- Verify eval CLI wiring on 0-2 examples only.
- Do not run large indexing, mining, sweeps, or training locally.

Local smoke commands:
```bash
export REPO_ROOT=/Users/joshuaflashner/Projects/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"

python "$REPO_ROOT/scripts/counterfactual/build_path_control_index.py" \
  --scenario-root "$REPO_ROOT/data/scenarionet_waymo_training_500" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v4" \
  --max-scenarios 10 \
  --seed 0 \
  --num-workers 1 \
  --max-candidates-per-scenario 1 \
  --max-agents-per-candidate 2 \
  --write-examples-manifest \
  --write-histograms \
  --write-summary

python "$REPO_ROOT/scripts/counterfactual/build_path_control_index.py" \
  --scenario-root "$REPO_ROOT/data/scenarionet_waymo_training_500" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v5" \
  --max-scenarios 25 \
  --seed 0 \
  --num-workers 1 \
  --max-candidates-per-scenario 1 \
  --max-agents-per-candidate 2 \
  --write-examples-manifest \
  --write-histograms \
  --write-summary

cp "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v5/path_support_summary.json" \
  "$REPO_ROOT/outputs/counterfactual_path_smoke/path_support_summary_smoke.json"
cp "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v5/path_label_histograms.json" \
  "$REPO_ROOT/outputs/counterfactual_path_smoke/path_label_histograms_smoke.json"
cp "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v5/path_examples_manifest.json" \
  "$REPO_ROOT/outputs/counterfactual_path_smoke/path_examples_manifest_smoke.json"
cp "$REPO_ROOT/outputs/counterfactual_path_smoke/tmp_index_v5/path_index.jsonl" \
  "$REPO_ROOT/outputs/counterfactual_path_smoke/path_index_smoke.jsonl"

# If the strict local path-index slice is empty, use the existing 4245 bundle
# only for path_only dataset wiring checks.
python "$REPO_ROOT/scripts/counterfactual/inspect_path_only_batch.py" \
  --out "$REPO_ROOT/outputs/counterfactual_path_smoke/path_only_batch_smoke.json" \
  --control-source "$REPO_ROOT/outputs/counterfactual_pr5_conditioning_4245/control_index.jsonl" \
  --data-dir "$REPO_ROOT/data/scenarionet_waymo_training_500" \
  --mode training \
  --batch-size 2

python "$REPO_ROOT/scripts/counterfactual/eval_path_control_sweep.py" \
  --control-index "$REPO_ROOT/outputs/counterfactual_path_smoke/path_index_smoke.jsonl" \
  --outdir "$REPO_ROOT/outputs/counterfactual_path_smoke/eval_tmp" \
  --num-examples 2 \
  --seed 0
```

Local artifact mapping:
- Path index build corresponds to remote path index build.
- `path_only_batch_smoke.json` corresponds to remote training input wiring.
  On a zero-support local slice, this is expected to come from the existing `outputs/counterfactual_pr5_conditioning_4245/control_index.jsonl` bundle rather than `path_index_smoke.jsonl`.
- `eval_smoke_summary.json` corresponds to remote eval sweep CLI wiring.

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
- `$OUTDIR/path_index.jsonl`
- `$OUTDIR/path_support_summary.json`
- `$OUTDIR/path_label_histograms.json`
- `$OUTDIR/path_examples_manifest.json`

How to resume/restart:
- Reuse the same `OUTDIR` only if you are comfortable overwriting prior summaries.
- Prefer a fresh `OUTDIR` per run size.

Common failure modes:
- `Could not be matched to scenario files`: stale control index or mismatched dataset root.
- Very slow runs: reduce `NUM_WORKERS` if memory pressure is high.
- Weak class support: inspect `path_support_summary.json` and rerun with larger `MAX_SCENARIOS`.

What artifacts to inspect first:
- `path_support_summary.json`
- `path_label_histograms.json`
- a few rows in `path_examples_manifest.json`

## Remote training run
Prerequisites:
- A forward-only checkpoint to finetune from.
- A path index JSONL from the previous step.
- GPU machine with enough memory for the chosen batch size.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export DATA_ROOT=/path/to/scenario_root
export CONTROL_INDEX=/path/to/path_index.jsonl
export FORWARD_CKPT=/path/to/forward_only.ckpt
export OUTDIR="$REPO_ROOT/logs/pr6_path_control"
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=8
export BATCH_SIZE=8
export MAX_STEPS=5000
export VAL_INTERVAL=500
export SEED=0
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
  DATA.COUNTERFACTUAL_CONTROL_INDEX="$REPO_ROOT/outputs/counterfactual_path_smoke/path_index_smoke.jsonl" \
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
- Reuse the same `OUTDIR` and point `ckpt` at `last.ckpt` if you want to resume.
- Keep the original `CONTROL_INDEX` fixed for a given run so class balance stays comparable.

Common failure modes:
- Missing checkpoint path.
- Mismatched `CONTROL_INDEX` and `DATA_ROOT`.
- OOM with large `BATCH_SIZE`; reduce batch size or workers.

What artifacts to inspect first:
- TensorBoard or WandB logs
- saved `config.yaml`
- path-head and anchor-loss curves

## Remote eval sweep
Prerequisites:
- A trained path-control checkpoint.
- The same path index used for training or a held-out one.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export CKPT=/path/to/path_control.ckpt
export CONTROL_INDEX=/path/to/path_index.jsonl
export OUTDIR="$REPO_ROOT/outputs/pr6_path_eval"
export NUM_EXAMPLES=200
export SEED=0
export BATCH_SIZE=1
export NUM_WORKERS=0
```

Exact copy-paste command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr6_eval_path_control.sh"
```

Expected output files:
- `$OUTDIR/eval_smoke_summary.json`
- `$OUTDIR/eval_smoke_examples.jsonl`

How to resume/restart:
- Use a fresh `OUTDIR` per checkpoint or eval split.
- Rerun with the same `SEED` for reproducible sampling.

Common failure modes:
- Missing checkpoint.
- Bad `scenario_pkl` paths inside the control index.
- Path index built from a different code revision than the checkpoint.

What artifacts to inspect first:
- `eval_smoke_summary.json`
- a handful of rows from `eval_smoke_examples.jsonl`

## Remote post-train runtime probe
Prerequisites:
- A trained strict-local checkpoint.
- The path index JSONL.

Environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export CKPT=/path/to/path_control.ckpt
export CONTROL_INDEX=/path/to/path_index.jsonl
export DATA_ROOT=/path/to/scenario_root
export OUTDIR="$REPO_ROOT/outputs/pr6_posttrain_probe"
export BATCH_SIZE=2
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
- Control index path passed as a directory instead of the JSONL file.
- `DATA_ROOT` not matching the scenarios referenced by the control index.
- Checkpoint config mismatch.

What artifacts to inspect first:
- `forward_control_runtime.json`
- `batch_control_summary.json`

## Artifact checklist to send back
- Path index build:
  `path_support_summary.json`
  `path_label_histograms.json`
  `path_examples_manifest.json`
  `path_index.jsonl`
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
