# PR10.1 Semantic-Only SDC Runbook

## Scope
- Use `sdc_semantic_control_v1` rows only.
- Runtime control is semantic token only.
- Candidate `sdc_paths` are privileged supervision only.
- Same-label paths form an acceptable semantic family.
- Divergence gating is family-vs-other-label, not exact-path imitation.
- Do not use raw path IDs as model inputs.

## Local smoke checklist
- Build a tiny semantic-only index slice first.
- Run the dataset batch smoke before any training.
- Run the no-grad loss smoke against the warm-start checkpoint.
- Inspect the family overlay, separability profile, and projection debug plots.
- Treat missing-key reports for the new semantic-only modules and frozen teacher as expected during warm-start.
- Treat very large projected family distances as a blocker to resolve before remote scaling.

Local smoke commands:
```bash
export REPO_ROOT=/Users/joshuaflashner/Projects/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"

.venv-mac/bin/python "$REPO_ROOT/scripts/counterfactual/build_sdc_semantic_control_index.py" \
  --semantics-index "$REPO_ROOT/outputs/waymax_sdc_postsplit_semantics_top50_diverse_alt_selection_w5_with_grid_offroute_trimmed/postsplit_semantics_index.jsonl" \
  --outdir "$REPO_ROOT/outputs/pr10_1_sdc_semantic_smoke" \
  --output-name sdc_semantic_control_index_smoke.jsonl \
  --max-examples 2 \
  --debug-max-rows 4 \
  --stage-vlm-artifacts

.venv-mac/bin/python "$REPO_ROOT/scripts/counterfactual/smoke_sdc_semantic_control.py" \
  --control-index "$REPO_ROOT/outputs/pr10_1_sdc_semantic_smoke/sdc_semantic_control_index_smoke.jsonl" \
  --data-dir "$REPO_ROOT/outputs/pr10_1_sdc_semantic_smoke/scenario_root" \
  --ckpt "$REPO_ROOT/outputs/pr6_eval_debug_bundle_20260403/logs/pr6_path_control_5000_wandb_20260401_run1/lightning_logs/infgen/pr6_path_control_5000_20260401_run1/checkpoints/last.ckpt" \
  --outdir "$REPO_ROOT/outputs/pr10_1_sdc_semantic_smoke" \
  --batch-size 2 \
  --debug-examples 2
```

Local artifacts to inspect first:
- `sdc_semantic_control_index_smoke.jsonl`
- `sdc_semantic_batch_smoke.json`
- `sdc_semantic_loss_smoke.json`
- `debug_manifest.json`
- `debug_examples/*/semantic_control_contact_sheet.png`

## Preparing a remote input index
Recommended shape:
- `sdc_semantic_control_index_train.jsonl`
- `sdc_semantic_control_index_val.jsonl`

The remote wrapper can also accept:
- a directory containing those two files
- a single `.jsonl` file, in which case train and val both point at that file
- a `_train.jsonl` file, in which case `_val.jsonl` is inferred if it exists

## Remote training prerequisites
- GPU machine with the full repo checkout.
- Python environment that can import `metadrive`, `scenarionet`, and `src/Adv-BMT`.
- A scenario root aligned with the `scenario_pkl` paths referenced by the semantic control index.
- A warm-start checkpoint for the original GT-trained policy.

Required environment variables:
```bash
export REPO_ROOT=/path/to/CounterBMT
export PYTHONPATH="$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT"
export DATA_ROOT=/path/to/scenario_root
export INPUT_INDEX=/path/to/sdc_semantic_control_index_train.jsonl
export FORWARD_CKPT=/path/to/forward_only.ckpt
export OUTDIR="$REPO_ROOT/logs/pr10_semantic_only_sdc"
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=8
export BATCH_SIZE=8
export MAX_STEPS=5000
export VAL_INTERVAL=500
export CKPT_LOAD_MODE=forgiving_state_dict
```

Optional environment variables:
```bash
export VAL_INDEX=/path/to/sdc_semantic_control_index_val.jsonl
export TEACHER_CKPT="$FORWARD_CKPT"
export SEED=0
export WANDB_ENABLED=true
export WANDB_PROJECT=infgen
export WANDB_ENTITY=your_wandb_entity
export WANDB_GROUP=pr10_semantic_only_sdc
export WANDB_RUN_NAME=pr10_semantic_only_sdc_run1
```

Exact remote training command:
```bash
bash "$REPO_ROOT/scripts/remote/run_pr10_semantic_only_sdc_train.sh"
```

## Expected outputs
- `$OUTDIR/lightning_logs/.../last.ckpt`
- `$OUTDIR/lightning_logs/.../config.yaml`
- training curves including:
  - `cf/sdc_semantic_acc`
  - `cf/sdc_family_guide_loss`
  - `cf/sdc_family_gate_mean`
  - `cf/sdc_family_distance_mean`
  - `cf/sdc_motion_gt_weight_mean`

## What to inspect first on remote
- Warm-start checkpoint load report.
- Batch-level `cf/sdc_control_available`.
- `cf/sdc_semantic_acc`.
- `cf/sdc_family_guide_loss`.
- `cf/sdc_family_gate_mean`.
- `cf/sdc_family_distance_mean`.
- Non-SDC drift under controlled vs no-control eval.

## Common failure modes
- Scenario root does not match `scenario_pkl` paths stored in the index.
- Train/val index resolution from `INPUT_INDEX` is wrong.
- OOM from large `BATCH_SIZE` or `NUM_WORKERS`.
- Family-path tensors becoming large when a semantic family contains long resampled paths.
- Warm-start reporting missing semantic-only keys. This is expected for the new modules.
- Very large `cf/sdc_family_distance_mean`, which likely indicates a frame mismatch between privileged path supervision and decoder state space.

## Current assumptions
- PR10.1 currently builds families from the highlighted, VLM-labeled paths in each postsplit example bundle.
- Off-route candidates can appear in the bundle upstream, but runtime control is still semantic token only.
- The frozen teacher is the original GT-trained checkpoint and acts as the realism prior in the family-teacher KL.
