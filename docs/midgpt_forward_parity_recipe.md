# MidGPT Forward-Parity Recipe

This note is the operational recipe for reproducing the released Adv-BMT
forward-only MidGPT setup as closely as possible with the `counter_bmt_v2`
rewrite.

## Targets

- Legacy config target: [0202_midgpt.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/0202_midgpt.yaml)
- v2 model preset: `midgpt_parity`
- v2 runtime preset: `legacy_midgpt_recipe`
- v2 tokenizer mode: `adv_bmt_parity`
- distribution mode: single-host `pmap`
- precision: `bf16-mixed`

## Why The H200 Script Uses `BATCH_SIZE=40`

Legacy Adv-BMT Lightning DDP treated `batch_size=10` as a per-process batch
size. The v2 `pmap` path treats `--batch-size` as the global batch size.

So on a 4 GPU host:

- legacy-equivalent per-device batch = `10`
- v2 global batch = `4 * 10 = 40`

That is why the provided launcher defaults to `BATCH_SIZE=40`.

## Exact Legacy Count Check

Before a serious training run, verify that the released legacy MidGPT model and
the v2 parity model are effectively the same size:

- released legacy `0202_midgpt`: `5,286,849` params
- current v2 `midgpt_parity`: `5,294,025` params
- delta: `7,176` params, about `0.136%`

To reproduce this locally on macOS, either install the parity extras during
bootstrap:

```bash
INSTALL_PARITY_TOOLS=1 tools/bootstrap_mac.sh
```

or install the parity audit profile directly:

```bash
python -m pip install -r requirements-mac-parity-tools.txt
python tools/verify_environment.py --profile mac-parity
```

```bash
.venv-mac/bin/python tools/count_midgpt_models.py \
  --json-out outputs/parity_counts/midgpt_counts.json
```

This uses the actual legacy `MotionLM` constructor with lightweight import
shims for non-counting dependencies, so the reported count reflects the real
released model wiring.

## 4x H200 Training Launch

Set your data roots and launch the prepared script:

```bash
export TRAIN_DATA_DIR=/path/to/scenarionet/waymo/training
export VAL_DATA_DIR=/path/to/scenarionet/waymo/validation
export OUTPUT_DIR=/path/to/outputs/h200_midgpt_parity
export PYTHON_BIN=python

bash tools/run_midgpt_parity_h200.sh
```

The launcher calls:

```bash
python src/counter_bmt_v2/cli/train_nnx_bmt.py \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --val-data-dir "$VAL_DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --runtime-preset legacy_midgpt_recipe \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --distributed-backend pmap \
  --precision bf16-mixed \
  --batch-size 40 \
  --epochs 30 \
  --strict-91-steps \
  --forward-export-artifacts
```

## Resume

To resume from a saved checkpoint:

```bash
export RESUME_CHECKPOINT=/path/to/checkpoint.pkl
bash tools/run_midgpt_parity_h200.sh
```

## Strict Forward Comparison

The launcher automatically runs strict comparison on exported forward artifacts
after training if `forward_eval_artifacts` exists.

You can also run it manually:

```bash
python src/scripts/parity/compare_forward_metrics.py \
  --artifact-dir "$OUTPUT_DIR/forward_eval_artifacts" \
  --output-json "$OUTPUT_DIR/forward_eval_strict/latest.json"
```

## Recommended Run Discipline

- Keep `mode=forward` and `reverse_probability=0.0` for this phase.
- Train the base parity model first, before Stage A/B/C DAG-latent work.
- Use the exact released tokenizer/runtime pairing:
  - `midgpt_parity`
  - `adv_bmt_parity`
  - `legacy_midgpt_recipe`
- Keep `strict_91_steps` enabled so horizon mismatch does not silently degrade
  parity.
- Export forward artifacts on every run so approximate metrics and strict
  parity checks can be compared offline from the same checkpoints.
