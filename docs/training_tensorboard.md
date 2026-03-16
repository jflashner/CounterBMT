# TensorBoard Monitoring for Supervised Training

This project now logs supervised training metrics to TensorBoard for both:

- `counter_bmt_v2.cli.train_nnx_bmt`
- `counter_bmt_v2.cli.train_nnx_bmt_dag_latent`

TensorBoard logging is enabled by default and writes to:

- `outputs/<run>/tensorboard/`

## Quick Start

Run training as usual:

```bash
python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/tb_demo \
  --max-steps 40 \
  --eval-every 20
```

Then launch TensorBoard:

```bash
tensorboard --logdir outputs/tb_demo/tensorboard --port 6006
```

Open `http://localhost:6006` in your browser.

## Useful Flags

- `--no-tensorboard`: disable TensorBoard logging.
- `--tensorboard-subdir <name>`: change log subdirectory under `--output-dir`.
- `--tensorboard-flush-secs <int>`: change writer flush cadence.
- `--no-tensorboard-log-run-config`: skip writing run config / summary text panels.

These flags are available in both supervised CLIs.

## What Gets Logged

Scalars only (no histograms/images yet):

- Train phase:
  - `train/lr`
  - `train/total_loss`, `train/accuracy`, `train/entropy`, `train/perplexity`, etc.
  - throughput metrics already in training (`train/train/steps_per_sec`, `train/train/tokens_per_sec`, ...)
- Eval phase:
  - `eval/*` metrics (including `eval/forward_approx/*`)
- Final eval:
  - `final_eval/*`
- Events:
  - `events/checkpoint_saved = 1` on checkpoint writes

Optional text entries:

- `run/config`
- `run/summary`

## Resume Behavior

When resuming in the same `--output-dir`, TensorBoard appends to the same
`tensorboard/` directory. This preserves a single continuous scalar history.

If you resumed into a different `--output-dir`, rebuild a merged TensorBoard run
from the two `metrics.jsonl` files:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/training/merge_tensorboard_runs.py \
  --run-dir outputs/first_run \
  --run-dir outputs/resumed_run \
  --output-dir outputs/merged_tensorboard_run \
  --overwrite
```

The merge prefers later `--run-dir` entries when both runs contain the same
`(phase, step)` pair, which is the right default for "original run + resumed
run". Then launch TensorBoard on the merged output:

```bash
tensorboard --logdir outputs/merged_tensorboard_run/tensorboard --port 6006
```

## Comparing Runs

For run comparison, start TensorBoard on a parent directory:

```bash
tensorboard --logdir outputs --port 6006
```

Then select different runs in TensorBoard’s UI.
