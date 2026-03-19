# MidGPT Learning Probe

This is the shortest end-to-end check that both the released Adv-BMT MidGPT
stack and the v2 `midgpt_parity` stack are learning and behaving similarly.

The probe script is:

- [run_midgpt_learning_probe.py](/Users/joshuaflashner/Projects/CounterBMT/tools/run_midgpt_learning_probe.py)

## What It Does

1. Builds a deterministic shared train/val subset by symlinking the same
   ScenarioNet files into a probe workspace.
2. Trains the v2 parity model for a short forward-only budget.
3. Trains the legacy Adv-BMT `0202_midgpt` model for a short forward-only budget.
4. Runs the existing head-to-head evaluator on the resulting checkpoints using
   the shared validation subset.
5. Writes a compact `probe_summary.json` with:
   - training commands
   - checkpoint paths
   - short training summaries
   - head-to-head report excerpts

## Why This Is Better Than Comparing Loss Alone

Training loss is only half the story. Two models can have roughly similar loss
curves but still diverge behaviorally at rollout time.

This probe therefore checks both:

- learning trend:
  - train loss / accuracy summaries from each trainer
- behavior:
  - head-to-head forward metrics on the shared validation subset

## Recommended First Probe

Use a small but non-trivial subset:

```bash
python tools/run_midgpt_learning_probe.py \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/midgpt_learning_probe \
  --train-scenarios 512 \
  --val-scenarios 64 \
  --batch-size 8 \
  --val-batch-size 4 \
  --v2-max-steps 200 \
  --v2-eval-batches 0 \
  --legacy-epochs 4 \
  --legacy-limit-train-batches 50 \
  --legacy-limit-val-batches 0
```

If you already have separate v2 and legacy environments and want to point the
probe at them explicitly:

```bash
.venv-v2/bin/python tools/run_midgpt_learning_probe.py \
  --train-data-dir /path/to/scenarionet/waymo/training \
  --val-data-dir /path/to/scenarionet/waymo/validation \
  --output-dir outputs/midgpt_learning_probe \
  --v2-python-bin .venv-v2/bin/python \
  --legacy-python-bin .venv-legacy-adv-bmt/bin/python \
  --head2head-python-bin .venv-v2/bin/python
```

## One-Command H200 Wrapper

The repo also includes a single wrapper that:

1. bootstraps the v2 env
2. bootstraps the separate legacy env
3. runs the paired learning probe

That launcher is:

- [run_midgpt_learning_probe_h200.sh](/Users/joshuaflashner/Projects/CounterBMT/tools/run_midgpt_learning_probe_h200.sh)

Example:

```bash
export TRAIN_DATA_DIR=/path/to/scenarionet/waymo/training
export VAL_DATA_DIR=/path/to/scenarionet/waymo/validation
export OUTPUT_DIR=outputs/h200_midgpt_learning_probe

bash tools/run_midgpt_learning_probe_h200.sh
```

`PYTHON_BIN` is optional. If unset, the wrapper auto-detects the first
available interpreter from `python3.10`, `python3`, then `python`.

The legacy bootstrap is handled separately inside the wrapper. By default it
tries to create the legacy env on Python `3.10`, then `3.11`, and if the host
only has Python `3.12` it will use `uv` to provision a managed `3.10`
interpreter automatically.

By default the H200 wrapper sets `CUDA_VISIBLE_DEVICES=0` so the probe uses one
GPU for a cleaner apples-to-apples learning comparison. Override
`CUDA_VISIBLE_DEVICES` if you want to exercise a wider device set.

Useful override knobs for the legacy env are:

- `LEGACY_PYTHON_SPEC=3.10|3.11`
- `LEGACY_BOOTSTRAP_PYTHON_BIN=/path/to/python3.10`
- `LEGACY_AUTO_INSTALL_UV=0|1`

Validation semantics in the probe are:

- `--v2-eval-batches 0`
  means the v2 trainer evaluates on the full shared validation subset
- `--legacy-limit-val-batches 0`
  together with `EVAL_MOTION=False` means the legacy trainer skips its internal
  validation loop during the short probe

We still compare the two models on the shared validation subset after training
using the head-to-head evaluator.

## Environment Note

This probe needs the legacy training runtime, but not the full Waymo/TensorFlow
evaluation stack. The simplest setup path is:

```bash
tools/bootstrap_legacy_adv_bmt.sh
```

That installs the separate legacy environment for training and disables the
legacy internal evaluator in the probe itself. If you later need the original
Waymo evaluator too, rerun the bootstrap with:

```bash
INSTALL_WAYMO_EVAL=1 tools/bootstrap_legacy_adv_bmt.sh
```

## Output Files

Under the chosen `--output-dir`, the key files are:

- `subset_manifest.json`
- `probe_summary.json`
- `v2_probe/checkpoints/last.pkl`
- `legacy_probe/.../last.ckpt`
- `head2head/report.json`
