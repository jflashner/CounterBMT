#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
CKPT="${CKPT:-/path/to/path_control.ckpt}"
CONTROL_INDEX="${CONTROL_INDEX:-/path/to/path_index.jsonl}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr6_posttrain_probe}"
BATCH_SIZE="${BATCH_SIZE:-2}"

cd "$REPO_ROOT"
export PYTHONPATH

python scripts/counterfactual/inspect_batch_control.py \
  --outdir "$OUTDIR" \
  --control-code-dir "$CONTROL_INDEX" \
  --data-dir "$DATA_ROOT" \
  --mode training \
  --batch-size "$BATCH_SIZE" \
  --run-forward \
  --forward-control-mode strict_local \
  --ckpt "$CKPT"
