#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
CKPT="${CKPT:-/path/to/path_control.ckpt}"
CONTROL_INDEX_DIR="${CONTROL_INDEX_DIR:-}"
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr6_posttrain_probe}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CKPT_LOAD_MODE="${CKPT_LOAD_MODE:-forgiving_state_dict}"

cd "$REPO_ROOT"
export PYTHONPATH

if [[ -n "$CONTROL_INDEX_DIR" ]]; then
  CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-$CONTROL_INDEX_DIR/path_index_curated_val.jsonl}"
fi
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-/path/to/path_index_curated_val.jsonl}"

python scripts/counterfactual/inspect_batch_control.py \
  --outdir "$OUTDIR" \
  --control-code-dir "$CONTROL_INDEX_VAL" \
  --data-dir "$DATA_ROOT" \
  --mode training \
  --batch-size "$BATCH_SIZE" \
  --run-forward \
  --forward-control-mode strict_local \
  --ckpt "$CKPT" \
  --load-mode "$CKPT_LOAD_MODE"
