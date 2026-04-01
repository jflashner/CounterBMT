#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
CKPT="${CKPT:-/path/to/path_control.ckpt}"
CONTROL_INDEX_DIR="${CONTROL_INDEX_DIR:-}"
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr6_path_eval}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CKPT_LOAD_MODE="${CKPT_LOAD_MODE:-forgiving_state_dict}"

cd "$REPO_ROOT"
export PYTHONPATH

if [[ -n "$CONTROL_INDEX_DIR" ]]; then
  CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-$CONTROL_INDEX_DIR/path_index_curated_val.jsonl}"
fi
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-/path/to/path_index_curated_val.jsonl}"

python scripts/counterfactual/eval_path_control_sweep.py \
  --ckpt "$CKPT" \
  --control-index "$CONTROL_INDEX_VAL" \
  --outdir "$OUTDIR" \
  --num-examples "$NUM_EXAMPLES" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --load-mode "$CKPT_LOAD_MODE"
