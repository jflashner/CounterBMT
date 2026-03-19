#!/usr/bin/env bash
set -euo pipefail

# 4x H200 forward-only MidGPT parity launcher.
#
# This script intentionally targets the paper-close forward recipe:
# - model preset:      midgpt_parity
# - runtime preset:    legacy_midgpt_recipe
# - tokenizer mode:    adv_bmt_parity
# - distributed mode:  single-host JAX pmap across 4 local GPUs
# - precision:         bf16 mixed
#
# Important batching note:
# The legacy Lightning recipe used batch_size=10 per process. v2 `pmap` treats
# `--batch-size` as the *global* batch size, so the paper-close 4-GPU match is
# `BATCH_SIZE=40`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No python interpreter found (checked PYTHON_BIN, python, python3)." >&2
  exit 1
fi

if [[ -z "${TRAIN_DATA_DIR:-}" ]]; then
  echo "Set TRAIN_DATA_DIR to the ScenarioNet/WOMD training split path." >&2
  exit 1
fi

if [[ -z "${VAL_DATA_DIR:-}" ]]; then
  echo "Set VAL_DATA_DIR to the validation/eval split path." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-src}"

# Keep memory behavior explicit on large boxes. Preallocation remains enabled by
# default; callers can override these if they have a preferred cluster policy.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/h200_midgpt_parity}"
BATCH_SIZE="${BATCH_SIZE:-40}"
EPOCHS="${EPOCHS:-30}"
EVAL_EVERY="${EVAL_EVERY:-2000}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2000}"
LOG_EVERY="${LOG_EVERY:-20}"
PRESCAN_WORKERS="${PRESCAN_WORKERS:-8}"
FORWARD_ARTIFACT_MAX_SCENARIOS="${FORWARD_ARTIFACT_MAX_SCENARIOS:-64}"
FORWARD_VIZ_MAX_SCENARIOS="${FORWARD_VIZ_MAX_SCENARIOS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
NUM_TRAIN_SCENARIOS="${NUM_TRAIN_SCENARIOS:-}"
NUM_VAL_SCENARIOS="${NUM_VAL_SCENARIOS:-}"

CMD=(
  "$PYTHON_BIN"
  src/counter_bmt_v2/cli/train_nnx_bmt.py
  --train-data-dir "$TRAIN_DATA_DIR"
  --val-data-dir "$VAL_DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --runtime-preset legacy_midgpt_recipe
  --model-preset midgpt_parity
  --tokenizer-mode adv_bmt_parity
  --distributed-backend pmap
  --precision bf16-mixed
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --strict-91-steps
  --sample-interval-training 1
  --sample-interval-test 1
  --prescan-workers "$PRESCAN_WORKERS"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --checkpoint-every "$CHECKPOINT_EVERY"
  --log-every "$LOG_EVERY"
  --forward-eval-modes 6
  --forward-eval-sampling topp
  --forward-eval-temperature 1.0
  --forward-eval-topp 0.95
  --forward-eval-topk 5
  --forward-export-artifacts
  --forward-artifact-max-scenarios "$FORWARD_ARTIFACT_MAX_SCENARIOS"
  --forward-viz-max-scenarios "$FORWARD_VIZ_MAX_SCENARIOS"
  --forward-viz-max-agents 10
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "$NUM_TRAIN_SCENARIOS" ]]; then
  CMD+=(--num-train-scenarios "$NUM_TRAIN_SCENARIOS")
fi

if [[ -n "$NUM_VAL_SCENARIOS" ]]; then
  CMD+=(--num-val-scenarios "$NUM_VAL_SCENARIOS")
fi

echo "Launching MidGPT parity training:"
printf '  %q' "${CMD[@]}"
echo
"${CMD[@]}"

ARTIFACT_DIR="$OUTPUT_DIR/forward_eval_artifacts"
STRICT_JSON="$OUTPUT_DIR/forward_eval_strict/latest.json"

if [[ -d "$ARTIFACT_DIR" ]]; then
  echo
  echo "Running strict forward-artifact comparison..."
  "$PYTHON_BIN" src/scripts/parity/compare_forward_metrics.py \
    --artifact-dir "$ARTIFACT_DIR" \
    --output-json "$STRICT_JSON"
  echo "Strict forward comparison written to $STRICT_JSON"
else
  echo
  echo "No forward_eval_artifacts directory found at $ARTIFACT_DIR; skipping strict comparison."
fi
