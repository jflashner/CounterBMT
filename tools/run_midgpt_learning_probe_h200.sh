#!/usr/bin/env bash
set -euo pipefail

# One-command H200 launcher for the paired MidGPT learning probe.
#
# This wrapper is intentionally conservative:
# - v2 and legacy use separate environments
# - both environments can be bootstrapped automatically
# - the probe defaults to one visible GPU for cleaner apples-to-apples learning
#   comparison, but callers can override CUDA_VISIBLE_DEVICES if they want
# - the head-to-head report is generated automatically at the end

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "$@"
  fi
}

if [[ -z "${TRAIN_DATA_DIR:-}" ]]; then
  echo "Set TRAIN_DATA_DIR to the ScenarioNet/WOMD training split path." >&2
  exit 1
fi

if [[ -z "${VAL_DATA_DIR:-}" ]]; then
  echo "Set VAL_DATA_DIR to the ScenarioNet/WOMD validation split path." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/h200_midgpt_learning_probe}"
V2_VENV_DIR="${V2_VENV_DIR:-.venv-v2}"
LEGACY_VENV_DIR="${LEGACY_VENV_DIR:-.venv-legacy-adv-bmt}"

BOOTSTRAP_V2="${BOOTSTRAP_V2:-1}"
BOOTSTRAP_LEGACY="${BOOTSTRAP_LEGACY:-1}"
RECREATE_V2_VENV="${RECREATE_V2_VENV:-0}"
RECREATE_LEGACY_VENV="${RECREATE_LEGACY_VENV:-0}"

LEGACY_PROFILE="${LEGACY_PROFILE:-linux-cu121}"
LEGACY_INSTALL_SIM_STACK="${LEGACY_INSTALL_SIM_STACK:-1}"
LEGACY_INSTALL_WAYMO_EVAL="${LEGACY_INSTALL_WAYMO_EVAL:-0}"
LEGACY_METADRIVE_SRC="${LEGACY_METADRIVE_SRC:-}"
LEGACY_SCENARIONET_SRC="${LEGACY_SCENARIONET_SRC:-}"
LEGACY_METADRIVE_REF="${LEGACY_METADRIVE_REF:-}"
LEGACY_SCENARIONET_REF="${LEGACY_SCENARIONET_REF:-}"

SEED="${SEED:-0}"
TRAIN_SCENARIOS="${TRAIN_SCENARIOS:-512}"
VAL_SCENARIOS="${VAL_SCENARIOS:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
V2_MAX_STEPS="${V2_MAX_STEPS:-200}"
V2_EVAL_BATCHES="${V2_EVAL_BATCHES:-0}"
LEGACY_EPOCHS="${LEGACY_EPOCHS:-4}"
LEGACY_LIMIT_TRAIN_BATCHES="${LEGACY_LIMIT_TRAIN_BATCHES:-50}"
LEGACY_LIMIT_VAL_BATCHES="${LEGACY_LIMIT_VAL_BATCHES:-0}"
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-none}"
PRECISION="${PRECISION:-fp32}"

# Default to a single visible GPU so the probe measures learning similarity
# rather than differences in the two frameworks' multi-GPU behavior. Override if
# you want to exercise a wider device set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ "$BOOTSTRAP_V2" == "1" ]]; then
  run_cmd env \
    PYTHON_BIN="$PYTHON_BIN" \
    VENV_DIR="$V2_VENV_DIR" \
    RECREATE_VENV="$RECREATE_V2_VENV" \
    bash tools/bootstrap_linux.sh v2
fi

if [[ "$BOOTSTRAP_LEGACY" == "1" ]]; then
  LEGACY_ARGS=(
    env
    PYTHON_BIN="$PYTHON_BIN"
    VENV_DIR="$LEGACY_VENV_DIR"
    RECREATE_VENV="$RECREATE_LEGACY_VENV"
    LEGACY_PROFILE="$LEGACY_PROFILE"
    INSTALL_SIM_STACK="$LEGACY_INSTALL_SIM_STACK"
    INSTALL_WAYMO_EVAL="$LEGACY_INSTALL_WAYMO_EVAL"
  )
  if [[ -n "$LEGACY_METADRIVE_SRC" ]]; then
    LEGACY_ARGS+=(METADRIVE_SRC="$LEGACY_METADRIVE_SRC")
  fi
  if [[ -n "$LEGACY_SCENARIONET_SRC" ]]; then
    LEGACY_ARGS+=(SCENARIONET_SRC="$LEGACY_SCENARIONET_SRC")
  fi
  if [[ -n "$LEGACY_METADRIVE_REF" ]]; then
    LEGACY_ARGS+=(METADRIVE_REF="$LEGACY_METADRIVE_REF")
  fi
  if [[ -n "$LEGACY_SCENARIONET_REF" ]]; then
    LEGACY_ARGS+=(SCENARIONET_REF="$LEGACY_SCENARIONET_REF")
  fi
  LEGACY_ARGS+=(bash tools/bootstrap_legacy_adv_bmt.sh)
  run_cmd "${LEGACY_ARGS[@]}"
fi

V2_PYTHON_BIN="$ROOT_DIR/$V2_VENV_DIR/bin/python"
LEGACY_PYTHON_BIN="$ROOT_DIR/$LEGACY_VENV_DIR/bin/python"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "+ $V2_PYTHON_BIN tools/run_midgpt_learning_probe.py --train-data-dir $TRAIN_DATA_DIR --val-data-dir $VAL_DATA_DIR --output-dir $OUTPUT_DIR --v2-python-bin $V2_PYTHON_BIN --legacy-python-bin $LEGACY_PYTHON_BIN --head2head-python-bin $V2_PYTHON_BIN --legacy-root src/Adv-BMT --seed $SEED --train-scenarios $TRAIN_SCENARIOS --val-scenarios $VAL_SCENARIOS --batch-size $BATCH_SIZE --val-batch-size $VAL_BATCH_SIZE --num-workers $NUM_WORKERS --v2-max-steps $V2_MAX_STEPS --v2-eval-batches $V2_EVAL_BATCHES --legacy-epochs $LEGACY_EPOCHS --legacy-limit-train-batches $LEGACY_LIMIT_TRAIN_BATCHES --legacy-limit-val-batches $LEGACY_LIMIT_VAL_BATCHES --distributed-backend $DISTRIBUTED_BACKEND --precision $PRECISION"
  echo
  echo "Dry run complete."
  exit 0
fi

if [[ ! -x "$V2_PYTHON_BIN" ]]; then
  echo "v2 python not found: $V2_PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -x "$LEGACY_PYTHON_BIN" ]]; then
  echo "legacy python not found: $LEGACY_PYTHON_BIN" >&2
  exit 1
fi

run_cmd "$V2_PYTHON_BIN" tools/run_midgpt_learning_probe.py \
  --train-data-dir "$TRAIN_DATA_DIR" \
  --val-data-dir "$VAL_DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --v2-python-bin "$V2_PYTHON_BIN" \
  --legacy-python-bin "$LEGACY_PYTHON_BIN" \
  --head2head-python-bin "$V2_PYTHON_BIN" \
  --legacy-root src/Adv-BMT \
  --seed "$SEED" \
  --train-scenarios "$TRAIN_SCENARIOS" \
  --val-scenarios "$VAL_SCENARIOS" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --v2-max-steps "$V2_MAX_STEPS" \
  --v2-eval-batches "$V2_EVAL_BATCHES" \
  --legacy-epochs "$LEGACY_EPOCHS" \
  --legacy-limit-train-batches "$LEGACY_LIMIT_TRAIN_BATCHES" \
  --legacy-limit-val-batches "$LEGACY_LIMIT_VAL_BATCHES" \
  --distributed-backend "$DISTRIBUTED_BACKEND" \
  --precision "$PRECISION"

echo
echo "Learning probe complete."
echo "Summary: $OUTPUT_DIR/probe_summary.json"
echo "Head-to-head report: $OUTPUT_DIR/head2head/report.json"
