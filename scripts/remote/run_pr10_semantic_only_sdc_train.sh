#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
INPUT_INDEX="${INPUT_INDEX:-/path/to/sdc_semantic_control_index.jsonl}"
FORWARD_CKPT="${FORWARD_CKPT:-/path/to/forward_only.ckpt}"
OUTDIR="${OUTDIR:-$REPO_ROOT/logs/pr10_semantic_only_sdc}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-5}"
MAX_STEPS="${MAX_STEPS:-5000}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
CKPT_LOAD_MODE="${CKPT_LOAD_MODE:-forgiving_state_dict}"
SEED="${SEED:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-infgen}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pr10_semantic_only_sdc}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
VAL_INDEX="${VAL_INDEX:-}"
TEACHER_CKPT="${TEACHER_CKPT:-$FORWARD_CKPT}"
GUIDE_LOSS_WEIGHT="${GUIDE_LOSS_WEIGHT:-0.2}"
SEMANTIC_AUX_LOSS_WEIGHT="${SEMANTIC_AUX_LOSS_WEIGHT:-0.0}"
FAMILY_PATH_PROX_WEIGHT="${FAMILY_PATH_PROX_WEIGHT:-1.0}"
FAMILY_HEADING_WEIGHT="${FAMILY_HEADING_WEIGHT:-0.75}"
FAMILY_BACKWARD_WEIGHT="${FAMILY_BACKWARD_WEIGHT:-0.5}"
FAMILY_GUIDE_TEMPERATURE="${FAMILY_GUIDE_TEMPERATURE:-1.0}"
FAMILY_GUIDE_BANDWIDTH_M="${FAMILY_GUIDE_BANDWIDTH_M:-6.0}"

resolve_indexes() {
  local input_index="$1"
  local train_index=""
  local val_index=""

  if [[ -d "$input_index" ]]; then
    if [[ -f "$input_index/sdc_semantic_control_index_train.jsonl" ]]; then
      train_index="$input_index/sdc_semantic_control_index_train.jsonl"
      val_index="$input_index/sdc_semantic_control_index_val.jsonl"
    elif [[ -f "$input_index/sdc_semantic_control_index.jsonl" ]]; then
      train_index="$input_index/sdc_semantic_control_index.jsonl"
      val_index="$input_index/sdc_semantic_control_index.jsonl"
    fi
  elif [[ -f "$input_index" ]]; then
    train_index="$input_index"
    case "$input_index" in
      *_train.jsonl)
        val_index="${input_index/_train.jsonl/_val.jsonl}"
        ;;
      *)
        val_index="$input_index"
        ;;
    esac
  fi

  if [[ -n "$VAL_INDEX" ]]; then
    val_index="$VAL_INDEX"
  fi

  if [[ -z "$train_index" ]]; then
    echo "Could not resolve training index from INPUT_INDEX=$input_index" >&2
    exit 1
  fi
  if [[ -z "$val_index" ]]; then
    echo "Could not resolve validation index from INPUT_INDEX=$input_index" >&2
    exit 1
  fi

  echo "$train_index|$val_index"
}

IFS="|" read -r CONTROL_INDEX_TRAIN CONTROL_INDEX_VAL < <(resolve_indexes "$INPUT_INDEX")

cd "$REPO_ROOT"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_GROUP
if [[ -n "$WANDB_RUN_NAME" ]]; then
  export WANDB_RUN_NAME
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv-legacy-adv-bmt/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv-legacy-adv-bmt/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

"$PYTHON_BIN" src/Adv-BMT/bmt/train_motion.py \
  --config-name motion_forward_sdc_semantic_only_strict_local.yaml \
  DATA.TRAINING_DATA_DIR="$DATA_ROOT" \
  DATA.TEST_DATA_DIR="$DATA_ROOT" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN="$CONTROL_INDEX_TRAIN" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$CONTROL_INDEX_VAL" \
  DATA.COUNTERFACTUAL_MODE="sdc_semantic_only" \
  batch_size="$BATCH_SIZE" \
  val_batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  val_num_workers="$NUM_WORKERS" \
  epochs="$EPOCHS" \
  pretrain="$FORWARD_CKPT" \
  CKPT_LOAD_MODE="$CKPT_LOAD_MODE" \
  MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT="$TEACHER_CKPT" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_GUIDE_LOSS_WEIGHT="$GUIDE_LOSS_WEIGHT" \
  MODEL.LOCAL_CONTROL_SDC_SEMANTIC_AUX_LOSS_WEIGHT="$SEMANTIC_AUX_LOSS_WEIGHT" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_PATH_PROX_WEIGHT="$FAMILY_PATH_PROX_WEIGHT" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_HEADING_WEIGHT="$FAMILY_HEADING_WEIGHT" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_BACKWARD_WEIGHT="$FAMILY_BACKWARD_WEIGHT" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_GUIDE_TEMPERATURE="$FAMILY_GUIDE_TEMPERATURE" \
  MODEL.LOCAL_CONTROL_SDC_FAMILY_GUIDE_BANDWIDTH_M="$FAMILY_GUIDE_BANDWIDTH_M" \
  log_dir="$OUTDIR" \
  seed="$SEED" \
  +max_steps="$MAX_STEPS" \
  +val_interval="$VAL_INTERVAL" \
  wandb="$WANDB_ENABLED"
