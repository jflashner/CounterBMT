#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
CONTROL_INDEX_DIR="${CONTROL_INDEX_DIR:-}"
CONTROL_INDEX_TRAIN="${CONTROL_INDEX_TRAIN:-}"
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-}"
FORWARD_CKPT="${FORWARD_CKPT:-/path/to/forward_only.ckpt}"
OUTDIR="${OUTDIR:-$REPO_ROOT/logs/pr6_path_control}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
SEED="${SEED:-0}"
CKPT_LOAD_MODE="${CKPT_LOAD_MODE:-forgiving_state_dict}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-infgen}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pr6_path_control}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

cd "$REPO_ROOT"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_GROUP
if [[ -n "$WANDB_RUN_NAME" ]]; then
  export WANDB_RUN_NAME
fi

if [[ -n "$CONTROL_INDEX_DIR" ]]; then
  CONTROL_INDEX_TRAIN="${CONTROL_INDEX_TRAIN:-$CONTROL_INDEX_DIR/path_index_curated_train.jsonl}"
  CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-$CONTROL_INDEX_DIR/path_index_curated_val.jsonl}"
fi
CONTROL_INDEX_TRAIN="${CONTROL_INDEX_TRAIN:-/path/to/path_index_curated_train.jsonl}"
CONTROL_INDEX_VAL="${CONTROL_INDEX_VAL:-/path/to/path_index_curated_val.jsonl}"

python src/Adv-BMT/bmt/train_motion.py \
  --config-name motion_forward_path_control_strict_local.yaml \
  DATA.TRAINING_DATA_DIR="$DATA_ROOT" \
  DATA.TEST_DATA_DIR="$DATA_ROOT" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN="$CONTROL_INDEX_TRAIN" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$CONTROL_INDEX_VAL" \
  batch_size="$BATCH_SIZE" \
  val_batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  val_num_workers="$NUM_WORKERS" \
  pretrain="$FORWARD_CKPT" \
  CKPT_LOAD_MODE="$CKPT_LOAD_MODE" \
  log_dir="$OUTDIR" \
  seed="$SEED" \
  +max_steps="$MAX_STEPS" \
  +val_interval="$VAL_INTERVAL" \
  wandb="$WANDB_ENABLED"
