#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
CONTROL_INDEX="${CONTROL_INDEX:-/path/to/path_index.jsonl}"
FORWARD_CKPT="${FORWARD_CKPT:-/path/to/forward_only.ckpt}"
OUTDIR="${OUTDIR:-$REPO_ROOT/logs/pr6_path_control}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-5000}"
VAL_INTERVAL="${VAL_INTERVAL:-500}"
SEED="${SEED:-0}"

cd "$REPO_ROOT"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES

python src/Adv-BMT/bmt/train_motion.py \
  --config-name motion_forward_path_control_strict_local.yaml \
  DATA.TRAINING_DATA_DIR="$DATA_ROOT" \
  DATA.TEST_DATA_DIR="$DATA_ROOT" \
  DATA.COUNTERFACTUAL_CONTROL_INDEX="$CONTROL_INDEX" \
  batch_size="$BATCH_SIZE" \
  val_batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  val_num_workers="$NUM_WORKERS" \
  pretrain="$FORWARD_CKPT" \
  log_dir="$OUTDIR" \
  seed="$SEED" \
  max_steps="$MAX_STEPS" \
  val_interval="$VAL_INTERVAL" \
  wandb=false
