#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/grads/jflashner/CounterBMT_run}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
PYTHON_BIN="${PYTHON_BIN:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
DEVICES="${DEVICES:-4}"

DATA_ROOT="${DATA_ROOT:-/data/home/grads/jflashner/CounterBMT/outputs/pr10_1_sdc_semantic_top859_full/scenario_root}"
TRAIN_INDEX="${TRAIN_INDEX:-/data/home/grads/jflashner/CounterBMT_run/outputs/pr10_1_sdc_semantic_top500_from859/sdc_semantic_control_index_train.jsonl}"
VAL_INDEX="${VAL_INDEX:-/data/home/grads/jflashner/CounterBMT_run/outputs/pr10_1_sdc_semantic_top500_from859/sdc_semantic_control_index_val.jsonl}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-/data/home/grads/jflashner/CounterBMT_run/logs/pr10_1_top500_actualwall_progresssoft_4gpu_h200_run3/lightning_logs/infgen/pr10_1_top500_actualwall_progresssoft_4gpu_h200_2026-04-10/checkpoints/last.ckpt}"
TEACHER_CKPT="${TEACHER_CKPT:-$PRETRAIN_CKPT}"

RUN_GROUP_ROOT="${RUN_GROUP_ROOT:-/data/home/grads/jflashner/CounterBMT_run/logs/progresssoft_topomcpo_gtmix_trafficcap_runs}"
EXP_NAME="${EXP_NAME:-pr10_1_top500_actualwall_progresssoft_topomcpo_gtmix_trafficcap_4gpu_h200_run1}"
OUTDIR="${OUTDIR:-$RUN_GROUP_ROOT/$EXP_NAME}"

MAX_STEPS="${MAX_STEPS:-12000}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
SEED="${SEED:-0}"
CKPT_LOAD_MODE="${CKPT_LOAD_MODE:-forgiving_state_dict}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

ROLLOUT_TRAIN_DEBUG_ENABLED="${ROLLOUT_TRAIN_DEBUG_ENABLED:-true}"
ROLLOUT_TRAIN_DEBUG_EVERY_N_STEPS="${ROLLOUT_TRAIN_DEBUG_EVERY_N_STEPS:-25}"
ROLLOUT_TRAIN_DEBUG_MAX_MATCHES="${ROLLOUT_TRAIN_DEBUG_MAX_MATCHES:-4}"

BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
VAL_BATCH_SIZE_OVERRIDE="${VAL_BATCH_SIZE_OVERRIDE:-}"
NUM_SANITY_VAL_STEPS_OVERRIDE="${NUM_SANITY_VAL_STEPS_OVERRIDE:-}"
ROLLOUT_GROUP_SIZE_OVERRIDE="${ROLLOUT_GROUP_SIZE_OVERRIDE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUTDIR"

cd "$REPO_ROOT"
export PYTHONPATH
export CUDA_VISIBLE_DEVICES
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    "$REPO_ROOT/.venv-legacy-adv-bmt/bin/python" \
    "/data/home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/python" \
    "$REPO_ROOT/.venv/bin/python" \
    "/data/home/grads/jflashner/CounterBMT/.venv/bin/python"
  do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

cmd=(
  "$PYTHON_BIN"
  src/Adv-BMT/bmt/train_motion.py
  --config-name
  motion_forward_sdc_semantic_only_progresssoft_topomcpo_dag_trafficcap.yaml
  exp_name="$EXP_NAME"
  pretrain="$PRETRAIN_CKPT"
  CKPT_LOAD_MODE="$CKPT_LOAD_MODE"
  MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT="$TEACHER_CKPT"
  DATA.TRAINING_DATA_DIR="$DATA_ROOT"
  DATA.TEST_DATA_DIR="$DATA_ROOT"
  DATA.COUNTERFACTUAL_CONTROL_INDEX_TRAIN="$TRAIN_INDEX"
  DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$VAL_INDEX"
  DATA.COUNTERFACTUAL_MODE="sdc_semantic_only"
  DATA.COUNTERFACTUAL_ALT_ONLY_TRAIN=false
  +devices="$DEVICES"
  +max_steps="$MAX_STEPS"
  +val_interval="$VAL_INTERVAL"
  accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES"
  ROLLOUT_TRAIN_DEBUG_ENABLED="$ROLLOUT_TRAIN_DEBUG_ENABLED"
  ROLLOUT_TRAIN_DEBUG_EVERY_N_STEPS="$ROLLOUT_TRAIN_DEBUG_EVERY_N_STEPS"
  ROLLOUT_TRAIN_DEBUG_MAX_MATCHES="$ROLLOUT_TRAIN_DEBUG_MAX_MATCHES"
  log_dir="$OUTDIR"
  seed="$SEED"
  wandb="$WANDB_ENABLED"
)

if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
  cmd+=(batch_size="$BATCH_SIZE_OVERRIDE")
fi
if [[ -n "$VAL_BATCH_SIZE_OVERRIDE" ]]; then
  cmd+=(val_batch_size="$VAL_BATCH_SIZE_OVERRIDE")
fi
if [[ -n "$NUM_SANITY_VAL_STEPS_OVERRIDE" ]]; then
  cmd+=(num_sanity_val_steps="$NUM_SANITY_VAL_STEPS_OVERRIDE")
fi
if [[ -n "$ROLLOUT_GROUP_SIZE_OVERRIDE" ]]; then
  cmd+=(MODEL.LOCAL_CONTROL_SDC_ROLLOUT_TUBE_GROUP_SIZE="$ROLLOUT_GROUP_SIZE_OVERRIDE")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_args <<< "$EXTRA_ARGS"
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"
