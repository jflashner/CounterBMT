#!/usr/bin/env bash
set -euo pipefail

EVAL_SPLIT="${EVAL_SPLIT:-natural}"
SEED="${1:-0}"

REPO_ROOT="${REPO_ROOT:-/data/home/grads/jflashner/CounterBMT_run}"
VIEWS_ROOT="${VIEWS_ROOT:-${REPO_ROOT}/eval_runs/advbmt_paperfaithful_td3_views_20260419}"

DATA_DIR="${DATA_DIR:-${VIEWS_ROOT}/train_counterbmt_mixed}"
EXP_STEM="${EXP_STEM:-td3_table4_advbmt_paperfaithful_train476}"

case "${EVAL_SPLIT}" in
  natural)
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-${VIEWS_ROOT}/eval_waymo_only}"
    EVAL_SUFFIX="eval_natural"
    ;;
  adversarial|adv)
    EVAL_DATA_DIR="${EVAL_DATA_DIR:-${VIEWS_ROOT}/eval_counterbmt_adversarial}"
    EVAL_SUFFIX="eval_adversarial"
    ;;
  *)
    echo "Unsupported EVAL_SPLIT=${EVAL_SPLIT}. Use natural or adversarial." >&2
    exit 1
    ;;
esac

EXP_NAME="${EXP_NAME:-${EXP_STEM}_${EVAL_SUFFIX}}"
USE_WANDB="${USE_WANDB:-false}"

export REPO_ROOT
export DATA_DIR
export EVAL_DATA_DIR
export EXP_NAME
export USE_WANDB

exec bash "${REPO_ROOT}/scripts/remote/run_td3_table4_openloop.sh" "${SEED}"
