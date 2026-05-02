#!/usr/bin/env bash
set -euo pipefail

RISK_VIEW="${RISK_VIEW:-medium_high}"
EVAL_SPLIT="${EVAL_SPLIT:-natural}"
SEED="${1:-0}"

REPO_ROOT="${REPO_ROOT:-/data/home/grads/jflashner/CounterBMT_run}"
RISK_VIEWS_ROOT="${RISK_VIEWS_ROOT:-${REPO_ROOT}/eval_runs/victim_centric_table4_td3_views_risk_scene_max_20260423}"

DATA_DIR="${RISK_VIEWS_ROOT}/train_counterbmt_mixed_risk_${RISK_VIEW}"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Risk-filtered TD3 train view does not exist: ${DATA_DIR}" >&2
  echo "Build it with scripts/agent_eval/prepare_td3_risk_ablation_views.py first." >&2
  exit 1
fi

case "${EVAL_SPLIT}" in
  natural)
    EVAL_DATA_DIR="${RISK_VIEWS_ROOT}/eval_waymo_only"
    EVAL_SUFFIX="eval_natural"
    ;;
  adversarial|adv)
    EVAL_DATA_DIR="${RISK_VIEWS_ROOT}/eval_counterbmt_adversarial"
    EVAL_SUFFIX="eval_adversarial"
    ;;
  *)
    echo "Unsupported EVAL_SPLIT=${EVAL_SPLIT}. Use natural or adversarial." >&2
    exit 1
    ;;
esac

if [[ ! -d "${EVAL_DATA_DIR}" ]]; then
  echo "TD3 eval view does not exist: ${EVAL_DATA_DIR}" >&2
  exit 1
fi

EXP_NAME="${EXP_NAME:-td3_table4_counterbmt_risk_${RISK_VIEW}_${EVAL_SUFFIX}}"

export REPO_ROOT
export DATA_DIR
export EVAL_DATA_DIR
export EXP_NAME

exec bash "${REPO_ROOT}/scripts/remote/run_td3_table4_openloop.sh" "${SEED}"
