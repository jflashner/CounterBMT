#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-0}"

REPO_ROOT="${REPO_ROOT:-/data/home/grads/jflashner/CounterBMT_run}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_ROOT}/src/Adv-BMT/bmt/rl_train/train/train_td3.py}"

DATA_DIR="${DATA_DIR:?Set DATA_DIR to a TD3-ready ScenarioNet directory}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:?Set EVAL_DATA_DIR to a TD3-ready ScenarioNet directory}"
SAVE_ROOT="${SAVE_ROOT:-${REPO_ROOT}/logs/td3_table4_runs}"
EXP_NAME="${EXP_NAME:-td3_table4_openloop}"

TRAINING_STEPS="${TRAINING_STEPS:-1000000}"
LR="${LR:-1e-4}"
HORIZON="${HORIZON:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-100}"
EVAL_FREQ="${EVAL_FREQ:-100000}"
EVAL_EP="${EVAL_EP:-100}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"

USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-scgen}"
WANDB_TEAM="${WANDB_TEAM:-drivingforce}"

SAVE_PATH="${SAVE_ROOT}/${EXP_NAME}_seed${SEED}"
mkdir -p "${SAVE_PATH}"

CMD=(
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}"
  --exp_name="${EXP_NAME}"
  --seed="${SEED}"
  --data_dir="${DATA_DIR}"
  --eval_data_dir="${EVAL_DATA_DIR}"
  --save_path="${SAVE_PATH}"
  --training_step="${TRAINING_STEPS}"
  --lr="${LR}"
  --eval_freq="${EVAL_FREQ}"
  --horizon="${HORIZON}"
  --eval_horizon="${EVAL_HORIZON}"
  --num_eval_envs="${NUM_EVAL_ENVS}"
  --eval_ep="${EVAL_EP}"
)

if [[ "${USE_WANDB}" == "true" ]]; then
  CMD+=(
    --wandb
    --wandb_project="${WANDB_PROJECT}"
    --wandb_team="${WANDB_TEAM}"
  )
fi

printf 'Running TD3 command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "${SAVE_PATH}/train.log"
