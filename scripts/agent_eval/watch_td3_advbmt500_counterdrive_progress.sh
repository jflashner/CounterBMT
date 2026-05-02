#!/usr/bin/env bash
set -euo pipefail

INTERVAL="${INTERVAL:-10}"
HOST="${HOST:-zhoulab-2.cs.vt.edu}"
TARGET_STEPS="${TARGET_STEPS:-1000000}"
ROOT="${ROOT:-/data/home/grads/jflashner/CounterBMT_run/logs/td3_advbmt500_counterdrive_progresssoft_nostop_cand2}"

python3 "$(dirname "$0")/watch_remote_build_progress.py" \
  --interval "${INTERVAL}" \
  --run "td3-natural|${HOST}|td3_advbmt500_counterdrive_natural_nowandb|${ROOT}/td3_advbmt500_counterdrive_progresssoft_nostop_cand2_eval_natural_nowandb_seed0|${TARGET_STEPS}" \
  --run "td3-adv|${HOST}|td3_advbmt500_counterdrive_adversarial_nowandb|${ROOT}/td3_advbmt500_counterdrive_progresssoft_nostop_cand2_eval_adversarial_nowandb_seed0|${TARGET_STEPS}"
