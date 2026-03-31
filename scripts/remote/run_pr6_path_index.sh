#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
SCENARIO_ROOT="${SCENARIO_ROOT:-/path/to/scenario_root}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr6_path_index}"
MAX_SCENARIOS="${MAX_SCENARIOS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"

cd "$REPO_ROOT"
export PYTHONPATH

python scripts/counterfactual/build_path_control_index.py \
  --scenario-root "$SCENARIO_ROOT" \
  --outdir "$OUTDIR" \
  --max-scenarios "$MAX_SCENARIOS" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --write-examples-manifest \
  --write-histograms \
  --write-summary
