#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
SCENARIO_ROOT="${SCENARIO_ROOT:-/path/to/scenario_root}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr6_path_index}"
MAX_SCENARIOS="${MAX_SCENARIOS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"

cd "$REPO_ROOT"
export PYTHONPATH
export PYTHONUNBUFFERED=1

python scripts/counterfactual/build_path_control_index.py \
  --scenario-root "$SCENARIO_ROOT" \
  --outdir "$OUTDIR" \
  --max-scenarios "$MAX_SCENARIOS" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  --artifact-mode minimal \
  --progress \
  --dedup-mode overlap_cluster \
  --decision-time-merge-frames 5 \
  --window-overlap-threshold 0.5 \
  --anchor-dist-threshold 5.0 \
  --anchor-heading-threshold-rad 0.35 \
  --path-only-safe-mode \
  --write-examples-manifest \
  --write-histograms \
  --write-summary

python scripts/counterfactual/split_path_index_by_scenario.py \
  --path-index "$OUTDIR/path_index_curated.jsonl" \
  --outdir "$OUTDIR" \
  --seed "$SEED" \
  --val-fraction "$VAL_FRACTION"
