#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
PYTHON_BIN="${PYTHON_BIN:-}"
RENDER_MANIFEST="${RENDER_MANIFEST:-$REPO_ROOT/outputs/pr8_vlm_semantics_bundle_local/vlm_render_manifest.json}"
PATH_INDEX="${PATH_INDEX:-$REPO_ROOT/outputs/pr6_eval_debug_bundle_20260403/outputs/pr6_path_index_5000/path_index_curated_val.jsonl}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr8_vlm_semantics_bundle_local/batch_local}"
OPENAI_DOTENV_PATH="${OPENAI_DOTENV_PATH:-$REPO_ROOT/.env}"
MODEL_DEFAULT="${MODEL_DEFAULT:-gpt-5.4-mini}"
MODEL_ESCALATE="${MODEL_ESCALATE:-gpt-5.4}"
IMAGE_DETAIL_DEFAULT="${IMAGE_DETAIL_DEFAULT:-low}"
IMAGE_DETAIL_ESCALATE="${IMAGE_DETAIL_ESCALATE:-high}"
ESCALATE_CONFIDENCE_THRESHOLD="${ESCALATE_CONFIDENCE_THRESHOLD:-0.65}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"
SUBMIT_BATCH="${SUBMIT_BATCH:-true}"
WAIT_FOR_BATCH="${WAIT_FOR_BATCH:-true}"
BATCH_OUTPUT_FILE="${BATCH_OUTPUT_FILE:-}"

cd "$REPO_ROOT"
export PYTHONPATH

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv-mac/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv-mac/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -f "$OPENAI_DOTENV_PATH" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$OPENAI_DOTENV_PATH"
  set +a
fi

LABEL_ARGS=(
  --render-manifest "$RENDER_MANIFEST"
  --outdir "$OUTDIR"
  --model-default "$MODEL_DEFAULT"
  --model-escalate "$MODEL_ESCALATE"
  --image-detail-default "$IMAGE_DETAIL_DEFAULT"
  --image-detail-escalate "$IMAGE_DETAIL_ESCALATE"
  --escalate-confidence-threshold "$ESCALATE_CONFIDENCE_THRESHOLD"
  --completion-window "$COMPLETION_WINDOW"
  --dotenv "$OPENAI_DOTENV_PATH"
  --batch-mode
)

if [[ "$SUBMIT_BATCH" == "true" ]]; then
  LABEL_ARGS+=(--submit-batch)
fi

if [[ "$WAIT_FOR_BATCH" == "true" ]]; then
  LABEL_ARGS+=(--wait-for-batch)
fi

if [[ -n "$BATCH_OUTPUT_FILE" ]]; then
  LABEL_ARGS+=(--batch-output-file "$BATCH_OUTPUT_FILE")
fi

"$PYTHON_BIN" scripts/counterfactual/label_vlm_semantics.py "${LABEL_ARGS[@]}"

if [[ -f "$OUTDIR/vlm_contracts_raw.jsonl" && -s "$OUTDIR/vlm_contracts_raw.jsonl" && -f "$PATH_INDEX" ]]; then
  "$PYTHON_BIN" scripts/counterfactual/fuse_vlm_semantic_contracts.py \
    --contracts "$OUTDIR/vlm_contracts_raw.jsonl" \
    --path-index "$PATH_INDEX" \
    --outdir "$OUTDIR"
fi
