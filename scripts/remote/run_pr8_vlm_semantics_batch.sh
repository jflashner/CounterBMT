#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/path/to/CounterBMT}"
PYTHONPATH="${PYTHONPATH:-$REPO_ROOT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"
DATA_ROOT="${DATA_ROOT:-/path/to/scenario_root}"
INPUT_INDEX="${INPUT_INDEX:-/path/to/path_index_curated_val.jsonl}"
OUTDIR="${OUTDIR:-$REPO_ROOT/outputs/pr8_vlm_semantics_batch}"
OPENAI_DOTENV_PATH="${OPENAI_DOTENV_PATH:-$REPO_ROOT/.env}"
MODEL_DEFAULT="${MODEL_DEFAULT:-gpt-5.4-mini}"
MODEL_ESCALATE="${MODEL_ESCALATE:-gpt-5.4}"
IMAGE_DETAIL_DEFAULT="${IMAGE_DETAIL_DEFAULT:-low}"
IMAGE_DETAIL_ESCALATE="${IMAGE_DETAIL_ESCALATE:-high}"
BATCH_MODE="${BATCH_MODE:-true}"
MAX_EXAMPLES="${MAX_EXAMPLES:-5000}"
ESCALATE_CONFIDENCE_THRESHOLD="${ESCALATE_CONFIDENCE_THRESHOLD:-0.65}"
WAIT_FOR_BATCH="${WAIT_FOR_BATCH:-true}"

cd "$REPO_ROOT"
export PYTHONPATH

if [[ -f "$OPENAI_DOTENV_PATH" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$OPENAI_DOTENV_PATH"
  set +a
fi

python scripts/counterfactual/materialize_path_debug_bundles.py \
  --path-index "$INPUT_INDEX" \
  --outdir "$OUTDIR/materialized_debug" \
  --sample-total "$MAX_EXAMPLES"

python scripts/counterfactual/render_vlm_semantic_views.py \
  --materialized-manifest "$OUTDIR/materialized_debug/debug_bundle_manifest.jsonl" \
  --path-index "$INPUT_INDEX" \
  --outdir "$OUTDIR"

LABEL_ARGS=(
  --render-manifest "$OUTDIR/vlm_render_manifest.json"
  --outdir "$OUTDIR"
  --model-default "$MODEL_DEFAULT"
  --model-escalate "$MODEL_ESCALATE"
  --image-detail-default "$IMAGE_DETAIL_DEFAULT"
  --image-detail-escalate "$IMAGE_DETAIL_ESCALATE"
  --escalate-confidence-threshold "$ESCALATE_CONFIDENCE_THRESHOLD"
  --dotenv "$OPENAI_DOTENV_PATH"
)

if [[ "$BATCH_MODE" == "true" ]]; then
  LABEL_ARGS+=(--batch-mode --submit-batch)
  if [[ "$WAIT_FOR_BATCH" == "true" ]]; then
    LABEL_ARGS+=(--wait-for-batch)
  fi
fi

python scripts/counterfactual/label_vlm_semantics.py "${LABEL_ARGS[@]}"

python scripts/counterfactual/fuse_vlm_semantic_contracts.py \
  --contracts "$OUTDIR/vlm_contracts_raw.jsonl" \
  --path-index "$INPUT_INDEX" \
  --outdir "$OUTDIR"
