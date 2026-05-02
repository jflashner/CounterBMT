#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/home/grads/jflashner/CounterBMT_run}"
ENV="${ENV:-/data/home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/activate}"
DATA_ROOT="${DATA_ROOT:-/data/home/grads/jflashner/CounterBMT/outputs/pr10_1_sdc_semantic_top859_full/scenario_root}"
INDEX_ROOT="${INDEX_ROOT:-$REPO/outputs/pr10_1_sdc_semantic_top500_from859}"
ALL_INDEX="${ALL_INDEX:-$INDEX_ROOT/sdc_semantic_control_index_val.jsonl}"
GT_INDEX="${GT_INDEX:-$INDEX_ROOT/sdc_semantic_control_index_val_factual_gt.jsonl}"
NOINTERVENTION_GT_INDEX="${NOINTERVENTION_GT_INDEX:-$INDEX_ROOT/sdc_semantic_control_index_val_nointervention_gt_only.jsonl}"
BASE_TEACHER="${BASE_TEACHER:-$REPO/src/Adv-BMT/bmt/ckpt/last.ckpt}"
CKPT="${CKPT:-$REPO/logs/progresssoft_nointervention_gtce50_repair_runs/pr10_1_top500_actualwall_progresssoft_nointervention_gtce50_repair_1gpu_a100_zh1_run2_resume_fastval/lightning_logs/infgen/pr10_1_top500_actualwall_progresssoft_nointervention_gtce50_repair_1gpu_a100_zh1_run2_resume_fastval_2026-04-23/checkpoints/last.ckpt}"
OUTROOT="${OUTROOT:-$REPO/eval_runs/nointervention_repair_20260424}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

LIMIT_ALL="${LIMIT_ALL:--1}"
LIMIT_GT="${LIMIT_GT:--1}"
RUN_NOINTERVENTION_GT="${RUN_NOINTERVENTION_GT:-true}"
TAG_PREFIX="${TAG_PREFIX:-0424_GPTmodel_nointerv_repair}"

export PYTHONPATH="/data/home/grads/jflashner/CounterBMT/metadrive:$REPO/scenarionet:$REPO/src/Adv-BMT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

source "$ENV"

run_eval() {
  local name="$1"
  local index="$2"
  local modeflag="$3"
  local workers="$4"
  local limit="$5"
  local outdir="$OUTROOT/$name"
  mkdir -p "$outdir"
  cd "$outdir"
  echo "[$(date -Is)] START $name ckpt=$CKPT index=$index flag=$modeflag limit=$limit gpu=$CUDA_VISIBLE_DEVICES"
  python "$REPO/src/Adv-BMT/bmt/eval/evaluate_scenario_metrics.py" --config-name motion_forward_sdc_semantic_only_strict_local.yaml \
    eval_mode=GPTmodel \
    multi_mode=true \
    +test_batch_size=1 \
    +limit_test_batches="$limit" \
    val_num_workers="$workers" \
    +pin_memory=false \
    +persistent_workers=false \
    "$modeflag" \
    ckpt="'$CKPT'" \
    CKPT_LOAD_MODE=forgiving_state_dict \
    DATA.TRAINING_DATA_DIR="$DATA_ROOT" \
    DATA.TEST_DATA_DIR="$DATA_ROOT" \
    DATA.COUNTERFACTUAL_MODE=sdc_semantic_only \
    DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$index" \
    DATA.COUNTERFACTUAL_CONTROL_INDEX="$index" \
    DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER=false \
    MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT="$BASE_TEACHER" \
    ++MODEL.LOCAL_CONTROL_SDC_SEMANTIC_EXT_ENABLED=true \
    ++MODEL.LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_MODEL_ENABLED=false \
    +save_tag="${TAG_PREFIX}_${name}"
  echo "[$(date -Is)] DONE $name"
}

run_eval alllabels "$ALL_INDEX" +key_metrics_only=true 4 "$LIMIT_ALL"
run_eval gt "$GT_INDEX" +start_metrics_only=true 0 "$LIMIT_GT"
if [[ "$RUN_NOINTERVENTION_GT" == "true" ]]; then
  run_eval nointervention_gt "$NOINTERVENTION_GT_INDEX" +start_metrics_only=true 0 "$LIMIT_GT"
fi

cd "$OUTROOT"
python - <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path

root = Path.cwd()
runs = ["alllabels", "gt", "nointervention_gt"]
keys = [
    "scenario_count",
    "sade_avg", "sade_min", "sfde_avg", "sfde_min",
    "sdc_ade_avg", "sdc_ade_min", "sdc_fde_avg", "sdc_fde_min",
    "sdc_gtarc_clip_ade_avg", "sdc_gtarc_clip_ade_min",
    "sdc_gtarc_clip_fde_avg", "sdc_gtarc_clip_fde_min",
    "vel_jsd", "acc_jsd", "ttc_jsd",
    "add", "fdd",
    "semantic_scene_alt_scene_count",
    "semantic_scene_alt_add", "semantic_scene_alt_fdd", "semantic_scene_alt_sdd",
    "semantic_scene_alt_rollout_count_mean",
]
rows = []
for run in runs:
    run_dir = root / run
    files = sorted(run_dir.glob("*_multi_mode_open_loop_results.json"))
    if not files:
        rows.append({"run": run, "status": "missing"})
        continue
    data = json.loads(files[-1].read_text())
    row = {"run": run, "status": "ok", "json": str(files[-1])}
    for key in keys:
        if key in data:
            row[key] = data[key]
    rows.append(row)

(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
fieldnames = ["run", "status", "json"] + keys
with (root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY
echo "[$(date -Is)] ALL DONE"
