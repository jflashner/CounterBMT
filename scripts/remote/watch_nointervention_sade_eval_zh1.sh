#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/home/grads/jflashner/CounterBMT_run}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTHONPATH="${PYTHONPATH:-/data/home/grads/jflashner/CounterBMT/metadrive:$REPO_ROOT/scenarionet:$REPO_ROOT/src/Adv-BMT}"

TRAIN_RUN_DIR="${TRAIN_RUN_DIR:?Set TRAIN_RUN_DIR to the training run output directory}"
DATA_ROOT="${DATA_ROOT:-/data/home/grads/jflashner/CounterBMT_run/outputs/nointervention_gtce_combined_waymo3394_waymax859/scenario_root}"
CONTROL_INDEX="${CONTROL_INDEX:-/data/home/grads/jflashner/CounterBMT_run/outputs/nointervention_gtce_combined_waymo3394_waymax859/sdc_semantic_control_index_val_nointervention_gtce_waymax172.jsonl}"
TEACHER_CKPT="${TEACHER_CKPT:-$REPO_ROOT/src/Adv-BMT/bmt/ckpt/last.ckpt}"

EVAL_ROOT="${EVAL_ROOT:-$TRAIN_RUN_DIR/sidecar_sade_eval}"
POLL_SECONDS="${POLL_SECONDS:-120}"
START_STEP="${START_STEP:-0}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
MAX_EVALS="${MAX_EVALS:-0}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:--1}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
MIN_CHECKPOINT_AGE_SECONDS="${MIN_CHECKPOINT_AGE_SECONDS:-30}"
CONFIG_NAME="${CONFIG_NAME:-motion_forward_sdc_semantic_only_strict_local.yaml}"
RUN_ONCE="${RUN_ONCE:-false}"
INITIAL_CKPT="${INITIAL_CKPT:-}"
SIDECAR_WANDB_ENABLED="${SIDECAR_WANDB_ENABLED:-true}"
SIDECAR_WANDB_ENTITY="${SIDECAR_WANDB_ENTITY:-flashner}"
SIDECAR_WANDB_PROJECT="${SIDECAR_WANDB_PROJECT:-infgen}"
SIDECAR_WANDB_RUN_NAME="${SIDECAR_WANDB_RUN_NAME:-$(basename "$TRAIN_RUN_DIR")_sidecar_sade}"
SIDECAR_WANDB_RUN_ID="${SIDECAR_WANDB_RUN_ID:-$(basename "$TRAIN_RUN_DIR" | tr -c '[:alnum:]_-' '_' | cut -c1-96)_sade}"

mkdir -p "$EVAL_ROOT"
export PYTHONPATH
export PYTHONUNBUFFERED=1
export SIDECAR_WANDB_ENABLED SIDECAR_WANDB_ENTITY SIDECAR_WANDB_PROJECT SIDECAR_WANDB_RUN_NAME SIDECAR_WANDB_RUN_ID
if [[ "$EVAL_DEVICE" == "cpu" ]]; then
  export CUDA_VISIBLE_DEVICES=""
elif [[ -n "$EVAL_DEVICE" ]]; then
  export CUDA_VISIBLE_DEVICES="$EVAL_DEVICE"
fi

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

state_file="$EVAL_ROOT/evaluated.tsv"
touch "$state_file"

step_from_path() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).name
matches = re.findall(r"step[=_-]?(\d+)", text)
if matches:
    print(int(matches[-1]))
else:
    print(-1)
PY
}

next_candidate() {
  local min_step="$1"
  local candidates=()
  mapfile -t candidates < <({
    if [[ -n "$INITIAL_CKPT" && -f "$INITIAL_CKPT" ]]; then
      printf '%s\n' "$INITIAL_CKPT"
    fi
    find "$TRAIN_RUN_DIR" -type f -path "*/checkpoints/*.ckpt" ! -name "last.ckpt" 2>/dev/null || true
  })
  "$PYTHON_BIN" - "$min_step" "$MIN_CHECKPOINT_AGE_SECONDS" "${candidates[@]}" <<'PY'
import re
import sys
import time
from pathlib import Path

min_step = int(sys.argv[1])
min_age = float(sys.argv[2])
best = None
now = time.time()
for raw in sys.argv[3:]:
    path = Path(raw)
    if not path.is_file():
        continue
    matches = re.findall(r"step[=_-]?(\d+)", path.name)
    if not matches:
        matches = re.findall(r"step_(\d+)", str(path))
    if not matches:
        continue
    step = int(matches[-1])
    if step < min_step:
        continue
    stat = path.stat()
    if min_age > 0 and now - stat.st_mtime < min_age:
        continue
    item = (step, stat.st_mtime, str(path))
    if best is None or item < best:
        best = item
if best is not None:
    print(best[2])
PY
}

already_done() {
  local path="$1"
  local sig
  sig="$(stat -c '%n|%s|%Y' "$path")"
  grep -Fqx "$sig" "$state_file"
}

mark_done() {
  local path="$1"
  stat -c '%n|%s|%Y' "$path" >> "$state_file"
}

append_summary() {
  local step="$1"
  local ckpt="$2"
  local eval_dir="$3"
  "$PYTHON_BIN" - "$step" "$ckpt" "$eval_dir" "$EVAL_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

step = int(sys.argv[1])
ckpt = sys.argv[2]
eval_dir = Path(sys.argv[3])
eval_root = Path(sys.argv[4])
files = sorted(eval_dir.glob("*_open_loop_results.json"))
row = {
    "step": step,
    "checkpoint": ckpt,
    "eval_dir": str(eval_dir),
    "status": "missing_results",
}
if files:
    data = json.loads(files[-1].read_text())
    keys = [
        "scenario_count",
        "sade_avg",
        "sade_min",
        "sfde_avg",
        "sfde_min",
        "sdc_ade_avg",
        "sdc_ade_min",
        "sdc_fde_avg",
        "sdc_fde_min",
        "sdc_gtarc_clip_ade_avg",
        "sdc_gtarc_clip_ade_min",
        "sdc_gtarc_clip_fde_avg",
        "sdc_gtarc_clip_fde_min",
    ]
    row.update({key: data[key] for key in keys if key in data})
    row["status"] = "ok"
    row["result_json"] = str(files[-1])

if str(os.environ.get("SIDECAR_WANDB_ENABLED", "true")).lower() in {"1", "true", "yes", "on"}:
    try:
        import wandb

        wandb_dir = eval_root / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": os.environ.get("SIDECAR_WANDB_PROJECT", "infgen"),
            "name": os.environ.get("SIDECAR_WANDB_RUN_NAME", "sidecar_sade"),
            "id": os.environ.get("SIDECAR_WANDB_RUN_ID", "sidecar_sade"),
            "resume": "allow",
            "dir": str(wandb_dir),
        }
        entity = os.environ.get("SIDECAR_WANDB_ENTITY", "")
        if entity:
            init_kwargs["entity"] = entity
        run = wandb.init(**init_kwargs)
        metrics = {
            f"sidecar_sade/{key}": value
            for key, value in row.items()
            if isinstance(value, (int, float))
        }
        metrics["sidecar_sade/status_ok"] = 1 if row.get("status") == "ok" else 0
        wandb.log(metrics, step=step)
        row["wandb_url"] = run.url
        wandb.finish()
    except Exception as exc:
        row["wandb_error"] = repr(exc)
with (eval_root / "summary.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps(row, indent=2, sort_keys=True))
PY
}

run_eval() {
  local ckpt="$1"
  local step="$2"
  local eval_dir="$EVAL_ROOT/step_$(printf '%06d' "$step")"
  mkdir -p "$eval_dir"
  echo "[$(date -Is)] START sidecar SADE step=$step ckpt=$ckpt"
  (
    cd "$eval_dir"
    "$PYTHON_BIN" "$REPO_ROOT/src/Adv-BMT/bmt/eval/evaluate_scenario_metrics.py" \
      --config-name "$CONFIG_NAME" \
      eval_mode=GPTmodel \
      multi_mode=true \
      ckpt="'$ckpt'" \
      CKPT_LOAD_MODE=forgiving_state_dict \
      DATA.TRAINING_DATA_DIR="$DATA_ROOT" \
      DATA.TEST_DATA_DIR="$DATA_ROOT" \
      DATA.COUNTERFACTUAL_CONTROL_INDEX_VAL="$CONTROL_INDEX" \
      DATA.COUNTERFACTUAL_CONTROL_INDEX="$CONTROL_INDEX" \
      DATA.COUNTERFACTUAL_MODE=sdc_semantic_only \
      DATA.COUNTERFACTUAL_WEIGHTED_SAMPLER=false \
      MODEL.LOCAL_CONTROL_SDC_POLICY_TEACHER_CKPT="$TEACHER_CKPT" \
      ++MODEL.LOCAL_CONTROL_SDC_SEMANTIC_EXT_ENABLED=true \
      ++MODEL.LOCAL_CONTROL_SDC_SEMANTIC_EXT_BEHAVIOR_MODEL_ENABLED=false \
      ++test_batch_size="$TEST_BATCH_SIZE" \
      ++limit_test_batches="$LIMIT_TEST_BATCHES" \
      ++val_num_workers="$VAL_NUM_WORKERS" \
      ++pin_memory=false \
      ++persistent_workers=false \
      ++key_metrics_only=true \
      ++start_metrics_only=false \
      ++save_tag="sidecar_sade_step_$(printf '%06d' "$step")" \
      hydra.run.dir=. \
      hydra.output_subdir=.hydra
  ) > "$eval_dir/eval.log" 2>&1
  append_summary "$step" "$ckpt" "$eval_dir"
  mark_done "$ckpt"
  echo "[$(date -Is)] DONE sidecar SADE step=$step"
}

echo "[$(date -Is)] Watching $TRAIN_RUN_DIR for checkpoints; eval_root=$EVAL_ROOT"
echo "[$(date -Is)] Sidecar checkpoint policy: oldest unevaluated checkpoint with step >= START_STEP and age >= ${MIN_CHECKPOINT_AGE_SECONDS}s"
eval_count=0
next_step="$START_STEP"
while true; do
  ckpt="$(next_candidate "$next_step" || true)"
  if [[ -n "$ckpt" && -f "$ckpt" ]]; then
    step="$(step_from_path "$ckpt")"
    if [[ "$step" -ge "$next_step" ]] && ! already_done "$ckpt"; then
      run_eval "$ckpt" "$step"
      eval_count=$((eval_count + 1))
      next_step=$((step + STEP_INTERVAL))
      if [[ "$MAX_EVALS" -gt 0 && "$eval_count" -ge "$MAX_EVALS" ]]; then
        echo "[$(date -Is)] Reached MAX_EVALS=$MAX_EVALS"
        exit 0
      fi
    fi
  fi
  if [[ "$RUN_ONCE" == "true" ]]; then
    echo "[$(date -Is)] RUN_ONCE complete"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
