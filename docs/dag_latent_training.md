# DAG-Latent Training + Evaluation Guide

This guide covers:
1. Recommended staged DAG-latent training commands (including H200 usage).
2. How to assess whether a long run is healthy.
3. How to generate trajectories from trained checkpoints and replay them in ScenarioNet/MetaDrive.

## 1) Recommended Training Commands

## 1.1 DAG-latent smoke (small, local-safe)
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/dag_latent_smoke \
  --model-preset paper_like_small \
  --tokenizer-mode paper_simple \
  --stage A_B_C \
  --stage-a-steps 40 \
  --stage-b-steps 40 \
  --stage-c-steps 40 \
  --batch-size 2
```

## 1.2 DAG-latent on H200 (opt-in path)
Use this only if you intentionally want DAG-latent conditioning.  
Default `train_nnx_bmt` remains unchanged and does not use DAG latents.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --train-data-dir data/_scenarionet_waymo_training_full_v12 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_dag_latent_full \
  --model-preset midgpt_dag_latent \
  --tokenizer-mode adv_bmt_parity \
  --distributed-backend pmap \
  --precision bf16-mixed \
  --batch-size 8 \
  --num-train-scenarios 486992 \
  --stage A_B_C \
  --stage-a-steps 60000 \
  --stage-b-steps 60000 \
  --stage-c-steps 180000 \
  --max-steps 300000 \
  --eval-every 2000 \
  --eval-batches 20 \
  --checkpoint-every 2000 \
  --log-every 50
```

Notes:
- Keep `num-train-scenarios` divisible by global batch for pmap stability.
- If memory is tight, reduce `max-map-features`, `max-vectors`, `max-agents`, or switch to `paper_like_small`.
- For stage B/C with strict cache mode, cache files must be compact v2:
  - `schema_version=counter_bmt_v2_dag_cache_v2_compact10`
  - v1 cache files are intentionally rejected.

## 2) Assessing a Finished Long Run

Assume:
```bash
RUN=outputs/counter_bmt_v2_training_womd_full
```

## 2.1 Quick sanity check from summary
```bash
cat "$RUN/summary.json"
```
Focus on:
- `best_eval_step` and `best_eval_loss`
- `final_checkpoint`
- `final_eval_metrics.forward_approx/*` (if forward eval enabled)

## 2.2 Trend check from `metrics.jsonl`
```bash
PYTHONPATH=src .venv-v2/bin/python - <<'PY'
import json, statistics
from pathlib import Path

run = Path("outputs/counter_bmt_v2_training_womd_full")
rows = [json.loads(l) for l in (run / "metrics.jsonl").open()]
eval_rows = [r for r in rows if r.get("phase") == "eval"]
if not eval_rows:
    print("No eval rows found.")
    raise SystemExit(0)

def seq(key):
    out = []
    for r in eval_rows:
        v = r.get("metrics", {}).get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out

keys = [
    "total_loss",
    "accuracy",
    "rate_default_pred",
    "forward_approx/sfde_min",
    "forward_approx/fdd",
    "forward_approx/vel_jsd",
    "forward_approx/acc_jsd",
    "forward_approx/ttc_jsd",
]

print(f"num_eval_points={len(eval_rows)}")
for k in keys:
    s = seq(k)
    if not s:
        continue
    n = min(5, len(s))
    head = statistics.mean(s[:n])
    tail = statistics.mean(s[-n:])
    print(f"{k}: first{n}_mean={head:.6f} last{n}_mean={tail:.6f}")
PY
```

Interpretation guidance:
- `total_loss` should generally decline then stabilize.
- `rate_default_pred` near `1.0` for long periods indicates token-collapse risk.
- `forward_approx/sfde_min` should improve over early checkpoints.
- JSD realism metrics (`vel_jsd`, `acc_jsd`, `ttc_jsd`) should not drift upward badly while loss decreases.

## 2.3 Inspect eval visualizations (if enabled)
```bash
find "$RUN/forward_eval_viz" -type f | head
```
These plots show rollout vs GT for selected scenarios and are the fastest qualitative check.

## 3) Generate Trajectories + Replay in ScenarioNet

If your long run used `--no-forward-export-artifacts`, first create a short artifact-export eval pass from your trained checkpoint.

## 3.1 Export forward artifacts from a trained checkpoint
```bash
CKPT=outputs/counter_bmt_v2_training_womd_full/checkpoints/step_00XXXXXX.pkl

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir data/_scenarionet_waymo_training_full_v12 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_eval_artifacts \
  --runtime-preset adv_bmt_runtime_parity \
  --distributed-backend none \
  --precision fp32 \
  --batch-size 1 \
  --num-train-scenarios 1 \
  --max-steps 1 \
  --lr 0 \
  --warmup-steps 1 \
  --resume-checkpoint "$CKPT" \
  --eval-every 1 \
  --eval-batches 20 \
  --checkpoint-every 100000 \
  --log-every 1 \
  --forward-export-artifacts \
  --forward-artifact-max-scenarios 20 \
  --forward-artifact-subdir forward_eval_artifacts
```

This runs one tiny step and exports per-scenario rollout tensors under:
`outputs/counter_bmt_v2_eval_artifacts/forward_eval_artifacts/step_*/`.

## 3.2 Convert one artifact to replayable ScenarioNet files
```bash
ART=$(find outputs/counter_bmt_v2_eval_artifacts/forward_eval_artifacts -name "*.npz" | head -n 1)

PYTHONPATH=src .venv-v2/bin/python src/scripts/replay/export_forward_artifact_to_scenario.py \
  --artifact-npz "$ART" \
  --scenario-root data/scenarionet_waymo_training_500 \
  --output-dir outputs/replay_from_forward_artifact \
  --mode-index 0 \
  --intervention-name model_rollout \
  --include-ground-truth
```

This writes:
- replay scenario pkls (`sd_*.pkl`)
- `dataset_summary.pkl` + `dataset_mapping.pkl`
- `replay_scenarios.py`

## 3.3 Replay in ScenarioNet simulator (recommended)
```bash
python -m scenarionet.sim -d outputs/replay_from_forward_artifact --render 2D
```

Optional MetaDrive replay:
```bash
PYTHONPATH=src .venv-v2/bin/python outputs/replay_from_forward_artifact/replay_scenarios.py --list
PYTHONPATH=src .venv-v2/bin/python outputs/replay_from_forward_artifact/replay_scenarios.py --scenario 0 --render
```

## 4) Useful Paths to Keep

- Run summary: `outputs/<run>/summary.json`
- Full logs: `outputs/<run>/metrics.jsonl`
- Checkpoints: `outputs/<run>/checkpoints/*.pkl`
- Forward eval visuals: `outputs/<run>/forward_eval_viz/step_*/`
- Forward eval artifacts: `outputs/<run>/forward_eval_artifacts/step_*/`
