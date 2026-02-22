# P4 Forward Metric Parity (Dual-Env, Offline Strict)

This document defines the P4 workflow for forward metric parity:
- training env computes and logs `forward_approx/*`,
- training env exports per-scenario forward-eval artifacts,
- strict parity script consumes artifacts and reports `forward_parity/*`.

## Scope
- Core + realism metrics only:
  - `sfde_min`, `sfde_avg`, `sade_min`, `sade_avg`, `ssde_min`, `ssde_avg`
  - `fdd`, `add`, `sdd`
  - `vel_jsd`, `acc_jsd`, `ttc_jsd`
- Collision/offroad/comfort strict parity is deferred.

## Artifact Layout
- Artifacts are written under:
  - `outputs/<run>/forward_eval_artifacts/step_<step>/scenario_<id>.npz`
  - `outputs/<run>/forward_eval_artifacts/step_<step>/manifest.json`

Each `.npz` contains strict-recompute tensors:
- `pred_pos_ktn2`, `pred_vel_ktn2`, `pred_speed_ktn`, `pred_valid_ktn`, `pred_heading_ktn`
- `gt_pos_tn2`, `gt_vel_tn2`, `gt_valid_tn`, `gt_heading_tn`
- `agent_shape_n3`, `dt_chunk_s`, `sdc_index`, `scenario_id`
- `forward_approx_metric_keys`, `forward_approx_metric_values`

## Commands

1. Approx path regression smoke
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p4_smoke_approx \
  --model-preset midgpt_parity \
  --tokenizer-mode adv_bmt_parity \
  --max-steps 2 --batch-size 1 \
  --eval-every 1 --eval-batches 1 --log-every 1 \
  --forward-export-artifacts
```

2. Artifact schema sanity
```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import numpy as np, glob
f = glob.glob('outputs/p4_smoke_approx/forward_eval_artifacts/step_*/*.npz')[0]
d = np.load(f, allow_pickle=True)
print(sorted(d.files))
PY
```

3. Strict comparison gate (legacy-compatible env)
```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/compare_forward_metrics.py \
  --artifact-dir outputs/p4_smoke_approx/forward_eval_artifacts \
  --output-json outputs/p4_forward_parity_report.json \
  --max-rel-error 0.01 \
  --min-corr 0.99
```
If TensorFlow/Waymo TTC operators are unavailable, the script falls back to approximate TTC and reports backend usage in `summary.ttc_backend_counts`.

4. Compatibility fallback (no strict deps needed)
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/p4_no_strict_deps \
  --max-steps 1 --batch-size 1 --eval-every 1 --eval-batches 1
```

## Pass Criteria
- Approx smoke:
  - train/eval completes,
  - `forward_approx/*` keys exist in eval log entries.
- Artifact export:
  - `.npz` files and `manifest.json` exist under step folders.
- Strict compare:
  - report JSON contains:
    - `forward_approx/*`
    - `forward_parity/*`
    - per-metric abs/rel error
    - scenario-level correlation
  - exits `0` when thresholds are met.
- Compatibility:
  - training/eval still works when strict dependencies are absent.
