# CounterBMT v2 Command Reference

This is the consolidated command cookbook for the current repo.

Use this when you want one place that answers:
- what each command does
- the most important options
- copy/paste examples for common workflows

For full flag details, run `--help` on each command.

## Conventions

Examples assume:
- repository root is current working directory
- `PYTHONPATH=src`
- one of:
  - local env: `.venv/bin/python`
  - training/eval env: `.venv-v2/bin/python`

If your env differs, replace the Python executable path.

## 1) Supervised Training (Base v2)

Command:
- `python -m counter_bmt_v2.cli.train_nnx_bmt`

What it does:
- trains the v2 motion model (non-DAG-latent path)
- supports explicit train/val splits or fallback split from one `--data-dir`
- supports pmap (`--distributed-backend pmap`) and bf16 mixed precision

Key options:
- data:
  - `--train-data-dir`, `--val-data-dir`
  - fallback: `--data-dir` + `--train-fraction`
  - `--num-train-scenarios`, `--num-val-scenarios`
  - `--sample-interval-training`, `--sample-interval-test`
- model/runtime:
  - `--model-preset {paper_like_small,paper_like_full,midgpt_parity}`
  - `--runtime-preset {none,adv_bmt_runtime_parity}`
  - `--tokenizer-mode {paper_simple,adv_bmt_parity}`
- optimization:
  - `--batch-size`, `--max-steps`, `--lr`, `--warmup-steps`
  - `--lr-schedule-mode {v2_cosine_minlr,legacy_cosine_zero}`
- scale/precision:
  - `--distributed-backend {none,pmap}`
  - `--precision {fp32,bf16-mixed}`
- eval/checkpoint/logging:
  - `--eval-every`, `--eval-batches`, `--checkpoint-every`, `--log-every`
  - `--forward-eval-*`, `--forward-viz-*`, `--forward-export-artifacts`
  - `--tensorboard`, `--tensorboard-subdir`
- resume:
  - `--resume-checkpoint`
  - `--no-resume-strict-determinism` for eval-only or changed split hashes

### Example: quick smoke (single device)

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/train_smoke_base \
  --model-preset paper_like_small \
  --max-steps 80 \
  --batch-size 2 \
  --eval-every 40 \
  --eval-batches 2 \
  --checkpoint-every 40 \
  --log-every 10
```

### Example: full-style H200 run (explicit split + pmap)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir data/_scenarionet_waymo_training_full_v12 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_womd_full \
  --runtime-preset adv_bmt_runtime_parity \
  --distributed-backend pmap \
  --precision bf16-mixed \
  --batch-size 8 \
  --num-train-scenarios 486992 \
  --max-steps 300000 \
  --eval-every 2000 \
  --eval-batches 20 \
  --checkpoint-every 2000 \
  --log-every 50 \
  --forward-eval-modes 6 \
  --forward-eval-sampling topp \
  --forward-eval-topp 0.95 \
  --no-forward-export-artifacts
```

### Example: resume run

```bash
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir data/_scenarionet_waymo_training_full_v12 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_womd_full \
  --runtime-preset adv_bmt_runtime_parity \
  --distributed-backend pmap \
  --precision bf16-mixed \
  --batch-size 8 \
  --num-train-scenarios 486992 \
  --max-steps 300000 \
  --resume-checkpoint outputs/counter_bmt_v2_training_womd_full/checkpoints/last.pkl
```

### Example: eval-only artifact export from checkpoint

```bash
CKPT=outputs/counter_bmt_v2_training_womd_full/checkpoints/step_0042000.pkl

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
  --no-resume-strict-determinism \
  --eval-every 1 \
  --eval-batches 20 \
  --checkpoint-every 100000 \
  --log-every 1 \
  --forward-export-artifacts \
  --forward-artifact-max-scenarios 20
```

## 2) Supervised Training (DAG-Latent Path)

Command:
- `python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent`

What it does:
- trains DAG-latent conditioned model with staged schedule A/B/C
- stage B can freeze non-DAG params
- uses DAG source resolver (`dual`, `cache`, `scene_derived`)

Key options:
- all major base training options from `train_nnx_bmt`
- DAG source:
  - `--dag-source-mode {dual,cache,scene_derived}`
  - `--dag-cache-dir`
  - `--dag-cache-strict`
  - `--dag-expected-schema {any,v2_compact10,v3_maneuver_outcome}`
- staged training:
  - `--stage {A,B,C,A_B_C}`
  - `--stage-a-steps`, `--stage-b-steps`, `--stage-c-steps`
  - `--stage-a-dag-dropout`, `--stage-b-dag-dropout`, `--stage-c-dag-dropout`
  - `--stage-b-freeze-non-dag`
  - `--stage-c-decoder-lr-scale`, `--stage-c-dag-lr-scale`
- prescan reuse:
  - `--prescan-cache`
  - `--prescan-cache-source <other_run/manifests/prescan_cache.pkl>`

### Example: quick staged smoke

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/dag_latent_smoke \
  --model-preset midgpt_dag_latent \
  --stage A_B_C \
  --stage-a-steps 40 \
  --stage-b-steps 40 \
  --stage-c-steps 40 \
  --batch-size 2
```

### Example: strict cache-only stage B

```bash
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/dag_latent_stage_b_cache \
  --model-preset midgpt_dag_latent \
  --stage B \
  --stage-b-steps 2000 \
  --dag-source-mode cache \
  --dag-cache-dir outputs/dag_cache_v3_mo/cache \
  --dag-cache-strict \
  --dag-expected-schema v3_maneuver_outcome \
  --batch-size 8
```

## 3) DAG Cache Build and Validation

### 3.1 Build v2 DAG cache from ScenarioNet (PromptBN + VLM frames)

Command:
- `python src/scripts/dag_cache/build_dag_cache_v2.py`

What it does:
- renders scenario frames (global-only by default; optional dual ego view)
- runs GPT-4o perception + PromptBN DAG build
- enforces DAG contract hard mode (`maneuver_outcome_v1` by default)
- writes cache JSON + examples + manifest

Key options:
- subset:
  - `--n-scenarios`, `--seed`
  - `--indices-file` or `--start-index/--end-index`
- VLM inputs:
  - `--num-frames`
  - `--frame-renderer {scenarionet,tensor,auto}`
  - `--annotate-vlm-frames`, `--annotation-style`
  - `--dual-view`, `--include-ego-context-text`
  - `--ego-color-hint`
- reliability:
  - `--max-retries`, `--retry-backoff-sec`, `--continue-on-error`
  - `--overwrite`
- contract:
  - `--dag-contract {maneuver_outcome_v1,compact10}`
  - `--dag-contract-mode hard`

Example:

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/dag_cache/build_dag_cache_v2.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --out-dir outputs/dag_cache_v3_mo_smoke \
  --n-scenarios 50 \
  --seed 0 \
  --strict-promptbn \
  --frame-renderer scenarionet \
  --num-frames 6 \
  --no-dual-view \
  --annotate-vlm-frames \
  --include-ego-context-text \
  --dag-contract maneuver_outcome_v1 \
  --dag-contract-mode hard
```

### 3.2 Validate cache contract compliance

Command:
- `python src/scripts/dag_cache/validate_cache_contract.py`

Example:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/validate_cache_contract.py \
  --cache-dir outputs/dag_cache_v3_mo_smoke/cache \
  --dag-contract maneuver_outcome_v1
```

### 3.3 Inspect generated DAG examples

Command:
- `python src/scripts/dag_cache/inspect_dag_examples.py`

Example:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/inspect_dag_examples.py \
  --cache-dir outputs/dag_cache_v3_mo_smoke/cache \
  --examples-dir outputs/dag_cache_v3_mo_smoke/examples \
  --n 5 \
  --seed 0 \
  --output-md outputs/dag_cache_v3_mo_smoke/inspect.md
```

### 3.4 Import legacy DAG JSONs into contract-aligned cache

Command:
- `python src/scripts/dag_cache/import_legacy_dag_json.py`

Example:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/import_legacy_dag_json.py \
  --legacy-root outputs/legacy_pipeline_runs \
  --out-dir data/dag_cache_from_legacy \
  --dag-contract maneuver_outcome_v1
```

## 4) Parity Suite

### 4.1 One-command parity run

Wrapper:
- `bash tools/run_parity_suite.sh`

Default behavior:
- profile: `quick`
- legacy policy: `required_if_available`
- P5 policy: `pass_with_waiver`

Examples:

```bash
bash tools/run_parity_suite.sh
```

```bash
bash tools/run_parity_suite.sh --profile full --p5-policy strict_fail
```

```bash
bash tools/run_parity_suite.sh --legacy-policy required --legacy-root /tmp/missing_legacy
```

### 4.2 Direct parity harness

Command:
- `python src/scripts/parity/parity_report.py`

Important flags:
- `--profile {quick,full,remote}`
- `--legacy-policy {required_if_available,required,optional}`
- `--p5-policy {pass_with_waiver,strict_fail,skip_overall}`
- `--forward-artifact-dir` to reuse existing P4 artifacts
- `--stop-on-fail`

Example:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/parity/parity_report.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/parity_report \
  --profile quick
```

## 5) Head-to-Head Model Comparison

Command:
- `python src/scripts/eval/compare_models_head2head.py --registry <yaml>`

What it does:
- evaluates multiple models on one deterministic scenario subset
- computes aggregate + pairwise metrics
- writes overlays and replay exports

Example:

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/eval/compare_models_head2head.py \
  --registry configs/eval/model_registry.example.yaml
```

Use `--reuse-artifacts` on reruns for speed:

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/eval/compare_models_head2head.py \
  --registry configs/eval/model_registry.example.yaml \
  --reuse-artifacts
```

See also: `docs/head2head_eval.md`.

## 6) Replay Export from Forward Artifacts

### 6.1 Single artifact -> replay package

Command:
- `python src/scripts/replay/export_forward_artifact_to_scenario.py`

Example:

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

Replay:

```bash
python -m scenarionet.sim -d outputs/replay_from_forward_artifact --render 2D
```

### 6.2 Batch export from head-to-head artifacts

Command:
- `python src/scripts/replay/export_forward_artifacts_batch.py`

Example:

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/replay/export_forward_artifacts_batch.py \
  --artifacts-root outputs/head2head_eval_example/artifacts \
  --dataset-dir data/scenarionet_waymo_training_500 \
  --scenario-subset-file outputs/head2head_eval_example/scenario_subset.json \
  --output-dir outputs/head2head_eval_example/replay \
  --max-scenarios 8
```

## 7) Legacy Adv-BMT Evaluation (Paper Protocol)

Command:
- `python src/scripts/eval/run_legacy_forward_paper_eval.py`

What it does:
- runs legacy evaluator stack for forward metrics
- useful for paper-protocol comparisons

Example:

```bash
PYTHONPATH=src .venv-legacy-paper/bin/python src/scripts/eval/run_legacy_forward_paper_eval.py \
  --legacy-root src/Adv-BMT \
  --checkpoint bmt/ckpt/last.ckpt \
  --dataset-dir data/scenarionet_waymo_training_500 \
  --output-prefix outputs/legacy_eval/legacy_paper_eval \
  --limit-test-batches 500 \
  --num-modes 6 \
  --sampling-method topp \
  --temperature 1.0 \
  --topp 0.95
```

## 8) RL Behavior-Manifold Training

Command:
- `python -m counter_bmt_v2.cli.train_rl_topo_mcpo`

What it does:
- runs RL loop with manifold embedding (`risk_vector`, `dag_gnn`, `topology_zpi`, `hybrid`)
- supports novelty, consensus, thermostat controls

Example:

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_dag \
  --embedding-mode dag_gnn \
  --group-size 8 \
  --steps 200
```

## 9) Vertical-Slice Pipeline Runner

Command:
- `python -m counter_bmt_v2.cli.run_pipeline`

What it does:
- executes perception -> DAG -> intervention -> rollout pipeline for one scenario
- supports demo mode and ScenarioNet mode

Example (demo mode):

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.run_pipeline \
  --scene-source demo \
  --scenario-id demo_001 \
  --n-samples 3 \
  --json-out outputs/pipeline_demo.json
```

Example (ScenarioNet mode):

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.run_pipeline \
  --scene-source scenarionet \
  --data-dir data/scenarionet_waymo_training_500 \
  --scenario-index 0 \
  --perception-backend gpt4o \
  --dag-backend promptbn \
  --json-out outputs/pipeline_scenarionet.json
```

## 10) TensorBoard

Training writes TensorBoard events by default (unless `--no-tensorboard`).

Run viewer:

```bash
tensorboard --logdir outputs/<run>/tensorboard --port 6006 --host 0.0.0.0
```

If TensorBoard dependency versions are problematic in your main env, run it in an isolated `uvx` env:

```bash
uvx --python 3.10 \
  --with "tensorboard==2.20.0" \
  --with "setuptools>=68,<81" \
  --with "six>=1.16.0" \
  tensorboard \
  --logdir outputs/<run>/tensorboard \
  --port 6006 \
  --host 0.0.0.0
```

See also: `docs/training_tensorboard.md`.

## 11) Most Common End-to-End Workflows

### A) Base model full training
1. `train_nnx_bmt` with explicit train/val dirs.
2. Monitor TensorBoard + `metrics.jsonl`.
3. Export forward artifacts from best checkpoint if needed.

### B) DAG-latent Stage B/C with strict cache
1. Build cache via `build_dag_cache_v2.py` (contract hard mode).
2. Validate with `validate_cache_contract.py`.
3. Train with `train_nnx_bmt_dag_latent --dag-source-mode cache --dag-cache-strict`.

### C) Compare multiple models
1. Prepare registry YAML.
2. Run `compare_models_head2head.py`.
3. Inspect `report.md`, `metrics/*.csv`, `viz/*.png`, and replay exports.

### D) Run full parity check before large run
1. `bash tools/run_parity_suite.sh` (quick).
2. Use `--profile full` for stricter gate coverage.
