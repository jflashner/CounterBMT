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

## Counterfactual Utilities

### `scripts/counterfactual/render_sdc_semantic_animation_examples.py`

What it does:
- renders multi-agent GIFs for SDC semantic-control validation rows
- now also supports arbitrary-agent semantic probe cases through `--non-sdc-cases-json`

Key modes:
- SDC validation index mode:
  - `--control-index`
  - `--data-dir`
  - `--num-scenes`
- non-SDC arbitrary-agent mode:
  - `--non-sdc-cases-json`
  - each case supplies:
    - `scenario_pkl`
    - `agent_id`
    - `semantic_label`
    - optional `start_step`, `end_step`, `semantic_confidence`, `case_name`

Non-SDC mode notes:
- now uses the same evaluation-style `preprocess_GPTmodel(...)` + `GPT_AR(..., teacher_forcing=False)` stack as the SDC semantic GIF path
- default control window is full horizon when `end_step` is omitted or negative

Outputs:
- SDC mode:
  - one ground-truth GIF plus one predicted GIF per slot
- non-SDC mode:
  - one reference GIF
  - one baseline-rollout GIF
  - one controlled-rollout GIF
  - per-case manifest entry

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
  - `--runtime-preset {none,adv_bmt_runtime_parity,legacy_midgpt_recipe}`
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

### Example: closest legacy MidGPT recipe on 4 GPUs

Notes:
- `legacy_midgpt_recipe` locks the missing legacy recipe knobs that matter most:
  - `midgpt_parity`
  - `adv_bmt_parity`
  - `legacy_cosine_zero`
  - `epochs=30`
  - `mode=forward`
  - `reverse_prob=0.0`
- Legacy Lightning DDP used `batch_size: 10` per process. v2 `pmap` uses a global batch size, so the closest 4-GPU match is `--batch-size 40`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src .venv-v2/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --train-data-dir data/_scenarionet_waymo_training_full_v12 \
  --val-data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_midgpt_legacy_recipe \
  --runtime-preset legacy_midgpt_recipe \
  --distributed-backend pmap \
  --precision bf16-mixed \
  --batch-size 40 \
  --num-train-scenarios 486992 \
  --num-val-scenarios 500 \
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

### 6.3 Create GIFs for convenient visualization

Command:
- `python src/scripts/replay/make_scenario_gif.py`

What it does:
- creates GIFs either from existing frame folders or by rendering scenarios directly
- useful for quick qualitative inspection without launching interactive replay

Example A: make GIF from existing frame folder (fast)

```bash
PYTHONPATH=src .venv/bin/python src/scripts/replay/make_scenario_gif.py \
  --frames-dir outputs/dag_cache_single_test_v2prompt_fix/examples/2ff20c0841a51211/frames_vlm \
  --frames-glob "global_t*.png" \
  --output-dir outputs/gif_exports \
  --output-name 2ff20c0841a51211_global.gif \
  --fps 3
```

Example B: render scenarios from ScenarioNet dataset and export GIFs

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/replay/make_scenario_gif.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/scenario_gifs \
  --scenario-indexes 0,1,2 \
  --num-frames 16 \
  --fps 4 \
  --continue-on-error
```

Example C: render by scenario ID directly

```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/replay/make_scenario_gif.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/scenario_gifs_by_id \
  --scenario-id 10af3d70d93ef629 \
  --num-frames 12 \
  --fps 4
```

Multiple IDs are supported via `--scenario-ids a,b,c` or `--scenario-ids-file`.

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
- runs Topo-MCPO-style RL with novelty, consensus, entropy thermostat, and reward logging
- supports the real checkpoint-backed NNX policy path (`--policy-backend nnx_checkpoint`) plus the older scaffold backend for compatibility
- samples full DAG assignments for the NNX path and conditions rollouts on the sampled DAG tensorization
- supports `judge` or `vlm_replace` alignment sources

Key options:
- policy backend:
  - `--policy-backend {nnx_checkpoint,scaffold}`
  - `--policy-checkpoint`
  - `--policy-model-preset`
  - `--policy-tokenizer-mode`
  - `--policy-skip-steps`
- DAG source:
  - `--dag-source-mode {dual,cache,scene_derived}`
  - `--dag-cache-dir`
  - `--dag-cache-strict`
  - `--dag-expected-schema {any,v2_compact10,v3_maneuver_outcome}`
- optimization:
  - `--clip-eps`
  - `--kl-beta`
  - `--policy-lr`
  - `--trainable-scope {decoder_dag,all}`
  - `--ppo-epochs`
- sampling:
  - `--candidate-multiplier`
  - `--enable-feasibility-mask`
  - `--feasible-max-speed-mps`
  - `--feasible-max-accel-delta`
  - `--feasible-max-yaw-delta`
  - `--store-rollout-traces`
- alignment:
  - `--alignment-source-mode {judge,vlm_replace}`
  - `--vlm-alignment-enabled`
  - `--vlm-alignment-backend {mock,gpt4o}`

Example: checkpoint-backed mock smoke

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_nnx_smoke \
  --steps 5 \
  --log-every 1 \
  --group-size 4 \
  --embedding-mode risk_vector \
  --policy-backend nnx_checkpoint \
  --policy-checkpoint outputs/dag_latent_stage_c/checkpoints/last.pkl \
  --policy-model-preset midgpt_dag_latent \
  --policy-tokenizer-mode paper_simple \
  --policy-skip-steps 1 \
  --dag-source-mode scene_derived
```

Example: checkpoint-backed run with DAG cache + `vlm_replace`

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_topo_mcpo_vlm \
  --steps 200 \
  --group-size 8 \
  --embedding-mode risk_vector \
  --policy-backend nnx_checkpoint \
  --policy-checkpoint outputs/dag_latent_stage_c/checkpoints/last.pkl \
  --policy-model-preset midgpt_dag_latent \
  --policy-tokenizer-mode paper_simple \
  --policy-skip-steps 1 \
  --dag-source-mode cache \
  --dag-cache-dir outputs/dag_cache_v3_mo/cache \
  --dag-cache-strict \
  --dag-expected-schema v3_maneuver_outcome \
  --alignment-source-mode vlm_replace \
  --vlm-alignment-enabled \
  --vlm-alignment-backend gpt4o
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

## 11) Legacy Counterfactual Probe Utilities

These utilities are useful when we want to inspect or debug the legacy
`src/Adv-BMT` counterfactual stack without launching a full training run.

### 11.1 Arbitrary-Agent Semantic Rollout Probe

Command:
- `python scripts/counterfactual/probe_agent_semantic_rollout.py`

What it does:
- loads a single raw scenario pickle
- preprocesses it into the legacy semantic-only model format
- selects an arbitrary modeled agent via `--agent-id`
- applies a semantic label to that agent with a controlled time window
- runs both:
  - a baseline rollout
  - a controlled rollout
- exports a small debug bundle:
  - `summary.json`
  - `trajectories.npz`
  - `overlay.png`
  - `victim_centric_overlay.png`
- can also:
  - default the victim to the scene SDC
  - auto-select a likely victim agent when `--victim-agent-id auto`
  - export a replayable victim-centric ground-truth/counterfactual scenario pair
  - create a MetaDrive/ScenarioNet replay script for that pair
  - dump a debug trace with:
    - preprocessed `cf/*` control tensors
    - runtime control-kind / control-available signals
    - first-step top logits for the targeted agent
    - optional SDC-vs-non-SDC comparison on the same scene

This is the safest first tool for victim-centric work because it lets us test
"can we intervene on a non-SDC actor with a semantic label?" before we build
the full scenario exporter.

Example:

```bash
PYTHONPATH=src/Adv-BMT .venv-mac/bin/python \
  scripts/counterfactual/probe_agent_semantic_rollout.py \
  --scenario-pkl outputs/pr10_1_sdc_semantic_top859_full/scenario_root/sd_waymo_v1.3.1_waymax_scene_00618.pkl \
  --ckpt /data/home/grads/jflashner/CounterBMT_run/logs/pr10_1_top500_actualwall_progresssoft_4gpu_h200_run3/lightning_logs/infgen/pr10_1_top500_actualwall_progresssoft_4gpu_h200_2026-04-10/checkpoints/last.ckpt \
  --config src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_progresssoft_topomcpo_dag_trafficcap.yaml \
  --agent-id 2948 \
  --semantic-label left \
  --semantic-confidence 1.0 \
  --start-step 0 \
  --end-step 15 \
  --rollout-sampling-method argmax \
  --outdir outputs/debug_probe_agent_semantic_rollout_scene618_agent2948_left
```

Victim-centric export example:

```bash
PYTHONPATH=src/Adv-BMT .venv-mac/bin/python \
  scripts/counterfactual/probe_agent_semantic_rollout.py \
  --scenario-pkl outputs/pr10_1_sdc_semantic_top859_full/scenario_root/sd_waymo_v1.3.1_waymax_scene_00618.pkl \
  --ckpt outputs/remote_checkpoints/pr10_1_semantic_only_sdc_top859_overnight_1gpu_run2/last.ckpt \
  --config src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_progresssoft_topomcpo_dag_trafficcap.yaml \
  --agent-id 1430 \
  --victim-agent-id auto \
  --semantic-label left \
  --semantic-confidence 1.0 \
  --start-step 0 \
  --end-step 15 \
  --rollout-sampling-method argmax \
  --export-victim-centric \
  --outdir outputs/debug_probe_agent_semantic_rollout_victim_export
```

Debug-trace example:

```bash
PYTHONPATH=src/Adv-BMT .venv-mac/bin/python \
  scripts/counterfactual/probe_agent_semantic_rollout.py \
  --scenario-pkl outputs/pr10_1_sdc_semantic_top859_full/scenario_root/sd_waymo_v1.3.1_waymax_scene_00057.pkl \
  --ckpt /data/home/grads/jflashner/CounterBMT_run/logs/pr10_1_top500_actualwall_progresssoft_4gpu_h200_run3/lightning_logs/infgen/pr10_1_top500_actualwall_progresssoft_4gpu_h200_2026-04-10/checkpoints/last.ckpt \
  --config src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml \
  --agent-id 1205 \
  --semantic-label left \
  --compare-label stop \
  --compare-label right \
  --start-step 0 \
  --end-step -1 \
  --rollout-sampling-method argmax \
  --debug-trace \
  --debug-compare-sdc \
  --outdir outputs/debug_probe_agent_semantic_rollout_trace_scene57
```

Useful outputs:
- `summary.json`
- `debug_trace.json`

### `scripts/agent_eval/build_victim_centric_table4_dataset.py`

What it does:
- builds the first offline victim-centric train/val dataset for Table 4 style RL
- uses the corrected arbitrary-agent semantic rollout path from
  `probe_agent_semantic_rollout.py`
- keeps the SDC as victim/ego by default
- chooses one non-SDC adversary intervention per base scene
- exports paired:
  - natural scenarios
  - adversarial victim-centric scenarios
- creates MetaDrive-compatible dataset summaries and JSON manifests

Core inputs:
- `--control-index`
- `--scenario-root`
- `--ckpt`
- `--config`
- `--outdir`

Important controls:
- `--num-scenes`
- `--scene-offset`
- repeated `--semantic-label`
- `--max-adversary-candidates`
- `--min-moving-speed-mps`
- `--max-distance-to-sdc-m`
- `--min-final-position-delta-m`
- `--min-changed-action-steps`

Notes:
- this is the right next step once non-SDC interventions are working
- it is intended to produce the offline augmented scenario bank before TD3 training
- the exporter functions themselves do not run rollout logic; they serialize the
  adversary trajectory generated by the corrected probe/eval stack

### `scripts/agent_eval/prepare_td3_table4_views.py`

What it does:
- builds TD3-ready ScenarioNet directory views from the offline natural and
  adversarial victim-centric banks
- matches scenarios by source `waymax_scene_*` id
- creates:
  - `train_waymo_only`
  - `train_counterbmt_mixed`
  - `eval_waymo_only`
  - `eval_counterbmt_adversarial`
- writes a manifest with the selected scene ids and the suggested TD3 dataset
  paths

Core inputs:
- `--train-natural-dir`
- `--train-adversarial-dir`
- `--val-natural-dir`
- `--val-adversarial-dir`
- `--outdir`

Important controls:
- `--target-train-pairs`
- `--target-val-pairs`
- `--shuffle-scenes`
- `--selection-seed`
- `--link-mode`

Notes:
- this is the clean bridge from the offline victim-centric banks into
  `train_td3.py`
- when fewer paired exports are available than requested, it clips to the
  available pair count deterministically

### `scripts/agent_eval/migrate_victim_centric_bank.py`

What it does:
- migrates an already-exported victim-centric natural/adversarial bank into
  MetaDrive-compatible ScenarioNet schema
- repairs older banks that were written before the ScenarioNet schema fix
- rewrites:
  - top-level `version`
  - map feature polygon/point fields
  - fresh `dataset_summary.pkl`
  - fresh `dataset_mapping.pkl`
- writes a `migration_report.json` with before/after counts

Core inputs:
- `--source-root`
- `--outdir`

Useful optional controls:
- `--max-scenarios-per-dir`
- `--copy-scene-analysis`
- `--overwrite`

Notes:
- this is the fastest repair path when intervention generation was already
  correct and only the ScenarioNet serialization was wrong
- use this before rebuilding TD3 views if the original bank predates the schema
  normalization fix

Example:

```bash
PYTHONPATH=src python scripts/agent_eval/migrate_victim_centric_bank.py \
  --source-root /data/home/grads/jflashner/CounterBMT_run/eval_runs/victim_centric_table4_train500_fullindex_zh2_20260418 \
  --outdir /data/home/grads/jflashner/CounterBMT_run/eval_runs/victim_centric_table4_train500_migrated_zh2_20260419 \
  --overwrite
```

### `scripts/agent_eval/watch_remote_build_progress.py`

What it does:
- polls remote build directories or TD3 run directories over SSH and renders a
  live in-place terminal view locally
- shows:
  - for build roots:
    - completed scenes
    - skip count
    - natural/adversarial scenario counts
    - `screen` liveness
    - active builder process count
  - for TD3 roots:
    - latest `total_timesteps`
    - `ep_rew_mean`
    - `ep_len_mean`
    - `fps`
    - checkpoint count and latest checkpoint step
    - `screen` liveness
    - active TD3 process count

Useful optional controls:
- `--interval`
- `--once`
- repeated `--run`

Run spec format:
- `label|host|screen_name|remote_root|target_total`

Notes:
- if no `--run` is provided, it defaults to the current paper-faithful
  Adv-BMT TD3 run on `zhoulab-1`
- this is meant to be run locally from your terminal, not on the remote host

Examples:

```bash
python scripts/agent_eval/watch_remote_build_progress.py
```

```bash
python scripts/agent_eval/watch_remote_build_progress.py --once
```

```bash
python scripts/agent_eval/watch_remote_build_progress.py \
  --run 'advbmt-td3|zhoulab-1.cs.vt.edu|td3_advbmt_paperfaithful_seed0|/data/home/grads/jflashner/CounterBMT_run/logs/td3_table4_runs/td3_table4_advbmt_paperfaithful_train476_eval_natural_seed0|1000000'
```

```bash
python scripts/agent_eval/watch_remote_build_progress.py \
  --run 'train|zhoulab-1.cs.vt.edu|advbmt_train476_20260419|/data/home/grads/jflashner/CounterBMT_run/eval_runs/advbmt_paperfaithful_train476_zh1_20260419|476' \
  --run 'val|zhoulab-1.cs.vt.edu|advbmt_val95_20260419|/data/home/grads/jflashner/CounterBMT_run/eval_runs/advbmt_paperfaithful_val95_zh1_20260419|95'
```

### `scripts/remote/run_td3_table4_openloop.sh`

What it does:
- launches the legacy open-loop TD3 trainer on a TD3-ready ScenarioNet dataset
- uses `train_td3.py` without the closed-loop online generators

Required environment variables:
- `DATA_DIR`
- `EVAL_DATA_DIR`

Useful optional environment variables:
- `REPO_ROOT`
- `PYTHON_BIN`
- `SAVE_ROOT`
- `EXP_NAME`
- `TRAINING_STEPS`
- `EVAL_FREQ`
- `EVAL_EP`
- `WANDB_PROJECT`
- `WANDB_TEAM`

Example:

```bash
DATA_DIR=/path/to/train_counterbmt_mixed \
EVAL_DATA_DIR=/path/to/eval_waymo_only \
EXP_NAME=td3_counterbmt_victim \
scripts/remote/run_td3_table4_openloop.sh 0
```

### `scripts/remote/run_td3_table4_train500_zh2.sh`

What it does:
- launches the concrete Table 4-style open-loop TD3 rows against the prepared
  TD3 view bank on the shared remote tree
- wraps `run_td3_table4_openloop.sh` so we only choose:
  - `ROW=waymo` or `ROW=counterbmt`
  - `EVAL_SPLIT=natural` or `EVAL_SPLIT=adversarial`

Defaults:
- `ROW=waymo`
- `EVAL_SPLIT=natural`

Examples:

```bash
ROW=waymo EVAL_SPLIT=natural scripts/remote/run_td3_table4_train500_zh2.sh 0
ROW=counterbmt EVAL_SPLIT=natural scripts/remote/run_td3_table4_train500_zh2.sh 0
ROW=counterbmt EVAL_SPLIT=adversarial scripts/remote/run_td3_table4_train500_zh2.sh 0
```

### `scripts/remote/run_pr10_progresssoft_topomcpo_dag_trafficcap_progresson_zh2.sh`

What it does:
- launches the progresssoft warm-start continuation on `zhoulab-2`
- keeps the loose traffic-speed cap
- keeps stall penalty disabled
- restores the old positive tube progress reward under semantic-ext through
  `LOCAL_CONTROL_SDC_SEMANTIC_EXT_ALLOW_PROGRESS_REWARD=true`

Primary preset:
- `src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_progresssoft_topomcpo_dag_trafficcap_progresson.yaml`

Use this when:
- the plain semantic-ext traffic-cap run starts collapsing to near-zero speed
- you want the old progresssoft forward incentive back without removing the cap

## 12) Most Common End-to-End Workflows

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
