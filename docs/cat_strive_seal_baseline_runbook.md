# Adv-BMT 500 Baseline Plan

Status: pivoted away from regenerating CAT / STRIVE / SEAL. We will use the Adv-BMT paper's reported CAT, STRIVE, SEAL, and Adv-BMT baseline numbers, and spend compute only on evaluating CounterDrive on the same original Adv-BMT 500-scene source bank.

## Source Split

Remote ScenarioNet source bank:

```text
/home/grads/jflashner/CounterBMT_run/data/scenarionet_waymo_training_500
```

Verified contents:

- 500 natural ScenarioNet Waymo scenarios.
- `dataset_summary.pkl` and `dataset_mapping.pkl` are present.
- The released baseline scenario folders that were added locally are subsets of this same source bank, so the Adv-BMT baseline table is the right comparison anchor for this split.
- This is a mirrored copy of `/home/grads/jflashner/CounterBMT/data/scenarionet_waymo_training_500`, kept under `CounterBMT_run` so generation and TD3 evaluation paths stay together.

## Reporting Strategy

Use Adv-BMT's paper table for baseline rows:

- CAT
- STRIVE
- SEAL
- Adv-BMT

Then add a new CounterDrive row produced on the same source split. In the paper text, phrase this as an evaluation on the original Adv-BMT 500-scene benchmark rather than the newer Waymax/CounterDrive split.

## CounterDrive Generation

Build a simple ScenarioNet control index:

```bash
cd /home/grads/jflashner/CounterBMT_run
PYTHONPATH=/home/grads/jflashner/CounterBMT_run/metadrive:/home/grads/jflashner/CounterBMT_run/scenarionet:/home/grads/jflashner/CounterBMT_run/src/Adv-BMT \
  /home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/python \
  /home/grads/jflashner/CounterBMT_run/scripts/counterfactual/build_no_intervention_scenarionet_index.py \
  --scenario-root /home/grads/jflashner/CounterBMT_run/data/scenarionet_waymo_training_500 \
  --output-index /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/index/sdc_semantic_control_index_advbmt500_nointervention.jsonl \
  --summary-json /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/index/index_summary.json \
  --source-tag advbmt500
```

Current result:

- 454 usable rows from 500 source scenarios.
- 46 skipped because the helper could not recover a usable SDC ground-truth route.

Generate CounterDrive victim-centric scenario pairs:

```bash
cd /home/grads/jflashner/CounterBMT_run
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/home/grads/jflashner/CounterBMT_run/metadrive:/home/grads/jflashner/CounterBMT_run/scenarionet:/home/grads/jflashner/CounterBMT_run/src:/home/grads/jflashner/CounterBMT_run/src/Adv-BMT \
/home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/python \
  /home/grads/jflashner/CounterBMT_run/scripts/agent_eval/build_victim_centric_table4_dataset.py \
  --control-index /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/index/sdc_semantic_control_index_advbmt500_nointervention.jsonl \
  --scenario-root /home/grads/jflashner/CounterBMT_run/data/scenarionet_waymo_training_500 \
  --ckpt /data/home/grads/jflashner/CounterBMT_run/logs/pr10_1_top500_actualwall_progresssoft_4gpu_h200_run3/lightning_logs/infgen/pr10_1_top500_actualwall_progresssoft_4gpu_h200_2026-04-10/checkpoints/last.ckpt \
  --config /home/grads/jflashner/CounterBMT_run/src/Adv-BMT/cfgs/motion_forward_sdc_semantic_only_strict_local.yaml \
  --outdir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/victim_centric_progresssoft_nostop_cand2 \
  --split-name train \
  --num-scenes 454 \
  --semantic-label left \
  --semantic-label right \
  --semantic-label left_lane_change \
  --semantic-label right_lane_change \
  --max-adversary-candidates 2 \
  --rollout-sampling-method argmax \
  --overwrite
```

Run a 1-scene smoke first by changing `--num-scenes 1` and writing to `victim_centric_progresssoft_nostop_cand2_smoke`.

## TD3 Agent Evaluation

After the train/val scenario banks exist, build TD3 views:

```bash
cd /home/grads/jflashner/CounterBMT_run
PYTHONPATH=/home/grads/jflashner/CounterBMT_run/src:/home/grads/jflashner/CounterBMT_run/src/Adv-BMT \
  /home/grads/jflashner/CounterBMT/.venv-legacy-adv-bmt/bin/python \
  /home/grads/jflashner/CounterBMT_run/scripts/agent_eval/prepare_td3_table4_views.py \
  --train-natural-dir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/victim_centric_progresssoft_nostop_cand2/natural_scenarios \
  --train-adversarial-dir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/victim_centric_progresssoft_nostop_cand2/adversarial_scenarios \
  --val-natural-dir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/victim_centric_progresssoft_nostop_cand2/natural_scenarios \
  --val-adversarial-dir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/victim_centric_progresssoft_nostop_cand2/adversarial_scenarios \
  --outdir /data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/td3_views_progresssoft_nostop_cand2 \
  --target-train-pairs 400 \
  --target-val-pairs 39 \
  --shuffle-scenes \
  --selection-seed 0 \
  --disjoint-val-from-train \
  --link-mode symlink
```

Then launch TD3 rows by pointing the existing launcher at those views:

```bash
VIEWS_ROOT=/data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/td3_views_progresssoft_nostop_cand2 \
ROW=counterbmt EVAL_SPLIT=natural \
/data/home/grads/jflashner/CounterBMT_run/scripts/remote/run_td3_table4_train500_zh2.sh 0

VIEWS_ROOT=/data/home/grads/jflashner/CounterBMT_run/eval_runs/counterdrive_advbmt500_20260424/td3_views_progresssoft_nostop_cand2 \
ROW=counterbmt EVAL_SPLIT=adversarial \
/data/home/grads/jflashner/CounterBMT_run/scripts/remote/run_td3_table4_train500_zh2.sh 0
```

The baseline numbers for CAT, STRIVE, SEAL, and Adv-BMT should be copied from the Adv-BMT paper table rather than recomputed locally.
