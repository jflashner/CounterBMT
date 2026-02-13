# CounterBMT v2 Day 2 Training Notes

This note documents the Day 2 training implementation for the NNX trajectory model rewrite.

## 1) New files

1. `src/counter_bmt_v2/training/supervised.py`
- New supervised training loop for motion-token learning.
- Uses Optax with:
  - AdamW
  - warmup + cosine decay schedule
  - global gradient clipping
- Includes:
  - forward/reverse/mixed supervision modes
  - token target creation from ScenarioNet kinematics
  - masked CE loss + token diagnostics
  - periodic eval
  - checkpoint save/load

2. `src/counter_bmt_v2/training/__init__.py`
- Exposes:
  - `ForwardPassEvalConfig`
  - `SupervisedTrainConfig`
  - `train_supervised`

3. `src/counter_bmt_v2/cli/train_nnx_bmt.py`
- CLI wrapper around `train_supervised`.

4. `src/counter_bmt_v2/training/forward_metrics.py`
- Adv-BMT-style forward-pass evaluator integrated into validation.
- Computes scenario-level metrics from sampled model rollouts.

## 2) Updated files

1. `src/counter_bmt_v2/data/scenarionet.py`
- `collate_nnx_scene_samples(...)` now supports fixed collate shapes:
  - `max_time_steps`
  - `max_agents`
  - `max_map_features`
  - `max_vectors_per_map_feature`
  - `max_traffic_lights`
- This is used by training to keep tensor shapes stable across batches.

2. `src/counter_bmt_v2/__init__.py`
- Exports training API at package level.

3. `src/counter_bmt_v2/ROADMAP.md`
- Added Day 2 progress section.

## 3) Supervision construction details

Paper intent reference:
- Adv-BMT trains on discrete motion-token predictions with CE objective.
- Day 2 follows this objective and uses bidirectional mode control.

Current pipeline:
1. Take collated agent velocity + heading + validity masks.
2. Downsample by `skip_steps` (default 5 for ~0.5s chunks on 10Hz data).
3. Compute per-transition:
- acceleration from speed delta
- yaw rate from wrapped heading delta
4. Quantize to token IDs in the 33x33 action space.
5. Build teacher-forcing sequence:
- `targets`: current token
- `prev_token_ids`: shifted targets with start token prepended
6. Build `continuous_motion` from previous tokens (token->(acc,yaw)).
7. For mixed mode, reverse selected samples and set `reverse_indicator`.

## 4) Metrics logged

The trainer logs token-level metrics that track the Adv-BMT-style CE setup:
- `total_loss` (masked cross-entropy)
- `accuracy`
- `entropy`
- `perplexity`, `gt_perplexity`
- `cluster_use`, `gt_cluster_use`
- `rate_default_gt`, `rate_default_pred`
- `num_trained_tokens`
- directional split metrics:
  - `accuracy_in_backward`, `accuracy_in_forward`
  - `loss_in_backward`, `loss_in_forward`
  - `entropy_in_backward`, `entropy_in_forward`
  - `backward_ratio`

Forward-pass scenario metrics are logged with `forward/` prefix:
- supervised fit: `forward/sfde_min`, `forward/sfde_avg`, `forward/sade_min`, `forward/sade_avg`, `forward/ssde_min`, `forward/ssde_avg`
- diversity: `forward/fdd`, `forward/add`, `forward/sdd`
- realism: `forward/vel_jsd`, `forward/acc_jsd`, `forward/ttc_jsd`
- safety/comfort: `forward/veh_coll_*`, `forward/coll_vel_*`, `forward/sdc_acc_*`, `forward/sdc_jerk_*`
- bookkeeping: `forward/scenario_count`

## 5) Checkpoint format

Saved under:
- `<output_dir>/checkpoints/step_XXXXXXX.pkl`
- `<output_dir>/checkpoints/last.pkl`

Payload includes:
- model state (`nnx.state(model)`)
- optimizer state + step
- train step
- serialized train/model configs
- latest metrics snapshot

## 6) Run examples

### Basic run
```bash
.venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training \
  --model-preset paper_like_small \
  --epochs 3 \
  --batch-size 4
```

### Mixed forward/backward mode
```bash
.venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_mixed \
  --mode mixed \
  --reverse-prob 0.5
```

### Forward-pass metric controls
```bash
.venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_forward_eval \
  --forward-eval-modes 6 \
  --forward-eval-sampling topp \
  --forward-eval-topp 0.95
```

Disable forward-pass eval when you only want token CE stats:
```bash
.venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_no_forward_eval \
  --no-forward-eval
```

### Resume from checkpoint
```bash
.venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/counter_bmt_v2_training_resume \
  --resume-checkpoint outputs/counter_bmt_v2_training/checkpoints/last.pkl
```

## 7) Current known limitations

1. Reverse supervision currently uses sequence reversal of tokenized transitions as a practical Day 2 approximation. A stricter backward-tokenization parity pass can be added in Day 3.
2. Forward-pass collision/TTC metrics are dependency-light approximations (NumPy/JAX) rather than the exact Waymo metric operators used in original Adv-BMT eval.
3. Forward-pass rollout currently uses a fixed-horizon iterative decoder path for stable compute; it is designed for practical training-time validation, not a full benchmark replacement script.
