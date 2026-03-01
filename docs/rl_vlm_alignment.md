# RL VLM Alignment (`vlm_replace`)

This mode replaces RL alignment reward with GPT-4o DAG-conformance scoring.

## What Changes
- Default mode (`judge`) is unchanged.
- In `vlm_replace` mode, RL uses VLM conformance score as `JudgeResult.reward`.
- `w_alignment` still controls alignment weight in reward composition.
- Safety/realism/novelty/consensus terms stay unchanged.

## Cost Controls (Default Conservative)
- `sample_rate=0.15`
- `every_n_steps=5`
- `max_calls_per_step=2`
- per-call timeout and step budget are bounded.

Unsampled rollouts use **step-mean fill**. If no rollout is scored in a step, neutral score is used.

## New Metrics
Logged in `metrics.jsonl`:
- `alignment/vlm_mean`
- `alignment/vlm_scored_fraction`
- `alignment/judge_original_mean`
- `alignment/vlm_calls_attempted`
- `alignment/vlm_calls_success`
- `alignment/vlm_cache_hits`
- `alignment/vlm_timeouts`
- `alignment/vlm_errors`
- `alignment/vlm_latency_ms_mean`
- `alignment/source_mode_vlm_replace`

## Commands

Mock smoke:

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_vlm_replace_smoke_mock \
  --steps 20 --log-every 5 \
  --alignment-source-mode vlm_replace \
  --vlm-alignment-enabled \
  --vlm-alignment-backend mock
```

Low-cost live GPT-4o:

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_rl_topo_mcpo \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/rl_vlm_replace_smoke_live \
  --steps 10 --log-every 1 \
  --alignment-source-mode vlm_replace \
  --vlm-alignment-enabled \
  --vlm-alignment-backend gpt4o \
  --vlm-alignment-sample-rate 0.15 \
  --vlm-alignment-every-n-steps 5 \
  --vlm-alignment-max-calls-per-step 2
```

## Evidence + Cache
- Evidence (if enabled): `output_dir/vlm_alignment_evidence/step_x/scenario_y/rollout_z`
- Cache: `--vlm-alignment-cache-dir` (default `outputs/rl_vlm_alignment_cache`)

## Failure Behavior
- API parse/timeouts/errors do not crash training.
- Failed scores fall back to neutral/step-mean policy.
