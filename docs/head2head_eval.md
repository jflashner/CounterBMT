# Head-to-Head Multi-Model Evaluation

## What It Does
`compare_models_head2head.py` compares multiple trajectory models (v2 + optional legacy Adv-BMT) on the same deterministic scenario subset, computes shared forward rollout metrics, and writes:
- per-scenario / aggregate / pairwise / ranking CSVs
- overlay trajectory plots (GT + model rollouts)
- replay exports for selected scenarios
- machine-readable `report.json` and summary `report.md`

## Main Command
```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/eval/compare_models_head2head.py \
  --registry configs/eval/model_registry.example.yaml
```

## Registry
Use `configs/eval/model_registry.example.yaml` as template.

Key run fields:
- `run.dataset_dir`
- `run.n_scenarios` + `run.seed` (deterministic subset)
- `run.metrics.mode`: `approx` or `strict_if_available`
- `run.legacy_policy`: `required_if_available`, `required`, `optional`
- `run.reuse_artifacts`
- `run.max_parallel_models`: optional parallel v2 model execution

Each model entry:
- `id`, `backend`, `checkpoint`, `runtime`
- v2 runtime: `model_preset`, `tokenizer_mode`, `skip_steps`, sampling args
- legacy runtime: `python_bin`, `legacy_root`, `skip_steps`, sampling args

## Output Layout
`outputs/<run>/...`
- `scenario_subset.json`
- `report.json`, `report.md`
- `artifacts/<model_id>/step_eval/*.npz`
- `metrics/per_scenario.csv`
- `metrics/aggregate.csv`
- `metrics/pairwise_deltas.csv`
- `metrics/rankings.csv`
- `viz/overlay_<scenario_id>.png`
- `replay/<model_id>/<scenario_id>/...`

## Replay Export (Batch Utility)
```bash
PYTHONPATH=src .venv-v2/bin/python src/scripts/replay/export_forward_artifacts_batch.py \
  --artifacts-root outputs/head2head_eval_example/artifacts \
  --dataset-dir data/scenarionet_waymo_training_500 \
  --scenario-subset-file outputs/head2head_eval_example/scenario_subset.json \
  --output-dir outputs/head2head_eval_example/replay \
  --max-scenarios 8
```

## Notes
- Legacy backend runs in subprocess to avoid dependency conflicts.
- If legacy env is unavailable and policy is `required_if_available`, legacy models are skipped with reason.
- Artifact reuse is hash-based on model spec + subset; reruns are much faster when unchanged.
