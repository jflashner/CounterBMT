# Compact DAG Contract v2 (`compact10`)

This document defines the strict DAG cache contract used by DAG-latent stage B/C training.

## Cache Schema

- `schema_version`: `counter_bmt_v2_dag_cache_v2_compact10`
- Required metadata keys:
  - `metadata.contract_name` (`compact10`)
  - `metadata.contract_version`
  - `metadata.contract_report` (summary object)

Old v1 cache files are intentionally incompatible and are rejected by default.

## Node Ontology

Allowed node types:
- `context`
- `ego_state`
- `interaction`
- `maneuver`
- `decision`
- `risk`
- `outcome`

Required anchors:
- `ego_initial_speed` (`ego_state`)
- at least one `maneuver_*`
- at least one `decision_*`
- `collision_outcome` (`outcome`)

## Behavior Vocabulary (`compact10`)

Maneuver classes:
- `straight`
- `left_turn`
- `right_turn`
- `lane_change_left`
- `lane_change_right`
- `stop`

Decision classes:
- `maintain_speed`
- `accelerate`
- `decelerate`
- `yield_or_proceed`

Free-text values are deterministically normalized into this set.

## Structural Caps

- `max_nodes = 14`
- `max_edges = 20`
- `max_parents_per_node = 3`
- `max_outgoing_per_node = 5`
- `max_depth = 4`

## Edge Tier Policy

Tier order:
- Tier 0: `context`, `ego_state`
- Tier 1: `interaction`, `maneuver`, `decision`, `risk`
- Tier 2: `outcome`

Allowed edge directions:
- parent tier `<` child tier
- one explicit same-tier exception: `maneuver -> decision`

## Canonicalization

Contract enforcement canonicalizes before validation:
- deterministic node IDs
- duplicate-edge removal (highest confidence kept)
- confidence clamp to `[0, 1]`
- mechanism normalization to bounded categories
- CPT rewiring after renaming
- CPT row normalization (sum to 1.0)
- uniform CPT fill for missing categorical CPTs

## Hard Enforcement Behavior

Mode used in v2 cache builder: `hard`

- Contract failure => scenario attempt fails
- Scenario retries up to configured retry budget
- If retries exhaust and strict mode is enabled => scenario skipped/fails
- No simple fallback DAG is written in strict cache generation mode

## Build / Validate Commands

Build cache:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/build_dag_cache_v2.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --out-dir outputs/dag_cache_v2_contract \
  --n-scenarios 50 \
  --strict-promptbn \
  --dag-contract compact10 \
  --dag-contract-mode hard
```

Validate cache:

```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/validate_cache_contract.py \
  --cache-dir outputs/dag_cache_v2_contract/cache
```

Train stage B/C using strict cache:

```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/dag_latent_contract_smoke \
  --dag-source-mode cache \
  --dag-cache-dir outputs/dag_cache_v2_contract/cache \
  --dag-cache-strict \
  --stage B \
  --stage-b-steps 20 \
  --batch-size 2
```
