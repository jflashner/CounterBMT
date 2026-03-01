# DAG Contract: `maneuver_outcome_v1`

## Purpose
This contract is a simplified, high-signal DAG format for DAG-latent training.
It keeps only maneuver and outcome semantics so latent capacity is spent on trajectory-relevant structure.

## Schema
- Cache schema version: `counter_bmt_v2_dag_cache_v3_maneuver_outcome`
- Contract metadata required in payload `metadata`:
  - `contract_name = maneuver_outcome_v1`
  - `contract_version`
  - `contract_report` with `passed=true`

## Node Types
Allowed node types:
- `maneuver`
- `outcome`

Any other node type is rejected in hard mode.

## Required Outcome Nodes
Each DAG must contain these anchors:
- `collision_outcome`
- `progress_outcome`
- `compliance_outcome`

## Maneuver Semantics
- Maximum maneuver nodes: 8 (extra nodes are deterministically capped before validation checks).
- Compact maneuver class vocabulary (12):
  - `straight`
  - `left_turn`
  - `right_turn`
  - `lane_change_left`
  - `lane_change_right`
  - `stop`
  - `accelerate`
  - `decelerate`
  - `yield`
  - `merge`
  - `u_turn`
  - `park`

Required interval metadata on each maneuver node (`node.metadata`):
- `start_s`
- `end_s`
- `duration_s`
- `mid_s`

## Edge Policy
Allowed edges only:
- `maneuver_* -> outcome_*`

Disallowed examples:
- `outcome -> maneuver`
- `outcome -> outcome`
- `maneuver -> maneuver`

## Tensorization (`d_node_in=24`)
For v3 payloads, feature packing is fixed to 24 dims:
- 2 dims: node type one-hot (`maneuver`, `outcome`)
- 12 dims: maneuver class one-hot
- 3 dims: outcome node class one-hot (`collision/progress/compliance`)
- 1 dim: observed flag
- 4 dims: normalized interval features (`start`, `end`, `duration`, `mid`)
- 2 dims: degree features (`in`, `out`)

## Commands
Build cache (global-only default):
```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/build_dag_cache_v2.py \
  --data-dir data/scenarionet_waymo_training_500 \
  --out-dir outputs/dag_cache_v3_mo_smoke \
  --n-scenarios 20 \
  --strict-promptbn \
  --dag-contract maneuver_outcome_v1 \
  --dag-contract-mode hard \
  --no-dual-view \
  --num-frames 6
```

Validate cache:
```bash
PYTHONPATH=src .venv/bin/python src/scripts/dag_cache/validate_cache_contract.py \
  --cache-dir outputs/dag_cache_v3_mo_smoke/cache \
  --dag-contract maneuver_outcome_v1
```

Train Stage B with strict schema:
```bash
PYTHONPATH=src .venv/bin/python -m counter_bmt_v2.cli.train_nnx_bmt_dag_latent \
  --data-dir data/scenarionet_waymo_training_500 \
  --output-dir outputs/dag_latent_mo_stageB_smoke \
  --dag-source-mode cache \
  --dag-cache-dir outputs/dag_cache_v3_mo_smoke/cache \
  --dag-cache-strict \
  --dag-expected-schema v3_maneuver_outcome \
  --stage B \
  --stage-b-steps 30 \
  --batch-size 2
```
