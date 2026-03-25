# Legacy Adv-BMT DAG-Latent Add-On

This folder is intentionally additive-only.

It does **not** modify the released legacy Adv-BMT model, trainer, configs, or
decoder internals. Instead it adds a new Torch model class:

- `bmt.dag_latent.model.MotionLMDAGLatent`

The training additions are:

- `bmt.dag_latent.lightning.MotionLMDAGLatentLightning`
- `bmt.dag_latent.train_stage_a`
- `bmt.dag_latent.train_stage_b`
- `bmt.dag_latent.train_stage_c`
- `cfgs/0202_midgpt_dag_stage_a.yaml`
- `cfgs/0202_midgpt_dag_stage_b.yaml`
- `cfgs/0202_midgpt_dag_stage_c.yaml`

## What It Does

`MotionLMDAGLatent` inherits from the legacy `bmt.models.motionlm.MotionLM`
and adds one small DAG control path:

1. Run the normal legacy scene encoder.
2. Build or read a DAG latent.
3. Inject that latent into `encoder/scenario_token` with a global gated
   residual.
4. Let the existing legacy decoder run unchanged.

This is the smallest clean hook that keeps the old model intact while still
making decoder behavior depend on the DAG latent.

## Supported DAG Inputs

The wrapper accepts either:

1. A precomputed latent:

```python
batch["dag_latent"] = torch.randn(batch_size, dag_latent_width)
```

2. A tensorized DAG, matching the v2 tensor contract from
`counter_bmt_v2.training.dag_tensorize.tensorize_dag_batch(...)`:

```python
batch["dag_node_feat"]  # [B, G, F_node]
batch["dag_node_mask"]  # [B, G]
batch["dag_edge_src"]   # [B, E]
batch["dag_edge_dst"]   # [B, E]
batch["dag_edge_feat"]  # [B, E, F_edge]
batch["dag_edge_mask"]  # [B, E]
batch["dag_global_feat"]  # optional [B, F_global]
```

The slash-style keys also work:

- `dag/node_feat`
- `dag/node_mask`
- `dag/edge_src`
- `dag/edge_dst`
- `dag/edge_feat`
- `dag/edge_mask`
- `dag/global_feat`

## Minimal Usage

```python
from bmt.dag_latent.encoder import DAGLatentConfig
from bmt.dag_latent.model import MotionLMDAGLatent

dag_cfg = DAGLatentConfig(
    enabled=True,
    use_graph_encoder=True,
    d_node_in=24,
    d_edge_in=8,
    d_hidden=128,
    n_layers=3,
)

model = MotionLMDAGLatent(config=legacy_cfg, dag_config=dag_cfg)
output = model(batch_dict)
```

## Stage A Training

Stage A keeps the legacy `0202_midgpt` recipe and swaps in the additive
DAG-latent model wrapper.

The provided Stage-A config:

- [0202_midgpt_dag_stage_a.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/0202_midgpt_dag_stage_a.yaml)

inherits the released MidGPT config and sets:

- `DAG_LATENT.ENABLED=True`
- `DAG_LATENT.DAG_DROPOUT_PROB=1.0`

So even if DAG tensors are present, the latent path has zero effect and Stage A
reduces to autoregressive pretraining.

Launch with:

```bash
PYTHONPATH=src/Adv-BMT python -m bmt.dag_latent.train_stage_a
```

## Stage B And Stage C

The additive legacy Stage B/C path is cache-backed for now. It reuses the v2
DAG cache schema and tensorization path, then attaches tensorized `dag_*`
fields to each legacy batch inside the additive datamodule.

What each stage does:

- Stage B: fit the DAG encoder and DAG conditioning adapters while freezing
  non-DAG legacy model parameters.
- Stage C: joint finetune all trainable modules, but with a lower decoder LR
  and a higher DAG-module LR.

The provided configs:

- [0202_midgpt_dag_stage_b.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/0202_midgpt_dag_stage_b.yaml)
- [0202_midgpt_dag_stage_c.yaml](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/cfgs/0202_midgpt_dag_stage_c.yaml)

Current assumptions:

- `DAG_LATENT.SOURCE_MODE="cache"`
- `DAG_LATENT.CACHE_DIR` points at a v2 DAG cache directory keyed by
  `scenario_id`
- `dag_alignment/*` validation metrics run in Stage B/C when the legacy
  scenario evaluator is not attached

Launch examples:

```bash
PYTHONPATH=src/Adv-BMT python -m bmt.dag_latent.train_stage_b \
  DAG_LATENT.CACHE_DIR=../../outputs/dag_cache

PYTHONPATH=src/Adv-BMT python -m bmt.dag_latent.train_stage_c \
  DAG_LATENT.CACHE_DIR=../../outputs/dag_cache
```

Use `pretrain=...` to branch from the previous stage's checkpoint and `ckpt=...`
to resume the same stage in place.

## Behavior Notes

- The DAG conditioning is applied once per batch dict and recorded under
  `dag/conditioning_applied`.
- Diagnostics are written back into the batch dict:
  - `dag/latent`
  - `dag/latent_norm`
  - `dag/gate_mean`
  - `dag/source_used`
- If you reuse the same batch dict across repeated decode calls, the wrapper
  will not add the DAG residual twice.
- If you want a pure no-DAG baseline, leave the DAG keys absent and keep
  `use_null_latent=False`.

## Why This Is Not A Full Port Of The v2 DAG Path

The v2 implementation injects the DAG latent deeper inside the rewritten JAX
decoder stack.

This folder intentionally does something smaller:
- keep the legacy decoder untouched
- preserve the legacy tensor contracts
- add the minimum control hook needed for DAG-conditioned behavior

That makes it a good first legacy-compatible DAG-latent bridge without taking
on a full decoder rewrite.
