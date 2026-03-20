# Legacy Adv-BMT DAG-Latent Add-On

This folder is intentionally additive-only.

It does **not** modify the released legacy Adv-BMT model, trainer, configs, or
decoder internals. Instead it adds a new Torch model class:

- `bmt.dag_latent.model.MotionLMDAGLatent`

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
