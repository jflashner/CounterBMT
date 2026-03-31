# Current DAG Schema And BMT Conditioning

This note captures the current DAG cache schema and the conditioning path currently implemented in Adv-BMT.

## Current DAG schema

Source of truth:
- `src/counter_bmt_v2/training/dag_cache_schema.py`

Current schema version:

```python
SCHEMA_VERSION_V3_MANEUVER_OUTCOME = "counter_bmt_v2_dag_cache_v3_maneuver_outcome"
SCHEMA_VERSION = SCHEMA_VERSION_V3_MANEUVER_OUTCOME

def schema_version_for_contract(contract_name: str) -> str:
    name = str(contract_name).strip().lower()
    if name == "maneuver_outcome_v1":
        return SCHEMA_VERSION_V3_MANEUVER_OUTCOME
    return SCHEMA_VERSION_V2_COMPACT10
```

So the current cache format for our DAGs is:
- schema version: `counter_bmt_v2_dag_cache_v3_maneuver_outcome`
- contract: `maneuver_outcome_v1`

Canonical payload shape:

```python
def dag_to_cache_payload(dag: BayesianDAG) -> Dict[str, Any]:
    nodes = []
    for node in dag.nodes.values():
        nodes.append(
            {
                "node_id": str(node.node_id),
                "node_type": str(node.node_type),
                "value": _jsonify(node.value),
                "timestamp_s": None if node.timestamp_s is None else float(node.timestamp_s),
                "metadata": _jsonify(dict(node.metadata)),
            }
        )

    edges = []
    for edge in dag.edges:
        edges.append(
            {
                "parent_id": str(edge.parent_id),
                "child_id": str(edge.child_id),
                "confidence": float(edge.confidence),
                "mechanism": str(edge.mechanism),
            }
        )

    ...

    return {
        "schema_version": schema_version,
        "scenario_id": str(dag.scenario_id),
        "nodes": nodes,
        "edges": edges,
        "cpts": _jsonify(dict(dag.cpts)),
        "metadata": metadata,
    }
```

In practice the cache JSON contains:

```json
{
  "schema_version": "counter_bmt_v2_dag_cache_v3_maneuver_outcome",
  "scenario_id": "...",
  "nodes": [
    {
      "node_id": "...",
      "node_type": "maneuver|outcome",
      "value": "...",
      "timestamp_s": 0.0,
      "metadata": {}
    }
  ],
  "edges": [
    {
      "parent_id": "...",
      "child_id": "...",
      "confidence": 0.9,
      "mechanism": "..."
    }
  ],
  "cpts": {},
  "metadata": {
    "contract_name": "maneuver_outcome_v1",
    "contract_version": "1",
    "contract_report": {
      "passed": true
    }
  }
}
```

Validation requirements:

```python
def validate_cache_payload(...):
    ...
    if schema_version == SCHEMA_VERSION_V3_MANEUVER_OUTCOME:
        expected_contract = "maneuver_outcome_v1"
    ...
    if expected_contract and contract_name != expected_contract:
        return False
    if not str(metadata.get("contract_version", "")).strip():
        return False
    report = metadata.get("contract_report")
    if not isinstance(report, Mapping):
        return False
    if "passed" not in report:
        return False
    if not bool(report.get("passed")):
        return False
```

## Current BMT conditioning path

Source of truth:
- `src/Adv-BMT/bmt/dag_latent/model.py`
- `src/Adv-BMT/bmt/models/layers/gpt_decoder_layer.py`
- `src/Adv-BMT/bmt/models/motion_decoder_gpt.py`

The model currently supports three DAG-conditioned paths:
- a global DAG latent gated onto `encoder/scenario_token`
- a timestep-aligned DAG control tensor `dag/time_control`
- optional maneuver-token cross-attention in the decoder

## Conditioning wrapper setup

The wrapper is `MotionLMDAGLatent(MotionLM)`:

```python
class MotionLMDAGLatent(MotionLM):
    _APPLIED_FLAG = "dag/conditioning_applied"

    def __init__(self, config: Any, dag_config: Optional[DAGLatentConfig] = None) -> None:
        super().__init__(config=config)
        self.dag_config = dag_config or DAGLatentConfig()
        self.d_model = int(self.config.MODEL.D_MODEL)

        if bool(self.dag_config.use_graph_encoder):
            self.dag_encoder = TorchDAGGraphEncoder(self.dag_config)
            self.dag_latent_in = 2 * int(self.dag_config.d_hidden)
        else:
            self.dag_encoder = None
            self.dag_latent_in = int(self.dag_config.latent_dim or self.d_model)

        self.dag_latent_proj = nn.Linear(self.dag_latent_in, self.d_model)
        self.dag_gate_proj = nn.Linear(self.dag_latent_in, self.d_model)
        self.null_dag_latent = nn.Parameter(
            torch.randn(self.dag_latent_in) * float(self.dag_config.null_latent_init_std)
        )

        self.dag_time_in = int(self.dag_config.time_guidance_feature_dim)
        if bool(self.dag_config.use_time_guidance) and bool(self.dag_config.time_guidance_use_global):
            self.dag_time_in += int(self.dag_latent_in)
        if bool(self.dag_config.use_time_guidance):
            self.dag_time_proj = nn.Linear(self.dag_time_in, self.d_model)
            self.dag_time_gate_proj = nn.Linear(self.dag_time_in, self.d_model)
            nn.init.constant_(
                self.dag_time_gate_proj.bias,
                float(self.dag_config.time_guidance_init_gate_bias),
            )
        else:
            self.dag_time_proj = None
            self.dag_time_gate_proj = None

        self.dag_maneuver_in = int(self.dag_config.maneuver_token_feature_dim)
        if bool(self.dag_config.use_maneuver_tokens) and bool(self.dag_config.maneuver_token_use_global):
            self.dag_maneuver_in += int(self.dag_latent_in)
        if bool(self.dag_config.use_maneuver_tokens):
            self.dag_maneuver_token_proj = nn.Linear(self.dag_maneuver_in, self.d_model)
        else:
            self.dag_maneuver_token_proj = None
```

## Global latent plus time guidance plus maneuver tokens

The actual injection happens in `apply_dag_conditioning(...)`:

```python
def apply_dag_conditioning(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    ...
    z_dag, meta = self.resolve_dag_latent(batch, device=scene_token.device, dtype=scene_token.dtype)
    dag_time_feat, time_meta = self.resolve_dag_time_guidance(
        batch,
        device=scene_token.device,
        dtype=scene_token.dtype,
        z_dag=z_dag,
    )
    dag_maneuver_feat, maneuver_meta = self.resolve_dag_maneuver_tokens(
        batch,
        device=scene_token.device,
        dtype=scene_token.dtype,
        z_dag=z_dag,
    )
    ...
    if z_dag is not None:
        bias = self.dag_latent_proj(z_dag)
        gate = torch.sigmoid(self.dag_gate_proj(z_dag))
        dag_bias = gate * bias
        ...
        batch["encoder/scenario_token"] = scene_token + dag_bias[:, None, :]
        batch["dag/latent"] = z_dag
        batch["dag/latent_norm"] = torch.linalg.norm(effective_z, dim=-1)
        batch["dag/gate_mean"] = effective_gate.mean(dim=-1)

    if dag_time_feat is not None:
        ...
        time_bias = self.dag_time_proj(dag_time_feat)
        if mode == "gated":
            time_gate = torch.sigmoid(self.dag_time_gate_proj(dag_time_feat))
            time_control = time_gate * time_bias
        elif mode == "additive":
            time_gate = torch.ones_like(time_bias)
            time_control = time_bias
        ...
        batch["dag/time_control"] = time_control
        batch["dag/time_mask"] = time_mask

    if dag_maneuver_feat is not None:
        ...
        maneuver_token = self.dag_maneuver_token_proj(dag_maneuver_feat)
        maneuver_token = maneuver_token * maneuver_mask[:, :, None].to(dtype=scene_token.dtype)
        ...
        batch["dag/maneuver_token"] = maneuver_token
        batch["dag/maneuver_mask"] = maneuver_mask
```

Interpretation:
- `encoder/scenario_token += gated(global_dag_latent)`
- `dag/time_control` gives a per-step control tensor
- `dag/maneuver_token` gives a small learned maneuver memory for decoder attention

## Decoder maneuver-token cross-attention

The decoder layer now has an explicit maneuver-memory attention block:

```python
self.cross_maneuver = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=nhead,
    dropout=dropout,
    batch_first=True,
)
...
self.maneuver_norm = nn.LayerNorm(d_model)
self.maneuver_residual_gate = nn.Parameter(torch.full((d_model,), float(maneuver_gate_init_bias)))
```

And the forward pass uses it like this:

```python
if (
    maneuver_token is not None
    and maneuver_token_mask is not None
    and maneuver_token.numel() > 0
    and bool(maneuver_token_mask.any())
):
    x = x.reshape(B, T * N, D)
    out = x
    if self.use_adaln:
        out = self.maneuver_adaln_norm(out)
        out = utils.modulate(
            out,
            shift_maneuver.reshape(B, T * N, D),
            scale_maneuver.reshape(B, T * N, D),
        )
    else:
        out = self.maneuver_norm(out)

    maneuver_token_mask = maneuver_token_mask.bool()
    empty_rows = ~maneuver_token_mask.any(dim=1)
    if bool(empty_rows.any()):
        maneuver_token = maneuver_token.clone()
        maneuver_token_mask = maneuver_token_mask.clone()
        maneuver_token[empty_rows, 0] = 0.0
        maneuver_token_mask[empty_rows, 0] = True

    out, _ = self.cross_maneuver(
        query=out,
        key=maneuver_token,
        value=maneuver_token,
        key_padding_mask=~maneuver_token_mask.bool(),
        need_weights=False,
    )
    if self.use_adaln:
        out = out * gate_maneuver.reshape(B, T * N, D)
    else:
        out = out * torch.sigmoid(self.maneuver_residual_gate).reshape(1, 1, D)
    x = x + out
    x = x.reshape(B, T, N, D)
```

So the decoder attends to maneuver tokens as an extra memory source, then adds that back through a learned residual gate.

## Decoder plumbing

The motion decoder receives the maneuver conditioning tensors here:

```python
decoded_tokens = self.decoder(
    agent_token=action_token,
    scene_token=scene_token,
    maneuver_token=input_dict.get("dag/maneuver_token", input_dict.get("dag_maneuver_token", None)),
    maneuver_token_mask=input_dict.get("dag/maneuver_mask", input_dict.get("dag_maneuver_mask", None)),
    a2a_info=a2a_info,
    a2t_info=a2t_info,
    a2s_info=a2s_info,
    condition_token=condition_token if self.use_adaln else None,
    use_cache=use_cache,
    past_key_value_list=past_key_value_list
)
```

## Practical note

The code currently supports:
- global DAG latent
- timestep guidance
- maneuver-token cross-attention

But the best-performing run so far was the time-guidance ablation:
- `DAG_LATENT.USE_TIME_GUIDANCE=true`
- `DAG_LATENT.USE_MANEUVER_TOKENS=false`

So the full conditioning stack exists in code, but the most promising training result so far has come from the simpler time-guidance-only setup.
