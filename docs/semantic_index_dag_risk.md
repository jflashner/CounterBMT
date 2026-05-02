# Semantic Index DAG Risk

This note documents the additive DAG-based risk prior attached to each row in the semantic SDC control index built by
[build_sdc_semantic_control_index.py](/Users/joshuaflashner/Projects/CounterBMT/scripts/counterfactual/build_sdc_semantic_control_index.py).

## Goal

We want each semantic maneuver row to carry a compact estimate of how safety-critical that maneuver is in the local scene context, so later we can:

- rank generated alternatives by safety criticality
- measure uplift relative to the factual GT maneuver
- mine the most dangerous but still plausible scenarios for downstream agent training

## What It Is

The new score is generated in
[dag_risk.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/dag_risk.py).

For each highlighted semantic slot, we build a small proxy causal context from:

- the semantic maneuver label for that slot
- the nearest relevant traffic light stop point and signal state
- conflict-agent ETA gaps near that stop point
- conflict-agent current speeds

From that proxy context we produce:

- `critical_event_prior`
- `risk_score_total`
- `risk_uplift_vs_gt`
- component risks:
  - `p_conflict_overlap`
  - `p_signal_violation`
  - `p_collision_or_near_miss`
  - `severity_score`
- a compact sparse DAG view under `dag_view`

## What It Is Not

This is currently a structured heuristic prior, not a calibrated probability model.

Important implications:

- `critical_event_prior` is a relative risk score in `[0, 1]`, not a validated probability of collision.
- `risk_uplift_vs_gt` is the main counterfactual ranking signal.
- the score is intended for mining and prioritization first, not for claiming absolute real-world risk.

The row field `calibrated_probability: false` is included on purpose to make that explicit.

## Where It Enters The Index

The builder now computes one `dag_risk` block per slot before writing rows:

- scorer call:
  [build_sdc_semantic_control_index.py](/Users/joshuaflashner/Projects/CounterBMT/scripts/counterfactual/build_sdc_semantic_control_index.py)
- row attachment:
  [build_sdc_semantic_control_index.py](/Users/joshuaflashner/Projects/CounterBMT/scripts/counterfactual/build_sdc_semantic_control_index.py)

Each output row gets:

```json
"dag_risk": {
  "version": "dag_risk_v1",
  "calibrated_probability": false,
  "semantic_label": "...",
  "risk_score_total": 0.0,
  "critical_event_prior": 0.0,
  "risk_uplift_vs_gt": 0.0,
  "critical_event_prior_uplift_vs_gt": 0.0,
  "risk_tier": "low|medium|high|critical",
  "risk_rank_within_example": 1,
  "signal_state_at_decision": "...",
  "signal_state_category": "go|stop|caution|unknown",
  "proxy_compliance_label": "...",
  "proxy_entry_timing": "...",
  "num_conflict_agents": 0,
  "min_conflict_eta_gap_s": null,
  "max_conflict_agent_speed_mps": 0.0,
  "risk_components": { ... },
  "dag_view": { ... }
}
```

## How To Use It Later

For post-training scenario mining, the recommended first-pass selector is:

- sort by `risk_uplift_vs_gt`
- break ties with `risk_score_total`
- then apply any rollout plausibility / realism filters

That ranking emphasizes maneuvers that make the scene more dangerous than the factual baseline, which is usually more useful than ranking by raw scene difficulty alone.
