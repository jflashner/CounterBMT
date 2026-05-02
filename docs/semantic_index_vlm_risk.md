# Semantic Index VLM Risk

This is the simple VLM-first risk path for semantic maneuver data.

## What Changed

The SDC path VLM contract now asks for a per-maneuver `risk_level`:

- `low`
- `medium`
- `high`

and a short explanation field:

- `risk_rationale_short`

The field is attached to each `highlighted_paths` entry in
[sdc_path_contract.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/vlm_semantics/sdc_path_contract.py)
and requested in the prompts in
[sdc_path_prompt.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/vlm_semantics/sdc_path_prompt.py).

## How It Flows

When the semantic control index is built in
[build_sdc_semantic_control_index.py](/Users/joshuaflashner/Projects/CounterBMT/scripts/counterfactual/build_sdc_semantic_control_index.py),
each row now carries:

- `requested_vlm_risk_level`
- `requested_vlm_risk_rationale_short`

The builder also supports:

- `--min-vlm-risk-level low|medium|high`

So we can construct:

- all semantic rows
- medium+ only rows
- high-only rows

without changing the rest of the training format.

## Intended Use

The intended use is post-training scenario mining, not training-row filtering.

In other words:

- generate trajectories from the trained model
- carry forward the VLM maneuver risk annotations
- select `high`-risk generated counterfactuals
- use those generated safety-critical scenarios for downstream agent training

The builder-side `--min-vlm-risk-level` filter is just a convenient mining utility. It does not automatically affect model training unless we explicitly choose to use it that way.

## Important Backward-Compatibility Note

Older VLM outputs do not contain `risk_level`.

For backward compatibility, normalization defaults missing values to `medium`.

That means:

- newly generated VLM contracts will have real `risk_level`
- old semantic datasets will still load, but their risk level is only the compatibility default unless they are regenerated

## Safe Backfill Workflow

For existing rendered bundles, the safest workflow is:

- copy the bundle first
- rerender the copy in place
- relabel the copy in place

The helper for that is
[backfill_postsplit_semantics_bundle_copy.py](/Users/joshuaflashner/Projects/CounterBMT/scripts/counterfactual/backfill_postsplit_semantics_bundle_copy.py).

It pins:

- `model = gpt-5.4`
- `image_detail = original`
- traffic-light rendering on

and leaves the source bundle untouched.

The rerender path also makes nearby agents a bit more visually salient in
[render.py](/Users/joshuaflashner/Projects/CounterBMT/src/Adv-BMT/bmt/counterfactual/vlm_semantics/render.py),
which helps the VLM use surrounding traffic as evidence for `risk_level` and `risk_rationale_short`.
