# Victim-Centric Adversarial Agent Eval

This note captures a better use of our generated safety-critical counterfactuals for downstream agent evaluation and training.

## Core Idea

Instead of treating the intervened actor's generated rollout as an ego behavior label to imitate, treat:

- the intervened actor as the adversary
- the most affected other agent in the scene as the ego

In this framing, the generated rollout matters directly because it is the adversary trajectory the evaluated agent must react to.

## Why This Is Better

This avoids the mismatch:

- we train a generator to produce meaningful risky maneuvers
- but then throw away the actual risky maneuver because we do not want the ego to imitate unsafe behavior

With a victim-centric adversarial setup:

- semantic maneuver learning still matters
- the generated rollout is used directly
- VLM risk labels remain useful for mining
- the resulting scenarios are naturally aligned with robustness evaluation

## Proposed Pipeline

1. Choose an actor `A` to intervene on.
2. Generate a counterfactual rollout for `A`.
3. Score the generated scenario for plausibility and safety criticality.
4. Identify the primary affected other agent `B`.
5. Recast `B` as ego for downstream evaluation or training.
6. Use `A`'s generated rollout as the adversary behavior.

## Selecting The New Ego

The new ego should be the primary victim, not an arbitrary other actor.

Candidate selection rules:

- minimum predicted TTC to the adversary
- earliest conflict-zone overlap
- largest right-of-way violation imposed by the adversary
- largest forced braking or evasive deviation

If multiple agents are strongly affected, a single generated adversarial rollout can produce multiple victim-centric evaluation scenarios.

## What From The Generated Rollout Matters

In this setup, the generated rollout is no longer just metadata. It becomes the adversary policy trace:

- path geometry
- timing
- speed profile
- merge / cut-in timing
- signal violation timing
- conflict ordering

That is exactly the behavior the victim ego must respond to safely.

## Fidelity Levels

There are three natural versions of this setup.

### Level 1

- adversary actor `A` follows the generated rollout
- all other agents replay logged futures

This is the easiest version and is useful for mining and short-horizon stress tests.

### Level 2

- adversary actor `A` follows the generated rollout
- directly affected agents are replanned or made reactive

This is likely a much better training target.

### Level 3

- full multi-agent counterfactual simulation

This is the cleanest version if the infrastructure supports it.

## Best Uses

This framing supports:

- adversarial scenario generation for victim-centric evaluation
- robustness fine-tuning for downstream agents
- mining structured failure benchmarks
- curriculum design by maneuver and risk type

Examples:

- left turn across path
- aggressive cut-in
- red-light violation
- forced merge
- blocked-turn overshoot

## Important Caveat

The generated ego rollout from the original intervention should usually **not** be used as the behavior target for the trained eval agent.

The main target is:

- survive and respond safely to the generated adversarial scenario

not:

- copy the risky maneuver

## Suggested Data Structure

A future `critical_scenario_bank` could store:

- map and traffic-light state
- adversary actor id
- adversary generated rollout
- victim ego actor id
- background actor trajectories
- maneuver label
- VLM risk level and rationale
- conflict metadata
- plausibility / degeneracy filters
- baseline-agent failure results

## Open Next Steps

- define the primary-victim selection rule
- decide whether Level 1 or Level 2 is the first implementation target
- define export format for victim-centric scenarios
- decide how these scenarios enter downstream agent training curricula

## Supporting Scripts

Current utilities we added for this workflow:

- `scripts/counterfactual/probe_agent_semantic_rollout.py`
  - probes arbitrary-agent semantic interventions
  - exports victim-centric overlays and optional scenario artifacts
  - now uses the same evaluation-style semantic rollout stack as the SDC GIF evaluator
  - defaults to a full-horizon control window unless overridden
  - can emit `debug_trace.json` with:
    - control tensors before and after preprocess
    - runtime control kind / control-available state
    - first-step top logits for the targeted agent
    - optional SDC-vs-non-SDC comparison on the same scene
- `scripts/counterfactual/render_sdc_semantic_animation_examples.py`
  - original SDC validation GIF renderer
  - now also supports arbitrary-agent GIF rendering via `--non-sdc-cases-json`
  - non-SDC GIF mode now uses the same evaluation-style semantic rollout path as SDC rendering
- `scripts/agent_eval/build_victim_centric_table4_dataset.py`
  - first offline dataset builder for Table 4 style victim-centric RL
  - loads the intervention checkpoint once, then sweeps one deterministic base row per scene
  - defaults to `victim = SDC`
  - ranks nearby moving non-SDC adversary candidates
  - evaluates a small semantic label set per candidate
  - exports paired:
    - natural scenarios
    - adversarial victim-centric scenarios
  - writes MetaDrive-compatible `dataset_summary.pkl` files and per-scene manifests
- `scripts/agent_eval/prepare_td3_table4_views.py`
  - builds TD3-ready ScenarioNet directory views from the offline natural and
    adversarial banks
  - matches natural/adversarial scenarios by `waymax_scene_*` source id
  - creates:
    - `train_waymo_only`
    - `train_counterbmt_mixed`
    - `eval_waymo_only`
    - `eval_counterbmt_adversarial`
  - clips the requested `500/100` pair counts to the number of paired exports
    we actually have
  - writes a manifest with selected scene ids and suggested TD3 paths
- `scripts/remote/run_td3_table4_openloop.sh`
  - generic open-loop TD3 launcher for the prepared Table 4 dataset views
  - points `train_td3.py` at the TD3-ready ScenarioNet dirs without using the
    closed-loop online generators

## Replicating Adv-BMT Table 4

This section translates the open-loop RL protocol from Table 4 of `Adv-BMT.pdf`
into our victim-centric CounterBMT setup.

### What Table 4 Does

The paper's open-loop table has a very specific protocol:

- start from `500` real-world WOMD training scenarios
- augment each with `1` generated collision scenario
- train a `TD3` policy for `1,000,000` steps with `8` random seeds in MetaDrive
- evaluate on:
  - `100` original WOMD validation scenarios
  - `100` generated collision validation scenarios

The important structure is not the exact baseline names. The important structure
is:

- fixed offline augmented training set
- `1:1` ratio between natural and generated scenarios
- training on one agent as ego in MetaDrive
- evaluation on both natural and adversarial validation sets

### Victim-Centric Translation

Our version should treat the intervened actor as the adversary and the most
affected other actor as the ego.

That gives us the clean Table 4 analogue:

- original training set:
  - `500` natural WOMD / ScenarioNet scenes
- augmented training set:
  - `500` victim-centric adversarial scenes
  - one per natural scene
- original validation set:
  - `100` natural validation scenes
- adversarial validation set:
  - `100` victim-centric adversarial validation scenes

The open-loop part matters here: the augmented scenarios are generated offline
before RL training. During RL training, the adversary is replayed from its
exported trajectory rather than being regenerated online.

### What Counts As Our Table 4 Rows

If we want a faithful analogue of Table 4 with our framework, the first three
rows to run are:

1. `Waymo`
   - Train TD3 on the original `500` natural scenes only.

2. `CounterBMT Victim-Centric`
   - Train TD3 on `500` natural scenes + `500` victim-centric adversarial
     scenes.
   - In these adversarial scenes:
     - the intervened actor follows the generated risky rollout
     - the selected victim actor is the ego
     - background actors replay logged futures

3. `CounterBMT Victim-Centric (Refined)`
   - Same dataset size and same `1:1` ratio.
   - But use refined exports where the adversary rollout is embedded in a more
     reactive scene, matching the spirit of Adv-BMT's forward refinement.
   - In our framework this should mean:
     - adversary follows generated rollout
     - directly affected neighbors and/or all modeled agents use the refined
       multi-agent rollout rather than plain logged futures

If we later want a wider comparison table, we can add extra rows such as:

- `CounterBMT Non-Victim-Recast`
  - keeps the original SDC as ego and uses the generated scene only as a
    perturbation baseline
- `CounterBMT Victim-Centric + High-Risk Only`
  - uses the same pipeline but mines only high-risk VLM-labeled exports

### Dataset Construction

The open-loop dataset build should be deterministic and split-aware.

#### Train Split

- Choose `500` natural training scenes.
- For each training scene:
  - generate candidate counterfactual maneuvers
  - score them for plausibility and risk
  - select the best adversarial candidate
  - choose the primary victim
  - export exactly one victim-centric adversarial scenario

This gives:

- `500` natural scenarios
- `500` adversarial victim-centric scenarios

#### Validation Split

- Choose `100` held-out natural validation scenes.
- For each validation scene, export exactly one victim-centric adversarial
  scenario using the same offline mining policy.

This gives:

- `100` natural validation scenarios
- `100` adversarial validation scenarios

To match the paper cleanly, train scenes and validation scenes must remain
disjoint at the base-scenario level.

### Scenario Selection Policy

For each base scene, generate multiple adversarial candidates, but export only
one per scene for Table 4.

The selection rule should prioritize:

- high VLM risk
- plausibility
- victim clarity
- scenario usefulness for agent learning

Recommended ranking:

1. `risk_level == high`
2. no obvious degeneracy
3. primary victim has the lowest TTC to adversary
4. victim is not the adversary itself
5. export candidate with strongest conflict signal

Useful tie-breakers:

- smallest min TTC
- largest required braking by victim
- largest right-of-way violation
- strongest conflict-zone overlap

### Export Semantics

To train RL like Table 4, the exported scenario should encode:

- map and traffic lights
- victim actor id as the new ego
- adversary actor id
- adversary generated rollout
- background actors
- metadata describing:
  - maneuver type
  - risk level
  - risk rationale
  - min TTC
  - conflict type
  - whether the scenario is refined

For the open-loop Table 4 analogue:

- the adversary trajectory is fixed
- the scene is replayable
- the RL agent controls only the victim ego

### Level 1 vs Refined Export

We should deliberately mirror the paper's `Adv-BMT` and `Adv-BMT (Refined)`
rows.

#### CounterBMT Victim-Centric

Use `Level 1` exports:

- adversary follows generated rollout
- other actors replay logged futures

This is the cleanest first reproduction.

#### CounterBMT Victim-Centric (Refined)

Use `Level 2` or refined exports:

- adversary follows generated rollout
- directly affected actors and/or all modeled agents use the refined
  multi-agent counterfactual rollout

This is the closest match to the paper's forward refinement idea.

### RL Training Protocol

We already have a TD3 training entrypoint in:

- `src/Adv-BMT/bmt/rl_train/train/train_td3.py`

That script already matches the paper surprisingly well:

- `ScenarioEnv` / MetaDrive
- `TD3`
- configurable train/eval data dirs
- configurable `1,000,000` training steps
- evaluation callbacks and episode metrics

So the practical Table 4 training plan is:

1. Build dataset dir `train_waymo_500/`
2. Build dataset dir `train_victim_adv_500/`
3. Merge them into `train_waymo_plus_victim_adv_1000/`
4. Build `val_waymo_100/`
5. Build `val_victim_adv_100/`
6. Run TD3 for `1,000,000` steps for `8` seeds
7. Evaluate each trained policy on both validation dirs

For our current wiring, the least invasive way to do this is:

1. export the offline natural/adversarial banks with
   `build_victim_centric_table4_dataset.py`
2. convert them into TD3-ready directory views with
   `prepare_td3_table4_views.py`
3. train open-loop TD3 with:
   - `train_waymo_only` for the natural baseline row
   - `train_counterbmt_mixed` for the victim-centric mixed row
4. evaluate the saved policy on:
   - `eval_waymo_only`
   - `eval_counterbmt_adversarial`

This keeps the RL path close to the original Adv-BMT open-loop setup while
letting us use our victim-centric scenario banks directly.

### Metrics To Report

To mirror Table 4, report:

- `Reward`
- `Cost`
- `Completion`
- `Collision`
- `Cost Sum`

The legacy TD3 script already logs the components needed for most of this:

- episode reward
- route completion
- crash
- cost

So our evaluation table can directly mirror:

- natural validation performance
- adversarial validation performance

### Recommended First Implementation Target

The first concrete milestone should be:

- reproduce the `Waymo` row
- reproduce `CounterBMT Victim-Centric`
- reproduce `CounterBMT Victim-Centric (Refined)`

That is enough to establish the key claim:

- whether victim-centric adversarial training improves robustness on adversarial
  validation environments without collapsing natural-scene performance

### Code Touchpoints

The minimum set of code we should extend is:

- `src/counter_bmt/scenario_export.py`
  - add victim-centric export that can:
    - set an arbitrary victim as ego
    - inject the adversary generated rollout
    - optionally export refined multi-agent rollouts

- `docs/victim_centric_adversarial_agent_eval.md`
  - this note becomes the design source of truth

- new script:
  - `scripts/agent_eval/build_victim_centric_table4_dataset.py`
  - responsibilities:
    - choose the `500/100` base splits
    - mine one export per scene
    - build merged train/eval dataset dirs
    - write manifest files for reproducibility

- existing trainer:
  - `src/Adv-BMT/bmt/rl_train/train/train_td3.py`
  - use this as the Table 4 training driver

- new script:
  - `scripts/agent_eval/prepare_td3_table4_views.py`
  - responsibilities:
    - take the exported natural/adversarial banks
    - select paired scene ids shared by both banks
    - materialize TD3-ready ScenarioNet views for:
      - `train_waymo_only`
      - `train_counterbmt_mixed`
      - `eval_waymo_only`
      - `eval_counterbmt_adversarial`

- new script:
    - `scripts/remote/run_td3_table4_train500_zh2.sh`
  - responsibilities:
    - launch the concrete `train500` Table 4 rows against the prepared shared
      TD3 view bank on `zhoulab-2`
    - expose only the row choice and eval split

- new script:
  - `scripts/agent_eval/migrate_victim_centric_bank.py`
  - responsibilities:
    - repair previously exported victim-centric banks that were generated
      before the ScenarioNet schema normalization fix
    - rewrite map features into the MetaDrive-compatible schema
    - fill missing top-level fields like `version`
    - regenerate `dataset_summary.pkl` and `dataset_mapping.pkl`
  - purpose:
    - salvage older `476/95` banks without rerunning intervention generation
    - provide a fast migration path before rebuilding TD3-ready views

### Utilities Added During This Workstream

To keep the implementation surface tidy, every new helper script should be
listed here as it is added.

- `scripts/counterfactual/probe_agent_semantic_rollout.py`
  - generation-only probe for semantic-only control on an arbitrary modeled
    agent
  - takes a raw scenario pickle, checkpoint, target `agent_id`, and semantic
    label
  - runs both a baseline rollout and a controlled rollout on the same scene
  - writes a compact artifact bundle:
    - `summary.json`
    - `trajectories.npz`
    - `overlay.png`
    - `victim_centric_overlay.png`
  - optional victim-centric replay exports when `--export-victim-centric` is
      enabled
  - purpose:
    - verify that arbitrary-agent semantic control works before we build the
      victim-centric exporter
    - inspect whether changing the controlled actor produces the adversarial
      behavior we actually want to export later
    - auto-select a plausible victim by nearest adversary-victim rollout
      interaction geometry
    - emit a first replayable pair:
      - victim-centric ground truth
      - victim-centric counterfactual
  - current default victim policy:
    - `sdc`
    - because the downstream agent we want to train/evaluate is the SDC
  - analysis mode still supports:
    - `--victim-agent-id auto`
    - or a specific non-SDC victim id

- `scripts/agent_eval/migrate_victim_centric_bank.py`
  - migrates an existing victim-centric natural/adversarial bank into
    MetaDrive-compatible ScenarioNet schema
  - writes:
    - migrated `natural_scenarios/`
    - migrated `adversarial_scenarios/`
    - `migration_report.json`
  - intended use:
    - older banks built before the schema fix
    - fast repair of TD3 inputs without recomputing interventions

This note and [command_reference.md](/Users/joshuaflashner/Projects/CounterBMT/docs/command_reference.md)
should stay in sync whenever we add more victim-centric export or dataset
builder scripts.

### Practical Recommendation

The cleanest path is:

1. implement victim-centric open-loop export first
2. build the `500 + 500` training dataset
3. prepare TD3-ready views from the exported banks
4. run the `Waymo` baseline and `CounterBMT Victim-Centric`
4. then add the refined export row

This gives us a direct analogue of Table 4 using our framework, while keeping
the experimental surface area small enough to debug.

### ScenarioNet Compatibility Note

One subtle but important export requirement for the TD3/MetaDrive path is that
the saved scenario bank must follow ScenarioNet-style map feature schema, not
just "roughly similar" Waymax raw schema.

In practice, the direct ScenarioNet files under:

- `/home/grads/jflashner/CounterBMT/data/scenarionet_waymo_training_500`

were the reference that made the failure mode obvious:

- polygonal map objects such as `CROSSWALK`, `SPEED_BUMP`, and `DRIVEWAY`
  carry `polygon`
- stop signs carry point-style fields such as `position`

Our reconstructed Waymax raw pickles did not always have those exact fields, so
MetaDrive could fail during `ScenarioEnv.reset()` with errors like
`KeyError: 'polygon'`.

The current fix lives in `src/counter_bmt/scenario_export.py`:

- normalize reconstructed Waymax raw scenarios into a minimal
  ScenarioNet/MetaDrive-compatible schema at export time
- preserve the original reconstructed track ids so victim/adversary ids still
  line up with the intervention outputs
- if an external source file is provided but does not contain the required
  victim/adversary track ids, automatically fall back to the normalized raw
  reconstructed scenario instead of hard-failing

This is the export path that should be used for the victim-centric Table 4
banks unless we later build a more exact source-scenario matching layer.

### Current Prepared TD3 Bank

The first wired bank currently lives at:

- `/data/home/grads/jflashner/CounterBMT_run/eval_runs/victim_centric_table4_td3_views_20260418`

with the concrete TD3 views:

- `train_waymo_only`
- `train_counterbmt_mixed`
- `eval_waymo_only`
- `eval_counterbmt_adversarial`

Current paired counts:

- train natural pairs available: `476`
- train adversarial pairs available: `476`
- val natural pairs available: `95`
- val adversarial pairs available: `95`
