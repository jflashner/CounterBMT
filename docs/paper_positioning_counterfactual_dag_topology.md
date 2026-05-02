# Paper Positioning: Counterfactual Semantic Control, Causal DAGs, and Topology-Aware Rollout Optimization

## Purpose
This document is a paper-facing summary of the scientific value of the current CounterBMT research thread. It is meant to capture:

- what we have actually built,
- what is scientifically novel,
- how the current setup connects to topology-aware RL ideas such as Topo-MCPO,
- how the work relates to `TEN-DM`-style topology representations, and
- how causal DAGs and counterfactual training fit together in a coherent research story.

The goal is to provide a strong starting point for an introduction, method section, contribution list, and discussion section in a paper.

---

## 1) Executive Summary

The core scientific idea is to train a motion model not merely to imitate observed trajectories, but to realize **semantically meaningful counterfactual decisions** under a **causal and geometrically grounded intervention view**.

In the current system:

- training examples are built as **local counterfactual interventions** around a decision point,
- the model is conditioned at runtime by a **compact semantic decision label** rather than an exact path identifier,
- privileged supervision is provided by **families of valid geometric alternatives**,
- behavior is refined with a **rollout-level RL objective** over sampled futures rather than only token-level imitation,
- progress reward is computed on a **topologically meaningful 1D trace derived from the valid tube geometry**, and
- causal DAG views provide a natural way to encode **decision-to-outcome structure** and connect this work to structured/topology-aware policy optimization.

This gives a framework that sits between:

- supervised trajectory prediction,
- causal counterfactual reasoning,
- geometry-aware control,
- and topology-aware RL.

---

## 2) The Scientific Problem

Standard motion forecasting and policy learning pipelines usually optimize one of two things:

1. exact imitation of the factual future, or
2. unconstrained reward maximization in a learned environment.

Neither is ideal for the research question we care about.

Our goal is different:

- We want to ask: **what would the SDC do under a different local decision?**
- We want that alternative to be **causally meaningful**, not an arbitrary perturbation.
- We want the resulting behavior to remain **dynamically plausible** and **geometrically valid**.
- We want to steer the model using **decision semantics** such as `left`, `right`, or `right_lane_change`, not brittle path IDs.

This makes the problem inherently counterfactual:

- hold the scene context fixed,
- intervene on the local decision,
- preserve realism and validity,
- and train the policy to realize the new branch.

That is the main scientific value of the project.

---

## 3) What We Have Built

## 3.1 A geometry-grounded counterfactual dataset

The semantic counterfactual dataset is built in the `sdc_semantic_only` pipeline, primarily in:

- `src/Adv-BMT/bmt/counterfactual/sdc_semantic_control.py`
- `src/Adv-BMT/bmt/counterfactual/branch_enumeration.py`
- `src/Adv-BMT/bmt/counterfactual/contract_local_intervention.py`

The dataset does not treat counterfactuals as free-form labels. Instead, it constructs them from:

- local branch geometry,
- recovered factual decision structure,
- branch commitment and branch margin,
- downstream validity,
- semantic grouping of alternatives.

The current top-859 build contains:

- `859` examples,
- `3436` total rows,
- `2577` alternative rows,
- `3392` rows marked usable for training.

From the current split:

- `687` train scenarios / `2748` train rows,
- `172` val scenarios / `688` val rows,
- `0` train/val scenario overlap.

This is scientifically important because the counterfactual supervision is not synthetic label noise. It is **grounded in valid local branch structure**.

## 3.2 A semantic-only runtime control interface

The runtime control interface is intentionally compact. The model is trained in `sdc_semantic_only` mode rather than exact-path mode:

- runtime input is a semantic request such as `left`, `right`, or `right_lane_change`,
- not a privileged exact path polyline,
- while training still has access to path-family geometry as supervision.

This separation matters. It lets the model learn:

- decision semantics at inference time,
- while using richer privileged structure during training.

That is a strong design choice for both scientific clarity and deployability.

## 3.3 A family-based realism prior

The system includes a semantic family guidance loss in:

- `src/Adv-BMT/bmt/models/motionlm_lightning.py`

This loss builds a **family teacher distribution** from:

- frozen teacher policy logits,
- path proximity,
- heading agreement,
- backward-progress penalties.

The student is then trained toward that family teacher by KL loss.

This is valuable scientifically because it provides a middle ground between:

- exact path imitation, and
- unconstrained RL.

It says: stay inside the **family of behaviors consistent with the requested intervention**, not necessarily one exact path.

## 3.4 A rollout-level counterfactual RL objective

The current main refinement mechanism is the rollout tube policy objective in:

- `src/Adv-BMT/bmt/models/motionlm_lightning.py`

At a high level, we:

1. sample grouped autoregressive rollouts,
2. score each rollout against a valid counterfactual tube,
3. add a progress-shaped reward,
4. compute discounted return-to-go,
5. normalize advantages within the rollout group,
6. apply a REINFORCE-style policy loss.

This is scientifically important because the model is no longer rewarded only for matching tokens. It is rewarded for producing **behavior-level commitment** to the requested counterfactual branch.

---

## 4) What Is Novel

## 4.1 Counterfactual supervision is built from local interventions, not generic trajectory relabeling

The intervention contract in:

- `src/Adv-BMT/bmt/counterfactual/contract_local_intervention.py`

explicitly separates:

- scene context,
- recovered factual decision,
- alternative decisions,
- provenance,
- commitment metrics,
- supervision gates,
- trainability/alignment metadata.

This gives the work a clear causal interpretation:

- the same context can support multiple locally valid alternative actions,
- and each alternative can be treated as an intervention candidate.

That is more principled than simply attaching a different trajectory target to the scene.

## 4.2 Semantic control without exact-path leakage at runtime

The training signal knows about candidate path families, but the runtime API stays semantic. This is novel in practice because it gives:

- high-level controllability,
- low-dimensional decision conditioning,
- and training-time geometric structure,

without making inference depend on privileged exact path annotations.

## 4.3 Rollout reward is defined over a valid region, not just a target line

The counterfactual objective does not force the policy to hit one polyline exactly. Instead it rewards behavior inside a **valid tube** around the selected alternative family.

This matters because counterfactual decisions like turning, lane changing, or yielding are naturally region-valued rather than single-curve-valued.


## 4.5 Causal DAGs are not an afterthought; they are a natural representation of the intervention

The code in:

- `src/Adv-BMT/bmt/counterfactual/dag_adapter.py`
- `docs/current_dag_schema_and_conditioning.md`
- `docs/dag_contract_maneuver_outcome_v1.md`

shows that local interventions can be projected into sparse DAGs with nodes such as:

- signal state,
- conflict ETA,
- path choice,
- compliance,
- entry timing,
- collision outcome,
- stopline crossing,
- interaction order.

Edges encode explicit causal hypotheses, for example:

- signal state -> compliance,
- conflict ETA -> entry timing,
- path choice/compliance/entry timing -> collision and crossing outcomes.

This makes the counterfactual story causally interpretable, not just geometrically interpretable.

---

## 5) Why This Is Scientifically Valuable

## 5.1 It reframes motion generation as intervention realization

The work shifts the target from:

- “predict the future”

to:

- “realize a plausible alternative future under a specified local intervention.”

That is a stronger scientific framing for controllable autonomy.

## 5.2 It separates decision semantics from geometric realization

The model is asked to realize a semantic branch choice, while geometry is used to define:

- what counts as valid,
- what counts as progress,
- what counts as a family-consistent behavior.

That disentanglement is one of the most valuable ideas in the project.

## 5.3 It creates a bridge between causal reasoning and trajectory learning

The DAG layer describes:

- which context variables affect the decision,
- which decision variables affect outcomes,
- and which outcomes should change under intervention.

The rollout tube objective then teaches the policy to **instantiate** those alternative decisions behaviorally.

This gives a coherent answer to a common gap in causal autonomy work:

- DAGs explain what should change,
- counterfactual rollout training teaches the policy how to change it.

---

## 6) Relation to Topo-MCPO

The strongest connection to Topo-MCPO is through **grouped rollout optimization over structured behavioral alternatives**.

The repo’s behavior-manifold notes are in:

- `docs/rl_behavior_manifold_implementation_details.md`

That document maps our broader RL stack to Topo-MCPO-style ideas such as:

- grouped rollout sampling,
- relative/group-normalized optimization,
- novelty-tilted selection,
- consensus-aware scoring,
- topology-aware behavior embeddings.

Our current Adv-BMT semantic rollout setup is not yet a verbatim end-to-end implementation of the full behavior-manifold Topo-MCPO stack. But it is highly aligned with that philosophy in several ways.

## 6.1 Grouped rollouts and relative optimization

The current tube policy objective:

- samples a group of rollouts per example,
- computes returns per rollout,
- normalizes advantages within the group,
- and optimizes relatively rather than absolutely.

This is directly in the spirit of group-relative policy optimization.

## 6.2 Structured behavior spaces instead of scalar reward only

Topo-MCPO is valuable because it treats policy optimization as happening over a structured space of behaviors, not just raw trajectories.

Our current setup already moves in that direction:

- semantic labels define decision families,
- path families define valid geometric support,
- progress traces define topology-aware forward structure,
- causal DAGs define interpretable decision/outcome structure.

In other words, our rollouts are not being compared as arbitrary curves. They are being compared as members of a **structured counterfactual behavior manifold**.

## 6.3 Natural next step toward full Topo-MCPO integration

A clean paper claim is that the present work provides the **counterfactual and causal substrate** on top of which a fuller Topo-MCPO layer can operate.

Concretely:

- the current rollout tube reward gives branch-validity and commitment,
- the DAG view gives structured intervention semantics,
- topology-aware behavior embeddings can then score novelty and consensus over these counterfactual behaviors.

So the relationship is best described as:

- **the current system is Topo-MCPO-compatible and behavior-manifold-ready**,
- not merely generic policy fine-tuning.

---

## 7) Relation to TEN-DM and Topology-Aware Representation Learning

The local paper `16261_TEN_DM_Topology_Enhanced (1).pdf` contributes a different but complementary idea:

- use graph abstractions and topology-aware representations to capture higher-order structure in spatio-temporal processes.

The main TEN-DM themes that matter here are:

- graph abstraction,
- topological summaries of temporal evolution,
- time-image representations,
- multi-scale structural reasoning,
- zigzag persistence style representations.

The current CounterBMT work connects to those ideas in three ways.

## 7.1 Graph abstraction

We explicitly convert local intervention structure into sparse DAGs. This mirrors the TEN-DM view that raw sequences alone are not enough; structural relations matter.

## 7.2 Topology-aware progress representation

Our biggest geometry lesson was that the relevant object is not the raw path polyline. It is the topology of the valid region and the ordering induced by that region.

The actual-right-wall progress trace is therefore topology-aware in a practical sense:

- it depends on the contour structure of the valid tube,
- it avoids artifacts from raw polyline ordering,
- and it uses geometry/topology to define a stable 1D progress coordinate.

## 7.3 Behavior-manifold/topology integration path

The repo’s RL behavior manifold layer already includes a topology branch with time-image and zigzag-persistence-inspired interfaces:

- `docs/rl_behavior_manifold_implementation_details.md`

This means the overall program has a coherent arc:

1. use counterfactual geometry and DAGs to define intervention-consistent behavior families,
2. use rollout RL to train commitment to those families,
3. use topology-aware behavior embeddings to compare, cluster, and score the resulting behavior manifold.

That is a very natural paper story.

---

## 8) Causal DAGs and Counterfactual Training

## 8.1 Why DAGs belong here

Counterfactual training is fundamentally causal. A useful counterfactual example answers:

- which variable was intervened on,
- which parts of the world were held fixed,
- which downstream outcomes should change,
- and which should remain invariant.

The local intervention contract and DAG projection make this explicit.

## 8.2 Current DAG representation

The current simplified DAG contract is:

- schema: `counter_bmt_v2_dag_cache_v3_maneuver_outcome`
- contract: `maneuver_outcome_v1`

with compact node types such as:

- maneuver,
- outcome.

This is already enough to carry decision-to-outcome semantics while keeping latent capacity focused on trajectory-relevant structure.

## 8.3 Why this matters for the paper

The paper should not describe DAGs as a side feature. The stronger framing is:

- DAGs provide the **causal semantic scaffold**,
- counterfactual dataset construction provides the **intervention instances**,
- rollout RL provides the **behavioral realization mechanism**.

That is a cohesive contribution.

---

## 9) The Current Training Objective

For the current semantic rollout training setup, the main ingredients are:

- semantic-only conditioning,
- optional family-guide realism prior,
- rollout tube policy loss,
- progress reward measured on the actual right-wall trace,
- group-relative return normalization.

At a high level, the per-step rollout reward is:

- positive when the realized rollout state remains inside the valid tube,
- negative when it leaves the tube,
- plus additional progress reward for moving forward along the capped progress trace.

This objective is interesting scientifically because it expresses the desired behavior as:

- stay in the valid counterfactual region,
- commit to the requested branch,
- and make forward decision-consistent progress,

rather than:

- exactly imitate one trajectory.

---

## 10) Important Scientific Lessons From the Development Process

These are not just engineering notes. They are methodological findings worth mentioning in a paper or appendix.

## 10.1 Naive progress along the raw path is wrong

Raw path order can contain:

- discontinuity connectors,
- backward zig-zags,
- parallel lane artifacts,
- long loops that remain spatially nearby.

Using that raw order as the progress signal mis-specifies the task.

## 10.2 A valid region needs its own progress coordinate

A 2D valid tube does not automatically define a stable 1D notion of progress. That coordinate must be constructed carefully.

This is what led to the current wall-trace solution.

## 10.3 Capped progress is necessary

If progress reward grows indefinitely with path length, the policy is pushed to:

- speed up too much,
- cut corners,
- and chase the far end of the route even when the path is only meant to indicate validity.

Bounding progress relative to divergence is therefore not a detail; it is part of the scientific formulation.

## 10.4 Family supervision and rollout optimization complement each other

The family-guide loss and rollout RL solve different problems:

- family guidance preserves realism and semantic-family consistency,
- rollout RL enforces branch commitment at behavior level.

That division of labor is important.

---

## 11) Strong Candidate Contribution Statements

If the paper needs a compact contribution list, the strongest claims are:

1. **A geometry-grounded counterfactual training pipeline for autonomous motion generation.**
   Counterfactual alternatives are constructed from local branch structure and causal intervention contracts rather than arbitrary relabeling.

2. **A semantic-only control formulation with privileged family supervision.**
   The model is controlled by compact decision semantics at runtime while learning from richer path-family geometry during training.

3. **A rollout-level counterfactual objective that optimizes branch validity and commitment.**
   Grouped sampled rollouts are scored against valid counterfactual tubes and optimized with relative advantages.

4. **A topology-aware progress representation derived from the actual valid-region contour.**
   Progress is measured on an actual right-wall trace of the valid tube rather than brittle raw path arc-length, solving a key failure mode in counterfactual route optimization.

5. **A causal DAG interface linking intervention semantics to downstream outcomes.**
   Local interventions are projected into sparse decision-outcome DAGs, making the control problem causally interpretable and compatible with structured topology-aware policy optimization.

6. **A bridge from causal counterfactual control to Topo-MCPO-style behavior-manifold RL.**
   The current system provides structured counterfactual rollout families, causal semantics, and topology-aware geometry that naturally support manifold-based novelty and consensus optimization.

---

## 12) Suggested Paper Positioning

The cleanest positioning is:

> We present a counterfactual semantic control framework for motion generation that combines geometry-grounded intervention construction, causal DAG structure, and topology-aware rollout optimization.

Then:

- the **causal** angle comes from local intervention contracts and DAGs,
- the **counterfactual** angle comes from alternative branch supervision,
- the **topology-aware** angle comes from valid-region-derived progress structure,
- the **RL** angle comes from grouped rollout optimization over counterfactual behavior families,
- and the **Topo-MCPO connection** comes from viewing these rollouts as members of a structured behavior manifold rather than standalone trajectories.

---

## 13) Good Figures To Include

Strong paper figures would be:

1. **Counterfactual intervention example**
   A scene with factual branch and semantic alternatives (`left`, `right`, `right_lane_change`).

2. **Valid tube and actual wall trace**
   Show the orange valid region and the right-wall progress path that fixes raw-path zig-zag artifacts.

3. **Causal DAG view of one intervention**
   Show context -> decision -> outcome edges for a local intersection example.

4. **Training objective visualization**
   Compare:
   - raw path arc progress,
   - invalid centerline progress,
   - actual wall-trace progress,
   - and resulting rollout rewards.

5. **Behavior-manifold connection**
   A conceptual diagram showing how counterfactual rollout families feed a topology-aware RL layer in the style of Topo-MCPO.

---

## 14) Honest Current Limitations

The document should also be candid about what is not yet complete.

- The current Adv-BMT semantic rollout pipeline is not yet a full end-to-end implementation of the larger behavior-manifold Topo-MCPO stack.
- TEN-DM-inspired topology representation is currently connected most strongly at the conceptual and infrastructure level, not as a full deployed zigzag-persistence representation in the semantic rollout training path.
- Some of the strongest methodology emerged through debugging and geometric refinement, so the final paper should present the chosen progress definition clearly and motivate why simpler alternatives fail.

These are acceptable limitations. They do not weaken the scientific story; they clarify where the present work sits in a larger research program.

---

## 15) Bottom Line

The strongest scientific claim is not just that we trained another controllable motion model.

It is that we built a framework where:

- counterfactual alternatives are causally and geometrically grounded,
- control is expressed in semantic decision space,
- realism is preserved through family-structured supervision,
- rollout RL optimizes branch commitment rather than only token imitation,
- progress is defined by the topology of the valid region itself,
- and causal DAGs provide a principled bridge to structured/topology-aware policy optimization.

That combination is the real novelty.
