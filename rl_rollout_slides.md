# CounterBMT RL Rollout (Draft Slides)

## 1. Title
- **CounterBMT: RL Rollout for Intervention Planning**
- Goal: learn low‑level interventions with VLM reward
- Date: 2026‑02‑01

---

## 2. Motivation
- Existing pipeline uses LLM planner + token biasing (works, but static)
- Want **policy learning**: Qwen proposes interventions, RL optimizes
- Reward from VLM: did the counterfactual show the intended maneuver?

---

## 3. End‑to‑End Flow (Current RL Loop)
- Scenario load (same path as `run_full_pipeline`)
- Qwen controller → high‑level intervention
- LLMInterventionPlanner → multi‑phase token bias plan
- BMT rollout → trajectory → replay export
- VLM judge → reward → GRPO update

---

## 4. Key Components Implemented
- `src/scripts/rl_controller_loop.py`
  - Multi‑scenario support (`--scenario-list`, `--scenario-range`, `--num-scenarios`)
  - Logging for LLM + VLM + reward
  - Replay export per iteration
- Qwen as planner client (`--planner-client qwen`)
- VLM debug artifacts saved per iteration

---

## 5. Logging & Artifacts
- `rollouts.jsonl`: reward + target + plan
- `llm_output.txt`: prompt + raw response + parsed plan
- `vlm_output.txt`: maneuvers/decisions summary
- `reward_report.json`: reward + frames + VLM raw responses
- `vlm_debug/frames/*.png` + `vlm_debug/*.json`

---

## 6. Current Bottlenecks
- Rewards often 0 → VLM sees “stop” or no maneuver
- Controller still picks limited actions (needs richer context)
- Planner quality depends on scenario motion + frame clarity

---

## 7. Next Improvements
- Feed richer context to Qwen planner (trajectory stats, current speed)
- Add reward shaping: distance / heading / speed deltas
- Option: train Qwen to output phases directly (skip planner)

---

## 8. Demo Commands
- Single scenario:
  - `python src/scripts/rl_controller_loop.py --data-dir data/scenarionet_waymo_training_500 --scenario-id 90078b3dd8e4b3b1 --bmt-checkpoint bmt/ckpt/last.ckpt --controller-model Qwen/Qwen3-4B --planner-client qwen`
- Batch:
  - `--scenario-range 0 49` or `--num-scenarios 50`

