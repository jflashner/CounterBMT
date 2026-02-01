"""
rl_controller_loop.py

Minimal RL loop skeleton:
    local LLM controller (Qwen3 8B) -> BMT rollout -> VLM judge -> reward

This is intentionally lightweight and hackable. It does NOT implement PPO/GRPO
updates yet; it logs rollouts so you can plug in your favorite trainer.
"""

import argparse
import json
import inspect
import logging
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np

from counter_bmt.bmt_generator import CounterBMTGenerator
from counter_bmt.scenario_export import export_trajectory_only, create_dataset_summary
from counter_bmt.scenarionet_visualizer import ScenarioNetVisualizer, ScenarioNetDatabase
from counter_bmt.vlm_extractor import (
    VLMSafetyCriticalExtractor,
    TimestampedImage,
    GPT4oClient,
    MockGPT4oClient,
)

# Reuse the scenario loader from the main pipeline
from scripts.run_full_pipeline import load_scenario_for_bmt_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rl_controller_loop")


# =============================================================================
# Controller (Qwen3 8B)
# =============================================================================

@dataclass
class InterventionPlan:
    target_type: str  # "maneuver" or "decision"
    target_value: str
    aggressiveness: str = "normal"
    timestamp: Optional[float] = None
    description: str = ""


class QwenController:
    """
    Minimal local LLM controller using Qwen3 8B (via transformers).
    Outputs a JSON plan for the desired intervention.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.9,
        model=None,
        tokenizer=None,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise ImportError(
                "transformers + torch required for local Qwen controller. "
                "Install with: pip install torch transformers"
            ) from e

        self.torch = torch
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device,
            )
            self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        allowed_maneuvers = [
            "straight", "accelerate", "decelerate", "stop",
            "lane_change_left", "lane_change_right", "left_turn", "right_turn",
        ]
        allowed_decisions = ["proceed", "yield", "slow_down", "speed_up", "merge"]

        return (
            "You are a driving intervention planner. "
            "Output a JSON object only (no extra text).\n\n"
            "Context:\n"
            f"- scenario_id: {context.get('scenario_id')}\n"
            f"- iteration: {context.get('iteration')}\n"
            f"- last_reward: {context.get('last_reward')}\n\n"
            "Choose a target intervention.\n"
            f"Allowed maneuvers: {allowed_maneuvers}\n"
            f"Allowed decisions: {allowed_decisions}\n\n"
            "JSON schema:\n"
            "{\n"
            '  "target_type": "maneuver" | "decision",\n'
            '  "target_value": "<value>",\n'
            '  "aggressiveness": "passive" | "normal" | "aggressive",\n'
            '  "timestamp": <float or null>,\n'
            '  "description": "<short description>"\n'
            "}\n"
        )

    def _extract_json(self, text: str) -> Optional[Dict]:
        # Grab the first JSON object by brace matching
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    def generate_plan(
        self,
        context: Dict[str, Any]
    ) -> Tuple[Optional[InterventionPlan], str, str, str, Optional[Any], Optional[Any]]:
        prompt = self._build_prompt(context)

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.model.device)
        input_len = input_ids.shape[-1]

        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = output_ids[0, input_len:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        parsed = self._extract_json(response_text) or self._extract_json(decoded)
        if not parsed:
            return None, decoded, prompt, response_text, input_ids[0], response_ids

        plan = InterventionPlan(
            target_type=parsed.get("target_type", "maneuver"),
            target_value=parsed.get("target_value", "straight"),
            aggressiveness=parsed.get("aggressiveness", "normal"),
            timestamp=parsed.get("timestamp", None),
            description=parsed.get("description", ""),
        )
        return plan, decoded, prompt, response_text, input_ids[0], response_ids


# =============================================================================
# Judge
# =============================================================================

def _normalize_choice(s: str) -> str:
    return s.lower().strip().replace(" ", "_").replace("-", "_")


def score_features(features, target_type: str, target_value: str) -> Tuple[float, Dict]:
    details = {"target_type": target_type, "target_value": target_value}

    if target_type == "maneuver":
        best = 0.0
        best_match = None
        for m in features.maneuver_sequence:
            if m.maneuver_type.value == target_value:
                conf = float(getattr(m, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = m
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    if target_type == "decision":
        best = 0.0
        best_match = None
        for d in features.critical_decisions:
            choice = _normalize_choice(d.ground_truth_choice)
            if choice == target_value:
                conf = float(getattr(d, "confidence", 1.0) or 1.0)
                if conf > best:
                    best = conf
                    best_match = d
        details["match"] = asdict(best_match) if best_match else None
        return best, details

    details["match"] = None
    return 0.0, details


def run_vlm_judge(
    replay_dir: Path,
    target_type: str,
    target_value: str,
    num_frames: int,
    client,
    debug_dir: Path,
) -> Tuple[float, Dict]:
    visualizer = ScenarioNetVisualizer(data_dir=str(replay_dir))
    extractor = VLMSafetyCriticalExtractor(
        client=client,
        debug=True,
        debug_output_dir=str(debug_dir),
    )

    try:
        saved_images, trajectory, scenario_id = visualizer.render_scenario(
            scenario_index=0,
            num_frames=num_frames,
            output_dir=str(debug_dir / "frames"),
        )
        images = [TimestampedImage(path=p, timestamp=t) for p, t in saved_images]
        features = extractor.extract(images, scenario_id=scenario_id, trajectory=trajectory)
        reward, details = score_features(features, target_type, target_value)
        return reward, details
    finally:
        visualizer.close()


# =============================================================================
# Loop
# =============================================================================

def build_intervention(plan: InterventionPlan) -> Dict[str, Any]:
    variable = "maneuver_0" if plan.target_type == "maneuver" else "decision_0"
    return {
        "variable": variable,
        "value": plan.target_value,
        "original_value": None,
        "aggressiveness": plan.aggressiveness,
        "timestamp": plan.timestamp,
        "description": plan.description,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RL loop skeleton for CounterBMT")
    parser.add_argument("--data-dir", required=True, help="ScenarioNet data directory")
    parser.add_argument("--scenario-index", type=int, default=0, help="Scenario index")
    parser.add_argument("--bmt-checkpoint", required=True, help="BMT checkpoint path")
    parser.add_argument("--num-iters", type=int, default=3, help="Number of RL iterations")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames for VLM judge")
    parser.add_argument("--n-samples", type=int, default=1, help="BMT samples per iteration")
    parser.add_argument("--temperature", type=float, default=None, help="BMT sampling temperature")
    parser.add_argument("--bias-strength", type=float, default=8.0, help="Token bias strength")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Random exploration rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--mock-vlm", action="store_true", help="Use mock VLM (no API)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--vlm-model", type=str, default="gpt-4o", help="VLM model name")
    parser.add_argument("--controller-model", type=str, default="Qwen/Qwen3-8B", help="Qwen model name")
    parser.add_argument("--use-grpo", action="store_true", help="Enable TRL GRPO updates")
    parser.add_argument("--use-trl", action="store_true", help="(deprecated) Alias for --use-grpo")
    parser.add_argument("--trl-lr", type=float, default=1e-5, help="TRL learning rate")
    parser.add_argument("--trl-epochs", type=int, default=1, help="TRL GRPO epochs")
    parser.add_argument("--trl-kl-coef", type=float, default=0.05, help="TRL KL/Beta coefficient")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "rl_loop" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # VLM client
    if args.mock_vlm:
        vlm_client = MockGPT4oClient()
    else:
        if args.api_key:
            os.environ["OPENAI_API_KEY"] = args.api_key
        vlm_client = GPT4oClient(model=args.vlm_model)

    # Load BMT model
    generator = CounterBMTGenerator.from_checkpoint(args.bmt_checkpoint)
    generator.compiler.DEFAULT_ENCOURAGE_BIAS = args.bias_strength

    # Load scenario
    scenario_data = load_scenario_for_bmt_input(
        data_dir=args.data_dir,
        scenario_index=args.scenario_index,
        config=generator.config,
        tokenizer=generator.tokenizer,
    )

    # Map center for exporting
    map_center = None
    preprocessed = scenario_data.get("preprocessed", {})
    if "metadata/map_center" in preprocessed:
        map_center = preprocessed["metadata/map_center"]

    # Controller (optional TRL GRPO)
    grpo_trainer = None
    use_grpo = args.use_grpo or args.use_trl
    if use_grpo:
        try:
            from transformers import AutoTokenizer
            from trl import GRPOConfig, GRPOTrainer, AutoModelForCausalLMWithValueHead
        except Exception as e:
            raise ImportError(
                "TRL required for --use-grpo. Install with: pip install trl"
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(args.controller_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.controller_model,
            trust_remote_code=True,
            torch_dtype=None,
            device_map="auto",
        )

        # Build GRPO config in a version-tolerant way
        config_kwargs = {
            "model_name": args.controller_model,
            "learning_rate": args.trl_lr,
            "batch_size": 1,
            "mini_batch_size": 1,
            "ppo_epochs": args.trl_epochs,
            "grpo_epochs": args.trl_epochs,
            "num_train_epochs": args.trl_epochs,
            "init_kl_coef": args.trl_kl_coef,
            "kl_coef": args.trl_kl_coef,
            "beta": args.trl_kl_coef,
        }
        cfg_sig = inspect.signature(GRPOConfig)
        filtered_cfg = {k: v for k, v in config_kwargs.items() if k in cfg_sig.parameters}
        grpo_config = GRPOConfig(**filtered_cfg)

        # Build GRPO trainer with best-effort kwargs
        trainer_kwargs = {
            "config": grpo_config,
            "model": model,
            "tokenizer": tokenizer,
        }
        tr_sig = inspect.signature(GRPOTrainer)
        filtered_tr = {k: v for k, v in trainer_kwargs.items() if k in tr_sig.parameters}
        grpo_trainer = GRPOTrainer(**filtered_tr)

        controller = QwenController(
            model_name=args.controller_model,
            model=grpo_trainer.model,
            tokenizer=grpo_trainer.tokenizer,
        )
        logger.info("TRL GRPO enabled for controller updates")
    else:
        controller = QwenController(model_name=args.controller_model)

    rollout_log = output_dir / "rollouts.jsonl"
    last_reward = 0.0

    for it in range(args.num_iters):
        context = {
            "scenario_id": scenario_data.get("scenario_id"),
            "iteration": it,
            "last_reward": last_reward,
        }

        # Exploration: random plan occasionally
        if random.random() < args.epsilon:
            plan = InterventionPlan(
                target_type="maneuver",
                target_value=random.choice([
                    "lane_change_left", "lane_change_right", "left_turn",
                    "right_turn", "accelerate", "decelerate", "straight"
                ]),
                aggressiveness=random.choice(["passive", "normal", "aggressive"]),
                timestamp=None,
                description="random exploration",
            )
            llm_raw = "<random>"
            prompt = ""
            response_text = ""
            query_ids = None
            response_ids = None
        else:
            plan, llm_raw, prompt, response_text, query_ids, response_ids = controller.generate_plan(context)
            if plan is None:
                plan = InterventionPlan(target_type="maneuver", target_value="straight")
                prompt = ""
                response_text = ""
                query_ids = None
                response_ids = None

        intervention = build_intervention(plan)

        # BMT rollout
        result = generator.generate_counterfactual(
            scenario_data=scenario_data["preprocessed"],
            intervention=intervention,
            n_samples=args.n_samples,
            temperature=args.temperature,
            return_baseline=False,
        )

        # Take first sample
        cf_traj = result["counterfactual_trajectories"][0]
        ego_positions = cf_traj["positions"][:, 0, :2]

        # Export for replay/judge
        iter_dir = output_dir / f"iter_{it:03d}"
        replay_dir = iter_dir / "replay_scenarios"
        replay_dir.mkdir(parents=True, exist_ok=True)

        export_path = replay_dir / f"sd_counterfactual_1.0_{scenario_data['scenario_id']}_iter{it:03d}.pkl"
        export_trajectory_only(
            trajectory=ego_positions,
            original_scenario=scenario_data["raw_data"],
            output_path=export_path,
            intervention_name=f"{plan.target_type}:{plan.target_value}",
            original_file_path=scenario_data.get("file_path"),
            map_center=map_center,
        )
        create_dataset_summary([export_path], replay_dir)

        # VLM judge
        reward, details = run_vlm_judge(
            replay_dir=replay_dir,
            target_type=plan.target_type,
            target_value=plan.target_value,
            num_frames=args.num_frames,
            client=vlm_client,
            debug_dir=iter_dir / "vlm_debug",
        )
        last_reward = reward

        log_entry = {
            "iteration": it,
            "plan": asdict(plan),
            "intervention": intervention,
            "reward": reward,
            "details": details,
            "llm_raw": llm_raw,
            "prompt": prompt,
            "response_text": response_text,
            "export_path": str(export_path),
        }
        with open(rollout_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.info(f"[iter {it}] reward={reward:.3f} target={plan.target_type}:{plan.target_value}")

        # TRL GRPO update (single-step, minimal)
        if grpo_trainer is not None and query_ids is not None and response_ids is not None:
            # TRL expects lists of tensors and scalar rewards
            grpo_stats = grpo_trainer.step(
                [query_ids],
                [response_ids],
                [reward],
            )
            with open(output_dir / "trl_grpo_stats.jsonl", "a") as f:
                f.write(json.dumps(grpo_stats, default=str) + "\n")

    summary = {
        "num_iters": args.num_iters,
        "output_dir": str(output_dir),
        "rollout_log": str(rollout_log),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. Logs in {output_dir}")


if __name__ == "__main__":
    main()
